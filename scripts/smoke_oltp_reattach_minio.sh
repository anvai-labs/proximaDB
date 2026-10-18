#!/usr/bin/env bash
# Copyright (C) 2026 ProximaDB
# SPDX-License-Identifier: Apache-2.0
#
# ADR-095 smoke: Postgres-compatible OLTP with the WAL on a reattachable volume
# and the writer-epoch authority in a REAL S3-compatible store (MinIO).
#
#   1. start MinIO on loopback, create a bucket;
#   2. host A: every durable root on "volume" vol-a, epoch log in s3://bucket/ctl,
#      [oltp] atomic_commit on;
#   3. bank-transfer workload over pgwire (psql): each transfer is ONE transaction
#      (debit + credit + history row); every 7th is forced to fail at COMMIT;
#   4. freeze A (SIGSTOP), "move" its volume to host B (copy), boot B: B must win
#      the next epoch from MinIO and serve every committed transfer;
#   5. resume A: it must self-fence and exit 75;
#   6. SIGKILL B, restart it: committed state survives crash recovery;
#   7. invariant after quiescing: total balance unchanged, history rows ==
#      committed transfers, and the bucket holds one epoch slot per boot.
#
# Prereqs: minio + mc (build: `go install github.com/minio/minio@<tag>` and
# `github.com/minio/mc@latest`; dl.min.io no longer serves binaries), psql, and
# a server built with S3 support:
#   cargo build -p proximadb-server --features aws
#
# Usage: scripts/smoke_oltp_reattach_minio.sh [--epoch-store-test]
#   PROXIMADB_BIN  server binary (default target/debug/proximadb-server)
#   MINIO_BIN / MC_BIN  (default ~/go/bin/minio, ~/go/bin/mc)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROXIMADB_BIN="${PROXIMADB_BIN:-$REPO_ROOT/target/debug/proximadb-server}"
MINIO_BIN="${MINIO_BIN:-$HOME/go/bin/minio}"
MC_BIN="${MC_BIN:-$HOME/go/bin/mc}"
RUN_EPOCH_STORE_TEST=0
[[ "${1:-}" == "--epoch-store-test" ]] && RUN_EPOCH_STORE_TEST=1

ACCOUNTS=10
START_BALANCE=1000
TRANSFERS=40
BUCKET=proximadb-smoke
DEPLOYMENT=smoke

for bin in "$PROXIMADB_BIN" "$MINIO_BIN" "$MC_BIN"; do
  [[ -x "$bin" ]] || { echo "missing executable: $bin" >&2; exit 2; }
done
command -v psql >/dev/null || { echo "psql is required" >&2; exit 2; }

WORK="$(mktemp -d -t proximadb-oltp-smoke-XXXXXX)"
PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    [[ -n "$pid" ]] && { kill -CONT "$pid" 2>/dev/null || true; kill -9 "$pid" 2>/dev/null || true; }
  done
  if [[ "${KEEP_WORK:-0}" == "1" ]]; then echo "work dir kept: $WORK"; else rm -rf "$WORK"; fi
}
trap cleanup EXIT

free_port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()'; }
log() { printf '\n== %s\n' "$*"; }
fail() { echo "SMOKE FAIL: $*" >&2; exit 1; }

# --- 1. MinIO ---------------------------------------------------------------
MINIO_PORT="$(free_port)"; MINIO_CONSOLE="$(free_port)"
log "starting MinIO on 127.0.0.1:$MINIO_PORT"
MINIO_ROOT_USER=minioadmin MINIO_ROOT_PASSWORD=minioadmin \
  "$MINIO_BIN" server "$WORK/minio" --address "127.0.0.1:$MINIO_PORT" \
  --console-address "127.0.0.1:$MINIO_CONSOLE" >"$WORK/minio.log" 2>&1 &
PIDS+=($!)
export MC_CONFIG_DIR="$WORK/mc"
for _ in $(seq 1 60); do
  "$MC_BIN" alias set smoke "http://127.0.0.1:$MINIO_PORT" minioadmin minioadmin >/dev/null 2>&1 && \
    "$MC_BIN" ls smoke >/dev/null 2>&1 && break
  sleep 0.5
done
"$MC_BIN" mb --ignore-existing "smoke/$BUCKET" >/dev/null || fail "could not create bucket"

export AWS_ENDPOINT="http://127.0.0.1:$MINIO_PORT"
export AWS_ALLOW_HTTP=true
export AWS_VIRTUAL_HOSTED_STYLE_REQUEST=false
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_REGION=us-east-1
CONTROL_URL="s3://$BUCKET/ctl"

if [[ "$RUN_EPOCH_STORE_TEST" == "1" ]]; then
  log "epoch-store contract test against MinIO"
  (cd "$REPO_ROOT" && PROXIMADB_TEST_EPOCH_S3_URL="s3://$BUCKET/epoch-test" \
    cargo test -p proximadb --features aws --lib cluster::writer_epoch -- --ignored --nocapture)
fi

# --- host helpers -----------------------------------------------------------
write_config() { # name volume rest grpc flight pg
  local name="$1" vol="$2" rest="$3" grpc="$4" flight="$5" pg="$6"
  cat >"$WORK/$name/proximadb.toml" <<EOF
[server]
node_id = "$name"
bind_address = "127.0.0.1"
port = $rest
data_dir = "$vol/data"

[server.tenant]
mode = "single_tenant"
default_tenant = "default"

[storage]
metadata_url = "file://$vol/metadata"
mmap_enabled = false

[[storage.storage_locations]]
url = "file://$vol/collections"
weight = 1
tags = ["oltp"]

[storage.optimization]
enable_mmap = false

[storage.wal_config]
write_buffer_directory = "$vol/wal"
enable_wal = true
sync_mode = "PerBatch"

[storage.sst_config]
data_directory = "$vol/collections"
mmap_enabled = false

[storage.viper_config]
data_directory = "$vol/collections"

[api]
rest_port = $rest
grpc_port = $grpc
arrow_flight_port = $flight
pg_port = $pg
unified_mode = false

[monitoring]
metrics_enabled = false
log_level = "info"

[oltp]
atomic_commit = true

[oltp.durable_volume]
root = "$vol"
deployment_id = "$DEPLOYMENT"
control_url = "$CONTROL_URL"
fence_check_interval_ms = 300
EOF
}

declare -A HOST_PID HOST_PG HOST_REST
start_host() { # name volume
  local name="$1" vol="$2"
  mkdir -p "$WORK/$name" "$vol"
  local rest grpc flight pg
  rest="$(free_port)"; grpc="$(free_port)"; flight="$(free_port)"; pg="$(free_port)"
  write_config "$name" "$vol" "$rest" "$grpc" "$flight" "$pg"
  (cd "$WORK/$name" && RUST_LOG=info exec "$PROXIMADB_BIN" --config proximadb.toml) \
    >>"$WORK/$name.log" 2>&1 &
  local pid=$!
  PIDS+=("$pid")
  HOST_PID[$name]=$pid; HOST_PG[$name]=$pg; HOST_REST[$name]=$rest
  for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$rest/health" >/dev/null 2>&1 && return 0
    kill -0 "$pid" 2>/dev/null || { tail -40 "$WORK/$name.log" >&2; fail "$name exited during boot"; }
    sleep 0.5
  done
  tail -40 "$WORK/$name.log" >&2; fail "$name did not become healthy"
}

sql() { # host sql...
  local name="$1"; shift
  PGCONNECT_TIMEOUT=5 psql -X -q -v ON_ERROR_STOP=1 -tA \
    "host=127.0.0.1 port=${HOST_PG[$name]} user=anonymous dbname=proximadb sslmode=disable" -c "$*"
}

balance_of() { sql "$1" "SELECT balance FROM accounts WHERE id = $2" | tr -d '[:space:]'; }

epoch_slots() {
  "$MC_BIN" ls --recursive "smoke/$BUCKET/ctl/_control/$DEPLOYMENT/writer/" 2>/dev/null | wc -l | tr -d ' '
}

# --- 2/3. host A + workload ---------------------------------------------------
VOL_A="$WORK/vol-a"; VOL_B="$WORK/vol-b"
log "host A on $VOL_A (epoch authority: $CONTROL_URL)"
start_host a "$VOL_A"
sql a "CREATE TABLE accounts (id BIGINT PRIMARY KEY, balance BIGINT)"
sql a "CREATE TABLE history (id BIGINT PRIMARY KEY, src BIGINT, dst BIGINT, amount BIGINT)"
for id in $(seq 0 $((ACCOUNTS - 1))); do
  sql a "INSERT INTO accounts (id, balance) VALUES ($id, $START_BALANCE)"
done
# A pre-existing history row that every 7th transfer collides with at COMMIT.
sql a "INSERT INTO history (id, src, dst, amount) VALUES (999999, -1, -1, 0)"

COMMITTED=0; FAILED=0
log "running $TRANSFERS transfer transactions (every 7th forced to fail at COMMIT)"
for i in $(seq 1 "$TRANSFERS"); do
  src=$(( i % ACCOUNTS )); dst=$(( (i * 3 + 1) % ACCOUNTS ))
  [[ "$src" == "$dst" ]] && dst=$(( (dst + 1) % ACCOUNTS ))
  amount=$(( (i % 9) + 1 ))
  sb="$(balance_of a "$src")"; db="$(balance_of a "$dst")"
  hid="$i"; (( i % 7 == 0 )) && hid=999999
  if sql a "BEGIN;
            UPDATE accounts SET balance = $((sb - amount)) WHERE id = $src;
            UPDATE accounts SET balance = $((db + amount)) WHERE id = $dst;
            INSERT INTO history (id, src, dst, amount) VALUES ($hid, $src, $dst, $amount);
            COMMIT" >/dev/null 2>"$WORK/txn.err"; then
    COMMITTED=$((COMMITTED + 1))
  else
    FAILED=$((FAILED + 1))
  fi
done
echo "committed=$COMMITTED failed=$FAILED"
(( FAILED > 0 )) || fail "the forced failures did not fail"

check_invariants() { # host label
  local total hist
  total="$(sql "$1" "SELECT SUM(balance) FROM accounts" | tr -d '[:space:]')"
  hist="$(sql "$1" "SELECT COUNT(*) FROM history" | tr -d '[:space:]')"
  echo "[$2] total=$total history=$hist (expect total=$((ACCOUNTS * START_BALANCE)) history=$((COMMITTED + 1)))"
  [[ "$total" == "$((ACCOUNTS * START_BALANCE))" ]] || fail "[$2] money was created or destroyed (partial commit)"
  [[ "$hist" == "$((COMMITTED + 1))" ]] || fail "[$2] history rows != committed transfers"
  # The sum alone is blind to a BALANCED partial transfer (debit + credit applied,
  # history row lost). Reconcile every account against the ledger instead:
  # balance == start + credits - debits, per account.
  sql "$1" "SELECT id, balance FROM accounts" >"$WORK/balances.txt"
  sql "$1" "SELECT src, dst, amount FROM history" >"$WORK/history.txt"
  python3 - "$WORK/balances.txt" "$WORK/history.txt" "$START_BALANCE" "$2" <<'PY' || fail "[$2] an account does not reconcile with the ledger (partial transaction leaked)"
import sys
balances = {int(i): int(b) for i, b in (l.split("|") for l in open(sys.argv[1]) if l.strip())}
expected = {i: int(sys.argv[3]) for i in balances}
for line in open(sys.argv[2]):
    if not line.strip():
        continue
    src, dst, amount = (int(v) for v in line.split("|"))
    if src >= 0:
        expected[src] -= amount
        expected[dst] += amount
bad = {i: (balances[i], expected[i]) for i in balances if balances[i] != expected[i]}
if bad:
    print(f"[{sys.argv[4]}] ledger mismatch (balance, expected): {bad}")
    sys.exit(1)
print(f"[{sys.argv[4]}] every account reconciles with the ledger")
PY
}
check_invariants a "host A"
[[ "$(epoch_slots)" == "1" ]] || fail "expected 1 epoch slot after host A boot, found $(epoch_slots)"

# --- 4. freeze A, move the volume, boot B -------------------------------------
log "freezing host A and moving its volume to host B"
kill -STOP "${HOST_PID[a]}"
cp -a "$VOL_A" "$VOL_B"
start_host b "$VOL_B"
[[ "$(epoch_slots)" == "2" ]] || fail "host B did not win a new epoch in MinIO"
check_invariants b "host B after reattach"

# --- 5. resume A: it must self-fence -----------------------------------------
log "resuming host A — it must observe epoch 2 and exit 75"
kill -CONT "${HOST_PID[a]}"
code=0
for _ in $(seq 1 40); do
  if ! kill -0 "${HOST_PID[a]}" 2>/dev/null; then break; fi
  sleep 0.25
done
if kill -0 "${HOST_PID[a]}" 2>/dev/null; then fail "host A kept running after being fenced"; fi
wait "${HOST_PID[a]}" || code=$?
[[ "$code" == "75" ]] || fail "host A exited $code, expected 75 (fenced)"
grep -q "writer fenced" "$WORK/a.log" || fail "host A did not log the fence"
echo "host A exited 75 (fenced)"

# --- 6. crash B, recover -------------------------------------------------------
log "SIGKILL host B and restart it on the same volume"
sql b "BEGIN; INSERT INTO history (id, src, dst, amount) VALUES (1000000, 0, 0, 0); COMMIT" >/dev/null
COMMITTED=$((COMMITTED + 1))
kill -9 "${HOST_PID[b]}"; wait "${HOST_PID[b]}" 2>/dev/null || true
start_host b2 "$VOL_B"
check_invariants b2 "host B after crash recovery"
[[ "$(epoch_slots)" == "3" ]] || fail "restart did not claim a fresh epoch"

log "SMOKE PASS: atomic transfers survived reattach + crash; stale owner fenced (epochs in MinIO: $(epoch_slots))"
