#!/bin/bash
# TD-CACHEC-1 smoke: per-collection survivor-cache attribution end-to-end
# (release-server boot -> REST create -> insert -> ADR-069 driver flush ->
# search -> scrape /metrics/prometheus). PASS = a
# proximadb_survivor_cache_collection_reads_total series exists for the
# smoke collection's CANONICAL id with non-zero counts. See TD-CACHEC-1 for
# the outcome semantics (first touch = cold; warm queries are served upstream
# of get_or_fetch and record nothing). Requires: curl, python3, cargo.
set -euo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Derive the port from the PID: two concurrent runs must never share a
# server (the health poll would happily answer against the other instance
# and invalidate the whole run).
PORT=$((20000 + $$ % 20000))
DATA=$(mktemp -d /tmp/cachec-smoke-data.XXXX)
CFG=$(mktemp /tmp/cachec-smoke-cfg.XXXX.toml)
LOG=$(mktemp /tmp/cachec-smoke-server.XXXX.log)
trap 'kill $SRV 2>/dev/null || true; rm -rf "$DATA" "$CFG" "$LOG"' EXIT

# Arm the ADR-069 auto-flush driver ([storage.wal_config] flush_interval_secs
# — the shipped default is a 300s RPO floor; 10s makes the drain fast) and
# move the public port ([api].unified_port — [server].port binds nothing, and
# the CLI -p flag is a no-op). flush_interval_s (no 'secs') is the io_trace
# ETL sink's unrelated timer — deliberately NOT touched here.
sed "s/^unified_mode = true/unified_mode = true\\nunified_port = $PORT/; s/^memory_flush_size_bytes = 16777216/memory_flush_size_bytes = 16777216\\nflush_interval_secs = 10/" \
  "$WT/config/config.toml" > "$CFG"
# Assert the edits landed: a drifted anchor would silently leave the driver
# disarmed and the smoke would fail far from the cause.
grep -q "^unified_port = $PORT$" "$CFG" || { echo "FATAL: unified_port override did not land"; exit 1; }
grep -q "^flush_interval_secs = 10$" "$CFG" || { echo "FATAL: flush_interval_secs override did not land"; exit 1; }

echo "== building release-server (if stale)"
CARGO_INCREMENTAL=0 cargo build --profile release-server --manifest-path "$WT/Cargo.toml" -p proximadb-server --bin proximadb-server

echo "== starting server on :$PORT (data: $DATA)"
RUST_LOG=info "$WT/target/release-server/proximadb-server" -c "$CFG" -d "$DATA" -p "$PORT" \
  > "$LOG" 2>&1 &
SRV=$!

for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health/live" >/dev/null 2>&1 && break
  sleep 1
done
curl -sf "http://127.0.0.1:$PORT/health/live" >/dev/null || { echo "server did not start"; tail -20 "$LOG"; exit 1; }
echo "== server up"

CID="smoke_cachec_$(date +%s)"
CREATE_RESP=$(curl -sf -X POST "http://127.0.0.1:$PORT/api/v2/collections" -H 'content-type: application/json' -d \
  "{\"name\":\"$CID\",\"dimension\":8,\"engine\":\"sst\",\"enable_proxima_record\":true}")
# The metric labels carry the catalog CANONICAL collection id, not the name.
CIDS=$(echo "$CREATE_RESP" | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["collection_id"])')
echo "== collection $CID created (canonical id: $CIDS)"

python3 -I - "$CID" "$PORT" <<'PY'
import json, sys, urllib.request
cid, port = sys.argv[1], sys.argv[2]
recs = [{"id": f"r_{i}",
         "vector": [((i*7+j*3) % 97) * 0.01 for j in range(8)],
         "props": {"grp": i % 5}} for i in range(300)]
body = json.dumps({"records": recs}).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/api/v2/collections/{cid}/records/batch",
    data=body, headers={"content-type": "application/json"}, method="POST")
print("insert:", urllib.request.urlopen(req).status)
PY
echo "== 300 records inserted; waiting for the ADR-069 drain (10s RPO floor)"
sleep 20

Q='[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]'
for i in 1 2 3; do
  curl -sf -X POST "http://127.0.0.1:$PORT/api/v2/collections/$CID/search" \
    -H 'content-type: application/json' -d "{\"vector\":$Q,\"top_k\":10}" >/dev/null
done
echo "== 3 searches done"

echo "== metric scrape:"
M=$(curl -sf "http://127.0.0.1:$PORT/metrics/prometheus" | grep "proximadb_survivor_cache_collection_reads_total" | grep -v '^#' || true)
echo "$M"
if echo "$M" | grep -q "collection_id=\"$CIDS\""; then
  echo "SMOKE PASS: per-collection series present for $CID (id $CIDS)"
  echo "$M" | grep "collection_id=\"$CIDS\"" | grep -v '} 0$' | head -3 \
    || { echo "  (all zero — check flush/survivor-cache enablement)"; exit 2; }
  echo "SMOKE PASS: non-zero counts recorded"
else
  echo "SMOKE FAIL: no per-collection series for canonical id $CIDS"
  exit 1
fi
