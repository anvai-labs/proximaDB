#!/bin/bash
# TD-CACHEC-1 smoke: per-collection survivor-cache attribution end-to-end
# (release-server boot -> REST create -> insert -> ADR-069 driver flush ->
# search -> scrape /metrics/prometheus). PASS = a
# proximadb_survivor_cache_collection_reads_total series exists for the
# smoke collection with non-zero counts. See TD-CACHEC-1 for the outcome
# semantics (first touch = cold; warm queries are served upstream of
# get_or_fetch and record nothing). Requires: curl, python3, cargo.
set -euo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT=5691
DATA=$(mktemp -d /tmp/cachec-smoke-data.XXXX)
CFG=$(mktemp /tmp/cachec-smoke-cfg.XXXX.toml)
# fast flush so the cascade sees segments without waiting 60s
sed 's/^# flush_interval_s = 60/flush_interval_s = 10/; s/^flush_interval_s = 60/flush_interval_s = 10/; s/^port = 5678/port = 5691/; s/^memory_flush_size_bytes = 16777216/memory_flush_size_bytes = 16777216\nflush_interval_secs = 10/; s/^unified_mode = true/unified_mode = true\nunified_port = 5691/' \
  "$WT/config/config.toml" > "$CFG"

echo "== building release-server (if stale)"
CARGO_INCREMENTAL=0 cargo build --profile release-server --manifest-path "$WT/Cargo.toml" -p proximadb-server --bin proximadb-server

echo "== starting server on :$PORT (data: $DATA)"
RUST_LOG=info "$WT/target/release-server/proximadb-server" -c "$CFG" -d "$DATA" -p $PORT \
  > /tmp/cachec-smoke-server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT

for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health/live" >/dev/null 2>&1 && break
  sleep 1
done
curl -sf "http://127.0.0.1:$PORT/health/live" >/dev/null || { echo "server did not start"; tail -20 /tmp/cachec-smoke-server.log; exit 1; }
echo "== server up"

CID="smoke_cachec_$(date +%s)"
curl -sf -X POST "http://127.0.0.1:$PORT/api/v2/collections" -H 'content-type: application/json' -d \
  "{\"name\":\"$CID\",\"dimension\":8,\"engine\":\"sst\",\"enable_proxima_record\":true}" >/dev/null
echo "== collection $CID created"

python3 -I - "$CID" <<'PY'
import json, sys, urllib.request
cid = sys.argv[1]
recs = [{"id": f"r_{i}",
         "vector": [((i*7+j*3) % 97) * 0.01 for j in range(8)],
         "props": {"grp": i % 5}} for i in range(300)]
body = json.dumps({"records": recs}).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:5691/api/v2/collections/{cid}/records/batch",
    data=body, headers={"content-type": "application/json"}, method="POST")
print(urllib.request.urlopen(req).status)
PY
echo "== 300 records inserted; waiting for flush_interval"
sleep 15

Q='[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]'
for i in 1 2 3; do
  curl -sf -X POST "http://127.0.0.1:$PORT/api/v2/collections/$CID/search" \
    -H 'content-type: application/json' -d "{\"vector\":$Q,\"top_k\":10}" >/dev/null
done
echo "== 3 searches done"

echo "== metric scrape:"
M=$(curl -sf "http://127.0.0.1:$PORT/metrics/prometheus" | grep "proximadb_survivor_cache_collection_reads_total" | grep -v '^#' || true)
echo "$M"
if echo "$M" | grep -q "$CID"; then
  echo "SMOKE PASS: per-collection series present for $CID"
  echo "$M" | grep "$CID" | grep -v '} 0$' | head -3 || { echo "  (all zero — check flush/survivor-cache enablement)"; exit 2; }
  echo "SMOKE PASS: non-zero counts recorded"
else
  echo "SMOKE FAIL: no per-collection series for $CID"
  exit 1
fi
