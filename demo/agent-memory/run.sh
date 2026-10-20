#!/usr/bin/env bash
# THE one command for the agent-memory demo:
#   models (if missing) → docker engine (build if missing) → seed/search/write/read-back
#
# Usage: ./run.sh            (engine stays up afterwards — ./run.sh down to stop)
#        ./run.sh down       (stop the engine; demo data PERSISTS in ./data —
#                            the volume is a host bind mount, not a named
#                            volume, so compose cannot remove it)
#        ./run.sh reset      (down + wipe ./data + wipe ./models: full clean slate)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "${HERE}"
CONTAINER="proximadb-agent-memory"

if [ "${1:-}" = "down" ]; then
  docker compose down -v
  echo "engine stopped. Demo data persists in ./data (rm -rf data, or ./run.sh reset, for a clean slate)."
  exit 0
fi
if [ "${1:-}" = "reset" ]; then
  docker compose down -v || true
  rm -rf ./data ./models
  echo "clean slate: engine stopped, ./data and ./models removed."
  exit 0
fi

echo "▶ step 1/3 — embedding model (BGE-small ONNX, ~133MB first run)"
./fetch-models.sh

echo "▶ step 2/3 — engine (docker build on first run: proximadb-server --features onnx)"
docker compose up -d --build

echo "▶ step 3/3 — waiting for the DEMO container to be healthy"
# Assert the container itself is healthy BEFORE trusting localhost:5678 —
# a foreign engine already bound to the host port would otherwise pass a
# bare curl while the demo container crash-loops on the port conflict.
for i in $(seq 1 60); do
  if [ "$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}}' "${CONTAINER}" 2>/dev/null)" = "healthy" ]; then
    break
  fi
  if [ "$i" = 60 ]; then
    echo "demo container never became healthy — check: docker compose logs"
    exit 1
  fi
  sleep 2
done
curl -sf http://localhost:5678/health >/dev/null

python3 seed_and_query.py http://localhost:5678

cat <<'EOF'

Engine stays up on http://localhost:5678 (loopback only, REST v2) — try your own queries:
  curl -s -X POST localhost:5678/api/v2/hybrid/search \
    -H 'content-type: application/json' \
    -d '{"collection":"agent_memory_demo","text_query":"kafka lag","top_k":3}' | jq

Teardown: ./run.sh down (keeps data) · ./run.sh reset (clean slate)
EOF
