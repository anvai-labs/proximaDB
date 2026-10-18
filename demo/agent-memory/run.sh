#!/usr/bin/env bash
# THE one command for the agent-memory demo:
#   models (if missing) → docker engine (build if missing) → seed/search/write/read-back
#
# Usage: ./run.sh            (engine stays up afterwards — ./run.sh down to stop)
#        ./run.sh down       (teardown: stop + remove volumes)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "${HERE}"

if [ "${1:-}" = "down" ]; then
  docker compose down -v
  exit 0
fi

echo "▶ step 1/3 — embedding model (BGE-small ONNX, ~133MB first run)"
./fetch-models.sh

echo "▶ step 2/3 — engine (docker build on first run: proximadb-server --features onnx)"
docker compose up -d --build

echo "▶ step 3/3 — waiting for health"
for i in $(seq 1 60); do
  if curl -sf http://localhost:5678/health >/dev/null 2>&1; then echo "   healthy"; break; fi
  [ "$i" = 60 ] && { echo "engine never became healthy — see: docker compose logs"; exit 1; }
  sleep 2
done

python3 seed_and_query.py http://localhost:5678

cat <<'EOF'

Engine stays up on http://localhost:5678 (REST v2) — try your own queries:
  curl -s -X POST localhost:5678/api/v2/hybrid/search \
    -H 'content-type: application/json' \
    -d '{"collection":"agent_memory_demo","text_query":"kafka lag","top_k":3}' | jq

Teardown: ./run.sh down
EOF
