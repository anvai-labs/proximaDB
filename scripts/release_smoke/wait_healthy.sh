#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Wait for a ProximaDB server's /health to answer, then assert it reports the
# expected version -- a server that is up but is not the release under test is
# a failure, not a pass.
#
# Usage: wait_healthy.sh <base_url> <expected_version> [timeout_seconds]
set -euo pipefail
BASE="${1:?base url}"; WANT="${2:?version}"; TIMEOUT="${3:-120}"
deadline=$(( $(date +%s) + TIMEOUT ))
until body="$(curl -fsS --max-time 5 "${BASE}/health" 2>/dev/null)"; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "::error::${BASE}/health did not answer within ${TIMEOUT}s"
    exit 1
  fi
  sleep 2
done
echo "health: ${body}"
got="$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("version",""))')"
status="$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("status",""))')"
if [ "$got" != "$WANT" ] || [ "$status" != "healthy" ]; then
  echo "::error::${BASE}/health reports status=${status} version=${got}; expected healthy ${WANT}"
  exit 1
fi
echo "OK   server healthy at ${BASE}, version ${got}"
