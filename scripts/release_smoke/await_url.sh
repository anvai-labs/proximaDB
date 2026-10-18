#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Wait (bounded) until a registry serves a URL whose body contains a needle.
#
# Registries propagate a fresh publish at their own pace (PyPI's JSON API and
# CDN, the npm packument, the crates.io index, Docker Hub tags). A single probe
# right after a publish reports "missing" for a version that is merely not
# visible yet, so every registry probe in release-smoke.yml goes through here:
# poll until the body contains NEEDLE, fail hard after MINUTES.
#
# Usage: await_url.sh <label> <url> <needle> <minutes> [poll_seconds]
# An empty needle means "HTTP 2xx is enough".
set -euo pipefail

LABEL="${1:?label}"
URL="${2:?url}"
NEEDLE="${3-}"
MINUTES="${4:?minutes}"
POLL="${5:-30}"
# crates.io rejects requests without a User-Agent, which reads like "missing".
UA='proximadb-release-smoke (+https://github.com/anvai-labs/proximaDB)'

deadline=$(( $(date +%s) + MINUTES * 60 ))
attempt=0
while :; do
  attempt=$((attempt + 1))
  if body="$(curl -fsSL --max-time 30 -A "$UA" "$URL" 2>/dev/null)"; then
    if [ -z "$NEEDLE" ] || printf '%s' "$body" | grep -qF -- "$NEEDLE"; then
      echo "OK   ${LABEL}: live (attempt ${attempt})"
      exit 0
    fi
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "::error::${LABEL}: not served after ${MINUTES} min (${URL})"
    exit 1
  fi
  echo "...  ${LABEL}: not visible yet (attempt ${attempt}); retrying in ${POLL}s"
  sleep "$POLL"
done
