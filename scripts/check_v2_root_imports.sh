#!/usr/bin/env bash
# ADR-094 ratchet: root-import count for the REST v2 handler tree.
#
# The v2 handlers under src/network/rest/v2/ are destined for
# crates/platform/proximadb-api (ADR-094 REST-tree convergence). Every
# `use crate::…` import ties a handler file to a root-internal module, which
# blocks the move (platform crates must not depend on the root monolith).
#
# This ratchet pins the TOTAL count at its committed ceiling: CI fails if it
# rises. Lower the ceiling in the same PR that removes imports.
#
# Baseline (2026-10-03): 52 across 12 files. After PR-3.1b (graphs.rs port
# rewrite): 51. After PR-3.2a (schema.rs moved to proximadb-api): 48.
# After PR-3.3a (query.rs + sql.rs moved): 44.
# After PR-3.3b (timeseries.rs + model_registry.rs moved): 38.

set -euo pipefail
cd "$(dirname "$0")/.."

CEILING="${V2_ROOT_IMPORTS_CEILING:-44}"
DIR="src/network/rest/v2"

count=$(grep -rh "^use crate::" "$DIR"/*.rs 2>/dev/null | wc -l | tr -d ' ')

echo "v2 root-import ratchet: $count / ceiling $CEILING"
if [ "$count" -gt "$CEILING" ]; then
  echo "FAIL: REST v2 handlers gained root imports ($count > $CEILING)." >&2
  echo "New handler code must go through ports (proximadb-runtime) or move its" >&2
  echo "dependencies to neutral crates per ADR-094." >&2
  grep -rn "^use crate::" "$DIR"/*.rs >&2
  exit 1
fi

echo "OK"
