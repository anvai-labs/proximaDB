#!/usr/bin/env bash
# ADR-094 ratchet: crates→root import count.
#
# Extracted crates must not reach back into the root monolith (`use
# proximadb::…`). The only sanctioned consumers are the embedded-binding
# crate files that intentionally compose the full engine for the Python/Spark
# surfaces — everything else is decomposition regression.
#
# Baseline (2026-10-04): 7 across 4 files, all in
# crates/binding/proximadb-embedded/ (the sanctioned in-process composition).

set -euo pipefail
cd "$(dirname "$0")/.."

CEILING="${CRATES_ROOT_IMPORTS_CEILING:-7}"

count=$(grep -rh "^use proximadb::" crates/ --include="*.rs" 2>/dev/null | wc -l | tr -d ' ')

echo "crates→root import ratchet: $count / ceiling $CEILING"
if [ "$count" -gt "$CEILING" ]; then
  echo "FAIL: extracted crates gained root-monomolith imports ($count > $CEILING)." >&2
  echo "Moved code must keep its dependencies inverted (ports, foundation types)." >&2
  grep -rn "^use proximadb::" crates/ --include="*.rs" >&2
  exit 1
fi

echo "OK"
