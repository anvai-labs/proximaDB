#!/usr/bin/env bash
# OSS docs decontamination lint (BLOCKING).
#
# ProximaDB OSS ships MECHANISM + NEUTRAL meters only. Commercial POLICY
# ($-pricing rates, $-ARR/revenue targets, $-budgets/salaries) lives in anvaiops
# — see docs/12-design/OSS_ENTERPRISE_BOUNDARY_2026_06_17.adoc. This lint BLOCKS
# accidental re-introduction of commercial $-content into the public docs.
#
# Catches: $-per-unit pricing ($0.50/hour, $100/DBU) + $-denominated amounts
# ($50M, $400K — covers ARR targets, budgets, salaries). Bare demo prices
# ($500 in a use-case example) are NOT matched (no K/M/B suffix, no /unit).
#
# ALLOWLISTED (legitimately use $ as DESIGN rationale — market-size context,
# cloud cost model, the boundary definition itself): the co-design mechanism
# docs. New policy $ there is caught by review, not this lint.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

PATTERN='\$[0-9]+(\.[0-9]+)?[[:space:]]*(per[[:space:]]*)?(TB|GB|MB|hour|hr|node|month|mo|credit|DBU|query|index|replica)|\$[0-9]+(\.[0-9]+)?[[:space:]]*[KMB]\b'

# Exclude: archives (deferred) + the mechanism/boundary docs that use $ as design rationale.
ALLOWLIST='/docs/12-design/CODESIGN_DIMENSIONAL_ARCHITECTURE_2026_06_19\.adoc|/docs/12-design/OUTGRESS_KOU_MULTICLOUD_COST_MODEL_2026_06_21\.adoc|/docs/12-design/OSS_ENTERPRISE_BOUNDARY_2026_06_17\.adoc|/scripts/check_no_commercial_docs\.sh'

mapfile -t HITS < <(grep -rniE "$PATTERN" --include='*.adoc' --include='*.md' "$ROOT/docs" 2>/dev/null \
  | grep -vE "$ALLOWLIST" || true)

if [ "${#HITS[@]}" -gt 0 ]; then
  echo "::error::Commercial \$-content (pricing / ARR / budgets / salaries) found in OSS docs." >&2
  echo "ProximaDB OSS ships mechanism + neutral meters only; \$ rates/revenue/budgets belong in anvaiops." >&2
  echo "See docs/12-design/OSS_ENTERPRISE_BOUNDARY_2026_06_17.adoc. If this is design-context \$ (market-size," >&2
  echo "cloud cost model), add the file to the ALLOWLIST in scripts/check_no_commercial_docs.sh with a reason." >&2
  echo "--- hits ---" >&2
  printf '%s\n' "${HITS[@]}" >&2
  exit 1
fi
echo "ok: no commercial \$-content in OSS docs (allowlisted mechanism docs aside)"
