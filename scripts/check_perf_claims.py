#!/usr/bin/env python3
"""Cross-check narrative-doc perf claims against docs/_internal/roadmap/BENCHMARK_EVIDENCE.toml.

The evidence ledger is the authority for perf claims (SUPPORTED_SURFACE:
"only status='measured' entries carry a dated artifact; do not quote a perf
figure as validated unless its ledger entry is measured"). validate_benchmark_evidence.py
checks the ledger's *internal* consistency (refs exist, status invariants). This
guard checks the *cross-doc* contract in two directions:

  1. FORWARD (stale-ref): each claim's `asserted_in` ref into a narrative doc
     (.adoc/.md) must actually contain the claim's number on/near the cited line.
     Catches docs that drifted past their refs (the number moved/was removed but
     the ledger still points at the old line).
  2. MEASURED-ONLY: a customer-facing CAPABILITY doc (README/concepts/guides/
     quick-start/api-reference) may cite ONLY `measured` claims — `unverified` and
     `aspirational` are not citable there as current results (measure them, or drop
     the citation). PRD/VISION (requirements/roadmap) and _internal are exempt —
     targets/unverified belong there.

Exit 0 = clean, 1 = violation(s), 2 = usage error.
"""

from __future__ import annotations

import re
import sys
from fnmatch import fnmatch
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "docs" / "_internal" / "roadmap" / "BENCHMARK_EVIDENCE.toml"

# Customer-facing CAPABILITY docs: what ProximaDB does today. An aspirational
# (target) claim cited here misrepresents a future goal as a current result.
# Excluded: docs/00-product (PRD/VISION — forward-looking), docs/10-quality
# (benchmarks — reference), docs/12-design, docs/_internal, roadmap.
CAPABILITY_GLOBS = [
    "README.adoc",
    "SUPPORTED_SURFACE.md",
    "docs/README.md",
    "docs/05-concepts/*.adoc",
    "docs/05-concepts/*.md",
    "docs/02-guides/*.md",
    "docs/01-quick-start/*.md",
    "docs/01-quick-start/*.adoc",
    "docs/03-api-reference/*.adoc",
]

NUMBER_RE = re.compile(r"\d+")
WINDOW = 3  # lines of tolerance around the cited line for the forward check


def _is_capability(rel: str) -> bool:
    return any(fnmatch(rel, g) for g in CAPABILITY_GLOBS)


def parse_reference(ref: str) -> tuple[Path, int | None]:
    if ":" in ref:
        path_part, line_part = ref.rsplit(":", 1)
        if line_part.isdigit():
            return ROOT / path_part, int(line_part)
    return ROOT / ref, None


def check_ref(claim_id: str, status: str, claim: str, ref: str, errors: list[str]) -> None:
    file_path, line_no = parse_reference(ref)
    rel = ref.rsplit(":", 1)[0] if ":" in ref else ref

    # MEASURED-ONLY: only measured claims may be cited in a customer-facing
    # capability doc (unverified/aspirational are not citable as current results).
    if status != "measured" and _is_capability(rel):
        errors.append(
            f"[{claim_id}] {status!r} claim cited in capability doc {rel!r} — only "
            f"measured claims are citable in customer-facing docs. Measure it "
            f"(promote to status='measured' with an artifact) or drop the citation."
        )

    # FORWARD: only for narrative docs with a line number.
    if line_no is None or file_path.suffix not in (".adoc", ".md"):
        return
    if not file_path.exists():
        return  # validate_benchmark_evidence.py owns existence checks
    numbers = NUMBER_RE.findall(claim)
    if not numbers:
        return  # non-numeric claim (e.g. "sub-millisecond") — nothing to anchor
    try:
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    lo = max(0, line_no - 1 - WINDOW)
    hi = min(len(lines), line_no + WINDOW)
    window = "\n".join(lines[lo:hi])
    if not any(n in window for n in numbers):
        preview = lines[line_no - 1][:80] if 0 <= line_no - 1 < len(lines) else "<out of range>"
        errors.append(
            f"[{claim_id}] stale ref: {ref} does not contain the claim's number "
            f"({numbers}) within ±{WINDOW} lines — line says: {preview!r}. "
            f"Update the asserted_in line (or the doc) so the citation is accurate."
        )


def main() -> int:
    if not LEDGER_PATH.exists():
        print(f"ERROR: Missing evidence ledger: {LEDGER_PATH}")
        return 1
    try:
        data = tomllib.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        print(f"ERROR: Invalid TOML in {LEDGER_PATH}: {exc}")
        return 1

    claims = data.get("claims")
    if not isinstance(claims, list) or not claims:
        print("ERROR: BENCHMARK_EVIDENCE.toml has no [[claims]]")
        return 1

    errors: list[str] = []
    seen_ids: set[str] = set()
    checked_refs = 0
    for idx, claim in enumerate(claims, start=1):
        claim_id = claim.get("id", f"<missing-id-{idx}>")
        if claim_id in seen_ids:
            continue  # validate_benchmark_evidence.py owns the duplicate-id check
        seen_ids.add(claim_id)
        status = claim.get("status", "")
        claim_text = claim.get("claim", "")
        for ref in claim.get("asserted_in", []) or []:
            if not isinstance(ref, str):
                continue
            check_ref(claim_id, status, claim_text, ref, errors)
            checked_refs += 1

    if errors:
        print("Perf-claim cross-check failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print(f"OK: perf-claim cross-check clean — {checked_refs} doc refs across {len(seen_ids)} claims verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
