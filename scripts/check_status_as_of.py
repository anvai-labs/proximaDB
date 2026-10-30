#!/usr/bin/env python3
"""Guard against status-of-record docs drifting stale behind the code.

The docs-claim guard greps for specific marketing phrases; nothing ties the
*status-of-record* docs (the ones that present current-state tables as the
authoritative baseline) to a freshness contract. That blind spot let
`SYSTEM_MAP` describe DataFusion as "all-Parquet-only" weeks after
`DataFusionLocal` routed live on pgwire — and an external architecture analysis
cited the stale table as current state (see TD-DOCS-2).

Contract: every doc in STATUS_OF_RECORD_DOCS carries an AsciiDoc comment tag

    // status-as-of: YYYY-MM-DD

meaning "the status/current-state sections were re-verified against HEAD on
this date". The tag is bumped ONLY after re-verifying — never mechanically.

What it checks (all hard errors):
  1. Every allowlisted doc exists and carries exactly one status-as-of tag.
  2. The tag parses as an ISO date, is not in the future, and is not older
     than --max-age-days (default 60).

Exit 0 = fresh; 1 = missing/stale/invalid tag; 2 = usage error.

Filing convention for the drift class: docs/10-quality/td/TD-DOCS-2-*.adoc.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date, datetime
from pathlib import Path

# Docs that present themselves as the current-state baseline. Extend this list
# when a doc graduates into a status-of-record role (see TD-DOCS-2).
STATUS_OF_RECORD_DOCS = [
    "docs/current/TRACKER.adoc",
    "docs/12-design/SYSTEM_MAP_2026_05_30.adoc",
    "docs/12-design/adr/README.adoc",
    "docs/SUPPORTED_SURFACE.adoc",
]

TAG_RE = re.compile(r"^//\s*status-as-of:\s*(\S+)\s*$", re.MULTILINE)

DEFAULT_MAX_AGE_DAYS = 60

REMEDY = (
    "  → Re-verify the doc's status/current-state sections against HEAD "
    "(routing, feature flags, shipped slices), fix any drift, then bump the "
    "tag to today: // status-as-of: {today}"
)


def check_doc(path: Path, max_age_days: int, today: date) -> list[str]:
    """Return a list of violation messages for one doc (empty = fresh)."""
    if not path.is_file():
        return [f"{path}: status-of-record doc is missing (allowlisted in {__file__})"]

    tags = TAG_RE.findall(path.read_text(encoding="utf-8"))
    if not tags:
        return [f"{path}: no '// status-as-of: YYYY-MM-DD' tag found"]
    if len(tags) > 1:
        return [f"{path}: {len(tags)} status-as-of tags found — keep exactly one"]

    try:
        tagged = datetime.strptime(tags[0], "%Y-%m-%d").date()
    except ValueError:
        return [f"{path}: unparseable status-as-of date {tags[0]!r} (want YYYY-MM-DD)"]

    if tagged > today:
        return [f"{path}: status-as-of {tagged} is in the future"]

    age = (today - tagged).days
    if age > max_age_days:
        return [
            f"{path}: status-as-of {tagged} is {age} days old (max {max_age_days})"
        ]
    return []


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Fail when a status-of-record doc's status-as-of tag is missing or stale."
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=DEFAULT_MAX_AGE_DAYS,
        help=f"maximum tag age in days (default {DEFAULT_MAX_AGE_DAYS})",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repo root (default: parent of scripts/)",
    )
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return 2
    if args.max_age_days <= 0:
        print("error: --max-age-days must be positive", file=sys.stderr)
        return 2

    today = date.today()
    violations: list[str] = []
    for rel in STATUS_OF_RECORD_DOCS:
        violations.extend(check_doc(args.root / rel, args.max_age_days, today))

    if violations:
        print("❌ status-as-of check failed:")
        for v in violations:
            print(f"  {v}")
        print(REMEDY.format(today=today.isoformat()))
        return 1

    print(f"✅ status-as-of: {len(STATUS_OF_RECORD_DOCS)} status-of-record doc(s) fresh")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
