#!/usr/bin/env python3
"""Duplication ratchet guard (redundancy pass 3 follow-up).

Runs the jscpd census (config: .jscpd.json — the same one `make dup-report`
uses) and compares the duplication percentage against a committed baseline.
Fails when duplication rises above baseline + tolerance, so the number can
only go down (or the baseline must be deliberately refreshed with a
justifying PR). Modeled on the v1-proto-usage baseline guards.

Modes:
    --mode check           compare against the baseline (CI gate)
    --mode report          print the current numbers (requires a baseline to exist)
    --mode update-baseline rewrite the baseline file with current numbers
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Clone counts can wobble by a handful on trivial diffs (comment moves,
# renames); require a real regression before failing.
TOLERANCE_PERCENTAGE_POINTS = 0.05

REPORT = Path("target/jscpd-report/jscpd-report.json")
BASELINE = Path("docs/10-quality/DUP_RATCHET_BASELINE.json")


def run_census() -> dict:
    # `make dup-report` owns the census flags (b0b0b7696 keeps them inline
    # there); reuse it rather than maintaining a second flag list.
    result = subprocess.run(
        ["make", "-s", "dup-report"], capture_output=True, text=True
    )
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        sys.exit(f"make dup-report failed with exit code {result.returncode}")
    if not REPORT.exists():
        sys.exit(f"expected jscpd JSON at {REPORT} (check the dup-report target)")
    return json.loads(REPORT.read_text())["statistics"]["total"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["check", "report", "update-baseline"], required=True)
    args = parser.parse_args()

    total = run_census()
    pct = total["percentage"]
    summary = (
        f"duplication {pct:.3f}%  clones={total['clones']}  "
        f"dup-lines={total['duplicatedLines']}  sources={total['sources']}"
    )

    if args.mode == "update-baseline":
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        BASELINE.write_text(
            json.dumps(
                {
                    "percentage": round(pct, 4),
                    "clones": total["clones"],
                    "duplicated_lines": total["duplicatedLines"],
                    "recorded_at_commit": sha,
                    "note": "Refresh with `make dup-ratchet-baseline` only alongside a "
                    "justification for the increase (see docs/10-quality/"
                    "DUPLICATION_CENSUS_2026_10_20.md for methodology).",
                },
                indent=2,
            )
            + "\n"
        )
        print(f"baseline updated: {summary} ({BASELINE})")
        return

    if not BASELINE.exists():
        sys.exit(f"missing baseline {BASELINE}; run `make dup-ratchet-baseline` first")

    baseline_pct = json.loads(BASELINE.read_text())["percentage"]
    limit = baseline_pct + TOLERANCE_PERCENTAGE_POINTS
    print(f"baseline {baseline_pct:.3f}%  limit {limit:.3f}%  current {summary}")

    if args.mode == "check" and pct > limit:
        sys.exit(
            f"duplication ratchet: {pct:.3f}% exceeds the baseline limit {limit:.3f}%.\n"
            "Refactor the introduced duplication, or (if the increase is deliberate "
            "and justified) refresh the baseline via `make dup-ratchet-baseline` "
            "in the same PR."
        )
    print("OK" if args.mode == "check" else "(report-only)")


if __name__ == "__main__":
    main()
