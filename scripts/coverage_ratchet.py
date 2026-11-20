#!/usr/bin/env python3
"""Coverage ratchet gate for the qa branch.

Enforces a NO-REGRESSION policy with an 80% target across components (Rust crates
and the Python SDK):

  - FAIL if any component's line coverage drops more than TOLERANCE below its
    committed baseline (a regression).
  - FAIL if a NEW component lands below the 80% target.
  - FAIL if a component REQUIRED for this invocation is absent from the report
    (`--require-components`) — an absent component is not a pass (the
    2026-11-20 CI review reproduced the false-green: a current report of `{}`
    exited zero against a non-empty baseline).
  - FAIL on an EMPTY or unexpectedly partial current report.
  - WARN (non-fatal) for existing components still under 80% — they ratchet up
    over time; regressions are what block.

Evidence rules (same review):
  * `normalize` REJECTS malformed or empty tool output instead of emitting an
    empty map: an llvm-cov report with no files, a coverage JSON with no
    totals, or a zero denominator all exit 2. A coverage number computed from
    zero executed tests is not evidence of coverage.
  * Rust crate attribution is repo-relative: paths are made relative to
    `--workspace-root` (default `.`) BEFORE crate extraction, so
    `apps/proximadb-server/src` and `clients/rust/src` map to their own
    crates instead of being swallowed into the root bucket by the old
    "any path containing src/" heuristic.

Usage:
  # compute a normalized {component: pct} map from raw tool output:
  coverage_ratchet.py normalize --rust llvm-cov.json --sdk sdk-cov.json -o current.json

  # check current against the committed baseline, requiring the Python
  # component be present in THIS report:
  coverage_ratchet.py check --baseline docs/_internal/roadmap/COVERAGE_BASELINE.json \
      --current current.json --require-components sdk:proximadb-python

  # (re)write the baseline from current (after an intentional change):
  coverage_ratchet.py write --current current.json \
      --baseline docs/_internal/roadmap/COVERAGE_BASELINE.json

Exit codes: 0 = pass, 1 = gate failure (regression / missing required
component / empty report), 2 = malformed input or usage error.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TARGET = 80.0  # coverage goal (%)
TOLERANCE = 0.5  # allowed downward drift before it's a regression (%)


class MalformedReport(Exception):
    """Tool output that cannot back a coverage claim."""


def _pct(covered: int, total: int) -> float:
    return 100.0 * covered / total if total else 100.0


def _relative(path: str, root: Path) -> str | None:
    """Make `path` repo-relative so crate attribution cannot be fooled by the
    checkout prefix (llvm-cov emits absolute paths like
    /home/runner/work/proximaDB/proximaDB/src/... — under the old heuristic the
    `proximaDB/src` prefix itself matched "dir/src" and every root-crate line
    was attributed to a crate named after the checkout directory). Returns
    None for files OUTSIDE the workspace: attributing a registry/vendored
    path by its `X/src/` segments would mint a spurious component (e.g.
    `rust:registry`) whose below-target coverage fails the gate as a phantom
    "new component" — so such files are skipped entirely."""
    p = Path(path)
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return None


def normalize_rust(path: Path, workspace_root: Path = Path(".")) -> dict[str, float]:
    """Parse `cargo llvm-cov report --json` into {crate: line_pct}.

    llvm-cov json: {"data":[{"files":[{"filename":..., "summary":{"lines":{"count","covered"}}}]}]}
    Files are grouped by workspace crate: `crates/<tier>/<crate>/src/...` maps
    to `<crate>`, `<crate-dir>/src/...` (apps/, clients/rust/, ...) maps to
    `<crate-dir>`, and repo-root `src/...` maps to the root `proximadb` crate.
    """
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise MalformedReport(f"{path}: not valid JSON ({e})") from e
    except OSError as e:
        raise MalformedReport(f"{path}: unreadable ({e})") from e
    blocks = doc.get("data") if isinstance(doc, dict) else None
    if not isinstance(blocks, list) or not blocks:
        raise MalformedReport(f"{path}: llvm-cov report has no `data` blocks")
    agg: dict[str, list[int]] = {}  # crate -> [covered, total]
    total_files = 0
    for block in blocks:
        for f in block.get("files", []):
            name = f.get("filename", "")
            lines = f.get("summary", {}).get("lines", {})
            covered = int(lines.get("covered", 0))
            total = int(lines.get("count", 0))
            rel = _relative(name, workspace_root)
            if rel is None:
                continue  # outside the workspace: not any component's line
            total_files += 1
            crate = _crate_of(rel)
            if crate is None or total == 0:
                continue
            slot = agg.setdefault(crate, [0, 0])
            slot[0] += covered
            slot[1] += total
    if total_files == 0:
        raise MalformedReport(f"{path}: llvm-cov report lists no files")
    if not agg:
        raise MalformedReport(
            f"{path}: no file in the report attributed to a workspace crate"
        )
    return {f"rust:{k}": round(_pct(v[0], v[1]), 2) for k, v in agg.items()}


_CRATE_DIR_SRC_RE = re.compile(r"(?:^|/)([A-Za-z0-9_][A-Za-z0-9_-]*)/src(?:/|$)")


def _crate_of(rel_path: str) -> str | None:
    """Crate owning `rel_path` (already repo-relative), or None to skip.

    Precedence: crates/<tier>/<crate>/src -> <crate>; <dir>/src -> <dir>
    (apps/proximadb-server, clients/rust, ...); bare src/ at the repo root ->
    the root `proximadb` crate. The root bucket therefore only receives the
    root monolith — apps/*/src and clients/rust/src keep their own components
    if the scope ever expands to them (2026-11-20 CI review finding).
    """
    parts = rel_path.split("/")
    if "crates" in parts:
        i = parts.index("crates")
        # crates/<tier>/<crate>/src/...
        if len(parts) > i + 2:
            return parts[i + 2]
    m = _CRATE_DIR_SRC_RE.search(rel_path)
    if m:
        return m.group(1)
    if re.match(r"^src(?:/|$)", rel_path):
        return "proximadb"
    return None


def normalize_sdk(path: Path) -> dict[str, float]:
    """Parse `coverage json` (pytest-cov) into {sdk: line_pct}.

    Rejects reports with no totals or a zero denominator: a coverage JSON
    produced by a run that executed no measurable code must fail loudly, not
    normalize to a number.
    """
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise MalformedReport(f"{path}: not valid JSON ({e})") from e
    except OSError as e:
        raise MalformedReport(f"{path}: unreadable ({e})") from e
    if not isinstance(doc, dict):
        raise MalformedReport(f"{path}: coverage JSON is not an object")
    totals = doc.get("totals")
    if not isinstance(totals, dict):
        raise MalformedReport(f"{path}: coverage JSON has no `totals` block")
    pct = totals.get("percent_covered")
    covered = totals.get("covered_lines")
    missing = totals.get("missing_lines")
    if pct is None:
        if covered is None or missing is None:
            raise MalformedReport(
                f"{path}: coverage totals carry neither percent_covered nor "
                "covered_lines/missing_lines"
            )
        total = int(covered) + int(missing)
        if total == 0:
            raise MalformedReport(
                f"{path}: zero coverage denominator — no measurable lines were "
                "executed; this report is not evidence of coverage"
            )
        pct = _pct(int(covered), total)
    return {"sdk:proximadb-python": round(float(pct), 2)}


def _load_map(path: Path, what: str) -> dict[str, float]:
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise MalformedReport(f"{path}: {what} is not valid JSON ({e})") from e
    except OSError as e:
        raise MalformedReport(f"{path}: {what} unreadable ({e})") from e
    if not isinstance(doc, dict) or not all(
        isinstance(v, (int, float)) for v in doc.values()
    ):
        raise MalformedReport(f"{path}: {what} must be a {{component: pct}} object")
    return doc


def cmd_normalize(args) -> int:
    if not args.rust and not args.sdk:
        print("normalize: nothing to normalize — pass --rust and/or --sdk", file=sys.stderr)
        return 2
    current: dict[str, float] = {}
    try:
        if args.rust:
            current.update(normalize_rust(Path(args.rust), Path(args.workspace_root)))
        if args.sdk:
            current.update(normalize_sdk(Path(args.sdk)))
    except MalformedReport as e:
        print(f"coverage normalize: {e}", file=sys.stderr)
        return 2
    Path(args.out).write_text(json.dumps(dict(sorted(current.items())), indent=2) + "\n")
    print(f"wrote {len(current)} components -> {args.out}")
    return 0


def cmd_write(args) -> int:
    current = _load_map(Path(args.current), "current")
    if not current:
        print("refusing to write an EMPTY baseline — nothing was measured", file=sys.stderr)
        return 2
    Path(args.baseline).write_text(json.dumps(dict(sorted(current.items())), indent=2) + "\n")
    print(f"baseline written: {len(current)} components -> {args.baseline}")
    return 0


def cmd_check(args) -> int:
    try:
        baseline = _load_map(Path(args.baseline), "baseline")
        current = _load_map(Path(args.current), "current")
    except MalformedReport as e:
        print(f"coverage check: {e}", file=sys.stderr)
        return 2

    required = [c.strip() for c in (args.require_components or "").split(",") if c.strip()]
    regressions, new_below, missing_required = [], [], []
    missing = sorted(set(baseline) - set(current))
    warnings = []

    # An EMPTY current report is a measurement failure, not a pass: every
    # baseline component would be "absent" and nothing could regress.
    if not current:
        print("coverage check: current report is EMPTY — no coverage was measured (FAIL)")
        print("\nqa coverage gate: FAIL")
        return 1
    for comp in required:
        if comp not in current:
            missing_required.append(comp)
    for comp, cur in sorted(current.items()):
        base = baseline.get(comp)
        if base is None:
            if cur + 1e-9 < TARGET:
                new_below.append((comp, cur))
            continue
        if cur + TOLERANCE < base:
            regressions.append((comp, base, cur))
        elif cur + 1e-9 < TARGET:
            warnings.append((comp, cur))

    fail = bool(regressions or new_below or missing_required)
    print(f"=== Coverage ratchet (target {TARGET}%, tolerance {TOLERANCE}%) ===")
    print(f"  components: {len(current)} current / {len(baseline)} baseline")
    if warnings:
        print(f"  below-target (warn, {len(warnings)}):")
        for c, p in warnings[:40]:
            print(f"    - {c}: {p}%")
    if missing:
        # Baseline components absent from THIS report: if the invocation's
        # scope legitimately excludes them, pass --require-components with the
        # components this run DOES cover — absence only warns for unrequired
        # components, and the gate refuses a report containing none of them.
        print(f"  WARNING: {len(missing)} baseline component(s) absent from current: {missing[:10]}")
    if missing_required:
        print(f"  REQUIRED component(s) absent from current (FAIL):")
        for c in missing_required:
            print(f"    - {c}")
    if new_below:
        print(f"  NEW component(s) below {TARGET}% (FAIL):")
        for c, p in new_below:
            print(f"    - {c}: {p}% (< {TARGET}%)")
    if regressions:
        print(f"  REGRESSIONS (FAIL):")
        for c, b, cur in regressions:
            print(f"    - {c}: {b}% -> {cur}% (drop {round(b - cur, 2)}%)")

    if fail:
        print("\nqa coverage gate: FAIL")
        return 1
    print("\nqa coverage gate: PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    n = sub.add_parser("normalize", help="normalize raw tool output -> {component: pct}")
    n.add_argument("--rust", help="cargo llvm-cov report --json output")
    n.add_argument(
        "--workspace-root",
        default=".",
        help="repo root used to make llvm-cov paths repo-relative (default: cwd)",
    )
    n.add_argument("--sdk", help="pytest coverage json output")
    n.add_argument("-o", "--out", required=True)
    n.set_defaults(func=cmd_normalize)

    c = sub.add_parser("check", help="check current vs baseline (ratchet)")
    c.add_argument("--baseline", required=True)
    c.add_argument("--current", required=True)
    c.add_argument(
        "--require-components",
        default="",
        help="comma-separated components this invocation MUST find in the "
        "current report; an absent one fails the gate instead of warning",
    )
    c.set_defaults(func=cmd_check)

    w = sub.add_parser("write", help="write baseline from current")
    w.add_argument("--current", required=True)
    w.add_argument("--baseline", required=True)
    w.set_defaults(func=cmd_write)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
