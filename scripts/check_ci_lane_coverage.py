#!/usr/bin/env python3
"""Every integration test file must be reachable from CI, or explicitly waived.

Release-review F2 (RELEASE_REVIEW_v0.4_to_v0.5.1.md): five flagship regression
e2e files were compiled by CI (`rust-compile (test)`) but executed by NO lane
in ANY workflow — the read-after-write, ON CONFLICT, declared-key and
extended-Describe fixes they pin were not ratcheted. A fix whose test never
runs is a fix that silently regresses.

Contract:
  * Every `tests/*.rs` (top-level integration targets only; `tests/common/`
    and `tests/scripts/` are modules, not targets) must be EXECUTED by some
    `.github/workflows/*.yml` — referenced inside a `run:` block as a
    `--test <name>` invocation, a bare for-loop name, a literal
    `tests/<name>.rs`, or a matching execution glob such as `tests/*wal*.rs`
    — OR appear in the allowlist below. Path FILTERS and compile-step globs
    (`tests/**`) do not count: a test that only triggers a lane without
    running in it is exactly the F2 shape.
  * Tests not yet executed enter a committed BASELINE
    (`scripts/ci_lane_baseline.txt`, one file name per line) that may only
    shrink: a NEW uncovered test fails the check, and a baseline entry that
    has since been covered also fails (regenerate with
    `--update-baseline`). This is the house ratchet pattern
    (`docs/10-quality/DUP_RATCHET_BASELINE.json`): the observed gap becomes
    explicit and monotone instead of silently growing. Burning the baseline
    down — wiring a test into a lane, or waiving it forever with a reason —
    is ordinary PR work.

Scope note: coverage is decided from `run:` blocks only (the execution
context), so a filter-only reference correctly fails. What this does NOT see
is whether the lane's `if:` gate actually fired for a given PR — that is a
triggering question, not an execution question.

Exit 0 = covered or baseline-consistent; 1 = uncovered new files or stale
baseline; 2 = usage error.
"""

from __future__ import annotations

import re
import sys
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
BASELINE = ROOT / "scripts" / "ci_lane_baseline.txt"

# Module directories under tests/ that are #[path]-included helpers, not targets.
NON_TARGET_DIRS = {"common", "scripts"}


def integration_targets() -> list[Path]:
    return sorted(
        p
        for p in TESTS_DIR.glob("*.rs")
        if p.is_file() and p.stem not in NON_TARGET_DIRS
    )


def run_block_text() -> str:
    """Concatenate `run:` block contents across workflows, comment-stripped,
    plus the bodies of repo scripts those blocks invoke.

    Coverage must mean EXECUTION. Three consequences:

    * Path filters (`changes:`/`filters:` lists) and compile-step globs
      (`tests/**`) reference files without running them, so they are never
      read.
    * Shell comments cannot certify: a run block saying
      `# FIXME: wire tests/alpha_e2e.rs into a lane` is the natural lazy
      non-fix, so each line is cut at its first `#`.
    * Execution routed through a repo script counts: a block running
      `bash scripts/run_cloud_emulator_tests.sh` executes whatever
      `--test` targets that script names, so invoked script bodies are
      appended to the search text (adversarial-review round 1 found
      `objstore_backend_contract_test.rs` certified exactly that way while
      the guard called it unexecuted).
    """
    chunks: list[str] = []
    invoked: set[Path] = set()

    def harvest_script_invocations(text: str) -> None:
        for match in re.finditer(r"(?:bash |sh |python3 )([^\s;|&]+\.sh\b)", text):
            candidate = ROOT / match.group(1).lstrip("./")
            if candidate.is_file():
                invoked.add(candidate)

    for path in sorted(WORKFLOWS_DIR.glob("*.yml")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not (stripped == "run: |" or stripped.startswith("run:")):
                continue
            if stripped == "run: |":
                indent = len(line) - len(line.lstrip())
                block: list[str] = []
                for follow in lines[index + 1:]:
                    if follow.strip() and len(follow) - len(follow.lstrip()) <= indent:
                        break
                    block.append(follow)
            else:
                block = [stripped[len("run:"):]]
            # Cut shell comments: nothing after '#' may certify coverage.
            block = [piece.split("#", 1)[0] for piece in block]
            harvest_script_invocations("\n".join(block))
            chunks.extend(block)
    for script in sorted(invoked):
        chunks.append(script.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def baseline_names() -> set[str]:
    if not BASELINE.is_file():
        return set()
    return {
        line.split("#", 1)[0].strip()
        for line in BASELINE.read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    }


# `for <var> in <words...>; do` on a single line; group 1 = the word list.
INLINE_FOR_RE = re.compile(r"\bfor\s+\w+\s+in\s+([^;\n]*);\s*do\b")


def is_covered(stem: str, file_name: str, text: str, patterns: list[str]) -> bool:
    """Covered = any reference anywhere in any workflow.

    Checks, in order: a `--test <stem>` invocation, the bare stem as a
    for-loop word, the literal file path, and finally glob expansion.
    """
    if re.search(rf"--test[= ]{re.escape(stem)}\b", text):
        return True
    # Bare for-loop word: alone, with a line-continuation backslash, or as
    # the final entry before `; do` / a comma.
    if re.search(
        rf"^\s*{re.escape(stem)}\s*(?:\\\s*|;\s*do\s*|,\s*)?$", text, re.M
    ):
        return True
    # Inline for-loop list: `for t in a b c; do` on one line. The bare-word
    # form above only sees a stem ALONE on its line, so an inline list was
    # invisible and its targets were baselined as never-run while a lane ran
    # them (#2502 review: the api-surface lane's two targets).
    for words in INLINE_FOR_RE.findall(text):
        if stem in words.split():
            return True
    if f"tests/{file_name}" in text:
        return True
    return any(fnmatch(f"tests/{file_name}", pat) for pat in patterns)


def main() -> int:
    if "--help" in sys.argv[1:]:
        print(__doc__)
        return 2
    update = "--update-baseline" in sys.argv[1:]
    targets = integration_targets()
    if not targets:
        print("check_ci_lane_coverage: no integration targets found under tests/", file=sys.stderr)
        return 2
    text = run_block_text()
    # Execution globs inside run blocks, e.g. tests/*wal*.rs in a for-loop.
    patterns = sorted(set(re.findall(r"(tests/[^\s'\";)]+)", text)))
    baseline = baseline_names()

    uncovered: set[str] = set()
    for path in targets:
        if not is_covered(path.stem, path.name, text, patterns):
            uncovered.add(path.name)

    if update:
        BASELINE.write_text("\n".join(sorted(uncovered)) + "\n", encoding="utf-8")
        print(f"baseline updated: {len(uncovered)} uncovered integration tests recorded")
        return 0

    new_uncovered = sorted(uncovered - baseline)
    stale = sorted(baseline - uncovered)
    if new_uncovered:
        print("CI lane coverage check failed — NEW integration tests no lane executes:", file=sys.stderr)
        for name in new_uncovered:
            print(f"  tests/{name}", file=sys.stderr)
        print(
            "\nWire the test into a lane's run list in .github/workflows/ci.yml (and its\n"
            "path filter), or record it in the baseline with --update-baseline only if it\n"
            "is deliberately local (say why in the PR).",
            file=sys.stderr,
        )
        return 1
    if stale:
        print("CI lane coverage check failed — baseline entries now covered (shrink it):", file=sys.stderr)
        for name in stale:
            print(f"  {name}", file=sys.stderr)
        print(f"\nRegenerate with: python3 scripts/check_ci_lane_coverage.py --update-baseline", file=sys.stderr)
        return 1
    covered = len(targets) - len(uncovered)
    print(
        f"CI lane coverage check passed: {covered}/{len(targets)} integration targets "
        f"executed by a lane; {len(uncovered)} on the burn-down baseline."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
