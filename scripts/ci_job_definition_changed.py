#!/usr/bin/env python3
"""Did a PR change the CI DEFINITION of a path-scoped job? (TD-CITIER-1)

A path-filtered gate must be able to see a change to itself -- otherwise a PR
that edits the gate's own `if:`/steps/filter skips the gate and reports green
("skipped is not passed", the lesson recorded in ci.yml's `rust:` filter). The
blunt way to get that is to list `.github/workflows/ci.yml` in every filter.
For the specialised heavy suites that is expensive: ci.yml was touched by 15 of
the 44 feat->develop PRs measured for TD-CITIER-1, almost never in those jobs,
and each match costs up to a 66-minute job on the critical path.

This script answers the precise question instead: for each `JOB=FILTER` pair,
did any text that defines that job change between the base and the working
tree? "Defines" means:

  * the job's own block (`  <job>:` up to the next top-level job key);
  * the whole `changes` job EXCEPT the bodies of `filters: |` block scalars:
    every output line (a redefined `test_tier`/`medium_tier` counts), every
    step, the paths-filter step's other `with:` keys (`predicate-quantifier:
    every` changes every filter's meaning), and the selfdef step's own
    `--job` list (deleting a pair would otherwise disarm this very check);
  * the filter block of every `changes` output the job reads
    (`needs.changes.outputs.<X>`, plus FILTER itself), with its globs, so a
    narrowed filter counts while edits to OTHER filters' globs do not.
  * the workflow header before `jobs:` (`on:`, `concurrency:`, global `env:`).

Comments count: the comparison is textual, so it errs towards running.

Output: one `<FILTER>=true|false` line per pair, in `$GITHUB_OUTPUT` format.
It FAILS OPEN: if the base cannot be read or a block cannot be found on either
side, the answer is `true` (run the suite). It never exits non-zero for a
content reason, so it cannot block CI; at worst it costs one extra run.

Dependency-free on purpose (no PyYAML): it runs in the `changes` job before
anything is installed, and the repo has been bitten by `import yaml` on a bare
runner before (see check_deterministic_commit_contract.py).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ".github/workflows/ci.yml"
_OUTPUT_REF_RE = re.compile(r"needs\.changes\.outputs\.([A-Za-z0-9_]+)")

_JOB_KEY_RE = re.compile(r"^  [A-Za-z0-9_-]+:\s*$")


def header(text: str) -> str | None:
    """Everything before the top-level `jobs:` key."""
    match = re.search(r"^jobs:\s*$", text, re.MULTILINE)
    return text[: match.start()] if match else None


def job_block(text: str, job: str) -> str | None:
    """The text of top-level job `job` (2-space key) up to the next job key."""
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.rstrip() == f"  {job}:":
            start = index
            break
    if start is None:
        return None
    end = start + 1
    while end < len(lines) and not _JOB_KEY_RE.match(lines[end]):
        end += 1
    return "\n".join(lines[start:end])


def filter_block(text: str, name: str) -> str | None:
    """The dorny/paths-filter entry `name:` and its glob list.

    Filters are keys inside a block scalar, so they are found by their exact
    indentation within the `changes` job and end at the next key of the same
    (or a shallower) indentation.
    """
    changes = job_block(text, "changes")
    if changes is None:
        return None
    lines = changes.splitlines()
    key_re = re.compile(rf"^(?P<indent> +){re.escape(name)}:\s*$")
    for index, line in enumerate(lines):
        match = key_re.match(line)
        if match is None:
            continue
        indent = len(match.group("indent"))
        end = index + 1
        while end < len(lines):
            stripped = lines[end].lstrip()
            depth = len(lines[end]) - len(stripped)
            if stripped and not stripped.startswith("#") and depth <= indent:
                break
            end += 1
        return "\n".join(lines[index:end])
    return None


def changes_skeleton(text: str) -> str | None:
    """The `changes` job with each `filters: |` block-scalar body removed.

    Edits to other suites' globs (most ci.yml PRs) stay invisible; everything
    else in the job -- outputs, steps, step options -- is compared.
    """
    changes = job_block(text, "changes")
    if changes is None:
        return None
    out: list[str] = []
    body_indent: int | None = None
    for line in changes.splitlines():
        stripped = line.lstrip()
        depth = len(line) - len(stripped)
        if body_indent is not None:
            if not stripped or depth > body_indent:
                continue
            body_indent = None
        out.append(line)
        if re.match(r"filters:\s*[|>][-+]?\s*$", stripped):
            body_indent = depth
    return "\n".join(out)


def output_line(text: str, name: str) -> str | None:
    """The `changes.outputs` entry `name: ...` (6-space key)."""
    changes = job_block(text, "changes")
    if changes is None:
        return None
    for line in changes.splitlines():
        if line.startswith(f"      {name}:"):
            return line
    return None


def referenced_outputs(text: str, job: str, name: str) -> set[str]:
    block = job_block(text, job) or ""
    return set(_OUTPUT_REF_RE.findall(block)) | {name}


def definition(text: str, job: str, names: set[str]) -> list[str | None]:
    parts: list[str | None] = [
        header(text),
        job_block(text, job),
        changes_skeleton(text),
    ]
    for output in sorted(names):
        line = output_line(text, output)
        parts.append(line)
        # Outputs computed from a path filter carry its globs too; outputs
        # that are plain expressions (test_tier, medium_tier, sweep) have no
        # filter block, and that absence is not a change.
        if line is not None and "steps.filter.outputs." in line:
            parts.append(filter_block(text, output))
    return parts


def changed(base: str | None, head: str, job: str, name: str) -> bool:
    if base is None:
        return True
    # Compare the union of the outputs either side reads, so the comparison is
    # symmetric. (A job that starts reading a different output has also
    # changed its own block, so that edit is caught there regardless.)
    names = referenced_outputs(base, job, name) | referenced_outputs(head, job, name)
    before = definition(base, job, names)
    after = definition(head, job, names)
    # A block that cannot be located on either side is treated as changed:
    # a renamed/removed job or filter is exactly the edit that must self-verify.
    if any(part is None for part in before + after):
        return True
    return before != after


def read_base(rev: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "show", f"{rev}:{WORKFLOW}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def parse_pair(raw: str) -> tuple[str, str]:
    job, sep, name = raw.partition("=")
    if not sep or not job or not name:
        raise argparse.ArgumentTypeError(f"expected JOB=FILTER, got {raw!r}")
    return job, name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--base", required=True, help="git rev of the base ci.yml")
    parser.add_argument(
        "--job",
        dest="pairs",
        action="append",
        type=parse_pair,
        required=True,
        help="JOB=FILTER (repeatable)",
    )
    args = parser.parse_args(argv)

    base = read_base(args.base)
    if base is None:
        print(
            f"::warning::could not read {WORKFLOW} at {args.base}; "
            "treating every path-scoped suite as changed (fail open)",
            file=sys.stderr,
        )
    head = (ROOT / WORKFLOW).read_text(encoding="utf-8")
    for job, name in args.pairs:
        verdict = changed(base, head, job, name)
        print(f"{name}={'true' if verdict else 'false'}")
        print(f"{job}: definition changed = {verdict}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
