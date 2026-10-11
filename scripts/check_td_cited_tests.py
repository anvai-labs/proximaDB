#!/usr/bin/env python3
# Copyright (C) 2026 ProximaDB
# SPDX-License-Identifier: Apache-2.0
"""Fail when a TD's Tests section cites a test function that does not exist.

Why this guard exists: on PR #1957 a TD cited a renamed test THREE times across
six review rounds. Each time a human (or an adversarial reviewer) found it by
reading. The citation is the part the next engineer acts on -- a TD that names a
test which no longer exists sends them looking for evidence that is not there.

Scope is deliberately narrow, because a noisy guard gets disabled:
  * only TD files that actually have a "Tests" section heading;
  * only backticked snake_case identifiers with >= 2 underscores, which is what a
    Rust test name looks like and what a prose word does not;
  * a citation is satisfied if the identifier occurs anywhere in the Rust
    sources as a whole word -- function, field, constant, whatever -- across
    `src`, `crates`, `apps`, `tests` and `benches`. The defect being caught is a
    name that vanished entirely (a rename), so "exists somewhere" is the right
    bar; tightening it to `fn <name>` flagged `valid_to_ns`, a field, and
    omitting `tests/` would flag every correct citation of an integration test;
  * a line that marks the name as historical ("earlier name", "pre-rename",
    "does not exist", "renamed to", "former") is skipped, since TDs legitimately
    record what something used to be called.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TD_DIR = ROOT / "docs" / "10-quality" / "td"

IDENT = re.compile(r"`([a-z][a-z0-9_]*)`")
HISTORICAL = re.compile(
    r"earlier name|pre-rename|does not exist|renamed to|former|used to be|no longer",
    re.IGNORECASE,
)


def exists_in_code(name: str) -> bool:
    """True when `name` occurs as a whole word anywhere in the Rust sources."""
    return (
        subprocess.run(
            # `tests` and `benches` are NOT optional: ProximaDB's integration
            # tests live flat in `tests/` (`*_integration_test.rs`, `*_e2e.rs`),
            # so omitting them made this guard fail on a CORRECT citation of an
            # integration test -- the author's only remedies being to delete the
            # citation or switch the step off, which is how a guard earns a
            # reputation for noise. Verified by running the predicate against a
            # real integration-test name before and after.
            ["grep", "-rqw", "--include=*.rs", name, "src", "crates", "apps", "tests", "benches"],
            cwd=ROOT,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def tests_section(text: str) -> list[tuple[int, str]]:
    """Lines of the first Tests section, with 1-based line numbers."""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^=+\s+Tests\b", line):
            start = i
            break
    if start is None:
        return []
    out = []
    for i in range(start + 1, len(lines)):
        if re.match(r"^=+\s+\S", lines[i]):
            break
        out.append((i + 1, lines[i]))
    return out


def main() -> int:
    if not (ROOT / "src").is_dir():
        print("check_td_cited_tests: no src/; skipping", file=sys.stderr)
        return 0

    errors: list[str] = []
    checked = 0
    for td in sorted(TD_DIR.glob("TD-*.adoc")):
        text = td.read_text(encoding="utf-8")
        for lineno, line in tests_section(text):
            if HISTORICAL.search(line):
                continue
            for name in IDENT.findall(line):
                if name.count("_") < 2:
                    continue
                checked += 1
                if not exists_in_code(name):
                    errors.append(
                        f"{td.relative_to(ROOT)}:{lineno}: cites `{name}`, "
                        "which does not occur anywhere in the Rust sources "
                        "(renamed or deleted?)"
                    )

    for e in errors:
        print(f"ERROR: {e}")
    print(f"check_td_cited_tests: {checked} citation(s) checked, {len(errors)} error(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
