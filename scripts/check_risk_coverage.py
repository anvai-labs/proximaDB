#!/usr/bin/env python3
"""Validate docs/10-quality/RISK_CONTRACT.toml — the risk-coverage contract.

The contract declares, per risk surface, the test(s) + scenarios that MUST exist
and the CI tier that MUST actually RUN them. This guard turns "is green enough
to merge?" from a per-PR human judgment into a check. Crucially, it fails if a
required test is *compiled but never run* by its declared tier — which is how
composite-FK / tenant-isolation / ANN-recall e2e tests were silently un-enforced
until they were wired into CI tiers (CI was green, but the risk was not covered).

Checks, for each [[risk]]:
  1. each required test resolves under its configured test directory (default tests/);
  2. each `required_scenarios` marker (`scenario: <name>`) appears in at least
     one of its required_tests;
  3. `must_run_in_tier` is a job listed in ci-success.needs; AND
  4. that tier's test-selection in .github/workflows/ci.yml (its `for test_file
     in <globs>` OR `for test_name in <list>`) actually includes each
     required_tests — the honest guarantee that green ⇒ the risk test ran; AND
  5. each required_tests is wired via an EXPLICIT `tests/<name>.rs` token (or
     membership in an explicit `for test_name in …` list) — NOT only via a
     wildcard glob. A wildcard glob that matches a declared test can silently
     sweep in other, unverified tests (the #558 → #567 regression:
     `tests/*recall*.rs` meant to wire filtered_ann_recall_bands also pulled in
     the known-broken td112_postflush_recall_e2e). When a wildcard glob is the
     only path, the undeclared tests it drags in are listed.

Exit 0 = clean, 1 = violation(s), 2 = usage error.
"""

from __future__ import annotations

import re
import sys
from fnmatch import fnmatch
from pathlib import Path
import tomllib

from check_ci_lane_coverage import package_runs_test

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs" / "10-quality" / "RISK_CONTRACT.toml"
CI_PATH = ROOT / ".github" / "workflows" / "ci.yml"

_NAME = r"[a-z][a-z0-9_-]*"


def _ci_text() -> str:
    return CI_PATH.read_text(encoding="utf-8")


def _job_block(ci: str, job: str) -> str | None:
    """Return the YAML text of a top-level (2-space) job block."""
    m = re.search(rf"^  {re.escape(job)}:\s*$", ci, re.MULTILINE)
    if not m:
        return None
    start = m.end()
    nxt = re.search(r"^  [A-Za-z][A-Za-z0-9_-]*:\s*$", ci[start:], re.MULTILINE)
    return ci[start : start + nxt.start()] if nxt else ci[start:]


def ci_success_needs(ci: str) -> set[str]:
    block = _job_block(ci, "ci-success")
    if not block:
        return set()
    m = re.search(r"^\s*needs:\s*\n((?:\s+-\s+[A-Za-z0-9_-]+\s*\n)+)", block, re.MULTILINE)
    return {ln.strip().lstrip("- ").strip() for ln in m.group(1).splitlines()} if m else set()


def tier_selection(ci: str, job: str) -> tuple[list[str], list[str]]:
    """Return (globs, explicit_names) the tier selects; ([], []) if none found."""
    block = _job_block(ci, job)
    if not block:
        return [], []
    globs: list[str] = []
    names: list[str] = []
    # Explicit list: `for test_name in <...>; do` (may span lines with `\`).
    m = re.search(r"for test_name in\s+(.*?)\s*; do", block, re.DOTALL)
    if m:
        for tok in re.split(r"\s+", m.group(1)):
            if re.fullmatch(_NAME, tok):
                names.append(tok)
    # Glob list: `for test_file in <globs>; do`.
    m = re.search(r"for test_file in\s+([^;]+);", block)
    if m:
        globs.extend(t for t in re.split(r"\s+", m.group(1)) if t)
    return globs, names


def tier_runs_test(ci: str, job: str, test_file: str, errors: list[str], rclass: str) -> None:
    globs, names = tier_selection(ci, job)
    test_name = Path(test_file).stem
    if names:
        if test_name not in names:
            errors.append(
                f"[{rclass}] tier {job!r} selects an explicit list that does NOT "
                f"include {test_name!r} — add it to the `for test_name in …` block "
                f"in ci.yml (green would not run this test)"
            )
        return
    if globs:
        if not any(fnmatch(test_file, g) for g in globs):
            errors.append(
                f"[{rclass}] tier {job!r} globs {globs} do NOT match {test_file} — "
                f"add a matching glob (e.g. tests/*<stem>*.rs) to the tier in ci.yml"
            )
        return
    errors.append(
        f"[{rclass}] tier {job!r} has no `for test_file/test_name in …` selection "
        f"in ci.yml — cannot confirm it runs {test_name}"
    )


def tier_wiring_precise(
    ci: str, job: str, test_file: str, declared_files: set[str], errors: list[str], rclass: str
) -> None:
    """Fail if a required test is wired ONLY via a wildcard glob.

    A wildcard glob that matches a declared test can silently sweep in OTHER,
    unverified tests (the #558 → #567 regression: `tests/*recall*.rs` was meant
    to wire filtered_ann_recall_bands but also pulled in the known-broken
    td112_postflush_recall_e2e). Require an explicit `tests/<name>.rs` token
    (or membership in an explicit `for test_name in …` list) for every declared
    test, and when a wildcard glob is the only path, list the undeclared tests it
    drags in.
    """
    globs, names = tier_selection(ci, job)
    if names:
        return  # explicit-list tier: membership (already asserted) is precise
    if not globs or test_file in globs:
        return  # no selection (flagged elsewhere) or an explicit filename token
    wildcards = [g for g in globs if "*" in g and fnmatch(test_file, g)]
    if not wildcards:
        return
    swept = sorted(
        {
            path
            for g in wildcards
            for p in (ROOT / Path(test_file).parent).glob("*.rs")
            if fnmatch(path := p.relative_to(ROOT).as_posix(), g) and path not in declared_files
        }
    )
    if swept:
        preview = ", ".join(swept[:8]) + (" …" if len(swept) > 8 else "")
        errors.append(
            f"[{rclass}] {test_file} is wired only via wildcard glob(s) {wildcards} "
            f"that ALSO match {len(swept)} undeclared test(s): {preview} — add an "
            f"explicit `{test_file}` token to the tier to avoid sweeping in unverified "
            f"tests (broad-glob sweep regression, cf. #558 → #567)"
        )


def main() -> int:
    if not CONTRACT_PATH.exists():
        print(f"ERROR: Missing risk contract: {CONTRACT_PATH}")
        return 1
    try:
        data = tomllib.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        print(f"ERROR: Invalid TOML in {CONTRACT_PATH}: {exc}")
        return 1

    risks = data.get("risk")
    if not isinstance(risks, list) or not risks:
        print("ERROR: RISK_CONTRACT.toml has no [[risk]] entries")
        return 1

    ci = _ci_text()
    needs = ci_success_needs(ci)
    errors: list[str] = []
    seen_classes: set[str] = set()
    declared_files: set[str] = {
        f"{risk.get('test_directory', 'tests')}/{t}.rs"
        for risk in risks
        for t in (risk.get("required_tests") or [])
        if isinstance(t, str)
    }

    for idx, risk in enumerate(risks, start=1):
        rclass = risk.get("class", f"<missing-class-{idx}>")
        if rclass in seen_classes:
            errors.append(f"[{rclass}] duplicate risk class")
        seen_classes.add(rclass)

        tests = risk.get("required_tests", [])
        if not isinstance(tests, list) or not tests:
            errors.append(f"[{rclass}] required_tests must be a non-empty list")
            tests = []
        directory = Path(risk.get("test_directory", "tests"))
        if directory.is_absolute() or ".." in directory.parts:
            errors.append(f"[{rclass}] test_directory must stay inside the repository")
            continue
        test_files = [(directory / f"{name}.rs").as_posix() for name in tests]
        for test_file in test_files:
            if not (ROOT / test_file).is_file():
                errors.append(f"[{rclass}] required test missing: {test_file}")

        scenarios = risk.get("required_scenarios", [])
        if scenarios:
            blob = "".join(
                (ROOT / test_file).read_text(encoding="utf-8", errors="replace")
                for test_file in test_files
                if (ROOT / test_file).is_file()
            )
            found = set(re.findall(r"scenario:\s*([a-z][a-z0-9-]*)", blob))
            for sc in scenarios:
                if sc not in found:
                    errors.append(
                        f"[{rclass}] scenario marker not found in required_tests: "
                        f"'scenario: {sc}' (add a `// scenario: {sc}` comment)"
                    )

        tier = risk.get("must_run_in_tier", "")
        if not tier:
            errors.append(f"[{rclass}] must_run_in_tier is empty")
            continue
        if tier not in needs:
            errors.append(
                f"[{rclass}] must_run_in_tier {tier!r} is not in ci-success.needs "
                f"— the tier is not a required gate"
            )
            continue
        block = _job_block(ci, tier)
        if block is None:
            errors.append(f"[{rclass}] must_run_in_tier {tier!r} is not a job in ci.yml")
            continue
        package = risk.get("test_package")
        for test_file in test_files:
            if package:
                name = Path(test_file).stem
                if not package_runs_test(block, package, name):
                    errors.append(f"[{rclass}] tier {tier!r} must run cargo test -p {package} --test {name}")
            else:
                tier_runs_test(ci, tier, test_file, errors, rclass)
                tier_wiring_precise(ci, tier, test_file, declared_files, errors, rclass)

    if errors:
        print("Risk-coverage contract validation failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print(f"OK: risk-coverage contract valid — {len(risks)} risk class(es), all wired to a required CI tier.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
