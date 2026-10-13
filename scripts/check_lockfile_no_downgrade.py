#!/usr/bin/env python3
"""Fail a PR whose `Cargo.lock` moves a package BACKWARDS versus the base ref.

TD-AUDITGATE-1. On 2026-10-13 two PRs merged hours apart: #2003 bumped `rustls`
to 0.23.45, closing RUSTSEC-2026-0285; #2004 regenerated `Cargo.lock` during a
rebase and silently moved it BACK to 0.23.44, reopening the advisory. #2004
passed 83/83 checks. Nothing automated noticed, because the Security Audit job
cannot fail -- it runs `cargo audit || echo "::warning::..."` under
`continue-on-error: true`, so the step is incapable of going red.

It was caught by a human reviewing an unrelated PR. This guard makes that
mechanical, and deliberately needs NO advisory database: a security fix is a
version going forwards, so a version going backwards is the signal. That also
catches plain accidental downgrades -- the common cause being a lockfile
conflict resolved with `git checkout --theirs`, which during a REBASE takes the
commit being replayed rather than the upstream branch.

Scope: only `Cargo.lock`, only *decreases*, only versions present on both sides.
Additions, removals and forward moves are all fine. A package that keeps several
majors (`rustls 0.21` + `0.23`) is compared per MAJOR line, so dropping an old
duplicate major is NOT a downgrade -- that is exactly what the TD-RUSTDEP-1 fix
did, and it must stay green.

Exit 0 = no package moved backwards; 1 = at least one did.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME_RE = re.compile(r'^name = "(.+)"', re.M)
VERSION_RE = re.compile(r'^version = "(.+)"', re.M)


def parse_lock(text: str) -> dict[str, set[str]]:
    """Map package name -> set of locked versions."""
    out: dict[str, set[str]] = {}
    for block in text.split("[[package]]")[1:]:
        name = NAME_RE.search(block)
        version = VERSION_RE.search(block)
        if name and version:
            out.setdefault(name.group(1), set()).add(version.group(1))
    return out


def version_key(raw: str) -> tuple[int, ...]:
    """Compare numerically; ignore pre-release/build metadata.

    `0.9.34+deprecated` and `1.10505.0` both appear in this lock, so split on
    the separators and keep the leading numeric components.
    """
    parts: list[int] = []
    for piece in re.split(r"[.\-+]", raw):
        if piece.isdigit():
            parts.append(int(piece))
        else:
            break
    return tuple(parts) if parts else (0,)


def major_line(raw: str) -> tuple[int, ...]:
    """The semver-compatible line a version belongs to.

    Cargo treats 0.x as breaking per minor, so the line is (0, minor) below 1.0
    and (major,) at or above it. Comparing within a line means dropping a stale
    duplicate major is not reported as a downgrade.
    """
    key = version_key(raw)
    if not key:
        return (0,)
    if key[0] == 0:
        return (0, key[1] if len(key) > 1 else 0)
    return (key[0],)


def worst_per_line(versions: set[str]) -> dict[tuple[int, ...], str]:
    """Highest version within each major line."""
    best: dict[tuple[int, ...], str] = {}
    for v in versions:
        line = major_line(v)
        if line not in best or version_key(v) > version_key(best[line]):
            best[line] = v
    return best


def git_show(ref: str, path: str) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{ref}:{path}"],
        capture_output=True,
        text=True,
    )
    return proc.stdout if proc.returncode == 0 else None


def find_downgrades(base_text: str, head_text: str) -> list[str]:
    base, head = parse_lock(base_text), parse_lock(head_text)
    findings: list[str] = []
    for name, base_versions in sorted(base.items()):
        head_versions = head.get(name)
        if not head_versions:
            continue  # removed entirely -- not a downgrade
        base_lines = worst_per_line(base_versions)
        head_lines = worst_per_line(head_versions)
        for line, base_v in base_lines.items():
            head_v = head_lines.get(line)
            if head_v is None:
                continue  # that whole major line is gone -- a dedup, not a downgrade
            if version_key(head_v) < version_key(base_v):
                findings.append(f"{name}: {base_v} -> {head_v}")
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base", nargs="?", default="origin/develop")
    ap.add_argument("--lock", default="Cargo.lock")
    ap.add_argument(
        "--require-base",
        action="store_true",
        help=(
            "Fail instead of skipping when the base lockfile cannot be read. CI "
            "passes this: a guard that silently skips is the failure mode this "
            "guard exists to fix, so an unfetched base must be loud, not green."
        ),
    )
    args = ap.parse_args()

    base_text = git_show(args.base, args.lock)
    if base_text is None:
        print("Lockfile downgrade guard")
        if args.require_base:
            print(
                f"FAILED: cannot read {args.base}:{args.lock}. The comparison did "
                f"not run, so this is reported as a failure rather than passing "
                f"silently. Fetch the base ref (fetch-depth: 0) and retry."
            )
            return 1
        print(f"SKIP: cannot read {args.base}:{args.lock}")
        return 0

    head_path = ROOT / args.lock
    if not head_path.is_file():
        print("Lockfile downgrade guard")
        if args.require_base:
            print(f"FAILED: {args.lock} not present; comparison did not run.")
            return 1
        print(f"SKIP: {args.lock} not present")
        return 0

    findings = find_downgrades(base_text, head_path.read_text(encoding="utf-8"))

    print("Lockfile downgrade guard")
    if not findings:
        print(f"OK: no package in {args.lock} moves backwards vs {args.base}.")
        return 0

    print(f"FAILED: {len(findings)} package(s) move BACKWARDS vs {args.base}")
    for f in findings:
        print(f"- {f}")
    print(
        "\nA version going backwards is how a merged security fix gets silently\n"
        "reverted (TD-AUDITGATE-1: #2004 undid #2003's rustls bump this way and\n"
        "passed every check). If the downgrade is deliberate, say so in the PR and\n"
        "pin it in the manifest; do not resolve a Cargo.lock conflict with\n"
        "`git checkout --theirs`, which during a rebase takes YOUR commit, not the\n"
        "base branch. Regenerate from the base lock plus `cargo metadata` instead."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
