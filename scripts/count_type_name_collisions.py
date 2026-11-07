#!/usr/bin/env python3
"""Ratchet redundant public Rust type definitions (TD-NAMING-1).

Counts `pub struct` / `pub enum` / `pub union` / `pub [unsafe] trait` NAMES
defined at more than one site in non-test, non-generated source. The same
name defined N times is how a fix lands in one copy while the others keep
the bug (`OperationType` is defined 12 times), and two same-named types are
a wrong-type footgun once both become reachable from one module (#2443:
`StorageAssignment`).

The ratchet is PER NAME and works both ways: `--mode no-regression` fails if
any name's count rises above its baseline (a fix elsewhere cannot pay for a
new collision) and if the total falls without the baseline being lowered in
the same PR (a baseline that silently drifts down stops being a ratchet).
`--mode baseline` lowers the baseline but REFUSES any per-name increase, so
the printed fix-it instruction cannot be used to bake in a new collision.

Counting rules:

* Sites, not files: same-named types in different modules of one file
  collide. cfg-gated alternatives of one type are one definition, whether the
  `#[cfg(..)]` sits on the items or on enclosing `mod` blocks.
* Not counted: `type` aliases (per-crate `pub type Result<T>` is idiomatic);
  generated protobuf (v1/v2 mirror domain names by design); test code —
  `tests/` `benches/` `examples/` trees, `*_test(s).rs` / `tests.rs` files,
  modules declared `#[cfg(test)] mod x;` out of line, and `#[cfg(test)]` /
  `#[cfg(all(test, ..))]` inline modules and items (`cfg(any(test, ..))`
  ships and IS counted); comments and string/char literal contents.
* Files come from `git ls-files` (a local baseline equals CI's); a file
  deleted from the work tree but not yet from the index is skipped; an
  unreadable tracked file fails closed.

Known gaps (TD-NAMING-1): `pub use` re-exports are NOT measured — the
re-export collision class, including the #2443 `StorageAssignment` case, needs
a resolving tool (rust-analyzer / `syn` + path resolution), not regexes.
Macro-generated definitions are not seen.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = (
    REPO_ROOT / "docs" / "_internal" / "roadmap" / "TYPE_NAME_COLLISION_BASELINE.json"
)
SCAN_ROOTS = ("src", "crates", "apps")
GENERATED_PREFIXES = ("crates/foundation/proximadb-proto/src/proto/",)
TEST_DIR_NAMES = {"tests", "benches", "examples", "target"}
TEST_FILE_SUFFIXES = ("_tests.rs", "_test.rs")

# Reuse the capability-facade ratchet's string/char/raw-string/comment-aware
# brace matcher rather than a second hand-rolled one (#2445 review).
_facades_spec = importlib.util.spec_from_file_location(
    "count_capability_facades", Path(__file__).resolve().parent / "count_capability_facades.py"
)
_facades = importlib.util.module_from_spec(_facades_spec)
_facades_spec.loader.exec_module(_facades)
match_brace = _facades._match_brace

ATTR = r"(?:#\[[^\]]*\]\s*)*"
VIS = r"pub(?:\([^)]*\))?\s+"
# An item starts at a line start or right after `{` / `}` / `;`.
DEF_RE = re.compile(
    rf"(?:^|(?<=[{{}};]))\s*({ATTR}){VIS}(?:unsafe\s+)?(?:struct|enum|union|trait)\s+([A-Z][A-Za-z0-9_]*)",
    re.MULTILINE,
)
TEST_CFG = r"#\[cfg\(\s*(?:test\s*\)|all\(\s*test\b)[^\]]*\]"
TEST_CFG_RE = re.compile(TEST_CFG)
CFG_ATTR_RE = re.compile(r"#\[cfg\(")
INLINE_MOD_RE = re.compile(rf"({ATTR})(?:{VIS})?mod\s+[A-Za-z_]\w*\s*\{{")
OUTLINE_TEST_MOD_RE = re.compile(rf"{TEST_CFG}\s*{ATTR}(?:{VIS})?mod\s+([A-Za-z_]\w*)\s*;")


class UnreadableSource(Exception):
    pass


def is_excluded_path(rel: str) -> bool:
    if rel.startswith(GENERATED_PREFIXES):
        return True
    parts = rel.split("/")
    if any(part in TEST_DIR_NAMES for part in parts[:-1]):
        return True
    return parts[-1] == "tests.rs" or parts[-1].endswith(TEST_FILE_SUFFIXES)


def mask_non_code(text: str) -> str:
    """Blank comments and string/char literal CONTENTS, keeping every offset
    and newline, so later regexes and brace counting only ever see code."""
    out = list(text)
    n, i = len(text), 0

    def blank(a: int, b: int) -> None:
        for k in range(a, min(b, n)):
            if out[k] != "\n":
                out[k] = " "

    def ident(ch: str) -> bool:
        return ch.isalnum() or ch == "_"

    while i < n:
        two = text[i : i + 2]
        c = text[i]
        if two == "//":
            j = text.find("\n", i)
            j = n if j == -1 else j
            blank(i, j)
            i = j
            continue
        if two == "/*":
            j = text.find("*/", i + 2)
            j = n if j == -1 else j + 2
            blank(i, j)
            i = j
            continue
        if c == "r" and (i == 0 or not ident(text[i - 1]) or text[i - 1] == "b"):
            j = i + 1
            while j < n and text[j] == "#":
                j += 1
            if j < n and text[j] == '"':
                close = '"' + "#" * (j - i - 1)
                k = text.find(close, j + 1)
                k = n if k == -1 else k + len(close)
                blank(j + 1, k - len(close))
                i = k
                continue
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            blank(i + 1, j)
            i = j + 1
            continue
        if c == "'":
            if i + 1 < n and text[i + 1] == "\\":
                k = text.find("'", i + 3)
                k = n if k == -1 else k
                blank(i + 1, k)
                i = k + 1
                continue
            if i + 2 < n and text[i + 2] == "'":
                blank(i + 1, i + 2)
                i += 3
                continue
        i += 1
    return "".join(out)


def test_and_gated_spans(code: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """(test-only module spans, cfg-gated non-test module spans)."""
    tests, gated = [], []
    for m in INLINE_MOD_RE.finditer(code):
        attrs = m.group(1)
        if not CFG_ATTR_RE.search(attrs):
            continue
        end = match_brace(code, m.end() - 1)
        (tests if TEST_CFG_RE.search(attrs) else gated).append((m.start(), end))
    return tests, gated


def inside(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(a <= pos < b for a, b in spans)


def enumerate_files(root: Path) -> list[str]:
    try:
        listed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--", *(f"{s}/*.rs" for s in SCAN_ROOTS)],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        if listed:
            return sorted(rel for rel in listed if (root / rel).exists())
    except (OSError, subprocess.CalledProcessError):
        pass
    return sorted(
        p.relative_to(root).as_posix()
        for s in SCAN_ROOTS if (root / s).is_dir()
        for p in (root / s).rglob("*.rs")
    )


def outline_test_module_files(rel: str, code: str) -> set[str]:
    """Files declared `#[cfg(test)] mod foo;` from `rel`."""
    parent = Path(rel).parent
    stem = Path(rel).stem
    base = parent if stem in ("mod", "lib", "main") else parent / stem
    found = set()
    for m in OUTLINE_TEST_MOD_RE.finditer(code):
        name = m.group(1)
        found.add((base / f"{name}.rs").as_posix())
        found.add((base / name / "mod.rs").as_posix())
    return found


def collect(root: Path) -> dict[str, int]:
    """name -> redundant definition sites (sites - 1), only for names > 1."""
    files = [rel for rel in enumerate_files(root) if not is_excluded_path(rel)]
    codes: dict[str, str] = {}
    for rel in files:
        try:
            codes[rel] = mask_non_code((root / rel).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as err:
            raise UnreadableSource(f"{rel}: {err}") from err
    excluded = set()
    for rel, code in codes.items():
        excluded |= outline_test_module_files(rel, code)

    sites: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for rel, code in codes.items():
        if rel in excluded:
            continue
        tests, gated = test_and_gated_spans(code)
        for m in DEF_RE.finditer(code):
            pos, attrs, name = m.start(2), m.group(1), m.group(2)
            if TEST_CFG_RE.search(attrs) or inside(pos, tests):
                continue
            is_gated = bool(CFG_ATTR_RE.search(attrs)) or inside(pos, gated)
            sites[name].append((rel, is_gated))

    result = {}
    for name, entries in sites.items():
        per_file: dict[str, list[bool]] = defaultdict(list)
        for rel, is_gated in entries:
            per_file[rel].append(is_gated)
        # In one file, all cfg-gated alternatives together are ONE definition.
        effective = sum(
            sum(1 for g in gs if not g) + (1 if any(gs) else 0) for gs in per_file.values()
        )
        if effective > 1:
            result[name] = effective - 1
    return result


def total(counts: dict[str, int]) -> int:
    return sum(counts.values())


def load_baseline(path: Path) -> dict[str, int] | None:
    try:
        return {k: int(v) for k, v in json.loads(path.read_text())["definitions"].items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("report", "no-regression", "baseline"), default="report")
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args(argv)

    try:
        counts = collect(args.root)
    except UnreadableSource as err:
        print(f"type-name ratchet: unreadable source, failing closed: {err}", file=sys.stderr)
        return 1

    if args.mode == "report":
        print(f"{total(counts)} redundant definitions across {len(counts)} names")
        for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: args.top]:
            print(f"  {n + 1:3d}x {name}")
        return 0

    baseline = load_baseline(args.baseline)
    rises = [] if baseline is None else [
        f"`{name}` {baseline.get(name, 0)} -> {n}"
        for name, n in sorted(counts.items()) if n > baseline.get(name, 0)
    ]

    if args.mode == "baseline":
        if rises:
            print("type-name ratchet: refusing to RAISE the baseline (it may only be lowered):",
                  file=sys.stderr)
            for r in rises:
                print(f"  {r}", file=sys.stderr)
            return 1
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(
            {"definitions": dict(sorted(counts.items())),
             "note": "TD-NAMING-1 per-name ratchet; lower via --mode baseline in the burn-down PR"},
            indent=2) + "\n")
        print(f"baseline written: {total(counts)} redundant definitions ({len(counts)} names)")
        return 0

    if baseline is None:
        print(f"type-name ratchet: no readable baseline at {args.baseline}", file=sys.stderr)
        return 1
    failures = list(rises)
    if total(counts) < total(baseline):
        failures.append(
            f"total {total(counts)} < baseline {total(baseline)} — collisions were burned down; "
            "lower the baseline in this PR: python3 scripts/count_type_name_collisions.py --mode baseline"
        )
    if failures:
        print("type-name ratchet FAILED (TD-NAMING-1):", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        print("A new pub struct/enum/union/trait reuses a name already defined elsewhere. Give it a\n"
              "qualifying name (e.g. ProtoStorageAssignment) or reuse the existing type.",
              file=sys.stderr)
        return 1
    print(f"type-name ratchet OK: {total(counts)} redundant definitions (= baseline)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
