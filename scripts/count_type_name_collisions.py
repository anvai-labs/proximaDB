#!/usr/bin/env python3
"""Ratchet redundant public Rust type definitions (TD-NAMING-1).

Counts `pub[(..)] struct` / `enum` / `union` / `[unsafe] trait` NAMES (restricted
visibility included — duplication harms within the root crate too)
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

def match_brace(code: str, open_idx: int) -> int:
    """Index just past the `}` matching `{` at open_idx, on MASKED code (no
    comments or literal contents remain, so a plain depth count is exact)."""
    depth = 0
    for i in range(open_idx, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return len(code)

ATTR = r"(?:#\[(?:[^\[\]]|\[[^\[\]]*\])*\]\s*)*"  # one level of nested [] (e.g. json!([1]))
VIS = r"pub(?:\([^)]*\))?\s+"  # restricted visibility counts: duplication harms within the root crate too
# An item starts at a line start or right after `{` / `}` / `;`.
DEF_RE = re.compile(
    rf"(?:^|(?<=[{{}};]))\s*({ATTR}){VIS}(?:unsafe\s+)?(?:struct|enum|union|trait)\s+([A-Z][A-Za-z0-9_]*)",
    re.MULTILINE,
)
CFG_ATTR_RE = re.compile(r"#!?\[cfg\(")


def _split_args(s: str) -> list[str]:
    """Split a cfg argument list on top-level commas."""
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _test_only(pred: str) -> bool:
    """True iff the cfg predicate can only be enabled in a test build:
    `test`; `all(..)` with any test-only child; `any(..)` whose children are
    ALL test-only. `not(..)` and everything else may ship."""
    pred = pred.strip()
    if pred == "test":
        return True
    m = re.fullmatch(r"(all|any|not)\s*\((.*)\)", pred, re.DOTALL)
    if not m:
        return False
    op, args = m.group(1), _split_args(m.group(2))
    if op == "all":
        return any(_test_only(a) for a in args)
    if op == "any":
        return bool(args) and all(_test_only(a) for a in args)
    return False  # not(..)


def has_test_only_cfg(attrs: str) -> bool:
    """Any `#[cfg(..)]` / `#![cfg(..)]` in `attrs` whose predicate is test-only."""
    for m in re.finditer(r"#!?\[cfg\(", attrs):
        depth, i = 1, m.end()
        while i < len(attrs) and depth:
            depth += {"(": 1, ")": -1}.get(attrs[i], 0)
            i += 1
        if _test_only(attrs[m.end() : i - 1]):
            return True
    return False
MODVIS = r"pub(?:\([^)]*\))?\s+"  # modules: any visibility (restricted test mods are common)
INLINE_MOD_RE = re.compile(rf"({ATTR})(?:{MODVIS})?mod\s+[A-Za-z_]\w*\s*\{{")
OUTLINE_MOD_RE = re.compile(rf"({ATTR})(?:{MODVIS})?mod\s+([A-Za-z_]\w*)\s*;")


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
            # Rust block comments NEST.
            depth, j = 1, i + 2
            while j < n and depth:
                if text[j : j + 2] == "/*":
                    depth, j = depth + 1, j + 2
                elif text[j : j + 2] == "*/":
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
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


def module_blocks(code: str) -> tuple[list[tuple[int, int]], list[tuple[int, int, str]]]:
    """ONE pass over inline `mod name { .. }` blocks: (test-only spans,
    (start, end, name) scopes)."""
    tests, scopes = [], []
    for m in INLINE_MOD_RE.finditer(code):
        end = match_brace(code, m.end() - 1)
        name = re.search(r"mod\s+([A-Za-z_]\w*)", m.group(0)).group(1)
        scopes.append((m.start(), end, name))
        if has_test_only_cfg(m.group(1)):
            tests.append((m.start(), end))
    return tests, scopes


def scope_of(pos: int, scopes: list[tuple[int, int, str]]) -> str:
    """Module path of `pos` inside its file, e.g. `outer::imp` (`` = file root)."""
    enclosing = sorted((a, name) for a, b, name in scopes if a <= pos < b)
    return "::".join(name for _, name in enclosing)


def inside(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(a <= pos < b for a, b in spans)


def enumerate_files(root: Path) -> list[str]:
    try:
        listed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--", *(f"{s}/*.rs" for s in SCAN_ROOTS)],
            capture_output=True, text=True, check=True,
        ).stdout.split("\0")
        listed = [rel for rel in listed if rel]
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
    for m in OUTLINE_MOD_RE.finditer(code):
        if not has_test_only_cfg(m.group(1)):
            continue
        name = m.group(2)
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
        # A whole file gated by an inner `#![cfg(test)]` is test code.
        inner = re.match(r"\s*((?:#!\[(?:[^\[\]]|\[[^\[\]]*\])*\]\s*)*)", code)
        if inner and has_test_only_cfg(inner.group(1)):
            excluded.add(rel)

    sites: dict[str, set[str]] = defaultdict(set)
    for rel, code in codes.items():
        if rel in excluded:
            continue
        tests, scopes = module_blocks(code)
        for m in DEF_RE.finditer(code):
            pos, attrs, name = m.start(2), m.group(1), m.group(2)
            if has_test_only_cfg(attrs) or inside(pos, tests):
                continue
            sites[name].add(f"{rel}::{scope_of(pos, scopes)}")

    result = {}
    for name, entries in sites.items():
        # Same-named items in ONE module scope can only coexist as mutually
        # exclusive cfg alternatives (rustc rejects anything else), so a scope
        # holds ONE definition however many cfg'd copies it has. Different
        # scopes — other modules, other files — each count.
        effective = len(entries)
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
    ap.add_argument(
        "--init",
        action="store_true",
        help="allow --mode baseline to create a baseline when none is readable",
    )
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
        if baseline is None and not args.init:
            print("type-name ratchet: no readable baseline to lower; refusing to write one "
                  "(a missing/corrupt baseline must not become a way to bake in collisions). "
                  "Pass --init only when deliberately creating it.", file=sys.stderr)
            return 1
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
