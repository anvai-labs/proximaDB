#!/usr/bin/env python3
"""Ratchet public Rust type-name collisions (TD-NAMING-1).

Two metrics, each ratcheted PER NAME against a committed baseline:

* **definitions** — a `pub struct` / `pub enum` / `pub union` /
  `pub [unsafe] trait` NAME defined at more than one site in non-test,
  non-generated source. The same concept defined N times is how a fix lands
  in one copy while the others keep the bug (`OperationType` is defined 12
  times). Sites, not files: two same-named types in different modules of one
  file collide too. Mutually exclusive `#[cfg(...)]` alternatives of one type
  in one file are ONE definition, not a collision.
* **reexports** — a `pub use` that exports a NAME which is defined at more
  than one site anywhere in the tree, generated protobuf included. This is
  the wrong-type footgun that motivated the guard: `path_resolver`
  re-exports the storage-ports `StorageAssignment`, while the proto
  `StorageAssignment` is a different type with the same name, and an
  obvious-looking signature compiled against the wrong one (#2443).

`--mode no-regression` fails if ANY name's count is above its baseline (so a
fix in one place cannot pay for a new collision elsewhere) and also if the
total is BELOW baseline without the baseline being lowered (a baseline that
silently drifts down stops being a ratchet — the same rule as
`count_capability_facades.py`). Lower it in the burn-down PR with
`--mode baseline`.

Excluded by construction (TD-NAMING-1): `type` aliases (per-crate
`pub type Result<T>` is idiomatic); generated protobuf from the
DEFINITIONS metric (v1/v2 mirror domain names by design); test code —
`tests/`, `benches/`, `examples/` trees, `*_test(s).rs` / `tests.rs` files,
`#[cfg(test)]` / `#[cfg(all(test, ...))]` modules and items; comments.

Files are enumerated with `git ls-files` so a local baseline equals what CI
measures; an unreadable tracked file fails closed.
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

ATTR = r"(?:#\[[^\]]*\]\s*)*"
VIS = r"pub(?:\([^)]*\))?\s+"
# An item starts at a line start OR right after `{` / `}` / `;` — so a
# one-line `pub mod v1 { pub struct Request; }` is still seen.
DEF_RE = re.compile(
    rf"(?:^|(?<=[{{}};]))\s*({ATTR}){VIS}(?:unsafe\s+)?(?:struct|enum|union|trait)\s+([A-Z][A-Za-z0-9_]*)",
    re.MULTILINE,
)
# `pub use a::b::Name;`, `pub use a::{X, Y as Z};`, `pub(crate) use ...`
PUB_USE_RE = re.compile(rf"^[ \t]*{ATTR}{VIS}use\s+([^;]+);", re.MULTILINE)
# A test-only cfg: `cfg(test)` or `cfg(all(test, ...))` — NOT `any(test, ..)`,
# which also compiles in non-test builds.
TEST_CFG_RE = re.compile(r"#\[cfg\(\s*(?:test\s*\)|all\(\s*test\b)")
TEST_MOD_RE = re.compile(
    rf"#\[cfg\(\s*(?:test\s*\)|all\(\s*test\b)[^\]]*\]\s*{ATTR}(?:{VIS})?mod\s+\w+\s*\{{"
)
CFG_ATTR_RE = re.compile(r"#\[cfg\(")


class UnreadableSource(Exception):
    pass


def is_excluded_path(rel: str) -> bool:
    parts = rel.split("/")
    if any(part in TEST_DIR_NAMES for part in parts[:-1]):
        return True
    name = parts[-1]
    return name == "tests.rs" or name.endswith(TEST_FILE_SUFFIXES)


def is_generated(rel: str) -> bool:
    return rel.startswith(GENERATED_PREFIXES)


def strip_comments(text: str) -> str:
    """Remove `/* */` and `//` comments (string literals are rare at item
    level; a comment-looking string can only HIDE an item, never add one)."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def strip_test_modules(text: str) -> str:
    """Remove `#[cfg(test)] mod x { ... }` bodies by brace matching."""
    out, pos = [], 0
    for match in TEST_MOD_RE.finditer(text):
        if match.start() < pos:
            continue
        depth, i = 1, match.end()
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        out.append(text[pos : match.start()])
        pos = i
    out.append(text[pos:])
    return "".join(out)


def enumerate_files(root: Path) -> list[str]:
    try:
        listed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--", *(f"{s}/*.rs" for s in SCAN_ROOTS)],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        if listed:
            return sorted(listed)
    except (OSError, subprocess.CalledProcessError):
        pass
    # Not a git checkout (unit-test fixtures): walk the tree.
    return sorted(
        p.relative_to(root).as_posix()
        for s in SCAN_ROOTS if (root / s).is_dir()
        for p in (root / s).rglob("*.rs")
    )


def read(root: Path, rel: str) -> str:
    try:
        return (root / rel).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as err:
        raise UnreadableSource(f"{rel}: {err}") from err


def exported_names(use_body: str) -> list[str]:
    """Leaf names a `pub use` body exports (`as` renames honoured)."""
    body = re.sub(r"\s+", " ", use_body)
    names = []
    for part in re.split(r"[{},]", body):
        part = part.strip()
        if not part or part.endswith("::") or part in ("self", "*"):
            continue
        leaf = part.split(" as ")[-1].split("::")[-1].strip()
        if leaf and leaf[0].isupper():
            names.append(leaf)
    return names


def collect(root: Path) -> tuple[dict[str, int], dict[str, int]]:
    """Return (definition collisions, re-export collisions), name -> sites."""
    def_sites: dict[str, list[tuple[str, bool]]] = defaultdict(list)  # (file, cfg-gated)
    all_defs: dict[str, int] = defaultdict(int)  # incl. generated proto
    reexports: dict[str, int] = defaultdict(int)
    for rel in enumerate_files(root):
        if is_excluded_path(rel):
            continue
        text = strip_test_modules(strip_comments(read(root, rel)))
        for match in DEF_RE.finditer(text):
            attrs, name = match.group(1), match.group(2)
            if TEST_CFG_RE.search(attrs):
                continue  # a single test-only item
            all_defs[name] += 1
            if not is_generated(rel):
                def_sites[name].append((rel, bool(CFG_ATTR_RE.search(attrs))))
        if is_generated(rel):
            continue
        for match in PUB_USE_RE.finditer(text):
            for name in exported_names(match.group(1)):
                reexports[name] += 1

    definitions: dict[str, int] = {}
    for name, sites in def_sites.items():
        # cfg-gated same-name items in ONE file are alternatives of one type.
        per_file: dict[str, int] = defaultdict(int)
        for rel, gated in sites:
            per_file[rel] += 0 if gated else 1
        effective = sum(max(n, 1) for n in per_file.values())
        if effective > 1:
            definitions[name] = effective - 1
    collisions_via_reexport = {
        name: count for name, count in reexports.items() if all_defs.get(name, 0) > 1
    }
    return definitions, collisions_via_reexport


def total(counts: dict[str, int]) -> int:
    return sum(counts.values())


def load_baseline(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
        return {"definitions": dict(data["definitions"]), "reexports": dict(data["reexports"])}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("report", "no-regression", "baseline"), default="report")
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args(argv)

    try:
        definitions, reexports = collect(args.root)
    except UnreadableSource as err:
        print(f"type-name ratchet: unreadable source, failing closed: {err}", file=sys.stderr)
        return 1

    if args.mode == "baseline":
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(
            {"definitions": dict(sorted(definitions.items())),
             "reexports": dict(sorted(reexports.items())),
             "note": "TD-NAMING-1 per-name ratchet; regenerate with --mode baseline in the burn-down PR"},
            indent=2) + "\n")
        print(f"baseline written: {total(definitions)} redundant definitions, "
              f"{total(reexports)} re-export collisions")
        return 0

    if args.mode == "report":
        print(f"{total(definitions)} redundant definitions across {len(definitions)} names; "
              f"{total(reexports)} re-export collisions across {len(reexports)} names")
        for label, counts in (("definitions", definitions), ("reexports", reexports)):
            print(f"-- {label}:")
            for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: args.top]:
                print(f"  {n + (1 if label == 'definitions' else 0):3d}x {name}")
        return 0

    baseline = load_baseline(args.baseline)
    if baseline is None:
        print(f"type-name ratchet: no readable baseline at {args.baseline}", file=sys.stderr)
        return 1
    failures = []
    for label, counts in (("definitions", definitions), ("reexports", reexports)):
        base = baseline[label]
        for name, n in sorted(counts.items()):
            if n > base.get(name, 0):
                failures.append(f"{label}: `{name}` {base.get(name, 0)} -> {n}")
        if total(counts) < total(base):
            failures.append(
                f"{label}: total {total(counts)} < baseline {total(base)} — collisions were burned "
                "down; lower the baseline in this PR: python3 scripts/count_type_name_collisions.py --mode baseline"
            )
    if failures:
        print("type-name ratchet FAILED (TD-NAMING-1):", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        print("A new pub type (or `pub use`) reuses a name already defined elsewhere. Give it a\n"
              "qualifying name (e.g. ProtoStorageAssignment) or reuse the existing type.",
              file=sys.stderr)
        return 1
    print(f"type-name ratchet OK: {total(definitions)} definitions, {total(reexports)} re-exports (= baseline)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
