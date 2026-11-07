#!/usr/bin/env python3
"""Ratchet public Rust type-name collisions (TD-NAMING-1).

A "collision" is a `pub struct` / `pub enum` / `pub trait` NAME that is
defined in more than one non-test, non-generated source file. Two same-named
types are a footgun the moment both become reachable from one module — the
case that motivated this guard: `trait_components::path_resolver` re-exports
the storage-ports `StorageAssignment` next to call sites that need the proto
`StorageAssignment`, and an obvious-looking signature compiled against the
wrong type. They are also how a primitive gets silently duplicated
(`CompactionConfig` defined 8 times): a fix lands in one copy while the
others keep the bug.

The count is a RATCHET, not a target: `--mode no-regression` fails only when
the number RISES above the committed baseline, so the guard stops new
collisions immediately without a big-bang rename that would conflict with
every open PR. Burn-down happens in themed batches; lower the baseline in
the same PR with `--mode baseline`.

Deliberately excluded (benign by construction, see TD-NAMING-1):

* `type` aliases — per-crate `pub type Result<T> = …` is idiomatic Rust;
  only `struct` / `enum` / `trait` definitions are counted.
* generated protobuf code (`crates/foundation/proximadb-proto/src/proto/`),
  where v1/v2 mirror domain names by design;
* test code — any path segment named `tests` / `benches` / `examples`, and
  files named `*_tests.rs` / `*_test.rs` / `tests.rs`;
* `target/` build output.

Modes:
  report         -> print the count and the worst offenders (default)
  no-regression  -> exit 1 if the count exceeds the baseline
  baseline       -> rewrite the baseline to the current count
"""

from __future__ import annotations

import argparse
import json
import re
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

# `pub struct Name`, `pub(crate) enum Name<..>`, `pub trait Name: ..` at the
# start of a line (after indentation). `pub(...)` restricted visibility is
# included: a crate-private duplicate still misleads readers of that crate.
DEF_RE = re.compile(
    r"^\s*pub(?:\([^)]*\))?\s+(?:struct|enum|trait)\s+([A-Z][A-Za-z0-9_]*)",
    re.MULTILINE,
)


def is_excluded(rel: str) -> bool:
    if rel.startswith(GENERATED_PREFIXES):
        return True
    parts = rel.split("/")
    if any(part in TEST_DIR_NAMES for part in parts[:-1]):
        return True
    name = parts[-1]
    return name == "tests.rs" or name.endswith(TEST_FILE_SUFFIXES)


def collect(root: Path) -> dict[str, list[str]]:
    """Map each type name to the (sorted, de-duplicated) files defining it."""
    sites: dict[str, set[str]] = defaultdict(set)
    for scan in SCAN_ROOTS:
        base = root / scan
        if not base.is_dir():
            continue
        for path in base.rglob("*.rs"):
            rel = path.relative_to(root).as_posix()
            if is_excluded(rel):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            # Strip `#[cfg(test)] mod tests { ... }` bodies coarsely: count
            # only definitions before the first `#[cfg(test)]`. Inline test
            # modules sit at the end of the file in this repo's convention.
            cut = text.find("#[cfg(test)]")
            if cut != -1:
                text = text[:cut]
            for match in DEF_RE.finditer(text):
                sites[match.group(1)].add(rel)
    return {name: sorted(files) for name, files in sites.items() if len(files) > 1}


def redundant_count(collisions: dict[str, list[str]]) -> int:
    """Definitions beyond the first, summed over every colliding name."""
    return sum(len(files) - 1 for files in collisions.values())


def load_baseline(path: Path) -> int | None:
    try:
        return int(json.loads(path.read_text())["redundant_definitions"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--mode", choices=("report", "no-regression", "baseline"), default="report"
    )
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args(argv)

    collisions = collect(args.root)
    count = redundant_count(collisions)

    if args.mode == "baseline":
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(
            json.dumps(
                {
                    "redundant_definitions": count,
                    "colliding_names": len(collisions),
                    "note": "TD-NAMING-1 ratchet; lower via --mode baseline in the PR that burns collisions down",
                },
                indent=2,
            )
            + "\n"
        )
        print(f"baseline written: {count} redundant definitions ({len(collisions)} names)")
        return 0

    if args.mode == "report":
        print(f"{count} redundant definitions across {len(collisions)} colliding names")
        worst = sorted(collisions.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        for name, files in worst[: args.top]:
            print(f"  {len(files):3d}x {name}")
        return 0

    baseline = load_baseline(args.baseline)
    if baseline is None:
        print(f"type-name ratchet: no readable baseline at {args.baseline}", file=sys.stderr)
        return 1
    if count > baseline:
        print(
            f"type-name ratchet FAILED: {count} redundant pub type definitions > baseline {baseline}.\n"
            "A new struct/enum/trait reuses a name already defined elsewhere. Give it a\n"
            "qualifying name (e.g. ProtoStorageAssignment, not StorageAssignment) or reuse the\n"
            "existing type. Run with --mode report to see the colliding names. (TD-NAMING-1)",
            file=sys.stderr,
        )
        return 1
    if count < baseline:
        print(
            f"type-name ratchet: {count} < baseline {baseline} — collisions were burned down; "
            "lower the baseline in this PR: python3 scripts/count_type_name_collisions.py --mode baseline"
        )
    else:
        print(f"type-name ratchet OK: {count} redundant definitions (baseline {baseline})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
