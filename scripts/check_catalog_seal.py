#!/usr/bin/env python3
"""Guard that `Catalog::create_table` stays sealed (TD-CAT-7).

`Catalog::create_table` is a *provided* trait method: it calls the backend's
``create_table_inner`` and then applies the identity post-condition — an
implementation that is an identity authority may not return a table with no
``object_id``, because authorization keys on it.

Rust has no ``final`` method, so an implementation could quietly override
``create_table`` and route around the post-condition. That is exactly the
failure shape TD-CAT-7 exists to close: a mechanism whose bypass looks like
ordinary code and reports success. This guard is the seal.

Backends implement ``create_table_inner`` and ``replace_table_inner``. Nothing implements the sealed ``create_table`` / ``replace_table`` wrappers.

Exit status:
* 0 - every ``impl Catalog for ...`` block defines only ``create_table_inner`` / ``replace_table_inner``.
* 1 - an implementation overrides ``create_table`` (the seal is broken), or the
      sealed default has gone missing from the trait itself.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = ("src", "crates", "apps", "tests")
SKIP_DIRS = {"target", ".git", "node_modules"}

IMPL_RE = re.compile(r"^\s*impl(?:<[^>]*>)?\s+Catalog\s+for\s+(\S+)")
OVERRIDE_RE = re.compile(
    r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(?:create_table|replace_table)\s*\("
)
TRAIT_FILE = ROOT / "crates/control/proximadb-catalog/src/lib.rs"


def rust_files():
    for root in SCAN_ROOTS:
        base = ROOT / root
        if not base.is_dir():
            continue
        for path in base.rglob("*.rs"):
            if SKIP_DIRS & set(path.relative_to(ROOT).parts):
                continue
            yield path


def scan(path: Path) -> list[str]:
    """Report `fn create_table(` defined inside an `impl Catalog for` block."""
    violations: list[str] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    depth = 0
    target = None
    for lineno, line in enumerate(lines, 1):
        if target is None:
            match = IMPL_RE.match(line)
            if match:
                target = match.group(1)
                depth = line.count("{") - line.count("}")
                if depth <= 0:  # single-line or trait-bound continuation
                    target = None
            continue
        kind = "create_table" if "fn create_table(" in line else "replace_table"
        if OVERRIDE_RE.match(line):
            violations.append(
                f"{path.relative_to(ROOT)}:{lineno}: `impl Catalog for {target}` "
                f"overrides `{kind}`; implement `{kind}_inner` instead "
                f"so the identity post-condition still runs (TD-CAT-7)"
            )
        depth += line.count("{") - line.count("}")
        if depth <= 0:
            target = None
    return violations


def main() -> int:
    violations: list[str] = []
    for path in rust_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if "impl Catalog for" not in text:
            continue
        violations.extend(scan(path))

    trait_src = TRAIT_FILE.read_text(encoding="utf-8", errors="replace")
    for inner in ("create_table_inner", "replace_table_inner"):
        if f"async fn {inner}(" not in trait_src:
            violations.append(
                f"{TRAIT_FILE.relative_to(ROOT)}: the `Catalog` trait no longer declares "
                f"`{inner}`; the sealed wrapper carrying the identity post-condition "
                f"has been removed (TD-CAT-7)"
            )

    if violations:
        print("❌ catalog seal broken:")
        for violation in violations:
            print(f"   {violation}")
        return 1
    print("✅ catalog seal intact: create_table and replace_table are provided, backends implement the _inner halves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
