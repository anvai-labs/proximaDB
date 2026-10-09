#!/usr/bin/env python3
# Copyright (C) 2025 ProximaDB
# SPDX-License-Identifier: Apache-2.0
"""Find `.rs` files inside crate `src/` trees that no build reaches (TD-ORPHAN-1).

A file under a crate's `src/` is compiled only if one of these holds:

* a module declares it -- `mod foo;` / `pub mod foo;`, possibly behind attributes;
* an attribute redirects a module at it -- `#[path = "f.rs"]` **or**
  `#[cfg_attr(test, path = "f.rs")]`, the second of which is easy to miss and
  accounts for four files an earlier revision of this detector wrongly reported;
* `include!("f.rs")` pulls it in;
* cargo autodiscovers it: `lib.rs`, `main.rs`, `build.rs`, `src/bin/*.rs` and
  `src/bin/*/main.rs` (unless `autobins = false`), or a crate-root `tests/`,
  `benches/`, `examples/` directory. Note a directory named `tests/` *inside* a
  `src/` tree is NOT autodiscovered and still needs a `mod`.

Comments are stripped before scanning, because a commented-out `mod x;` is not a
declaration -- several `viper/tests/*.rs` files are orphaned exactly because their
declarations sit behind `// TEMPORARILY DISABLED`.

Exit 1 and list the files when the count exceeds BASELINE, so the number can only
go down. Verify a hit with the compiler using `--all-targets` (or
`cargo test --no-run --lib` for root-crate files) -- a plain `cargo check -p <crate>`
builds the lib only and cannot see a `cfg(test)`-gated file.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = 41  # TD-ORPHAN-1; lower this as files are removed or re-attached.

MOD_RE = re.compile(
    r"^\s*(?:#\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;",
    re.M,
)
PATH_RE = re.compile(r'#\[\s*(?:cfg_attr\s*\([^)]*?,\s*)?path\s*=\s*"([^"]+)"')
INCLUDE_RE = re.compile(r'include!\s*\(\s*"([^"]+)"')
SRC_TREE_RE = re.compile(r"^(?:src/|(?:crates|apps|clients)/[^/]+(?:/[^/]+)*?/src/)")


def _strip(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//.*", "", text)


def orphans() -> list[tuple[int, str]]:
    listing = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    files = [f for f in listing if f.endswith(".rs")]
    raw = {}
    for f in files:
        try:
            raw[f] = (ROOT / f).read_text(encoding="utf-8", errors="replace")
        except OSError:
            raw[f] = ""
    stripped = {f: _strip(t) for f, t in raw.items()}

    declared: dict[str, set[str]] = {}
    reached: set[str] = set()
    generated: set[str] = set()
    for f, text in stripped.items():
        d, base = os.path.dirname(f), os.path.basename(f)
        scope = d if base in ("mod.rs", "lib.rs", "main.rs") else os.path.join(d, base[:-3])
        for m in MOD_RE.finditer(text):
            declared.setdefault(scope, set()).add(m.group(1))
        for pattern in (PATH_RE, INCLUDE_RE):
            for m in pattern.finditer(text):
                reached.add(os.path.normpath(os.path.join(d, m.group(1))))
        if "include_proto!" in text:
            generated.add(d)

    out: list[tuple[int, str]] = []
    for f in files:
        if not SRC_TREE_RE.match(f):
            continue
        base, d = os.path.basename(f), os.path.dirname(f)
        if base in ("lib.rs", "main.rs", "build.rs"):
            continue
        # cargo bin autodiscovery: src/bin/*.rs and src/bin/*/main.rs
        if re.search(r"(^|/)src/bin/[^/]+\.rs$", f) or re.search(r"(^|/)src/bin/[^/]+/main\.rs$", f):
            continue
        if f in reached or d in generated or "/proto/" in f:
            continue
        name = os.path.basename(d) if base == "mod.rs" else base[:-3]
        parent = os.path.dirname(d) if base == "mod.rs" else d
        if name not in declared.get(parent, set()):
            out.append((len((ROOT / f).read_text(errors="replace").splitlines()), f))
    out.sort(reverse=True)
    return out


def main() -> int:
    found = orphans()
    total = sum(n for n, _ in found)
    if len(found) > BASELINE:
        print(f"orphan-rust-files: FAILED — {len(found)} files ({total:,} LOC), baseline {BASELINE}")
        for n, f in found:
            print(f"  {n:6d}  {f}")
        return 1
    print(
        f"orphan-rust-files: OK — {len(found)} unreachable file(s), {total:,} LOC "
        f"(baseline {BASELINE}) — TD-ORPHAN-1"
    )
    if len(found) < BASELINE:
        print(f"  baseline can be lowered to {len(found)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
