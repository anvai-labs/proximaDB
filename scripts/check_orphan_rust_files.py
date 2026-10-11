#!/usr/bin/env python3
# Copyright (C) 2025 ProximaDB
# SPDX-License-Identifier: Apache-2.0
"""Find `.rs` files that no build reaches (TD-ORPHAN-1).

Reachability, not a local name check. Roots come from `cargo metadata` -- every
target's `src_path`, so autodiscovered bins/tests/benches/examples and explicit
`[[bin]]` entries are all handled without guessing -- and the walk follows
`mod x;`, `#[path = "..."]`, `#[cfg_attr(..., path = "...")]` and
`include!("...")` from each root until nothing new is found.

Transitivity is the whole point. A local rule -- "is this file's stem named in a
`mod` somewhere in its parent directory" -- counts the children of an *orphaned*
`mod.rs` as reached, because the declaration exists but nothing reaches the file
making it. `src/storage/engines/sst/tests/mod.rs` is the live example: nothing
declares it, and it declares 15 test modules totalling ~8.2k LOC that no build
compiles.

Comments are stripped before scanning, because a commented-out `mod x;` is not a
declaration -- four `viper/tests/*.rs` files are unreachable exactly because
their declarations sit behind `// TEMPORARILY DISABLED`.

Verify an individual hit with the compiler using `--all-targets`, or
`cargo test --no-run --lib` for root-crate files. A bare `cargo check -p <crate>`
builds the lib only and cannot see a `#[cfg(test)]`-gated file, so it will
wrongly confirm one as unreachable.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = 58  # TD-ORPHAN-1; lower this as files are removed or re-attached.

MOD_RE = re.compile(
    r"^\s*(?:#\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;",
    re.M,
)
PATH_RE = re.compile(r'#\[\s*(?:cfg_attr\s*\([^)]*?,\s*)?path\s*=\s*"([^"]+)"')
INCLUDE_RE = re.compile(r'include!\s*\(\s*"([^"]+)"')
# `include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/proto/x.rs"))` and friends:
# take the last string literal, resolved against the crate root rather than the file.
INCLUDE_CONCAT_RE = re.compile(r'include!\s*\(\s*concat!\((?P<args>[^)]*(?:\)[^)]*)*?)\)\s*\)')


def _strip_comments(text: str) -> str:
    """Remove comments without being fooled by `//` or `/*` inside a string."""
    out, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            out.append(ch)
            i += 1
            while i < n:
                if text[i] == "\\":
                    out.append(text[i : i + 2])
                    i += 2
                    continue
                out.append(text[i])
                if text[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        if text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
            continue
        if text.startswith("/*", i):
            depth, i = 1, i + 2
            while i < n and depth:
                if text.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif text.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _target_roots() -> set[str]:
    meta = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--no-deps", "--offline"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    roots = set()
    for pkg in meta["packages"]:
        for tgt in pkg["targets"]:
            try:
                roots.add(os.path.relpath(tgt["src_path"], ROOT))
            except ValueError:
                pass
    return roots


def _module_dir(rel: str) -> str:
    base = os.path.basename(rel)
    d = os.path.dirname(rel)
    return d if base in ("mod.rs", "lib.rs", "main.rs") else os.path.join(d, base[:-3])


def unreachable() -> list[tuple[int, str]]:
    tracked = [
        f
        for f in subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.split()
        if f.endswith(".rs")
    ]
    known = set(tracked)
    text: dict[str, str] = {}
    for f in tracked:
        try:
            text[f] = _strip_comments((ROOT / f).read_text(encoding="utf-8", errors="replace"))
        except OSError:
            text[f] = ""

    # cargo roots, plus every crate root by convention: a crate outside the
    # workspace (clients/rust/codegen) has no entry in `cargo metadata --no-deps`,
    # so its `main.rs` would look unreachable although it is a real entry point.
    roots = {r for r in _target_roots() if r in known}
    roots |= {f for f in tracked if os.path.basename(f) in ("lib.rs", "main.rs", "build.rs")}
    seen: set[str] = set()
    queue = list(roots)
    while queue:
        cur = queue.pop()
        if cur in seen or cur not in known:
            continue
        seen.add(cur)
        body = text.get(cur, "")
        mdir = _module_dir(cur)
        crate_root = cur.split("/src/")[0] if "/src/" in cur else os.path.dirname(cur)

        explicit: set[str] = set()
        # `#[path]` and `include!` resolve against the directory holding THIS file
        # (`protocol.rs` + `#[path = "protocol_tests.rs"]` is a sibling), whereas a
        # conventional `mod x;` resolves inside the module directory. Resolving the
        # former against the module dir loses every sibling redirect -- 22 files on
        # this tree. Both bases are tried, which over-approximates reachability:
        # the safe direction for a guard whose false positives mean deleting live code.
        fdir = os.path.dirname(cur)
        for m in PATH_RE.finditer(body):
            for base in (fdir, mdir):
                explicit.add(os.path.normpath(os.path.join(base, m.group(1))))
        for m in INCLUDE_RE.finditer(body):
            for base in (fdir, mdir):
                explicit.add(os.path.normpath(os.path.join(base, m.group(1))))
        for m in INCLUDE_CONCAT_RE.finditer(body):
            lits = re.findall(r'"([^"]+)"', m.group("args"))
            if lits:
                tail = lits[-1].lstrip("/")
                for base in (crate_root, fdir, mdir):
                    explicit.add(os.path.normpath(os.path.join(base, tail)))
        for cand in explicit:
            if cand in known:
                queue.append(cand)

        declared = {m.group(1) for m in MOD_RE.finditer(body)}
        # A `#[path]`-redirected module name must not also resolve conventionally.
        for name in declared:
            for cand in (
                os.path.join(mdir, f"{name}.rs"),
                os.path.join(mdir, name, "mod.rs"),
            ):
                cand = os.path.normpath(cand)
                if cand in known:
                    queue.append(cand)

    out = []
    for f in tracked:
        if f in seen:
            continue
        if not re.match(r"^(?:src/|(?:crates|apps|clients)/[^/]+(?:/[^/]+)*?/src/)", f):
            continue
        out.append((len((ROOT / f).read_text(errors="replace").splitlines()), f))
    out.sort(reverse=True)
    return out


def main() -> int:
    found = unreachable()
    total = sum(n for n, _ in found)
    if len(found) > BASELINE:
        print(
            f"orphan-rust-files: FAILED — {len(found)} files ({total:,} LOC), baseline {BASELINE}"
        )
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
