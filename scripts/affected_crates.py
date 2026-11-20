#!/usr/bin/env python3
"""Print the workspace crates whose source changed vs a base ref.

Maps each changed file to the package whose manifest directory is the file's
longest path prefix, counting only source-ish files (src/, tests/, benches/,
build.rs, Cargo.toml) — so doc/CI/script edits map to no crate. Files under the
repo root's src/ map to the root `proximadb` crate.

This is DIRECT crates only — reverse-dependents are not expanded. For a change
to a low-level foundation crate, also run the full suite (`make test`); a
v2 can walk `cargo metadata`'s resolve graph for dependents.

`--exclude-dependents-of PKG` (repeatable) additionally drops PKG *and every
workspace member that transitively depends on it* (any dependency kind —
normal, dev, or build; optional deps count, conservatively). This exists for
budget-capped fast lanes: selecting a crate with `-p` compiles its whole
dependency graph, so a "leaf" crate that path-depends on the ~850K-LOC root
`proximadb` monolith is not cheap at all — naming the root alone is not enough
to keep such a lane fast. Excluded crates are reported on stderr; stdout stays
the space-separated crate list.

Usage: affected_crates.py [--exclude-dependents-of PKG]... [base-ref]
                                                 (base-ref default: origin/develop)
Prints space-separated crate names to stdout (empty if no crate source changed).
"""
import argparse
import json
import os
import subprocess
import sys


def git(args, cwd):
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd).stdout


def workspace_dependents_of(meta, roots):
    """Return `roots` plus every workspace member transitively depending on them.

    Uses the manifest-level dependency lists from `cargo metadata --no-deps`
    (cheap: no registry resolve), restricted to workspace-internal edges.
    """
    names = {p["name"] for p in meta["packages"]}
    rdeps = {}
    for p in meta["packages"]:
        for d in p["dependencies"]:
            if d["name"] in names:
                rdeps.setdefault(d["name"], set()).add(p["name"])
    closure = {r for r in roots if r in names}
    queue = list(closure)
    while queue:
        for dependent in rdeps.get(queue.pop(), ()):
            if dependent not in closure:
                closure.add(dependent)
                queue.append(dependent)
    return closure


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("base", nargs="?", default="origin/develop")
    ap.add_argument("--exclude-dependents-of", action="append", default=[], metavar="PKG")
    args = ap.parse_args()
    base = args.base
    root = git(["rev-parse", "--show-toplevel"], cwd=".").strip()
    if not root:
        sys.exit(0)
    # Committed changes vs base + any uncommitted working-tree changes.
    diff = git(["diff", "--name-only", f"{base}...HEAD"], root)
    diff += git(["diff", "--name-only", "HEAD"], root)
    # `git diff` never lists UNTRACKED files — a brand-new src file (the common
    # shape for a fresh module) was invisible to the selector, so the affected
    # package check silently skipped the crate it belongs to.
    diff += git(["ls-files", "--others", "--exclude-standard"], root)
    changed = sorted({ln.strip() for ln in diff.splitlines() if ln.strip()})
    if not changed:
        return

    meta = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],
            capture_output=True, text=True, cwd=root,
        ).stdout
    )
    # (manifest-dir-relative-with-trailing-slash, crate-name), longest dir first
    # so a nested crate wins over the root crate.
    pkgs = []
    for p in meta["packages"]:
        d = os.path.relpath(os.path.dirname(p["manifest_path"]), root)
        pkgs.append(("" if d == "." else d + "/", p["name"]))
    pkgs.sort(key=lambda x: -len(x[0]))

    def crate_for(f):
        for d, name in pkgs:
            if d == "":
                rest = f
            elif f.startswith(d):
                rest = f[len(d):]
            else:
                continue
            if rest in ("Cargo.toml", "build.rs") or rest.startswith(("src/", "tests/", "benches/")):
                return name
        return None

    seen = []
    for f in changed:
        c = crate_for(f)
        if c and c not in seen:
            seen.append(c)

    if args.exclude_dependents_of:
        excluded = workspace_dependents_of(meta, args.exclude_dependents_of)
        dropped = [c for c in seen if c in excluded]
        seen = [c for c in seen if c not in excluded]
        if dropped:
            print(
                "affected_crates: excluded (depends on %s): %s"
                % (", ".join(args.exclude_dependents_of), " ".join(dropped)),
                file=sys.stderr,
            )

    if seen:
        print(" ".join(seen))


if __name__ == "__main__":
    main()
