#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static check of the Homebrew formula users install `proximadb` from.

Release evidence, part (B), for the one channel that is not a registry upload:
the docs tell users `brew install proximadb`, so the tap formula must

  * exist (Formula/proximadb.rb in the tap repo);
  * reference this release: every release-asset URL it carries points at tag
    `vX.Y.Z` and names an asset that tag's GitHub Release actually has;
  * pin each such URL with a `sha256` equal to the published SHA256SUMS.txt
    entry for that asset (a formula without sha256 cannot be installed by
    `brew` at all);
  * cover linux x86_64, the platform the release ships a server binary for.

Static on purpose: running `brew` on an ubuntu runner costs minutes of
Homebrew bootstrap and proves little more than these comparisons.

Usage: homebrew_check.py --repo OWNER/NAME --version X.Y.Z --tap OWNER/homebrew-tap
                         [--formula Formula/proximadb.rb] [--out evidence.json]
"""
from __future__ import annotations

import argparse
import base64
import json
import pathlib
import re
import subprocess
import sys
from typing import Any

URL_RE = re.compile(r'^\s*url\s+"(?P<url>[^"]+)"')
SHA_RE = re.compile(r'^\s*sha256\s+"(?P<sha>[0-9a-f]{64})"')
LINUX_X86 = "x86_64-unknown-linux-gnu"


def gh_json(path: str) -> Any:
    proc = subprocess.run(["gh", "api", path], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise LookupError(f"gh api {path}: {proc.stderr.strip() or proc.stdout.strip()}")
    return json.loads(proc.stdout)


def release_sums(repo: str, tag: str) -> tuple[dict[str, str], set[str]]:
    rel = gh_json(f"repos/{repo}/releases/tags/{tag}")
    assets = {a["name"]: a["id"] for a in rel.get("assets", [])}
    if "SHA256SUMS.txt" not in assets:
        raise LookupError(f"release {tag} has no SHA256SUMS.txt asset")
    proc = subprocess.run(
        ["gh", "api", "-H", "Accept: application/octet-stream",
         f"repos/{repo}/releases/assets/{assets['SHA256SUMS.txt']}"],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise LookupError(f"cannot download SHA256SUMS.txt: {proc.stderr.strip()}")
    sums: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1].lstrip("*")] = parts[0].lower()
    return sums, set(assets)


def formula_pairs(text: str) -> list[tuple[str, str | None]]:
    """(url, sha256-or-None) for each `url` line; the sha256 is the next
    `sha256` line before the next `url`."""
    pairs: list[tuple[str, str | None]] = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = URL_RE.match(line)
        if m:
            pairs.append((m.group("url"), None))
            continue
        s = SHA_RE.match(line)
        if s and pairs and pairs[-1][1] is None:
            pairs[-1] = (pairs[-1][0], s.group("sha"))
    return pairs


def evaluate(repo: str, version: str, formula: str | None, sums: dict[str, str], assets: set[str]) -> list[dict[str, Any]]:
    """Pure: formula text + release facts -> checks."""
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    if formula is None:
        add("formula_exists", False, "no proximadb formula in the tap -- `brew install proximadb` cannot work")
        return checks
    add("formula_exists", True, "formula present")
    tag = f"v{version}"
    prefix = f"https://github.com/{repo}/releases/download/"
    ours = [(u, s) for u, s in formula_pairs(formula) if u.startswith(prefix)]
    add("references_release", bool(ours), f"{len(ours)} release-asset url(s) in the formula")
    for url, sha in ours:
        rest = url[len(prefix):]
        url_tag, _, asset = rest.partition("/")
        if url_tag != tag:
            add(f"url:{asset}", False, f"points at {url_tag}, not {tag}")
            continue
        if asset not in assets:
            add(f"url:{asset}", False, f"{tag} has no asset named {asset}")
            continue
        want = sums.get(asset)
        if sha is None:
            add(f"sha256:{asset}", False, "url has no sha256 -- brew refuses it")
        elif want is None:
            add(f"sha256:{asset}", False, "asset missing from SHA256SUMS.txt")
        else:
            add(f"sha256:{asset}", sha == want, f"formula {sha[:16]}... vs release {want[:16]}...")
    linux = [u for u, _ in ours if LINUX_X86 in u]
    add("covers_linux_x86_64", bool(linux), "linux x86_64 url present" if linux else f"no {LINUX_X86} url")
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--tap", required=True)
    ap.add_argument("--formula", default="Formula/proximadb.rb")
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    formula: str | None
    try:
        blob = gh_json(f"repos/{args.tap}/contents/{args.formula}")
        formula = base64.b64decode(blob["content"]).decode("utf-8")
    except LookupError:
        formula = None
    try:
        sums, assets = release_sums(args.repo, f"v{args.version}")
        checks = evaluate(args.repo, args.version, formula, sums, assets)
    except LookupError as exc:
        checks = [{"name": "release_assets", "ok": False, "detail": str(exc)}]
    doc = {
        "artifact": f"homebrew:{args.tap}/{args.formula}",
        "version": args.version,
        "passed": all(c["ok"] for c in checks),
        "checks": checks,
    }
    rendered = json.dumps(doc, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not doc["passed"]:
        print(f"::error::homebrew formula check FAILED for v{args.version}: "
              + "; ".join(f"{c['name']}: {c['detail']}" for c in checks if not c["ok"]))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
