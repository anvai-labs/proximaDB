#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render the Homebrew formula for a published release `vX.Y.Z`.

The formula users install lives in the tap, anvai-labs/homebrew-tap
(`Formula/proximadb.rb`, `brew install anvai-labs/tap/proximadb`). This
renders the version that tap file must carry for a release:

* only for archives the release ACTUALLY has. release.yml builds one
  Homebrew-installable archive, linux x86_64; the old generated formula also
  named macOS and aarch64 archives that were never built, and carried no
  sha256 at all, so brew could install none of it. If the release grows a
  second unix archive this refuses, because the tap formula has one top-level
  `url` and a multi-platform formula needs on_linux/on_macos blocks (and a
  matching docs change) -- not a silent first-match;
* with each sha256 taken from the release's SHA256SUMS.txt AND recomputed from
  the downloaded bytes (they must agree);
* by rewriting the tap's CURRENT formula (url + sha256 only), so the tap keeps
  ownership of its structure, style fixes and tests -- the same edit the tap's
  own scripts/update-proximadb.py makes. A fresh formula (same shape as the
  tap's) is rendered only when the tap has none. A downgrade is refused.

Usage:
  release_homebrew_formula.py --repo OWNER/NAME --version X.Y.Z --out proximadb.rb
                              [--base Formula/proximadb.rb] [--skip-download]
Needs `gh` (GH_TOKEN) to read the release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys

STABLE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
LINUX_X86 = "x86_64-unknown-linux-gnu"
# Archive targets a Homebrew formula could install. Windows zips never can.
UNIX_TARGETS = (
    LINUX_X86,
    "aarch64-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
)

TEMPLATE = """class Proximadb < Formula
  desc "Vector, document, and graph database server"
  homepage "https://github.com/{repo}"
  url "{url}"
  sha256 "{sha}"
  license "Apache-2.0"

  livecheck do
    url :stable
    strategy :github_latest
  end

  depends_on arch: :x86_64
  depends_on :linux
  depends_on "openssl@3"
  depends_on "xz"

  def install
    bin.install "proximadb-server", "proximadb-bench", "proximadb-migrate"
    pkgshare.install "config"
    prefix.install "README.adoc"
  end

  def caveats
    <<~EOS
      This upstream release supports Linux x86_64 only.
      Example configuration files are in #{{pkgshare}}/config.
      Copy and configure one before starting proximadb-server --config <path>.
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{{bin}}/proximadb-server --version")
  end
end
"""


class RenderError(RuntimeError):
    pass


def _ver(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def select_asset(version: str, assets: set[str], sums: dict[str, str]) -> tuple[str, str]:
    """Pure: the one Homebrew-installable archive of the release, with its sha256."""
    built = [t for t in UNIX_TARGETS if f"proximadb-{version}-{t}.tar.gz" in assets]
    if built != [LINUX_X86]:
        raise RenderError(
            f"release v{version} has unix archives for {built or 'no target'}; the tap formula "
            f"supports exactly [{LINUX_X86}] -- add on_linux/on_macos blocks before shipping more"
        )
    asset = f"proximadb-{version}-{LINUX_X86}.tar.gz"
    sha = sums.get(asset)
    if not sha or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise RenderError(f"SHA256SUMS.txt has no valid entry for {asset}")
    return asset, sha


def rewrite(base: str | None, repo: str, version: str, asset: str, sha: str) -> str:
    """Pure: the tap formula with url/sha256 pointing at this release."""
    url = f"https://github.com/{repo}/releases/download/v{version}/{asset}"
    if base is None:
        return TEMPLATE.format(repo=repo, url=url, sha=sha)
    urls = re.findall(r'^  url "([^"]+)"$', base, flags=re.M)
    shas = re.findall(r'^  sha256 "[0-9a-f]{64}"$', base, flags=re.M)
    if len(urls) != 1 or len(shas) != 1:
        raise RenderError(
            f"tap formula has {len(urls)} top-level url and {len(shas)} sha256 lines; expected 1 each"
        )
    m = re.search(r"/releases/download/v([0-9.]+)/", urls[0])
    if m and STABLE.match(m.group(1)) and _ver(version) < _ver(m.group(1)):
        raise RenderError(f"refusing to downgrade the tap formula from {m.group(1)} to {version}")
    out = re.sub(r'^  url ".*"$', f'  url "{url}"', base, count=1, flags=re.M)
    return re.sub(r'^  sha256 ".*"$', f'  sha256 "{sha}"', out, count=1, flags=re.M)


def _gh(args: list[str], binary: bool = False):
    proc = subprocess.run(["gh", *args], capture_output=True, check=False, text=not binary)
    if proc.returncode != 0:
        err = proc.stderr if isinstance(proc.stderr, str) else proc.stderr.decode(errors="replace")
        raise RenderError(f"gh {' '.join(args)}: {err.strip()}")
    return proc.stdout


def release_facts(repo: str, version: str) -> tuple[set[str], dict[str, str], dict[str, int]]:
    rel = json.loads(_gh(["api", f"repos/{repo}/releases/tags/v{version}"]))
    if rel.get("draft") or rel.get("prerelease"):
        raise RenderError(f"v{version} is not a published stable release")
    ids = {a["name"]: a["id"] for a in rel.get("assets", [])}
    if "SHA256SUMS.txt" not in ids:
        raise RenderError(f"v{version} has no SHA256SUMS.txt")
    text = _gh(["api", "-H", "Accept: application/octet-stream",
                f"repos/{repo}/releases/assets/{ids['SHA256SUMS.txt']}"])
    sums = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1].lstrip("*")] = parts[0].lower()
    return set(ids), sums, ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--base", type=pathlib.Path, help="the tap's current Formula/proximadb.rb, if any")
    ap.add_argument("--skip-download", action="store_true",
                    help="trust SHA256SUMS.txt without re-hashing the archive (tests only)")
    args = ap.parse_args()
    try:
        if not STABLE.match(args.version):
            raise RenderError(f"{args.version} is not a stable X.Y.Z version")
        assets, sums, ids = release_facts(args.repo, args.version)
        asset, sha = select_asset(args.version, assets, sums)
        if not args.skip_download:
            blob = _gh(["api", "-H", "Accept: application/octet-stream",
                        f"repos/{args.repo}/releases/assets/{ids[asset]}"], binary=True)
            got = hashlib.sha256(blob).hexdigest()
            if got != sha:
                raise RenderError(f"{asset}: downloaded bytes hash to {got}, SHA256SUMS.txt says {sha}")
        base = args.base.read_text(encoding="utf-8") if args.base and args.base.is_file() else None
        formula = rewrite(base, args.repo, args.version, asset, sha)
    except RenderError as exc:
        print(f"::error::homebrew formula: {exc}")
        return 1
    args.out.write_text(formula, encoding="utf-8")
    print(f"rendered {args.out}: {asset} sha256 {sha} ({'tap formula rewritten' if base else 'fresh template'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
