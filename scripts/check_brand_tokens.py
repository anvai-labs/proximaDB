#!/usr/bin/env python3
"""Fail when this docs site drifts from the Anvai family brand.

<docs_dir>/stylesheets/tokens.css is a VERBATIM copy of anvaiops
design/tokens.css (the family's single source of truth) and the other pinned
files are the shared Material mapping and self-hosted font, identical in
every Anvai docs site. All are pinned by digest so a local edit fails; to
adopt a change, re-copy the files and update the digests in every repo.
"""

import hashlib
import re
import sys
from pathlib import Path

CANONICAL = "anvaiops design/tokens.css (as of e8215fc) + shared brand.css v1.1"
PINNED = {
    "tokens.css": "2ea50eda33844d7c1ae58a264c8356b5f08841c665365f97aa61df53b175bca6",
    "brand.css": "53338f26b8d1d0f08d170bb8557f067143206ac382ae83164259d7e7ed2583cf",
    "fonts/inter-latin-wght-normal.woff2": "313809049cc15a226aaca190f5aee5f18d8125a5219eafe1cd0b52098a9b9699",
    "fonts/Inter-OFL.txt": "3b0a5fca3d17942cde889069889dedbbbd075e9b599968c82a95f4d944e9b345",
}
REQUIRED_EXTRA_CSS = ("stylesheets/tokens.css", "stylesheets/brand.css")


def extra_css_entries(mkdocs: str) -> list[str]:
    """List items of the top-level extra_css key (comments elsewhere do not count)."""
    block = re.search(r"^extra_css:[ \t]*\n((?:[ \t]+-[^\n]*\n?)+)", mkdocs, re.MULTILINE)
    if block is None:
        return []
    return [m.strip().strip("'\"") for m in re.findall(r"^[ \t]+-[ \t]*([^#\n]+)", block.group(1), re.MULTILINE)]


def main(root: Path) -> int:
    mkdocs = (root / "mkdocs.yml").read_text(encoding="utf-8")
    match = re.search(r"^docs_dir:\s*(\S+)", mkdocs, re.MULTILINE)
    docs_dir = root / (match.group(1) if match else "docs")
    failures = []
    for name, digest in PINNED.items():
        path = docs_dir / "stylesheets" / name
        if not path.is_file():
            failures.append(f"{path.relative_to(root)} is missing")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            failures.append(f"{path.relative_to(root)} drifted (sha256 {actual}); re-copy it")
    entries = extra_css_entries(mkdocs)
    if [e for e in entries if e in REQUIRED_EXTRA_CSS] != list(REQUIRED_EXTRA_CSS):
        failures.append(f"mkdocs.yml extra_css must list {' then '.join(REQUIRED_EXTRA_CSS)} (found {entries})")
    if not re.search(r"^[ \t]+primary:[ \t]*custom[ \t]*$", mkdocs, re.MULTILINE):
        failures.append("mkdocs.yml palette must set primary: custom")
    for failure in failures:
        print(f"brand: {failure}", file=sys.stderr)
    if not failures:
        print(f"brand tokens OK ({CANONICAL})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()))
