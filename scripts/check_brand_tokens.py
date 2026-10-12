#!/usr/bin/env python3
"""Fail when this docs site drifts from the Anvai family brand.

<docs_dir>/stylesheets/tokens.css is a VERBATIM copy of anvaiops
design/tokens.css (the family's single source of truth) and
<docs_dir>/stylesheets/brand.css is the shared Material mapping, identical in
every Anvai docs site. Both are pinned by digest so a local edit fails; to
adopt a change, re-copy the files and update the digests in every repo.
"""

import hashlib
import re
import sys
from pathlib import Path

CANONICAL = "anvaiops@e8215fc design/tokens.css + shared brand.css v1"
PINNED = {
    "tokens.css": "2ea50eda33844d7c1ae58a264c8356b5f08841c665365f97aa61df53b175bca6",
    "brand.css": "7ffe54af26dde70b8f275f3b8c833f480ee9c76dad0bfe31aa371aaad43a0402",
    "fonts/inter-latin-wght-normal.woff2": "3100e775e8616cd2611beecfa23a4263d7037586789b43f035236a2e6fbd4c62",
}


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
    for needle in ("stylesheets/tokens.css", "stylesheets/brand.css", "primary: custom"):
        if needle not in mkdocs:
            failures.append(f"mkdocs.yml is missing {needle!r}")
    for failure in failures:
        print(f"brand: {failure}", file=sys.stderr)
    if not failures:
        print(f"brand tokens OK ({CANONICAL})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()))
