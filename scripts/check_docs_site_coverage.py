#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Guard: no user-facing AsciiDoc page is silently absent from the docs site.

`mkdocs build --strict` cannot catch this. mkdocs reports a link to a file
matched by `exclude_docs` at INFO *unconditionally* — there is no `validation:`
category to raise it to a warning (verified against mkdocs 1.6.1 by setting
every category to `warn`: the excluded-file messages stay INFO). And a page that
is excluded is simply not built, so nothing complains that it vanished.

Since `exclude_docs` drops `*.adoc` wholesale (mkdocs renders no AsciiDoc), every
`.adoc` under a published directory is a page users are pointed at but cannot
read. This guard freezes that set: the existing ones are listed in the baseline,
a NEW one fails, and converting one makes its stale baseline entry fail. So the
list can only shrink, and the failure mode stops being silence. See TD-DOCSITE-1.

The published directories are derived from mkdocs.yml rather than restated here,
so the guard cannot drift from the site it guards.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / "docs" / ".site-coverage-baseline.json"


def load_mkdocs_exclusions() -> tuple[Path, list[str]]:
    """Return (docs_dir, exclude patterns) from mkdocs.yml.

    Parsed directly instead of via PyYAML so the guard runs in a bare CI step
    with no docs toolchain installed.
    """
    cfg = (REPO / "mkdocs.yml").read_text().splitlines()
    docs_dir, patterns, in_block = "docs", [], False
    for line in cfg:
        if line.startswith("docs_dir:"):
            docs_dir = line.split(":", 1)[1].strip()
        elif line.startswith("exclude_docs:"):
            in_block = True
        elif in_block:
            # the block scalar ends at the first non-indented, non-blank line
            if line.strip() and not line.startswith((" ", "\t")):
                in_block = False
            elif line.strip() and not line.strip().startswith("#"):
                patterns.append(line.strip())
    return REPO / docs_dir, patterns


def unpublished_asciidoc(docs: Path, patterns: list[str]) -> list[str]:
    """`.adoc` files under a directory the site publishes.

    Only directory patterns are consulted: a `.adoc` is excluded by the blanket
    `*.adoc` pattern no matter where it sits, which is exactly the problem, so
    that pattern must not be treated as "intentionally unpublished".
    """
    excluded_dirs = [p.rstrip("/") for p in patterns if p.endswith("/")]
    found = []
    for path in sorted(docs.rglob("*.adoc")):
        rel = path.relative_to(docs)
        if any(part in excluded_dirs for part in rel.parts[:-1]):
            continue
        found.append(rel.as_posix())
    return found


def main() -> int:
    docs, patterns = load_mkdocs_exclusions()
    if not docs.is_dir():
        print(f"error: docs_dir {docs} does not exist", file=sys.stderr)
        return 2
    if not any(p.endswith("/") for p in patterns):
        # A parse failure would silently empty `excluded_dirs` and report every
        # internal .adoc as a violation; fail loudly instead.
        print("error: parsed no directory exclusions from mkdocs.yml", file=sys.stderr)
        return 2

    actual = unpublished_asciidoc(docs, patterns)
    baseline = json.loads(BASELINE.read_text())["unpublished_asciidoc"]

    new = [p for p in actual if p not in baseline]
    gone = [p for p in baseline if p not in actual]
    rc = 0

    if new:
        rc = 1
        print(
            f"error: {len(new)} user-facing AsciiDoc page(s) are not rendered by the "
            "docs site, and are not in the baseline:",
            file=sys.stderr,
        )
        for p in new:
            print(f"  + {p}", file=sys.stderr)
        print(
            "\n  mkdocs cannot render AsciiDoc, so these pages publish nowhere while\n"
            "  links to them stay silently broken. Write the page in Markdown, or move\n"
            "  it under a directory excluded from the site (see mkdocs.yml). Adding it\n"
            "  to the baseline is only appropriate alongside a TD-DOCSITE-1 conversion\n"
            "  plan for it.",
            file=sys.stderr,
        )

    if gone:
        rc = 1
        print(
            f"\nerror: {len(gone)} baseline entr(y/ies) no longer exist — the baseline "
            "must shrink with them:",
            file=sys.stderr,
        )
        for p in gone:
            print(f"  - {p}", file=sys.stderr)
        print(
            f"\n  Remove them from {BASELINE.relative_to(REPO)} so the remaining count "
            "stays truthful.",
            file=sys.stderr,
        )

    if rc == 0:
        print(
            f"docs-site-coverage: OK ({len(actual)} known-unpublished AsciiDoc page(s), "
            "none new) — TD-DOCSITE-1"
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
