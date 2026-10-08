#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Guard: nothing under a published docs directory is silently mishandled.

Two failure modes, neither of which `mkdocs build --strict` can catch.

**1. A page mkdocs cannot render.** `exclude_docs` drops `*.adoc` wholesale
because mkdocs renders no AsciiDoc, so every `.adoc` under a published directory
is a page users are pointed at but cannot read. mkdocs reports a link to an
excluded file at INFO *unconditionally* — there is no `validation:` category to
raise it (verified against mkdocs 1.6.1 by setting every category to `warn`:
those messages stay INFO) — and an excluded page is simply not built, so nothing
complains that it vanished.

**2. A non-page file served as raw source.** mkdocs copies ANY non-excluded,
non-Markdown file into `site/` verbatim. That is right for images, and wrong for
everything else: `docs/00-product/MVP_TRUST_CORRIDOR.toml` (an internal MVP
scorecard with tier claims) was being published at `<site>/00-product/
MVP_TRUST_CORRIDOR.toml`. Neither `--strict` nor a `.adoc`-only check sees it.

Both are frozen in a baseline: the existing entries are listed, a NEW one fails,
and removing one fails until its stale baseline entry is dropped. So the list can
only shrink. See TD-DOCSITE-1.

The published directories and the exclusion patterns are derived from
`mkdocs.yml` rather than restated here, so the guard cannot drift from the site
it guards.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / "docs" / ".site-coverage-baseline.json"

# Files mkdocs renders into pages. This is mkdocs's OWN tuple
# (`mkdocs.utils.markdown_extensions`), matched the way mkdocs matches it:
# `path.endswith(markdown_extensions)`, i.e. CASE-SENSITIVELY.
#
# An earlier revision used `{".md"}` compared against `entry.suffix.lower()`.
# That is wrong in both directions: `.MD` is not a page to mkdocs (so it is
# copied into site/ as raw source) while `.lower()` classified it as renderable
# and skipped it — re-opening the exact leak this guard closes — and `.markdown`
# IS a page to mkdocs while the guard reported it as unservable.
RENDERABLE = (".markdown", ".mdown", ".mkdn", ".mkd", ".md")

# Files it copies verbatim and SHOULD: a real page can reference them.
ASSETS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".avif",
    ".woff", ".woff2", ".ttf", ".eot",
    ".css", ".js", ".map",
    ".pdf", ".mp4", ".webm",
}


def load_mkdocs_config() -> tuple[Path, list[str]]:
    """Return (docs_dir, exclude_docs patterns) from mkdocs.yml.

    Parsed by hand rather than with PyYAML for two reasons: the guard must run in
    a bare CI step with no docs toolchain installed, and mkdocs.yml carries a
    `!!python/name:` tag for pymdownx.superfences that `yaml.safe_load` refuses
    to construct.
    """
    cfg = (REPO / "mkdocs.yml").read_text().splitlines()
    docs_dir, patterns, in_block = "docs", [], False
    for line in cfg:
        if line.startswith("docs_dir:"):
            docs_dir = line.split(":", 1)[1].split("#")[0].strip().strip("'\"")
        elif line.startswith("exclude_docs:"):
            in_block = True
        elif in_block:
            # A block scalar ends at the first non-indented, non-blank line.
            if line.strip() and not line.startswith((" ", "\t")):
                in_block = False
            elif line.strip() and not line.strip().startswith("#"):
                patterns.append(line.strip())
    return REPO / docs_dir, patterns


def is_excluded(rel: PurePosixPath, patterns: list[str]) -> bool:
    """Does any `exclude_docs` pattern cover this path?

    `*.adoc` is deliberately NOT consulted for the renderability question — see
    `unpublished()`. Here it is, since the caller asks about every pattern.
    """
    for pat in patterns:
        if pat.endswith("/"):
            if pat.rstrip("/") in rel.parts[:-1]:
                return True
        elif pat.startswith("*."):
            if rel.name.endswith(pat[1:]):
                return True
        elif rel.as_posix() == pat or rel.name == pat:
            return True
    return False


def unpublished(docs: Path, patterns: list[str]) -> list[str]:
    """Files under a published directory that the site cannot properly serve.

    Only *directory* and *exact-path* exclusions count as "intentionally
    unpublished". The blanket `*.adoc` is skipped on purpose: it is the very
    pattern that hides these pages, so honouring it would make the guard
    vacuous.
    """
    structural = [p for p in patterns if not p.startswith("*.")]
    found = []
    # Walked explicitly rather than with rglob, which does not descend symlinked
    # directories and so would hide a whole tree.
    #
    # Symlinks are FLAGGED, never followed. Following them made the verdict
    # depend on directory sort order: a link from a published directory into an
    # excluded one was skipped only when the real directory had already been
    # visited, so `02-guides/r -> rfcs` passed while `02-guides/p -> 00-product`
    # failed. Exclusion is evaluated on the logical path, so the two cannot be
    # reconciled by following. There are no symlinks under docs/ today; if one is
    # ever wanted, decide its publication explicitly rather than inheriting it
    # from readdir order.
    stack = [docs]
    while stack:
        d = stack.pop()
        for entry in sorted(d.iterdir()):
            rel_entry = PurePosixPath(entry.relative_to(docs).as_posix())
            if entry.is_symlink():
                if not is_excluded(rel_entry, structural):
                    found.append(rel_entry.as_posix())
                continue
            if entry.is_dir():
                # Skip an excluded directory here rather than per-file, so the
                # decision cannot depend on traversal order.
                if not is_excluded(
                    PurePosixPath(rel_entry.as_posix() + "/x"), structural
                ):
                    stack.append(entry)
                continue
            rel = PurePosixPath(entry.relative_to(docs).as_posix())
            if is_excluded(rel, structural):
                continue
            # Renderability is case-SENSITIVE, matching mkdocs. Asset classification
            # is not: a `.PNG` is still an image a page may reference.
            if entry.name.endswith(RENDERABLE) or entry.suffix.lower() in ASSETS:
                continue
            found.append(rel.as_posix())
    return sorted(found)


def main() -> int:
    docs, patterns = load_mkdocs_config()
    if not docs.is_dir():
        print(f"error: docs_dir {docs} does not exist", file=sys.stderr)
        return 2
    # `exclude_docs` is gitignore syntax, where a leading `!` RE-INCLUDES a path.
    # This parser does not implement that, and ignoring it is not safe: a
    # negation under a directory exclusion silently republishes the file, and the
    # guard reported OK while mkdocs served it. Fail closed instead of guessing.
    negations = [p for p in patterns if p.startswith("!")]
    if negations:
        print(
            "error: exclude_docs contains gitignore-style negation(s) this guard "
            "does not interpret, so it cannot tell what the site publishes:",
            file=sys.stderr,
        )
        for p in negations:
            print(f"  {p}", file=sys.stderr)
        print(
            "\n  Remove the negation and express the intent as an explicit include,\n"
            "  or teach is_excluded() to honour it. Until then this is fail-closed\n"
            "  on purpose: a negation under a directory exclusion republishes the\n"
            "  file while this guard says OK.",
            file=sys.stderr,
        )
        return 2

    if not any(p.endswith("/") for p in patterns):
        # A parse failure would silently empty the exclusion set and report every
        # internal file as a violation; fail loudly instead of noisily.
        print("error: parsed no directory exclusions from mkdocs.yml", file=sys.stderr)
        return 2

    try:
        baseline = json.loads(BASELINE.read_text())["unpublished_asciidoc"]
    except FileNotFoundError:
        print(f"error: baseline {BASELINE} is missing", file=sys.stderr)
        return 2
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(
            f"error: baseline {BASELINE} is unreadable ({type(e).__name__}: {e}); "
            "it must be a JSON object with an 'unpublished_asciidoc' list",
            file=sys.stderr,
        )
        return 2

    actual = unpublished(docs, patterns)
    new = [p for p in actual if p not in baseline]
    gone = [p for p in baseline if p not in actual]
    rc = 0

    if new:
        rc = 1
        print(
            f"error: {len(new)} file(s) under a published docs directory cannot be "
            "served as a page, and are not in the baseline:",
            file=sys.stderr,
        )
        for p in new:
            print(f"  + {p}", file=sys.stderr)
        print(
            "\n  mkdocs renders only Markdown. An .adoc publishes nowhere while links\n"
            "  to it stay silently broken; anything else is copied into site/ as raw\n"
            "  source and served publicly. Write the page in Markdown, or exclude it\n"
            "  in mkdocs.yml. Adding it to the baseline is only appropriate alongside\n"
            "  a TD-DOCSITE-1 conversion plan.",
            file=sys.stderr,
        )

    if gone:
        rc = 1
        print(
            f"\nerror: {len(gone)} baseline entr(y/ies) no longer apply — the baseline "
            "must shrink with them:",
            file=sys.stderr,
        )
        for p in gone:
            print(f"  - {p}", file=sys.stderr)
        print(
            f"\n  Remove them from {BASELINE.relative_to(REPO)} so the count stays "
            "truthful.",
            file=sys.stderr,
        )

    if rc == 0:
        print(
            f"docs-site-coverage: OK ({len(actual)} known-unservable file(s) under "
            "published directories, none new) — TD-DOCSITE-1"
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
