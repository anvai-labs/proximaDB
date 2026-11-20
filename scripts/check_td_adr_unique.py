#!/usr/bin/env python3
"""Guard against duplicate TD and ADR identifiers across the repo.

Concurrent agent sessions repeatedly collided on TD/ADR *numbers* — two sessions
off an old base both pick the same "next" number, discovered only at merge (or
silently coexisting, e.g. two `ADR-031-*.adoc`). Sequential numbering in a shared
space has no atomic allocator, so this is a CI **backstop**: it fails the PR the
moment a duplicate id appears in the tree. Because a rebased PR re-runs CI, branch
B's `ADR-037` collides with develop's `ADR-037` on rebase and is caught *before*
merge.

What it checks (all hard errors):
  1. No two ADR files share a number — `docs/12-design/adr/ADR-<NNN>-<slug>.adoc`.
  2. No two TDs share an id — across BOTH the legacy `TECHNICAL_DEBT.adoc` table
     rows (`| TD-… |`) AND the per-file TDs (`docs/10-quality/td/TD-<id>-…`), so a
     new per-file TD can't reuse a legacy id. Non-numeric topic-scoped ids
     (`TD-CAT-1`, `TD-SC-2`) and variants (`TD-162a-2`) compare as full tokens.

What it warns (does not fail — the README index drift is a tracked backlog):
  3. An ADR file not listed in the ADR README index.

Exit 0 = unique; 1 = a duplicate (or, with --strict, any warning).

Filing convention: see `docs/12-design/HOW_TO_FILE_TD_AND_ADR.adoc`.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADR_DIR = ROOT / "docs" / "12-design" / "adr"
ADR_README = ADR_DIR / "README.adoc"
TD_TABLE = ROOT / "docs" / "10-quality" / "TECHNICAL_DEBT.adoc"
TD_DIR = ROOT / "docs" / "10-quality" / "td"

# ADR filename: ADR-<digits><optional letter>-<slug>.adoc  → capture the number.
ADR_FILE_RE = re.compile(r"^ADR-(\d+[a-z]?)-.+\.adoc$")
# A TD id token: TD- followed by an alnum/hyphen run (TD-167, TD-162a-2, TD-CAT-1).
TD_ID_RE = re.compile(r"\bTD-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\b")
# A TD id at the START of an AsciiDoc table cell: `| TD-... |` (the id column).
TD_ROW_RE = re.compile(r"^\|\s*(TD-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)\b")


def adr_numbers() -> dict[str, list[str]]:
    """Map ADR number -> list of filenames declaring it (>1 ⇒ collision)."""
    by_num: dict[str, list[str]] = defaultdict(list)
    if not ADR_DIR.is_dir():
        return by_num
    for p in sorted(ADR_DIR.glob("ADR-*.adoc")):
        m = ADR_FILE_RE.match(p.name)
        if m:
            by_num[m.group(1)].append(p.name)
    return by_num


def td_ids() -> dict[str, list[str]]:
    """Map TD id -> list of sources declaring it (legacy table rows + per-file)."""
    by_id: dict[str, list[str]] = defaultdict(list)
    # Legacy monolithic table: one declaration per id-column row.
    # A row that links to its own per-file entry (`td/TD-181-….adoc`) is a
    # migration POINTER, not a second declaration (HOW_TO_FILE_TD_AND_ADR: the
    # legacy table migrates to per-file in batches; the row stays as a pointer).
    if TD_TABLE.is_file():
        lines = TD_TABLE.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            m = TD_ROW_RE.match(line)
            if m:
                row_text = line + (lines[i + 1] if i + 1 < len(lines) else "")
                if re.search(rf"td/{re.escape(m.group(1))}-[^\s`\[\]]+\.adoc", row_text):
                    continue
                by_id[m.group(1)].append(f"{TD_TABLE.name} (table row)")
    # Per-file TDs: the id is read from the file's first `= TD-<id>: ...` heading
    # (NOT the filename — a slug uses hyphens too, so `TD-167-foo` is ambiguous;
    # in the heading the id is delimited by `:`/space, matching how the legacy
    # table row and ADR titles declare their id). Filename is cosmetic.
    if TD_DIR.is_dir():
        # `.md` per-file TDs exist too (TD-SSO-2/3, TD-SPECRAT-1/2); before 2026-11-19
        # they escaped the uniqueness check entirely.
        for p in sorted([*TD_DIR.glob("TD-*.adoc"), *TD_DIR.glob("TD-*.md")]):
            if p.name.lower() == "readme.adoc":
                continue
            lines = p.read_text(encoding="utf-8").splitlines()
            heading = next((line for line in lines if line.startswith("= ")), "")
            if p.suffix == ".md" and not heading:
                # Markdown TDs may use `# TD-<ID>: title`. A `#` heading WITHOUT the
                # `:` after the id (`# TD-SEAM-14 S0 Audit`) names its parent TD and
                # is a companion document, not a declaration.
                md = next((line for line in lines if line.startswith("# ")), "")
                if re.match(r"^#\s+TD-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*:", md):
                    heading = md
            m = TD_ID_RE.search(heading)
            if m:
                by_id[m.group(0)].append(p.name)
            else:
                # A per-file TD with no `= TD-…:` heading can't be uniqueness-checked.
                by_id[f"<no-id-heading:{p.name}>"].append(p.name)
    return by_id


def archived_td_ids() -> dict[str, str]:
    """TD ids closed into an archive register (`TECHNICAL_DEBT_CLOSED_*.adoc`).

    Archived ids are history, not live declarations: a reference to one is
    valid, but a NEW entry must never reuse one (rule 4: never reuse a number).
    """
    out: dict[str, str] = {}
    for p in sorted(TD_TABLE.parent.glob("TECHNICAL_DEBT_CLOSED_*.adoc")):
        for line in p.read_text(encoding="utf-8").splitlines():
            m = TD_ROW_RE.match(line)
            if m:
                out.setdefault(m.group(1), p.name)
    return out


def indexed_adrs() -> set[str]:
    """ADR filenames linked from the ADR README index."""
    if not ADR_README.is_file():
        return set()
    text = ADR_README.read_text(encoding="utf-8")
    return set(re.findall(r"ADR-\d+[a-z]?-[A-Za-z0-9._-]+\.adoc", text))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--strict",
        action="store_true",
        help="treat warnings (unindexed ADRs) as failures too",
    )
    args = ap.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    for num, files in sorted(adr_numbers().items()):
        if len(files) > 1:
            errors.append(
                f"ADR number {num} is used by {len(files)} files: {', '.join(files)} "
                f"— renumber to the next free ADR (scan {ADR_DIR.relative_to(ROOT)}/)."
            )

    for tid, sources in sorted(td_ids().items()):
        if len(sources) > 1:
            errors.append(
                f"TD id {tid} is declared {len(sources)}×: {', '.join(sources)} "
                f"— a TD id must be unique across the legacy table and the per-file dir."
            )

    archived = archived_td_ids()
    for tid, sources in sorted(td_ids().items()):
        if tid in archived:
            errors.append(
                f"TD id {tid} ({', '.join(sources)}) reuses an id closed into {archived[tid]} "
                f"— pick a new id; archived ids are reserved forever."
            )

    # A heading id that swallowed the filename slug (`= TD-SC-5-multi-pod-…:`)
    # registers as a different id than the one everyone cites, so a second
    # `TD-SC-5` would never collide. Ten entries carried this until 2026-11-19,
    # hiding a real duplicate (TD-181). Ids are uppercase/digit segments
    # (`TD-162a-3`, `TD-SEAM-12-F5`); a lowercase-word segment is a slug.
    for tid, sources in sorted(td_ids().items()):
        if tid.startswith("<"):
            continue  # companion doc without an id heading (e.g. an audit appendix)
        if any(re.match(r"^[a-z][a-z_]", seg) for seg in tid.split("-")[1:]):
            errors.append(
                f"TD heading id {tid!r} in {', '.join(sources)} contains a filename slug — "
                f"write the heading as `= TD-<ID>: <title>`."
            )

    indexed = indexed_adrs()
    if indexed:  # only warn when an index exists to compare against
        for p in sorted(ADR_DIR.glob("ADR-*.adoc")):
            if p.name not in indexed:
                warnings.append(f"ADR file not listed in the README index: {p.name}")

    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)

    n_adr = sum(len(v) for v in adr_numbers().values())
    n_td = len(td_ids())
    print(
        f"checked {n_adr} ADR files, {n_td} TD ids — "
        f"{len(errors)} error(s), {len(warnings)} warning(s)."
    )

    if errors or (args.strict and warnings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
