#!/usr/bin/env python3
"""Ratchet broken documentation references: dead link targets and unresolvable TD/ADR ids.

Two failure classes this guards (both observed while reconciling the metadata
spine, ADR-098, 2026-11-19):

* **Dead link targets.** ``link:``/``xref:``/``include::`` (AsciiDoc) and
  ``](path)`` (Markdown) relative targets that name no tracked file. An ADR
  filed with guessed sibling filenames shipped four of these before review;
  CI's former "Check documentation links" step only ran ``find | head`` and
  could never fail.
* **Unresolvable ids.** A ``TD-…`` or ``ADR-NNN`` mention that matches no
  per-file TD heading, no legacy ``TECHNICAL_DEBT.adoc`` row and no ADR file —
  a hallucinated id, a typo, or a pointer left behind by a renumbering.

The ratchet is PER FILE and bidirectional, like
``count_type_name_collisions.py``: ``--mode no-regression`` fails when any
file's count rises above its baseline (a fix elsewhere cannot pay for a new
break) and when the total falls without the baseline being lowered in the same
PR. ``--mode baseline`` rewrites the baseline but refuses any per-file
increase. ``--mode report`` prints everything and never fails.

Files come from ``git ls-files`` so a local run equals CI's. Stdlib only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "docs" / "_internal" / "roadmap" / "DOC_REFERENCES_BASELINE.json"

ADOC_LINK_RE = re.compile(r"(?:link|xref):([^\s\[\]]+)\[|include::([^\s\[\]]+)\[")
MD_LINK_RE = re.compile(r"\]\(<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\)")
ADR_MENTION_RE = re.compile(r"\bADR-(\d{3,4}[a-z]?)\b")


def _load_unique_checker():
    """Import check_td_adr_unique by path (immune to a foreign `scripts` package on sys.path)."""
    spec = importlib.util.spec_from_file_location(
        "_check_td_adr_unique", ROOT / "scripts" / "check_td_adr_unique.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load scripts/check_td_adr_unique.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tracked_docs() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "docs", "*.adoc", "*.md"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    return sorted({ROOT / p for p in out if p.endswith((".adoc", ".md"))})


def tracked_files() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.split()
    return set(out)


def _is_external(target: str) -> bool:
    return (
        "://" in target
        or target.startswith(("mailto:", "#", "{", "/"))
        or "{" in target
    )


def _looks_like_file(target: str) -> bool:
    # `xref:_d6_supersessions[` is an in-document anchor, not a file.
    name = target.rsplit("/", 1)[-1]
    return "/" in target or "." in name


FENCE_RE = re.compile(r"^(```|~~~|----|\.\.\.\.)\s*$")
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")


def prose(text: str) -> str:
    """Text with code blocks and inline code removed: examples there are not references."""
    out: list[str] = []
    fenced = False
    for line in text.splitlines():
        if FENCE_RE.match(line.strip()):
            fenced = not fenced
            continue
        if not fenced:
            out.append(INLINE_CODE_RE.sub("", line))
    return "\n".join(out)


def link_targets(path: Path, text: str) -> list[str]:
    targets: list[str] = []
    if path.suffix == ".adoc":
        for m in ADOC_LINK_RE.finditer(text):
            targets.append(m.group(1) or m.group(2))
    else:
        targets.extend(m.group(1) for m in MD_LINK_RE.finditer(text))
    return targets


def broken_links(path: Path, text: str, files: set[str]) -> list[str]:
    broken: list[str] = []
    for raw in link_targets(path, text):
        target = raw.split("#", 1)[0]
        if not target or _is_external(raw) or not _looks_like_file(target):
            continue
        resolved = (path.parent / target).resolve()
        try:
            rel = resolved.relative_to(ROOT).as_posix()
        except ValueError:
            broken.append(raw)
            continue
        if rel in files or any(f.startswith(rel.rstrip("/") + "/") for f in files):
            continue
        broken.append(raw)
    return broken


def _adr_key(num: str) -> str:
    """`ADR-0053` and `ADR-053` name the same record: compare without leading zeros."""
    digits = num.rstrip("abcdefghijklmnopqrstuvwxyz")
    return str(int(digits)) + num[len(digits):]


def unresolved_ids(text: str, td_known: set[str], adr_known: set[str], td_re) -> list[str]:
    bad: list[str] = []
    for m in td_re.finditer(text):
        mention = m.group(0)
        # `TD-IOTRACE` / `TD-SEAM-*` name a family, not one entry.
        if not any(ch.isdigit() for ch in mention):
            continue
        # File slugs (TD-PAXDIR-1-collection-…) and suffixes: trim trailing
        # segments until a known id matches.
        # Only lowercase slug words are trimmed; an uppercase/digit tail
        # (`TD-SEAM-12-F99`) is a sub-id and must exist in its own right.
        # A tail with a lowercase word is a filename slug and may be trimmed
        # (`TD-DEPS-1-bincode-2x-…`); a tail of only uppercase/digit segments is
        # a sub-id (`TD-SEAM-12-F99`, `TD-162a-3`) and must exist in its own right.
        parts = mention.split("-")
        resolved = any(
            "-".join(parts[:k]) in td_known
            and (k == len(parts) or any(seg[:1].islower() for seg in parts[k:]))
            for k in range(len(parts), 1, -1)
        )
        if not resolved:
            bad.append(mention)
    for m in ADR_MENTION_RE.finditer(text):
        if _adr_key(m.group(1)) not in adr_known:
            bad.append(m.group(0))
    return bad


def scan() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    uniq = _load_unique_checker()
    # Live declarations plus ids closed into the archive register: a mention of
    # an archived TD is history, not an orphan.
    td_known = set(uniq.td_ids()) | set(uniq.archived_td_ids())
    adr_known = {_adr_key(n) for n in uniq.adr_numbers()}
    files = tracked_files()
    links: dict[str, list[str]] = {}
    ids: dict[str, list[str]] = {}
    for path in tracked_docs():
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(ROOT).as_posix()
        body = prose(text)
        bl = broken_links(path, body, files)
        if bl:
            links[rel] = bl
        ui = unresolved_ids(body, td_known, adr_known, uniq.TD_ID_RE)
        if ui:
            ids[rel] = ui
    return links, ids


def counts(found: dict[str, list[str]]) -> dict[str, int]:
    return {k: len(v) for k, v in sorted(found.items())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("no-regression", "baseline", "report"), default="no-regression")
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    args = ap.parse_args()

    links, ids = scan()
    current = {"links": counts(links), "ids": counts(ids)}

    if args.mode == "report":
        for kind, found in (("dead link", links), ("unresolved id", ids)):
            for f, items in sorted(found.items()):
                print(f"{f}: {kind}: {', '.join(sorted(set(items)))}")
        print(f"total: {sum(current['links'].values())} dead links, {sum(current['ids'].values())} unresolved ids")
        return 0

    base = {"links": {}, "ids": {}}
    if args.baseline.exists():
        base = json.loads(args.baseline.read_text(encoding="utf-8"))

    increases: list[str] = []
    for kind, found in (("links", links), ("ids", ids)):
        for f, n in current[kind].items():
            allowed = base.get(kind, {}).get(f, 0)
            if n > allowed:
                sample = ", ".join(sorted(set(found[f])))
                increases.append(f"{f}: {kind} {allowed} -> {n} ({sample})")

    if args.mode == "baseline":
        # First run: no baseline yet, so record the current debt as the floor.
        if increases and args.baseline.exists():
            print("refusing to raise the baseline; fix these instead:", file=sys.stderr)
            print("\n".join(f"  {line}" for line in increases), file=sys.stderr)
            return 1
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"baseline written: {sum(current['links'].values())} dead links, {sum(current['ids'].values())} unresolved ids")
        return 0

    failed = False
    if increases:
        failed = True
        print("doc-references: NEW broken references (fix the target or the id):", file=sys.stderr)
        print("\n".join(f"  {line}" for line in increases), file=sys.stderr)
    for kind in ("links", "ids"):
        cur_total = sum(current[kind].values())
        base_total = sum(base.get(kind, {}).values())
        if cur_total < base_total:
            failed = True
            print(
                f"doc-references: {kind} fell {base_total} -> {cur_total}; lower the baseline in this PR:\n"
                f"  python3 scripts/check_doc_references.py --mode baseline",
                file=sys.stderr,
            )
    if failed:
        return 1
    print(
        f"doc-references: OK ({sum(current['links'].values())} dead links, "
        f"{sum(current['ids'].values())} unresolved ids; none new)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
