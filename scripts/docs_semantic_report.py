#!/usr/bin/env python3
"""Semantic (near-duplicate) documentation report — guardrail tier 2.

Agentic coding's classic docs failure is the paraphrased re-implementation:
the same fact restated in different words, both claiming authority, drifting
apart silently (this repo pruned a generation of them in b95294651, "one
authority per question"). Token-based CPD (tier 1, the jscpd ratchet) can
only see VERBATIM copies; this report surfaces NEAR-duplicate section pairs
so a reviewer can decide which authority survives.

Deterministic by design: chunks (sections split on markdown/asciidoc
headings) are vectorized with TF-IDF over character 3-5-grams (char_wb,
sklearn + numpy — no model download, no version drift, same output every
run). An optional embedding backend can be plugged in later behind the
same cosine threshold; it must be pinned in-repo before any of this
becomes a blocking gate. The default operating point (cosine >= 0.80,
min-chunk-chars 300) was validated by threshold sweep on 2026-10-21:
17 pairs with near-perfect precision — a 200-char floor admits short
template/header noise, a 400-char floor over-prunes.

Output: top-N similar chunk pairs (cosine >= threshold) with file, heading,
and line span — as a table (stdout/markdown) or JSON. Advisory only: exits
0 unless --fail-on-count is given (reserved for a future ratchet once the
threshold proves stable on real history).

Usage:
    python3 scripts/docs_semantic_report.py [--threshold 0.80] [--top 20]
        [--min-chunk-chars 300] [--json] [--docs-root docs]
"""

import argparse
import json
import re
import sys
from pathlib import Path

MIN_HEADING_LEN = 3

MD_HEADING = re.compile(r"^#{1,6}\s+\S")
ADOC_TITLE = re.compile(r"^={1,6}\s+\S[^=]*[^=\s]$")


def is_heading(line: str) -> bool:
    return bool(MD_HEADING.match(line) or ADOC_TITLE.match(line))


def chunk_file(path: Path, docs_root: Path):
    """Yield (title, text, start_line) sections split on headings."""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return
    lines = text.splitlines()
    sections, title, buf, start = [], None, [], 0
    for i, line in enumerate(lines, 1):
        if is_heading(line):
            if title is not None and buf:
                sections.append((title, "\n".join(buf), start))
            title, buf, start = line.strip("=# \t"), [], i
        else:
            buf.append(line)
    if title is not None and buf:
        sections.append((title, "\n".join(buf), start))
    rel = str(path.relative_to(docs_root))
    for title, body, start in sections:
        yield {
            "file": rel,
            "heading": title,
            "start_line": start,
            "end_line": start + body.count("\n") + 1,
            "text": body,
        }


def vectorize(chunks):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [c["text"] for c in chunks]
    # Character n-grams catch phrasing drift (the paraphrase signal)
    # while remaining robust to terminology substitutions.
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    matrix = vec.fit_transform(texts)
    return cosine_similarity(matrix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--min-chunk-chars", type=int, default=300)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--docs-root", default="docs")
    args = parser.parse_args()

    try:
        import sklearn  # noqa: F401  (graceful skip on sklearn-less boxes)
    except ImportError:
        print("scikit-learn unavailable; skipping semantic report (advisory)")
        return

    root = Path(args.docs_root)
    files = sorted(
        p for p in root.rglob("*") if p.suffix in (".adoc", ".md") and p.is_file()
        and "node_modules" not in p.parts
    )
    chunks = [
        c
        for f in files
        for c in chunk_file(f, root)
        if len(c["text"].strip()) >= args.min_chunk_chars
    ]
    if len(chunks) < 2:
        print("fewer than 2 chunks; nothing to compare")
        return

    sims = vectorize(chunks)
    pairs = []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            a, b = chunks[i], chunks[j]
            if a["file"] == b["file"] and a["heading"] == b["heading"]:
                continue  # repeated same-name sections within one file
            if sims[i][j] >= args.threshold:
                pairs.append((float(sims[i][j]), a, b))
    pairs.sort(key=lambda p: -p[0])

    if args.json:
        print(json.dumps(
            [
                {
                    "cosine": round(s, 4),
                    "a": {k: a[k] for k in ("file", "heading", "start_line", "end_line")},
                    "b": {k: b[k] for k in ("file", "heading", "start_line", "end_line")},
                }
                for s, a, b in pairs[: args.top]
            ],
            indent=1,
        ))
        return

    print(f"docs semantic report: {len(files)} files, {len(chunks)} sections, "
          f"{len(pairs)} pair(s) >= {args.threshold:.2f} cosine")
    for s, a, b in pairs[: args.top]:
        print(f"  {s:.3f}  {a['file']} :: {a['heading'][:40]} (L{a['start_line']})")
        print(f"         <->  {b['file']} :: {b['heading'][:40]} (L{b['start_line']})")
    if not pairs:
        print("  (no near-duplicate section pairs above threshold)")


if __name__ == "__main__":
    main()
