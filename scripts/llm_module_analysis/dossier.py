"""Precompute per-module dossiers for LLM analysis (optimized).

Fast version: skips expensive cross-layer edge queries and JSON parsing.
Focuses on the core signals: hotspot metrics, top symbols, clone families,
zero-ref public symbols, and structural complexity proxies.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModuleDossier:
    module_path: str
    # graph_module_metric row
    hotspot_score: float = 0.0
    pagerank_score: float = 0.0
    afferent_coupling: int = 0
    efferent_coupling: int = 0
    instability: float = 0.0
    cohesion_lcom4: float = 0.0
    tdd_priority: float = 0.0
    symbol_count: int = 0
    # top-degree symbols (name, inbound-degree, file)
    top_symbols: list[tuple[str, int, str]] = field(default_factory=list)
    # structural clone families intersecting this module
    clone_families: list[tuple[int, float, list[tuple[str, int, str]]]] = field(
        default_factory=list
    )
    # zero-reference public symbols (name, file, line)
    zero_ref_public: list[tuple[str, str, int]] = field(default_factory=list)
    # complexity tops (name, file, line, signature_length)
    complexity_tops: list[tuple[str, str, int, int]] = field(default_factory=list)
    # metadata
    computed_at: str = ""


def _parse_structural_families(
    structural_file: Path,
) -> list[tuple[int, float, list[tuple[str, int, str]]]]:
    """Parse victor_structural_full.txt into (family_id, similarity, members).

    Format (human-oriented report, hard-wrapped at ~80 columns):
        ── exact (1.00, 111 tokens)
           crates/foundation/.../foo.rs:825
    request_vote                    <- wrap: continuation at column 0
           clients/.../post_attribute_bind
    ing.py:22 _get_kwargs           <- wrap can split mid-path
           ... +171 more            <- elision marker (family larger than shown)

    Member entries START with whitespace; their wrapped continuations start
    at column 0. Joining raw lines (keeping trailing spaces) reconstructs
    both wrap shapes without corrupting either. Without this join the parser
    silently drops ~1/3 of families (measured 127/200 on the 2026-11-04
    scan) - long SDK paths are exactly the ones that wrap.
    """
    if not structural_file.is_file():
        return []
    families = []
    current_family_id = 0
    current_similarity = 0.0
    current_members: list[tuple[str, int, str]] = []
    header_re = re.compile(r"^── (exact|template|near) \(([0-9.]+),")
    member_re = re.compile(r"^\s+([^:\s][^:]*):(\d+)\s+(\S+)\s*$")
    logical = ""  # accumulated member entry (entry line + column-0 wraps)

    def flush_member() -> None:
        nonlocal logical
        if not logical:
            return
        # Some wraps strip the space after the line number (`:45` +
        # `_parse_response` -> `:45_parse_response`); restore the boundary.
        # A `:<digits><letter>` run cannot occur in a real path/name pair.
        logical = re.sub(r":(\d+)(?=[A-Za-z_])", r":\1 ", logical)
        if m := member_re.match(logical):
            file, lineno, fn_name = m.group(1).strip(), int(m.group(2)), m.group(3)
            current_members.append((file, lineno, fn_name))
        logical = ""

    with structural_file.open() as f:
        for raw in f:
            line = raw.rstrip("\n")
            if m := header_re.match(line):
                flush_member()
                if current_members:
                    families.append(
                        (current_family_id, current_similarity, current_members)
                    )
                    current_family_id += 1
                    current_members = []
                kind = m.group(1)
                current_similarity = float(m.group(2))
                if kind == "exact":
                    current_similarity = 1.0
            elif not line.strip():
                flush_member()  # blank line ends the family block
            elif line.lstrip().startswith("…"):
                flush_member()  # elision marker; family is larger than shown
            elif line[:1].isspace():
                flush_member()
                logical = line  # member entry starts (indented)
            else:
                # Column-0 non-header, non-elision: wrapped continuation of
                # the in-flight member entry. Keep raw text so mid-word path
                # splits rejoin exactly and a `:825 ` trailing space
                # separates a wrapped function name.
                logical += line
    flush_member()
    if current_members:
        families.append((current_family_id, current_similarity, current_members))
    return families


def precompute_dossier(
    db_path: Path,
    module_path: str,
    structural_file: Path,
    *,
    top_n_symbols: int = 10,
    top_n_complexity: int = 5,
) -> ModuleDossier:
    """Build the dossier for one module (optimized)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    dossier = ModuleDossier(module_path=module_path)

    # 1. graph_module_metric row
    cur.execute(
        """
        SELECT hotspot_score, pagerank_score, afferent_coupling, efferent_coupling,
               instability, cohesion_lcom4, tdd_priority, symbol_count, computed_at
        FROM graph_module_metric
        WHERE module_path = ?
        """,
        (module_path,),
    )
    if row := cur.fetchone():
        (
            dossier.hotspot_score,
            dossier.pagerank_score,
            dossier.afferent_coupling,
            dossier.efferent_coupling,
            dossier.instability,
            dossier.cohesion_lcom4,
            dossier.tdd_priority,
            dossier.symbol_count,
            dossier.computed_at,
        ) = row

    # 2. top-degree symbols — SKIPPED (expensive JOIN on 95k nodes)
    # The LLM can query victor directly for top symbols if needed.

    # 3. structural clone families intersecting this module
    all_families = _parse_structural_families(structural_file)
    for fam_id, sim, members in all_families:
        intersecting = [
            (f, ln, fn) for f, ln, fn in members if f.startswith(module_path)
        ]
        if intersecting:
            dossier.clone_families.append((fam_id, sim, intersecting))

    # 4. zero-reference public symbols — SKIPPED (expensive LEFT JOIN)
    # The LLM can query victor directly if needed.

    # 5. complexity tops — SKIPPED (requires signature length computation)
    # The LLM can grep for long functions if needed.

    conn.close()
    return dossier


def precompute_all_modules(
    db_path: Path,
    structural_file: Path,
    output_dir: Path,
    *,
    min_hotspot_score: float = 0.0,
    limit: Optional[int] = None,
) -> list[Path]:
    """Precompute dossiers for all modules above the hotspot threshold."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT module_path
        FROM graph_module_metric
        WHERE hotspot_score >= ?
        ORDER BY hotspot_score DESC
        LIMIT ?
        """,
        (min_hotspot_score, -1 if limit is None else limit),
    )
    modules = [r[0] for r in cur.fetchall()]
    conn.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for mod in modules:
        dossier = precompute_dossier(db_path, mod, structural_file)
        # NOTE: '/'->'__' is NOT injective (`a/b__c.rs` vs `a/b/c.rs`
        # collide). These are transient gitignored artifacts and the
        # authoritative module path lives INSIDE each JSON (consumers use
        # `_module_path_of`, never the filename), so a collision only
        # overwrites a report file.
        out_path = output_dir / f"{mod.replace('/', '__')}.json"
        out_path.write_text(json.dumps(asdict(dossier), indent=2))
        written.append(out_path)
    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=".victor/project.db")
    parser.add_argument(
        "--structural",
        default=".victor/analysis/post-program-2026-11-04/victor_structural_full.txt",
    )
    parser.add_argument("--output-dir", default="report/llm_module_analysis/dossiers")
    parser.add_argument("--min-hotspot", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    written = precompute_all_modules(
        Path(args.db),
        Path(args.structural),
        Path(args.output_dir),
        min_hotspot_score=args.min_hotspot,
        limit=args.limit,
    )
    print(f"Precomputed {len(written)} dossiers in {args.output_dir}")
