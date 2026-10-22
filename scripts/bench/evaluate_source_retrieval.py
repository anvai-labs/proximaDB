#!/usr/bin/env python3
"""Evaluate chunk retrieval against document qrels after source collapse.

The evaluator assigns each source the maximum score of any retrieved chunk.
Quality runs should use a corpus built without exact deduplication: one stored
vector mapping to multiple source documents makes document ranking ambiguous.
Alias expansion is therefore an explicit diagnostic opt-in.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# corpus_io is a sibling module in scripts/bench/. Inject that directory so the
# import works both when run as a script and when loaded via importlib in tests.
_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from corpus_io import (  # noqa: E402  — path-injected sibling bench module
    atomic_json_write as _atomic_json,
    atomic_jsonl_write as _atomic_jsonl,
    finite_number as _finite_number,
    histogram as _histogram,
    iter_jsonl as _jsonl,
    required_text as _required_text,
    sha256_file as _sha256,
)


def _load_occurrences(
    path: Path, *, allow_deduplicated_aliases: bool
) -> tuple[dict[str, tuple[str, ...]], int, int, frozenset[str]]:
    sources_by_corpus: dict[str, set[str]] = defaultdict(set)
    occurrence_count = 0
    alias_count = 0
    for line_number, record in _jsonl(path):
        context = f"occurrences line {line_number}"
        corpus_id = _required_text(record, "corpus_id", context)
        source_id = _required_text(record, "source_id", context)
        alias = record.get("deduplicated_alias")
        if not isinstance(alias, bool):
            raise ValueError(f"{context}: deduplicated_alias must be boolean")
        sources_by_corpus[corpus_id].add(source_id)
        occurrence_count += 1
        alias_count += int(alias)
    if not occurrence_count:
        raise ValueError("occurrence sidecar contains no rows")
    if alias_count and not allow_deduplicated_aliases:
        raise ValueError(
            "occurrence sidecar contains deduplicated aliases; rebuild quality "
            "corpora with deduplicate=none or explicitly allow diagnostic expansion"
        )
    return (
        {
            corpus_id: tuple(sorted(source_ids))
            for corpus_id, source_ids in sources_by_corpus.items()
        },
        occurrence_count,
        alias_count,
        frozenset(
            source_id
            for source_ids in sources_by_corpus.values()
            for source_id in source_ids
        ),
    )


def _load_qrels(path: Path) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    for line_number, record in _jsonl(path):
        context = f"source qrels line {line_number}"
        if record.get("judgment_granularity") != "document":
            raise ValueError(f"{context}: judgment_granularity must be 'document'")
        query_id = _required_text(record, "query_id", context)
        source_id = _required_text(record, "source_id", context)
        relevance = _finite_number(record, "relevance", context)
        if relevance <= 0:
            raise ValueError(f"{context}: relevance must be positive")
        qrels[query_id][source_id] = max(relevance, qrels[query_id].get(source_id, 0.0))
    if not qrels:
        raise ValueError("source qrels contain no relations")
    return dict(qrels)


def _collapse_run(
    path: Path,
    sources_by_corpus: dict[str, tuple[str, ...]],
    known_queries: set[str],
    known_sources: frozenset[str],
    *,
    run_granularity: str,
) -> tuple[dict[str, dict[str, float]], int]:
    source_scores: dict[str, dict[str, float]] = defaultdict(dict)
    row_count = 0
    for line_number, record in _jsonl(path):
        context = f"run line {line_number}"
        query_id = _required_text(record, "query_id", context)
        score = _finite_number(record, "score", context)
        if query_id not in known_queries:
            raise ValueError(f"{context}: query_id {query_id!r} has no qrels")
        if run_granularity == "chunk":
            corpus_id = _required_text(record, "corpus_id", context)
            source_ids = sources_by_corpus.get(corpus_id)
            if source_ids is None:
                raise ValueError(f"{context}: unknown corpus_id {corpus_id!r}")
        else:
            source_id = _required_text(record, "source_id", context)
            if source_id not in known_sources:
                raise ValueError(f"{context}: unknown source_id {source_id!r}")
            source_ids = (source_id,)
        for source_id in source_ids:
            source_scores[query_id][source_id] = max(
                score, source_scores[query_id].get(source_id, -math.inf)
            )
        row_count += 1
    if not row_count:
        raise ValueError("retrieval run contains no rows")
    return dict(source_scores), row_count


def _metrics_at_k(
    ranked_sources: list[str], qrels: dict[str, float], k: int
) -> dict[str, float]:
    retrieved = ranked_sources[:k]
    relevant = set(qrels)
    relevant_ranks = [
        rank for rank, source_id in enumerate(retrieved, 1) if source_id in relevant
    ]
    hit_count = len(relevant_ranks)
    recall = hit_count / len(relevant)
    perfect_recall_ceiling = min(k, len(relevant)) / len(relevant)
    capped_recall = hit_count / min(k, len(relevant))
    first_relevant = next(
        iter(relevant_ranks),
        None,
    )
    reciprocal_rank = 0.0 if first_relevant is None else 1.0 / first_relevant
    dcg = sum(
        qrels.get(source_id, 0.0) / math.log2(rank + 1)
        for rank, source_id in enumerate(retrieved, 1)
    )
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(
        relevance / math.log2(rank + 1) for rank, relevance in enumerate(ideal, 1)
    )
    average_precision = sum(
        seen / rank for seen, rank in enumerate(relevant_ranks, 1)
    ) / min(k, len(relevant))
    return {
        "average_precision": average_precision,
        "capped_recall": capped_recall,
        "hit_rate": float(hit_count > 0),
        "mrr": reciprocal_rank,
        "ndcg": dcg / idcg,
        "perfect_recall_ceiling": perfect_recall_ceiling,
        "precision": hit_count / k,
        "recall": recall,
    }


def evaluate_source_retrieval(
    run_path: Path,
    source_qrels_path: Path,
    occurrences_path: Path,
    *,
    k_values: tuple[int, ...] = (1, 5, 10, 20, 100),
    allow_deduplicated_aliases: bool = False,
    run_granularity: str = "chunk",
    require_complete_k: tuple[int, ...] = (),
    collapsed_run_output: Path | None = None,
    per_query_output: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Collapse chunk scores to sources and compute macro document metrics."""

    if not k_values or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in k_values
    ):
        raise ValueError("k_values must contain positive integers")
    chosen_k = tuple(sorted(set(k_values)))
    if run_granularity not in {"chunk", "source"}:
        raise ValueError("run_granularity must be 'chunk' or 'source'")
    if any(value not in chosen_k for value in require_complete_k):
        raise ValueError("require_complete_k must be a subset of k_values")
    sources_by_corpus, occurrence_count, alias_count, known_sources = _load_occurrences(
        occurrences_path,
        allow_deduplicated_aliases=allow_deduplicated_aliases,
    )
    qrels = _load_qrels(source_qrels_path)
    source_scores, input_run_row_count = _collapse_run(
        run_path,
        sources_by_corpus,
        set(qrels),
        known_sources,
        run_granularity=run_granularity,
    )

    metric_names = (
        "average_precision",
        "capped_recall",
        "hit_rate",
        "mrr",
        "ndcg",
        "perfect_recall_ceiling",
        "precision",
        "recall",
    )
    totals = {k: {metric: 0.0 for metric in metric_names} for k in chosen_k}
    candidate_counts: list[int] = []
    relevant_counts: list[int] = []
    collapsed_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []
    source_run_row_count = 0
    for query_id in sorted(qrels):
        scores = source_scores.get(query_id, {})
        ranked = [
            source_id
            for source_id, _score in sorted(
                scores.items(), key=lambda item: (-item[1], item[0])
            )
        ]
        candidate_counts.append(len(ranked))
        relevant_counts.append(len(qrels[query_id]))
        source_run_row_count += len(ranked)
        if collapsed_run_output is not None:
            collapsed_rows.extend(
                {
                    "query_id": query_id,
                    "rank": rank,
                    "score": scores[source_id],
                    "source_id": source_id,
                }
                for rank, source_id in enumerate(ranked, 1)
            )
        query_metrics = {
            str(k): _metrics_at_k(ranked, qrels[query_id], k) for k in chosen_k
        }
        for k in chosen_k:
            for metric, value in query_metrics[str(k)].items():
                totals[k][metric] += value
        if per_query_output is not None:
            per_query_rows.append(
                {
                    "candidate_count": len(ranked),
                    "metrics": query_metrics,
                    "query_id": query_id,
                    "relevant_document_count": len(qrels[query_id]),
                }
            )

    query_count = len(qrels)
    candidate_completeness = {}
    for k in chosen_k:
        required = min(k, len(known_sources))
        complete_queries = sum(count >= required for count in candidate_counts)
        candidate_completeness[str(k)] = {
            "complete": complete_queries == query_count,
            "complete_queries": complete_queries,
            "incomplete_queries": query_count - complete_queries,
            "required_candidates_per_query": required,
        }
        if k in require_complete_k and complete_queries != query_count:
            raise ValueError(
                f"retrieval run does not contain complete source candidates at k={k}: "
                f"{query_count - complete_queries}/{query_count} queries are incomplete"
            )
    aggregated_metrics = {}
    for k in chosen_k:
        values = {
            metric: total / query_count for metric, total in sorted(totals[k].items())
        }
        ceiling = values["perfect_recall_ceiling"]
        values["ceiling_normalized_recall"] = values["recall"] / ceiling
        aggregated_metrics[str(k)] = dict(sorted(values.items()))

    result: dict[str, Any] = {
        "schema_version": 2,
        "producer_sha256": _sha256(Path(__file__).resolve()),
        "metric_semantics": {
            "collapse": "maximum chunk score per source",
            "ndcg_gain": "linear relevance",
            "recall": "binary positive relevance",
            "capped_recall": "binary hits divided by min(k, relevant documents)",
            "perfect_recall_ceiling": "macro mean of min(k, relevant) / relevant",
            "ceiling_normalized_recall": "macro recall divided by macro perfect ceiling",
            "precision": "binary positive relevance with denominator k",
            "average_precision": "binary AP normalized by min(k, relevant documents)",
            "hit_rate": "at least one positive relevance",
            "mrr": "first positive relevance",
            "aggregation": "macro mean over every qrels query",
            "tie_break": "source_id ascending",
        },
        "run_path": str(run_path.resolve()),
        "run_sha256": _sha256(run_path),
        "source_qrels_path": str(source_qrels_path.resolve()),
        "source_qrels_sha256": _sha256(source_qrels_path),
        "occurrences_path": str(occurrences_path.resolve()),
        "occurrences_sha256": _sha256(occurrences_path),
        "query_count": query_count,
        "run_granularity": run_granularity,
        "occurrence_count": occurrence_count,
        "source_count": len(known_sources),
        "deduplicated_alias_count": alias_count,
        "diagnostic_alias_expansion": allow_deduplicated_aliases,
        "input_run_row_count": input_run_row_count,
        "source_run_row_count": source_run_row_count,
        "source_candidates_per_query": _histogram(candidate_counts),
        "relevant_documents_per_query": _histogram(relevant_counts),
        "candidate_completeness": candidate_completeness,
        "metrics": aggregated_metrics,
        "limitations": [
            "document qrels do not measure citation-span precision",
            "latency and generator answer quality are outside this evaluator",
        ],
    }
    if allow_deduplicated_aliases and alias_count:
        result["limitations"].append(
            "diagnostic alias expansion credits every source sharing a retrieved vector"
        )
    incomplete_k = [
        k for k in chosen_k if not candidate_completeness[str(k)]["complete"]
    ]
    if incomplete_k:
        rendered = ", ".join(f"@{k}" for k in incomplete_k)
        result["limitations"].append(
            f"source candidate depth is incomplete for metrics {rendered}; "
            "those values are lower-bound diagnostics"
        )
    if collapsed_run_output is not None:
        _atomic_jsonl(collapsed_rows, collapsed_run_output)
        result["collapsed_run"] = {
            "path": str(collapsed_run_output.resolve()),
            "sha256": _sha256(collapsed_run_output),
            "rows": len(collapsed_rows),
        }
    if per_query_output is not None:
        _atomic_jsonl(per_query_rows, per_query_output)
        result["per_query_diagnostics"] = {
            "path": str(per_query_output.resolve()),
            "sha256": _sha256(per_query_output),
            "rows": len(per_query_rows),
        }
    if output_path is not None:
        result["output_path"] = str(output_path.resolve())
        _atomic_json(result, output_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-qrels", type=Path, required=True)
    parser.add_argument("--occurrences-jsonl", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20, 100])
    parser.add_argument("--allow-deduplicated-aliases", action="store_true")
    parser.add_argument(
        "--run-granularity", choices=("chunk", "source"), default="chunk"
    )
    parser.add_argument("--require-complete-k", type=int, nargs="*", default=[])
    parser.add_argument("--collapsed-run-output", type=Path)
    parser.add_argument("--per-query-output", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_source_retrieval(
        args.run,
        args.source_qrels,
        args.occurrences_jsonl,
        k_values=tuple(args.k),
        allow_deduplicated_aliases=args.allow_deduplicated_aliases,
        run_granularity=args.run_granularity,
        require_complete_k=tuple(args.require_complete_k),
        collapsed_run_output=args.collapsed_run_output,
        per_query_output=args.per_query_output,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
