#!/usr/bin/env python3
"""Fuse sealed source-level retrieval runs with deterministic weighted RRF."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

# corpus_io is a sibling module in scripts/bench/. Inject that directory so the
# import works both when run as a script and when loaded via importlib in tests.
_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from corpus_io import (  # noqa: E402  — path-injected sibling bench module
    atomic_json_write as _atomic_json,
    atomic_jsonl_write as _atomic_jsonl,
    sha256_file as _sha256,
)


def _load_run(path: Path) -> dict[str, dict[str, int]]:
    rankings: dict[str, dict[str, int]] = defaultdict(dict)
    ranks_by_query: dict[str, set[int]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} line {line_number}: expected an object")
            query_id = value.get("query_id")
            source_id = value.get("source_id")
            rank = value.get("rank")
            if not isinstance(query_id, str) or not query_id.strip():
                raise ValueError(f"{path} line {line_number}: query_id is required")
            if not isinstance(source_id, str) or not source_id.strip():
                raise ValueError(f"{path} line {line_number}: source_id is required")
            if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
                raise ValueError(f"{path} line {line_number}: rank must be positive")
            if source_id in rankings[query_id]:
                raise ValueError(
                    f"{path} line {line_number}: duplicate source {source_id!r} "
                    f"for query {query_id!r}"
                )
            if rank in ranks_by_query[query_id]:
                raise ValueError(
                    f"{path} line {line_number}: duplicate rank {rank} "
                    f"for query {query_id!r}"
                )
            rankings[query_id][source_id] = rank
            ranks_by_query[query_id].add(rank)
    if not rankings:
        raise ValueError(f"{path} contains no run rows")
    for query_id, ranks in ranks_by_query.items():
        expected = set(range(1, len(ranks) + 1))
        if ranks != expected:
            raise ValueError(f"{path}: ranks for query {query_id!r} are not contiguous")
    return dict(rankings)


def _load_and_validate_manifest(
    manifest_path: Path,
    run_path: Path,
    run: dict[str, dict[str, int]],
) -> dict[str, Any]:
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{manifest_path}: expected a JSON object")
    expected_sha256 = value.get("run_sha256")
    actual_sha256 = _sha256(run_path)
    if expected_sha256 != actual_sha256:
        raise ValueError(f"{manifest_path}: run_sha256 does not match {run_path}")
    declared_path = value.get("run_path")
    if (
        not isinstance(declared_path, str)
        or Path(declared_path).resolve() != run_path.resolve()
    ):
        raise ValueError(f"{manifest_path}: run_path does not match {run_path}")
    query_count = value.get("query_count")
    if query_count != len(run):
        raise ValueError(f"{manifest_path}: query_count does not match the run")
    row_count = sum(len(ranking) for ranking in run.values())
    if value.get("run_row_count") != row_count:
        raise ValueError(f"{manifest_path}: run_row_count does not match the run")
    granularity = value.get("candidate_granularity")
    if granularity is not None and granularity != "source":
        raise ValueError(f"{manifest_path}: candidate_granularity must be source")
    declared_top_k = value.get("top_k")
    if declared_top_k is not None:
        if isinstance(declared_top_k, bool) or not isinstance(declared_top_k, int):
            raise ValueError(f"{manifest_path}: top_k must be an integer")
        if any(len(ranking) != declared_top_k for ranking in run.values()):
            raise ValueError(f"{manifest_path}: top_k does not match every query depth")
    return value


def _percentile(values: Sequence[int], fraction: float) -> int:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def fuse_source_runs(
    run_paths: Sequence[Path],
    output_path: Path,
    *,
    labels: Sequence[str],
    top_k: int,
    rrf_k: int = 60,
    weights: Sequence[float] | None = None,
    manifest_path: Path | None = None,
    manifest_paths: Sequence[Path] | None = None,
    require_complete_top_k: bool = False,
) -> dict[str, Any]:
    """Fuse runs by rank, requiring identical query coverage across every leg."""

    if len(run_paths) < 2:
        raise ValueError("at least two source runs are required")
    if len(labels) != len(run_paths) or any(not label.strip() for label in labels):
        raise ValueError("labels must contain one non-empty value per run")
    if len(set(labels)) != len(labels):
        raise ValueError("labels must be unique")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 0:
        raise ValueError("rrf_k must be a non-negative integer")
    chosen_weights = tuple(weights or (1.0,) * len(run_paths))
    if len(chosen_weights) != len(run_paths) or any(
        not math.isfinite(weight) or weight <= 0 for weight in chosen_weights
    ):
        raise ValueError("weights must contain one finite positive value per run")
    chosen_manifests = tuple(manifest_paths or ())
    if chosen_manifests and len(chosen_manifests) != len(run_paths):
        raise ValueError("manifest_paths must contain one value per run")

    runs = [_load_run(path) for path in run_paths]
    validated_manifests = (
        [
            _load_and_validate_manifest(path, run_path, run)
            for path, run_path, run in zip(
                chosen_manifests, run_paths, runs, strict=True
            )
        ]
        if chosen_manifests
        else []
    )
    query_ids = set(runs[0])
    for label, run in zip(labels[1:], runs[1:], strict=True):
        if set(run) != query_ids:
            missing = sorted(query_ids - set(run))
            extra = sorted(set(run) - query_ids)
            raise ValueError(
                f"run {label!r} query coverage differs: missing={missing[:3]} "
                f"extra={extra[:3]}"
            )

    rows: list[dict[str, Any]] = []
    candidate_counts: list[int] = []
    for query_id in sorted(query_ids):
        fused: dict[str, float] = defaultdict(float)
        for run, weight in zip(runs, chosen_weights, strict=True):
            for source_id, rank in run[query_id].items():
                fused[source_id] += weight / (rrf_k + rank)
        ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        candidate_counts.append(len(ranked))
        rows.extend(
            {
                "query_id": query_id,
                "rank": rank,
                "score": score,
                "source_id": source_id,
            }
            for rank, (source_id, score) in enumerate(ranked, 1)
        )

    complete_query_count = sum(count >= top_k for count in candidate_counts)
    if require_complete_top_k and complete_query_count != len(query_ids):
        raise ValueError(
            f"{len(query_ids) - complete_query_count} queries have fewer than required "
            f"top_k={top_k} candidates"
        )

    _atomic_jsonl(rows, output_path)
    manifest_path = manifest_path or output_path.with_name(
        f"{output_path.stem}.fusion.manifest.json"
    )
    result = {
        "schema_version": 2,
        "producer_sha256": _sha256(Path(__file__).resolve()),
        "fusion": "reciprocal_rank_fusion",
        "parameters": {"rrf_k": rrf_k},
        "tie_break": "source_id ascending",
        "top_k": top_k,
        "query_count": len(query_ids),
        "candidate_count": {
            "min": min(candidate_counts),
            "p50": _percentile(candidate_counts, 0.50),
            "p90": _percentile(candidate_counts, 0.90),
            "p99": _percentile(candidate_counts, 0.99),
            "max": max(candidate_counts),
        },
        "complete_query_count": complete_query_count,
        "require_complete_top_k": require_complete_top_k,
        "input_manifests_validated": bool(validated_manifests),
        "inputs": [
            {
                "label": label,
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "weight": weight,
                **(
                    {
                        "manifest_path": str(input_manifest_path.resolve()),
                        "manifest_sha256": _sha256(input_manifest_path),
                    }
                    if chosen_manifests
                    else {}
                ),
            }
            for label, path, weight, input_manifest_path in zip(
                labels,
                run_paths,
                chosen_weights,
                chosen_manifests or (None,) * len(run_paths),
                strict=True,
            )
        ],
        "run_path": str(output_path.resolve()),
        "run_sha256": _sha256(output_path),
        "run_row_count": len(rows),
    }
    _atomic_json(result, manifest_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--run-manifest", type=Path, action="append")
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--weight", type=float, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--require-complete-top-k", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = fuse_source_runs(
        args.run,
        args.output,
        labels=args.label,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        weights=args.weight,
        manifest_path=args.manifest,
        manifest_paths=args.run_manifest,
        require_complete_top_k=args.require_complete_top_k,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
