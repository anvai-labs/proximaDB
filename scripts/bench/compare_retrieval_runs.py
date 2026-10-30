#!/usr/bin/env python3
"""Compare query-aligned retrieval metrics with a deterministic paired bootstrap."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

# corpus_io is a sibling module in scripts/bench/. Inject that directory so the
# import works both when run as a script and when loaded via importlib in tests.
_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from corpus_io import (  # noqa: E402  — path-injected sibling bench module
    atomic_json_write as _atomic_json,
    sha256_file as _sha256,
)


def _load_metric(path: Path, *, k: int, metric: str) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path} line {line_number}: expected an object")
            query_id = record.get("query_id")
            if not isinstance(query_id, str) or not query_id.strip():
                raise ValueError(f"{path} line {line_number}: query_id is required")
            if query_id in values:
                raise ValueError(
                    f"{path} line {line_number}: duplicate query {query_id!r}"
                )
            metrics = record.get("metrics")
            at_k = metrics.get(str(k)) if isinstance(metrics, dict) else None
            value = at_k.get(metric) if isinstance(at_k, dict) else None
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"{path} line {line_number}: metric {metric!r} at k={k} "
                    "must be finite"
                )
            values[query_id] = float(value)
    if not values:
        raise ValueError(f"{path} contains no per-query metrics")
    return values


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def compare_runs(
    baseline_path: Path,
    candidate_path: Path,
    *,
    k: int | None = None,
    baseline_k: int | None = None,
    candidate_k: int | None = None,
    metric: str,
    bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 17,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Return a paired delta and percentile confidence interval over queries."""

    if k is not None and (baseline_k is not None or candidate_k is not None):
        raise ValueError("use either k or the baseline_k/candidate_k pair")
    if k is not None:
        baseline_k = candidate_k = k
    if baseline_k is None or candidate_k is None:
        raise ValueError("k or both baseline_k and candidate_k are required")
    for name, depth in (("baseline_k", baseline_k), ("candidate_k", candidate_k)):
        if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not metric.strip():
        raise ValueError("metric is required")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples <= 0
    ):
        raise ValueError("bootstrap_samples must be a positive integer")
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")

    baseline = _load_metric(baseline_path, k=baseline_k, metric=metric)
    candidate = _load_metric(candidate_path, k=candidate_k, metric=metric)
    if set(baseline) != set(candidate):
        missing = sorted(set(baseline) - set(candidate))
        extra = sorted(set(candidate) - set(baseline))
        raise ValueError(
            f"candidate query coverage differs: missing={missing[:3]} extra={extra[:3]}"
        )

    query_ids = sorted(baseline)
    deltas = [candidate[query_id] - baseline[query_id] for query_id in query_ids]
    query_count = len(query_ids)
    random_source = random.Random(seed)
    bootstrap_means = [
        sum(deltas[random_source.randrange(query_count)] for _ in range(query_count))
        / query_count
        for _ in range(bootstrap_samples)
    ]
    tail = (1.0 - confidence) / 2.0
    mean_delta = sum(deltas) / query_count
    result = {
        "schema_version": 2,
        "producer_sha256": _sha256(Path(__file__).resolve()),
        "baseline_path": str(baseline_path.resolve()),
        "baseline_sha256": _sha256(baseline_path),
        "candidate_path": str(candidate_path.resolve()),
        "candidate_sha256": _sha256(candidate_path),
        "k": baseline_k if baseline_k == candidate_k else None,
        "baseline_k": baseline_k,
        "candidate_k": candidate_k,
        "metric": metric,
        "query_count": query_count,
        "baseline_mean": sum(baseline.values()) / query_count,
        "candidate_mean": sum(candidate.values()) / query_count,
        "mean_delta": mean_delta,
        "relative_delta": (
            mean_delta / (sum(baseline.values()) / query_count)
            if any(baseline.values())
            else None
        ),
        "paired_outcomes": {
            "candidate_wins": sum(delta > 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "losses": sum(delta < 0 for delta in deltas),
        },
        "confidence_interval": {
            "method": "paired query bootstrap percentile",
            "confidence": confidence,
            "low": _percentile(bootstrap_means, tail),
            "high": _percentile(bootstrap_means, 1.0 - tail),
            "samples": bootstrap_samples,
            "seed": seed,
        },
        "interpretation": (
            "interval_excludes_zero"
            if _percentile(bootstrap_means, tail) > 0
            or _percentile(bootstrap_means, 1.0 - tail) < 0
            else "interval_includes_zero"
        ),
    }
    if output_path is not None:
        result["output_path"] = str(output_path.resolve())
        _atomic_json(result, output_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--k", type=int)
    parser.add_argument("--baseline-k", type=int)
    parser.add_argument("--candidate-k", type=int)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            compare_runs(
                args.baseline,
                args.candidate,
                k=args.k,
                baseline_k=args.baseline_k,
                candidate_k=args.candidate_k,
                metric=args.metric,
                bootstrap_samples=args.bootstrap_samples,
                confidence=args.confidence,
                seed=args.seed,
                output_path=args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
