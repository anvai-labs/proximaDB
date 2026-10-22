#!/usr/bin/env python3
"""Score and materialize source-aware cross-encoder reranking runs.

The scorer never asks the model tokenizer to truncate. Long source documents
are split into query-aware token windows, each window is scored, and the
source score is the maximum window score. Expensive model scores are cached
per query so rerank-depth policies can be swept without repeating inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

# corpus_io is a sibling module in scripts/bench/. Inject that directory so the
# import works both when run as a script and when loaded via importlib in tests.
_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from corpus_io import (  # noqa: E402  — path-injected sibling bench module
    atomic_json_write as _atomic_json,
    atomic_jsonl_write as _atomic_jsonl,
    finite as _finite,
    iter_jsonl as _jsonl,
    required_text as _required_text,
    sha256_file as _sha256,
)


@dataclass(frozen=True)
class PairWindow:
    text: str
    body_token_start: int
    body_token_end: int
    input_tokens: int


class PairTokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool) -> str: ...

    def num_special_tokens_to_add(self, *, pair: bool) -> int: ...


class PairScorer(Protocol):
    tokenizer: PairTokenizer

    def predict(
        self, pairs: list[tuple[str, str]], *, batch_size: int
    ) -> list[float]: ...


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pair_length(tokenizer: PairTokenizer, query: str, document: str) -> int:
    return (
        len(tokenizer.encode(query, add_special_tokens=False))
        + len(tokenizer.encode(document, add_special_tokens=False))
        + tokenizer.num_special_tokens_to_add(pair=True)
    )


def build_pair_windows(
    tokenizer: PairTokenizer,
    *,
    query: str,
    title: str,
    body: str,
    max_length: int,
    overlap_tokens: int,
) -> list[PairWindow]:
    """Cover every body token with pair inputs that fit ``max_length``.

    Title context is propagated to every body window. A title that leaves no
    body capacity is rejected explicitly instead of silently dropping text.
    """

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be non-negative")
    query = query.strip()
    title = title.strip()
    body = body.strip()
    if not query:
        raise ValueError("query is required")
    if not title and not body:
        raise ValueError("document title or body is required")

    special = tokenizer.num_special_tokens_to_add(pair=True)
    query_tokens = tokenizer.encode(query, add_special_tokens=False)
    document_budget = max_length - special - len(query_tokens)
    if document_budget <= 0:
        raise ValueError(
            f"query requires {len(query_tokens) + special} tokens, leaving no document budget"
        )

    prefix = f"{title}\n\n" if title and body else title
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    if body and len(prefix_tokens) >= document_budget:
        raise ValueError(
            f"title context requires {len(prefix_tokens)} of {document_budget} document tokens"
        )
    if not body:
        input_tokens = _pair_length(tokenizer, query, prefix)
        if input_tokens > max_length:
            raise ValueError(
                f"title-only document requires {input_tokens} tokens, limit is {max_length}"
            )
        return [PairWindow(prefix, 0, 0, input_tokens)]

    body_tokens = tokenizer.encode(body, add_special_tokens=False)
    body_budget = document_budget - len(prefix_tokens)
    if overlap_tokens >= body_budget:
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) must be below body capacity ({body_budget})"
        )

    windows: list[PairWindow] = []
    start = 0
    while start < len(body_tokens):
        end = min(start + body_budget, len(body_tokens))
        while end > start:
            decoded = tokenizer.decode(
                body_tokens[start:end], skip_special_tokens=True
            ).strip()
            document = f"{prefix}{decoded}"
            input_tokens = _pair_length(tokenizer, query, document)
            if input_tokens <= max_length:
                break
            end -= 1
        if end <= start:
            raise ValueError(
                f"could not fit body token {start} inside max_length={max_length}"
            )
        windows.append(PairWindow(document, start, end, input_tokens))
        if end == len(body_tokens):
            break
        next_start = end - overlap_tokens
        if next_start <= start:
            raise ValueError("windowing did not make forward progress")
        start = next_start

    if windows[0].body_token_start != 0 or windows[-1].body_token_end != len(
        body_tokens
    ):
        raise AssertionError("body token coverage is incomplete")
    if any(window.input_tokens > max_length for window in windows):
        raise AssertionError("tokenizer would truncate a generated pair window")
    return windows


def _load_documents(path: Path) -> dict[str, dict[str, str]]:
    documents: dict[str, dict[str, str]] = {}
    for line_number, record in _jsonl(path):
        context = f"documents line {line_number}"
        source_id = _required_text(record, "id", context)
        if source_id in documents:
            raise ValueError(f"{context}: duplicate id {source_id!r}")
        title = record.get("title", "")
        body = record.get("body", record.get("text", ""))
        if not isinstance(title, str) or not isinstance(body, str):
            raise ValueError(f"{context}: title and body must be strings")
        if not title.strip() and not body.strip():
            raise ValueError(f"{context}: title or body is required")
        documents[source_id] = {"title": title, "body": body}
    if not documents:
        raise ValueError("documents input is empty")
    return documents


def _load_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    for line_number, record in _jsonl(path):
        context = f"queries line {line_number}"
        query_id = _required_text(record, "id", context)
        if query_id in queries:
            raise ValueError(f"{context}: duplicate id {query_id!r}")
        queries[query_id] = _required_text(record, "text", context)
    if not queries:
        raise ValueError("queries input is empty")
    return queries


def _load_run(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for line_number, record in _jsonl(path):
        context = f"run line {line_number}"
        query_id = _required_text(record, "query_id", context)
        source_id = _required_text(record, "source_id", context)
        rank = record.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"{context}: rank must be a positive integer")
        score = _finite(record.get("score"), f"{context} score")
        candidates = grouped.setdefault(query_id, [])
        if rank != len(candidates) + 1:
            raise ValueError(f"{context}: ranks must be contiguous from one")
        candidates.append(
            {"source_id": source_id, "baseline_rank": rank, "baseline_score": score}
        )
    if not grouped:
        raise ValueError("baseline run is empty")
    return grouped


def _query_cache_path(cache_dir: Path, query_id: str) -> Path:
    name = hashlib.sha256(query_id.encode("utf-8")).hexdigest()
    return cache_dir / "queries" / f"{name}.json"


def score_query(
    scorer: PairScorer,
    *,
    query_id: str,
    query: str,
    candidates: Sequence[dict[str, Any]],
    documents: dict[str, dict[str, str]],
    max_length: int,
    overlap_tokens: int,
    batch_size: int,
    contract_fingerprint: str,
) -> dict[str, Any]:
    pairs: list[tuple[str, str]] = []
    owners: list[tuple[int, PairWindow]] = []
    candidate_records: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        source_id = candidate["source_id"]
        document = documents.get(source_id)
        if document is None:
            raise ValueError(f"query {query_id!r}: unknown source {source_id!r}")
        windows = build_pair_windows(
            scorer.tokenizer,
            query=query,
            title=document["title"],
            body=document["body"],
            max_length=max_length,
            overlap_tokens=overlap_tokens,
        )
        candidate_records.append(
            {
                **candidate,
                "window_count": len(windows),
                "input_token_count": sum(window.input_tokens for window in windows),
                "max_input_tokens": max(window.input_tokens for window in windows),
            }
        )
        for window in windows:
            pairs.append((query, window.text))
            owners.append((candidate_index, window))

    raw_scores = scorer.predict(pairs, batch_size=batch_size)
    if len(raw_scores) != len(pairs):
        raise ValueError(
            f"scorer returned {len(raw_scores)} scores for {len(pairs)} pairs"
        )
    best: list[tuple[float, PairWindow] | None] = [None] * len(candidate_records)
    for raw_score, (candidate_index, window) in zip(raw_scores, owners, strict=True):
        score = _finite(raw_score, "reranker score")
        current = best[candidate_index]
        if current is None or score > current[0]:
            best[candidate_index] = (score, window)
    for candidate, winner in zip(candidate_records, best, strict=True):
        if winner is None:
            raise AssertionError("candidate had no scored windows")
        score, window = winner
        candidate["rerank_score"] = score
        candidate["winning_window"] = asdict(window)

    return {
        "schema_version": 1,
        "contract_fingerprint": contract_fingerprint,
        "query_id": query_id,
        "candidate_count": len(candidate_records),
        "window_count": len(pairs),
        "zero_truncation_asserted": all(
            candidate["max_input_tokens"] <= max_length
            for candidate in candidate_records
        ),
        "candidates": candidate_records,
    }


class HFCrossEncoderScorer:
    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        device: str,
        max_length: int,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(
            model_id,
            revision=revision,
            device=device,
            max_length=max_length,
        )
        self.tokenizer = self._model.tokenizer

    def predict(self, pairs: list[tuple[str, str]], *, batch_size: int) -> list[float]:
        values = self._model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(value) for value in values.reshape(-1)]


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def score_cache(args: argparse.Namespace) -> dict[str, Any]:
    rerank_all = bool(getattr(args, "rerank_all", False))
    max_rerank = None if rerank_all else args.max_rerank
    if (max_rerank is not None and max_rerank <= 0) or args.batch_size <= 0:
        raise ValueError("max-rerank and batch-size must be positive")
    documents = _load_documents(args.documents)
    queries = _load_queries(args.queries)
    run = _load_run(args.run)
    if set(run) != set(queries):
        raise ValueError("baseline run and query IDs differ")
    if max_rerank is not None and any(
        len(candidates) < max_rerank for candidates in run.values()
    ):
        raise ValueError("baseline run has fewer candidates than max-rerank")

    producer_sha256 = _sha256(Path(__file__).resolve())
    contract = {
        "producer_sha256": producer_sha256,
        "model": {"id": args.model, "revision": args.revision},
        "rerank_policy": (
            {"mode": "all_candidates"}
            if max_rerank is None
            else {"mode": "prefix", "count": max_rerank}
        ),
        "max_length": args.max_length,
        "overlap_tokens": args.overlap_tokens,
        "documents_sha256": _sha256(args.documents),
        "queries_sha256": _sha256(args.queries),
        "run_sha256": _sha256(args.run),
    }
    contract_fingerprint = _fingerprint(contract)
    existing_manifest_path = args.output / "score-cache.manifest.json"
    if existing_manifest_path.exists():
        existing_manifest = json.loads(
            existing_manifest_path.read_text(encoding="utf-8")
        )
        if existing_manifest.get("contract_fingerprint") != contract_fingerprint:
            raise ValueError(
                "score cache contract differs; use a new output directory instead of mixing evidence"
            )

    device = _resolve_device(args.device)
    scorer = HFCrossEncoderScorer(
        args.model, args.revision, device=device, max_length=args.max_length
    )
    started = time.monotonic()
    scored = 0
    ordered_query_ids = sorted(queries)
    if args.query_limit is not None:
        ordered_query_ids = ordered_query_ids[: args.query_limit]
    for query_id in ordered_query_ids:
        output_path = _query_cache_path(args.output, query_id)
        if output_path.exists():
            cached = json.loads(output_path.read_text(encoding="utf-8"))
            if cached.get("query_id") != query_id:
                raise ValueError(f"cache identity mismatch at {output_path}")
            if cached.get("contract_fingerprint") != contract_fingerprint:
                raise ValueError(
                    f"cache contract mismatch at {output_path}; use a new output directory"
                )
            continue
        result = score_query(
            scorer,
            query_id=query_id,
            query=queries[query_id],
            candidates=(
                run[query_id] if max_rerank is None else run[query_id][:max_rerank]
            ),
            documents=documents,
            max_length=args.max_length,
            overlap_tokens=args.overlap_tokens,
            batch_size=args.batch_size,
            contract_fingerprint=contract_fingerprint,
        )
        _atomic_json(result, output_path)
        scored += 1
        if scored == 1 or scored % 10 == 0:
            print(f"scored {scored} new queries ({query_id})", flush=True)

    cache_files = [
        _query_cache_path(args.output, query_id) for query_id in ordered_query_ids
    ]
    records = [json.loads(path.read_text(encoding="utf-8")) for path in cache_files]
    if not all(record.get("zero_truncation_asserted") is True for record in records):
        raise AssertionError("at least one cached query would truncate")
    manifest = {
        "schema_version": 1,
        "producer_sha256": producer_sha256,
        "contract": contract,
        "contract_fingerprint": contract_fingerprint,
        "model": {"id": args.model, "revision": args.revision},
        "tokenization": {
            "max_length": args.max_length,
            "overlap_tokens": args.overlap_tokens,
            "overflow_policy": "split_source_body_and_max_reduce_windows",
            "title_policy": "propagate_to_every_window_or_fail_if_it_cannot_fit",
            "zero_truncation_asserted": True,
        },
        "execution": {"device": device, "batch_size": args.batch_size},
        "inputs": {
            "documents": str(args.documents.resolve()),
            "documents_sha256": _sha256(args.documents),
            "queries": str(args.queries.resolve()),
            "queries_sha256": _sha256(args.queries),
            "run": str(args.run.resolve()),
            "run_sha256": _sha256(args.run),
        },
        "query_count": len(records),
        "candidate_count_per_query": {
            "min": min(record["candidate_count"] for record in records),
            "max": max(record["candidate_count"] for record in records),
        },
        "candidate_count": sum(record["candidate_count"] for record in records),
        "window_count": sum(record["window_count"] for record in records),
        "input_token_count": sum(
            candidate["input_token_count"]
            for record in records
            for candidate in record["candidates"]
        ),
        "max_input_tokens": max(
            candidate["max_input_tokens"]
            for record in records
            for candidate in record["candidates"]
        ),
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "new_queries_this_invocation": scored,
    }
    _atomic_json(manifest, args.output / "score-cache.manifest.json")
    return manifest


def materialize_run(
    baseline: dict[str, list[dict[str, Any]]],
    cached: dict[str, dict[str, Any]],
    *,
    rerank_count: int | None,
) -> list[dict[str, Any]]:
    if rerank_count is not None and rerank_count <= 0:
        raise ValueError("rerank_count must be positive")
    output: list[dict[str, Any]] = []
    for query_id, candidates in baseline.items():
        record = cached.get(query_id)
        if record is None:
            raise ValueError(f"missing cached scores for query {query_id!r}")
        effective_rerank_count = (
            len(candidates) if rerank_count is None else rerank_count
        )
        score_rows = record.get("candidates")
        if not isinstance(score_rows, list) or len(score_rows) < effective_rerank_count:
            raise ValueError(f"query {query_id!r} has insufficient cached candidates")
        if rerank_count is None and len(score_rows) != len(candidates):
            raise ValueError(f"query {query_id!r} cache does not cover every candidate")
        prefix_ids = [
            candidate["source_id"] for candidate in candidates[:effective_rerank_count]
        ]
        cached_ids = [
            candidate["source_id"] for candidate in score_rows[:effective_rerank_count]
        ]
        if prefix_ids != cached_ids:
            raise ValueError(f"query {query_id!r} cache does not match baseline prefix")
        reranked = sorted(
            score_rows[:effective_rerank_count],
            key=lambda row: (
                -_finite(row.get("rerank_score"), "rerank_score"),
                row["source_id"],
            ),
        )
        merged = [
            {
                "source_id": row["source_id"],
                "baseline_rank": row["baseline_rank"],
                "baseline_score": row["baseline_score"],
                "rerank_score": row["rerank_score"],
            }
            for row in reranked
        ]
        merged.extend(candidates[effective_rerank_count:])
        if {row["source_id"] for row in merged} != {
            row["source_id"] for row in candidates
        }:
            raise AssertionError("reranking changed candidate-set membership")
        for rank, row in enumerate(merged, 1):
            output.append(
                {
                    "query_id": query_id,
                    "source_id": row["source_id"],
                    "rank": rank,
                    "score": float(len(merged) - rank + 1),
                    "baseline_rank": row["baseline_rank"],
                    "baseline_score": row["baseline_score"],
                    **(
                        {"rerank_score": row["rerank_score"]}
                        if "rerank_score" in row
                        else {}
                    ),
                }
            )
    return output


def materialize_cache(args: argparse.Namespace) -> dict[str, Any]:
    baseline = _load_run(args.run)
    cached: dict[str, dict[str, Any]] = {}
    for query_id in baseline:
        path = _query_cache_path(args.cache, query_id)
        if not path.exists():
            raise ValueError(f"missing cache file for query {query_id!r}")
        cached[query_id] = json.loads(path.read_text(encoding="utf-8"))
    rerank_count = None if getattr(args, "rerank_all", False) else args.rerank_count
    records = materialize_run(baseline, cached, rerank_count=rerank_count)
    _atomic_jsonl(records, args.output)
    manifest = {
        "schema_version": 1,
        "producer_sha256": _sha256(Path(__file__).resolve()),
        "policy": (
            "rerank_all_candidates"
            if rerank_count is None
            else "rerank_prefix_then_append_untouched_tail"
        ),
        "score_field": "ordinal_rank_score; raw model value is rerank_score",
        "rerank_count": "all" if rerank_count is None else rerank_count,
        "query_count": len(baseline),
        "row_count": len(records),
        "candidate_membership_preserved": True,
        "recall_at_full_run_depth_invariant": True,
        "inputs": {
            "run": str(args.run.resolve()),
            "run_sha256": _sha256(args.run),
            "cache_manifest": str((args.cache / "score-cache.manifest.json").resolve()),
            "cache_manifest_sha256": _sha256(args.cache / "score-cache.manifest.json"),
        },
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
    }
    _atomic_json(
        manifest, args.output.with_suffix(args.output.suffix + ".manifest.json")
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score", help="score and cache query-source windows")
    score.add_argument("--documents", type=Path, required=True)
    score.add_argument("--queries", type=Path, required=True)
    score.add_argument("--run", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--model", required=True)
    score.add_argument("--revision", required=True)
    score.add_argument("--max-rerank", type=int, default=100)
    score.add_argument(
        "--rerank-all",
        action="store_true",
        help="score every candidate, including variable-depth query pools",
    )
    score.add_argument("--max-length", type=int, default=512)
    score.add_argument("--overlap-tokens", type=int, default=32)
    score.add_argument("--batch-size", type=int, default=32)
    score.add_argument("--device", default="auto")
    score.add_argument("--query-limit", type=int)

    materialize = subparsers.add_parser(
        "materialize", help="build a ranked run from cached model scores"
    )
    materialize.add_argument("--run", type=Path, required=True)
    materialize.add_argument("--cache", type=Path, required=True)
    materialize_selection = materialize.add_mutually_exclusive_group(required=True)
    materialize_selection.add_argument("--rerank-count", type=int)
    materialize_selection.add_argument(
        "--rerank-all",
        action="store_true",
        help="rerank every cached candidate for each query",
    )
    materialize.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = score_cache(args) if args.command == "score" else materialize_cache(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
