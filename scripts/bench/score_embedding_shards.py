#!/usr/bin/env python3
"""Create a sealed exact retrieval run from normalized float32 shards.

This scorer isolates embedding/chunking quality from ANN and uint8 transport
effects. It validates corpus and row-order lineage before using exact inner
product over the cached passage- and query-role embeddings.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# corpus_io is a sibling module in scripts/bench/. Inject that directory so the
# import works both when run as a script and when loaded via importlib in tests.
_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from corpus_io import (  # noqa: E402  — path-injected sibling bench module
    atomic_json_write as _atomic_json,
    required_text as _required_text,
    sha256_file as _sha256,
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_chunk_ids(path: Path) -> tuple[str, ...]:
    identifiers: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"chunks line {line_number}: expected an object")
            chunk_id = _required_text(record, "chunk_id", f"chunks line {line_number}")
            if chunk_id in seen:
                raise ValueError(f"chunks line {line_number}: duplicate chunk_id")
            seen.add(chunk_id)
            identifiers.append(chunk_id)
    if not identifiers:
        raise ValueError("chunks JSONL contains no rows")
    return tuple(identifiers)


def _load_source_mapping(
    path: Path, corpus_ids: tuple[str, ...]
) -> tuple[dict[str, tuple[str, ...]], int, int]:
    known_corpus = set(corpus_ids)
    sources_by_corpus: dict[str, set[str]] = {}
    source_ids: set[str] = set()
    alias_count = 0
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"occurrences line {line_number}: expected an object")
            corpus_id = _required_text(
                record, "corpus_id", f"occurrences line {line_number}"
            )
            source_id = _required_text(
                record, "source_id", f"occurrences line {line_number}"
            )
            if corpus_id not in known_corpus:
                raise ValueError(
                    f"occurrences line {line_number}: unknown corpus_id {corpus_id!r}"
                )
            sources_by_corpus.setdefault(corpus_id, set()).add(source_id)
            source_ids.add(source_id)
            if record.get("deduplicated_alias") is True:
                alias_count += 1
    missing = sorted(known_corpus - set(sources_by_corpus))
    if missing:
        raise ValueError(f"occurrences do not cover corpus IDs: {missing[:3]}")
    return (
        {
            corpus_id: tuple(sorted(source_ids_for_corpus))
            for corpus_id, source_ids_for_corpus in sources_by_corpus.items()
        },
        len(source_ids),
        alias_count,
    )


def _load_query_ids(path: Path) -> tuple[str, ...]:
    values = _json(path)
    if not isinstance(values, list) or not values:
        raise ValueError("query IDs must be a non-empty JSON array")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("query IDs must be non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError("query IDs must be unique")
    return tuple(values)


def _load_shard_identity(
    embedding_manifest: dict[str, Any], role: str
) -> tuple[Path, dict[str, Any]]:
    reference = embedding_manifest.get("shards", {}).get(role)
    if not isinstance(reference, dict):
        raise ValueError(f"embedding manifest has no {role} shard identity")
    path = Path(_required_text(reference, "manifest_path", f"{role} shards"))
    expected_sha = _required_text(reference, "manifest_sha256", f"{role} shards")
    if _sha256(path) != expected_sha:
        raise ValueError(f"{role} shard identity digest mismatch")
    identity = _json(path)
    if not isinstance(identity, dict):
        raise ValueError(f"{role} shard identity must be an object")
    return path, identity


def _validate_common_identity(
    embedding: dict[str, Any],
    passage: dict[str, Any],
    query: dict[str, Any],
) -> None:
    expected = {
        "model_id": embedding.get("model_id"),
        "model_revision": embedding.get("model_revision"),
        "contract_fingerprint": embedding.get("contract_fingerprint"),
        "dimension": embedding.get("dimension"),
        "runtime": embedding.get("runtime"),
        "normalize_embeddings": True,
        "artifact_dtype": "float32",
    }
    for role, input_role, identity in (
        ("passage", "document", passage),
        ("query", "query", query),
    ):
        for name, value in expected.items():
            if identity.get(name) != value:
                raise ValueError(f"{role} shard identity disagrees on {name}")
        if identity.get("input_role") != input_role:
            raise ValueError(f"{role} shard identity has the wrong input_role")


def _shard_paths(identity_path: Path, identity: dict[str, Any]) -> list[Path]:
    rows = identity.get("rows")
    shard_size = identity.get("shard_size")
    dimension = identity.get("dimension")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (rows, shard_size, dimension)
    ):
        raise ValueError(
            "shard identity rows, shard_size and dimension must be positive"
        )
    paths: list[Path] = []
    for start in range(0, rows, shard_size):
        path = identity_path.parent / f"{start:08d}.npy"
        if not path.exists():
            raise ValueError(f"embedding shard is missing: {path}")
        values = np.load(path, mmap_mode="r")
        expected = (min(shard_size, rows - start), dimension)
        if values.shape != expected or values.dtype != np.float32:
            raise ValueError(
                f"invalid embedding shard {path}: {values.shape} {values.dtype}; "
                f"expected {expected} float32"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"embedding shard contains non-finite values: {path}")
        paths.append(path)
    return paths


def _merge_top_k(
    previous: list[tuple[float, str]],
    scores: np.ndarray,
    corpus_ids: tuple[str, ...],
    top_k: int,
) -> list[tuple[float, str]]:
    candidates = previous + [
        (float(score), corpus_id)
        for score, corpus_id in zip(scores, corpus_ids, strict=True)
    ]
    if len(candidates) > top_k:
        numeric = np.fromiter((item[0] for item in candidates), dtype=np.float32)
        cutoff = float(
            np.partition(numeric, len(numeric) - top_k)[len(numeric) - top_k]
        )
        candidates = [item for item in candidates if item[0] >= cutoff]
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[:top_k]


def score_embedding_shards(
    embedding_manifest_path: Path,
    corpus_manifest_path: Path,
    chunks_path: Path,
    output_path: Path,
    *,
    top_k: int,
    candidate_granularity: str = "chunk",
    occurrences_path: Path | None = None,
    scoring_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Score every query exactly and emit deterministic ranked JSONL rows."""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if candidate_granularity not in {"chunk", "source"}:
        raise ValueError("candidate_granularity must be 'chunk' or 'source'")
    if candidate_granularity == "source" and occurrences_path is None:
        raise ValueError("source candidate scoring requires occurrences_path")
    if candidate_granularity == "chunk" and occurrences_path is not None:
        raise ValueError("occurrences_path is valid only for source candidate scoring")
    embedding = _json(embedding_manifest_path)
    corpus = _json(corpus_manifest_path)
    if not isinstance(embedding, dict) or not isinstance(corpus, dict):
        raise ValueError("embedding and corpus manifests must be objects")
    if embedding.get("evaluation_mode") != "qrels":
        raise ValueError("exact quality scoring requires evaluation_mode=qrels")
    if corpus.get("texts_sha256") != embedding.get("corpus_sha256"):
        raise ValueError("embedding manifest does not match corpus texts")
    chunks_sha256 = _sha256(chunks_path)
    if corpus.get("chunks_sha256") != chunks_sha256:
        raise ValueError("chunks JSONL does not match the corpus manifest")
    if embedding.get("qrels", {}).get("chunks_sha256") != chunks_sha256:
        raise ValueError("chunks JSONL does not match the embedding evaluation inputs")

    passage_identity_path, passage_identity = _load_shard_identity(embedding, "passage")
    query_identity_path, query_identity = _load_shard_identity(embedding, "query")
    _validate_common_identity(embedding, passage_identity, query_identity)
    if passage_identity.get("corpus_sha256") != corpus.get("texts_sha256"):
        raise ValueError("passage shard identity does not match corpus texts")
    if query_identity.get("corpus_sha256") != embedding.get("qrels", {}).get(
        "queries_sha256"
    ):
        raise ValueError("query shard identity does not match evaluation queries")

    query_ids_record = embedding.get("query_ids")
    if not isinstance(query_ids_record, dict):
        raise ValueError("embedding manifest has no query ID sidecar")
    query_ids_path = Path(_required_text(query_ids_record, "path", "query ID sidecar"))
    if _sha256(query_ids_path) != _required_text(
        query_ids_record, "sha256", "query ID sidecar"
    ):
        raise ValueError("query ID sidecar digest mismatch")
    query_ids = _load_query_ids(query_ids_path)
    corpus_ids = _load_chunk_ids(chunks_path)
    sources_by_corpus: dict[str, tuple[str, ...]] | None = None
    source_count: int | None = None
    alias_count: int | None = None
    occurrences_sha256: str | None = None
    if occurrences_path is not None:
        occurrences_sha256 = _sha256(occurrences_path)
        if corpus.get("chunk_occurrences_sha256") != occurrences_sha256:
            raise ValueError("occurrences JSONL does not match the corpus manifest")
        sources_by_corpus, source_count, alias_count = _load_source_mapping(
            occurrences_path, corpus_ids
        )
        declared_source_count = corpus.get("source_count")
        if (
            isinstance(declared_source_count, bool)
            or not isinstance(declared_source_count, int)
            or declared_source_count <= 0
        ):
            raise ValueError("source candidate scoring requires manifest source_count")
        if source_count != declared_source_count:
            raise ValueError(
                "occurrence source count disagrees with the corpus manifest"
            )
    if len(corpus_ids) != passage_identity.get("rows"):
        raise ValueError("chunk row count does not match passage shards")
    if len(query_ids) != query_identity.get("rows"):
        raise ValueError("query ID row count does not match query shards")
    if query_ids_record.get("rows") != len(query_ids):
        raise ValueError("query ID sidecar row count disagrees with its manifest")
    candidate_count = source_count if source_count is not None else len(corpus_ids)
    if top_k > candidate_count:
        raise ValueError("top_k cannot exceed the candidate count")

    passage_paths = _shard_paths(passage_identity_path, passage_identity)
    query_paths = _shard_paths(query_identity_path, query_identity)
    query_vectors = np.concatenate([np.load(path) for path in query_paths], axis=0)
    best: list[list[tuple[float, str]]] = [[] for _ in query_ids]
    source_best: list[dict[str, float]] | None = (
        [dict() for _ in query_ids] if sources_by_corpus is not None else None
    )
    offset = 0
    for passage_path in passage_paths:
        passages = np.load(passage_path)
        block_ids = corpus_ids[offset : offset + len(passages)]
        scores = query_vectors @ passages.T
        for query_index in range(len(query_ids)):
            if source_best is None:
                best[query_index] = _merge_top_k(
                    best[query_index], scores[query_index], block_ids, top_k
                )
                continue
            query_source_scores = source_best[query_index]
            for score, corpus_id in zip(scores[query_index], block_ids, strict=True):
                numeric_score = float(score)
                for source_id in sources_by_corpus[corpus_id]:
                    previous = query_source_scores.get(source_id, -math.inf)
                    if numeric_score > previous:
                        query_source_scores[source_id] = numeric_score
        offset += len(passages)
    if source_best is not None:
        best = [
            sorted(values.items(), key=lambda item: (-item[1], item[0]))[:top_k]
            for values in source_best
        ]
        best = [[(score, source_id) for source_id, score in rows] for rows in best]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as output:
            id_field = "source_id" if candidate_granularity == "source" else "corpus_id"
            for query_id, candidates in zip(query_ids, best, strict=True):
                for rank, (score, candidate_id) in enumerate(candidates, 1):
                    if not math.isfinite(score):
                        raise RuntimeError("exact scorer produced a non-finite score")
                    output.write(
                        json.dumps(
                            {
                                id_field: candidate_id,
                                "query_id": query_id,
                                "rank": rank,
                                "score": score,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)

    scoring_manifest_path = scoring_manifest_path or output_path.with_name(
        f"{output_path.stem}.scoring.manifest.json"
    )
    result = {
        "schema_version": 2,
        "producer_sha256": _sha256(Path(__file__).resolve()),
        "scoring": "exact normalized float32 inner product",
        "candidate_granularity": candidate_granularity,
        "tie_break": f"{candidate_granularity}_id ascending",
        "top_k": top_k,
        "query_count": len(query_ids),
        "corpus_row_count": len(corpus_ids),
        "source_count": source_count,
        "deduplicated_alias_count": alias_count,
        "run_row_count": len(query_ids) * top_k,
        "embedding_manifest_path": str(embedding_manifest_path.resolve()),
        "embedding_manifest_sha256": _sha256(embedding_manifest_path),
        "corpus_manifest_path": str(corpus_manifest_path.resolve()),
        "corpus_manifest_sha256": _sha256(corpus_manifest_path),
        "chunks_path": str(chunks_path.resolve()),
        "chunks_sha256": chunks_sha256,
        "occurrences_path": (
            str(occurrences_path.resolve()) if occurrences_path is not None else None
        ),
        "occurrences_sha256": occurrences_sha256,
        "query_ids_path": str(query_ids_path.resolve()),
        "query_ids_sha256": _sha256(query_ids_path),
        "passage_shard_manifest_sha256": _sha256(passage_identity_path),
        "query_shard_manifest_sha256": _sha256(query_identity_path),
        "run_path": str(output_path.resolve()),
        "run_sha256": _sha256(output_path),
        "limitations": [
            "exact float scoring does not measure ANN recall or serving latency",
            (
                "source candidates are exact maximum chunk scores"
                if candidate_granularity == "source"
                else "document metrics still require source collapse against original qrels"
            ),
        ],
    }
    _atomic_json(result, scoring_manifest_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--chunks-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--candidate-granularity", choices=("chunk", "source"), default="chunk"
    )
    parser.add_argument("--occurrences-jsonl", type=Path)
    parser.add_argument("--scoring-manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = score_embedding_shards(
        args.embedding_manifest,
        args.corpus_manifest,
        args.chunks_jsonl,
        args.output,
        top_k=args.top_k,
        candidate_granularity=args.candidate_granularity,
        occurrences_path=args.occurrences_jsonl,
        scoring_manifest_path=args.scoring_manifest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
