#!/usr/bin/env python3
"""Resumable embedding transport for a token-budget corpus and pinned contract.

The corpus manifest is authoritative for text and input-contract identity. This
transport owns embedding shards and benchmark binary emission; it contains no
chunking logic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from proximadb_sdk.chunking_strategies import InputRole
from proximadb_sdk.embedding_providers.providers.local.open_weights import (
    create_open_model_provider,
)

# corpus_io is a sibling module in scripts/bench/. Inject that directory so the
# import works both when run as a script and when loaded via importlib in tests.
_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from corpus_io import (  # noqa: E402  — path-injected sibling bench module
    atomic_json_write as _atomic_json,
    sha256_file as _sha256,
)


def _contract_for_model(
    corpus_manifest: dict[str, Any], model_id: str
) -> dict[str, Any]:
    contracts = corpus_manifest.get("input_contract", {}).get("contracts", [])
    matches = [item for item in contracts if item.get("model_id") == model_id]
    if len(matches) != 1:
        raise ValueError(
            f"corpus manifest must contain exactly one contract for {model_id}"
        )
    return matches[0]


def load_query_records(
    path: Path, *, id_field: str = "id", text_field: str = "text"
) -> tuple[list[str], tuple[str, ...]]:
    texts: list[str] = []
    query_ids: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            query_id = record.get(id_field)
            text = record.get(text_field)
            if not isinstance(query_id, str) or not query_id.strip():
                raise ValueError(f"query line {line_number}: {id_field!r} is required")
            if query_id in seen:
                raise ValueError(f"duplicate query id {query_id!r}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"query line {line_number}: {text_field!r} is required"
                )
            seen.add(query_id)
            query_ids.append(query_id)
            texts.append(text)
    if not texts:
        raise ValueError("queries JSONL contains no query records")
    return texts, tuple(query_ids)


def load_corpus_ids(path: Path) -> set[str]:
    corpus_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            corpus_id = record.get("chunk_id")
            if not isinstance(corpus_id, str) or not corpus_id.strip():
                raise ValueError(f"chunk line {line_number}: chunk_id is required")
            if corpus_id in corpus_ids:
                raise ValueError(
                    f"chunk line {line_number}: duplicate chunk_id {corpus_id!r}"
                )
            corpus_ids.add(corpus_id)
    if not corpus_ids:
        raise ValueError("chunks JSONL contains no corpus rows")
    return corpus_ids


def validate_qrels(
    path: Path, query_ids: Sequence[str], corpus_ids: set[str] | None = None
) -> dict[str, Any]:
    known_queries = set(query_ids)
    covered_queries: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    row_count = 0
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            query_id = record.get("query_id")
            corpus_id = record.get("corpus_id")
            relevance = record.get("relevance")
            if query_id not in known_queries:
                raise ValueError(
                    f"qrels line {line_number}: unknown query_id {query_id!r}"
                )
            if not isinstance(corpus_id, str) or not corpus_id.strip():
                raise ValueError(f"qrels line {line_number}: corpus_id is required")
            if corpus_ids is not None and corpus_id not in corpus_ids:
                raise ValueError(
                    f"qrels line {line_number}: unknown corpus_id {corpus_id!r}"
                )
            if (
                isinstance(relevance, bool)
                or not isinstance(relevance, (int, float))
                or not math.isfinite(float(relevance))
                or relevance <= 0
            ):
                raise ValueError(
                    f"qrels line {line_number}: relevance must be a positive number"
                )
            pair = (query_id, corpus_id)
            if pair in seen_pairs:
                raise ValueError(f"qrels line {line_number}: duplicate relation {pair}")
            seen_pairs.add(pair)
            covered_queries.add(query_id)
            row_count += 1
    missing = known_queries - covered_queries
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise ValueError(f"qrels have no relevant corpus rows for queries: {preview}")
    return {
        "sha256": _sha256(path),
        "row_count": row_count,
        "query_count": len(covered_queries),
    }


def prepare_shards(
    *,
    output_dir: Path,
    corpus_sha256: str,
    model_id: str,
    revision: str,
    contract_fingerprint: str,
    dimension: int,
    shard_size: int,
    rows: int,
    runtime: dict[str, Any],
    input_role: str = "document",
) -> list[Path]:
    if input_role not in {"document", "query"}:
        raise ValueError(f"unsupported embedding input role {input_role!r}")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "--", model_id).strip("-")
    runtime_slug = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "--",
        "-".join(
            str(runtime.get(key) or "none")
            for key in ("backend", "device", "compute_dtype")
        ),
    ).strip("-")
    shard_dir = (
        output_dir
        / "shards"
        / f"{corpus_sha256[:16]}-{slug}-{revision[:12]}-{dimension}d-{input_role}-{runtime_slug}"
    )
    shard_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": 2,
        "transport_sha256": _sha256(Path(__file__).resolve()),
        "corpus_sha256": corpus_sha256,
        "model_id": model_id,
        "model_revision": revision,
        "contract_fingerprint": contract_fingerprint,
        "dimension": dimension,
        "normalize_embeddings": True,
        "shard_size": shard_size,
        "rows": rows,
        "input_role": input_role,
        "runtime": runtime,
        "artifact_dtype": "float32",
    }
    identity_path = shard_dir / "manifest.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise RuntimeError(f"embedding shard identity mismatch in {shard_dir}")
    else:
        _atomic_json(identity, identity_path)
    return [shard_dir / f"{start:08d}.npy" for start in range(0, rows, shard_size)]


def pending_shards(
    paths: Sequence[Path], *, rows: int, dimension: int, shard_size: int
) -> list[tuple[int, Path]]:
    pending = []
    for start, path in zip(range(0, rows, shard_size), paths, strict=True):
        expected = (min(shard_size, rows - start), dimension)
        if path.exists():
            saved = np.load(path, mmap_mode="r")
            if saved.shape != expected or saved.dtype != np.float32:
                raise RuntimeError(
                    f"invalid cached shard {path}: {saved.shape} {saved.dtype}; expected {expected} float32"
                )
        else:
            pending.append((start, path))
    return pending


def shard_manifest_reference(paths: Sequence[Path]) -> dict[str, str]:
    """Bind the identity manifest shared by one ordered shard set."""
    if not paths:
        raise ValueError("embedding shard set cannot be empty")
    parent = paths[0].parent
    if any(path.parent != parent for path in paths):
        raise ValueError("embedding shard paths do not share one identity directory")
    manifest_path = parent / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"embedding shard identity is missing: {manifest_path}")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
    }


def _atomic_numpy(array: np.ndarray, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def finalize_embeddings(
    shard_paths: Sequence[Path],
    *,
    output_dir: Path,
    prefix: str,
    dimension: int,
    query_rows: int,
    query_shard_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    passage_rows = 0
    value_min = float("inf")
    value_max = float("-inf")
    sum_x = np.zeros(dimension, dtype=np.float64)
    sum_xx = np.zeros((dimension, dimension), dtype=np.float64)
    for path in shard_paths:
        embeddings = np.load(path)
        if embeddings.ndim != 2 or embeddings.shape[1] != dimension:
            raise RuntimeError(f"invalid embedding shape in {path}: {embeddings.shape}")
        passage_rows += embeddings.shape[0]
        value_min = min(value_min, float(embeddings.min()))
        value_max = max(value_max, float(embeddings.max()))
        values = embeddings.astype(np.float64)
        sum_x += values.sum(axis=0)
        sum_xx += values.T @ values

    separate_queries = query_shard_paths is not None
    actual_query_rows = 0
    query_value_min: float | None = None
    query_value_max: float | None = None
    query_clip_low_count = 0
    query_clip_high_count = 0
    if separate_queries:
        if query_rows != 0:
            raise ValueError("query_rows must be zero when query shards are provided")
        for path in query_shard_paths:
            embeddings = np.load(path)
            if embeddings.ndim != 2 or embeddings.shape[1] != dimension:
                raise RuntimeError(
                    f"invalid query embedding shape in {path}: {embeddings.shape}"
                )
            actual_query_rows += embeddings.shape[0]
            shard_min = float(embeddings.min())
            shard_max = float(embeddings.max())
            query_value_min = (
                shard_min
                if query_value_min is None
                else min(query_value_min, shard_min)
            )
            query_value_max = (
                shard_max
                if query_value_max is None
                else max(query_value_max, shard_max)
            )
            query_clip_low_count += int(np.count_nonzero(embeddings < value_min))
            query_clip_high_count += int(np.count_nonzero(embeddings > value_max))
        if actual_query_rows == 0:
            raise ValueError("query shard set contains no embeddings")
        base_rows = passage_rows
    else:
        if passage_rows <= query_rows:
            raise ValueError(f"need more than {query_rows} rows, got {passage_rows}")
        actual_query_rows = query_rows
        base_rows = passage_rows - query_rows
    if not value_max > value_min:
        raise ValueError("embedding quantization range is empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    query_path = output_dir / f"{prefix}_query.u8bin"
    base_path = output_dir / f"{prefix}_base.u8bin"
    query_temporary = query_path.with_name(f".{query_path.name}.tmp")
    base_temporary = base_path.with_name(f".{base_path.name}.tmp")
    position = 0
    try:
        with query_temporary.open("wb") as query, base_temporary.open("wb") as base:
            np.asarray([actual_query_rows, dimension], dtype="<i4").tofile(query)
            np.asarray([base_rows, dimension], dtype="<i4").tofile(base)

            def quantized(path: Path) -> np.ndarray:
                embeddings = np.load(path)
                return np.clip(
                    np.rint((embeddings - value_min) / (value_max - value_min) * 255.0),
                    0,
                    255,
                ).astype(np.uint8)

            if separate_queries:
                for path in query_shard_paths:
                    quantized(path).tofile(query)
                for path in shard_paths:
                    quantized(path).tofile(base)
            else:
                for path in shard_paths:
                    values = quantized(path)
                    query_end = min(
                        values.shape[0], max(actual_query_rows - position, 0)
                    )
                    values[:query_end].tofile(query)
                    values[query_end:].tofile(base)
                    position += values.shape[0]
        os.replace(query_temporary, query_path)
        os.replace(base_temporary, base_path)
    finally:
        query_temporary.unlink(missing_ok=True)
        base_temporary.unlink(missing_ok=True)

    mean = sum_x / passage_rows
    covariance = sum_xx / passage_rows - np.outer(mean, mean)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    cumulative = np.cumsum(eigenvalues)
    total_variance = cumulative[-1]
    if not np.isfinite(total_variance) or total_variance <= 0:
        raise ValueError("embedding covariance has no positive variance")
    cumulative /= total_variance
    spectrum = {
        str(threshold): int(np.searchsorted(cumulative, threshold) + 1)
        for threshold in (0.70, 0.79, 0.85, 0.90, 0.95)
    }
    return {
        "rows": base_rows + actual_query_rows,
        "passage_rows": passage_rows,
        "base_rows": base_rows,
        "query_rows": actual_query_rows,
        "evaluation_mode": "qrels" if separate_queries else "geometry_probe",
        "dimension": dimension,
        "quantization_min": value_min,
        "quantization_max": value_max,
        "query_value_min": query_value_min,
        "query_value_max": query_value_max,
        "query_clip_low_count": query_clip_low_count,
        "query_clip_high_count": query_clip_high_count,
        "spectrum_full": spectrum,
        "base_path": str(base_path),
        "query_path": str(query_path),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for name in ("dimension", "shard_size", "batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    queries_path = getattr(args, "queries_jsonl", None)
    qrels_path = getattr(args, "qrels", None)
    chunks_path = getattr(args, "chunks_jsonl", None)
    evaluation_inputs = (queries_path, qrels_path, chunks_path)
    if any(path is not None for path in evaluation_inputs) and not all(
        path is not None for path in evaluation_inputs
    ):
        raise ValueError(
            "--queries-jsonl, --qrels, and --chunks-jsonl must be provided together"
        )
    if queries_path is None and args.query_rows <= 0:
        raise ValueError("query_rows must be positive in geometry-probe mode")
    if re.fullmatch(r"[0-9a-f]{40}", args.revision) is None:
        raise ValueError("revision must be an immutable 40-character commit SHA")
    texts_sha256 = _sha256(args.texts)
    corpus_manifest = json.loads(args.corpus_manifest.read_text(encoding="utf-8"))
    if corpus_manifest.get("texts_sha256") != texts_sha256:
        raise ValueError("texts.json does not match the corpus manifest")
    if chunks_path is not None and corpus_manifest.get("chunks_sha256") != _sha256(
        chunks_path
    ):
        raise ValueError("chunks.jsonl does not match the corpus manifest")
    declared_contract = _contract_for_model(corpus_manifest, args.model)
    if declared_contract.get("model_revision") != args.revision:
        raise ValueError("requested model revision does not match the corpus contract")
    if declared_contract.get("output_dimension") != args.dimension:
        raise ValueError("requested dimension does not match the corpus contract")
    texts = json.loads(args.texts.read_text(encoding="utf-8"))
    if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        raise ValueError("texts.json must be an array of strings")

    provider = create_open_model_provider(
        args.model,
        revision=args.revision,
        dimension=args.dimension,
        device=args.device,
        backend=args.backend,
        batch_size=args.batch_size,
        normalize=True,
    )
    resolved_contract = provider.get_input_contract()
    if resolved_contract.fingerprint != declared_contract.get("contract_fingerprint"):
        raise ValueError("loaded runtime contract does not match the corpus contract")
    if provider.get_dimension() != args.dimension:
        raise ValueError(
            "loaded runtime dimension does not match the requested dimension"
        )
    runtime = provider.get_runtime_info()

    query_texts: list[str] | None = None
    query_ids: tuple[str, ...] = ()
    qrels_summary: dict[str, Any] | None = None
    if queries_path is not None:
        query_texts, query_ids = load_query_records(
            queries_path,
            id_field=getattr(args, "query_id_field", "id"),
            text_field=getattr(args, "query_text_field", "text"),
        )
        corpus_ids = load_corpus_ids(chunks_path)
        query_token_counts = [
            resolved_contract.validate(text, InputRole.QUERY) for text in query_texts
        ]
        qrels_summary = {
            **validate_qrels(qrels_path, query_ids, corpus_ids),
            "path": str(qrels_path.resolve()),
            "queries_sha256": _sha256(queries_path),
            "queries_path": str(queries_path.resolve()),
            "chunks_sha256": _sha256(chunks_path),
            "chunks_path": str(chunks_path.resolve()),
            "token_min": min(query_token_counts),
            "token_max": max(query_token_counts),
        }

    shard_paths = prepare_shards(
        output_dir=args.output_dir,
        corpus_sha256=texts_sha256,
        model_id=args.model,
        revision=args.revision,
        contract_fingerprint=resolved_contract.fingerprint,
        dimension=args.dimension,
        shard_size=args.shard_size,
        rows=len(texts),
        runtime=runtime,
    )
    pending = pending_shards(
        shard_paths,
        rows=len(texts),
        dimension=args.dimension,
        shard_size=args.shard_size,
    )
    for index, (start, path) in enumerate(pending, 1):
        started = time.monotonic()
        embeddings = np.asarray(
            provider.embed_passages(texts[start : start + args.shard_size]),
            dtype=np.float32,
        )
        expected = (min(args.shard_size, len(texts) - start), args.dimension)
        if embeddings.shape != expected:
            raise RuntimeError(
                f"runtime emitted {embeddings.shape} for shard {start}; expected {expected}"
            )
        _atomic_numpy(embeddings, path)
        elapsed = time.monotonic() - started
        remaining = len(pending) - index
        print(
            f"shard {start:>8,}: {elapsed:.1f}s ({index}/{len(pending)}, eta {remaining * elapsed / 60:.1f}m)",
            flush=True,
        )

    query_shard_paths: list[Path] | None = None
    if query_texts is not None:
        if qrels_summary is None:
            raise RuntimeError("validated query inputs are missing qrels metadata")
        query_shard_paths = prepare_shards(
            output_dir=args.output_dir,
            corpus_sha256=qrels_summary["queries_sha256"],
            model_id=args.model,
            revision=args.revision,
            contract_fingerprint=resolved_contract.fingerprint,
            dimension=args.dimension,
            shard_size=args.shard_size,
            rows=len(query_texts),
            input_role="query",
            runtime=runtime,
        )
        query_pending = pending_shards(
            query_shard_paths,
            rows=len(query_texts),
            dimension=args.dimension,
            shard_size=args.shard_size,
        )
        for index, (start, path) in enumerate(query_pending, 1):
            started = time.monotonic()
            embeddings = np.asarray(
                provider.embed_queries(query_texts[start : start + args.shard_size]),
                dtype=np.float32,
            )
            expected = (
                min(args.shard_size, len(query_texts) - start),
                args.dimension,
            )
            if embeddings.shape != expected:
                raise RuntimeError(
                    f"runtime emitted {embeddings.shape} for query shard {start}; "
                    f"expected {expected}"
                )
            _atomic_numpy(embeddings, path)
            elapsed = time.monotonic() - started
            remaining = len(query_pending) - index
            print(
                f"query shard {start:>8,}: {elapsed:.1f}s "
                f"({index}/{len(query_pending)}, eta {remaining * elapsed / 60:.1f}m)",
                flush=True,
            )

    query_ids_manifest: dict[str, Any] | None = None
    if query_ids:
        query_ids_path = args.output_dir / f"{args.prefix}_query_ids.json"
        _atomic_json(list(query_ids), query_ids_path)
        query_ids_manifest = {
            "path": str(query_ids_path),
            "sha256": _sha256(query_ids_path),
            "rows": len(query_ids),
        }

    result = finalize_embeddings(
        shard_paths,
        output_dir=args.output_dir,
        prefix=args.prefix,
        dimension=args.dimension,
        query_rows=0 if query_shard_paths is not None else args.query_rows,
        query_shard_paths=query_shard_paths,
    )
    manifest = {
        "schema_version": 2,
        "transport_sha256": _sha256(Path(__file__).resolve()),
        "model_id": args.model,
        "model_revision": args.revision,
        "contract_fingerprint": resolved_contract.fingerprint,
        "corpus_sha256": texts_sha256,
        "shard_size": args.shard_size,
        "normalize_embeddings": True,
        "runtime": runtime,
        "artifact_dtype": "float32",
        "qrels": qrels_summary,
        "query_ids": query_ids_manifest,
        "shards": {
            "passage": shard_manifest_reference(shard_paths),
            "query": (
                shard_manifest_reference(query_shard_paths)
                if query_shard_paths is not None
                else None
            ),
        },
        **result,
    }
    _atomic_json(manifest, args.output_dir / f"{args.prefix}.embedding.manifest.json")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--shard-size", type=int, default=20_000)
    parser.add_argument("--query-rows", type=int, default=1_000)
    parser.add_argument(
        "--queries-jsonl",
        type=Path,
        help="role-correct query JSONL for qrels-backed evaluation",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        help="JSONL rows with query_id, corpus_id, and positive relevance",
    )
    parser.add_argument(
        "--chunks-jsonl",
        type=Path,
        help="exact corpus chunk manifest whose chunk_id values qrels reference",
    )
    parser.add_argument("--query-id-field", default="id")
    parser.add_argument("--query-text-field", default="text")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device")
    parser.add_argument("--backend", choices=("torch", "onnx", "openvino"))
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
