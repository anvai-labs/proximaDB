#!/usr/bin/env python3
"""Measure token-only row, storage, compute-proxy, and update economics.

This transport consumes a sealed token-budget corpus. It does not load model
weights and does not claim retrieval quality or wall-clock inference cost.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# corpus_io is a sibling module in scripts/bench/. Inject that directory so the
# import works both when run as a script and when loaded via importlib in tests.
_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from corpus_io import (  # noqa: E402  — path-injected sibling bench module
    atomic_json_write as _atomic_json,
    histogram as _histogram,
    iter_jsonl as _jsonl,
    sha256_file as _sha256,
)

SCALAR_BYTES = {"float32": 4, "float16": 2, "sq8": 1}


def _validate_span(record: dict[str, Any], *, context: str) -> tuple[int, int]:
    start = record.get("start_pos")
    end = record.get("end_pos")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        raise ValueError(f"{context}: require 0 <= start_pos < end_pos")
    return start, end


def _dimensions_from_manifest(manifest: dict[str, Any]) -> tuple[int, ...]:
    dimensions: set[int] = set()
    for contract in manifest.get("input_contract", {}).get("contracts", []):
        value = contract.get("output_dimension") or contract.get("native_dimension")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            dimensions.add(value)
    if not dimensions:
        raise ValueError("no output dimension declared; pass --dimension explicitly")
    return tuple(sorted(dimensions))


def analyze_corpus(
    manifest_path: Path,
    *,
    chunks_path: Path | None = None,
    occurrences_path: Path | None = None,
    dimensions: tuple[int, ...] = (),
    fixed_bytes_per_row: int = 0,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Return exact structural economics and clearly-labelled compute proxies."""

    if fixed_bytes_per_row < 0:
        raise ValueError("fixed_bytes_per_row cannot be negative")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in dimensions
    ):
        raise ValueError("dimensions must be positive integers")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if require_complete and manifest.get("corpus_scope") != "complete_source":
        raise ValueError("economics evidence requires corpus_scope=complete_source")
    if not manifest.get("zero_truncation_asserted"):
        raise ValueError("economics evidence requires zero_truncation_asserted=true")

    root = manifest_path.parent
    chunks_path = chunks_path or root / "chunks.jsonl"
    declared_chunks_sha = manifest.get("chunks_sha256")
    if declared_chunks_sha != _sha256(chunks_path):
        raise ValueError("chunks.jsonl does not match the corpus manifest")

    chunk_ids: set[str] = set()
    token_lengths: dict[str, list[int]] = defaultdict(list)
    canonical_span_chars = 0
    canonical_source_counts: Counter[str] = Counter()
    for line_number, record in _jsonl(chunks_path):
        chunk_id = record.get("chunk_id")
        source_id = record.get("source_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(f"chunks line {line_number}: chunk_id is required")
        if chunk_id in chunk_ids:
            raise ValueError(
                f"chunks line {line_number}: duplicate chunk_id {chunk_id!r}"
            )
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"chunks line {line_number}: source_id is required")
        start, end = _validate_span(record, context=f"chunks line {line_number}")
        counts = record.get("token_counts")
        if not isinstance(counts, dict) or not counts:
            raise ValueError(f"chunks line {line_number}: token_counts are required")
        for model_id, count in counts.items():
            if (
                not isinstance(model_id, str)
                or not model_id
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
            ):
                raise ValueError(
                    f"chunks line {line_number}: invalid token count {model_id!r}={count!r}"
                )
            token_lengths[model_id].append(count)
        chunk_ids.add(chunk_id)
        canonical_source_counts[source_id] += 1
        canonical_span_chars += end - start

    canonical_rows = len(chunk_ids)
    if canonical_rows != manifest.get("chunk_count"):
        raise ValueError(
            f"manifest chunk_count {manifest.get('chunk_count')!r} != {canonical_rows}"
        )

    limitations = [
        "attention_quadratic_proxy is sum(rendered_tokens^2), not measured latency",
        "storage excludes model-specific ANN/index overhead except caller-supplied fixed bytes per row",
        "retrieval quality, citation precision, and generator payload require qrels-backed evaluation",
    ]
    declared_occurrences_sha = manifest.get("chunk_occurrences_sha256")
    if occurrences_path is None and declared_occurrences_sha is not None:
        occurrences_path = root / "chunk_occurrences.jsonl"

    occurrence_source_counts: Counter[str] = Counter()
    source_occurrence_rows = 0
    deduplicated_alias_rows = 0
    occurrence_span_chars = 0
    if occurrences_path is not None:
        if declared_occurrences_sha != _sha256(occurrences_path):
            raise ValueError(
                "chunk_occurrences.jsonl does not match the corpus manifest"
            )
        occurrence_basis = "chunk_occurrences"
        for line_number, record in _jsonl(occurrences_path):
            corpus_id = record.get("corpus_id")
            source_id = record.get("source_id")
            if corpus_id not in chunk_ids:
                raise ValueError(
                    f"occurrences line {line_number}: unknown corpus_id {corpus_id!r}"
                )
            if not isinstance(source_id, str) or not source_id:
                raise ValueError(
                    f"occurrences line {line_number}: source_id is required"
                )
            start, end = _validate_span(
                record, context=f"occurrences line {line_number}"
            )
            alias = record.get("deduplicated_alias")
            if not isinstance(alias, bool):
                raise ValueError(
                    f"occurrences line {line_number}: deduplicated_alias must be boolean"
                )
            occurrence_source_counts[source_id] += 1
            source_occurrence_rows += 1
            deduplicated_alias_rows += int(alias)
            occurrence_span_chars += end - start
        if source_occurrence_rows != manifest.get("chunk_occurrence_count"):
            raise ValueError(
                "manifest chunk_occurrence_count does not match the occurrence sidecar"
            )
        if deduplicated_alias_rows != manifest.get("duplicate_chunk_count", 0):
            raise ValueError(
                "manifest duplicate_chunk_count does not match occurrence aliases"
            )
        if source_occurrence_rows - deduplicated_alias_rows != canonical_rows:
            raise ValueError(
                "non-alias occurrences do not map one-to-one to canonical chunks"
            )
    else:
        occurrence_basis = "canonical_chunks_fallback"
        occurrence_source_counts.update(canonical_source_counts)
        source_occurrence_rows = canonical_rows
        occurrence_span_chars = canonical_span_chars
        if manifest.get("duplicate_chunk_count", 0):
            limitations.append(
                "legacy corpus lacks chunk occurrences; deduplicated source aliases and update amplification are undercounted"
            )

    source_count = manifest.get("source_count")
    if (
        isinstance(source_count, bool)
        or not isinstance(source_count, int)
        or source_count <= 0
    ):
        raise ValueError("manifest source_count must be a positive integer")
    sources_with_occurrences = len(occurrence_source_counts)
    if sources_with_occurrences > source_count:
        raise ValueError("occurrence sidecar has more sources than the corpus manifest")

    chosen_dimensions = tuple(
        sorted(set(dimensions or _dimensions_from_manifest(manifest)))
    )
    storage: dict[str, Any] = {}
    for dimension in chosen_dimensions:
        payload = {
            name: canonical_rows * dimension * scalar_bytes
            for name, scalar_bytes in SCALAR_BYTES.items()
        }
        update_sweep = {
            name: source_occurrence_rows * dimension * scalar_bytes
            for name, scalar_bytes in SCALAR_BYTES.items()
        }
        fixed_overhead = canonical_rows * fixed_bytes_per_row
        storage[str(dimension)] = {
            "canonical_rows": canonical_rows,
            "vector_payload_bytes": payload,
            "fixed_bytes_per_row": fixed_bytes_per_row,
            "fixed_row_overhead_bytes": fixed_overhead,
            "payload_plus_fixed_overhead_bytes": {
                name: value + fixed_overhead for name, value in payload.items()
            },
            "one_update_per_source_vector_bytes": update_sweep,
        }

    return {
        "schema_version": 1,
        "producer_sha256": _sha256(Path(__file__).resolve()),
        "corpus": {
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": _sha256(manifest_path),
            "chunks_path": str(chunks_path.resolve()),
            "chunks_sha256": declared_chunks_sha,
            "occurrences_path": (
                str(occurrences_path.resolve())
                if occurrences_path is not None
                else None
            ),
            "occurrences_sha256": declared_occurrences_sha,
            "occurrence_basis": occurrence_basis,
            "corpus_scope": manifest.get("corpus_scope"),
            "deduplication_policy": manifest.get("deduplication_policy"),
            "boundary_strategy": manifest.get("boundary_strategy"),
            "boundary_char_size": manifest.get("boundary_char_size"),
            "token_budget": manifest.get("token_budget"),
        },
        "row_economics": {
            "canonical_rows": canonical_rows,
            "source_occurrence_rows": source_occurrence_rows,
            "deduplicated_alias_rows": deduplicated_alias_rows,
            "source_count": source_count,
            "sources_with_occurrences": sources_with_occurrences,
            "sources_without_occurrences": source_count - sources_with_occurrences,
            "occurrences_per_source": _histogram(
                list(occurrence_source_counts.values())
            ),
        },
        "span_economics": {
            "canonical_span_characters": canonical_span_chars,
            "source_occurrence_span_characters": occurrence_span_chars,
        },
        "tokens_by_model": {
            model_id: {
                "rendered_tokens": sum(values),
                "attention_quadratic_proxy": sum(value * value for value in values),
                "histogram": _histogram(values),
            }
            for model_id, values in sorted(token_lengths.items())
        },
        "storage_by_dimension": storage,
        "limitations": limitations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--chunks-jsonl", type=Path)
    parser.add_argument("--occurrences-jsonl", type=Path)
    parser.add_argument("--dimension", type=int, action="append", default=[])
    parser.add_argument("--fixed-bytes-per-row", type=int, default=0)
    parser.add_argument("--allow-partial-corpus", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_corpus(
        args.corpus_manifest,
        chunks_path=args.chunks_jsonl,
        occurrences_path=args.occurrences_jsonl,
        dimensions=tuple(args.dimension),
        fixed_bytes_per_row=args.fixed_bytes_per_row,
        require_complete=not args.allow_partial_corpus,
    )
    if args.output is not None:
        _atomic_json(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
