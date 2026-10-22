#!/usr/bin/env python3
"""Project source/span relevance judgments onto chunk occurrence IDs.

Source qrels survive changes in tokenizer and token budget. The occurrence
sidecar preserves provenance even when multiple source spans share one exact-
deduplicated canonical vector.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
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
    iter_jsonl as _jsonl,
    required_text as _required_text,
    sha256_file as _sha256,
)


def _span(
    record: dict[str, Any], *, context: str, optional: bool
) -> tuple[int, int] | None:
    start = record.get("start_pos")
    end = record.get("end_pos")
    if optional and start is None and end is None:
        return None
    if start is None or end is None:
        raise ValueError(f"{context}: start_pos and end_pos are required together")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or start < 0
    ):
        raise ValueError(f"{context}: span positions must be non-negative integers")
    if end <= start:
        raise ValueError(f"{context}: end_pos must be greater than start_pos")
    return start, end


def project_qrels(
    source_qrels_path: Path,
    corpus_manifest_path: Path,
    occurrences_path: Path,
    output_path: Path,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    """Project half-open source spans; repeated judgments keep max relevance."""

    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    if corpus_manifest.get("corpus_scope") != "complete_source":
        raise ValueError("qrels projection requires corpus_scope=complete_source")
    if not corpus_manifest.get("zero_truncation_asserted"):
        raise ValueError("qrels projection requires zero_truncation_asserted=true")
    if corpus_manifest.get("chunk_occurrences_sha256") != _sha256(occurrences_path):
        raise ValueError("chunk_occurrences.jsonl does not match the corpus manifest")

    occurrences_by_source: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    occurrence_count = 0
    for line_number, record in _jsonl(occurrences_path):
        context = f"occurrences line {line_number}"
        corpus_id = _required_text(record, "corpus_id", context)
        source_id = _required_text(record, "source_id", context)
        span = _span(record, context=context, optional=False)
        if span is None:
            raise RuntimeError("required occurrence span unexpectedly missing")
        occurrences_by_source[source_id].append((corpus_id, *span))
        occurrence_count += 1
    if not occurrence_count:
        raise ValueError("occurrence sidecar contains no rows")

    projected: dict[tuple[str, str], float] = {}
    source_relation_count = 0
    span_relation_count = 0
    source_level_relation_count = 0
    query_ids: set[str] = set()
    for line_number, record in _jsonl(source_qrels_path):
        context = f"source qrels line {line_number}"
        query_id = _required_text(record, "query_id", context)
        source_id = _required_text(record, "source_id", context)
        relevance = record.get("relevance")
        if (
            isinstance(relevance, bool)
            or not isinstance(relevance, (int, float))
            or not math.isfinite(float(relevance))
            or relevance <= 0
        ):
            raise ValueError(f"{context}: relevance must be a positive number")
        requested_span = _span(record, context=context, optional=True)
        matches: set[str] = set()
        for corpus_id, occurrence_start, occurrence_end in occurrences_by_source.get(
            source_id, ()
        ):
            if requested_span is None or (
                occurrence_start < requested_span[1]
                and occurrence_end > requested_span[0]
            ):
                matches.add(corpus_id)
        if not matches:
            raise ValueError(
                f"{context}: relation matched no corpus occurrences for source {source_id!r}"
            )
        numeric_relevance = float(relevance)
        for corpus_id in matches:
            pair = (query_id, corpus_id)
            projected[pair] = max(projected.get(pair, 0.0), numeric_relevance)
        source_relation_count += 1
        span_relation_count += int(requested_span is not None)
        source_level_relation_count += int(requested_span is None)
        query_ids.add(query_id)

    if not source_relation_count:
        raise ValueError("source qrels contain no relations")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for (query_id, corpus_id), relevance in sorted(projected.items()):
                output.write(
                    json.dumps(
                        {
                            "query_id": query_id,
                            "corpus_id": corpus_id,
                            "relevance": relevance,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)

    summary_path = summary_path or output_path.with_name(
        f"{output_path.stem}.projection.manifest.json"
    )
    summary = {
        "schema_version": 1,
        "producer_sha256": _sha256(Path(__file__).resolve()),
        "corpus_manifest_path": str(corpus_manifest_path.resolve()),
        "corpus_manifest_sha256": _sha256(corpus_manifest_path),
        "source_qrels_path": str(source_qrels_path.resolve()),
        "source_qrels_sha256": _sha256(source_qrels_path),
        "occurrences_path": str(occurrences_path.resolve()),
        "occurrences_sha256": _sha256(occurrences_path),
        "output_path": str(output_path.resolve()),
        "output_sha256": _sha256(output_path),
        "occurrence_count": occurrence_count,
        "source_relation_count": source_relation_count,
        "span_relation_count": span_relation_count,
        "source_level_relation_count": source_level_relation_count,
        "projected_relation_count": len(projected),
        "query_count": len(query_ids),
        "projection_manifest_path": str(summary_path.resolve()),
    }
    _atomic_json(summary, summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-qrels", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--occurrences-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--projection-manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            project_qrels(
                args.source_qrels,
                args.corpus_manifest,
                args.occurrences_jsonl,
                args.output,
                args.projection_manifest,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
