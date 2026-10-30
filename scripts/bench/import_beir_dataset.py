#!/usr/bin/env python3
"""Normalize a verified BEIR dataset into corpus-builder and source-qrels inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# corpus_io is a sibling module in scripts/bench/. Inject that directory so the
# import works both when run as a script and when loaded via importlib in tests.
_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from corpus_io import (  # noqa: E402  — path-injected sibling bench module
    atomic_json_write as _atomic_json,
)


def _digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path, *, kind: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{kind} line {line_number}: expected an object")
            record_id = record.get("_id")
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError(f"{kind} line {line_number}: _id is required")
            if record_id in records:
                raise ValueError(
                    f"{kind} line {line_number}: duplicate id {record_id!r}"
                )
            records[record_id] = record
    if not records:
        raise ValueError(f"{kind} file contains no records")
    return records


def _atomic_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for record in records:
                output.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _document_fields(record: dict[str, Any], *, document_id: str) -> dict[str, str]:
    title = record.get("title", "")
    text = record.get("text", "")
    if title is None:
        title = ""
    if text is None:
        text = ""
    if not isinstance(title, str) or not isinstance(text, str):
        raise ValueError(f"corpus document {document_id!r}: title/text must be strings")
    title = title.strip()
    text = text.strip()
    combined = f"{title}\n\n{text}" if title and text else title or text
    return {"body": text, "text": combined, "title": title}


def import_dataset(
    dataset_root: Path,
    *,
    split: str,
    output_dir: Path,
    dataset_name: str,
    source_url: str,
    archive_path: Path | None = None,
    expected_archive_md5: str | None = None,
) -> dict[str, Any]:
    """Normalize BEIR files while preserving document-level qrels semantics."""

    if not dataset_name.strip():
        raise ValueError("dataset_name is required")
    if not source_url.strip():
        raise ValueError("source_url is required")
    if expected_archive_md5 is not None and archive_path is None:
        raise ValueError("expected_archive_md5 requires archive_path")
    archive: dict[str, Any] | None = None
    if archive_path is not None:
        actual_md5 = _digest(archive_path, "md5")
        if expected_archive_md5 is not None and actual_md5 != expected_archive_md5:
            raise ValueError(
                f"archive MD5 mismatch: {actual_md5} != {expected_archive_md5}"
            )
        archive = {
            "path": str(archive_path.resolve()),
            "md5": actual_md5,
            "sha256": _digest(archive_path),
            "published_md5": expected_archive_md5,
        }

    corpus_path = dataset_root / "corpus.jsonl"
    queries_path = dataset_root / "queries.jsonl"
    qrels_path = dataset_root / "qrels" / f"{split}.tsv"
    corpus = _load_jsonl(corpus_path, kind="corpus")
    queries = _load_jsonl(queries_path, kind="queries")

    relations: dict[tuple[str, str], float] = {}
    dropped_nonpositive = 0
    with qrels_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        expected_fields = {"query-id", "corpus-id", "score"}
        if reader.fieldnames is None or set(reader.fieldnames) != expected_fields:
            raise ValueError(
                f"qrels header must contain {sorted(expected_fields)}, got {reader.fieldnames}"
            )
        for line_number, row in enumerate(reader, 2):
            query_id = row.get("query-id")
            corpus_id = row.get("corpus-id")
            score_text = row.get("score")
            if query_id not in queries:
                raise ValueError(
                    f"qrels line {line_number}: unknown query id {query_id!r}"
                )
            if corpus_id not in corpus:
                raise ValueError(
                    f"qrels line {line_number}: unknown corpus id {corpus_id!r}"
                )
            try:
                score = float(score_text) if score_text is not None else float("nan")
            except ValueError as exc:
                raise ValueError(
                    f"qrels line {line_number}: invalid score {score_text!r}"
                ) from exc
            if not math.isfinite(score):
                raise ValueError(f"qrels line {line_number}: score must be finite")
            if score <= 0:
                dropped_nonpositive += 1
                continue
            pair = (query_id, corpus_id)
            if pair in relations:
                raise ValueError(f"qrels line {line_number}: duplicate relation {pair}")
            relations[pair] = score
    if not relations:
        raise ValueError("qrels contain no positive relations")

    judged_query_ids = {query_id for query_id, _corpus_id in relations}
    documents_output = output_dir / "documents.jsonl"
    queries_output = output_dir / "queries.jsonl"
    qrels_output = output_dir / "source_qrels.jsonl"
    _atomic_jsonl(
        [
            {
                "id": document_id,
                **_document_fields(record, document_id=document_id),
            }
            for document_id, record in sorted(corpus.items())
        ],
        documents_output,
    )
    normalized_queries = []
    for query_id in sorted(judged_query_ids):
        query_text = queries[query_id].get("text")
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError(f"query {query_id!r}: text is required")
        normalized_queries.append({"id": query_id, "text": query_text.strip()})
    _atomic_jsonl(normalized_queries, queries_output)
    _atomic_jsonl(
        [
            {
                "query_id": query_id,
                "source_id": corpus_id,
                "relevance": relevance,
                "judgment_granularity": "document",
            }
            for (query_id, corpus_id), relevance in sorted(relations.items())
        ],
        qrels_output,
    )

    manifest = {
        "schema_version": 2,
        "producer_sha256": _digest(Path(__file__).resolve()),
        "dataset_name": dataset_name,
        "split": split,
        "source_url": source_url,
        "license_review_required": True,
        "judgment_granularity": "document",
        "document_fields": {
            "body": "BEIR corpus text",
            "text": "backward-compatible title plus body",
            "title": "BEIR corpus title",
        },
        "archive": archive,
        "inputs": {
            "corpus": {
                "path": str(corpus_path.resolve()),
                "sha256": _digest(corpus_path),
            },
            "queries": {
                "path": str(queries_path.resolve()),
                "sha256": _digest(queries_path),
            },
            "qrels": {"path": str(qrels_path.resolve()), "sha256": _digest(qrels_path)},
        },
        "outputs": {
            "documents": {
                "path": str(documents_output.resolve()),
                "sha256": _digest(documents_output),
            },
            "queries": {
                "path": str(queries_output.resolve()),
                "sha256": _digest(queries_output),
            },
            "source_qrels": {
                "path": str(qrels_output.resolve()),
                "sha256": _digest(qrels_output),
            },
        },
        "document_count": len(corpus),
        "query_count": len(judged_query_ids),
        "relation_count": len(relations),
        "dropped_nonpositive_relation_count": dropped_nonpositive,
    }
    _atomic_json(manifest, output_dir / "beir_import.manifest.json")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--expected-archive-md5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = import_dataset(
        args.dataset_root,
        split=args.split,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        source_url=args.source_url,
        archive_path=args.archive,
        expected_archive_md5=args.expected_archive_md5,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
