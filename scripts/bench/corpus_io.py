#!/usr/bin/env python3
"""Shared local IO helpers for ``scripts/bench`` (TD-CTXCORR-1 Slice 2b).

SIFT-family ``.fvecs`` / ``.ivecs`` readers and checksum helpers. Benchmark
vectors are kept **local and gitignored** and verified by SHA256 — the checksum
becomes the report ``dataset_hash`` (satisfies ``require_dataset_hash``). These
are the file-format + fetch foundation the SIFT1M / Wikipedia providers
(Slice 2b-ii) build on; nothing here downloads by itself except the injectable
``fetch_corpus`` helper.

This module is also the canonical home for the atomic JSON/JSONL writers,
JSONL readers, record validators, and percentile/histogram statistics that the
bench scripts previously duplicated per-script (redundancy pass 6). Scripts
adopt them via ``sys.path`` injection of this directory plus
``from corpus_io import ...`` (see ``context_corridor.py`` / ``fetch_corpus.py``
for the pattern).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

_HEADER = struct.Struct("<i")


def _read_vecs(path: Path | str, code: str) -> Iterator[list[float]]:
    # .fvecs/.ivecs layout: for each vector, an int32 dimension followed by
    # `dimension` little-endian values of the given struct code ('f' or 'i').
    with open(path, "rb") as handle:
        while True:
            header = handle.read(4)
            if not header:
                return
            if len(header) != 4:
                raise ValueError(f"truncated .{code}vecs header in {path}")
            (dimension,) = _HEADER.unpack(header)
            if dimension < 0:
                raise ValueError(f"invalid .{code}vecs dimension {dimension} in {path}")
            body = handle.read(4 * dimension)
            if len(body) != 4 * dimension:
                raise ValueError(f"truncated .{code}vecs vector body in {path}")
            yield list(struct.unpack(f"<{dimension}{code}", body))


def read_fvecs(path: Path | str) -> Iterator[list[float]]:
    """Stream float32 vectors from a ``.fvecs`` file (SIFT base/query vectors)."""
    return _read_vecs(path, "f")


def read_ivecs(path: Path | str) -> Iterator[list[int]]:
    """Stream int32 vectors from an ``.ivecs`` file (SIFT ground-truth neighbours)."""
    return (
        [int(value) for value in vector] for vector in _read_vecs(path, "i")
    )


def sha256_file(path: Path | str, *, chunk_size: int = 1 << 20) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_checksum(path: Path | str, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"checksum mismatch for {path}: expected {expected}, got {actual}"
        )
    return actual


def fetch_corpus(
    url: str,
    dest: Path | str,
    expected_sha256: str,
    *,
    download: Callable[[str, str], object] | None = None,
) -> str:
    """Fetch ``url`` to ``dest`` and verify its SHA256; return the checksum.

    Idempotent: an existing ``dest`` with the right checksum is reused (no
    download). Downloads to a ``.part`` sidecar and only promotes it on a
    checksum match, so a mismatch never leaves a corrupt corpus in place. The
    downloader is injectable for testing.
    """
    download = download or urllib.request.urlretrieve
    dest = Path(dest)
    if dest.exists():
        actual = sha256_file(dest)
        if actual == expected_sha256:
            return actual
        raise ValueError(
            f"{dest} exists but its checksum {actual} != expected "
            f"{expected_sha256}; delete it to re-fetch"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    try:
        download(url, str(partial))
        actual = sha256_file(partial)
        if actual != expected_sha256:
            raise ValueError(
                f"checksum mismatch for {url}: expected {expected_sha256}, "
                f"got {actual}"
            )
        partial.replace(dest)
    finally:
        if partial.exists():
            partial.unlink()
    return actual


def atomic_json_write(value: Any, path: Path) -> None:
    """Atomically write ``value`` as sorted, pretty-printed JSON to ``path``.

    Writes to a ``.tmp`` sibling via ``tempfile.mkstemp`` and promotes it with
    ``os.replace``, so a crash mid-write never leaves a truncated report.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_jsonl_write(records: Sequence[dict[str, Any]], path: Path) -> None:
    """Atomically write ``records`` as one sorted-key JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Stream ``(line_number, object)`` pairs from a JSONL file of objects.

    Blank lines are skipped; a non-object line is a ``ValueError`` carrying the
    1-based line number.
    """
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} line {line_number}: expected an object")
            yield line_number, value


def read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Eager form of :func:`iter_jsonl` — the whole file as a list of pairs."""
    return list(iter_jsonl(path))


def required_text(record: dict[str, Any], name: str, context: str) -> str:
    """Return ``record[name]`` as a non-empty string or raise ``ValueError``."""
    value = record.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: {name} is required")
    return value


def finite(value: Any, context: str) -> float:
    """Return ``value`` as a finite float or raise ``ValueError`` (bools rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}: expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context}: expected a finite number")
    return result


def finite_number(record: dict[str, Any], name: str, context: str) -> float:
    """Return ``record[name]`` as a finite float or raise ``ValueError``."""
    value = record.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{context}: {name} must be a finite number")
    return float(value)


def percentile(values: list[int], fraction: float) -> int | None:
    """Nearest-rank-ish percentile of ``values`` at ``fraction`` (sorted copy)."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def histogram(values: list[int]) -> dict[str, int | None]:
    """min/p50/p90/p99/max summary of ``values`` (``None`` entries when empty)."""
    return {
        "min": min(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }
