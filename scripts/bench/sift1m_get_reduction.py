#!/usr/bin/env python3
"""Auditable SIFT1M GET-reduction acceptance harness.

This harness owns the server processes it measures and refuses to publish a
result unless:

* the binary comes from an optimized Cargo profile;
* the SIFT files have the requested cardinality and dimension;
* live PAX footer row counts sum to the full corpus;
* segment paths/sizes/row counts are stable after async compaction;
* the collection's unflushed-WAL byte gauge is zero;
* the physical object-store GET/byte counters are present; and
* recall is computed against full-corpus SIFT ground truth.

It measures three distinct states with fresh result/DRAM caches:

1. post-write: cache_on_write=all, before restart;
2. local_disk_warm: restart with the persistent local-disk tier; and
3. object_cold: restart without the local-disk tier (diagnostic baseline).

The default local backend meters the same physical read seam used by
object-store backends. Pass ``--storage-url az://... --azurite`` to exercise
the production Azure backend over HTTP. Azurite latency is local-emulator
evidence, not a production Azure WAN-latency claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

PAX_MAGIC = b"PAXSEG01"
PAX_HEADER_MAGIC = b"PXH1"
A0_MAGIC = b"PXA0"
AZURE_COALESCE_GAP_BYTES = 1024 * 1024
AZURE_COALESCE_RANGE_BYTES = 4 * 1024 * 1024
METRICS = (
    "proximadb_object_store_gets_total",
    "proximadb_object_store_bytes_read_total",
    "proximadb_survivor_cache_hits",
    "proximadb_survivor_cache_misses",
    "proximadb_survivor_cache_bytes",
    "proximadb_segment_invariants_cache_hits_total",
    "proximadb_segment_invariants_cache_misses_total",
    "proximadb_segment_invariants_cache_bytes",
    "proximadb_cache_local_disk_hits_total",
    "proximadb_cache_local_disk_misses_total",
    "proximadb_cache_local_disk_bytes",
    "proximadb_compactions_total",
    "proximadb_compaction_bytes_read_total",
    "proximadb_compaction_bytes_written_total",
    "proximadb_compaction_memory_reserved_bytes",
    "proximadb_compaction_scratch_reserved_bytes",
    "proximadb_compaction_spill_total",
    "proximadb_wal_size_bytes",
    "proximadb_ivf_cells_total",
    "proximadb_ivf_cells_probed_total",
    "proximadb_ivf_probed_rows_total",
    "proximadb_ivf_region_a_bytes_read_total",
    "proximadb_ivf_region_b_bytes_read_total",
    "proximadb_ivf_fetch_rounds_total",
    "proximadb_ivf_whole_region_fallback_total",
    "proximadb_read_ranges_fetch_rounds_total",
    "proximadb_read_ranges_max_inflight",
    "proximadb_pax_metadata_ops_agg_total",
    "proximadb_ivf_probe_rank_us_agg_total",
    "proximadb_ivf_probe_rank_calls_agg_total",
    "proximadb_region_b_rerank_us_agg_total",
    "proximadb_region_d_fetch_decode_us_agg_total",
    "proximadb_rehydrate_us_agg_total",
    "proximadb_cascade_total_us_agg_total",
)


def directory_size_bytes(root: Path) -> int:
    """Return allocated logical bytes below ``root`` without following links."""
    if not root.exists():
        return 0
    total = 0
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except FileNotFoundError:
                        # Spill phases atomically retire runs while sampling.
                        # A disappearing entry contributes zero to this sample.
                        continue
        except FileNotFoundError:
            # A completed task may reclaim its whole directory after it was
            # queued by the parent scan.
            continue
    return total


class ProcessScratchSampler:
    """Low-rate external sampler for one server and its admitted scratch root."""

    def __init__(
        self,
        process_id: int,
        scratch_root: Path,
        interval_seconds: float = 0.25,
    ):
        if process_id <= 0:
            raise RuntimeError("resource sampler requires a positive process id")
        if interval_seconds <= 0:
            raise RuntimeError("resource sampler interval must be positive")
        self.process_id = process_id
        self.scratch_root = scratch_root
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._sample_count = 0
        self._baseline_rss: int | None = None
        self._peak_rss = 0
        self._baseline_scratch: int | None = None
        self._peak_scratch = 0

    def _process_rss_bytes(self) -> int:
        proc_statm = Path(f"/proc/{self.process_id}/statm")
        if proc_statm.exists():
            fields = proc_statm.read_text().split()
            if len(fields) < 2:
                raise RuntimeError(
                    f"cannot parse RSS for server process {self.process_id}"
                )
            return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(self.process_id)],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        raw = completed.stdout.strip()
        if completed.returncode != 0 or not raw:
            raise RuntimeError(
                f"cannot sample RSS for server process {self.process_id}"
            )
        return int(raw.splitlines()[-1].strip()) * 1024

    def _scratch_bytes(self) -> int:
        return directory_size_bytes(self.scratch_root)

    def sample_once(self) -> None:
        rss = self._process_rss_bytes()
        scratch = self._scratch_bytes()
        if self._baseline_rss is None:
            self._baseline_rss = rss
            self._baseline_scratch = scratch
        self._peak_rss = max(self._peak_rss, rss)
        self._peak_scratch = max(self._peak_scratch, scratch)
        self._sample_count += 1

    def _run(self) -> None:
        try:
            while not self._stop.wait(self.interval_seconds):
                self.sample_once()
        except Exception as error:  # surfaced synchronously by stop()
            self._error = error
            self._stop.set()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("resource sampler was already started")
        self.sample_once()
        self._thread = threading.Thread(
            target=self._run,
            name="proximadb-benchmark-resource-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(5.0, self.interval_seconds * 4))
        self._thread = None
        if self._error is not None:
            raise RuntimeError(f"resource sampler failed: {self._error}")

    def snapshot(self) -> dict:
        if self._sample_count == 0:
            raise RuntimeError("resource sampler recorded no samples")
        baseline_rss = self._baseline_rss or 0
        baseline_scratch = self._baseline_scratch or 0
        return {
            "sample_interval_seconds": self.interval_seconds,
            "sample_count": self._sample_count,
            "baseline_process_rss_bytes": baseline_rss,
            "peak_process_rss_bytes": self._peak_rss,
            "peak_process_rss_delta_bytes": max(0, self._peak_rss - baseline_rss),
            "baseline_scratch_bytes": baseline_scratch,
            "peak_scratch_bytes": self._peak_scratch,
            "peak_scratch_delta_bytes": max(0, self._peak_scratch - baseline_scratch),
        }


def compute_profile(machine: str | None = None) -> dict:
    """Describe the kernels this fixed Euclidean PAX benchmark exercises."""
    architecture = (machine or platform.machine()).lower()
    if architecture in {"arm64", "aarch64"}:
        sq8_kernel = "neon_fused_decode_distance"
        dispatch = "compile_time_aarch64"
    elif architecture in {"x86_64", "amd64"}:
        sq8_kernel = "avx2_or_scalar_fused_decode_distance"
        dispatch = "runtime_feature_detection"
    else:
        sq8_kernel = "scalar_fused_decode_distance"
        dispatch = "portable_fallback"
    return {
        "architecture": architecture,
        "distance_metric": "euclidean_l2",
        "region_a_filter_kernel": "rabitq_query_bound_lookup_table",
        "region_b_sq8_l2_kernel": sq8_kernel,
        "dispatch": dispatch,
        "gpu_role": "not_used_by_pax_rabitq_sq8_search",
        "source": {
            "rerank_call": "src/storage/engines/sst/segment_format.rs",
            "sq8_dispatch": (
                "crates/horizontal/proximadb-codec/src/baseline/functions/sq8.rs"
            ),
        },
    }


def request_json(
    url: str, method: str = "GET", body: object | None = None, timeout: int = 180
) -> dict:
    data = None if body is None else json.dumps(body).encode()
    headers = {} if data is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    return json.loads(payload) if payload else {}


def count_fixed_records(path: Path, scalar_bytes: int) -> tuple[int, int]:
    with path.open("rb") as source:
        header = source.read(4)
    if len(header) != 4:
        raise RuntimeError(f"{path}: missing vector header")
    dimension = struct.unpack("<i", header)[0]
    if dimension <= 0:
        raise RuntimeError(f"{path}: invalid dimension {dimension}")
    record_bytes = 4 + dimension * scalar_bytes
    size = path.stat().st_size
    if size % record_bytes:
        raise RuntimeError(f"{path}: {size} bytes is not a multiple of {record_bytes}")
    return size // record_bytes, dimension


def read_fvecs(path: Path, start: int, count: int) -> list[list[float]]:
    total, dimension = count_fixed_records(path, 4)
    if start < 0 or count <= 0 or start + count > total:
        raise RuntimeError(f"{path}: requested [{start}, {start + count}) of {total}")
    record_bytes = 4 + 4 * dimension
    vectors: list[list[float]] = []
    with path.open("rb") as source:
        source.seek(start * record_bytes)
        for _ in range(count):
            record = source.read(record_bytes)
            encoded_dimension = struct.unpack_from("<i", record, 0)[0]
            if encoded_dimension != dimension:
                raise RuntimeError(f"{path}: variable dimension encountered")
            vectors.append(list(struct.unpack_from(f"<{dimension}f", record, 4)))
    return vectors


def inspect_u8bin(path: Path) -> tuple[int, int, int]:
    """Return physical rows, dimension, and source-declared rows.

    BIGANN publishes one 1B-row object and defines smaller corpora as byte
    prefixes. A prefix therefore retains the 1B source header while its
    physical payload contains fewer rows. Both counts are evidence: callers
    must bound reads by the physical payload and record the declared count.
    """
    with path.open("rb") as source:
        header = source.read(8)
    if len(header) != 8:
        raise RuntimeError(f"{path}: missing u8bin header")
    declared_rows, dimension = struct.unpack("<II", header)
    if declared_rows <= 0 or dimension <= 0:
        raise RuntimeError(
            f"{path}: invalid u8bin shape ({declared_rows}, {dimension})"
        )
    payload_bytes = path.stat().st_size - len(header)
    if payload_bytes < 0 or payload_bytes % dimension:
        raise RuntimeError(f"{path}: partial dense row in u8bin payload")
    physical_rows = payload_bytes // dimension
    if physical_rows <= 0 or physical_rows > declared_rows:
        raise RuntimeError(
            f"{path}: physical rows {physical_rows} are outside declared "
            f"range 1..{declared_rows}"
        )
    return physical_rows, dimension, declared_rows


def vector_source_geometry(path: Path, vector_format: str) -> tuple[int, int, int]:
    if vector_format == "fvecs":
        rows, dimension = count_fixed_records(path, 4)
        return rows, dimension, rows
    if vector_format == "u8bin":
        return inspect_u8bin(path)
    raise RuntimeError(f"unsupported vector format: {vector_format}")


def read_u8bin(path: Path, start: int, count: int) -> list[list[float]]:
    total, dimension, _ = inspect_u8bin(path)
    if start < 0 or count <= 0 or start + count > total:
        raise RuntimeError(f"{path}: requested [{start}, {start + count}) of {total}")
    vectors: list[list[float]] = []
    with path.open("rb") as source:
        source.seek(8 + start * dimension)
        for _ in range(count):
            record = source.read(dimension)
            if len(record) != dimension:
                raise RuntimeError(f"{path}: truncated u8bin vector")
            vectors.append([float(value) for value in record])
    return vectors


def read_vectors(
    path: Path, vector_format: str, start: int, count: int
) -> list[list[float]]:
    if vector_format == "fvecs":
        return read_fvecs(path, start, count)
    if vector_format == "u8bin":
        return read_u8bin(path, start, count)
    raise RuntimeError(f"unsupported vector format: {vector_format}")


def read_ivecs(path: Path, start: int, count: int) -> list[list[int]]:
    total, dimension = count_fixed_records(path, 4)
    if start < 0 or count <= 0 or start + count > total:
        raise RuntimeError(f"{path}: requested [{start}, {start + count}) of {total}")
    record_bytes = 4 + 4 * dimension
    vectors: list[list[int]] = []
    with path.open("rb") as source:
        source.seek(start * record_bytes)
        for _ in range(count):
            record = source.read(record_bytes)
            encoded_dimension = struct.unpack_from("<i", record, 0)[0]
            if encoded_dimension != dimension:
                raise RuntimeError(f"{path}: variable dimension encountered")
            vectors.append(list(struct.unpack_from(f"<{dimension}i", record, 4)))
    return vectors


def count_bigann_truth_records(path: Path) -> tuple[int, int]:
    with path.open("rb") as source:
        header = source.read(8)
    if len(header) != 8:
        raise RuntimeError(f"{path}: missing BIGANN ground-truth header")
    rows, width = struct.unpack("<II", header)
    if rows <= 0 or width <= 0:
        raise RuntimeError(
            f"{path}: invalid BIGANN ground-truth shape ({rows}, {width})"
        )
    expected_bytes = 8 + rows * width * 8
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"{path}: BIGANN ground truth has {actual_bytes} bytes, "
            f"expected {expected_bytes}"
        )
    return rows, width


def count_truth_records(path: Path, truth_format: str) -> tuple[int, int]:
    if truth_format == "ivecs":
        return count_fixed_records(path, 4)
    if truth_format == "bigann-bin":
        return count_bigann_truth_records(path)
    raise RuntimeError(f"unsupported ground-truth format: {truth_format}")


def read_bigann_truth_ids(path: Path, start: int, count: int) -> list[list[int]]:
    total, width = count_bigann_truth_records(path)
    if start < 0 or count <= 0 or start + count > total:
        raise RuntimeError(f"{path}: requested [{start}, {start + count}) of {total}")
    row_bytes = width * 4
    vectors: list[list[int]] = []
    with path.open("rb") as source:
        source.seek(8 + start * row_bytes)
        for _ in range(count):
            record = source.read(row_bytes)
            if len(record) != row_bytes:
                raise RuntimeError(f"{path}: truncated BIGANN truth ID row")
            vectors.append(list(struct.unpack(f"<{width}i", record)))
    return vectors


def read_truth_ids(
    path: Path, truth_format: str, start: int, count: int
) -> list[list[int]]:
    if truth_format == "ivecs":
        return read_ivecs(path, start, count)
    if truth_format == "bigann-bin":
        return read_bigann_truth_ids(path, start, count)
    raise RuntimeError(f"unsupported ground-truth format: {truth_format}")


def iter_fvec_batches(path: Path, count: int, batch_size: int):
    total, dimension = count_fixed_records(path, 4)
    if count > total:
        raise RuntimeError(f"{path}: requested {count} of {total} vectors")
    record_bytes = 4 + 4 * dimension
    with path.open("rb") as source:
        next_id = 0
        while next_id < count:
            batch = []
            for _ in range(min(batch_size, count - next_id)):
                record = source.read(record_bytes)
                encoded_dimension = struct.unpack_from("<i", record, 0)[0]
                if encoded_dimension != dimension:
                    raise RuntimeError(f"{path}: variable dimension encountered")
                batch.append(
                    {
                        "id": f"v{next_id}",
                        "vector": list(struct.unpack_from(f"<{dimension}f", record, 4)),
                    }
                )
                next_id += 1
            yield batch


def iter_u8bin_batches(path: Path, count: int, batch_size: int):
    total, dimension, _ = inspect_u8bin(path)
    if count > total:
        raise RuntimeError(f"{path}: requested {count} of {total} vectors")
    with path.open("rb") as source:
        source.seek(8)
        next_id = 0
        while next_id < count:
            batch = []
            for _ in range(min(batch_size, count - next_id)):
                record = source.read(dimension)
                if len(record) != dimension:
                    raise RuntimeError(f"{path}: truncated u8bin vector")
                batch.append(
                    {
                        "id": f"v{next_id}",
                        "vector": [float(value) for value in record],
                    }
                )
                next_id += 1
            yield batch


def iter_vector_batches(path: Path, vector_format: str, count: int, batch_size: int):
    if vector_format == "fvecs":
        yield from iter_fvec_batches(path, count, batch_size)
        return
    if vector_format == "u8bin":
        yield from iter_u8bin_batches(path, count, batch_size)
        return
    raise RuntimeError(f"unsupported vector format: {vector_format}")


def iter_vector_arrow_batches(
    path: Path,
    vector_format: str,
    count: int,
    batch_size: int,
    arrow_module=None,
):
    if arrow_module is None:
        try:
            import pyarrow as arrow_module
        except ImportError as error:
            raise RuntimeError(
                "Arrow Flight ingest requires PyArrow; run the harness with "
                "the repository Python environment"
            ) from error
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "Arrow Flight ingest requires NumPy; run the harness with the "
            "repository Python environment"
        ) from error

    total, dimension, _ = vector_source_geometry(path, vector_format)
    if count > total:
        raise RuntimeError(f"{path}: requested {count} of {total} vectors")
    if vector_format == "fvecs":
        header_bytes = 0
        source_row_bytes = 4 + dimension * 4
    elif vector_format == "u8bin":
        header_bytes = 8
        source_row_bytes = dimension
    else:
        raise RuntimeError(f"unsupported vector format: {vector_format}")

    with path.open("rb") as source:
        source.seek(header_bytes)
        next_id = 0
        while next_id < count:
            rows = min(batch_size, count - next_id)
            encoded = source.read(rows * source_row_bytes)
            if len(encoded) != rows * source_row_bytes:
                raise RuntimeError(f"{path}: truncated {vector_format} batch")
            if vector_format == "fvecs":
                raw_ints = np.frombuffer(encoded, dtype="<i4").reshape(
                    rows, dimension + 1
                )
                if not np.all(raw_ints[:, 0] == dimension):
                    raise RuntimeError(f"{path}: variable fvec dimensions")
                vectors = (
                    np.frombuffer(encoded, dtype="<f4")
                    .reshape(rows, dimension + 1)[:, 1:]
                    .copy()
                )
            else:
                vectors = (
                    np.frombuffer(encoded, dtype=np.uint8)
                    .reshape(rows, dimension)
                    .astype(np.float32)
                )
            ids = arrow_module.array(
                [f"v{row}" for row in range(next_id, next_id + rows)],
                type=arrow_module.utf8(),
            )
            values = arrow_module.array(
                vectors.reshape(-1), type=arrow_module.float32()
            )
            vector_array = arrow_module.FixedSizeListArray.from_arrays(
                values, dimension
            )
            yield arrow_module.record_batch(
                [ids, vector_array],
                names=["id", "vector"],
            )
            next_id += rows


def summarize_values(values: list[int] | list[float]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    return {
        "min": ordered[0],
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def fnv1a64(data: bytes) -> int:
    value = 0xCBF29CE484222325
    for byte in data:
        value ^= byte
        value = (value * 0x00000100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


def parse_a0_geometry(a0: bytes) -> dict:
    """Parse the persisted coarse-model shape without interpreting vectors."""
    if len(a0) < 48 or a0[:4] != A0_MAGIC:
        raise RuntimeError("invalid A0 coarse directory")
    if a0[4] != 1:
        raise RuntimeError(f"unsupported A0 version {a0[4]}")
    n_comp = struct.unpack_from("<H", a0, 6)[0]
    cells = struct.unpack_from("<I", a0, 8)[0]
    dimension = struct.unpack_from("<I", a0, 12)[0]
    seed = struct.unpack_from("<Q", a0, 16)[0]
    trained_rows = struct.unpack_from("<Q", a0, 24)[0]
    covered_rows = struct.unpack_from("<Q", a0, 32)[0]
    expected_length = (
        40
        + dimension * 4
        + n_comp * dimension * 4
        + cells * n_comp * 4
        + cells * 4
        + cells * 72
        + 8
    )
    if len(a0) != expected_length:
        raise RuntimeError(f"A0 length {len(a0)} != expected {expected_length}")
    stored_checksum = struct.unpack_from("<Q", a0, len(a0) - 8)[0]
    if fnv1a64(a0[:-8]) != stored_checksum:
        raise RuntimeError("A0 checksum mismatch")

    offset = 40 + dimension * 4 + n_comp * dimension * 4 + cells * n_comp * 4
    radii = list(struct.unpack_from(f"<{cells}f", a0, offset))
    offset += cells * 4
    cell_rows = []
    for _ in range(cells):
        row_begin, row_end = struct.unpack_from("<QQ", a0, offset)
        if row_end < row_begin:
            raise RuntimeError("A0 cell has a descending row range")
        cell_rows.append(row_end - row_begin)
        offset += 72
    if sum(cell_rows) != covered_rows:
        raise RuntimeError(
            f"A0 cell rows {sum(cell_rows)} != covered rows {covered_rows}"
        )
    nonempty = [rows for rows in cell_rows if rows]
    row_summary = summarize_values(cell_rows)
    mean_rows = row_summary["mean"] or 0.0
    return {
        "coarse_cells": cells,
        "coarse_components": n_comp,
        "coarse_dimension": dimension,
        "coarse_seed": seed,
        "coarse_trained_rows": trained_rows,
        "coarse_rows_covered": covered_rows,
        "training_rows_per_cell": (trained_rows / cells if cells else None),
        "empty_cells": cells - len(nonempty),
        "empty_cell_fraction": ((cells - len(nonempty)) / cells if cells else None),
        "cell_rows": cell_rows,
        "cell_row_summary": row_summary,
        "cell_row_max_to_mean": (max(cell_rows) / mean_rows if mean_rows else None),
        "radii": radii,
        "radius_summary": summarize_values(radii),
    }


def parse_pax(path: Path, root: Path) -> dict:
    size = path.stat().st_size
    if size < 25:
        raise RuntimeError(f"{path}: too short for a coalesced PAX segment")
    with path.open("rb") as segment:
        header = segment.read(72)
        segment.seek(-16, os.SEEK_END)
        tail = segment.read(16)
        if tail[8:] != PAX_MAGIC:
            raise RuntimeError(f"{path}: missing coalesced PAX tail")
        footer_len = struct.unpack_from("<Q", tail, 0)[0]
        if footer_len < 9 or footer_len + 16 > size:
            raise RuntimeError(f"{path}: invalid footer length {footer_len}")
        segment.seek(-(16 + footer_len), os.SEEK_END)
        footer_prefix = segment.read(9)
    if footer_prefix[0] != 1:
        raise RuntimeError(f"{path}: unsupported footer version {footer_prefix[0]}")
    if header[:4] != PAX_HEADER_MAGIC:
        raise RuntimeError(f"{path}: legacy PAX layout cannot prove row count")
    coarse = {}
    a0_offset, a0_length = struct.unpack_from("<QQ", header, 56)
    if a0_length > 0:
        if a0_length < 48:
            raise RuntimeError(f"{path}: invalid A0 length {a0_length}")
        if a0_offset + a0_length > size:
            raise RuntimeError(
                f"{path}: A0 extents [{a0_offset}+{a0_length}] exceed the "
                f"segment ({size} B) — offsets 56..72 are body bytes on the "
                "pre-collapse 56-byte-header v1 layout (legacy layouts "
                "unsupported; harness beds are always freshly written)"
            )
        with path.open("rb") as segment:
            segment.seek(a0_offset)
            a0 = segment.read(a0_length)
        if len(a0) != a0_length:
            raise RuntimeError(f"{path}: truncated A0 directory")
        coarse = parse_a0_geometry(a0)
    return {
        "path": str(path.relative_to(root)),
        "bytes": size,
        "rows": struct.unpack_from("<Q", footer_prefix, 1)[0],
        "layout_version": header[4],
        "mtime_ns": path.stat().st_mtime_ns,
        **coarse,
    }


def pax_geometry(root: Path) -> dict:
    segments = [
        parse_pax(path, root) for path in sorted(root.rglob("*.pax")) if path.is_file()
    ]
    return {
        "segment_count": len(segments),
        "row_count": sum(item["rows"] for item in segments),
        "bytes": sum(item["bytes"] for item in segments),
        "segments": segments,
    }


def materialize_footer_geometry(fetch_range, inventory: dict) -> dict:
    """TD-BENCH-2 fix: footer-only remote geometry, shared by the Azure and
    S3 readers. Proving a segment's row count needs the 72-byte header, the
    16-byte tail (footer length + PAX magic), the 9-byte footer prefix (row
    count), and optionally the small A0 directory — KBs of ranged reads per
    segment instead of whole-segment downloads (~267 MB), so settle cannot
    lose a materialize-vs-deadline race at high RTT. ``fetch_range(name,
    start, end)`` is the backend's inclusive byte-range primitive.
    """
    segments = []
    for blob in inventory["segments"]:
        name = blob["path"]
        size = int(blob["bytes"])
        if size < 25:
            raise RuntimeError(f"{name}: too short for a coalesced PAX segment")
        header = fetch_range(name, 0, 71)
        if header[:4] != PAX_HEADER_MAGIC:
            raise RuntimeError(f"{name}: legacy PAX layout cannot prove row count")
        tail = fetch_range(name, size - 16, size - 1)
        if tail[8:] != PAX_MAGIC:
            raise RuntimeError(f"{name}: missing coalesced PAX tail")
        footer_len = struct.unpack_from("<Q", tail, 0)[0]
        if footer_len < 9 or footer_len + 16 > size:
            raise RuntimeError(f"{name}: invalid footer length {footer_len}")
        footer_start = size - (16 + footer_len)
        footer_prefix = fetch_range(name, footer_start, footer_start + 8)
        if footer_prefix[0] != 1:
            raise RuntimeError(f"{name}: unsupported footer version {footer_prefix[0]}")
        coarse = {}
        # A0 presence is authoritative from the header's a0 extents (same
        # offsets in every layout version; the post-collapse v1 writer emits
        # the coarse directory in v1 segments — header[4]==3 is the retired
        # two-level marker).
        a0_offset, a0_length = struct.unpack_from("<QQ", header, 56)
        if a0_length >= 48:
            if a0_offset + a0_length > size:
                # Offsets 56..72 hold segment BODY bytes on the pre-collapse
                # 56-byte-header v1 layout (SEG_HEADER_PREFIX_LEN was 56 at
                # 47bef62e); a junk u64 there means a legacy/misread header.
                # Fresh-prefix beds are always written by the current binary,
                # so reaching this is a provenance bug — fail loudly.
                raise RuntimeError(
                    f"{name}: A0 extents [{a0_offset}+{a0_length}] exceed the "
                    f"object ({size} B) — offsets 56..72 are body bytes, i.e. a "
                    "pre-collapse 56-byte-header v1 segment (legacy layouts "
                    "unsupported; harness beds are always freshly written)"
                )
            a0 = fetch_range(name, a0_offset, a0_offset + a0_length - 1)
            if len(a0) == a0_length:
                coarse = parse_a0_geometry(a0)
        segments.append(
            {
                "path": name,
                "bytes": size,
                "rows": struct.unpack_from("<Q", footer_prefix, 1)[0],
                "layout_version": header[4],
                "mtime_ns": 0,
                "blob_etag": str(blob.get("etag", "")),
                **coarse,
            }
        )
    return {
        "segment_count": len(segments),
        "row_count": sum(item["rows"] for item in segments),
        "bytes": sum(item["bytes"] for item in segments),
        "segments": segments,
    }


class AzureCliPaxGeometry:
    """Read final PAX geometry from Azure/Azurite without touching server metrics.

    Query GET counters live inside the measured server process. The Azure CLI
    inventory/download calls here are out-of-process evidence reads performed
    before query sweeps. They cannot inflate or warm ProximaDB's cache counters,
    although they may warm Azurite's host page cache.
    """

    def __init__(self, storage_url: str, snapshot_root: Path):
        parsed = urlparse(storage_url)
        if parsed.scheme not in {"adls", "az", "azure"}:
            raise RuntimeError(
                f"Azure geometry requires canonical az:// storage "
                f"(azure:// and adls:// aliases are accepted), got {storage_url}"
            )
        if not parsed.netloc:
            raise RuntimeError(f"Azure storage URL has no container: {storage_url}")
        self.container = parsed.netloc
        self.prefix = parsed.path.strip("/")
        self.snapshot_root = snapshot_root

    _AZ_TIMEOUT_SECS = 300

    @staticmethod
    def _run_bounded(command: list[str], timeout: int) -> str:
        """TD-RDSTRAT-12 §3 defensive: az CLI calls are bounded — a hung
        subprocess (telemetry/DNS/proxy stall) previously froze the settle
        loop forever with no error and no progress. Timeouts surface as
        RuntimeError, which the settle loop already treats as transient."""
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"az CLI timed out after {timeout}s: {command[2:4]}"
            ) from error
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"az CLI failed ({error.returncode}): "
                f"{(error.stderr or '')[:400]}"
            ) from error
        return completed.stdout

    @staticmethod
    def _authentication_args() -> list[str]:
        """Forward an explicit connection string when the caller supplied one.

        Azure CLI does not consistently consume AZURE_STORAGE_CONNECTION_STRING
        as implicit auth across CLI versions. The benchmark already requires it
        for Azurite, so forwarding it makes inventory and snapshot reads use the
        same emulator endpoint as the measured server. Real Azure runs without
        this variable retain the CLI's normal identity/account resolution.
        """
        connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        return (
            ["--connection-string", connection_string]
            if connection_string is not None
            else []
        )

    def _list_blobs(self) -> list[dict]:
        command = [
            "az",
            "storage",
            "blob",
            "list",
            "--container-name",
            self.container,
            "--output",
            "json",
        ]
        if self.prefix:
            command.extend(["--prefix", self.prefix])
        command.extend(self._authentication_args())
        payload = json.loads(self._run_bounded(command, self._AZ_TIMEOUT_SECS))
        if not isinstance(payload, list):
            raise RuntimeError("Azure CLI blob inventory was not a JSON list")
        # Azure's server-side prefix match is lexical, not path-segment aware:
        # requesting ``run-3m3`` also returns ``run-3m3-ncomp128``. Enforce the
        # canonical object-prefix boundary locally so geometry, empty-prefix
        # checks, and evidence snapshots cannot mix sibling benchmark beds.
        if self.prefix:
            descendant_prefix = f"{self.prefix}/"
            payload = [
                item
                for item in payload
                if str(item.get("name", "")) == self.prefix
                or str(item.get("name", "")).startswith(descendant_prefix)
            ]
        return payload

    def require_empty_prefix(self) -> None:
        blobs = self._list_blobs()
        if blobs:
            names = [item.get("name", "<unnamed>") for item in blobs[:5]]
            raise RuntimeError(
                "Azure benchmark prefix is not empty: "
                f"adls://{self.container}/{self.prefix}, first_blobs={names}"
            )

    def _snapshot_target(self, blob_name: str) -> Path:
        relative = PurePosixPath(blob_name)
        if relative.is_absolute() or any(
            component in {"", ".", ".."} for component in relative.parts
        ):
            raise RuntimeError(
                f"unsafe Azure blob name in geometry snapshot: {blob_name!r}"
            )
        return self.snapshot_root.joinpath(*relative.parts)

    def inventory(self) -> dict:
        segments = []
        for blob in self._list_blobs():
            name = str(blob.get("name", ""))
            if not name.endswith(".pax"):
                continue
            properties = blob.get("properties") or {}
            size = properties.get("contentLength", blob.get("contentLength"))
            if size is None:
                raise RuntimeError(f"Azure blob has no content length: {name}")
            segments.append(
                {
                    "path": name,
                    "bytes": int(size),
                    "etag": str(properties.get("etag", blob.get("etag", ""))),
                    "last_modified": str(
                        properties.get("lastModified", blob.get("lastModified", ""))
                    ),
                }
            )
        segments.sort(key=lambda item: item["path"])
        return {
            "segment_count": len(segments),
            "bytes": sum(item["bytes"] for item in segments),
            "segments": segments,
        }

    @staticmethod
    def stable_signature(inventory: dict) -> tuple:
        return tuple(
            (item["path"], item["bytes"], item["etag"])
            for item in inventory["segments"]
        )

    def _download_range(self, blob_name: str, start: int, end: int) -> bytes:
        """TD-BENCH-2 fix: inclusive byte-range fetch (KBs) instead of
        whole-segment downloads (~267 MB) — removes the
        materialize-vs-deadline race that silently discarded high-RTT arms."""
        target = self.snapshot_root / "range-scratch.bin"
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        command = [
            "az", "storage", "blob", "download",
            "--container-name", self.container,
            "--name", blob_name,
            "--file", str(target),
            "--overwrite", "true",
            "--start-range", str(start),
            "--end-range", str(end),
            "--no-progress", "--output", "none",
        ]
        command.extend(self._authentication_args())
        try:
            subprocess.run(
                command, check=True, capture_output=True,
                text=True, timeout=300,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"az ranged download timed out after 300s: {blob_name}[{start},{end}]"
            ) from error
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"az ranged download failed ({error.returncode}) for "
                f"{blob_name}[{start},{end}]: {(error.stderr or '')[:400]}"
            ) from error
        return target.read_bytes()

    def materialize(self, inventory: dict) -> dict:
        return materialize_footer_geometry(self._download_range, inventory)


class S3PaxGeometry:
    """Read final PAX geometry from an S3-shaped endpoint (e.g. MinIO) via
    boto3, without touching server metrics.

    Same out-of-process contract as ``AzureCliPaxGeometry``: inventory and
    footer reads here cannot inflate or warm the measured ProximaDB process's
    cache counters. boto3 calls carry explicit connect/read timeouts so a hung
    endpoint fails the settle loop loudly instead of freezing it (the
    TD-BENCH-2 defensive posture, ported from the az CLI runner).
    """

    _CONNECT_TIMEOUT_SECS = 10
    _READ_TIMEOUT_SECS = 120

    def __init__(
        self,
        storage_url: str,
        snapshot_root: Path,
        endpoint_url: str,
        region: str = "us-east-1",
    ):
        parsed = urlparse(storage_url)
        if parsed.scheme != "s3":
            raise RuntimeError(
                f"S3 geometry requires canonical s3:// storage, got {storage_url}"
            )
        if not parsed.netloc:
            raise RuntimeError(f"S3 storage URL has no bucket: {storage_url}")
        if not endpoint_url:
            raise RuntimeError("S3 geometry requires an explicit endpoint URL")
        self.bucket = parsed.netloc
        self.prefix = parsed.path.strip("/")
        self.endpoint_url = endpoint_url
        self.region = region
        self.snapshot_root = snapshot_root
        try:
            import boto3
            import botocore.config
        except ImportError as error:
            raise RuntimeError(
                "S3 geometry requires boto3; run the harness with the "
                "repository Python environment"
            ) from error
        access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        if not access_key or not secret_key:
            raise RuntimeError(
                "S3 geometry requires AWS_ACCESS_KEY_ID and "
                "AWS_SECRET_ACCESS_KEY in the environment"
            )
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=botocore.config.Config(
                s3={"addressing_style": "path"},
                connect_timeout=self._CONNECT_TIMEOUT_SECS,
                read_timeout=self._READ_TIMEOUT_SECS,
                retries={"max_attempts": 2},
            ),
        )

    def _list_objects(self) -> list[dict]:
        """Paginated key inventory under the configured prefix."""
        objects: list[dict] = []
        paginator = self._client.get_paginator("list_objects_v2")
        try:
            pages = (
                paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)
                if self.prefix
                else paginator.paginate(Bucket=self.bucket)
            )
            for page in pages:
                for item in page.get("Contents", []):
                    objects.append(
                        {
                            "name": str(item["Key"]),
                            "size": int(item["Size"]),
                            "etag": str(item.get("ETag", "")).strip('"'),
                        }
                    )
        except Exception as error:
            raise RuntimeError(f"S3 list_objects_v2 failed: {error}") from error
        # S3's server-side prefix match is lexical, not path-segment aware:
        # prefix "run-1" also returns "run-10/...". Enforce the canonical
        # object-prefix boundary locally so geometry, empty-prefix checks,
        # and evidence snapshots cannot mix sibling benchmark beds (same
        # defense as the Azure reader).
        if self.prefix:
            descendant_prefix = f"{self.prefix}/"
            objects = [
                item
                for item in objects
                if item["name"] == self.prefix
                or item["name"].startswith(descendant_prefix)
            ]
        return objects

    def require_empty_prefix(self) -> None:
        objects = self._list_objects()
        if objects:
            names = [item["name"] for item in objects[:5]]
            raise RuntimeError(
                f"S3 benchmark prefix is not empty: s3://{self.bucket}/{self.prefix}, "
                f"first_objects={names}"
            )

    def inventory(self) -> dict:
        segments = [
            {
                "path": item["name"],
                "bytes": item["size"],
                "etag": item["etag"],
            }
            for item in self._list_objects()
            if item["name"].endswith(".pax")
        ]
        segments.sort(key=lambda item: item["path"])
        return {
            "segment_count": len(segments),
            "bytes": sum(item["bytes"] for item in segments),
            "segments": segments,
        }

    @staticmethod
    def stable_signature(inventory: dict) -> tuple:
        return tuple(
            (item["path"], item["bytes"], item["etag"])
            for item in inventory["segments"]
        )

    def _download_range(self, key: str, start: int, end: int) -> bytes:
        """TD-BENCH-2 ranged fetch: inclusive byte range, KBs per read."""
        try:
            self.snapshot_root.mkdir(parents=True, exist_ok=True)
            response = self._client.get_object(
                Bucket=self.bucket, Key=key, Range=f"bytes={start}-{end}"
            )
            with response["Body"] as stream:
                return stream.read()
        except Exception as error:
            raise RuntimeError(
                f"S3 ranged read failed for {key}[{start},{end}]: {error}"
            ) from error

    def materialize(self, inventory: dict) -> dict:
        return materialize_footer_geometry(self._download_range, inventory)


class LocalPaxGeometry:
    """file:// adapter for the explicit-flush epoch proof: the same
    inventory / stable_signature contract as the remote readers, backed by
    the local PAX footer parser. Out-of-process by construction (reads the
    settle root, not server metrics)."""

    def __init__(self, sst_root: Path):
        self.sst_root = sst_root

    def require_empty_prefix(self) -> None:
        pass  # the fresh-root check is require_empty_directory(root)

    def inventory(self) -> dict:
        return pax_geometry(self.sst_root)

    @staticmethod
    def stable_signature(inventory: dict) -> tuple:
        return stable_signature(inventory)


def stable_signature(geometry: dict) -> tuple:
    return tuple(
        (item["path"], item["bytes"], item["rows"], item["mtime_ns"])
        for item in geometry["segments"]
    )


def wal_is_quiescent(wal_bytes: float | None) -> bool:
    """Treat an absent per-collection gauge as zero after WAL retirement.

    The metrics registry removes the labelled sample when the collection has no
    remaining unflushed WAL. Materialization still requires a stable PAX epoch
    and an exact footer row count, so absence cannot by itself admit an
    incomplete segment.
    """
    return wal_bytes is None or wal_bytes == 0


def layout_candidate_is_ready(
    geometry: dict,
    required_layout_version: int | None,
    azure_inventory: bool,
) -> bool:
    """Reject known-transient layouts before starting the stable window.

    Local geometry includes parsed headers and can prove the layout directly.
    Azure inventory is deliberately metadata-only to avoid repeatedly
    downloading a large segment. For the v3 two-level layout, an L0 object is
    the untrained flush artifact and therefore cannot be terminal; training
    compaction publishes it at L1 or above. The eventual downloaded header is
    still the authority before the function returns.
    """
    if required_layout_version is None:
        return True
    segments = geometry.get("segments", [])
    if segments and all("layout_version" in segment for segment in segments):
        # A settled segment must ALSO carry a coarse A0 directory: the
        # required-layout-version=1 default retired the L0-filename gate, and
        # an untrained flush artifact is a v1 segment with a valid row count —
        # only the missing coarse directory distinguishes it from a trained
        # one. Untrained segments are transient (training compaction replaces
        # them), so not-ready keeps the settle loop polling.
        return all(
            segment["layout_version"] == required_layout_version
            and segment.get("coarse_cells")
            for segment in segments
        )
    if not azure_inventory or required_layout_version <= 1:
        return True
    for segment in segments:
        name = Path(segment["path"]).name
        match = re.match(r"L(\d+)_", name)
        if match is None or int(match.group(1)) == 0:
            return False
    return bool(segments)


def wait_for_materialization(
    root: Path,
    server: str,
    collection_id: str,
    expected_rows: int,
    max_segments: int,
    timeout_seconds: int,
    stable_seconds: int,
    azure_geometry: AzureCliPaxGeometry | S3PaxGeometry | None = None,
    required_layout_version: int | None = None,
) -> dict:
    deadline = time.monotonic() + timeout_seconds
    stable_since = None
    prior = None
    last_report = 0.0
    last_parse_error = None
    # TD-RDSTRAT-13 campaign finding: geometry can settle on the training-
    # compaction chain's INTERMEDIATE level (2×L2) and the chain then publishes
    # a higher level (L3) AFTER the stable window, leaving overlapping segments
    # that double-serve rows at query time. A stable window is only trustworthy
    # when the compaction counter is quiet for its whole duration.
    compactions_at_window_start = None
    while time.monotonic() < deadline:
        now = time.monotonic()
        try:
            geometry = (
                azure_geometry.inventory()
                if azure_geometry is not None
                else pax_geometry(root)
            )
            last_parse_error = None
        except RuntimeError as error:
            # Compaction output is visible while it is still being streamed.
            # A missing tail is therefore transient, but it can never satisfy
            # the stable/quiescent gate below.
            last_parse_error = str(error)
            stable_since = None
            prior = None
            if now - last_report >= 15:
                print(f"settle: waiting for valid PAX tail: {error}", flush=True)
                last_report = now
            time.sleep(3)
            continue
        signature = (
            azure_geometry.stable_signature(geometry)
            if azure_geometry is not None
            else stable_signature(geometry)
        )
        metrics_text = scrape_text(server)
        wal_bytes = labelled_metric(
            metrics_text,
            "proximadb_wal_size_bytes",
            "collection",
            collection_id,
        )
        # Unlabeled aggregate counter (sst::metrics) — plain token parse, the
        # labelled helper only matches `name{...}` lines.
        compactions_total = None
        for raw_line in metrics_text.splitlines():
            line = raw_line.strip()
            if line.startswith("proximadb_compactions_total"):
                try:
                    compactions_total = float(line.split()[1])
                except (ValueError, IndexError):
                    compactions_total = None
                break
        observed_rows = geometry.get("row_count")
        row_status = (
            f"{observed_rows:,}/{expected_rows:,}"
            if observed_rows is not None
            else f"pending-footer-snapshot/{expected_rows:,}"
        )
        if now - last_report >= 15:
            print(
                "settle:"
                f" rows={row_status}"
                f" segments={geometry['segment_count']}"
                f" bytes={geometry['bytes']:,}"
                f" wal_unflushed={wal_bytes!r}",
                flush=True,
            )
            last_report = now
        # Compaction publishes its output before deleting inputs, so a transient
        # row sum above `expected_rows` is legal. It must disappear before the
        # stable window; a persistent duplicate/stale segment fails by timeout.
        complete = (
            (observed_rows == expected_rows if azure_geometry is None else True)
            and 0 < geometry["segment_count"] <= max_segments
            and wal_is_quiescent(wal_bytes)
            and layout_candidate_is_ready(
                geometry,
                required_layout_version,
                azure_geometry is not None,
            )
        )
        if complete and signature == prior:
            stable_since = stable_since or now
            if compactions_at_window_start is None:
                compactions_at_window_start = compactions_total
            if compactions_total != compactions_at_window_start:
                # A compaction ran during the window: the settled geometry is
                # an intermediate artifact of an in-flight chain. Restart the
                # window on the post-compaction geometry.
                print(
                    f"settle: compaction advanced ({compactions_at_window_start} -> "
                    f"{compactions_total}); restarting stable window",
                    flush=True,
                )
                stable_since = now
                compactions_at_window_start = compactions_total
            if now - stable_since >= stable_seconds:
                if azure_geometry is not None:
                    geometry = azure_geometry.materialize(geometry)
                    if geometry["row_count"] != expected_rows:
                        raise RuntimeError(
                            "quiescent Azure PAX footer row mismatch: "
                            f"{geometry['row_count']} != {expected_rows}"
                        )
                    if not layout_candidate_is_ready(
                        geometry,
                        required_layout_version,
                        azure_inventory=False,
                    ):
                        stable_since = None
                        prior = signature
                        time.sleep(3)
                        continue
                geometry["wal_unflushed_bytes"] = wal_bytes
                return geometry
        else:
            stable_since = now if complete else None
            compactions_at_window_start = None
        prior = signature
        time.sleep(3)
    try:
        geometry = (
            azure_geometry.inventory()
            if azure_geometry is not None
            else pax_geometry(root)
        )
    except RuntimeError as error:
        last_parse_error = str(error)
        geometry = {"row_count": 0, "segment_count": 0}
    raise RuntimeError(
        "materialization/compaction did not quiesce: "
        f"rows={geometry.get('row_count', 'not-snapshotted')}/{expected_rows}, "
        f"segments={geometry['segment_count']} (max {max_segments}), "
        f"last_parse_error={last_parse_error!r}"
    )


def parse_prometheus(text: str) -> dict[str, float]:
    totals = dict.fromkeys(METRICS, 0.0)
    present: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        token, _, raw_value = line.partition(" ")
        name = token.split("{", 1)[0]
        if name not in totals:
            continue
        try:
            value = float(raw_value.strip().split()[0])
        except (ValueError, IndexError):
            continue
        totals[name] += value
        present.add(name)
    totals["_present"] = sorted(present)  # type: ignore[assignment]
    return totals


# A settle poll is a LIVENESS probe against a server that is deliberately busy:
# training compaction is CPU-bound k-means and can stall the metrics handler for
# far longer than a single request timeout. A 30s ceiling with no retry made one
# slow response fatal to a multi-hour bed build — observed killing a 3.3M build
# twice, and again once PROXIMADB_IVF_TRAIN_SAMPLE was doubled (more training =
# longer stalls). Retry with backoff so a transient stall is survivable, but
# still raise once the budget is spent so a genuinely dead server fails loudly.
SCRAPE_TIMEOUT_SECONDS = 120
SCRAPE_ATTEMPTS = 4

# Graceful-shutdown budget. Scales with collection size because SIGTERM triggers
# a flush of everything still unflushed; see ServerProcess.stop().
SHUTDOWN_GRACE_SECONDS = 120


def scrape_text(server: str) -> str:
    last: Exception | None = None
    for attempt in range(SCRAPE_ATTEMPTS):
        try:
            with urllib.request.urlopen(
                server + "/metrics/prometheus", timeout=SCRAPE_TIMEOUT_SECONDS
            ) as response:
                return response.read().decode()
        except (TimeoutError, OSError, urllib.error.URLError) as error:
            last = error
            if attempt + 1 < SCRAPE_ATTEMPTS:
                print(
                    f"scrape: transient failure ({error!r}); "
                    f"retry {attempt + 1}/{SCRAPE_ATTEMPTS - 1}",
                    flush=True,
                )
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(
        f"metrics scrape failed after {SCRAPE_ATTEMPTS} attempts: {last!r}"
    )


def scrape(server: str) -> dict[str, float]:
    return parse_prometheus(scrape_text(server))


def labelled_metric(
    text: str, name: str, label: str, expected_value: str
) -> float | None:
    """Return one exact labelled Prometheus sample without summing tenants."""
    label_pattern = re.compile(rf'(?:^|,){re.escape(label)}="([^"\\]*(?:\\.[^"\\]*)*)"')
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith(name + "{"):
            continue
        token, _, raw_value = line.partition(" ")
        labels = token[len(name) + 1 : -1]
        match = label_pattern.search(labels)
        if match is None or match.group(1) != expected_value:
            continue
        try:
            return float(raw_value.strip().split()[0])
        except (ValueError, IndexError):
            return None
    return None


def metric_delta(before: dict, after: dict, name: str) -> float:
    present_before = set(before.get("_present", []))
    present_after = set(after.get("_present", []))
    if name not in present_after and name not in present_before:
        # Prometheus collectors may register counters lazily on the first
        # increment. Absence on both sides is therefore an observed zero
        # delta, which is the expected state for a fully eliminated GET path.
        return 0.0
    if name not in present_after:
        raise RuntimeError(f"required metric {name} is absent after the sweep")
    delta = after[name] - before.get(name, 0.0)
    if delta < 0:
        raise RuntimeError(f"metric {name} moved backwards across one process")
    return delta


def percentile(sorted_values: list[float], quantile: float) -> float:
    rank = max(0, math.ceil(quantile * len(sorted_values)) - 1)
    return sorted_values[rank]


def prefix_quality_checkpoints(
    recalls: list[float], latencies: list[float]
) -> list[dict]:
    """Summarize nested 100/1K/10K samples without rerunning any query.

    Recall and latency are query-local, so their prefix estimates can share a
    single execution. Physical-I/O counters remain attached to the complete
    sweep because scraping at each prefix would add control-plane noise.
    """
    if not recalls or len(recalls) != len(latencies):
        raise RuntimeError("quality checkpoints require paired query samples")
    checkpoints = sorted(
        {len(recalls), *[n for n in (100, 1_000, 10_000) if n <= len(recalls)]}
    )
    result = []
    for query_count in checkpoints:
        prefix_latencies = sorted(latencies[:query_count])
        result.append(
            {
                "query_count": query_count,
                "recall_at_k": sum(recalls[:query_count]) / query_count,
                "latency_ms": {
                    "p50": percentile(prefix_latencies, 0.50),
                    "p95": percentile(prefix_latencies, 0.95),
                    "mean": sum(prefix_latencies) / query_count,
                },
            }
        )
    return result


def query_result_identity(returned_ids: list[str]) -> dict[str, object]:
    """Hash one query's result IDs without persisting corpus identifiers.

    Both forms are needed when a behavior-neutral storage experiment changes
    an aggregate recall value: the ordered digest detects rank changes, while
    the set digest distinguishes membership changes from tie-only reordering.
    Each query is hashed separately so the evidence can localize divergence
    without retaining corpus identifiers.
    """
    ordered_json = json.dumps(
        returned_ids, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    set_json = json.dumps(
        sorted(set(returned_ids)), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return {
        "ordered_ids_sha256": hashlib.sha256(ordered_json).hexdigest(),
        "set_ids_sha256": hashlib.sha256(set_json).hexdigest(),
        "result_count": len(returned_ids),
        "unique_result_count": len(set(returned_ids)),
    }


def run_query_sweep(
    server: str,
    collection_id: str,
    query_path: Path,
    groundtruth_path: Path,
    query_start: int,
    query_count: int,
    top_k: int,
    phase: str,
    query_format: str = "fvecs",
    groundtruth_format: str = "ivecs",
    progress_guard=None,
    concurrency: int = 1,
) -> dict:
    if concurrency <= 0:
        raise RuntimeError("query concurrency must be positive")
    queries = read_vectors(query_path, query_format, query_start, query_count)
    groundtruth = read_truth_ids(
        groundtruth_path, groundtruth_format, query_start, query_count
    )
    before = scrape(server)
    latencies = []
    recalls = []
    ordered_result_digests = []
    set_result_digests = []
    result_counts = []
    unique_result_counts = []
    recall_hits_by_query = []
    in_flight = 0
    peak_in_flight = 0
    in_flight_lock = threading.Lock()

    def execute_query(indexed_query):
        nonlocal in_flight, peak_in_flight
        offset, query = indexed_query
        if progress_guard is not None:
            progress_guard()
        with in_flight_lock:
            in_flight += 1
            peak_in_flight = max(peak_in_flight, in_flight)
        try:
            started = time.perf_counter()
            response = request_json(
                f"{server}/api/v2/collections/{collection_id}/search",
                method="POST",
                body={"vector": query, "top_k": top_k},
                timeout=300,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            return offset, latency_ms, response
        finally:
            with in_flight_lock:
                in_flight -= 1

    wall_started = time.perf_counter()
    indexed_queries = list(enumerate(queries))
    if concurrency == 1:
        outcomes = [execute_query(item) for item in indexed_queries]
    else:
        with ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix="pax-query",
        ) as executor:
            # Map returns in input order. Recall and result hashes therefore
            # remain tied to query index, not nondeterministic completion order.
            outcomes = list(executor.map(execute_query, indexed_queries))
    wall_seconds = time.perf_counter() - wall_started

    for expected_offset, (offset, latency_ms, response) in enumerate(outcomes):
        if offset != expected_offset:
            raise RuntimeError(f"{phase}: concurrent query order was not preserved")
        latencies.append(latency_ms)
        response_results = response.get("results")
        if not isinstance(response_results, list):
            raise RuntimeError(f"{phase}: query {offset} returned no result list")
        returned_ids = []
        for item in response_results:
            oid = item.get("id") if isinstance(item, dict) else None
            if not isinstance(oid, str):
                raise RuntimeError(
                    f"{phase}: query {offset} returned a result without a string id"
                )
            returned_ids.append(oid)
        identity = query_result_identity(returned_ids)
        ordered_result_digests.append(identity["ordered_ids_sha256"])
        set_result_digests.append(identity["set_ids_sha256"])
        result_counts.append(identity["result_count"])
        unique_result_counts.append(identity["unique_result_count"])
        returned = set(returned_ids)
        expected = {f"v{row}" for row in groundtruth[offset][:top_k]}
        recall_hits = len(returned & expected)
        recall_hits_by_query.append(recall_hits)
        recalls.append(recall_hits / top_k)
    if progress_guard is not None:
        progress_guard()
    after = scrape(server)
    quality_checkpoints = prefix_quality_checkpoints(recalls, latencies)
    latencies.sort()
    gets = metric_delta(before, after, "proximadb_object_store_gets_total")
    bytes_read = metric_delta(before, after, "proximadb_object_store_bytes_read_total")
    survivor_hits = metric_delta(before, after, "proximadb_survivor_cache_hits")
    survivor_misses = metric_delta(before, after, "proximadb_survivor_cache_misses")
    invariant_hits = metric_delta(
        before, after, "proximadb_segment_invariants_cache_hits_total"
    )
    invariant_misses = metric_delta(
        before, after, "proximadb_segment_invariants_cache_misses_total"
    )
    local_hits = metric_delta(before, after, "proximadb_cache_local_disk_hits_total")
    local_misses = metric_delta(
        before, after, "proximadb_cache_local_disk_misses_total"
    )
    ivf_cells_total = metric_delta(before, after, "proximadb_ivf_cells_total")
    ivf_cells_probed = metric_delta(before, after, "proximadb_ivf_cells_probed_total")
    ivf_probed_rows = metric_delta(before, after, "proximadb_ivf_probed_rows_total")
    ivf_region_a_bytes = metric_delta(
        before, after, "proximadb_ivf_region_a_bytes_read_total"
    )
    ivf_region_b_bytes = metric_delta(
        before, after, "proximadb_ivf_region_b_bytes_read_total"
    )
    ivf_fetch_rounds = metric_delta(before, after, "proximadb_ivf_fetch_rounds_total")
    ivf_whole_region_fallbacks = metric_delta(
        before, after, "proximadb_ivf_whole_region_fallback_total"
    )
    # TD-RDSTRAT-12 §2: bounded-concurrent wave meters. fetch_rounds is a
    # counter delta across the phase; max_inflight is a GAUGE, so its "delta"
    # is the value after the phase = the LAST wave's peak width (not a sum).
    rr_fetch_rounds = metric_delta(
        before, after, "proximadb_read_ranges_fetch_rounds_total"
    )
    rr_max_inflight = after.get("proximadb_read_ranges_max_inflight", 0.0)
    # TD-RDSTRAT-13: per-query metadata ops (HEAD — C3 removes it) and the
    # coarse-probe rank compute term (mean us = total/calls — C2 parallelizes it).
    metadata_ops = metric_delta(before, after, "proximadb_pax_metadata_ops_agg_total")
    rank_us_total = metric_delta(before, after, "proximadb_ivf_probe_rank_us_agg_total")
    rank_calls = metric_delta(before, after, "proximadb_ivf_probe_rank_calls_agg_total")
    rank_us_mean = rank_us_total / rank_calls if rank_calls else 0.0
    # TD-RDSTRAT-13 PR-B tail attribution (per-phase wall deltas).
    tail_attribution = {
        name: metric_delta(before, after, name)
        for name in (
            "proximadb_region_b_rerank_us_agg_total",
            "proximadb_region_d_fetch_decode_us_agg_total",
            "proximadb_rehydrate_us_agg_total",
            "proximadb_cascade_total_us_agg_total",
        )
    }
    survivor_total = survivor_hits + survivor_misses
    invariant_total = invariant_hits + invariant_misses
    result = {
        "phase": phase,
        "query_start": query_start,
        "query_count": query_count,
        "load": {
            "model": "closed_loop_bounded",
            "configured_concurrency": concurrency,
            "peak_in_flight": peak_in_flight,
            "wall_seconds": wall_seconds,
            "qps": query_count / wall_seconds,
        },
        "top_k": top_k,
        "recall_at_k": sum(recalls) / len(recalls),
        "recall_hits_total": sum(recall_hits_by_query),
        "recall_denominator": query_count * top_k,
        "result_identity": {
            "schema": "sha256-query-result-ids-v1",
            "ordered_ids_sha256_by_query": ordered_result_digests,
            "set_ids_sha256_by_query": set_result_digests,
            "recall_hits_by_query": recall_hits_by_query,
            "result_count_by_query": result_counts,
            "unique_result_count_by_query": unique_result_counts,
        },
        "prefix_quality_checkpoints": quality_checkpoints,
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "mean": sum(latencies) / len(latencies),
        },
        "physical_gets": gets,
        "gets_per_query": gets / query_count,
        "bytes_read": bytes_read,
        "bytes_per_query": bytes_read / query_count,
        "survivor": {
            "hits": survivor_hits,
            "misses": survivor_misses,
            "hit_ratio": (survivor_hits / survivor_total if survivor_total else None),
            "resident_bytes": after["proximadb_survivor_cache_bytes"],
        },
        "invariants": {
            "hits": invariant_hits,
            "misses": invariant_misses,
            "hit_ratio": (
                invariant_hits / invariant_total if invariant_total else None
            ),
            "resident_bytes": after["proximadb_segment_invariants_cache_bytes"],
        },
        "local_disk": {
            "hits": local_hits,
            "misses": local_misses,
            "resident_bytes": after["proximadb_cache_local_disk_bytes"],
        },
        "ivf": {
            "cells_total": ivf_cells_total,
            "cells_total_per_query": ivf_cells_total / query_count,
            "cells_probed": ivf_cells_probed,
            "cells_probed_per_query": ivf_cells_probed / query_count,
            "probed_rows": ivf_probed_rows,
            "probed_rows_per_query": ivf_probed_rows / query_count,
            "region_a_bytes": ivf_region_a_bytes,
            "region_a_bytes_per_query": (ivf_region_a_bytes / query_count),
            "region_b_bytes": ivf_region_b_bytes,
            "region_b_bytes_per_query": (ivf_region_b_bytes / query_count),
            "fetch_rounds": ivf_fetch_rounds,
            "fetch_rounds_per_query": ivf_fetch_rounds / query_count,
            "whole_region_fallbacks": ivf_whole_region_fallbacks,
        },
        "rank_and_head": {
            "metadata_ops_per_query": metadata_ops / query_count,
            "probe_rank_us_mean": rank_us_mean,
            "probe_rank_calls": rank_calls,
            "tail_attribution_us": tail_attribution,
        },
        "read_ranges_wave": {
            # TD-RDSTRAT-12 §2: observable proof the bounded-concurrent wave
            # fired (fetch_rounds > 0) and its width (max_inflight). Without
            # this, "the wave fired" is an inference, not a measurement.
            "fetch_rounds": rr_fetch_rounds,
            "fetch_rounds_per_query": rr_fetch_rounds / query_count,
            "max_inflight_last": rr_max_inflight,
        },
        # TD-RDSTRAT-12 §3 round 4: the settled bed's 100 overlapping queries
        # mix a genuinely cold first cohort with warm repeats — the aggregate
        # p50 hides the wave's effect. The cold cohort is the fair verdict
        # population for the >=2x acceptance.
        "cold_cohort": {
            "query_count": min(10, query_count),
            "latency_ms": {
                "p50": percentile(sorted(latencies[: min(10, len(latencies))]), 0.50),
                "p95": percentile(sorted(latencies[: min(10, len(latencies))]), 0.95),
                "mean": (
                    sum(latencies[: min(10, len(latencies))])
                    / min(10, len(latencies))
                    if latencies
                    else 0.0
                ),
            },
            "recall_at_k": (
                sum(recalls[: min(10, len(recalls))]) / min(10, len(recalls))
                if recalls
                else 0.0
            ),
        },
    }
    print(
        f"{phase}: recall@{top_k}={result['recall_at_k']:.4f} "
        f"GET/q={result['gets_per_query']:.2f} "
        f"bytes/q={result['bytes_per_query'] / 1_000_000:.2f}MB "
        f"p50={result['latency_ms']['p50']:.2f}ms "
        f"p95={result['latency_ms']['p95']:.2f}ms "
        f"c={concurrency} qps={result['load']['qps']:.2f} "
        f"cells/q={result['ivf']['cells_probed_per_query']:.2f} "
        f"rows/q={result['ivf']['probed_rows_per_query']:.0f} "
        f"rounds/q={result['ivf']['fetch_rounds_per_query']:.2f} "
        f"local_hits={local_hits:.0f}",
        flush=True,
    )
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_complete_insert(response: dict, expected_count: int) -> None:
    inserted = response.get("inserted_count")
    failed = response.get("failed_count")
    if inserted != expected_count or failed != 0:
        raise RuntimeError(
            "batch insert was not fully admitted: "
            f"expected={expected_count}, inserted_count={inserted!r}, "
            f"failed_count={failed!r}, errors={response.get('errors')!r}"
        )


def write_config(
    path: Path,
    root: Path,
    port: int,
    write_buffer_mb: int,
    flush_vector_threshold: int,
    storage_url: str,
    flush_interval_secs: int = 12,
    flush_floor_predicted_mb: int = 128,
    compaction_max_memory_mb: int = 0,
    compaction_spill_enabled: bool = False,
    compaction_spill_directory: Path | None = None,
    compaction_spill_working_memory_mb: int = 512,
    compaction_spill_scratch_amplification_factor: float = 10.0,
    compaction_spill_available_disk_fraction: float = 0.5,
    compaction_spill_max_disk_mb: int = 0,
) -> None:
    data = root / "data"
    if compaction_spill_enabled and compaction_spill_directory is None:
        raise RuntimeError("enabled compaction spill requires a local directory")
    spill_directory_line = (
        f'spill_directory = "{compaction_spill_directory}"\n'
        if compaction_spill_directory is not None
        else ""
    )
    config = f"""[server]
node_id = "sift1m-get-reduction"
bind_address = "127.0.0.1"
port = {port}
data_dir = "{data}"

[server.tenant]
mode = "single_tenant"
default_tenant = "default"

[storage]
metadata_url = "file://{data / "metadata"}"
mmap_enabled = false

[storage.optimization]
enable_mmap = false

[storage.compaction_config]
memory_amplification_factor = 12.0
memory_budget_fraction = 0.25
available_memory_fraction = 0.5
max_memory_mb = {compaction_max_memory_mb}
spill_enabled = {str(compaction_spill_enabled).lower()}
{spill_directory_line}spill_working_memory_mb = {compaction_spill_working_memory_mb}
spill_scratch_amplification_factor = {compaction_spill_scratch_amplification_factor}
spill_available_disk_fraction = {compaction_spill_available_disk_fraction}
spill_max_disk_mb = {compaction_spill_max_disk_mb}

[[storage.storage_locations]]
url = "{storage_url}"
weight = 1
tags = ["durable", "benchmark"]

[storage.wal_config]
write_buffer_directory = "file://{data / "wal"}"
# TD-IOBUDGET-2: the global WAL manifest must not live on the object store —
# GlobalManifestService builds its own bare FilesystemFactory (create_default),
# which cannot open s3:// without endpoint config. Pin the manifest next to the
# local write buffer.
global_manifest_url = "file://{data / "manifest"}"
enable_wal = true
sync_mode = "PerBatch"
write_buffer_size_mb = {write_buffer_mb}
flush_interval_secs = {flush_interval_secs}
flush_floor_predicted_mb = {flush_floor_predicted_mb}

[storage.sst_config]
data_directory = "{data / "sst"}"
mmap_enabled = false
segment_invariants_cache_mb = 256
survivor_cache_mb = 1024
cache_on_write = "all"

[api]
unified_mode = true
unified_port = {port}
rest_port = {port}
grpc_port = {port + 1}
arrow_flight_port = {port + 2}
pg_port = {port + 3}
internal_mux_port = {port + 10001}

[monitoring]
metrics_enabled = true
log_level = "info"
"""
    path.write_text(config)


def effective_flush_interval(
    explicit_flush_every_rows: int | None,
    configured_interval_secs: int | None,
) -> int:
    if configured_interval_secs is not None:
        if configured_interval_secs <= 0:
            raise RuntimeError("--flush-interval-secs must be positive")
        return configured_interval_secs
    return 3600 if explicit_flush_every_rows is not None else 12


class OwnedServer:
    def __init__(
        self,
        binary: Path,
        config: Path,
        server: str,
        log_path: Path,
        local_disk_path: Path | None,
        ivf_k: int | None = None,
        n_comp: int | None = None,
        nprobe: int | None = None,
        training_compaction_min_mb: int | None = None,
        azure_emulator: bool = False,
        s3_endpoint: str | None = None,
        s3_region: str = "us-east-1",
        coalesce_gap_bytes: int = AZURE_COALESCE_GAP_BYTES,
        coalesce_range_bytes: int = AZURE_COALESCE_RANGE_BYTES,
        adaptive_read_strategy: bool = False,
        read_ranges_inflight: int = 0,
        read_ranges_wave_split_mb: int = 0,
        rank_degree: int = 0,
    ):
        self.binary = binary
        self.config = config
        self.server = server
        self.log_path = log_path
        self.local_disk_path = local_disk_path
        self.ivf_k = ivf_k
        self.n_comp = n_comp
        self.nprobe = nprobe
        self.training_compaction_min_mb = training_compaction_min_mb
        self.azure_emulator = azure_emulator
        self.s3_endpoint = s3_endpoint
        self.s3_region = s3_region
        if coalesce_gap_bytes < 0 or coalesce_range_bytes <= 0:
            raise RuntimeError("coalescing gap must be non-negative and range positive")
        self.coalesce_gap_bytes = coalesce_gap_bytes
        self.coalesce_range_bytes = coalesce_range_bytes
        self.adaptive_read_strategy = adaptive_read_strategy
        self.read_ranges_inflight = read_ranges_inflight
        self.read_ranges_wave_split_mb = read_ranges_wave_split_mb
        self.rank_degree = rank_degree
        self.process: subprocess.Popen | None = None
        self.log_file = None

    def start(self) -> None:
        environment = os.environ.copy()
        environment.update(
            {
                "PROXIMADB_COUNT_FS_IO": "1",
                "PROXIMADB_CACHE_PREFILL": "0",
                "PROXIMADB_CACHE_ON_WRITE": "all",
                "PROXIMADB_L0_COMPACTION_ENABLED": "1",
            }
        )
        # The diagnostic object-cold phase must remain cold even when the
        # invoking shell exports a persistent-cache config mirror.
        for inherited_gate in (
            "PROXIMADB_CACHE_LOCAL_DISK_PATH",
            "PROXIMADB_CACHE_LOCAL_DISK_MAX_GB",
            "PROXIMADB_CACHE_NVME_PATH",
            "PROXIMADB_CACHE_NVME_MAX_GB",
            "PROXIMADB_IVF_K",
            "PROXIMADB_IVF_NCOMP",
            "PROXIMADB_PAX_READ_COARSE_NPROBE",
            "PROXIMADB_TRAINING_COMPACTION_MIN_MB",
            "PROXIMADB_STORAGE_READ_STRATEGY_CHOOSER",
            "PROXIMADB_PAX_VECTOR_COALESCE_GAP",
            "PROXIMADB_PAX_VECTOR_COALESCE_RANGE",
            "PROXIMADB_PAX_COALESCE_GAP",
            "PROXIMADB_PAX_COALESCE_RANGE",
            # TD-RDSTRAT-12 §3: arm gates must come from the CLI flags only —
            # an invoking shell's exports would silently merge arms.
            "PROXIMADB_READ_RANGES_INFLIGHT",
            "PROXIMADB_READ_RANGES_WAVE_SPLIT_MB",
            "PROXIMADB_SEARCH_MORSEL_DEGREE",
            # S3 endpoint identity comes from --s3-endpoint/--s3-region only;
            # ambient AWS_* (SDK convention files, endpoint-url variants,
            # session tokens) and PROXIMADB_S3_* overrides would silently
            # re-point or re-auth the backend.
            "PROXIMADB_S3_ENDPOINT",
            "PROXIMADB_S3_REGION",
            "PROXIMADB_S3_FORCE_PATH_STYLE",
            "AWS_ENDPOINT",
            "AWS_ENDPOINT_URL",
            "AWS_ENDPOINT_URL_S3",
            "AWS_REGION",
            "AWS_DEFAULT_REGION",
            "AWS_ALLOW_HTTP",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
        ):
            environment.pop(inherited_gate, None)
        if self.s3_endpoint is not None:
            # The server's AwsS3FileSystem reads PROXIMADB_S3_ENDPOINT /
            # PROXIMADB_S3_FORCE_PATH_STYLE (mod.rs aws_s3_config_from_env);
            # MinIO requires path-style (no bucket DNS under the endpoint
            # host). AWS_ENDPOINT/creds additionally cover the object_store
            # parse_url_opts path, and plain HTTP needs the explicit allow
            # flag there.
            environment.update(
                {
                    "PROXIMADB_S3_ENDPOINT": self.s3_endpoint,
                    "PROXIMADB_S3_FORCE_PATH_STYLE": "1",
                    "AWS_ENDPOINT": self.s3_endpoint,
                    "AWS_REGION": self.s3_region,
                    "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
                    "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
                    **(
                        {"AWS_ALLOW_HTTP": "true"}
                        if self.s3_endpoint.startswith("http://")
                        else {}
                    ),
                }
            )
        if self.read_ranges_inflight > 0:
            environment["PROXIMADB_READ_RANGES_INFLIGHT"] = str(
                self.read_ranges_inflight
            )
        if self.read_ranges_wave_split_mb > 0:
            environment["PROXIMADB_READ_RANGES_WAVE_SPLIT_MB"] = str(
                self.read_ranges_wave_split_mb
            )
        if self.rank_degree > 0:
            environment["PROXIMADB_SEARCH_MORSEL_DEGREE"] = str(self.rank_degree)
        if self.adaptive_read_strategy:
            environment["PROXIMADB_STORAGE_READ_STRATEGY_CHOOSER"] = "1"
            print("[chooser] ON arm engaged: PROXIMADB_STORAGE_READ_STRATEGY_CHOOSER=1")
        else:
            # TD-IOBUDGET-2: an inherited gate must NOT leak into the OFF arm —
            # the 2026-08-31 A/B compared two un-engaged arms because the env
            # pop at construction was the only thing standing between an
            # operator's exported gate and the "OFF" baseline.
            if environment.get("PROXIMADB_STORAGE_READ_STRATEGY_CHOOSER") is not None:
                raise RuntimeError(
                    "chooser gate present in the OFF arm's server env; the "
                    "baseline would silently run engaged"
                )
            # Keep the planner fixed across local and Azure profiles so
            # backend choice does not silently change controlled sweeps.
            environment.update(
                {
                    "PROXIMADB_PAX_VECTOR_COALESCE_GAP": str(
                        self.coalesce_gap_bytes
                    ),
                    "PROXIMADB_PAX_VECTOR_COALESCE_RANGE": str(
                        self.coalesce_range_bytes
                    ),
                    "PROXIMADB_PAX_COALESCE_GAP": str(self.coalesce_gap_bytes),
                    "PROXIMADB_PAX_COALESCE_RANGE": str(self.coalesce_range_bytes),
                }
            )
        if self.ivf_k is not None:
            environment["PROXIMADB_IVF_K"] = str(self.ivf_k)
        if self.n_comp is not None:
            environment["PROXIMADB_IVF_NCOMP"] = str(self.n_comp)
        if self.nprobe is not None:
            environment["PROXIMADB_PAX_READ_COARSE_NPROBE"] = str(self.nprobe)
        if self.training_compaction_min_mb is not None:
            environment["PROXIMADB_TRAINING_COMPACTION_MIN_MB"] = str(
                self.training_compaction_min_mb
            )
        if self.azure_emulator:
            environment.update(
                {
                    "PROXIMADB_AZURE_EMULATOR": "1",
                    "AZURE_STORAGE_USE_EMULATOR": "true",
                    "AZURE_ALLOW_HTTP": "true",
                    "AZURE_STORAGE_ACCOUNT": "devstoreaccount1",
                    "AZURE_STORAGE_ACCOUNT_NAME": "devstoreaccount1",
                }
            )
        if self.local_disk_path is not None:
            environment["PROXIMADB_CACHE_LOCAL_DISK_PATH"] = str(self.local_disk_path)
            environment["PROXIMADB_CACHE_LOCAL_DISK_MAX_GB"] = "10"
        self.log_file = self.log_path.open("wb")
        self.process = subprocess.Popen(
            [str(self.binary), "-c", str(self.config)],
            cwd=self.log_path.parent,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"server exited with {self.process.returncode}; see {self.log_path}"
                )
            try:
                request_json(self.server + "/health/live", timeout=5)
                return
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                time.sleep(1)
        raise RuntimeError(f"server did not become live; see {self.log_path}")

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                # Graceful shutdown flushes unflushed data, so the time needed
                # scales with collection size — a 30M bed logged "Storage engine
                # stop timeout" with a collection still draining. SIGKILL during
                # that drain leaves a half-written bed that only fails later, at
                # geometry validation, after the ingest cost is already sunk.
                self.process.wait(timeout=SHUTDOWN_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.log_file is not None:
            self.log_file.close()
        self.process = None


class FlightUpsertStream:
    def __init__(
        self,
        host: str,
        port: int,
        collection_id: str,
        schema,
        flight_module=None,
    ):
        if flight_module is None:
            try:
                import pyarrow.flight as flight_module
            except ImportError as error:
                raise RuntimeError(
                    "Arrow Flight ingest requires PyArrow Flight; run the "
                    "harness with the repository Python environment"
                ) from error
        command = json.dumps(
            {
                "collection_id": collection_id,
                # The benchmark owns a fresh collection prefix and deterministic
                # unique IDs. UPSERT preserves that final state without paying
                # insert-only's required point lookup for every row.
                "operation": "upsert",
                "write_mode": "wal",
                "trigger_compaction": False,
            }
        ).encode()
        location = flight_module.Location.for_grpc_tcp(host, port)
        self.client = flight_module.FlightClient(location)
        descriptor = flight_module.FlightDescriptor.for_command(command)
        self.writer, self.reader = self.client.do_put(descriptor, schema)
        self.rows = 0
        self.closed = False

    def write_batch(self, batch) -> None:
        if self.closed:
            raise RuntimeError("cannot write to a closed Flight upsert stream")
        self.writer.write_batch(batch)
        self.rows += batch.num_rows

    def close(self) -> dict:
        if self.closed:
            raise RuntimeError("Flight upsert stream was already closed")
        self.closed = True
        try:
            self.writer.done_writing()
            payload = self.reader.read()
            encoded = payload.to_pybytes() if payload is not None else b""
            result = json.loads(encoded) if encoded else {}
        finally:
            self.writer.close()
            self.client.close()
        metrics = result.get("metrics", {})
        successful = metrics.get("successful_count")
        processed = metrics.get("total_processed")
        failed = metrics.get("failed_count")
        if (
            result.get("success") is not True
            or successful != self.rows
            or processed != self.rows
            or failed != 0
        ):
            raise RuntimeError(
                "Flight DoPut did not fully admit its stream: "
                f"rows={self.rows}, result={result}"
            )
        return result


def storage_action_via_flight(
    host: str,
    port: int,
    collection_id: str,
    action_type: str,
    expected_operation: str,
    flight_module=None,
) -> dict:
    if flight_module is None:
        try:
            import pyarrow.flight as flight_module
        except ImportError as error:
            raise RuntimeError(
                "--explicit-flush-every-rows requires PyArrow Flight; "
                "run the harness with the repository Python environment"
            ) from error

    location = flight_module.Location.for_grpc_tcp(host, port)
    client = flight_module.FlightClient(location)
    try:
        action = flight_module.Action(
            action_type,
            json.dumps({"collection_id": collection_id}).encode(),
        )
        responses = list(client.do_action(action))
    finally:
        client.close()
    if len(responses) != 1:
        raise RuntimeError(
            f"Flight {action_type} returned an unexpected response count: "
            f"{len(responses)}"
        )
    payload = json.loads(bytes(responses[0].body))
    if (
        payload.get("success") is not True
        or payload.get("operation") != expected_operation
    ):
        raise RuntimeError(
            f"Flight {action_type} did not succeed as expected: {payload}"
        )
    return payload


def force_flush_via_flight(
    host: str, port: int, collection_id: str, flight_module=None
) -> dict:
    return storage_action_via_flight(
        host,
        port,
        collection_id,
        "flush_collection",
        "flush",
        flight_module,
    )


def compact_via_flight(
    host: str, port: int, collection_id: str, flight_module=None
) -> dict:
    return storage_action_via_flight(
        host,
        port,
        collection_id,
        "compact_collection",
        "compact",
        flight_module,
    )


def wait_for_pax_epoch(
    server: str,
    collection_id: str,
    geometry: AzureCliPaxGeometry,
    before_inventory: dict,
    timeout_seconds: int = 300,
) -> tuple[float, dict, float | None]:
    started = time.monotonic()
    deadline = started + timeout_seconds
    before_signature = geometry.stable_signature(before_inventory)
    prior_signature = None
    stable_observations = 0
    while time.monotonic() < deadline:
        inventory = geometry.inventory()
        signature = geometry.stable_signature(inventory)
        if signature != before_signature:
            stable_observations = (
                stable_observations + 1 if signature == prior_signature else 1
            )
        else:
            stable_observations = 0
        prior_signature = signature
        wal_bytes = labelled_metric(
            scrape_text(server),
            "proximadb_wal_size_bytes",
            "collection",
            collection_id,
        )
        if stable_observations >= 2 and wal_is_quiescent(wal_bytes):
            return time.monotonic() - started, inventory, wal_bytes
        time.sleep(0.5)
    raise RuntimeError(
        "storage action did not publish a stable PAX epoch within "
        f"{timeout_seconds}s: collection={collection_id}"
    )


def post_flush_compaction_observation(
    explicit_flush_every_rows: int | None,
) -> dict | None:
    if explicit_flush_every_rows is None:
        return None
    return {
        "requested": False,
        "reason": (
            "automatic threshold compaction is observed by the stable "
            "materialization gate"
        ),
    }


def ingest(
    server: str,
    base_path: Path,
    base_format: str,
    expected_rows: int,
    batch_size: int,
    flight_host: str,
    flight_port: int,
    ingest_transport: str,
    explicit_flush_every_rows: int | None,
    explicit_geometry: AzureCliPaxGeometry | None,
) -> tuple[str, float, dict[int, int], list[dict], dict | None]:
    _, dimension, _ = vector_source_geometry(base_path, base_format)
    response = request_json(
        server + "/api/v2/collections",
        method="POST",
        body={
            "name": "sift1m_get_reduction",
            "dimension": dimension,
            "engine": "sst",
            "enable_proxima_record": True,
            "distance_metric": "euclidean",
        },
    )
    collection_id = str(response.get("collection_id") or response.get("id") or "")
    if not collection_id:
        raise RuntimeError(f"create response has no collection id: {response}")
    try:
        int(collection_id)
    except ValueError as error:
        raise RuntimeError(
            f"v2 create returned non-numeric catalog object id {collection_id!r}"
        ) from error
    started = time.perf_counter()
    inserted = 0
    retry_status_counts: dict[int, int] = {}
    explicit_flushes = []
    flight_stream = None
    if ingest_transport == "flight":
        batches = iter_vector_arrow_batches(
            base_path, base_format, expected_rows, batch_size
        )
    elif ingest_transport == "rest":
        batches = iter_vector_batches(base_path, base_format, expected_rows, batch_size)
    else:
        raise RuntimeError(f"unsupported ingest transport: {ingest_transport}")
    for batch in batches:
        if ingest_transport == "flight":
            batch_rows = batch.num_rows
            if flight_stream is None:
                flight_stream = FlightUpsertStream(
                    flight_host,
                    flight_port,
                    collection_id,
                    batch.schema,
                )
            flight_stream.write_batch(batch)
        else:
            batch_rows = len(batch)
            for attempt in range(8):
                try:
                    response = request_json(
                        f"{server}/api/v2/collections/{collection_id}/records/batch",
                        method="POST",
                        body={"records": batch},
                        timeout=300,
                    )
                    require_complete_insert(response, batch_rows)
                    break
                except urllib.error.HTTPError as error:
                    if error.code not in (429, 503) or attempt == 7:
                        raise
                    retry_status_counts[error.code] = (
                        retry_status_counts.get(error.code, 0) + 1
                    )
                    time.sleep(min(10, 0.25 * (2**attempt)))
        inserted += batch_rows
        if (
            explicit_flush_every_rows is not None
            and inserted % explicit_flush_every_rows == 0
        ):
            flight_ack = None
            if flight_stream is not None:
                flight_ack = flight_stream.close()
                flight_stream = None
            if explicit_geometry is None:
                raise RuntimeError(
                    "explicit flush epochs require Azure PAX inventory evidence"
                )
            inventory_before = explicit_geometry.inventory()
            wal_before = labelled_metric(
                scrape_text(server),
                "proximadb_wal_size_bytes",
                "collection",
                collection_id,
            )
            flush_started = time.perf_counter()
            response = force_flush_via_flight(
                flight_host,
                flight_port,
                collection_id,
            )
            action_seconds = time.perf_counter() - flush_started
            (
                publish_seconds,
                inventory_after,
                wal_after,
            ) = wait_for_pax_epoch(
                server,
                collection_id,
                explicit_geometry,
                inventory_before,
            )
            explicit_flushes.append(
                {
                    "after_rows": inserted,
                    "wal_bytes_before": wal_before,
                    "wal_bytes_after": wal_after,
                    "action_seconds": action_seconds,
                    "publish_seconds": publish_seconds,
                    "segments_before": inventory_before["segments"],
                    "segments_after": inventory_after["segments"],
                    "flight_ingest_ack": flight_ack,
                    "response": response,
                }
            )
            print(
                f"explicit flush: rows={inserted:,} "
                f"action={action_seconds:.3f}s "
                f"publish={publish_seconds:.3f}s "
                f"segments={inventory_after['segment_count']}",
                flush=True,
            )
        if inserted % 100_000 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"ingest: {inserted:,}/{expected_rows:,} "
                f"({inserted / elapsed:,.0f} vectors/s)",
                flush=True,
            )
    if flight_stream is not None:
        flight_stream.close()
    if inserted != expected_rows:
        raise RuntimeError(
            f"ingest iterator admitted {inserted} rows, expected {expected_rows}"
        )
    # Do not issue a second compaction after the final explicit flush. The
    # fifth L0 epoch arms automatic compaction at the configured threshold,
    # and wait_for_materialization below is the authoritative quiescence gate.
    # If automatic compaction already finished, the operator action is a valid
    # no-op and therefore cannot produce the "new PAX epoch" this harness used
    # to wait for. Requiring one caused a false 300-second timeout; racing the
    # automatic morsel also risked measuring redundant work rather than the
    # settled write path.
    explicit_compaction = post_flush_compaction_observation(explicit_flush_every_rows)
    elapsed = time.perf_counter() - started
    return (
        collection_id,
        elapsed,
        retry_status_counts,
        explicit_flushes,
        explicit_compaction,
    )


def require_empty_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"{path} is not empty; use a fresh benchmark root")
    path.mkdir(parents=True, exist_ok=True)


def require_groundtruth_scope(rows: int, groundtruth_scope: int) -> None:
    if groundtruth_scope != rows:
        raise RuntimeError(
            "ground-truth corpus mismatch: "
            f"--rows={rows}, groundtruth_scope={groundtruth_scope}; "
            "subset runs require an ivecs file generated against exactly the "
            "same corpus prefix"
        )


def gate_failures(
    phase: str,
    result: dict,
    max_gets: float | None,
    min_recall: float,
    max_p50_ms: float | None,
    require_local_hit: bool,
) -> list[str]:
    failures = []
    if max_gets is not None and result["gets_per_query"] > max_gets:
        failures.append(
            f"{phase}: GET/q {result['gets_per_query']:.2f} > {max_gets:.2f}"
        )
    if result["recall_at_k"] < min_recall:
        failures.append(
            f"{phase}: recall {result['recall_at_k']:.4f} < {min_recall:.4f}"
        )
    if max_p50_ms is not None and result["latency_ms"]["p50"] > max_p50_ms:
        failures.append(
            f"{phase}: p50 {result['latency_ms']['p50']:.2f}ms > {max_p50_ms:.2f}ms"
        )
    if require_local_hit and result["local_disk"]["hits"] <= 0:
        failures.append(f"{phase}: local-disk phase recorded zero local hits")
    return failures


def ivf_byte_attribution_failure(phase: str, result: dict) -> str | None:
    ivf = result["ivf"]
    if (
        ivf["cells_probed"] > 0
        and result["physical_gets"] > 0
        and ivf["region_a_bytes"] + ivf["region_b_bytes"] <= 0
    ):
        return (
            f"{phase}: IVF probe issued physical GETs but attributed zero "
            "Region-A/B bytes"
        )
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("target/release-server/proximadb-server"),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--sift-dir",
        type=Path,
        default=Path(os.environ.get("SIFT_DIR", "/Users/vijaysingh/sift1m")),
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        help="base vectors; defaults to <sift-dir>/sift_base.fvecs",
    )
    parser.add_argument(
        "--base-format",
        choices=("fvecs", "u8bin"),
        default="fvecs",
    )
    parser.add_argument(
        "--query-path",
        type=Path,
        help="query vectors; defaults to <sift-dir>/sift_query.fvecs",
    )
    parser.add_argument(
        "--query-format",
        choices=("fvecs", "u8bin"),
        default="fvecs",
    )
    parser.add_argument(
        "--groundtruth-path",
        type=Path,
        help=(
            "ivecs ground truth for exactly --rows corpus rows; defaults to "
            "<sift-dir>/sift_groundtruth.ivecs (the full base corpus)"
        ),
    )
    parser.add_argument(
        "--groundtruth-format",
        choices=("ivecs", "bigann-bin"),
        default="ivecs",
    )
    parser.add_argument(
        "--groundtruth-scope-rows",
        type=int,
        help=(
            "corpus cardinality used to generate --groundtruth-path; required "
            "with a custom path and required to equal --rows"
        ),
    )
    parser.add_argument("--port", type=int, default=5690)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=2_000)
    parser.add_argument(
        "--ingest-transport",
        choices=("flight", "rest"),
        default="flight",
        help=(
            "bulk data-plane transport; Flight is the canonical columnar "
            "default, while REST is retained as an explicit control"
        ),
    )
    parser.add_argument("--write-buffer-mb", type=int, default=4096)
    parser.add_argument(
        "--compaction-max-memory-mb",
        type=int,
        default=0,
        help=(
            "optional absolute ceiling for process-wide projected compaction "
            "memory; zero keeps cgroup/live-memory auto-sizing"
        ),
    )
    parser.add_argument(
        "--compaction-spill",
        action="store_true",
        help=(
            "enable deterministic application-managed local spill; defaults "
            "to a scratch directory below --root"
        ),
    )
    parser.add_argument(
        "--compaction-spill-directory",
        type=Path,
        help="explicit local/managed-disk scratch directory",
    )
    parser.add_argument(
        "--compaction-spill-working-memory-mb",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--compaction-spill-scratch-amplification-factor",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--compaction-spill-available-disk-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--compaction-spill-max-disk-mb",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--flush-vector-threshold",
        type=int,
        default=100_000,
        help=(
            "RETIRED no-op (#1526): the server-side vector_count_threshold knob "
            "was removed; kept only for checkpoint-identity stability"
        ),
    )
    parser.add_argument(
        "--flush-interval-secs",
        type=int,
        help=(
            "time-based WAL flush interval; defaults to 12 without explicit "
            "epochs and 3600 with --explicit-flush-every-rows so the timer "
            "cannot race the controlled admission plan"
        ),
    )
    parser.add_argument(
        "--flush-floor-predicted-mb",
        type=int,
        default=128,
        help=(
            "minimum predicted quantized PAX MiB before an inline size flush; "
            "the production default is 128. For a single-segment geometry bed, "
            "set this above the corpus prediction and use one explicit flush"
        ),
    )
    parser.add_argument(
        "--explicit-flush-every-rows",
        type=int,
        help=(
            "issue the supported Arrow Flight flush action after each exact "
            "row interval; intended for controlled small-corpus geometry"
        ),
    )
    parser.add_argument("--queries", type=int, default=1_000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--settle-timeout-secs", type=int, default=1_200)
    parser.add_argument(
        "--read-ranges-inflight",
        type=int,
        default=0,
        help="Arm PROXIMADB_READ_RANGES_INFLIGHT (bounded-concurrent wave width); 0 keeps the sequential default",
    )
    parser.add_argument(
        "--read-ranges-wave-split-mb",
        type=int,
        default=0,
        help="Arm PROXIMADB_READ_RANGES_WAVE_SPLIT_MB (opt-in Region-B mega-GET wave split); 0 keeps the single-GET default",
    )
    parser.add_argument(
        "--rank-degree",
        type=int,
        default=0,
        help="Set PROXIMADB_SEARCH_MORSEL_DEGREE (coarse-probe rank workers, TD-RDSTRAT-13); 0 keeps the server default (adaptive)",
    )
    parser.add_argument("--stable-secs", type=int, default=30)
    parser.add_argument("--max-segments", type=int, default=2)
    parser.add_argument(
        "--require-layout-version",
        type=int,
        default=1,
        help=(
            "expected on-disk segment layout version asserted against settled "
            "footers; the current writer emits 1 (post-collapse single layout)"
        ),
    )
    parser.add_argument("--post-write-max-gets", type=float, default=12.5)
    parser.add_argument("--local-warm-max-gets", type=float, default=12.5)
    parser.add_argument("--object-cold-max-gets", type=float, default=20.0)
    # TD-IOBUDGET-2 (product decision 2026-09-03): the recall floor drops
    # 0.98 -> 0.975 to make the economical default geometry (k=122 on file://,
    # bytes/q -51%, GETs/q -13% at recall 0.9794) shippable. Paired deltas
    # remain REPORTED as context; the gate is the absolute floor.
    parser.add_argument("--min-recall", type=float, default=0.975)
    parser.add_argument("--max-p50-ms", type=float, default=50.0)
    parser.add_argument(
        "--storage-url",
        help=(
            "durable segment base URL; defaults to a file:// directory under "
            "--root. az:// requires Azure CLI for footer geometry."
        ),
    )
    parser.add_argument(
        "--azurite",
        action="store_true",
        help=(
            "run the production Azure backend against local Azurite; requires "
            "--storage-url az://... and AZURE_STORAGE_CONNECTION_STRING for "
            "out-of-process geometry snapshots"
        ),
    )
    parser.add_argument(
        "--s3-endpoint",
        help=(
            "S3-shaped endpoint URL for --storage-url s3://... (e.g. "
            "http://minio-host:9000); forwarded to the server as AWS_ENDPOINT"
        ),
    )
    parser.add_argument(
        "--s3-region",
        default="us-east-1",
        help="S3 region sentinel forwarded to the server as AWS_REGION (default us-east-1)",
    )
    parser.add_argument(
        "--local-warm-max-p50-ms",
        type=float,
        help=(
            "optional local-tier latency gate; unset by default because the "
            "acceptance latency target is object-cold and local file:// "
            "latency is not an Azure WAN proxy"
        ),
    )
    parser.add_argument(
        "--binary-source-revision",
        help=(
            "Git revision used to build --binary; defaults to current HEAD. "
            "Use only when later commits touch docs, benchmark harnesses, or "
            "the focused Python contract for this harness."
        ),
    )
    parser.add_argument(
        "--ivf-k",
        type=int,
        help="force compaction-time coarse cell count (scale experiments only)",
    )
    parser.add_argument(
        "--n-comp",
        type=int,
        help=(
            "force compaction-time coarse-PCA projection width (width "
            "experiments only). Forced AND verified against the persisted A0 "
            "model, unlike a bare PROXIMADB_IVF_NCOMP env var, whose value the "
            "bed never recorded (TD-IVF-3)"
        ),
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        help="force query-time coarse cells (scale experiments only)",
    )
    parser.add_argument(
        "--adaptive-read-strategy",
        action="store_true",
        help=(
            "engage the read-strategy chooser in the spawned server "
            "(PROXIMADB_STORAGE_READ_STRATEGY_CHOOSER=1). TD-IOBUDGET-2: this "
            "flag is the ONLY wire from the CLI to the gate — the 2026-08-31 "
            "chooser A/B ran its ON arm un-engaged because nothing set it."
        ),
    )
    parser.add_argument(
        "--training-compaction-min-mb",
        type=int,
        help=(
            "override the production 32 MiB untrained-L0 floor; intended only "
            "to make sub-floor corpus points comparable in geometry experiments"
        ),
    )
    args = parser.parse_args()
    if args.write_buffer_mb <= 0:
        raise RuntimeError("--write-buffer-mb must be positive")
    if args.compaction_max_memory_mb < 0:
        raise RuntimeError("--compaction-max-memory-mb must be non-negative")
    if args.compaction_spill_directory is not None and not args.compaction_spill:
        raise RuntimeError("--compaction-spill-directory requires --compaction-spill")
    if args.compaction_spill_working_memory_mb <= 0:
        raise RuntimeError("--compaction-spill-working-memory-mb must be positive")
    if args.compaction_spill_scratch_amplification_factor < 1.0:
        raise RuntimeError(
            "--compaction-spill-scratch-amplification-factor must be at least 1"
        )
    if not 0 < args.compaction_spill_available_disk_fraction <= 1.0:
        raise RuntimeError(
            "--compaction-spill-available-disk-fraction must be in (0, 1]"
        )
    if args.compaction_spill_max_disk_mb < 0:
        raise RuntimeError("--compaction-spill-max-disk-mb must be non-negative")
    if args.flush_vector_threshold <= 0:
        raise RuntimeError("--flush-vector-threshold must be positive")
    if args.flush_floor_predicted_mb < 0:
        raise RuntimeError("--flush-floor-predicted-mb must be non-negative")
    if args.explicit_flush_every_rows is not None:
        if args.explicit_flush_every_rows <= 0:
            raise RuntimeError("--explicit-flush-every-rows must be positive")
        if args.explicit_flush_every_rows % args.batch_size:
            raise RuntimeError(
                "--explicit-flush-every-rows must be a multiple of --batch-size"
            )
        if args.rows % args.explicit_flush_every_rows:
            raise RuntimeError(
                "--rows must be a multiple of --explicit-flush-every-rows"
            )
    if args.ivf_k is not None and args.ivf_k <= 0:
        raise RuntimeError("--ivf-k must be positive")
    if args.n_comp is not None and args.n_comp <= 0:
        raise RuntimeError("--n-comp must be positive")
    if args.nprobe is not None and args.nprobe <= 0:
        raise RuntimeError("--nprobe must be positive")
    if (
        args.training_compaction_min_mb is not None
        and args.training_compaction_min_mb <= 0
    ):
        raise RuntimeError("--training-compaction-min-mb must be positive")
    flush_interval_secs = effective_flush_interval(
        args.explicit_flush_every_rows,
        args.flush_interval_secs,
    )
    if args.azurite and not (
        args.storage_url
        and urlparse(args.storage_url).scheme in {"adls", "az", "azure"}
    ):
        raise RuntimeError(
            "--azurite requires canonical --storage-url az://... "
            "(azure:// and adls:// aliases are accepted)"
        )
    if args.azurite and not os.environ.get("AZURE_STORAGE_CONNECTION_STRING"):
        raise RuntimeError(
            "--azurite requires AZURE_STORAGE_CONNECTION_STRING for geometry"
        )
    if (
        urlparse(args.storage_url or "").scheme == "s3"
    ) and args.s3_endpoint is None:
        raise RuntimeError(
            "--storage-url s3://... requires --s3-endpoint (e.g. http://minio-host:9000)"
        )
    if args.s3_endpoint is not None and not (
        os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")
    ):
        raise RuntimeError(
            "S3 storage requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
            "in the harness environment; they are forwarded to the server"
        )

    binary = args.binary.resolve()
    if not binary.is_file():
        raise RuntimeError(f"release binary not found: {binary}")
    binary_text = str(binary)
    if (
        "/target/release/" not in binary_text
        and "/target/release-server/" not in binary_text
    ):
        raise RuntimeError(
            "benchmark binary must come from target/release or target/release-server"
        )
    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tracked_status:
        raise RuntimeError(
            "benchmark refuses a dirty worktree; commit the exact "
            "source state before building and measuring"
        )
    binary_source_revision = args.binary_source_revision or git_revision
    subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            binary_source_revision,
            git_revision,
        ],
        check=True,
    )
    if binary_source_revision != git_revision:
        changed_since_build = subprocess.run(
            [
                "git",
                "diff",
                "--name-only",
                f"{binary_source_revision}..{git_revision}",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        # Python harness tests cannot alter the Rust release executable. Keep
        # the provenance boundary language-based instead of maintaining a
        # filename allowlist every time a new benchmark protocol lands.
        unsafe_changes = [
            path
            for path in changed_since_build
            if not path.startswith(("docs/", "scripts/", "tests/python/"))
        ]
        if unsafe_changes:
            raise RuntimeError(
                "binary source revision differs from executable source: "
                f"{unsafe_changes}; rebuild the release binary"
            )

    base_path = (
        args.base_path.resolve()
        if args.base_path is not None
        else args.sift_dir / "sift_base.fvecs"
    )
    query_path = (
        args.query_path.resolve()
        if args.query_path is not None
        else args.sift_dir / "sift_query.fvecs"
    )
    (
        base_count,
        base_dimension,
        base_declared_rows,
    ) = vector_source_geometry(base_path, args.base_format)
    groundtruth_path = (
        args.groundtruth_path.resolve()
        if args.groundtruth_path is not None
        else args.sift_dir / "sift_groundtruth.ivecs"
    )
    if args.groundtruth_path is not None and args.groundtruth_scope_rows is None:
        raise RuntimeError(
            "--groundtruth-scope-rows is required with --groundtruth-path"
        )
    groundtruth_scope = (
        args.groundtruth_scope_rows
        if args.groundtruth_scope_rows is not None
        else base_count
    )
    require_groundtruth_scope(args.rows, groundtruth_scope)
    (
        query_count,
        query_dimension,
        query_declared_rows,
    ) = vector_source_geometry(query_path, args.query_format)
    gt_count, truth_width = count_truth_records(
        groundtruth_path, args.groundtruth_format
    )
    if args.rows > base_count:
        raise RuntimeError(
            f"base corpus too small: rows={base_count} < requested {args.rows}"
        )
    # Dimension is validated for plausibility, not pinned to SIFT's 128: the
    # geometry beds now cover neural-embedding corpora (384/768/1024-d) where
    # k_c = rows*dim/iop_target and the coarse-PCA width behave very differently.
    # Base/query agreement is asserted separately below and is the check that
    # actually protects correctness.
    if not 2 <= base_dimension <= 4096:
        raise RuntimeError(f"implausible base dimension: {base_dimension}")
    measured_queries = args.queries * 3
    if measured_queries > query_count or measured_queries > gt_count:
        raise RuntimeError(
            "not enough SIFT queries/ground-truth rows for three disjoint "
            f"{args.queries}-query phases"
        )
    if query_dimension != base_dimension:
        raise RuntimeError("base/query dimensions differ")
    if args.top_k > truth_width:
        raise RuntimeError(
            f"top_k={args.top_k} exceeds ground-truth width {truth_width}"
        )

    root = args.root.resolve()
    require_empty_directory(root)
    spill_directory = (
        args.compaction_spill_directory.resolve()
        if args.compaction_spill_directory is not None
        else root / "compaction-scratch"
    )
    storage_url = args.storage_url or f"file://{root / 'data' / 'sst'}"
    storage_scheme = urlparse(storage_url).scheme
    if storage_scheme not in {"file", "adls", "az", "azure", "s3"}:
        raise RuntimeError(
            f"unsupported benchmark storage URL scheme: {storage_scheme}"
        )
    # Remote geometry reader, duck-typed across backends: inventory /
    # require_empty_prefix / stable_signature / materialize. Azure uses the
    # az CLI; S3-shaped endpoints use boto3. Both read out-of-process and
    # cannot touch the measured server's counters.
    remote_geometry = None
    if storage_scheme in {"adls", "az", "azure"}:
        remote_geometry = AzureCliPaxGeometry(storage_url, root / "azure-pax-snapshot")
    elif storage_scheme == "s3":
        remote_geometry = S3PaxGeometry(
            storage_url, root / "s3-pax-snapshot", args.s3_endpoint, args.s3_region
        )
    if remote_geometry is not None:
        remote_geometry.require_empty_prefix()
    # Epoch proof for --explicit-flush-every-rows: remote readers use object
    # identity; a local (file://) bed proves the same epochs from the PAX
    # footers via the LocalPaxGeometry adapter (same inventory/signature
    # contract), so the deterministic settle protocol is not remote-only.
    epoch_geometry = remote_geometry
    if args.explicit_flush_every_rows is not None and epoch_geometry is None:
        if storage_url.startswith("file://"):
            epoch_geometry = LocalPaxGeometry(root / "data" / "sst")
        else:
            raise RuntimeError(
                "--explicit-flush-every-rows requires Azure/S3 remote storage "
                "or a local file:// bed so each durable PAX epoch can be "
                "proven by object identity"
            )
    config = root / "benchmark.toml"
    write_config(
        config,
        root,
        args.port,
        args.write_buffer_mb,
        args.flush_vector_threshold,
        storage_url,
        flush_interval_secs,
        args.flush_floor_predicted_mb,
        args.compaction_max_memory_mb,
        args.compaction_spill,
        spill_directory if args.compaction_spill else None,
        args.compaction_spill_working_memory_mb,
        args.compaction_spill_scratch_amplification_factor,
        args.compaction_spill_available_disk_fraction,
        args.compaction_spill_max_disk_mb,
    )
    server_url = f"http://127.0.0.1:{args.port}"
    local_disk = root / "local-disk-cache"
    backend_profile = (
        "azure_blob_azurite"
        if args.azurite
        else (
            "s3_endpoint"
            if storage_scheme == "s3"
            else ("azure_blob" if remote_geometry is not None else "local_file")
        )
    )
    filesystem_note = (
        "Azure backend and HTTP request-path evidence against Azurite. "
        "GET counters are emitted by the measured ProximaDB process. "
        "Out-of-process Azure CLI inventory/footer reads are excluded from "
        "those counters but may warm the emulator host page cache; latency is "
        "local-emulator evidence, not production Azure WAN evidence."
        if args.azurite
        else (
            "GET count is physical-I/O-seam evidence. Latency is local "
            "filesystem evidence, not Azure WAN evidence."
            if remote_geometry is None
            else (
                "S3-shaped backend (e.g. MinIO) over HTTP; GET counters are "
                "emitted by the measured ProximaDB process. Out-of-process "
                "boto3 inventory/footer reads are excluded from those "
                "counters but may warm the endpoint's host page cache; "
                "latency is endpoint-network evidence, not production-cloud "
                "evidence."
                if storage_scheme == "s3"
                else (
                    "Azure backend evidence. Out-of-process Azure CLI "
                    "inventory/footer reads are excluded from the measured "
                    "server counters."
                )
            )
        )
    )
    result = {
        "protocol": "pax_get_reduction",
        "git_revision": git_revision,
        "binary": {
            "path": str(binary),
            "sha256": sha256(binary),
            "bytes": binary.stat().st_size,
            "source_revision": binary_source_revision,
            "profile": (
                "release-server"
                if "/target/release-server/" in binary_text
                else "release"
            ),
        },
        "harness": {
            "git_revision": git_revision,
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "dataset": {
            "base": str(base_path),
            "base_format": args.base_format,
            "available_rows": base_count,
            "base_declared_rows": base_declared_rows,
            "measured_rows": args.rows,
            "dimension": base_dimension,
            "queries_path": str(query_path),
            "query_format": args.query_format,
            "query_available_rows": query_count,
            "query_declared_rows": query_declared_rows,
            "query_count": args.queries,
            "phase_query_ranges": {
                "post_write": [0, args.queries],
                "local_disk_warm": [args.queries, args.queries * 2],
                "object_cold": [args.queries * 2, args.queries * 3],
            },
            "groundtruth": str(groundtruth_path),
            "groundtruth_format": args.groundtruth_format,
            "groundtruth_scope": groundtruth_scope,
            "groundtruth_width": truth_width,
        },
        "filesystem_profile": {
            "segment_backend": backend_profile,
            "storage_url": storage_url,
            "azurite": args.azurite,
            "s3_endpoint": args.s3_endpoint,
            "geometry_evidence": (
                (
                    "s3_boto3_inventory_and_footer_snapshot"
                    if storage_scheme == "s3"
                    else "azure_cli_inventory_and_footer_snapshot"
                )
                if remote_geometry is not None
                else "local_pax_footer"
            ),
            "local_disk_path": str(local_disk),
            "range_planner": {
                "profile": "azure",
                "max_gap_bytes": AZURE_COALESCE_GAP_BYTES,
                "max_range_bytes": AZURE_COALESCE_RANGE_BYTES,
            },
            "note": filesystem_note,
        },
        "probe_policy": {
            "ivf_k_override": args.ivf_k,
            "n_comp_override": args.n_comp,
            "nprobe_override": args.nprobe,
            "training_compaction_min_mb_override": (
                args.training_compaction_min_mb
            ),
            "required_layout_version": args.require_layout_version,
        },
        "ingest_config": {
            "transport": args.ingest_transport,
            "flight_operation": (
                "upsert" if args.ingest_transport == "flight" else None
            ),
            "batch_size": args.batch_size,
            "write_buffer_mb": args.write_buffer_mb,
            "flush_vector_threshold": args.flush_vector_threshold,
            "flush_interval_secs": flush_interval_secs,
            "flush_floor_predicted_mb": args.flush_floor_predicted_mb,
            "explicit_flush_every_rows": args.explicit_flush_every_rows,
            "explicit_flush_flight_port": (
                args.port if args.explicit_flush_every_rows is not None else None
            ),
        },
        "query_config": {
            "transport": "rest_v2",
            "reason": (
                "REST v2 remains the fixed cross-scale reference transport; "
                "canonical Flight v2 parity and transport overhead are measured "
                "separately by flight_vs_rest_v2_bed.py"
            ),
        },
        "compute_profile": compute_profile(),
        "compaction_memory_policy": {
            "memory_amplification_factor": 12.0,
            "memory_budget_fraction": 0.25,
            "available_memory_fraction": 0.5,
            "max_memory_mb": args.compaction_max_memory_mb,
            "spill_enabled": args.compaction_spill,
            "spill_directory": (
                str(spill_directory) if args.compaction_spill else None
            ),
            "spill_working_memory_mb": (args.compaction_spill_working_memory_mb),
            "spill_scratch_amplification_factor": (
                args.compaction_spill_scratch_amplification_factor
            ),
            "spill_available_disk_fraction": (
                args.compaction_spill_available_disk_fraction
            ),
            "spill_max_disk_mb": args.compaction_spill_max_disk_mb,
        },
        "thresholds": {
            "post_write_max_gets_per_query": args.post_write_max_gets,
            "local_disk_warm_max_gets_per_query": args.local_warm_max_gets,
            "object_cold_max_gets_per_query": args.object_cold_max_gets,
            "min_recall_at_k": args.min_recall,
            "post_write_max_p50_ms": args.max_p50_ms,
            "local_disk_warm_max_p50_ms": args.local_warm_max_p50_ms,
            "object_cold_max_p50_ms": args.max_p50_ms,
            "max_segments": args.max_segments,
        },
        "phases": {},
    }

    active: OwnedServer | None = None
    resource_sampler: ProcessScratchSampler | None = None
    failures: list[str] = []
    try:
        active = OwnedServer(
            binary=binary,
            config=config,
            server=server_url,
            log_path=root / "server-ingest.log",
            local_disk_path=local_disk,
            ivf_k=args.ivf_k,
            n_comp=args.n_comp,
            nprobe=args.nprobe,
            training_compaction_min_mb=args.training_compaction_min_mb,
            azure_emulator=args.azurite,
            s3_endpoint=args.s3_endpoint,
            s3_region=args.s3_region,
            rank_degree=args.rank_degree,
            read_ranges_inflight=args.read_ranges_inflight,
            read_ranges_wave_split_mb=args.read_ranges_wave_split_mb,
            adaptive_read_strategy=args.adaptive_read_strategy,
        )
        active.start()
        if args.compaction_spill:
            if active.process is None:
                raise RuntimeError("spill resource sampling requires a live server")
            resource_sampler = ProcessScratchSampler(
                active.process.pid,
                spill_directory,
            )
            resource_sampler.start()
        (
            collection_id,
            ingest_seconds,
            retry_status_counts,
            explicit_flushes,
            explicit_compaction,
        ) = ingest(
            server_url,
            base_path,
            args.base_format,
            args.rows,
            args.batch_size,
            "127.0.0.1",
            args.port,
            args.ingest_transport,
            args.explicit_flush_every_rows,
            epoch_geometry,
        )
        result["collection_id"] = collection_id
        result["ingest"] = {
            "seconds": ingest_seconds,
            "vectors_per_second": args.rows / ingest_seconds,
            "retry_status_counts": retry_status_counts,
            "explicit_flushes": explicit_flushes,
            "explicit_compaction": explicit_compaction,
        }
        geometry = wait_for_materialization(
            root / "data" / "sst",
            server_url,
            collection_id,
            args.rows,
            args.max_segments,
            args.settle_timeout_secs,
            args.stable_secs,
            remote_geometry,
            required_layout_version=args.require_layout_version,
        )
        wrong_layouts = [
            segment
            for segment in geometry["segments"]
            if segment["layout_version"] != args.require_layout_version
        ]
        if wrong_layouts:
            raise RuntimeError(
                "settled segment layout mismatch: "
                f"required v{args.require_layout_version}, got {wrong_layouts}"
            )
        untrained = [
            segment["path"]
            for segment in geometry["segments"]
            if not segment.get("coarse_cells")
        ]
        if untrained:
            raise RuntimeError(
                "settled segment(s) lack a coarse A0 directory "
                f"(untrained flush artifact?): {untrained}"
            )
        if args.ivf_k is not None:
            wrong_cells = [
                segment
                for segment in geometry["segments"]
                if segment.get("coarse_cells") != args.ivf_k
            ]
            if wrong_cells:
                raise RuntimeError(
                    f"forced ivf_k={args.ivf_k} was not persisted: {wrong_cells}"
                )
        if args.n_comp is not None:
            wrong_width = [
                segment
                for segment in geometry["segments"]
                if segment.get("coarse_components") != args.n_comp
            ]
            if wrong_width:
                raise RuntimeError(
                    f"forced n_comp={args.n_comp} was not persisted: {wrong_width}"
                )
        result["settled_geometry"] = geometry
        if resource_sampler is not None:
            resource_sampler.stop()
            result["compaction_resource_observation"] = resource_sampler.snapshot()
            resource_sampler = None
        materialized_metrics = scrape(server_url)
        result["compaction_metrics_after_materialization"] = {
            name: materialized_metrics[name]
            for name in (
                "proximadb_compactions_total",
                "proximadb_compaction_bytes_read_total",
                "proximadb_compaction_bytes_written_total",
                "proximadb_compaction_memory_reserved_bytes",
                "proximadb_compaction_scratch_reserved_bytes",
                "proximadb_compaction_spill_total",
            )
        }
        post_write = run_query_sweep(
            server_url,
            collection_id,
            query_path,
            groundtruth_path,
            0,
            args.queries,
            args.top_k,
            "post_write",
            args.query_format,
            args.groundtruth_format,
        )
        result["phases"]["post_write"] = post_write
        failures.extend(
            gate_failures(
                "post_write",
                post_write,
                args.post_write_max_gets,
                args.min_recall,
                args.max_p50_ms,
                require_local_hit=False,
            )
        )
        active.stop()

        active = OwnedServer(
            binary=binary,
            config=config,
            server=server_url,
            log_path=root / "server-local-disk-warm.log",
            local_disk_path=local_disk,
            ivf_k=args.ivf_k,
            nprobe=args.nprobe,
            training_compaction_min_mb=args.training_compaction_min_mb,
            azure_emulator=args.azurite,
            s3_endpoint=args.s3_endpoint,
            s3_region=args.s3_region,
            rank_degree=args.rank_degree,
            read_ranges_inflight=args.read_ranges_inflight,
            read_ranges_wave_split_mb=args.read_ranges_wave_split_mb,
            adaptive_read_strategy=args.adaptive_read_strategy,
        )
        active.start()
        local_warm = run_query_sweep(
            server_url,
            collection_id,
            query_path,
            groundtruth_path,
            args.queries,
            args.queries,
            args.top_k,
            "local_disk_warm",
            args.query_format,
            args.groundtruth_format,
        )
        result["phases"]["local_disk_warm"] = local_warm
        failures.extend(
            gate_failures(
                "local_disk_warm",
                local_warm,
                args.local_warm_max_gets,
                args.min_recall,
                args.local_warm_max_p50_ms,
                require_local_hit=True,
            )
        )
        active.stop()

        active = OwnedServer(
            binary=binary,
            config=config,
            server=server_url,
            log_path=root / "server-object-cold.log",
            local_disk_path=None,
            ivf_k=args.ivf_k,
            nprobe=args.nprobe,
            training_compaction_min_mb=args.training_compaction_min_mb,
            azure_emulator=args.azurite,
            s3_endpoint=args.s3_endpoint,
            s3_region=args.s3_region,
            rank_degree=args.rank_degree,
            read_ranges_inflight=args.read_ranges_inflight,
            read_ranges_wave_split_mb=args.read_ranges_wave_split_mb,
            adaptive_read_strategy=args.adaptive_read_strategy,
        )
        active.start()
        object_cold = run_query_sweep(
            server_url,
            collection_id,
            query_path,
            groundtruth_path,
            args.queries * 2,
            args.queries,
            args.top_k,
            "object_cold",
            args.query_format,
            args.groundtruth_format,
        )
        result["phases"]["object_cold"] = object_cold
        failures.extend(
            gate_failures(
                "object_cold",
                object_cold,
                args.object_cold_max_gets,
                args.min_recall,
                args.max_p50_ms,
                require_local_hit=False,
            )
        )
        if attribution_failure := ivf_byte_attribution_failure(
            "object_cold", object_cold
        ):
            failures.append(attribution_failure)
        if failures:
            result["gate_failures"] = failures
            raise RuntimeError("; ".join(failures))
        result["status"] = "pass"
    except Exception as error:
        result["status"] = "fail"
        result["error"] = str(error)
        raise
    finally:
        if resource_sampler is not None:
            try:
                resource_sampler.stop()
                result["compaction_resource_observation"] = resource_sampler.snapshot()
            except Exception as sampler_error:
                result["resource_sampler_error"] = str(sampler_error)
        if active is not None:
            active.stop()
        output = root / "result.json"
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"result: {output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
