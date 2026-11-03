#!/usr/bin/env python3
"""TD-TIER-1 SC6 (2b): multi-tenant soak-bed harness.

Runs the TIER1_OPT_IN_ROLLOUT_RUNBOOK Phase-2 exit gates as phases against a
single spawned server, and pins per-tenant/per-phase gauges into result.json:

* phase ``ready_pin``       — every PERFORMANCE collection reaches READY(g)
  (``storage_class_status``) and stays, with its grant charged in the ONE
  shared pool;
* phase ``recall_parity_warm`` — recall spot-check per granted collection plus
  a cross-class parity pair (the same data slice served granted vs balanced);
* phase ``concurrent_shared_pool`` — concurrent ingest+query across tenants
  while sampling the pool ledger: exit gate "no capacity overcommit under
  concurrency (SC-TE3)", including an over-admission refusal drill;
* phase ``idle_reclaim``    — one tenant idles past ``idle_reclaim_secs``:
  exit gate "idle reservation accrual matches the ledger" (the accrual tail
  closes at the reclaim) and re-admission is NOT automatic;
* phase ``drop_release``    — exit gate "drops/updates release cleanly": a
  balanced drop never touches the ledger, a granted drop releases exactly its
  effective bytes, and a recreate re-admits through the normal path. A REST
  storage-class UPDATE surface does not exist today (manager-internal only),
  so the "update" arm runs as drop+recreate per the runbook's own "(or
  recreate)" wording.

Bed shape (fixed, sized by knobs): three tenants — ``sc6-perf-a`` owns
``perf-a`` (performance) and ``bal-a`` (balanced, same data slice: the parity
pair), ``sc6-perf-b`` owns ``perf-b`` (performance, the idle-reclaim subject),
``sc6-bal`` owns ``bal-c`` (balanced). Both PERFORMANCE grants charge ONE
reservation pool, so capacity contention is real.

Evidence discipline (SC6 plan): per-query GETs/bytes come from the response
``search_plan_trace`` (request ``debug: true``), cross-checked against the
same Prometheus counters the sift1m harness gates on. No retracted GETs/q
figures are quoted anywhere. Latency is reported, never gated — both arms
miss the 50 ms cold-p50 gate identically (inherited open caveat).

Usage (SIFT): multi_tenant_bed.py --root <bed> --binary <server> \
    --sift-dir <sift1m>
Usage (no SIFT, smoke): multi_tenant_bed.py --root <bed> --synthetic
Plan only:             multi_tenant_bed.py --root <bed> --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROTOCOL = "sc6_multi_tenant_soak"
POOL_ID = "sc6-shared"
PORT_DEFAULT = 5995  # 599x range; the 589x/1589x range belongs to another lane
INTERNAL_MUX_OFFSET = 10_001
SHUTDOWN_GRACE_SECONDS = 120
HEALTH_TIMEOUT_SECONDS = 180
API_TIMEOUT_SECONDS = 300  # warm phases stack object-store round trips
# Harness-side pacing cap, NOT a server limit: the REST default allows 64MB
# bodies (rest/server.rs max_request_size_mb). Keeping batches ~1.8MB paces
# ingest at a steady, retryable granularity instead of shipping a few giant
# bodies.
BODY_SAFETY_BYTES = 1_800_000
INGEST_RETRIES = 8
POLL_SECONDS = 2.0
LEDGER_NAME = "reservations.json"
METERING_INTERVAL_ENV = "PROXIMADB_METERING_FLUSH_INTERVAL_SECS"
# Same counter names the sift1m harness gates on (cross-check source).
PROM_COUNTERS = (
    "proximadb_object_store_gets_total",
    "proximadb_object_store_bytes_read_total",
    "proximadb_object_store_requests_total",
    "proximadb_pax_resident_hits_total",
)

# The bed plan. Slice labels map each collection to a corpus row range; ids
# stay the global ``v<row>`` form so sliced ground truth maps without renaming.
# perf-a and bal-a share slice A (the cross-class parity pair).
BED = (
    ("sc6-perf-a", "perf-a", "performance", "A"),
    ("sc6-perf-a", "bal-a", "balanced", "A"),
    ("sc6-perf-b", "perf-b", "performance", "B"),
    ("sc6-bal", "bal-c", "balanced", "C"),
)
PERF_TENANT_IDLE_SUBJECT = "sc6-perf-b"  # the tenant that goes idle in phase 4


class BedError(RuntimeError):
    """Gate or harness failure that fails the run with context."""


# ── HTTP helpers ────────────────────────────────────────────────────────────


def request_json(
    url: str,
    method: str = "GET",
    body: object | None = None,
    timeout: int = API_TIMEOUT_SECONDS,
    tenant: str | None = None,
) -> dict:
    data = None if body is None else json.dumps(body).encode()
    headers = {} if data is None else {"Content-Type": "application/json"}
    if tenant is not None:
        headers["X-Tenant-ID"] = tenant
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    return json.loads(payload) if payload else {}


def request_json_status(
    url: str,
    method: str = "GET",
    body: object | None = None,
    timeout: int = API_TIMEOUT_SECONDS,
    tenant: str | None = None,
) -> tuple[int, str, dict | None]:
    """One attempt returning (status, text, parsed-json-or-None). HTTPError is
    NOT raised: the admission-refusal drill must inspect the refusal body."""
    data = None if body is None else json.dumps(body).encode()
    headers = {} if data is None else {"Content-Type": "application/json"}
    if tenant is not None:
        headers["X-Tenant-ID"] = tenant
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode(errors="replace")
            status = response.status
    except urllib.error.HTTPError as error:
        text = error.read().decode(errors="replace")
        status = error.code
    except (TimeoutError, OSError) as error:
        return 0, f"transport error: {error!r}", None
    try:
        return status, text, (json.loads(text) if text else None)
    except json.JSONDecodeError:
        return status, text, None


def post_batch_with_retry(
    server: str, collection_id: str, records: list[dict], tenant: str
) -> dict[int, int]:
    """Ingest one batch under the known 429/503 backoff contract. Returns the
    per-status retry counts (recorded as ingest evidence)."""
    retry_counts: dict[int, int] = {}
    for attempt in range(INGEST_RETRIES):
        try:
            response = request_json(
                f"{server}/api/v2/collections/{collection_id}/records/batch",
                method="POST",
                body={"records": records},
                tenant=tenant,
            )
            inserted = response.get("inserted_count")
            failed = response.get("failed_count")
            if inserted != len(records) or failed != 0:
                raise BedError(
                    "batch insert was not fully admitted: "
                    f"expected={len(records)}, inserted_count={inserted!r}, "
                    f"failed_count={failed!r}, errors={response.get('errors')!r}"
                )
            return retry_counts
        except urllib.error.HTTPError as error:
            if error.code not in (429, 503) or attempt == INGEST_RETRIES - 1:
                raise
            retry_counts[error.code] = retry_counts.get(error.code, 0) + 1
            time.sleep(min(10.0, 0.25 * (2**attempt)))
        except (TimeoutError, OSError) as error:
            if attempt == INGEST_RETRIES - 1:
                raise BedError(
                    f"batch insert failed after retries: {error!r}"
                ) from error
            time.sleep(min(10.0, 0.25 * (2**attempt)))
    raise BedError("batch insert retry loop exhausted without a verdict")


def scrape(server: str) -> dict[str, float]:
    """Sum every sample of each counter (process-global; cross-check only —
    per-tenant attribution comes from the per-query search_plan_trace)."""
    with urllib.request.urlopen(server + "/metrics/prometheus", timeout=30) as response:
        text = response.read().decode()
    totals: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or " " not in line:
            continue
        name, _, raw_value = line.partition(" ")
        metric = name.split("{")[0]
        if metric not in PROM_COUNTERS:
            continue
        try:
            totals[metric] = totals.get(metric, 0.0) + float(raw_value.strip())
        except ValueError:
            continue
    return totals


def metric_delta(before: dict[str, float], after: dict[str, float], name: str) -> float:
    return after.get(name, 0.0) - before.get(name, 0.0)


def labelled_wal_bytes(server: str, collection_id: str) -> float | None:
    """The collection's unflushed-WAL gauge, or None when not exposed."""
    with urllib.request.urlopen(server + "/metrics/prometheus", timeout=30) as response:
        text = response.read().decode()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("proximadb_wal_size_bytes{") and (
            f'collection="{collection_id}"' in line
        ):
            try:
                return float(line.rpartition(" ")[2].strip())
            except ValueError:
                return None
    return None


# ── Ledger (the durable pool state under <pool path>/reservations.json) ─────


def effective_bytes(grant_bytes: int, overhead_factor: float) -> int:
    """Mirror of ``ReservationPoolConfig::effective_bytes``: the per-grant
    capacity charge is grant bytes plus ceil(grant x overhead)."""
    return grant_bytes + math.ceil(grant_bytes * overhead_factor)


def read_ledger(pool_dir: Path) -> dict:
    """Read the pool ledger. The server persists via fsync+rename, so a read
    sees either the old or the new file — retry on a torn/absent file."""
    path = pool_dir / LEDGER_NAME
    for attempt in range(3):
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as error:
            if attempt == 2:
                raise BedError(f"ledger unreadable at {path}: {error!r}") from error
            time.sleep(0.2)
    raise BedError("unreachable")


def grant_state_label(state: object) -> str:
    """GrantState is externally tagged: plain variants are strings, Degraded is
    ``{"Degraded": {"reason": ...}}``."""
    if isinstance(state, str):
        return state
    if isinstance(state, dict) and state:
        return next(iter(state))
    return "Unknown"


def charging_grants(ledger: dict) -> list[dict]:
    return [
        grant
        for grant in ledger.get("grants", [])
        if grant_state_label(grant.get("state")) != "Released"
    ]


def charged_effective(ledger: dict, overhead_factor: float) -> int:
    """Mirror of ``ReservationAuthority::charged_effective``: the ledger stores
    RAW grant bytes; the effective charge applies the overhead at sum time."""
    return sum(
        effective_bytes(int(grant.get("bytes", 0)), overhead_factor)
        for grant in charging_grants(ledger)
    )


def live_grant_for(ledger: dict, tenant: str, collection_id: str) -> dict | None:
    for grant in charging_grants(ledger):
        if (
            grant.get("tenant_id") == tenant
            and grant.get("collection_id") == collection_id
        ):
            return grant
    return None


def any_grant_for(ledger: dict, collection_id: str) -> dict | None:
    for grant in ledger.get("grants", []):
        if grant.get("collection_id") == collection_id:
            return grant
    return None


def ledger_view(ledger: dict, overhead_factor: float, note: str) -> dict:
    return {
        "note": note,
        "charged_effective": charged_effective(ledger, overhead_factor),
        "charging": [
            {
                "grant_id": grant.get("id"),
                "tenant_id": grant.get("tenant_id"),
                "collection_id": grant.get("collection_id"),
                "bytes": grant.get("bytes"),
                "state": grant_state_label(grant.get("state")),
            }
            for grant in charging_grants(ledger)
        ],
    }


# ── Vector corpus ───────────────────────────────────────────────────────────


def count_fixed_records(path: Path, scalar_bytes: int) -> tuple[int, int]:
    with path.open("rb") as source:
        header = source.read(4)
    if len(header) != 4:
        raise BedError(f"{path}: missing vector header")
    dimension = struct.unpack("<i", header)[0]
    if dimension <= 0:
        raise BedError(f"{path}: invalid dimension {dimension}")
    record_bytes = 4 + dimension * scalar_bytes
    size = path.stat().st_size
    if size % record_bytes:
        raise BedError(f"{path}: {size} bytes is not a multiple of {record_bytes}")
    return size // record_bytes, dimension


def read_fvecs(path: Path, start: int, count: int) -> list[list[float]]:
    total, dimension = count_fixed_records(path, 4)
    if start < 0 or count <= 0 or start + count > total:
        raise BedError(f"{path}: requested [{start}, {start + count}) of {total}")
    record_bytes = 4 + 4 * dimension
    vectors: list[list[float]] = []
    with path.open("rb") as source:
        source.seek(start * record_bytes)
        for _ in range(count):
            record = source.read(record_bytes)
            encoded = struct.unpack_from("<i", record, 0)[0]
            if encoded != dimension:
                raise BedError(f"{path}: variable dimension encountered")
            vectors.append(list(struct.unpack_from(f"<{dimension}f", record, 4)))
    return vectors


def read_ivecs(path: Path, start: int, count: int) -> list[list[int]]:
    total, width = count_fixed_records(path, 4)
    if start < 0 or count <= 0 or start + count > total:
        raise BedError(f"{path}: requested [{start}, {start + count}) of {total}")
    record_bytes = 4 + 4 * width
    rows: list[list[int]] = []
    with path.open("rb") as source:
        source.seek(start * record_bytes)
        for _ in range(count):
            record = source.read(record_bytes)
            encoded = struct.unpack_from("<i", record, 0)[0]
            if encoded != width:
                raise BedError(f"{path}: variable width encountered")
            rows.append(list(struct.unpack_from(f"<{width}i", record, 4)))
    return rows


def synthetic_vectors(seed: int, rows: int, dimension: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(dimension)] for _ in range(rows)]


def brute_force_truth(
    corpus: list[list[float]], queries: list[list[float]], top_k: int
) -> list[list[int]]:
    truth: list[list[int]] = []
    for query in queries:
        scored = sorted(
            range(len(corpus)),
            key=lambda row: sum((a - b) * (a - b) for a, b in zip(query, corpus[row])),
        )
        truth.append(scored[:top_k])
    return truth


def resolve_dataset(
    args: argparse.Namespace, rows_per_tenant: int, delta_rows: int, dry_run: bool
) -> dict:
    """Load (or synthesize) the shared corpus and the per-slice expected truth.

    The corpus spans 3 slices plus the perf-b delta rows; ids are global
    ``v<row>`` so the SIFT ground truth maps without renaming. Truth is the
    global ground truth restricted to the slice (SIFT) or in-slice brute force
    (synthetic) — both normalized to per-query id SETS.
    """
    slices = {
        "A": (0, rows_per_tenant),
        "B": (rows_per_tenant, rows_per_tenant),
        "C": (2 * rows_per_tenant, rows_per_tenant),
    }
    corpus_rows = 3 * rows_per_tenant + delta_rows

    if args.synthetic and dry_run:
        return {
            "mode": "synthetic",
            "dimension": args.synthetic_dimension,
            "corpus": [],
            "queries": [],
            "slices": slices,
            "truth": {},
            "skipped_no_truth": {},
            "note": "dry-run: synthetic corpus not generated",
        }

    if args.synthetic:
        dimension = args.synthetic_dimension
        corpus = synthetic_vectors(args.synthetic_seed, corpus_rows, dimension)
        queries = synthetic_vectors(args.synthetic_seed + 1, args.queries, dimension)
        truth: dict[str, list[set[str]]] = {}
        skipped: dict[str, int] = {}
        for label, (start, count) in slices.items():
            truth[label] = [
                {f"v{start + row}" for row in rows}
                for rows in brute_force_truth(
                    corpus[start : start + count], queries, args.top_k
                )
            ]
            skipped[label] = 0
        return {
            "mode": "synthetic",
            "dimension": dimension,
            "corpus": corpus,
            "queries": queries,
            "slices": slices,
            "truth": truth,
            "skipped_no_truth": skipped,
            "note": "synthetic corpus; truth is in-slice brute force",
        }

    sift_dir = args.sift_dir or Path(os.environ.get("SIFT_DIR", ""))
    base_path = args.base_path or (sift_dir / "sift_base.fvecs" if sift_dir else None)
    query_path = args.query_path or (
        sift_dir / "sift_query.fvecs" if sift_dir else None
    )
    truth_path = args.groundtruth_path or (
        sift_dir / "sift_groundtruth.ivecs" if sift_dir else None
    )
    if dry_run and (base_path is None or query_path is None or truth_path is None):
        return {
            "mode": "sift",
            "dimension": None,
            "corpus": [],
            "queries": [],
            "slices": slices,
            "truth": {},
            "skipped_no_truth": {},
            "note": "dry-run: dataset unresolved (SIFT paths absent)",
        }
    if base_path is None or query_path is None or truth_path is None:
        raise BedError(
            "SIFT mode needs --sift-dir or --base-path/--query-path/--groundtruth-path"
        )
    total_base, dimension = count_fixed_records(base_path, 4)
    total_queries, _ = count_fixed_records(query_path, 4)
    if corpus_rows > total_base:
        raise BedError(
            f"corpus too small: {corpus_rows} rows needed (3 slices + delta), "
            f"{total_base} available"
        )
    corpus = read_fvecs(base_path, 0, corpus_rows)
    queries = read_fvecs(query_path, 0, min(args.queries, total_queries))
    groundtruth = read_ivecs(truth_path, 0, len(queries))
    truth = {}
    skipped = {}
    for label, (start, count) in slices.items():
        allowed = {f"v{row}" for row in range(start, start + count)}
        per_query: list[set[str]] = []
        skip = 0
        for row in groundtruth:
            in_slice = [
                f"v{value}" for value in row[: args.top_k * 4] if f"v{value}" in allowed
            ][: args.top_k]
            if in_slice:
                per_query.append(set(in_slice))
            else:
                skip += 1
        truth[label] = per_query
        skipped[label] = skip
    return {
        "mode": "sift",
        "dimension": dimension,
        "corpus": corpus,
        "queries": queries,
        "slices": slices,
        "truth": truth,
        "skipped_no_truth": skipped,
        "note": "SIFT corpus; truth is the global ground truth filtered to the slice",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(sorted_values: list[float], fraction: float) -> float:
    # Same rank convention as the sift1m harness (ceil(q*n)-1) so p50/p99
    # compare across the two beds.
    if not sorted_values:
        return 0.0
    rank = max(0, math.ceil(fraction * len(sorted_values)) - 1)
    return sorted_values[rank]


# ── Server lifecycle ────────────────────────────────────────────────────────


def wait_for_port_free(port: int, timeout: float = 120.0) -> None:
    """Every port the server will bind must be free: the unified port plus
    gRPC/Flight/pgwire (+1..+3) and the internal mux (+10001) — a contended
    satellite port fails the server mid-startup, not this check."""
    needed = [port, port + 1, port + 2, port + 3, port + INTERNAL_MUX_OFFSET]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        busy = []
        for candidate in needed:
            with socket.socket() as probe:
                try:
                    probe.bind(("127.0.0.1", candidate))
                except OSError:
                    busy.append(candidate)
        if not busy:
            return
        time.sleep(1.0)
    raise BedError(f"ports {needed} never freed before the run")


class OwnedServer:
    """Spawn/stop the measured server with environment hygiene: the arm must
    come from CLI flags only — an invoking shell's exports must not re-point
    the backend or re-arm gates (pattern from the sift1m harness)."""

    STRIP_ENV = (
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
        METERING_INTERVAL_ENV,
        "PROXIMADB_TENANT_HEADER_TRUST",
        "PROXIMADB_TIER_HEADER_TRUST",
        # The residency store must come from the emitted config only — an
        # ambient export would silently point the bed's resident tier (and
        # the idle-clock read hits) somewhere else, e.g. tmpfs.
        "PROXIMADB_CACHE_LOCAL_DISK_PATH",
        "PROXIMADB_CACHE_LOCAL_DISK_MAX_GB",
        # Kill switch: an operator export would refuse every admission and
        # the run would fail for a reason the result never names.
        "PROXIMADB_TIER1_RESERVATIONS_OFF",
    )

    def __init__(
        self,
        binary: Path,
        config: Path,
        server: str,
        log_path: Path,
        metering_interval_secs: int,
        minio_endpoint: str | None,
        s3_region: str,
    ) -> None:
        self.binary = binary
        self.config = config
        self.server = server
        self.log_path = log_path
        self.metering_interval_secs = metering_interval_secs
        self.minio_endpoint = minio_endpoint
        self.s3_region = s3_region
        self.process: subprocess.Popen | None = None
        self.log_file = None

    def start(self) -> None:
        environment = dict(os.environ)
        for inherited in self.STRIP_ENV:
            environment.pop(inherited, None)
        # TD-TIER-1 SC6 S2: the idle-reclaim sweep and the reservation-usage
        # interval drain ride the metering loop's cadence (database.rs); the
        # production default is hourly, which a soak must compress.
        environment[METERING_INTERVAL_ENV] = str(self.metering_interval_secs)
        environment["RUST_LOG"] = "proximadb=info"
        if self.minio_endpoint is not None:
            for required in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
                if not os.environ.get(required):
                    raise BedError(
                        f"--minio-endpoint requires {required} in the invoking environment"
                    )
            environment.update(
                {
                    "PROXIMADB_S3_ENDPOINT": self.minio_endpoint,
                    "PROXIMADB_S3_FORCE_PATH_STYLE": "1",
                    "AWS_ENDPOINT": self.minio_endpoint,
                    "AWS_REGION": self.s3_region,
                    "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
                    "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
                    **(
                        {"AWS_ALLOW_HTTP": "true"}
                        if self.minio_endpoint.startswith("http://")
                        else {}
                    ),
                }
            )
        self.log_file = self.log_path.open("wb")
        self.process = subprocess.Popen(
            [str(self.binary), "-c", str(self.config)],
            cwd=str(self.log_path.parent),
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise BedError(
                    f"server exited with {self.process.returncode}; see {self.log_path}"
                )
            try:
                request_json(self.server + "/health/live", timeout=5)
                return
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                time.sleep(1.0)
        raise BedError(f"server did not become live; see {self.log_path}")

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                # Graceful shutdown flushes unflushed data; a SIGKILL mid-drain
                # leaves a half-written bed that fails later (sift1m lesson).
                self.process.wait(timeout=SHUTDOWN_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.log_file is not None:
            self.log_file.close()
        self.process = None


# ── Config emission ─────────────────────────────────────────────────────────


def write_config(
    path: Path,
    root: Path,
    data_dir: Path,
    port: int,
    storage_url: str,
    write_buffer_mb: int,
    pool: dict[str, Any],
) -> None:
    """Emit the server TOML. Pool field names are verified against
    ``ReservationPoolConfig`` (src/services/collection/reservation.rs) and the
    ``storage.reservation_pools`` table (src/core/config.rs)."""
    data = data_dir
    # BLOCKER fix (SC6 2b review): arm the process residency store. Without
    # a cache local-disk path the server registers NO PersistentByteStore
    # (sst/core.rs: the env var or SstConfig.cache_local_disk_path are the
    # only registrations), which degrades every grant ("no persistent local
    # store is configured" — READY unreachable) and starves the idle clock
    # (read hits come only from store reads), so the reclaim sweep would
    # release BOTH grants on age alone. The default budget (100 GiB) is
    # non-zero, so the path alone arms the store. Real local disk, not
    # tmpfs (runbook "what NOT to do").
    cache_local_disk = root / "local-disk-cache"
    pool_lines = [
        "# TD-TIER-1 SC6 (2b): ONE shared pool for every tenant's grants.",
        "[[storage.reservation_pools]]",
        f'id = "{pool["id"]}"',
        f'path = "{pool["path"]}"',
        f'allocatable_bytes = {pool["allocatable_bytes"]}',
        f'tenant_limit_bytes = {pool["tenant_limit_bytes"]}',
        f"overhead_factor = {pool['overhead_factor']}",
        f"default_grant_bytes = {pool['default_grant_bytes']}",
        "is_default = true",
        f"idle_reclaim_secs = {pool['idle_reclaim_secs']}",
        f"idle_reclaim_min_lifetime_secs = {pool['idle_reclaim_min_lifetime_secs']}",
        f"warm_bandwidth_mib_per_s = {pool['warm_bandwidth_mib_per_s']}",
    ]
    if pool["dram_budget_bytes"] is not None:
        pool_lines.append(f'dram_budget_bytes = {pool["dram_budget_bytes"]}')
    config = f"""# TD-TIER-1 SC6 (2b) multi-tenant soak bed — generated by multi_tenant_bed.py
[server]
node_id = "sc6-multi-tenant-bed"
bind_address = "127.0.0.1"
port = {port}
data_dir = "{data}"

# SaaS-shaped bed: every request carries X-Tenant-ID (fail-closed without it).
[server.tenant]
mode = "multi_tenant"
default_tenant = "default"

# Bench-local identity trust (the harness stamps X-Tenant-ID per request).
[security.tenant]
header_trust = "open"

[storage]
metadata_url = "file://{data / 'metadata'}"
mmap_enabled = false

[storage.optimization]
enable_mmap = false

[[storage.storage_locations]]
url = "{storage_url}"
weight = 1
tags = ["durable", "benchmark"]

[storage.wal_config]
write_buffer_directory = "file://{data / 'wal'}"
# The global WAL manifest must not live on the object store (TD-IOBUDGET-2).
global_manifest_url = "file://{data / 'manifest'}"
enable_wal = true
sync_mode = "PerBatch"
write_buffer_size_mb = {write_buffer_mb}
flush_interval_secs = 12

[storage.sst_config]
data_directory = "{data / 'sst'}"
mmap_enabled = false
segment_invariants_cache_mb = 256
survivor_cache_mb = 1024
cache_on_write = "all"
# TD-TIER-1 SC6 (2b): the process residency store — REQUIRED for grants to
# reach READY and for idle-reclaim's read-hit clock. Budget defaults to
# cache_local_disk_max_gb (100 GiB); the path alone arms the store.
cache_local_disk_path = "{cache_local_disk}"

{chr(10).join(pool_lines)}

[api]
# Warm phases stack object-store round trips behind the added-RTT proxy; the
# API timeout is not a measured dimension (unconditional, sift1m protocol).
timeout_seconds = {API_TIMEOUT_SECONDS}
unified_mode = true
unified_port = {port}
rest_port = {port}
grpc_port = {port + 1}
arrow_flight_port = {port + 2}
pg_port = {port + 3}
internal_mux_port = {port + INTERNAL_MUX_OFFSET}

[monitoring]
metrics_enabled = true
log_level = "info"
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config)


# ── Collection / status helpers ─────────────────────────────────────────────


def create_collection(
    server: str,
    tenant: str,
    name: str,
    dimension: int,
    storage_class: str,
    pool_id: str | None = None,
) -> str:
    body: dict[str, Any] = {
        "name": name,
        "dimension": dimension,
        "engine": "sst",
        "enable_proxima_record": True,
        "distance_metric": "euclidean",
    }
    if storage_class != "balanced":
        body["storage_class"] = storage_class
    if pool_id is not None:
        body["pool_id"] = pool_id
    response = request_json(
        server + "/api/v2/collections", method="POST", body=body, tenant=tenant
    )
    collection_id = str(response.get("collection_id") or response.get("id") or "")
    if not collection_id:
        raise BedError(f"create response for {name} has no collection id: {response}")
    return collection_id


def collection_detail(server: str, tenant: str, collection_id: str) -> dict:
    return request_json(
        f"{server}/api/v2/collections/{collection_id}", method="GET", tenant=tenant
    )


def collection_status(server: str, tenant: str, collection_id: str) -> dict:
    detail = collection_detail(server, tenant, collection_id)
    status = detail.get("storage_class_status")
    return status if isinstance(status, dict) else {}


def status_state(status: dict) -> str:
    return str(status.get("state", "")).upper()


def wait_until_ready(
    server: str, tenant: str, collection_id: str, timeout_secs: int, label: str
) -> dict:
    deadline = time.monotonic() + timeout_secs
    status: dict = {}
    while time.monotonic() < deadline:
        status = collection_status(server, tenant, collection_id)
        if status_state(status) == "READY":
            return status
        time.sleep(POLL_SECONDS)
    raise BedError(
        f"ready_pin: {label} never reached READY within {timeout_secs}s: {status!r}"
    )


def pin_status(status: dict) -> dict:
    """Pin the READY(g) evidence fields (sift1m residency-pin pattern)."""
    return {
        "state": status.get("state"),
        "reason": status.get("reason"),
        "policy_generation": status.get("policy_generation"),
        "reserved_bytes": status.get("reserved_bytes"),
        "resident_bytes": status.get("resident_bytes"),
        "covered_bytes": status.get("covered_bytes"),
        "covered_generation": status.get("covered_generation"),
    }


def delete_collection(server: str, tenant: str, collection_id: str) -> None:
    request_json(
        f"{server}/api/v2/collections/{collection_id}", method="DELETE", tenant=tenant
    )


# ── Query machinery ─────────────────────────────────────────────────────────


class QueryOutcome:
    __slots__ = ("latency_ms", "gets", "bytes_read", "ids", "error", "trace_present")

    def __init__(
        self,
        latency_ms: float,
        gets: int,
        bytes_read: int,
        ids: list[str],
        error: str | None = None,
        trace_present: bool = False,
    ) -> None:
        self.latency_ms = latency_ms
        self.gets = gets
        self.bytes_read = bytes_read
        self.ids = ids
        self.error = error
        self.trace_present = trace_present


def search_once(
    server: str, tenant: str, collection_id: str, vector: list[float], top_k: int
) -> QueryOutcome:
    """One search with ``debug: true`` so the response carries the per-query
    ``search_plan_trace`` (object_store_gets / object_store_bytes_read)."""
    started = time.perf_counter()
    try:
        response = request_json(
            f"{server}/api/v2/collections/{collection_id}/search",
            method="POST",
            body={"vector": vector, "top_k": top_k, "debug": True},
            tenant=tenant,
        )
    except urllib.error.HTTPError as error:
        return QueryOutcome(
            (time.perf_counter() - started) * 1000.0, 0, 0, [], f"HTTP {error.code}"
        )
    except (TimeoutError, OSError) as error:
        return QueryOutcome(
            (time.perf_counter() - started) * 1000.0,
            0,
            0,
            [],
            f"transport {error!r}",
        )
    latency_ms = (time.perf_counter() - started) * 1000.0
    trace = response.get("search_plan_trace")
    gets = 0
    bytes_read = 0
    trace_present = isinstance(trace, dict)
    if trace_present:
        gets = int(trace.get("object_store_gets", 0) or 0)
        bytes_read = int(trace.get("object_store_bytes_read", 0) or 0)
    ids = [
        item.get("id")
        for item in (response.get("results") or [])
        if isinstance(item, dict)
    ]
    return QueryOutcome(latency_ms, gets, bytes_read, ids, trace_present=trace_present)


def summarize_outcomes(outcomes: list[QueryOutcome]) -> dict:
    """Per-tenant/per-phase gauge block (counter sources identical to the
    sift1m harness; per-query attribution via search_plan_trace)."""
    ok = [outcome for outcome in outcomes if outcome.error is None]
    latencies = sorted(outcome.latency_ms for outcome in ok)
    gets = [outcome.gets for outcome in ok]
    bytes_read = [outcome.bytes_read for outcome in ok]
    n = len(ok)
    return {
        "queries": len(outcomes),
        "errors": len(outcomes) - n,
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p99": percentile(latencies, 0.99),
            "mean": (sum(latencies) / n) if n else 0.0,
        },
        "gets_per_query": {
            "mean": (sum(gets) / n) if n else 0.0,
            "max": max(gets, default=0),
        },
        "bytes_per_query": {"mean": (sum(bytes_read) / n) if n else 0.0},
        # Warm fraction: queries served with zero object-store GETs.
        "warm_fraction": (sum(1 for value in gets if value == 0) / n) if n else 0.0,
    }


def trace_counter_gate(
    phase: str, prom_gets_delta: float, outcomes: list[QueryOutcome]
) -> tuple[list[str], str | None]:
    """The per-query search_plan_trace and the process-global Prometheus GET
    counter must agree in direction. Without this gate, a server-side break
    of the trace attachment (debug:true ignored) would silently zero every
    gets_per_query gauge and read as a perfectly warm bed.

    Returns (problems, vacated): ``vacated`` names why the gate could not run
    (no queries, or none succeeded) and MUST be recorded in the phase record
    — a vacated gate is never a silent skip; the caller's own zero-success
    failure makes the run red in that case."""
    problems: list[str] = []
    ok = [outcome for outcome in outcomes if outcome.error is None]
    if not outcomes:
        return problems, "no queries ran in this phase"
    if not ok:
        return (
            problems,
            f"all {len(outcomes)} queries failed — trace attribution vacuous",
        )
    if not any(outcome.trace_present for outcome in ok):
        problems.append(
            f"trace_vs_counter[{phase}]: search_plan_trace absent on every "
            "successful response (debug:true not honored?) — per-tenant GET "
            "attribution would be silently zero"
        )
    trace_gets = sum(outcome.gets for outcome in ok)
    # Accepted-as-is residual (round-2 review): a background GET during an
    # all-warm sweep can advance the process counter while every trace reads
    # 0 — a spurious red. Kept loud rather than silenced.
    if prom_gets_delta > 0 and trace_gets == 0:
        problems.append(
            f"trace_vs_counter[{phase}]: object_store_gets_total advanced by "
            f"{prom_gets_delta:.0f} while summed trace GETs == 0 — trace "
            "attachment is broken or mis-attributing server-side (or a "
            "background GET raced an all-warm sweep)"
        )
    return problems, None


def recall_for(outcomes: list[QueryOutcome], expected: list[set[str]]) -> dict:
    hits = 0
    denominator = 0
    compared = 0
    for index in range(min(len(outcomes), len(expected))):
        outcome = outcomes[index]
        truth_ids = expected[index]
        if not truth_ids or outcome.error is not None:
            continue
        compared += 1
        denominator += len(truth_ids)
        hits += len(set(outcome.ids) & truth_ids)
    return {
        "compared": compared,
        "hits": hits,
        "expected_ids": denominator,
        "recall": (hits / denominator) if denominator else 0.0,
    }


def parity_for(granted: list[QueryOutcome], balanced: list[QueryOutcome]) -> dict:
    overlaps: list[float] = []
    exact = 0
    compared = 0
    for a, b in zip(granted, balanced):
        if a.error is not None or b.error is not None:
            continue
        compared += 1
        set_a, set_b = set(a.ids), set(b.ids)
        union = set_a | set_b
        overlaps.append(len(set_a & set_b) / len(union) if union else 1.0)
        if set_a == set_b:
            exact += 1
    return {
        "compared": compared,
        "mean_overlap": (sum(overlaps) / len(overlaps)) if overlaps else 0.0,
        "exact_set_equality_fraction": (exact / compared) if compared else 0.0,
    }


# ── Ingest / settle ─────────────────────────────────────────────────────────


def capped_batches(records: list[dict], batch_size: int) -> list[list[dict]]:
    if not records:
        return []
    per_record = max(1, len(json.dumps(records[0]).encode()))
    cap = max(1, BODY_SAFETY_BYTES // per_record)
    effective = min(batch_size, cap)
    return [records[i : i + effective] for i in range(0, len(records), effective)]


def ingest_slice(
    server: str,
    tenant: str,
    collection_id: str,
    corpus: list[list[float]],
    start: int,
    count: int,
    batch_size: int,
    label: str,
) -> dict:
    records = [
        {"id": f"v{start + offset}", "vector": corpus[start + offset]}
        for offset in range(count)
    ]
    batches = capped_batches(records, batch_size)
    if not batches:
        raise BedError(f"ingest slice {label} produced no batches")
    started = time.perf_counter()
    retry_counts: dict[int, int] = {}
    for batch in batches:
        counts = post_batch_with_retry(server, collection_id, batch, tenant)
        for status, hits in counts.items():
            retry_counts[status] = retry_counts.get(status, 0) + hits
    elapsed = time.perf_counter() - started
    return {
        "label": label,
        "rows": count,
        "seconds": elapsed,
        "vectors_per_second": count / elapsed if elapsed > 0 else 0.0,
        "batches": len(batches),
        "batch_cap_rows": len(batches[0]),
        "retry_status_counts": retry_counts,
    }


def wait_settled(
    server: str,
    tenant: str,
    collection_id: str,
    expected_rows: int,
    timeout_secs: int,
    stable_secs: int,
) -> dict:
    """Wait until the collection's unflushed WAL is quiescent for
    ``stable_secs``. ``record_count`` stats are best-effort context only — the
    REST counter is Option and not maintained by the write path (#1527)."""
    deadline = time.monotonic() + timeout_secs
    stable_since: float | None = None
    last: dict = {}
    while time.monotonic() < deadline:
        wal_bytes = labelled_wal_bytes(server, collection_id)
        detail = collection_detail(server, tenant, collection_id)
        stats = detail.get("stats") if isinstance(detail.get("stats"), dict) else {}
        record_count = stats.get("record_count")
        # An absent per-collection gauge means no remaining unflushed WAL (the
        # registry removes the labelled sample on retirement) — quiescent.
        settled = wal_bytes is None or wal_bytes == 0.0
        if record_count is not None:
            settled = settled and int(record_count) >= expected_rows
        if settled:
            stable_since = stable_since or time.monotonic()
            if time.monotonic() - stable_since >= stable_secs:
                return {
                    "settled": True,
                    "wal_bytes": wal_bytes,
                    "record_count": record_count,
                }
        else:
            stable_since = None
        last = {"wal_bytes": wal_bytes, "record_count": record_count}
        time.sleep(POLL_SECONDS)
    raise BedError(
        f"collection {collection_id} never settled within {timeout_secs}s: {last!r}"
    )


def usage_records(storage_root: Path | None, pool_id: str) -> list[dict]:
    """Read the durable reservation-usage records (file:// beds). The server
    writes them PER TENANT via DrPathBuilder —
    ``<storage_root>/data/{tenant_id}/_metering/reservations/{pool}/
    usage-{epoch}-{sequence}.json`` (metering_writer.rs +
    DrTenantSystemPath::metering_subprefix; account_id=None on this path) —
    so the glob spans every tenant directory. Each record is one accrual
    interval; the dedup identity (tenant, pool, producer_epoch, sequence) is
    embedded in the object name."""
    if storage_root is None:
        return []
    records = []
    for path in sorted(
        storage_root.glob(f"data/*/_metering/reservations/{pool_id}/usage-*.json")
    ):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def contiguous_accrual(records: list[dict], grant_id: str) -> dict:
    """Idle-accrual evidence: this grant's intervals tile its live lifetime
    without gaps or overlaps (coalesced-interval semantics). Absent records
    are disclosed as None, not failed."""
    mine = sorted(
        (
            int(record.get("interval_start_unix_secs", 0)),
            int(record.get("interval_end_unix_secs", 0)),
        )
        for record in records
        if record.get("grant_id") == grant_id
    )
    if not mine:
        return {"records": 0, "contiguous": None}
    gaps = sum(
        1
        for (_start_a, end_a), (start_b, _end_b) in zip(mine, mine[1:])
        if start_b != end_a
    )
    return {
        "records": len(mine),
        "span": [mine[0][0], mine[-1][1]],
        "gaps": gaps,
        "contiguous": gaps == 0,
    }


# ── Main ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TD-TIER-1 SC6 (2b) multi-tenant soak bed harness"
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("target/release-server/proximadb-server"),
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="bed root (unique per run; never a shared /tmp path)",
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        help="generated server TOML path (default <root>/config.toml)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="server data directory (default <root>/data)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=PORT_DEFAULT,
        help=f"unified REST port (599x range; default {PORT_DEFAULT})",
    )
    parser.add_argument(
        "--storage-url",
        help="durable segment base URL; defaults to file://<root>/objects",
    )
    parser.add_argument(
        "--minio-endpoint",
        help="S3-shaped endpoint (e.g. http://127.0.0.1:12100); forwards "
        "AWS_ENDPOINT/PROXIMADB_S3_ENDPOINT to the server and switches the "
        "default storage URL to s3://<bucket>",
    )
    parser.add_argument("--minio-bucket", default="sc6-multi-tenant-bed")
    parser.add_argument("--s3-region", default="us-east-1")

    corpus = parser.add_mutually_exclusive_group()
    corpus.add_argument(
        "--synthetic",
        action="store_true",
        help="generate a deterministic synthetic corpus (no SIFT files needed)",
    )
    corpus.add_argument("--sift-dir", type=Path)
    parser.add_argument("--base-path", type=Path)
    parser.add_argument("--query-path", type=Path)
    parser.add_argument("--groundtruth-path", type=Path)
    parser.add_argument("--synthetic-seed", type=int, default=20261103)
    parser.add_argument("--synthetic-dimension", type=int, default=96)

    parser.add_argument(
        "--rows-per-tenant",
        type=int,
        default=0,
        help="corpus rows per tenant slice (default 20000 SIFT / 2500 synthetic)",
    )
    parser.add_argument(
        "--delta-rows",
        type=int,
        default=0,
        help="extra rows perf-b ingests during the concurrency phase "
        "(default 10% of rows)",
    )
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="requested batch rows; capped to ~1.8MB JSON bodies — a "
        "harness-side pacing choice (the REST server's default body limit is "
        "64MB, not the reason for the cap)",
    )
    parser.add_argument("--write-buffer-mb", type=int, default=512)

    parser.add_argument("--pool-allocatable-gb", type=float, default=3.0)
    parser.add_argument("--pool-grant-gb", type=float, default=1.0)
    parser.add_argument(
        "--tenant-limit-gb",
        type=float,
        default=0.0,
        help="per-tenant grant ceiling in GiB (0 = the pool allocatable)",
    )
    parser.add_argument(
        "--overhead-factor",
        type=float,
        default=0.1,
        help="must match the server pool's overhead_factor (charge arithmetic)",
    )
    parser.add_argument("--idle-reclaim-secs", type=int, default=120)
    parser.add_argument("--idle-reclaim-min-lifetime-secs", type=int, default=60)
    parser.add_argument("--warm-bandwidth-mib-per-s", type=int, default=64)
    parser.add_argument(
        "--dram-budget-gb",
        type=float,
        default=0.0,
        help="pool dram_budget_bytes ceiling in GiB (0 = uncapped/omitted)",
    )
    parser.add_argument(
        "--metering-interval-secs",
        type=int,
        default=15,
        help="PROXIMADB_METERING_FLUSH_INTERVAL_SECS for the server (the "
        "idle-reclaim sweep and interval drain ride this cadence)",
    )
    parser.add_argument(
        "--duration-secs", type=int, default=120, help="concurrency-phase window"
    )
    parser.add_argument(
        "--idle-wait-secs",
        type=int,
        default=0,
        help="idle-phase observation window (0 = auto from the TTL)",
    )
    parser.add_argument("--residency-ready-timeout-secs", type=int, default=600)
    parser.add_argument("--settle-timeout-secs", type=int, default=1_200)
    parser.add_argument("--stable-secs", type=int, default=30)
    parser.add_argument("--min-recall", type=float, default=0.975)
    parser.add_argument(
        "--min-parity-recall",
        type=float,
        default=0.95,
        help="granted-vs-balanced top-k overlap floor (exact-set-equality "
        "fraction is reported alongside; score ties can flip one id)",
    )

    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="reuse a prior run's settled collections (same --root/data-dir); "
        "requires --prior-result — ids are registered from it and verified "
        "against this server's catalog, no re-create, no re-ingest. Note the "
        "drop_release phase still DROPS collections: reuse re-runs the gauge "
        "phases, then consumes the bed",
    )
    parser.add_argument(
        "--prior-result", type=Path, help="result.json of the build run"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="emit the config TOML and the bed plan, start nothing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if args.skip_ingest and not args.prior_result:
        raise SystemExit("--skip-ingest requires --prior-result")
    if args.skip_ingest and args.dry_run:
        raise SystemExit(
            "--dry-run starts nothing; --skip-ingest is meaningless with it"
        )

    config_out = args.config_out or (root / "config.toml")
    # Resolve before interpolating into the TOML and recording: a relative
    # --data-dir must not depend on the spawned server inheriting the bed cwd,
    # and result.json must name the absolute path the server actually uses.
    data_dir = (args.data_dir or (root / "data")).resolve()
    pool_dir = root / "pool"
    if args.storage_url:
        storage_url = args.storage_url
    elif args.minio_endpoint:
        storage_url = f"s3://{args.minio_bucket}"
    else:
        storage_url = f"file://{root / 'objects'}"

    grant_bytes = int(args.pool_grant_gb * (1 << 30))
    allocatable_bytes = int(args.pool_allocatable_gb * (1 << 30))
    tenant_limit_bytes = (
        int(args.tenant_limit_gb * (1 << 30))
        if args.tenant_limit_gb > 0
        else allocatable_bytes
    )
    dram_budget_bytes = (
        int(args.dram_budget_gb * (1 << 30)) if args.dram_budget_gb > 0 else None
    )
    expected_effective = effective_bytes(grant_bytes, args.overhead_factor)
    pool = {
        "id": POOL_ID,
        "path": str(pool_dir),
        "allocatable_bytes": allocatable_bytes,
        "tenant_limit_bytes": tenant_limit_bytes,
        "overhead_factor": args.overhead_factor,
        "default_grant_bytes": grant_bytes,
        "idle_reclaim_secs": args.idle_reclaim_secs,
        "idle_reclaim_min_lifetime_secs": args.idle_reclaim_min_lifetime_secs,
        "warm_bandwidth_mib_per_s": args.warm_bandwidth_mib_per_s,
        "dram_budget_bytes": dram_budget_bytes,
    }

    rows_per_tenant = args.rows_per_tenant or (2_500 if args.synthetic else 20_000)
    delta_rows = args.delta_rows or max(1, rows_per_tenant // 10)
    idle_wait = args.idle_wait_secs or (
        args.idle_reclaim_secs + 6 * args.metering_interval_secs + 30
    )

    dataset = resolve_dataset(args, rows_per_tenant, delta_rows, args.dry_run)

    if args.dry_run:
        # Emit the config + the bed plan; start nothing.
        root.mkdir(parents=True, exist_ok=True)
        write_config(
            config_out,
            root,
            data_dir,
            args.port,
            storage_url,
            args.write_buffer_mb,
            pool,
        )
        plan = {
            "protocol": PROTOCOL,
            "dry_run": True,
            "root": str(root),
            "config_out": str(config_out),
            "data_dir": str(data_dir),
            "residency_store": str(root / "local-disk-cache"),
            "storage_url": storage_url,
            "pool": pool,
            "expected_effective_bytes_per_grant": expected_effective,
            "expected_grants": 2,
            "expected_charged_after_setup": 2 * expected_effective,
            "over_admission_expectation": "refused (a 3rd grant would exceed allocatable)",
            "tenants": {
                tenant: {
                    name: {"storage_class": klass, "slice": label}
                    for t, name, klass, label in BED
                    if t == tenant
                }
                for tenant in ("sc6-perf-a", "sc6-perf-b", "sc6-bal")
            },
            "phases": [
                "ready_pin (runbook: status reaches READY(g) and stays)",
                "recall_parity_warm (same correctness/recall across classes)",
                "concurrent_shared_pool (no capacity overcommit under concurrency, SC-TE3)",
                "idle_reclaim (idle accrual matches the ledger; no auto re-admission)",
                "drop_release (drops/updates release cleanly; update = drop+recreate)",
            ],
            "dataset": {
                "mode": dataset["mode"],
                "dimension": dataset["dimension"],
                "note": dataset["note"],
                "slices": {
                    label: list(bounds) for label, bounds in dataset["slices"].items()
                },
                "rows_per_tenant": rows_per_tenant,
                "delta_rows": delta_rows,
            },
            "knobs": {
                "metering_interval_secs": args.metering_interval_secs,
                "idle_reclaim_secs": args.idle_reclaim_secs,
                "duration_secs": args.duration_secs,
                "queries": args.queries,
                "top_k": args.top_k,
            },
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        print(f"dry-run: config written to {config_out}; nothing started")
        return 0

    # ── Real run ────────────────────────────────────────────────────────────
    if not args.binary.exists():
        raise BedError(f"server binary not found: {args.binary}")
    root.mkdir(parents=True, exist_ok=True)
    write_config(
        config_out, root, data_dir, args.port, storage_url, args.write_buffer_mb, pool
    )

    result: dict[str, Any] = {
        "protocol": PROTOCOL,
        "binary": {"path": str(args.binary), "sha256": sha256(args.binary)},
        "harness": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "root": str(root),
        "config_out": str(config_out),
        "data_dir": str(data_dir),
        "residency_store": str(root / "local-disk-cache"),
        "storage_url": storage_url,
        "port": args.port,
        "pool": pool,
        "expected_effective_bytes_per_grant": expected_effective,
        "dataset": {
            "mode": dataset["mode"],
            "dimension": dataset["dimension"],
            "rows_per_tenant": rows_per_tenant,
            "delta_rows": delta_rows,
            "queries": len(dataset["queries"]),
            "top_k": args.top_k,
            "note": dataset["note"],
            "skipped_no_in_slice_truth": dataset["skipped_no_truth"],
        },
        "phases": {},
        "tenants": {},
        "residency": {},
        "ledger": {},
        "ingest": {},
        "collection_ids": {},
        "gates_note": (
            "latency is reported, never gated (inherited SC6 caveat: both arms "
            "miss the 50ms cold-p50 gate identically); per-query GETs/bytes "
            "come from search_plan_trace (debug:true), gated against the same "
            "Prometheus counters as the sift1m harness (a trace/counter "
            "contradiction or trace-less sweep fails the run)"
        ),
    }

    server = f"http://127.0.0.1:{args.port}"
    active: OwnedServer | None = None
    failures: list[str] = []
    lock = threading.Lock()
    outcome_log: dict[tuple[str, str, str], list[QueryOutcome]] = {}

    def record_outcomes(tenant: str, name: str, phase: str, outcomes: list) -> None:
        with lock:
            outcome_log.setdefault((tenant, name, phase), []).extend(outcomes)

    def snapshot_ledger(note: str) -> dict:
        return ledger_view(read_ledger(pool_dir), args.overhead_factor, note)

    try:
        wait_for_port_free(args.port)
        active = OwnedServer(
            binary=args.binary,
            config=config_out,
            server=server,
            log_path=root / "server.log",
            metering_interval_secs=args.metering_interval_secs,
            minio_endpoint=args.minio_endpoint,
            s3_region=args.s3_region,
        )
        active.start()
        dimension = dataset["dimension"]
        corpus = dataset["corpus"]

        # ── Setup: collections (skip-ingest re-registers verified ids) ──────
        prior_collections: dict[str, dict[str, str]] = {}
        if args.skip_ingest:
            prior = json.loads(args.prior_result.read_text())
            prior_collections = prior.get("collection_ids", {})
            if not prior_collections:
                raise BedError("--prior-result carries no collection_ids map")

        collection_ids: dict[str, dict[str, str]] = {}
        if args.skip_ingest:
            # Bed reuse: do NOT re-create (a same-root rerun would hit
            # AlreadyExists and a fresh-root rerun would mint new ids against
            # the old object prefixes). Register the prior run's ids and
            # verify each resolves in THIS server's catalog, fail-closed.
            for tenant, name, _storage_class, _slice in BED:
                prior_id = prior_collections.get(tenant, {}).get(name)
                if not prior_id:
                    raise BedError(f"prior result lacks {tenant}/{name}")
                detail = collection_detail(server, tenant, prior_id)
                if detail.get("name") != name:
                    raise BedError(
                        f"skip-ingest fixture mismatch for {tenant}/{name}: id "
                        f"{prior_id} resolves to name {detail.get('name')!r}"
                    )
                collection_ids.setdefault(tenant, {})[name] = prior_id
        else:
            for tenant, name, storage_class, _slice in BED:
                collection_ids.setdefault(tenant, {})[name] = create_collection(
                    server,
                    tenant,
                    name,
                    dimension,
                    storage_class,
                    pool_id=(POOL_ID if storage_class == "performance" else None),
                )
        result["collection_ids"] = collection_ids

        if args.skip_ingest:
            result["ingest"] = {"skipped": True, "source": str(args.prior_result)}
        else:
            for tenant, name, _storage_class, slice_label in BED:
                start, count = dataset["slices"][slice_label]
                result["ingest"][name] = ingest_slice(
                    server,
                    tenant,
                    collection_ids[tenant][name],
                    corpus,
                    start,
                    count,
                    args.batch_size,
                    name,
                )
            settle: dict[str, Any] = {}
            for tenant, name, _storage_class, slice_label in BED:
                start, count = dataset["slices"][slice_label]
                settle[name] = wait_settled(
                    server,
                    tenant,
                    collection_ids[tenant][name],
                    count,
                    args.settle_timeout_secs,
                    args.stable_secs,
                )
            result["settle"] = settle

        # ── Phase: ready_pin ────────────────────────────────────────────────
        ready_pins: dict[str, Any] = {}
        for tenant, name, storage_class, _slice in BED:
            if storage_class != "performance":
                continue
            status = wait_until_ready(
                server,
                tenant,
                collection_ids[tenant][name],
                args.residency_ready_timeout_secs,
                f"{tenant}/{name}",
            )
            pinned = pin_status(status)
            # READY(g) must survive beyond the first observation (runbook
            # phase 1: "reaches READY(g) and stays through cycles").
            time.sleep(args.stable_secs)
            still = collection_status(server, tenant, collection_ids[tenant][name])
            pinned["state_after_stable_secs"] = still.get("state")
            ready_pins[name] = pinned
            if status_state(still) != "READY":
                failures.append(
                    f"ready_pin: {name} left READY within {args.stable_secs}s "
                    f"({still.get('state')!r})"
                )
        result["residency"]["ready_pins"] = ready_pins

        setup_ledger = snapshot_ledger("after setup")
        result["ledger"]["after_setup"] = setup_ledger
        expected_setup_charge = 2 * expected_effective
        if setup_ledger["charged_effective"] > allocatable_bytes:
            failures.append(
                "no_overcommit: setup itself overcommitted: "
                f"{setup_ledger['charged_effective']} > {allocatable_bytes}"
            )
        if setup_ledger["charged_effective"] != expected_setup_charge:
            failures.append(
                "no_overcommit: setup charge "
                f"{setup_ledger['charged_effective']} != expected "
                f"{expected_setup_charge} (2 grants x effective)"
            )

        # ── Phase: recall_parity_warm ───────────────────────────────────────
        prom_before = scrape(server)
        warm_outcomes: dict[tuple[str, str], list[QueryOutcome]] = {}
        for tenant, name, _storage_class, _slice in BED:
            outcomes = [
                search_once(
                    server, tenant, collection_ids[tenant][name], vector, args.top_k
                )
                for vector in dataset["queries"]
            ]
            warm_outcomes[(tenant, name)] = outcomes
            record_outcomes(tenant, name, "recall_parity_warm", outcomes)
        prom_after = scrape(server)
        result["phases"]["recall_parity_warm_prometheus"] = {
            name: metric_delta(prom_before, prom_after, name) for name in PROM_COUNTERS
        }
        warm_all = [outcome for pair in warm_outcomes.values() for outcome in pair]
        warm_problems, warm_vacated = trace_counter_gate(
            "recall_parity_warm",
            metric_delta(
                prom_before,
                prom_after,
                "proximadb_object_store_gets_total",
            ),
            warm_all,
        )
        result["phases"]["recall_parity_warm_trace_gate"] = {"vacated": warm_vacated}
        failures.extend(warm_problems)
        # Zero successful warm queries already fails this phase below via the
        # granted collections' no-comparable-queries recall gate (and the
        # parity gate's compared==0 branch) — stated here so the vacated
        # record above is not read as an unbackstopped skip.

        recalls: dict[str, Any] = {}
        for tenant, name, storage_class, slice_label in BED:
            recall = recall_for(
                warm_outcomes[(tenant, name)], dataset["truth"][slice_label]
            )
            recalls[name] = recall
            if storage_class == "performance":
                if recall["compared"] == 0:
                    failures.append(f"recall: {name} had no comparable queries")
                elif recall["recall"] < args.min_recall:
                    failures.append(
                        f"recall: {name} warm recall {recall['recall']:.4f} "
                        f"< floor {args.min_recall}"
                    )
        result["phases"]["recall"] = recalls
        parity = parity_for(
            warm_outcomes[("sc6-perf-a", "perf-a")],
            warm_outcomes[("sc6-perf-a", "bal-a")],
        )
        result["phases"]["parity_perf_a_vs_bal_a"] = parity
        if parity["compared"] == 0:
            failures.append("parity: no comparable granted-vs-balanced queries")
        elif parity["mean_overlap"] < args.min_parity_recall:
            failures.append(
                f"parity: granted-vs-balanced overlap {parity['mean_overlap']:.4f} "
                f"< floor {args.min_parity_recall}"
            )

        # ── Phase: concurrent_shared_pool (SC-TE3) ──────────────────────────
        stop_flag = threading.Event()
        ledger_samples: list[dict] = []
        sampler_errors: list[str] = []

        def ledger_sampler() -> None:
            while not stop_flag.is_set():
                try:
                    ledger_samples.append(snapshot_ledger("concurrency sample"))
                except Exception as error:  # recorded, not raised
                    sampler_errors.append(repr(error))
                stop_flag.wait(POLL_SECONDS)

        def query_worker(tenant: str, name: str) -> None:
            queries = dataset["queries"]
            outcomes: list[QueryOutcome] = []
            index = 0
            while not stop_flag.is_set():
                outcomes.append(
                    search_once(
                        server,
                        tenant,
                        collection_ids[tenant][name],
                        queries[index % len(queries)],
                        args.top_k,
                    )
                )
                index += 1
            record_outcomes(tenant, name, "concurrent_shared_pool", outcomes)

        def delta_ingest_worker() -> None:
            if args.skip_ingest:
                return  # bed reuse must not add rows to a settled fixture
            start, count = dataset["slices"]["B"]
            try:
                stats = ingest_slice(
                    server,
                    PERF_TENANT_IDLE_SUBJECT,
                    collection_ids[PERF_TENANT_IDLE_SUBJECT]["perf-b"],
                    corpus,
                    start + count,
                    delta_rows,
                    args.batch_size,
                    "perf-b delta (concurrency)",
                )
                with lock:
                    result["ingest"]["perf-b-delta"] = stats
            except Exception as error:  # recorded as a gate failure
                with lock:
                    failures.append(f"concurrent ingest failed: {error!r}")

        sampler = threading.Thread(target=ledger_sampler, daemon=True)
        workers = [
            threading.Thread(target=query_worker, args=(tenant, name), daemon=True)
            for tenant, name in (
                ("sc6-perf-a", "perf-a"),
                ("sc6-perf-a", "bal-a"),
                ("sc6-perf-b", "perf-b"),
                ("sc6-bal", "bal-c"),
            )
        ]
        prom_conc_before = scrape(server)
        sampler.start()
        for worker in workers:
            worker.start()
        ingest_thread = threading.Thread(target=delta_ingest_worker, daemon=True)
        ingest_thread.start()
        time.sleep(args.duration_secs)
        stop_flag.set()
        for worker in workers:
            worker.join(timeout=API_TIMEOUT_SECONDS)
        ingest_thread.join(timeout=API_TIMEOUT_SECONDS)
        sampler.join(timeout=10)
        prom_conc_after = scrape(server)
        result["phases"]["concurrent_shared_pool_prometheus"] = {
            name: metric_delta(prom_conc_before, prom_conc_after, name)
            for name in PROM_COUNTERS
        }
        with lock:
            conc_outcomes = [
                outcome
                for (_tenant, _name, phase), outcomes in outcome_log.items()
                if phase == "concurrent_shared_pool"
                for outcome in outcomes
            ]
        conc_succeeded = sum(1 for outcome in conc_outcomes if outcome.error is None)
        result["phases"]["concurrent_shared_pool_queries"] = {
            "issued": len(conc_outcomes),
            "succeeded": conc_succeeded,
        }
        if conc_succeeded == 0:
            # A concurrency phase with zero successful queries vacates every
            # gauge and the trace gate — that is a failed phase, not a skip
            # (the server may be rejecting searches while ingest/ledger stay
            # healthy, which is exactly the shape this must catch).
            failures.append(
                "concurrent_shared_pool: zero queries succeeded "
                f"({len(conc_outcomes)} issued) — phase vacated"
            )
        conc_problems, conc_vacated = trace_counter_gate(
            "concurrent_shared_pool",
            metric_delta(
                prom_conc_before,
                prom_conc_after,
                "proximadb_object_store_gets_total",
            ),
            conc_outcomes,
        )
        result["phases"]["concurrent_shared_pool_trace_gate"] = {
            "vacated": conc_vacated
        }
        failures.extend(conc_problems)

        if sampler_errors:
            failures.append(
                f"ledger sampler failed {len(sampler_errors)} times: "
                f"{sampler_errors[0]}"
            )
        overcommits = [
            sample
            for sample in ledger_samples
            if sample["charged_effective"] > allocatable_bytes
        ]
        result["ledger"]["concurrency_samples"] = {
            "n": len(ledger_samples),
            "min_charged": min(
                (sample["charged_effective"] for sample in ledger_samples), default=None
            ),
            "max_charged": max(
                (sample["charged_effective"] for sample in ledger_samples), default=None
            ),
            "max_utilization": (
                max(
                    sample["charged_effective"] / allocatable_bytes
                    for sample in ledger_samples
                )
                if ledger_samples
                else None
            ),
            "overcommit_samples": len(overcommits),
        }
        if overcommits:
            failures.append(
                "no_overcommit (SC-TE3): "
                f"{len(overcommits)}/{len(ledger_samples)} concurrency samples "
                f"exceeded allocatable {allocatable_bytes}"
            )

        # Over-admission drill: a third PERFORMANCE grant must be refused and
        # must leave the ledger unchanged (SC-TE4 capacity pressure).
        before_drill = snapshot_ledger("before over-admission drill")
        status_code, text, payload = request_json_status(
            server + "/api/v2/collections",
            method="POST",
            body={
                "name": "perf-over",
                "dimension": dimension,
                "engine": "sst",
                "enable_proxima_record": True,
                "distance_metric": "euclidean",
                "storage_class": "performance",
                "pool_id": POOL_ID,
            },
            tenant="sc6-bal",
        )
        # NOTE on the matcher: the manager wraps EVERY admission failure in
        # one outer context ("TD-TIER-1 SC3: PERFORMANCE admission refused"),
        # so the specific inner cause (capacity vs tenant-limit vs kill
        # switch) does not reach the HTTP body — matching the outer phrase is
        # the best a client can do today. Tightening needs a server-side fix
        # (error-code plumbing); do not widen this matcher further.
        refused = status_code >= 400 and (
            "pool capacity exceeded" in text or "admission refused" in text
        )
        time.sleep(2 * POLL_SECONDS)
        after_drill = snapshot_ledger("after over-admission drill")
        drill = {
            "status": status_code,
            "refusal_match": refused,
            "body_snippet": text[:300],
            "charged_before": before_drill["charged_effective"],
            "charged_after": after_drill["charged_effective"],
        }
        result["ledger"]["over_admission_drill"] = drill
        if status_code == 0:
            failures.append(f"admission_refusal: drill transport failure: {text!r}")
        elif status_code < 400:
            failures.append(
                "admission_refusal: over-admission create unexpectedly SUCCEEDED "
                f"(status {status_code}) — the pool overcommitted; cleaning up"
            )
            created = str(
                (payload or {}).get("collection_id") or (payload or {}).get("id") or ""
            )
            if created:
                try:
                    delete_collection(server, "sc6-bal", created)
                except Exception as cleanup_error:
                    failures.append(f"drill cleanup failed: {cleanup_error!r}")
        elif not refused:
            failures.append(
                f"admission_refusal: create refused (status {status_code}) but the "
                "body names neither 'pool capacity exceeded' nor 'admission refused'"
            )
        if after_drill["charged_effective"] != before_drill["charged_effective"]:
            failures.append(
                "admission_refusal: ledger changed across the drill: "
                f"{before_drill['charged_effective']} -> {after_drill['charged_effective']}"
            )

        # ── Phase: idle_reclaim ─────────────────────────────────────────────
        perf_b_id = collection_ids[PERF_TENANT_IDLE_SUBJECT]["perf-b"]
        pre_idle = read_ledger(pool_dir)
        grant_b = live_grant_for(pre_idle, PERF_TENANT_IDLE_SUBJECT, perf_b_id)
        if grant_b is None:
            failures.append(
                "idle_reclaim: perf-b holds no live grant before the idle phase"
            )
            grant_b = any_grant_for(pre_idle, perf_b_id) or {}
        idle_started_wall = int(time.time())
        result["ledger"]["idle_phase"] = {
            "grant_id": grant_b.get("id"),
            "state_before": grant_state_label(grant_b.get("state")),
            "bytes": grant_b.get("bytes"),
            "last_interval_end_before": grant_b.get("last_interval_end_unix_secs"),
            "created_unix_secs": grant_b.get("created_unix_secs"),
            "idle_started_unix_secs": idle_started_wall,
            "idle_reclaim_secs": args.idle_reclaim_secs,
        }

        # Keep sc6-perf-a's grant hot (read hits reset its idle clock) while
        # the idle subject's collection receives NO traffic at all.
        # Accepted-as-is residual (round-2 review): a keepalive served
        # entirely from DRAM L1 does not tick the persistent-store clock, so
        # the grant can still age out — loud-not-green, no action.
        keepalive: list[QueryOutcome] = []
        deadline = time.monotonic() + idle_wait
        reclaimed_at: dict | None = None
        while time.monotonic() < deadline:
            keepalive.append(
                search_once(
                    server,
                    "sc6-perf-a",
                    collection_ids["sc6-perf-a"]["perf-a"],
                    dataset["queries"][0],
                    args.top_k,
                )
            )
            grant_now = any_grant_for(read_ledger(pool_dir), perf_b_id)
            if (
                grant_now is not None
                and grant_state_label(grant_now.get("state")) == "Released"
            ):
                reclaimed_at = grant_now
                break
            time.sleep(max(POLL_SECONDS, args.metering_interval_secs / 2))
        record_outcomes("sc6-perf-a", "perf-a", "idle_reclaim_keepalive", keepalive)

        if reclaimed_at is None:
            failures.append(
                f"idle_reclaim: perf-b grant not Released within {idle_wait}s "
                f"(idle_reclaim_secs={args.idle_reclaim_secs}, sweep cadence "
                f"{args.metering_interval_secs}s)"
            )
        else:
            status_b = collection_status(server, PERF_TENANT_IDLE_SUBJECT, perf_b_id)
            tail = int(reclaimed_at.get("last_interval_end_unix_secs") or 0)
            result["ledger"]["idle_phase"].update(
                {
                    "released": True,
                    "status_after_reclaim": pin_status(status_b),
                    "released_last_interval_end": tail,
                    "released_observed_unix_secs": int(time.time()),
                }
            )
            if status_state(status_b) == "READY":
                failures.append(
                    "idle_reclaim: status still READY after the grant was reclaimed"
                )
            if tail < idle_started_wall:
                failures.append(
                    "idle_reclaim: accrual tail closed BEFORE the idle phase "
                    f"(last_interval_end={tail} < idle_start={idle_started_wall}) — "
                    "accrual does not match the ledger"
                )
            # Re-admission is NOT automatic: wait past another sweep window and
            # confirm no new grant was minted for the collection.
            time.sleep(2 * args.metering_interval_secs + POLL_SECONDS)
            resurrected = live_grant_for(
                read_ledger(pool_dir), PERF_TENANT_IDLE_SUBJECT, perf_b_id
            )
            result["ledger"]["idle_phase"]["re_admitted_after_extra_window"] = (
                resurrected is not None
            )
            if resurrected is not None:
                failures.append(
                    "idle_reclaim: grant auto re-admitted without a storage-class "
                    "update or recreate (must follow the normal admission path)"
                )
            # Durable accrual evidence. On a file:// bed a grant that lived
            # several metering ticks MUST have written usage records — zero
            # records there means the accrual evidence path itself is broken
            # (e.g. the writer is disabled or the path moved), which would
            # otherwise gate as a silent null every time.
            storage_root = (
                Path(storage_url.removeprefix("file://"))
                if storage_url.startswith("file://")
                else None
            )
            accrual = contiguous_accrual(
                usage_records(storage_root, POOL_ID), str(grant_b.get("id"))
            )
            result["ledger"]["idle_phase"]["accrual_records"] = accrual
            if accrual["contiguous"] is False:
                failures.append(
                    f"idle_reclaim: accrual intervals not contiguous ({accrual})"
                )
            if storage_root is not None and accrual["records"] == 0:
                created = int(grant_b.get("created_unix_secs") or 0)
                lifetime_ticks = (
                    (tail - created) / args.metering_interval_secs
                    if created and tail > created
                    else 0
                )
                if lifetime_ticks >= 3:
                    failures.append(
                        "idle_reclaim: no durable usage records for grant "
                        f"{grant_b.get('id')} on a file:// bed despite living "
                        f"~{lifetime_ticks:.0f} metering ticks — the accrual "
                        "evidence path is broken (expected records under "
                        f"{storage_root}/data/*/_metering/reservations/{POOL_ID}/)"
                    )

        # ── Phase: drop_release ─────────────────────────────────────────────
        pre_drop = snapshot_ledger("before drop phase")

        # Balanced drop: never touches the ledger (no grant exists).
        bal_c_id = collection_ids["sc6-bal"]["bal-c"]
        if live_grant_for(pre_drop, "sc6-bal", bal_c_id) is not None:
            failures.append(
                "drop_release: balanced collection unexpectedly holds a grant"
            )
        delete_collection(server, "sc6-bal", bal_c_id)
        after_bal_drop = snapshot_ledger("after balanced drop")
        if after_bal_drop["charged_effective"] != pre_drop["charged_effective"]:
            failures.append(
                "drop_release: balanced drop changed the ledger: "
                f"{pre_drop['charged_effective']} -> {after_bal_drop['charged_effective']}"
            )

        # Granted drop: releases exactly its effective bytes (release runs in a
        # detached task — poll for the terminal state).
        perf_a_id = collection_ids["sc6-perf-a"]["perf-a"]
        delete_collection(server, "sc6-perf-a", perf_a_id)
        deadline = time.monotonic() + 60
        released_a: dict | None = None
        while time.monotonic() < deadline:
            grant_now = any_grant_for(read_ledger(pool_dir), perf_a_id)
            if (
                grant_now is not None
                and grant_state_label(grant_now.get("state")) == "Released"
            ):
                released_a = grant_now
                break
            time.sleep(POLL_SECONDS)
        after_perf_drop = snapshot_ledger("after granted drop")
        result["ledger"]["drop_phase"] = {
            "charged_before": pre_drop["charged_effective"],
            "charged_after_balanced_drop": after_bal_drop["charged_effective"],
            "charged_after_granted_drop": after_perf_drop["charged_effective"],
            "granted_drop_released": released_a is not None,
        }
        if released_a is None:
            failures.append(
                "drop_release: granted drop never Released its grant within 60s"
            )
        else:
            expected_after = pre_drop["charged_effective"] - expected_effective
            if after_perf_drop["charged_effective"] != expected_after:
                failures.append(
                    "drop_release: granted drop leaked reserved bytes: "
                    f"{after_perf_drop['charged_effective']} != {expected_after} "
                    "(charged_before - one effective grant)"
                )

        # Recreate arm (the runbook's "(or recreate)" re-admission path; a
        # REST storage-class UPDATE surface does not exist today).
        recreated_id = create_collection(
            server, "sc6-perf-a", "perf-a2", dimension, "performance", POOL_ID
        )
        deadline = time.monotonic() + 30
        re_admitted: dict | None = None
        while time.monotonic() < deadline:
            re_admitted = live_grant_for(
                read_ledger(pool_dir), "sc6-perf-a", recreated_id
            )
            if re_admitted is not None:
                break
            time.sleep(POLL_SECONDS)
        after_recreate = snapshot_ledger("after recreate")
        result["ledger"]["recreate_arm"] = {
            "collection_id": recreated_id,
            "re_admitted": re_admitted is not None,
            "charged_after": after_recreate["charged_effective"],
            "expected_after": pre_drop["charged_effective"],
        }
        if re_admitted is None:
            failures.append(
                "drop_release: recreate did not re-admit a grant (normal path)"
            )
        elif after_recreate["charged_effective"] != pre_drop["charged_effective"]:
            failures.append(
                "drop_release: recreate charge mismatch: "
                f"{after_recreate['charged_effective']} != "
                f"{pre_drop['charged_effective']} (double charge or leak)"
            )
        delete_collection(server, "sc6-perf-a", recreated_id)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            grant_now = any_grant_for(read_ledger(pool_dir), recreated_id)
            if (
                grant_now is not None
                and grant_state_label(grant_now.get("state")) == "Released"
            ):
                break
            time.sleep(POLL_SECONDS)
        final_ledger = snapshot_ledger("final")
        result["ledger"]["final"] = final_ledger
        # Every grant created in this bed is terminally Released by now — any
        # residue is leaked reserved bytes.
        if final_ledger["charged_effective"] != 0:
            failures.append(
                "drop_release: leaked reserved bytes at the end: "
                f"{final_ledger['charged_effective']} still charging "
                f"({final_ledger['charging']!r})"
            )

        # ── Per-tenant gauge roll-up ────────────────────────────────────────
        for (tenant, name, phase), outcomes in sorted(outcome_log.items()):
            result["tenants"].setdefault(tenant, {}).setdefault(name, {})[phase] = (
                summarize_outcomes(outcomes)
            )

        result["status"] = "pass" if not failures else "fail"
        if failures:
            result["gate_failures"] = failures
    except Exception as error:
        result["status"] = "fail"
        result["error"] = str(error)
        raise
    finally:
        if active is not None:
            active.stop()
        output = root / "result.json"
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"result: {output}", flush=True)
    if failures:
        raise BedError("; ".join(failures))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
