#!/usr/bin/env python3
"""Compare fixed and adaptive PAX reads across object, DRAM, and disk tiers.

Fixed and adaptive policies run adjacently at the same requested concurrency,
and every measured phase searches the same query slice.  DRAM is warmed with a
disjoint slice so a process-local query-result cache cannot impersonate a PAX
range-cache hit.  Each policy owns a separate persistent-cache directory.  Its
disk tier is populated in measured-then-warmup order: the second, disjoint
slice exerts eviction pressure on the first slice, spilling its ranges before
the server restarts with empty DRAM.

Azurite's append-only debug log is reconciled with application counters for
each measured phase.  It proves HTTP request shape, not production Azure
latency.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
MIB = 1024 * 1024
AZURE_HOT_READ_USD_PER_10K = 0.005
MEASURED_PHASES = ("object_cold", "dram_warm", "disk_warm")
CACHE_PHASES = ("dram_warm", "disk_warm")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load benchmark module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ACCEPTANCE = load_module(
    "cache_tier_acceptance", SCRIPT_ROOT / "sift1m_get_reduction.py"
)
NPROBE = load_module("cache_tier_nprobe", SCRIPT_ROOT / "nprobe_sweep.py")
RANGE = load_module("cache_tier_range", SCRIPT_ROOT / "range_cap_sweep.py")


def query_slices(
    query_start: int,
    warmup_queries: int,
    measured_queries: int,
    available_queries: int,
    available_truth: int,
) -> dict[str, list[int]]:
    if query_start < 0:
        raise RuntimeError("query start must be non-negative")
    if warmup_queries <= 0 or measured_queries <= 0:
        raise RuntimeError("warmup and measured query counts must be positive")
    warmup_end = query_start + warmup_queries
    measured_end = warmup_end + measured_queries
    if measured_end > available_queries or measured_end > available_truth:
        raise RuntimeError("warmup plus measured query slice exceeds available rows")
    return {
        "warmup": [query_start, warmup_end],
        "measured": [warmup_end, measured_end],
    }


def disk_population_order() -> tuple[str, str]:
    """Order that makes measured ranges eligible for L1-to-disk spill."""
    return ("measured", "warmup")


def range_policies(include_adaptive: bool) -> tuple[str, ...]:
    return ("fixed", "adaptive") if include_adaptive else ("fixed",)


def disk_path_for_attempt(run_root: Path, policy: str, attempt: int) -> Path:
    if attempt < 0:
        raise RuntimeError("attempt must be non-negative")
    if policy not in range_policies(True):
        raise RuntimeError(f"unsupported range policy: {policy}")
    return run_root / f"local-disk-cache-{policy}-attempt-{attempt}"


def remove_discarded_cache_paths(run_root: Path, paths: dict[str, Path]) -> None:
    """Remove only policy cache directories owned by a rejected attempt."""
    resolved_root = run_root.resolve()
    for policy, path in paths.items():
        expected_prefix = f"local-disk-cache-{policy}-attempt-"
        resolved = path.resolve()
        if (
            policy not in range_policies(True)
            or resolved.parent != resolved_root
            or not resolved.name.startswith(expected_prefix)
        ):
            raise RuntimeError(f"refusing to remove unowned cache path: {path}")
        if resolved.exists():
            shutil.rmtree(resolved)


def hit_ratio(hits: float, misses: float) -> float | None:
    total = hits + misses
    return hits / total if total else None


def add_cache_ratios(point: dict) -> dict:
    for tier in ("survivor", "invariants", "local_disk"):
        counters = point[tier]
        counters["hit_ratio"] = hit_ratio(counters["hits"], counters["misses"])
    return point


def ratio_with_zero_identity(candidate: float, baseline: float) -> float | None:
    """Return a ratio where equal zero means no change, not missing evidence."""
    if baseline:
        return candidate / baseline
    return 1.0 if candidate == 0 else None


def reduction_with_zero_identity(candidate: float, baseline: float) -> float | None:
    """Return relative reduction where equal zero means zero reduction."""
    ratio = ratio_with_zero_identity(candidate, baseline)
    return 1.0 - ratio if ratio is not None else None


def wire_ranges_reconciled(point: dict, tolerance: float = 0.01) -> bool:
    """Reconcile application reads with ranged wire GETs, including zero I/O."""
    application_gets = point["physical_gets"]
    range_gets = point["wire_http"].get("range_get_requests")
    if range_gets is None:
        ratio = point.get("wire_range_to_application_get_ratio")
        return ratio is not None and abs(ratio - 1.0) <= tolerance
    if application_gets == 0:
        return range_gets == 0
    return abs(range_gets / application_gets - 1.0) <= tolerance


def compare_points(candidate: dict, baseline: dict, query_count: int) -> dict:
    if query_count <= 0:
        raise RuntimeError("query count must be positive")
    candidate_gets = candidate["physical_gets"]
    baseline_gets = baseline["physical_gets"]
    candidate_wire_gets = candidate["wire_http"]["get_requests"]
    baseline_wire_gets = baseline["wire_http"]["get_requests"]
    candidate_bytes = candidate["bytes_read"]
    baseline_bytes = baseline["bytes_read"]
    candidate_wire_gets_per_query = candidate_wire_gets / query_count
    candidate_peak = candidate["process_rss"]["peak_bytes"]
    baseline_peak = baseline["process_rss"]["peak_bytes"]
    candidate_qps = candidate["load"]["qps"]
    baseline_qps = baseline["load"]["qps"]
    return {
        "application_get_reduction": reduction_with_zero_identity(
            candidate_gets, baseline_gets
        ),
        "wire_get_reduction": reduction_with_zero_identity(
            candidate_wire_gets, baseline_wire_gets
        ),
        "bytes_ratio": ratio_with_zero_identity(candidate_bytes, baseline_bytes),
        "p50_ratio": (candidate["latency_ms"]["p50"] / baseline["latency_ms"]["p50"]),
        "p95_ratio": (candidate["latency_ms"]["p95"] / baseline["latency_ms"]["p95"]),
        "rss_ratio": (
            candidate_peak / baseline_peak
            if candidate_peak is not None and baseline_peak
            else None
        ),
        "qps_ratio": candidate_qps / baseline_qps if baseline_qps else None,
        "recall_delta": candidate["recall_at_k"] - baseline["recall_at_k"],
        "result_identity_diagnostics": RANGE.result_identity_diagnostics(
            candidate, baseline
        ),
        "azure_hot_read_cogs_per_million_queries_usd": (
            candidate_wire_gets_per_query
            * 1_000_000
            / 10_000
            * AZURE_HOT_READ_USD_PER_10K
        ),
    }


def evaluate_promotion(
    policy_results: dict,
    *,
    query_count: int,
    concurrency: int,
    target_recall: float,
    max_recall_regression: float,
    min_disk_get_reduction: float,
    min_adaptive_cold_get_reduction: float,
    max_adaptive_warm_get_ratio: float,
    max_byte_amplification: float,
    max_latency_ratio: float,
    max_rss_ratio: float,
    min_qps_ratio: float,
) -> dict:
    """Evaluate cache benefit within each policy and adaptive safety by tier."""
    checks = {}
    cache_comparisons = {}
    for policy, policy_result in policy_results.items():
        phases = policy_result["phases"]
        cache_comparisons[policy] = {
            phase: compare_points(phases[phase], phases["object_cold"], query_count)
            for phase in CACHE_PHASES
        }
        for phase in MEASURED_PHASES:
            point = phases[phase]
            checks[f"{policy}.{phase}.target_recall"] = (
                point["recall_at_k"] >= target_recall
            )
            checks[f"{policy}.{phase}.achieved_concurrency"] = (
                point["load"]["peak_in_flight"] == concurrency
            )
            checks[f"{policy}.{phase}.wire_range_reconciled"] = wire_ranges_reconciled(
                point
            )
        disk = phases["disk_warm"]
        checks[f"{policy}.disk_warm.local_disk_hit"] = disk["local_disk"]["hits"] > 0
        disk_get_reduction = cache_comparisons[policy]["disk_warm"][
            "wire_get_reduction"
        ]
        checks[f"{policy}.disk_warm.get_reduction"] = (
            disk_get_reduction is not None
            and disk_get_reduction >= min_disk_get_reduction
        )

    paired_comparisons = {}
    if "adaptive" in policy_results:
        fixed = policy_results["fixed"]["phases"]
        adaptive = policy_results["adaptive"]["phases"]
        for phase in MEASURED_PHASES:
            comparison = compare_points(adaptive[phase], fixed[phase], query_count)
            paired_comparisons[phase] = comparison
            checks[f"paired.{phase}.recall_noninferior"] = (
                comparison["recall_delta"] >= -max_recall_regression
            )
            checks[f"paired.{phase}.bytes_bounded"] = (
                comparison["bytes_ratio"] is not None
                and comparison["bytes_ratio"] <= max_byte_amplification
            )
            checks[f"paired.{phase}.p50_bounded"] = (
                comparison["p50_ratio"] <= max_latency_ratio
            )
            checks[f"paired.{phase}.p95_bounded"] = (
                comparison["p95_ratio"] <= max_latency_ratio
            )
            checks[f"paired.{phase}.rss_bounded"] = (
                comparison["rss_ratio"] is not None
                and comparison["rss_ratio"] <= max_rss_ratio
            )
            checks[f"paired.{phase}.qps_bounded"] = (
                comparison["qps_ratio"] is not None
                and comparison["qps_ratio"] >= min_qps_ratio
            )

        cold_reduction = paired_comparisons["object_cold"]["wire_get_reduction"]
        checks["paired.object_cold.get_reduction"] = (
            cold_reduction is not None
            and cold_reduction >= min_adaptive_cold_get_reduction
        )
        for phase in CACHE_PHASES:
            warm_reduction = paired_comparisons[phase]["wire_get_reduction"]
            checks[f"paired.{phase}.get_not_regressed"] = (
                warm_reduction is not None
                and 1.0 - warm_reduction <= max_adaptive_warm_get_ratio
            )

    failures = [name for name, passed in checks.items() if not passed]
    return {
        "cache_comparisons": cache_comparisons,
        "paired_comparisons": paired_comparisons,
        "checks": checks,
        "gate_failures": failures,
        "measurement_valid": not failures,
        "promotion_eligible": not failures and "adaptive" in policy_results,
    }


def require_new_run(output: Path, run_root: Path) -> None:
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    if run_root.exists() and any(run_root.iterdir()):
        raise RuntimeError(f"run root must be empty: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)


def write_result(output: Path, result: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def run_measured_phase(
    *,
    label: str,
    server,
    server_url: str,
    collection_id: str,
    query_path: Path,
    groundtruth_path: Path,
    query_format: str,
    groundtruth_format: str,
    query_start: int,
    query_count: int,
    top_k: int,
    concurrency: int,
    wire_log,
) -> dict:
    if server.process is None:
        raise RuntimeError("owned server did not expose its process")
    wire_offset = wire_log.snapshot()
    contention = RANGE.HostContentionMonitor()
    contention.start()
    sampler = RANGE.RssSampler(server.process.pid)
    sampler.start()
    try:
        point = ACCEPTANCE.run_query_sweep(
            server_url,
            collection_id,
            query_path,
            groundtruth_path,
            query_start,
            query_count,
            top_k,
            label,
            query_format,
            groundtruth_format,
            contention.raise_if_conflict,
            concurrency=concurrency,
        )
    finally:
        rss = sampler.stop()
        contention_observation = contention.stop()
    contention.raise_if_conflict()
    point["process_rss"] = rss
    point["host_contention"] = contention_observation
    point["wire_http"] = wire_log.sample(wire_offset)
    RANGE.validate_wire_observation(label, point)
    point["wire_gets_per_query"] = point["wire_http"]["get_requests"] / query_count
    point["wire_range_to_application_get_ratio"] = (
        point["wire_http"]["range_get_requests"] / point["physical_gets"]
        if point["physical_gets"]
        else None
    )
    return add_cache_ratios(point)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-source-revision", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--storage-url", required=True)
    parser.add_argument("--azurite-debug-log", type=Path, required=True)
    parser.add_argument("--base-path", type=Path, required=True)
    parser.add_argument("--base-format", choices=("fvecs", "u8bin"), default="fvecs")
    parser.add_argument("--query-path", type=Path, required=True)
    parser.add_argument("--query-format", choices=("fvecs", "u8bin"), default="fvecs")
    parser.add_argument("--groundtruth-path", type=Path, required=True)
    parser.add_argument(
        "--groundtruth-format", choices=("ivecs", "bigann-bin"), default="ivecs"
    )
    parser.add_argument("--groundtruth-scope-rows", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--nprobe", type=int, required=True)
    parser.add_argument("--range-cap-mib", type=int, required=True)
    parser.add_argument(
        "--include-adaptive",
        action="store_true",
        help="pair the fixed cap with the default-OFF exact-plan chooser",
    )
    parser.add_argument("--coalesce-gap-mib", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--warmup-queries", type=int, default=500)
    parser.add_argument("--measured-queries", type=int, default=500)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--port", type=int, default=5790)
    parser.add_argument("--max-segments", type=int, default=1)
    parser.add_argument("--required-layout-version", type=int, default=3)
    parser.add_argument("--target-recall", type=float, default=0.975)
    parser.add_argument("--max-recall-regression", type=float, default=0.0005)
    parser.add_argument("--min-disk-get-reduction", type=float, default=0.20)
    parser.add_argument("--min-adaptive-cold-get-reduction", type=float, default=0.10)
    parser.add_argument("--max-adaptive-warm-get-ratio", type=float, default=1.05)
    parser.add_argument("--max-byte-amplification", type=float, default=1.25)
    parser.add_argument("--max-latency-ratio", type=float, default=1.10)
    parser.add_argument("--max-rss-ratio", type=float, default=1.10)
    parser.add_argument("--min-qps-ratio", type=float, default=0.95)
    parser.add_argument("--host-quiet-window-secs", type=float, default=0)
    parser.add_argument("--host-quiet-timeout-secs", type=float, default=3600)
    parser.add_argument("--max-contention-retries", type=int, default=0)
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[2]
    binary_source = args.binary.resolve()
    config = args.config.resolve()
    run_root = args.run_root.resolve()
    output = args.output.resolve()
    require_new_run(output, run_root)
    _matrix_lock = NPROBE.acquire_matrix_lock(output)
    if not args.collection_id.isdecimal():
        raise RuntimeError("--collection-id must be a decimal catalog object id")
    if args.range_cap_mib <= 0 or args.nprobe <= 0 or args.concurrency <= 0:
        raise RuntimeError("range cap, nprobe, and concurrency must be positive")
    if args.coalesce_gap_mib < 0:
        raise RuntimeError("coalescing gap must be non-negative")
    if args.max_contention_retries < 0:
        raise RuntimeError("contention retries must be non-negative")
    if args.groundtruth_scope_rows != args.rows:
        raise RuntimeError("ground-truth scope must equal measured corpus rows")
    NPROBE.require_config_port(config, args.port)
    current_revision, profile = NPROBE.require_release_provenance(
        repository, binary_source, args.binary_source_revision
    )
    binary = RANGE.snapshot_binary(binary_source, run_root)

    base_path = args.base_path.resolve()
    query_path = args.query_path.resolve()
    groundtruth_path = args.groundtruth_path.resolve()
    base_count, dimension, base_declared = ACCEPTANCE.vector_source_geometry(
        base_path, args.base_format
    )
    query_count, query_dimension, query_declared = ACCEPTANCE.vector_source_geometry(
        query_path, args.query_format
    )
    truth_count, truth_width = ACCEPTANCE.count_truth_records(
        groundtruth_path, args.groundtruth_format
    )
    if args.rows > base_count or query_dimension != dimension:
        raise RuntimeError("dataset cardinality or dimension does not match the bed")
    if args.top_k > truth_width:
        raise RuntimeError("top-k exceeds ground-truth width")
    slices = query_slices(
        args.query_start,
        args.warmup_queries,
        args.measured_queries,
        query_count,
        truth_count,
    )

    container, prefix = RANGE.azure_storage_scope(args.storage_url)
    wire_log = RANGE.AzuriteWireLog(args.azurite_debug_log.resolve(), container, prefix)
    wire_log.snapshot()
    geometry_reader = ACCEPTANCE.AzureCliPaxGeometry(
        args.storage_url, run_root / "pax-snapshot"
    )
    geometry = geometry_reader.materialize(geometry_reader.inventory())
    NPROBE.validate_geometry(
        geometry, args.rows, args.max_segments, args.required_layout_version
    )
    max_cells = max(segment["coarse_cells"] for segment in geometry["segments"])
    if args.nprobe > max_cells:
        raise RuntimeError(
            f"nprobe={args.nprobe} exceeds persisted max k_c={max_cells}"
        )

    result = {
        "protocol": "pax_paired_cache_tier_sweep",
        "status": "running",
        "git_revision": current_revision,
        "collection_id": args.collection_id,
        "binary": {
            "path": str(binary),
            "sha256": ACCEPTANCE.sha256(binary),
            "bytes": binary.stat().st_size,
            "source_revision": args.binary_source_revision,
            "profile": profile,
        },
        "bed_config": {"path": str(config), "sha256": ACCEPTANCE.sha256(config)},
        "dataset": {
            "base": str(base_path),
            "base_format": args.base_format,
            "available_rows": base_count,
            "base_declared_rows": base_declared,
            "measured_rows": args.rows,
            "dimension": dimension,
            "queries": str(query_path),
            "query_format": args.query_format,
            "query_available_rows": query_count,
            "query_declared_rows": query_declared,
            "groundtruth": str(groundtruth_path),
            "groundtruth_format": args.groundtruth_format,
            "groundtruth_scope_rows": args.groundtruth_scope_rows,
            "groundtruth_width": truth_width,
            "query_slices": slices,
        },
        "settled_geometry": geometry,
        "compute_profile": ACCEPTANCE.compute_profile(),
        "experiment": {
            "fixed_nprobe": args.nprobe,
            "fixed_range_cap_mib": args.range_cap_mib,
            "fixed_coalesce_gap_mib": args.coalesce_gap_mib,
            "range_policies": list(range_policies(args.include_adaptive)),
            "top_k": args.top_k,
            "concurrency": args.concurrency,
            "disk_population_order": list(disk_population_order()),
            "disk_population_reason": (
                "measured ranges are touched before a disjoint slice exerts "
                "eviction pressure; restart then clears DRAM and result cache"
            ),
            # OwnedServer deliberately fixes the isolated benchmark tier at
            # 10 GiB.  Report the actual control instead of exposing a CLI
            # knob that the shared launcher cannot honor.
            "local_disk_max_gb": 10,
            "azure_hot_read_usd_per_10k": AZURE_HOT_READ_USD_PER_10K,
            "host_quiet_window_seconds": args.host_quiet_window_secs,
            "host_quiet_timeout_seconds": args.host_quiet_timeout_secs,
            "max_contention_retries": args.max_contention_retries,
            "decision_thresholds": {
                "target_recall": args.target_recall,
                "max_recall_regression": args.max_recall_regression,
                "min_disk_get_reduction": args.min_disk_get_reduction,
                "min_adaptive_cold_get_reduction": (
                    args.min_adaptive_cold_get_reduction
                ),
                "max_adaptive_warm_get_ratio": args.max_adaptive_warm_get_ratio,
                "max_byte_amplification": args.max_byte_amplification,
                "max_latency_ratio": args.max_latency_ratio,
                "max_rss_ratio": args.max_rss_ratio,
                "min_qps_ratio": args.min_qps_ratio,
            },
        },
        "policy_results": {},
        "cache_comparisons": {},
        "paired_comparisons": {},
        "checks": {},
        "gate_failures": [],
        "rejected_attempts": [],
    }
    write_result(output, result)

    server_url = f"http://127.0.0.1:{args.port}"

    def new_server(label: str, disk_path: Path | None, policy: str):
        if policy not in range_policies(True):
            raise RuntimeError(f"unsupported range policy: {policy}")
        return ACCEPTANCE.OwnedServer(
            binary=binary,
            config=config,
            server=server_url,
            log_path=run_root / f"{label}.log",
            local_disk_path=disk_path,
            nprobe=args.nprobe,
            azure_emulator=True,
            coalesce_gap_bytes=args.coalesce_gap_mib * MIB,
            coalesce_range_bytes=args.range_cap_mib * MIB,
            adaptive_read_strategy=policy == "adaptive",
        )

    def query(label: str, server, bounds: list[int]) -> dict:
        return run_measured_phase(
            label=label,
            server=server,
            server_url=server_url,
            collection_id=args.collection_id,
            query_path=query_path,
            groundtruth_path=groundtruth_path,
            query_format=args.query_format,
            groundtruth_format=args.groundtruth_format,
            query_start=bounds[0],
            query_count=bounds[1] - bounds[0],
            top_k=args.top_k,
            concurrency=args.concurrency,
            wire_log=wire_log,
        )

    policies = range_policies(args.include_adaptive)
    active = None
    try:
        attempt = 0
        while True:
            attempt_results = {policy: {"phases": {}} for policy in policies}
            local_disks = {
                policy: disk_path_for_attempt(run_root, policy, attempt)
                for policy in policies
            }
            try:
                for policy in policies:
                    RANGE.wait_for_host_quiet(
                        args.host_quiet_window_secs,
                        args.host_quiet_timeout_secs,
                    )
                    active = new_server(
                        f"{policy}-object-cold-attempt-{attempt}", None, policy
                    )
                    active.start()
                    attempt_results[policy]["phases"]["object_cold"] = query(
                        f"{policy}.object_cold", active, slices["measured"]
                    )
                    active.stop()
                    active = None

                for policy in policies:
                    RANGE.wait_for_host_quiet(
                        args.host_quiet_window_secs,
                        args.host_quiet_timeout_secs,
                    )
                    active = new_server(
                        f"{policy}-dram-warm-attempt-{attempt}", None, policy
                    )
                    active.start()
                    attempt_results[policy]["phases"]["dram_warmup"] = query(
                        f"{policy}.dram_warmup", active, slices["warmup"]
                    )
                    attempt_results[policy]["phases"]["dram_warm"] = query(
                        f"{policy}.dram_warm", active, slices["measured"]
                    )
                    active.stop()
                    active = None

                for policy in policies:
                    local_disk = local_disks[policy]
                    RANGE.wait_for_host_quiet(
                        args.host_quiet_window_secs,
                        args.host_quiet_timeout_secs,
                    )
                    active = new_server(
                        f"{policy}-disk-populate-attempt-{attempt}",
                        local_disk,
                        policy,
                    )
                    active.start()
                    for slice_name in disk_population_order():
                        phase = f"disk_population_{slice_name}"
                        attempt_results[policy]["phases"][phase] = query(
                            f"{policy}.{phase}", active, slices[slice_name]
                        )
                    active.stop()
                    active = None

                    RANGE.wait_for_host_quiet(
                        args.host_quiet_window_secs,
                        args.host_quiet_timeout_secs,
                    )
                    active = new_server(
                        f"{policy}-disk-warm-attempt-{attempt}",
                        local_disk,
                        policy,
                    )
                    active.start()
                    attempt_results[policy]["phases"]["disk_warm"] = query(
                        f"{policy}.disk_warm", active, slices["measured"]
                    )
                    active.stop()
                    active = None
            except (Exception, KeyboardInterrupt) as error:
                if active is not None:
                    active.stop()
                    active = None
                if RANGE.is_host_contention_error(error) and (
                    attempt < args.max_contention_retries
                ):
                    result["rejected_attempts"].append(
                        {
                            "attempt": attempt,
                            "reason": f"{type(error).__name__}: {error}",
                            "discarded_phases": {
                                policy: sorted(policy_result["phases"])
                                for policy, policy_result in attempt_results.items()
                            },
                            "discarded_local_disk_paths": {
                                policy: str(path)
                                for policy, path in local_disks.items()
                            },
                        }
                    )
                    remove_discarded_cache_paths(run_root, local_disks)
                    write_result(output, result)
                    attempt += 1
                    continue
                raise
            result["policy_results"] = attempt_results
            result["experiment"]["accepted_attempt"] = attempt
            result["experiment"]["accepted_local_disk_paths"] = {
                policy: str(path) for policy, path in local_disks.items()
            }
            write_result(output, result)
            break

        evaluation = evaluate_promotion(
            result["policy_results"],
            query_count=args.measured_queries,
            concurrency=args.concurrency,
            target_recall=args.target_recall,
            max_recall_regression=args.max_recall_regression,
            min_disk_get_reduction=args.min_disk_get_reduction,
            min_adaptive_cold_get_reduction=args.min_adaptive_cold_get_reduction,
            max_adaptive_warm_get_ratio=args.max_adaptive_warm_get_ratio,
            max_byte_amplification=args.max_byte_amplification,
            max_latency_ratio=args.max_latency_ratio,
            max_rss_ratio=args.max_rss_ratio,
            min_qps_ratio=args.min_qps_ratio,
        )
        result.update(evaluation)
        result["status"] = "pass" if evaluation["measurement_valid"] else "fail"
    except (Exception, KeyboardInterrupt) as error:
        result["status"] = "incomplete"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if active is not None:
            active.stop()
        write_result(output, result)

    print(f"cache-tier result: {output}", flush=True)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
