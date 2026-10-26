"""Focused contracts for the auditable SIFT GET-reduction harness."""

from __future__ import annotations

import importlib.util
import json
import struct
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = REPOSITORY_ROOT / "scripts" / "bench" / "sift1m_get_reduction.py"
SPEC = importlib.util.spec_from_file_location(
    "sift1m_get_reduction_harness", HARNESS_PATH
)
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)
MATRIX_PATH = REPOSITORY_ROOT / "scripts" / "bench" / "nprobe_sweep.py"
MATRIX_SPEC = importlib.util.spec_from_file_location("sift_nprobe_matrix", MATRIX_PATH)
assert MATRIX_SPEC is not None and MATRIX_SPEC.loader is not None
MATRIX = importlib.util.module_from_spec(MATRIX_SPEC)
MATRIX_SPEC.loader.exec_module(MATRIX)


def test_matrix_contract_rejects_duplicate_probes_and_wrong_geometry() -> None:
    assert MATRIX.comma_separated_ints("1,2,4", "--nprobes") == [1, 2, 4]
    with pytest.raises(RuntimeError, match="duplicates"):
        MATRIX.comma_separated_ints("1,2,1", "--nprobes")
    with pytest.raises(RuntimeError, match="rows"):
        MATRIX.validate_geometry(
            {
                "row_count": 99,
                "segment_count": 1,
                "segments": [{"layout_version": 3, "coarse_cells": 4}],
            },
            rows=100,
            max_segments=1,
            layout_version=3,
        )


def test_matrix_rejects_port_different_from_immutable_bed(tmp_path: Path) -> None:
    config = tmp_path / "benchmark.toml"
    config.write_text(
        "[server]\nport = 1111\n\n[api]\nunified_port = 5790 # measured port\n"
    )

    MATRIX.require_config_port(config, 5790)
    with pytest.raises(RuntimeError, match="does not match"):
        MATRIX.require_config_port(config, 5800)


def test_object_cold_ivf_requires_physical_region_byte_attribution() -> None:
    result = {
        "physical_gets": 12,
        "ivf": {
            "cells_probed": 20,
            "region_a_bytes": 0,
            "region_b_bytes": 0,
        },
    }
    assert HARNESS.ivf_byte_attribution_failure("object_cold", result) == (
        "object_cold: IVF probe issued physical GETs but attributed zero "
        "Region-A/B bytes"
    )
    result["ivf"]["region_b_bytes"] = 4096
    assert HARNESS.ivf_byte_attribution_failure("object_cold", result) is None


def test_azure_geometry_uses_canonical_az_scheme(tmp_path: Path) -> None:
    geometry = HARNESS.AzureCliPaxGeometry(
        "az://benchmark-container/five-point/100k",
        tmp_path,
    )

    assert geometry.container == "benchmark-container"
    assert geometry.prefix == "five-point/100k"


def test_compute_profile_names_the_actual_pax_distance_dispatch() -> None:
    arm = HARNESS.compute_profile("arm64")
    x86 = HARNESS.compute_profile("x86_64")

    assert arm["region_b_sq8_l2_kernel"] == "neon_fused_decode_distance"
    assert arm["dispatch"] == "compile_time_aarch64"
    assert x86["region_b_sq8_l2_kernel"] == "avx2_or_scalar_fused_decode_distance"
    assert x86["dispatch"] == "runtime_feature_detection"
    assert arm["gpu_role"] == "not_used_by_pax_rabitq_sq8_search"


def test_prefix_quality_checkpoints_reuse_one_query_execution() -> None:
    recalls = [1.0] * 100 + [0.9] * 900 + [0.8] * 9_000
    latencies = [float(value) for value in range(1, 10_001)]

    checkpoints = HARNESS.prefix_quality_checkpoints(recalls, latencies)

    assert [point["query_count"] for point in checkpoints] == [100, 1_000, 10_000]
    assert checkpoints[0]["recall_at_k"] == 1.0
    assert checkpoints[1]["recall_at_k"] == pytest.approx(0.91)
    assert checkpoints[2]["recall_at_k"] == pytest.approx(0.811)
    assert checkpoints[0]["latency_ms"]["p50"] == 50.0
    assert checkpoints[1]["latency_ms"]["p95"] == 950.0


def test_query_result_identity_separates_membership_from_order() -> None:
    first = HARNESS.query_result_identity(["v2", "v1", "v1"])
    reordered = HARNESS.query_result_identity(["v1", "v2", "v1"])
    changed = HARNESS.query_result_identity(["v1", "v3", "v1"])

    assert first["ordered_ids_sha256"] != reordered["ordered_ids_sha256"]
    assert first["set_ids_sha256"] == reordered["set_ids_sha256"]
    assert first["set_ids_sha256"] != changed["set_ids_sha256"]
    assert first["result_count"] == 3
    assert first["unique_result_count"] == 2


def test_query_sweep_checks_progress_guard_before_queries_and_after_measurement() -> None:
    guard_calls = []
    metrics = dict.fromkeys(HARNESS.METRICS, 0.0)
    with (
        patch.object(HARNESS, "read_vectors", return_value=[[0.0, 1.0]]),
        patch.object(HARNESS, "read_truth_ids", return_value=[[1]]),
        patch.object(HARNESS, "scrape", side_effect=[metrics, metrics]),
        patch.object(
            HARNESS,
            "request_json",
            return_value={"results": [{"id": "v1"}]},
        ),
    ):
        result = HARNESS.run_query_sweep(
            "http://127.0.0.1:5678",
            "1",
            Path("queries.u8bin"),
            Path("truth.ivecs"),
            0,
            1,
            1,
            "guarded",
            "u8bin",
            "ivecs",
            lambda: guard_calls.append("checked"),
        )

    assert result["recall_at_k"] == 1.0
    assert guard_calls == ["checked", "checked"]


def test_query_sweep_runs_bounded_concurrency_and_preserves_query_order() -> None:
    metrics = dict.fromkeys(HARNESS.METRICS, 0.0)
    lock = threading.Lock()
    first_pair = threading.Barrier(2, timeout=1)
    request_count = 0

    def request(_url, *, method, body, timeout):
        nonlocal request_count
        assert method == "POST"
        assert timeout == 300
        query_id = int(body["vector"][0])
        with lock:
            request_count += 1
            ordinal = request_count
        if ordinal <= 2:
            first_pair.wait()
        time.sleep(0.01)
        return {"results": [{"id": f"v{query_id}"}]}

    with (
        patch.object(
            HARNESS,
            "read_vectors",
            return_value=[[float(index)] for index in range(4)],
        ),
        patch.object(HARNESS, "read_truth_ids", return_value=[[i] for i in range(4)]),
        patch.object(HARNESS, "scrape", side_effect=[metrics, metrics]),
        patch.object(HARNESS, "request_json", side_effect=request),
    ):
        result = HARNESS.run_query_sweep(
            "http://127.0.0.1:5678",
            "1",
            Path("queries.u8bin"),
            Path("truth.ivecs"),
            0,
            4,
            1,
            "concurrent",
            "u8bin",
            "ivecs",
            concurrency=2,
        )

    expected = [HARNESS.query_result_identity([f"v{i}"]) for i in range(4)]
    assert result["recall_at_k"] == 1.0
    assert result["result_identity"]["ordered_ids_sha256_by_query"] == [
        item["ordered_ids_sha256"] for item in expected
    ]
    assert result["load"] == {
        "model": "closed_loop_bounded",
        "configured_concurrency": 2,
        "peak_in_flight": 2,
        "wall_seconds": pytest.approx(result["load"]["wall_seconds"]),
        "qps": pytest.approx(4 / result["load"]["wall_seconds"]),
    }


def test_query_sweep_rejects_nonpositive_concurrency() -> None:
    with pytest.raises(RuntimeError, match="concurrency must be positive"):
        HARNESS.run_query_sweep(
            "http://127.0.0.1:5678",
            "1",
            Path("queries.u8bin"),
            Path("truth.ivecs"),
            0,
            1,
            1,
            "invalid",
            concurrency=0,
        )


def test_u8bin_prefix_uses_physical_rows_and_preserves_declared_rows(
    tmp_path: Path,
) -> None:
    pa = pytest.importorskip("pyarrow")
    path = tmp_path / "base-prefix.u8bin"
    path.write_bytes(
        struct.pack("<II", 1_000_000_000, 4)
        + bytes(
            [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
            ]
        )
    )

    physical_rows, dimension, declared_rows = HARNESS.inspect_u8bin(path)

    assert (physical_rows, dimension, declared_rows) == (
        3,
        4,
        1_000_000_000,
    )
    assert HARNESS.read_vectors(path, "u8bin", 1, 2) == [
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
    ]
    assert list(HARNESS.iter_vector_batches(path, "u8bin", 3, 2)) == [
        [
            {"id": "v0", "vector": [1.0, 2.0, 3.0, 4.0]},
            {"id": "v1", "vector": [5.0, 6.0, 7.0, 8.0]},
        ],
        [{"id": "v2", "vector": [9.0, 10.0, 11.0, 12.0]}],
    ]
    arrow_batches = list(
        HARNESS.iter_vector_arrow_batches(path, "u8bin", 3, 2, arrow_module=pa)
    )
    assert [batch.num_rows for batch in arrow_batches] == [2, 1]
    assert arrow_batches[0].column("id").to_pylist() == ["v0", "v1"]
    assert arrow_batches[0].column("vector").to_pylist() == [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
    ]


def test_u8bin_rejects_partial_dense_row(tmp_path: Path) -> None:
    path = tmp_path / "partial.u8bin"
    path.write_bytes(struct.pack("<II", 3, 4) + bytes([1, 2, 3, 4, 5]))

    with pytest.raises(RuntimeError, match="partial dense row"):
        HARNESS.inspect_u8bin(path)


def test_bigann_groundtruth_reads_id_matrix_before_distances(
    tmp_path: Path,
) -> None:
    path = tmp_path / "groundtruth.bin"
    ids = (7, 3, 9, 4, 2, 8)
    distances = (1.0, 2.0, 3.0, 0.5, 1.5, 2.5)
    path.write_bytes(
        struct.pack("<II", 2, 3)
        + struct.pack("<6i", *ids)
        + struct.pack("<6f", *distances)
    )

    assert HARNESS.count_truth_records(path, "bigann-bin") == (2, 3)
    assert HARNESS.read_truth_ids(path, "bigann-bin", 0, 2) == [
        [7, 3, 9],
        [4, 2, 8],
    ]


def test_a0_geometry_reports_cell_shape_and_verifies_checksum() -> None:
    dimension = 2
    components = 1
    rows = [3, 0]
    radii = [1.5, 0.0]
    encoded = bytearray(b"PXA0")
    encoded.extend(bytes([1, 0]))
    encoded.extend(
        struct.pack("<HIIQQQ", components, len(rows), dimension, 7, 5, sum(rows))
    )
    encoded.extend(struct.pack("<2f", 0.0, 0.0))
    encoded.extend(struct.pack("<2f", 1.0, 0.0))
    encoded.extend(struct.pack("<2f", 0.0, 2.0))
    encoded.extend(struct.pack("<2f", *radii))
    row_begin = 0
    for row_count in rows:
        row_end = row_begin + row_count
        encoded.extend(
            struct.pack(
                "<QQQQQQQQII",
                row_begin,
                row_end,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            )
        )
        row_begin = row_end
    encoded.extend(struct.pack("<Q", HARNESS.fnv1a64(encoded)))

    geometry = HARNESS.parse_a0_geometry(bytes(encoded))

    assert geometry["coarse_cells"] == 2
    assert geometry["coarse_trained_rows"] == 5
    assert geometry["cell_rows"] == rows
    assert geometry["empty_cell_fraction"] == 0.5
    assert geometry["cell_row_max_to_mean"] == 2.0
    assert geometry["radii"] == radii

    encoded[-1] ^= 1
    with pytest.raises(RuntimeError, match="checksum"):
        HARNESS.parse_a0_geometry(bytes(encoded))


def test_azure_inventory_scopes_prefix_and_records_stable_identity(
    tmp_path: Path,
) -> None:
    payload = [
        {
            "name": "run-1/collections/7/segment.pax",
            "properties": {
                "contentLength": 1234,
                "etag": "etag-1",
                "lastModified": "2026-07-30T00:00:00Z",
            },
        },
        {
            "name": "run-1/collections/7/manifest.json",
            "properties": {"contentLength": 50, "etag": "etag-2"},
        },
        {
            "name": "run-1-sibling/collections/7/segment.pax",
            "properties": {
                "contentLength": 9999,
                "etag": "sibling-etag",
                "lastModified": "2026-07-30T00:00:00Z",
            },
        },
    ]
    completed = SimpleNamespace(stdout=json.dumps(payload))
    geometry = HARNESS.AzureCliPaxGeometry("adls://benchmarks/run-1", tmp_path)

    with patch.object(HARNESS.subprocess, "run", return_value=completed) as run:
        inventory = geometry.inventory()

    command = run.call_args.args[0]
    assert command[command.index("--container-name") + 1] == "benchmarks"
    assert command[command.index("--prefix") + 1] == "run-1"
    assert inventory == {
        "segment_count": 1,
        "bytes": 1234,
        "segments": [
            {
                "path": "run-1/collections/7/segment.pax",
                "bytes": 1234,
                "etag": "etag-1",
                "last_modified": "2026-07-30T00:00:00Z",
            }
        ],
    }
    assert geometry.stable_signature(inventory) == (
        ("run-1/collections/7/segment.pax", 1234, "etag-1"),
    )


def test_azure_inventory_forwards_explicit_connection_string(tmp_path: Path) -> None:
    completed = SimpleNamespace(stdout="[]")
    geometry = HARNESS.AzureCliPaxGeometry("az://benchmarks/run-1", tmp_path)

    with (
        patch.dict(
            HARNESS.os.environ,
            {"AZURE_STORAGE_CONNECTION_STRING": "emulator-connection"},
        ),
        patch.object(HARNESS.subprocess, "run", return_value=completed) as run,
    ):
        geometry.inventory()

    command = run.call_args.args[0]
    assert command[command.index("--connection-string") + 1] == (
        "emulator-connection"
    )


@pytest.mark.parametrize("blob_name", ["/absolute.pax", "../escape.pax"])
def test_azure_snapshot_rejects_paths_outside_evidence_root(
    tmp_path: Path, blob_name: str
) -> None:
    geometry = HARNESS.AzureCliPaxGeometry("adls://benchmarks/run-1", tmp_path)

    with pytest.raises(RuntimeError, match="unsafe Azure blob name"):
        geometry._snapshot_target(blob_name)


def test_config_preserves_object_store_url(tmp_path: Path) -> None:
    config_path = tmp_path / "benchmark.toml"
    storage_url = "adls://benchmarks/run-1"

    HARNESS.write_config(
        config_path,
        tmp_path,
        5690,
        128,
        20_000,
        storage_url,
        compaction_max_memory_mb=4096,
    )

    config = config_path.read_text()
    assert f'url = "{storage_url}"' in config
    assert "[storage.optimization]\nenable_mmap = false" in config
    assert "vector_count_threshold" not in config  # retired knob (#1526)
    assert "flush_floor_predicted_mb = 128" in config
    assert "memory_amplification_factor = 12.0" in config
    assert "memory_budget_fraction = 0.25" in config
    assert "available_memory_fraction = 0.5" in config
    assert "max_memory_mb = 4096" in config


def test_config_arms_bounded_local_spill_explicitly(tmp_path: Path) -> None:
    config_path = tmp_path / "benchmark.toml"
    scratch = tmp_path / "managed-disk" / "compaction"

    HARNESS.write_config(
        config_path,
        tmp_path,
        5690,
        128,
        20_000,
        "az://benchmarks/spill",
        compaction_max_memory_mb=4096,
        compaction_spill_enabled=True,
        compaction_spill_directory=scratch,
        compaction_spill_working_memory_mb=384,
        compaction_spill_scratch_amplification_factor=3.5,
        compaction_spill_available_disk_fraction=0.4,
        compaction_spill_max_disk_mb=8192,
    )

    config = config_path.read_text()
    assert "spill_enabled = true" in config
    assert f'spill_directory = "{scratch}"' in config
    assert "spill_working_memory_mb = 384" in config
    assert "spill_scratch_amplification_factor = 3.5" in config
    assert "spill_available_disk_fraction = 0.4" in config
    assert "spill_max_disk_mb = 8192" in config


def test_resource_sampler_reports_process_and_scratch_peaks(tmp_path: Path) -> None:
    sampler = HARNESS.ProcessScratchSampler(123, tmp_path, interval_seconds=0.25)

    with (
        patch.object(sampler, "_process_rss_bytes", side_effect=[100, 180, 140]),
        patch.object(sampler, "_scratch_bytes", side_effect=[10, 70, 20]),
    ):
        sampler.sample_once()
        sampler.sample_once()
        sampler.sample_once()

    assert sampler.snapshot() == {
        "sample_interval_seconds": 0.25,
        "sample_count": 3,
        "baseline_process_rss_bytes": 100,
        "peak_process_rss_bytes": 180,
        "peak_process_rss_delta_bytes": 80,
        "baseline_scratch_bytes": 10,
        "peak_scratch_bytes": 70,
        "peak_scratch_delta_bytes": 60,
    }


def test_scratch_sampler_tolerates_run_retirement_during_scan(tmp_path: Path) -> None:
    run = tmp_path / "retired.pxrun"
    run.write_bytes(b"run")

    def disappearing_stat(*, follow_symlinks: bool):
        assert not follow_symlinks
        run.unlink()
        raise FileNotFoundError(run)

    entry = SimpleNamespace(
        path=str(run),
        is_dir=lambda *, follow_symlinks: False,
        is_file=lambda *, follow_symlinks: True,
        stat=disappearing_stat,
    )

    class Entries:
        def __enter__(self):
            return iter([entry])

        def __exit__(self, *_args):
            return False

    with patch.object(HARNESS.os, "scandir", return_value=Entries()):
        assert HARNESS.directory_size_bytes(tmp_path) == 0


def test_config_records_controlled_geometry_flush_floor(tmp_path: Path) -> None:
    config_path = tmp_path / "benchmark.toml"

    HARNESS.write_config(
        config_path,
        tmp_path,
        5690,
        2048,
        2_000_000,
        "az://benchmarks/run-1",
        flush_interval_secs=3600,
        flush_floor_predicted_mb=256,
    )

    config = config_path.read_text()
    assert "flush_interval_secs = 3600" in config
    assert "flush_floor_predicted_mb = 256" in config


def test_explicit_flush_disables_timer_races_by_default() -> None:
    assert HARNESS.effective_flush_interval(None, None) == 12
    assert HARNESS.effective_flush_interval(20_000, None) == 3600
    assert HARNESS.effective_flush_interval(20_000, 60) == 60
    with pytest.raises(RuntimeError, match="positive"):
        HARNESS.effective_flush_interval(20_000, 0)


def test_flight_upsert_stream_writes_batches_and_validates_ack() -> None:
    calls = {"rows": 0}

    class FakeLocation:
        @staticmethod
        def for_grpc_tcp(host: str, port: int) -> tuple[str, int]:
            return host, port

    class FakeDescriptor:
        @staticmethod
        def for_command(command: bytes):
            calls["command"] = json.loads(command)
            return command

    class FakeWriter:
        def write_batch(self, batch) -> None:
            calls["rows"] += batch.num_rows

        def done_writing(self) -> None:
            calls["done_writing"] = True

        def close(self) -> None:
            calls["writer_closed"] = True

    class FakeReader:
        @staticmethod
        def read():
            return SimpleNamespace(
                to_pybytes=lambda: json.dumps(
                    {
                        "success": True,
                        "metrics": {
                            "total_processed": 5,
                            "successful_count": 5,
                            "failed_count": 0,
                        },
                    }
                ).encode()
            )

    class FakeClient:
        def __init__(self, location):
            calls["location"] = location

        @staticmethod
        def do_put(descriptor, schema):
            calls["descriptor"] = descriptor
            calls["schema"] = schema
            return FakeWriter(), FakeReader()

        @staticmethod
        def close() -> None:
            calls["client_closed"] = True

    flight = SimpleNamespace(
        Location=FakeLocation,
        FlightDescriptor=FakeDescriptor,
        FlightClient=FakeClient,
    )
    stream = HARNESS.FlightUpsertStream(
        "127.0.0.1",
        5692,
        "7",
        schema="vector-schema",
        flight_module=flight,
    )

    stream.write_batch(SimpleNamespace(num_rows=2))
    stream.write_batch(SimpleNamespace(num_rows=3))
    result = stream.close()

    assert calls["location"] == ("127.0.0.1", 5692)
    assert calls["command"] == {
        "collection_id": "7",
        "operation": "upsert",
        "write_mode": "wal",
        "trigger_compaction": False,
    }
    assert calls["schema"] == "vector-schema"
    assert calls["rows"] == 5
    assert calls["done_writing"] is True
    assert calls["writer_closed"] is True
    assert calls["client_closed"] is True
    assert result["metrics"]["successful_count"] == 5


def test_explicit_flush_uses_supported_flight_action() -> None:
    calls = {}

    class FakeLocation:
        @staticmethod
        def for_grpc_tcp(host: str, port: int) -> tuple[str, int]:
            return host, port

    class FakeClient:
        def __init__(self, location: tuple[str, int]):
            calls["location"] = location

        def do_action(self, action):
            calls["action"] = action
            return [
                SimpleNamespace(
                    body=json.dumps(
                        {
                            "success": True,
                            "collection_id": "7",
                            "operation": "flush",
                        }
                    ).encode()
                )
            ]

        def close(self) -> None:
            calls["closed"] = True

    class FakeAction:
        def __init__(self, action_type: str, body: bytes):
            self.type = action_type
            self.body = body

    flight = SimpleNamespace(
        Location=FakeLocation,
        FlightClient=FakeClient,
        Action=FakeAction,
    )

    response = HARNESS.force_flush_via_flight("127.0.0.1", 5692, "7", flight)

    assert response["success"] is True
    assert calls["location"] == ("127.0.0.1", 5692)
    assert calls["action"].type == "flush_collection"
    assert json.loads(calls["action"].body) == {"collection_id": "7"}
    assert calls["closed"] is True


def test_explicit_compaction_uses_supported_flight_action() -> None:
    calls = {}

    class FakeLocation:
        @staticmethod
        def for_grpc_tcp(host: str, port: int) -> tuple[str, int]:
            return host, port

    class FakeClient:
        def __init__(self, location: tuple[str, int]):
            calls["location"] = location

        def do_action(self, action):
            calls["action"] = action
            return [
                SimpleNamespace(
                    body=json.dumps(
                        {
                            "success": True,
                            "collection_id": "7",
                            "operation": "compact",
                        }
                    ).encode()
                )
            ]

        def close(self) -> None:
            calls["closed"] = True

    class FakeAction:
        def __init__(self, action_type: str, body: bytes):
            self.type = action_type
            self.body = body

    flight = SimpleNamespace(
        Location=FakeLocation,
        FlightClient=FakeClient,
        Action=FakeAction,
    )

    response = HARNESS.compact_via_flight("127.0.0.1", 5690, "7", flight)

    assert response["success"] is True
    assert calls["location"] == ("127.0.0.1", 5690)
    assert calls["action"].type == "compact_collection"
    assert json.loads(calls["action"].body) == {"collection_id": "7"}
    assert calls["closed"] is True


def test_explicit_flush_bed_relies_on_automatic_compaction_quiescence() -> None:
    assert HARNESS.post_flush_compaction_observation(None) is None
    observation = HARNESS.post_flush_compaction_observation(20_000)

    assert observation is not None
    assert observation["requested"] is False
    assert "materialization gate" in observation["reason"]


def test_explicit_flush_waits_for_stable_new_pax_epoch() -> None:
    before = {"segments": [], "segment_count": 0, "bytes": 0}
    after = {
        "segments": [{"path": "7/data/L0.pax", "bytes": 4096, "etag": "e1"}],
        "segment_count": 1,
        "bytes": 4096,
    }

    class FakeGeometry:
        def __init__(self):
            self.calls = 0

        def inventory(self) -> dict:
            self.calls += 1
            return after

        @staticmethod
        def stable_signature(inventory: dict) -> tuple:
            return tuple(
                (item["path"], item["bytes"], item["etag"])
                for item in inventory["segments"]
            )

    with (
        patch.object(HARNESS, "scrape_text", return_value=""),
        patch.object(HARNESS.time, "sleep") as sleep,
    ):
        geometry = FakeGeometry()
        elapsed, observed, wal_bytes = HARNESS.wait_for_pax_epoch(
            "http://server", "7", geometry, before
        )

    assert elapsed >= 0
    assert observed == after
    assert wal_bytes is None
    assert geometry.calls == 2
    sleep.assert_called_once_with(0.5)


def test_materialization_accepts_absent_wal_gauge_after_exact_azure_footer(
    tmp_path: Path,
) -> None:
    inventory = {
        "segments": [{"path": "7/data/L0.pax", "bytes": 4096, "etag": "e1"}],
        "segment_count": 1,
        "bytes": 4096,
    }

    class FakeGeometry:
        @staticmethod
        def inventory() -> dict:
            return inventory

        @staticmethod
        def stable_signature(observed: dict) -> tuple:
            return tuple(
                (item["path"], item["bytes"], item["etag"])
                for item in observed["segments"]
            )

        @staticmethod
        def materialize(observed: dict) -> dict:
            return {
                **observed,
                "row_count": 100,
                "segments": [
                    {
                        **observed["segments"][0],
                        "rows": 100,
                        "layout_version": 3,
                    }
                ],
            }

    with (
        patch.object(HARNESS, "scrape_text", return_value=""),
        patch.object(HARNESS.time, "sleep"),
        patch.object(HARNESS.time, "monotonic", side_effect=[0, 1, 1, 2, 2]),
    ):
        settled = HARNESS.wait_for_materialization(
            tmp_path,
            "http://server",
            "7",
            expected_rows=100,
            max_segments=1,
            timeout_seconds=10,
            stable_seconds=0,
            azure_geometry=FakeGeometry(),
        )

    assert settled["row_count"] == 100
    assert settled["wal_unflushed_bytes"] is None
    assert HARNESS.wal_is_quiescent(None)
    assert HARNESS.wal_is_quiescent(0)
    assert not HARNESS.wal_is_quiescent(1)


def test_v3_azure_settle_treats_l0_as_transient_until_training_compaction() -> None:
    l0 = {
        "segments": [{"path": "7/data/L0_20260731T000000_a.pax"}],
        "segment_count": 1,
    }
    l1 = {
        "segments": [{"path": "7/data/L1_20260731T000100_b.pax"}],
        "segment_count": 1,
    }
    parsed_l1_v1 = {
        "segments": [{"path": l1["segments"][0]["path"], "layout_version": 1}],
        "segment_count": 1,
    }

    assert not HARNESS.layout_candidate_is_ready(l0, 3, azure_inventory=True)
    assert HARNESS.layout_candidate_is_ready(l1, 3, azure_inventory=True)
    assert not HARNESS.layout_candidate_is_ready(
        parsed_l1_v1, 3, azure_inventory=False
    )


def test_server_records_explicit_sub_floor_training_override(tmp_path: Path) -> None:
    process = SimpleNamespace(poll=lambda: None)
    server = HARNESS.OwnedServer(
        tmp_path / "proximadb-server",
        tmp_path / "benchmark.toml",
        "http://127.0.0.1:5790",
        tmp_path / "server.log",
        None,
        training_compaction_min_mb=1,
    )

    with (
        patch.dict(
            HARNESS.os.environ,
            {"PROXIMADB_TRAINING_COMPACTION_MIN_MB": "99"},
        ),
        patch.object(HARNESS.subprocess, "Popen", return_value=process) as popen,
        patch.object(HARNESS, "request_json", return_value={}),
    ):
        server.start()

    environment = popen.call_args.kwargs["env"]
    assert environment["PROXIMADB_TRAINING_COMPACTION_MIN_MB"] == "1"


def test_server_records_explicit_query_range_policy(tmp_path: Path) -> None:
    process = SimpleNamespace(poll=lambda: None)
    server = HARNESS.OwnedServer(
        tmp_path / "proximadb-server",
        tmp_path / "benchmark.toml",
        "http://127.0.0.1:5790",
        tmp_path / "server.log",
        None,
        coalesce_gap_bytes=1024 * 1024,
        coalesce_range_bytes=16 * 1024 * 1024,
    )

    with (
        patch.object(HARNESS.subprocess, "Popen", return_value=process) as popen,
        patch.object(HARNESS, "request_json", return_value={}),
    ):
        server.start()

    environment = popen.call_args.kwargs["env"]
    assert environment["PROXIMADB_PAX_VECTOR_COALESCE_GAP"] == "1048576"
    assert environment["PROXIMADB_PAX_COALESCE_GAP"] == "1048576"
    assert environment["PROXIMADB_PAX_VECTOR_COALESCE_RANGE"] == "16777216"
    assert environment["PROXIMADB_PAX_COALESCE_RANGE"] == "16777216"


def test_server_adaptive_range_policy_removes_inherited_overrides(tmp_path: Path) -> None:
    process = SimpleNamespace(poll=lambda: None)
    server = HARNESS.OwnedServer(
        tmp_path / "proximadb-server",
        tmp_path / "benchmark.toml",
        "http://127.0.0.1:5790",
        tmp_path / "server.log",
        None,
        adaptive_read_strategy=True,
    )

    inherited = {
        "PROXIMADB_STORAGE_READ_STRATEGY_CHOOSER": "0",
        "PROXIMADB_PAX_VECTOR_COALESCE_GAP": "7",
        "PROXIMADB_PAX_VECTOR_COALESCE_RANGE": "8",
        "PROXIMADB_PAX_COALESCE_GAP": "9",
        "PROXIMADB_PAX_COALESCE_RANGE": "10",
    }
    with (
        patch.dict(HARNESS.os.environ, inherited),
        patch.object(HARNESS.subprocess, "Popen", return_value=process) as popen,
        patch.object(HARNESS, "request_json", return_value={}),
    ):
        server.start()

    environment = popen.call_args.kwargs["env"]
    assert environment["PROXIMADB_STORAGE_READ_STRATEGY_CHOOSER"] == "1"
    for name in (
        "PROXIMADB_PAX_VECTOR_COALESCE_GAP",
        "PROXIMADB_PAX_VECTOR_COALESCE_RANGE",
        "PROXIMADB_PAX_COALESCE_GAP",
        "PROXIMADB_PAX_COALESCE_RANGE",
    ):
        assert name not in environment


def _wave_scrape(counts: dict[str, float]) -> str:
    """A Prometheus text body carrying one wave histogram (cumulative)."""
    lines = ["# TYPE proximadb_pax_query_waves histogram"]
    for bound, count in counts.items():
        lines.append(f'proximadb_pax_query_waves_bucket{{le="{bound}"}} {count}')
    return "\n".join(lines) + "\n"


def test_wave_quantiles_come_from_the_histogram_delta() -> None:
    # TD-PAXDIR-1 D7: before the sweep 5 queries took 7 waves; the sweep adds
    # 4 queries at 3 waves and 1 at 4. The delta, not the totals, is measured.
    before = _wave_scrape({"2": 0, "3": 0, "4": 0, "7": 5, "+Inf": 5})
    after = _wave_scrape({"2": 0, "3": 4, "4": 5, "7": 10, "+Inf": 10})
    waves = HARNESS.histogram_delta_quantiles(
        before, after, HARNESS.WAVES_HISTOGRAM, (0.5, 0.95)
    )
    assert waves == {"observed": 5.0, "p50": 3.0, "p95": 4.0}
    empty = HARNESS.histogram_delta_quantiles(
        after, after, HARNESS.WAVES_HISTOGRAM, (0.5,)
    )
    assert empty == {"observed": 0.0, "p50": None}
    with pytest.raises(RuntimeError, match="backwards"):
        HARNESS.histogram_delta_quantiles(
            after, before, HARNESS.WAVES_HISTOGRAM, (0.5,)
        )


def test_request_and_wave_gates_report_breaches_and_missing_waves() -> None:
    result = {
        "gets_per_query": 10.0,
        "requests_per_query": 21.5,
        "recall_at_k": 0.99,
        "latency_ms": {"p50": 1.0},
        "waves": {"observed": 10.0, "p50": 4.0, "p95": 5.0},
    }
    failures = HARNESS.gate_failures(
        "object_cold",
        result,
        20.0,
        0.98,
        None,
        require_local_hit=False,
        max_requests=20.0,
        max_p50_waves=3.0,
    )
    assert failures == [
        "object_cold: requests/q 21.50 > 20.00",
        "object_cold: p50 waves 4 > 3",
    ]
    result["waves"] = {"observed": 0.0, "p50": None, "p95": None}
    assert HARNESS.gate_failures(
        "object_cold",
        result,
        None,
        0.98,
        None,
        require_local_hit=False,
        max_p50_waves=3.0,
    ) == ["object_cold: no per-query wave observations"]
    # Unset gates only report.
    assert HARNESS.gate_failures(
        "object_cold", result, None, 0.98, None, require_local_hit=False
    ) == []


def test_centroid_directory_arm_sets_every_gate_and_the_global_nprobe(
    tmp_path: Path,
) -> None:
    process = SimpleNamespace(poll=lambda: None)

    def environment(centroid_directory: str) -> dict:
        server = HARNESS.OwnedServer(
            tmp_path / "proximadb-server",
            tmp_path / "benchmark.toml",
            "http://127.0.0.1:5790",
            tmp_path / "server.log",
            None,
            nprobe=12,
            centroid_directory=centroid_directory,
        )
        inherited = dict.fromkeys(HARNESS.CENTROID_DIRECTORY_GATES, "1")
        inherited["PROXIMADB_PAX_READ_CENTROID_DIRECTORY_NPROBE"] = "99"
        with (
            patch.dict(HARNESS.os.environ, inherited),
            patch.object(HARNESS.subprocess, "Popen", return_value=process) as popen,
            patch.object(HARNESS, "request_json", return_value={}),
        ):
            server.start()
        return popen.call_args.kwargs["env"]

    on = environment("on")
    for gate in HARNESS.CENTROID_DIRECTORY_GATES:
        assert on[gate] == "1", gate
    # An explicit per-segment nprobe would disable directory planning.
    assert "PROXIMADB_PAX_READ_COARSE_NPROBE" not in on
    assert on["PROXIMADB_PAX_READ_CENTROID_DIRECTORY_NPROBE"] == "12"

    # The gates default ON since D8: the off arm disables them explicitly.
    off = environment("off")
    for gate in HARNESS.CENTROID_DIRECTORY_GATES:
        assert off[gate] == "0", gate
    assert off["PROXIMADB_PAX_READ_COARSE_NPROBE"] == "12"

    # The default arm scrubs inherited exports and leaves the gates unset,
    # so the server's (D8 default-ON) behavior applies.
    unset = environment("default")
    for gate in HARNESS.CENTROID_DIRECTORY_GATES:
        assert gate not in unset, f"inherited {gate} leaked into the default arm"
    assert unset["PROXIMADB_PAX_READ_CENTROID_DIRECTORY_NPROBE"] == "12"


def test_local_geometry_lists_published_directories(tmp_path: Path) -> None:
    collection = tmp_path / "t" / "c"
    collection.mkdir(parents=True)
    (collection / "L1_a.pax").write_bytes(b"x")
    (collection / "centroids-0000000000000002.pxcd").write_bytes(b"d")
    geometry = HARNESS.LocalPaxGeometry(tmp_path)
    assert geometry.directory_objects() == ["t/c/centroids-0000000000000002.pxcd"]


def test_unready_segments_name_untrained_tails_and_wrong_layouts() -> None:
    # TD-PAXDIR-1 D7 finding: a bed can settle on a small untrained L0 flush
    # tail that training compaction never replaces; the settle loop must name
    # it instead of timing out silently.
    geometry = {
        "segments": [
            {"path": "L2_a.pax", "layout_version": 1, "coarse_cells": 7, "bytes": 9},
            {"path": "L0_b.pax", "layout_version": 1, "coarse_cells": 0, "bytes": 2_950_953},
            {"path": "L1_c.pax", "layout_version": 3, "coarse_cells": 4, "bytes": 5},
            {"path": "unparsed.pax"},
        ]
    }
    assert HARNESS.unready_segments(geometry, 1) == [
        "L0_b.pax: untrained (no coarse A0; 2,950,953 bytes)",
        "L1_c.pax: layout v3",
    ]
    # Downloaded (materialized) geometry is fully parsed; there the untrained
    # tail alone keeps the settle loop waiting.
    parsed = {"segments": geometry["segments"][:2]}
    assert not HARNESS.layout_candidate_is_ready(parsed, 1, azure_inventory=False)
    assert HARNESS.unready_segments(parsed, 1) == [
        "L0_b.pax: untrained (no coarse A0; 2,950,953 bytes)"
    ]


def test_coalesce_cap_and_gap_come_from_the_cli_and_reach_the_server(
    tmp_path: Path,
) -> None:
    # TD-TIER-1 SC0: the range cap/gap were hard-coded constants in every
    # arm; they are experiment inputs now, so a sweep arm must actually
    # change what the server sees (and result.json must record it).
    process = SimpleNamespace(poll=lambda: None)

    def environment(gap_mb: float, range_mb: float) -> dict:
        server = HARNESS.OwnedServer(
            tmp_path / "proximadb-server",
            tmp_path / "benchmark.toml",
            "http://127.0.0.1:5790",
            tmp_path / "server.log",
            None,
            coalesce_gap_bytes=int(gap_mb * 1024 * 1024),
            coalesce_range_bytes=int(range_mb * 1024 * 1024),
        )
        with (
            patch.object(HARNESS.subprocess, "Popen", return_value=process) as popen,
            patch.object(HARNESS, "request_json", return_value={}),
        ):
            server.start()
        return popen.call_args.kwargs["env"]

    arm = environment(0.5, 16.0)
    for name in (
        "PROXIMADB_PAX_VECTOR_COALESCE_GAP",
        "PROXIMADB_PAX_COALESCE_GAP",
    ):
        assert arm[name] == str(int(0.5 * 1024 * 1024)), name
    for name in (
        "PROXIMADB_PAX_VECTOR_COALESCE_RANGE",
        "PROXIMADB_PAX_COALESCE_RANGE",
    ):
        assert arm[name] == str(int(16.0 * 1024 * 1024)), name

    # The default arm keeps the historical 1 MiB / 4 MiB constants, so old
    # arms reproduce byte-for-byte until a sweep opts in.
    default = environment(
        HARNESS.AZURE_COALESCE_GAP_BYTES / (1024 * 1024),
        HARNESS.AZURE_COALESCE_RANGE_BYTES / (1024 * 1024),
    )
    assert default["PROXIMADB_PAX_COALESCE_GAP"] == str(HARNESS.AZURE_COALESCE_GAP_BYTES)
    assert default["PROXIMADB_PAX_COALESCE_RANGE"] == str(HARNESS.AZURE_COALESCE_RANGE_BYTES)


def test_server_proc_sampler_parses_rss_and_cpu() -> None:
    # TD-TIER-1 E1 telemetry: the sampler must parse real /proc shapes and
    # derive CPU seconds from the tick delta, never from a silent zero.
    status = "Name: server\nVmRSS:      1234 kB\nThreads: 8\n"
    # comm contains spaces+parens; utime=100, stime=50 ticks at the start.
    stat_start = "1 (proximadb server) R 2 3 4 5 6 7 8 9 10 11 100 50 0 0\n"
    stat_end = "1 (proximadb server) R 2 3 4 5 6 7 8 9 10 11 250 90 0 0\n"

    files = {}

    def fake_read(path: str) -> str:
        return files[path]

    sampler = HARNESS.ServerProcSampler(pid=99, interval_s=0.001, read=fake_read)
    files[f"/proc/99/status"] = status
    files[f"/proc/99/stat"] = stat_start
    sampler.start()  # real background thread (daemon); start samples once
    # utime 100->250, stime 50->90 = 190 ticks.
    files[f"/proc/99/stat"] = stat_end
    import time as _time
    _time.sleep(0.01)  # let the thread observe the end state
    out = sampler.stop()
    assert out["peak_rss_bytes"] == 1234 * 1024
    assert out["cpu_seconds"] == 1.9


def test_skip_ingest_requires_collection_id() -> None:
    # The fixed-object mode is unusable without the collection to reuse —
    # fail at argparse time, not after a server has booted.
    import subprocess, sys as _sys

    proc = subprocess.run(
        [
            _sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "bench" / "sift1m_get_reduction.py"),
            "--root",
            "/tmp/whatever",
            "--skip-ingest",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode != 0
    assert "--skip-ingest requires --collection-id" in (proc.stderr or proc.stdout)
