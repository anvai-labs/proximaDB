"""Offline unit tests for metadata_utils, performance/data_models, and
unified_client_async.

Fully offline: no network, no server, no model downloads. The async unified
client is exercised by injecting AsyncMock REST / MagicMock gRPC backends onto
its private attributes so no real transport is ever opened.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from proximadb_sdk import metadata_utils as mu
from proximadb_sdk import unified_client_async as uca
from proximadb_sdk.config import Protocol
from proximadb_sdk.performance import data_models as dm

# ---------------------------------------------------------------------------
# metadata_utils
# ---------------------------------------------------------------------------


def test_dict_to_proto_metadata_all_types():
    items = mu.dict_to_proto_metadata(
        {
            "s": "hello",
            "i": 7,
            "f": 1.5,
            "b": True,
            "none": None,
            "other": [1, 2, 3],
        }
    )
    by_key = {it.key: it for it in items}
    assert by_key["s"].string_value == "hello"
    # bool must be detected before int/float
    assert by_key["b"].bool_value is True
    assert by_key["i"].number_value == 7.0
    assert by_key["f"].number_value == 1.5
    # None -> empty string
    assert by_key["none"].string_value == ""
    # unsupported types -> stringified
    assert by_key["other"].string_value == str([1, 2, 3])


def test_proto_metadata_roundtrip():
    items = mu.dict_to_proto_metadata(
        {"s": "x", "n": 3, "fl": 1.25, "flag": False, "empty": None}
    )
    result = mu.proto_metadata_to_dict(items)
    assert result["s"] == "x"
    assert result["n"] == 3.0
    assert result["fl"] == 1.25
    assert result["flag"] is False
    # None encoded as empty string round-trips back to ""
    assert result["empty"] == ""


def test_proto_metadata_to_dict_no_value_set():
    # An item with key but no value field set -> None
    pb = mu.v1_vector_types_pb2
    item = pb.MetadataItem()
    item.key = "lonely"
    result = mu.proto_metadata_to_dict([item])
    assert result == {"lonely": None}


def test_has_field_handles_non_proto():
    # Object without HasField -> False (AttributeError path)
    class Dummy:
        pass

    assert mu._has_field(Dummy(), "string_value") is False

    # Object whose HasField raises ValueError
    class Raiser:
        def HasField(self, name):
            raise ValueError("bad field")

    assert mu._has_field(Raiser(), "string_value") is False


def test_json_compatible_value():
    assert mu.json_compatible_value(True) is True
    assert mu.json_compatible_value(False) is False
    assert mu.json_compatible_value(5) == 5.0
    assert mu.json_compatible_value(2.5) == 2.5
    assert mu.json_compatible_value("text") == "text"
    assert mu.json_compatible_value(None) is None
    assert mu.json_compatible_value({"a": 1}) == str({"a": 1})


def test_dict_to_proto_metadata_raises_without_grpc(monkeypatch):
    monkeypatch.setattr(mu, "GRPC_AVAILABLE", False)
    monkeypatch.setattr(mu, "v1_vector_types_pb2", None)
    with pytest.raises(ImportError):
        mu.dict_to_proto_metadata({"a": 1})


# ---------------------------------------------------------------------------
# performance/data_models
# ---------------------------------------------------------------------------


def test_validation_status_values():
    assert dm.ValidationStatus.PASS.value == "pass"
    assert dm.ValidationStatus.WARN.value == "warn"
    assert dm.ValidationStatus.FAIL.value == "fail"
    assert dm.ValidationStatus.SKIP.value == "skip"


def test_latency_stats_from_samples_empty():
    s = dm.LatencyStats.from_samples([])
    assert s.min_ms == 0
    assert s.max_ms == 0
    assert s.avg_ms == 0
    assert s.std_dev_ms == 0


def test_latency_stats_from_samples_single():
    s = dm.LatencyStats.from_samples([5.0])
    assert s.min_ms == 5.0
    assert s.max_ms == 5.0
    assert s.avg_ms == 5.0
    # single sample -> stdev branch returns 0
    assert s.std_dev_ms == 0


def test_latency_stats_from_samples_many():
    samples = [float(x) for x in range(1, 101)]
    s = dm.LatencyStats.from_samples(samples)
    assert s.min_ms == 1.0
    assert s.max_ms == 100.0
    assert 50 <= s.avg_ms <= 51
    assert s.p50_ms > 0
    assert s.p95_ms > s.p50_ms
    assert s.p99_ms >= s.p95_ms
    assert s.std_dev_ms > 0


def test_throughput_and_memory_metrics():
    t = dm.ThroughputMetrics(operations_per_second=10.0)
    assert t.vectors_per_second == 0.0
    assert t.total_operations == 0
    m = dm.MemoryMetrics(peak_memory_mb=128.0)
    assert m.avg_memory_mb == 0.0
    assert m.gc_collections == 0


def test_benchmark_metrics_nested():
    bm = dm.BenchmarkMetrics(
        latency=dm.LatencyStats(min_ms=1, max_ms=2, avg_ms=1.5),
        throughput=dm.ThroughputMetrics(operations_per_second=100.0),
        memory=dm.MemoryMetrics(peak_memory_mb=64.0),
        recall=0.95,
        precision=0.9,
    )
    assert bm.latency.avg_ms == 1.5
    assert bm.throughput.operations_per_second == 100.0
    assert bm.recall == 0.95


def test_engine_performance_defaults():
    ep = dm.EnginePerformance(engine_name="nova")
    assert ep.engine_name == "nova"
    assert ep.insert_metrics is None
    assert ep.flush_time_ms is None


def test_benchmark_result_and_report():
    result = dm.BenchmarkResult(
        benchmark_name="insert-bench",
        duration_seconds=12.5,
        vector_count=1000,
        dimension=384,
        engine_results=[dm.EnginePerformance(engine_name="nova")],
        metadata={"host": "ci"},
    )
    assert result.success is True
    assert isinstance(result.timestamp, datetime)
    assert result.engine_results[0].engine_name == "nova"

    report = dm.PerformanceReport(report_id="r-1")
    assert report.report_id == "r-1"
    assert isinstance(report.generated_at, datetime)
    assert report.benchmark_results == []
    assert report.environment == {}


def test_performance_summary():
    summary = dm.PerformanceSummary(
        total_vectors_tested=10000,
        avg_insert_latency_ms=2.0,
        avg_search_latency_ms=5.0,
        recommendations=["use viper"],
    )
    assert summary.total_queries_executed == 0
    assert summary.best_engine_insert is None
    assert summary.recommendations == ["use viper"]


def test_create_latency_stats_helpers():
    s = dm.create_latency_stats(1.0, 10.0, 5.0)
    assert s.p50_ms == 5.0
    # p99 defaults to max_ms when not supplied
    assert s.p95_ms == 10.0
    assert s.p99_ms == 10.0

    s2 = dm.create_latency_stats(1.0, 10.0, 5.0, p99_ms=8.0)
    assert s2.p95_ms == 8.0
    assert s2.p99_ms == 8.0


def test_create_throughput_metrics_helper():
    t = dm.create_throughput_metrics(50.0, 500, 10000.0)
    assert t.operations_per_second == 50.0
    assert t.total_operations == 500
    assert t.total_duration_ms == 10000.0


def test_create_validation_result_threshold_pass_and_fail():
    ok = dm.create_validation_result(
        "recall", 0.95, 0.9, threshold=0.9, comparator=">="
    )
    assert ok.status == dm.ValidationStatus.PASS

    bad = dm.create_validation_result(
        "recall", 0.85, 0.9, threshold=0.9, comparator=">="
    )
    assert bad.status == dm.ValidationStatus.FAIL

    le = dm.create_validation_result("lat", 2.0, 5.0, threshold=5.0, comparator="<=")
    assert le.status == dm.ValidationStatus.PASS

    eq = dm.create_validation_result("n", 5, 5, threshold=5, comparator="==")
    assert eq.status == dm.ValidationStatus.PASS

    unknown = dm.create_validation_result("x", 1, 2, threshold=1, comparator="!!")
    # unknown comparator -> passed True
    assert unknown.status == dm.ValidationStatus.PASS


def test_create_validation_result_no_threshold_equality():
    same = dm.create_validation_result("eq", "a", "a")
    assert same.status == dm.ValidationStatus.PASS
    diff = dm.create_validation_result("eq", "a", "b")
    assert diff.status == dm.ValidationStatus.FAIL
    assert "actual=a" in diff.message


# ---------------------------------------------------------------------------
# unified_client_async
# ---------------------------------------------------------------------------


def _make_unified():
    return uca.ProximaDBAsyncUnified(url="http://testserver", timeout=5.0)


def test_init_defaults():
    c = uca.ProximaDBAsyncUnified(url="http://testserver")
    assert c.grpc_endpoint == "localhost:5679"
    assert c.protocol == Protocol.AUTO
    assert c._grpc is None
    assert c._rest is None


def test_init_protocol_string_coercion():
    c = uca.ProximaDBAsyncUnified(url="http://x", protocol="rest")
    assert c.protocol == Protocol.REST
    assert c.rest_url == "http://x"


@pytest.mark.asyncio
async def test_astart_rest_path(monkeypatch):
    fake_rest = AsyncMock()
    monkeypatch.setattr(uca, "RestAsyncClient", MagicMock(return_value=fake_rest))
    c = uca.ProximaDBAsyncUnified(url="http://testserver", protocol=Protocol.REST)
    await c.astart()
    assert c._rest is fake_rest
    assert c._grpc is None


@pytest.mark.asyncio
async def test_astart_grpc_path(monkeypatch):
    fake_grpc = MagicMock()
    fake_grpc.connect = AsyncMock()
    monkeypatch.setattr(uca, "_grpc_available", lambda: True)
    monkeypatch.setattr(uca, "RestAsyncClient", MagicMock(return_value=AsyncMock()))
    import proximadb_sdk.protocols.grpc_async as grpc_async

    monkeypatch.setattr(
        grpc_async, "ProximaDBAsyncGrpcClient", MagicMock(return_value=fake_grpc)
    )
    c = uca.ProximaDBAsyncUnified(url="http://testserver", protocol=Protocol.GRPC)
    await c.astart()
    assert c._grpc is fake_grpc
    assert c._use_grpc is True


@pytest.mark.asyncio
async def test_astart_grpc_failure_falls_back_to_rest(monkeypatch):
    fake_rest = AsyncMock()
    monkeypatch.setattr(uca, "_grpc_available", lambda: True)
    monkeypatch.setattr(uca, "RestAsyncClient", MagicMock(return_value=fake_rest))
    import proximadb_sdk.protocols.grpc_async as grpc_async

    monkeypatch.setattr(
        grpc_async,
        "ProximaDBAsyncGrpcClient",
        MagicMock(side_effect=RuntimeError("no channel")),
    )
    c = uca.ProximaDBAsyncUnified(url="http://testserver", protocol=Protocol.GRPC)
    await c.astart()
    assert c._grpc is None
    assert c._use_grpc is False
    assert c._rest is fake_rest


@pytest.mark.asyncio
async def test_aclose_with_rest():
    c = _make_unified()
    fake_rest = AsyncMock()
    c._rest = fake_rest
    await c.aclose()
    # aclose() resets _rest to None after closing, so assert on the captured mock.
    fake_rest.aclose.assert_awaited_once()
    assert c._rest is None


@pytest.mark.asyncio
async def test_aclose_without_rest_noop():
    c = _make_unified()
    # no _rest set -> should not raise
    await c.aclose()


@pytest.mark.asyncio
async def test_graph_shortest_path_grpc():
    c = _make_unified()
    grpc = MagicMock()
    grpc.shortest_path.return_value = {"path": ["a", "b"]}
    c._grpc = grpc
    out = await c.graph_shortest_path("a", "b", max_depth=5, algorithm="DIJKSTRA")
    assert out == {"path": ["a", "b"]}
    grpc.shortest_path.assert_called_once()


@pytest.mark.asyncio
async def test_graph_shortest_path_rest():
    c = _make_unified()
    rest = AsyncMock()
    rest.graph_shortest_path.return_value = {"path": ["x"]}
    c._rest = rest
    out = await c.graph_shortest_path("x", "y")
    assert out == {"path": ["x"]}
    rest.graph_shortest_path.assert_awaited_once()


@pytest.mark.asyncio
async def test_graph_shortest_path_not_started():
    c = _make_unified()
    with pytest.raises(RuntimeError):
        await c.graph_shortest_path("a", "b")


@pytest.mark.asyncio
async def test_graph_traverse_rest():
    c = _make_unified()
    rest = AsyncMock()
    rest.graph_traverse.return_value = {"nodes": ["a"]}
    c._rest = rest
    out = await c.graph_traverse("a", max_depth=2, algorithm="BFS", limit=10)
    assert out == {"nodes": ["a"]}
    rest.graph_traverse.assert_awaited_once()


@pytest.mark.asyncio
async def test_graph_traverse_not_started():
    c = _make_unified()
    with pytest.raises(RuntimeError):
        await c.graph_traverse("a")
