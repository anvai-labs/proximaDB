"""Offline unit tests for proximadb_sdk.timeseries.

Fully offline: a hand-fake backend client is injected; no network, sockets,
sleeps, or real DB boot. Exercises models, builders, the repository, and the
high-level ProximaDBTimeSeries API including server-path and local-fallback
branches.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from proximadb_sdk.timeseries import (
    AggregatedMetric,
    AggregationType,
    CompressionCodec,
    DownsampleMode,
    Metric,
    ProximaDBTimeSeries,
    TimeSeriesCollectionConfig,
    TimeSeriesFilter,
    TimeSeriesQueryResponse,
    TimeSeriesRepository,
    ValueColumn,
    ValueType,
    create_timeseries_api,
)


@pytest.fixture(autouse=True)
def _clear_shared_state():
    """Reset the class-level shared dicts before/after each test."""
    TimeSeriesRepository._shared_batch_buffer.clear()
    TimeSeriesRepository._shared_collections.clear()
    TimeSeriesRepository._shared_points.clear()
    yield
    TimeSeriesRepository._shared_batch_buffer.clear()
    TimeSeriesRepository._shared_collections.clear()
    TimeSeriesRepository._shared_points.clear()


class FakeClient:
    """Hand fake backend that records calls and returns canned dicts."""

    def __init__(
        self,
        create_result=None,
        ingest_result=None,
        query_result=None,
        create_raises=False,
        ingest_raises=False,
        query_raises=False,
    ):
        self.create_result = create_result if create_result is not None else {}
        self.ingest_result = (
            ingest_result
            if ingest_result is not None
            else {"success": True, "ingested_count": 0}
        )
        self.query_result = query_result if query_result is not None else {}
        self.create_raises = create_raises
        self.ingest_raises = ingest_raises
        self.query_raises = query_raises
        self.calls: list[tuple] = []

    def create_timeseries_collection(self, **kwargs):
        self.calls.append(("create", kwargs))
        if self.create_raises:
            raise RuntimeError("boom-create")
        return self.create_result

    def ingest_timeseries(self, **kwargs):
        self.calls.append(("ingest", kwargs))
        if self.ingest_raises:
            raise RuntimeError("boom-ingest")
        return self.ingest_result

    def query_timeseries(self, **kwargs):
        self.calls.append(("query", kwargs))
        if self.query_raises:
            raise RuntimeError("boom-query")
        return self.query_result


# ---------------------------------------------------------------------------
# Models / enums / builders
# ---------------------------------------------------------------------------


def test_value_column_defaults_and_to_dict():
    vc = ValueColumn(name="cpu")
    assert vc.data_type == ValueType.FLOAT
    assert vc.aggregation == AggregationType.AVG
    assert vc.type == ValueType.FLOAT
    d = vc.to_dict()
    assert d == {
        "name": "cpu",
        "data_type": "float",
        "aggregation": "avg",
        "unit": None,
        "description": None,
    }


def test_value_column_string_coercion_and_type_alias_and_setter():
    vc = ValueColumn(name="loc", type="int", aggregation="sum", unit="lines")
    assert vc.data_type == ValueType.INT
    assert vc.aggregation == AggregationType.SUM
    assert vc.unit == "lines"
    vc.type = "bool"
    assert vc.data_type == ValueType.BOOL
    vc.type = ValueType.STRING
    assert vc.data_type == ValueType.STRING


def test_collection_config_retention_parsing_and_aliases():
    cfg = TimeSeriesCollectionConfig(
        name="m",
        value_columns=[{"name": "x", "type": "float"}],
        tags_columns=["host"],
        retention="2d",
        default_compression="zigzag",
    )
    assert cfg.retention_ms == 2 * 24 * 60 * 60 * 1000
    assert cfg.retention == f"{cfg.retention_ms}ms"
    assert cfg.tags_columns == ["host"]
    assert cfg.compression == CompressionCodec.ZIGZAG
    assert isinstance(cfg.value_columns[0], ValueColumn)
    d = cfg.to_dict()
    assert d["name"] == "m"
    assert d["compression"] == "zigzag"
    assert d["tag_columns"] == ["host"]


def test_collection_config_retention_none_and_unknown_suffix():
    assert TimeSeriesCollectionConfig._parse_retention_ms(None) is None
    assert TimeSeriesCollectionConfig._parse_retention_ms("123x") is None
    cfg = TimeSeriesCollectionConfig(name="n")
    assert cfg.retention is None
    assert cfg.retention_ms is None


def test_collection_config_explicit_retention_ms_and_compression_enum():
    cfg = TimeSeriesCollectionConfig(
        name="n",
        retention_ms=5000,
        compression=CompressionCodec.SNP,
    )
    assert cfg.retention_ms == 5000
    assert cfg.compression == CompressionCodec.SNP


def test_metric_to_dict_string_and_datetime():
    m1 = Metric(timestamp="2026-01-01T00:00:00Z", values={"v": 1}, tags={"h": "a"})
    d1 = m1.to_dict()
    assert d1["timestamp"] == "2026-01-01T00:00:00Z"
    assert d1["v"] == 1 and d1["h"] == "a"

    dt = datetime(2026, 1, 2, 3, 4, 5)
    m2 = Metric(timestamp=dt, values={"v": 2})
    assert m2.to_dict()["timestamp"] == dt.isoformat()


def test_aggregated_metric_to_dict():
    dt = datetime(2026, 1, 1)
    am = AggregatedMetric(timestamp=dt, values={"avg": 3.0}, count=4, tags={"h": "x"})
    d = am.to_dict()
    assert d["_count"] == 4
    assert d["avg"] == 3.0
    assert d["h"] == "x"
    assert d["timestamp"] == dt.isoformat()


def test_query_response_dict_like():
    resp = TimeSeriesQueryResponse(metrics=[{"a": 1}], total_points=1, query_time_ms=5)
    assert len(resp) == 1
    assert list(resp) == [{"a": 1}]
    assert resp.get("total_points") == 1
    assert resp.get("missing", "d") == "d"
    assert resp.to_dict()["query_time_ms"] == 5

    resp2 = TimeSeriesQueryResponse(raw_points=[{"b": 2}, {"c": 3}])
    assert len(resp2) == 2
    assert list(resp2) == [{"b": 2}, {"c": 3}]


def test_timeseries_filter_builder_to_dict():
    f = (
        TimeSeriesFilter()
        .tag("lang", "py")
        .tag_in("host", ["a", "b"])
        .gte("c", 10)
        .lte("c", 100)
        .gt("d", 1)
        .lt("d", 9)
        .or_()
        .and_()
        .limit(50)
        .time_range("2026-01-01T00:00:00", datetime(2026, 2, 1))
    )
    d = f.to_dict()
    assert d["logic"] == "AND"
    assert d["limit"] == 50
    assert d["start_time"] is not None and d["end_time"] is not None
    assert {"key": "lang", "op": "eq", "value": "py"} in d["tag_filters"]
    assert {"key": "host", "op": "in", "value": ["a", "b"]} in d["tag_filters"]
    ops = {vf["op"] for vf in d["value_filters"]}
    assert ops == {"gte", "lte", "gt", "lt"}


def test_filter_to_dict_empty_times():
    d = TimeSeriesFilter().to_dict()
    assert d["start_time"] is None
    assert d["end_time"] is None


# ---------------------------------------------------------------------------
# Repository static helpers
# ---------------------------------------------------------------------------


def test_parse_and_format_timestamp():
    dt = TimeSeriesRepository._parse_timestamp("2026-01-01T12:00:00Z")
    assert dt.tzinfo is None
    naive = TimeSeriesRepository._parse_timestamp(datetime(2026, 1, 1, 5))
    assert naive == datetime(2026, 1, 1, 5)
    aware = TimeSeriesRepository._parse_timestamp(
        datetime(2026, 1, 1, 5, tzinfo=UTC)
    )
    assert aware.tzinfo is None
    formatted = TimeSeriesRepository._format_timestamp(datetime(2026, 1, 1))
    assert formatted.endswith("Z")


def test_normalize_aggregation():
    assert TimeSeriesRepository._normalize_aggregation(None) is None
    assert (
        TimeSeriesRepository._normalize_aggregation(AggregationType.SUM)
        == AggregationType.SUM
    )
    assert TimeSeriesRepository._normalize_aggregation("max") == AggregationType.MAX


def test_interval_to_bucket_ms():
    assert TimeSeriesRepository._interval_to_bucket_ms(None) is None
    assert TimeSeriesRepository._interval_to_bucket_ms("1h") == 3600 * 1000
    assert TimeSeriesRepository._interval_to_bucket_ms("5m") == 300 * 1000
    assert TimeSeriesRepository._interval_to_bucket_ms("xyz") is None


def test_infer_value_type():
    assert TimeSeriesRepository._infer_value_type(True) == ValueType.BOOL
    assert TimeSeriesRepository._infer_value_type(3) == ValueType.INT
    assert TimeSeriesRepository._infer_value_type(3.5) == ValueType.FLOAT
    assert TimeSeriesRepository._infer_value_type("s") == ValueType.STRING


def test_aggregate_value_branches():
    repo = TimeSeriesRepository(FakeClient())
    assert repo._aggregate_value([1, 2, 3], AggregationType.COUNT) == 3
    assert repo._aggregate_value([], AggregationType.SUM) is None
    assert repo._aggregate_value([1, 2, 3], AggregationType.SUM) == 6
    assert repo._aggregate_value([2, 4], AggregationType.AVG) == 3
    assert repo._aggregate_value([5, 1], AggregationType.MIN) == 1
    assert repo._aggregate_value([5, 1], AggregationType.MAX) == 5
    assert repo._aggregate_value([7, 8], AggregationType.FIRST) == 7
    assert repo._aggregate_value([7, 8], AggregationType.LAST) == 8
    # Unhandled aggregation falls through to mean
    assert repo._aggregate_value([2, 4], AggregationType.MEDIAN) == 3


def test_bucket_start():
    repo = TimeSeriesRepository(FakeClient())
    dt = datetime(2026, 1, 1, 1, 30, 45)
    assert repo._bucket_start(dt, None) == dt
    bucketed = repo._bucket_start(dt, 3600 * 1000)
    assert bucketed.minute == 0 and bucketed.second == 0


# ---------------------------------------------------------------------------
# Collection management via repository
# ---------------------------------------------------------------------------


def test_create_collection_success_and_cache():
    client = FakeClient(create_result={"collection_id": "cid-1"})
    repo = TimeSeriesRepository(client)
    cfg = TimeSeriesCollectionConfig(
        name="m", value_columns=[ValueColumn(name="x", type="float")], tag_columns=["h"]
    )
    cid = repo.create_collection(cfg)
    assert cid == "cid-1"
    assert "cid-1" in repo._collections
    assert repo.get_collection("cid-1") is not None


def test_create_collection_default_id_and_error():
    client = FakeClient(create_result={})  # no collection_id -> uses config.name
    repo = TimeSeriesRepository(client)
    cfg = TimeSeriesCollectionConfig(name="named")
    assert repo.create_collection(cfg) == "named"

    err_client = FakeClient(create_raises=True)
    repo2 = TimeSeriesRepository(err_client)
    from proximadb_sdk.exceptions import ProximaDBError

    with pytest.raises(ProximaDBError):
        repo2.create_collection(TimeSeriesCollectionConfig(name="bad"))


def test_get_collection_missing_returns_none():
    repo = TimeSeriesRepository(FakeClient())
    assert repo.get_collection("nope") is None


def test_list_and_delete_collection():
    client = FakeClient(create_result={"collection_id": "c"})
    repo = TimeSeriesRepository(client)
    repo.create_collection(TimeSeriesCollectionConfig(name="c"))
    listed = repo.list_collections()
    assert any(item["id"] == "c" for item in listed)
    assert repo.delete_collection("c") is True
    assert repo.get_collection("c") is None


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def test_ingest_empty():
    repo = TimeSeriesRepository(FakeClient())
    res = repo.ingest("c", [])
    assert res["ingested_count"] == 0


def test_ingest_server_path_and_local_cache():
    client = FakeClient(ingest_result={"success": True, "ingested_count": 2})
    repo = TimeSeriesRepository(client)
    metrics = [
        Metric(timestamp="2026-01-01T00:00:00Z", values={"v": 1.0}, tags={"h": "a"}),
        {"timestamp": "2026-01-01T01:00:00Z", "values": {"v": 2.0}, "tags": {"h": "a"}},
    ]
    res = repo.ingest("c", metrics)
    assert res == {"success": True, "ingested_count": 2}
    assert len(repo._points["c"]) == 2
    # Collection inferred
    assert "c" in repo._collections


def test_ingest_fallback_on_server_error():
    client = FakeClient(ingest_raises=True)
    repo = TimeSeriesRepository(client)
    res = repo.ingest("c", [{"timestamp": "2026-01-01T00:00:00Z", "values": {"v": 1}}])
    assert res["fallback"] == "local"
    assert res["ingested_count"] == 1


def test_ingest_auto_flush_when_batch_full():
    client = FakeClient(ingest_result={"success": True})
    repo = TimeSeriesRepository(client, batch_size=2)
    repo.ingest(
        "c",
        [
            {"timestamp": "2026-01-01T00:00:00Z", "values": {"v": 1}},
            {"timestamp": "2026-01-01T01:00:00Z", "values": {"v": 2}},
        ],
    )
    # Buffer auto-flushed
    assert repo._batch_buffer["c"] == []


def test_ingest_batch_flushes():
    client = FakeClient(ingest_result={"success": True})
    repo = TimeSeriesRepository(client, batch_size=1000)
    res = repo.ingest_batch(
        "c", [{"timestamp": "2026-01-01T00:00:00Z", "values": {"v": 1}}]
    )
    assert res["flushed_count"] == 1


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def test_query_server_metrics_path():
    client = FakeClient(
        query_result={
            "metrics": [
                {"timestamp": "2026-01-01T00:00:00Z", "values": {"v": 1}, "tags": {}}
            ],
            "total_points": 1,
        }
    )
    repo = TimeSeriesRepository(client)
    resp = repo.query("c", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z")
    assert resp.total_points == 1
    assert len(resp.metrics) == 1


def test_query_server_raw_points_path():
    client = FakeClient(
        query_result={"points": [{"timestamp": "t", "values": {}}], "total_points": 1}
    )
    repo = TimeSeriesRepository(client)
    resp = repo.query("c", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z")
    assert resp.total_points == 1
    assert len(resp.raw_points) == 1


def test_query_local_fallback_raw():
    client = FakeClient(query_raises=True)
    repo = TimeSeriesRepository(client)
    # Seed local points
    repo._ensure_collection("c")
    repo._points["c"].append(
        {
            "timestamp": TimeSeriesRepository._parse_timestamp("2026-01-01T06:00:00Z"),
            "values": {"v": 5},
            "tags": {"h": "a"},
        }
    )
    with pytest.warns(UserWarning):
        resp = repo.query("c", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z")
    assert resp.total_points == 1
    assert len(resp.raw_points) == 1


def test_query_local_fallback_aggregated_and_filter():
    client = FakeClient(query_raises=True)
    repo = TimeSeriesRepository(client)
    repo._ensure_collection("c")
    for hour, v in [(0, 10), (0, 20), (1, 30)]:
        repo._points["c"].append(
            {
                "timestamp": datetime(2026, 1, 1, hour),
                "values": {"v": v},
                "tags": {"h": "a"},
            }
        )
    flt = TimeSeriesFilter().tag("h", "a")
    with pytest.warns(UserWarning):
        resp = repo.query(
            "c",
            "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z",
            filter=flt,
            aggregation="avg",
            interval="1h",
        )
    assert len(resp.metrics) == 2  # two hourly buckets


def test_query_local_ohlc():
    client = FakeClient(query_raises=True)
    repo = TimeSeriesRepository(client)
    repo._collections["c"] = TimeSeriesCollectionConfig(
        name="c", value_columns=[ValueColumn(name="v", type="float")]
    )
    repo._ensure_collection("c")
    for minute, v in [(0, 10.0), (1, 50.0), (2, 5.0), (3, 30.0)]:
        repo._points["c"].append(
            {
                "timestamp": datetime(2026, 1, 1, 0, minute),
                "values": {"v": v},
                "tags": {},
            }
        )
    with pytest.warns(UserWarning):
        resp = repo.query(
            "c",
            "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z",
            aggregation=AggregationType.OHLC,
            interval="1h",
        )
    m = resp.metrics[0]
    assert m["open"] == 10.0 and m["high"] == 50.0
    assert m["low"] == 5.0 and m["close"] == 30.0


# ---------------------------------------------------------------------------
# get_latest
# ---------------------------------------------------------------------------


def test_get_latest_and_batch():
    repo = TimeSeriesRepository(FakeClient())
    repo._ensure_collection("c")
    repo._points["c"].extend(
        [
            {"timestamp": datetime(2026, 1, 1), "values": {"v": 1}, "tags": {"h": "a"}},
            {"timestamp": datetime(2026, 1, 2), "values": {"v": 2}, "tags": {"h": "a"}},
            {"timestamp": datetime(2026, 1, 3), "values": {"v": 9}, "tags": {"h": "b"}},
        ]
    )
    latest = repo.get_latest("c", {"h": "a"})
    assert latest is not None and latest.values["v"] == 2
    assert repo.get_latest("c", {"h": "missing"}) is None
    batch = repo.get_latest_batch("c", [{"h": "a"}, {"h": "missing"}])
    assert batch[0] is not None and batch[1] is None


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def test_aggregate_simple_server_metrics():
    client = FakeClient(
        query_result={
            "metrics": [
                {"timestamp": "2026-01-01T00:00:00Z", "values": {"v": 1}, "tags": {}}
            ],
            "total_points": 1,
        }
    )
    repo = TimeSeriesRepository(client)
    res = repo.aggregate(
        "c",
        "2026-01-01T00:00:00Z",
        "2026-01-02T00:00:00Z",
        aggregation="avg",
        interval="1h",
        value_column="v",
    )
    assert "results" in res and "query_time_ms" in res


def test_aggregate_pipeline_group_by_and_default():
    client = FakeClient(query_raises=True)
    repo = TimeSeriesRepository(client)
    repo._ensure_collection("c")
    for hour, host, v in [(0, "a", 10), (0, "b", 20), (1, "a", 30)]:
        repo._points["c"].append(
            {
                "timestamp": datetime(2026, 1, 1, hour),
                "values": {"v": v},
                "tags": {"host": host},
            }
        )
    pipeline = [
        {
            "stage": "group_by",
            "aggregation": "avg",
            "tag_columns": ["host"],
            "bucket_ms": 3600000,
        },
    ]
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = repo.aggregate(
            "c", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", pipeline=pipeline
        )
    assert res["results"]
    # grouped tags present
    assert any("tags" in m for m in res["results"])


def test_aggregate_pipeline_non_group_stage():
    client = FakeClient(
        query_result={
            "metrics": [{"timestamp": "2026-01-01T00:00:00Z", "values": {"v": 5}}],
            "total_points": 1,
        }
    )
    repo = TimeSeriesRepository(client)
    pipeline = [{"stage": "aggregate", "aggregation": "sum", "bucket_ms": 3600000}]
    res = repo.aggregate(
        "c", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", pipeline=pipeline
    )
    assert "results" in res


# ---------------------------------------------------------------------------
# downsample / flush
# ---------------------------------------------------------------------------


def test_downsample_stub():
    repo = TimeSeriesRepository(FakeClient())
    res = repo.downsample("src", "dst", "1h", mode=DownsampleMode.SMA)
    assert res == {"success": True, "downsampled": 0}


def test_flush_batch_branches():
    repo = TimeSeriesRepository(FakeClient())
    # collection not in buffer
    assert repo.flush_batch("missing") == {"success": True, "flushed": 0}
    # empty buffer
    repo._ensure_collection("c")
    assert repo.flush_batch("c") == {"success": True, "flushed": 0}
    # with data
    repo._batch_buffer["c"].append({"x": 1})
    res = repo.flush_batch("c")
    assert res["flushed"] == 1
    assert repo._batch_buffer["c"] == []


# ---------------------------------------------------------------------------
# matches_filter direct branches
# ---------------------------------------------------------------------------


def test_matches_filter_branches():
    repo = TimeSeriesRepository(FakeClient())
    point = {
        "timestamp": datetime(2026, 1, 1, 12),
        "values": {"v": 50},
        "tags": {"h": "a", "lang": "py"},
    }
    # tag_filters dict mismatch
    assert repo._matches_filter(point, None, tag_filters={"h": "b"}) is False
    # None filter, matching tag_filters
    assert repo._matches_filter(point, None, tag_filters={"h": "a"}) is True
    # dict filter form with tag_filters as dict + value filters + time range + OR
    filter_dict = {
        "logic": "OR",
        "tag_filters": {"lang": "py"},
        "value_filters": [
            {"column": "v", "op": "gte", "value": 10},
            {"column": "v", "op": "lte", "value": 100},
            {"column": "v", "op": "gt", "value": 0},
            {"column": "v", "op": "lt", "value": 999},
            {"column": "v", "op": "eq", "value": 50},
        ],
        "start_time": "2026-01-01T00:00:00Z",
        "end_time": "2026-01-02T00:00:00Z",
    }
    assert repo._matches_filter(point, filter_dict) is True
    # in op on tag list via list form
    filter_in = {"tag_filters": [{"key": "h", "op": "in", "value": ["a", "z"]}]}
    assert repo._matches_filter(point, filter_in) is True
    # TimeSeriesFilter object path
    tf = TimeSeriesFilter().tag("h", "a")
    assert repo._matches_filter(point, tf) is True


# ---------------------------------------------------------------------------
# High-level API + factory
# ---------------------------------------------------------------------------


def test_high_level_api_full_flow():
    client = FakeClient(
        create_result={"collection_id": "code_metrics"},
        ingest_result={"success": True, "ingested_count": 1},
        query_raises=True,
    )
    ts = ProximaDBTimeSeries(client, batch_size=1000)

    created = ts.create_collection(
        name="code_metrics",
        value_columns=[ValueColumn(name="v", type="float")],
        tags_columns=["h"],
        retention="30d",
    )
    assert created["success"] is True
    assert created["collection_id"] == "code_metrics"

    ing = ts.ingest(
        "code_metrics",
        points=[
            Metric(timestamp="2026-01-01T00:00:00Z", values={"v": 1.0}, tags={"h": "a"})
        ],
    )
    assert ing["success"] is True

    with pytest.warns(UserWarning):
        resp = ts.query("code_metrics", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z")
    assert isinstance(resp, TimeSeriesQueryResponse)

    latest = ts.get_latest("code_metrics", {"h": "a"})
    assert latest is not None

    assert any(c["id"] == "code_metrics" for c in ts.list_collections())

    agg = ts.aggregate(
        "code_metrics",
        "2026-01-01T00:00:00Z",
        "2026-01-02T00:00:00Z",
        aggregation="avg",
        interval="1h",
    )
    assert "results" in agg

    assert ts.flush("code_metrics")["success"] is True
    assert ts.delete_collection("code_metrics") is True


def test_high_level_create_with_config_object():
    client = FakeClient(create_result={"collection_id": "x"})
    ts = ProximaDBTimeSeries(client)
    cfg = TimeSeriesCollectionConfig(name="x")
    res = ts.create_collection(config=cfg)
    assert res["collection_id"] == "x"


def test_high_level_ingest_metrics_kwarg():
    client = FakeClient(ingest_result={"success": True})
    ts = ProximaDBTimeSeries(client)
    res = ts.ingest(
        "c", metrics=[{"timestamp": "2026-01-01T00:00:00Z", "values": {"v": 1}}]
    )
    assert res["success"] is True


def test_factory_returns_instance():
    api = create_timeseries_api(FakeClient(), batch_size=10)
    assert isinstance(api, ProximaDBTimeSeries)
