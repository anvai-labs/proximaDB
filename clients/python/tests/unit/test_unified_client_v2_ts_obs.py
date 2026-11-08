"""TD-SDK-3 batch 3: the timeseries + observability families onto v2.

Mirrors the batch contract:

1. Facade delegation — every method reaches the adapter with the kwargs
   v1's dispatch produces; return coercions (int/list/bool) per contract.
2. Wire equivalence — the REAL REST adapter runs the spec-driven
   generated ops for timeseries (the previous hand-rolled ``_session``
   code never executed) and for namespace/logs/metrics.
3. Transport ownership — the traces family stays on the honest base
   defaults (no REST spec endpoints; embedded-only, exactly as v1
   discloses); gRPC stays on base defaults for the whole batch.
4. Parity census ratchet — the ported sets are pinned; they only grow.
"""

from http import HTTPStatus
from unittest.mock import MagicMock, patch

import pytest

import proximadb_sdk.unified_client_v2 as uc
from proximadb_sdk._generated.rest.types import Response
from proximadb_sdk.adapters.base import BaseProtocolAdapter
from proximadb_sdk.adapters.embedded_adapter import EmbeddedProtocolAdapter
from proximadb_sdk.adapters.grpc_adapter import GrpcProtocolAdapter
from proximadb_sdk.adapters.rest_adapter import RestProtocolAdapter
from proximadb_sdk.config import Protocol
from proximadb_sdk.exceptions import (
    ProximaDBError,
    ProximaDBNotSupportedError,
)
from proximadb_sdk.unified_client_v2 import ProximaDBClient

# ---------------------------------------------------------------------------
# Parity census (TD-SDK-3 program step 2 — the ratchet)
# ---------------------------------------------------------------------------

#: The timeseries methods ported in TD-SDK-3 batch 3.
PORTED_TIMESERIES_METHODS = frozenset(
    {
        "create_timeseries_collection",
        "ingest_timeseries",
        "query_timeseries",
        "list_timeseries_collections",
        "delete_timeseries_collection",
    }
)
assert len(PORTED_TIMESERIES_METHODS) == 5, "batch 3 ported exactly 5"

#: The observability methods ported in TD-SDK-3 batch 3.
PORTED_OBSERVABILITY_METHODS = frozenset(
    {
        "create_observability_namespace",
        "ingest_logs",
        "query_logs",
        "ingest_metrics",
        "aggregate_metrics",
        "ingest_traces",
        "query_traces",
        "get_trace",
    }
)
assert len(PORTED_OBSERVABILITY_METHODS) == 8, "batch 3 ported exactly 8"

#: The name-filter keys for the observability census. Lexical, not
#: semantic — and deliberately not exact on the v1 side until batch 4
#: lands: TWO batch-4 names also contain these keys
#: (get_protocol_metrics, reset_routing_metrics) and belong to THAT
#: pinned set, not this one. (`logout` contains no key and escapes the
#: filter entirely — the v2-side exactness below is exact over the
#: filtered vocabulary only.)
_OBS_KEYS = ("observ", "logs", "trace", "namespace", "metric")


def _ok(parsed):
    return Response(status_code=HTTPStatus.OK, content=b"{}", headers={}, parsed=parsed)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_adapter():
    adapter = MagicMock()
    adapter.protocol_name = "rest"
    return adapter


@pytest.fixture
def client(mock_adapter):
    with patch.object(uc, "create_adapter", return_value=mock_adapter):
        c = ProximaDBClient(url="http://testserver:5678", protocol=Protocol.REST)
    return c


# ---------------------------------------------------------------------------
# 1. Facade delegation
# ---------------------------------------------------------------------------


class TestDelegation:
    def test_create_timeseries_collection(self, client, mock_adapter):
        mock_adapter.create_timeseries_collection.return_value = {"name": "ts"}
        out = client.create_timeseries_collection("ts", {"retention_ms": 1000})
        mock_adapter.create_timeseries_collection.assert_called_once_with(
            "ts", {"retention_ms": 1000}
        )
        assert out == {"name": "ts"}

    def test_ingest_timeseries(self, client, mock_adapter):
        mock_adapter.ingest_timeseries.return_value = {"ingested": 2}
        out = client.ingest_timeseries("ts", [{"timestamp": 1}])
        mock_adapter.ingest_timeseries.assert_called_once_with("ts", [{"timestamp": 1}])
        assert out == {"ingested": 2}

    def test_query_timeseries(self, client, mock_adapter):
        mock_adapter.query_timeseries.return_value = {"points": []}
        out = client.query_timeseries(
            "ts", 1, 2, aggregation="avg", bucket_ms=60, tag_filters={"h": "a"}
        )
        mock_adapter.query_timeseries.assert_called_once_with(
            "ts",
            1,
            2,
            aggregation="avg",
            bucket_ms=60,
            tag_filters={"h": "a"},
        )
        assert out == {"points": []}

    def test_observability_namespace(self, client, mock_adapter):
        mock_adapter.create_observability_namespace.return_value = {
            "success": True,
            "namespace": "ops",
        }
        out = client.create_observability_namespace("ops", retention_days=7)
        mock_adapter.create_observability_namespace.assert_called_once_with(
            "ops", retention_days=7
        )
        assert out["namespace"] == "ops"

    def test_log_and_metric_ingests_coerce_int(self, client, mock_adapter):
        mock_adapter.ingest_logs.return_value = 3
        assert client.ingest_logs("ops", [{"message": "x"}]) == 3
        mock_adapter.ingest_logs.assert_called_once_with("ops", [{"message": "x"}])
        mock_adapter.ingest_metrics.return_value = 5
        assert client.ingest_metrics("ops", [{"name": "cpu", "value": 1}]) == 5
        mock_adapter.ingest_metrics.assert_called_once_with(
            "ops", [{"name": "cpu", "value": 1}]
        )
        mock_adapter.ingest_traces.return_value = 2
        assert client.ingest_traces("ops", [{"id": "t"}]) == 2
        mock_adapter.ingest_traces.assert_called_once_with("ops", [{"id": "t"}])

    def test_query_logs_and_aggregate_coerce_list(self, client, mock_adapter):
        mock_adapter.query_logs.return_value = [{"message": "x"}]
        out = client.query_logs("ops", 0, 10, query="err", limit=5)
        mock_adapter.query_logs.assert_called_once_with(
            "ops", start_time_ns=0, end_time_ns=10, query="err", limit=5
        )
        assert out == [{"message": "x"}]

        mock_adapter.aggregate_metrics.return_value = [{"points": []}]
        out = client.aggregate_metrics(
            "ops", "cpu", aggregation="max", start_time="1", end_time="2"
        )
        mock_adapter.aggregate_metrics.assert_called_once_with(
            "ops",
            metric_name="cpu",
            aggregation="max",
            start_time="1",
            end_time="2",
            step_seconds=60,
        )
        assert out == [{"points": []}]

    def test_trace_queries(self, client, mock_adapter):
        mock_adapter.query_traces.return_value = [{"id": "t1"}]
        out = client.query_traces("ops", 0, 10, service="svc", status="error", limit=3)
        mock_adapter.query_traces.assert_called_once_with(
            "ops",
            start_time_ns=0,
            end_time_ns=10,
            trace_id=None,
            service="svc",
            operation=None,
            min_duration_ns=None,
            status="error",
            limit=3,
        )
        assert out == [{"id": "t1"}]

        mock_adapter.get_trace.return_value = {"spans": []}
        out = client.get_trace("ops", "t1")
        mock_adapter.get_trace.assert_called_once_with("ops", trace_id="t1")
        assert out == {"spans": []}

    def test_unsupported_surfaces_typed_error(self, client, mock_adapter):
        mock_adapter.query_traces.side_effect = NotImplementedError("no traces")
        with pytest.raises(ProximaDBNotSupportedError) as exc:
            client.query_traces("ops", 0, 10)
        assert isinstance(exc.value, ProximaDBError)


# ---------------------------------------------------------------------------
# 2. Wire equivalence — the REAL REST adapter over the generated ops
# ---------------------------------------------------------------------------


class TestRestWireEquivalence:
    """FACADE-level wire pins only; the per-op body/error pinning lives in
    test_rest_adapter_cov.py's timeseries section (one layer down) — the
    tests here cover what the facade adds: config-sourced client identity,
    undeclared-field pass-through, coercion, and the observability ops
    (which the cov file does not cover)."""

    @pytest.fixture
    def rest_client(self):
        adapter = RestProtocolAdapter(url="http://testserver:5678")
        with patch.object(uc, "create_adapter", return_value=adapter):
            c = ProximaDBClient(url="http://testserver:5678", protocol=Protocol.REST)
        return c

    def test_create_timeseries_body(self, rest_client):
        from proximadb_sdk._generated.rest.models.ts_create_response import (
            TsCreateResponse,
        )

        op = (
            "proximadb_sdk._generated.rest.api.timeseries"
            ".create_timeseries_collection.sync_detailed"
        )
        with patch(op) as mock_op:
            mock_op.return_value = _ok(TsCreateResponse.from_dict({"name": "ts"}))
            out = rest_client.create_timeseries_collection("ts", {"retention_ms": 5})
        assert out["name"] == "ts"
        kwargs = mock_op.call_args.kwargs
        assert kwargs["body"].to_dict()["name"] == "ts"
        assert kwargs["body"].to_dict()["retention_ms"] == 5
        assert kwargs["client"] is rest_client._adapter._get_generated_document_client()

    def test_query_timeseries_body_and_extras(self, rest_client):
        from proximadb_sdk._generated.rest.models.ts_query_response import (
            TsQueryResponse,
        )

        op = (
            "proximadb_sdk._generated.rest.api.timeseries"
            ".query_timeseries.sync_detailed"
        )
        with patch(op) as mock_op:
            mock_op.return_value = _ok(
                TsQueryResponse.from_dict({"points": [{"timestamp": 1}]})
            )
            out = rest_client.query_timeseries("ts", 1, 2)
        assert out["points"] == [{"timestamp": 1}]
        body = mock_op.call_args.kwargs["body"].to_dict()
        assert body["start_time"] == 1
        assert body["end_time"] == 2

    def test_query_aggregation_raises_honestly(self, rest_client):
        """The server's QueryRequest drops unknown fields — shipping
        aggregation/bucket_ms/tag_filters would silently return raw
        points. The adapter raises instead (v2: typed capability error;
        v1: falls back to the local repo, which honors them)."""
        with pytest.raises(ProximaDBNotSupportedError):
            rest_client.query_timeseries("ts", 1, 2, aggregation="max")

    def test_list_and_delete_timeseries(self, rest_client):
        from proximadb_sdk._generated.rest.models.ts_delete_response import (
            TsDeleteResponse,
        )
        from proximadb_sdk._generated.rest.models.ts_list_response import (
            TsListResponse,
        )

        list_op = (
            "proximadb_sdk._generated.rest.api.timeseries"
            ".list_timeseries_collections.sync_detailed"
        )
        with patch(list_op) as mock_op:
            mock_op.return_value = _ok(
                TsListResponse.from_dict({"collections": [{"name": "ts"}]})
            )
            assert rest_client.list_timeseries_collections() == [{"name": "ts"}]

        del_op = (
            "proximadb_sdk._generated.rest.api.timeseries"
            ".delete_timeseries_collection.sync_detailed"
        )
        with patch(del_op) as mock_op:
            mock_op.return_value = _ok(TsDeleteResponse.from_dict({"success": True}))
            assert rest_client.delete_timeseries_collection("ts") is True
            assert mock_op.call_args.kwargs["collection_id"] == "ts"

    def test_observability_namespace_maps_retention(self, rest_client):
        from proximadb_sdk._generated.rest.models.create_observability_namespace_response import (
            CreateObservabilityNamespaceResponse,
        )

        op = (
            "proximadb_sdk._generated.rest.api.observability"
            ".create_observability_namespace.sync_detailed"
        )
        with patch(op) as mock_op:
            mock_op.return_value = _ok(
                CreateObservabilityNamespaceResponse.from_dict(
                    {"success": True, "namespace": "ops"}
                )
            )
            out = rest_client.create_observability_namespace("ops", retention_days=7)
        assert out["namespace"] == "ops"
        body = mock_op.call_args.kwargs["body"].to_dict()
        assert body["name"] == "ops"
        assert body["hot_retention_days"] == 7

    def test_log_ingest_and_query(self, rest_client):
        from proximadb_sdk._generated.rest.models.observability_ingest_response import (
            ObservabilityIngestResponse,
        )
        from proximadb_sdk._generated.rest.models.query_logs_response_200 import (
            QueryLogsResponse200,
        )

        ingest_op = (
            "proximadb_sdk._generated.rest.api.observability"
            ".ingest_logs.sync_detailed"
        )
        with patch(ingest_op) as mock_op:
            mock_op.return_value = _ok(
                ObservabilityIngestResponse.from_dict(
                    {"ingested": 2, "failed": 0, "success": True}
                )
            )
            assert rest_client.ingest_logs("ops", [{"message": "x"}]) == 2
            kwargs = mock_op.call_args.kwargs
            assert kwargs["namespace"] == "ops"
            assert len(kwargs["body"].to_dict()["logs"]) == 1

        query_op = (
            "proximadb_sdk._generated.rest.api.observability"
            ".query_logs.sync_detailed"
        )
        with patch(query_op) as mock_op:
            mock_op.return_value = _ok(
                QueryLogsResponse200.from_dict({"logs": [{"message": "x"}]})
            )
            out = rest_client.query_logs("ops", 0, 10, query="err", limit=5)
        assert out == [{"message": "x"}]
        body = mock_op.call_args.kwargs["body"].to_dict()
        assert body["start_time_ns"] == 0
        assert body["end_time_ns"] == 10
        assert body["query"] == "err"
        assert body["limit"] == 5

    def test_metric_ingest_and_aggregate(self, rest_client):
        from proximadb_sdk._generated.rest.models.metric_aggregation_result import (
            MetricAggregationResult,
        )
        from proximadb_sdk._generated.rest.models.observability_ingest_response import (
            ObservabilityIngestResponse,
        )

        ingest_op = (
            "proximadb_sdk._generated.rest.api.observability"
            ".ingest_metrics.sync_detailed"
        )
        with patch(ingest_op) as mock_op:
            mock_op.return_value = _ok(
                ObservabilityIngestResponse.from_dict(
                    {"ingested": 4, "failed": 0, "success": True}
                )
            )
            assert rest_client.ingest_metrics("ops", [{"name": "cpu", "value": 1}]) == 4
            assert len(mock_op.call_args.kwargs["body"].to_dict()["metrics"]) == 1

        agg_op = (
            "proximadb_sdk._generated.rest.api.observability"
            ".aggregate_metrics.sync_detailed"
        )
        with patch(agg_op) as mock_op:
            mock_op.return_value = _ok(
                MetricAggregationResult.from_dict({"series": [], "query_time_ms": 1})
            )
            out = rest_client.aggregate_metrics(
                "ops", "cpu", start_time=1, end_time=2, step_seconds=30
            )
        assert out == []
        body = mock_op.call_args.kwargs["body"].to_dict()
        assert body["metric_name"] == "cpu"
        assert body["step_seconds"] == 30
        assert body["start_time_ns"] == 1
        assert body["end_time_ns"] == 2

    def test_malformed_payload_maps_to_proximadb_error(self, rest_client):
        """A point missing the spec's REQUIRED fields (timestamp) surfaces
        as ProximaDBError — not a raw KeyError from model construction."""
        op = (
            "proximadb_sdk._generated.rest.api.timeseries"
            ".ingest_timeseries.sync_detailed"
        )
        with patch(op):
            with pytest.raises(ProximaDBError, match="malformed"):
                rest_client.ingest_timeseries("ts", [{"values": {"v": 1}}])


# ---------------------------------------------------------------------------
# 3. Transport ownership
# ---------------------------------------------------------------------------


class TestTransportOwnership:
    def test_grpc_family_stays_on_base_defaults(self):
        for name in sorted(PORTED_TIMESERIES_METHODS | PORTED_OBSERVABILITY_METHODS):
            assert getattr(GrpcProtocolAdapter, name) is getattr(
                BaseProtocolAdapter, name
            ), f"{name} grew a gRPC impl — update the census + tests"

    def test_rest_timeseries_and_observability_override(self):
        rest_impls = PORTED_TIMESERIES_METHODS | (
            PORTED_OBSERVABILITY_METHODS
            - {"ingest_traces", "query_traces", "get_trace"}
        )
        for name in sorted(rest_impls):
            assert getattr(RestProtocolAdapter, name) is not getattr(
                BaseProtocolAdapter, name
            ), f"{name} dropped its REST impl"

    def test_traces_have_no_rest_spec_endpoints(self):
        """The traces family deliberately stays on the base defaults for
        REST (no spec endpoints) — embedded-only, exactly as v1 discloses."""
        for name in ("ingest_traces", "query_traces", "get_trace"):
            assert getattr(RestProtocolAdapter, name) is getattr(
                BaseProtocolAdapter, name
            )

    def test_embedded_overrides_everything(self):
        for name in sorted(PORTED_TIMESERIES_METHODS | PORTED_OBSERVABILITY_METHODS):
            assert getattr(EmbeddedProtocolAdapter, name) is not getattr(
                BaseProtocolAdapter, name
            ), f"{name} dropped its embedded impl"


# ---------------------------------------------------------------------------
# 4. Parity census ratchet
# ---------------------------------------------------------------------------


class TestParityCensus:
    def test_every_ported_method_exists_on_v2_and_base(self):
        for name in PORTED_TIMESERIES_METHODS | PORTED_OBSERVABILITY_METHODS:
            assert callable(
                getattr(ProximaDBClient, name, None)
            ), f"TD-SDK-3 ratchet: {name} missing from the v2 client"
            assert hasattr(
                BaseProtocolAdapter, name
            ), f"TD-SDK-3 ratchet: {name} missing from BaseProtocolAdapter"

    def test_timeseries_pin_matches_v1_exactly(self):
        from proximadb_sdk.unified_client import ProximaDBClient as V1Client

        v1_ts = {
            name
            for name in dir(V1Client)
            if not name.startswith("_") and "timeseries" in name
        }
        assert v1_ts == set(PORTED_TIMESERIES_METHODS), (
            f"v1 timeseries surface drifted: "
            f"extra={v1_ts - set(PORTED_TIMESERIES_METHODS)} "
            f"missing={set(PORTED_TIMESERIES_METHODS) - v1_ts}"
        )

    def test_observability_pin_is_subset_of_v1_until_batch4(self):
        """v1's name-filter set also contains TWO batch-4 names
        (get_protocol_metrics, reset_routing_metrics), so this census is
        subset-directional on the v1 side until batch 4 pins them; the
        v2 side below is exact over the filtered vocabulary (`logout`
        contains no filter key and escapes it entirely)."""
        from proximadb_sdk.unified_client import ProximaDBClient as V1Client

        v1_names = {name for name in dir(V1Client) if not name.startswith("_")}
        invented = set(PORTED_OBSERVABILITY_METHODS) - v1_names
        assert not invented, f"v2 invented observability names not on v1: {invented}"

    def test_v2_surface_equals_the_pins(self):
        v2_ts = {
            name
            for name in dir(ProximaDBClient)
            if not name.startswith("_") and "timeseries" in name
        }
        assert v2_ts == set(PORTED_TIMESERIES_METHODS)
        v2_obs = {
            name
            for name in dir(ProximaDBClient)
            if not name.startswith("_") and any(k in name for k in _OBS_KEYS)
        }
        # Batch 4 ported the two "metric"-keyed stragglers
        # (get_protocol_metrics, reset_routing_metrics) — the v2 surface is
        # the 8 observability methods plus those routing/metrics surfaces.
        assert v2_obs == set(PORTED_OBSERVABILITY_METHODS) | {
            "get_protocol_metrics",
            "reset_routing_metrics",
        }, (
            f"v2 observability surface drifted: "
            f"extra={v2_obs - set(PORTED_OBSERVABILITY_METHODS)} "
            f"missing={set(PORTED_OBSERVABILITY_METHODS) - v2_obs}"
        )
