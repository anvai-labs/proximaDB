"""TD-SDK-3 batch 4: routing/auth compat surfaces + the search/scan/query
extensions — the batch that completes method parity (v1 ⊆ v2).

1. Delegation for the real ops (scan_page/envelope/hybrid/sql/unified).
2. Wire equivalence: hybrid_search and execute_sql over the REAL adapter
   and the spec-driven generated ops (replacing never-executable
   ``_session`` code — the last live sites in the adapter).
3. Compositions: scan's cursor loop (bounds + repeat-cursor guard);
   search_iter's envelope paging + single-page fallback.
4. Compat surfaces: routing/selector/auth keep v1's response shapes over
   v2's single-adapter reality; force_protocol_switch rebuilds.
5. THE COMPLETION CENSUS: v1's public surface ⊆ v2's (program step 2 is
   DONE — the swap gate is parity + soak, per the TD).
"""

from http import HTTPStatus
from unittest.mock import MagicMock, patch

import pytest

import proximadb_sdk.unified_client_v2 as uc
from proximadb_sdk._generated.rest.types import Response
from proximadb_sdk.adapters.base import BaseProtocolAdapter
from proximadb_sdk.adapters.grpc_adapter import GrpcProtocolAdapter
from proximadb_sdk.adapters.rest_adapter import RestProtocolAdapter
from proximadb_sdk.config import Protocol
from proximadb_sdk.exceptions import (
    ProximaDBNotSupportedError,
    ServerError,
)
from proximadb_sdk.unified_client_v2 import ProximaDBClient

BATCH4_METHODS = frozenset(
    {
        "add_routing_rule",
        "execute_sql",
        "execute_unified_query",
        "force_protocol_switch",
        "get_auth_status",
        "get_protocol_metrics",
        "get_routing_stats",
        "get_selection_stats",
        "hybrid_search",
        "logout",
        "refresh_authentication",
        "reset_routing_metrics",
        "scan",
        "scan_page",
        "search_envelope",
        "search_iter",
    }
)
assert len(BATCH4_METHODS) == 16, "batch 4 ported exactly 16"


def _ok(parsed):
    return Response(status_code=HTTPStatus.OK, content=b"{}", headers={}, parsed=parsed)


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
# 1. Delegation
# ---------------------------------------------------------------------------


class TestDelegation:
    def test_scan_page(self, client, mock_adapter):
        mock_adapter.scan_page.return_value = ([], None)
        out = client.scan_page("c1", filter={"a": 1}, limit=5, cursor="k")
        mock_adapter.scan_page.assert_called_once_with(
            "c1",
            filter={"a": 1},
            limit=5,
            cursor="k",
            include_vectors=False,
            include_text=False,
        )
        assert out == ([], None)

    def test_hybrid_and_sql_and_unified(self, client, mock_adapter):
        mock_adapter.hybrid_search.return_value = {"results": []}
        out = client.hybrid_search("c1", "hello", [0.1], fusion_params={"k": 60})
        mock_adapter.hybrid_search.assert_called_once_with(
            "c1",
            "hello",
            [0.1],
            fusion_strategy="rrf",
            top_k=10,
            fusion_params={"k": 60},
        )
        assert out == {"results": []}

        mock_adapter.execute_sql.return_value = {"rows": []}
        out = client.execute_sql("SELECT 1", parameters=[1], collection="c1")
        mock_adapter.execute_sql.assert_called_once_with(
            "SELECT 1", parameters=[1], collection="c1"
        )
        assert out == {"rows": []}

        mock_adapter.execute_unified_query.return_value = [{"id": "r1"}]
        out = client.execute_unified_query("UQL", query_vector=[0.1])
        mock_adapter.execute_unified_query.assert_called_once_with(
            "UQL", query_vector=[0.1], fusion_strategy=None
        )
        assert out == [{"id": "r1"}]


# ---------------------------------------------------------------------------
# 2. Wire equivalence (REAL adapter; generated ops patched)
# ---------------------------------------------------------------------------


class TestRestWireEquivalence:
    @pytest.fixture
    def rest_client(self):
        adapter = RestProtocolAdapter(url="http://testserver:5678")
        with patch.object(uc, "create_adapter", return_value=adapter):
            c = ProximaDBClient(url="http://testserver:5678", protocol=Protocol.REST)
        return c

    def test_hybrid_search_body(self, rest_client):
        from proximadb_sdk._generated.rest.models.hybrid_search_response_200 import (
            HybridSearchResponse200,
        )

        op = "proximadb_sdk._generated.rest.api.hybrid.hybrid_search.sync_detailed"
        with patch(op) as mock_op:
            mock_op.return_value = _ok(
                HybridSearchResponse200.from_dict({"results": []})
            )
            out = rest_client.hybrid_search("c1", "hello", [0.1, 0.2], top_k=5)
        assert out["results"] == []
        body = mock_op.call_args.kwargs["body"].to_dict()
        assert body["collection"] == "c1"
        assert body["text_query"] == "hello"
        assert body["vector"] == [0.1, 0.2]
        assert body["fusion_strategy"] == "rrf"
        assert body["top_k"] == 5

    def test_hybrid_fusion_params_raise_honestly(self, rest_client):
        """The server's hybrid body has NO fusion_params field (it reads
        rrf_k/vector_weight; serde drops unknown keys) — shipping it would
        be a silent no-op. The adapter raises instead (v2: typed error)."""
        with pytest.raises(ProximaDBNotSupportedError):
            rest_client.hybrid_search("c1", "hello", [0.1], fusion_params={"k": 60})

    def test_execute_sql_body_and_error(self, rest_client):
        from proximadb_sdk._generated.rest.models.error_body import ErrorBody
        from proximadb_sdk._generated.rest.models.error_response import ErrorResponse
        from proximadb_sdk._generated.rest.models.sql_response import SqlResponse

        # parameters are rejected on REST: the server's SQL contract
        # deliberately does not advertise parameter binding ("callers must
        # not be offered a field that an adapter would silently ignore").
        with pytest.raises(ProximaDBNotSupportedError):
            rest_client.execute_sql("SELECT 1", parameters=[1])

        op = "proximadb_sdk._generated.rest.api.sql.execute_sql.sync_detailed"
        with patch(op) as mock_op:
            mock_op.return_value = _ok(
                SqlResponse.from_dict(
                    {
                        "columns": ["a"],
                        "column_types": ["f64"],
                        "rows": [{"a": 1}],
                        "rows_returned": 1,
                        "rows_scanned": 1,
                        "execution_time_ms": 1,
                        "request_id": "q1",
                    }
                )
            )
            out = rest_client.execute_sql("SELECT 1", collection="c1")
        assert out["rows"] == [{"a": 1}]
        body = mock_op.call_args.kwargs["body"].to_dict()
        assert body["query"] == "SELECT 1"
        assert body["collection"] == "c1"

        with patch(op) as mock_op:
            mock_op.return_value = Response(
                status_code=HTTPStatus.BAD_REQUEST,
                content=b"{}",
                headers={},
                parsed=ErrorResponse(
                    error=ErrorBody(code=400, message="bad sql", type_="invalid")
                ),
            )
            with pytest.raises(ServerError, match="bad sql"):
                rest_client.execute_sql("NOPE")

    def test_scan_and_envelope_delegate_to_protocol_client(self, rest_client):
        protocol_client = rest_client._adapter._client
        with patch.object(
            protocol_client,
            "scan_records",
            return_value=([], None),
        ) as scan_rec:
            out = rest_client.scan_page("c1", limit=5)
        # The protocol surface's kwarg is the SINGULAR include_vector —
        # pinned against the REAL protocol client (round 1's mock-only
        # test shipped the plural and crashed every production call).
        scan_rec.assert_called_once_with(
            "c1",
            filter=None,
            limit=5,
            cursor=None,
            include_vector=False,
            include_text=False,
        )
        assert out == ([], None)

    def test_search_envelope_reads_the_real_model(self, rest_client):
        """The protocol client returns models.SearchEnvelope (.items/.cursor/
        .has_more) — NOT a results/next_cursor dict. Pin the real shape."""
        from proximadb_sdk.models import SearchEnvelope, SearchResult

        env = SearchEnvelope(
            items=[SearchResult(id="r1", score=0.9, metadata={})],
            total=1,
            cursor=None,
            has_more=False,
        )
        with patch.object(
            rest_client._adapter._client, "search_envelope", return_value=env
        ) as se:
            out = rest_client.search_envelope("c1", [0.1])
        assert out is env
        se.assert_called_once_with(
            collection_id="c1",
            vector=[0.1],
            top_k=10,
            include_vectors=False,
            include_metadata=True,
        )


# ---------------------------------------------------------------------------
# 3. Compositions
# ---------------------------------------------------------------------------


class TestCompositions:
    def test_scan_follows_cursors_and_caps(self, client, mock_adapter):
        pages = [
            ([{"id": 1}], "k2"),
            ([{"id": 2}], "k3"),
            ([{"id": 3}], None),
        ]
        mock_adapter.scan_page.side_effect = pages
        out = client.scan("c1", page_size=2)
        assert [r["id"] for r in out] == [1, 2, 3]
        assert mock_adapter.scan_page.call_count == 3

        mock_adapter.scan_page.side_effect = [([{"id": 1}], "k1"), ([{"id": 2}], "k1")]
        out = client.scan("c1")  # repeating cursor -> guard stops the loop
        assert len(out) == 2

        mock_adapter.scan_page.side_effect = [
            ([{"id": 1}], "k2"),
            ([{"id": 2}], None),
        ]
        out = client.scan("c1", max_rows=1)
        assert out == [{"id": 1}]

    def test_search_iter_yields_items_over_real_envelopes(self, client):
        """v1's yield semantics: ITEMS, not pages — over the REAL
        models.SearchEnvelope shape (.items/.cursor/.has_more)."""
        from proximadb_sdk.models import SearchEnvelope, SearchResult

        def item(i):
            return SearchResult(id=f"r{i}", score=0.9, metadata={})

        envs = [
            SearchEnvelope(
                items=[item(1), item(2)], total=2, cursor="k2", has_more=True
            ),
            SearchEnvelope(items=[item(3)], total=3, cursor=None, has_more=False),
        ]
        client._adapter.search_envelope = MagicMock(side_effect=envs)
        got = list(client.search_iter("c1", [0.1]))
        assert [r.id for r in got] == ["r1", "r2", "r3"]

    def test_search_iter_stops_on_empty_page(self, client):
        from proximadb_sdk.models import SearchEnvelope

        envs = [
            SearchEnvelope(items=[], total=0, cursor="k2", has_more=True),
            SearchEnvelope(items=[], total=0, cursor="k3", has_more=True),
        ]
        client._adapter.search_envelope = MagicMock(side_effect=envs)
        assert list(client.search_iter("c1", [0.1])) == []
        assert client._adapter.search_envelope.call_count == 2  # empty page stops

    def test_search_iter_falls_back_to_single_search(self, client, mock_adapter):
        mock_adapter.search_envelope.side_effect = ProximaDBNotSupportedError("x")
        result = MagicMock(results=[MagicMock(id="r9")])
        mock_adapter.search.return_value = result
        got = list(client.search_iter("c1", [0.1]))
        assert [r.id for r in got] == ["r9"]
        # fallback forwards the include flags
        _, kwargs = mock_adapter.search.call_args
        assert kwargs["include_vectors"] is False

    def test_force_switch_failure_keeps_old_adapter(self, mock_adapter):
        with patch.object(uc, "create_adapter", return_value=mock_adapter):
            c = ProximaDBClient(url="http://testserver:5678", protocol=Protocol.REST)
        with patch.object(
            uc, "create_adapter", side_effect=ValueError("unknown protocol")
        ):
            with pytest.raises(ValueError):
                c.force_protocol_switch("bogus")
        # build-then-swap: the OLD adapter is untouched and still usable
        assert c._adapter is mock_adapter
        mock_adapter.close.assert_not_called()

    def test_force_switch_passes_resolved_config(self, mock_adapter):
        with patch.object(uc, "create_adapter", return_value=mock_adapter):
            c = ProximaDBClient(url="http://testserver:5678", protocol=Protocol.REST)
        new_adapter = MagicMock()
        new_adapter.protocol_name = "grpc"
        with patch.object(uc, "create_adapter", return_value=new_adapter) as mk:
            c.force_protocol_switch(Protocol.GRPC)
        kwargs = mk.call_args.kwargs
        # the resolved config rides through (endpoint/auth/timeout) instead
        # of ad-hoc url/api_key kwargs the gRPC adapter would ignore
        assert kwargs["config"] is c.config
        assert c._adapter is new_adapter
        mock_adapter.close.assert_called_once()


# ---------------------------------------------------------------------------
# 4. Compat surfaces
# ---------------------------------------------------------------------------


class TestCompatSurfaces:
    def test_routing_shapes_match_v1(self, client):
        assert client.get_routing_stats() == {"error": "Operation routing not enabled"}
        assert client.get_selection_stats() == {
            "error": "Intelligent protocol selection not enabled"
        }
        assert client.add_routing_rule(object()) is None  # warns, no raise
        assert client.reset_routing_metrics() is None
        metrics = client.get_protocol_metrics()
        assert metrics["protocol"] == "rest"
        assert "selection" in metrics

    def test_auth_shapes(self, client):
        status = client.get_auth_status()
        assert status == {"authenticated": False, "method": None}
        assert client.refresh_authentication() is False
        assert client.logout() is True

    def test_force_protocol_switch_rebuilds_adapter(self, mock_adapter):
        with patch.object(uc, "create_adapter", return_value=mock_adapter):
            c = ProximaDBClient(url="http://testserver:5678", protocol=Protocol.REST)
        new_adapter = MagicMock()
        new_adapter.protocol_name = "grpc"
        with patch.object(uc, "create_adapter", return_value=new_adapter) as mk2:
            c.force_protocol_switch(Protocol.GRPC)
        assert mk2.call_count == 1
        assert c._adapter is new_adapter
        mock_adapter.close.assert_called_once()


# ---------------------------------------------------------------------------
# 5. Transport ownership + THE COMPLETION CENSUS
# ---------------------------------------------------------------------------


class TestOwnershipAndCensus:
    def test_grpc_on_base_defaults_for_ops(self):
        for name in (
            "scan_page",
            "search_envelope",
            "execute_sql",
            "execute_unified_query",
        ):
            assert getattr(GrpcProtocolAdapter, name) is getattr(
                BaseProtocolAdapter, name
            ), f"{name} grew a gRPC impl"

    def test_rest_overrides_ops(self):
        for name in ("scan_page", "search_envelope", "execute_sql", "hybrid_search"):
            assert getattr(RestProtocolAdapter, name) is not getattr(
                BaseProtocolAdapter, name
            )

    def test_every_batch4_method_on_v2_and_base_where_applicable(self):
        for name in BATCH4_METHODS:
            assert callable(getattr(ProximaDBClient, name, None)), name

    def test_V1_PUBLIC_SURFACE_IS_NOW_A_SUBSET_OF_V2(self):
        """THE COMPLETION CENSUS (TD-SDK-3 program step 2): every v1 public
        method and the scan constants exist on v2. The 54-method gap is
        closed; the remaining swap gate is parity + soak per the TD."""
        from proximadb_sdk.unified_client import ProximaDBClient as V1Client

        v1_public = {
            name
            for name in dir(V1Client)
            if not name.startswith("_") and callable(getattr(V1Client, name))
        }
        missing = {
            name for name in v1_public if not callable(getattr(ProximaDBClient, name))
        }
        assert not missing, f"v1 methods still missing from v2: {missing}"
        assert ProximaDBClient.SCAN_MAX_PAGE == V1Client.SCAN_MAX_PAGE
        assert ProximaDBClient.SCAN_DEFAULT_MAX_ROWS == V1Client.SCAN_DEFAULT_MAX_ROWS
