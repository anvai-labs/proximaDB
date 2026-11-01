"""TD-SDK-3 batch 2: the document-collections family ported onto the v2 client.

Mirrors the batch contract in the TD onto the document family:

1. Facade delegation — every ported method reaches the adapter with the
   kwargs v1's ``_call_document_adapter`` dispatch produces.
2. Facade validation — the protocol-independent ingest-record checks fire
   before any adapter contact.
3. Error translation — adapter ``NotImplementedError`` surfaces as the typed
   ``ProximaDBNotSupportedError``; ``allow_local_fallback`` is the only local
   path (v1's silent fallback on ANY adapter error is deliberately gone).
4. Wire equivalence — the FIXED REST adapter ingest path runs the shared
   ``_document_ingest`` protocol (same request body, same per-call headers,
   same outcome mapping as v1's canonical generated-op path).
5. Parity census ratchet — the ported set is pinned; it may only grow.
"""

from http import HTTPStatus
from unittest.mock import MagicMock, patch

import httpx
import pytest

import proximadb_sdk.unified_client_v2 as uc
from proximadb_sdk._generated.rest.models.error_body import ErrorBody
from proximadb_sdk._generated.rest.models.error_response import ErrorResponse
from proximadb_sdk._generated.rest.models.ingest_documents_response import (
    IngestDocumentsResponse,
)
from proximadb_sdk._generated.rest.models.ingested_record import IngestedRecord
from proximadb_sdk._generated.rest.types import Response
from proximadb_sdk.adapters.base import BaseProtocolAdapter
from proximadb_sdk.adapters.embedded_adapter import EmbeddedProtocolAdapter
from proximadb_sdk.adapters.grpc_adapter import GrpcProtocolAdapter
from proximadb_sdk.adapters.rest_adapter import RestProtocolAdapter
from proximadb_sdk.config import Protocol
from proximadb_sdk.exceptions import (
    NetworkError,
    ProximaDBError,
    ProximaDBNotSupportedError,
    ServerError,
)
from proximadb_sdk.unified_client_v2 import ProximaDBClient

# The lazy import inside the shared ingest protocol re-reads this attribute
# at call time, so patching the module attribute is what every consumer sees.
INGEST_MOD = "proximadb_sdk._generated.rest.api.documents.ingest_documents"

# ---------------------------------------------------------------------------
# Parity census (TD-SDK-3 program step 2 — the ratchet)
# ---------------------------------------------------------------------------

#: The document-collections methods ported in TD-SDK-3 batch 2. Batches
#: append their own pinned sets; the count is pinned so a removal is a
#: deliberate, visible edit.
PORTED_DOCUMENT_METHODS = frozenset(
    {
        "create_document_collection",
        "ingest_documents",
        "get_document",
        "query_documents",
        "update_document",
        "delete_document",
        "list_document_collections",
        "delete_document_collection",
    }
)
assert len(PORTED_DOCUMENT_METHODS) == 8, "TD-SDK-3 batch 2 ported exactly 8"


def _ok_response():
    parsed = IngestDocumentsResponse(
        mode="native", records=[IngestedRecord(dim=4, id="r1")]
    )
    return Response(status_code=HTTPStatus.OK, content=b"{}", headers={}, parsed=parsed)


def _err_response(message="bad dims", code=400):
    err = ErrorResponse(
        error=ErrorBody(code=code, message=message, type_="invalid_argument")
    )
    return Response(status_code=HTTPStatus(code), content=b"{}", headers={}, parsed=err)


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


class TestDocumentDelegation:
    def test_create_document_collection(self, client, mock_adapter):
        mock_adapter.create_document_collection.return_value = {"name": "docs"}
        result = client.create_document_collection("docs", {"enable_fulltext": True})
        mock_adapter.create_document_collection.assert_called_once_with(
            "docs", {"enable_fulltext": True}
        )
        assert result == {"name": "docs"}

    def test_create_document_collection_none_config(self, client, mock_adapter):
        client.create_document_collection("docs")
        mock_adapter.create_document_collection.assert_called_once_with("docs", None)

    def test_get_document_none_passthrough(self, client, mock_adapter):
        mock_adapter.get_document.return_value = None
        assert client.get_document("docs", "d1") is None
        mock_adapter.get_document.assert_called_once_with("docs", "d1", None)

    def test_get_document_with_projection(self, client, mock_adapter):
        mock_adapter.get_document.return_value = {"id": "d1"}
        out = client.get_document("docs", "d1", projection=["$.a"])
        mock_adapter.get_document.assert_called_once_with("docs", "d1", ["$.a"])
        assert out == {"id": "d1"}

    def test_query_documents(self, client, mock_adapter):
        mock_adapter.query_documents.return_value = {"documents": []}
        out = client.query_documents(
            "docs", filter={"a": 1}, projection=["$.a"], limit=5
        )
        mock_adapter.query_documents.assert_called_once_with(
            "docs", filter={"a": 1}, projection=["$.a"], limit=5
        )
        assert out == {"documents": []}

    def test_query_documents_defaults(self, client, mock_adapter):
        client.query_documents("docs")
        mock_adapter.query_documents.assert_called_once_with(
            "docs", filter=None, projection=None, limit=100
        )

    def test_update_document(self, client, mock_adapter):
        mock_adapter.update_document.return_value = {"success": True}
        out = client.update_document("docs", "d1", [{"op": "set", "path": "$.a"}])
        mock_adapter.update_document.assert_called_once_with(
            "docs", "d1", [{"op": "set", "path": "$.a"}]
        )
        assert out == {"success": True}

    @pytest.mark.parametrize("ret,expected", [(0, False), ({"deleted": True}, True)])
    def test_delete_document_bool_coerced(self, client, mock_adapter, ret, expected):
        mock_adapter.delete_document.return_value = ret
        assert client.delete_document("docs", "d1") is expected
        mock_adapter.delete_document.assert_called_once_with("docs", "d1")

    def test_list_document_collections(self, client, mock_adapter):
        mock_adapter.list_document_collections.return_value = [{"name": "docs"}]
        assert client.list_document_collections() == [{"name": "docs"}]
        mock_adapter.list_document_collections.assert_called_once_with()

    def test_delete_document_collection_bool_coerced(self, client, mock_adapter):
        mock_adapter.delete_document_collection.return_value = {"success": True}
        assert client.delete_document_collection("docs") is True
        mock_adapter.delete_document_collection.assert_called_once_with("docs")

    def test_ingest_documents_delegates(self, client, mock_adapter):
        mock_adapter.ingest_documents.return_value = {"mode": "native"}
        out = client.ingest_documents(
            "coll",
            [{"id": "r1", "text": "hi"}],
            tenant_id="t1",
            ingest_mode="streaming",
        )
        # allow_local_fallback is a facade concern — never forwarded.
        mock_adapter.ingest_documents.assert_called_once_with(
            "coll",
            [{"id": "r1", "text": "hi"}],
            tenant_id="t1",
            ingest_mode="streaming",
            embed_source="native",
        )
        assert out == {"mode": "native"}


# ---------------------------------------------------------------------------
# 2. Facade validation (before any adapter contact)
# ---------------------------------------------------------------------------


class TestDocumentValidation:
    def test_record_missing_id_raises_before_adapter(self, client, mock_adapter):
        with pytest.raises(ProximaDBError, match="missing 'id'"):
            client.ingest_documents("coll", [{"text": "hi"}])
        mock_adapter.ingest_documents.assert_not_called()

    def test_non_dict_record_raises_before_adapter(self, client, mock_adapter):
        with pytest.raises(ProximaDBError, match="dict or IngestDocument"):
            client.ingest_documents("coll", [42])
        mock_adapter.ingest_documents.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Error translation + the caller-opted local path
# ---------------------------------------------------------------------------


def test_adapter_not_implemented_becomes_typed_capability_error(client, mock_adapter):
    mock_adapter.create_document_collection.side_effect = NotImplementedError(
        "no documents"
    )
    with pytest.raises(ProximaDBNotSupportedError) as exc:
        client.create_document_collection("docs")
    assert isinstance(exc.value, ProximaDBError)
    assert "create_document_collection" in str(exc.value)


class TestLocalFallback:
    def _unsupported(self, mock_adapter):
        mock_adapter.ingest_documents.side_effect = NotImplementedError(
            "grpc has no document ingest"
        )

    def test_without_fallback_raises(self, client, mock_adapter):
        self._unsupported(mock_adapter)
        with pytest.raises(ProximaDBNotSupportedError):
            client.ingest_documents("coll", [{"id": "r1", "text": "hi"}])

    def test_with_fallback_uses_local_repo(self, client, mock_adapter):
        self._unsupported(mock_adapter)
        repo = MagicMock()
        repo.insert.return_value = MagicMock(id="r1", version=1)
        client._document_repository = repo

        out = client.ingest_documents(
            "coll", [{"id": "r1", "text": "hi"}], allow_local_fallback=True
        )

        assert out["mode"] == "local"
        assert out["records"] == [{"id": "r1", "version": 1}]
        repo.insert.assert_called_once_with("coll", {"id": "r1", "text": "hi"}, "r1")

    def test_fallback_never_masks_transport_errors(self, client, mock_adapter):
        mock_adapter.ingest_documents.side_effect = NetworkError("boom")
        with pytest.raises(NetworkError):
            client.ingest_documents(
                "coll", [{"id": "r1", "text": "hi"}], allow_local_fallback=True
            )


# ---------------------------------------------------------------------------
# 4. Wire equivalence — the FIXED REST adapter runs the shared ADR-041
#    protocol (the same one v1's canonical path uses)
# ---------------------------------------------------------------------------


class TestRestIngestWireEquivalence:
    @pytest.fixture
    def rest_client(self):
        """A v2 client whose adapter is the REAL REST adapter (offline)."""
        adapter = RestProtocolAdapter(
            url="http://testserver:5678", api_key="test-api-key-0123456789abcdef"
        )
        with patch.object(uc, "create_adapter", return_value=adapter):
            c = ProximaDBClient(url="http://testserver:5678", protocol=Protocol.REST)
        return c

    @patch(f"{INGEST_MOD}.sync_detailed")
    def test_request_body_matches_v1_shape(self, mock_sync, rest_client):
        mock_sync.return_value = _ok_response()

        rest_client.ingest_documents(
            "coll",
            [{"id": "r1", "text": "hello", "metadata": {"k": "v"}, "vector": [0.1]}],
        )

        body = mock_sync.call_args.kwargs["body"]
        assert body.to_dict() == {
            "records": [
                {"id": "r1", "text": "hello", "metadata": {"k": "v"}, "vector": [0.1]}
            ]
        }
        assert mock_sync.call_args.kwargs["collection_id"] == "coll"

    @patch(f"{INGEST_MOD}.sync_detailed")
    def test_per_call_headers_ride_the_evolved_client(self, mock_sync, rest_client):
        mock_sync.return_value = _ok_response()

        rest_client.ingest_documents(
            "coll", [{"id": "r1", "text": "hi"}], tenant_id="t1", ingest_mode="batch"
        )

        wire_client = mock_sync.call_args.kwargs["client"]
        assert wire_client._headers["X-Tenant-ID"] == "t1"
        assert wire_client._headers["X-Ingest-Mode"] == "batch"
        assert wire_client._headers["X-Embed-Source"] == "native"

    @patch(f"{INGEST_MOD}.sync_detailed")
    def test_per_call_headers_do_not_accumulate(self, mock_sync, rest_client):
        mock_sync.return_value = _ok_response()

        rest_client.ingest_documents("coll", [{"id": "r1"}], tenant_id="t1")
        rest_client.ingest_documents("coll", [{"id": "r2"}], tenant_id="t2")

        first = mock_sync.call_args_list[0].kwargs["client"]
        second = mock_sync.call_args_list[1].kwargs["client"]
        assert first._headers["X-Tenant-ID"] == "t1"
        assert second._headers["X-Tenant-ID"] == "t2"
        # The shared base client is never mutated (with_headers evolves).
        base = rest_client._adapter._get_generated_document_client()
        assert "X-Tenant-ID" not in base._headers

    @patch(f"{INGEST_MOD}.sync_detailed")
    def test_empty_embed_source_omits_header(self, mock_sync, rest_client):
        mock_sync.return_value = _ok_response()

        rest_client.ingest_documents("coll", [{"id": "r1"}], embed_source="")

        wire_client = mock_sync.call_args.kwargs["client"]
        assert "X-Embed-Source" not in wire_client._headers

    def test_auth_headers_survive_the_adapter_boundary(self, rest_client):
        base = rest_client._adapter._get_generated_document_client()
        expected = rest_client._adapter._client.config.get_base_headers()
        for key, value in expected.items():
            assert (
                base._headers.get(key) == value
            ), f"auth/base header {key!r} dropped at the adapter boundary"
        assert (
            base._headers.get("Authorization") == "Bearer test-api-key-0123456789abcdef"
        )

    @patch(f"{INGEST_MOD}.sync_detailed")
    def test_success_returns_parsed_dict(self, mock_sync, rest_client):
        mock_sync.return_value = _ok_response()
        out = rest_client.ingest_documents("coll", [{"id": "r1", "text": "hi"}])
        assert out["mode"] == "native"
        assert out["records"] == [{"dim": 4, "id": "r1"}]

    @patch(f"{INGEST_MOD}.sync_detailed")
    def test_error_response_mapped_to_server_error(self, mock_sync, rest_client):
        mock_sync.return_value = _err_response(message="bad dims", code=400)
        with pytest.raises(ServerError, match="bad dims") as exc:
            rest_client.ingest_documents("coll", [{"id": "r1", "text": "hi"}])
        assert exc.value.status_code == 400

    @patch(f"{INGEST_MOD}.sync_detailed")
    def test_transport_error_mapped_to_network_error(self, mock_sync, rest_client):
        mock_sync.side_effect = httpx.ConnectError("boom")
        with pytest.raises(NetworkError, match="transport error"):
            rest_client.ingest_documents("coll", [{"id": "r1", "text": "hi"}])


# ---------------------------------------------------------------------------
# Transport ownership: gRPC stays on the honest base defaults; REST and
# embedded override the whole family.
# ---------------------------------------------------------------------------


class TestTransportOwnership:
    def test_grpc_family_stays_on_base_defaults(self):
        for name in sorted(PORTED_DOCUMENT_METHODS):
            assert getattr(GrpcProtocolAdapter, name) is getattr(
                BaseProtocolAdapter, name
            ), f"{name} grew a gRPC impl — update the census + tests"

    def test_rest_and_embedded_override_the_whole_family(self):
        for adapter_cls in (RestProtocolAdapter, EmbeddedProtocolAdapter):
            for name in sorted(PORTED_DOCUMENT_METHODS):
                assert getattr(adapter_cls, name) is not getattr(
                    BaseProtocolAdapter, name
                ), f"{adapter_cls.__name__} dropped its {name} impl"


# ---------------------------------------------------------------------------
# 5. Parity census ratchet
# ---------------------------------------------------------------------------


class TestParityCensus:
    def test_every_ported_method_exists_on_v2(self):
        for name in PORTED_DOCUMENT_METHODS:
            assert callable(
                getattr(ProximaDBClient, name, None)
            ), f"TD-SDK-3 ratchet: {name} missing from the v2 client"

    def test_base_adapter_defines_the_whole_family(self):
        """The canonical protocol surface exists on the base class, so every
        adapter (real or mocked) accepts the full family."""
        for name in PORTED_DOCUMENT_METHODS:
            assert hasattr(
                BaseProtocolAdapter, name
            ), f"TD-SDK-3 ratchet: {name} missing from BaseProtocolAdapter"

    def test_ported_set_matches_v1_public_document_surface(self):
        """The pinned set IS v1's public document family (the 2026-11-01 AST
        census: 8 public methods). If this fails, v1's surface changed —
        re-census and update the pin deliberately."""
        from proximadb_sdk.unified_client import ProximaDBClient as V1Client

        v1_document_names = {
            name
            for name in dir(V1Client)
            if not name.startswith("_") and "document" in name
        }
        # Batch 2 claims the ENTIRE document family, so the sets must be equal.
        assert v1_document_names == set(PORTED_DOCUMENT_METHODS), (
            f"v1 document surface drifted: "
            f"extra={v1_document_names - set(PORTED_DOCUMENT_METHODS)} "
            f"missing={set(PORTED_DOCUMENT_METHODS) - v1_document_names}"
        )

    def test_v2_document_surface_equals_the_pinned_set(self):
        """Symmetric ratchet: v2's document-visible surface IS the pinned set —
        a document method added to v2 outside PORTED_DOCUMENT_METHODS
        (skipping the census pin) fails here rather than drifting silently."""
        v2_document_names = {
            name
            for name in dir(ProximaDBClient)
            if not name.startswith("_") and "document" in name
        }
        assert v2_document_names == set(PORTED_DOCUMENT_METHODS), (
            f"v2 document surface drifted: "
            f"extra={v2_document_names - set(PORTED_DOCUMENT_METHODS)} "
            f"missing={set(PORTED_DOCUMENT_METHODS) - v2_document_names}"
        )

    def test_v2_does_not_invent_document_names(self):
        from proximadb_sdk.unified_client import ProximaDBClient as V1Client

        v1_names = {name for name in dir(V1Client) if not name.startswith("_")}
        invented = set(PORTED_DOCUMENT_METHODS) - v1_names
        assert not invented, f"v2 invented document names not on v1: {invented}"
