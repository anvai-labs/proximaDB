"""Offline unit tests for ``ProximaDBClient.ingest_documents`` (ADR-041, P2).

The hand-written ``ingest_documents`` delegates to the GENERATED openapi op
(``_generated/rest/api/documents/ingest_documents.py``) — NOT the broken
``adapters/rest_adapter.py`` path whose ``insert_document`` silently falls back
to an in-memory repository on any error. These tests patch the generated
``sync_detailed`` so we assert request-shaping, per-call header passthrough,
header non-accumulation, and outcome mapping — with no server on the wire.
"""

from http import HTTPStatus
from unittest.mock import MagicMock, patch

import httpx
import pytest

from proximadb_sdk._generated.rest.models.error_body import ErrorBody
from proximadb_sdk._generated.rest.models.error_response import ErrorResponse
from proximadb_sdk._generated.rest.models.ingest_documents_response import (
    IngestDocumentsResponse,
)
from proximadb_sdk._generated.rest.models.ingested_record import IngestedRecord
from proximadb_sdk._generated.rest.types import Response
from proximadb_sdk.config import Protocol
from proximadb_sdk.exceptions import NetworkError, ProximaDBError, ServerError
from proximadb_sdk.unified_client import ProximaDBClient

# The shared _document_ingest protocol resolves sync_detailed as a module
# attribute at call time, so patching the module attribute is what every
# consumer (v1 client, v2 facade, REST adapter) sees.
INGEST_MOD = "proximadb_sdk._generated.rest.api.documents.ingest_documents"


def make_client(url="http://testserver:5678"):
    """A REST-mode client whose internals are inert MagicMocks (offline)."""
    c = ProximaDBClient(url=url, protocol="rest")
    c._adapter = MagicMock()
    c._client = MagicMock()
    c._active_protocol = Protocol.REST
    return c


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
# No server configured: raise vs. caller-opted local fallback
# ---------------------------------------------------------------------------


def test_no_server_without_fallback_raises():
    c = make_client()
    c._url = None
    c._generated_document_client = None
    with pytest.raises(ProximaDBError, match="server URL"):
        c.ingest_documents("coll", [{"id": "r1", "text": "hi"}])


def test_no_server_with_fallback_uses_local_repo():
    c = make_client()
    c._url = None
    c._generated_document_client = None
    repo = MagicMock()
    repo.insert.return_value = MagicMock(id="r1", version=1)
    c._get_document_repository = MagicMock(return_value=repo)

    out = c.ingest_documents(
        "coll", [{"id": "r1", "text": "hi"}], allow_local_fallback=True
    )

    assert out["mode"] == "local"
    assert out["records"] == [{"id": "r1", "version": 1}]
    repo.insert.assert_called_once_with("coll", {"id": "r1", "text": "hi"}, "r1")


# ---------------------------------------------------------------------------
# Per-call header passthrough + non-accumulation
# ---------------------------------------------------------------------------


@patch(f"{INGEST_MOD}.sync_detailed")
def test_headers_passed_via_with_headers(mock_sync):
    c = make_client()
    mock_sync.return_value = _ok_response()

    c.ingest_documents(
        "coll",
        [{"id": "r1", "text": "hi"}],
        tenant_id="tenant-a",
        ingest_mode="streaming",
    )

    client = mock_sync.call_args.kwargs["client"]
    assert client._headers["X-Tenant-ID"] == "tenant-a"
    assert client._headers["X-Ingest-Mode"] == "streaming"
    assert client._headers["X-Embed-Source"] == "native"


@patch(f"{INGEST_MOD}.sync_detailed")
def test_per_call_headers_do_not_accumulate(mock_sync):
    c = make_client()
    mock_sync.return_value = _ok_response()

    base = c._get_generated_document_client()
    assert base is not None
    base_headers_before = dict(base._headers)

    c.ingest_documents("coll", [{"id": "r1", "text": "a"}], tenant_id="t1")
    c.ingest_documents("coll", [{"id": "r2", "text": "b"}], tenant_id="t2")

    # Each evolved client carries only its own tenant header...
    client1 = mock_sync.call_args_list[0].kwargs["client"]
    client2 = mock_sync.call_args_list[1].kwargs["client"]
    assert client1._headers["X-Tenant-ID"] == "t1"
    assert client2._headers["X-Tenant-ID"] == "t2"
    # ...and the shared base client is never mutated (with_headers uses evolve).
    assert "X-Tenant-ID" not in base_headers_before
    assert base._headers == base_headers_before
    assert "X-Tenant-ID" not in base._headers


@patch(f"{INGEST_MOD}.sync_detailed")
def test_empty_embed_source_omits_header(mock_sync):
    c = make_client()
    mock_sync.return_value = _ok_response()

    c.ingest_documents("coll", [{"id": "r1", "text": "hi"}], embed_source="")

    client = mock_sync.call_args.kwargs["client"]
    assert "X-Embed-Source" not in client._headers


# ---------------------------------------------------------------------------
# Record dict -> IngestDocument conversion
# ---------------------------------------------------------------------------


@patch(f"{INGEST_MOD}.sync_detailed")
def test_record_dict_converted_to_request_body(mock_sync):
    c = make_client()
    mock_sync.return_value = _ok_response()

    c.ingest_documents(
        "coll",
        [{"id": "r1", "text": "hello", "metadata": {"k": "v"}, "vector": [0.1, 0.2]}],
    )

    body = mock_sync.call_args.kwargs["body"]
    assert body.to_dict() == {
        "records": [
            {"id": "r1", "text": "hello", "metadata": {"k": "v"}, "vector": [0.1, 0.2]}
        ]
    }


@patch(f"{INGEST_MOD}.sync_detailed")
def test_record_without_text_or_metadata_is_omitted(mock_sync):
    c = make_client()
    mock_sync.return_value = _ok_response()

    c.ingest_documents("coll", [{"id": "r1", "vector": [0.1]}])

    rec = mock_sync.call_args.kwargs["body"].to_dict()["records"][0]
    assert rec == {"id": "r1", "vector": [0.1]}
    assert "text" not in rec and "metadata" not in rec


@patch(f"{INGEST_MOD}.sync_detailed")
def test_ingestdocument_record_passed_through_unchanged(mock_sync):
    from proximadb_sdk._generated.rest.models.ingest_document import IngestDocument

    c = make_client()
    mock_sync.return_value = _ok_response()
    doc = IngestDocument(id="r9", text="raw")

    c.ingest_documents("coll", [doc])

    body = mock_sync.call_args.kwargs["body"]
    assert body.records[0] is doc


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_record_missing_id_raises():
    c = make_client()
    with pytest.raises(ProximaDBError, match="missing 'id'"):
        c.ingest_documents("coll", [{"text": "hi"}])


def test_non_dict_record_raises():
    c = make_client()
    with pytest.raises(ProximaDBError, match="dict or IngestDocument"):
        c.ingest_documents("coll", [42])


# ---------------------------------------------------------------------------
# Outcome mapping
# ---------------------------------------------------------------------------


@patch(f"{INGEST_MOD}.sync_detailed")
def test_success_returns_parsed_dict(mock_sync):
    c = make_client()
    mock_sync.return_value = _ok_response()

    out = c.ingest_documents("coll", [{"id": "r1", "text": "hi"}])

    assert out["mode"] == "native"
    assert out["records"] == [{"dim": 4, "id": "r1"}]


@patch(f"{INGEST_MOD}.sync_detailed")
def test_error_response_mapped_to_server_error(mock_sync):
    c = make_client()
    mock_sync.return_value = _err_response(message="bad dims", code=400)

    with pytest.raises(ServerError, match="bad dims") as exc:
        c.ingest_documents("coll", [{"id": "r1", "text": "hi"}])

    assert exc.value.status_code == 400


@patch(f"{INGEST_MOD}.sync_detailed")
def test_transport_error_mapped_to_network_error(mock_sync):
    c = make_client()
    mock_sync.side_effect = httpx.ConnectError("boom")

    with pytest.raises(NetworkError, match="transport error"):
        c.ingest_documents("coll", [{"id": "r1", "text": "hi"}])
