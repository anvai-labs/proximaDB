"""Shared protocol for ``ingest_documents`` (ADR-041) — one implementation.

The canonical document-ingest surface is protocol logic, not client logic:
record validation, ``IngestDocument`` request-body building, per-call header
shaping (``X-Embed-Source`` / ``X-Tenant-ID`` / ``X-Ingest-Mode``), and the
generated-op outcome mapping (transport → ``NetworkError``, error envelope →
``ServerError``). It is consumed by the v1 unified client, the v2 facade, and
the REST protocol adapter, so the three cannot drift apart — the exact
failure mode the TD-SDK-3 convergence program exists to remove.
"""

from __future__ import annotations

from typing import Any

import httpx

from ._generated.rest.api.documents import ingest_documents as _ingest_op
from ._generated.rest.errors import UnexpectedStatus
from ._generated.rest.models.error_response import ErrorResponse
from ._generated.rest.models.ingest_document import IngestDocument
from ._generated.rest.models.ingest_document_metadata import (
    IngestDocumentMetadata,
)
from ._generated.rest.models.ingest_documents_request import (
    IngestDocumentsRequest,
)
from ._generated.rest.types import UNSET
from .exceptions import NetworkError, ProximaDBError, ServerError


def validate_ingest_records(records: list[dict[str, Any] | IngestDocument]) -> None:
    """Validate ingest records with the protocol-independent checks.

    Raises:
        ProximaDBError: A record is not a ``dict``/``IngestDocument`` or is
            missing the required ``id`` key — before any wire activity.
    """
    for rec in records:
        if isinstance(rec, IngestDocument):
            continue
        if not isinstance(rec, dict):
            raise ProximaDBError(
                "ingest_documents record must be a dict or IngestDocument, "
                f"got {type(rec).__name__}"
            )
        if "id" not in rec:
            raise ProximaDBError("ingest_documents record missing 'id'")


def build_ingest_request_body(
    records: list[dict[str, Any] | IngestDocument],
) -> IngestDocumentsRequest:
    """Build the generated request body from ergonomic record dicts.

    ``IngestDocument`` records pass through unchanged; dict records map
    ``{id, text?, metadata?, vector?}`` onto the generated model with unset
    fields omitted.
    """
    validate_ingest_records(records)
    body_records: list[IngestDocument] = []
    for rec in records:
        if isinstance(rec, IngestDocument):
            body_records.append(rec)
            continue
        metadata = rec.get("metadata")
        body_records.append(
            IngestDocument(
                id=str(rec["id"]),
                text=rec.get("text", UNSET),
                vector=rec.get("vector", UNSET),
                metadata=(
                    IngestDocumentMetadata.from_dict(metadata)
                    if metadata is not None
                    else UNSET
                ),
            )
        )
    return IngestDocumentsRequest(records=body_records)


def ingest_per_call_headers(
    tenant_id: str | None,
    ingest_mode: str | None,
    embed_source: str | None,
) -> dict[str, str]:
    """Per-call headers for the ingest op (empty ``embed_source`` omits it)."""
    headers: dict[str, str] = {}
    if embed_source:
        headers["X-Embed-Source"] = embed_source
    if tenant_id is not None:
        headers["X-Tenant-ID"] = str(tenant_id)
    if ingest_mode is not None:
        headers["X-Ingest-Mode"] = str(ingest_mode)
    return headers


def insert_records_locally(
    repo: Any,
    collection_id: str,
    records: list[dict[str, Any] | IngestDocument],
) -> dict[str, Any]:
    """Caller-opted local fallback: store records in a ``DocumentRepository``.

    Both the v1 client and the v2 facade expose this only behind
    ``allow_local_fallback=True`` — never as a silent fallback for adapter
    errors — and share one implementation so the local shape cannot drift.
    """
    inserted = []
    for rec in records:
        doc = rec if isinstance(rec, dict) else rec.to_dict()
        doc_id = rec.get("id") if isinstance(rec, dict) else rec.id
        inserted.append(repo.insert(collection_id, doc, doc_id))
    return {
        "mode": "local",
        "records": [
            {
                "id": getattr(d, "id", None),
                "version": getattr(d, "version", None),
            }
            for d in inserted
        ],
    }


def execute_ingest_documents(
    client: Any,
    collection_id: str,
    body: IngestDocumentsRequest,
) -> dict[str, Any]:
    """Call the generated ingest op and map outcomes to SDK exceptions.

    Args:
        client: An evolved generated ``Client`` carrying the per-call headers.
        collection_id: Target collection id.
        body: The request body from :func:`build_ingest_request_body`.

    Returns:
        The parsed ``IngestDocumentsResponse`` as a dict.

    Raises:
        NetworkError: Transport-level failure.
        ServerError: Documented error envelope or an unexpected status.
    """
    try:
        # Attribute lookup at call time (not a frozen import) so the offline
        # test seam — patching the generated module's ``sync_detailed`` —
        # keeps working for every consumer of this shared protocol.
        response = _ingest_op.sync_detailed(
            collection_id=collection_id, client=client, body=body
        )
    except httpx.HTTPError as e:
        raise NetworkError(
            f"ingest_documents transport error: {e}", original_error=e
        ) from e
    except UnexpectedStatus as e:
        # raise_on_unexpected_status=True raises BEFORE the response is
        # returned, so undocumented statuses (e.g. HTTP 500) map to
        # ServerError here — the generated exception never escapes the SDK.
        raise ServerError(
            f"ingest_documents unexpected status: {e.status_code}",
            status_code=e.status_code,
        ) from e

    parsed = response.parsed
    if isinstance(parsed, ErrorResponse):
        detail = getattr(getattr(parsed, "error", None), "message", None)
        raise ServerError(
            "ingest_documents rejected by server" f"{f': {detail}' if detail else ''}",
            status_code=response.status_code,
        )
    if parsed is None:
        # raise_on_unexpected_status=True makes this unreachable in practice;
        # guard anyway so a future regression can't silently return None.
        raise ServerError(
            "ingest_documents returned no body",
            status_code=response.status_code,
        )
    return parsed.to_dict()
