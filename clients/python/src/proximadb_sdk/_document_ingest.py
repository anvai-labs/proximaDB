"""Shared protocol for ``ingest_documents`` (ADR-041) — one implementation.

The canonical document-ingest surface is protocol logic, not client logic:
record validation, ``IngestDocument`` request-body building, per-call header
shaping (``X-Embed-Source`` / ``X-Tenant-ID`` / ``X-Ingest-Mode``), and the
generated-op outcome mapping (transport → ``NetworkError``, error envelope →
``ServerError``). It is consumed by the v1 unified client, the v2 facade, and
the REST protocol adapter, so the three cannot drift. The outcome-mapping
executor itself lives in ``_generated_op_exec`` (shared with the
timeseries/observability families since TD-SDK-3 batch 3) apart — the exact
failure mode the TD-SDK-3 convergence program exists to remove.
"""

from __future__ import annotations

from typing import Any

from ._generated.rest.api.documents import ingest_documents as _ingest_op
from ._generated.rest.models.ingest_document import IngestDocument
from ._generated.rest.models.ingest_documents_request import (
    IngestDocumentsRequest,
)
from ._generated_op_exec import execute_generated_op
from .exceptions import ProximaDBError


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
    fields omitted. Keys beyond the four known fields ride through the
    model's ``additional_properties`` so a rebuilt record (the v2 facade
    normalizes ``IngestDocument``→dict before delegating) is wire-equal to
    one passed straight through (the v1 path).
    """
    validate_ingest_records(records)
    body_records: list[IngestDocument] = []
    for rec in records:
        if isinstance(rec, IngestDocument):
            body_records.append(rec)
            continue
        # Build through from_dict: it maps the known fields (converting
        # ``metadata`` dicts). Unknown keys are DROPPED here by design —
        # the ingest surface denies unknown fields (TD-REC-1 policy,
        # additionalProperties:false in the spec): the contract is the
        # declared fields plus the props/metadata alias, and the server
        # rejects anything else with a 400 naming the field. A rebuilt
        # record (the v2 facade normalizes ``IngestDocument``→dict before
        # delegating) is therefore wire-equal to one passed straight
        # through (the v1 path).
        item: dict[str, Any] = {"id": str(rec["id"])}
        if "text" in rec:
            item["text"] = rec["text"]
        if "vector" in rec:
            item["vector"] = rec["vector"]
        if rec.get("metadata") is not None:
            item["metadata"] = rec["metadata"]
        body_records.append(IngestDocument.from_dict(item))
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
    # Attribute lookup at call time (not a frozen import) so the offline
    # test seam — patching the generated module's ``sync_detailed`` —
    # keeps working for every consumer of this shared protocol.
    return execute_generated_op(
        "ingest_documents",
        lambda: _ingest_op.sync_detailed(
            collection_id=collection_id, client=client, body=body
        ),
    )
