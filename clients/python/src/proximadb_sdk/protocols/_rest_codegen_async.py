"""Spec-driven **async** request adapter over the generated REST transport.

This is the async analog of :mod:`_rest_codegen` (TD-126 Phase 4 / the
native-async follow-up). Where the sync adapter sources only the operation's
``method``/``url``/body-shape from the generated ``_get_kwargs`` (the facade then
executes the call through its own ``_make_request`` seam), the async path goes
one step further and wires the GENERATED ``asyncio_detailed`` endpoint functions
directly — exactly as the sync ``sync``/``sync_detailed`` functions are wired —
so the async client *also* gets the spec-governed response parsing
(``Response.parsed`` typed models) for free, with zero hand-written HTTP requests.

Each helper takes the generated ``Client`` (whose shared ``httpx2.AsyncClient`` is
owned by the facade) plus the facade-built body dict, builds the generated attrs
request model (reusing the same ``from_dict`` round-trip as the sync adapter so
the wire body matches the published spec), and ``await``s the generated
``asyncio_detailed``. It returns the generated ``Response`` so the facade can
read the typed ``.parsed`` model and coerce it into the SDK's public Pydantic
types — the same return types the sync facade produces.

Do NOT hand-roll REST request-building or response-parsing here (Core Directive
#15 / GEMINI #29): the request shape, URL, method, and response decoding are all
the generated client's job; this module only binds them onto a shared async
transport.
"""

from __future__ import annotations

from typing import Any, Union

from .._generated.rest.api.collections import (
    create_collection as _gen_create_collection,
)
from .._generated.rest.api.collections import (
    delete_collection as _gen_delete_collection,
)
from .._generated.rest.api.collections import get_collection as _gen_get_collection
from .._generated.rest.api.collections import list_collections as _gen_list_collections
from .._generated.rest.api.records import delete_record as _gen_delete_record
from .._generated.rest.api.records import get_record as _gen_get_record
from .._generated.rest.api.records import insert_records as _gen_insert_records
from .._generated.rest.api.records import scan_records as _gen_scan_records
from .._generated.rest.api.search import search_records as _gen_search_records
from .._generated.rest.client import AuthenticatedClient, Client
from .._generated.rest.models.create_collection_v2_request import (
    CreateCollectionV2Request,
)
from .._generated.rest.models.insert_records_request import InsertRecordsRequest
from .._generated.rest.models.scan_records_request import ScanRecordsRequest
from .._generated.rest.models.typed_search_request import TypedSearchRequest
from .._generated.rest.types import UNSET, Response
from ._rest_codegen import _from_dict

GenClient = Union[AuthenticatedClient, Client]


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


async def create_collection(client: GenClient, body: dict[str, Any]) -> Response[Any]:
    model = _from_dict(CreateCollectionV2Request, body)
    return await _gen_create_collection.asyncio_detailed(client=client, body=model)


async def get_collection(client: GenClient, collection_id: str) -> Response[Any]:
    return await _gen_get_collection.asyncio_detailed(collection_id, client=client)


async def delete_collection(client: GenClient, collection_id: str) -> Response[Any]:
    return await _gen_delete_collection.asyncio_detailed(collection_id, client=client)


async def list_collections(
    client: GenClient,
    *,
    limit: int | None = None,
    offset: int | None = None,
    include_stats: bool | None = None,
) -> Response[Any]:
    return await _gen_list_collections.asyncio_detailed(
        client=client,
        limit=UNSET if limit is None else limit,
        offset=UNSET if offset is None else offset,
        include_stats=UNSET if include_stats is None else include_stats,
    )


# ---------------------------------------------------------------------------
# Records / vectors
# ---------------------------------------------------------------------------


async def insert_records(
    client: GenClient, collection_id: str, body: dict[str, Any]
) -> Response[Any]:
    model = _from_dict(InsertRecordsRequest, body)
    return await _gen_insert_records.asyncio_detailed(
        collection_id=collection_id, client=client, body=model
    )


async def get_record(
    client: GenClient,
    collection_id: str,
    record_id: str,
    *,
    include_vector: bool | None = None,
    include_text: bool | None = None,
) -> Response[Any]:
    return await _gen_get_record.asyncio_detailed(
        collection_id=collection_id,
        record_id=record_id,
        client=client,
        include_vector=UNSET if include_vector is None else include_vector,
        include_text=UNSET if include_text is None else include_text,
    )


async def delete_record(
    client: GenClient, collection_id: str, record_id: str
) -> Response[Any]:
    return await _gen_delete_record.asyncio_detailed(
        collection_id=collection_id, record_id=record_id, client=client
    )


async def scan_records(
    client: GenClient, collection_id: str, body: dict[str, Any]
) -> Response[Any]:
    """Vector-free, metadata-filtered, cursor-paginated record scan (async).

    Mirrors the sync :func:`_rest_codegen.scan_records`: the body is the OpenAPI
    ``ScanRecordsRequest`` (``filter`` / ``limit`` / ``cursor`` / ``include_*``);
    the metadata ``filter`` is pushed into the scan predicate server-side.
    """
    model = _from_dict(ScanRecordsRequest, body)
    return await _gen_scan_records.asyncio_detailed(
        collection_id=collection_id, client=client, body=model
    )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


async def search_records(
    client: GenClient, collection_id: str, body: dict[str, Any]
) -> Response[Any]:
    model = _from_dict(TypedSearchRequest, body)
    return await _gen_search_records.asyncio_detailed(
        collection_id=collection_id, client=client, body=model
    )
