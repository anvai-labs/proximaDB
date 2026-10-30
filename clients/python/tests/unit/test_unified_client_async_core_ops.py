"""Async core-ops tests for ProximaDBAsyncUnified.

These assert the native-async facade wires the GENERATED ``asyncio_detailed``
endpoint functions (spec-driven transport, TD-126) for each core op:

  * Every request hits the spec-derived method + URL the generated client
    builds (proving the generated endpoint module — not a hand-rolled route —
    was invoked).
  * The parsed result is coerced into the SAME public return type the sync
    sibling produces (Collection / list[Collection] / BatchResult /
    list[SearchResult] / DeleteResult / dict for get_vector).

We mock only the HTTP transport (``httpx.MockTransport`` injected into the
generated Client's shared AsyncClient), so the real generated request-building
and response-parsing run end to end.
"""

import httpx
import pytest

from proximadb_sdk.models import (
    BatchResult,
    Collection,
    CollectionConfig,
    DeleteResult,
    SearchResult,
)
from proximadb_sdk.unified_client_async import ProximaDBAsyncUnified


class _Recorder:
    """Captures each request and serves a canned JSON response per route."""

    def __init__(self, routes):
        # routes: list of (method, url_predicate, status, json_body)
        self.routes = routes
        self.seen = []  # (method, path)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.seen.append((request.method, request.url.path))
        for method, pred, status, body in self.routes:
            if request.method == method and pred(request.url.path):
                return httpx.Response(status, json=body)
        return httpx.Response(404, json={"error": "no mock route"})


async def _started_client(recorder: _Recorder) -> ProximaDBAsyncUnified:
    client = ProximaDBAsyncUnified(url="http://localhost:5678", protocol="rest")
    await client.astart()
    # Swap the generated client's shared async transport for our mock so the
    # real generated asyncio_detailed runs against canned responses.
    mock = httpx.AsyncClient(
        base_url="http://localhost:5678",
        transport=httpx.MockTransport(recorder.handler),
    )
    client._async_http = mock
    client._gen_client.set_async_httpx_client(mock)
    return client


@pytest.mark.asyncio
async def test_create_collection_hits_generated_endpoint():
    rec = _Recorder(
        [
            (
                "POST",
                lambda p: p == "/api/v2/collections",
                200,
                {
                    "collection_id": "products",
                    "content_revision": 1,
                    "content_revision_token": "test-incarnation:1",
                    "created_at": "now",
                    "dimension": 8,
                    "engine": "sst",
                    "name": "products",
                    "proxima_record_enabled": True,
                    "storage_class": "balanced",
                    "storage_class_status": {
                        "state": "ACTIVE",
                        "reason": "",
                        "policy_generation": 0,
                        "reserved_bytes": 0,
                        "resident_bytes": 0,
                        "covered_bytes": 0,
                    },
                },
            )
        ]
    )
    client = await _started_client(rec)
    try:
        out = await client.create_collection(
            "products", CollectionConfig(name="products", dimension=8)
        )
    finally:
        await client.aclose()

    assert isinstance(out, Collection)
    assert out.id == "products"
    assert out.config.dimension == 8
    # Spec-derived route from the generated create_collection endpoint.
    assert ("POST", "/api/v2/collections") in rec.seen


@pytest.mark.asyncio
async def test_get_collection_hits_generated_endpoint():
    rec = _Recorder(
        [
            (
                "GET",
                lambda p: p == "/api/v2/collections/products",
                200,
                {
                    "collection_id": "products",
                    "content_revision": 1,
                    "content_revision_token": "test-incarnation:1",
                    "created_at": "now",
                    "dimension": 16,
                    "distance_metric": "cosine",
                    "engine": "sst",
                    "index_specs": [],
                    "name": "products",
                    "proxima_record_enabled": True,
                    "storage_class": "balanced",
                    "storage_class_status": {
                        "state": "ACTIVE",
                        "reason": "",
                        "policy_generation": 0,
                        "reserved_bytes": 0,
                        "resident_bytes": 0,
                        "covered_bytes": 0,
                    },
                    "stats": {
                        "indexed_fields": 0,
                        "record_count": 3,
                        "storage_size_bytes": 0,
                        "text_field_count": 0,
                    },
                },
            )
        ]
    )
    client = await _started_client(rec)
    try:
        out = await client.get_collection("products")
    finally:
        await client.aclose()

    assert isinstance(out, Collection)
    assert out.id == "products"
    assert out.stats.vector_count == 3
    assert ("GET", "/api/v2/collections/products") in rec.seen


@pytest.mark.asyncio
async def test_list_collections_hits_generated_endpoint():
    rec = _Recorder(
        [
            (
                "GET",
                lambda p: p == "/api/v2/collections",
                200,
                {
                    "collections": [
                        {
                            "collection_id": "products",
                            "dimension": 8,
                            "engine": "sst",
                            "name": "products",
                            "proxima_record_enabled": True,
                            "storage_class": "balanced",
                            "storage_class_status": {
                                "state": "ACTIVE",
                                "reason": "",
                                "policy_generation": 0,
                                "reserved_bytes": 0,
                                "resident_bytes": 0,
                                "covered_bytes": 0,
                            },
                            "record_count": 5,
                        }
                    ],
                    "has_more": False,
                    "limit": 100,
                    "offset": 0,
                    "total": 1,
                },
            )
        ]
    )
    client = await _started_client(rec)
    try:
        out = await client.list_collections()
    finally:
        await client.aclose()

    assert isinstance(out, list)
    assert all(isinstance(c, Collection) for c in out)
    assert out[0].id == "products"
    assert out[0].stats.vector_count == 5
    assert ("GET", "/api/v2/collections") in rec.seen


@pytest.mark.asyncio
async def test_delete_collection_hits_generated_endpoint():
    rec = _Recorder(
        [
            (
                "DELETE",
                lambda p: p == "/api/v2/collections/products",
                200,
                {"collection_id": "products", "deleted": True, "success": True},
            )
        ]
    )
    client = await _started_client(rec)
    try:
        ok = await client.delete_collection("products")
    finally:
        await client.aclose()

    assert ok is True
    assert ("DELETE", "/api/v2/collections/products") in rec.seen


@pytest.mark.asyncio
async def test_insert_and_upsert_records_hit_generated_endpoint():
    rec = _Recorder(
        [
            (
                "POST",
                lambda p: p == "/api/v2/collections/products/records/batch",
                200,
                {
                    "errors": [],
                    "failed_count": 0,
                    "inserted_count": 2,
                    "inserted_ids": ["a", "b"],
                },
            )
        ]
    )
    client = await _started_client(rec)
    try:
        res = await client.insert_records(
            "products",
            [
                {"id": "a", "vector": [0.1, 0.2], "props": {"k": "v"}},
                {"id": "b", "vector": [0.3, 0.4]},
            ],
        )
        up = await client.upsert_records(
            "products", [{"id": "a", "vector": [0.1, 0.2]}]
        )
    finally:
        await client.aclose()

    assert isinstance(res, BatchResult)
    assert res.success == 2 and res.failed == 0 and res.total == 2
    assert isinstance(up, BatchResult)
    assert ("POST", "/api/v2/collections/products/records/batch") in rec.seen


@pytest.mark.asyncio
async def test_search_hits_generated_endpoint():
    rec = _Recorder(
        [
            (
                "POST",
                lambda p: p == "/api/v2/collections/products/search",
                200,
                {
                    "latency_ms": 1,
                    "request_id": "r1",
                    "results": [
                        {"id": "a", "props": {"k": "v"}, "score": 0.9},
                        {"id": "b", "props": {}, "score": 0.7},
                    ],
                },
            )
        ]
    )
    client = await _started_client(rec)
    try:
        results = await client.search("products", [0.1, 0.2], top_k=2)
    finally:
        await client.aclose()

    assert isinstance(results, list)
    assert all(isinstance(r, SearchResult) for r in results)
    assert [r.id for r in results] == ["a", "b"]
    assert results[0].rank == 1
    assert ("POST", "/api/v2/collections/products/search") in rec.seen


@pytest.mark.asyncio
async def test_get_vector_hits_generated_endpoint():
    rec = _Recorder(
        [
            (
                "GET",
                lambda p: p == "/api/v2/collections/products/records/a",
                200,
                {"id": "a", "props": {"k": "v"}, "vector": [0.1, 0.2]},
            )
        ]
    )
    client = await _started_client(rec)
    try:
        out = await client.get_vector("products", "a")
    finally:
        await client.aclose()

    assert isinstance(out, dict)
    assert out["id"] == "a"
    assert ("GET", "/api/v2/collections/products/records/a") in rec.seen


@pytest.mark.asyncio
async def test_delete_vectors_hit_generated_endpoint():
    rec = _Recorder(
        [
            (
                "DELETE",
                lambda p: p.startswith("/api/v2/collections/products/records/"),
                200,
                {"id": "x", "processing_time_us": 1, "success": True},
            )
        ]
    )
    client = await _started_client(rec)
    try:
        single = await client.delete_vector("products", "a")
        batch = await client.delete_vectors("products", ["b", "c"])
    finally:
        await client.aclose()

    assert isinstance(single, DeleteResult)
    assert single.success is True and single.deleted_count == 1
    assert isinstance(batch, DeleteResult)
    assert batch.deleted_count == 2 and batch.success is True
    deletes = [s for s in rec.seen if s[0] == "DELETE"]
    assert len(deletes) == 3  # 1 single + 2 batch


@pytest.mark.asyncio
async def test_context_manager_lifecycle():
    rec = _Recorder([])
    client = ProximaDBAsyncUnified(url="http://localhost:5678", protocol="rest")
    async with client:
        assert client._gen_client is not None
        assert client._async_http is not None
    # aclose ran on __aexit__: shared transport torn down.
    assert client._async_http is None
    assert client._gen_client is None


@pytest.mark.asyncio
async def test_methods_require_astart():
    client = ProximaDBAsyncUnified(url="http://localhost:5678", protocol="rest")
    with pytest.raises(RuntimeError):
        await client.get_collection("products")
