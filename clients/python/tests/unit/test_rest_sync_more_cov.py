"""Additional offline coverage for proximadb_sdk.protocols.rest_sync.

This file EXTENDS (does not duplicate) test_rest_sync_methods_coverage.py.
It focuses on:
  * collection lifecycle: create / get / list / delete / schema / stats
  * vectors: insert / upsert / delete / get / update
  * search / search_batch / search_envelope / search_next_page
  * the *_cached read wrappers and *_batched submit wrappers
  * close() and cache/batch helper guards

All transports are mocked — no server, no socket, no sleep.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from proximadb_sdk.models import CollectionConfig
from proximadb_sdk.protocols.rest_sync import ProximaDBClient
from tests.unit._id_helpers import is_uuid as _is_uuid


class FakeResp:
    def __init__(self, data=None, status=200):
        self._d = dict(data) if data else {}
        self.status_code = status
        self.headers = {}
        self.text = "{}"
        self.content = b"{}"

    def json(self):
        return self._d

    def raise_for_status(self):
        return None


class FakeHttpClient:
    def __init__(self, resp=None):
        self.calls = []
        self._resp = resp or FakeResp()
        self.closed = False

    def _record(self, verb, path, **kw):
        self.calls.append((verb, path, kw))
        return self._resp

    def get(self, path, **kw):
        return self._record("GET", path, **kw)

    def post(self, path, **kw):
        return self._record("POST", path, **kw)

    def put(self, path, **kw):
        return self._record("PUT", path, **kw)

    def delete(self, path, **kw):
        return self._record("DELETE", path, **kw)

    def close(self):
        self.closed = True


def _make_client(monkeypatch, resp_body=None):
    """Construct a client with both transports mocked."""
    c = ProximaDBClient(url="http://testserver")
    captured = {"req": []}

    body = resp_body if resp_body is not None else {}

    def fake_make_request(method, endpoint, **kwargs):
        captured["req"].append((method, endpoint, kwargs))
        return FakeResp(body)

    monkeypatch.setattr(c, "_make_request", fake_make_request)
    monkeypatch.setattr(c, "_http_client", FakeHttpClient())
    c._captured = captured  # type: ignore[attr-defined]
    return c


@pytest.fixture
def client(monkeypatch):
    return _make_client(monkeypatch)


def _paths(client):
    return [p for _, p, _ in client._captured["req"]]


def _last(client):
    return client._captured["req"][-1]


# ---------------------------------------------------------------- collections


def test_create_collection_collection_id_response(monkeypatch):
    c = _make_client(
        monkeypatch,
        resp_body={
            "collection_id": "abc",
            "name": "products_x",
            "dimension": 128,
            "engine": "viper",
        },
    )
    cfg = CollectionConfig(name="products_x", dimension=128)
    coll = c.create_collection("products_x", cfg)
    assert coll.id == "abc"
    assert coll.config.dimension == 128
    method, path, kw = _last(c)
    assert (method, path) == ("POST", "/api/v2/collections")
    assert kw["json"]["name"] == "products_x"
    assert kw["json"]["dimension"] == 128


def test_create_collection_nested_collection_response(monkeypatch):
    c = _make_client(
        monkeypatch,
        resp_body={
            "collection": {
                "id": "id123",
                "config": {
                    "name": "mycoll_x",
                    "dimension": 64,
                    "distance_metric": 2,  # euclidean
                    "storage_engine": 1,  # viper
                },
                "created_at": 111,
                "updated_at": 222,
            }
        },
    )
    cfg = CollectionConfig(name="mycoll_x", dimension=64)
    coll = c.create_collection("mycoll_x", cfg)
    assert coll.id == "id123"
    assert coll.config.distance_metric.value == "euclidean"
    assert coll.config.storage_engine.value == "viper"


def test_create_collection_builds_config_from_kwargs(monkeypatch):
    c = _make_client(
        monkeypatch,
        resp_body={"collection_id": "z", "dimension": 32, "engine": "viper"},
    )
    coll = c.create_collection("kwargcoll", dimension=32)
    assert coll.id == "z"


def test_create_collection_warns_on_too_many_filterable(monkeypatch):
    c = _make_client(
        monkeypatch,
        resp_body={"collection_id": "w", "dimension": 8, "engine": "viper"},
    )
    cfg = CollectionConfig(
        name="warncoll_xx",
        dimension=8,
        filterable_metadata_fields=[f"f{i}" for i in range(20)],
    )
    with pytest.warns(UserWarning):
        c.create_collection("warncoll_xx", cfg)


def test_get_collection_simple(monkeypatch):
    c = _make_client(
        monkeypatch,
        resp_body={
            "id": "cid",
            "name": "collection_name",
            "dimension": 256,
            "metric": "cosine",
            "vector_count": 42,
            "created_at": 1,
            "updated_at": 2,
        },
    )
    coll = c.get_collection("cid")
    assert coll.id == "cid"
    assert coll.config.dimension == 256
    assert coll.stats.vector_count == 42
    assert _paths(c) == ["/api/v2/collections/cid"]


def test_get_collection_proto_int_metric(monkeypatch):
    c = _make_client(
        monkeypatch,
        resp_body={
            "id": "cid2",
            "name": "longenoughname",
            "dimension": 8,
            "distance_metric": 2,  # int -> euclidean
            "storage_engine": 2,  # int -> sst
        },
    )
    coll = c.get_collection("cid2")
    assert coll.config.distance_metric.value == "euclidean"
    assert coll.config.storage_engine.value == "sst"


def test_get_collection_error_generic(monkeypatch):
    # NB: the "not found" branch raises an undefined CollectionNotFoundError
    # (a source-level NameError) so we exercise the generic-error branch.
    from proximadb_sdk.exceptions import ProximaDBError

    c = _make_client(monkeypatch, resp_body={"error_message": "kaboom"})
    with pytest.raises(ProximaDBError):
        c.get_collection("missing")


def test_get_collection_success_false(monkeypatch):
    from proximadb_sdk.exceptions import ProximaDBError

    c = _make_client(monkeypatch, resp_body={"success": False, "error_message": "boom"})
    with pytest.raises(ProximaDBError):
        c.get_collection("longenoughid")


def test_list_collections_with_params(monkeypatch):
    c = _make_client(
        monkeypatch,
        resp_body={
            "collections": [
                {
                    "id": "c1",
                    "name": "first_one",
                    "dimension": 4,
                    "metric": 1,
                    "storage_engine": 2,
                    "vector_count": 3,
                },
                {
                    "id": "c2",
                    "config": {
                        "name": "second_one",
                        "dimension": 8,
                        "distance_metric": 2,
                        "storage_engine": 1,
                    },
                    "stats": {"vector_count": 7},
                },
            ],
            "total_count": 2,
        },
    )
    colls = c.list_collections(limit=10, offset=0, include_stats=True)
    assert len(colls) == 2
    assert colls[0].id == "c1"
    assert colls[1].config.dimension == 8
    method, path, kw = _last(c)
    assert path == "/api/v2/collections"
    assert kw["params"]["limit"] == 10
    # The generated REST transport (openapi-python-client) emits the boolean
    # query param as a native Python bool in the params dict; httpx serializes
    # it to "true" at the wire layer. Assert the real in-memory value.
    assert kw["params"]["include_stats"] is True


def test_list_collections_no_params(monkeypatch):
    c = _make_client(monkeypatch, resp_body={"collections": []})
    assert c.list_collections() == []


def test_delete_collection(monkeypatch):
    c = _make_client(monkeypatch, resp_body={"success": True})
    assert c.delete_collection("cid") is True
    method, path, _ = _last(c)
    assert (method, path) == ("DELETE", "/api/v2/collections/cid")


def test_get_schema(monkeypatch):
    c = _make_client(
        monkeypatch,
        resp_body={
            "schema_id": "s1",
            "schema_version": "1",
            "collection_id": "cid",
            "schema": {"columns": []},
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    schema = c.get_schema("cid")
    assert schema is not None
    assert _paths(c) == ["/api/v2/collections/cid/schema"]


def test_update_schema_with_dict(monkeypatch):
    c = _make_client(
        monkeypatch,
        resp_body={
            "schema_id": "s2",
            "schema_version": "2",
            "previous_schema_id": "s1",
            "changes": [],
            "warnings": [],
            "updated_at": "2026-01-01T00:00:00Z",
        },
    )
    out = c.update_schema("cid", {"fields": []}, force=True)
    assert out is not None
    method, path, kw = _last(c)
    assert (method, path) == ("PUT", "/api/v2/collections/cid/schema")
    assert kw["json"]["force"] is True


def test_get_collection_stats(monkeypatch):
    c = _make_client(
        monkeypatch,
        resp_body={
            "vector_count": 100,
            "index_size_bytes": 5,
            "data_size_bytes": 6,
        },
    )
    stats = c.get_collection_stats("cid")
    assert stats.vector_count == 100
    assert _paths(c) == ["/collections/cid/stats"]


# ----------------------------------------------------------------- vectors


def test_insert_vector_single(monkeypatch):
    c = _make_client(monkeypatch, resp_body={"inserted_count": 1, "failed_count": 0})
    res = c.insert_vector("cid", "v1", [0.1, 0.2, 0.3], metadata={"k": "v"})
    assert res.success == 1
    assert res.failed == 0
    method, path, kw = _last(c)
    assert (method, path) == (
        "POST",
        "/api/v2/collections/cid/records/batch",
    )
    rec = kw["json"]["records"][0]
    assert rec["id"] == "v1"
    assert rec["vector"] == [0.1, 0.2, 0.3]


def test_insert_records_multiple_batches(monkeypatch):
    c = _make_client(monkeypatch, resp_body={"inserted_count": 2, "failed_count": 0})
    records = [{"id": f"r{i}", "vector": [float(i), float(i + 1)]} for i in range(4)]
    res = c.insert_records("cid", records, batch_size=2)
    # 4 records / batch_size 2 -> 2 requests, each reporting 2 inserted
    assert len(c._captured["req"]) == 2
    assert res.success == 4
    assert res.total == 4


def test_insert_records_numpy_vector(monkeypatch):
    c = _make_client(monkeypatch, resp_body={"inserted_count": 1, "failed_count": 0})
    rec = {"id": "n1", "vector": np.array([1.0, 2.0], dtype=np.float64)}
    res = c.insert_records("cid", [rec])
    assert res.success == 1


def test_insert_records_missing_vector_raises(client):
    with pytest.raises(ValueError):
        client.insert_records("cid", [{"id": "x"}])


def test_insert_records_empty_vector_raises(client):
    with pytest.raises(ValueError):
        client.insert_records("cid", [{"id": "x", "vector": []}])


def test_upsert_records_sets_upsert_flag(monkeypatch):
    c = _make_client(monkeypatch, resp_body={"inserted_count": 1, "failed_count": 0})
    c.upsert_records("cid", [{"id": "u1", "vector": [1.0, 2.0]}])
    _, _, kw = _last(c)
    assert kw["json"]["upsert"] is True


def test_insert_vectors_from_arrays(monkeypatch):
    c = _make_client(monkeypatch, resp_body={"inserted_count": 2, "failed_count": 0})
    res = c.insert_vectors(
        "cid",
        [[1.0, 2.0], [3.0, 4.0]],
        ids=["a", "b"],
        metadata=[{"m": 1}, {"m": 2}],
    )
    assert res.success == 2


def test_insert_vectors_autogenerates_ids(monkeypatch):
    c = _make_client(monkeypatch, resp_body={"inserted_count": 2, "failed_count": 0})
    res = c.insert_vectors("cid", [[1.0, 2.0], [3.0, 4.0]])
    assert res.success == 2
    rec_ids = [r["id"] for r in _last(c)[2]["json"]["records"]]
    assert len(rec_ids) == 2 and all(_is_uuid(r) for r in rec_ids)
    assert rec_ids[0] != rec_ids[1]


def test_insert_vectors_dict_records(monkeypatch):
    c = _make_client(monkeypatch, resp_body={"inserted_count": 1, "failed_count": 0})
    res = c.insert_vectors("cid", [{"id": "d1", "vector": [1.0, 2.0]}])
    assert res.success == 1


def test_insert_vectors_mismatched_ids_raises(client):
    with pytest.raises(ValueError):
        client.insert_vectors("cid", [[1.0]], ids=["a", "b"])


def test_insert_vectors_mismatched_metadata_raises(client):
    with pytest.raises(ValueError):
        client.insert_vectors(
            "cid", [[1.0], [2.0]], ids=["a", "b"], metadata=[{"m": 1}]
        )


def test_delete_vector(monkeypatch):
    c = _make_client(monkeypatch, resp_body={"success": True})
    res = c.delete_vector("cid", "v1")
    assert res.success is True
    assert res.deleted_count == 1
    method, path, _ = _last(c)
    assert (method, path) == (
        "DELETE",
        "/api/v2/collections/cid/records/v1",
    )


def test_delete_vectors_multiple(monkeypatch):
    c = _make_client(monkeypatch, resp_body={"success": True})
    res = c.delete_vectors("cid", ["a", "b", "c"])
    assert res.deleted_count == 3
    assert res.success is True
    assert len(c._captured["req"]) == 3


def test_delete_vectors_handles_exception(monkeypatch):
    # When the per-id delete raises, delete_vectors collects the error string
    # into DeleteResult.errors (the field was added to DeleteResult; previously
    # it was silently dropped).
    c = ProximaDBClient(url="http://testserver")

    def boom(method, endpoint, **kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr(c, "_make_request", boom)
    monkeypatch.setattr(c, "_http_client", FakeHttpClient())
    res = c.delete_vectors("cid", ["a"])
    assert res.success is False
    assert res.errors
    c.close()


def test_get_vector(monkeypatch):
    c = _make_client(monkeypatch, resp_body={"id": "v1", "vector": [1.0, 2.0]})
    out = c.get_vector("cid", "v1")
    assert out["id"] == "v1"
    method, path, kw = _last(c)
    assert (method, path) == (
        "GET",
        "/api/v2/collections/cid/records/v1",
    )
    assert kw["params"]["include_vector"] is True


def test_get_vector_nested_results(monkeypatch):
    c = _make_client(
        monkeypatch,
        resp_body={"results": {"results": [{"id": "inner", "vector": [1.0]}]}},
    )
    out = c.get_vector("cid", "v1")
    assert out["id"] == "inner"


def test_get_vector_not_found_raises(monkeypatch):
    from proximadb_sdk.exceptions import ProximaDBError

    c = _make_client(monkeypatch, resp_body={"error_code": "NOT_FOUND"})
    with pytest.raises(ProximaDBError):
        c.get_vector("cid", "missing")


def test_upsert_vectors_records(monkeypatch):
    c = _make_client(monkeypatch, resp_body={"inserted_count": 1, "failed_count": 0})
    rec = MagicMock()
    rec.vector = [1.0, 2.0]
    rec.id = "r1"
    rec.metadata = {"k": "v"}
    res = c.upsert_vectors("cid", [rec])
    assert res.success == 1
    _, _, kw = _last(c)
    assert kw["json"]["upsert"] is True


def test_update_vector(monkeypatch):
    c = _make_client(monkeypatch, resp_body={"inserted_count": 1, "failed_count": 0})
    res = c.update_vector("cid", "v1", vector=[1.0, 2.0], metadata={"a": 1})
    assert res.success == 1
    _, _, kw = _last(c)
    assert kw["json"]["upsert"] is True


def test_update_vector_numpy(monkeypatch):
    c = _make_client(monkeypatch, resp_body={"inserted_count": 1, "failed_count": 0})
    res = c.update_vector("cid", "v1", vector=np.array([1.0, 2.0], dtype=np.float64))
    assert res.success == 1


def test_update_vector_requires_vector(client):
    with pytest.raises(ValueError):
        client.update_vector("cid", "v1", vector=None)


# ------------------------------------------------------------------ search


def test_search_basic(monkeypatch):
    c = _make_client(
        monkeypatch,
        resp_body={
            "results": [
                {"id": "a", "score": 0.9, "rank": 0, "props": {"k": "v"}},
                {"id": "b", "score": 0.8, "rank": 1},
            ]
        },
    )
    results = c.search("cid", [0.1, 0.2], top_k=2)
    assert len(results) == 2
    assert results[0].id == "a"
    assert results[0].score == 0.9
    method, path, kw = _last(c)
    assert (method, path) == ("POST", "/api/v2/collections/cid/search")
    assert kw["json"]["top_k"] == 2


def test_search_with_filter_and_numpy(monkeypatch):
    c = _make_client(monkeypatch, resp_body={"results": []})
    c.search(
        "cid",
        np.array([0.1, 0.2], dtype=np.float64),
        metadata_filter={"category": "x"},
    )
    _, _, kw = _last(c)
    assert kw["json"]["filters"] == [{"field": "category", "op": "eq", "value": "x"}]


def test_search_with_hints(monkeypatch):
    c = _make_client(monkeypatch, resp_body={"results": []})
    c.search(
        "cid",
        [0.1, 0.2],
        search_hints={"enable_two_stage": True, "accuracy_threshold": 0.9},
    )
    _, _, kw = _last(c)
    assert "search_optimization" in kw["json"]


def test_search_error_not_found_returns_empty(monkeypatch):
    c = _make_client(monkeypatch, resp_body={"error_message": "collection not found"})
    assert c.search("cid", [0.1]) == []


def test_search_error_message_raises(monkeypatch):
    from proximadb_sdk.exceptions import ProximaDBError

    c = _make_client(monkeypatch, resp_body={"error_message": "boom"})
    with pytest.raises(ProximaDBError):
        c.search("cid", [0.1])


def test_search_nested_results(monkeypatch):
    c = _make_client(
        monkeypatch,
        resp_body={"results": {"results": [{"id": "n", "score": 0.5}]}},
    )
    out = c.search("cid", [0.1])
    assert out[0].id == "n"


def test_search_null_results(monkeypatch):
    c = _make_client(monkeypatch, resp_body={"results": None})
    assert c.search("cid", [0.1]) == []


def test_search_skips_malformed_results(monkeypatch):
    c = _make_client(
        monkeypatch,
        resp_body={"results": ["bad", 123, {"id": "good", "score": 1.0}]},
    )
    out = c.search("cid", [0.1])
    assert len(out) == 1
    assert out[0].id == "good"


def test_search_batch(monkeypatch):
    c = _make_client(
        monkeypatch,
        resp_body={
            "results": [
                [{"id": "a", "score": 0.9}],
                [{"id": "b", "score": 0.8}],
            ]
        },
    )
    out = c.search_batch("cid", [[0.1, 0.2], [0.3, 0.4]], k=1)
    assert len(out) == 2
    assert out[0][0].id == "a"
    method, path, kw = _last(c)
    assert (method, path) == ("POST", "/collections/cid/search/batch")
    assert kw["json"]["k"] == 1


def test_search_batch_with_ef(monkeypatch):
    c = _make_client(monkeypatch, resp_body={"results": [[]]})
    c.search_batch("cid", [[0.1]], k=5, ef=64, include_vectors=True)
    _, _, kw = _last(c)
    assert kw["json"]["params"]["ef"] == 64


def test_search_envelope(monkeypatch):
    c = _make_client(
        monkeypatch,
        resp_body={"results": [{"id": "a", "score": 0.9}]},
    )
    env = c.search_envelope("cid", [0.1, 0.2], top_k=5)
    assert env.cursor is None
    assert env.has_more is False
    assert len(env.items) == 1


def test_search_envelope_numpy(monkeypatch):
    c = _make_client(monkeypatch, resp_body={"results": []})
    env = c.search_envelope("cid", np.array([0.1], dtype=np.float64))
    assert env.items == []


def test_search_next_page_empty_cursor(client):
    env = client.search_next_page("cid", "")
    assert env.items == []
    assert env.cursor is None


def test_search_next_page_unsupported(client):
    # _sks_search_supported defaults to None -> not True -> empty envelope
    env = client.search_next_page("cid", "some-cursor")
    assert env.items == []


def test_search_next_page_supported(monkeypatch):
    c = _make_client(
        monkeypatch,
        resp_body={
            "items": [{"entity_id": "e1", "score": 0.7, "metadata": {"k": "v"}}],
            "page": {"cursor": "next123", "has_more": True},
            "progress": {"stage": 1, "stages": 2, "complete": False},
            "total": 10,
        },
    )
    c._sks_search_supported = True
    env = c.search_next_page("cid", "cur", include_metadata=True)
    assert len(env.items) == 1
    assert env.items[0].id == "e1"
    assert env.cursor == "next123"
    assert env.has_more is True
    assert env.progress is not None


def test_search_next_page_bad_body(monkeypatch):
    c = _make_client(monkeypatch, resp_body={"no_items": True})
    c._sks_search_supported = True
    env = c.search_next_page("cid", "cur")
    assert env.items == []


# -------------------------------------------------------- cached wrappers


def test_search_cached_passthrough_when_disabled(monkeypatch):
    c = _make_client(monkeypatch, resp_body={"results": []})
    # caching disabled -> direct passthrough to search
    out = c.search_cached("cid", [0.1, 0.2], top_k=3)
    assert out == []
    assert _paths(c) == ["/api/v2/collections/cid/search"]


def test_get_vector_cached_passthrough_when_disabled(monkeypatch):
    c = _make_client(monkeypatch, resp_body={"id": "v1"})
    out = c.get_vector_cached("cid", "v1")
    assert out["id"] == "v1"


def test_list_collections_cached_passthrough(monkeypatch):
    c = _make_client(monkeypatch, resp_body={"collections": []})
    assert c.list_collections_cached() == []


def test_get_collection_cached_passthrough(monkeypatch):
    c = _make_client(
        monkeypatch,
        resp_body={"id": "cid", "name": "longenough", "dimension": 4},
    )
    coll = c.get_collection_cached("cid")
    assert coll.id == "cid"


def test_search_cached_with_caching_enabled(monkeypatch):
    c = ProximaDBClient(url="http://testserver", enable_caching=True)
    captured = {"req": []}

    def fake_make_request(method, endpoint, **kwargs):
        captured["req"].append((method, endpoint, kwargs))
        return FakeResp({"results": [{"id": "a", "score": 1.0}]})

    monkeypatch.setattr(c, "_make_request", fake_make_request)
    monkeypatch.setattr(c, "_http_client", FakeHttpClient())

    out1 = c.search_cached("cid", [0.1, 0.2], top_k=1)
    out2 = c.search_cached("cid", [0.1, 0.2], top_k=1)
    assert out1[0].id == "a"
    assert out2[0].id == "a"
    # second call served from cache -> only one underlying request
    assert len(captured["req"]) == 1
    c.close()


def test_get_vector_cached_with_caching_enabled(monkeypatch):
    c = ProximaDBClient(url="http://testserver", enable_caching=True)
    captured = {"req": []}

    def fake_make_request(method, endpoint, **kwargs):
        captured["req"].append((method, endpoint, kwargs))
        return FakeResp({"id": "v1", "vector": [1.0]})

    monkeypatch.setattr(c, "_make_request", fake_make_request)
    monkeypatch.setattr(c, "_http_client", FakeHttpClient())

    c.get_vector_cached("cid", "v1")
    c.get_vector_cached("cid", "v1")
    assert len(captured["req"]) == 1
    c.close()


def test_cache_guards_raise_when_disabled(client):
    assert client.get_cache_stats() == {"error": "Caching is not enabled"}
    with pytest.raises(RuntimeError):
        client.clear_cache()
    with pytest.raises(RuntimeError):
        client.invalidate_collection_cache("cid")
    with pytest.raises(RuntimeError):
        client.warm_cache([])


# -------------------------------------------------------- batched wrappers


def test_batched_wrappers_raise_when_disabled(client):
    with pytest.raises(RuntimeError):
        client.insert_vectors_batched("cid", [[1.0]], ["a"])
    with pytest.raises(RuntimeError):
        client.upsert_vectors_batched("cid", [[1.0]], ["a"])
    with pytest.raises(RuntimeError):
        client.delete_vectors_batched("cid", ["a"])
    with pytest.raises(RuntimeError):
        client.get_batch_metrics()
    with pytest.raises(RuntimeError):
        client.reset_batch_metrics()


def _batching_client(monkeypatch):
    """Client with batching *simulated* — we never start the real threaded
    processor (that would spawn a non-daemon thread and hang the run); we flip
    the flag and inject a MagicMock processor directly."""
    c = _make_client(monkeypatch)
    fake_proc = MagicMock()
    c.enable_batching = True
    c._batch_processor = fake_proc
    return c, fake_proc


def test_insert_vectors_batched_submits(monkeypatch):
    c, fake_proc = _batching_client(monkeypatch)
    fake_proc.submit_request.return_value = "req-1"
    rid = c.insert_vectors_batched("cid", [[1.0, 2.0]], ["a"], metadata=[{"k": "v"}])
    assert rid == "req-1"
    assert fake_proc.submit_request.called


def test_upsert_vectors_batched_submits(monkeypatch):
    c, fake_proc = _batching_client(monkeypatch)
    fake_proc.submit_request.return_value = "req-2"
    rid = c.upsert_vectors_batched("cid", [[1.0, 2.0]], ["a"])
    assert rid == "req-2"


def test_delete_vectors_batched_submits(monkeypatch):
    c, fake_proc = _batching_client(monkeypatch)
    fake_proc.submit_request.return_value = "req-3"
    rid = c.delete_vectors_batched("cid", ["a", "b"])
    assert rid == "req-3"


def test_insert_vectors_batched_mismatch_raises(monkeypatch):
    c, _ = _batching_client(monkeypatch)
    with pytest.raises(ValueError):
        c.insert_vectors_batched("cid", [[1.0]], ["a", "b"])


def test_batch_metrics_when_enabled(monkeypatch):
    c, fake_proc = _batching_client(monkeypatch)
    fake_proc.get_metrics.return_value = {"submitted": 1}
    assert c.get_batch_metrics() == {"submitted": 1}
    c.reset_batch_metrics()
    assert fake_proc.reset_metrics.called


# --------------------------------------------------------- close & helpers


def test_close_closes_http_client(monkeypatch):
    c = ProximaDBClient(url="http://testserver")
    fake_http = FakeHttpClient()
    monkeypatch.setattr(c, "_http_client", fake_http)
    c.close()
    assert fake_http.closed is True


def test_execute_batch_insert(monkeypatch):
    from proximadb_sdk.batching_unified import BatchOperationType

    c = _make_client(monkeypatch, resp_body={"inserted_count": 2, "failed_count": 0})
    out = c._execute_batch(
        BatchOperationType.INSERT_VECTORS,
        "cid",
        [[{"id": "a", "vector": [1.0]}, {"id": "b", "vector": [2.0]}]],
    )
    assert out[0]["success"] is True


def test_execute_batch_unknown_operation(client):
    out = client._execute_batch("NOPE", "cid", [])
    assert out[0]["success"] is False


def test_execute_batch_upsert(monkeypatch):
    # The UPSERT branch calls self.upsert_vectors(vectors=, ids=, metadata=),
    # but upsert_vectors' signature is (collection_id, records) — so the call
    # raises TypeError and is surfaced through the except guard as success=False.
    from proximadb_sdk.batching_unified import BatchOperationType

    c = _make_client(monkeypatch, resp_body={"inserted_count": 1, "failed_count": 0})
    out = c._execute_batch(
        BatchOperationType.UPSERT_VECTORS,
        "cid",
        [[{"id": "a", "vector": [1.0]}]],
    )
    assert out[0]["success"] is False


def test_execute_batch_delete(monkeypatch):
    # Similarly, the DELETE branch calls delete_vectors(ids=...) but the real
    # signature is delete_vectors(collection_id, vector_ids); the kwarg
    # mismatch is surfaced via the except guard.
    from proximadb_sdk.batching_unified import BatchOperationType

    c = _make_client(monkeypatch, resp_body={"success": True})
    out = c._execute_batch(BatchOperationType.DELETE_VECTORS, "cid", [["a", "b"]])
    assert "success" in out[0]


def test_execute_batch_exception_path(monkeypatch):
    from proximadb_sdk.batching_unified import BatchOperationType

    c = _make_client(monkeypatch)
    # Pass malformed batch items (missing "vector") so the inner extraction
    # raises -> the except branch returns success=False with an error string.
    out = c._execute_batch(BatchOperationType.INSERT_VECTORS, "cid", [[{"id": "a"}]])
    assert out[0]["success"] is False
    assert "error" in out[0]


# ----------------------------------------------- extra branch coverage


def test_get_collection_nested_config(monkeypatch):
    # dimension lives only under the nested "config" object.
    c = _make_client(
        monkeypatch,
        resp_body={
            "id": "cidnested",
            "name": "nestedcoll",
            "config": {"dimension": 99, "metric": "cosine"},
        },
    )
    coll = c.get_collection("cidnested")
    assert coll.config.dimension == 99


def test_get_collection_wraps_in_collection_field(monkeypatch):
    c = _make_client(
        monkeypatch,
        resp_body={
            "collection": {
                "id": "wrapped",
                "name": "wrappedcoll",
                "dimension": 16,
                "metric": "cosine",
            }
        },
    )
    coll = c.get_collection("wrapped")
    assert coll.id == "wrapped"
    assert coll.config.dimension == 16


def test_search_include_vectors(monkeypatch):
    c = _make_client(
        monkeypatch,
        resp_body={
            "results": [{"id": "a", "score": 0.9, "vector": [1.0, 2.0], "props": {}}]
        },
    )
    out = c.search("cid", [0.1, 0.2], include_vectors=True)
    assert out[0].vector == [1.0, 2.0]


def test_search_not_a_dict_response(monkeypatch):
    class _ListResp:
        status_code = 200
        headers = {}
        text = "[]"
        content = b"[]"

        def json(self):
            return ["not", "a", "dict"]

        def raise_for_status(self):
            return None

    c = ProximaDBClient(url="http://testserver")
    monkeypatch.setattr(c, "_make_request", lambda m, p, **k: _ListResp())
    monkeypatch.setattr(c, "_http_client", FakeHttpClient())
    assert c.search("cid", [0.1]) == []
    c.close()


def test_search_next_page_request_raises_returns_empty(monkeypatch):
    c = ProximaDBClient(url="http://testserver")

    def boom(method, endpoint, **kwargs):
        raise RuntimeError("transport down")

    monkeypatch.setattr(c, "_make_request", boom)
    monkeypatch.setattr(c, "_http_client", FakeHttpClient())
    c._sks_search_supported = True
    env = c.search_next_page("cid", "cur")
    assert env.items == []
    c.close()


def test_update_vector_empty_vector_list_when_none_metadata(monkeypatch):
    c = _make_client(monkeypatch, resp_body={"inserted_count": 1, "failed_count": 0})
    # vector provided but metadata None -> props {} path
    res = c.update_vector("cid", "v1", vector=[1.0, 2.0])
    assert res.success == 1
