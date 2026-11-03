"""Offline unit tests for proximadb_sdk.adapters.embedded_adapter.

Everything is mocked: we never boot a real embedded DB. We inject a fake
native DB object (plain object with only the methods we want present) so the
adapter's translation logic is exercised without any network / native binding.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from proximadb_sdk.adapters.embedded_adapter import EmbeddedProtocolAdapter
from proximadb_sdk.models import (
    Collection,
    CollectionConfig,
    CollectionStats,
    DistanceMetric,
    SearchResult,
    VectorOperationResponse,
    VectorRecord,
)

COLL = "mycollection"  # collection names must be >= 8 chars per the model


# ---------------------------------------------------------------------------
# Fake native DB helpers
# ---------------------------------------------------------------------------


class FakeDB:
    """A bare native-DB stand-in with no methods; tests add attributes as needed."""


def make_adapter(db: Any) -> EmbeddedProtocolAdapter:
    """Construct adapter with an injected fake DB (no native import path)."""
    return EmbeddedProtocolAdapter(embedded_db=db)


def build_collection(name: str = COLL, dimension: int = 4) -> Collection:
    return Collection(
        id=name,
        config=CollectionConfig(
            name=name,
            dimension=dimension,
            distance_metric=DistanceMetric.COSINE,
            storage_engine="sst",
        ),
        stats=CollectionStats(vector_count=0),
    )


# ---------------------------------------------------------------------------
# Construction / properties
# ---------------------------------------------------------------------------


def test_init_with_injected_db_is_connected():
    db = FakeDB()
    a = make_adapter(db)
    assert a.is_connected is True
    assert a.protocol_name == "embedded"
    assert a._db is db


# NOTE: We intentionally do NOT test the native-import fallback init branch.
# In this environment the legacy `proximadb` package imports successfully and
# would attempt to boot a real embedded DB, violating the offline rule. The
# injected-db construction path (used by every other test) is covered instead.


# ---------------------------------------------------------------------------
# Health
#
# NOTE: the SDK's HealthStatus model requires `version` and `uptime_seconds`,
# which the adapter's health() does not supply. As a result health() raises a
# pydantic ValidationError at runtime for every branch. We assert that real
# behavior rather than inventing a contract the code does not honor.
# ---------------------------------------------------------------------------


def test_health_not_connected_raises_validation_error():
    from pydantic import ValidationError

    a = make_adapter(FakeDB())
    a._connected = False
    a._db = None
    with pytest.raises(ValidationError):
        a.health()


def test_health_connected_raises_validation_error():
    from pydantic import ValidationError

    db = FakeDB()
    db.list_collections = lambda: ["a", "b"]
    a = make_adapter(db)
    with pytest.raises(ValidationError):
        a.health()


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


def test_create_collection_positional_with_config():
    db = FakeDB()
    calls = []
    db.create_collection = lambda *a, **k: calls.append((a, k))
    a = make_adapter(db)
    cfg = CollectionConfig(
        name=COLL,
        dimension=8,
        distance_metric=DistanceMetric.COSINE,
        storage_engine="sst",
    )
    coll = a.create_collection(COLL, config=cfg)
    assert coll.name == COLL
    assert coll.dimension == 8
    assert COLL in a._collections


def test_create_collection_typeerror_falls_back_to_kwargs():
    db = FakeDB()
    state = {"first": True}

    def cc(*args, **kwargs):
        if state["first"] and args:  # positional call
            state["first"] = False
            raise TypeError("positional not supported")
        return None

    db.create_collection = cc
    a = make_adapter(db)
    coll = a.create_collection(COLL, dimension=16, storage_engine="viper")
    assert coll.dimension == 16


def test_create_collection_engine_kwarg_via_proto():
    db = FakeDB()
    db.create_collection = lambda *a, **k: None
    a = make_adapter(db)
    coll = a.create_collection(COLL, engine="sst")
    assert coll.name == COLL


def test_create_collection_raises_propagates():
    db = FakeDB()

    def cc(*a, **k):
        raise ValueError("boom")

    db.create_collection = cc
    a = make_adapter(db)
    with pytest.raises(ValueError):
        a.create_collection(COLL, dimension=4)


def test_get_collection_from_cache():
    a = make_adapter(FakeDB())
    coll = build_collection()
    a._collections[COLL] = coll
    assert a.get_collection(COLL) is coll


def test_get_collection_via_db_object():
    db = FakeDB()

    class Result:
        name = COLL
        dimension = 32
        engine = "sst"
        vector_count = 5

    db.get_collection = lambda cid: Result()
    a = make_adapter(db)
    coll = a.get_collection(COLL)
    assert coll is not None
    assert coll.dimension == 32
    assert COLL in a._collections


def test_get_collection_fallback_to_list():
    db = FakeDB()
    db.get_collection = lambda cid: None
    # list_collections returns a real Collection object so the fallback finds it
    db.list_collections = lambda: [build_collection(COLL)]
    a = make_adapter(db)
    coll = a.get_collection(COLL)
    assert coll is not None
    assert coll.name == COLL


def test_get_collection_not_found_returns_none():
    db = FakeDB()
    db.get_collection = lambda cid: None
    db.list_collections = lambda: []
    a = make_adapter(db)
    assert a.get_collection("missingcoll") is None


def test_get_collection_exception_returns_none():
    db = FakeDB()

    def boom(cid):
        raise RuntimeError("x")

    db.get_collection = boom
    a = make_adapter(db)
    assert a.get_collection("missingcoll") is None


def test_list_collections_mixed_items():
    # NOTE: string items hit _build_collection_model(dimension=0) which fails
    # validation; since list_collections wraps everything in try/except and
    # returns [] on ANY error, we only exercise items that build successfully
    # (Collection passthrough + attr-bearing object with dimension >= 1).
    db = FakeDB()
    existing = build_collection("existingcol")

    class Obj:
        name = "objcollection"
        dimension = 12
        engine = "sst"
        vector_count = 3

    db.list_collections = lambda: [existing, Obj()]
    a = make_adapter(db)
    out = a.list_collections()
    names = {c.name for c in out}
    assert "existingcol" in names
    assert "objcollection" in names


def test_list_collections_no_attr_returns_cache():
    db = FakeDB()  # no list_collections
    a = make_adapter(db)
    a._collections[COLL] = build_collection()
    out = a.list_collections()
    assert out == [a._collections[COLL]]


def test_list_collections_exception_returns_empty():
    db = FakeDB()

    def boom():
        raise RuntimeError("x")

    db.list_collections = boom
    a = make_adapter(db)
    assert a.list_collections() == []


def test_delete_collection_success():
    db = FakeDB()
    db.delete_collection = lambda cid: None
    a = make_adapter(db)
    a._collections[COLL] = build_collection()
    assert a.delete_collection(COLL) is True
    assert COLL not in a._collections


def test_delete_collection_no_attr_still_true():
    a = make_adapter(FakeDB())
    assert a.delete_collection(COLL) is True


def test_delete_collection_exception_false():
    db = FakeDB()

    def boom(cid):
        raise RuntimeError("x")

    db.delete_collection = boom
    a = make_adapter(db)
    assert a.delete_collection(COLL) is False


# ---------------------------------------------------------------------------
# numpy insert / upsert
# ---------------------------------------------------------------------------


def test_insert_numpy_with_insert_numpy_method():
    db = FakeDB()
    db.insert_numpy = lambda cid, ids, arr, meta: len(ids)
    a = make_adapter(db)
    resp = a.insert_numpy(COLL, ["a", "b"], [[1.0, 2.0], [3.0, 4.0]])
    assert isinstance(resp, VectorOperationResponse)
    assert resp.success
    assert resp.metrics.successful_count == 2


def test_insert_numpy_falls_back_to_insert():
    db = FakeDB()
    db.insert = lambda cid, ids, arr, meta: len(ids)
    a = make_adapter(db)
    resp = a.insert_numpy(COLL, ["a"], [[1.0, 2.0]], [{"k": "v"}])
    assert resp.success
    assert resp.metrics.successful_count == 1


def test_upsert_numpy_with_upsert_numpy_method():
    db = FakeDB()
    db.upsert_numpy = lambda cid, ids, arr, meta: len(ids)
    a = make_adapter(db)
    resp = a.upsert_numpy(COLL, ["a", "b", "c"], [[1, 2], [3, 4], [5, 6]])
    assert resp.success
    assert resp.operation == "UPSERT"
    assert resp.metrics.successful_count == 3


def test_upsert_numpy_falls_back_to_upsert():
    db = FakeDB()  # no upsert_numpy
    db.upsert = lambda cid, ids, arr, meta: len(ids)
    a = make_adapter(db)
    resp = a.upsert_numpy(COLL, ["a"], [[1.0, 2.0]])
    assert resp.success


def test_numpy_batch_partial_success_count():
    db = FakeDB()
    db.insert_numpy = lambda cid, ids, arr, meta: 1  # only 1 of 2 succeeded
    a = make_adapter(db)
    resp = a.insert_numpy(COLL, ["a", "b"], [[1, 2], [3, 4]])
    assert resp.metrics.successful_count == 1
    assert resp.metrics.failed_count == 1


def test_numpy_batch_bad_shape_raises_value_error():
    db = FakeDB()
    db.insert_numpy = lambda *a, **k: 1
    a = make_adapter(db)
    # 1D array -> ndim != 2 -> ValueError raised before try; surfaces to caller
    with pytest.raises(ValueError):
        a.insert_numpy(COLL, ["a"], [1.0, 2.0])


def test_numpy_batch_db_exception_returns_error_response():
    db = FakeDB()

    def boom(*a, **k):
        raise RuntimeError("db down")

    db.insert_numpy = boom
    a = make_adapter(db)
    resp = a.insert_numpy(COLL, ["a"], [[1, 2]])
    assert not resp.success
    assert "db down" in resp.error_message


# ---------------------------------------------------------------------------
# record insert / upsert
# ---------------------------------------------------------------------------


def test_insert_records_native_method():
    db = FakeDB()
    db.insert_records = lambda cid, recs, **k: len(recs)
    a = make_adapter(db)
    res = a.insert_records(COLL, [{"id": "x", "vector": [1, 2]}])
    assert res.total == 1
    assert res.success == 1
    assert res.failed == 0


def test_insert_records_native_exception():
    db = FakeDB()

    def boom(cid, recs, **k):
        raise RuntimeError("write fail")

    db.insert_records = boom
    a = make_adapter(db)
    res = a.insert_records(COLL, [{"id": "x", "vector": [1, 2]}])
    assert res.failed == 1
    assert "write fail" in res.errors[0]


def test_insert_records_fallback_to_numpy():
    db = FakeDB()  # no insert_records
    db.insert_numpy = lambda cid, ids, arr, meta: len(ids)
    a = make_adapter(db)
    res = a.insert_records(
        COLL,
        [
            {"id": "x", "vector": [1.0, 2.0], "metadata": {"k": "v"}},
            {"id": "y", "vector": [3.0, 4.0]},
        ],
    )
    assert res.total == 2
    assert res.success == 2


def test_upsert_records_native_tuple_result():
    db = FakeDB()
    db.upsert_records = lambda cid, recs, **k: (len(recs), 0)
    a = make_adapter(db)
    res = a.upsert_records(COLL, [{"id": "x", "vector": [1, 2]}])
    assert res.success == 1


def test_upsert_records_native_exception():
    db = FakeDB()

    def boom(cid, recs, **k):
        raise RuntimeError("upsert fail")

    db.upsert_records = boom
    a = make_adapter(db)
    res = a.upsert_records(COLL, [{"id": "x", "vector": [1, 2]}])
    assert res.failed == 1


def test_upsert_records_fallback_to_numpy():
    db = FakeDB()
    db.upsert_numpy = lambda cid, ids, arr, meta: len(ids)
    a = make_adapter(db)
    res = a.upsert_records(COLL, [{"id": "x", "vector": [1.0, 2.0]}])
    assert res.success == 1


def test_normalize_records_with_model_dump():
    """Exercise VectorRecord (model_dump) and no-metadata branches."""
    db = FakeDB()
    db.insert_numpy = lambda cid, ids, arr, meta: len(ids)
    a = make_adapter(db)
    rec = VectorRecord(id="r1", vector=[1.0, 2.0], metadata={"a": "b"})
    rec2 = VectorRecord(id="r2", vector=[3.0, 4.0])  # no metadata
    res = a.insert_records(COLL, [rec, rec2])
    assert res.success == 2


# ---------------------------------------------------------------------------
# vector compatibility aliases
# ---------------------------------------------------------------------------


def test_insert_vectors_alias():
    db = FakeDB()
    db.insert_numpy = lambda cid, ids, arr, meta: len(ids)
    a = make_adapter(db)
    resp = a.insert_vectors(COLL, [{"id": "x", "vector": [1.0, 2.0]}])
    assert isinstance(resp, VectorOperationResponse)
    assert resp.operation == "INSERT"
    assert resp.success


def test_upsert_vectors_alias():
    db = FakeDB()
    db.upsert_numpy = lambda cid, ids, arr, meta: len(ids)
    a = make_adapter(db)
    resp = a.upsert_vectors(COLL, [{"id": "x", "vector": [1.0, 2.0]}])
    assert resp.operation == "UPSERT"


# ---------------------------------------------------------------------------
# get_vectors
# ---------------------------------------------------------------------------


def test_get_vectors_via_get_vectors_dict():
    db = FakeDB()
    db.get_vectors = lambda cid, ids: [{"id": "a", "vector": [1.0], "metadata": {}}]
    a = make_adapter(db)
    out = a.get_vectors(COLL, ["a"])
    assert len(out) == 1
    assert out[0].id == "a"


def test_get_vectors_via_get_vector_objects():
    db = FakeDB()

    class V:
        id = "a"
        vector = [1.0, 2.0]
        metadata = {"k": "v"}

    db.get_vector = lambda cid, vid: V()
    a = make_adapter(db)
    out = a.get_vectors(COLL, ["a"], include_vectors=True)
    assert out[0].vector == [1.0, 2.0]


def test_get_vectors_passes_through_vectorrecord():
    db = FakeDB()
    vr = VectorRecord(id="a", vector=[1.0])
    db.get_vectors = lambda cid, ids: [vr]
    a = make_adapter(db)
    out = a.get_vectors(COLL, ["a"])
    assert out[0] is vr


def test_get_vectors_no_method_returns_empty():
    a = make_adapter(FakeDB())
    assert a.get_vectors(COLL, ["a"]) == []


def test_get_vectors_exception_returns_empty():
    db = FakeDB()

    def boom(cid, ids):
        raise RuntimeError("x")

    db.get_vectors = boom
    a = make_adapter(db)
    assert a.get_vectors(COLL, ["a"]) == []


# ---------------------------------------------------------------------------
# delete_vectors
# ---------------------------------------------------------------------------


def test_delete_vectors_int_result():
    db = FakeDB()
    db.delete_vectors = lambda cid, ids: len(ids)
    a = make_adapter(db)
    resp = a.delete_vectors(COLL, ["a", "b"])
    assert resp.success
    assert resp.metrics.successful_count == 2


def test_delete_vectors_non_int_result():
    # A non-int result (a list) takes the "assume all succeeded" branch.
    db = FakeDB()
    db.delete_vectors = lambda cid, ids: ["a", "b"]
    a = make_adapter(db)
    resp = a.delete_vectors(COLL, ["a", "b"])
    assert resp.success
    assert resp.metrics.successful_count == 2


def test_delete_vectors_single_via_delete_vector():
    db = FakeDB()
    db.delete_vector = lambda cid, vid: True
    a = make_adapter(db)
    resp = a.delete_vectors(COLL, ["a"])
    assert resp.success
    assert resp.metrics.successful_count == 1


def test_delete_vectors_not_implemented():
    # The "not implemented" branch builds a VectorOperationResponse without the
    # required `metrics` field -> ValidationError, caught by the outer except,
    # which returns an error response carrying the validation message.
    a = make_adapter(FakeDB())
    resp = a.delete_vectors(COLL, ["a", "b"])
    assert not resp.success
    assert resp.error_message  # non-empty (the validation error text)


def test_delete_vectors_exception():
    db = FakeDB()

    def boom(cid, ids):
        raise RuntimeError("del fail")

    db.delete_vectors = boom
    a = make_adapter(db)
    resp = a.delete_vectors(COLL, ["a"])
    assert not resp.success
    assert "del fail" in resp.error_message


# ---------------------------------------------------------------------------
# update_vector_metadata
# ---------------------------------------------------------------------------


def test_update_vector_metadata_native():
    db = FakeDB()
    db.update_metadata = lambda cid, vid, md: None
    a = make_adapter(db)
    resp = a.update_vector_metadata(COLL, "a", {"k": "v"})
    assert resp.success
    assert resp.operation == "UPDATE"


def test_update_vector_metadata_fallback_found():
    db = FakeDB()
    db.get_vectors = lambda cid, ids: [
        {"id": "a", "vector": [1.0], "metadata": {"old": 1}}
    ]
    db.upsert_numpy = lambda cid, ids, arr, meta: len(ids)
    a = make_adapter(db)
    resp = a.update_vector_metadata(COLL, "a", {"new": 2})
    assert resp.success


def test_update_vector_metadata_fallback_not_found():
    db = FakeDB()
    db.get_vectors = lambda cid, ids: []
    a = make_adapter(db)
    resp = a.update_vector_metadata(COLL, "a", {"k": "v"})
    assert not resp.success
    assert "not found" in resp.error_message


def test_update_vector_metadata_exception():
    db = FakeDB()

    def boom(cid, vid, md):
        raise RuntimeError("upd fail")

    db.update_metadata = boom
    a = make_adapter(db)
    resp = a.update_vector_metadata(COLL, "a", {"k": "v"})
    assert not resp.success
    assert "upd fail" in resp.error_message


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_numpy_tuples():
    db = FakeDB()
    db.search_numpy = lambda cid, q, top_k, filter: [
        ("a", 0.9, {"m": 1}, [1.0, 2.0]),
        ("b", 0.5),
    ]
    a = make_adapter(db)
    out = a.search(
        COLL, [1.0, 2.0], top_k=2, include_vectors=True, include_metadata=True
    )
    assert len(out) == 2
    assert out[0].id == "a"
    assert out[0].score == 0.9
    assert out[0].vector == [1.0, 2.0]


def test_search_with_filter_and_numpy_query():
    db = FakeDB()
    captured = {}

    def sn(cid, q, top_k, filter):
        captured["filter"] = filter
        return [{"id": "a", "score": 0.7}]

    db.search_numpy = sn
    a = make_adapter(db)
    out = a.search(
        COLL,
        np.array([1.0, 2.0], dtype=np.float32),
        filter={"env": "prod", "team": "x"},
    )
    assert "env = 'prod'" in captured["filter"]
    assert "team = 'x'" in captured["filter"]
    assert out[0].id == "a"


def test_search_fallback_to_search_method_dicts():
    db = FakeDB()  # no search_numpy
    db.search = lambda cid, q, top_k, filter: [
        {"vector_id": "z", "distance": 0.1, "metadata": {"a": 1}}
    ]
    a = make_adapter(db)
    out = a.search(COLL, [1.0, 2.0])
    assert out[0].id == "z"
    assert out[0].score == 0.1


def test_search_object_results():
    db = FakeDB()

    class R:
        id = "obj1"
        score = 0.42
        vector = [9.0]
        metadata = {"k": "v"}

    db.search_numpy = lambda cid, q, top_k, filter: [R()]
    a = make_adapter(db)
    out = a.search(COLL, [1.0], include_vectors=True, include_metadata=True)
    assert out[0].id == "obj1"
    assert out[0].vector == [9.0]


def test_search_passes_through_searchresult():
    db = FakeDB()
    sr = SearchResult(id="a", score=1.0)
    db.search_numpy = lambda cid, q, top_k, filter: [sr]
    a = make_adapter(db)
    out = a.search(COLL, [1.0])
    assert out[0] is sr


def test_search_exception_returns_empty():
    db = FakeDB()

    def boom(*a, **k):
        raise RuntimeError("search fail")

    db.search_numpy = boom
    a = make_adapter(db)
    assert a.search(COLL, [1.0]) == []


def test_to_search_results_none():
    a = make_adapter(FakeDB())
    assert a._to_search_results(None, False, False) == []


def test_to_search_results_conversion_error_skipped():
    a = make_adapter(FakeDB())

    class Bad:
        id = "x"

        @property
        def score(self):
            raise ValueError("bad score")

    out = a._to_search_results([Bad()], False, False)
    assert out == []


# ---------------------------------------------------------------------------
# batch_search
# ---------------------------------------------------------------------------


def test_batch_search_native():
    db = FakeDB()
    db.batch_search = lambda cid, qs, k, filter, include_vectors, include_metadata: [
        [("a", 0.9)],
        [("b", 0.8)],
    ]
    a = make_adapter(db)
    out = a.batch_search(COLL, [[1.0], np.array([2.0], dtype=np.float32)])
    assert len(out) == 2
    assert out[0][0].id == "a"


def test_batch_search_fallback_individual():
    db = FakeDB()  # no batch_search
    db.search_numpy = lambda cid, q, top_k, filter: [("a", 0.5)]
    a = make_adapter(db)
    out = a.batch_search(COLL, [[1.0], [2.0]])
    assert len(out) == 2
    assert out[0][0].id == "a"


def test_batch_search_exception_returns_empty_per_query():
    db = FakeDB()

    def boom(*a, **k):
        raise RuntimeError("batch fail")

    db.batch_search = boom
    a = make_adapter(db)
    out = a.batch_search(COLL, [[1.0], [2.0]])
    assert out == [[], []]


# ---------------------------------------------------------------------------
# graph operations
# ---------------------------------------------------------------------------


def test_create_graph_positional():
    db = FakeDB()
    db.create_graph = lambda gid, engine: None
    a = make_adapter(db)
    res = a.create_graph("g1", engine="sst")
    assert res == {"success": True, "graph_id": "g1"}


def test_create_graph_typeerror_falls_back():
    db = FakeDB()
    state = {"first": True}

    def cg(*args, **kwargs):
        if state["first"] and args:
            state["first"] = False
            raise TypeError("no positional")
        return None

    db.create_graph = cg
    a = make_adapter(db)
    res = a.create_graph("g1")
    assert res["success"] is True


def test_create_graph_not_implemented():
    a = make_adapter(FakeDB())
    with pytest.raises(NotImplementedError):
        a.create_graph("g1")


def test_delete_graph():
    db = FakeDB()
    db.delete_graph = lambda gid: None
    a = make_adapter(db)
    assert a.delete_graph("g1")["success"] is True


def test_delete_graph_not_implemented():
    a = make_adapter(FakeDB())
    with pytest.raises(NotImplementedError):
        a.delete_graph("g1")


def test_query_nodes_dict_and_obj():
    db = FakeDB()

    class Node:
        id = "n2"
        labels = ["L"]
        properties = {"p": 1}

    db.query_nodes = lambda graph_id, labels, properties, limit, offset: [
        {"id": "n1", "labels": ["A"], "properties": {"k": "v"}},
        Node(),
        None,
    ]
    a = make_adapter(db)
    res = a.query_nodes(graph="g1", labels=["A"])
    assert res["total_count"] == 3
    assert res["nodes"][0]["id"] == "n1"
    assert res["nodes"][1]["id"] == "n2"
    assert res["nodes"][2] is None


def test_query_nodes_not_implemented():
    a = make_adapter(FakeDB())
    with pytest.raises(NotImplementedError):
        a.query_nodes(graph="g1")


def test_query_nodes_graph_id_from_kwargs():
    db = FakeDB()
    captured = {}

    def qn(graph_id, labels, properties, limit, offset):
        captured["gid"] = graph_id
        return []

    db.query_nodes = qn
    a = make_adapter(db)
    a.query_nodes(graph_id="from_kwargs")
    assert captured["gid"] == "from_kwargs"


def test_traverse_graph_dict_result():
    db = FakeDB()
    db.traverse_graph = lambda **k: {"nodes": [1], "edges": [2]}
    a = make_adapter(db)
    res = a.traverse_graph("start", graph="g1")
    assert res["nodes"] == [1]


def test_traverse_graph_non_dict_result():
    db = FakeDB()
    db.traverse_graph = lambda **k: "weird"
    a = make_adapter(db)
    res = a.traverse_graph("start")
    assert res == {"nodes": [], "edges": []}


def test_traverse_graph_not_implemented():
    a = make_adapter(FakeDB())
    with pytest.raises(NotImplementedError):
        a.traverse_graph("start")


def test_create_node():
    db = FakeDB()
    db.create_node = lambda **k: "ok"
    a = make_adapter(db)
    res = a.create_node("n1", ["L"], {"p": 1}, graph="g1")
    assert res["node_id"] == "n1"
    assert res["result"] == "ok"


def test_create_node_not_implemented():
    a = make_adapter(FakeDB())
    with pytest.raises(NotImplementedError):
        a.create_node("n1", ["L"], {})


def test_create_edge():
    db = FakeDB()
    db.create_edge = lambda **k: "edge_ok"
    a = make_adapter(db)
    res = a.create_edge("e1", "KNOWS", from_node="n1", to_node="n2", weight=1.0)
    assert res["edge_id"] == "e1"


def test_create_edge_not_implemented():
    a = make_adapter(FakeDB())
    with pytest.raises(NotImplementedError):
        a.create_edge("e1", "KNOWS")


def test_get_node_found():
    db = FakeDB()
    db.get_node = lambda graph_id, node_id: {"id": "n1", "labels": [], "properties": {}}
    a = make_adapter(db)
    res = a.get_node("n1", graph="g1")
    assert res["id"] == "n1"


def test_get_node_none():
    db = FakeDB()
    db.get_node = lambda graph_id, node_id: None
    a = make_adapter(db)
    assert a.get_node("n1") is None


def test_get_node_not_implemented():
    a = make_adapter(FakeDB())
    with pytest.raises(NotImplementedError):
        a.get_node("n1")


def test_get_outgoing_edges():
    db = FakeDB()
    db.get_outgoing_edges = lambda graph_id, node_id, edge_types: [
        {"id": "e1", "from_node": "n1", "to_node": "n2", "edge_type": "KNOWS"}
    ]
    a = make_adapter(db)
    edges = a.get_outgoing_edges("n1", graph="g1")
    assert edges[0]["from_node_id"] == "n1"


def test_get_outgoing_edges_not_implemented():
    a = make_adapter(FakeDB())
    with pytest.raises(NotImplementedError):
        a.get_outgoing_edges("n1")


def test_get_incoming_edges_obj():
    db = FakeDB()

    class Edge:
        id = "e1"
        from_node = "n0"
        to_node = "n1"
        edge_type = "REF"
        weight = 2.0
        properties = {"x": 1}

    db.get_incoming_edges = lambda graph_id, node_id, edge_types: [Edge()]
    a = make_adapter(db)
    edges = a.get_incoming_edges("n1")
    assert edges[0]["to_node_id"] == "n1"
    assert edges[0]["weight"] == 2.0


def test_get_incoming_edges_not_implemented():
    a = make_adapter(FakeDB())
    with pytest.raises(NotImplementedError):
        a.get_incoming_edges("n1")


def test_delete_node():
    db = FakeDB()
    db.delete_node = lambda graph_id, node_id: True
    a = make_adapter(db)
    assert a.delete_node("n1", graph="g1") is True


def test_delete_node_not_implemented():
    a = make_adapter(FakeDB())
    with pytest.raises(NotImplementedError):
        a.delete_node("n1")


def test_get_graph_stats():
    db = FakeDB()

    class Stats:
        total_nodes = 5
        total_edges = 7

    db.graph_stats = lambda gid: Stats()
    a = make_adapter(db)
    res = a.get_graph_stats("g1")
    assert res == {"total_nodes": 5, "total_edges": 7}


def test_get_graph_stats_not_implemented():
    a = make_adapter(FakeDB())
    with pytest.raises(NotImplementedError):
        a.get_graph_stats("g1")


def test_execute_graph_query_native():
    db = FakeDB()
    db.execute_graph_query = lambda graph, query: ["row"]
    a = make_adapter(db)
    res = a.execute_graph_query("g1", "MATCH (n) RETURN n")
    assert res["results"] == ["row"]


def test_execute_graph_query_multimodal_fallback_import_error():
    # The multi-modal fallback imports MultiModalQuery/QueryComponent from
    # proximadb_sdk.models, which no longer exports them -> ImportError is
    # raised (caught + re-raised by the method).
    db = FakeDB()  # no execute_graph_query
    db.execute_multi_modal_query = lambda mm: ["mm_row"]
    a = make_adapter(db)
    with pytest.raises(ImportError):
        a.execute_graph_query("g1", "SELECT *")


def test_execute_graph_query_not_implemented():
    a = make_adapter(FakeDB())
    with pytest.raises(NotImplementedError):
        a.execute_graph_query("g1", "Q")


# ---------------------------------------------------------------------------
# document operations
# ---------------------------------------------------------------------------


def test_create_document_collection_native():
    db = FakeDB()
    db.create_document_collection = lambda name, indexed_paths: "ok"
    a = make_adapter(db)
    res = a.create_document_collection("docscoll", config={"indexed_paths": ["a"]})
    assert res["success"] is True
    assert res["collection_id"] == "docscoll"


def test_create_document_collection_vector_fallback_raises():
    # The vector fallback calls create_collection(name, config={"dimension": N});
    # create_collection then does `config.dimension` on a dict -> AttributeError
    # (caught + re-raised by create_document_collection).
    db = FakeDB()  # no create_document_collection
    db.create_collection = lambda *a, **k: None
    a = make_adapter(db)
    with pytest.raises(AttributeError):
        a.create_document_collection("docscoll", config={"dimension": 256})


def test_create_document_collection_exception():
    db = FakeDB()

    def boom(name, indexed_paths):
        raise RuntimeError("x")

    db.create_document_collection = boom
    a = make_adapter(db)
    with pytest.raises(RuntimeError):
        a.create_document_collection("docscoll")


def test_insert_document_native():
    db = FakeDB()
    db.insert_document = lambda c, doc, did: ("doc1", 3)
    a = make_adapter(db)
    res = a.insert_document("docscoll", {"text": "hi"}, id="doc1")
    assert res == {"id": "doc1", "success": True, "version": 3}


def test_insert_document_vector_fallback():
    db = FakeDB()  # no insert_document
    db.insert_numpy = lambda cid, ids, arr, meta: len(ids)
    a = make_adapter(db)
    res = a.insert_document(
        "docscoll", {"text": "hi", "metadata": {"k": "v"}}, id="doc1"
    )
    assert res["implementation"] == "vector_fallback"
    assert res["id"] == "doc1"


def test_insert_document_exception():
    db = FakeDB()

    def boom(c, doc, did):
        raise RuntimeError("x")

    db.insert_document = boom
    a = make_adapter(db)
    with pytest.raises(RuntimeError):
        a.insert_document("docscoll", {"text": "hi"})


def test_get_document_native():
    db = FakeDB()
    db.get_document = lambda c, did: {"id": did, "document": {"x": 1}}
    a = make_adapter(db)
    res = a.get_document("docscoll", "d1")
    assert res["id"] == "d1"


def test_get_document_vector_fallback_with_json_source():
    import json

    db = FakeDB()
    db.get_vectors = lambda cid, ids, **k: [
        VectorRecord(
            id="d1", vector=[0.0], source=json.dumps({"a": 1}), metadata={"m": 1}
        )
    ]
    a = make_adapter(db)
    res = a.get_document("docscoll", "d1")
    assert res["document"] == {"a": 1}


def test_get_document_vector_fallback_non_json_source():
    db = FakeDB()
    db.get_vectors = lambda cid, ids, **k: [
        VectorRecord(id="d1", vector=[0.0], source="not-json", metadata={})
    ]
    a = make_adapter(db)
    res = a.get_document("docscoll", "d1")
    assert res["document"] == {"source": "not-json"}


def test_get_document_vector_fallback_not_found():
    db = FakeDB()
    db.get_vectors = lambda cid, ids, **k: []
    a = make_adapter(db)
    assert a.get_document("docscoll", "d1") is None


def test_get_document_exception_returns_none():
    db = FakeDB()

    def boom(c, did):
        raise RuntimeError("x")

    db.get_document = boom
    a = make_adapter(db)
    assert a.get_document("docscoll", "d1") is None


def test_query_documents_native():
    db = FakeDB()
    db.query_documents = lambda c, f, limit: [("d1", {"a": 1}), ("d2", {"b": 2})]
    a = make_adapter(db)
    res = a.query_documents("docscoll", filter={"k": "v"}, limit=10)
    assert res["count"] == 2


def test_query_documents_vector_fallback():
    db = FakeDB()  # no query_documents
    a = make_adapter(db)
    res = a.query_documents("docscoll")
    assert res["implementation"] == "vector_fallback"


def test_query_documents_exception():
    db = FakeDB()

    def boom(c, f, limit):
        raise RuntimeError("x")

    db.query_documents = boom
    a = make_adapter(db)
    with pytest.raises(RuntimeError):
        a.query_documents("docscoll", filter={"k": "v"})


def test_update_document_native():
    db = FakeDB()
    db.update_document = lambda c, did, m: None
    a = make_adapter(db)
    res = a.update_document("docscoll", "d1", [{"path": "$.a", "value": 1}])
    assert res["success"] is True


def test_update_document_vector_fallback_found():
    db = FakeDB()  # no update_document
    db.get_vectors = lambda cid, ids, **k: [
        VectorRecord(id="d1", vector=[0.0], source='{"a": {"b": 1}}', metadata={})
    ]
    db.insert_numpy = lambda cid, ids, arr, meta: len(ids)
    a = make_adapter(db)
    res = a.update_document(
        "docscoll", "d1", [{"path": "$.a.c", "value": 9, "operation": "SET"}]
    )
    assert res["implementation"] == "vector_fallback"


def test_update_document_vector_fallback_not_found():
    db = FakeDB()
    db.get_vectors = lambda cid, ids, **k: []
    a = make_adapter(db)
    res = a.update_document("docscoll", "d1", [{"path": "$.a", "value": 1}])
    assert res["success"] is False


def test_update_document_exception():
    db = FakeDB()

    def boom(c, did, m):
        raise RuntimeError("x")

    db.update_document = boom
    a = make_adapter(db)
    with pytest.raises(RuntimeError):
        a.update_document("docscoll", "d1", [{"path": "$.a", "value": 1}])


def test_delete_document_native():
    db = FakeDB()
    db.delete_document = lambda c, did: True
    a = make_adapter(db)
    assert a.delete_document("docscoll", "d1") is True


def test_delete_document_vector_fallback():
    db = FakeDB()  # no delete_document
    db.delete_vectors = lambda cid, ids: len(ids)
    a = make_adapter(db)
    assert a.delete_document("docscoll", "d1") is True


def test_delete_document_exception_returns_false():
    db = FakeDB()

    def boom(c, did):
        raise RuntimeError("x")

    db.delete_document = boom
    a = make_adapter(db)
    assert a.delete_document("docscoll", "d1") is False


def test_list_document_collections_native():
    db = FakeDB()
    db.list_document_collections = lambda: [{"name": "d"}]
    a = make_adapter(db)
    assert a.list_document_collections() == [{"name": "d"}]


def test_list_document_collections_non_list_native():
    db = FakeDB()
    db.list_document_collections = lambda: "notalist"
    a = make_adapter(db)
    assert a.list_document_collections() == []


def test_list_document_collections_vector_fallback():
    db = FakeDB()  # no list_document_collections
    db.list_collections = lambda: [build_collection("vectorcoll")]
    a = make_adapter(db)
    out = a.list_document_collections()
    assert out[0]["name"] == "vectorcoll"


def test_list_document_collections_exception():
    db = FakeDB()

    def boom():
        raise RuntimeError("x")

    db.list_document_collections = boom
    a = make_adapter(db)
    assert a.list_document_collections() == []


def test_delete_document_collection_native():
    db = FakeDB()
    db.delete_document_collection = lambda c: True
    a = make_adapter(db)
    assert a.delete_document_collection("docscoll") is True


def test_delete_document_collection_vector_fallback():
    db = FakeDB()  # no delete_document_collection
    db.delete_collection = lambda c: None
    a = make_adapter(db)
    assert a.delete_document_collection("docscoll") is True


def test_delete_document_collection_exception():
    db = FakeDB()

    def boom(c):
        raise RuntimeError("x")

    db.delete_document_collection = boom
    a = make_adapter(db)
    assert a.delete_document_collection("docscoll") is False


# ---------------------------------------------------------------------------
# hybrid search
# ---------------------------------------------------------------------------


def test_hybrid_search_native():
    db = FakeDB()
    db.hybrid_search = lambda **k: {"results": []}
    a = make_adapter(db)
    res = a.hybrid_search("hybcoll", "text", [1.0, 2.0])
    assert res == {"results": []}


def test_hybrid_search_vector_fallback():
    db = FakeDB()  # no hybrid_search
    db.search_numpy = lambda cid, q, top_k, filter: [("a", 0.9, {"m": 1})]
    a = make_adapter(db)
    res = a.hybrid_search("hybcoll", "text", [1.0, 2.0])
    assert res["fusion_strategy"] == "vector_only"
    assert res["results"][0]["id"] == "a"


def test_hybrid_search_exception():
    db = FakeDB()

    def boom(**k):
        raise RuntimeError("x")

    db.hybrid_search = boom
    a = make_adapter(db)
    with pytest.raises(RuntimeError):
        a.hybrid_search("hybcoll", "text", [1.0])


# ---------------------------------------------------------------------------
# time-series
# ---------------------------------------------------------------------------


def test_create_timeseries_collection_native():
    db = FakeDB()
    db.create_timeseries_collection = lambda name, config: "ok"
    a = make_adapter(db)
    res = a.create_timeseries_collection("tscoll001", config={"x": 1})
    assert res["success"] is True


def test_create_timeseries_collection_vector_fallback_raises():
    # Same dict-config bug as the document fallback: create_collection accesses
    # config.dimension on a dict -> AttributeError, re-raised.
    db = FakeDB()  # no create_timeseries_collection
    db.create_collection = lambda *a, **k: None
    a = make_adapter(db)
    with pytest.raises(AttributeError):
        a.create_timeseries_collection("tscoll001", config={"dimension": 64})


def test_create_timeseries_collection_exception():
    db = FakeDB()

    def boom(name, config):
        raise RuntimeError("x")

    db.create_timeseries_collection = boom
    a = make_adapter(db)
    with pytest.raises(RuntimeError):
        a.create_timeseries_collection("tscoll001")


def test_ingest_timeseries_native():
    db = FakeDB()
    db.ingest_timeseries = lambda collection, points: {"ingested_count": 1}
    a = make_adapter(db)
    res = a.ingest_timeseries("tscoll001", [{"timestamp": "t", "values": {"v": 1}}])
    assert res["ingested_count"] == 1


def test_ingest_timeseries_vector_fallback_raises():
    # The vector fallback builds a VectorRecord with metadata={"tags": {...}};
    # a nested dict is not a valid metadata value -> ValidationError, re-raised.
    db = FakeDB()  # no ingest_timeseries
    db.insert_numpy = lambda cid, ids, arr, meta: len(ids)
    a = make_adapter(db)
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        a.ingest_timeseries(
            "tscoll001",
            [{"timestamp": "t1", "values": {"cpu": 0.5}, "tags": {"host": "h"}}],
        )


def test_ingest_timeseries_exception():
    db = FakeDB()

    def boom(collection, points):
        raise RuntimeError("x")

    db.ingest_timeseries = boom
    a = make_adapter(db)
    with pytest.raises(RuntimeError):
        a.ingest_timeseries("tscoll001", [])


def test_query_timeseries_native():
    db = FakeDB()
    db.query_timeseries = lambda **k: {"raw_points": []}
    a = make_adapter(db)
    res = a.query_timeseries("tscoll001", "t0", "t1")
    assert res == {"raw_points": []}


def test_query_timeseries_vector_fallback():
    # The adapter's get_vectors converts via the db; we hand it dict-shaped
    # records (metadata has flat values only — nested dicts are invalid). p1 is
    # inside [t0, t5]; p2 (t9) is past end_time and filtered out.
    db = FakeDB()  # no query_timeseries
    import json

    db.get_vectors = lambda cid, ids: [
        {
            "id": "p1",
            "vector": [0.0],
            "source": json.dumps({"values": {"cpu": 0.5}}),
            "metadata": {"timestamp": "t3"},
        },
        {
            "id": "p2",
            "vector": [0.0],
            "source": "bad-json",
            "metadata": {"timestamp": "t9"},
        },
    ]
    a = make_adapter(db)
    res = a.query_timeseries("tscoll001", "t0", "t5")
    assert res["implementation"] == "vector_fallback"
    # only p1 falls inside the [t0, t5] window
    assert res["total_points"] == 1


def test_query_timeseries_fallback_rejects_explicit_aggregation():
    """TD-EMB-1: the vector fallback drops aggregation/bucket_ms — an
    explicit ask must raise the honest signal, not return raw points
    that look aggregated."""
    db = FakeDB()  # no query_timeseries -> vector fallback
    db.get_vectors = lambda cid, ids: []
    a = make_adapter(db)
    # default (None) still returns the fallback's raw shape
    res = a.query_timeseries("tscoll001", "t0", "t5")
    assert res["implementation"] == "vector_fallback"
    with pytest.raises(NotImplementedError, match="tag_filters are honored"):
        a.query_timeseries("tscoll001", "t0", "t5", aggregation="avg")
    with pytest.raises(NotImplementedError, match="TD-EMB-1"):
        a.query_timeseries("tscoll001", "t0", "t5", bucket_ms=60)


def test_query_timeseries_subprocess_async_guard():
    """TD-EMB-1 direction 3: the subprocess flavor's query_timeseries is
    `async def` — the sync adapter must NOT call it (an un-awaited
    coroutine would be returned as a success-shaped object)."""

    class AsyncFlavor:
        async def query_timeseries(self, **kwargs):  # pragma: no cover
            raise AssertionError("must never be awaited from the sync adapter")

    db = FakeDB()
    db.get_vectors = lambda cid, ids: []
    db.query_timeseries = AsyncFlavor().query_timeseries
    a = make_adapter(db)
    res = a.query_timeseries("tscoll001", "t0", "t5")
    assert res["implementation"] == "vector_fallback"


def test_query_timeseries_exception():
    db = FakeDB()

    def boom(**k):
        raise RuntimeError("x")

    db.query_timeseries = boom
    a = make_adapter(db)
    with pytest.raises(RuntimeError):
        a.query_timeseries("tscoll001", "t0", "t1")


def test_list_timeseries_collections_native():
    db = FakeDB()
    db.list_timeseries_collections = lambda: [{"name": "ts"}]
    a = make_adapter(db)
    assert a.list_timeseries_collections() == [{"name": "ts"}]


def test_list_timeseries_collections_non_list():
    db = FakeDB()
    db.list_timeseries_collections = lambda: None
    a = make_adapter(db)
    assert a.list_timeseries_collections() == []


def test_list_timeseries_collections_vector_fallback():
    db = FakeDB()  # no list_timeseries_collections
    db.list_collections = lambda: [build_collection("tsfallbk1")]
    a = make_adapter(db)
    out = a.list_timeseries_collections()
    assert out[0]["name"] == "tsfallbk1"


def test_list_timeseries_collections_exception():
    db = FakeDB()

    def boom():
        raise RuntimeError("x")

    db.list_timeseries_collections = boom
    a = make_adapter(db)
    assert a.list_timeseries_collections() == []


def test_delete_timeseries_collection_native_dict():
    db = FakeDB()
    db.delete_timeseries_collection = lambda collection: {"success": True}
    a = make_adapter(db)
    assert a.delete_timeseries_collection("tscoll001") is True


def test_delete_timeseries_collection_native_bool():
    db = FakeDB()
    db.delete_timeseries_collection = lambda collection: True
    a = make_adapter(db)
    assert a.delete_timeseries_collection("tscoll001") is True


def test_delete_timeseries_collection_vector_fallback():
    db = FakeDB()  # no delete_timeseries_collection
    db.delete_collection = lambda c: None
    a = make_adapter(db)
    assert a.delete_timeseries_collection("tscoll001") is True


def test_delete_timeseries_collection_exception():
    db = FakeDB()

    def boom(collection):
        raise RuntimeError("x")

    db.delete_timeseries_collection = boom
    a = make_adapter(db)
    assert a.delete_timeseries_collection("tscoll001") is False


# ---------------------------------------------------------------------------
# sql / unified
# ---------------------------------------------------------------------------


def test_execute_sql_native():
    db = FakeDB()
    db.execute_sql = lambda q, p, c: {"rows": []}
    a = make_adapter(db)
    assert a.execute_sql("SELECT 1") == {"rows": []}


def test_execute_sql_not_implemented():
    a = make_adapter(FakeDB())
    with pytest.raises(NotImplementedError):
        a.execute_sql("SELECT 1")


def test_execute_unified_query_native():
    db = FakeDB()
    db.execute_unified_query = lambda q, qv, fs: ["row"]
    a = make_adapter(db)
    assert a.execute_unified_query("Q", [1.0], "rrf") == ["row"]


def test_execute_unified_query_none_result():
    db = FakeDB()
    db.execute_unified_query = lambda q, qv, fs: None
    a = make_adapter(db)
    assert a.execute_unified_query("Q") == []


def test_execute_unified_query_not_implemented():
    a = make_adapter(FakeDB())
    with pytest.raises(NotImplementedError):
        a.execute_unified_query("Q")


# ---------------------------------------------------------------------------
# observability
# ---------------------------------------------------------------------------


def test_create_observability_namespace():
    db = FakeDB()
    db.create_observability_namespace = lambda name, retention: None
    a = make_adapter(db)
    assert a.create_observability_namespace("ns", retention_days=7)["success"] is True


def test_create_observability_namespace_not_implemented():
    a = make_adapter(FakeDB())
    with pytest.raises(NotImplementedError):
        a.create_observability_namespace("ns")


def test_create_observability_namespace_exception():
    db = FakeDB()

    def boom(name, retention):
        raise RuntimeError("x")

    db.create_observability_namespace = boom
    a = make_adapter(db)
    with pytest.raises(RuntimeError):
        a.create_observability_namespace("ns")


def test_ingest_logs():
    db = FakeDB()
    db.ingest_logs = lambda ns, logs: 3
    a = make_adapter(db)
    assert a.ingest_logs("ns", [{}]) == 3


def test_ingest_logs_not_implemented():
    a = make_adapter(FakeDB())
    with pytest.raises(NotImplementedError):
        a.ingest_logs("ns", [])


def test_query_logs():
    db = FakeDB()
    db.query_logs = lambda ns, s, e, q, limit: [{"log": 1}]
    a = make_adapter(db)
    assert a.query_logs("ns", 0, 1) == [{"log": 1}]


def test_query_logs_not_implemented():
    a = make_adapter(FakeDB())
    with pytest.raises(NotImplementedError):
        a.query_logs("ns", 0, 1)


def test_ingest_metrics():
    db = FakeDB()
    db.ingest_metrics = lambda ns, samples: 5
    a = make_adapter(db)
    assert a.ingest_metrics("ns", [{}]) == 5


def test_ingest_metrics_not_implemented():
    a = make_adapter(FakeDB())
    with pytest.raises(NotImplementedError):
        a.ingest_metrics("ns", [])


def test_aggregate_metrics():
    db = FakeDB()
    db.aggregate_metrics = lambda *a: [{"v": 1}]
    a = make_adapter(db)
    assert a.aggregate_metrics("ns", "cpu") == [{"v": 1}]


def test_aggregate_metrics_not_implemented():
    a = make_adapter(FakeDB())
    with pytest.raises(NotImplementedError):
        a.aggregate_metrics("ns", "cpu")


def test_ingest_traces():
    db = FakeDB()
    db.ingest_traces = lambda ns, traces: 2
    a = make_adapter(db)
    assert a.ingest_traces("ns", [{}]) == 2


def test_ingest_traces_not_implemented():
    a = make_adapter(FakeDB())
    with pytest.raises(NotImplementedError):
        a.ingest_traces("ns", [])


def test_query_traces():
    db = FakeDB()
    db.query_traces = lambda *a: [{"trace": 1}]
    a = make_adapter(db)
    assert a.query_traces("ns", 0, 1) == [{"trace": 1}]


def test_query_traces_not_implemented():
    a = make_adapter(FakeDB())
    with pytest.raises(NotImplementedError):
        a.query_traces("ns", 0, 1)


def test_get_trace():
    db = FakeDB()
    db.get_trace = lambda ns, tid: {"id": tid}
    a = make_adapter(db)
    assert a.get_trace("ns", "t1") == {"id": "t1"}


def test_get_trace_not_implemented():
    a = make_adapter(FakeDB())
    with pytest.raises(NotImplementedError):
        a.get_trace("ns", "t1")


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------


def test_close_via_close():
    db = FakeDB()
    closed = {"v": False}
    db.close = lambda: closed.__setitem__("v", True)
    a = make_adapter(db)
    a.close()
    assert closed["v"] is True
    assert a._db is None
    assert a._connected is False


def test_close_via_shutdown():
    db = FakeDB()
    shut = {"v": False}
    db.shutdown = lambda: shut.__setitem__("v", True)
    a = make_adapter(db)
    a.close()
    assert shut["v"] is True
    assert a._db is None


def test_close_exception_is_swallowed():
    db = FakeDB()

    def boom():
        raise RuntimeError("close fail")

    db.close = boom
    a = make_adapter(db)
    a.close()  # should not raise
    assert a._db is None


def test_close_when_db_none():
    a = make_adapter(FakeDB())
    a._db = None
    a.close()  # no-op, no error
    assert a._db is None
