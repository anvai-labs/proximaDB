"""Offline unit tests for proximadb_sdk.client_v1.ProximaDBClientV1.

Fully offline: every transport (requests + gRPC stubs) is mocked. No real
network, server, or channel I/O happens. gRPC channels are created lazily by
grpc.insecure_channel and never connected because we replace every stub with a
MagicMock before invoking any RPC.
"""

from unittest.mock import MagicMock

import grpc
import pytest

import proximadb_sdk.client_v1 as cv1
from proximadb_sdk.client_v1 import ProximaDBClientV1, create_client_v1
from proximadb_sdk.exceptions import NetworkError, ProximaDBError
from proximadb_sdk.models import DistanceMetric, StorageEngine, VectorRecord
from proximadb_sdk.v1 import (
    collection_types_pb2,
    graph_pb2,
    types_pb2,
    vector_types_pb2,
)
from proximadb_sdk.v2 import record_pb2


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------
class FakeResp:
    def __init__(self, json_data=None, status_code=200):
        self._json = json_data if json_data is not None else {}
        self.status_code = status_code
        self.headers = {}
        self.text = "fake"
        self.content = b"fake"

    def json(self):
        return self._json

    def raise_for_status(self):
        return None


class FakeRpcError(grpc.RpcError):
    def __init__(self, code=grpc.StatusCode.INTERNAL, details="boom"):
        self._code = code
        self._details = details

    def code(self):
        return self._code

    def details(self):
        return self._details


@pytest.fixture
def rest_client():
    return ProximaDBClientV1(url="http://testserver", protocol="rest")


@pytest.fixture
def grpc_client():
    c = ProximaDBClientV1(url="http://testserver:5679", protocol="grpc")
    # Replace every stub so no real RPC can hit the wire.
    c.vector_stub = MagicMock()
    c.collection_stub = MagicMock()
    c.sql_stub = MagicMock()
    c.graph_stub = MagicMock()
    c.record_stub = MagicMock()
    return c


# --------------------------------------------------------------------------
# Construction / protocol resolution
# --------------------------------------------------------------------------
def test_init_rest_default():
    c = ProximaDBClientV1()
    assert c.protocol == "rest"
    assert c.base_url == "http://localhost:5678"


def test_init_auto_grpc_by_port():
    c = ProximaDBClientV1(url="http://h:5679", protocol="auto")
    assert c.protocol == "grpc"
    assert hasattr(c, "channel")
    c.close()


def test_init_auto_grpc_by_scheme():
    c = ProximaDBClientV1(url="grpc://h:1234", protocol="auto")
    assert c.protocol == "grpc"
    c.close()


def test_init_explicit_grpc_sets_stubs():
    c = ProximaDBClientV1(url="http://h:5679", protocol="grpc")
    assert hasattr(c, "vector_stub")
    assert hasattr(c, "collection_stub")
    assert hasattr(c, "sql_stub")
    assert hasattr(c, "graph_stub")
    c.close()


def test_close_no_channel(rest_client):
    # No channel attribute on REST clients; close must be a no-op.
    assert not hasattr(rest_client, "channel")
    rest_client.close()


def test_create_client_v1_factory():
    c = create_client_v1(url="http://testserver", protocol="rest")
    assert isinstance(c, ProximaDBClientV1)
    assert c.protocol == "rest"


# --------------------------------------------------------------------------
# Collections - REST
# --------------------------------------------------------------------------
def test_create_collection_rest(rest_client, monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None, **kw):
        captured["url"] = url
        captured["json"] = json
        return FakeResp(
            {"collection_id": "c1", "name": "docs_col", "dimension": 8, "engine": "sst"}
        )

    monkeypatch.setattr(cv1.requests, "post", fake_post)
    col = rest_client.create_collection(
        "docs_col", 8, DistanceMetric.COSINE, StorageEngine.SST
    )
    assert col.id == "c1"
    assert col.config.dimension == 8
    assert "/api/v2/collections" in captured["url"]
    assert captured["json"]["enable_proxima_record"] is True


def test_create_collection_rest_string_enums(rest_client, monkeypatch):
    monkeypatch.setattr(
        cv1.requests, "post", lambda *a, **k: FakeResp({"id": "x", "dimension": 4})
    )
    col = rest_client.create_collection(
        "colnamed", 4, distance_metric="euclidean", storage_engine="nova"
    )
    assert col.config.distance_metric == DistanceMetric.EUCLIDEAN


def test_create_collection_rest_network_error(rest_client, monkeypatch):
    def boom(*a, **k):
        raise cv1.requests.RequestException("down")

    monkeypatch.setattr(cv1.requests, "post", boom)
    with pytest.raises(NetworkError):
        rest_client.create_collection("c", 4)


def test_get_collection_rest_found(rest_client, monkeypatch):
    monkeypatch.setattr(
        cv1.requests,
        "get",
        lambda *a, **k: FakeResp(
            {
                "id": "c1",
                "name": "docs_col",
                "dimension": 8,
                "distance_metric": "COSINE",
                "engine": "SST",
            }
        ),
    )
    col = rest_client.get_collection("docs_col")
    assert col.config.dimension == 8
    assert col.config.distance_metric == DistanceMetric.COSINE


def test_get_collection_rest_not_found(rest_client, monkeypatch):
    monkeypatch.setattr(
        cv1.requests, "get", lambda *a, **k: FakeResp({}, status_code=404)
    )
    assert rest_client.get_collection("missing") is None


def test_get_collection_rest_network_error(rest_client, monkeypatch):
    def boom(*a, **k):
        raise cv1.requests.RequestException("oops")

    monkeypatch.setattr(cv1.requests, "get", boom)
    with pytest.raises(NetworkError):
        rest_client.get_collection("docs")


def test_list_collections_rest(rest_client, monkeypatch):
    monkeypatch.setattr(
        cv1.requests,
        "get",
        lambda *a, **k: FakeResp(
            {"collections": [{"name": "alpha_col", "dimension": 4, "engine": "sst"}]}
        ),
    )
    cols = rest_client.list_collections()
    assert len(cols) == 1
    assert cols[0].config.name == "alpha_col"


def test_list_collections_rest_network_error(rest_client, monkeypatch):
    def boom(*a, **k):
        raise cv1.requests.RequestException("oops")

    monkeypatch.setattr(cv1.requests, "get", boom)
    with pytest.raises(NetworkError):
        rest_client.list_collections()


# --------------------------------------------------------------------------
# Collections - gRPC
# --------------------------------------------------------------------------
def test_create_collection_grpc(grpc_client):
    resp = collection_types_pb2.Collection(
        id="c1",
        config=collection_types_pb2.CollectionConfig(
            name="docs_col",
            dimension=8,
            distance_metric=vector_types_pb2.COSINE,
            storage_engine=vector_types_pb2.SST,
        ),
        stats=collection_types_pb2.CollectionStats(vector_count=3),
        created_at=2000,
        updated_at=4000,
    )
    grpc_client.collection_stub.CreateCollection.return_value = resp
    col = grpc_client.create_collection("docs_col", 8, "cosine", "sst")
    assert col.id == "c1"
    assert col.config.dimension == 8
    assert col.stats.vector_count == 3
    assert col.created_at_ms == 2  # micros -> millis


def test_create_collection_grpc_error(grpc_client):
    grpc_client.collection_stub.CreateCollection.side_effect = FakeRpcError(
        details="bad"
    )
    with pytest.raises(ProximaDBError):
        grpc_client.create_collection("docs", 8)


def test_get_collection_grpc_not_found(grpc_client):
    grpc_client.collection_stub.GetCollection.side_effect = FakeRpcError(
        code=grpc.StatusCode.NOT_FOUND, details="nope"
    )
    assert grpc_client.get_collection("x") is None


def test_get_collection_grpc_other_error(grpc_client):
    grpc_client.collection_stub.GetCollection.side_effect = FakeRpcError(
        code=grpc.StatusCode.INTERNAL, details="boom"
    )
    with pytest.raises(ProximaDBError):
        grpc_client.get_collection("x")


def test_list_collections_grpc_error(grpc_client):
    grpc_client.collection_stub.ListCollections.side_effect = FakeRpcError()
    with pytest.raises(ProximaDBError):
        grpc_client.list_collections()


# --------------------------------------------------------------------------
# Records / vectors - REST
# --------------------------------------------------------------------------
def test_insert_records_rest_dict_payloads(rest_client, monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None, **kw):
        captured["url"] = url
        captured["json"] = json
        return FakeResp({"success": True, "success_count": 2})

    monkeypatch.setattr(cv1.requests, "post", fake_post)
    out = rest_client.insert_records(
        "col",
        [
            {"id": "r1", "vector": [0.1], "metadata": {"k": "v"}},
            {"oid": "r2", "vector": [0.2]},
        ],
    )
    assert out["success"] is True
    recs = captured["json"]["records"]
    # metadata renamed to props; id derived from oid for second record
    assert recs[0]["props"] == {"k": "v"}
    assert recs[1]["id"] == "r2"
    assert "/records/batch" in captured["url"]


def test_insert_records_rest_vectorrecord(rest_client, monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None, **kw):
        captured["json"] = json
        return FakeResp({"success": True})

    monkeypatch.setattr(cv1.requests, "post", fake_post)
    rec = VectorRecord(id="v1", vector=[1.0, 2.0], metadata={"a": 1}, source="hello")
    rest_client.insert_vectors("col", [rec])
    payload = captured["json"]["records"][0]
    assert payload["id"] == "v1"
    # TD-REC-1: no top-level `source` on the wire (it was always silently
    # dropped server-side and is now a contract 400) — the content rides in
    # text_fields, where it persists.
    assert "source" not in payload
    assert payload["text_fields"][0]["content"] == "hello"


def test_insert_records_rest_no_id_fallback(rest_client, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        cv1.requests,
        "post",
        lambda url, json=None, **k: captured.update(json=json) or FakeResp({}),
    )
    rec = VectorRecord(vector=[1.0])
    rest_client.insert_records("col", [rec])
    assert captured["json"]["records"][0]["id"] == "record_0"


def test_insert_records_rest_network_error(rest_client, monkeypatch):
    def boom(*a, **k):
        raise cv1.requests.RequestException("net")

    monkeypatch.setattr(cv1.requests, "post", boom)
    with pytest.raises(NetworkError):
        rest_client.insert_records("col", [{"id": "x", "vector": [1.0]}])


# --------------------------------------------------------------------------
# Records / vectors - gRPC
# --------------------------------------------------------------------------
def test_insert_records_grpc(grpc_client):
    resp = record_pb2.ProximaRecordBatchResponse(
        success=True,
        total_processed=2,
        success_count=1,
        failed_count=1,
        inserted_ids=["r1"],
        errors=[
            record_pb2.BatchError(
                record_index=1, record_id="r2", error_code="DUP", error_message="dup"
            )
        ],
    )
    grpc_client.record_stub.InsertRecords.return_value = resp
    out = grpc_client.insert_records(
        "col",
        [
            {
                "id": "r1",
                "vector": [0.1],
                "props": {"n": 1, "b": True, "f": 1.5, "s": "x", "z": None},
            },
            {
                "id": "r2",
                "vector": [0.2],
                "source": "txt",
                "text_fields": [{"name": "t", "content": "body"}],
            },
        ],
    )
    assert out["success"] is True
    assert out["success_count"] == 1
    assert out["inserted_ids"] == ["r1"]
    assert out["errors"][0]["record_id"] == "r2"


def test_insert_records_grpc_error(grpc_client):
    grpc_client.record_stub.InsertRecords.side_effect = FakeRpcError()
    with pytest.raises(ProximaDBError):
        grpc_client.insert_records("col", [{"id": "r1", "vector": [0.1]}])


def test_insert_records_grpc_missing_stub():
    c = ProximaDBClientV1(url="http://h:5679", protocol="grpc")
    # Simulate v2 stubs unavailable.
    if hasattr(c, "record_stub"):
        delattr(c, "record_stub")
    with pytest.raises(ProximaDBError):
        c.insert_records("col", [{"id": "r1", "vector": [0.1]}])
    c.close()


def test_typed_value_branches(grpc_client):
    assert grpc_client._typed_value(None).is_null is True
    assert grpc_client._typed_value(True).boolean_value is True
    assert grpc_client._typed_value(5).integer_value == 5
    assert grpc_client._typed_value(2.5).float_value == 2.5
    assert grpc_client._typed_value("hi").text_value == "hi"


def test_typed_value_unavailable(grpc_client, monkeypatch):
    monkeypatch.setattr(cv1, "record_pb2", None)
    with pytest.raises(ProximaDBError):
        grpc_client._typed_value(1)


# --------------------------------------------------------------------------
# Search - REST / gRPC
# --------------------------------------------------------------------------
def test_search_vectors_grpc(grpc_client):
    inner = vector_types_pb2.SearchResult(
        results=[
            vector_types_pb2.SearchVectorRecord(id="r1", score=0.9, vector=[0.1, 0.2])
        ]
    )
    resp = vector_types_pb2.VectorOperationResponse(success=True, results=inner)
    grpc_client.vector_stub.VectorSearch.return_value = resp
    # filters left None: the gRPC SearchQuery.filters map expects SqlValue
    # messages, so the source's ``filters or {}`` only works for the empty case.
    results = grpc_client.search_vectors("col", [0.1, 0.2], top_k=5)
    assert results[0].id == "r1"
    assert results[0].score == pytest.approx(0.9)


def test_search_vectors_grpc_empty(grpc_client):
    resp = vector_types_pb2.VectorOperationResponse(success=True)
    grpc_client.vector_stub.VectorSearch.return_value = resp
    assert grpc_client.search_vectors("col", [0.1]) == []


def test_search_vectors_grpc_error(grpc_client):
    grpc_client.vector_stub.VectorSearch.side_effect = FakeRpcError()
    with pytest.raises(ProximaDBError):
        grpc_client.search_vectors("col", [0.1])


def test_search_vectors_rest_network_error(rest_client, monkeypatch):
    def boom(*a, **k):
        raise cv1.requests.RequestException("net")

    monkeypatch.setattr(cv1.requests, "post", boom)
    with pytest.raises(NetworkError):
        rest_client.search_vectors("col", [0.1], filters={"k": "v"})


def test_get_vector_grpc_found(grpc_client):
    inner = vector_types_pb2.SearchResult(
        results=[vector_types_pb2.SearchVectorRecord(id="v1", score=1.0, vector=[0.1])]
    )
    resp = vector_types_pb2.VectorOperationResponse(success=True, results=inner)
    grpc_client.vector_stub.VectorGet.return_value = resp
    rec = grpc_client.get_vector("col", "v1")
    assert rec.id == "v1"
    assert rec.vector == [pytest.approx(0.1)]


def test_get_vector_grpc_none(grpc_client):
    resp = vector_types_pb2.VectorOperationResponse(success=False)
    grpc_client.vector_stub.VectorGet.return_value = resp
    assert grpc_client.get_vector("col", "v1") is None


def test_get_vector_grpc_not_found_error(grpc_client):
    grpc_client.vector_stub.VectorGet.side_effect = FakeRpcError(
        code=grpc.StatusCode.NOT_FOUND
    )
    assert grpc_client.get_vector("col", "v1") is None


def test_get_vector_grpc_other_error(grpc_client):
    grpc_client.vector_stub.VectorGet.side_effect = FakeRpcError()
    with pytest.raises(ProximaDBError):
        grpc_client.get_vector("col", "v1")


def test_get_vector_rest_found(rest_client, monkeypatch):
    monkeypatch.setattr(
        cv1.requests,
        "get",
        lambda *a, **k: FakeResp({"id": "v1", "vector": [0.1], "props": {"k": "v"}}),
    )
    rec = rest_client.get_vector("col", "v1")
    assert rec.id == "v1"
    assert rec.metadata == {"k": "v"}


def test_get_vector_rest_404(rest_client, monkeypatch):
    monkeypatch.setattr(
        cv1.requests, "get", lambda *a, **k: FakeResp({}, status_code=404)
    )
    assert rest_client.get_vector("col", "v1") is None


def test_get_vector_rest_empty_body(rest_client, monkeypatch):
    monkeypatch.setattr(cv1.requests, "get", lambda *a, **k: FakeResp({}))
    assert rest_client.get_vector("col", "v1") is None


def test_get_vector_rest_network_error(rest_client, monkeypatch):
    def boom(*a, **k):
        raise cv1.requests.RequestException("net")

    monkeypatch.setattr(cv1.requests, "get", boom)
    with pytest.raises(NetworkError):
        rest_client.get_vector("col", "v1")


def test_advanced_vector_search_rest(rest_client, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        cv1.requests,
        "post",
        lambda url, json=None, **k: captured.update(url=url, json=json)
        or FakeResp({"results": []}),
    )
    out = rest_client.advanced_vector_search(
        "col",
        [0.1],
        filters={"k": "v"},
        accuracy_threshold=0.8,
        search_params={"timeout_ms": 100},
    )
    assert out == {"results": []}
    assert captured["json"]["accuracy_threshold"] == 0.8


def test_advanced_vector_search_rest_error(rest_client, monkeypatch):
    def boom(*a, **k):
        raise cv1.requests.RequestException("net")

    monkeypatch.setattr(cv1.requests, "post", boom)
    with pytest.raises(NetworkError):
        rest_client.advanced_vector_search("col", [0.1])


def test_advanced_vector_search_grpc(grpc_client):
    # The source iterates ``response.results`` then ``result_list.results`` —
    # i.e. it expects a list-of-result-lists shape. Provide a matching fake
    # response so the parsing loop is exercised end to end. ``filters`` is left
    # None because the SearchQuery.filters map only accepts SqlValue messages.
    inner = vector_types_pb2.SearchResult(
        results=[vector_types_pb2.SearchVectorRecord(id="r1", score=0.5, vector=[0.1])]
    )

    class FakeResults:
        results = [inner]
        execution_time_ms = 7

    grpc_client.vector_stub.SearchVectors.return_value = FakeResults()
    out = grpc_client.advanced_vector_search(
        "col",
        [0.1],
        accuracy_threshold=0.9,
        search_params={
            "timeout_ms": 50,
            "enable_two_stage": True,
            "enable_clustering_hint": True,
            "enable_metadata_filtering_hint": True,
        },
    )
    assert out["total_count"] == 1
    assert out["results"][0]["id"] == "r1"
    assert out["execution_time_ms"] == 7


def test_advanced_vector_search_grpc_error(grpc_client):
    grpc_client.vector_stub.SearchVectors.side_effect = FakeRpcError()
    with pytest.raises(ProximaDBError):
        grpc_client.advanced_vector_search("col", [0.1])


# --------------------------------------------------------------------------
# SQL
# --------------------------------------------------------------------------
def test_execute_sql_rest(rest_client, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        cv1.requests,
        "post",
        lambda url, json=None, **k: captured.update(url=url, json=json)
        or FakeResp({"rows": [{"a": 1}]}),
    )
    out = rest_client.execute_sql("SELECT 1", parameters=[1, "x"])
    assert out["rows"] == [{"a": 1}]
    assert captured["json"]["language"] == "uql"
    assert captured["json"]["parameters"] == [1, "x"]
    assert "/api/v2/query" in captured["url"]


def test_execute_sql_rest_network_error(rest_client, monkeypatch):
    def boom(*a, **k):
        raise cv1.requests.RequestException("net")

    monkeypatch.setattr(cv1.requests, "post", boom)
    with pytest.raises(NetworkError):
        rest_client.execute_sql("SELECT 1")


def test_execute_sql_grpc(grpc_client):
    row = types_pb2.SqlRow(
        fields=[
            types_pb2.SqlRowField(
                key="name", value=types_pb2.SqlValue(string_value="abc")
            ),
            types_pb2.SqlRowField(key="num", value=types_pb2.SqlValue(int64_value=42)),
        ]
    )
    resp = types_pb2.ExecuteQueryResponse(rows=[row], rows_scanned=10, rows_returned=1)
    grpc_client.sql_stub.ExecuteQuery.return_value = resp
    out = grpc_client.execute_sql(
        "SELECT * FROM t WHERE x = ?",
        parameters=["s", 1, 1.5, True, None, b"bytes", [1, 2], {"k": "v"}],
    )
    assert out["rows"][0]["name"] == "abc"
    assert out["rows"][0]["num"] == 42
    assert out["rows_scanned"] == 10


def test_execute_sql_grpc_error(grpc_client):
    grpc_client.sql_stub.ExecuteQuery.side_effect = FakeRpcError()
    with pytest.raises(ProximaDBError):
        grpc_client.execute_sql("SELECT 1")


def test_convert_to_from_sql_value_roundtrip(rest_client):
    cases = ["str", 7, 3.14, True, None, b"data", [1, "two"], {"k": 1}]
    for original in cases:
        proto = rest_client._convert_to_sql_value(original)
        back = rest_client._convert_from_sql_value(proto)
        if isinstance(original, bytes):
            assert back == original
        elif isinstance(original, list):
            assert back == [1, "two"]
        elif isinstance(original, dict):
            assert back == {"k": 1}
        else:
            assert back == original


def test_convert_to_sql_value_fallback(rest_client):
    class Weird:
        def __str__(self):
            return "weird"

    proto = rest_client._convert_to_sql_value(Weird())
    assert proto.string_value == "weird"


def test_convert_metadata_to_sql_value(rest_client):
    out = rest_client._convert_metadata_to_sql_value({"a": 1, "b": "x"})
    assert out["a"].int64_value == 1
    assert out["b"].string_value == "x"
    assert rest_client._convert_metadata_to_sql_value(None) == {}


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------
def test_health_check(rest_client, monkeypatch):
    monkeypatch.setattr(cv1.requests, "get", lambda *a, **k: FakeResp({"status": "ok"}))
    assert rest_client.health_check() == {"status": "ok"}


def test_health_check_error(rest_client, monkeypatch):
    def boom(*a, **k):
        raise cv1.requests.RequestException("dead")

    monkeypatch.setattr(cv1.requests, "get", boom)
    with pytest.raises(NetworkError):
        rest_client.health_check()


# --------------------------------------------------------------------------
# Graph - REST
# --------------------------------------------------------------------------
def test_create_node_rest(rest_client, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        cv1.requests,
        "post",
        lambda url, json=None, **k: captured.update(url=url, json=json)
        or FakeResp({"id": "n1"}),
    )
    out = rest_client.create_node("n1", ["Person"], {"name": "Bob"}, embedding=[0.1])
    assert out == {"id": "n1"}
    assert captured["json"]["embedding"] == [0.1]
    assert "/nodes" in captured["url"]


def test_create_node_rest_error(rest_client, monkeypatch):
    def boom(*a, **k):
        raise cv1.requests.RequestException("net")

    monkeypatch.setattr(cv1.requests, "post", boom)
    with pytest.raises(NetworkError):
        rest_client.create_node("n1", ["L"])


def test_create_edge_rest(rest_client, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        cv1.requests,
        "post",
        lambda url, json=None, **k: captured.update(json=json)
        or FakeResp({"id": "e1"}),
    )
    out = rest_client.create_edge("e1", "a", "b", "KNOWS", {"since": 2020}, weight=0.5)
    assert out == {"id": "e1"}
    assert captured["json"]["weight"] == 0.5


def test_create_edge_rest_error(rest_client, monkeypatch):
    def boom(*a, **k):
        raise cv1.requests.RequestException("net")

    monkeypatch.setattr(cv1.requests, "post", boom)
    with pytest.raises(NetworkError):
        rest_client.create_edge("e1", "a", "b", "KNOWS")


def test_traverse_graph_rest(rest_client, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        cv1.requests,
        "post",
        lambda url, json=None, **k: captured.update(json=json)
        or FakeResp({"nodes": []}),
    )
    out = rest_client.traverse_graph("n1", max_depth=2, edge_types=["KNOWS"], limit=5)
    assert out == {"nodes": []}
    assert captured["json"]["algorithm"] == "BFS"
    assert captured["json"]["limit"] == 5


def test_traverse_graph_rest_error(rest_client, monkeypatch):
    def boom(*a, **k):
        raise cv1.requests.RequestException("net")

    monkeypatch.setattr(cv1.requests, "post", boom)
    with pytest.raises(NetworkError):
        rest_client.traverse_graph("n1")


def test_query_nodes_rest(rest_client, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        cv1.requests,
        "post",
        lambda url, json=None, **k: captured.update(json=json)
        or FakeResp({"nodes": []}),
    )
    out = rest_client.query_nodes(
        labels=["Person"], properties={"x": 1}, limit=10, offset=2
    )
    assert out == {"nodes": []}
    assert captured["json"]["limit"] == 10
    assert captured["json"]["offset"] == 2


def test_query_nodes_rest_error(rest_client, monkeypatch):
    def boom(*a, **k):
        raise cv1.requests.RequestException("net")

    monkeypatch.setattr(cv1.requests, "post", boom)
    with pytest.raises(NetworkError):
        rest_client.query_nodes()


def test_hybrid_search_rest(rest_client, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        cv1.requests,
        "post",
        lambda url, json=None, **k: captured.update(json=json)
        or FakeResp({"nodes": []}),
    )
    out = rest_client.hybrid_search(
        "col",
        [0.1],
        top_k=5,
        start_node_id="n1",
        max_depth=3,
        combination_strategy="balanced",
        edge_types=["KNOWS"],
        limit=7,
    )
    assert out == {"nodes": []}
    assert captured["json"]["combination_strategy"] == "BALANCED"
    assert captured["json"]["graph_traversal"]["start_node_id"] == "n1"
    assert captured["json"]["limit"] == 7


def test_hybrid_search_rest_no_start_node(rest_client, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        cv1.requests,
        "post",
        lambda url, json=None, **k: captured.update(json=json) or FakeResp({}),
    )
    rest_client.hybrid_search("col", [0.1])
    assert "graph_traversal" not in captured["json"]


def test_hybrid_search_rest_error(rest_client, monkeypatch):
    def boom(*a, **k):
        raise cv1.requests.RequestException("net")

    monkeypatch.setattr(cv1.requests, "post", boom)
    with pytest.raises(NetworkError):
        rest_client.hybrid_search("col", [0.1])


# --------------------------------------------------------------------------
# Graph - gRPC (error paths + request building)
# --------------------------------------------------------------------------
def test_create_node_grpc_error(grpc_client):
    grpc_client.graph_stub.CreateNode.side_effect = FakeRpcError()
    with pytest.raises(ProximaDBError):
        grpc_client.create_node("n1", ["L"], {"k": "v"}, embedding=[0.1])


def test_create_edge_grpc_error(grpc_client):
    grpc_client.graph_stub.CreateEdge.side_effect = FakeRpcError()
    with pytest.raises(ProximaDBError):
        grpc_client.create_edge("e1", "a", "b", "T", {"k": 1}, weight=0.5)


def test_traverse_graph_grpc_error(grpc_client):
    grpc_client.graph_stub.TraverseGraph.side_effect = FakeRpcError()
    with pytest.raises(ProximaDBError):
        grpc_client.traverse_graph("n1", algorithm="DFS", limit=3)


def test_traverse_graph_grpc_parallel_bfs_error(grpc_client):
    grpc_client.graph_stub.TraverseGraph.side_effect = FakeRpcError()
    with pytest.raises(ProximaDBError):
        grpc_client.traverse_graph("n1", algorithm="PARALLEL_BFS")


def test_query_nodes_grpc_error(grpc_client):
    grpc_client.graph_stub.QueryNodes.side_effect = FakeRpcError()
    with pytest.raises(ProximaDBError):
        grpc_client.query_nodes(labels=["L"], properties={"k": "v"}, limit=5, offset=1)


def test_hybrid_search_grpc_error(grpc_client):
    grpc_client.graph_stub.ExecuteHybridQuery.side_effect = FakeRpcError()
    with pytest.raises(ProximaDBError):
        grpc_client.hybrid_search(
            "col",
            [0.1],
            combination_strategy="GRAPH_THEN_VECTOR",
            limit=5,
        )


def test_hybrid_search_grpc_balanced_error(grpc_client):
    grpc_client.graph_stub.ExecuteHybridQuery.side_effect = FakeRpcError()
    with pytest.raises(ProximaDBError):
        grpc_client.hybrid_search("col", [0.1], combination_strategy="BALANCED")


def test_hybrid_search_grpc_filters_source_bug(grpc_client):
    # Source bug: SearchQuery.filters is a map<string, SqlValue>; assigning a
    # Python value via ``search_query.filters[key] = ...`` raises ValueError.
    with pytest.raises(ValueError):
        grpc_client.hybrid_search("col", [0.1], vector_filters={"k": "v"})


def test_advanced_vector_search_grpc_filters_source_bug(grpc_client):
    # Same map-assignment bug on the advanced search gRPC path.
    with pytest.raises(ValueError):
        grpc_client.advanced_vector_search("col", [0.1], filters={"k": "v"})


def test_search_vectors_grpc_filters_source_bug(grpc_client):
    # SearchQuery(filters={...}) with string values cannot be constructed; the
    # proto map expects SqlValue messages.
    with pytest.raises(TypeError):
        grpc_client.search_vectors("col", [0.1], filters={"k": "v"})


# --------------------------------------------------------------------------
# Property value conversions
# --------------------------------------------------------------------------
def test_convert_property_value_branches(rest_client):
    c = rest_client
    assert c._convert_to_property_value("s").string_value == "s"
    assert c._convert_to_property_value(True).bool_value is True
    assert c._convert_to_property_value(5).int_value == 5
    assert c._convert_to_property_value(2.5).double_value == 2.5
    assert c._convert_to_property_value(b"x").bytes_value == b"x"
    arr = c._convert_to_property_value([1, "a"])
    assert len(arr.array_value.values) == 2
    obj = c._convert_to_property_value({"k": 1})
    assert "k" in obj.object_value.fields


def test_convert_property_value_fallback(rest_client):
    class Weird:
        def __str__(self):
            return "w"

    assert rest_client._convert_to_property_value(Weird()).string_value == "w"


def test_convert_from_property_value_branches(rest_client):
    c = rest_client
    assert (
        c._convert_from_property_value(graph_pb2.PropertyValue(string_value="s")) == "s"
    )
    assert c._convert_from_property_value(graph_pb2.PropertyValue(int_value=3)) == 3
    assert (
        c._convert_from_property_value(graph_pb2.PropertyValue(double_value=1.5)) == 1.5
    )
    assert (
        c._convert_from_property_value(graph_pb2.PropertyValue(bool_value=True)) is True
    )
    assert (
        c._convert_from_property_value(graph_pb2.PropertyValue(bytes_value=b"x"))
        == b"x"
    )
    arr = graph_pb2.PropertyValue(
        array_value=graph_pb2.PropertyArray(
            values=[graph_pb2.PropertyValue(int_value=1)]
        )
    )
    assert c._convert_from_property_value(arr) == [1]
    obj = graph_pb2.PropertyValue(
        object_value=graph_pb2.PropertyObject(
            fields={"k": graph_pb2.PropertyValue(string_value="v")}
        )
    )
    assert c._convert_from_property_value(obj) == {"k": "v"}
    # Unset field -> None
    assert c._convert_from_property_value(graph_pb2.PropertyValue()) is None


def test_convert_edge_from_proto_raises_on_timestamp_field(rest_client):
    # Source bug: _convert_edge_from_proto probes HasField("created_at") but the
    # proto field is "created_at_ms", so protobuf raises ValueError. Pin the
    # current behavior so the path is still exercised offline.
    edge = graph_pb2.Edge(id="e1", from_node_id="a", to_node_id="b", edge_type="T")
    with pytest.raises(ValueError):
        rest_client._convert_edge_from_proto(edge)


def test_convert_path_from_proto(rest_client):
    class PathWithIds:
        node_ids = ["a", "b"]

    class PathNoIds:
        pass

    assert rest_client._convert_path_from_proto(PathWithIds()) == ["a", "b"]
    assert rest_client._convert_path_from_proto(PathNoIds()) == []


def test_convert_search_result_from_proto(rest_client):
    rec = vector_types_pb2.SearchVectorRecord(id="r1", score=0.7, vector=[0.1])
    out = rest_client._convert_search_result_from_proto(rec)
    assert out["id"] == "r1"
    assert out["score"] == pytest.approx(0.7)
    assert out["vector"] == [pytest.approx(0.1)]
