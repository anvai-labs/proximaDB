"""Offline unit tests for proximadb_sdk.protocols.grpc_sync.

Fully offline: the gRPC connection pool is replaced with a dummy that never
opens a real channel, and every *Stub class on the module is monkeypatched so
RPC wrappers return real *_pb2 response messages (or hand fakes). No socket,
no sleep, no server.
"""

import json

import grpc
import pytest

import proximadb_sdk.protocols.grpc_sync as gs
from proximadb.v2 import record_pb2 as r
from proximadb_sdk.exceptions import ProximaDBError
from proximadb_sdk.v1 import collection_types_pb2 as c
from proximadb_sdk.v1 import types_pb2 as v1_types


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeRpcError(grpc.RpcError):
    """A grpc.RpcError with code()/details() that the wrappers inspect."""

    def __init__(self, code, details):
        self._code = code
        self._details = details

    def code(self):
        return self._code

    def details(self):
        return self._details


class DummyChannel:
    """Fake gRPC channel.

    Real generated *Stub constructors call ``channel.unary_unary(...)`` etc. to
    build callables. We return a MagicMock for any such factory so constructing
    a *real* stub never touches a socket. The methods we actually exercise are
    overridden on the monkeypatched stub classes, so these factory-produced
    callables are never invoked.
    """

    def __getattr__(self, name):
        from unittest.mock import MagicMock

        return MagicMock()


class DummyPool:
    """Stand-in for GrpcConnectionPool — never connects."""

    def __init__(self, *args, **kwargs):
        self.closed = False

    def get_channel(self):
        return DummyChannel()

    def return_channel(self, channel, success=True, response_time_ms=0.0):
        return None

    def get_metrics(self):
        return {"requests_served": 7}

    def close(self):
        self.closed = True


@pytest.fixture
def client(monkeypatch):
    """A ProximaDBSyncGrpcClient whose pool is a DummyPool (offline)."""
    monkeypatch.setattr(gs, "GrpcConnectionPool", DummyPool)
    cl = gs.ProximaDBSyncGrpcClient("localhost:5678", timeout=1.0)
    return cl


def make_stub_factory(method_name, behavior):
    """Return a Stub class whose ``method_name`` calls ``behavior(req)``."""

    class _Stub:
        def __init__(self, channel):
            self.channel = channel

    def _method(self, req, timeout=None, metadata=None):
        return behavior(req)

    setattr(_Stub, method_name, _method)
    return _Stub


# --------------------------------------------------------------------------- #
# Wrapper helper classes
# --------------------------------------------------------------------------- #
def test_collection_wrapper():
    coll = c.Collection(id="cid", config=c.CollectionConfig(name="n", dimension=8))
    w = gs.CollectionWrapper(coll)
    assert w.name == "n"
    assert w.dimension == 8
    assert w.id == "cid"
    assert w.config is not None
    assert w.stats is not None
    assert w.created_at == coll.created_at
    assert "CollectionWrapper" in repr(w)


def test_collection_wrapper_missing_config():
    w = gs.CollectionWrapper(object())
    assert w.name is None
    assert w.dimension is None
    assert w.id is None


def test_search_results_wrapper():
    w = gs.SearchResultsWrapper([1, 2, 3])
    assert len(w) == 3
    assert list(iter(w)) == [1, 2, 3]
    assert w[0] == 1
    assert w.results == [1, 2, 3]
    assert "count=3" in repr(w)


def test_vector_wrapper():
    w = gs.VectorWrapper({"id": "v1", "vector": [0.1]})
    assert w.id == "v1"
    assert w["vector"] == [0.1]
    assert w.get("vector") == [0.1]
    assert w.get("missing", "d") == "d"
    assert "v1" in repr(w)


def test_dict_wrapper():
    w = gs.DictWrapper({"status": "ok", "count": 2})
    assert w.status == "ok"
    assert w["count"] == 2
    assert w.get("count") == 2
    assert w.get("nope") is None
    assert "status" in repr(w)


# --------------------------------------------------------------------------- #
# Construction / lifecycle / pool
# --------------------------------------------------------------------------- #
def test_init_with_compression(monkeypatch):
    monkeypatch.setattr(gs, "GrpcConnectionPool", DummyPool)
    cl = gs.ProximaDBSyncGrpcClient(
        "localhost:5678", enable_compression=True, compression_algorithm="gzip"
    )
    assert cl._connection_pool is cl._pool


def test_init_with_deflate_compression(monkeypatch):
    monkeypatch.setattr(gs, "GrpcConnectionPool", DummyPool)
    cl = gs.ProximaDBSyncGrpcClient(
        "localhost:5678", enable_compression=True, compression_algorithm="deflate"
    )
    assert cl is not None


def test_init_with_unknown_compression(monkeypatch):
    monkeypatch.setattr(gs, "GrpcConnectionPool", DummyPool)
    cl = gs.ProximaDBSyncGrpcClient(
        "localhost:5678", enable_compression=True, compression_algorithm="bogus"
    )
    assert cl is not None


def test_init_failure_wrapped(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(gs, "GrpcConnectionPool", boom)
    with pytest.raises(ProximaDBError):
        gs.ProximaDBSyncGrpcClient("localhost:5678")


def test_get_pool_metrics(client):
    assert client.get_pool_metrics() == {"requests_served": 7}


def test_get_pool_metrics_none(client):
    client._connection_pool = None
    assert client.get_pool_metrics() is None


def test_close_and_context_manager(monkeypatch):
    monkeypatch.setattr(gs, "GrpcConnectionPool", DummyPool)
    with gs.ProximaDBSyncGrpcClient("localhost:5678") as cl:
        pool = cl._connection_pool
    assert pool.closed is True


def test_close_swallows_errors(client):
    class BadPool:
        def close(self):
            raise RuntimeError("x")

    client._connection_pool = BadPool()
    client.close()  # should not raise


# --------------------------------------------------------------------------- #
# SqlValue encode/decode
# --------------------------------------------------------------------------- #
def test_python_to_sql_value_roundtrip(client):
    cases = [
        None,
        True,
        42,
        3.14,
        b"bytes",
        "hello",
        [1, "two", 3.0],
        {"k": "v", "n": 5},
    ]
    for value in cases:
        sv = client._python_to_sql_value(value)
        back = client._sql_value_to_python(sv)
        if isinstance(value, bytes):
            assert back == value
        elif value is None:
            assert back is None
        elif isinstance(value, list):
            assert back == [1, "two", 3.0]
        elif isinstance(value, dict):
            assert back == {"k": "v", "n": 5}
        else:
            assert back == value


def test_sql_value_to_python_unknown_kind(client):
    sv = v1_types.SqlValue()  # no oneof set
    assert client._sql_value_to_python(sv) is None


def test_sql_value_to_python_jsonb_preserves_wire_bytes(client):
    """The legacy SqlValue adapter must not silently drop tag-9 JSONB."""
    sv = v1_types.SqlValue()
    sv.jsonb_value = b"\x81\xa1x\x01"

    assert client._sql_value_to_python(sv) == b"\x81\xa1x\x01"


# --------------------------------------------------------------------------- #
# v2 TypedValue encode/decode
# --------------------------------------------------------------------------- #
def test_v2_typed_value_roundtrip(client):
    cases = [
        (None, None),
        (True, True),
        (7, 7),
        (1.5, 1.5),
        (b"bin", b"bin"),
        ("text", "text"),
        ({"a": 1}, {"a": 1}),
    ]
    for value, expected in cases:
        tv = client._python_to_v2_typed_value(value)
        assert client._v2_typed_value_to_python(tv) == expected


def test_v2_typed_value_type_hints(client):
    tv32 = client._python_to_v2_typed_value({"type": "float32", "value": 1.25})
    assert abs(client._v2_typed_value_to_python(tv32) - 1.25) < 1e-3
    sym = client._python_to_v2_typed_value({"type": "symbol", "value": "S"})
    assert client._v2_typed_value_to_python(sym) == "S"


def test_v2_typed_value_array(client):
    tv = client._python_to_v2_typed_value([1, 2, 3])
    assert client._v2_typed_value_to_python(tv) == [1, 2, 3]


def test_v2_typed_value_jsonb_decode(client):
    tv = r.TypedValue()
    tv.jsonb_value = json.dumps({"x": 1}).encode("utf-8")
    assert client._v2_typed_value_to_python(tv) == {"x": 1}


def test_v2_typed_value_fallback_object(client):
    tv = client._python_to_v2_typed_value(object())  # str() fallback path
    assert isinstance(client._v2_typed_value_to_python(tv), str)


def test_v2_typed_value_more_kinds(client):
    # timestamp / date / time / duration / uuid / binary / json / float32
    tv = r.TypedValue()
    tv.timestamp_value = 1700
    assert client._v2_typed_value_to_python(tv) == 1700

    tv = r.TypedValue()
    tv.date_value = 5
    assert client._v2_typed_value_to_python(tv) == 5

    tv = r.TypedValue()
    tv.time_value = 9
    assert client._v2_typed_value_to_python(tv) == 9

    tv = r.TypedValue()
    tv.duration_value = 12
    assert client._v2_typed_value_to_python(tv) == 12

    tv = r.TypedValue()
    tv.uuid_value = bytes.fromhex("00112233445566778899aabbccddeeff")
    assert client._v2_typed_value_to_python(tv) == "00112233445566778899aabbccddeeff"

    tv = r.TypedValue()
    tv.binary_value = b"\x01\x02"
    assert client._v2_typed_value_to_python(tv) == b"\x01\x02"

    tv = r.TypedValue()
    tv.json_value = '{"a": 1}'
    assert client._v2_typed_value_to_python(tv) == {"a": 1}

    tv = r.TypedValue()
    tv.float32_value = 0.5
    assert abs(client._v2_typed_value_to_python(tv) - 0.5) < 1e-3


def test_v2_typed_value_json_invalid(client):
    tv = r.TypedValue()
    tv.json_value = "not-json"
    assert client._v2_typed_value_to_python(tv) == "not-json"


def test_v2_typed_value_jsonb_invalid_bytes(client):
    tv = r.TypedValue()
    tv.jsonb_value = b"\xff\xfe"
    assert client._v2_typed_value_to_python(tv) == b"\xff\xfe"


def test_v2_typed_value_array_field(client):
    tv = r.TypedValue()
    tv.integer_array.values.extend([1, 2, 3])
    assert client._v2_typed_value_to_python(tv) == [1, 2, 3]


# --------------------------------------------------------------------------- #
# normalize vector alias records / record proto builder
# --------------------------------------------------------------------------- #
def test_normalize_vector_alias_records_variants(client):
    class HasModelDump:
        def model_dump(self, exclude_none=False):
            return {"id": "md", "vector": [0.1], "metadata": {"k": "v"}}

    class HasDict:
        def __init__(self):
            self.id = "dd"
            self.vector = [0.2]
            self._private = "skip"

        def dict(self, exclude_none=False):
            return {"id": "dd", "vector": [0.2]}

    inputs = [
        {"id": "a", "vector": [0.1], "props": {"p": 1}},
        {"oid": "b", "vector": [0.2]},
        HasModelDump(),
        HasDict(),
        [0.5, 0.6],  # bare vector
    ]
    out = client._normalize_vector_alias_records(inputs)
    assert out[0]["id"] == "a"
    assert out[1]["id"] == "b"
    assert out[2]["id"] == "md"
    assert out[3]["id"] == "dd"
    assert out[4]["id"] == "record_4"
    assert out[4]["vector"] == [0.5, 0.6]


def test_record_proto_for_grpc_full(client):
    record = {
        "id": "rec1",
        "vector": [0.1, 0.2],
        "vector_dimension": 2,
        "props": {"category": "x"},
        "typed_fields": {"score": {"value_type": "float32", "value": 0.9}},
        "timestamp_ms": 1234,
        "version": 2,
        "partition_values": {"region": "us"},
        "custom_metadata": {"src": "test"},
    }
    proto = client._record_proto_for_grpc(record)
    assert proto.id == "rec1"
    assert list(proto.vector) == pytest.approx([0.1, 0.2])
    assert "category" in proto.props
    assert proto.timestamp_ms == 1234


def test_record_proto_from_embeddings(client):
    record = {"id": "e1", "embeddings": [{"values": [0.3, 0.4]}]}
    proto = client._record_proto_for_grpc(record)
    assert list(proto.vector) == pytest.approx([0.3, 0.4])


def test_record_proto_missing_vector(client):
    with pytest.raises(ValueError):
        client._record_proto_for_grpc({"id": "x"})


def test_record_proto_bad_type(client):
    with pytest.raises(TypeError):
        client._record_proto_for_grpc(12345)


# --------------------------------------------------------------------------- #
# Collection operations — v2 ProximaRecordService (collection RPCs are served
# by ProximaRecordServiceStub via _execute_collection_with_pool). The create/get
# responses are wrapped in CollectionWrapper, which reads .config.name /
# .config.dimension / .id off the proto.
# --------------------------------------------------------------------------- #
class _FakeCollectionConfig:
    def __init__(self, name, dimension):
        self.name = name
        self.dimension = dimension


class _FakeCollection:
    """Minimal object satisfying CollectionWrapper's attribute access."""

    def __init__(self, name, dimension, cid="cid"):
        self.id = cid
        self.config = _FakeCollectionConfig(name, dimension)
        self.stats = None
        self.created_at = None


def _patch_collection_stub(monkeypatch, method, behavior):
    monkeypatch.setattr(
        gs.v2_record_pb2_grpc,
        "ProximaRecordServiceStub",
        make_stub_factory(method, behavior),
    )


def _patch_collection_v1_stub(monkeypatch, method, behavior):
    """Patch the v1 CollectionServiceStub used by the *_v1 collection helpers
    (they construct their stub inside _op and route through _execute_with_pool)."""
    monkeypatch.setattr(
        gs.v1_collection_pb2_grpc,
        "CollectionServiceStub",
        make_stub_factory(method, behavior),
    )


def test_create_collection(client, monkeypatch):
    def behavior(cfg):
        return _FakeCollection(name=cfg.name, dimension=cfg.dimension)

    _patch_collection_stub(monkeypatch, "CreateCollection", behavior)
    res = client.create_collection(
        "mycoll", dimension=16, distance_metric=1, storage_engine=1
    )
    assert res.name == "mycoll"
    assert res.dimension == 16


def test_create_collection_engine_alias_and_string(client, monkeypatch):
    def behavior(cfg):
        return _FakeCollection(name=cfg.name, dimension=cfg.dimension)

    _patch_collection_stub(monkeypatch, "CreateCollection", behavior)
    res = client.create_collection(
        "c2",
        dimension=4,
        indexing_algorithm=1,
        engine="viper",
    )
    assert res.name == "c2"


def test_create_collection_bad_storage_engine(client):
    with pytest.raises(ValueError):
        client.create_collection("c3", dimension=4, storage_engine="not-a-real-engine")


def test_get_collection(client, monkeypatch):
    def behavior(req):
        assert req.collection_id == "mycoll"
        return _FakeCollection(name="mycoll", dimension=8)

    _patch_collection_stub(monkeypatch, "GetCollection", behavior)
    res = client.get_collection("mycoll")
    assert res.name == "mycoll"


def test_list_collections(client, monkeypatch):
    def behavior(req):
        class _Resp:
            collections = [
                _FakeCollection(name="a", dimension=2, cid="a"),
                _FakeCollection(name="b", dimension=3, cid="b"),
            ]

        return _Resp()

    _patch_collection_stub(monkeypatch, "ListCollections", behavior)
    res = client.list_collections()
    assert [coll.name for coll in res] == ["a", "b"]


def test_delete_collection(client, monkeypatch):
    def behavior(req):
        # v2 V2DeleteCollectionResponse only carries `success`.
        return r.V2DeleteCollectionResponse(success=True)

    _patch_collection_stub(monkeypatch, "DeleteCollection", behavior)
    res = client.delete_collection("dead")
    assert res.success is True
    assert res.collection_id == "dead"
    assert res.status == "deleted"


def test_create_collection_v1(client, monkeypatch):
    def behavior(cfg):
        return c.Collection(
            id="x", config=c.CollectionConfig(name=cfg.name, dimension=cfg.dimension)
        )

    _patch_collection_v1_stub(monkeypatch, "CreateCollection", behavior)
    res = client.create_collection_v1(
        "v1c", dimension=5, distance_metric=1, storage_engine=1, tags=["t"]
    )
    assert res.config.name == "v1c"


def test_get_collection_v1(client, monkeypatch):
    _patch_collection_v1_stub(
        monkeypatch, "GetCollection", lambda req: c.Collection(id=req.collection_id)
    )
    res = client.get_collection_v1("gid")
    assert res.id == "gid"


def test_list_collections_v1(client, monkeypatch):
    _patch_collection_v1_stub(
        monkeypatch, "ListCollections", lambda req: c.ListCollectionsResponse()
    )
    res = client.list_collections_v1(limit=5, offset=1, include_stats=True)
    assert len(res.collections) == 0


def test_delete_collection_v1(client, monkeypatch):
    _patch_collection_v1_stub(
        monkeypatch,
        "DeleteCollection",
        lambda req: c.DeleteCollectionResponse(success=True),
    )
    res = client.delete_collection_v1("zid")
    assert res.success is True


# --------------------------------------------------------------------------- #
# Health check — uses v2 ProximaRecordService.ListCollections as a probe.
# --------------------------------------------------------------------------- #
def test_health_check_ok(client, monkeypatch):
    _patch_collection_stub(
        monkeypatch, "ListCollections", lambda req: r.V2ListCollectionsResponse()
    )
    res = client.health_check()
    assert res.healthy is True
    assert res.status == "connected"
    assert res.server_address == "localhost:5678"


def test_health_check_rpc_error(client, monkeypatch):
    def behavior(req):
        raise FakeRpcError(grpc.StatusCode.UNAVAILABLE, "down")

    _patch_collection_stub(monkeypatch, "ListCollections", behavior)
    res = client.health_check()
    assert res.healthy is False
    assert res.details == "down"


def test_health_check_generic_error(client, monkeypatch):
    def behavior(req):
        raise RuntimeError("boom")

    _patch_collection_stub(monkeypatch, "ListCollections", behavior)
    res = client.health_check()
    assert res.healthy is False
    assert "boom" in res.details


# --------------------------------------------------------------------------- #
# Record / vector operations (v2 ProximaRecordService)
# --------------------------------------------------------------------------- #
def _patch_record_stub(monkeypatch, method, behavior):
    monkeypatch.setattr(
        gs.v2_record_pb2_grpc,
        "ProximaRecordServiceStub",
        make_stub_factory(method, behavior),
    )


def _batch_resp(success=2, failed=0, errors=None):
    resp = r.ProximaRecordBatchResponse(
        success=failed == 0,
        total_processed=success + failed,
        success_count=success,
        failed_count=failed,
        processing_time_us=123,
    )
    for err in errors or []:
        resp.errors.add(record_id=err[0], error_message=err[1])
    return resp


def test_insert_records(client, monkeypatch):
    _patch_record_stub(monkeypatch, "InsertRecords", lambda req: _batch_resp(success=2))
    res = client.insert_records(
        "col", [{"id": "1", "vector": [0.1]}, {"id": "2", "vector": [0.2]}]
    )
    assert res.success == 2
    assert res.failed == 0


def test_insert_records_with_errors(client, monkeypatch):
    _patch_record_stub(
        monkeypatch,
        "InsertRecords",
        lambda req: _batch_resp(success=1, failed=1, errors=[("2", "bad")]),
    )
    res = client.insert_records(
        "col", [{"id": "1", "vector": [0.1]}, {"id": "2", "vector": [0.2]}]
    )
    assert res.failed == 1
    assert "bad" in res.errors[0]


def test_insert_records_upsert_kwarg_routes(client, monkeypatch):
    _patch_record_stub(monkeypatch, "UpsertRecords", lambda req: _batch_resp(success=1))
    res = client.insert_records("col", [{"id": "1", "vector": [0.1]}], upsert=True)
    assert res.success == 1


def test_upsert_records(client, monkeypatch):
    _patch_record_stub(monkeypatch, "UpsertRecords", lambda req: _batch_resp(success=1))
    res = client.upsert_records("col", [{"id": "1", "vector": [0.1]}], schema_id="s1")
    assert res.success == 1


def test_insert_vectors_and_insert_vector(client, monkeypatch):
    _patch_record_stub(monkeypatch, "InsertRecords", lambda req: _batch_resp(success=1))
    res = client.insert_vectors(
        "col", [{"id": "v1", "vector": [0.1], "metadata": {"k": "v"}}]
    )
    assert res.success is True
    assert res.operation == "INSERT"
    assert res.vector_ids == ["v1"]

    res2 = client.insert_vector("col", "vx", [0.5], metadata={"m": 1})
    assert res2.vector_ids == ["vx"]


def test_insert_vectors_upsert(client, monkeypatch):
    _patch_record_stub(monkeypatch, "UpsertRecords", lambda req: _batch_resp(success=1))
    res = client.insert_vectors("col", [{"id": "v1", "vector": [0.1]}], upsert=True)
    assert res.operation == "UPSERT"


def test_update_vector(client, monkeypatch):
    _patch_record_stub(monkeypatch, "UpsertRecords", lambda req: _batch_resp(success=1))
    res = client.update_vector("col", "v1", vector=[0.9], metadata={"k": "v"})
    assert res["status"] == "updated"
    assert res["success"] is True


def test_search_vectors(client, monkeypatch):
    def behavior(req):
        resp = r.TypedSearchResponse()
        item = resp.results.add(id="v1", score=0.9)
        item.props["cat"].CopyFrom(client._python_to_v2_typed_value("news"))
        item.vector.extend([0.1, 0.2])
        return resp

    _patch_record_stub(monkeypatch, "Search", behavior)
    res = client.search_vectors(
        "col",
        query_vector=[0.1, 0.2],
        top_k=3,
        metadata_filters={"cat": "news"},
        include_vectors=True,
        include_metadata=True,
        search_hints={"ef": 64},
    )
    assert len(res) == 1
    assert res[0].id == "v1"
    assert res[0].metadata == {"cat": "news"}
    assert res[0].vector == pytest.approx([0.1, 0.2])


def test_search_vectors_requires_query(client):
    with pytest.raises(ValueError):
        client.search_vectors("col")


def test_search_alias(client, monkeypatch):
    _patch_record_stub(monkeypatch, "Search", lambda req: r.TypedSearchResponse())
    res = client.search(collection_name="col", query_vector=[0.1], k=5)
    assert len(res) == 0


def test_search_requires_collection(client):
    with pytest.raises(ValueError):
        client.search(query_vector=[0.1])


def test_get_vector(client, monkeypatch):
    # v2 ProximaRecordService.GetRecord returns {found, record}.
    record = r.ProximaRecord(id="v1")
    record.vector.extend([0.1, 0.2])

    class FakeResponse:
        found = True

        def __init__(self):
            self.record = record

        def HasField(self, name):
            return name == "record"

    _patch_collection_stub(monkeypatch, "GetRecord", lambda req: FakeResponse())
    res = client.get_vector("col", "v1", include_vector=True, include_metadata=False)
    assert res.id == "v1"
    assert res["vector"] == pytest.approx([0.1, 0.2])


def test_get_vector_with_metadata_and_fields(client, monkeypatch):
    record = r.ProximaRecord(id="v1")
    record.vector.extend([0.1, 0.2])
    # v2 props map: str -> TypedValue (decoded via _v2_typed_value_to_python).
    record.props["cat"].CopyFrom(client._python_to_v2_typed_value("news"))
    record.props["rank"].CopyFrom(client._python_to_v2_typed_value(3))

    class FakeResponse:
        found = True

        def __init__(self):
            self.record = record

        def HasField(self, name):
            return name == "record"

    _patch_collection_stub(monkeypatch, "GetRecord", lambda req: FakeResponse())
    res = client.get_vector("col", "v1", include_vector=True, include_metadata=True)
    assert res.id == "v1"
    assert res["metadata"]["cat"] == "news"
    assert res["metadata"]["rank"] == 3


def test_get_vector_empty_results(client, monkeypatch):
    class FakeResponse:
        found = False

        def HasField(self, name):
            return False

    _patch_collection_stub(monkeypatch, "GetRecord", lambda req: FakeResponse())
    with pytest.raises(ProximaDBError):
        client.get_vector("col", "v1")


def test_get_vector_not_found(client, monkeypatch):
    class FakeResponse:
        found = False

        def HasField(self, name):
            return False

    _patch_collection_stub(monkeypatch, "GetRecord", lambda req: FakeResponse())
    with pytest.raises(ProximaDBError):
        client.get_vector("col", "missing")


def test_delete_vector(client, monkeypatch):
    _patch_record_stub(
        monkeypatch, "DeleteRecords", lambda req: _batch_resp(success=1, failed=0)
    )
    res = client.delete_vector("col", "v1")
    assert res["status"] == "deleted"
    assert res["success"] is True


def test_delete_vectors(client, monkeypatch):
    _patch_record_stub(
        monkeypatch, "DeleteRecords", lambda req: _batch_resp(success=2, failed=0)
    )
    res = client.delete_vectors("col", ["v1", "v2"])
    assert res["status"] == "completed"
    assert res["deleted_count"] == 2
    assert res["total_requested"] == 2


# --------------------------------------------------------------------------- #
# SQL — v2 ProximaRecordService.ExecuteQuery (inline GrpcChannelContext path).
# --------------------------------------------------------------------------- #
def test_execute_sql(client, monkeypatch):
    def behavior(req):
        resp = r.V2QueryResponse(rows_returned=1, execution_time_ms=5)
        resp.columns.extend(["name"])
        row = resp.rows.add()
        row.values["name"].CopyFrom(client._python_to_v2_typed_value("alice"))
        return resp

    _patch_collection_stub(monkeypatch, "ExecuteQuery", behavior)
    res = client.execute_sql("SELECT name", parameters=["x", 5], collection="col")
    assert res["row_count"] == 1
    assert res["rows"][0]["name"] == "alice"
    assert res["columns"] == ["name"]


def test_execute_sql_rpc_error(client, monkeypatch):
    def behavior(req):
        raise FakeRpcError(grpc.StatusCode.INTERNAL, "sql-boom")

    _patch_collection_stub(monkeypatch, "ExecuteQuery", behavior)
    with pytest.raises(ProximaDBError):
        client.execute_sql("SELECT 1")


# --------------------------------------------------------------------------- #
# Error mapping in the pooled executors
# --------------------------------------------------------------------------- #
def test_execute_with_pool_unavailable_maps_connection(client, monkeypatch):
    def behavior(req):
        raise FakeRpcError(grpc.StatusCode.UNAVAILABLE, "no route")

    monkeypatch.setattr(
        gs.v2_record_pb2_grpc,
        "ProximaRecordServiceStub",
        make_stub_factory("DeleteRecords", behavior),
    )
    with pytest.raises(ProximaDBError) as ei:
        client.delete_vector("col", "v1")
    assert "connection failed" in str(ei.value)


def test_execute_with_pool_other_rpc_error(client, monkeypatch):
    def behavior(req):
        raise FakeRpcError(grpc.StatusCode.INTERNAL, "internal")

    monkeypatch.setattr(
        gs.v2_record_pb2_grpc,
        "ProximaRecordServiceStub",
        make_stub_factory("DeleteRecords", behavior),
    )
    with pytest.raises(ProximaDBError) as ei:
        client.delete_vector("col", "v1")
    assert "RPC failed" in str(ei.value)


def test_execute_with_pool_generic_error(client, monkeypatch):
    def behavior(req):
        raise RuntimeError("generic")

    monkeypatch.setattr(
        gs.v2_record_pb2_grpc,
        "ProximaRecordServiceStub",
        make_stub_factory("DeleteRecords", behavior),
    )
    with pytest.raises(ProximaDBError) as ei:
        client.delete_vector("col", "v1")
    assert "failed" in str(ei.value)


def test_collection_pool_rpc_error(client, monkeypatch):
    _patch_collection_stub(
        monkeypatch,
        "GetCollection",
        lambda req: (_ for _ in ()).throw(FakeRpcError(grpc.StatusCode.INTERNAL, "x")),
    )
    with pytest.raises(ProximaDBError):
        client.get_collection("c")


# --------------------------------------------------------------------------- #
# GRPC_AVAILABLE guard paths
# --------------------------------------------------------------------------- #
def test_grpc_unavailable_guards(client, monkeypatch):
    monkeypatch.setattr(gs, "GRPC_AVAILABLE", False)
    with pytest.raises(ProximaDBError):
        client.health_check()
    with pytest.raises(ProximaDBError):
        client.execute_sql("SELECT 1")
    with pytest.raises(ProximaDBError):
        client.get_collection("c")
    with pytest.raises(ProximaDBError):
        client.delete_vector("c", "v")


def test_record_stub_unavailable(client, monkeypatch):
    monkeypatch.setattr(gs, "v2_record_pb2_grpc", None)
    with pytest.raises(ProximaDBError):
        client.delete_vector("c", "v")


# --------------------------------------------------------------------------- #
# Graph operations (v2 ProximaGraphService) — property conversions + RPC
# wrappers. Graph ops use an inline GrpcChannelContext constructing
# v2_graph_pb2_grpc.ProximaGraphServiceStub directly. Responses decode via
# _convert_node_from_proto / _convert_edge_from_proto which expose a
# ``properties`` key (NOT ``metadata``).
# --------------------------------------------------------------------------- #
def _patch_graph_stub(monkeypatch, method, behavior):
    monkeypatch.setattr(
        gs.v2_graph_pb2_grpc,
        "ProximaGraphServiceStub",
        make_stub_factory(method, behavior),
    )


def test_property_value_roundtrip(client):
    for value in ["s", True, 5, 2.5, b"b", [1, "a"], {"k": "v"}]:
        pv = client._convert_to_property_value(value)
        back = client._convert_from_property_value(pv)
        if isinstance(value, list):
            assert back == [1, "a"]
        elif isinstance(value, dict):
            assert back == {"k": "v"}
        elif isinstance(value, bytes):
            assert back == b"b"
        else:
            assert back == value


def test_property_value_fallback(client):
    pv = client._convert_to_property_value(object())
    assert isinstance(client._convert_from_property_value(pv), str)


def test_create_node(client, monkeypatch):
    gp = gs.v2_graph_pb2

    def behavior(req):
        # Echo the request's encoded properties back so the round-trip is verified.
        node = gp.GraphNode(
            id="n1", labels=["L"], created_at_ms=1000, updated_at_ms=2000
        )
        node.properties["k"].CopyFrom(req.node.properties["k"])
        return gp.GraphNodeResponse(node=node)

    _patch_graph_stub(monkeypatch, "CreateNode", behavior)
    res = client.create_node("n1", ["L"], properties={"k": "v"}, graph_id="g")
    assert res["id"] == "n1"
    assert res["labels"] == ["L"]
    assert res["properties"] == {"k": "v"}
    assert res["created_at"] is not None


def test_create_edge(client, monkeypatch):
    gp = gs.v2_graph_pb2

    def behavior(req):
        edge = gp.GraphEdge(
            id="e1", from_node_id="a", to_node_id="b", edge_type="REL", weight=1.5
        )
        edge.properties["k"].CopyFrom(req.edge.properties["k"])
        return gp.GraphEdgeResponse(edge=edge)

    _patch_graph_stub(monkeypatch, "CreateEdge", behavior)
    res = client.create_edge("e1", "a", "b", "REL", properties={"k": 1}, weight=1.5)
    assert res["id"] == "e1"
    assert res["from_node_id"] == "a"
    assert res["weight"] == pytest.approx(1.5)
    assert res["properties"] == {"k": 1}


def test_traverse_graph(client, monkeypatch):
    gp = gs.v2_graph_pb2

    def behavior(req):
        resp = gp.TraverseGraphResponse()
        resp.nodes.add(id="n1")
        resp.edges.add(id="e1", from_node_id="n1", to_node_id="n2", edge_type="R")
        return resp

    _patch_graph_stub(monkeypatch, "TraverseGraph", behavior)
    res = client.traverse_graph("n1", algorithm="DFS", limit=10, edge_types=["R"])
    assert res["nodes"][0]["id"] == "n1"
    assert res["edges"][0]["id"] == "e1"
    assert "stats" in res


def test_query_nodes(client, monkeypatch):
    gp = gs.v2_graph_pb2

    def behavior(req):
        resp = gp.QueryGraphNodesResponse()
        resp.nodes.add(id="n1")
        return resp

    _patch_graph_stub(monkeypatch, "QueryNodes", behavior)
    res = client.query_nodes(labels=["L"], properties={"k": "v"}, limit=5, offset=0)
    assert res["total_count"] == 1
    assert res["nodes"][0]["id"] == "n1"


def test_query_edges(client, monkeypatch):
    gp = gs.v2_graph_pb2

    def behavior(req):
        resp = gp.QueryGraphEdgesResponse()
        resp.edges.add(id="e1", from_node_id="a", to_node_id="b", edge_type="R")
        return resp

    _patch_graph_stub(monkeypatch, "QueryEdges", behavior)
    res = client.query_edges(
        edge_type="R", from_node_id="a", to_node_id="b", limit=5, offset=1
    )
    assert res["total_count"] == 1


def test_get_node(client, monkeypatch):
    gp = gs.v2_graph_pb2

    def behavior(req):
        node = gp.GraphNode(id="n1", labels=["L"])
        return gp.GraphNodeResponse(node=node)

    _patch_graph_stub(monkeypatch, "GetNode", behavior)
    res = client.get_node("n1", graph_id="g")
    assert res["id"] == "n1"


def test_delete_node(client, monkeypatch):
    gp = gs.v2_graph_pb2

    def behavior(req):
        # DeleteNode returns DeleteGraphNodeResponse{deleted, node}
        node = gp.GraphNode(id="n1")
        return gp.DeleteGraphNodeResponse(deleted=True, node=node)

    _patch_graph_stub(monkeypatch, "DeleteNode", behavior)
    res = client.delete_node("n1")
    assert res["id"] == "n1"


def test_get_outgoing_and_incoming_edges(client, monkeypatch):
    gp = gs.v2_graph_pb2

    def behavior(req):
        resp = gp.QueryGraphEdgesResponse()
        resp.edges.add(id="e1", from_node_id="a", to_node_id="b", edge_type="R")
        return resp

    _patch_graph_stub(monkeypatch, "QueryEdges", behavior)
    out = client.get_outgoing_edges("a", edge_types=["R"])
    inc = client.get_incoming_edges("b", edge_types=["R"])
    assert out[0]["id"] == "e1"
    assert inc[0]["id"] == "e1"


def test_shortest_path(client, monkeypatch):
    gp = gs.v2_graph_pb2

    def behavior(req):
        return gp.GraphShortestPathResponse(
            node_ids=["a", "b"], total_weight=1.0, found=True
        )

    _patch_graph_stub(monkeypatch, "ShortestPath", behavior)
    res = client.shortest_path(
        "a",
        "b",
        max_depth=3,
        edge_types=["R"],
        algorithm="ASTAR",
        enable_prefetch=True,
        prefetch_budget=100,
    )
    assert res is not None


def test_graph_rpc_error(client, monkeypatch):
    def behavior(req):
        raise FakeRpcError(grpc.StatusCode.INTERNAL, "graph-boom")

    _patch_graph_stub(monkeypatch, "CreateNode", behavior)
    with pytest.raises(ProximaDBError):
        client.create_node("n1", ["L"])


def test_graph_generic_error(client, monkeypatch):
    def behavior(req):
        raise RuntimeError("generic-graph")

    _patch_graph_stub(monkeypatch, "GetNode", behavior)
    with pytest.raises(ProximaDBError):
        client.get_node("n1")


def test_graph_unavailable_guard(client, monkeypatch):
    monkeypatch.setattr(gs, "v2_graph_pb2_grpc", None)
    with pytest.raises(ProximaDBError):
        client.create_node("n1", ["L"])
    with pytest.raises(ProximaDBError):
        client.shortest_path("a", "b")


def test_convert_path_from_proto(client):
    class FakePath:
        node_ids = ["n1", "n2"]

    assert client._convert_path_from_proto(FakePath()) == ["n1", "n2"]
    assert client._convert_path_from_proto(object()) == []
