"""Offline unit tests for proximadb_sdk.unified_client_v2.

Strategy: patch the adapter factory (`create_adapter`) so the client's
`_setup_adapter` returns a MagicMock adapter. No network, no server, no
heavy deps. We exercise the v2 facade's request-shaping and response
unwrapping logic directly against the mock adapter.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import proximadb_sdk.unified_client_v2 as uc
from proximadb_sdk.config import Protocol
from proximadb_sdk.exceptions import ProximaDBError
from proximadb_sdk.models import (
    BatchResult,
    Collection,
    CollectionConfig,
    CollectionStats,
    OperationMetrics,
    SearchResult,
    VectorRecord,
)
from proximadb_sdk.unified_client_v2 import (
    ProximaDBClient,
    connect,
    connect_embedded,
    connect_grpc,
    connect_rest,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_NAME = "test_collection_name"  # >= 8 chars (model constraint)


@pytest.fixture
def mock_adapter():
    """A MagicMock adapter with a protocol_name attribute."""
    adapter = MagicMock()
    adapter.protocol_name = "rest"
    return adapter


@pytest.fixture
def client(mock_adapter):
    """ProximaDBClient whose adapter factory is patched to return a mock."""
    with patch.object(uc, "create_adapter", return_value=mock_adapter) as factory:
        c = ProximaDBClient(url="http://testserver:5678", protocol=Protocol.REST)
    c._test_factory = factory
    return c


def _batch_result(success=2, total=2, errors=None):
    return BatchResult(
        total=total,
        success=success,
        failed=total - success,
        errors=errors or [],
        metrics=OperationMetrics(total_processed=total, successful_count=success),
    )


def _records(n=2):
    return [
        VectorRecord(id=f"v{i}", vector=[0.1 * (i + 1), 0.2], metadata={"k": str(i)})
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Construction / adapter setup
# ---------------------------------------------------------------------------


def test_init_rest_protocol(mock_adapter):
    with patch.object(uc, "create_adapter", return_value=mock_adapter) as factory:
        c = ProximaDBClient(url="http://h:5678", protocol="rest")
    assert c.active_protocol == Protocol.REST
    assert c.adapter is mock_adapter
    # rest branch passes url kwarg
    _, kwargs = factory.call_args
    assert kwargs["url"] == "http://h:5678"


def test_init_grpc_protocol(mock_adapter):
    with patch.object(uc, "create_adapter", return_value=mock_adapter) as factory:
        c = ProximaDBClient(url="http://h:5678", protocol=Protocol.GRPC)
    assert c.active_protocol == Protocol.GRPC
    args, kwargs = factory.call_args
    assert args[0] == "grpc"
    # gRPC address is host:port from the unified-mode config
    assert kwargs["server_address"] == "h:5678"


def test_init_embedded_protocol(mock_adapter):
    with patch.object(uc, "create_adapter", return_value=mock_adapter) as factory:
        c = ProximaDBClient(url="http://h:5678", protocol="embedded", data_dir="/tmp/x")
    assert c.active_protocol == Protocol.EMBEDDED
    args, kwargs = factory.call_args
    assert args[0] == "embedded"
    assert kwargs["data_dir"] == "/tmp/x"


def test_init_embedded_default_data_dir(mock_adapter):
    with patch.object(uc, "create_adapter", return_value=mock_adapter) as factory:
        ProximaDBClient(url="http://h:5678", protocol="embedded")
    _, kwargs = factory.call_args
    assert kwargs["data_dir"] == "/tmp/proximadb/data"


def test_init_auto_selects_grpc(mock_adapter):
    with patch.object(uc, "create_adapter", return_value=mock_adapter) as factory:
        c = ProximaDBClient(url="http://h:5678", protocol=Protocol.AUTO)
    assert c.active_protocol == Protocol.GRPC
    assert factory.call_args[0][0] == "grpc"


def test_init_auto_falls_back_to_rest(mock_adapter):
    calls = {"n": 0}

    def factory(protocol, **kwargs):
        calls["n"] += 1
        if protocol == "grpc":
            raise ImportError("no grpc")
        return mock_adapter

    with patch.object(uc, "create_adapter", side_effect=factory):
        c = ProximaDBClient(url="http://h:5678", protocol=Protocol.AUTO)
    assert c.active_protocol == Protocol.REST
    assert calls["n"] == 2


def test_init_unknown_protocol_raises():
    # Bypass Protocol enum coercion by passing an object with a bogus .value
    from proximadb_sdk.config import ClientConfig

    class Fake:
        value = "carrier-pigeon"

    cfg = ClientConfig(url="http://h:5678")
    with pytest.raises(ValueError, match="Unknown protocol"):
        ProximaDBClient(config=cfg, protocol=Fake())


def test_init_with_explicit_config(mock_adapter):
    from proximadb_sdk.config import ClientConfig

    cfg = ClientConfig(url="http://h:5678")
    with patch.object(uc, "create_adapter", return_value=mock_adapter):
        c = ProximaDBClient(config=cfg, protocol="rest")
    assert c.config is cfg


# ---------------------------------------------------------------------------
# _get_grpc_url branches
# ---------------------------------------------------------------------------


def test_get_grpc_url_from_config(mock_adapter):
    with patch.object(uc, "create_adapter", return_value=mock_adapter):
        c = ProximaDBClient(url="http://h:5678", protocol="rest")

    # config is a (possibly frozen) pydantic model; swap in a lightweight
    # stand-in exposing the get_protocol_url hook the method looks for.
    class CfgStub:
        def get_protocol_url(self, proto):
            return "from-config:9999"

    c.config = CfgStub()
    assert c._get_grpc_url() == "from-config:9999"


def test_get_grpc_url_parsed_from_url(mock_adapter):
    with patch.object(uc, "create_adapter", return_value=mock_adapter):
        c = ProximaDBClient(url="http://h:5678", protocol="rest")
    # No get_protocol_url on config -> parse from self._url; grpc = port+1
    c.config = object()
    c._url = "http://myhost:7000"
    assert c._get_grpc_url() == "myhost:7001"


def test_get_grpc_url_default_no_url(mock_adapter):
    with patch.object(uc, "create_adapter", return_value=mock_adapter):
        c = ProximaDBClient(url="http://h:5678", protocol="rest")
    # Strip config method and url to hit the localhost:5679 default
    c.config = object()
    c._url = None
    assert c._get_grpc_url() == "localhost:5679"


# ---------------------------------------------------------------------------
# Health & collections
# ---------------------------------------------------------------------------


def test_health(client, mock_adapter):
    sentinel = object()
    mock_adapter.health.return_value = sentinel
    assert client.health() is sentinel
    mock_adapter.health.assert_called_once()


def test_create_collection_builds_config(client, mock_adapter):
    mock_adapter.create_collection.return_value = "coll"
    out = client.create_collection(VALID_NAME, dimension=128)
    assert out == "coll"
    name_arg, cfg_arg = mock_adapter.create_collection.call_args[0][:2]
    assert name_arg == VALID_NAME
    assert isinstance(cfg_arg, CollectionConfig)
    assert cfg_arg.dimension == 128


def test_create_collection_with_explicit_config(client, mock_adapter):
    cfg = CollectionConfig(name=VALID_NAME, dimension=64)
    client.create_collection(VALID_NAME, config=cfg)
    assert mock_adapter.create_collection.call_args[0][1] is cfg


def test_get_collection(client, mock_adapter):
    mock_adapter.get_collection.return_value = "c"
    assert client.get_collection("cid") == "c"
    mock_adapter.get_collection.assert_called_once_with("cid")


def test_list_collections(client, mock_adapter):
    mock_adapter.list_collections.return_value = ["a", "b"]
    assert client.list_collections() == ["a", "b"]


def test_delete_collection(client, mock_adapter):
    mock_adapter.delete_collection.return_value = True
    assert client.delete_collection("cid") is True


# ---------------------------------------------------------------------------
# Record operations
# ---------------------------------------------------------------------------


def test_insert_records(client, mock_adapter):
    br = _batch_result()
    mock_adapter.insert_records.return_value = br
    recs = _records()
    assert client.insert_records("cid", recs) is br
    mock_adapter.insert_records.assert_called_once_with("cid", recs)


def test_insert_records_empty_raises(client):
    with pytest.raises(ValueError, match="must be provided"):
        client.insert_records("cid", [])
    with pytest.raises(ValueError):
        client.insert_records("cid", None)


def test_upsert_records(client, mock_adapter):
    br = _batch_result()
    mock_adapter.upsert_records.return_value = br
    recs = _records()
    assert client.upsert_records("cid", recs) is br


def test_upsert_records_empty_raises(client):
    with pytest.raises(ValueError):
        client.upsert_records("cid", [])


# ---------------------------------------------------------------------------
# Vector compatibility aliases
# ---------------------------------------------------------------------------


def test_batch_to_vector_response_success():
    br = _batch_result(success=2, total=2)
    resp = ProximaDBClient._batch_to_vector_response(br, "INSERT")
    assert resp.operation == "INSERT"
    assert resp.error_message is None
    assert resp.metrics.successful_count == 2


def test_batch_to_vector_response_with_errors():
    br = _batch_result(success=1, total=2, errors=["boom", "bang"])
    resp = ProximaDBClient._batch_to_vector_response(br, "INSERT")
    assert resp.error_message == "boom; bang"


def test_insert_vectors_with_records(client, mock_adapter):
    mock_adapter.insert_records.return_value = _batch_result()
    recs = _records()
    resp = client.insert_vectors("cid", records=recs)
    assert resp.operation == "INSERT"
    mock_adapter.insert_records.assert_called_once_with("cid", recs)


def test_insert_vectors_from_raw_lists(client, mock_adapter):
    mock_adapter.insert_records.return_value = _batch_result()
    resp = client.insert_vectors(
        "cid",
        vectors=[[1.0, 2.0], [3.0, 4.0]],
        ids=["a", "b"],
        metadata=[{"m": "1"}, {"m": "2"}],
    )
    assert resp.operation == "INSERT"
    passed = mock_adapter.insert_records.call_args[0][1]
    assert len(passed) == 2
    assert all(isinstance(r, VectorRecord) for r in passed)
    assert passed[0].id == "a"
    assert passed[1].metadata == {"m": "2"}


def test_insert_vectors_from_numpy(client, mock_adapter):
    mock_adapter.insert_records.return_value = _batch_result()
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    client.insert_vectors("cid", vectors=arr)
    passed = mock_adapter.insert_records.call_args[0][1]
    assert len(passed) == 2
    assert passed[0].vector == [1.0, 2.0]


def test_insert_vectors_detects_vectorrecord_list(client, mock_adapter):
    mock_adapter.insert_records.return_value = _batch_result()
    recs = _records(2)
    # Pass already-built VectorRecords positionally as `vectors`
    client.insert_vectors("cid", vectors=recs)
    passed = mock_adapter.insert_records.call_args[0][1]
    assert passed == recs


def test_insert_vectors_no_input_raises(client):
    with pytest.raises(ValueError, match="Either 'records' or 'vectors'"):
        client.insert_vectors("cid")


def test_upsert_vectors(client, mock_adapter):
    mock_adapter.upsert_records.return_value = _batch_result()
    recs = _records()
    resp = client.upsert_vectors("cid", recs)
    assert resp.operation == "UPSERT"


def test_get_vectors(client, mock_adapter):
    mock_adapter.get_vectors.return_value = _records(1)
    out = client.get_vectors("cid", ["v0"])
    assert len(out) == 1
    mock_adapter.get_vectors.assert_called_once_with("cid", ["v0"], True)


def test_get_vector_found(client, mock_adapter):
    rec = _records(1)
    mock_adapter.get_vectors.return_value = rec
    assert client.get_vector("cid", "v0") is rec[0]


def test_get_vector_not_found(client, mock_adapter):
    mock_adapter.get_vectors.return_value = []
    assert client.get_vector("cid", "missing") is None


def test_delete_vectors(client, mock_adapter):
    mock_adapter.delete_vectors.return_value = "resp"
    assert client.delete_vectors("cid", ["a", "b"]) == "resp"
    mock_adapter.delete_vectors.assert_called_once_with("cid", ["a", "b"])


def test_delete_vector(client, mock_adapter):
    mock_adapter.delete_vectors.return_value = "resp"
    assert client.delete_vector("cid", "a") == "resp"
    mock_adapter.delete_vectors.assert_called_once_with("cid", ["a"])


def test_update_vector_metadata(client, mock_adapter):
    mock_adapter.update_vector_metadata.return_value = "ok"
    out = client.update_vector_metadata("cid", "v0", {"x": "1"})
    assert out == "ok"
    mock_adapter.update_vector_metadata.assert_called_once_with("cid", "v0", {"x": "1"})


def test_insert_vector_insert_path(client, mock_adapter):
    mock_adapter.insert_records.return_value = _batch_result(success=1, total=1)
    resp = client.insert_vector("cid", "v0", [1.0, 2.0], metadata={"m": "1"})
    assert resp.operation == "INSERT"
    passed = mock_adapter.insert_records.call_args[0][1]
    assert passed[0].id == "v0"


def test_insert_vector_upsert_path(client, mock_adapter):
    mock_adapter.upsert_records.return_value = _batch_result(success=1, total=1)
    resp = client.insert_vector("cid", "v0", [1.0, 2.0], upsert=True)
    assert resp.operation == "UPSERT"


def test_insert_vector_numpy(client, mock_adapter):
    mock_adapter.insert_records.return_value = _batch_result(success=1, total=1)
    client.insert_vector("cid", "v0", np.array([1.0, 2.0]))
    passed = mock_adapter.insert_records.call_args[0][1]
    assert passed[0].vector == [1.0, 2.0]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def _search_result():
    return SearchResult(id="v0", score=0.9)


def test_search_list_vector(client, mock_adapter):
    mock_adapter.search.return_value = [_search_result()]
    out = client.search("cid", [1.0, 2.0], top_k=5)
    assert len(out) == 1
    kwargs = mock_adapter.search.call_args[1]
    assert kwargs["query_vector"] == [1.0, 2.0]
    assert kwargs["top_k"] == 5


def test_search_numpy_vector(client, mock_adapter):
    mock_adapter.search.return_value = []
    client.search("cid", np.array([1.0, 2.0]))
    assert mock_adapter.search.call_args[1]["query_vector"] == [1.0, 2.0]


def test_search_invalid_top_k(client):
    with pytest.raises(ProximaDBError, match="top_k must be positive"):
        client.search("cid", [1.0, 2.0], top_k=0)


def test_search_single(client, mock_adapter):
    mock_adapter.search.return_value = ["r"]
    out = client.search_single("cid", [1.0, 2.0], top_k=3)
    assert out == ["r"]
    assert mock_adapter.search.call_args[1]["top_k"] == 3


def test_search_batch_list(client, mock_adapter):
    mock_adapter.batch_search.return_value = [["r1"], ["r2"]]
    out = client.search_batch("cid", [[1.0], [2.0]], top_k=2)
    assert out == [["r1"], ["r2"]]
    assert mock_adapter.batch_search.call_args[1]["query_vectors"] == [[1.0], [2.0]]


def test_search_batch_numpy(client, mock_adapter):
    mock_adapter.batch_search.return_value = []
    client.search_batch("cid", np.array([[1.0, 2.0]]))
    assert mock_adapter.batch_search.call_args[1]["query_vectors"] == [[1.0, 2.0]]


# ---------------------------------------------------------------------------
# Legacy compat
# ---------------------------------------------------------------------------


def test_legacy_insert(client, mock_adapter):
    mock_adapter.insert_records.return_value = _batch_result()
    resp = client.insert("cid", [[1.0, 2.0]], ids=["a"], metadata=[{"m": "1"}])
    assert resp.operation == "INSERT"


def test_legacy_upsert_list(client, mock_adapter):
    mock_adapter.upsert_records.return_value = _batch_result()
    resp = client.upsert("cid", [[1.0, 2.0], [3.0, 4.0]], ids=["a", "b"])
    assert resp.operation == "UPSERT"
    passed = mock_adapter.upsert_records.call_args[0][1]
    assert passed[0].id == "a"


def test_legacy_upsert_numpy(client, mock_adapter):
    mock_adapter.upsert_records.return_value = _batch_result()
    client.upsert("cid", np.array([[1.0, 2.0]]), ids=["a"], metadata=[{"m": "1"}])
    passed = mock_adapter.upsert_records.call_args[0][1]
    assert passed[0].vector == [1.0, 2.0]
    assert passed[0].metadata == {"m": "1"}


def test_legacy_delete(client, mock_adapter):
    mock_adapter.delete_vectors.return_value = "d"
    assert client.delete("cid", ["a"]) == "d"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _collection():
    return Collection(
        id="cid",
        config=CollectionConfig(name=VALID_NAME, dimension=128),
        stats=CollectionStats(),
    )


def test_get_collection_stats_found(client, mock_adapter):
    mock_adapter.get_collection.return_value = _collection()
    stats = client.get_collection_stats("cid")
    assert stats["id"] == "cid"
    assert stats["name"] == VALID_NAME
    assert stats["dimension"] == 128


def test_get_collection_stats_missing(client, mock_adapter):
    # TD-SDK-3 swap: get_collection restores v1's public contract — a missing
    # collection raises CollectionNotFoundError (previously returned None and
    # stats silently degraded to {}).
    from proximadb_sdk.exceptions import CollectionNotFoundError

    mock_adapter.get_collection.return_value = None
    with pytest.raises(CollectionNotFoundError):
        client.get_collection_stats("cid")


def test_get_performance_info_grpc(mock_adapter):
    with patch.object(uc, "create_adapter", return_value=mock_adapter):
        c = ProximaDBClient(url="http://h:5678", protocol=Protocol.GRPC)
    info = c.get_performance_info()
    assert info["protocol"] == "gRPC"


def test_get_performance_info_rest(client):
    info = client.get_performance_info()
    assert info["protocol"] == "REST"


def test_get_performance_info_embedded(mock_adapter):
    with patch.object(uc, "create_adapter", return_value=mock_adapter):
        c = ProximaDBClient(url="http://h:5678", protocol="embedded")
    info = c.get_performance_info()
    assert info["protocol"] == "Embedded"


# ---------------------------------------------------------------------------
# Query operations
# ---------------------------------------------------------------------------


def test_execute_query(client, mock_adapter):
    mock_adapter.execute_query.return_value = {"rows": []}
    out = client.execute_query("SELECT 1", language="uql", limit=10)
    assert out == {"rows": []}
    kwargs = mock_adapter.execute_query.call_args[1]
    assert kwargs["language"] == "uql"
    assert kwargs["limit"] == 10


def test_execute_query_unsupported(client, mock_adapter):
    # Remove execute_query so hasattr is False
    del mock_adapter.execute_query
    mock_adapter.protocol_name = "embedded"
    with pytest.raises(ProximaDBError, match="does not support execute_query"):
        client.execute_query("SELECT 1")


def test_execute_uql(client, mock_adapter):
    mock_adapter.execute_query.return_value = {"ok": True}
    client.execute_uql("Q", parameters=[1], collection="c", limit=5)
    assert mock_adapter.execute_query.call_args[1]["language"] == "uql"


def test_execute_aql(client, mock_adapter):
    mock_adapter.execute_query.return_value = {}
    client.execute_aql("Q")
    assert mock_adapter.execute_query.call_args[1]["language"] == "aql"


def test_execute_federated(client, mock_adapter):
    mock_adapter.execute_query.return_value = {}
    client.execute_federated("Q")
    assert mock_adapter.execute_query.call_args[1]["language"] == "federated"


def test_explain_query(client, mock_adapter):
    mock_adapter.explain_query.return_value = {"plan": "x"}
    out = client.explain_query("Q", language="aql", collection="c")
    assert out == {"plan": "x"}
    assert mock_adapter.explain_query.call_args[1]["language"] == "aql"


def test_explain_query_unsupported(client, mock_adapter):
    del mock_adapter.explain_query
    mock_adapter.protocol_name = "grpc"
    with pytest.raises(ProximaDBError, match="does not support explain_query"):
        client.explain_query("Q")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_close(client, mock_adapter):
    client.close()
    mock_adapter.close.assert_called_once()
    assert client._adapter is None
    # idempotent
    client.close()


def test_context_manager(mock_adapter):
    with patch.object(uc, "create_adapter", return_value=mock_adapter):
        with ProximaDBClient(url="http://h:5678", protocol="rest") as c:
            assert c is not None
    mock_adapter.close.assert_called_once()


def test_del_swallows_exceptions(mock_adapter):
    with patch.object(uc, "create_adapter", return_value=mock_adapter):
        c = ProximaDBClient(url="http://h:5678", protocol="rest")
    mock_adapter.close.side_effect = RuntimeError("boom")
    # __del__ must not raise
    c.__del__()


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def test_connect(mock_adapter):
    with patch.object(uc, "create_adapter", return_value=mock_adapter):
        c = connect(url="http://h:5678", protocol="rest")
    assert isinstance(c, ProximaDBClient)


def test_connect_grpc(mock_adapter):
    with patch.object(uc, "create_adapter", return_value=mock_adapter):
        c = connect_grpc(url="http://h:5678")
    assert c.active_protocol == Protocol.GRPC


def test_connect_rest(mock_adapter):
    with patch.object(uc, "create_adapter", return_value=mock_adapter):
        c = connect_rest(url="http://h:5678")
    assert c.active_protocol == Protocol.REST


def test_connect_embedded(mock_adapter):
    with patch.object(uc, "create_adapter", return_value=mock_adapter):
        c = connect_embedded(data_dir="/tmp/y", url="http://h:5678")
    assert c.active_protocol == Protocol.EMBEDDED
