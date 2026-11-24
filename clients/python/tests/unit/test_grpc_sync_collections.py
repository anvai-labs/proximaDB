"""
Tests for gRPC sync client collection operations

These tests verify collection CRUD operations using mocked gRPC stubs.
"""

import types
from unittest.mock import Mock

import pytest

try:
    import grpc

    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False


class FakeCollectionStub:
    """Fake gRPC collection stub for testing"""

    def __init__(self, channel):
        pass

    def CreateCollection(self, req, timeout=None):
        return types.SimpleNamespace(id="col-xyz")

    def GetCollection(self, req, timeout=None):
        return types.SimpleNamespace(id="col-xyz", name="test_docs")

    def ListCollections(self, req, timeout=None):
        return types.SimpleNamespace(
            collections=[types.SimpleNamespace(id="c1"), types.SimpleNamespace(id="c2")]
        )

    def DeleteCollection(self, req, timeout=None):
        return types.SimpleNamespace(success=True)


class FakePool:
    """Fake connection pool for testing"""

    def __init__(self, *a, **k):
        self.pool_size = k.get("pool_size", 5)
        self.max_message_size = k.get("max_message_size", 64 * 1024 * 1024)
        self.endpoint = k.get("endpoint", "localhost:5679")

    def get_metrics(self):
        return types.SimpleNamespace(
            total_connections=self.pool_size,
            active_connections=0,
            idle_connections=self.pool_size,
            health_status=types.SimpleNamespace(value="healthy"),
        )

    def close(self):
        pass


class FakeCtx:
    """Fake channel context for testing"""

    def __init__(self, pool):
        pass

    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


@pytest.mark.skipif(not GRPC_AVAILABLE, reason="gRPC not available")
def test_grpc_sync_collection_ops(monkeypatch):
    """Test gRPC sync collection operations"""

    # Create fake request classes
    class FakeCollectionConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class FakeListCollectionsRequest:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class FakeGetCollectionRequest:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class FakeDeleteCollectionRequest:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    # Patch grpc.insecure_channel first
    monkeypatch.setattr("grpc.insecure_channel", lambda *a, **k: Mock())

    # Import the module after patching
    import proximadb_sdk.protocols.grpc_sync as grpc_mod

    # Patch collection stub and connection pool
    monkeypatch.setattr(
        grpc_mod,
        "v1_collection_pb2_grpc",
        types.SimpleNamespace(CollectionServiceStub=FakeCollectionStub),
        raising=False,
    )

    # Patch message modules to simple namespaces to avoid import dependence
    monkeypatch.setattr(
        grpc_mod,
        "v1_collection_types_pb2",
        types.SimpleNamespace(
            CollectionConfig=FakeCollectionConfig,
            ListCollectionsRequest=FakeListCollectionsRequest,
            GetCollectionRequest=FakeGetCollectionRequest,
            DeleteCollectionRequest=FakeDeleteCollectionRequest,
        ),
        raising=False,
    )

    # Patch connection pool classes
    monkeypatch.setattr(
        "proximadb_sdk.protocols.connection_pools.GrpcConnectionPool", FakePool
    )
    monkeypatch.setattr(
        "proximadb_sdk.protocols.connection_pools.GrpcChannelContext", FakeCtx
    )

    from proximadb_sdk.models import DistanceMetricType, StorageEngineType
    from proximadb_sdk.protocols.grpc_sync import ProximaDBSyncGrpcClient

    client = ProximaDBSyncGrpcClient("localhost:5679")

    # Test collection operations
    client.create_collection_v1(
        name="test_docs",  # Must be >8 chars
        dimension=128,
        distance_metric=DistanceMetricType.COSINE,
        storage_engine=StorageEngineType.NOVA,
    )
    client.get_collection_v1("test_docs")
    client.list_collections_v1()
    client.delete_collection_v1("test_docs")

    client.close()
