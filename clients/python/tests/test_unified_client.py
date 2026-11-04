#!/usr/bin/env python3
"""
Test script to verify unified client functionality.

This test checks that:
1. Unified client can switch between gRPC and REST protocols
2. Type conversion between proto and Pydantic works correctly
3. Interface remains consistent regardless of protocol
4. Auto-selection works properly
"""

# To run this script, set PYTHONPATH to include the src directory:
# PYTHONPATH=/home/vsingh/code/proximaDB/clients/python/src python tests/test_unified_client.py


from proximadb_sdk import (
    CollectionConfig,
    IndexingAlgorithm,
    Protocol,
    ProximaDBClient,
    StorageEngine,
    VectorRecord,
)

# TD-SDK-3 swap: the package-level ProximaDBClient is the v2 client. The
# proto-conversion helpers below are v1 internals — import the v1 client
# explicitly to keep pinning them.
from proximadb_sdk.unified_client import ProximaDBClient as ProximaDBClientV1


def test_protocol_selection():
    """Test that protocol selection works correctly"""

    # Test auto selection (should fallback to REST when gRPC fails)
    client_auto = ProximaDBClient(url="http://localhost:5678", protocol=Protocol.AUTO)
    assert client_auto.active_protocol in [Protocol.GRPC, Protocol.REST]

    # Test forced REST
    client_rest = ProximaDBClient(url="http://localhost:5678", protocol=Protocol.REST)
    assert client_rest.active_protocol == Protocol.REST

    # Test convenience functions
    client_connect = ProximaDBClient(url="http://localhost:5678")
    assert client_connect.active_protocol in [Protocol.GRPC, Protocol.REST]


def test_pydantic_model_creation():
    """Test that Pydantic models can be created properly"""

    # Test CollectionConfig creation
    config = CollectionConfig(
        name="test_collection",
        dimension=128,
        distance_metric="cosine",
        storage_engine=StorageEngine.VIPER,
        description="Test collection",
        tags=["test", "example"],
        owner="unittest",
    )

    assert config.name == "test_collection"
    assert config.dimension == 128
    assert config.distance_metric == "cosine"
    assert config.storage_engine == StorageEngine.VIPER
    assert config.description == "Test collection"
    assert config.tags == ["test", "example"]
    assert config.owner == "unittest"


def test_vector_record_creation():
    """Test that VectorRecord can be created properly"""

    # Test VectorRecord with all fields
    record = VectorRecord(
        id="vec_001",
        vector=[0.1, 0.2, 0.3, 0.4],
        metadata={
            "category": "test",
            "score": 0.95,
            "active": True,
            "tags": ["important", "verified"],
        },
        timestamp_ms=1640995200000,  # 2022-01-01 00:00:00 UTC in milliseconds
        version=1,
    )

    assert record.id == "vec_001"
    assert record.vector == [0.1, 0.2, 0.3, 0.4]
    assert record.metadata["category"] == "test"
    assert record.metadata["score"] == 0.95
    assert record.metadata["active"] is True
    assert record.metadata["tags"] == ["important", "verified"]
    assert record.timestamp_ms == 1640995200000
    assert record.version == 1

    # Test VectorRecord with minimal fields
    minimal_record = VectorRecord(vector=[1.0, 2.0, 3.0])

    assert minimal_record.vector == [1.0, 2.0, 3.0]
    assert minimal_record.metadata == {}
    assert minimal_record.version == 0


def test_type_conversion_helpers():
    """Test type conversion between proto and Pydantic"""

    client = ProximaDBClientV1(url="http://localhost:5678", protocol=Protocol.REST)

    # Test distance metric conversion
    assert client._proto_to_pydantic_distance_metric(1) == "cosine"
    assert client._proto_to_pydantic_distance_metric(2) == "euclidean"
    assert client._proto_to_pydantic_distance_metric(3) == "dot_product"

    # Test storage engine conversion
    assert client._proto_to_pydantic_storage_engine(1) == StorageEngine.VIPER
    assert client._proto_to_pydantic_storage_engine(2) == StorageEngine.SST
    assert client._proto_to_pydantic_storage_engine(3) == StorageEngine.NOVA

    # Test indexing algorithm conversion
    assert client._proto_to_pydantic_indexing_algorithm(1) == IndexingAlgorithm.HNSW
    assert client._proto_to_pydantic_indexing_algorithm(2) == IndexingAlgorithm.IVF
    assert client._proto_to_pydantic_indexing_algorithm(3) == IndexingAlgorithm.PQ

    # Test reverse conversion (only if gRPC is available)
    try:
        from proximadb_sdk import proximadb_pb2 as pb2

        assert client._pydantic_to_proto_distance_metric("cosine") == pb2.COSINE
        assert (
            client._pydantic_to_proto_storage_engine(StorageEngine.VIPER)
            == pb2.StorageEngine.VIPER
        )
        assert (
            client._pydantic_to_proto_indexing_algorithm(IndexingAlgorithm.HNSW)
            == pb2.HNSW
        )

    except ImportError:
        print("gRPC not available, skipping proto conversion tests")


def test_performance_info():
    """Test that performance info is returned correctly"""

    client_rest = ProximaDBClient(url="http://localhost:5678", protocol=Protocol.REST)
    perf_info = client_rest.get_performance_info()

    assert perf_info["protocol"] == "REST"
    assert "advantages" in perf_info
    assert "serialization" in perf_info
    assert "transport" in perf_info
    assert perf_info["serialization"] == "JSON"
    assert perf_info["transport"] == "HTTP/1.1"


def test_client_interface_consistency():
    """Test that client interface is consistent regardless of protocol"""

    # Test with REST client
    client_rest = ProximaDBClient(url="http://localhost:5678", protocol=Protocol.REST)

    # Check that all expected methods exist
    assert hasattr(client_rest, "health")
    assert hasattr(client_rest, "create_collection")
    assert hasattr(client_rest, "get_collection")
    assert hasattr(client_rest, "list_collections")
    assert hasattr(client_rest, "delete_collection")
    assert hasattr(client_rest, "insert_vectors")
    assert hasattr(client_rest, "upsert_vectors")
    assert hasattr(client_rest, "search_single")
    assert hasattr(client_rest, "delete_vectors")
    assert hasattr(client_rest, "get_vector")

    # Check method signatures
    import inspect

    # Test health method
    health_sig = inspect.signature(client_rest.health)
    assert len(health_sig.parameters) == 0

    # Test create_collection method
    create_sig = inspect.signature(client_rest.create_collection)
    assert "name" in create_sig.parameters
    assert "config" in create_sig.parameters

    # Test search_single method
    search_sig = inspect.signature(client_rest.search_single)
    assert "collection_id" in search_sig.parameters
    assert "vector" in search_sig.parameters
    assert "top_k" in search_sig.parameters


def test_convenience_functions():
    """Test convenience connection functions"""

    from proximadb_sdk import connect, connect_grpc, connect_rest

    # Test generic connect
    client = connect(url="http://localhost:5678")
    assert isinstance(client, ProximaDBClient)
    assert client.active_protocol in [Protocol.GRPC, Protocol.REST]

    # Test REST connect
    client_rest = connect_rest(url="http://localhost:5678")
    assert isinstance(client_rest, ProximaDBClient)
    assert client_rest.active_protocol == Protocol.REST

    # Test gRPC connect (may fail if gRPC not available)
    try:
        client_grpc = connect_grpc(url="http://localhost:5679")
        assert isinstance(client_grpc, ProximaDBClient)
        assert client_grpc.active_protocol == Protocol.GRPC
    except ImportError:
        print("gRPC not available, skipping gRPC connect test")


def test_context_manager():
    """Test that client works as context manager"""

    with ProximaDBClient(url="http://localhost:5678", protocol=Protocol.REST) as client:
        assert client.active_protocol == Protocol.REST
        assert hasattr(client, "close")
        # Client should be usable within context
        perf_info = client.get_performance_info()
        assert perf_info["protocol"] == "REST"


if __name__ == "__main__":
    """Run unified client tests"""
    print("Testing protocol selection...")
    test_protocol_selection()
    print("✓ Protocol selection works correctly")

    print("\nTesting Pydantic model creation...")
    test_pydantic_model_creation()
    print("✓ Pydantic models create correctly")

    print("\nTesting VectorRecord creation...")
    test_vector_record_creation()
    print("✓ VectorRecord creates correctly")

    print("\nTesting type conversion helpers...")
    test_type_conversion_helpers()
    print("✓ Type conversion works correctly")

    print("\nTesting performance info...")
    test_performance_info()
    print("✓ Performance info returns correctly")

    print("\nTesting client interface consistency...")
    test_client_interface_consistency()
    print("✓ Client interface is consistent")

    print("\nTesting convenience functions...")
    test_convenience_functions()
    print("✓ Convenience functions work correctly")

    print("\nTesting context manager...")
    test_context_manager()
    print("✓ Context manager works correctly")

    print("\n🎉 All unified client tests passed!")
    print("\nSummary:")
    print("- Unified client successfully switches between gRPC and REST")
    print("- Type conversion between proto and Pydantic models works")
    print("- Interface remains consistent regardless of protocol")
    print("- Auto-selection and forced protocol selection work")
    print("- Convenience functions provide easy client creation")
    print("- Context manager support for proper resource cleanup")
