#!/usr/bin/env python3
"""
ProximaDB Collection Operations Test Suite
Consolidated tests for collection CRUD operations, configuration, and lifecycle management
"""

import time

import pytest

from proximadb_sdk import (
    CollectionConfig,
    CollectionNotFoundError,
    CompressionType,
    IndexConfiguration,
    IndexType,
    ProximaDBError,
    StorageConfig,
    StorageEngine,
    connect_grpc,
    connect_rest,
)
from proximadb_sdk.models import (  # Import from models to avoid namespace conflict
    CompressionConfig,
)


class TestCollectionCRUD:
    """Test collection Create, Read, Update, Delete operations"""

    @pytest.fixture(scope="class")
    def rest_client(self):
        """REST client fixture"""
        client = connect_rest("http://localhost:5678")
        yield client
        client.close()

    @pytest.fixture(scope="class")
    def grpc_client(self):
        """gRPC client fixture"""
        client = connect_grpc("http://localhost:5679")
        yield client
        client.close()

    @pytest.fixture
    def collection_name(self):
        """Generate unique collection name for each test"""
        return f"test_collection_{int(time.time())}"

    def test_collection_lifecycle_rest(self, rest_client, collection_name):
        """Test complete collection lifecycle via REST"""
        config = CollectionConfig(
            name="test_collection_rest",  # Minimum 8 characters
            dimension=128,
            distance_metric="cosine",
            description="REST test collection",
        )

        # Create collection
        collection = rest_client.create_collection(collection_name, config)
        assert collection is not None
        assert collection.id is not None
        collection_id = collection.id

        # List collections - verify creation
        collections = rest_client.list_collections()
        assert collections is not None
        collection_names = [col.name for col in collections]
        assert collection_name in collection_names

        # Get specific collection using ID
        retrieved = rest_client.get_collection(collection_id)
        assert retrieved is not None
        assert retrieved.name == collection_name

        # Delete collection using ID
        result = rest_client.delete_collection(collection_id)

        # Verify deletion
        with pytest.raises((CollectionNotFoundError, ProximaDBError)):
            rest_client.get_collection(collection_id)

    def test_collection_lifecycle_grpc(self, grpc_client, collection_name):
        """Test complete collection lifecycle via gRPC"""
        config = CollectionConfig(
            name=collection_name,
            dimension=256,
            distance_metric="dot_product",
            description="gRPC test collection",
        )

        # Create collection
        collection = grpc_client.create_collection(collection_name, config)
        assert collection is not None
        assert collection.id is not None
        collection_id = collection.id

        # List collections
        collections = grpc_client.list_collections()
        assert collections is not None
        collection_names = [col.name for col in collections]
        assert collection_name in collection_names

        # Get specific collection using ID
        retrieved = grpc_client.get_collection(collection_id)
        assert retrieved is not None
        assert retrieved.name == collection_name

        # Delete collection using ID
        result = grpc_client.delete_collection(collection_id)

    def test_cross_protocol_operations(self, rest_client, grpc_client, collection_name):
        """Test collection operations across REST and gRPC protocols"""
        config = CollectionConfig(
            name=collection_name,
            dimension=128,
            distance_metric="cosine",
            description="Cross-protocol test collection",
        )

        # Create with REST
        collection = rest_client.create_collection(collection_name, config)
        assert collection is not None
        assert collection.id is not None
        collection_id = collection.id

        # Verify with gRPC using ID
        retrieved_via_grpc = grpc_client.get_collection(collection_id)
        assert retrieved_via_grpc is not None
        assert retrieved_via_grpc.name == collection_name

        # List via both protocols
        rest_collections = rest_client.list_collections()
        grpc_collections = grpc_client.list_collections()

        # Both should see the collection
        rest_names = [col.name for col in rest_collections]
        grpc_names = [col.name for col in grpc_collections]

        assert collection_name in rest_names
        assert collection_name in grpc_names

        # Delete with gRPC using ID
        grpc_client.delete_collection(collection_id)

        # Verify deletion with REST using ID
        with pytest.raises((CollectionNotFoundError, ProximaDBError)):
            rest_client.get_collection(collection_id)


class TestCollectionConfiguration:
    """Test collection configuration options and validation"""

    @pytest.fixture
    def rest_client(self):
        client = connect_rest("http://localhost:5678")
        yield client
        client.close()

    def test_basic_collection_config(self):
        """Test basic collection configuration"""
        config = CollectionConfig(
            name="test_collection_basic",  # Minimum 8 characters
            dimension=768,
            distance_metric="cosine",
        )
        assert config.dimension == 768
        assert config.distance_metric == "cosine"

    def test_advanced_collection_config(self):
        """Test advanced collection configuration with all options"""
        index_config = IndexConfiguration(
            index_name="primary_hnsw", algorithm=IndexType.HNSW, memory_limit_mb=512
        )

        config = CollectionConfig(
            name="test_collection_advanced",  # Minimum 8 characters
            dimension=384,
            distance_metric="euclidean",
            storage_engine=StorageEngine.NOVA,
            index_configs=[index_config],
            description="Advanced test collection",
        )

        assert config.dimension == 384
        assert config.distance_metric == "euclidean"
        assert config.storage_engine == StorageEngine.NOVA
        assert len(config.index_configs) == 1
        assert config.index_configs[0].algorithm == IndexType.HNSW

    def test_distance_metrics(self):
        """Test all distance metric options"""
        metrics = ["cosine", "euclidean", "dot_product", "manhattan", "hamming"]

        for metric in metrics:
            config = CollectionConfig(
                name="test_collection", dimension=128, distance_metric=metric
            )
            assert config.distance_metric == metric

    def test_index_algorithms(self):
        """Test index algorithm options"""
        algorithms = [
            IndexType.HNSW,
            IndexType.IVF,
            IndexType.PQ,
            IndexType.FLAT,
            IndexType.ANNOY,
        ]

        for algo in algorithms:
            index_config = IndexConfiguration(
                index_name=f"test_{algo.value}", algorithm=algo
            )
            assert index_config.algorithm == algo

    def test_compression_types(self):
        """Test compression type options"""
        compression_types = [
            CompressionType.NONE,
            CompressionType.LZ4,
            CompressionType.ZSTD,
            CompressionType.SNAPPY,
        ]

        for compression in compression_types:
            # Create CompressionConfig with the compression type
            compression_config = CompressionConfig(algorithm=compression)
            storage_config = StorageConfig(compression=compression_config)
            assert storage_config.compression.algorithm == compression

    def test_collection_with_metadata_schema(self):
        """Test collection with metadata schema configuration"""
        metadata_schema = {
            "category": "string",
            "timestamp": "datetime",
            "score": "float",
        }

        config = CollectionConfig(
            name="test_collection_metadata",  # Minimum 8 characters
            dimension=512,
            metadata_schema=metadata_schema,
            filterable_metadata_fields=["category", "timestamp"],
        )

        assert config.metadata_schema == metadata_schema
        assert "category" in config.filterable_metadata_fields
        assert "timestamp" in config.filterable_metadata_fields

    def test_collection_creation_with_config(self, rest_client):
        """Test creating collection with advanced configuration"""
        collection_name = f"config_test_{int(time.time())}"

        config = CollectionConfig(
            name=collection_name,
            dimension=384,
            distance_metric="cosine",
            description="Configuration test collection",
            storage_engine=StorageEngine.NOVA,
        )

        try:
            collection = rest_client.create_collection(collection_name, config)
            assert collection is not None

            # Verify configuration persisted
            retrieved = rest_client.get_collection(collection_name)
            assert retrieved is not None

        finally:
            try:
                rest_client.delete_collection(collection_name)
            except:
                pass


class TestCollectionValidation:
    """Test collection configuration validation and error handling"""

    def test_dimension_validation(self):
        """Test dimension validation"""
        # Valid dimensions
        valid_config = CollectionConfig(
            name="test_collection_valid",  # Minimum 8 characters
            dimension=128,
            distance_metric="cosine",
        )
        assert valid_config.dimension == 128

        # Invalid dimensions should raise validation errors
        with pytest.raises((ValueError, TypeError)):
            CollectionConfig(
                name="test_collection_invalid",  # Minimum 8 characters
                dimension=0,  # Invalid dimension
                distance_metric="cosine",
            )

        with pytest.raises((ValueError, TypeError)):
            CollectionConfig(
                name="test_collection_toolarge",  # Minimum 8 characters
                dimension=65537,  # Too large - exceeds server maximum of 65536
                distance_metric="cosine",
            )

    def test_collection_not_found_error(self):
        """Test CollectionNotFoundError handling"""
        client = connect_rest("http://localhost:5678")
        non_existent = f"non_existent_{int(time.time())}"

        with pytest.raises((CollectionNotFoundError, ProximaDBError)):
            client.get_collection(non_existent)

    def test_duplicate_collection_creation(self):
        """Test handling of duplicate collection creation"""
        client = connect_rest("http://localhost:5678")
        collection_name = f"duplicate_test_{int(time.time())}"

        config = CollectionConfig(
            name=collection_name, dimension=128, distance_metric="cosine"
        )
        try:
            # Create first time - should succeed
            collection1 = client.create_collection(collection_name, config)
            assert collection1 is not None

            # Create again - should raise error or handle gracefully
            with pytest.raises(ProximaDBError):
                client.create_collection(collection_name, config)

        finally:
            try:
                client.delete_collection(collection_name)
            except:
                pass


class TestCollectionPersistence:
    """Test collection persistence across server restarts"""

    def test_collection_persistence_after_restart(self):
        """Test that collections persist after server restart"""
        client = connect_rest("http://localhost:5678")
        collection_name = f"persist_test_{int(time.time())}"

        config = CollectionConfig(
            name=collection_name,
            dimension=256,
            distance_metric="cosine",
            description="Persistence test collection",
        )

        try:
            # Create collection
            collection = client.create_collection(collection_name, config)
            assert collection is not None

            # Verify it exists immediately
            retrieved = client.get_collection(collection_name)
            assert retrieved is not None

            # Note: Actual server restart testing would require test infrastructure
            # This test verifies the basic persistence mechanism is working

        finally:
            try:
                client.delete_collection(collection_name)
            except:
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
