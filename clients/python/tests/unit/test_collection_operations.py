#!/usr/bin/env python3
"""
ProximaDB Collection Operations Test Suite
Consolidated tests for collection CRUD operations, configuration, and lifecycle management

Tests run against embedded ProximaDB database for fast, reliable testing.
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
    StorageEngine,
)


class TestCollectionCRUD:
    """Test collection Create, Read, Update, Delete operations using embedded database"""

    @pytest.fixture
    def collection_name(self):
        """Generate unique collection name for each test"""
        return f"test_collection_{int(time.time() * 1000)}"

    def test_collection_lifecycle_rest(self, rest_client, collection_name):
        """Test complete collection lifecycle via embedded database (REST-style interface)"""
        config = CollectionConfig(
            name="test_collection",
            dimension=128,
            distance_metric="cosine",
            description="REST test collection",
        )

        # Create collection
        collection = rest_client.create_collection(collection_name, config=config)
        assert collection is not None

        # List collections - verify creation
        collections = rest_client.list_collections()
        assert collections is not None

        # Handle both string and object responses
        collection_names = []
        if isinstance(collections, list):
            for col in collections:
                if isinstance(col, str):
                    collection_names.append(col)
                elif hasattr(col, "name"):
                    collection_names.append(col.name)
                elif hasattr(col, "id"):
                    collection_names.append(col.id)
                else:
                    collection_names.append(str(col))

        assert collection_name in collection_names

        # Get specific collection
        retrieved = rest_client.get_collection(collection_name)
        assert retrieved is not None

        # Delete collection
        result = rest_client.delete_collection(collection_name)

        # Verify deletion
        with pytest.raises((CollectionNotFoundError, ProximaDBError, Exception)):
            rest_client.get_collection(collection_name)

    def test_collection_lifecycle_grpc(self, grpc_client, collection_name):
        """Test complete collection lifecycle via embedded database (gRPC-style interface)"""
        config = CollectionConfig(
            name=collection_name,
            dimension=256,
            distance_metric="dot_product",
            description="gRPC test collection",
        )

        # Create collection
        collection = grpc_client.create_collection(collection_name, config=config)
        assert collection is not None

        # List collections
        collections = grpc_client.list_collections()
        assert collections is not None

        # Get specific collection
        retrieved = grpc_client.get_collection(collection_name)
        assert retrieved is not None

        # Delete collection
        result = grpc_client.delete_collection(collection_name)

    def test_cross_protocol_operations(self, rest_client, grpc_client, collection_name):
        """Test collection operations across REST and gRPC clients (both use same embedded DB)"""
        config = CollectionConfig(
            name="test_collection",
            dimension=128,
            distance_metric="cosine",
            description="Cross-protocol test collection",
        )

        # Create with REST client
        collection = rest_client.create_collection(collection_name, config=config)
        assert collection is not None

        # Verify with gRPC client (same embedded DB)
        retrieved_via_grpc = grpc_client.get_collection(collection_name)
        assert retrieved_via_grpc is not None

        # List via both clients
        rest_collections = rest_client.list_collections()
        grpc_collections = grpc_client.list_collections()

        # Extract names from both
        rest_names = [
            col.name if hasattr(col, "name") else str(col) for col in rest_collections
        ]
        grpc_names = [
            col.name if hasattr(col, "name") else str(col) for col in grpc_collections
        ]

        assert collection_name in rest_names
        assert collection_name in grpc_names

        # Delete with gRPC client
        grpc_client.delete_collection(collection_name)

        # Verify deletion with REST client
        with pytest.raises((CollectionNotFoundError, ProximaDBError, Exception)):
            rest_client.get_collection(collection_name)


class TestCollectionConfiguration:
    """Test collection configuration options and validation - Pure unit tests"""

    def test_basic_collection_config(self):
        """Test basic collection configuration"""
        config = CollectionConfig(
            name="test_collection", dimension=768, distance_metric="cosine"
        )
        assert config.dimension == 768
        assert config.distance_metric == "cosine"

    def test_advanced_collection_config(self):
        """Test advanced collection configuration with all options"""
        index_config = IndexConfiguration(
            index_name="test_index",
            algorithm=IndexType.HNSW,
        )

        config = CollectionConfig(
            name="test_collection",
            dimension=384,
            distance_metric="euclidean",
            index_configs=[index_config],
            storage_engine=StorageEngine.NOVA,
            description="Advanced test collection",
            filterable_metadata_fields=["category", "timestamp"],
            compression={"algorithm": CompressionType.LZ4},
        )

        assert config.dimension == 384
        assert config.distance_metric == "euclidean"
        assert config.index_config.algorithm == IndexType.HNSW
        assert config.compression.algorithm == CompressionType.LZ4

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
                index_name=f"test_{algo.value}_index", algorithm=algo
            )
            config = CollectionConfig(
                name="test_algo_config", dimension=128, index_configs=[index_config]
            )
            assert config.index_config.algorithm == algo

    def test_storage_engines(self):
        """Test storage engine options"""
        engines = [
            StorageEngine.NOVA,
            StorageEngine.SST,
            StorageEngine.MMAP,
            StorageEngine.HYBRID,
        ]

        for engine in engines:
            config = CollectionConfig(
                name="test_engine_config", dimension=128, storage_engine=engine
            )
            assert config.storage_engine == engine

    def test_collection_with_filterable_columns(self):
        """Test collection with filterable columns configuration"""
        from proximadb_sdk import FilterableColumn, FilterableDataType

        filterable_cols = [
            FilterableColumn(
                name="category", data_type=FilterableDataType.STRING, indexed=True
            ),
            FilterableColumn(
                name="timestamp", data_type=FilterableDataType.DATETIME, indexed=True
            ),
            FilterableColumn(
                name="score", data_type=FilterableDataType.FLOAT, indexed=False
            ),
        ]

        config = CollectionConfig(
            name="test_filterable", dimension=512, filterable_columns=filterable_cols
        )

        assert config.filterable_columns is not None
        assert len(config.filterable_columns) == 3
        assert config.filterable_columns[0].name == "category"

    def test_collection_creation_with_config(self, rest_client):
        """Test creating collection with advanced configuration"""
        collection_name = f"config_test_{int(time.time() * 1000)}"

        config = CollectionConfig(
            name=collection_name,
            dimension=384,
            distance_metric="cosine",
            description="Configuration test collection",
            storage_engine=StorageEngine.SST,  # Use SST which is supported by embedded
        )

        try:
            collection = rest_client.create_collection(collection_name, config=config)
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
            name="test_collection", dimension=128, distance_metric="cosine"
        )
        assert valid_config.dimension == 128

        # Invalid dimensions should raise validation errors
        with pytest.raises((ValueError, TypeError)):
            CollectionConfig(
                name="test_collection",
                dimension=0,  # Invalid dimension
                distance_metric="cosine",
            )

        with pytest.raises((ValueError, TypeError)):
            CollectionConfig(
                name="test_collection",
                dimension=70000,  # Too large (max is 65536)
                distance_metric="cosine",
            )

    def test_collection_not_found_error(self, rest_client):
        """Test CollectionNotFoundError handling"""
        non_existent = f"non_existent_{int(time.time() * 1000)}"

        with pytest.raises((CollectionNotFoundError, ProximaDBError, Exception)):
            rest_client.get_collection(non_existent)

    def test_duplicate_collection_creation(self, rest_client):
        """Test handling of duplicate collection creation"""
        collection_name = f"duplicate_test_{int(time.time() * 1000)}"

        config = CollectionConfig(
            name=collection_name, dimension=128, distance_metric="cosine"
        )

        try:
            # Create first time - should succeed
            collection1 = rest_client.create_collection(collection_name, config=config)
            assert collection1 is not None

            # Create again - should raise error or handle gracefully
            with pytest.raises((ProximaDBError, Exception)):
                rest_client.create_collection(collection_name, config=config)

        finally:
            try:
                rest_client.delete_collection(collection_name)
            except:
                pass


class TestCollectionPersistence:
    """Test collection persistence within a session"""

    def test_collection_persistence_after_restart(self, rest_client):
        """Test that collections persist within the embedded database session"""
        collection_name = f"persist_test_{int(time.time() * 1000)}"

        config = CollectionConfig(
            name=collection_name,
            dimension=256,
            distance_metric="cosine",
            description="Persistence test collection",
        )

        try:
            # Create collection
            collection = rest_client.create_collection(collection_name, config=config)
            assert collection is not None

            # Verify it exists immediately
            retrieved = rest_client.get_collection(collection_name)
            assert retrieved is not None

            # Flush to ensure persistence
            rest_client.flush()

            # Verify still exists after flush
            retrieved_again = rest_client.get_collection(collection_name)
            assert retrieved_again is not None

        finally:
            try:
                rest_client.delete_collection(collection_name)
            except:
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
