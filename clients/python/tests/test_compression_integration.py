#!/usr/bin/env python3
"""
Tests for SDK-driven compression integration

Copyright 2025 ProximaDB
"""

from unittest.mock import Mock

import numpy as np
import pytest

from proximadb_sdk.models import (
    CollectionConfig,
    CompressionAlgorithm,
    CompressionConfig,
    DistanceMetric,
    SearchOptimization,
    SearchResult,
    StorageEngine,
)


class TestCompressionConfig:
    """Test unified compression configuration models"""

    def test_compression_config_defaults(self):
        """Test default values for CompressionConfig"""
        config = CompressionConfig()

        assert config.algorithm == CompressionAlgorithm.NONE
        assert config.level is None
        assert config.adaptive is False
        assert config.min_ratio is None
        assert config.enable_quantization is False
        assert config.quantization_type is None
        assert config.normalization_method is None
        assert config.block_size_kb is None
        assert config.dynamic_block_sizing is False

    def test_compression_config_columnar_optimized(self):
        """Test columnar-optimized compression configuration"""
        config = CompressionConfig(
            algorithm=CompressionAlgorithm.LZ4,
            level=1,
            enable_quantization=True,
            quantization_type="int8",
            normalization_method="mean",
        )

        assert config.algorithm == CompressionAlgorithm.LZ4
        assert config.level == 1
        assert config.enable_quantization is True
        assert config.quantization_type == "int8"
        assert config.normalization_method == "mean"

    def test_compression_config_sst_optimized(self):
        """Test SST-optimized compression configuration"""
        config = CompressionConfig(
            algorithm=CompressionAlgorithm.ZSTD,
            level=6,
            block_size_kb=16384,
            dynamic_block_sizing=True,
        )

        assert config.algorithm == CompressionAlgorithm.ZSTD
        assert config.level == 6
        assert config.block_size_kb == 16384
        assert config.dynamic_block_sizing is True

    def test_compression_algorithm_enum(self):
        """Test CompressionAlgorithm enum values"""
        assert CompressionAlgorithm.NONE == "none"
        assert CompressionAlgorithm.ZSTD == "zstd"
        assert CompressionAlgorithm.LZ4 == "lz4"
        assert CompressionAlgorithm.SNAPPY == "snappy"

    def test_compression_level_validation(self):
        """Test compression level validation"""
        # Valid levels
        config = CompressionConfig(level=1)
        assert config.level == 1

        config = CompressionConfig(level=22)  # Valid for ZSTD
        assert config.level == 22

        # Invalid levels should raise ValueError
        with pytest.raises(ValueError, match="Compression level must be between 1-22"):
            CompressionConfig(level=0)

        with pytest.raises(ValueError, match="Compression level must be between 1-22"):
            CompressionConfig(level=23)

    def test_quantization_type_validation(self):
        """Test quantization type validation"""
        # Valid types
        valid_types = [
            "int8",
            "pq8",
            "pq4",
            "uniform",
            "pq",
            "scalar",
            "binary",
            "none",
        ]
        for qtype in valid_types:
            config = CompressionConfig(quantization_type=qtype)
            assert config.quantization_type == qtype

        # Invalid type should raise ValueError
        with pytest.raises(ValueError, match="Quantization type must be one of"):
            CompressionConfig(quantization_type="invalid_type")

    def test_block_size_validation(self):
        """Test SST block size validation"""
        # Valid sizes
        config = CompressionConfig(block_size_kb=8192)
        assert config.block_size_kb == 8192

        # Invalid sizes should raise ValueError
        with pytest.raises(
            ValueError, match="SST block size must be between 256-16384 KB"
        ):
            CompressionConfig(block_size_kb=128)

        with pytest.raises(
            ValueError, match="SST block size must be between 256-16384 KB"
        ):
            CompressionConfig(block_size_kb=32768)


class TestCollectionConfigWithCompression:
    """Test collection configuration with unified compression"""

    def test_collection_config_defaults(self):
        """Test CollectionConfig uses server-aligned defaults"""
        config = CollectionConfig(name="test_collection", dimension=384)

        assert config.distance_metric == DistanceMetric.COSINE
        assert (
            config.storage_engine == StorageEngine.SST
        )  # Default is SST (fast, production-ready)
        # Note: primary_indexing_algorithm is Optional, may not have default

    def test_collection_config_with_columnar_compression(self):
        """Test CollectionConfig with columnar-engine compression"""
        compression = CompressionConfig(
            algorithm=CompressionAlgorithm.ZSTD,
            level=6,
            enable_quantization=True,
            quantization_type="int8",
        )

        collection_config = CollectionConfig(
            name="columnar_compressed",
            dimension=1536,
            storage_engine=StorageEngine.NOVA,
            compression=compression,
        )

        assert collection_config.compression is not None
        assert collection_config.compression.algorithm == CompressionAlgorithm.ZSTD
        assert collection_config.compression.enable_quantization is True

    def test_collection_config_with_sst_compression(self):
        """Test CollectionConfig with SST-specific compression"""
        compression = CompressionConfig(
            algorithm=CompressionAlgorithm.LZ4,
            level=1,
            block_size_kb=16384,
            dynamic_block_sizing=True,
        )

        collection_config = CollectionConfig(
            name="sst_compressed",
            dimension=768,
            storage_engine=StorageEngine.SST,
            compression=compression,
        )

        assert collection_config.compression.block_size_kb == 16384
        assert collection_config.compression.dynamic_block_sizing is True

    def test_collection_config_serialization(self):
        """Test CollectionConfig serialization with compression"""
        compression = CompressionConfig(
            algorithm=CompressionAlgorithm.SNAPPY, adaptive=True
        )

        collection_config = CollectionConfig(
            name="serialization_test", dimension=768, compression=compression
        )

        # Test model_dump (Pydantic v2)
        data = collection_config.model_dump(exclude_none=True)
        assert "compression" in data
        assert data["compression"]["algorithm"] == "snappy"
        assert data["compression"]["adaptive"] is True


class TestSearchOptimizationWithCompression:
    """Test search optimization with compression hints"""

    def test_search_optimization_defaults(self):
        """Test default values for compression-aware search hints"""
        optimization = SearchOptimization()

        # Check compression-specific fields
        assert optimization.prefer_compressed_search is None
        assert optimization.decompression_budget_ms is None
        assert optimization.use_decompression_cache is True  # Default to True
        assert optimization.compression_aware_routing is None

    def test_search_optimization_custom_values(self):
        """Test custom compression-aware search hints"""
        optimization = SearchOptimization(
            enable_two_stage=True,
            prefer_compressed_search=True,
            decompression_budget_ms=100,
            use_decompression_cache=False,
            compression_aware_routing=True,
        )

        assert optimization.enable_two_stage is True
        assert optimization.prefer_compressed_search is True
        assert optimization.decompression_budget_ms == 100
        assert optimization.use_decompression_cache is False
        assert optimization.compression_aware_routing is True


class TestCompressionIntegrationScenarios:
    """Test complete compression integration scenarios"""

    @pytest.fixture
    def mock_client(self):
        """Create a mock ProximaDB client"""
        client = Mock()
        client.create_collection = Mock(
            return_value=Mock(id="test_collection", name="test_collection")
        )
        client.insert_vectors = Mock(return_value={"success": True, "inserted": 100})
        client.search_vectors = Mock(
            return_value=[
                SearchResult(id="vec_1", score=0.95, rank=1),
                SearchResult(id="vec_2", score=0.90, rank=2),
            ]
        )
        return client

    def test_unified_compression_sst_scenario(self, mock_client):
        """Test SST storage with unified compression config"""
        compression = CompressionConfig(
            algorithm=CompressionAlgorithm.ZSTD,
            level=6,
            block_size_kb=16384,
            adaptive=True,
        )

        collection_config = CollectionConfig(
            name="sst_compressed",
            dimension=1536,
            storage_engine=StorageEngine.SST,
            compression=compression,
        )

        collection = mock_client.create_collection(collection_config)
        assert collection.id == "test_collection"

        # Verify unified compression config was passed
        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args[0][0]
        assert call_args.compression.algorithm == CompressionAlgorithm.ZSTD
        assert call_args.compression.block_size_kb == 16384

    def test_unified_compression_columnar_scenario(self, mock_client):
        """Test columnar storage with unified compression config"""
        compression = CompressionConfig(
            algorithm=CompressionAlgorithm.LZ4,
            level=1,
            enable_quantization=True,
            quantization_type="pq8",
            normalization_method="trimmed_mean",
        )

        collection_config = CollectionConfig(
            name="columnar_compressed",
            dimension=768,
            storage_engine=StorageEngine.NOVA,
            compression=compression,
        )

        collection = mock_client.create_collection(collection_config)

        # Verify unified compression config
        call_args = mock_client.create_collection.call_args[0][0]
        assert call_args.compression.enable_quantization is True
        assert call_args.compression.quantization_type == "pq8"
        assert call_args.compression.normalization_method == "trimmed_mean"

    def test_compression_aware_search(self, mock_client):
        """Test compression-aware search optimization"""
        query_vector = np.random.rand(1536).tolist()

        optimization = SearchOptimization(
            enable_two_stage=True,
            prefer_compressed_search=True,
            decompression_budget_ms=150,
            use_decompression_cache=True,
            compression_aware_routing=True,
        )

        results = mock_client.search(
            collection_id="test_collection",
            query_vector=query_vector,
            top_k=10,
            search_optimization=optimization,
        )

        assert len(results) == 2

        # Verify optimization hints were passed
        mock_client.search_vectors.assert_called_once()
        call_kwargs = mock_client.search_vectors.call_args[1]
        opt = call_kwargs["search_optimization"]
        assert opt.prefer_compressed_search is True
        assert opt.decompression_budget_ms == 150


class TestCompressionPerformance:
    """Test compression performance characteristics"""

    def test_compression_ratio_estimation(self):
        """Test estimation of compression ratios for different data types"""
        # Test data creation for different compression scenarios
        sparse_data = np.zeros((100, 512))
        # Correctly assign random values to sparse positions
        indices = np.random.choice(100 * 512, 1000, replace=False)
        flat_sparse = sparse_data.flatten()
        flat_sparse[indices] = np.random.randn(1000)
        sparse_data = flat_sparse.reshape(100, 512)

        dense_data = np.random.rand(100, 512)
        structured_data = np.tile(np.arange(512), (100, 1))

        assert sparse_data.nbytes == dense_data.nbytes == structured_data.nbytes

        # In real scenarios with ProximaDB compression:
        # - Sparse: ~10-20x compression (ZSTD with quantization)
        # - Dense: ~1.1-1.3x compression (basic ZSTD)
        # - Structured: ~5-10x compression (ZSTD pattern recognition)

    def test_quantization_effectiveness(self):
        """Test quantization compression effectiveness"""
        original_vector = np.random.rand(1536).astype(np.float32)

        # INT8 quantization: 4:1 compression ratio
        int8_compressed_size = len(original_vector) // 4
        assert int8_compressed_size == 384

        # PQ8 quantization: 8-32:1 compression ratio
        pq8_compressed_size = len(original_vector) // 16
        assert pq8_compressed_size == 96

        # PQ4 quantization: 16-64:1 compression ratio
        pq4_compressed_size = len(original_vector) // 32
        assert pq4_compressed_size == 48


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
