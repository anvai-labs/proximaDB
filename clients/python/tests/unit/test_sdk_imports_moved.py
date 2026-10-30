#!/usr/bin/env python3
"""
Test module to verify all ProximaDB SDK imports are working correctly
"""

import sys
from pathlib import Path

import pytest

# Add the src directory to path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestImports:
    """Test all ProximaDB SDK imports"""

    def test_basic_imports(self):
        """Test basic module imports"""
        import proximadb_sdk

        assert proximadb_sdk.__version__
        assert proximadb_sdk.__author__

    def test_client_imports(self):
        """Test client imports"""
        from proximadb_sdk import (
            Protocol,
            ProximaDBClient,
            connect,
            connect_grpc,
            connect_rest,
        )

        assert ProximaDBClient is not None
        assert callable(connect)
        assert callable(connect_grpc)
        assert callable(connect_rest)
        assert Protocol.REST.value == "rest"
        assert Protocol.GRPC.value == "grpc"

    def test_config_imports(self):
        """Test configuration imports"""
        from proximadb_sdk import ClientConfig, CompressionConfig

        # Test ClientConfig
        config = ClientConfig(url="http://localhost:5678")
        assert config.url == "http://localhost:5678"

        # Test CompressionConfig with actual API
        compression = CompressionConfig(enabled=True, algorithm="gzip")
        assert compression.enabled is True
        assert compression.algorithm == "gzip"

    def test_model_imports(self):
        """Test model imports"""
        from proximadb_sdk import (
            DistanceMetric,
            IndexingAlgorithm,
            QuantizationType,
            StorageEngine,
        )

        # Test enums
        assert DistanceMetric.COSINE
        assert IndexingAlgorithm.HNSW
        assert StorageEngine.VIPER
        assert QuantizationType.UNIFORM

    def test_exception_imports(self):
        """Test exception imports"""
        from proximadb_sdk import (
            AuthenticationError,
            CollectionNotFoundError,
            ProximaDBError,
        )

        # Test exception hierarchy
        assert issubclass(AuthenticationError, ProximaDBError)
        assert issubclass(CollectionNotFoundError, ProximaDBError)

    def test_chunking_imports(self):
        """Test text chunking imports (opt-in `[chunking]` extra)."""
        # Chunking public names are exported only when the `[chunking]` extra
        # (tree-sitter) is installed; skip when it isn't, matching the SDK's
        # opt-in design (see proximadb_sdk.__init__._chunking_concern_available).
        pytest.importorskip("tree_sitter")

        from proximadb_sdk import ChunkingStrategy, chunk_by_sentences, create_chunker

        # Test ChunkingStrategy enum
        assert ChunkingStrategy.SENTENCE
        assert ChunkingStrategy.SLIDING_WINDOW

        # Test callable functions
        assert callable(create_chunker)
        assert callable(chunk_by_sentences)

    def test_filter_imports(self):
        """Test filter API imports"""
        from proximadb_sdk import FilterOp, LogicalOp, and_filters, eq

        # Test enums
        assert FilterOp.EQUALS
        assert LogicalOp.AND

        # Test callable functions
        assert callable(eq)
        assert callable(and_filters)

    def test_all_exports(self):
        """Test that all items in __all__ are importable"""
        import proximadb_sdk

        for item in proximadb_sdk.__all__:
            assert hasattr(proximadb_sdk, item), f"Missing export: {item}"

    def test_optional_grpc_imports(self):
        """Test optional gRPC imports"""
        try:
            from proximadb_sdk import proximadb_pb2, proximadb_pb2_grpc

            # If we get here, gRPC is available
            assert (
                proximadb_pb2 is not None or proximadb_pb2 is None
            )  # Could be None if not installed
            assert proximadb_pb2_grpc is not None or proximadb_pb2_grpc is None
        except ImportError:
            # gRPC not available, which is fine
            pass

    def test_backwards_compatibility(self):
        """Test backwards compatibility aliases"""
        from proximadb_sdk import IndexConfig, IndexConfiguration, Vector, VectorRecord

        # These should be the same
        assert IndexConfig is IndexConfiguration
        assert Vector is VectorRecord

    def test_from_import_patterns(self):
        """Test common from-import patterns"""
        # Pattern 1: Import everything from proximadb
        from proximadb_sdk.chunking import TextChunker

        # Pattern 2: Import from submodules
        from proximadb_sdk.config import ClientConfig
        from proximadb_sdk.exceptions import ProximaDBError
        from proximadb_sdk.models import Collection

        # All imports should work
        assert ClientConfig is not None
        assert Collection is not None
        assert ProximaDBError is not None
        assert TextChunker is not None


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
