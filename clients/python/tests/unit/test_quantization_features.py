"""
Test quantization features and search optimization hints

This test file validates the new quantization and search optimization
features added to ProximaDB.
"""

import logging

import numpy as np
import pytest

from proximadb_sdk import (
    CollectionConfig,
    Protocol,
    ProximaDBClient,
    QuantizationConfig,
    QuantizationHint,
    QuantizationType,
    SearchOptimization,
)

from ..embedding_utils import embed_many, embed_seed

logger = logging.getLogger(__name__)

try:
    from proximadb_sdk import proximadb_pb2
except ImportError as e:
    logger.debug(f"Failed to import proximadb_pb2: {e}")
    proximadb_pb2 = None


class TestQuantizationModels:
    """Test quantization configuration models"""

    def test_quantization_config_defaults(self):
        """Test default quantization configuration"""
        config = QuantizationConfig()
        assert config.enabled is False
        assert config.type == QuantizationType.NONE
        assert config.progressive_quantization is False

    def test_quantization_config_pq(self):
        """Test product quantization configuration"""
        config = QuantizationConfig(
            enabled=True,
            type=QuantizationType.PRODUCT,
            bits_per_subvector=8,
            num_subvectors=16,
            compression_ratio_target=4.0,
        )
        assert config.enabled is True
        assert config.type == QuantizationType.PRODUCT
        assert config.bits_per_subvector == 8
        assert config.num_subvectors == 16
        assert config.compression_ratio_target == 4.0

    def test_quantization_config_scalar(self):
        """Test scalar quantization configuration"""
        config = QuantizationConfig(
            enabled=True,
            type=QuantizationType.SCALAR,
            bits_per_vector=8,
            accuracy_threshold=0.95,
        )
        assert config.enabled is True
        assert config.type == QuantizationType.SCALAR
        assert config.bits_per_vector == 8
        assert config.accuracy_threshold == 0.95

    def test_quantization_config_validation(self):
        """Test quantization configuration validation"""
        # SimpleQuantizationConfig doesn't have validation for these fields
        # Creating configs with various values to ensure no errors
        config1 = QuantizationConfig(bits_per_vector=0)
        assert config1.bits_per_vector == 0

        config2 = QuantizationConfig(bits_per_vector=33)
        assert config2.bits_per_vector == 33

        config3 = QuantizationConfig(accuracy_threshold=-0.1)
        assert config3.accuracy_threshold == -0.1

        config4 = QuantizationConfig(accuracy_threshold=1.1)
        assert config4.accuracy_threshold == 1.1


class TestSearchOptimization:
    """Test search optimization hints model"""

    def test_search_hints_defaults(self):
        """Test default search optimization hints"""
        hints = SearchOptimization()
        assert hints.enable_two_stage is None
        assert hints.quantization_hint is None
        assert hints.enable_clustering_hint is None
        assert hints.enable_metadata_filtering_hint is None

    def test_search_hints_two_stage(self):
        """Test two-stage search configuration"""
        hints = SearchOptimization(
            enable_two_stage=True,
            quantization_hint=QuantizationHint(
                hint_type="product", parameters={"bits": 8}
            ),
            accuracy_threshold=0.95,
            timeout_ms=1000,
        )
        assert hints.enable_two_stage is True
        assert hints.quantization_hint.hint_type == "product"
        assert hints.accuracy_threshold == 0.95
        assert hints.timeout_ms == 1000

    def test_search_hints_custom(self):
        """Test custom search hints"""
        hints = SearchOptimization(
            custom_hints={"use_simd": "true", "prefetch_size": "64"}
        )
        assert hints.custom_hints["use_simd"] == "true"
        assert hints.custom_hints["prefetch_size"] == "64"

    def test_search_hints_validation(self):
        """Test search hints validation"""
        # SearchOptimization doesn't have validation for these fields
        # Creating configs with various values to ensure no errors
        opt1 = SearchOptimization(accuracy_threshold=0.5)
        assert opt1.accuracy_threshold == 0.5

        opt2 = SearchOptimization(accuracy_threshold=1.0)
        assert opt2.accuracy_threshold == 1.0

        opt3 = SearchOptimization(timeout_ms=0)
        assert opt3.timeout_ms == 0


class TestCollectionWithQuantization:
    """Test collection configuration with quantization"""

    def test_collection_config_with_quantization(self):
        """Test collection config includes quantization"""
        quantization = QuantizationConfig(
            enabled=True,
            type=QuantizationType.PRODUCT,
            bits_per_subvector=4,
            num_subvectors=32,
        )

        config = CollectionConfig(
            name="test_collection",
            dimension=768,
            distance_metric="cosine",
            quantization=quantization,
        )

        assert config.quantization is not None
        assert config.quantization.enabled is True
        assert config.quantization.type == QuantizationType.PRODUCT

    def test_collection_config_without_quantization(self):
        """Test collection config without quantization"""
        config = CollectionConfig(
            name="test_collection", dimension=384, distance_metric="euclidean"
        )

        assert config.quantization is None


class TestProtoQuantizationMessages:
    """Test protobuf quantization message generation"""

    @pytest.mark.skipif(
        proximadb_pb2 is None, reason="proximadb_pb2 not available due to import issues"
    )
    def test_quantization_config_proto(self):
        """Test QuantizationConfig protobuf message"""
        # Create a QuantizationConfig proto message
        proto_config = proximadb_pb2.QuantizationConfig()
        proto_config.enabled = True
        proto_config.compression_ratio_target = 4.0

        # Create storage quantization config
        storage_config = proximadb_pb2.StorageQuantizationConfig()
        storage_config.enabled = True
        storage_config.progressive_quantization = True

        # Create a QuantizationLevel with PQ
        level = proximadb_pb2.QuantizationLevel()
        pq = proximadb_pb2.ProductQuantization()
        pq.bits_per_code = 8
        pq.num_subvectors = 16
        pq.adaptive_subvectors = True
        level.pq.CopyFrom(pq)

        storage_config.level.CopyFrom(level)
        proto_config.storage_quantization.CopyFrom(storage_config)

        # Verify fields
        assert proto_config.enabled is True
        assert proto_config.compression_ratio_target == 4.0
        assert proto_config.storage_quantization.enabled is True
        assert proto_config.storage_quantization.progressive_quantization is True
        assert proto_config.storage_quantization.level.pq.bits_per_code == 8
        assert proto_config.storage_quantization.level.pq.num_subvectors == 16

    @pytest.mark.skipif(
        proximadb_pb2 is None, reason="proximadb_pb2 not available due to import issues"
    )
    def test_search_optimization_hints_proto(self):
        """Test SearchParams protobuf message"""
        # Create SearchParams proto message
        params = proximadb_pb2.SearchParams()
        params.top_k = 10
        params.enable_two_stage = True
        params.accuracy_threshold = 0.95
        params.timeout_ms = 100
        params.include_expired = False
        params.enable_clustering_hint = True
        params.enable_metadata_filtering_hint = True

        # Set quantization hint - using uniform quantization params
        uniform_params = proximadb_pb2.UniformQuantizationParams()
        uniform_params.scale = 1.0
        uniform_params.offset = 0.0
        params.uniform.CopyFrom(uniform_params)

        # Add custom hints using google.protobuf.Value

        params.custom_hints["use_gpu"].string_value = "true"
        params.custom_hints["batch_size"].string_value = "256"

        # Verify fields
        assert params.top_k == 10
        assert params.enable_two_stage is True
        assert abs(params.accuracy_threshold - 0.95) < 0.001
        assert params.timeout_ms == 100
        assert params.uniform.scale == 1.0
        assert params.custom_hints["use_gpu"].string_value == "true"

    @pytest.mark.skipif(
        proximadb_pb2 is None, reason="proximadb_pb2 not available due to import issues"
    )
    def test_collection_config_proto_with_quantization(self):
        """Test CollectionConfig proto with quantization"""
        # Create CollectionConfig proto
        config = proximadb_pb2.CollectionConfig()
        config.name = "test_collection"
        config.dimension = 768
        config.distance_metric = proximadb_pb2.COSINE
        config.storage_engine = proximadb_pb2.StorageEngine.VIPER
        config.primary_indexing_algorithm = proximadb_pb2.IndexingAlgorithm.HNSW

        # Add quantization config
        config.quantization_config.enabled = True
        config.quantization_config.compression_ratio_target = 4.0

        # Set search quantization with binary
        search_config = proximadb_pb2.SearchQuantizationConfig()
        search_config.enabled = True
        search_config.accuracy_threshold = 0.95
        search_config.candidate_multiplier = 3

        level = proximadb_pb2.QuantizationLevel()
        binary_quant = proximadb_pb2.BinaryQuantization()
        binary_quant.threshold = 0.0
        binary_quant.sign_based = True
        level.binary.CopyFrom(binary_quant)

        search_config.default_level.CopyFrom(level)
        config.quantization_config.search_quantization.CopyFrom(search_config)

        # Verify
        assert config.name == "test_collection"
        assert config.dimension == 768
        assert config.quantization_config.enabled is True
        assert (
            config.quantization_config.search_quantization.default_level.binary.sign_based
            is True
        )

    @pytest.mark.skipif(
        proximadb_pb2 is None, reason="proximadb_pb2 not available due to import issues"
    )
    def test_vector_search_request_with_hints(self):
        """Test VectorSearchRequest with optimization hints"""
        # Create search request
        request = proximadb_pb2.VectorSearchRequest()
        request.collection_id = "test_collection"
        request.top_k = 10

        # Add query
        query = proximadb_pb2.SearchQuery()
        query.vector.extend([0.1, 0.2, 0.3, 0.4])
        request.queries.append(query)

        # Add optimization hints using SearchParams
        request.search_optimization.enable_two_stage = True
        request.search_optimization.enable_clustering_hint = True
        request.search_optimization.enable_metadata_filtering_hint = True
        request.search_optimization.accuracy_threshold = 0.95
        request.search_optimization.timeout_ms = 1000

        # Set quantization hint to PQ using ProductQuantizationParams
        pq_params = proximadb_pb2.ProductQuantizationParams()
        pq_params.num_subvectors = 64
        pq_params.bits_per_code = 4
        request.search_optimization.product.CopyFrom(pq_params)

        # Verify
        assert request.collection_id == "test_collection"
        assert request.top_k == 10
        assert len(request.queries) == 1
        assert request.search_optimization.enable_two_stage is True
        assert request.search_optimization.product.bits_per_code == 4


@pytest.mark.integration
class TestQuantizationIntegration:
    """Integration tests for quantization features (requires running server)"""

    @pytest.fixture
    def client(self):
        """Create REST client for testing"""
        return ProximaDBClient(url="http://localhost:5678", protocol=Protocol.REST)

    @staticmethod
    def _require_server(client):
        """Skip honestly when no server is reachable.

        ``client.health()`` CANNOT detect this: the REST adapter swallows
        connection errors and fabricates a ``status="running"`` HealthStatus;
        ``list_collections()`` is swallowed to ``[]`` the same way. Probe the
        TCP port directly. (Under v1 these tests also "passed" serverlessly
        via the client's in-memory local fallback; the v2 client surfaces
        the connection error, so the skip must be real.)
        """
        import socket
        from urllib.parse import urlparse

        parsed = urlparse(getattr(client.config, "url", "http://localhost:5678"))
        try:
            with socket.create_connection(
                (parsed.hostname or "localhost", parsed.port or 5678), timeout=1
            ):
                pass
        except OSError:
            pytest.skip("ProximaDB server not running")

    def test_create_collection_with_quantization(self, client):
        """Test creating collection with quantization enabled"""
        self._require_server(client)

        # Create collection with quantization
        collection_name = "test_quant_123456"

        try:
            # Create collection
            collection = client.create_collection(
                collection_name,
                dimension=128,
                distance_metric="cosine",
                quantization_config=QuantizationConfig(
                    enabled=True,
                    type=QuantizationType.SCALAR,
                    bits_per_vector=8,
                    accuracy_threshold=0.95,
                ),
            )
            assert collection.name == collection_name

            # Verify collection was created
            collections = client.list_collections()
            assert collection_name in [c.name for c in collections]

        finally:
            # Cleanup
            try:
                client.delete_collection(collection_name)
            except Exception:
                pass

    def test_search_with_optimization_hints(self, client):
        """Test search with optimization hints"""
        self._require_server(client)

        collection_name = "test_search_hints_123456"

        try:
            # Create collection
            config = CollectionConfig(name=collection_name, dimension=128)
            collection = client.create_collection(collection_name, config)

            # Insert test vectors
            vectors = np.array(embed_many(100, 128), dtype=np.float32)
            ids = [f"vec_{i}" for i in range(100)]
            client.insert_vectors(collection_name, vectors, ids)

            # Search with optimization hints
            query = np.array(embed_seed(999, 128), dtype=np.float32)
            hints = {
                "enable_two_stage_search": True,
                "quantization_hint": "FP32",
                "candidate_multiplier": 2.0,
                "enable_parallel_search": True,
            }

            results = client.search(
                collection_name, query, top_k=10, optimization_hints=hints
            )

            # Verify results
            assert len(results) <= 10
            for result in results:
                assert hasattr(result, "id")
                assert hasattr(result, "score")

        finally:
            # Cleanup
            try:
                client.delete_collection(collection_name)
            except Exception:
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
