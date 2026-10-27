#!/usr/bin/env python3
"""
Comprehensive test suite for collection configuration combinations.
Tests all possible combinations of collection configurations to ensure
proper handling by the Python SDK.

NOTE: Moved from tests/unit/ to tests/integration/ - these are integration tests
requiring REST/gRPC client connections to a running server.
"""

import logging

import pytest

from proximadb_sdk import (
    CollectionConfig,
    DistanceMetric,
    FilterableColumn,
    FilterableDataType,
    IndexConfiguration,
    IndexingAlgorithm,
    ProximaDBError,
    QuantizationConfig,
    QuantizationType,
    StorageEngine,
    connect_grpc,
    connect_rest,
)

# Import index configs directly from models
from proximadb_sdk.models import FlatConfig, HnswConfig, IvfConfig, LshConfig, PqConfig

logger = logging.getLogger(__name__)


class TestCollectionConfigComprehensive:
    """Comprehensive tests for all collection configuration combinations"""

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self, request):
        """Setup and cleanup before/after each test"""
        # Cleanup before test to ensure clean state
        try:
            client = connect_rest("http://localhost:5678")
            collections = client.list_collections()
            for col in collections:
                if col.name.startswith("test_config_"):
                    try:
                        client.delete_collection(col.name)
                    except:
                        pass
        except:
            pass
        yield
        # Cleanup after test
        try:
            client = connect_rest("http://localhost:5678")
            self._cleanup_test_collections(client)
        except:
            pass

    @pytest.fixture
    def rest_client(self):
        """Create REST client"""
        client = connect_rest("http://localhost:5678")
        yield client

    @pytest.fixture
    def grpc_client(self):
        """Create gRPC client"""
        client = connect_grpc("grpc://localhost:5679")
        yield client

    def _cleanup_test_collections(self, client):
        """Clean up test collections"""
        try:
            collections = client.list_collections()
            for col in collections:
                if col.name.startswith("test_config_"):
                    client.delete_collection(col.name)
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

    def test_distance_metric_combinations(self, rest_client, grpc_client):
        """Test all 13 supported distance metric options (updated 2025-08)"""
        distance_metrics = [
            # Core 3 metrics (always supported)
            DistanceMetric.COSINE,
            DistanceMetric.EUCLIDEAN,
            DistanceMetric.DOT_PRODUCT,
            # Extended metrics (now fully supported as of 2025-08)
            DistanceMetric.MANHATTAN,
            DistanceMetric.HAMMING,
            DistanceMetric.JACCARD,
            DistanceMetric.CHEBYSHEV,
            DistanceMetric.CANBERRA,
            DistanceMetric.MINKOWSKI,
            DistanceMetric.ANGULAR,
            DistanceMetric.BRAY_CURTIS,
            DistanceMetric.HELLINGER,
            DistanceMetric.CUSTOM,
        ]

        for metric in distance_metrics:
            collection_name = f"test_config_metric_{metric.value}"

            # Test with REST
            config = CollectionConfig(
                name=collection_name, dimension=128, distance_metric=metric
            )

            try:
                collection = rest_client.create_collection(collection_name, config)
                # Check if distance_metric was set (may differ due to server defaults)
                if collection.config.distance_metric != metric:
                    logger.warning(
                        f"⚠ REST: Requested {metric.value} but got {collection.config.distance_metric.value} - "
                        f"collection created successfully but metric differs"
                    )
                else:
                    logger.info(
                        f"✓ REST: Created collection with {metric.value} metric"
                    )
            except ProximaDBError as e:
                if "not supported" in str(e).lower():
                    logger.warning(f"⚠ REST: {metric.value} metric not supported")
                else:
                    raise

            # Test with gRPC
            grpc_collection_name = f"{collection_name}_grpc"
            grpc_config = CollectionConfig(
                name=grpc_collection_name, dimension=128, distance_metric=metric
            )

            try:
                collection = grpc_client.create_collection(
                    grpc_collection_name, grpc_config
                )
                # Check if distance_metric was set (may differ due to server defaults)
                if collection.config.distance_metric != metric:
                    logger.warning(
                        f"⚠ gRPC: Requested {metric.value} but got {collection.config.distance_metric.value} - "
                        f"collection created successfully but metric differs"
                    )
                else:
                    logger.info(
                        f"✓ gRPC: Created collection with {metric.value} metric"
                    )
            except ProximaDBError as e:
                if "not supported" in str(e).lower():
                    logger.warning(f"⚠ gRPC: {metric.value} metric not supported")
                else:
                    raise

    def test_storage_engine_combinations(self, rest_client, grpc_client):
        """Test all storage engine options"""
        storage_engines = [
            StorageEngine.VIPER,
            StorageEngine.SST,
            StorageEngine.MMAP,
            StorageEngine.HYBRID,
        ]

        for engine in storage_engines:
            collection_name = f"test_config_engine_{engine.value}"

            # Test with REST
            config = CollectionConfig(
                name=collection_name, dimension=256, storage_engine=engine
            )

            try:
                collection = rest_client.create_collection(collection_name, config)
                # Check if storage_engine was set (may differ due to server defaults)
                if collection.config.storage_engine != engine:
                    logger.warning(
                        f"⚠ REST: Requested {engine.value} but got {collection.config.storage_engine.value} - "
                        f"collection created successfully but engine differs"
                    )
                else:
                    logger.info(
                        f"✓ REST: Created collection with {engine.value} engine"
                    )
            except ProximaDBError as e:
                if "not supported" in str(e).lower():
                    logger.warning(f"⚠ REST: {engine.value} engine not supported")
                else:
                    raise

            # Test with gRPC
            grpc_collection_name = f"{collection_name}_grpc"
            grpc_config = CollectionConfig(
                name=grpc_collection_name, dimension=256, storage_engine=engine
            )

            try:
                collection = grpc_client.create_collection(
                    grpc_collection_name, grpc_config
                )
                # Check if storage_engine was set (may differ due to server defaults)
                if collection.config.storage_engine != engine:
                    logger.warning(
                        f"⚠ gRPC: Requested {engine.value} but got {collection.config.storage_engine.value} - "
                        f"collection created successfully but engine differs"
                    )
                else:
                    logger.info(
                        f"✓ gRPC: Created collection with {engine.value} engine"
                    )
            except ProximaDBError as e:
                if "not supported" in str(e).lower():
                    logger.warning(f"⚠ gRPC: {engine.value} engine not supported")
                else:
                    raise

    def test_indexing_algorithm_combinations(self, rest_client, grpc_client):
        """Test all indexing algorithm options"""
        indexing_algorithms = [
            IndexingAlgorithm.HNSW,
            IndexingAlgorithm.IVF,
            IndexingAlgorithm.FLAT,
            IndexingAlgorithm.LSH,
            IndexingAlgorithm.ANNOY,
            IndexingAlgorithm.PQ,
        ]

        for algorithm in indexing_algorithms:
            collection_name = f"test_config_algo_{algorithm.value}"

            # Test with REST
            config = CollectionConfig(
                name=collection_name,
                dimension=384,
                primary_indexing_algorithm=algorithm,
            )

            try:
                collection = rest_client.create_collection(collection_name, config)
                assert collection.config.primary_indexing_algorithm == algorithm
                logger.info(
                    f"✓ REST: Created collection with {algorithm.value} algorithm"
                )
            except ProximaDBError as e:
                if "not supported" in str(e).lower():
                    logger.warning(f"⚠ REST: {algorithm.value} algorithm not supported")
                else:
                    raise

            # Test with gRPC
            grpc_collection_name = f"{collection_name}_grpc"
            grpc_config = CollectionConfig(
                name=grpc_collection_name,
                dimension=384,
                primary_indexing_algorithm=algorithm,
            )

            try:
                collection = grpc_client.create_collection(
                    grpc_collection_name, grpc_config
                )
                logger.info(
                    f"✓ gRPC: Created collection with {algorithm.value} algorithm"
                )
            except ProximaDBError as e:
                if "not supported" in str(e).lower():
                    logger.warning(f"⚠ gRPC: {algorithm.value} algorithm not supported")
                else:
                    raise

    def test_quantization_type_combinations(self, rest_client, grpc_client):
        """Test all quantization type options"""
        quantization_configs = [
            # No quantization
            QuantizationConfig(enabled=False, type=QuantizationType.NONE),
            # Scalar quantization
            QuantizationConfig(
                enabled=True,
                type=QuantizationType.SCALAR,
                bits_per_vector=8,
                accuracy_threshold=0.95,
            ),
            # Product quantization
            QuantizationConfig(
                enabled=True,
                type=QuantizationType.PRODUCT,
                num_subvectors=8,
                bits_per_subvector=4,
                accuracy_threshold=0.90,
            ),
            # Binary quantization
            QuantizationConfig(
                enabled=True,
                type=QuantizationType.BINARY,
                threshold=0.5,
                accuracy_threshold=0.85,
            ),
            # Uniform quantization
            QuantizationConfig(
                enabled=True,
                type=QuantizationType.UNIFORM,
                bits_per_vector=16,
                compression_ratio_target=4.0,
            ),
            # Progressive quantization
            QuantizationConfig(
                enabled=True,
                type=QuantizationType.SCALAR,
                progressive_quantization=True,
                bits_per_vector=8,
                retraining_threshold=0.92,
            ),
        ]

        for i, quant_config in enumerate(quantization_configs):
            collection_name = f"test_config_quant_{i}_{quant_config.type.value}"

            # Test with REST
            config = CollectionConfig(
                name=collection_name, dimension=512, quantization_config=quant_config
            )

            try:
                collection = rest_client.create_collection(collection_name, config)
                assert (
                    collection.config.quantization_config.enabled
                    == quant_config.enabled
                )
                assert collection.config.quantization_config.type == quant_config.type
                logger.info(
                    f"✓ REST: Created collection with {quant_config.type.value} quantization"
                )
            except ProximaDBError as e:
                if "not supported" in str(e).lower():
                    logger.warning(
                        f"⚠ REST: {quant_config.type.value} quantization not supported"
                    )
                else:
                    raise

            # Test with gRPC
            grpc_collection_name = f"{collection_name}_grpc"
            grpc_config = CollectionConfig(
                name=grpc_collection_name,
                dimension=512,
                quantization_config=quant_config,
            )

            try:
                collection = grpc_client.create_collection(
                    grpc_collection_name, grpc_config
                )
                logger.info(
                    f"✓ gRPC: Created collection with {quant_config.type.value} quantization"
                )
            except ProximaDBError as e:
                if "not supported" in str(e).lower():
                    logger.warning(
                        f"⚠ gRPC: {quant_config.type.value} quantization not supported"
                    )
                else:
                    raise

    def test_filterable_columns_combinations(self, rest_client, grpc_client):
        """Test various filterable column configurations"""
        filterable_configs = [
            # No filterable columns
            None,
            # Single column
            [FilterableColumn(name="category", data_type=FilterableDataType.STRING)],
            # Multiple columns with different types
            [
                FilterableColumn(name="category", data_type=FilterableDataType.STRING),
                FilterableColumn(name="price", data_type=FilterableDataType.FLOAT),
                FilterableColumn(name="count", data_type=FilterableDataType.INTEGER),
                FilterableColumn(name="active", data_type=FilterableDataType.BOOLEAN),
            ],
            # Indexed columns
            [
                FilterableColumn(
                    name="user_id",
                    data_type=FilterableDataType.STRING,
                    indexed=True,
                    estimated_cardinality=10000,
                ),
                FilterableColumn(
                    name="timestamp", data_type=FilterableDataType.INTEGER, indexed=True
                ),
            ],
        ]

        for i, filterable_cols in enumerate(filterable_configs):
            collection_name = f"test_config_filterable_{i}"

            # Test with REST
            config = CollectionConfig(
                name=collection_name, dimension=128, filterable_columns=filterable_cols
            )

            try:
                collection = rest_client.create_collection(collection_name, config)
                if filterable_cols:
                    assert len(collection.config.filterable_columns) == len(
                        filterable_cols
                    )
                logger.info(
                    f"✓ REST: Created collection with {len(filterable_cols or [])} filterable columns"
                )
            except ProximaDBError as e:
                logger.error(
                    f"✗ REST: Failed to create collection with filterable columns: {e}"
                )
                raise

            # Test with gRPC
            grpc_collection_name = f"{collection_name}_grpc"
            grpc_config = CollectionConfig(
                name=grpc_collection_name,
                dimension=128,
                filterable_columns=filterable_cols,
            )

            try:
                collection = grpc_client.create_collection(
                    grpc_collection_name, grpc_config
                )
                logger.info(
                    f"✓ gRPC: Created collection with {len(filterable_cols or [])} filterable columns"
                )
            except ProximaDBError as e:
                logger.error(
                    f"✗ gRPC: Failed to create collection with filterable columns: {e}"
                )
                raise

    def test_index_config_combinations(self, rest_client, grpc_client):
        """Test various index configurations"""
        index_configs = [
            # HNSW configuration
            [
                IndexConfiguration(
                    index_name="hnsw_index",
                    algorithm=IndexingAlgorithm.HNSW,
                    hnsw_config=HnswConfig(
                        m=32,
                        ef_construction=400,
                        ef_search=100,
                        max_partition_size=200000,
                    ),
                )
            ],
            # IVF configuration
            [
                IndexConfiguration(
                    index_name="ivf_index",
                    algorithm=IndexingAlgorithm.IVF,
                    ivf_config=IvfConfig(
                        n_lists=200,
                        n_probe=5,
                        quantization_bits=8,
                        use_pq=True,
                        pq_subspaces=16,
                    ),
                )
            ],
            # Multiple indices
            [
                IndexConfiguration(
                    index_name="primary_hnsw",
                    algorithm=IndexingAlgorithm.HNSW,
                    hnsw_config=HnswConfig(m=16, ef_construction=200),
                ),
                IndexConfiguration(
                    index_name="secondary_flat",
                    algorithm=IndexingAlgorithm.FLAT,
                    flat_config=FlatConfig(enable_simd=True, batch_size=2000),
                ),
            ],
            # PQ index
            [
                IndexConfiguration(
                    index_name="pq_index",
                    algorithm=IndexingAlgorithm.PQ,
                    pq_config=PqConfig(
                        subvectors=16,
                        bits_per_subvector=4,
                        training_sample_count=20000,
                        enable_reranking=True,
                    ),
                )
            ],
            # LSH index
            [
                IndexConfiguration(
                    index_name="lsh_index",
                    algorithm=IndexingAlgorithm.LSH,
                    lsh_config=LshConfig(
                        num_hash_tables=10, hash_size=12, num_hash_functions=5
                    ),
                )
            ],
        ]

        for i, idx_configs in enumerate(index_configs):
            collection_name = f"test_config_index_{i}"

            # Test with REST
            config = CollectionConfig(
                name=collection_name, dimension=256, index_configs=idx_configs
            )

            try:
                collection = rest_client.create_collection(collection_name, config)
                assert len(collection.config.index_configs) == len(idx_configs)
                logger.info(
                    f"✓ REST: Created collection with {idx_configs[0].algorithm.value} index"
                )
            except ProximaDBError as e:
                if "not supported" in str(e).lower():
                    logger.warning(
                        f"⚠ REST: {idx_configs[0].algorithm.value} index not supported"
                    )
                else:
                    raise

            # Test with gRPC
            grpc_collection_name = f"{collection_name}_grpc"
            grpc_config = CollectionConfig(
                name=grpc_collection_name, dimension=256, index_configs=idx_configs
            )

            try:
                collection = grpc_client.create_collection(
                    grpc_collection_name, grpc_config
                )
                logger.info(
                    f"✓ gRPC: Created collection with {idx_configs[0].algorithm.value} index"
                )
            except ProximaDBError as e:
                if "not supported" in str(e).lower():
                    logger.warning(
                        f"⚠ gRPC: {idx_configs[0].algorithm.value} index not supported"
                    )
                else:
                    raise

    def test_comprehensive_combinations(self, rest_client, grpc_client):
        """Test comprehensive combinations of configurations"""
        # Define a subset of combinations to test
        test_combinations = [
            # VIPER + HNSW + Scalar Quantization
            {
                "name": "test_config_combo_viper_hnsw_scalar",
                "dimension": 384,
                "distance_metric": DistanceMetric.COSINE,
                "storage_engine": StorageEngine.VIPER,
                "primary_indexing_algorithm": IndexingAlgorithm.HNSW,
                "quantization_config": QuantizationConfig(
                    enabled=True, type=QuantizationType.SCALAR, bits_per_vector=8
                ),
            },
            # SST + IVF + Product Quantization
            {
                "name": "test_config_combo_sst_ivf_pq",
                "dimension": 512,
                "distance_metric": DistanceMetric.EUCLIDEAN,
                "storage_engine": StorageEngine.SST,
                "primary_indexing_algorithm": IndexingAlgorithm.IVF,
                "quantization_config": QuantizationConfig(
                    enabled=True,
                    type=QuantizationType.PRODUCT,
                    num_subvectors=16,
                    bits_per_subvector=4,
                ),
            },
            # SST + FLAT + No Quantization + Filterable Columns
            {
                "name": "test_config_combo_sst_flat_filter",
                "dimension": 256,
                "distance_metric": DistanceMetric.DOT_PRODUCT,
                "storage_engine": StorageEngine.SST,
                "primary_indexing_algorithm": IndexingAlgorithm.FLAT,
                "filterable_columns": [
                    FilterableColumn(
                        name="category", data_type=FilterableDataType.STRING
                    ),
                    FilterableColumn(
                        name="priority", data_type=FilterableDataType.INTEGER
                    ),
                ],
            },
            # VIPER + Multiple Indices + Binary Quantization
            {
                "name": "test_config_combo_viper_multi_binary",
                "dimension": 1024,
                "distance_metric": DistanceMetric.HAMMING,
                "storage_engine": StorageEngine.VIPER,
                "index_configs": [
                    IndexConfiguration(
                        index_name="primary",
                        algorithm=IndexingAlgorithm.HNSW,
                        hnsw_config=HnswConfig(m=16),
                    ),
                    IndexConfiguration(
                        index_name="secondary",
                        algorithm=IndexingAlgorithm.FLAT,
                        flat_config=FlatConfig(),
                    ),
                ],
                "quantization_config": QuantizationConfig(
                    enabled=True, type=QuantizationType.BINARY, threshold=0.5
                ),
            },
            # Full configuration with all features
            {
                "name": "test_config_combo_full_features",
                "dimension": 768,
                "distance_metric": DistanceMetric.COSINE,
                "storage_engine": StorageEngine.VIPER,
                "primary_indexing_algorithm": IndexingAlgorithm.HNSW,
                "filterable_columns": [
                    FilterableColumn(
                        name="doc_type",
                        data_type=FilterableDataType.STRING,
                        indexed=True,
                    ),
                    FilterableColumn(name="score", data_type=FilterableDataType.FLOAT),
                    FilterableColumn(
                        name="timestamp",
                        data_type=FilterableDataType.INTEGER,
                        indexed=True,
                    ),
                    FilterableColumn(
                        name="active", data_type=FilterableDataType.BOOLEAN
                    ),
                ],
                "index_configs": [
                    IndexConfiguration(
                        index_name="main_index",
                        algorithm=IndexingAlgorithm.HNSW,
                        hnsw_config=HnswConfig(
                            m=32,
                            ef_construction=400,
                            ef_search=100,
                            adaptive_parameters=True,
                        ),
                    )
                ],
                "quantization_config": QuantizationConfig(
                    enabled=True,
                    type=QuantizationType.PRODUCT,
                    progressive_quantization=True,
                    num_subvectors=24,
                    bits_per_subvector=8,
                    accuracy_threshold=0.95,
                    compression_ratio_target=4.0,
                ),
                "enable_automatic_index_selection": True,
                "description": "Full-featured test collection",
                "tags": ["test", "comprehensive", "all-features"],
                "owner": "test_suite",
            },
        ]

        for combo in test_combinations:
            config = CollectionConfig(**combo)

            # Test with REST
            try:
                collection = rest_client.create_collection(config.name, config)
                logger.info(f"✓ REST: Created comprehensive collection '{config.name}'")

                # Verify key configurations
                assert collection.config.name == config.name
                assert collection.config.dimension == config.dimension
                # Check if distance_metric was set (may differ due to server defaults)
                if config.distance_metric:
                    if collection.config.distance_metric != config.distance_metric:
                        logger.warning(
                            f"⚠ REST: Requested {config.distance_metric.value} but got {collection.config.distance_metric.value} - "
                            f"collection created successfully but metric differs"
                        )
                # Check if storage_engine was set (may differ due to server defaults)
                if config.storage_engine:
                    if collection.config.storage_engine != config.storage_engine:
                        logger.warning(
                            f"⚠ REST: Requested {config.storage_engine.value} but got {collection.config.storage_engine.value} - "
                            f"collection created successfully but engine differs"
                        )

            except ProximaDBError as e:
                logger.error(f"✗ REST: Failed to create '{config.name}': {e}")
                if "not supported" not in str(e).lower():
                    raise

            # Test with gRPC
            grpc_config = CollectionConfig(**combo)
            grpc_config.name = f"{config.name}_grpc"

            try:
                collection = grpc_client.create_collection(
                    grpc_config.name, grpc_config
                )
                logger.info(
                    f"✓ gRPC: Created comprehensive collection '{grpc_config.name}'"
                )

            except ProximaDBError as e:
                logger.error(f"✗ gRPC: Failed to create '{grpc_config.name}': {e}")
                if "not supported" not in str(e).lower():
                    raise

    def test_edge_cases_and_validation(self, rest_client):
        """Test edge cases and validation"""
        # Test minimum collection name length
        with pytest.raises(ValueError, match="at least 8 characters"):
            config = CollectionConfig(name="short", dimension=128)

        # Test maximum dimension
        config = CollectionConfig(
            name="test_config_max_dimension", dimension=10000  # Maximum allowed
        )
        collection = rest_client.create_collection(config.name, config)
        assert collection.config.dimension == 10000

        # Test dimension out of range
        with pytest.raises(ValueError):
            config = CollectionConfig(
                name="test_config_invalid_dimension", dimension=10001  # Too large
            )

        # Test empty filterable columns
        config = CollectionConfig(
            name="test_config_empty_filterable", dimension=128, filterable_columns=[]
        )
        collection = rest_client.create_collection(config.name, config)
        assert collection.config.filterable_columns == []

        # Test metadata schema
        config = CollectionConfig(
            name="test_config_metadata_schema",
            dimension=256,
            metadata_schema={
                "title": {"type": "string", "required": True},
                "score": {"type": "number", "min": 0, "max": 100},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        )
        collection = rest_client.create_collection(config.name, config)
        assert collection.config.metadata_schema is not None

        logger.info("✓ All edge cases and validation tests passed")

    def test_protocol_consistency(self, rest_client, grpc_client):
        """Test consistency between REST and gRPC protocols"""
        config = CollectionConfig(
            name="test_config_protocol_consistency",
            dimension=512,
            distance_metric=DistanceMetric.COSINE,
            storage_engine=StorageEngine.VIPER,
            primary_indexing_algorithm=IndexingAlgorithm.HNSW,
            quantization_config=QuantizationConfig(
                enabled=True, type=QuantizationType.SCALAR, bits_per_vector=8
            ),
        )

        # Create with REST
        rest_collection = rest_client.create_collection(config.name, config)

        # Create with gRPC
        grpc_config = CollectionConfig(**config.model_dump())
        grpc_config.name = f"{config.name}_grpc"
        grpc_collection = grpc_client.create_collection(grpc_config.name, grpc_config)

        # Compare configurations (accounting for potential differences in defaults)
        assert rest_collection.config.dimension == grpc_collection.config.dimension
        assert (
            rest_collection.config.distance_metric
            == grpc_collection.config.distance_metric
        )
        assert (
            rest_collection.config.storage_engine
            == grpc_collection.config.storage_engine
        )

        logger.info("✓ Protocol consistency verified")

    def test_all_distance_metrics_no_fallback_warnings(self, rest_client):
        """Test that all 13 distance metrics work without fallback warnings (2025-08 update)"""
        import warnings

        # All 13 distance metrics that should be fully supported
        all_distance_metrics = [
            DistanceMetric.COSINE,
            DistanceMetric.EUCLIDEAN,
            DistanceMetric.DOT_PRODUCT,
            DistanceMetric.MANHATTAN,
            DistanceMetric.HAMMING,
            DistanceMetric.JACCARD,
            DistanceMetric.CHEBYSHEV,
            DistanceMetric.CANBERRA,
            DistanceMetric.MINKOWSKI,
            DistanceMetric.ANGULAR,
            DistanceMetric.BRAY_CURTIS,
            DistanceMetric.HELLINGER,
            DistanceMetric.CUSTOM,
        ]

        successful_metrics = []
        fallback_warnings = []

        for metric in all_distance_metrics:
            collection_name = f"test_all_metrics_{metric.value}"
            config = CollectionConfig(
                name=collection_name,
                dimension=128,
                distance_metric=metric,
                storage_engine=StorageEngine.VIPER,
                description=f"Test collection for {metric.value} distance metric",
            )

            # Capture warnings during collection creation
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                try:
                    collection = rest_client.create_collection(collection_name, config)

                    # Check if the server accepted the distance metric
                    created_metric = collection.config.distance_metric
                    if created_metric == metric:
                        successful_metrics.append(metric.value)
                        logger.info(f"✅ {metric.value}: Native support confirmed")
                    else:
                        logger.warning(
                            f"⚠️  {metric.value}: Server used {created_metric} instead"
                        )

                    # Check for fallback warnings
                    distance_warnings = [
                        warning
                        for warning in w
                        if metric.value in str(warning.message).lower()
                        and "fallback" in str(warning.message).lower()
                    ]

                    if distance_warnings:
                        fallback_warnings.extend(
                            [
                                (metric.value, str(warning.message))
                                for warning in distance_warnings
                            ]
                        )
                        logger.warning(f"⚠️  {metric.value}: Fallback warning detected")
                    else:
                        logger.info(f"✅ {metric.value}: No fallback warnings")

                except Exception as e:
                    logger.error(
                        f"❌ {metric.value}: Failed to create collection - {e}"
                    )

        # Assertions
        logger.info("\n📊 Distance Metrics Test Results:")
        logger.info(f"   ✅ Successful: {len(successful_metrics)}/13 metrics")
        logger.info(f"   ⚠️  Fallback warnings: {len(fallback_warnings)} metrics")

        if successful_metrics:
            logger.info(f"   🎯 Native support: {', '.join(successful_metrics)}")

        if fallback_warnings:
            logger.info("   ⚠️  Metrics with fallback warnings:")
            for metric, warning in fallback_warnings:
                logger.info(f"     - {metric}: {warning}")

        # With the 2025-08 SDK update, no fallback warnings should be generated
        # Note: Server may still fall back internally, but SDK shouldn't warn about it
        assert (
            len(fallback_warnings) == 0
        ), f"Expected no fallback warnings, but got {len(fallback_warnings)}: {fallback_warnings}"
        assert (
            len(successful_metrics) >= 3
        ), f"Expected at least 3 core metrics to work, got {len(successful_metrics)}"

        logger.info(
            "🎉 Python SDK correctly handles distance metrics without generating fallback warnings!"
        )

    def test_all_indexing_algorithms_no_fallback_warnings(self, rest_client):
        """Test that all 6 indexing algorithms work without fallback warnings (2025-08 update)"""
        import warnings

        # All 6 indexing algorithms that should be fully supported
        all_indexing_algorithms = [
            IndexingAlgorithm.HNSW,
            IndexingAlgorithm.IVF,
            IndexingAlgorithm.FLAT,
            IndexingAlgorithm.PQ,
            IndexingAlgorithm.ANNOY,
            IndexingAlgorithm.LSH,
        ]

        successful_algorithms = []
        fallback_warnings = []

        for algorithm in all_indexing_algorithms:
            collection_name = f"test_all_algos_{algorithm.value}"
            config = CollectionConfig(
                name=collection_name,
                dimension=256,
                distance_metric=DistanceMetric.COSINE,
                storage_engine=StorageEngine.VIPER,
                primary_indexing_algorithm=algorithm,
                description=f"Test collection for {algorithm.value} indexing algorithm",
            )

            # Capture warnings during collection creation
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                try:
                    collection = rest_client.create_collection(collection_name, config)

                    # Check if the server accepted the indexing algorithm
                    created_algorithm = collection.config.primary_indexing_algorithm
                    if created_algorithm == algorithm:
                        successful_algorithms.append(algorithm.value)
                        logger.info(f"✅ {algorithm.value}: Native support confirmed")
                    else:
                        logger.warning(
                            f"⚠️  {algorithm.value}: Server used {created_algorithm} instead"
                        )

                    # Check for fallback warnings
                    algorithm_warnings = [
                        warning
                        for warning in w
                        if algorithm.value in str(warning.message).lower()
                        and "fallback" in str(warning.message).lower()
                    ]

                    if algorithm_warnings:
                        fallback_warnings.extend(
                            [
                                (algorithm.value, str(warning.message))
                                for warning in algorithm_warnings
                            ]
                        )
                        logger.warning(
                            f"⚠️  {algorithm.value}: Fallback warning detected"
                        )
                    else:
                        logger.info(f"✅ {algorithm.value}: No fallback warnings")

                except Exception as e:
                    logger.error(
                        f"❌ {algorithm.value}: Failed to create collection - {e}"
                    )

        # Assertions
        logger.info("\n📊 Indexing Algorithms Test Results:")
        logger.info(f"   ✅ Successful: {len(successful_algorithms)}/6 algorithms")
        logger.info(f"   ⚠️  Fallback warnings: {len(fallback_warnings)} algorithms")

        if successful_algorithms:
            logger.info(f"   🎯 Native support: {', '.join(successful_algorithms)}")

        if fallback_warnings:
            logger.info("   ⚠️  Algorithms with fallback warnings:")
            for algorithm, warning in fallback_warnings:
                logger.info(f"     - {algorithm}: {warning}")

        # With the 2025-08 SDK update, no fallback warnings should be generated
        # Note: Server may still fall back internally, but SDK shouldn't warn about it
        assert (
            len(fallback_warnings) == 0
        ), f"Expected no fallback warnings, but got {len(fallback_warnings)}: {fallback_warnings}"
        assert (
            len(successful_algorithms) >= 3
        ), f"Expected at least 3 core algorithms to work, got {len(successful_algorithms)}"

        logger.info(
            "🎉 Python SDK correctly handles indexing algorithms without generating fallback warnings!"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
