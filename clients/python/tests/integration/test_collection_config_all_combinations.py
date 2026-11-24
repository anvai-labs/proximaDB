#!/usr/bin/env python3
"""
Test all possible collection configuration combinations.
This test verifies what configurations are actually supported by the server.

NOTE: Moved from tests/unit/ to tests/integration/ - these are integration tests
requiring REST/gRPC client connections to a running server.
"""

import logging
from typing import Any

import pytest

from proximadb_sdk import (
    CollectionConfig,
    DistanceMetric,
    FilterableColumn,
    FilterableDataType,
    IndexingAlgorithm,
    ProximaDBError,
    QuantizationConfig,
    QuantizationType,
    StorageEngine,
    connect_grpc,
    connect_rest,
)

logger = logging.getLogger(__name__)


class TestAllCollectionConfigurations:
    """Test all collection configuration combinations"""

    @pytest.fixture
    def rest_client(self):
        """Create REST client"""
        client = connect_rest("http://localhost:5678")
        yield client
        self._cleanup_collections(client)

    @pytest.fixture
    def grpc_client(self):
        """Create gRPC client"""
        client = connect_grpc("grpc://localhost:5679")
        yield client
        self._cleanup_collections(client)

    def _cleanup_collections(self, client):
        """Clean up test collections"""
        try:
            collections = client.list_collections()
            for col in collections:
                if col.name.startswith("test_all_config_"):
                    try:
                        client.delete_collection(col.name)
                    except Exception:
                        pass
        except Exception:
            pass

    def _test_config(
        self, client, protocol: str, config: CollectionConfig
    ) -> dict[str, Any]:
        """Test a single configuration and return results"""
        result = {
            "protocol": protocol,
            "config": config.model_dump(),
            "success": False,
            "error": None,
            "actual_config": None,
        }

        try:
            collection = client.create_collection(config.name, config)
            result["success"] = True
            result["actual_config"] = (
                collection.config.model_dump() if collection.config else None
            )

            # Log successful configuration
            logger.info(f"✓ {protocol}: Created collection '{config.name}'")

            # Check if actual matches requested
            if collection.config:
                mismatches = []
                if (
                    config.distance_metric
                    and collection.config.distance_metric != config.distance_metric
                ):
                    mismatches.append(
                        f"distance_metric: requested={config.distance_metric.value}, actual={collection.config.distance_metric.value}"
                    )
                if (
                    config.storage_engine
                    and collection.config.storage_engine != config.storage_engine
                ):
                    mismatches.append(
                        f"storage_engine: requested={config.storage_engine.value}, actual={collection.config.storage_engine.value}"
                    )
                if (
                    config.primary_indexing_algorithm
                    and collection.config.primary_indexing_algorithm
                    != config.primary_indexing_algorithm
                ):
                    mismatches.append(
                        f"indexing_algorithm: requested={config.primary_indexing_algorithm.value}, actual={collection.config.primary_indexing_algorithm.value}"
                    )

                if mismatches:
                    result["mismatches"] = mismatches
                    logger.warning(f"  ⚠ Mismatches: {'; '.join(mismatches)}")

        except ProximaDBError as e:
            result["error"] = str(e)
            logger.error(f"✗ {protocol}: Failed to create '{config.name}': {e}")
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
            logger.error(f"✗ {protocol}: Unexpected error for '{config.name}': {e}")

        return result

    def test_all_distance_metrics(self, rest_client, grpc_client):
        """Test all distance metric combinations"""
        metrics = [
            DistanceMetric.COSINE,
            DistanceMetric.EUCLIDEAN,
            DistanceMetric.DOT_PRODUCT,
            DistanceMetric.MANHATTAN,
            DistanceMetric.HAMMING,
            DistanceMetric.JACCARD,
        ]

        results = {"rest": [], "grpc": []}

        for i, metric in enumerate(metrics):
            # Test REST
            config = CollectionConfig(
                name=f"test_all_config_metric_rest_{i}",
                dimension=128,
                distance_metric=metric,
            )
            results["rest"].append(self._test_config(rest_client, "REST", config))

            # Test gRPC
            config.name = f"test_all_config_metric_grpc_{i}"
            results["grpc"].append(self._test_config(grpc_client, "gRPC", config))

        # Summary
        self._print_summary("Distance Metrics", results)

    def test_all_storage_engines(self, rest_client, grpc_client):
        """Test all storage engine combinations"""
        engines = [
            StorageEngine.NOVA,
            StorageEngine.SST,
            StorageEngine.MMAP,
            StorageEngine.HYBRID,
        ]

        results = {"rest": [], "grpc": []}

        for i, engine in enumerate(engines):
            # Test REST
            config = CollectionConfig(
                name=f"test_all_config_engine_rest_{i}",
                dimension=256,
                storage_engine=engine,
            )
            results["rest"].append(self._test_config(rest_client, "REST", config))

            # Test gRPC
            config.name = f"test_all_config_engine_grpc_{i}"
            results["grpc"].append(self._test_config(grpc_client, "gRPC", config))

        # Summary
        self._print_summary("Storage Engines", results)

    def test_all_indexing_algorithms(self, rest_client, grpc_client):
        """Test all indexing algorithm combinations"""
        algorithms = [
            IndexingAlgorithm.HNSW,
            IndexingAlgorithm.IVF,
            IndexingAlgorithm.FLAT,
            IndexingAlgorithm.LSH,
            IndexingAlgorithm.ANNOY,
            IndexingAlgorithm.PQ,
        ]

        results = {"rest": [], "grpc": []}

        for i, algo in enumerate(algorithms):
            # Test REST
            config = CollectionConfig(
                name=f"test_all_config_algo_rest_{i}",
                dimension=384,
                primary_indexing_algorithm=algo,
            )
            results["rest"].append(self._test_config(rest_client, "REST", config))

            # Test gRPC
            config.name = f"test_all_config_algo_grpc_{i}"
            results["grpc"].append(self._test_config(grpc_client, "gRPC", config))

        # Summary
        self._print_summary("Indexing Algorithms", results)

    def test_all_quantization_types(self, rest_client, grpc_client):
        """Test all quantization type combinations"""
        quant_configs = [
            ("none", QuantizationConfig(enabled=False, type=QuantizationType.NONE)),
            (
                "scalar_8bit",
                QuantizationConfig(
                    enabled=True,
                    type=QuantizationType.SCALAR,
                    bits_per_vector=8,
                    accuracy_threshold=0.95,
                ),
            ),
            (
                "scalar_16bit",
                QuantizationConfig(
                    enabled=True,
                    type=QuantizationType.SCALAR,
                    bits_per_vector=16,
                    accuracy_threshold=0.98,
                ),
            ),
            (
                "product_4bit",
                QuantizationConfig(
                    enabled=True,
                    type=QuantizationType.PRODUCT,
                    num_subvectors=8,
                    bits_per_subvector=4,
                    accuracy_threshold=0.90,
                ),
            ),
            (
                "product_8bit",
                QuantizationConfig(
                    enabled=True,
                    type=QuantizationType.PRODUCT,
                    num_subvectors=16,
                    bits_per_subvector=8,
                    accuracy_threshold=0.95,
                ),
            ),
            (
                "binary",
                QuantizationConfig(
                    enabled=True,
                    type=QuantizationType.BINARY,
                    threshold=0.5,
                    accuracy_threshold=0.85,
                ),
            ),
            (
                "uniform",
                QuantizationConfig(
                    enabled=True,
                    type=QuantizationType.UNIFORM,
                    bits_per_vector=16,
                    compression_ratio_target=4.0,
                ),
            ),
        ]

        results = {"rest": [], "grpc": []}

        for i, (name, quant_config) in enumerate(quant_configs):
            # Test REST
            config = CollectionConfig(
                name=f"test_all_config_quant_{name}_rest",
                dimension=512,
                quantization_config=quant_config,
            )
            results["rest"].append(self._test_config(rest_client, "REST", config))

            # Test gRPC
            config.name = f"test_all_config_quant_{name}_grpc"
            results["grpc"].append(self._test_config(grpc_client, "gRPC", config))

        # Summary
        self._print_summary("Quantization Types", results)

    def test_filterable_columns(self, rest_client, grpc_client):
        """Test filterable column configurations"""
        column_configs = [
            ("none", None),
            (
                "single_string",
                [
                    FilterableColumn(
                        name="category", data_type=FilterableDataType.STRING
                    )
                ],
            ),
            (
                "multiple_types",
                [
                    FilterableColumn(
                        name="category", data_type=FilterableDataType.STRING
                    ),
                    FilterableColumn(name="price", data_type=FilterableDataType.FLOAT),
                    FilterableColumn(
                        name="count", data_type=FilterableDataType.INTEGER
                    ),
                    FilterableColumn(
                        name="active", data_type=FilterableDataType.BOOLEAN
                    ),
                ],
            ),
            (
                "indexed",
                [
                    FilterableColumn(
                        name="user_id",
                        data_type=FilterableDataType.STRING,
                        indexed=True,
                    ),
                    FilterableColumn(
                        name="timestamp",
                        data_type=FilterableDataType.INTEGER,
                        indexed=True,
                    ),
                ],
            ),
            (
                "arrays",
                [
                    FilterableColumn(
                        name="tags", data_type=FilterableDataType.ARRAY_STRING
                    ),
                    FilterableColumn(
                        name="scores", data_type=FilterableDataType.ARRAY_FLOAT
                    ),
                ],
            ),
        ]

        results = {"rest": [], "grpc": []}

        for i, (name, columns) in enumerate(column_configs):
            # Test REST
            config = CollectionConfig(
                name=f"test_all_config_cols_{name}_rest",
                dimension=128,
                filterable_columns=columns,
            )
            results["rest"].append(self._test_config(rest_client, "REST", config))

            # Test gRPC
            config.name = f"test_all_config_cols_{name}_grpc"
            results["grpc"].append(self._test_config(grpc_client, "gRPC", config))

        # Summary
        self._print_summary("Filterable Columns", results)

    def test_comprehensive_combinations(self, rest_client, grpc_client):
        """Test comprehensive configuration combinations"""
        combinations = [
            # Basic combination
            {
                "name": "test_all_config_basic",
                "dimension": 128,
                "distance_metric": DistanceMetric.COSINE,
                "storage_engine": StorageEngine.NOVA,
                "primary_indexing_algorithm": IndexingAlgorithm.HNSW,
            },
            # With quantization
            {
                "name": "test_all_config_with_quant",
                "dimension": 384,
                "distance_metric": DistanceMetric.EUCLIDEAN,
                "storage_engine": StorageEngine.SST,
                "primary_indexing_algorithm": IndexingAlgorithm.IVF,
                "quantization_config": QuantizationConfig(
                    enabled=True, type=QuantizationType.SCALAR, bits_per_vector=8
                ),
            },
            # With filterable columns
            {
                "name": "test_all_config_with_filters",
                "dimension": 256,
                "distance_metric": DistanceMetric.DOT_PRODUCT,
                "storage_engine": StorageEngine.NOVA,
                "primary_indexing_algorithm": IndexingAlgorithm.FLAT,
                "filterable_columns": [
                    FilterableColumn(name="type", data_type=FilterableDataType.STRING),
                    FilterableColumn(name="score", data_type=FilterableDataType.FLOAT),
                ],
            },
            # Full featured
            {
                "name": "test_all_config_full",
                "dimension": 768,
                "distance_metric": DistanceMetric.COSINE,
                "storage_engine": StorageEngine.NOVA,
                "primary_indexing_algorithm": IndexingAlgorithm.HNSW,
                "quantization_config": QuantizationConfig(
                    enabled=True,
                    type=QuantizationType.PRODUCT,
                    num_subvectors=24,
                    bits_per_subvector=8,
                    progressive_quantization=True,
                ),
                "filterable_columns": [
                    FilterableColumn(
                        name="doc_type",
                        data_type=FilterableDataType.STRING,
                        indexed=True,
                    ),
                    FilterableColumn(
                        name="timestamp",
                        data_type=FilterableDataType.INTEGER,
                        indexed=True,
                    ),
                    FilterableColumn(
                        name="tags", data_type=FilterableDataType.ARRAY_STRING
                    ),
                ],
                "description": "Full featured test collection",
                "tags": ["test", "comprehensive"],
            },
        ]

        results = {"rest": [], "grpc": []}

        for combo in combinations:
            # Test REST
            config = CollectionConfig(**combo)
            config.name = f"{combo['name']}_rest"
            results["rest"].append(self._test_config(rest_client, "REST", config))

            # Test gRPC
            config.name = f"{combo['name']}_grpc"
            results["grpc"].append(self._test_config(grpc_client, "gRPC", config))

        # Summary
        self._print_summary("Comprehensive Combinations", results)

    def _print_summary(self, test_name: str, results: dict[str, list[dict]]):
        """Print test summary"""
        print(f"\n{'='*60}")
        print(f"{test_name} Summary")
        print(f"{'='*60}")

        for protocol in ["rest", "grpc"]:
            protocol_results = results[protocol]
            total = len(protocol_results)
            successful = sum(1 for r in protocol_results if r["success"])
            failed = total - successful

            print(f"\n{protocol.upper()}:")
            print(f"  Total: {total}")
            print(f"  ✓ Successful: {successful}")
            print(f"  ✗ Failed: {failed}")

            if failed > 0:
                print("\n  Failed configurations:")
                for r in protocol_results:
                    if not r["success"]:
                        config_desc = self._get_config_description(r["config"])
                        print(f"    - {config_desc}: {r['error']}")

            # Check for mismatches
            mismatched = [r for r in protocol_results if r.get("mismatches")]
            if mismatched:
                print("\n  Configuration mismatches:")
                for r in mismatched:
                    config_desc = self._get_config_description(r["config"])
                    print(f"    - {config_desc}:")
                    for mismatch in r["mismatches"]:
                        print(f"      • {mismatch}")

    def _get_config_description(self, config: dict) -> str:
        """Get a concise description of a configuration"""
        parts = []
        if "distance_metric" in config and config["distance_metric"]:
            parts.append(f"metric={config['distance_metric']}")
        if "storage_engine" in config and config["storage_engine"]:
            parts.append(f"engine={config['storage_engine']}")
        if (
            "primary_indexing_algorithm" in config
            and config["primary_indexing_algorithm"]
        ):
            parts.append(f"index={config['primary_indexing_algorithm']}")
        if "quantization_config" in config and config["quantization_config"]:
            quant = config["quantization_config"]
            if quant.get("enabled"):
                parts.append(f"quant={quant.get('type', 'unknown')}")
        if "filterable_columns" in config and config["filterable_columns"]:
            parts.append(f"filters={len(config['filterable_columns'])}")

        return ", ".join(parts) if parts else config.get("name", "unknown")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
