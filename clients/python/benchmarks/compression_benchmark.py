#!/usr/bin/env python3
"""
ProximaDB Compression Benchmark Suite

Measures performance characteristics of different compression configurations.

Copyright 2025 ProximaDB
"""

import asyncio
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

from proximadb import ProximaDBClient
from proximadb.models import (
    CollectionConfig,
    CompressionConfig,
    CompressionAlgorithm,
    DistanceMetric,
    StorageEngine,
    VectorRecord,
    SearchOptimization,
)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""

    config_name: str
    compression_algorithm: str
    compression_level: int
    storage_engine: str

    # Ingestion metrics
    vectors_inserted: int
    ingestion_time_s: float
    ingestion_rate_vec_s: float

    # Storage metrics
    storage_size_mb: float
    compression_ratio: float

    # Search metrics
    search_time_ms: float
    search_with_cache_ms: float
    cache_speedup: float

    # Memory metrics
    peak_memory_mb: float
    cache_hit_rate: float


class CompressionBenchmark:
    """Compression benchmark suite"""

    def __init__(self, client: ProximaDBClient, dimension: int = 1536):
        self.client = client
        self.dimension = dimension
        self.results: List[BenchmarkResult] = []

    def generate_vectors(
        self, num_vectors: int, sparsity: float = 0.0
    ) -> List[VectorRecord]:
        """Generate test vectors with configurable sparsity"""
        vectors = []

        for i in range(num_vectors):
            if sparsity > 0:
                # Generate sparse vector
                vec = np.zeros(self.dimension)
                non_zero = int(self.dimension * (1 - sparsity))
                indices = np.random.choice(self.dimension, non_zero, replace=False)
                vec[indices] = np.random.randn(non_zero)
            else:
                # Generate dense vector
                vec = np.random.randn(self.dimension)

            vectors.append(
                VectorRecord(
                    id=f"vec_{i:06d}",
                    vector=vec.tolist(),
                    metadata={
                        "timestamp": int(time.time()),
                        "category": f"cat_{i % 100}",
                        "sparsity": sparsity,
                    },
                )
            )

        return vectors

    async def benchmark_configuration(
        self,
        config_name: str,
        compression_config: CompressionConfig,
        storage_engine: StorageEngine,
        num_vectors: int = 10000,
        sparsity: float = 0.0,
    ) -> BenchmarkResult:
        """Benchmark a single compression configuration"""

        print(f"\n{'='*60}")
        print(f"Benchmarking: {config_name}")
        print(f"{'='*60}")

        # Create collection
        collection_name = f"bench_{config_name}_{int(time.time())}"
        collection_config = CollectionConfig(
            name=collection_name,
            dimension=self.dimension,
            distance_metric=DistanceMetric.COSINE,
            storage_engine=storage_engine,
            compression_config=compression_config,
        )

        print(f"Creating collection: {collection_name}")
        collection = await self.client.create_collection(collection_config)

        try:
            # Generate test data
            print(f"Generating {num_vectors} vectors (sparsity={sparsity})...")
            vectors = self.generate_vectors(num_vectors, sparsity)

            # Benchmark ingestion
            print("Benchmarking ingestion...")
            start_time = time.time()
            await self.client.insert_vectors(collection_name, vectors)
            ingestion_time = time.time() - start_time
            ingestion_rate = num_vectors / ingestion_time

            print(f"  Inserted {num_vectors} vectors in {ingestion_time:.2f}s")
            print(f"  Rate: {ingestion_rate:.0f} vectors/second")

            # Get storage metrics
            await asyncio.sleep(2)  # Wait for flush
            metrics = await self.client.get_collection_metrics(collection_name)
            storage_size_mb = metrics.get("storage_bytes", 0) / (1024 * 1024)
            original_size_mb = num_vectors * self.dimension * 4 / (1024 * 1024)  # FP32
            compression_ratio = original_size_mb / max(storage_size_mb, 0.001)

            print(
                f"  Storage: {storage_size_mb:.1f}MB (ratio: {compression_ratio:.2f}x)"
            )

            # Benchmark search (cold cache)
            print("Benchmarking search...")
            query_vector = np.random.randn(self.dimension).tolist()

            start_time = time.time()
            results = await self.client.search_vectors(
                collection_id=collection_name,
                query_vector=query_vector,
                top_k=100,
            )
            cold_search_time = (time.time() - start_time) * 1000  # ms

            print(f"  Cold search: {cold_search_time:.2f}ms")

            # Benchmark search with cache hints
            optimization = SearchOptimization(
                use_decompression_cache=True,
                prefer_compressed_search=True,
                compression_aware_routing=True,
            )

            # Warm up cache
            for _ in range(3):
                await self.client.search_vectors(
                    collection_id=collection_name,
                    query_vector=query_vector,
                    top_k=100,
                    search_optimization=optimization,
                )

            # Measure cached search
            start_time = time.time()
            results = await self.client.search_vectors(
                collection_id=collection_name,
                query_vector=query_vector,
                top_k=100,
                search_optimization=optimization,
            )
            cached_search_time = (time.time() - start_time) * 1000  # ms

            print(f"  Cached search: {cached_search_time:.2f}ms")

            cache_speedup = cold_search_time / max(cached_search_time, 0.001)
            print(f"  Cache speedup: {cache_speedup:.2f}x")

            # Get cache statistics
            cache_stats = await self.client.get_cache_stats()
            cache_hit_rate = cache_stats.get("hit_rate", 0.0)

            # Create result
            result = BenchmarkResult(
                config_name=config_name,
                compression_algorithm=compression_config.sst_compression_algorithm
                or "none",
                compression_level=compression_config.sst_compression_level or 0,
                storage_engine=storage_engine,
                vectors_inserted=num_vectors,
                ingestion_time_s=ingestion_time,
                ingestion_rate_vec_s=ingestion_rate,
                storage_size_mb=storage_size_mb,
                compression_ratio=compression_ratio,
                search_time_ms=cold_search_time,
                search_with_cache_ms=cached_search_time,
                cache_speedup=cache_speedup,
                peak_memory_mb=0,  # Would need system monitoring
                cache_hit_rate=cache_hit_rate,
            )

            return result

        finally:
            # Cleanup
            print(f"Cleaning up collection: {collection_name}")
            await self.client.delete_collection(collection_name)

    async def run_benchmark_suite(self, num_vectors: int = 10000):
        """Run complete benchmark suite"""

        print("\n" + "=" * 80)
        print("ProximaDB Compression Benchmark Suite")
        print("=" * 80)
        print(f"Dimension: {self.dimension}")
        print(f"Vectors: {num_vectors}")
        print("=" * 80)

        # Define test configurations
        configs = [
            # Baseline (no compression)
            ("baseline_sst", CompressionConfig(), StorageEngine.SST, 0.0),
            # SST with different algorithms
            (
                "sst_zstd_fast",
                CompressionConfig(
                    sst_compression_algorithm=CompressionAlgorithm.ZSTD,
                    sst_compression_level=3,
                    sst_block_size=16384,
                ),
                StorageEngine.SST,
                0.0,
            ),
            (
                "sst_zstd_balanced",
                CompressionConfig(
                    sst_compression_algorithm=CompressionAlgorithm.ZSTD,
                    sst_compression_level=6,
                    sst_block_size=32768,
                ),
                StorageEngine.SST,
                0.0,
            ),
            (
                "sst_zstd_high",
                CompressionConfig(
                    sst_compression_algorithm=CompressionAlgorithm.ZSTD,
                    sst_compression_level=9,
                    sst_block_size=65536,
                ),
                StorageEngine.SST,
                0.0,
            ),
            (
                "sst_lz4",
                CompressionConfig(
                    sst_compression_algorithm=CompressionAlgorithm.LZ4,
                    sst_compression_level=1,
                    sst_block_size=32768,
                ),
                StorageEngine.SST,
                0.0,
            ),
            (
                "sst_snappy",
                CompressionConfig(
                    sst_compression_algorithm=CompressionAlgorithm.SNAPPY,
                    sst_block_size=32768,
                ),
                StorageEngine.SST,
                0.0,
            ),
            # Test with sparse data (should compress better)
            (
                "sst_zstd_sparse",
                CompressionConfig(
                    sst_compression_algorithm=CompressionAlgorithm.ZSTD,
                    sst_compression_level=6,
                    sst_block_size=32768,
                ),
                StorageEngine.SST,
                0.8,
            ),  # 80% sparse
        ]

        # Run benchmarks
        for config_name, compression_config, storage_engine, sparsity in configs:
            try:
                result = await self.benchmark_configuration(
                    config_name=config_name,
                    compression_config=compression_config,
                    storage_engine=storage_engine,
                    num_vectors=num_vectors,
                    sparsity=sparsity,
                )
                self.results.append(result)
            except Exception as e:
                print(f"Error benchmarking {config_name}: {e}")

        # Generate report
        self.generate_report()

    def generate_report(self):
        """Generate benchmark report"""

        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)

        # Convert results to DataFrame
        df = pd.DataFrame(
            [
                {
                    "Configuration": r.config_name,
                    "Algorithm": r.compression_algorithm,
                    "Level": r.compression_level,
                    "Engine": r.storage_engine,
                    "Ingestion (vec/s)": f"{r.ingestion_rate_vec_s:.0f}",
                    "Storage (MB)": f"{r.storage_size_mb:.1f}",
                    "Compression": f"{r.compression_ratio:.2f}x",
                    "Search (ms)": f"{r.search_time_ms:.2f}",
                    "Cached (ms)": f"{r.search_with_cache_ms:.2f}",
                    "Speedup": f"{r.cache_speedup:.2f}x",
                }
                for r in self.results
            ]
        )

        print(df.to_string(index=False))

        # Summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        if self.results:
            baseline = next(
                (r for r in self.results if r.config_name == "baseline_sst"), None
            )
            if baseline:
                print(f"\nCompression Impact vs Baseline:")
                for r in self.results:
                    if r.config_name != "baseline_sst":
                        storage_reduction = (
                            1 - r.storage_size_mb / baseline.storage_size_mb
                        ) * 100
                        search_overhead = (
                            r.search_time_ms / baseline.search_time_ms - 1
                        ) * 100
                        print(f"  {r.config_name}:")
                        print(f"    Storage reduction: {storage_reduction:.1f}%")
                        print(f"    Search overhead: {search_overhead:+.1f}%")

        # Save results to JSON
        output_file = f"compression_benchmark_{int(time.time())}.json"
        with open(output_file, "w") as f:
            json.dump(
                [r.__dict__ for r in self.results],
                f,
                indent=2,
                default=str,
            )
        print(f"\nResults saved to: {output_file}")


async def main():
    """Main benchmark function"""

    # Initialize client
    client = ProximaDBClient(
        url="http://localhost:5678",
        grpc_url="http://localhost:5679",
    )

    # Run benchmarks
    benchmark = CompressionBenchmark(client, dimension=1536)

    # Quick benchmark (1K vectors)
    # await benchmark.run_benchmark_suite(num_vectors=1000)

    # Full benchmark (10K vectors)
    await benchmark.run_benchmark_suite(num_vectors=10000)


if __name__ == "__main__":
    asyncio.run(main())
