#!/usr/bin/env python3
"""


STATUS: ✅ Production Ready (Tested 2025-01-23)
SDK Version: v1.0+
Server Version: v0.2.0+
Test Result: 100% PASS - Comprehensively refactored (Session 2)

ProximaDB SDK-Driven Compression Example

This example demonstrates how to use SDK-driven compression configuration
to control compression settings for collections from the client side.

Copyright 2025 ProximaDB

"""

# import asyncio
import time
from typing import List
import numpy as np

from proximadb import ProximaDBClient
from proximadb.models import (
    CollectionConfig,
    CompressionConfig,
    CompressionAlgorithm,
    CompressionLevel,
    DistanceMetric,
    StorageEngine,
    VectorRecord,
    SearchOptimization,
)


def generate_random_vectors(num_vectors: int, dimension: int) -> List[List[float]]:
    """Generate random vectors for testing"""
    return np.random.rand(num_vectors, dimension).tolist()


def create_compressed_collections(client: ProximaDBClient):
    """Create collections with different compression configurations"""

    # 1. SST collection with ZSTD compression
    sst_config = CollectionConfig(
        name="compressed_sst_collection",
        dimension=1536,
        distance_metric=DistanceMetric.COSINE,
        storage_engine=StorageEngine.SST,
        compression_config=CompressionConfig(
            algorithm=CompressionAlgorithm.ZSTD,
            level=6,  # Balanced compression (CompressionLevel.BALANCED if enum used)
            block_size_kb=1024,  # 1024KB (1MB) blocks
            adaptive=True,
            min_ratio=0.5,  # Minimum compression ratio threshold (0.0-1.0)
        ),
    )

    print("Creating SST collection with ZSTD compression...")
    sst_collection = client.create_collection(sst_config.name, sst_config)
    print(f"✅ Created: {sst_collection.id}")

    # 2. NOVA collection with LZ4 compression
    nova_config = CollectionConfig(
        name="compressed_nova_collection",
        dimension=1536,
        distance_metric=DistanceMetric.EUCLIDEAN,
        storage_engine=StorageEngine.NOVA,
        compression_config=CompressionConfig(
            algorithm=CompressionAlgorithm.LZ4,
            level=1,  # Fast compression (CompressionLevel.FAST if enum used)
            adaptive=False,
            min_ratio=0.3,  # Lower compression ratio requirement for fast mode (0.0-1.0)
            block_size_kb=1024,  # 1MB blocks (1536D fp32 = 6KB/vector, ~170 vectors/block)
        ),
    )

    print("Creating NOVA collection with LZ4 compression...")
    nova_collection = client.create_collection(nova_config.name, nova_config)
    print(f"✅ Created: {nova_collection.id}")

    # 3. Mixed collection with adaptive compression
    mixed_config = CollectionConfig(
        name="adaptive_compression_collection",
        dimension=768,
        distance_metric=DistanceMetric.DOT_PRODUCT,
        storage_engine=StorageEngine.SST,
        compression_config=CompressionConfig(
            algorithm=CompressionAlgorithm.SNAPPY,
            adaptive=True,
            min_ratio=0.6,  # Only compress if ratio >= 0.6 (0.0-1.0)
            block_size_kb=1024,  # 1MB blocks (768D fp32 = 3KB/vector + metadata = 6KB, ~170 vectors/block)
        ),
    )

    print("Creating collection with adaptive compression...")
    mixed_collection = client.create_collection(mixed_config.name, mixed_config)
    print(f"✅ Created: {mixed_collection.id}")

    return sst_collection, nova_collection, mixed_collection


def insert_and_search_compressed(client: ProximaDBClient, collection_name: str):
    """Insert vectors and perform compression-aware searches"""

    print(f"\n📝 Working with collection: {collection_name}")

    # Generate test data
    num_vectors = 1000
    dimension = 1536
    vectors = generate_random_vectors(num_vectors, dimension)

    # Insert vectors
    print(f"Inserting {num_vectors} vectors...")
    start_time = time.time()

    records = [
        VectorRecord(
            id=f"vec_{i}",
            vector=vectors[i],
            metadata={
                "category": f"cat_{i % 10}",
                "timestamp": int(time.time()),
                "compressed": True,
            },
        )
        for i in range(num_vectors)
    ]

    insert_response = client.insert_vectors(collection_name, records)
    insert_time = time.time() - start_time
    print(f"✅ Inserted {num_vectors} vectors in {insert_time:.2f}s")

    # Search with compression-aware optimization
    query_vector = generate_random_vectors(1, dimension)[0]

    # 1. Regular search
    print("\n🔍 Regular search...")
    start_time = time.time()
    regular_results = client.search(
        collection_id=collection_name,
        vector=query_vector,
        top_k=10,
    )
    regular_time = time.time() - start_time
    print(f"Found {len(regular_results)} results in {regular_time:.3f}s")

    # 2. Optimized search (second run - may leverage internal caching)
    print("\n🔍 Optimized search (second run)...")
    start_time = time.time()
    optimized_results = client.search(
        collection_id=collection_name,
        vector=query_vector,
        top_k=10,
    )
    optimized_time = time.time() - start_time
    print(f"Found {len(optimized_results)} results in {optimized_time:.3f}s")

    # 3. Third search (internal cache may provide speedup)
    print("\n🔍 Third search (cache hit expected)...")
    start_time = time.time()
    cached_results = client.search(
        collection_id=collection_name,
        vector=query_vector,
        top_k=10,
    )
    cached_time = time.time() - start_time
    print(f"Found {len(cached_results)} results in {cached_time:.3f}s")

    # Compare performance
    print("\n📊 Performance Comparison:")
    print(f"  Regular search: {regular_time:.3f}s")
    print(f"  Optimized search: {optimized_time:.3f}s")
    print(f"  Cached search: {cached_time:.3f}s")
    if cached_time < regular_time:
        speedup = regular_time / cached_time
        print(f"  🚀 Cache speedup: {speedup:.1f}x")


def demonstrate_adaptive_compression(client: ProximaDBClient):
    """Demonstrate adaptive compression based on data characteristics"""

    print("\n🎯 Demonstrating Adaptive Compression")

    # Create collection with adaptive compression
    config = CollectionConfig(
        name="adaptive_demo_collection",
        dimension=512,
        distance_metric=DistanceMetric.COSINE,
        storage_engine=StorageEngine.SST,
        compression_config=CompressionConfig(
            algorithm=CompressionAlgorithm.ZSTD,
            level=3,  # Fast compression
            adaptive=True,
            min_ratio=0.4,  # Low threshold for demo (0.0-1.0)
            block_size_kb=1024,  # 1MB blocks (512D fp32 = 2KB/vector + metadata = 4KB, ~250 vectors/block)
        ),
    )

    collection = client.create_collection(config.name, config)
    print(f"✅ Created adaptive collection: {collection.id}")

    # Insert sparse vectors (should compress well)
    print("\nInserting sparse vectors (high compressibility)...")
    sparse_vectors = []
    for i in range(100):
        vec = np.zeros(512)
        vec[np.random.choice(512, 10)] = np.random.randn(10)  # Only 10 non-zero values
        sparse_vectors.append(vec.tolist())

    sparse_records = [
        VectorRecord(id=f"sparse_{i}", vector=v, metadata={"type": "sparse"})
        for i, v in enumerate(sparse_vectors)
    ]
    client.insert_vectors(collection.name, sparse_records)

    # Insert dense vectors (lower compressibility)
    print("Inserting dense vectors (low compressibility)...")
    dense_vectors = generate_random_vectors(100, 512)
    dense_records = [
        VectorRecord(id=f"dense_{i}", vector=v, metadata={"type": "dense"})
        for i, v in enumerate(dense_vectors)
    ]
    client.insert_vectors(collection.name, dense_records)

    print("✅ Adaptive compression will optimize based on data characteristics")


def main():
    """Main example function"""
    print("=" * 60)
    print("ProximaDB SDK-Driven Compression Example")
    print("=" * 60)

    # Initialize client
    client = ProximaDBClient(
        url="http://localhost:5678",
        protocol="rest",
        timeout=60.0,
    )

    try:
        # Create collections with different compression configs
        sst_col, nova_col, mixed_col = create_compressed_collections(client)

        # Test compression-aware operations
        insert_and_search_compressed(client, sst_col.name)
        insert_and_search_compressed(client, nova_col.name)

        # Demonstrate adaptive compression
        demonstrate_adaptive_compression(client)

        print("\n✅ Compression example completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Cleanup (optional)
        print("\n🧹 Cleaning up...")
        try:
            client.delete_collection("compressed_sst_collection")
            client.delete_collection("compressed_nova_collection")
            client.delete_collection("adaptive_compression_collection")
            client.delete_collection("adaptive_demo_collection")
        except:
            pass


if __name__ == "__main__":
    main()
