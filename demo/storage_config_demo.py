#!/usr/bin/env python3
"""
STATUS: 🚧 Future Feature - Advanced Storage Configuration
SDK Version: v1.1+ (requires StorageEngineConfig and related classes)
Server Version: v0.2.0+
Test Result: SKIP - Requires storage configuration API not yet in SDK

ProximaDB Storage Engine Configuration Demo

This script demonstrates how to use the new per-collection storage engine
configuration feature to optimize performance for different use cases.

NOTE: This demo requires StorageEngineConfig, ParquetWriterSettings, FooterCacheSettings,
and other advanced storage configuration classes that will be added in SDK v1.1+.
For current storage engine selection, see: demo/quickstart/basic_demo.py
"""

import sys
import time
import numpy as np
from typing import List, Dict, Any

# Add parent directory to path for local imports
sys.path.append('../clients/python/src')

from proximadb import ProximaDBClient
from proximadb.models import (
    CollectionConfig,
    StorageEngineConfig,
    ParquetWriterSettings,
    FooterCacheSettings,
    HybridWriterSettings,
    SstEngineSettings,
    AccessPattern,
    DataDensity,
    StorageEngine,
    DistanceMetric,
    CompressionAlgorithm
)


def create_cloud_optimized_collection(client: ProximaDBClient):
    """Create a collection optimized for cloud storage with S3/GCS/Azure"""
    print("\n=== Creating Cloud-Optimized Collection ===")
    
    config = CollectionConfig(
        name="cloud_vectors",
        dimension=768,
        distance_metric=DistanceMetric.COSINE,
        storage_engine=StorageEngine.NOVA,
        storage_engine_config=StorageEngineConfig(
            # Optimization hints
            access_pattern=AccessPattern.READ_HEAVY,
            data_density=DataDensity.DENSE,
            expected_size_gb=500,
            read_write_ratio=10.0,  # 10:1 read:write ratio
            
            # Use cloud-optimized preset
            preset="cloud_optimized",
            
            # Enable all optimizations
            enable_all_optimizations=True,
            
            # Custom Parquet settings for cloud
            parquet_writer=ParquetWriterSettings(
                row_group_size=50000,  # Large row groups for fewer files
                page_size=2097152,     # 2MB pages
                enable_bloom_filters=True,
                bloom_filter_fpp=0.001,  # 0.1% false positive rate
                enable_column_index=True,
                enable_offset_index=True,
                enable_pq_sorting=True,  # Better compression
                enable_native_metadata=True
            ),
            
            # Aggressive footer caching for cloud
            footer_cache=FooterCacheSettings(
                enable=True,
                max_entries=20000,
                ttl_seconds=7200,  # 2 hour TTL
                enable_prefetch=True,
                prefetch_threshold=5,
                enable_compression=True
            ),
            
            # Batch mode for cloud writes
            hybrid_writer=HybridWriterSettings(
                enable=True,
                initial_mode="batch",
                max_buffer_size=100000,
                buffer_time_limit_seconds=60
            )
        )
    )
    
    collection = client.create_collection(config=config)
    print(f"✅ Created cloud-optimized collection: {collection.id}")
    
    # Insert some sample data
    vectors = [
        {
            "id": f"cloud_vec_{i}",
            "vector": np.random.rand(768).tolist(),
            "metadata": {
                "source": "s3",
                "region": "us-west-2",
                "timestamp": int(time.time())
            }
        }
        for i in range(100)
    ]
    
    result = client.insert_vectors("cloud_vectors", vectors)
    print(f"✅ Inserted {len(vectors)} vectors")
    return collection


def create_real_time_collection(client: ProximaDBClient):
    """Create a collection optimized for real-time, low-latency operations"""
    print("\n=== Creating Real-Time Collection ===")
    
    config = CollectionConfig(
        name="realtime_vectors",
        dimension=256,
        distance_metric=DistanceMetric.EUCLIDEAN,
        storage_engine=StorageEngine.SST,  # SST for low latency
        storage_engine_config=StorageEngineConfig(
            # Optimization hints
            access_pattern=AccessPattern.WRITE_HEAVY,
            frequent_updates=True,
            read_write_ratio=0.5,  # 1:2 read:write ratio
            
            # Use real-time preset
            preset="real_time",
            
            # SST-specific settings for low latency
            sst_settings=SstEngineSettings(
                enable_bloom_filters=True,
                compression=CompressionAlgorithm.LZ4,  # Fast compression
                compression_level=1,  # Fastest
                write_buffer_size=33554432,  # 32MB buffer
                max_write_buffers=4,
                block_size_kb=1024,  # 1MB blocks
                dynamic_block_sizing=True
            )
        )
    )
    
    collection = client.create_collection(config=config)
    print(f"✅ Created real-time collection: {collection.id}")
    
    # Insert streaming data
    for batch in range(5):
        vectors = [
            {
                "id": f"rt_vec_{batch}_{i}",
                "vector": np.random.rand(256).tolist(),
                "metadata": {
                    "batch": batch,
                    "timestamp": time.time()
                }
            }
            for i in range(20)
        ]
        
        result = client.insert_vectors("realtime_vectors", vectors)
        print(f"  Batch {batch + 1}: Inserted {len(vectors)} vectors")
        time.sleep(0.1)  # Simulate streaming
    
    return collection


def create_archive_collection(client: ProximaDBClient):
    """Create a collection optimized for long-term archival storage"""
    print("\n=== Creating Archive Collection ===")
    
    config = CollectionConfig(
        name="archive_vectors",
        dimension=1024,
        distance_metric=DistanceMetric.DOT_PRODUCT,
        storage_engine=StorageEngine.NOVA,
        storage_engine_config=StorageEngineConfig(
            # Optimization hints
            access_pattern=AccessPattern.ARCHIVE,
            data_density=DataDensity.SPARSE,
            expected_size_gb=10000,  # 10TB expected
            frequent_updates=False,
            
            # Maximum compression for archival
            parquet_writer=ParquetWriterSettings(
                row_group_size=100000,  # Very large row groups
                enable_bloom_filters=False,  # Save space
                enable_pq_sorting=True,  # Maximum compression
                enable_dictionary=True,
                dictionary_threshold=0.9  # Aggressive dictionary encoding
            ),
            
            
            # Minimal footer cache
            footer_cache=FooterCacheSettings(
                enable=True,
                max_entries=100,  # Very small cache
                enable_prefetch=False,  # No prefetching
                enable_compression=True  # Compress cached data
            ),
            
            # Streaming mode for writes
            hybrid_writer=HybridWriterSettings(
                enable=True,
                initial_mode="streaming",
                max_buffer_size=1000,  # Small buffer
                enable_concurrent_writes=False  # Save memory
            )
        )
    )
    
    collection = client.create_collection(config=config)
    print(f"✅ Created memory-constrained collection: {collection.id}")
    return collection


def test_search_performance(client: ProximaDBClient, collection_name: str):
    """Test search performance on a collection"""
    print(f"\n  Testing search on {collection_name}...")
    
    # Get collection info
    collection = client.get_collection(collection_name)
    if not collection:
        print(f"  ❌ Collection {collection_name} not found")
        return
    
    # Create a random query vector
    query_vector = np.random.rand(collection.config.dimension).tolist()
    
    # Perform search
    start_time = time.time()
    results = client.search_vectors(
        collection_id=collection_name,
        query_vector=query_vector,
        top_k=10
    )
    search_time = (time.time() - start_time) * 1000
    
    print(f"  ✅ Search completed in {search_time:.2f}ms")
    print(f"     Found {len(results.get('results', []))} results")


def main():
    """Main demo function"""
    print("=" * 60)
    print("ProximaDB Storage Engine Configuration Demo")
    print("=" * 60)
    
    # Initialize client
    client = ProximaDBClient(
        url="http://localhost:5678",
        protocol="auto"
    )
    
    # Check server health
    health = client.health()
    print(f"\nServer Status: {health.status}")
    print(f"Server Version: {health.version}")
    
    try:
        # Create collections with different storage configurations
        collections = []
        
        # 1. Cloud-optimized collection
        cloud_collection = create_cloud_optimized_collection(client)
        collections.append(cloud_collection.config.name)
        
        # 2. Real-time collection
        rt_collection = create_real_time_collection(client)
        collections.append(rt_collection.config.name)
        
        # 3. Archive collection
        archive_collection = create_archive_collection(client)
        collections.append(archive_collection.config.name)
        
        # 4. Memory-constrained collection
        memory_collection = create_memory_constrained_collection(client)
        collections.append(memory_collection.config.name)
        
        # Test search performance on each collection
        print("\n" + "=" * 60)
        print("Search Performance Tests")
        print("=" * 60)
        
        for collection_name in collections:
            test_search_performance(client, collection_name)
        
        # List all collections
        print("\n" + "=" * 60)
        print("All Collections")
        print("=" * 60)
        
        all_collections = client.list_collections()
        for col in all_collections:
            if col.config.name in collections:
                storage_config = "Custom" if hasattr(col.config, 'storage_engine_config') and col.config.storage_engine_config else "Default"
                print(f"  - {col.config.name}: {col.config.dimension}D, {col.config.storage_engine}, {storage_config} config")
        
        # Cleanup (optional)
        print("\n" + "=" * 60)
        print("Cleanup")
        print("=" * 60)
        
        cleanup = input("\nDelete demo collections? (y/n): ").strip().lower()
        if cleanup == 'y':
            for collection_name in collections:
                try:
                    client.delete_collection(collection_name)
                    print(f"  ✅ Deleted {collection_name}")
                except Exception as e:
                    print(f"  ❌ Failed to delete {collection_name}: {e}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()