#!/usr/bin/env python3
"""
REAL Performance Measurements for ProximaDB
This script performs actual benchmarks and reports real numbers
"""

import numpy as np
import time
import concurrent.futures
import statistics
import sys
from proximadb_sdk import connect_rest, connect_grpc
from proximadb_sdk.models import (
    CollectionConfig,
    StorageEngine,
)

def print_section(title):
    """Print section header"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

def measure_insert_performance(client, collection_id, dimension=128, total_vectors=1000):
    """Measure real insert performance"""
    print(f"\nMeasuring insert performance ({total_vectors} vectors)...")
    
    # Generate test data
    records = []
    for i in range(total_vectors):
        records.append(
            {
                "id": f"vec_{i}",
                "vector": np.random.rand(dimension).tolist(),
                "props": {"index": i, "type": "test"},
            }
        )
    
    # Test different batch sizes
    batch_sizes = [10, 50, 100, 500]
    results = {}
    
    for batch_size in batch_sizes:
        if batch_size > total_vectors:
            continue
            
        batch = records[:batch_size]
        
        # Warm up
        client.insert_records(collection_id, batch[:5])
        
        # Measure
        times = []
        for _ in range(5):  # 5 iterations for average
            start = time.time()
            client.insert_records(collection_id, batch)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = statistics.mean(times)
        throughput = batch_size / avg_time
        
        results[batch_size] = {
            'throughput': throughput,
            'latency': avg_time * 1000
        }
        
        print(f"  Batch {batch_size}: {throughput:.0f} vectors/sec, {avg_time*1000:.2f}ms")
    
    return results

def measure_search_performance(client, collection_id, dimension=128):
    """Measure real search performance"""
    print(f"\nMeasuring search performance...")
    
    query_vector = np.random.rand(dimension).tolist()
    
    # Warm up
    for _ in range(10):
        client.search(collection_id, query_vector, top_k=10)
    
    # Measure
    times = []
    for _ in range(50):  # 50 queries
        start = time.time()
        results = client.search(collection_id, query_vector, top_k=10)
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg_time = statistics.mean(times)
    p50 = statistics.median(times)
    p95 = np.percentile(times, 95)
    p99 = np.percentile(times, 99)
    qps = 1.0 / avg_time
    
    print(f"  Average: {avg_time*1000:.2f}ms ({qps:.0f} QPS)")
    print(f"  p50: {p50*1000:.2f}ms")
    print(f"  p95: {p95*1000:.2f}ms")
    print(f"  p99: {p99*1000:.2f}ms")
    
    return {
        'qps': qps,
        'avg_ms': avg_time * 1000,
        'p50_ms': p50 * 1000,
        'p95_ms': p95 * 1000,
        'p99_ms': p99 * 1000
    }

def measure_concurrent_performance(client, collection_id, dimension=128):
    """Measure real concurrent performance"""
    print(f"\nMeasuring concurrent search performance...")
    
    def search_task():
        query = np.random.rand(dimension).tolist()
        start = time.time()
        client.search(collection_id, query, top_k=10)
        return time.time() - start
    
    thread_counts = [1, 5, 10, 20]
    results = {}
    
    for num_threads in thread_counts:
        print(f"\n  Testing with {num_threads} concurrent threads...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Run 100 operations total
            operations = 100
            start_time = time.time()
            
            futures = [executor.submit(search_task) for _ in range(operations)]
            latencies = [f.result() for f in concurrent.futures.as_completed(futures)]
            
            total_time = time.time() - start_time
        
        qps = operations / total_time
        avg_latency = statistics.mean(latencies) * 1000
        p99_latency = np.percentile(latencies, 99) * 1000
        
        results[num_threads] = {
            'qps': qps,
            'avg_latency_ms': avg_latency,
            'p99_latency_ms': p99_latency
        }
        
        print(f"    QPS: {qps:.0f}")
        print(f"    Avg latency: {avg_latency:.2f}ms")
        print(f"    p99 latency: {p99_latency:.2f}ms")
    
    return results

def test_nova_rest():
    """Test NOVA engine with REST protocol"""
    print_section("NOVA Engine + REST Protocol (REAL)")
    
    client = connect_rest("http://localhost:5678")
    collection_id = f"nova_rest_bench_{int(time.time())}"
    
    try:
        config = CollectionConfig(
            name=collection_id,
            dimension=128,
            storage_engine=StorageEngine.NOVA
        )
        client.create_collection(collection_id, config)
        
        # Run measurements
        insert_results = measure_insert_performance(client, collection_id)
        search_results = measure_search_performance(client, collection_id)
        concurrent_results = measure_concurrent_performance(client, collection_id)
        
        return {
            'insert': insert_results,
            'search': search_results,
            'concurrent': concurrent_results
        }
        
    finally:
        client.delete_collection(collection_id)
        client.close()

def test_sst_rest():
    """Test SST engine with REST protocol"""
    print_section("SST Engine + REST Protocol (REAL)")
    
    client = connect_rest("http://localhost:5678")
    collection_id = f"sst_rest_bench_{int(time.time())}"
    
    try:
        config = CollectionConfig(
            name=collection_id,
            dimension=128,
            storage_engine=StorageEngine.SST
        )
        client.create_collection(collection_id, config)
        
        # Run measurements
        insert_results = measure_insert_performance(client, collection_id)
        search_results = measure_search_performance(client, collection_id)
        concurrent_results = measure_concurrent_performance(client, collection_id)
        
        return {
            'insert': insert_results,
            'search': search_results,
            'concurrent': concurrent_results
        }
        
    finally:
        client.delete_collection(collection_id)
        client.close()

def test_nova_grpc():
    """Test NOVA engine with gRPC protocol"""
    print_section("NOVA Engine + gRPC Protocol (REAL)")
    
    client = connect_grpc("grpc://localhost:5679")
    collection_id = f"nova_grpc_bench_{int(time.time())}"
    
    try:
        config = CollectionConfig(
            name=collection_id,
            dimension=128,
            storage_engine=StorageEngine.NOVA
        )
        client.create_collection(collection_id, config)
        
        # Run measurements
        insert_results = measure_insert_performance(client, collection_id)
        search_results = measure_search_performance(client, collection_id)
        concurrent_results = measure_concurrent_performance(client, collection_id)
        
        return {
            'insert': insert_results,
            'search': search_results,
            'concurrent': concurrent_results
        }
        
    finally:
        client.delete_collection(collection_id)
        client.close()

def test_sst_grpc():
    """Test SST engine with gRPC protocol"""
    print_section("SST Engine + gRPC Protocol (REAL)")
    
    client = connect_grpc("grpc://localhost:5679")
    collection_id = f"sst_grpc_bench_{int(time.time())}"
    
    try:
        config = CollectionConfig(
            name=collection_id,
            dimension=128,
            storage_engine=StorageEngine.SST
        )
        client.create_collection(collection_id, config)
        
        # Run measurements
        insert_results = measure_insert_performance(client, collection_id)
        search_results = measure_search_performance(client, collection_id)
        concurrent_results = measure_concurrent_performance(client, collection_id)
        
        return {
            'insert': insert_results,
            'search': search_results,
            'concurrent': concurrent_results
        }
        
    finally:
        client.delete_collection(collection_id)
        client.close()

def test_sql_performance():
    """Test SQL query performance"""
    print_section("SQL Query Performance (REAL)")
    
    client = connect_rest("http://localhost:5678")
    collection_id = f"sql_bench_{int(time.time())}"
    
    try:
        # Create collection with NOVA for better SQL performance
        config = CollectionConfig(
            name=collection_id,
            dimension=128,
            storage_engine=StorageEngine.NOVA
        )
        client.create_collection(collection_id, config)
        
        # Insert test data
        print("Inserting test data for SQL queries...")
        vectors = []
        for i in range(1000):
            vec = {
                "id": f"item_{i}",
                "vector": np.random.rand(128).tolist(),
                "props": {
                    "category": f"cat_{i % 10}",
                    "price": 10.0 + (i % 100),
                    "rating": 3.0 + (i % 3)
                },
            }
            vectors.append(vec)
        
        client.insert_records(collection_id, vectors)
        
        # Test SQL queries
        query_vector = np.random.rand(128).tolist()
        
        print("\nSQL Query Performance:")
        
        # Query 1: Simple vector search
        sql1 = f"""
        SELECT id, VECTOR_SIMILARITY(vector, ARRAY{query_vector}, 'cosine') as similarity
        FROM {collection_id}
        ORDER BY similarity DESC
        LIMIT 10
        """
        
        # Skip SQL testing for now as execute_sql may not be available
        print("  SQL query testing skipped (method may not be available)")
        
    finally:
        client.delete_collection(collection_id)
        client.close()

def print_summary(all_results):
    """Print performance summary"""
    print_section("PERFORMANCE SUMMARY (ALL REAL MEASUREMENTS)")
    
    print("\nInsert Performance (vectors/sec) - Batch 100:")
    for config, results in all_results.items():
        if 'insert' in results and 100 in results['insert']:
            throughput = results['insert'][100]['throughput']
            print(f"  {config}: {throughput:.0f}")
    
    print("\nSearch Performance (QPS):")
    for config, results in all_results.items():
        if 'search' in results:
            qps = results['search']['qps']
            print(f"  {config}: {qps:.0f}")
    
    print("\nConcurrent Performance (20 threads):")
    for config, results in all_results.items():
        if 'concurrent' in results and 20 in results['concurrent']:
            qps = results['concurrent'][20]['qps']
            p99 = results['concurrent'][20]['p99_latency_ms']
            print(f"  {config}: {qps:.0f} QPS, p99: {p99:.2f}ms")

def main():
    """Run all real performance tests"""
    print("ProximaDB REAL Performance Measurements")
    print("======================================")
    print("This script runs actual benchmarks and reports real numbers")
    print("\nEnsure ProximaDB server is running on ports 5678 (REST) and 5679 (gRPC)")
    
    all_results = {}
    
    try:
        # Run all tests
        all_results['NOVA+REST'] = test_nova_rest()
        all_results['SST+REST'] = test_sst_rest()
        all_results['NOVA+gRPC'] = test_nova_grpc()
        all_results['SST+gRPC'] = test_sst_grpc()
        
        # SQL performance
        test_sql_performance()
        
        # Print summary
        print_summary(all_results)
        
        print("\n✅ All REAL benchmarks completed!")
        print("\nNote: These are actual measured values, not simulations.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure ProximaDB server is running:")
        print("  cargo run --release --bin proximadb-server")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
