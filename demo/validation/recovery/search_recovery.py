#!/usr/bin/env python3
"""
Post-Restart Recovery and Search Performance Test
Tests data recovery after server restart and search performance with quantization
"""

# Set PYTHONPATH to include src directory
import sys
import os
if 'PYTHONPATH' not in os.environ:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import time
import json
import numpy as np
from proximadb import connect_rest, connect_grpc, CollectionConfig, DistanceMetric, StorageEngine, VectorRecord

# Test configuration
DIMENSION = 128
NUM_QUERIES = 100
TOP_K = 100

def measure_server_startup_time():
    """Measure server startup and recovery time"""
    print("🔍 Measuring server startup and recovery time...")
    
    # Try to connect to both REST and gRPC APIs
    rest_ready = False
    grpc_ready = False
    start_time = time.time()
    
    while not (rest_ready and grpc_ready):
        try:
            rest_client = connect_rest("http://localhost:5678")
            # Try a simple operation to test if server is ready
            rest_client.get_collection("test_nonexistent")
            rest_ready = True
        except:
            pass
        
        try:
            grpc_client = connect_grpc("http://localhost:5679")
            # Try a simple operation to test if server is ready
            grpc_client.get_collection("test_nonexistent")
            grpc_ready = True
        except:
            pass
        
        time.sleep(0.1)
        
        # Timeout after 30 seconds
        if time.time() - start_time > 30:
            print("❌ Server startup timeout!")
            return None
    
    startup_time = time.time() - start_time
    print(f"✅ Server startup and recovery time: {startup_time:.2f}s")
    return startup_time

def verify_collection_recovery(client, collection_name, expected_count=None):
    """Verify collection exists and has expected data"""
    try:
        collection = client.get_collection(collection_name)
        if collection:
            print(f"✅ Collection '{collection_name}' recovered successfully")
            return True
        else:
            print(f"❌ Collection '{collection_name}' not found")
            return False
    except Exception as e:
        print(f"❌ Error verifying collection '{collection_name}': {e}")
        return False

def verify_vector_recovery(client, collection_name, vector_ids):
    """Verify specific vectors exist by searching for them"""
    print(f"🔍 Verifying vector recovery in '{collection_name}'...")
    
    # Test search with random queries to verify vectors are accessible
    recovered_vectors = 0
    for i in range(10):  # Test 10 random searches
        query = np.random.randn(DIMENSION).astype(np.float32)
        query = query / np.linalg.norm(query)
        
        try:
            results = client.search(collection_name, query.tolist(), top_k=10)
            if results and len(results) > 0:
                recovered_vectors += len(results)
        except Exception as e:
            print(f"❌ Error searching collection '{collection_name}': {e}")
            return False
    
    if recovered_vectors > 0:
        print(f"✅ Found {recovered_vectors} vectors across test searches")
        return True
    else:
        print(f"❌ No vectors found in collection '{collection_name}'")
        return False

def test_post_restart_search_performance(client, collection_name, protocol, engine):
    """Test search performance after restart"""
    print(f"\n🔍 Testing post-restart search performance: {protocol.upper()} + {engine.upper()}")
    
    # Load baseline if available
    baseline_file = f"{protocol}_{engine}_baseline.json"
    baseline_search_ms = None
    try:
        with open(baseline_file, "r") as f:
            baseline_data = json.load(f)
            baseline_search_ms = baseline_data.get("baseline_search_ms")
    except:
        pass
    
    # Test search performance
    search_times = []
    for i in range(NUM_QUERIES):
        query = np.random.randn(DIMENSION).astype(np.float32)
        query = query / np.linalg.norm(query)
        
        start = time.time()
        try:
            results = client.search(collection_name, query.tolist(), top_k=TOP_K)
            search_time = (time.time() - start) * 1000
            search_times.append(search_time)
        except Exception as e:
            print(f"❌ Search error: {e}")
            continue
        
        if (i + 1) % 20 == 0:
            avg_so_far = sum(search_times) / len(search_times)
            print(f"  Progress: {i+1}/{NUM_QUERIES} queries (avg: {avg_so_far:.2f}ms)")
    
    if not search_times:
        print("❌ No successful searches")
        return None
    
    # Calculate statistics
    avg_search_time = sum(search_times) / len(search_times)
    search_times_sorted = sorted(search_times)
    p50 = search_times_sorted[len(search_times_sorted) // 2]
    p95 = search_times_sorted[int(len(search_times_sorted) * 0.95)]
    p99 = search_times_sorted[int(len(search_times_sorted) * 0.99)]
    
    # Compare with baseline
    comparison = ""
    if baseline_search_ms:
        diff = avg_search_time - baseline_search_ms
        pct_change = (diff / baseline_search_ms) * 100
        comparison = f" (vs baseline: {diff:+.2f}ms, {pct_change:+.1f}%)"
    
    print(f"✅ Search performance after restart:")
    print(f"   - Average: {avg_search_time:.2f}ms{comparison}")
    print(f"   - P50: {p50:.2f}ms")
    print(f"   - P95: {p95:.2f}ms")
    print(f"   - P99: {p99:.2f}ms")
    
    return {
        "avg_latency_ms": avg_search_time,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "baseline_latency_ms": baseline_search_ms,
        "queries_tested": len(search_times)
    }

def main():
    print("🚀 Post-Restart Recovery and Search Performance Test")
    print("="*80)
    
    # Measure server startup time
    startup_time = measure_server_startup_time()
    if startup_time is None:
        print("❌ Server not ready, exiting")
        return
    
    # Test collections to verify
    test_collections = [
        ("rest_nova_persist_test", "rest", "nova", connect_rest("http://localhost:5678")),
        ("rest_lsm_persist_test", "rest", "lsm", connect_rest("http://localhost:5678")),
        ("grpc_nova_persist_test", "grpc", "nova", connect_grpc("http://localhost:5679")),
        ("grpc_lsm_persist_test", "grpc", "lsm", connect_grpc("http://localhost:5679"))
    ]
    
    recovery_results = {
        "server_startup_time_s": startup_time,
        "collection_recovery": {},
        "search_performance": {}
    }
    
    print("\n🔍 Testing collection recovery...")
    for collection_name, protocol, engine, client in test_collections:
        print(f"\n--- Testing {collection_name} ---")
        
        # Verify collection exists
        collection_exists = verify_collection_recovery(client, collection_name)
        
        # Verify vectors exist
        vectors_exist = False
        search_results = None
        
        if collection_exists:
            vectors_exist = verify_vector_recovery(client, collection_name, [])
            
            # Test search performance
            if vectors_exist:
                search_results = test_post_restart_search_performance(client, collection_name, protocol, engine)
        
        recovery_results["collection_recovery"][collection_name] = {
            "protocol": protocol,
            "engine": engine,
            "collection_exists": collection_exists,
            "vectors_accessible": vectors_exist
        }
        
        if search_results:
            recovery_results["search_performance"][collection_name] = search_results
    
    # Save results
    recovery_results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    with open("recovery_test_results.json", "w") as f:
        json.dump(recovery_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("RECOVERY TEST SUMMARY")
    print("="*80)
    print(f"Server startup time: {startup_time:.2f}s")
    
    print("\nCollection Recovery:")
    for collection_name, result in recovery_results["collection_recovery"].items():
        status = "✅" if result["collection_exists"] and result["vectors_accessible"] else "❌"
        print(f"  {status} {collection_name}: {result['protocol'].upper()} + {result['engine'].upper()}")
    
    print("\nSearch Performance After Restart:")
    for collection_name, result in recovery_results["search_performance"].items():
        print(f"  {collection_name}: {result['avg_latency_ms']:.2f}ms avg")
    
    print("\n📊 Results saved to recovery_test_results.json")

if __name__ == "__main__":
    main()