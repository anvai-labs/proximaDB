#!/usr/bin/env python3
"""Performance comparison between original VectorService and optimized DirectVectorService"""

import time
import requests
import json
import statistics
from typing import List, Dict

def create_collection(collection_name: str, port: int = 5678) -> bool:
    """Create a test collection."""
    url = f"http://localhost:{port}/api/v1/collection"
    data = {
        "operation": "create",
        "config": {
            "name": collection_name,
            "dimension": 128,
            "distance_metric": "cosine",
            "storage_engine": "NOVA"
        }
    }
    response = requests.post(url, json=data)
    return response.status_code == 200

def insert_vectors_batch(collection_name: str, batch_size: int, port: int = 5678) -> Dict:
    """Insert a batch of vectors and measure performance."""
    url = f"http://localhost:{port}/api/v1/vector/batch"
    
    vectors = []
    for i in range(batch_size):
        vector = {
            "id": f"vec_{i}",
            "vector": [float(j % 10) / 10.0 for j in range(128)],
            "metadata": {"batch": str(i // 100)}
        }
        vectors.append(vector)
    
    data = {
        "operation": "upsert",
        "collection_id": collection_name,
        "vectors": vectors
    }
    
    start_time = time.time()
    response = requests.post(url, json=data)
    end_time = time.time()
    
    return {
        "success": response.status_code == 200,
        "duration_ms": (end_time - start_time) * 1000,
        "vectors_per_second": batch_size / (end_time - start_time) if end_time > start_time else 0,
        "batch_size": batch_size
    }

def search_vectors(collection_name: str, port: int = 5678) -> Dict:
    """Search vectors and measure performance."""
    url = f"http://localhost:{port}/api/v1/vector/search"
    query_vector = [0.1] * 128
    
    data = {
        "collection_id": collection_name,
        "queries": [{
            "vector": query_vector
        }],
        "top_k": 10
    }
    
    start_time = time.time()
    response = requests.post(url, json=data)
    end_time = time.time()
    
    results = []
    if response.status_code == 200:
        results = response.json().get('results', [])
    
    return {
        "success": response.status_code == 200,
        "duration_ms": (end_time - start_time) * 1000,
        "results_found": len(results),
    }

def run_performance_test(service_name: str, port: int, iterations: int = 10) -> Dict:
    """Run comprehensive performance test."""
    print(f"\n=== Testing {service_name} (port {port}) ===")
    
    collection_name = f"perf_test_{service_name.lower()}"
    
    # Create collection
    if not create_collection(collection_name, port):
        print(f"❌ Failed to create collection for {service_name}")
        return {}
    
    # Test different batch sizes
    batch_sizes = [10, 50, 100, 500]
    insert_results = {}
    
    for batch_size in batch_sizes:
        print(f"📝 Testing insert performance with batch size {batch_size}")
        
        durations = []
        throughputs = []
        
        for i in range(iterations):
            result = insert_vectors_batch(f"{collection_name}_batch_{batch_size}", batch_size, port)
            if result["success"]:
                durations.append(result["duration_ms"])
                throughputs.append(result["vectors_per_second"])
            else:
                print(f"⚠️ Insert failed for batch {i}")
        
        if durations:
            insert_results[batch_size] = {
                "avg_duration_ms": statistics.mean(durations),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations),
                "avg_throughput": statistics.mean(throughputs),
                "max_throughput": max(throughputs)
            }
            
            print(f"   Batch {batch_size}: {insert_results[batch_size]['avg_duration_ms']:.2f}ms avg, "
                  f"{insert_results[batch_size]['avg_throughput']:.0f} vectors/sec avg")
    
    # Test search performance
    print(f"🔍 Testing search performance")
    search_durations = []
    
    for i in range(iterations * 2):  # More search tests
        result = search_vectors(collection_name, port)
        if result["success"]:
            search_durations.append(result["duration_ms"])
            if i == 0:  # Log first search result count
                print(f"   Found {result['results_found']} results in first search")
    
    search_results = {}
    if search_durations:
        search_results = {
            "avg_duration_ms": statistics.mean(search_durations),
            "min_duration_ms": min(search_durations),
            "max_duration_ms": max(search_durations),
            "searches_per_second": 1000 / statistics.mean(search_durations)
        }
        
        print(f"   Search: {search_results['avg_duration_ms']:.2f}ms avg, "
              f"{search_results['searches_per_second']:.0f} searches/sec")
    
    return {
        "service": service_name,
        "port": port,
        "insert_results": insert_results,
        "search_results": search_results
    }

def compare_performance(original_results: Dict, optimized_results: Dict):
    """Compare performance between original and optimized services."""
    print(f"\n🏆 === PERFORMANCE COMPARISON ===")
    
    if not original_results or not optimized_results:
        print("❌ Cannot compare - missing results")
        return
    
    # Compare insert performance
    print(f"\n📝 INSERT PERFORMANCE:")
    for batch_size in [10, 50, 100, 500]:
        if batch_size in original_results["insert_results"] and batch_size in optimized_results["insert_results"]:
            orig = original_results["insert_results"][batch_size]
            opt = optimized_results["insert_results"][batch_size]
            
            duration_improvement = ((orig["avg_duration_ms"] - opt["avg_duration_ms"]) / orig["avg_duration_ms"]) * 100
            throughput_improvement = ((opt["avg_throughput"] - orig["avg_throughput"]) / orig["avg_throughput"]) * 100
            
            print(f"   Batch {batch_size:3d}: Duration {duration_improvement:+6.1f}%, Throughput {throughput_improvement:+6.1f}%")
    
    # Compare search performance
    print(f"\n🔍 SEARCH PERFORMANCE:")
    if original_results["search_results"] and optimized_results["search_results"]:
        orig_search = original_results["search_results"]
        opt_search = optimized_results["search_results"]
        
        search_improvement = ((orig_search["avg_duration_ms"] - opt_search["avg_duration_ms"]) / orig_search["avg_duration_ms"]) * 100
        search_throughput_improvement = ((opt_search["searches_per_second"] - orig_search["searches_per_second"]) / orig_search["searches_per_second"]) * 100
        
        print(f"   Search Latency: {search_improvement:+6.1f}%")
        print(f"   Search Throughput: {search_throughput_improvement:+6.1f}%")
    
    # Overall summary
    print(f"\n📊 SUMMARY:")
    print(f"   🎯 Expected improvements from eliminating WAL Manager Registry:")
    print(f"      • Insert latency: 40-60% reduction")
    print(f"      • Memory allocations: ~2 fewer per operation")
    print(f"      • HashMap lookups: Eliminated")
    print(f"   📈 Measured improvements will be visible once DirectVectorService is integrated")

def main():
    """Run performance comparison test."""
    print("🚀 ProximaDB Performance Comparison Test")
    print("   Comparing original VectorService vs optimized DirectVectorService")
    
    # Test original service (current implementation)
    print("\n⏳ Please ensure ProximaDB server is running on port 5678...")
    input("Press Enter when ready...")
    
    original_results = run_performance_test("Original VectorService", 5678, iterations=5)
    
    # TODO: Test optimized service once integrated
    print(f"\n📋 DirectVectorService integration pending - will test once implemented")
    optimized_results = {}
    
    # Show comparison
    compare_performance(original_results, optimized_results)
    
    # Save results
    results = {
        "timestamp": time.time(),
        "original": original_results,
        "optimized": optimized_results
    }
    
    with open("performance_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to performance_comparison_results.json")

if __name__ == "__main__":
    main()