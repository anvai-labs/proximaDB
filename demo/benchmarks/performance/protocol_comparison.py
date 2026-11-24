#!/usr/bin/env python3
"""
Comprehensive ProximaDB Performance Test
Tests REST and gRPC protocols with SST and NOVA engines
Including quantization performance (16-bit, 8-bit, 4-bit)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import time
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from proximadb_sdk import (
    CollectionConfig,
    DistanceMetric,
    StorageEngine,
    connect_grpc,
    connect_rest,
)
from proximadb_sdk.models import QuantizationConfig, QuantizationType

# Test configuration
DIMENSION = 128
NUM_VECTORS = 1000
BATCH_SIZE = 100
NUM_QUERIES = 100
TOP_K = 10

def generate_test_vectors(num_vectors: int, dimension: int) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    """Generate test vectors and queries"""
    vectors = []
    raw_vectors = []
    
    for i in range(num_vectors):
        vec_data = np.random.randn(dimension).astype(np.float32)
        # Normalize to unit length for cosine similarity
        vec_data = vec_data / np.linalg.norm(vec_data)
        raw_vectors.append(vec_data)
        
        vectors.append(
            {
                "id": f"vec_{i}",
                "vector": vec_data.tolist(),
                "props": {"index": i, "category": f"cat_{i % 10}"},
            }
        )
    
    # Generate query vectors
    queries = []
    for i in range(NUM_QUERIES):
        query = np.random.randn(dimension).astype(np.float32)
        query = query / np.linalg.norm(query)
        queries.append(query)
    
    return vectors, queries, raw_vectors

def calculate_recall(ground_truth: List[str], results: List[str]) -> float:
    """Calculate recall@k"""
    if not ground_truth:
        return 0.0
    
    hits = len(set(ground_truth) & set(results))
    return hits / len(ground_truth)

def get_ground_truth(query: np.ndarray, vectors: List[np.ndarray], vector_ids: List[str], k: int) -> List[str]:
    """Get ground truth nearest neighbors using exact search"""
    similarities = []
    for i, vec in enumerate(vectors):
        sim = np.dot(query, vec)  # Cosine similarity for normalized vectors
        similarities.append((vector_ids[i], sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in similarities[:k]]

def time_operation(func, *args, **kwargs):
    """Time an operation and return result with timing"""
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, (end - start) * 1000  # Convert to ms

def test_protocol_engine_combination(
    protocol: str,
    engine: str,
    quantization: str = None
) -> Dict[str, Any]:
    """Test a specific protocol/engine/quantization combination"""
    
    print(f"\n{'='*60}")
    print(f"Testing: {protocol.upper()} + {engine.upper()} Engine" + (f" + {quantization}" if quantization else ""))
    print(f"{'='*60}")
    
    results = {
        "protocol": protocol,
        "engine": engine,
        "quantization": quantization,
        "metrics": {}
    }
    
    # Connect to server
    if protocol == "rest":
        client = connect_rest("http://localhost:5678")
    else:
        client = connect_grpc("grpc://localhost:5679")
    
    # Create collection with specified engine
    collection_name = f"perf_{protocol}_{engine}_{quantization or 'fp32'}_{int(time.time())}"
    
    config = CollectionConfig(
        name=collection_name,
        dimension=DIMENSION,
        distance_metric=DistanceMetric.COSINE,
        storage_engine=StorageEngine.NOVA if engine == "nova" else StorageEngine.SST,
        description=f"Performance test: {protocol}/{engine}/{quantization or 'FP32'}"
    )
    
    # Add quantization config if specified
    if quantization:
        if quantization == "pq16":
            config.quantization_config = QuantizationConfig(
                type=QuantizationType.PRODUCT,
                bits=16,
                num_subvectors=16
            )
        elif quantization == "pq8":
            config.quantization_config = QuantizationConfig(
                type=QuantizationType.PRODUCT,
                bits=8,
                num_subvectors=16
            )
        elif quantization == "pq4":
            config.quantization_config = QuantizationConfig(
                type=QuantizationType.PRODUCT,
                bits=4,
                num_subvectors=16
            )
    
    # Create collection
    collection, create_time = time_operation(client.create_collection, collection_name, config)
    results["metrics"]["create_collection_ms"] = create_time
    print(f"✅ Collection created: {create_time:.2f}ms")
    
    # Generate test data
    print(f"\n📊 Generating {NUM_VECTORS} test vectors...")
    vectors, queries, raw_vectors = generate_test_vectors(NUM_VECTORS, DIMENSION)
    vector_ids = [v["id"] for v in vectors]
    
    # Test batch insert
    print(f"\n📝 Testing batch insert ({BATCH_SIZE} vectors per batch)...")
    insert_times = []
    total_start = time.time()
    
    for i in range(0, NUM_VECTORS, BATCH_SIZE):
        batch = vectors[i:i+BATCH_SIZE]
        _, batch_time = time_operation(client.insert_records, collection_name, batch)
        insert_times.append(batch_time)
    
    total_insert_time = (time.time() - total_start) * 1000
    avg_batch_time = sum(insert_times) / len(insert_times)
    
    results["metrics"]["insert"] = {
        "total_vectors": NUM_VECTORS,
        "batch_size": BATCH_SIZE,
        "avg_batch_ms": avg_batch_time,
        "total_ms": total_insert_time,
        "vectors_per_second": (NUM_VECTORS / total_insert_time) * 1000
    }
    
    print(f"✅ Insert complete: {avg_batch_time:.2f}ms per batch, {(NUM_VECTORS / total_insert_time) * 1000:.0f} vectors/sec")
    
    # Wait for data to be flushed if NOVA
    if engine == "nova":
        print("⏳ Waiting for NOVA flush...")
        time.sleep(2)
    
    # Test search performance and accuracy
    print(f"\n🔍 Testing search performance (top-{TOP_K})...")
    search_times = []
    recalls = []
    
    for i, query in enumerate(queries[:20]):  # Test first 20 queries
        # Get ground truth
        ground_truth = get_ground_truth(query, raw_vectors, vector_ids, TOP_K)
        
        # Search
        search_results, search_time = time_operation(
            client.search, 
            collection_name, 
            query.tolist(), 
            top_k=TOP_K
        )
        search_times.append(search_time)
        
        # Calculate recall
        result_ids = [r.id for r in search_results]
        recall = calculate_recall(ground_truth[:len(result_ids)], result_ids)
        recalls.append(recall)
    
    avg_search_time = sum(search_times) / len(search_times)
    avg_recall = sum(recalls) / len(recalls)
    
    results["metrics"]["search"] = {
        "top_k": TOP_K,
        "queries_tested": len(search_times),
        "avg_latency_ms": avg_search_time,
        "searches_per_second": 1000 / avg_search_time,
        "avg_recall": avg_recall,
        "accuracy_percentage": avg_recall * 100
    }
    
    print(f"✅ Search complete: {avg_search_time:.2f}ms avg latency, {avg_recall*100:.1f}% recall@{TOP_K}")
    
    # Cleanup
    client.delete_collection(collection_name)
    
    return results

def run_comprehensive_tests():
    """Run all test combinations"""
    
    all_results = {
        "test_config": {
            "dimension": DIMENSION,
            "num_vectors": NUM_VECTORS,
            "batch_size": BATCH_SIZE,
            "num_queries": NUM_QUERIES,
            "top_k": TOP_K
        },
        "results": []
    }
    
    # Test combinations
    test_cases = [
        # REST tests
        ("rest", "nova", None),      # REST + NOVA + FP32
        ("rest", "nova", "pq16"),    # REST + NOVA + PQ16
        ("rest", "nova", "pq8"),     # REST + NOVA + PQ8
        ("rest", "nova", "pq4"),     # REST + NOVA + PQ4
        ("rest", "sst", None),        # REST + SST + FP32
        ("rest", "sst", "pq16"),      # REST + SST + PQ16
        ("rest", "sst", "pq8"),       # REST + SST + PQ8
        ("rest", "sst", "pq4"),       # REST + SST + PQ4
        
        # gRPC tests
        ("grpc", "nova", None),      # gRPC + NOVA + FP32
        ("grpc", "nova", "pq16"),    # gRPC + NOVA + PQ16
        ("grpc", "nova", "pq8"),     # gRPC + NOVA + PQ8
        ("grpc", "nova", "pq4"),     # gRPC + NOVA + PQ4
        ("grpc", "sst", None),        # gRPC + SST + FP32
        ("grpc", "sst", "pq16"),      # gRPC + SST + PQ16
        ("grpc", "sst", "pq8"),       # gRPC + SST + PQ8
        ("grpc", "sst", "pq4"),       # gRPC + SST + PQ4
    ]
    
    for protocol, engine, quantization in test_cases:
        try:
            result = test_protocol_engine_combination(protocol, engine, quantization)
            all_results["results"].append(result)
        except Exception as e:
            print(f"❌ Error testing {protocol}/{engine}/{quantization}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    with open("comprehensive_perf_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print_performance_summary(all_results)
    
    return all_results

def print_performance_summary(results: Dict[str, Any]):
    """Print a formatted summary of all results"""
    
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    # Group by protocol
    for protocol in ["rest", "grpc"]:
        print(f"\n{protocol.upper()} Protocol Results:")
        print("-"*60)
        
        protocol_results = [r for r in results["results"] if r["protocol"] == protocol]
        
        # Print header
        print(f"{'Engine':<8} {'Quant':<6} {'Insert (vec/s)':<15} {'Search (ms)':<12} {'Recall@10':<10}")
        print("-"*60)
        
        for result in protocol_results:
            engine = result["engine"].upper()
            quant = result["quantization"] or "FP32"
            insert_rate = result["metrics"]["insert"]["vectors_per_second"]
            search_latency = result["metrics"]["search"]["avg_latency_ms"]
            recall = result["metrics"]["search"]["avg_recall"] * 100
            
            print(f"{engine:<8} {quant:<6} {insert_rate:<15.0f} {search_latency:<12.2f} {recall:<10.1f}%")
    
    # Print accuracy vs performance tradeoff
    print("\n" + "="*80)
    print("QUANTIZATION TRADEOFFS")
    print("="*80)
    
    for engine in ["nova", "sst"]:
        print(f"\n{engine.upper()} Engine:")
        print("-"*60)
        
        # Get FP32 baseline
        fp32_results = [r for r in results["results"] 
                       if r["engine"] == engine and r["quantization"] is None]
        
        if fp32_results:
            fp32_search = fp32_results[0]["metrics"]["search"]["avg_latency_ms"]
            fp32_recall = fp32_results[0]["metrics"]["search"]["avg_recall"] * 100
            
            print(f"{'Quantization':<12} {'Speedup':<10} {'Recall':<10} {'Accuracy Drop':<15}")
            print("-"*60)
            
            for quant in [None, "pq16", "pq8", "pq4"]:
                quant_results = [r for r in results["results"] 
                               if r["engine"] == engine and r["quantization"] == quant]
                
                if quant_results:
                    search_ms = quant_results[0]["metrics"]["search"]["avg_latency_ms"]
                    recall = quant_results[0]["metrics"]["search"]["avg_recall"] * 100
                    speedup = fp32_search / search_ms if search_ms > 0 else 1.0
                    accuracy_drop = fp32_recall - recall
                    
                    quant_str = quant or "FP32"
                    print(f"{quant_str:<12} {speedup:<10.2f}x {recall:<10.1f}% {accuracy_drop:<15.1f}%")

if __name__ == "__main__":
    print("🚀 Starting Comprehensive ProximaDB Performance Test")
    print("   Testing REST and gRPC with SST and NOVA engines")
    print("   Including PQ16, PQ8, and PQ4 quantization")
    
    run_comprehensive_tests()
