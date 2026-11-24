#!/usr/bin/env python3
"""Benchmark compression performance for ProximaDB"""

import sys
import time
import numpy as np
import json
import gzip
from datetime import datetime
from typing import Dict, List, Any

# Use PYTHONPATH instead of sys.path
# export PYTHONPATH=/home/vsingh/code/proximaDB/clients/python/src

from proximadb_sdk import ProximaDBClient, connect_grpc, connect_rest
from proximadb_sdk.config import ClientConfig, CompressionConfig, Protocol
from proximadb_sdk.models import CollectionConfig, DistanceMetric, StorageEngine


class CompressionBenchmark:
    """Benchmark compression performance"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "benchmarks": []
        }
    
    def measure_data_sizes(self, num_vectors: int, dimension: int) -> Dict[str, Any]:
        """Measure data sizes with and without compression"""
        
        # Generate test data
        vectors = np.random.rand(num_vectors, dimension).astype(np.float32)
        ids = [f"vec_{i}" for i in range(num_vectors)]
        metadata = [{"category": f"cat_{i % 10}", "value": i, "description": f"Test vector {i}"} for i in range(num_vectors)]
        
        # Create JSON payload
        payload = {
            "vectors": vectors.tolist(),
            "ids": ids,
            "metadata": metadata
        }
        
        # Measure sizes
        json_data = json.dumps(payload)
        json_bytes = json_data.encode('utf-8')
        uncompressed_size = len(json_bytes)
        
        # Compress with different algorithms
        compression_results = {}
        
        # Gzip
        gzip_data = gzip.compress(json_bytes, compresslevel=6)
        compression_results['gzip'] = {
            'size': len(gzip_data),
            'ratio': (1 - len(gzip_data) / uncompressed_size) * 100
        }
        
        # Deflate (zlib)
        import zlib
        deflate_data = zlib.compress(json_bytes, level=6)
        compression_results['deflate'] = {
            'size': len(deflate_data),
            'ratio': (1 - len(deflate_data) / uncompressed_size) * 100
        }
        
        # Try optional compression libraries
        try:
            import zstandard
            cctx = zstandard.ZstdCompressor(level=3)
            zstd_data = cctx.compress(json_bytes)
            compression_results['zstd'] = {
                'size': len(zstd_data),
                'ratio': (1 - len(zstd_data) / uncompressed_size) * 100
            }
        except ImportError:
            pass
        
        try:
            import brotli
            br_data = brotli.compress(json_bytes, quality=4)
            compression_results['brotli'] = {
                'size': len(br_data),
                'ratio': (1 - len(br_data) / uncompressed_size) * 100
            }
        except ImportError:
            pass
        
        return {
            "num_vectors": num_vectors,
            "dimension": dimension,
            "uncompressed_size_mb": uncompressed_size / (1024 * 1024),
            "compression_results": compression_results,
            "vectors": vectors,
            "ids": ids,
            "metadata": metadata
        }
    
    def benchmark_protocol(self, protocol: str, num_vectors: int, dimension: int, 
                          enable_compression: bool) -> Dict[str, Any]:
        """Benchmark a specific protocol with/without compression"""
        
        print(f"\n{'='*60}")
        print(f"Protocol: {protocol}, Compression: {enable_compression}")
        print(f"Vectors: {num_vectors}, Dimension: {dimension}")
        print('='*60)
        
        # Prepare test data
        data_info = self.measure_data_sizes(num_vectors, dimension)
        
        if enable_compression:
            print(f"📊 Data sizes:")
            print(f"   Uncompressed: {data_info['uncompressed_size_mb']:.2f} MB")
            print(f"   Compression results:")
            for algo, stats in data_info['compression_results'].items():
                print(f"     {algo}: {stats['size'] / (1024*1024):.2f} MB ({stats['ratio']:.1f}% reduction)")
        
        # Create client with proper configuration
        if protocol == "REST":
            if enable_compression:
                config = ClientConfig(
                    url="http://localhost:5678",
                    protocol=Protocol.REST,
                    compression=CompressionConfig(
                        enabled=True,
                        algorithm="gzip",
                        threshold_bytes=1024,
                        level=6
                    )
                )
                client = ProximaDBClient(config=config)
            else:
                client = connect_rest("http://localhost:5678")
        else:  # gRPC
            client = connect_grpc(
                "http://localhost:5679",
                enable_compression=enable_compression
            )
        
        try:
            # Create collection with unified API
            collection_name = f"compress_bench_{int(time.time())}"
            config = CollectionConfig(
                name=collection_name,
                dimension=dimension,
                distance_metric=DistanceMetric.COSINE,
                storage_engine=StorageEngine.SST
            )
            client.create_collection(collection_name, config)
            
            # Benchmark insert with the record-native SDK API.
            start_time = time.time()
            records = []
            for i in range(num_vectors):
                records.append(
                    {
                        "id": data_info["ids"][i],
                        "vector": data_info["vectors"][i].tolist(),
                        "props": data_info["metadata"][i],
                    }
                )
            
            client.insert_records(collection_name, records)
            insert_time = time.time() - start_time
            insert_throughput = num_vectors / insert_time
            
            print(f"\n📥 Insert Performance:")
            print(f"   Time: {insert_time:.2f}s")
            print(f"   Throughput: {insert_throughput:.0f} vectors/sec")
            
            # Benchmark search
            query_vector = np.random.rand(dimension).tolist()
            search_times = []
            
            for i in range(10):
                start_time = time.time()
                # Use unified search method
                results = client.search(
                    collection_name,
                    query_vector,
                    top_k=10
                )
                search_times.append((time.time() - start_time) * 1000)
            
            avg_search_time = np.mean(search_times)
            
            print(f"\n🔍 Search Performance (10 runs):")
            print(f"   Average: {avg_search_time:.2f}ms")
            print(f"   Min: {min(search_times):.2f}ms")
            print(f"   Max: {max(search_times):.2f}ms")
            
            # Cleanup
            client.delete_collection(collection_name)
            
            return {
                "protocol": protocol,
                "compression_enabled": enable_compression,
                "num_vectors": num_vectors,
                "dimension": dimension,
                "data_size_mb": data_info['uncompressed_size_mb'],
                "compressed_size_mb": data_info['compression_results']['gzip']['size'] / (1024*1024) if enable_compression else None,
                "compression_ratio": data_info['compression_results']['gzip']['ratio'] if enable_compression else None,
                "insert_time_s": insert_time,
                "insert_throughput": insert_throughput,
                "avg_search_time_ms": avg_search_time,
                "min_search_time_ms": min(search_times),
                "max_search_time_ms": max(search_times)
            }
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return None
    
    def benchmark_compression_algorithms(self):
        """Test different compression algorithms on REST API"""
        print(f"\n🗜️  Compression Algorithm Comparison")
        print("=" * 80)
        
        algorithms = ["deflate", "gzip", "zstd", "br"]
        test_data_sizes = [(100, 384), (1000, 768)]  # (num_vectors, dimension)
        
        for num_vectors, dimension in test_data_sizes:
            print(f"\n📊 Testing {num_vectors} vectors, {dimension} dimensions")
            data_info = self.measure_data_sizes(num_vectors, dimension)
            
            for algo in algorithms:
                try:
                    config = ClientConfig(
                        url="http://localhost:5678",
                        protocol=Protocol.REST,
                        compression=CompressionConfig(
                            enabled=True,
                            algorithm=algo,
                            threshold_bytes=1024
                        )
                    )
                    client = ProximaDBClient(config=config)
                    
                    collection_name = f"algo_test_{algo}_{int(time.time())}"
                    
                    # Create collection
                    col_config = CollectionConfig(
                        name=collection_name,
                        dimension=dimension,
                        distance_metric=DistanceMetric.COSINE,
                        storage_engine=StorageEngine.NOVA
                    )
                    client.create_collection(collection_name, col_config)
                    
                    # Measure insert time through the record-native SDK API.
                    records = []
                    for i in range(num_vectors):
                        records.append(
                            {
                                "id": data_info["ids"][i],
                                "vector": data_info["vectors"][i].tolist(),
                                "props": data_info["metadata"][i],
                            }
                        )
                    
                    start = time.time()
                    client.insert_records(collection_name, records)
                    insert_time = time.time() - start
                    
                    # Cleanup
                    client.delete_collection(collection_name)
                    
                    # Get compression stats if available
                    if algo in data_info['compression_results']:
                        comp_stats = data_info['compression_results'][algo]
                        print(f"  {algo}: {insert_time:.2f}s insert, {comp_stats['ratio']:.1f}% reduction")
                    else:
                        print(f"  {algo}: {insert_time:.2f}s insert (library not available)")
                        
                except Exception as e:
                    print(f"  {algo}: Failed - {e}")
    
    def run_benchmarks(self):
        """Run all compression benchmarks"""
        
        print(f"\n🗜️  ProximaDB Compression Benchmark - {datetime.now()}")
        print("=" * 80)
        
        # First test different algorithms
        self.benchmark_compression_algorithms()
        
        # Then run protocol comparison
        print(f"\n\n📊 Protocol Compression Comparison")
        print("=" * 80)
        
        # Test configurations
        test_configs = [
            (100, 128),    # Small: 100 vectors, 128 dimensions
            (1000, 384),   # Medium: 1000 vectors, 384 dimensions
            (5000, 768),   # Large: 5000 vectors, 768 dimensions
        ]
        
        for num_vectors, dimension in test_configs:
            for protocol in ["REST", "gRPC"]:
                for compression in [False, True]:
                    result = self.benchmark_protocol(
                        protocol, num_vectors, dimension, compression
                    )
                    if result:
                        self.results["benchmarks"].append(result)
                    time.sleep(1)  # Brief pause between tests
        
        # Generate summary
        self.print_summary()
        
        # Save results
        with open(f"compression_benchmark_{int(time.time())}.json", "w") as f:
            json.dump(self.results, f, indent=2)
    
    def print_summary(self):
        """Print benchmark summary"""
        
        print("\n" + "=" * 80)
        print("📊 COMPRESSION BENCHMARK SUMMARY")
        print("=" * 80)
        
        # Group results
        rest_results = [r for r in self.results["benchmarks"] if r and r["protocol"] == "REST"]
        grpc_results = [r for r in self.results["benchmarks"] if r and r["protocol"] == "gRPC"]
        
        # Compare compressed vs uncompressed
        print("\n🚀 Performance Impact of Compression:")
        
        for protocol, results in [("REST", rest_results), ("gRPC", grpc_results)]:
            print(f"\n{protocol}:")
            
            # Group by vector count
            by_size = {}
            for r in results:
                key = r["num_vectors"]
                if key not in by_size:
                    by_size[key] = {"compressed": None, "uncompressed": None}
                
                if r["compression_enabled"]:
                    by_size[key]["compressed"] = r
                else:
                    by_size[key]["uncompressed"] = r
            
            # Compare
            for num_vectors in sorted(by_size.keys()):
                data = by_size[num_vectors]
                if data["compressed"] and data["uncompressed"]:
                    comp = data["compressed"]
                    uncomp = data["uncompressed"]
                    
                    insert_diff = ((comp["insert_throughput"] / uncomp["insert_throughput"]) - 1) * 100
                    search_diff = ((uncomp["avg_search_time_ms"] / comp["avg_search_time_ms"]) - 1) * 100
                    
                    print(f"\n  {num_vectors} vectors (dim={comp['dimension']}):")
                    print(f"    Data reduction: {comp['compression_ratio']:.1f}%")
                    print(f"    Insert performance: {insert_diff:+.1f}%")
                    print(f"    Search performance: {search_diff:+.1f}%")
        
        print("\n" + "=" * 80)


if __name__ == "__main__":
    benchmark = CompressionBenchmark()
    benchmark.run_benchmarks()
