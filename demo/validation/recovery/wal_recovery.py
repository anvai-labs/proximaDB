#!/usr/bin/env python3
"""
Vector WAL Recovery Test
Specifically tests vector data recovery from WAL after server restart
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

def test_vector_wal_recovery():
    """Test vector WAL recovery specifically"""
    
    print("🚀 Vector WAL Recovery Test")
    print("="*80)
    print("Testing vector data recovery from WAL after server restart")
    print("="*80)
    
    # Test configuration
    collection_name = "vector_wal_recovery_test"
    
    # Connect to server
    print("🔗 Connecting to server...")
    client = connect_rest("http://localhost:5678")
    
    # Check if collection exists and has vectors
    print(f"🔍 Checking collection '{collection_name}' for persisted vectors...")
    
    recovery_results = {
        "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "collection_found": False,
        "vectors_recovered": 0,
        "search_working": False,
        "recovery_successful": False
    }
    
    try:
        # Get collection
        existing_collection = client.get_collection(collection_name)
        recovery_results["collection_found"] = True
        print(f"✅ Collection found: {existing_collection.config.name}")
        
        # Test search to see if vectors were recovered
        print("🔍 Testing search for recovered vectors...")
        
        # Use a known test vector pattern
        test_query = [float(i) for i in range(128)]  # Sequential pattern
        
        try:
            search_results = client.search(collection_name, test_query, top_k=10)
            recovery_results["vectors_recovered"] = len(search_results)
            recovery_results["search_working"] = True
            
            if len(search_results) > 0:
                print(f"✅ VECTOR WAL RECOVERY SUCCESS!")
                print(f"   Found {len(search_results)} vectors recovered from WAL")
                
                for i, result in enumerate(search_results):
                    print(f"   Vector {i+1}: ID={result.id}, Score={getattr(result, 'score', 'N/A')}")
                    
                    # Check if this is one of our test vectors
                    if hasattr(result, 'metadata') and result.metadata:
                        print(f"     Metadata: {result.metadata}")
                
                recovery_results["recovery_successful"] = True
                
                # Try to identify which phase these vectors are from
                phase_counts = {}
                for result in search_results:
                    if hasattr(result, 'metadata') and result.metadata:
                        phase = result.metadata.get('phase', 'unknown')
                        phase_counts[phase] = phase_counts.get(phase, 0) + 1
                
                if phase_counts:
                    print(f"   Recovery by phase: {phase_counts}")
                
            else:
                print("❌ No vectors recovered from WAL")
                recovery_results["recovery_successful"] = False
                
        except Exception as e:
            print(f"❌ Search failed: {e}")
            recovery_results["search_working"] = False
        
        print(f"\n🎉 WAL RECOVERY TEST COMPLETE")
        print(f"   - Collection persisted: ✅")
        print(f"   - Vectors recovered: {recovery_results['vectors_recovered']}")
        print(f"   - Recovery successful: {'✅' if recovery_results['recovery_successful'] else '❌'}")
        
        return recovery_results
        
    except Exception as e:
        print(f"📦 Collection not found: {e}")
        recovery_results["collection_found"] = False
    
    # If collection doesn't exist, create it and insert test vectors
    print(f"📦 Creating new collection for WAL testing...")
    
    # Create collection
    config = CollectionConfig(
        name=collection_name,
        dimension=128,
        distance_metric=DistanceMetric.COSINE,
        storage_engine=StorageEngine.NOVA,
        description="Vector WAL recovery test collection"
    )
    
    collection = client.create_collection(collection_name, config)
    print(f"✅ Collection created: {collection.config.name}")
    
    # Insert test vectors in multiple phases to test WAL recovery
    phases = [
        {"name": "phase1", "count": 100, "pattern": "sequential"},
        {"name": "phase2", "count": 200, "pattern": "random"},
        {"name": "phase3", "count": 150, "pattern": "clustered"}
    ]
    
    insertion_results = {}
    total_vectors = 0
    
    for phase in phases:
        print(f"\n📝 Inserting {phase['count']} vectors for {phase['name']}...")
        
        vectors = []
        for i in range(phase['count']):
            vector_id = f"{phase['name']}_vec_{i}"
            
            # Generate different patterns
            if phase['pattern'] == 'sequential':
                vector_data = [float(j + i) for j in range(128)]
            elif phase['pattern'] == 'random':
                np.random.seed(42 + i)  # Reproducible random
                vector_data = np.random.randn(128).astype(np.float32).tolist()
            else:  # clustered
                cluster_center = [float(i % 10)] * 128
                noise = np.random.randn(128).astype(np.float32) * 0.1
                vector_data = [cluster_center[j] + noise[j] for j in range(128)]
            
            # Normalize
            norm = np.linalg.norm(vector_data)
            if norm > 0:
                vector_data = [x / norm for x in vector_data]
            
            vector = VectorRecord(
                id=vector_id,
                vector=vector_data,
                metadata={
                    "phase": phase['name'],
                    "pattern": phase['pattern'],
                    "index": i,
                    "total_index": total_vectors + i,
                    "timestamp": int(time.time())
                }
            )
            vectors.append(vector)
        
        # Insert vectors
        start_time = time.time()
        result = client.insert_vectors(collection_name, vectors)
        insert_time = time.time() - start_time
        
        insertion_results[phase['name']] = {
            "vectors_inserted": len(vectors),
            "insert_time_ms": insert_time * 1000,
            "vectors_per_second": len(vectors) / insert_time
        }
        
        total_vectors += len(vectors)
        print(f"✅ {phase['name']}: {len(vectors)} vectors in {insert_time:.2f}s")
    
    print(f"\n📊 Total vectors inserted: {total_vectors}")
    
    # Test immediate search
    print(f"\n🔍 Testing immediate search before restart...")
    test_query = [float(i) for i in range(128)]
    
    try:
        immediate_results = client.search(collection_name, test_query, top_k=10)
        print(f"✅ Immediate search: {len(immediate_results)} results")
        
        if len(immediate_results) > 0:
            print("   Top results:")
            for i, result in enumerate(immediate_results):
                phase = result.metadata.get('phase', 'unknown') if hasattr(result, 'metadata') and result.metadata else 'unknown'
                print(f"   {i+1}. {result.id} (phase: {phase})")
        
    except Exception as e:
        print(f"❌ Immediate search failed: {e}")
    
    # Save test state
    test_state = {
        "collection_name": collection_name,
        "total_vectors_inserted": total_vectors,
        "insertion_phases": insertion_results,
        "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ready_for_restart": True
    }
    
    with open("vector_wal_test_state.json", "w") as f:
        json.dump(test_state, f, indent=2)
    
    print(f"\n📊 Test state saved to: vector_wal_test_state.json")
    print(f"🔄 RESTART SERVER NOW to test vector WAL recovery!")
    print(f"   Then run this test again to verify {total_vectors} vectors are recovered")
    
    return {
        "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "collection_found": False,
        "vectors_inserted": total_vectors,
        "ready_for_restart": True
    }

def analyze_wal_recovery():
    """Analyze WAL recovery results"""
    
    print("\n📊 WAL Recovery Analysis")
    print("="*60)
    
    # Load test state
    try:
        with open("vector_wal_test_state.json", "r") as f:
            test_state = json.load(f)
        
        print(f"📄 Test state loaded:")
        print(f"   - Original vectors: {test_state['total_vectors_inserted']}")
        print(f"   - Test time: {test_state['test_timestamp']}")
        print(f"   - Phases: {list(test_state['insertion_phases'].keys())}")
        
    except FileNotFoundError:
        print("❌ No test state file found")
        return
    
    # Connect and test recovery
    client = connect_rest("http://localhost:5678")
    collection_name = test_state["collection_name"]
    
    try:
        # Get collection
        collection = client.get_collection(collection_name)
        print(f"✅ Collection recovered: {collection.config.name}")
        
        # Test search with different queries
        test_queries = [
            ([float(i) for i in range(128)], "sequential"),
            ([1.0] * 128, "uniform"),
            ([float(i % 10) for i in range(128)], "modulo")
        ]
        
        recovery_summary = {
            "total_expected": test_state['total_vectors_inserted'],
            "total_recovered": 0,
            "phase_recovery": {},
            "search_tests": []
        }
        
        for query, query_name in test_queries:
            print(f"\n🔍 Testing {query_name} query...")
            
            try:
                # Normalize query
                norm = np.linalg.norm(query)
                if norm > 0:
                    query = [x / norm for x in query]
                
                results = client.search(collection_name, query, top_k=20)
                
                print(f"   Found {len(results)} results")
                
                # Analyze by phase
                phase_counts = {}
                for result in results:
                    if hasattr(result, 'metadata') and result.metadata:
                        phase = result.metadata.get('phase', 'unknown')
                        phase_counts[phase] = phase_counts.get(phase, 0) + 1
                
                recovery_summary["search_tests"].append({
                    "query_name": query_name,
                    "results_count": len(results),
                    "phase_breakdown": phase_counts
                })
                
                if phase_counts:
                    print(f"   Phase breakdown: {phase_counts}")
                
                # Update total recovered (use max found)
                recovery_summary["total_recovered"] = max(
                    recovery_summary["total_recovered"], 
                    len(results)
                )
                
            except Exception as e:
                print(f"   ❌ Search failed: {e}")
        
        # Summary
        expected = recovery_summary["total_expected"]
        recovered = recovery_summary["total_recovered"]
        recovery_rate = (recovered / expected) * 100 if expected > 0 else 0
        
        print(f"\n📊 VECTOR WAL RECOVERY SUMMARY:")
        print(f"   - Expected vectors: {expected}")
        print(f"   - Recovered vectors: {recovered}")
        print(f"   - Recovery rate: {recovery_rate:.1f}%")
        
        if recovery_rate > 90:
            print(f"   ✅ EXCELLENT WAL RECOVERY")
        elif recovery_rate > 70:
            print(f"   ⚠️  GOOD WAL RECOVERY")
        else:
            print(f"   ❌ POOR WAL RECOVERY")
        
        # Save recovery results
        with open("vector_wal_recovery_results.json", "w") as f:
            json.dump(recovery_summary, f, indent=2)
        
        return recovery_summary
        
    except Exception as e:
        print(f"❌ Collection recovery failed: {e}")
        return None

if __name__ == "__main__":
    # Run the test
    result = test_vector_wal_recovery()
    
    # If we found existing data, analyze the recovery
    if result and result.get("recovery_successful"):
        analyze_wal_recovery()
    
    print("\n📋 NEXT STEPS:")
    print("  1. If vectors were just inserted, restart the server")
    print("  2. Run this test again to verify WAL recovery")
    print("  3. Check vector_wal_recovery_results.json for detailed analysis")