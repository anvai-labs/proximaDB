#!/usr/bin/env python3
"""
Simple test for gRPC VectorGet functionality
"""

import time
import numpy as np
from proximadb import ProximaDBClient


def test_grpc_vector_get():
    """Test gRPC vector insert and get operations"""

    print("\n🔍 Testing gRPC VectorGet Fix")
    print("=" * 60)

    # Initialize clients
    grpc_client = ProximaDBClient("grpc://localhost:5679", protocol="grpc")
    rest_client = ProximaDBClient("http://localhost:5678", protocol="rest")

    # Create collection
    collection_name = f"test_grpc_get_{int(time.time())}"
    print(f"\n📦 Creating collection: {collection_name}")

    grpc_client.create_collection(
        name=collection_name,
        dimension=128,
        distance_metric="cosine",
        storage_engine="nova",
    )

    # Test vectors
    vectors = [np.random.rand(128).tolist(), np.random.rand(128).tolist()]
    ids = ["vec_001", "vec_002"]
    metadatas = [{"type": "test", "index": 1}, {"type": "test", "index": 2}]

    # Test 1: gRPC insert -> gRPC get
    print("\n🧪 Test 1: gRPC insert -> gRPC get")
    grpc_client.insert_vectors(collection_name, vectors, ids=ids, metadata=metadatas)
    print("✅ Vectors inserted via gRPC")

    time.sleep(1)  # Allow time for processing

    # Try to get vectors via gRPC
    for vec_id in ids:
        try:
            result = grpc_client.get_vector(collection_name, vec_id)
            if result:
                print(f"✅ Found vector {vec_id} via gRPC")
                print(f"   - ID: {result.get('id')}")
                print(f"   - Has vector: {result.get('vector') is not None}")
                print(f"   - Has metadata: {result.get('metadata') is not None}")
            else:
                print(f"❌ Vector {vec_id} NOT FOUND via gRPC")
        except Exception as e:
            print(f"❌ Error getting vector {vec_id}: {e}")

    # Test 2: REST get for comparison
    print("\n🧪 Test 2: Same vectors via REST")
    for vec_id in ids:
        try:
            result = rest_client.get_vector(collection_name, vec_id)
            if result:
                print(f"✅ Found vector {vec_id} via REST")
            else:
                print(f"❌ Vector {vec_id} NOT FOUND via REST")
        except Exception as e:
            print(f"❌ Error getting vector {vec_id}: {e}")

    # Test 3: Search to verify vectors exist
    print("\n🧪 Test 3: Search to verify vectors exist")
    search_results = grpc_client.search_vectors(
        collection_name, query_vector=vectors[0], top_k=5
    )
    print(f"📊 Search found {len(search_results)} results")
    for i, result in enumerate(search_results):
        print(
            f"   [{i+1}] ID: {result.get('id', 'Unknown')}, Score: {result.get('score', 0):.4f}"
        )

    print("\n✅ Test completed")


if __name__ == "__main__":
    try:
        test_grpc_vector_get()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
