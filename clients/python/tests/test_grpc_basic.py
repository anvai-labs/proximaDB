#!/usr/bin/env python3
"""
Basic gRPC API Test
Tests fundamental gRPC operations using unified client
"""

import time

import numpy as np

from proximadb_sdk import Protocol, ProximaDBClient, VectorRecord


def test_grpc_basic():
    """Test basic gRPC operations"""

    print("🧪 Testing Basic gRPC API with Unified Client")
    print("=" * 60)

    # Initialize client with gRPC protocol
    client = ProximaDBClient(url="localhost", protocol=Protocol.GRPC)
    print("✅ gRPC client initialized")

    # Test 1: Health check
    print("\n1️⃣ Testing health check...")
    try:
        health = client.health()
        print(f"✅ Server is healthy: {health.status}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")

    # Test 2: Create collection
    print("\n2️⃣ Creating collection...")
    collection_name = f"grpc_test_{int(time.time())}"

    try:
        collection = client.create_collection(
            name=collection_name,
            dimension=128,
            distance_metric="cosine",
            storage_engine="nova",
        )
        print(f"✅ Collection created: {collection.config.name}")
        print(f"   ID: {collection.id}")
        print(f"   Dimension: {collection.config.dimension}")
    except Exception as e:
        print(f"❌ Collection creation failed: {e}")
        return

    # Test 3: Insert vectors using unified client API
    print("\n3️⃣ Testing vector insertion...")

    # Prepare test vectors as VectorRecord objects
    records = []
    for i in range(10):
        record = VectorRecord(
            id=f"vec_{i}",
            vector=np.random.rand(128).astype(np.float32).tolist(),
            metadata={"index": str(i), "type": "test"},
        )
        records.append(record)

    try:
        response = client.insert_vectors(collection_id=collection_name, records=records)

        print("✅ Vectors inserted successfully")
        print(f"   Successful: {response.metrics.successful_count}")
        print(f"   Failed: {response.metrics.failed_count}")
        if hasattr(response.metrics, "processing_time_us"):
            print(f"   Time: {response.metrics.processing_time_us / 1000:.2f}ms")
    except Exception as e:
        print(f"❌ Vector insertion exception: {e}")

    # Test 4: Search using unified client API
    print("\n4️⃣ Testing vector search...")
    try:
        query_vector = np.random.rand(128).astype(np.float32).tolist()

        results = client.search_single(
            collection_id=collection_name, vector=query_vector, top_k=5
        )

        print(f"✅ Search completed, found {len(results)} results")
        for i, result in enumerate(results[:3]):
            print(f"   {i+1}. ID: {result.id}, Score: {result.score:.4f}")
    except Exception as e:
        print(f"❌ Search failed: {e}")

    # Test 5: Get collection
    print("\n5️⃣ Getting collection info...")
    try:
        collection = client.get_collection(collection_name)
        if collection:
            print(f"✅ Collection retrieved: {collection.config.name}")
            print(f"   Storage engine: {collection.config.storage_engine}")
        else:
            print("❌ Collection not found")
    except Exception as e:
        print(f"❌ Get collection failed: {e}")

    # Test 6: List collections
    print("\n6️⃣ Listing collections...")
    try:
        collections = client.list_collections()
        print(f"✅ Found {len(collections)} collections")
        for coll in collections[:3]:
            print(f"   - {coll.config.name} (dim: {coll.config.dimension})")
    except Exception as e:
        print(f"❌ List collections failed: {e}")

    # Test 7: Delete collection
    print("\n7️⃣ Cleaning up...")
    try:
        success = client.delete_collection(collection_name)
        if success:
            print(f"✅ Collection deleted: {collection_name}")
        else:
            print("❌ Failed to delete collection")
    except Exception as e:
        print(f"❌ Delete collection failed: {e}")

    print("\n✅ Basic gRPC test completed!")


if __name__ == "__main__":
    test_grpc_basic()
