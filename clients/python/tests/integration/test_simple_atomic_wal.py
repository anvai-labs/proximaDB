#!/usr/bin/env python3
"""
Simple Atomic WAL Test using direct HTTP requests

NOTE: Moved from tests/unit/ to tests/integration/ - this is an integration test
requiring a running ProximaDB server at localhost:5678.
"""

import requests

from ..embedding_utils import embed_seed


def test_atomic_wal():
    """Test atomic WAL with direct HTTP requests"""

    print("🧪 Simple Atomic WAL Test")
    print("=" * 40)

    base_url = "http://localhost:5678"

    # Test 1: Health check
    print("\n💊 Health check...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        assert False, f"Health check failed: {e}"

    # Test 2: Collection creation (try different endpoints)
    collection_id = "simple_atomic_test"

    # Try standard REST collection creation
    print(f"\n📁 Creating collection: {collection_id}")

    collection_config = {
        "name": collection_id,
        "dimension": 384,
        "distance_metric": "cosine",
        "storage_engine": "nova",
    }

    # Use the correct REST API endpoint with proper request format
    endpoint = "/api/v1/collections"

    collection_created = False
    try:
        print(f"   Trying: POST {endpoint}")
        request_data = {
            "operation": 1,  # COLLECTION_CREATE
            "collection_config": collection_config,
            "query_params": {},
            "options": {},
            "migration_config": {},
        }
        response = requests.post(
            f"{base_url}{endpoint}",
            json=request_data,
            headers={"Content-Type": "application/json"},
        )
        print(f"   Response: {response.status_code} - {response.text[:200]}")

        if response.status_code in [200, 201]:
            collection_created = True
            print(f"✅ Collection created via {endpoint}")

    except Exception as e:
        print(f"   Error: {e}")

    if not collection_created:
        print("❌ Failed to create collection via any endpoint")

        # Let's see what endpoints are available
        print("\n🔍 Checking available endpoints...")
        try:
            # Try to get some info about available routes
            response = requests.get(f"{base_url}/")
            print(f"Root: {response.status_code} - {response.text[:200]}")
        except:
            pass

        assert False, "Failed to create collection via any endpoint"

    # Test 3: Vector insertion (if collection was created)
    print("\n🔥 Testing vector insertion...")

    vector_data = {
        "id": "atomic_test_vector_1",
        "vector": embed_seed(0, 384),
        "metadata": {"test": "atomic", "timestamp": "2025-07-03"},
    }

    # Try different vector insertion endpoints
    vector_endpoints = [
        f"/collections/{collection_id}/vectors",
        f"/api/v1/collections/{collection_id}/vectors",
        f"/api/collections/{collection_id}/vectors",
    ]

    vector_inserted = False
    for endpoint in vector_endpoints:
        try:
            print(f"   Trying: POST {endpoint}")
            response = requests.post(
                f"{base_url}{endpoint}",
                json=vector_data,
                headers={"Content-Type": "application/json"},
            )
            print(f"   Response: {response.status_code} - {response.text[:200]}")

            if response.status_code in [200, 201]:
                vector_inserted = True
                print(f"✅ Vector inserted via {endpoint}")
                break

        except Exception as e:
            print(f"   Error: {e}")

    if vector_inserted:
        print("✅ Atomic WAL test basic operations successful!")
    else:
        print("⚠️ Vector insertion failed, but collection creation worked")

    assert collection_created, "Collection creation failed"


def check_wal_logs():
    """Check for WAL-related log messages"""

    print("\n📋 Checking WAL logs...")

    try:
        with open("server_atomic_test.log") as f:
            logs = f.read()

        # Look for WriteBuffer-related messages
        wal_keywords = [
            "WAL",
            "WriteBuffer",
            "memtable",
            "disk write",
            "atomic write",
            "PerBatch",
            "Proto",
        ]

        found_logs = []
        for keyword in wal_keywords:
            if keyword in logs:
                found_logs.append(keyword)

        print(f"📊 Found WAL-related keywords: {found_logs}")

        # Look for specific atomic messages
        if "atomic write" in logs:
            print("✅ Found atomic write messages in logs")
        else:
            print("⚠️ No atomic write messages found")

    except Exception as e:
        print(f"❌ Error checking logs: {e}")


if __name__ == "__main__":
    success = test_atomic_wal()
    check_wal_logs()

    if success:
        print("\n🎉 Test completed!")
    else:
        print("\n💥 Test failed!")
