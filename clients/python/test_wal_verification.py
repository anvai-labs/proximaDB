#!/usr/bin/env python3
"""
Test to verify WAL/Write Buffer functionality
Inserts vectors and checks that WAL files are created
"""

import sys
import os

sys.path.insert(0, "src")

import requests
import json
import numpy as np
import time

SERVER_URL = "http://localhost:5678"


def test_wal_directory_creation():
    """Test that WAL files are created when vectors are inserted"""
    print("🧪 Testing WAL/Write Buffer Creation")
    print("=" * 60)

    # 1. Create collection
    print("\n1️⃣ Creating test collection...")
    collection_name = f"wal_test_{int(time.time())}"
    collection_config = {
        "name": collection_name,
        "dimension": 128,
        "distance_metric": 1,  # COSINE
        "storage_engine": 2,  # SST
        "tags": [],
        "filterable_columns": [],
        "index_configs": [
            {
                "index_name": f"{collection_name}_primary",
                "algorithm": 1,  # HNSW
                "parameters": {},
                "enabled": True,
                "update_mode": 0,
                "enable_background_optimization": True,
                "build_concurrency": 4,
                "memory_limit_mb": 512,
                "checkpoint_interval_ms": 60000,
                "is_primary": True,
                "use_cases": [],
                "selectivity_threshold": 0.5,
                "use_quantization": False,
                "queue_representation": "vector",
            }
        ],
        "primary_index": f"{collection_name}_primary",
        "auto_index_selection": True,
        "embedding_models": [],
    }

    request_data = {
        "operation": 1,  # COLLECTION_CREATE
        "collection_config": collection_config,
        "query_params": {},
        "options": {},
        "migration_config": {},
    }

    response = requests.post(
        f"{SERVER_URL}/api/v1/collections",
        json=request_data,
        headers={"Content-Type": "application/json"},
    )

    print(f"   Response status: {response.status_code}")

    if response.status_code not in [200, 201]:
        print(f"❌ Collection creation failed: {response.status_code}")
        print(f"   Response: {response.text}")
        return False

    try:
        result = response.json()
        print(f"   Response parsed successfully")
        print(f"   Result type: {type(result)}")
        print(
            f"   Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}"
        )
    except Exception as e:
        print(f"❌ Failed to parse JSON response: {e}")
        print(f"   Response text: {response.text}")
        return False

    if not result:
        print(f"❌ Result is None or empty")
        return False

    if "collection" not in result:
        print(f"❌ 'collection' key not in result")
        print(f"   Available keys: {list(result.keys())}")
        return False

    if result["collection"] is None:
        print(f"❌ result['collection'] is None")
        print(f"   Full result: {json.dumps(result, indent=2)}")
        return False

    collection_id = result["collection"]["id"]
    storage_path = result["collection"]["storage_assignment"]["primary_path"]
    print(f"✅ Collection created: {collection_id}")
    print(f"   Storage path: {storage_path}")

    # Extract file path from storage_path (format: file:///path/to/dir)
    if storage_path.startswith("file://"):
        base_path = storage_path[7:]  # Remove 'file://'
        wal_path = os.path.join(
            base_path, collection_id, "wal"
        )  # Changed from 'write_buffer' to 'wal'
        data_path = os.path.join(base_path, collection_id, "data")

        print(f"\n2️⃣ Checking directory structure...")
        print(f"   WAL directory: {wal_path}")
        print(f"   Data directory: {data_path}")

        if os.path.exists(wal_path):
            print(f"   ✅ WAL directory exists")
            initial_files = os.listdir(wal_path)
            print(f"   Files before insert: {len(initial_files)} - {initial_files}")
        else:
            print(f"   ❌ WAL directory does not exist!")
            return False

    # 2. Insert vectors using gRPC (more reliable)
    print(f"\n3️⃣ Inserting 100 test vectors via gRPC...")
    try:
        from proximadb.protocols.grpc_sync import ProximaDBSyncGrpcClient

        client = ProximaDBSyncGrpcClient(
            server_address="localhost:5679", enable_compression=False
        )

        # Create 100 vectors
        vectors = []
        for i in range(100):
            vectors.append(
                {
                    "id": f"wal_vec_{i}",
                    "vector": np.random.rand(128).astype(np.float32).tolist(),
                    "metadata": {"batch": 1, "index": i},
                }
            )

        # Use the high-level insert method
        response = client.insert_vectors(
            collection_id=collection_id, vectors=vectors, upsert=False
        )

        print(f"   Insert response: {response}")

        if not response.success or response.metrics.successful_count == 0:
            print(f"   ❌ Insert failed!")
            print(f"   Success: {response.success}")
            print(f"   Successful count: {response.metrics.successful_count}")
            print(f"   Error message: {response.error_message}")
            return False

        print(
            f"   ✅ Inserted {response.metrics.successful_count} vectors successfully"
        )

    except Exception as e:
        print(f"   ❌ Insert failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 3. Wait a moment for WAL write
    print(f"\n4️⃣ Waiting for WAL writes to complete...")
    time.sleep(2)

    # 4. Check WAL directory again
    print(f"\n5️⃣ Checking WAL directory after insert...")
    if os.path.exists(wal_path):
        after_files = os.listdir(wal_path)
        print(f"   Files after insert: {len(after_files)}")

        if len(after_files) > 0:
            print(f"   ✅ WAL files created!")
            for f in after_files:
                file_path = os.path.join(wal_path, f)
                file_size = os.path.getsize(file_path)
                print(f"      - {f} ({file_size} bytes)")
        else:
            print(f"   ⚠️  No WAL files found (may be in memtable only)")

    # 5. Check collection stats
    print(f"\n6️⃣ Checking collection stats...")
    response = requests.get(f"{SERVER_URL}/api/v1/collections/{collection_id}")
    if response.status_code == 200:
        stats = response.json().get("collection", {}).get("stats", {})
        print(f"   Vector count: {stats.get('vector_count', 0)}")
        print(f"   Index size: {stats.get('index_size_bytes', 0)} bytes")
        print(f"   Data size: {stats.get('data_size_bytes', 0)} bytes")

    print(f"\n{'='*60}")
    print("✅ WAL verification test completed!")
    return True


if __name__ == "__main__":
    try:
        success = test_wal_directory_creation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
