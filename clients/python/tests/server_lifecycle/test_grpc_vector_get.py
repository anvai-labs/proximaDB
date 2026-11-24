#!/usr/bin/env python3
"""
Test gRPC VectorGet functionality after fix
"""

import atexit
import os
import subprocess
import time

import numpy as np

from proximadb_sdk import ProximaDBClient

# Server process
server_process = None


def start_server():
    """Start ProximaDB server for testing

    NOTE: This function kills and restarts the server.
    This test is in tests/server_lifecycle/ and should be run separately.
    """
    global server_process
    print("🚀 Starting ProximaDB server...")

    # Kill any existing server (OK in server_lifecycle tests)
    os.system("pkill -f proximadb-server")
    time.sleep(1)

    # Find project root (assumes test is in clients/python/tests/server_lifecycle/)
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(test_dir, "../../../.."))
    server_binary = os.path.join(project_root, "target/release/proximadb-server")
    config_file = os.path.join(project_root, "config/config.toml")

    if not os.path.exists(server_binary):
        print(f"❌ Server binary not found at: {server_binary}")
        print("   Please build with: cargo build --release")
        return

    # Start server
    server_process = subprocess.Popen(
        [server_binary, "--config", config_file],
        cwd=project_root,
        env={**os.environ, "RUST_LOG": "info"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to start
    time.sleep(3)
    print("✅ Server started")


def stop_server():
    """Stop ProximaDB server

    NOTE: This is OK in server_lifecycle tests which run in isolation.
    """
    global server_process
    if server_process:
        print("🛑 Stopping server...")
        server_process.terminate()
        server_process.wait()
        server_process = None


# Register cleanup - OK in server_lifecycle tests
atexit.register(stop_server)


def test_grpc_vector_get():
    """Test gRPC vector insert and get operations"""

    print("\n🔍 Testing gRPC VectorGet Fix")
    print("=" * 60)

    # Start server
    start_server()

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
    test_vectors = [
        {
            "id": "vec_001",
            "vector": np.random.rand(128).tolist(),
            "metadata": {"type": "test", "index": 1},
        },
        {
            "id": "vec_002",
            "vector": np.random.rand(128).tolist(),
            "metadata": {"type": "test", "index": 2},
        },
    ]

    # Test 1: gRPC insert -> gRPC get
    print("\n🧪 Test 1: gRPC insert -> gRPC get")
    grpc_client.insert_vectors(collection_name, test_vectors)
    print("✅ Vectors inserted via gRPC")

    time.sleep(1)  # Allow time for processing

    # Try to get vectors via gRPC
    for vec in test_vectors:
        try:
            result = grpc_client.get_vector(collection_name, vec["id"])
            if result:
                print(f"✅ Found vector {vec['id']} via gRPC")
                print(f"   - Has vector: {result.get('vector') is not None}")
                print(f"   - Has metadata: {result.get('metadata') is not None}")
            else:
                print(f"❌ Vector {vec['id']} NOT FOUND via gRPC")
        except Exception as e:
            print(f"❌ Error getting vector {vec['id']}: {e}")

    # Test 2: REST get for comparison
    print("\n🧪 Test 2: Same vectors via REST")
    for vec in test_vectors:
        try:
            result = rest_client.get_vector(collection_name, vec["id"])
            if result:
                print(f"✅ Found vector {vec['id']} via REST")
            else:
                print(f"❌ Vector {vec['id']} NOT FOUND via REST")
        except Exception as e:
            print(f"❌ Error getting vector {vec['id']}: {e}")

    # Test 3: Search to verify vectors exist
    print("\n🧪 Test 3: Search to verify vectors exist")
    search_results = grpc_client.search(
        collection_name, query_vector=test_vectors[0]["vector"], top_k=5
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
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        stop_server()
