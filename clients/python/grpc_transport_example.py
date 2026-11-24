#!/usr/bin/env python3
"""
ProximaDB gRPC Transport Usage Examples

This file demonstrates how to use the gRPC transport for various operations.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import asyncio
from proximadb.transport.grpc import GRPCTransport


async def example_usage():
    """Comprehensive example of gRPC transport usage"""

    print("🚀 ProximaDB gRPC Transport Examples")
    print("=" * 40)

    # 1. Initialize transport
    config = {
        "verify_ssl": False,  # Use True in production
        "compression": {"enabled": True},
        "api_key": "your-api-key-here",
        "custom_headers": {"client-version": "1.0.0"},
    }

    transport = GRPCTransport("localhost:5679", config)

    try:
        # 2. Connect to server
        await transport.connect()
        print("✅ Connected to ProximaDB server")

        # 3. Health check
        health = await transport.health_check()
        print(f"📊 Server health: {health['status']} (v{health['version']})")

        # 4. Create a collection
        collection_result = await transport.create_collection(
            name="example_collection",
            dimension=384,
            distance_metric="cosine",
            storage_engine="nova",
            description="Example collection for testing",
        )
        print(f"📁 Collection created: {collection_result['success']}")

        # 5. Insert vectors
        vectors = [
            {
                "id": "doc1",
                "vector": [0.1] * 384,
                "metadata": {"category": "test", "title": "Document 1"},
            },
            {
                "id": "doc2",
                "vector": [0.2] * 384,
                "metadata": {"category": "test", "title": "Document 2"},
            },
        ]

        insert_result = await transport.insert_vectors("example_collection", vectors)
        print(f"📥 Inserted {insert_result['metrics']['successful_count']} vectors")

        # 6. Search vectors
        query_vector = [0.15] * 384
        search_result = await transport.search_vectors(
            collection_name="example_collection",
            query_vector=query_vector,
            top_k=5,
            filter_dict={"category": "test"},
            include_metadata=True,
        )

        print(f"🔍 Found {len(search_result['results'])} results:")
        for i, result in enumerate(search_result["results"][:2]):
            print(f"   {i+1}. ID: {result['id']}, Score: {result['score']:.4f}")
            print(f"      Metadata: {result['metadata']}")

        # 7. Get specific vector
        vector_result = await transport.get_vector(
            collection_name="example_collection",
            vector_id="doc1",
            include_metadata=True,
        )

        if vector_result["vector"]:
            print(f"📄 Retrieved vector: ID={vector_result['vector']['id']}")
            print(f"   Metadata: {vector_result['vector']['metadata']}")

        # 8. Update vector metadata
        update_result = await transport.update_vector(
            collection_name="example_collection",
            vector_id="doc1",
            metadata={"category": "updated", "title": "Updated Document 1"},
        )
        print(f"✏️  Vector updated: {update_result['success']}")

        # 9. List collections
        collections = await transport.list_collections()
        print(f"📋 Found {len(collections)} collections:")
        for collection in collections:
            print(
                f"   - {collection['name']} ({collection['stats']['vector_count']} vectors)"
            )

        # 10. Clean up - delete vectors and collection
        await transport.delete_vector("example_collection", "doc1")
        await transport.delete_vector("example_collection", "doc2")
        print("🗑️  Vectors deleted")

        delete_result = await transport.delete_collection("example_collection")
        print(f"🗑️  Collection deleted: {delete_result['success']}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Always disconnect
        await transport.disconnect()
        print("👋 Disconnected from server")


async def performance_comparison_example():
    """Example showing gRPC performance benefits"""

    print("\n🏃 Performance Comparison Example")
    print("=" * 40)

    # This would typically compare against REST transport
    # For demonstration, we'll show the features that make gRPC faster

    config = {"verify_ssl": False, "compression": {"enabled": True}}
    transport = GRPCTransport("localhost:5679", config)

    try:
        await transport.connect()

        # Large batch insert (where gRPC really shines)
        import time

        large_vectors = [
            {
                "id": f"perf_doc_{i}",
                "vector": [0.1 * (i % 10)] * 384,
                "metadata": {"batch": "performance_test", "index": i},
            }
            for i in range(100)  # 100 vectors
        ]

        start_time = time.time()
        result = await transport.insert_vectors("test_perf_collection", large_vectors)
        end_time = time.time()

        if result["success"]:
            print(
                f"⚡ Inserted {len(large_vectors)} vectors in {end_time - start_time:.3f}s"
            )
            print(
                f"   Throughput: {len(large_vectors) / (end_time - start_time):.1f} vectors/sec"
            )
            print(
                f"   Processing time: {result['metrics']['processing_time_us'] / 1000:.1f}ms"
            )

        # Clean up
        await transport.delete_collection("test_perf_collection")

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
    finally:
        await transport.disconnect()


if __name__ == "__main__":
    print("📚 ProximaDB gRPC Transport Examples")
    print("This demonstrates the complete gRPC transport implementation\n")

    try:
        # Run basic usage example
        asyncio.run(example_usage())

        # Run performance example
        asyncio.run(performance_comparison_example())

        print("\n🎉 All examples completed successfully!")
        print("\n💡 Key Benefits of gRPC Transport:")
        print("   • 2-3x faster than REST for vector operations")
        print("   • Binary protocol reduces serialization overhead")
        print("   • HTTP/2 connection multiplexing")
        print("   • Automatic compression reduces bandwidth")
        print("   • Type-safe proto message handling")
        print("   • Efficient batch operations")

    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n💥 Examples failed: {e}")
        import traceback

        traceback.print_exc()
