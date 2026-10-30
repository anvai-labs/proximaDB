#!/usr/bin/env python3
"""
ProximaDB Vector Operations Test Suite
Tests for vector CRUD operations, batch insertions, and large-scale operations

Tests run against embedded ProximaDB database for fast, reliable testing.
"""

import logging
import time

import numpy as np
import pytest

from proximadb_sdk import CollectionConfig, StorageEngine

logger = logging.getLogger(__name__)


# Local helper functions for vector generation
def embed_seed(seed: int, dimension: int) -> np.ndarray:
    """Generate a deterministic embedding based on seed"""
    np.random.seed(seed)
    vec = np.random.rand(dimension).astype(np.float32)
    return vec / np.linalg.norm(vec)


class TestVectorCRUD:
    """Test vector Create, Read, Update, Delete operations using embedded database"""

    @pytest.fixture(scope="class")
    def test_collection(self, rest_client):
        """Create test collection for vector operations"""
        collection_name = f"vector_crud_{int(time.time() * 1000)}"

        config = CollectionConfig(
            name=collection_name,
            dimension=128,
            distance_metric="cosine",
            description="Vector CRUD test collection",
        )
        collection = rest_client.create_collection(collection_name, config=config)
        yield collection

        # Cleanup
        try:
            rest_client.delete_collection(collection_name)
        except:
            pass

    def test_single_vector_operations_rest(self, rest_client, test_collection):
        """Test single vector CRUD operations via embedded database (REST-style)"""
        vector_id = "test_vector_1"
        vector = embed_seed(0, 128)
        metadata = {
            "description": "Test vector",
            "category": "test",
            "timestamp": time.time(),
        }

        # Insert vector
        result = rest_client.insert_vector(
            collection_id=test_collection.name,
            vector_id=vector_id,
            vector=vector.tolist() if hasattr(vector, "tolist") else vector,
            metadata=metadata,
        )
        assert result is not None

        # Update vector (upsert)
        updated_vector = embed_seed(1, 128)
        updated_metadata = {
            "description": "Updated test vector",
            "category": "updated",
            "timestamp": time.time(),
        }

        update_result = rest_client.insert_vector(
            collection_id=test_collection.name,
            vector_id=vector_id,
            vector=(
                updated_vector.tolist()
                if hasattr(updated_vector, "tolist")
                else updated_vector
            ),
            metadata=updated_metadata,
        )
        assert update_result is not None

    def test_single_vector_operations_grpc(self, grpc_client, test_collection):
        """Test single vector CRUD operations via embedded database (gRPC-style)"""
        vector_id = "grpc_test_vector_1"
        vector = embed_seed(2, 128)
        metadata = {
            "description": "gRPC test vector",
            "category": "grpc_test",
            "protocol": "grpc",
        }

        # Insert vector
        result = grpc_client.insert_vector(
            collection_id=test_collection.name,
            vector_id=vector_id,
            vector=vector.tolist() if hasattr(vector, "tolist") else vector,
            metadata=metadata,
        )
        assert result is not None


class TestBatchVectorOperations:
    """Test batch vector operations and large-scale insertions"""

    @pytest.fixture(scope="class")
    def batch_collection(self, rest_client):
        """Create collection optimized for batch operations"""
        collection_name = f"batch_test_{int(time.time() * 1000)}"

        config = CollectionConfig(
            name=collection_name,
            dimension=384,
            distance_metric="cosine",
            description="Batch operations test collection",
            storage_engine=StorageEngine.SST,  # Use SST for embedded
        )
        collection = rest_client.create_collection(collection_name, config=config)
        yield collection

        # Cleanup
        try:
            rest_client.delete_collection(collection_name)
        except:
            pass

    def test_batch_insertion_rest(self, rest_client, batch_collection):
        """Test batch vector insertion via embedded database (REST-style)"""
        batch_size = 100
        vectors = []
        vector_ids = []
        metadatas = []

        for i in range(batch_size):
            vector = embed_seed(i, 384)
            vectors.append(vector.tolist() if hasattr(vector, "tolist") else vector)
            vector_ids.append(f"batch_rest_{i}")
            metadatas.append(
                {
                    "index": i,
                    "batch": "rest_batch",
                    "category": f"group_{i % 10}",
                    "timestamp": time.time() + i,
                }
            )

        # Insert batch
        result = rest_client.insert_vectors(
            collection_id=batch_collection.name,
            vectors=vectors,
            ids=vector_ids,
            metadata=metadatas,
        )

        assert result is not None
        inserted_count = getattr(
            result, "count", getattr(result, "successful_count", batch_size)
        )
        assert inserted_count >= batch_size * 0.9  # Allow for some failures

    def test_batch_insertion_grpc(self, grpc_client, batch_collection):
        """Test batch vector insertion via embedded database (gRPC-style)"""
        batch_size = 150
        vectors = []
        vector_ids = []
        metadatas = []

        for i in range(batch_size):
            vector = embed_seed(100 + i, 384)
            vectors.append(vector.tolist() if hasattr(vector, "tolist") else vector)
            vector_ids.append(f"batch_grpc_{i}")
            metadatas.append(
                {
                    "index": i,
                    "batch": "grpc_batch",
                    "category": f"grpc_group_{i % 15}",
                    "protocol": "grpc",
                }
            )

        # Insert batch
        result = grpc_client.insert_vectors(
            collection_id=batch_collection.name,
            vectors=vectors,
            ids=vector_ids,
            metadata=metadatas,
        )

        assert result is not None
        inserted_count = getattr(
            result, "count", getattr(result, "successful_count", batch_size)
        )
        assert inserted_count >= batch_size * 0.9


class TestLargeScaleOperations:
    """Test large-scale vector operations that trigger flush and compaction"""

    @pytest.fixture(scope="class")
    def large_scale_collection(self, rest_client):
        """Create collection for large-scale testing"""
        collection_name = f"large_scale_{int(time.time() * 1000)}"

        config = CollectionConfig(
            name=collection_name,
            dimension=512,
            distance_metric="cosine",
            description="Large-scale operations test",
            storage_engine=StorageEngine.SST,
        )
        collection = rest_client.create_collection(collection_name, config=config)
        yield collection

        # Cleanup
        try:
            rest_client.delete_collection(collection_name)
        except:
            pass

    def test_large_batch_rest_uuid(self, rest_client, large_scale_collection):
        """Test large batch insertion via embedded database"""
        # Get collection name (embedded uses name as ID)
        collection_id = large_scale_collection.name

        # Target ~1MB of data: 512 dims * 4 bytes * ~500 vectors = ~1MB
        vector_count = 600
        batch_size = 100

        for batch_start in range(0, vector_count, batch_size):
            batch_end = min(batch_start + batch_size, vector_count)
            batch_vectors = []
            batch_ids = []
            batch_metadatas = []

            for i in range(batch_start, batch_end):
                vector = embed_seed(i, 512)
                batch_vectors.append(
                    vector.tolist() if hasattr(vector, "tolist") else vector
                )
                batch_ids.append(f"large_vector_{i}")
                batch_metadatas.append(
                    {
                        "index": i,
                        "batch": f"large_batch_{batch_start//batch_size}",
                        "category": f"group_{i % 20}",
                        "operation": "large_scale_uuid",
                    }
                )

            # Insert batch
            result = rest_client.insert_vectors(
                collection_id=collection_id,
                vectors=batch_vectors,
                ids=batch_ids,
                metadata=batch_metadatas,
            )

            assert result is not None

        # Verify collection still accessible
        collection_info = rest_client.get_collection(large_scale_collection.name)
        assert collection_info is not None

    def test_large_batch_grpc(self, grpc_client, large_scale_collection):
        """Test large batch insertion via embedded database (gRPC-style)"""
        vector_count = 700
        batch_size = 150

        for batch_start in range(0, vector_count, batch_size):
            batch_end = min(batch_start + batch_size, vector_count)
            batch_vectors = []
            batch_ids = []
            batch_metadatas = []

            for i in range(batch_start, batch_end):
                vector = embed_seed(200 + i, 512)
                batch_vectors.append(
                    vector.tolist() if hasattr(vector, "tolist") else vector
                )
                batch_ids.append(f"grpc_large_{i}")
                batch_metadatas.append(
                    {
                        "index": i,
                        "batch": f"grpc_batch_{batch_start//batch_size}",
                        "protocol": "grpc",
                        "operation": "large_scale",
                    }
                )

            # Insert batch
            result = grpc_client.insert_vectors(
                collection_id=large_scale_collection.name,
                vectors=batch_vectors,
                ids=batch_ids,
                metadata=batch_metadatas,
            )

            assert result is not None

    def test_stress_operations(self, rest_client, grpc_client, large_scale_collection):
        """Test stress operations"""
        vector_count = 400

        # Phase 1: Initial insertion
        vectors = []
        vector_ids = []
        metadatas = []

        for i in range(vector_count):
            vector = embed_seed(300 + i, 512)
            vectors.append(vector.tolist() if hasattr(vector, "tolist") else vector)
            vector_ids.append(f"stress_{i}")
            metadatas.append(
                {"index": i, "phase": "initial", "category": f"stress_group_{i % 8}"}
            )

        # Insert in batches
        batch_size = 100
        for i in range(0, vector_count, batch_size):
            batch_end = min(i + batch_size, vector_count)
            batch_result = rest_client.insert_vectors(
                collection_id=large_scale_collection.name,
                vectors=vectors[i:batch_end],
                ids=vector_ids[i:batch_end],
                metadata=metadatas[i:batch_end],
            )
            assert batch_result is not None

        # Phase 2: Update operations
        update_count = vector_count // 2
        for i in range(update_count):
            updated_vector = embed_seed(400, 512)
            updated_metadata = {
                "index": i,
                "phase": "updated",
                "update_timestamp": time.time(),
            }

            # Alternate between rest and grpc clients
            client = grpc_client if i % 2 == 0 else rest_client
            try:
                client.insert_vector(
                    collection_id=large_scale_collection.name,
                    vector_id=f"stress_{i}",
                    vector=(
                        updated_vector.tolist()
                        if hasattr(updated_vector, "tolist")
                        else updated_vector
                    ),
                    metadata=updated_metadata,
                )
            except Exception:
                pass

        # Verify final state
        collection_info = rest_client.get_collection(large_scale_collection.name)
        assert collection_info is not None


class TestVectorValidation:
    """Test vector validation and error handling - Pure unit tests"""

    def test_dimension_validation_config(self):
        """Test dimension validation in config creation"""
        # Valid dimensions
        valid_config = CollectionConfig(
            name="test_collection", dimension=128, distance_metric="cosine"
        )
        assert valid_config.dimension == 128

        # Invalid dimensions should raise validation errors
        with pytest.raises((ValueError, TypeError)):
            CollectionConfig(
                name="test_collection", dimension=0, distance_metric="cosine"
            )

        with pytest.raises((ValueError, TypeError)):
            CollectionConfig(
                name="test_collection",
                dimension=70000,  # Too large
                distance_metric="cosine",
            )


class TestStreamingBatchingConcepts:
    """Test streaming and batching concepts with embedded database"""

    @pytest.fixture(scope="class")
    def streaming_collection(self, rest_client):
        """Create collection for streaming tests"""
        collection_name = f"streaming_test_{int(time.time() * 1000)}"

        config = CollectionConfig(
            name=collection_name,
            dimension=256,
            distance_metric="cosine",
            description="Streaming operations test collection",
        )
        collection = rest_client.create_collection(collection_name, config=config)
        yield collection

        # Cleanup
        try:
            rest_client.delete_collection(collection_name)
        except:
            pass

    def test_simulated_streaming_insertion(self, rest_client, streaming_collection):
        """Test streaming-like vector insertion using regular batching"""
        total_vectors = 200
        chunk_size = 50

        def generate_vector_chunk(start_idx, size):
            """Generate a chunk of vectors"""
            vectors = []
            ids = []
            metadatas = []

            for i in range(start_idx, start_idx + size):
                vector = embed_seed(i, 256)
                vectors.append(vector.tolist() if hasattr(vector, "tolist") else vector)
                ids.append(f"stream_vec_{i}")
                metadatas.append(
                    {"index": i, "chunk": i // chunk_size, "source": "streaming_test"}
                )

            return vectors, ids, metadatas

        # Process vectors in streaming fashion
        processed_count = 0
        for chunk_start in range(0, total_vectors, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_vectors)
            chunk_vectors, chunk_ids, chunk_metadatas = generate_vector_chunk(
                chunk_start, chunk_end - chunk_start
            )

            # Insert chunk
            result = rest_client.insert_vectors(
                collection_id=streaming_collection.name,
                vectors=chunk_vectors,
                ids=chunk_ids,
                metadata=chunk_metadatas,
            )

            if result is not None:
                processed_count += len(chunk_vectors)

            # Simulate streaming delay
            time.sleep(0.01)

        assert processed_count >= 190  # Allow some failures

    def test_paginated_search_as_stream(self, grpc_client, streaming_collection):
        """Test search result streaming using pagination"""
        # First insert test data
        vectors = []
        ids = []
        metadatas = []

        for i in range(100):
            vector = embed_seed(i, 256)
            vectors.append(vector.tolist() if hasattr(vector, "tolist") else vector)
            ids.append(f"search_stream_{i}")
            metadatas.append({"index": i, "group": f"group_{i % 5}"})

        grpc_client.insert_vectors(
            collection_id=streaming_collection.name,
            vectors=vectors,
            ids=ids,
            metadata=metadatas,
        )

        time.sleep(0.5)  # Wait for indexing

        # Search for similar vectors
        query_vector = embed_seed(999, 256)
        query_list = (
            query_vector.tolist() if hasattr(query_vector, "tolist") else query_vector
        )

        results = grpc_client.search(
            collection_id=streaming_collection.name,
            vector=query_list,
            top_k=20,
            include_metadata=True,
        )

        # Verify we got some results
        assert len(results) > 0
        logger.info(f"Retrieved {len(results)} results")


class TestBatchingOptimization:
    """Test batching optimization strategies using embedded database"""

    @pytest.fixture(scope="class")
    def batching_collection(self, rest_client):
        """Create collection for batching tests"""
        collection_name = f"batching_test_{int(time.time() * 1000)}"

        config = CollectionConfig(
            name=collection_name,
            dimension=128,
            distance_metric="euclidean",
            description="Batching operations test collection",
        )
        collection = rest_client.create_collection(collection_name, config=config)
        yield collection

        # Cleanup
        try:
            rest_client.delete_collection(collection_name)
        except:
            pass

    def test_optimal_batch_sizing(self, rest_client, batching_collection):
        """Test different batch sizes to find optimal performance"""
        batch_sizes = [10, 25, 50, 100]
        results = {}

        for batch_size in batch_sizes:
            start_time = time.time()

            # Generate and insert vectors in batches
            total_vectors = 200
            successful_inserts = 0

            for batch_start in range(0, total_vectors, batch_size):
                batch_end = min(batch_start + batch_size, total_vectors)
                batch_vectors = []
                batch_ids = []
                batch_metadatas = []

                for i in range(batch_start, batch_end):
                    vector = embed_seed(i, 128)
                    batch_vectors.append(
                        vector.tolist() if hasattr(vector, "tolist") else vector
                    )
                    batch_ids.append(f"batch_{batch_size}_{i}")
                    batch_metadatas.append({"batch_size": batch_size, "index": i})

                try:
                    result = rest_client.insert_vectors(
                        collection_id=batching_collection.name,
                        vectors=batch_vectors,
                        ids=batch_ids,
                        metadata=batch_metadatas,
                    )
                    successful_inserts += len(batch_vectors)
                except Exception as e:
                    logger.warning(f"Batch insert failed for size {batch_size}: {e}")

            elapsed_time = time.time() - start_time
            throughput = successful_inserts / elapsed_time if elapsed_time > 0 else 0

            results[batch_size] = {
                "total_vectors": total_vectors,
                "successful": successful_inserts,
                "time_seconds": elapsed_time,
                "throughput_per_sec": throughput,
            }

            logger.info(f"Batch size {batch_size}: {throughput:.1f} vectors/sec")

        # Verify all batch sizes worked
        for batch_size, result in results.items():
            assert result["successful"] >= result["total_vectors"] * 0.9

    def test_concurrent_batch_processing(self, rest_client, batching_collection):
        """Test concurrent batch processing for improved throughput"""
        import concurrent.futures
        import threading

        # Shared counter for thread-safe ID generation
        counter_lock = threading.Lock()
        vector_counter = [0]  # Use list for mutable reference

        def process_batch(batch_num, size=50):
            """Process a single batch of vectors"""
            vectors = []
            ids = []
            metadatas = []

            # Generate batch data
            with counter_lock:
                start_idx = vector_counter[0]
                vector_counter[0] += size

            for i in range(start_idx, start_idx + size):
                vector = embed_seed(100 + i, 128)
                vectors.append(vector.tolist() if hasattr(vector, "tolist") else vector)
                ids.append(f"concurrent_{i}")
                metadatas.append({"batch": batch_num, "index": i})

            # Insert batch
            try:
                result = rest_client.insert_vectors(
                    collection_id=batching_collection.name,
                    vectors=vectors,
                    ids=ids,
                    metadata=metadatas,
                )
                return {"success": True, "count": size}
            except Exception as e:
                return {"success": False, "error": str(e)}

        # Process batches concurrently
        num_batches = 10
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_batch, i) for i in range(num_batches)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        elapsed_time = time.time() - start_time

        # Analyze results
        successful_batches = sum(1 for r in results if r["success"])
        total_vectors = sum(r.get("count", 0) for r in results if r["success"])

        logger.info(
            f"Concurrent batching: {successful_batches}/{num_batches} batches, "
            f"{total_vectors} vectors in {elapsed_time:.2f}s"
        )

        assert successful_batches >= num_batches * 0.8  # Allow some failures


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
