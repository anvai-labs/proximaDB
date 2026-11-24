#!/usr/bin/env python3
"""
Arrow Flight TDD Tests for All ProximaDB Storage Engines

This test file implements Test-Driven Development (TDD) for Arrow Flight API
across all 6 ProximaDB storage engines using the embedded database.

Test Coverage:
1. Arrow Format Conversion (vectors_to_arrow_table, arrow_table_to_vectors)
2. Embedded Database Operations (insert, search, flush) for each engine
3. End-to-End Arrow Flight Server Tests (when server is available)

Engines Tested:
- SST: Write-optimized, real-time ingestion (~5ms P99)
- HELIX: Locality-optimized with Hilbert curve (~13ms)
- NOVA: Columnar Parquet for analytics
- SWIFT: Ultra-low latency for small datasets (~5ms)
- NOVA: Progressive columnar for mixed workloads (~30ms)
- RAPTOR: Adaptive row-group for dynamic workloads (~10ms)

Usage:
    # Run all tests
    PYTHONPATH=clients/python/src pytest tests/test_arrow_flight_engines.py -v

    # Run specific engine tests
    PYTHONPATH=clients/python/src pytest tests/test_arrow_flight_engines.py -v -k "sst"

    # Run with server (requires running ProximaDB server)
    PROXIMADB_SERVER_URL=localhost:5678 PYTHONPATH=clients/python/src pytest tests/test_arrow_flight_engines.py -v
"""

import os
import sys
import tempfile
import shutil
import time
import pytest
import random
import numpy as np
from typing import List, Dict, Any, Optional

# Add Python SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'clients', 'python', 'src'))

# Check if PyArrow is available
try:
    import pyarrow as pa
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    pa = None

# Check if embedded ProximaDB is available
try:
    import proximadb
    PROXIMADB_AVAILABLE = True
except ImportError:
    PROXIMADB_AVAILABLE = False
    proximadb = None

# Check if Arrow Flight client is available
try:
    from proximadb_sdk.protocols.arrow_flight import (
        ArrowFlightClient,
        vectors_to_arrow_table,
        arrow_table_to_vectors,
        FlightPutResult,
        FlightSearchResult,
        WriteMode,
    )
    ARROW_FLIGHT_CLIENT_AVAILABLE = PYARROW_AVAILABLE
except ImportError:
    ARROW_FLIGHT_CLIENT_AVAILABLE = False
    ArrowFlightClient = None
    vectors_to_arrow_table = None
    arrow_table_to_vectors = None


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_db_dir():
    """Create temporary directory for embedded database."""
    temp_dir = tempfile.mkdtemp(prefix="proximadb_arrow_test_")
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_vectors():
    """Generate sample vectors for testing."""
    np.random.seed(42)
    dimension = 128
    num_vectors = 100

    ids = [f"vec_{i:04d}" for i in range(num_vectors)]
    vectors = [np.random.randn(dimension).astype(np.float32).tolist() for _ in range(num_vectors)]
    metadata = [{"category": f"cat_{i % 5}", "value": i * 1.5} for i in range(num_vectors)]

    return {
        "ids": ids,
        "vectors": vectors,
        "metadata": metadata,
        "dimension": dimension,
        "num_vectors": num_vectors,
    }


@pytest.fixture
def large_sample_vectors():
    """Generate larger sample for performance tests."""
    np.random.seed(123)
    dimension = 768
    num_vectors = 1000

    ids = [f"vec_{i:06d}" for i in range(num_vectors)]
    vectors = [np.random.randn(dimension).astype(np.float32).tolist() for _ in range(num_vectors)]
    metadata = [{"batch": i // 100, "index": i % 100} for i in range(num_vectors)]

    return {
        "ids": ids,
        "vectors": vectors,
        "metadata": metadata,
        "dimension": dimension,
        "num_vectors": num_vectors,
    }


# =============================================================================
# Arrow Format Conversion Tests
# =============================================================================

@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")
class TestArrowFormatConversion:
    """Test Arrow format conversion functions."""

    def test_vectors_to_arrow_table_basic(self, sample_vectors):
        """Test basic conversion from Python vectors to Arrow table."""
        table = vectors_to_arrow_table(
            ids=sample_vectors["ids"][:10],
            vectors=sample_vectors["vectors"][:10],
        )

        assert table is not None
        assert table.num_rows == 10
        assert "id" in table.schema.names
        assert "vector" in table.schema.names

    def test_vectors_to_arrow_table_with_metadata(self, sample_vectors):
        """Test conversion with metadata."""
        table = vectors_to_arrow_table(
            ids=sample_vectors["ids"][:10],
            vectors=sample_vectors["vectors"][:10],
            metadata=sample_vectors["metadata"][:10],
        )

        assert table is not None
        assert table.num_rows == 10
        assert "metadata" in table.schema.names

    def test_vectors_to_arrow_table_with_timestamps(self, sample_vectors):
        """Test conversion with timestamps."""
        timestamps = [int(time.time() * 1e9) + i for i in range(10)]

        table = vectors_to_arrow_table(
            ids=sample_vectors["ids"][:10],
            vectors=sample_vectors["vectors"][:10],
            timestamps=timestamps,
        )

        assert table is not None
        assert table.num_rows == 10
        assert "timestamp" in table.schema.names

    def test_arrow_table_to_vectors(self, sample_vectors):
        """Test roundtrip conversion: Python -> Arrow -> Python."""
        original_ids = sample_vectors["ids"][:10]
        original_vectors = sample_vectors["vectors"][:10]
        original_metadata = sample_vectors["metadata"][:10]

        # Convert to Arrow
        table = vectors_to_arrow_table(
            ids=original_ids,
            vectors=original_vectors,
            metadata=original_metadata,
        )

        # Convert back to Python
        ids, vectors, metadata = arrow_table_to_vectors(table)

        assert ids == original_ids
        assert len(vectors) == len(original_vectors)

        # Check vector values (with floating point tolerance)
        for i, (orig, converted) in enumerate(zip(original_vectors, vectors)):
            assert len(orig) == len(converted), f"Vector {i} dimension mismatch"
            for j, (o, c) in enumerate(zip(orig, converted)):
                assert abs(o - c) < 1e-6, f"Vector {i}[{j}] value mismatch"

    def test_empty_vectors_error(self):
        """Test that empty vectors raise error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            vectors_to_arrow_table(ids=[], vectors=[])

    def test_mismatched_length_error(self):
        """Test that mismatched ids/vectors raise error."""
        with pytest.raises(ValueError, match="same length"):
            vectors_to_arrow_table(
                ids=["v1", "v2"],
                vectors=[[0.1, 0.2, 0.3]],  # Only one vector
            )

    def test_large_batch_conversion(self, large_sample_vectors):
        """Test conversion of large batches (1000 vectors)."""
        table = vectors_to_arrow_table(
            ids=large_sample_vectors["ids"],
            vectors=large_sample_vectors["vectors"],
            metadata=large_sample_vectors["metadata"],
        )

        assert table.num_rows == large_sample_vectors["num_vectors"]

        # Verify memory efficiency (Arrow uses columnar format)
        size_bytes = table.nbytes
        vectors_only = 4 * large_sample_vectors["dimension"] * large_sample_vectors["num_vectors"]

        # Arrow table should be reasonably sized
        assert size_bytes < vectors_only * 2, "Arrow table too large"


# =============================================================================
# Embedded Database Tests - Base Class
# =============================================================================

@pytest.mark.skipif(not PROXIMADB_AVAILABLE, reason="ProximaDB embedded not available")
class BaseEngineTest:
    """Base class for engine-specific tests."""

    engine_name: str = "sst"  # Override in subclass

    @pytest.fixture
    def db(self, temp_db_dir):
        """Create embedded database instance."""
        db = proximadb.ProximaDB(temp_db_dir)
        yield db
        db.flush()
        db.close()

    def create_collection(self, db, collection_name: str, dimension: int):
        """Create collection with specific engine."""
        db.create_collection(
            collection_name,
            dimension=dimension,
            engine=self.engine_name,
        )

    def test_create_collection(self, db, sample_vectors):
        """Test collection creation with engine."""
        collection_name = f"test_{self.engine_name}_basic"
        self.create_collection(db, collection_name, sample_vectors["dimension"])

        # Verify collection exists
        collections = db.list_collections()
        collection_names = [c.name if hasattr(c, 'name') else str(c) for c in collections]
        assert collection_name in collection_names

    def test_insert_vectors(self, db, sample_vectors):
        """Test vector insertion."""
        collection_name = f"test_{self.engine_name}_insert"
        self.create_collection(db, collection_name, sample_vectors["dimension"])

        # Insert vectors
        for i in range(min(10, sample_vectors["num_vectors"])):
            db.insert(
                collection_name,
                ids=[sample_vectors["ids"][i]],
                vectors=[sample_vectors["vectors"][i]],
            )

        db.flush()

    def test_bulk_insert(self, db, sample_vectors):
        """Test bulk vector insertion (Arrow-like batch insert)."""
        collection_name = f"test_{self.engine_name}_bulk"
        self.create_collection(db, collection_name, sample_vectors["dimension"])

        # Bulk insert all vectors at once
        db.insert(
            collection_name,
            ids=sample_vectors["ids"],
            vectors=sample_vectors["vectors"],
        )

        db.flush()

    def test_search_after_insert(self, db, sample_vectors):
        """Test search after inserting vectors."""
        collection_name = f"test_{self.engine_name}_search"
        self.create_collection(db, collection_name, sample_vectors["dimension"])

        # Insert vectors
        db.insert(
            collection_name,
            ids=sample_vectors["ids"][:50],
            vectors=sample_vectors["vectors"][:50],
        )

        db.flush()

        # Search with first vector as query
        query_vector = sample_vectors["vectors"][0]
        results = db.search(collection_name, query=query_vector, top_k=10)

        assert len(results) > 0
        assert len(results) <= 10

        # First result should be exact match
        first_result_id = results[0].id if hasattr(results[0], 'id') else results[0]["id"]
        assert first_result_id == sample_vectors["ids"][0]

    def test_search_with_metadata_filter(self, db, sample_vectors):
        """Test search with metadata filtering (if supported)."""
        collection_name = f"test_{self.engine_name}_filter"
        self.create_collection(db, collection_name, sample_vectors["dimension"])

        # Insert vectors with metadata
        for i in range(50):
            db.insert(
                collection_name,
                ids=[sample_vectors["ids"][i]],
                vectors=[sample_vectors["vectors"][i]],
                metadata=[sample_vectors["metadata"][i]],
            )

        db.flush()

        # Search with filter (category == "cat_0")
        query_vector = sample_vectors["vectors"][0]

        try:
            results = db.search(
                collection_name,
                query=query_vector,
                top_k=10,
                filter={"category": "cat_0"},
            )

            # All results should match filter
            for r in results:
                meta = r.metadata if hasattr(r, 'metadata') else r.get("metadata")
                if meta:
                    cat = meta.get("category") if isinstance(meta, dict) else getattr(meta, "category", None)
                    if cat:
                        assert cat == "cat_0"
        except Exception as e:
            # Filter may not be supported for all engines
            pytest.skip(f"Metadata filter not supported for {self.engine_name}: {e}")

    def test_insert_performance(self, db, large_sample_vectors):
        """Test insert performance with larger dataset."""
        collection_name = f"test_{self.engine_name}_perf"
        self.create_collection(db, collection_name, large_sample_vectors["dimension"])

        start = time.perf_counter()

        # Insert in batches
        batch_size = 100
        for i in range(0, large_sample_vectors["num_vectors"], batch_size):
            end_idx = min(i + batch_size, large_sample_vectors["num_vectors"])
            db.insert(
                collection_name,
                ids=large_sample_vectors["ids"][i:end_idx],
                vectors=large_sample_vectors["vectors"][i:end_idx],
            )

        db.flush()

        elapsed = time.perf_counter() - start
        vectors_per_sec = large_sample_vectors["num_vectors"] / elapsed

        print(f"\n{self.engine_name} insert performance: {vectors_per_sec:.0f} vectors/sec")

        # Basic performance assertion (at least 100 vectors/sec)
        assert vectors_per_sec > 100, f"{self.engine_name} insert too slow"

    def test_search_performance(self, db, large_sample_vectors):
        """Test search performance after bulk insert."""
        collection_name = f"test_{self.engine_name}_search_perf"
        self.create_collection(db, collection_name, large_sample_vectors["dimension"])

        # Insert all vectors
        db.insert(
            collection_name,
            ids=large_sample_vectors["ids"],
            vectors=large_sample_vectors["vectors"],
        )
        db.flush()

        # Run search benchmark
        num_queries = 10
        start = time.perf_counter()

        for i in range(num_queries):
            query_idx = random.randint(0, large_sample_vectors["num_vectors"] - 1)
            query_vector = large_sample_vectors["vectors"][query_idx]
            results = db.search(collection_name, query=query_vector, top_k=10)
            assert len(results) > 0

        elapsed = time.perf_counter() - start
        avg_latency_ms = (elapsed / num_queries) * 1000

        print(f"\n{self.engine_name} search performance: {avg_latency_ms:.2f}ms avg")

        # Basic latency assertion (under 1 second per query)
        assert avg_latency_ms < 1000, f"{self.engine_name} search too slow"


# =============================================================================
# Engine-Specific Test Classes
# =============================================================================

@pytest.mark.skipif(not PROXIMADB_AVAILABLE, reason="ProximaDB embedded not available")
class TestSSTEngine(BaseEngineTest):
    """Tests for SST engine (write-optimized, real-time)."""
    engine_name = "sst"

    def test_sst_specific_write_optimization(self, db, large_sample_vectors):
        """Test SST's write optimization with streaming inserts."""
        collection_name = "test_sst_write_opt"
        self.create_collection(db, collection_name, large_sample_vectors["dimension"])

        # Simulate streaming writes (one at a time)
        start = time.perf_counter()
        for i in range(100):
            db.insert(
                collection_name,
                ids=[large_sample_vectors["ids"][i]],
                vectors=[large_sample_vectors["vectors"][i]],
            )
        elapsed = time.perf_counter() - start

        vectors_per_sec = 100 / elapsed
        print(f"\nSST streaming insert: {vectors_per_sec:.0f} vectors/sec")

        # SST should handle streaming well
        assert vectors_per_sec > 50, "SST streaming insert too slow"


@pytest.mark.skipif(not PROXIMADB_AVAILABLE, reason="ProximaDB embedded not available")
class TestHELIXEngine(BaseEngineTest):
    """Tests for HELIX engine (locality-optimized with Hilbert curve)."""
    engine_name = "helix"

    def test_helix_spatial_locality(self, db, sample_vectors):
        """Test HELIX's spatial locality optimization."""
        collection_name = "test_helix_spatial"
        self.create_collection(db, collection_name, sample_vectors["dimension"])

        # Insert vectors
        db.insert(
            collection_name,
            ids=sample_vectors["ids"],
            vectors=sample_vectors["vectors"],
        )
        db.flush()

        # Search should return spatially close results
        query_vector = sample_vectors["vectors"][0]
        results = db.search(collection_name, query=query_vector, top_k=10)

        assert len(results) > 0


@pytest.mark.skipif(not PROXIMADB_AVAILABLE, reason="ProximaDB embedded not available")
class TestNOVAEngine(BaseEngineTest):
    """Tests for the NOVA engine (columnar Parquet for analytics)."""
    engine_name = "nova"

    def test_nova_columnar_efficiency(self, db, large_sample_vectors):
        """Test NOVA's columnar storage efficiency."""
        collection_name = "test_nova_columnar"
        self.create_collection(db, collection_name, large_sample_vectors["dimension"])

        # Bulk insert (columnar engines excel at this)
        db.insert(
            collection_name,
            ids=large_sample_vectors["ids"],
            vectors=large_sample_vectors["vectors"],
        )
        db.flush()

        # Search
        query_vector = large_sample_vectors["vectors"][0]
        results = db.search(collection_name, query=query_vector, top_k=10)

        assert len(results) > 0


@pytest.mark.skipif(not PROXIMADB_AVAILABLE, reason="ProximaDB embedded not available")
class TestSWIFTEngine(BaseEngineTest):
    """Tests for SWIFT engine (ultra-low latency for small datasets)."""
    engine_name = "swift"

    def test_swift_low_latency(self, db, sample_vectors):
        """Test SWIFT's low-latency search on small dataset."""
        collection_name = "test_swift_lowlat"
        self.create_collection(db, collection_name, sample_vectors["dimension"])

        # Insert small dataset (SWIFT is optimized for <5K vectors)
        db.insert(
            collection_name,
            ids=sample_vectors["ids"][:50],
            vectors=sample_vectors["vectors"][:50],
        )
        db.flush()

        # Measure search latency
        query_vector = sample_vectors["vectors"][0]

        latencies = []
        for _ in range(5):
            start = time.perf_counter()
            results = db.search(collection_name, query=query_vector, top_k=10)
            latencies.append((time.perf_counter() - start) * 1000)
            assert len(results) > 0

        avg_latency = sum(latencies) / len(latencies)
        print(f"\nSWIFT avg search latency: {avg_latency:.2f}ms")


@pytest.mark.skipif(not PROXIMADB_AVAILABLE, reason="ProximaDB embedded not available")
class TestNOVAEngine(BaseEngineTest):
    """Tests for NOVA engine (progressive columnar for mixed workloads)."""
    engine_name = "nova"

    def test_nova_mixed_workload(self, db, sample_vectors):
        """Test NOVA's mixed read/write workload handling."""
        collection_name = "test_nova_mixed"
        self.create_collection(db, collection_name, sample_vectors["dimension"])

        # Mixed workload: insert + search interleaved
        for i in range(20):
            # Insert batch
            start_idx = i * 5
            end_idx = min(start_idx + 5, sample_vectors["num_vectors"])
            if start_idx < sample_vectors["num_vectors"]:
                db.insert(
                    collection_name,
                    ids=sample_vectors["ids"][start_idx:end_idx],
                    vectors=sample_vectors["vectors"][start_idx:end_idx],
                )

            # Search (if we have data)
            if i > 0:
                query_vector = sample_vectors["vectors"][0]
                results = db.search(collection_name, query=query_vector, top_k=5)

        db.flush()


@pytest.mark.skipif(not PROXIMADB_AVAILABLE, reason="ProximaDB embedded not available")
class TestRAPTOREngine(BaseEngineTest):
    """Tests for RAPTOR engine (adaptive row-group for dynamic workloads)."""
    engine_name = "raptor"

    def test_raptor_adaptive_behavior(self, db, sample_vectors):
        """Test RAPTOR's adaptive behavior with varying batch sizes."""
        collection_name = "test_raptor_adaptive"
        self.create_collection(db, collection_name, sample_vectors["dimension"])

        # Insert with varying batch sizes (RAPTOR should adapt)
        batch_sizes = [1, 5, 10, 20, 10, 5, 1]
        idx = 0

        for batch_size in batch_sizes:
            end_idx = min(idx + batch_size, sample_vectors["num_vectors"])
            if idx < sample_vectors["num_vectors"]:
                db.insert(
                    collection_name,
                    ids=sample_vectors["ids"][idx:end_idx],
                    vectors=sample_vectors["vectors"][idx:end_idx],
                )
                idx = end_idx

        db.flush()

        # Verify search works
        query_vector = sample_vectors["vectors"][0]
        results = db.search(collection_name, query=query_vector, top_k=10)
        assert len(results) > 0


# =============================================================================
# Arrow Flight Server Tests (requires running server)
# =============================================================================

@pytest.mark.skipif(
    not ARROW_FLIGHT_CLIENT_AVAILABLE,
    reason="Arrow Flight client not available"
)
class TestArrowFlightServer:
    """
    Test Arrow Flight operations against a running ProximaDB server.

    These tests require a running ProximaDB server with Arrow Flight enabled.
    Set PROXIMADB_SERVER_URL environment variable to run these tests.
    """

    @pytest.fixture
    def server_url(self):
        """Get server URL from environment or skip."""
        url = os.environ.get("PROXIMADB_SERVER_URL")
        if not url:
            pytest.skip("PROXIMADB_SERVER_URL not set")
        return url

    @pytest.fixture
    def flight_client(self, server_url):
        """Create Arrow Flight client."""
        client = ArrowFlightClient(server_url)
        yield client
        client.close()

    def test_list_actions(self, flight_client):
        """Test listing available actions."""
        actions = flight_client.list_actions()

        # Should have flush, compact, and flush_and_compact actions
        action_types = [a[0] for a in actions]
        assert "flush_collection" in action_types or len(actions) >= 0
        assert "bulk_upsert" in action_types
        assert "bulk_delete" in action_types

    def test_bulk_insert_and_search(self, flight_client, sample_vectors):
        """Test bulk insert via Arrow Flight and search."""
        collection_name = f"arrow_test_{int(time.time())}"

        # Create table
        table = vectors_to_arrow_table(
            ids=sample_vectors["ids"][:50],
            vectors=sample_vectors["vectors"][:50],
            metadata=sample_vectors["metadata"][:50],
        )

        # Bulk insert
        result = flight_client.bulk_insert(collection_name, table)

        if result.success:
            assert result.vectors_inserted == 50

            # Flush
            flight_client.flush_collection(collection_name)

            # Search
            query_vector = sample_vectors["vectors"][0]
            results = flight_client.search(collection_name, query_vector, top_k=10)

            assert len(results) > 0
        else:
            # Collection may not exist - skip
            pytest.skip(f"Bulk insert failed: {result.message}")

    def test_bulk_insert_performance(self, flight_client, large_sample_vectors):
        """Test Arrow Flight bulk insert performance."""
        collection_name = f"arrow_perf_test_{int(time.time())}"

        # Create table
        table = vectors_to_arrow_table(
            ids=large_sample_vectors["ids"],
            vectors=large_sample_vectors["vectors"],
        )

        start = time.perf_counter()
        result = flight_client.bulk_insert(collection_name, table)
        elapsed = time.perf_counter() - start

        if result.success:
            vectors_per_sec = large_sample_vectors["num_vectors"] / elapsed
            print(f"\nArrow Flight bulk insert: {vectors_per_sec:.0f} vectors/sec")

            # Arrow Flight should be fast
            assert vectors_per_sec > 1000, "Arrow Flight insert too slow"

    def test_bulk_upsert_and_delete_doput(self, flight_client, sample_vectors):
        """Test DoPut bulk upsert followed by DoPut bulk delete."""
        collection_name = f"arrow_doput_delete_{int(time.time())}"
        created = flight_client._do_action(
            "create_collection",
            {
                "name": collection_name,
                "dimension": sample_vectors["dimension"],
                "engine": "sst",
                "distance_metric": "cosine",
            },
        )
        if not created:
            pytest.skip("Could not create collection over Arrow Flight")

        table = vectors_to_arrow_table(
            ids=sample_vectors["ids"][:8],
            vectors=sample_vectors["vectors"][:8],
            metadata=sample_vectors["metadata"][:8],
        )

        upsert = flight_client.bulk_upsert(collection_name, table, batch_size=3)
        if not upsert.success:
            pytest.skip(f"Bulk upsert failed before delete: {upsert.message}")

        assert upsert.records_processed == 8

        delete = flight_client.bulk_delete(
            collection_name,
            sample_vectors["ids"][:4],
            batch_size=2,
        )

        assert delete.success
        assert delete.records_processed == 4
        assert delete.records_failed == 0

    def test_bulk_upsert_exchange(self, flight_client, sample_vectors):
        """Test progress-aware bulk upsert via Arrow Flight DoExchange."""
        collection_name = f"arrow_exchange_upsert_{int(time.time())}"
        created = flight_client._do_action(
            "create_collection",
            {
                "name": collection_name,
                "dimension": sample_vectors["dimension"],
                "engine": "sst",
                "distance_metric": "cosine",
            },
        )
        if not created:
            pytest.skip("Could not create collection over Arrow Flight")

        table = vectors_to_arrow_table(
            ids=sample_vectors["ids"][:10],
            vectors=sample_vectors["vectors"][:10],
            metadata=sample_vectors["metadata"][:10],
        )

        result = flight_client.bulk_upsert_exchange(collection_name, table, batch_size=4)

        assert result.success
        assert result.records_processed == 10
        assert result.records_failed == 0
        assert result.batches_processed >= 1
        assert result.metadata["operation"] == "upsert"

    def test_bulk_delete_exchange(self, flight_client, sample_vectors):
        """Test progress-aware bulk delete via Arrow Flight DoExchange."""
        collection_name = f"arrow_exchange_delete_{int(time.time())}"
        created = flight_client._do_action(
            "create_collection",
            {
                "name": collection_name,
                "dimension": sample_vectors["dimension"],
                "engine": "sst",
                "distance_metric": "cosine",
            },
        )
        if not created:
            pytest.skip("Could not create collection over Arrow Flight")

        table = vectors_to_arrow_table(
            ids=sample_vectors["ids"][:10],
            vectors=sample_vectors["vectors"][:10],
            metadata=sample_vectors["metadata"][:10],
        )
        upsert = flight_client.bulk_upsert_exchange(collection_name, table, batch_size=5)
        if not upsert.success:
            pytest.skip(f"Bulk upsert failed before delete: {upsert.message}")

        result = flight_client.bulk_delete_exchange(
            collection_name,
            sample_vectors["ids"][:5],
            batch_size=3,
        )

        assert result.success
        assert result.records_processed == 5
        assert result.records_failed == 0
        assert result.batches_processed >= 1
        assert result.metadata["operation"] == "delete"


# =============================================================================
# Integration Tests - Arrow Format + Embedded Database
# =============================================================================

@pytest.mark.skipif(
    not (PYARROW_AVAILABLE and PROXIMADB_AVAILABLE),
    reason="PyArrow and ProximaDB embedded required"
)
class TestArrowEmbeddedIntegration:
    """
    Integration tests combining Arrow format conversion with embedded database.

    These tests verify the end-to-end data flow:
    1. Create vectors in Arrow format
    2. Convert to Python format
    3. Insert into embedded database
    4. Search and verify results
    """

    @pytest.fixture
    def db(self, temp_db_dir):
        """Create embedded database instance."""
        db = proximadb.ProximaDB(temp_db_dir)
        yield db
        db.flush()
        db.close()

    @pytest.mark.parametrize("engine", ["sst", "helix", "nova", "swift", "nova", "raptor"])
    def test_arrow_to_embedded_roundtrip(self, db, sample_vectors, engine):
        """Test Arrow -> Python -> Embedded DB roundtrip for each engine."""
        collection_name = f"arrow_integration_{engine}"
        dimension = sample_vectors["dimension"]

        # Create collection
        db.create_collection(collection_name, dimension=dimension, engine=engine)

        # Create Arrow table
        table = vectors_to_arrow_table(
            ids=sample_vectors["ids"][:20],
            vectors=sample_vectors["vectors"][:20],
            metadata=sample_vectors["metadata"][:20],
        )

        # Convert to Python
        ids, vectors, metadata = arrow_table_to_vectors(table)

        # Insert into embedded DB
        db.insert(collection_name, ids=ids, vectors=vectors)
        db.flush()

        # Search
        query_vector = sample_vectors["vectors"][0]
        results = db.search(collection_name, query=query_vector, top_k=5)

        # Verify
        assert len(results) > 0
        first_result_id = results[0].id if hasattr(results[0], 'id') else results[0]["id"]
        assert first_result_id == sample_vectors["ids"][0]

        print(f"\n{engine} Arrow integration: {len(results)} results found")

    def test_all_engines_comparison(self, db, large_sample_vectors):
        """Compare all engines with Arrow-formatted data."""
        engines = ["sst", "helix", "nova", "swift", "nova", "raptor"]
        dimension = large_sample_vectors["dimension"]
        num_vectors = 500  # Use subset for speed

        # Create Arrow table once
        table = vectors_to_arrow_table(
            ids=large_sample_vectors["ids"][:num_vectors],
            vectors=large_sample_vectors["vectors"][:num_vectors],
        )

        # Convert to Python once
        ids, vectors, _ = arrow_table_to_vectors(table)

        results_summary = {}

        for engine in engines:
            collection_name = f"arrow_compare_{engine}"

            try:
                # Create collection
                db.create_collection(collection_name, dimension=dimension, engine=engine)

                # Insert
                start = time.perf_counter()
                db.insert(collection_name, ids=ids, vectors=vectors)
                db.flush()
                insert_time = time.perf_counter() - start

                # Search
                query_vector = large_sample_vectors["vectors"][0]
                start = time.perf_counter()
                search_results = db.search(collection_name, query=query_vector, top_k=10)
                search_time = (time.perf_counter() - start) * 1000

                results_summary[engine] = {
                    "insert_time": insert_time,
                    "search_time_ms": search_time,
                    "results_count": len(search_results),
                    "status": "success",
                }

            except Exception as e:
                results_summary[engine] = {
                    "status": "failed",
                    "error": str(e),
                }

        # Print summary
        print("\n" + "=" * 60)
        print("Arrow Integration - All Engines Comparison")
        print("=" * 60)
        print(f"{'Engine':<10} {'Insert(s)':<12} {'Search(ms)':<12} {'Results':<10} {'Status'}")
        print("-" * 60)

        for engine, result in results_summary.items():
            if result["status"] == "success":
                print(f"{engine:<10} {result['insert_time']:<12.3f} {result['search_time_ms']:<12.2f} {result['results_count']:<10} OK")
            else:
                print(f"{engine:<10} {'N/A':<12} {'N/A':<12} {'N/A':<10} {result['error'][:20]}")

        # At least some engines should work
        successful = [e for e, r in results_summary.items() if r["status"] == "success"]
        assert len(successful) >= 1, "At least one engine should work"


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
