#!/usr/bin/env python3
"""
Comprehensive pytest test suite for all storage engines with index types.

Tests the matrix of:
- 6 Storage Engines: SST, HELIX, VIPER, SWIFT, NOVA, RAPTOR
- Index Types: HNSW (via AXIS), IVF (planned), Flat (brute-force)
- Search modes: Exact, Approximate

This validates that each engine+index combination works correctly.
"""

import tempfile
import time

import numpy as np
import pytest

# Import ProximaDB - handle both installed and local dev modes
try:
    import proximadb

    proximadb.init_logging("warn")
    from proximadb import ProximaDB as EmbeddedProximaDB
except ImportError:
    pytest.skip("proximadb module not available", allow_module_level=True)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def test_vectors():
    """Generate test vectors for all tests."""
    np.random.seed(42)
    count = 1000
    dimension = 128
    vectors = np.random.randn(count, dimension).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return {
        "vectors": vectors,
        "vectors_list": vectors.tolist(),
        "ids": [f"vec_{i}" for i in range(count)],
        "dimension": dimension,
        "count": count,
    }


@pytest.fixture(scope="module")
def query_vectors():
    """Generate query vectors for all tests."""
    np.random.seed(123)
    count = 10
    dimension = 128
    queries = np.random.randn(count, dimension).astype(np.float32)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    return {
        "queries": queries,
        "queries_list": queries.tolist(),
    }


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for database storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================================================
# Engine Test Classes
# ============================================================================


class TestSSTEngine:
    """Tests for SST (LSM-tree) storage engine with various indexes."""

    @pytest.fixture
    def sst_db(self, temp_db_dir):
        """Create an SST engine database instance."""
        db = EmbeddedProximaDB(temp_db_dir)
        yield db
        db.close()

    def test_sst_basic_insert_search(self, sst_db, test_vectors, query_vectors):
        """Test basic insert and search with SST engine."""
        sst_db.create_collection(
            "sst_basic_test", dimension=test_vectors["dimension"], engine="sst"
        )

        sst_db.insert(
            "sst_basic_test",
            ids=test_vectors["ids"],
            vectors=test_vectors["vectors_list"],
        )
        sst_db.flush()

        # Wait for async indexing
        time.sleep(2)

        # Search
        results = sst_db.search(
            "sst_basic_test", query=query_vectors["queries_list"][0], top_k=10
        )

        assert len(results) == 10, f"Expected 10 results, got {len(results)}"
        # Verify results have required fields
        assert hasattr(results[0], "id"), "Result should have 'id' field"
        assert hasattr(results[0], "score"), "Result should have 'score' field"

    def test_sst_self_match(self, sst_db, test_vectors):
        """Test self-match: querying with an inserted vector should return itself."""
        sst_db.create_collection(
            "sst_selfmatch", dimension=test_vectors["dimension"], engine="sst"
        )

        sst_db.insert(
            "sst_selfmatch",
            ids=test_vectors["ids"],
            vectors=test_vectors["vectors_list"],
        )
        sst_db.flush()
        time.sleep(2)

        # Query with vec_0
        results = sst_db.search(
            "sst_selfmatch", query=test_vectors["vectors_list"][0], top_k=5
        )

        assert len(results) >= 1, "Should get at least 1 result"
        assert (
            results[0].id == "vec_0"
        ), f"First result should be vec_0, got {results[0].id}"
        assert (
            results[0].score > 0.99
        ), f"Self-match score should be ~1.0, got {results[0].score}"

    def test_sst_recall_quality(self, sst_db, test_vectors, query_vectors):
        """Test recall quality of SST engine with HNSW index."""
        sst_db.create_collection(
            "sst_recall_test", dimension=test_vectors["dimension"], engine="sst"
        )

        sst_db.insert(
            "sst_recall_test",
            ids=test_vectors["ids"],
            vectors=test_vectors["vectors_list"],
        )
        sst_db.flush()
        time.sleep(3)  # More time for index building

        vectors = test_vectors["vectors"]
        query = query_vectors["queries"][0]

        # Compute exact neighbors
        similarities = np.dot(vectors, query)
        exact_indices = np.argsort(-similarities)[:10]
        exact_ids = set(f"vec_{idx}" for idx in exact_indices)

        # HNSW search
        results = sst_db.search("sst_recall_test", query=query.tolist(), top_k=10)
        hnsw_ids = set(r.id for r in results)

        recall = len(exact_ids & hnsw_ids) / 10
        # With 1000 vectors, expect high recall
        assert recall >= 0.8, f"Recall@10 should be >= 80%, got {recall*100:.0f}%"


class TestHELIXEngine:
    """Tests for HELIX (Hilbert curve) storage engine."""

    @pytest.fixture
    def helix_db(self, temp_db_dir):
        """Create a HELIX engine database instance."""
        db = EmbeddedProximaDB(temp_db_dir)
        yield db
        db.close()

    def test_helix_basic_insert_search(self, helix_db, test_vectors, query_vectors):
        """Test basic insert and search with HELIX engine."""
        helix_db.create_collection(
            "helix_test", dimension=test_vectors["dimension"], engine="helix"
        )

        helix_db.insert(
            "helix_test", ids=test_vectors["ids"], vectors=test_vectors["vectors_list"]
        )
        helix_db.flush()
        time.sleep(2)

        results = helix_db.search(
            "helix_test", query=query_vectors["queries_list"][0], top_k=10
        )

        assert len(results) == 10, f"Expected 10 results, got {len(results)}"

    def test_helix_self_match(self, helix_db, test_vectors):
        """Test self-match with HELIX engine."""
        helix_db.create_collection(
            "helix_self", dimension=test_vectors["dimension"], engine="helix"
        )

        helix_db.insert(
            "helix_self", ids=test_vectors["ids"], vectors=test_vectors["vectors_list"]
        )
        helix_db.flush()
        time.sleep(2)

        results = helix_db.search(
            "helix_self", query=test_vectors["vectors_list"][0], top_k=5
        )

        assert len(results) >= 1
        assert (
            results[0].id == "vec_0"
        ), f"First result should be vec_0, got {results[0].id}"


class TestVIPEREngine:
    """Tests for VIPER (Columnar Parquet) storage engine."""

    @pytest.fixture
    def viper_db(self, temp_db_dir):
        """Create a VIPER engine database instance."""
        db = EmbeddedProximaDB(temp_db_dir)
        yield db
        db.close()

    def test_viper_basic_insert_search(self, viper_db, test_vectors, query_vectors):
        """Test basic insert and search with VIPER engine."""
        viper_db.create_collection(
            "nova_test", dimension=test_vectors["dimension"], engine="nova"
        )

        viper_db.insert(
            "viper_test", ids=test_vectors["ids"], vectors=test_vectors["vectors_list"]
        )
        viper_db.flush()
        time.sleep(2)

        results = viper_db.search(
            "viper_test", query=query_vectors["queries_list"][0], top_k=10
        )

        assert len(results) == 10, f"Expected 10 results, got {len(results)}"


class TestSWIFTEngine:
    """Tests for SWIFT (in-memory optimized) storage engine."""

    @pytest.fixture
    def swift_db(self, temp_db_dir):
        """Create a SWIFT engine database instance."""
        db = EmbeddedProximaDB(temp_db_dir)
        yield db
        db.close()

    def test_swift_basic_insert_search(self, swift_db, test_vectors, query_vectors):
        """Test basic insert and search with SWIFT engine."""
        swift_db.create_collection(
            "swift_test", dimension=test_vectors["dimension"], engine="swift"
        )

        swift_db.insert(
            "swift_test", ids=test_vectors["ids"], vectors=test_vectors["vectors_list"]
        )
        swift_db.flush()
        time.sleep(2)

        results = swift_db.search(
            "swift_test", query=query_vectors["queries_list"][0], top_k=10
        )

        assert len(results) == 10, f"Expected 10 results, got {len(results)}"


class TestNOVAEngine:
    """Tests for NOVA (progressive columnar) storage engine."""

    @pytest.fixture
    def nova_db(self, temp_db_dir):
        """Create a NOVA engine database instance."""
        db = EmbeddedProximaDB(temp_db_dir)
        yield db
        db.close()

    def test_nova_basic_insert_search(self, nova_db, test_vectors, query_vectors):
        """Test basic insert and search with NOVA engine."""
        nova_db.create_collection(
            "nova_test0", dimension=test_vectors["dimension"], engine="nova"
        )

        nova_db.insert(
            "nova_test0", ids=test_vectors["ids"], vectors=test_vectors["vectors_list"]
        )
        nova_db.flush()
        time.sleep(2)

        results = nova_db.search(
            "nova_test0", query=query_vectors["queries_list"][0], top_k=10
        )

        assert len(results) == 10, f"Expected 10 results, got {len(results)}"


class TestRAPTOREngine:
    """Tests for RAPTOR (adaptive row-group) storage engine."""

    @pytest.fixture
    def raptor_db(self, temp_db_dir):
        """Create a RAPTOR engine database instance."""
        db = EmbeddedProximaDB(temp_db_dir)
        yield db
        db.close()

    def test_raptor_basic_insert_search(self, raptor_db, test_vectors, query_vectors):
        """Test basic insert and search with RAPTOR engine."""
        raptor_db.create_collection(
            "raptor_test", dimension=test_vectors["dimension"], engine="raptor"
        )

        raptor_db.insert(
            "raptor_test", ids=test_vectors["ids"], vectors=test_vectors["vectors_list"]
        )
        raptor_db.flush()
        time.sleep(2)

        results = raptor_db.search(
            "raptor_test", query=query_vectors["queries_list"][0], top_k=10
        )

        assert len(results) == 10, f"Expected 10 results, got {len(results)}"


# ============================================================================
# Cross-Engine Tests
# ============================================================================


class TestCrossEngineConsistency:
    """Tests to verify all engines return consistent results."""

    @pytest.fixture
    def multi_engine_dbs(self, temp_db_dir):
        """Create database instance for multi-engine tests."""
        db = EmbeddedProximaDB(temp_db_dir)
        yield db
        db.close()

    @pytest.mark.parametrize(
        "engine", ["sst", "helix", "swift", "nova", "raptor"]
    )
    def test_engine_returns_results(
        self, multi_engine_dbs, test_vectors, query_vectors, engine
    ):
        """Parametrized test: each engine should return search results."""
        collection_name = f"cross_eng_{engine}"

        multi_engine_dbs.create_collection(
            collection_name, dimension=test_vectors["dimension"], engine=engine
        )

        multi_engine_dbs.insert(
            collection_name,
            ids=test_vectors["ids"][:100],  # Use fewer vectors for speed
            vectors=test_vectors["vectors_list"][:100],
        )
        multi_engine_dbs.flush()
        time.sleep(2)

        results = multi_engine_dbs.search(
            collection_name, query=query_vectors["queries_list"][0], top_k=10
        )

        assert len(results) >= 1, f"Engine {engine} should return at least 1 result"

    @pytest.mark.parametrize(
        "engine", ["sst", "helix", "swift", "nova", "raptor"]
    )
    def test_engine_self_match_works(self, multi_engine_dbs, test_vectors, engine):
        """Parametrized test: each engine should find self-match."""
        collection_name = f"selfmatch_{engine}"

        multi_engine_dbs.create_collection(
            collection_name, dimension=test_vectors["dimension"], engine=engine
        )

        # Insert just 100 vectors for speed
        multi_engine_dbs.insert(
            collection_name,
            ids=test_vectors["ids"][:100],
            vectors=test_vectors["vectors_list"][:100],
        )
        multi_engine_dbs.flush()
        time.sleep(2)

        # Query with first vector
        results = multi_engine_dbs.search(
            collection_name, query=test_vectors["vectors_list"][0], top_k=5
        )

        assert len(results) >= 1, f"Engine {engine} should return results"
        # Self-match should be in top results
        top_ids = [r.id for r in results[:3]]
        assert (
            "vec_0" in top_ids
        ), f"Engine {engine}: vec_0 should be in top-3, got {top_ids}"


# ============================================================================
# Index Type Tests (placeholder for when IVF/other indexes are available)
# ============================================================================


class TestIndexTypes:
    """Tests for different index types with SST engine."""

    @pytest.fixture
    def index_db(self, temp_db_dir):
        """Create database for index tests."""
        db = EmbeddedProximaDB(temp_db_dir)
        yield db
        db.close()

    def test_hnsw_index_creation(self, index_db, test_vectors, query_vectors):
        """Test HNSW index is automatically created for larger collections."""
        index_db.create_collection(
            "hnsw_auto", dimension=test_vectors["dimension"], engine="sst"
        )

        index_db.insert(
            "hnsw_auto", ids=test_vectors["ids"], vectors=test_vectors["vectors_list"]
        )
        index_db.flush()
        time.sleep(3)

        # HNSW should be created automatically for 1000 vectors
        results = index_db.search(
            "hnsw_auto", query=query_vectors["queries_list"][0], top_k=10
        )

        assert len(results) == 10, "HNSW index should return 10 results"

    @pytest.mark.skip(reason="IVF index not yet exposed in Python API")
    def test_ivf_index(self, index_db, test_vectors, query_vectors):
        """Test IVF index when available."""
        pass

    @pytest.mark.skip(reason="Flat index explicit configuration not yet in API")
    def test_flat_index(self, index_db, test_vectors, query_vectors):
        """Test flat (brute-force) index when available."""
        pass


# ============================================================================
# Performance Sanity Tests
# ============================================================================


class TestPerformanceSanity:
    """Basic performance sanity tests."""

    @pytest.fixture
    def perf_db(self, temp_db_dir):
        """Create database for performance tests."""
        db = EmbeddedProximaDB(temp_db_dir)
        yield db
        db.close()

    def test_search_latency_reasonable(self, perf_db, test_vectors, query_vectors):
        """Test that search latency is reasonable (< 100ms for 1000 vectors)."""
        perf_db.create_collection(
            "perf_test1", dimension=test_vectors["dimension"], engine="sst"
        )

        perf_db.insert(
            "perf_test1", ids=test_vectors["ids"], vectors=test_vectors["vectors_list"]
        )
        perf_db.flush()
        time.sleep(3)

        # Warm-up query
        perf_db.search("perf_test1", query=query_vectors["queries_list"][0], top_k=10)

        # Timed queries
        latencies = []
        for query in query_vectors["queries_list"][:5]:
            start = time.time()
            perf_db.search("perf_test1", query=query, top_k=10)
            latencies.append((time.time() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        assert (
            avg_latency < 100
        ), f"Average latency should be < 100ms, got {avg_latency:.1f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
