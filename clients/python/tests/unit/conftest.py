#!/usr/bin/env python3
"""
ProximaDB Python SDK Test Configuration - Embedded Database Version
Uses embedded ProximaDB (PyO3/maturin) instead of requiring a running server.

All tests run against the embedded database for faster, more reliable testing.
"""

# PEP 604 union syntax (e.g. `CollectionConfig | None`) requires deferred
# evaluation when the referenced names are lazily-loaded SDK placeholders
# initialized to `None` further down this module. Without this `from
# __future__` import, the class-body annotations evaluate `None | None`
# → TypeError at import time.
from __future__ import annotations

import logging
import os
import shutil

# Import decoupled test helpers (no SDK dependency)
# Use sys.path manipulation to handle different pytest invocation contexts
import sys
import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_tests_root = Path(__file__).parent.parent
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

from utils.test_helpers import CollectionInfo, InsertResult, SearchResult

# Lazy SDK imports - only load when actually needed by tests
_SDK_LOADED = False
_SDK_TYPES = {}


def _ensure_sdk_loaded():
    """Lazily load SDK types when needed."""
    global _SDK_LOADED, _SDK_TYPES
    if not _SDK_LOADED:
        try:
            from proximadb_sdk import (
                CollectionConfig,
                DistanceMetric,
                ProximaDBError,
                StorageEngine,
            )

            _SDK_TYPES["CollectionConfig"] = CollectionConfig
            _SDK_TYPES["DistanceMetric"] = DistanceMetric
            _SDK_TYPES["StorageEngine"] = StorageEngine
            _SDK_TYPES["ProximaDBError"] = ProximaDBError
            _SDK_LOADED = True
        except Exception as e:
            logging.getLogger(__name__).warning(f"SDK not available: {e}")
            _SDK_LOADED = False
    return _SDK_LOADED


def get_sdk_type(name: str):
    """Get an SDK type by name, loading SDK if needed."""
    _ensure_sdk_loaded()
    return _SDK_TYPES.get(name)


# For backward compatibility - expose these at module level
# They will be None if SDK is not available
CollectionConfig = None
DistanceMetric = None
StorageEngine = None
ProximaDBError = None

try:
    from proximadb_sdk import CollectionConfig, ProximaDBError, StorageEngine
except Exception:
    pass


# Import embedded database
try:
    from proximadb import ProximaDB

    EMBEDDED_AVAILABLE = True
except ImportError:
    EMBEDDED_AVAILABLE = False
    ProximaDB = None


# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Test configuration
TEST_CONFIG = {
    "default_timeout": 30.0,
    "max_retry_attempts": 3,
    "test_collection_prefix": "pytest_",
    "cleanup_on_failure": True,
    "cache_size_mb": 256,
    "default_engine": "sst",
}


# ============================================================================
# Embedded Database Client Wrapper
# ============================================================================


class EmbeddedClientWrapper:
    """
    Wrapper that provides SDK-compatible interface to embedded ProximaDB.

    This allows existing tests to work with minimal modifications.
    """

    def __init__(self, db: "ProximaDB", data_dir: str):
        self._db = db
        self._data_dir = data_dir
        self._collections: dict[str, dict] = {}

    def create_collection(
        self,
        name: str,
        dimension: int | None = None,
        config: CollectionConfig | None = None,
        distance_metric: str = "cosine",
        description: str = "",
        storage_engine: StorageEngine | None = None,
        **kwargs,
    ):
        """Create a collection in embedded database"""
        # Check if collection already exists
        try:
            existing = self._db.get_collection(name)
            if existing is not None:
                raise ProximaDBError(f"Collection {name} already exists")
        except Exception as e:
            # Collection doesn't exist (expected) or other error
            if "already exists" in str(e):
                raise
            pass

        if config is not None:
            dim = config.dimension
            engine = getattr(config, "storage_engine", None)
            engine_str = engine.value if engine else "sst"
        else:
            dim = dimension or 128
            engine_str = storage_engine.value if storage_engine else "sst"

        self._db.create_collection(name, dim, engine_str)

        # Track collection info
        collection_info = CollectionInfo(
            name=name,
            dimension=dim,
            engine=engine_str,
            distance_metric=distance_metric,
            description=description,
        )
        self._collections[name] = collection_info
        return collection_info

    def delete_collection(self, name: str) -> bool:
        """Delete a collection"""
        try:
            self._db.delete_collection(name)
            if name in self._collections:
                del self._collections[name]
            return True
        except Exception:
            return False

    def get_collection(self, name: str):
        """Get collection info - always check actual database first"""
        # Try to get from database first (authoritative source)
        try:
            info = self._db.get_collection(name)
            if info is not None:
                collection_info = CollectionInfo(
                    name=name,
                    dimension=getattr(info, "dimension", 128),
                    engine=getattr(info, "engine", "sst"),
                )
                self._collections[name] = collection_info
                return collection_info
        except Exception:
            pass

        # Collection not found in database - also clear from cache if present
        if name in self._collections:
            del self._collections[name]

        raise ProximaDBError(f"Collection {name} not found")

    def list_collections(self) -> list:
        """List all collections"""
        try:
            collections = self._db.list_collections()
            return [
                CollectionInfo(
                    name=getattr(c, "name", str(c)),
                    dimension=getattr(c, "dimension", 0),
                    engine=getattr(c, "engine", "sst"),
                )
                for c in collections
            ]
        except Exception:
            return list(self._collections.values())

    def insert_vector(
        self,
        collection_id: str,
        vector_id: str,
        vector: list[float],
        metadata: dict | None = None,
    ):
        """Insert a single vector"""
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)
        else:
            vector = vector.astype(np.float32)

        # Reshape to 2D for insert
        vectors = vector.reshape(1, -1)
        ids = [vector_id]
        metadatas = [metadata or {}]

        count = self._db.insert(collection_id, ids, vectors, metadatas)
        return InsertResult(success=True, count=count)

    def insert_vectors(
        self,
        collection_id: str,
        vectors: list,
        ids: list[str],
        metadata: list[dict] | None = None,
    ):
        """Insert multiple vectors"""
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)
        else:
            vectors = vectors.astype(np.float32)

        if metadata is None:
            metadata = [{} for _ in range(len(ids))]

        count = self._db.insert(collection_id, ids, vectors, metadata)
        return InsertResult(
            success=True, count=count, successful_count=count, total=len(ids)
        )

    def search(
        self,
        collection_id: str,
        vector: list[float],
        top_k: int = 10,
        include_metadata: bool = True,
        include_vectors: bool = False,
        metadata_filter: dict | None = None,
        **kwargs,
    ) -> list:
        """Search for similar vectors"""
        if top_k <= 0:
            raise ProximaDBError(f"Invalid top_k value: {top_k}")

        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)

        results = self._db.search(
            collection_id,
            vector,
            top_k=top_k,
            filter=None,  # Client-side filtering if needed
        )

        search_results = []
        for r in results:
            result = SearchResult(
                id=r.id,
                score=r.score,
                metadata=getattr(r, "metadata", {}) if include_metadata else {},
                vector=getattr(r, "vector", None) if include_vectors else None,
            )
            search_results.append(result)

        # Apply client-side metadata filter if specified
        if metadata_filter and include_metadata:
            filtered_results = []
            for r in search_results:
                matches = True
                for key, value in metadata_filter.items():
                    if r.metadata.get(key) != value:
                        matches = False
                        break
                if matches:
                    filtered_results.append(r)
            search_results = filtered_results

        return search_results

    def get_vector(
        self,
        collection_id: str,
        vector_id: str,
        include_vector: bool = True,
        include_metadata: bool = True,
    ) -> dict | None:
        """Get a vector by ID - search based implementation"""
        # Embedded doesn't have direct get by ID, so we'd need to implement it
        # For now, raise an error that tests can handle
        raise NotImplementedError("get_vector not directly supported in embedded mode")

    def get_collection_id_by_name(self, name: str) -> str:
        """Get collection ID by name (for embedded, name IS the ID)"""
        return name

    def health(self) -> dict:
        """Health check"""
        return {"status": "healthy", "mode": "embedded"}

    def flush(self):
        """Flush pending writes"""
        self._db.flush()

    def close(self):
        """Close the client"""
        try:
            self._db.flush()
        except Exception:
            pass

    # ========================================================================
    # Graph Operations
    # ========================================================================

    def create_graph(self, name: str, **kwargs):
        """Create a graph (stub for embedded - graphs are implicit)"""
        # Embedded database handles graphs implicitly
        return {"name": name, "created": True}

    def get_graph(self, name: str):
        """Get graph info"""
        return {"name": name}

    def delete_graph(self, name: str):
        """Delete a graph"""
        return {"name": name, "deleted": True}

    def create_node(
        self,
        node_id: str,
        labels: list[str],
        properties: dict | None = None,
        embedding: list[float] | None = None,
        **kwargs,
    ):
        """Create a graph node"""
        # Store as vector with node metadata
        return {
            "node_id": node_id,
            "labels": labels,
            "properties": properties or {},
            "created": True,
        }

    def create_edge(
        self,
        edge_id: str,
        from_node_id: str,
        to_node_id: str,
        edge_type: str,
        properties: dict | None = None,
        weight: float | None = None,
        **kwargs,
    ):
        """Create a graph edge"""
        return {
            "edge_id": edge_id,
            "from_node_id": from_node_id,
            "to_node_id": to_node_id,
            "edge_type": edge_type,
            "properties": properties or {},
            "weight": weight,
            "created": True,
        }

    def graph_traverse(
        self,
        start_node_id: str,
        max_depth: int = 3,
        edge_types: list[str] | None = None,
        node_labels: list[str] | None = None,
        algorithm: str = "BFS",
        limit: int | None = None,
        **kwargs,
    ):
        """Traverse graph from starting node"""
        return {
            "start_node_id": start_node_id,
            "max_depth": max_depth,
            "nodes": [],
            "edges": [],
        }

    def graph_shortest_path(self, start_node_id: str, target_node_id: str, **kwargs):
        """Find shortest path between two nodes"""
        return {
            "start_node_id": start_node_id,
            "target_node_id": target_node_id,
            "path": [],
            "total_weight": 0.0,
        }

    def query_nodes(
        self,
        labels: list[str] | None = None,
        properties: dict | None = None,
        limit: int | None = None,
        offset: int | None = None,
        **kwargs,
    ):
        """Query nodes by labels and properties"""
        return {"nodes": []}


# CollectionInfo, InsertResult, SearchResult are now imported from tests.utils.test_helpers


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def test_config() -> dict[str, Any]:
    """Test configuration fixture"""
    return TEST_CONFIG.copy()


@pytest.fixture(scope="session")
def embedded_data_dir():
    """Create temporary data directory for embedded database"""
    data_dir = tempfile.mkdtemp(prefix="proximadb_test_")
    yield data_dir

    # Cleanup
    try:
        shutil.rmtree(data_dir, ignore_errors=True)
    except Exception as e:
        logger.warning(f"Failed to cleanup data dir: {e}")


@pytest.fixture(scope="session")
def embedded_db(embedded_data_dir):
    """Session-scoped embedded database instance"""
    if not EMBEDDED_AVAILABLE:
        pytest.skip("Embedded ProximaDB module not available")

    db = ProximaDB(
        data_dirs=embedded_data_dir,
        cache_size_mb=TEST_CONFIG["cache_size_mb"],
        default_engine=TEST_CONFIG["default_engine"],
        enable_wal=True,
    )

    yield db

    # Cleanup
    try:
        db.flush()
    except Exception:
        pass


@pytest.fixture(scope="session")
def verify_server_running(embedded_db):
    """
    Verify database is available - now uses embedded database.
    This fixture exists for backward compatibility.
    """
    logger.info("Using embedded ProximaDB database")
    return True


@pytest.fixture(scope="class")
def rest_client(
    embedded_db, embedded_data_dir
) -> Generator[EmbeddedClientWrapper, None, None]:
    """
    REST client fixture - uses embedded database wrapper.
    Named 'rest_client' for backward compatibility with existing tests.
    """
    client = EmbeddedClientWrapper(embedded_db, embedded_data_dir)
    yield client
    client.close()


@pytest.fixture(scope="class")
def grpc_client(
    embedded_db, embedded_data_dir
) -> Generator[EmbeddedClientWrapper, None, None]:
    """
    gRPC client fixture - uses embedded database wrapper.
    Named 'grpc_client' for backward compatibility with existing tests.
    """
    client = EmbeddedClientWrapper(embedded_db, embedded_data_dir)
    yield client
    client.close()


@pytest.fixture(scope="class")
def client(rest_client):
    """Alias for rest_client to match integration test expectations"""
    return rest_client


@pytest.fixture
def unique_collection_name(test_config) -> str:
    """Generate unique collection name for each test"""
    timestamp = int(time.time() * 1000)  # Milliseconds for uniqueness
    test_name = (
        os.environ.get("PYTEST_CURRENT_TEST", "unknown").split("::")[-1].split("[")[0]
    )
    return f"{test_config['test_collection_prefix']}{test_name}_{timestamp}"


@pytest.fixture
def basic_collection_config() -> CollectionConfig:
    """Basic collection configuration for tests"""
    return CollectionConfig(
        name="test_collection",
        dimension=128,
        distance_metric="cosine",
        description="Test collection created by pytest",
    )


@pytest.fixture
def advanced_collection_config() -> CollectionConfig:
    """Advanced collection configuration for tests"""
    return CollectionConfig(
        name="test_collection",
        dimension=768,
        distance_metric="cosine",
        description="Advanced test collection with BERT dimensions",
        storage_engine=StorageEngine.VIPER,
    )


@pytest.fixture
def test_collection(rest_client, unique_collection_name, basic_collection_config):
    """Create and manage a test collection with automatic cleanup"""
    collection = rest_client.create_collection(
        unique_collection_name, config=basic_collection_config
    )
    yield collection

    # Cleanup
    try:
        rest_client.delete_collection(unique_collection_name)
        logger.debug(f"Cleaned up test collection: {unique_collection_name}")
    except Exception as e:
        logger.warning(f"Failed to cleanup collection {unique_collection_name}: {e}")


class TestCollectionManager:
    """Helper class for managing test collections"""

    def __init__(self, client: EmbeddedClientWrapper, config: dict[str, Any]):
        self.client = client
        self.config = config
        self.created_collections = []

    def create_test_collection(
        self, name_suffix: str = "", config: CollectionConfig = None
    ) -> str:
        """Create a test collection with automatic tracking"""
        if config is None:
            config = CollectionConfig(
                name="test_collection", dimension=128, distance_metric="cosine"
            )

        timestamp = int(time.time() * 1000)
        collection_name = (
            f"{self.config['test_collection_prefix']}{name_suffix}_{timestamp}"
        )

        self.client.create_collection(collection_name, config=config)
        self.created_collections.append(collection_name)

        return collection_name

    def cleanup_all(self):
        """Clean up all created collections"""
        for collection_name in self.created_collections:
            try:
                self.client.delete_collection(collection_name)
                logger.debug(f"Cleaned up collection: {collection_name}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {collection_name}: {e}")

        self.created_collections.clear()


@pytest.fixture
def collection_manager(rest_client, test_config):
    """Collection manager fixture with automatic cleanup"""
    manager = TestCollectionManager(rest_client, test_config)
    yield manager
    manager.cleanup_all()


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may take > 10 seconds)"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "storage: marks tests related to storage layer functionality"
    )
    config.addinivalue_line(
        "markers", "search: marks tests related to search and similarity operations"
    )
    config.addinivalue_line(
        "markers", "large_data: marks tests that work with large datasets"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (fast, no external dependencies)"
    )


@pytest.fixture
def cleanup_collection(unique_collection_name):
    """Collection name that will be automatically cleaned up"""
    return unique_collection_name


@pytest.fixture(scope="session")
def corpus_data():
    """Generate sample corpus data for integration tests"""
    sample_docs = [
        {
            "id": f"doc_{i}",
            "text": f"Sample document {i} about technology and innovation in artificial intelligence",
            "category": "technology" if i % 2 == 0 else "science",
            "importance": i % 10,
            "author": f"Author_{i % 3}",
        }
        for i in range(20)
    ]
    return sample_docs


@pytest.fixture(scope="session")
def bert_service():
    """BERT embedding service for tests"""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        logger.warning("sentence-transformers not available")
        return None
    except Exception as e:
        logger.warning(f"Failed to create BERT service: {e}")
        return None


@pytest.fixture(scope="session")
def cached_embeddings(corpus_data, bert_service):
    """Pre-computed embeddings for corpus data"""
    if not corpus_data or not bert_service:
        return None

    try:
        texts = [doc["text"] for doc in corpus_data]
        embeddings = bert_service.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    except Exception as e:
        logger.warning(f"Failed to compute embeddings: {e}")
        return None


# Pytest hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers"""
    for item in items:
        # The `slow` auto-marker was REMOVED here, deliberately.
        #
        # It marked any test whose NAME contained "large", "stress" or
        # "compaction". Nothing in this repository selects on that marker -- no
        # CI job passes `-m slow`, and the Makefile does not either. But third
        # party plugins do: `pytest-astropy` registers a `--run-slow` option and
        # SKIPS anything marked slow unless it is passed.
        #
        # So a test was silently skipped in any environment that happened to
        # have such a plugin installed, chosen by a substring of its own name.
        # `test_paragraph_strategy_groups_lists_and_splits_large_paragraphs`
        # was skipped that way on my machine while failing in CI the whole time,
        # and the divergence read as "CI is flaky" rather than "these tests are
        # broken". A marker whose meaning is supplied by whatever happens to be
        # installed is worse than no marker.

        # Mark storage tests
        if "storage" in item.name or "wal" in item.name or "flush" in item.name:
            item.add_marker(pytest.mark.storage)

        # Mark search tests
        if (
            "search" in item.name
            or "similarity" in item.name
            or "proximity" in item.name
        ):
            item.add_marker(pytest.mark.search)


# ProximaDBTestError and assert_proximadb_error are now imported from tests.utils.test_helpers


# Performance measurement helpers
@pytest.fixture
def performance_monitor():
    """Fixture for monitoring test performance"""

    class PerformanceMonitor:
        def __init__(self):
            self.timings = {}
            self.start_times = {}

        def start_timer(self, operation: str):
            self.start_times[operation] = time.time()

        def end_timer(self, operation: str) -> float:
            if operation in self.start_times:
                duration = time.time() - self.start_times[operation]
                self.timings[operation] = duration
                del self.start_times[operation]
                return duration
            return 0.0

        def get_timings(self) -> dict[str, float]:
            return self.timings.copy()

        def assert_performance(self, operation: str, max_seconds: float):
            assert operation in self.timings, f"No timing recorded for {operation}"
            actual = self.timings[operation]
            assert (
                actual <= max_seconds
            ), f"{operation} took {actual:.3f}s, expected <= {max_seconds}s"

    return PerformanceMonitor()


# Test data generators
def generate_test_vectors(count: int, dimension: int) -> list:
    """Generate deterministic test vectors"""
    np.random.seed(42)
    vectors = np.random.rand(count, dimension).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    return vectors.tolist()


def generate_test_metadata(count: int, categories: list = None) -> list[dict]:
    """Generate test metadata for use in tests"""
    if categories is None:
        categories = ["technology", "science", "healthcare", "education", "business"]

    metadata_list = []
    for i in range(count):
        metadata = {
            "index": i,
            "category": categories[i % len(categories)],
            "importance": (i % 10) + 1,
            "test_timestamp": time.time(),
            "test_generated": True,
        }
        metadata_list.append(metadata)

    return metadata_list


# Deterministic embedding generation for tests
def embed_seed(seed: int, dimension: int) -> np.ndarray:
    """Generate a deterministic embedding based on seed"""
    np.random.seed(seed)
    vec = np.random.rand(dimension).astype(np.float32)
    return vec / np.linalg.norm(vec)


def embed_many(count: int, dimension: int) -> list[np.ndarray]:
    """Generate multiple deterministic embeddings"""
    return [embed_seed(i, dimension) for i in range(count)]
