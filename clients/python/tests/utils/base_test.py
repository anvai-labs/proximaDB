"""
Base test class for ProximaDB Python SDK tests

Provides common functionality for tests using embedded ProximaDB.
No external server required - uses PyO3/maturin embedded database.
"""

import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

# Add SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from proximadb_sdk.models import VectorRecord

from .embedded_client_adapter import PROXIMADB_IMPORT_ERROR, EmbeddedClientAdapter


class BaseProximaDBTest:
    """
    Base class for ProximaDB tests using embedded database

    Provides:
    - Automatic embedded database setup
    - Test collection management
    - Common test utilities
    - No external server required
    """

    # Class-level shared resources
    _shared_data_dir = None
    _rest_client = None
    _grpc_client = None

    @classmethod
    def setup_class(cls):
        """Initialize embedded database for tests"""
        if PROXIMADB_IMPORT_ERROR is not None:
            pytest.skip(
                "Native ProximaDB embedded module is not installed",
                allow_module_level=False,
            )

        # Create temporary directory for test data
        cls._shared_data_dir = tempfile.mkdtemp(prefix="proximadb_test_")

        # Create embedded client adapters (both point to same embedded DB)
        cls._rest_client = EmbeddedClientAdapter(
            data_dir=cls._shared_data_dir,
            cache_size_mb=256,
            default_engine="sst",
        )
        # For compatibility, grpc_client is same as rest_client in embedded mode
        cls._grpc_client = cls._rest_client

        # Also expose as instance attributes
        cls.rest_client = cls._rest_client
        cls.grpc_client = cls._grpc_client

    @classmethod
    def teardown_class(cls):
        """Clean up embedded database"""
        if cls._rest_client is not None:
            try:
                cls._rest_client.close()
            except Exception:
                pass
            cls._rest_client = None
            cls._grpc_client = None

        # Clean up temporary directory
        if cls._shared_data_dir and Path(cls._shared_data_dir).exists():
            try:
                shutil.rmtree(cls._shared_data_dir)
            except Exception:
                pass

    def setup_method(self):
        """Setup for each test method"""
        self.test_collection_name = f"test_{uuid.uuid4().hex[:8]}"
        self.created_collections = []
        # Ensure instance has access to clients
        self.rest_client = self._rest_client
        self.grpc_client = self._grpc_client

    def teardown_method(self):
        """Cleanup after each test method"""
        # Clean up created collections
        for collection_name in self.created_collections:
            try:
                self.rest_client.delete_collection(collection_name)
            except Exception:
                pass

    def create_collection(
        self,
        client: EmbeddedClientAdapter | None = None,
        name: str | None = None,
        dimension: int = 384,
        engine: str = "sst",
    ) -> str:
        """
        Create a test collection

        Args:
            client: Client to use (defaults to rest_client)
            name: Collection name (auto-generated if None)
            dimension: Vector dimension
            engine: Storage engine

        Returns:
            Collection name
        """
        client = client or self.rest_client
        name = name or f"test_{uuid.uuid4().hex[:8]}"

        client.create_collection(name, dimension, engine)
        self.created_collections.append(name)

        return name

    def insert_test_vectors(
        self,
        client: EmbeddedClientAdapter,
        collection_name: str,
        count: int = 10,
        dimension: int = 384,
        metadata_template: dict[str, Any] | None = None,
    ) -> list[VectorRecord]:
        """
        Insert test vectors into collection

        Args:
            client: Client to use
            collection_name: Collection to insert into
            count: Number of vectors to insert
            dimension: Vector dimension
            metadata_template: Base metadata for vectors

        Returns:
            List of inserted vector records
        """
        from ..embedding_utils import embed_seed

        vectors = []
        for i in range(count):
            # Deterministic realistic embedding based on index
            vector = embed_seed(i, dimension)

            metadata = metadata_template.copy() if metadata_template else {}
            metadata.update(
                {
                    "index": i,
                    "test": True,
                    "category": f"cat_{i % 3}",
                    "value": float(i),
                }
            )

            record = VectorRecord(id=f"vec_{i:04d}", vector=vector, metadata=metadata)
            vectors.append(record)

        # Insert in batch
        response = client.insert_vectors(collection_name, records=vectors)

        # Brief pause for indexing (shorter for embedded)
        time.sleep(0.1)

        return vectors

    def verify_search_results(
        self, results: list[Any], expected_count: int, check_scores: bool = True
    ):
        """
        Verify search results are valid

        Args:
            results: Search results
            expected_count: Expected number of results
            check_scores: Whether to verify scores are descending
        """
        assert (
            len(results) == expected_count
        ), f"Expected {expected_count} results, got {len(results)}"

        if check_scores and len(results) > 1:
            # Verify scores are in descending order
            scores = [
                r.score if hasattr(r, "score") else r.get("score", 0) for r in results
            ]
            assert scores == sorted(
                scores, reverse=True
            ), "Scores not in descending order"

        # Verify each result has required fields
        for result in results:
            # Handle both Pydantic models and dicts
            if hasattr(result, "id"):
                assert result.id is not None
                assert result.score is not None
                assert isinstance(result.score, (int, float))
            else:
                assert "id" in result
                assert "score" in result
                assert isinstance(result["score"], (int, float))

    def wait_for_indexing(self, duration: float = 0.1):
        """Wait for vectors to be indexed (shorter for embedded)"""
        time.sleep(duration)


# Backward compatibility: keep the old function signatures
def ensure_server_running():
    """No-op for embedded mode - server not needed"""
    pass


def create_test_collection(client, name, dimension, engine):
    """Create a test collection using client adapter"""
    return client.create_collection(name, dimension, engine)


def cleanup_test_collections(client):
    """Clean up test collections"""
    try:
        collections = client.list_collections()
        for col in collections:
            col_name = col.name if hasattr(col, "name") else str(col)
            if col_name.startswith("test_"):
                try:
                    client.delete_collection(col_name)
                except Exception:
                    pass
    except Exception:
        pass
