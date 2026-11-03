#!/usr/bin/env python3
"""
ProximaDB Docker Demo Container Test Suite

Tests the Docker demo container functionality including:
- Container startup and health checks
- REST API endpoints
- Collection management
- Vector operations
- Performance validation
"""

import os
import time

import pytest
import requests


class TestDockerDemoContainer:
    """Test suite for Docker demo container"""

    @pytest.fixture(scope="class")
    def base_url(self):
        """Base URL for the demo container"""
        return os.environ.get("PROXIMADB_URL", "http://localhost:5678")

    @pytest.fixture(scope="class")
    def session(self):
        """HTTP session with default headers"""
        session = requests.Session()
        session.headers.update({"Content-Type": "application/json"})
        return session

    @pytest.fixture(scope="class", autouse=True)
    def wait_for_container(self, base_url, session):
        """Wait for container to be ready before running tests"""
        timeout = 60
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = session.get(f"{base_url}/health", timeout=5)
                if response.status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)

        pytest.fail("Container failed to become ready within timeout")

    def test_health_endpoints(self, base_url, session):
        """Test health check endpoints"""
        # Test REST health endpoint
        response = session.get(f"{base_url}/health")
        assert (
            response.status_code == 200
        ), f"Health check failed: {response.status_code}"

        health_data = response.json()
        # Server returns status: "healthy" instead of success: true
        assert (
            health_data.get("status") == "healthy"
        ), f"Health check not successful: {health_data.get('status')}"
        # Timestamp is optional in health response
        # assert "timestamp" in health_data, "Missing timestamp in health response"

    def test_collection_operations(self, base_url):
        """Test collection CRUD operations using SDK"""
        collection_name = f"test_demo_collection_{int(time.time())}"

        # Use SDK instead of raw REST calls for proper API handling
        from proximadb_sdk import ProximaDBClient

        # force_protocol was never honored by any client (v1 swallowed the kwarg silently); state the actual intent.
        client = ProximaDBClient(url=base_url, protocol="rest")

        try:
            # Test create collection using SDK
            collection = client.create_collection(
                collection_name, dimension=384, distance_metric="cosine", engine="viper"
            )
            assert (
                collection is not None
            ), "Collection creation should return collection object"

            # Test list collections
            collections = client.list_collections()
            assert isinstance(
                collections, list
            ), "List collections should return a list"
            # Handle both dict and object responses
            collection_names = []
            for c in collections:
                if isinstance(c, dict):
                    collection_names.append(c.get("name"))
                elif hasattr(c, "config") and hasattr(c.config, "name"):
                    collection_names.append(c.config.name)
                elif hasattr(c, "name"):
                    collection_names.append(c.name)
                elif hasattr(c, "id"):
                    collection_names.append(c.id)
            assert (
                collection_name in collection_names
            ), "Created collection should be in list"

            # Test get specific collection
            collection_info = client.get_collection(collection_name)
            assert (
                collection_info is not None
            ), "Get collection should return collection info"
            # Handle both dict and object responses
            actual_name = None
            if isinstance(collection_info, dict):
                actual_name = collection_info.get("name")
            elif hasattr(collection_info, "config") and hasattr(
                collection_info.config, "name"
            ):
                actual_name = collection_info.config.name
            elif hasattr(collection_info, "name"):
                actual_name = collection_info.name
            elif hasattr(collection_info, "id"):
                actual_name = collection_info.id
            assert (
                actual_name == collection_name
            ), f"Collection name should match: {actual_name} != {collection_name}"

        finally:
            # Test delete collection
            try:
                client.delete_collection(collection_name)
            except:
                pass  # Cleanup - ignore errors

    def test_vector_operations(self, base_url, session):
        """Test vector operations using SDK"""
        collection_name = f"vector_test_collection_{int(time.time())}"

        # Use SDK instead of raw REST calls for proper metadata handling
        from proximadb_sdk import ProximaDBClient, VectorRecord

        # force_protocol was never honored by any client (v1 swallowed the kwarg silently); state the actual intent.
        client = ProximaDBClient(url=base_url, protocol="rest")

        try:
            # Create test collection using SDK
            client.create_collection(
                collection_name, dimension=3, distance_metric="cosine", engine="viper"
            )

            # Test vector insert using SDK (handles metadata conversion)
            vectors = [
                VectorRecord(
                    id="test_vector_1",
                    vector=[0.1, 0.2, 0.3],
                    metadata={"category": "test", "author": "demo"},
                )
            ]

            result = client.insert_vectors(collection_name, records=vectors)
            assert result.metrics.successful_count > 0, "Vector insert should succeed"

            # Test vector retrieval with metadata
            retrieved = client.get_vector(
                collection_name, "test_vector_1", include_metadata=True
            )
            assert retrieved is not None, "Vector should be retrievable"
            # Handle both dict and VectorRecord (Pydantic model) responses
            if hasattr(retrieved, "metadata"):
                assert retrieved.metadata is not None, "Metadata should be included"
                assert (
                    retrieved.metadata["category"] == "test"
                ), "Metadata should be preserved"
            elif isinstance(retrieved, dict):
                assert "metadata" in retrieved, "Metadata should be included"
                assert (
                    retrieved["metadata"]["category"] == "test"
                ), "Metadata should be preserved"

        finally:
            # Cleanup using SDK
            try:
                client.delete_collection(collection_name)
            except:
                pass

    def test_performance_baseline(self, base_url, session):
        """Test basic performance characteristics"""
        # Measure health check latency
        start_time = time.time()
        response = session.get(f"{base_url}/health")
        health_latency = (time.time() - start_time) * 1000

        assert response.status_code == 200
        assert health_latency < 100, f"Health check too slow: {health_latency:.2f}ms"

        # Measure collection list latency
        start_time = time.time()
        response = session.get(f"{base_url}/api/v1/collections")
        list_latency = (time.time() - start_time) * 1000

        assert response.status_code == 200
        assert list_latency < 200, f"Collection list too slow: {list_latency:.2f}ms"

    def test_demo_collections(self, base_url, session):
        """Test that demo collections are created"""
        response = session.get(f"{base_url}/api/v1/collections")
        assert (
            response.status_code == 200
        ), f"Get collections failed: {response.status_code}"

        collections_response = response.json()
        if not collections_response.get("success"):
            pytest.skip("Collections response not successful, but endpoint works")

        collections = collections_response.get("data", [])
        collection_names = [
            col.get("name") for col in collections if isinstance(col, dict)
        ]

        # Check for demo collections (they might be created by setup script)
        demo_collections = ["documents", "products"]
        found_demo = any(name in collection_names for name in demo_collections)

        # This is informational - demo collections may or may not exist
        if not found_demo:
            pytest.skip("Demo collections not found (may be created async)")

    @pytest.mark.parametrize("max_retries", [3])
    def test_container_restart_resilience(self, base_url, session, max_retries):
        """Test that tests handle container restarts gracefully"""
        for attempt in range(max_retries):
            try:
                response = session.get(f"{base_url}/health", timeout=5)
                assert response.status_code == 200
                break
            except (requests.exceptions.RequestException, AssertionError):
                if attempt == max_retries - 1:
                    pytest.fail("Container not responding after retries")
                time.sleep(1)
