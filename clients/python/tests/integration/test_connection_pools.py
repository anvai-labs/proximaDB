"""
Tests for ProximaDB Connection Pooling with Real Server

Validates connection pool functionality for both gRPC and REST protocols
using real ProximaDB server connections.

NOTE: Moved from tests/unit/ to tests/integration/ - these are integration tests
requiring a running ProximaDB server and real connections.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.base_test import BaseProximaDBTest
from utils.server_utils import ensure_server_running

try:
    import grpc

    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False

import httpx

from proximadb_sdk.protocols.connection_pools import (
    GrpcChannelContext,
    GrpcConnectionPool,
    PoolHealth,
    PoolMetrics,
    RestClientContext,
    RestConnectionPool,
)


class TestGrpcConnectionPool(BaseProximaDBTest):
    """Test gRPC connection pooling functionality with real server"""

    @pytest.fixture
    def pool_config(self):
        """Standard pool configuration for real server"""
        return {
            "endpoint": "localhost:5679",
            "pool_size": 3,
            "max_message_size": 32 * 1024 * 1024,
            "compression": "gzip",
            "keepalive_time_ms": 10000,
            "keepalive_timeout_ms": 5000,
            "http2_max_pings_without_data": 0,
            "keepalive_permit_without_calls": True,
            "options": [
                ("grpc.max_receive_message_length", 32 * 1024 * 1024),
                ("grpc.max_send_message_length", 32 * 1024 * 1024),
            ],
        }

    @pytest.mark.skipif(not GRPC_AVAILABLE, reason="gRPC not available")
    def test_pool_initialization(self, pool_config):
        """Test connection pool initialization with real server"""
        pool = GrpcConnectionPool(**pool_config)

        assert pool.endpoint == "localhost:5679"
        assert pool.pool_size == 3
        assert pool.max_message_size == 32 * 1024 * 1024
        assert pool.compression == "gzip"

        # Pool should start with no connections
        assert pool.get_active_connections() == 0
        pool.close()

    @pytest.mark.skipif(not GRPC_AVAILABLE, reason="gRPC not available")
    def test_connection_acquisition_real(self, pool_config):
        """Test acquiring connections from pool with real server"""
        ensure_server_running()
        pool = GrpcConnectionPool(**pool_config)

        # Acquire connection
        with pool.get_connection() as channel_ctx:
            assert isinstance(channel_ctx, GrpcChannelContext)
            assert channel_ctx.channel is not None

            # Test actual connection by creating a stub
            from proximadb_sdk.v1 import vector_pb2_grpc

            stub = vector_pb2_grpc.VectorServiceStub(channel_ctx.channel)

            # Verify stub is created (actual call would need proper request)
            assert stub is not None

            # Active connections should be 1
            assert pool.get_active_connections() == 1

        # After context exit, connection should be available
        time.sleep(0.1)  # Brief pause for connection return
        assert pool.get_active_connections() == 0

        pool.close()

    @pytest.mark.skipif(not GRPC_AVAILABLE, reason="gRPC not available")
    def test_concurrent_connections_real(self, pool_config):
        """Test concurrent connection usage with real server"""
        ensure_server_running()
        pool_config["pool_size"] = 5
        pool = GrpcConnectionPool(**pool_config)

        results = []

        def use_connection(conn_id):
            with pool.get_connection() as channel_ctx:
                # Simulate work with real channel
                from proximadb_sdk.v1 import collection_pb2_grpc, collection_types_pb2

                stub = collection_pb2_grpc.CollectionServiceStub(channel_ctx.channel)

                # Create a simple request
                request = collection_types_pb2.ListCollectionsRequest(limit=1)

                try:
                    # Make actual gRPC call
                    response = stub.ListCollections(request, timeout=5.0)
                    results.append((conn_id, True, len(response.collections)))
                except Exception as e:
                    results.append((conn_id, False, str(e)))

                time.sleep(0.1)  # Simulate work

        # Run concurrent connections
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(use_connection, i) for i in range(10)]
            for future in as_completed(futures):
                future.result()

        # Check results
        assert len(results) == 10
        successful = [r for r in results if r[1]]
        assert len(successful) > 0, "At least some connections should succeed"

        pool.close()

    @pytest.mark.skipif(not GRPC_AVAILABLE, reason="gRPC not available")
    def test_pool_exhaustion_real(self, pool_config):
        """Test pool exhaustion behavior with real connections"""
        ensure_server_running()
        pool_config["pool_size"] = 2
        pool_config["wait_timeout"] = 1.0
        pool = GrpcConnectionPool(**pool_config)

        # Acquire all connections
        contexts = []
        for i in range(2):
            ctx = pool.get_connection()
            ctx.__enter__()
            contexts.append(ctx)

        assert pool.get_active_connections() == 2

        # Try to acquire one more - should timeout
        start_time = time.time()
        with pytest.raises(TimeoutError):
            with pool.get_connection():
                pass

        elapsed = time.time() - start_time
        assert 0.9 < elapsed < 1.5, "Should timeout after ~1 second"

        # Release connections
        for ctx in contexts:
            ctx.__exit__(None, None, None)

        pool.close()

    @pytest.mark.skipif(not GRPC_AVAILABLE, reason="gRPC not available")
    def test_health_check_real(self, pool_config):
        """Test health check with real server"""
        ensure_server_running()
        pool = GrpcConnectionPool(**pool_config)

        # Check health
        health = pool.check_health()

        assert health == PoolHealth.HEALTHY

        # Get metrics
        metrics = pool.get_metrics()
        assert isinstance(metrics, PoolMetrics)
        assert metrics.total_connections >= 0
        assert metrics.active_connections >= 0
        assert metrics.idle_connections >= 0

        pool.close()


class TestRestConnectionPool(BaseProximaDBTest):
    """Test REST connection pooling functionality with real server"""

    @pytest.fixture
    def pool_config(self):
        """Standard pool configuration for real server"""
        return {
            "base_url": "http://localhost:5678",
            "pool_size": 5,
            "timeout": 30.0,
            "max_connections": 10,
            "max_keepalive_connections": 5,
            "keepalive_expiry": 300.0,
            "compression": True,
        }

    def test_pool_initialization(self, pool_config):
        """Test REST connection pool initialization"""
        pool = RestConnectionPool(**pool_config)

        assert pool.base_url == "http://localhost:5678"
        assert pool.pool_size == 5
        assert pool.timeout == 30.0
        assert pool.compression == True

        pool.close()

    def test_connection_acquisition_real(self, pool_config):
        """Test acquiring REST connections with real server"""
        ensure_server_running()
        pool = RestConnectionPool(**pool_config)

        # Acquire connection
        with pool.get_connection() as client_ctx:
            assert isinstance(client_ctx, RestClientContext)
            assert isinstance(client_ctx.client, httpx.Client)

            # Test actual connection
            response = client_ctx.client.get("/health")
            assert response.status_code == 200

            # Check response content
            data = response.json()
            assert data.get("healthy") is True

        pool.close()

    def test_concurrent_requests_real(self, pool_config):
        """Test concurrent REST requests with real server"""
        ensure_server_running()
        pool = RestConnectionPool(**pool_config)

        results = []

        def make_request(req_id):
            with pool.get_connection() as client_ctx:
                try:
                    # Make actual HTTP request
                    response = client_ctx.client.get("/health")
                    results.append((req_id, response.status_code, response.json()))
                except Exception as e:
                    results.append((req_id, -1, str(e)))

                time.sleep(0.05)  # Simulate work

        # Run concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(20)]
            for future in as_completed(futures):
                future.result()

        # Check results
        assert len(results) == 20
        successful = [r for r in results if r[1] == 200]
        assert len(successful) == 20, "All requests should succeed"

        # Verify all got healthy response
        for _, status, data in successful:
            assert data.get("healthy") is True

        pool.close()

    def test_connection_retry_real(self, pool_config):
        """Test connection retry with real server"""
        ensure_server_running()
        pool = RestConnectionPool(**pool_config)

        with pool.get_connection() as client_ctx:
            # Test successful request
            response = client_ctx.client.get("/health")
            assert response.status_code == 200

            # Test 404 (server running but endpoint doesn't exist)
            response = client_ctx.client.get("/nonexistent", follow_redirects=False)
            assert response.status_code == 404

        pool.close()

    def test_compression_real(self, pool_config):
        """Test compression support with real server"""
        ensure_server_running()
        pool = RestConnectionPool(**pool_config)

        with pool.get_connection() as client_ctx:
            # Create some data to send
            test_data = {"data": "x" * 1000}  # Compressible data

            # Make request with compression
            response = client_ctx.client.post(
                "/collections",
                json={
                    "id": f"test_compression_{int(time.time())}",
                    "dimension": 384,
                    "engine": "nova",
                },
                headers={"Accept-Encoding": "gzip"},
            )

            # Should succeed or fail with specific error (not compression error)
            assert response.status_code in [200, 201, 400, 409]

        pool.close()

    def test_metrics_real(self, pool_config):
        """Test metrics collection with real server"""
        ensure_server_running()
        pool = RestConnectionPool(**pool_config)

        # Make some requests
        for i in range(5):
            with pool.get_connection() as client_ctx:
                response = client_ctx.client.get("/health")
                assert response.status_code == 200

        # Check metrics
        metrics = pool.get_metrics()
        assert metrics.total_connections > 0
        assert metrics.total_requests >= 5
        assert metrics.successful_requests >= 5
        assert metrics.failed_requests == 0

        pool.close()


class TestConnectionPoolIntegration(BaseProximaDBTest):
    """Integration tests for connection pools with real operations"""

    def test_grpc_pool_vector_operations(self):
        """Test gRPC pool with actual vector operations"""
        if not GRPC_AVAILABLE:
            pytest.skip("gRPC not available")

        ensure_server_running()

        pool = GrpcConnectionPool(endpoint="localhost:5679", pool_size=3)

        # Create collection and insert vectors using pool
        collection_name = self.create_collection(self.grpc_client)

        with pool.get_connection() as channel_ctx:
            from proximadb_sdk import proximadb_pb2, proximadb_pb2_grpc

            stub = proximadb_pb2_grpc.ProximaDBStub(channel_ctx.channel)

            # Insert vector
            vector_record = proximadb_pb2.VectorRecord(
                id="test_vec_1", vector=[0.1] * 384, metadata={"test": True}
            )

            request = proximadb_pb2.InsertVectorsRequest(
                collection_id=collection_name, vectors=[vector_record]
            )

            response = stub.InsertVectors(request)
            assert response.success

        pool.close()

    def test_rest_pool_collection_operations(self):
        """Test REST pool with actual collection operations"""
        ensure_server_running()

        pool = RestConnectionPool(base_url="http://localhost:5678", pool_size=5)

        collection_name = f"test_pool_{int(time.time())}"

        with pool.get_connection() as client_ctx:
            # Create collection
            response = client_ctx.client.post(
                "/collections",
                json={"id": collection_name, "dimension": 128, "engine": "sst"},
            )
            assert response.status_code in [200, 201]

            # List collections
            response = client_ctx.client.get("/collections")
            assert response.status_code == 200
            collections = response.json()
            assert any(
                c.get("id") == collection_name or c.get("name") == collection_name
                for c in collections
            )

            # Delete collection
            response = client_ctx.client.delete(f"/collections/{collection_name}")
            assert response.status_code in [200, 204]

        pool.close()
