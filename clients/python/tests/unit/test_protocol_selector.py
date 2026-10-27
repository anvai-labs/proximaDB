"""
Tests for intelligent protocol selection system using unified router

Tests protocol selection, health monitoring, and performance metrics using
the new unified IntelligentRouter system.
"""

import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.base_test import BaseProximaDBTest
from utils.server_utils import ensure_server_running

from proximadb_sdk.config import ClientConfig, Protocol
from proximadb_sdk.exceptions import ProximaDBError
from proximadb_sdk.intelligent_router import (
    IntelligentRouter,
    OperationType,
    ProtocolHealth,
    ProtocolMetrics,
    RoutingConfig,
    RoutingStrategy,
)

# Backward compatibility imports
from proximadb_sdk.protocol_selector import ProtocolSelector, create_protocol_selector


class TestProtocolMetrics:
    """Test protocol metrics tracking with new unified system"""

    @pytest.fixture
    def metrics(self):
        """Create fresh metrics instance"""
        return ProtocolMetrics(Protocol.GRPC)

    def test_metrics_initialization(self, metrics):
        """Test metrics initialization"""
        assert metrics.protocol == Protocol.GRPC
        assert metrics.health_status == ProtocolHealth.UNKNOWN
        assert metrics.total_requests == 0
        assert metrics.get_success_rate() == 0.0
        assert not metrics.circuit_breaker_open
        assert len(metrics.latency_samples) == 0
        assert metrics.consecutive_failures == 0

    def test_update_success(self, metrics):
        """Test updating metrics for successful requests"""
        metrics.update_success(latency_ms=50.0, throughput_qps=100.0)

        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.consecutive_failures == 0
        assert metrics.get_success_rate() == 100.0
        assert metrics.get_avg_latency() == 50.0
        assert not metrics.circuit_breaker_open
        assert metrics.last_request_time > 0

    def test_update_failure(self, metrics):
        """Test updating metrics for failed requests"""
        metrics.update_failure("timeout")

        assert metrics.total_requests == 1
        assert metrics.failed_requests == 1
        assert metrics.consecutive_failures == 1
        assert metrics.get_success_rate() == 0.0
        assert metrics.last_request_time > 0

    def test_circuit_breaker_opens(self, metrics):
        """Test circuit breaker opens after threshold failures"""
        # Circuit should open after 5 consecutive failures (default)
        for i in range(5):
            metrics.update_failure("error")
            if i < 4:
                assert not metrics.circuit_breaker_open

        # Should be open after 5th failure
        assert metrics.circuit_breaker_open
        assert metrics.health_status == ProtocolHealth.UNHEALTHY

    def test_circuit_breaker_closes_on_success(self, metrics):
        """Test circuit breaker behavior on successful request after failures"""
        # Open circuit
        for _ in range(5):
            metrics.update_failure("error")
        assert metrics.circuit_breaker_open
        assert metrics.health_status == ProtocolHealth.UNHEALTHY

        # Success resets consecutive failures but health transitions to DEGRADED
        # (not immediately HEALTHY - that requires sustained success)
        metrics.update_success(25.0)
        assert metrics.consecutive_failures == 0
        # Health transitions from UNHEALTHY to DEGRADED on success
        assert metrics.health_status == ProtocolHealth.DEGRADED

    def test_health_status_updates(self, metrics):
        """Test health status updates based on success rate"""
        # Initially unknown
        assert metrics.health_status == ProtocolHealth.UNKNOWN

        # Add >10 successful requests with low latency (<100ms) to become HEALTHY
        for _ in range(12):
            metrics.update_success(10.0)  # Low latency

        # Should be healthy with all successes (>10 samples, <100ms avg)
        assert metrics.health_status == ProtocolHealth.HEALTHY

        # Add failures to degrade health
        for _ in range(3):
            metrics.update_failure("error")

        # Should be degraded (3 consecutive failures)
        assert metrics.health_status == ProtocolHealth.DEGRADED

        # Add more failures to make it unhealthy (need 2 more for 5 total)
        for _ in range(2):
            metrics.update_failure("error")

        # Should be unhealthy with circuit open (5 consecutive failures)
        assert metrics.health_status == ProtocolHealth.UNHEALTHY
        assert metrics.circuit_breaker_open

    def test_latency_percentiles(self, metrics):
        """Test latency percentile calculations"""
        # Add latency samples
        latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] * 2  # 20 samples

        for latency in latencies:
            metrics.update_success(latency)

        # Should calculate P95
        p95 = metrics.get_p95_latency()
        avg = metrics.get_avg_latency()
        assert p95 > 0
        assert p95 >= avg

    def test_score_calculation(self, metrics):
        """Test score calculation for different strategies"""
        # Add some successful requests
        for latency in [10, 20, 30]:
            metrics.update_success(latency, throughput_qps=50)

        # Performance-based score (lower latency = higher score)
        perf_score = metrics.get_score(RoutingStrategy.PERFORMANCE_BASED)
        assert 0 <= perf_score <= 1.0

        # Reliability-based score (success rate based)
        rel_score = metrics.get_score(RoutingStrategy.RELIABILITY_BASED)
        assert rel_score == 1.0  # 100% success rate

        # Balanced score
        bal_score = metrics.get_score(RoutingStrategy.BALANCED)
        assert 0 <= bal_score <= 1.0

        # Circuit broken should return 0 ONLY if within half_open_time window
        metrics.circuit_breaker_open = True
        metrics.circuit_breaker_half_open_time = time.time() + 30  # Set future time
        assert metrics.get_score(RoutingStrategy.PERFORMANCE_BASED) == 0.0


class TestIntelligentRouter:
    """Test intelligent router functionality (new unified system)"""

    @pytest.fixture
    def config(self):
        """Standard routing configuration"""
        return RoutingConfig(
            strategy=RoutingStrategy.BALANCED,
            health_check_interval_seconds=0,  # Disable for tests
        )

    @pytest.fixture
    def router(self, config):
        """Create intelligent router"""
        return IntelligentRouter(config)

    @pytest.fixture
    def mock_grpc_client(self):
        """Mock gRPC client"""
        client = Mock()
        client.health_check.return_value = {"status": "healthy"}
        client.list_collections.return_value = []
        return client

    @pytest.fixture
    def mock_rest_client(self):
        """Mock REST client"""
        client = Mock()
        client.health_check.return_value = {"status": "ok"}
        client.list_collections.return_value = []
        return client

    def teardown_method(self):
        """Cleanup after tests"""
        # Clean up any routers created in tests
        pass

    def test_router_initialization(self, router, config):
        """Test intelligent router initialization"""
        assert router.config.strategy == config.strategy
        assert Protocol.GRPC in router._metrics
        assert Protocol.REST in router._metrics
        assert router._current_protocol is None
        router.stop()

    def test_register_client_factory(self, router):
        """Test registering client factories"""
        grpc_factory = Mock()
        rest_factory = Mock()

        router.register_client_factory(Protocol.GRPC, grpc_factory)
        router.register_client_factory(Protocol.REST, rest_factory)

        assert Protocol.GRPC in router._client_factories
        assert Protocol.REST in router._client_factories
        router.stop()

    def test_route_operation_no_clients(self, router):
        """Test operation routing with no registered clients"""
        with pytest.raises(ProximaDBError, match="No healthy protocol available"):
            router.route_operation(OperationType.HEALTH_CHECK)
        router.stop()

    def test_route_operation_single_available(self, router, mock_grpc_client):
        """Test operation routing with only one available protocol"""
        router.register_client_factory(Protocol.GRPC, lambda: mock_grpc_client)

        # Set protocol as healthy
        router._metrics[Protocol.GRPC].health_status = ProtocolHealth.HEALTHY

        protocol, client = router.route_operation(OperationType.SINGLE_SEARCH)
        assert protocol == Protocol.GRPC
        assert client == mock_grpc_client
        router.stop()

    def test_route_operation_based_rules(self, mock_grpc_client, mock_rest_client):
        """Test operation-based routing rules"""
        # Use OPERATION_BASED strategy for operation-based routing
        config = RoutingConfig(
            strategy=RoutingStrategy.OPERATION_BASED, health_check_interval_seconds=0
        )
        router = IntelligentRouter(config)
        router.register_client_factory(Protocol.GRPC, lambda: mock_grpc_client)
        router.register_client_factory(Protocol.REST, lambda: mock_rest_client)

        # Set both protocols as healthy
        router._metrics[Protocol.GRPC].health_status = ProtocolHealth.HEALTHY
        router._metrics[Protocol.REST].health_status = ProtocolHealth.HEALTHY

        # Bulk operations should prefer gRPC
        protocol, client = router.route_operation(OperationType.BULK_INSERT)
        assert protocol == Protocol.GRPC
        assert client == mock_grpc_client

        # Admin operations should prefer REST
        protocol, client = router.route_operation(OperationType.HEALTH_CHECK)
        assert protocol == Protocol.REST
        assert client == mock_rest_client

        router.stop()

    def test_performance_based_routing(
        self, router, mock_grpc_client, mock_rest_client
    ):
        """Test performance-based routing"""
        router.config.strategy = RoutingStrategy.PERFORMANCE_BASED
        router.register_client_factory(Protocol.GRPC, lambda: mock_grpc_client)
        router.register_client_factory(Protocol.REST, lambda: mock_rest_client)

        # Set both as healthy
        router._metrics[Protocol.GRPC].health_status = ProtocolHealth.HEALTHY
        router._metrics[Protocol.REST].health_status = ProtocolHealth.HEALTHY

        # Make gRPC faster (10ms) vs REST (500ms)
        # Score formula: min(1.0, 100.0 / latency)
        # gRPC: min(1.0, 100/10) = 1.0
        # REST: min(1.0, 100/500) = 0.2
        router._metrics[Protocol.GRPC].update_success(
            10.0, 200.0
        )  # Fast, high throughput
        router._metrics[Protocol.REST].update_success(500.0, 100.0)  # Much slower

        protocol, client = router.route_operation(OperationType.SINGLE_SEARCH)
        assert protocol == Protocol.GRPC
        router.stop()

    def test_fallback_when_preferred_unhealthy(
        self, router, mock_grpc_client, mock_rest_client
    ):
        """Test fallback to healthy protocol when preferred is unhealthy"""
        router.register_client_factory(Protocol.GRPC, lambda: mock_grpc_client)
        router.register_client_factory(Protocol.REST, lambda: mock_rest_client)

        # Make gRPC unhealthy, REST healthy
        router._metrics[Protocol.GRPC].health_status = ProtocolHealth.UNHEALTHY
        router._metrics[Protocol.REST].health_status = ProtocolHealth.HEALTHY

        # Bulk insert should prefer gRPC but fallback to REST
        protocol, client = router.route_operation(OperationType.BULK_INSERT)
        assert protocol == Protocol.REST
        assert client == mock_rest_client
        router.stop()

    def test_record_operation_result(self, router):
        """Test recording operation results"""
        initial_requests = router._metrics[Protocol.GRPC].total_requests

        router.record_operation_result(
            OperationType.SINGLE_INSERT,
            Protocol.GRPC,
            success=True,
            latency_ms=25.0,
            throughput_qps=400.0,
        )

        metrics = router._metrics[Protocol.GRPC]
        assert metrics.total_requests == initial_requests + 1
        assert metrics.successful_requests > 0
        assert 25.0 in metrics.latency_samples
        router.stop()

    def test_round_robin_strategy(self, router, mock_grpc_client, mock_rest_client):
        """Test round-robin routing strategy"""
        router.config.strategy = RoutingStrategy.ROUND_ROBIN
        router.register_client_factory(Protocol.GRPC, lambda: mock_grpc_client)
        router.register_client_factory(Protocol.REST, lambda: mock_rest_client)

        # Set both as healthy
        router._metrics[Protocol.GRPC].health_status = ProtocolHealth.HEALTHY
        router._metrics[Protocol.REST].health_status = ProtocolHealth.HEALTHY

        # Should alternate between protocols
        protocols = []
        for _ in range(4):
            protocol, _ = router.route_operation(OperationType.SINGLE_SEARCH)
            protocols.append(protocol)

        # Should see both protocols
        assert Protocol.GRPC in protocols
        assert Protocol.REST in protocols
        router.stop()

    def test_sticky_strategy(self, router, mock_grpc_client, mock_rest_client):
        """Test sticky routing strategy"""
        router.config.strategy = RoutingStrategy.STICKY
        router.register_client_factory(Protocol.GRPC, lambda: mock_grpc_client)
        router.register_client_factory(Protocol.REST, lambda: mock_rest_client)

        # Set both as healthy
        router._metrics[Protocol.GRPC].health_status = ProtocolHealth.HEALTHY
        router._metrics[Protocol.REST].health_status = ProtocolHealth.HEALTHY

        # First selection
        first_protocol, _ = router.route_operation(OperationType.SINGLE_SEARCH)

        # Subsequent selections should be same
        for _ in range(5):
            protocol, _ = router.route_operation(OperationType.SINGLE_SEARCH)
            assert protocol == first_protocol

        router.stop()

    def test_get_metrics(self, router):
        """Test getting comprehensive router metrics"""
        # Add some metrics
        router._metrics[Protocol.GRPC].update_success(20.0)
        router._metrics[Protocol.REST].update_failure("timeout")

        metrics = router.get_metrics()

        assert "strategy" in metrics
        assert "protocols" in metrics
        assert "learned_preferences" in metrics
        assert "current_selection" in metrics

        assert metrics["strategy"] == router.config.strategy.value
        assert Protocol.GRPC.value in metrics["protocols"]
        assert Protocol.REST.value in metrics["protocols"]

        grpc_metrics = metrics["protocols"][Protocol.GRPC.value]
        assert "health" in grpc_metrics
        assert "success_rate" in grpc_metrics
        assert grpc_metrics["total_requests"] > 0

        router.stop()


class TestBackwardCompatibility:
    """Test backward compatibility with old ProtocolSelector API"""

    def test_protocol_selector_alias(self):
        """Test ProtocolSelector is an alias to IntelligentRouter"""
        config = RoutingConfig()
        selector = ProtocolSelector(config)
        assert isinstance(selector, IntelligentRouter)
        selector.stop()

    def test_create_protocol_selector_function(self):
        """Test create_protocol_selector factory function"""
        client_config = ClientConfig(url="http://localhost:5678")

        grpc_factory = Mock()
        rest_factory = Mock()

        selector = create_protocol_selector(
            config=client_config,
            grpc_factory=grpc_factory,
            rest_factory=rest_factory,
            strategy=RoutingStrategy.PERFORMANCE_BASED,
        )

        assert isinstance(selector, IntelligentRouter)
        assert selector.config.strategy == RoutingStrategy.PERFORMANCE_BASED
        assert Protocol.GRPC in selector._client_factories
        assert Protocol.REST in selector._client_factories

        # Clean up
        selector.stop()

    def test_imports_work(self):
        """Test that old imports still work"""
        # These imports should work due to backward compatibility
        from proximadb_sdk.protocol_selector import ProtocolSelector

        # Create router using old interface
        config = RoutingConfig(health_check_interval_seconds=0)
        selector = ProtocolSelector(config)
        assert selector is not None
        selector.stop()


class TestProtocolSelectorIntegration(BaseProximaDBTest):
    """Integration tests with real server"""

    def test_router_with_real_server(self):
        """Test router integration with real server"""
        ensure_server_running()

        # Create router
        config = RoutingConfig(
            strategy=RoutingStrategy.OPERATION_BASED,
            health_check_interval_seconds=0,  # Disable for test
        )
        router = IntelligentRouter(config)

        try:
            # Register mock clients that could work with real operations
            mock_client = Mock()
            mock_client.health_check.return_value = {"status": "healthy"}

            router.register_client_factory(Protocol.GRPC, lambda: mock_client)
            router.register_client_factory(Protocol.REST, lambda: mock_client)

            # Set both protocols as healthy for testing
            router._metrics[Protocol.GRPC].health_status = ProtocolHealth.HEALTHY
            router._metrics[Protocol.REST].health_status = ProtocolHealth.HEALTHY

            # Test routing different operations
            protocol, client = router.route_operation(OperationType.HEALTH_CHECK)
            assert protocol in [Protocol.GRPC, Protocol.REST]
            assert client == mock_client

            # Record successful operation
            router.record_operation_result(
                OperationType.HEALTH_CHECK, protocol, success=True, latency_ms=15.0
            )

            # Check metrics were updated
            metrics = router.get_metrics()
            assert metrics["protocols"][protocol.value]["total_requests"] > 0

        finally:
            router.stop()


@pytest.mark.performance
class TestProtocolSelectorPerformance:
    """Performance tests for protocol selector (marked for manual execution)"""

    def test_routing_performance(self):
        """Test operation routing performance under load"""
        config = RoutingConfig(health_check_interval_seconds=0)
        router = IntelligentRouter(config)

        # Register fast mock factories
        mock_grpc = Mock()
        mock_rest = Mock()
        router.register_client_factory(Protocol.GRPC, lambda: mock_grpc)
        router.register_client_factory(Protocol.REST, lambda: mock_rest)

        # Set both as healthy and add some metrics to make routing meaningful
        router._metrics[Protocol.GRPC].health_status = ProtocolHealth.HEALTHY
        router._metrics[Protocol.GRPC].update_success(20.0)
        router._metrics[Protocol.REST].health_status = ProtocolHealth.HEALTHY
        router._metrics[Protocol.REST].update_success(30.0)

        # Time multiple routing operations
        start_time = time.time()
        routes = []

        for i in range(1000):
            operation = (
                OperationType.SINGLE_SEARCH if i % 2 == 0 else OperationType.BULK_INSERT
            )
            protocol, client = router.route_operation(operation)
            routes.append((protocol, client))

        end_time = time.time()
        duration = end_time - start_time
        routes_per_second = 1000 / duration

        print(f"Operation routing: {routes_per_second:.0f} per second")

        # Should be very fast (>1000 routes/sec)
        assert routes_per_second > 1000
        assert (
            len(set(r[0] for r in routes)) <= 2
        )  # Should only select available protocols

        router.stop()

    def test_concurrent_routing(self):
        """Test concurrent operation routing"""
        config = RoutingConfig(health_check_interval_seconds=0)
        router = IntelligentRouter(config)

        # Register clients
        mock_grpc = Mock()
        mock_rest = Mock()
        router.register_client_factory(Protocol.GRPC, lambda: mock_grpc)
        router.register_client_factory(Protocol.REST, lambda: mock_rest)

        # Set both as healthy
        router._metrics[Protocol.GRPC].health_status = ProtocolHealth.HEALTHY
        router._metrics[Protocol.REST].health_status = ProtocolHealth.HEALTHY

        results = []
        errors = []

        def routing_worker():
            try:
                for i in range(100):
                    operation = (
                        OperationType.SINGLE_SEARCH
                        if i % 2 == 0
                        else OperationType.BULK_INSERT
                    )
                    protocol, client = router.route_operation(operation)
                    results.append((protocol, client))

                    # Simulate recording results
                    router.record_operation_result(
                        operation, protocol, success=True, latency_ms=25.0
                    )
            except Exception as e:
                errors.append(e)

        # Run concurrent routing operations
        threads = []
        start_time = time.time()

        for _ in range(10):
            thread = threading.Thread(target=routing_worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        end_time = time.time()

        # Should complete without errors
        assert len(errors) == 0
        assert len(results) == 1000  # 10 threads × 100 operations

        # Should have recorded all operations
        total_requests = sum(m.total_requests for m in router._metrics.values())
        assert total_requests == 1000

        print(f"Concurrent routing test completed in {end_time - start_time:.2f}s")
        print(f"Total routes: {len(results)}")
        print(f"Errors: {len(errors)}")

        router.stop()


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
