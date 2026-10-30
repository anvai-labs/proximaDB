"""
Tests for intelligent routing functionality using unified router

Tests operation routing, protocol selection, and performance metrics using
the new unified IntelligentRouter system.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.base_test import BaseProximaDBTest
from utils.server_utils import ensure_server_running

from proximadb_sdk.config import Protocol
from proximadb_sdk.intelligent_router import (
    IntelligentRouter,
    OperationType,
    ProtocolHealth,
    ProtocolMetrics,
    RoutingConfig,
    RoutingRule,
    RoutingStrategy,
)

# Backward compatibility
from proximadb_sdk.operation_router import OperationRouter, create_operation_router


class TestProtocolMetrics:
    """Test protocol performance metrics"""

    def test_metrics_initialization(self):
        """Test metrics initialization"""
        metrics = ProtocolMetrics(Protocol.GRPC)

        assert metrics.protocol == Protocol.GRPC
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.health_status == ProtocolHealth.UNKNOWN
        assert metrics.get_success_rate() == 0.0
        assert metrics.consecutive_failures == 0

    def test_record_success(self):
        """Test recording successful operations"""
        metrics = ProtocolMetrics(Protocol.REST)

        # Record some successful operations
        metrics.update_success(10.0, 100.0)
        metrics.update_success(20.0, 150.0)
        metrics.update_success(15.0, 125.0)

        assert metrics.total_requests == 3
        assert metrics.successful_requests == 3
        assert metrics.failed_requests == 0
        assert metrics.get_success_rate() == 100.0  # Returns percentage
        assert metrics.get_avg_latency() == 15.0  # (10+20+15)/3
        assert metrics.consecutive_failures == 0

    def test_record_failure(self):
        """Test recording failed operations"""
        metrics = ProtocolMetrics(Protocol.GRPC)

        # Record success first
        metrics.update_success(10.0)

        # Record failures
        metrics.update_failure("Connection error")
        metrics.update_failure("Timeout")

        assert metrics.total_requests == 3
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 2
        assert abs(metrics.get_success_rate() - 33.33) < 1  # About 33.33%
        assert metrics.consecutive_failures == 2

    def test_health_degradation(self):
        """Test health status degradation with failures"""
        metrics = ProtocolMetrics(Protocol.GRPC)

        # Multiple failures should degrade health
        for _ in range(5):
            metrics.update_failure("error")

        assert metrics.health_status == ProtocolHealth.UNHEALTHY
        assert metrics.circuit_breaker_open == True

    def test_routing_score_calculation(self):
        """Test routing score calculation for different strategies"""
        metrics = ProtocolMetrics(Protocol.REST)

        # Record good performance
        for _ in range(10):
            metrics.update_success(50.0)  # 50ms latency

        # Test different strategies
        perf_score = metrics.get_score(RoutingStrategy.PERFORMANCE_BASED)
        reliability_score = metrics.get_score(RoutingStrategy.RELIABILITY_BASED)
        balanced_score = metrics.get_score(RoutingStrategy.BALANCED)

        assert perf_score > 0
        assert reliability_score == 1.0  # 100% success rate
        assert 0 < balanced_score <= 1.0


class TestRoutingConfig:
    """Test routing configuration"""

    def test_default_config(self):
        """Test default routing configuration"""
        config = RoutingConfig()

        assert config.strategy == RoutingStrategy.BALANCED
        assert config.health_check_interval_seconds == 30.0
        assert config.enable_fallback == True
        assert config.max_fallback_attempts == 2
        assert config.circuit_breaker_failure_threshold == 5
        assert config.enable_adaptive_learning == True

    def test_custom_config(self):
        """Test custom routing configuration"""
        config = RoutingConfig(
            strategy=RoutingStrategy.OPERATION_BASED,
            health_check_interval_seconds=60.0,
            enable_fallback=False,
            max_fallback_attempts=1,
            circuit_breaker_failure_threshold=3,
        )

        assert config.strategy == RoutingStrategy.OPERATION_BASED
        assert config.health_check_interval_seconds == 60.0
        assert config.enable_fallback == False
        assert config.max_fallback_attempts == 1
        assert config.circuit_breaker_failure_threshold == 3


class TestRoutingRules:
    """Test routing rule matching"""

    def test_rule_creation(self):
        """Test routing rule creation"""
        rule = RoutingRule(
            operation=OperationType.BULK_INSERT,
            preferred_protocol=Protocol.GRPC,
            priority=10,
            min_data_size_bytes=1024,
        )

        assert rule.operation == OperationType.BULK_INSERT
        assert rule.preferred_protocol == Protocol.GRPC
        assert rule.priority == 10
        assert rule.min_data_size_bytes == 1024

    def test_rule_matching(self):
        """Test routing rule matching logic"""
        rule = RoutingRule(
            operation=OperationType.SINGLE_INSERT,
            preferred_protocol=Protocol.GRPC,
            min_data_size_bytes=100,
            max_data_size_bytes=10240,
        )

        # Should match
        assert rule.matches(OperationType.SINGLE_INSERT, 5000)

        # Should not match - wrong operation
        assert not rule.matches(OperationType.BULK_INSERT, 5000)

        # Should not match - data too small
        assert not rule.matches(OperationType.SINGLE_INSERT, 50)

        # Should not match - data too large
        assert not rule.matches(OperationType.SINGLE_INSERT, 20000)

    def test_operation_types_available(self):
        """Test all operation types are available"""
        operations = [
            OperationType.BULK_INSERT,
            OperationType.BULK_UPSERT,
            OperationType.BULK_DELETE,
            OperationType.BATCH_SEARCH,
            OperationType.SINGLE_INSERT,
            OperationType.SINGLE_UPSERT,
            OperationType.SINGLE_DELETE,
            OperationType.SINGLE_SEARCH,
            OperationType.GET_VECTOR,
            OperationType.CREATE_COLLECTION,
            OperationType.DELETE_COLLECTION,
            OperationType.UPDATE_COLLECTION,
            OperationType.LIST_COLLECTIONS,
            OperationType.GET_COLLECTION_INFO,
            OperationType.HEALTH_CHECK,
            OperationType.GET_METRICS,
            OperationType.SQL_QUERY,
        ]

        for operation in operations:
            rule = RoutingRule(operation=operation, preferred_protocol=Protocol.GRPC)
            assert rule.operation == operation


class TestIntelligentRouter:
    """Test intelligent router functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.config = RoutingConfig(
            strategy=RoutingStrategy.OPERATION_BASED,
            health_check_interval_seconds=0,  # Disable background monitoring in tests
            enable_fallback=True,
        )
        self.router = IntelligentRouter(self.config)

        # Mock client factories
        self.mock_grpc_client = Mock()
        self.mock_rest_client = Mock()

        self.router.register_client_factory(
            Protocol.GRPC, lambda: self.mock_grpc_client
        )
        self.router.register_client_factory(
            Protocol.REST, lambda: self.mock_rest_client
        )

    def teardown_method(self):
        """Cleanup after tests"""
        if hasattr(self, "router"):
            self.router.stop()

    def test_router_creation(self):
        """Test router can be created"""
        assert self.router is not None
        assert self.router.config.strategy == RoutingStrategy.OPERATION_BASED

    def test_bulk_operations_routed_to_grpc(self):
        """Test bulk operations are routed to gRPC"""
        # Mock healthy protocols
        self.router._metrics[Protocol.GRPC].health_status = ProtocolHealth.HEALTHY
        self.router._metrics[Protocol.REST].health_status = ProtocolHealth.HEALTHY

        protocol, client = self.router.route_operation(OperationType.BULK_INSERT)
        assert protocol == Protocol.GRPC
        assert client == self.mock_grpc_client

    def test_admin_operations_routed_to_rest(self):
        """Test admin operations are routed to REST"""
        # Mock healthy protocols
        self.router._metrics[Protocol.GRPC].health_status = ProtocolHealth.HEALTHY
        self.router._metrics[Protocol.REST].health_status = ProtocolHealth.HEALTHY

        protocol, client = self.router.route_operation(OperationType.HEALTH_CHECK)
        assert protocol == Protocol.REST
        assert client == self.mock_rest_client

    def test_fallback_when_preferred_unhealthy(self):
        """Test fallback to healthy protocol when preferred is unhealthy"""
        # Make gRPC unhealthy, REST healthy
        self.router._metrics[Protocol.GRPC].health_status = ProtocolHealth.UNHEALTHY
        self.router._metrics[Protocol.REST].health_status = ProtocolHealth.HEALTHY

        # Bulk insert should prefer gRPC but fallback to REST
        protocol, client = self.router.route_operation(OperationType.BULK_INSERT)
        assert protocol == Protocol.REST
        assert client == self.mock_rest_client

    def test_routing_strategies(self):
        """Test different routing strategies"""
        strategies = [
            RoutingStrategy.OPERATION_BASED,
            RoutingStrategy.PERFORMANCE_BASED,
            RoutingStrategy.RELIABILITY_BASED,
            RoutingStrategy.BALANCED,
            RoutingStrategy.HYBRID,
            RoutingStrategy.ROUND_ROBIN,
            RoutingStrategy.STICKY,
            RoutingStrategy.ADAPTIVE,
        ]

        for strategy in strategies:
            config = RoutingConfig(strategy=strategy, health_check_interval_seconds=0)
            router = IntelligentRouter(config)
            assert router.config.strategy == strategy
            router.stop()

    def test_operation_result_recording(self):
        """Test recording operation results for metrics"""
        # Record successful operation
        self.router.record_operation_result(
            OperationType.SINGLE_INSERT,
            Protocol.GRPC,
            success=True,
            latency_ms=25.0,
            throughput_qps=400.0,
        )

        metrics = self.router._metrics[Protocol.GRPC]
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0

        # Record failed operation
        self.router.record_operation_result(
            OperationType.SINGLE_INSERT, Protocol.GRPC, success=False, latency_ms=1000.0
        )

        assert metrics.total_requests == 2
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 1

    def test_get_metrics(self):
        """Test getting router metrics"""
        metrics = self.router.get_metrics()

        assert "strategy" in metrics
        assert "protocols" in metrics
        assert "learned_preferences" in metrics
        assert "current_selection" in metrics

        assert metrics["strategy"] == self.config.strategy.value
        assert Protocol.GRPC.value in metrics["protocols"]
        assert Protocol.REST.value in metrics["protocols"]


class TestBackwardCompatibility:
    """Test backward compatibility with old operation router API"""

    def test_operation_router_alias(self):
        """Test OperationRouter is an alias to IntelligentRouter"""
        config = RoutingConfig()
        router = OperationRouter(config)
        assert isinstance(router, IntelligentRouter)
        router.stop()

    def test_create_operation_router_function(self):
        """Test create_operation_router factory function"""
        config = RoutingConfig(strategy=RoutingStrategy.HYBRID)
        router = create_operation_router(config)
        assert isinstance(router, IntelligentRouter)
        assert router.config.strategy == RoutingStrategy.HYBRID
        router.stop()

    def test_imports_work(self):
        """Test that old imports still work"""
        # These imports should work due to backward compatibility
        from proximadb_sdk.operation_router import create_operation_router

        # Create router using old interface
        router = create_operation_router()
        assert router is not None
        router.stop()


class TestOperationRouterIntegration(BaseProximaDBTest):
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

            # Test routing
            protocol, client = router.route_operation(OperationType.HEALTH_CHECK)
            assert protocol in [Protocol.GRPC, Protocol.REST]
            assert client == mock_client

        finally:
            router.stop()
