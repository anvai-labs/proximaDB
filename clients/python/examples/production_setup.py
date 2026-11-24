#!/usr/bin/env python3
"""


STATUS: ⚠️  Requires Future Feature
SDK Version: v1.1+ (requires ResilientProximaDBClient)
Server Version: v0.2.0+
Test Result: SKIP - Resilient client wrapper not yet implemented

Production Setup Example for ProximaDB Python SDK v1.0

This example demonstrates production-ready configurations:
- Connection pooling and health checks
- Circuit breakers for fault tolerance
- Retry strategies with backoff
- Request interceptors for auth/logging
- Caching for performance
- Telemetry and monitoring
- Graceful shutdown

"""

# import asyncio
import logging
import os
import signal
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from proximadb import ResilientProximaDBClient, ClientConfig
from proximadb.models import (
    CollectionConfig,
    VectorRecord,
    DistanceMetric,
    StorageEngine,
)
from proximadb.interceptors import (
    InterceptorChain,
    AuthenticationInterceptor,
    LoggingInterceptor,
    MetadataInterceptor,
    ValidationInterceptor,
    MetricsInterceptor,
    RetryInterceptor,
    CachingInterceptor,
)
from proximadb.retry import RetryConfig, BackoffStrategy
from proximadb.cache import CacheManager, QueryCache, EvictionPolicy
from proximadb.telemetry import init_telemetry, ConsoleExporter, HTTPExporter
from proximadb.exceptions import ProximaDBError, TransportError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ProductionProximaDBClient:
    """Production-ready ProximaDB client with all enterprise features"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client: Optional[ResilientProximaDBClient] = None
        self.telemetry = None
        self._shutdown = False

    def initialize(self):
        """Initialize client with production configuration"""
        logger.info("Initializing production ProximaDB client...")

        # 1. Initialize telemetry
        self._setup_telemetry()

        # 2. Create client configuration
        client_config = ClientConfig(
            url=self.config["url"],
            timeout=self.config.get("timeout", 30.0),
            max_retries=self.config.get("max_retries", 3),
            headers={
                "User-Agent": f"ProximaDB-Python/{self.config.get('app_name', 'production')}",
                "X-Request-ID": "will-be-set-by-interceptor",
            },
        )

        # 3. Create resilient client with pooling and circuit breakers
        self.client = ResilientProximaDBClient(
            config=client_config,
            pool_config=self._get_pool_config(),
            circuit_breaker_config=self._get_circuit_breaker_config(),
        )

        # 4. Set up interceptors
        self.client.set_interceptors(self._create_interceptor_chain())

        # 5. Configure retry strategy
        self.client.set_retry_config(self._get_retry_config())

        # 6. Set up caching
        self.client.set_cache_manager(self._create_cache_manager())

        # 7. Connect and verify
        self._verify_connection()

        logger.info("Production client initialized successfully")

    def _get_pool_config(self) -> Dict[str, Any]:
        """Get connection pool configuration"""
        return {
            "min_size": self.config.get("pool_min_size", 10),
            "max_size": self.config.get("pool_max_size", 50),
            "max_idle_time": self.config.get("pool_idle_timeout", 300),
            "health_check_interval": 30,
            "connection_timeout": 10.0,
            "enable_keep_alive": True,
            "keep_alive_interval": 60,
        }

    def _get_circuit_breaker_config(self) -> Dict[str, Any]:
        """Get circuit breaker configuration"""
        return {
            "failure_threshold": 5,  # Failures before opening
            "success_threshold": 2,  # Successes to close
            "timeout": 60.0,  # Seconds before half-open
            "error_threshold_percentage": 50.0,
            "min_requests": 10,  # Min requests before evaluation
            "excluded_exceptions": [ValidationError],  # Don't trip on client errors
        }

    def _get_retry_config(self) -> RetryConfig:
        """Get retry configuration"""
        return RetryConfig(
            max_attempts=5,
            initial_delay=0.5,
            max_delay=30.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER,
            retry_on_exceptions={TransportError, ConnectionError, TimeoutError},
            retry_on_status_codes={429, 502, 503, 504},
            on_retry=lambda attempt, delay: logger.warning(
                f"Retry attempt {attempt} after {delay:.2f}s delay"
            ),
        )

    def _create_interceptor_chain(self) -> InterceptorChain:
        """Create interceptor chain for production"""
        interceptors = []

        # Authentication
        if self.config.get("api_key"):
            interceptors.append(
                AuthenticationInterceptor(
                    auth_token=self.config["api_key"],
                    auth_scheme="Bearer",
                    auth_header="Authorization",
                )
            )

        # Request metadata
        interceptors.append(
            MetadataInterceptor(
                {
                    "app_name": self.config.get("app_name", "production"),
                    "environment": self.config.get("environment", "production"),
                    "version": self.config.get("version", "1.0.0"),
                }
            )
        )

        # Validation
        interceptors.append(
            ValidationInterceptor(
                validate_vectors=True,
                validate_metadata=True,
                max_vector_dimension=self.config.get("max_dimension", 2048),
                max_metadata_size=1024 * 1024,  # 1MB
            )
        )

        # Structured logging
        if self.config.get("enable_request_logging", True):
            interceptors.append(
                LoggingInterceptor(
                    log_level=logging.INFO,
                    log_request_body=False,  # Don't log sensitive data
                    log_response_body=False,
                    max_body_length=500,
                    logger=logger,
                )
            )

        # Caching
        interceptors.append(
            CachingInterceptor(
                cache_ttl=300.0,  # 5 minutes
                cache_operations=["get_collection", "get_vector", "list_collections"],
                max_cache_size=1000,
            )
        )

        # Metrics collection
        interceptors.append(MetricsInterceptor())

        # Retry marking
        interceptors.append(
            RetryInterceptor(
                max_retries=3,
                retry_delay=1.0,
                retry_on_errors=[ConnectionError, TimeoutError],
            )
        )

        return InterceptorChain(interceptors)

    def _create_cache_manager(self) -> CacheManager:
        """Create cache manager for production"""
        return CacheManager(
            query_cache=QueryCache(
                default_ttl=300.0,  # 5 minutes
                cache_search=True,
                cache_get=True,
                max_size=10000,
                eviction_policy=EvictionPolicy.LRU,
            ),
            enable_caching=self.config.get("enable_caching", True),
        )

    def _setup_telemetry(self):
        """Set up telemetry and monitoring"""
        exporters = []

        # Console exporter for development
        if self.config.get("telemetry_console", False):
            exporters.append(ConsoleExporter())

        # HTTP exporter for production
        if metrics_endpoint := self.config.get("metrics_endpoint"):
            exporters.append(
                HTTPExporter(
                    endpoint=metrics_endpoint,
                    headers={
                        "Authorization": f"Bearer {self.config.get('metrics_api_key', '')}"
                    },
                )
            )

        self.telemetry = init_telemetry(
            exporters=exporters,
            export_interval=60.0,  # Export every minute
            service_name=self.config.get("app_name", "proximadb-client"),
            service_version=self.config.get("version", "1.0.0"),
        )

        self.telemetry.start()

    def _verify_connection(self):
        """Verify client can connect to server"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                collections = self.client.alist_collections()
                logger.info(
                    f"Connected to ProximaDB server, found {len(collections)} collections"
                )
                return
            except Exception as e:
                if attempt < max_attempts - 1:
                    logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                    time.sleep(2**attempt)
                else:
                    raise ConnectionError(f"Failed to connect to ProximaDB: {e}")

    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = {"status": "healthy", "timestamp": time.time(), "checks": {}}

        # Check connection pool
        try:
            pool_stats = self.client.get_pool_stats()
            health_status["checks"]["connection_pool"] = {
                "status": "healthy",
                "active_connections": pool_stats["in_use_connections"],
                "available_connections": pool_stats["available_connections"],
                "total_connections": pool_stats["total_connections"],
            }
        except Exception as e:
            health_status["checks"]["connection_pool"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["status"] = "degraded"

        # Check circuit breakers
        try:
            cb_stats = self.client.get_circuit_breaker_stats()
            all_closed = all(stats["state"] == "closed" for stats in cb_stats.values())
            health_status["checks"]["circuit_breakers"] = {
                "status": "healthy" if all_closed else "degraded",
                "breakers": cb_stats,
            }
            if not all_closed:
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["checks"]["circuit_breakers"] = {
                "status": "unhealthy",
                "error": str(e),
            }

        # Check server connectivity
        try:
            start = time.time()
            self.client.alist_collections()
            latency = (time.time() - start) * 1000
            health_status["checks"]["server_connectivity"] = {
                "status": "healthy",
                "latency_ms": latency,
            }
        except Exception as e:
            health_status["checks"]["server_connectivity"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["status"] = "unhealthy"

        # Check cache performance
        if self.client._cache_manager:
            cache_stats = self.client._cache_manager.get_stats()
            health_status["checks"]["cache"] = {
                "status": "healthy",
                "hit_rate": cache_stats["query_cache"]["hit_rate"],
                "entries": cache_stats["query_cache"]["entries"],
            }

        return health_status

    def graceful_shutdown(self):
        """Perform graceful shutdown"""
        if self._shutdown:
            return

        self._shutdown = True
        logger.info("Starting graceful shutdown...")

        # 1. Stop accepting new requests
        logger.info("Stopping new requests...")

        # 2. Wait for in-flight requests (with timeout)
        logger.info("Waiting for in-flight requests...")
        time.sleep(2)  # Simple wait, could be more sophisticated

        # 3. Export final metrics
        if self.telemetry:
            logger.info("Exporting final metrics...")
            self.telemetry.flush()
            self.telemetry.stop()

        # 4. Close connections
        if self.client:
            logger.info("Closing connections...")
            self.client.adisconnect()

        logger.info("Graceful shutdown completed")

    def __getattr__(self, name):
        """Proxy attribute access to underlying client"""
        return getattr(self.client, name)


@asynccontextmanager
def create_production_client(config: Dict[str, Any]):
    """Context manager for production client lifecycle"""
    client = ProductionProximaDBClient(config)

    try:
        client.initialize()
        yield client
    finally:
        client.graceful_shutdown()


def demo_production_operations(client: ProductionProximaDBClient):
    """Demonstrate production operations with monitoring"""
    collection_name = "production_demo"

    # Create collection with production settings
    logger.info(f"Creating collection '{collection_name}'...")

    config = CollectionConfig(
        name=collection_name,
        dimension=768,  # BERT-base dimension
        distance_metric=DistanceMetric.COSINE,
        storage_engine=StorageEngine.NOVA,
        metadata={
            "description": "Production demo collection",
            "created_by": "production_setup.py",
            "environment": "production",
        },
    )

    try:
        client.adelete_collection(collection_name)
    except:
        pass

    collection = client.acreate_collection(config)
    logger.info(f"Collection created: {collection.id}")

    # Insert vectors with monitoring
    logger.info("Inserting vectors...")
    vectors = []
    for i in range(100):
        vectors.append(
            VectorRecord(
                id=f"prod_vec_{i:04d}",
                vector=np.random.randn(768).tolist(),
                metadata={
                    "index": i,
                    "timestamp": time.time(),
                    "environment": "production",
                },
            )
        )

    response = client.ainsert_vectors(collection_name, vectors)
    logger.info(f"Inserted {response.success_count} vectors")

    # Perform searches with caching
    logger.info("Performing cached searches...")
    query_vector = vectors[0].vector

    # First search (cache miss)
    start = time.time()
    results1 = client.search(collection_name, query_vector, top_k=10)
    time1 = time.time() - start
    logger.info(f"First search took {time1*1000:.2f}ms (cache miss)")

    # Second search (cache hit)
    start = time.time()
    results2 = client.search(collection_name, query_vector, top_k=10)
    time2 = time.time() - start
    logger.info(f"Second search took {time2*1000:.2f}ms (cache hit)")

    # Health check
    health = client.health_check()
    logger.info(f"Health status: {health['status']}")

    # Clean up
    client.adelete_collection(collection_name)
    logger.info("Demo collection deleted")


def main():
    """Main function demonstrating production setup"""

    # Production configuration (typically from environment/config file)
    config = {
        "url": os.getenv("PROXIMADB_URL", "http://localhost:5678"),
        "api_key": os.getenv("PROXIMADB_API_KEY"),
        "app_name": "production-demo",
        "environment": os.getenv("ENVIRONMENT", "production"),
        "version": "1.0.0",
        # Connection settings
        "timeout": 30.0,
        "max_retries": 3,
        "pool_min_size": 10,
        "pool_max_size": 50,
        "pool_idle_timeout": 300,
        # Features
        "enable_caching": True,
        "enable_request_logging": True,
        "telemetry_console": True,
        "metrics_endpoint": os.getenv("METRICS_ENDPOINT"),
        "metrics_api_key": os.getenv("METRICS_API_KEY"),
        # Limits
        "max_dimension": 2048,
    }

    print("🚀 Production Setup Example for ProximaDB")
    print("=" * 50)

    # Set up signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create and use production client
    with create_production_client(config) as client:
        # Run demo operations
        demo_production_operations(client)

        # Show production metrics
        print("\n📊 Production Metrics:")
        print("-" * 50)

        # Pool stats
        pool_stats = client.get_pool_stats()
        print(f"Connection Pool:")
        print(f"  - Active: {pool_stats['in_use_connections']}")
        print(f"  - Available: {pool_stats['available_connections']}")
        print(f"  - Total: {pool_stats['total_connections']}")

        # Circuit breaker stats
        cb_stats = client.get_circuit_breaker_stats()
        print(f"\nCircuit Breakers:")
        for operation, stats in cb_stats.items():
            print(
                f"  - {operation}: {stats['state']} "
                f"(errors: {stats['error_percentage']:.1f}%)"
            )

        # Cache stats
        cache_stats = client._cache_manager.get_stats()
        print(f"\nCache Performance:")
        print(f"  - Hit rate: {cache_stats['query_cache']['hit_rate']:.1%}")
        print(f"  - Entries: {cache_stats['query_cache']['entries']}")

        # Health check
        health = client.health_check()
        print(f"\nHealth Status: {health['status'].upper()}")

        # Wait for shutdown signal (in production)
        print("\n✅ Production client running. Press Ctrl+C to shutdown...")
        try:
            shutdown_event.wait()
        except KeyboardInterrupt:
            pass

    print("\n✅ Production example completed with graceful shutdown!")


if __name__ == "__main__":
    import numpy as np  # For demo data generation

    main()
