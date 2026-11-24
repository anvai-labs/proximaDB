"""
ProximaDB Adapter for VectorDBBench.

The default mode is embedded and uses direct PyO3 calls into the in-process
Rust service facade. Remote gRPC/Arrow Flight benchmarks should use a separate
adapter that exercises those protocol surfaces explicitly.
"""

import logging
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ProximaDBVectorDBClient:
    """ProximaDB client for VectorDBBench"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ProximaDB client

        Args:
            config: Configuration dictionary with host, port, etc.
        """
        self.mode = config.get("mode", "embedded")
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 5678)
        self.collection_name = None
        self.dimension = None
        self.metric_type = None
        self.storage_engine = config.get("engine", "sst")
        self._tmpdir: Optional[tempfile.TemporaryDirectory] = None
        self._db = None

        if self.mode != "embedded":
            raise NotImplementedError(
                "VectorDBBench adapter currently supports mode='embedded' only. "
                "Use a dedicated gRPC or Arrow Flight benchmark adapter for remote "
                "protocol measurements."
            )

        try:
            from proximadb_embedded import ProximaDB
        except ImportError as exc:
            raise ImportError(
                "Embedded VectorDBBench requires the proximadb_embedded PyO3 package. "
                "Build it with: cd clients/python-embedded && "
                "maturin develop -m ../../Cargo.toml --release --features python,pylib"
            ) from exc

        data_dir = config.get("data_dir")
        if data_dir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="proximadb-vdbbench-")
            data_dir = self._tmpdir.name

        self._db = ProximaDB(
            data_dirs=data_dir,
            cache_size_mb=int(config.get("cache_size_mb", 1024)),
            default_engine=config.get("engine", "sst"),
            enable_wal=bool(config.get("enable_wal", True)),
        )
        logger.info("Connected to embedded ProximaDB at %s", data_dir)

    def create_collection(self, collection_name: str, dimension: int, metric_type: str):
        """
        Create a collection for benchmarking

        Args:
            collection_name: Name of the collection
            dimension: Vector dimension
            metric_type: Distance metric (L2, IP, COSINE)
        """
        self.collection_name = collection_name
        self.dimension = dimension
        self.metric_type = metric_type

        requested_engine = str(self.storage_engine).lower()
        if requested_engine in {"sst", "nova", "swift", "raptor", "helix"}:
            storage_engine = requested_engine
        else:
            storage_engine = "sst"

        self._db.create_collection(
            collection_name,
            dimension=dimension,
            engine=storage_engine,
        )
        logger.info(
            "Created embedded collection: %s (dim=%s, metric=%s, engine=%s)",
            collection_name,
            dimension,
            metric_type,
            storage_engine,
        )

    def insert_vectors(self, vectors: np.ndarray, ids: List[int] = None):
        """
        Insert vectors into the collection

        Args:
            vectors: Numpy array of vectors (N x D)
            ids: Optional list of vector IDs
        """
        if ids is None:
            ids = list(range(len(vectors)))

        if self.collection_name is None:
            raise RuntimeError("create_collection must be called before insert_vectors")

        vector_array = np.asarray(vectors, dtype=np.float32)
        string_ids = [str(vector_id) for vector_id in ids]
        if hasattr(self._db, "insert_numpy"):
            inserted = self._db.insert_numpy(self.collection_name, string_ids, vector_array)
        else:
            inserted = self._db.insert(
                self.collection_name,
                string_ids,
                vector_array.tolist(),
            )
        logger.info("Inserted %s vectors through embedded bindings", inserted)

    def create_index(self, index_params: Dict[str, Any]):
        """
        Create index on the collection

        Args:
            index_params: Index parameters (e.g., HNSW M, efConstruction)
        """
        # The embedded collection API currently chooses index behavior through
        # collection configuration and engine defaults. Keep the hook explicit so
        # VectorDBBench configs remain visible without pretending to build a
        # separate durable index.
        logger.info("Index params recorded for embedded benchmark: %s", index_params)

    def search(
        self, query_vectors: np.ndarray, k: int
    ) -> Tuple[List[List[Any]], List[List[float]]]:
        """
        Search for nearest neighbors

        Args:
            query_vectors: Query vectors (N x D)
            k: Number of neighbors to return

        Returns:
            Tuple of (ids, distances) for each query
        """
        start_time = time.time()

        if self.collection_name is None:
            raise RuntimeError("create_collection must be called before search")

        ids: List[List[Any]] = []
        distances: List[List[float]] = []
        for query in np.asarray(query_vectors, dtype=np.float32):
            if hasattr(self._db, "search_numpy"):
                results = self._db.search_numpy(self.collection_name, query, top_k=k)
            else:
                results = self._db.search(self.collection_name, query.tolist(), top_k=k)

            result_ids: List[Any] = []
            result_distances: List[float] = []
            for result in results:
                raw_id = getattr(result, "id", "")
                try:
                    result_ids.append(int(raw_id))
                except (TypeError, ValueError):
                    result_ids.append(raw_id)
                result_distances.append(float(getattr(result, "score", 0.0)))
            ids.append(result_ids)
            distances.append(result_distances)

        elapsed = time.time() - start_time
        logger.debug("Search completed for %s queries in %.3fs", len(query_vectors), elapsed)

        return ids, distances

    def get_memory_usage(self) -> int:
        """
        Get current memory usage in MB

        Returns:
            Memory usage in MB
        """
        # Process RSS is what VectorDBBench expects for an embedded benchmark:
        # the database lives in this Python process.
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def disconnect(self):
        """Disconnect from ProximaDB"""
        if self._db is not None and hasattr(self._db, "close"):
            self._db.close()
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
        logger.info("Disconnected from embedded ProximaDB")


class VectorDBBenchAdapter:
    """Adapter for running VectorDBBench with ProximaDB"""

    @staticmethod
    def run_benchmark(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run VectorDBBench benchmark

        Args:
            config: Benchmark configuration

        Returns:
            Benchmark results dictionary
        """
        client = ProximaDBVectorDBClient(config)

        results = {
            'timestamp': time.time(),
            'config': config,
            'metrics': {}
        }

        try:
            # Setup
            collection_name = config.get('collection_name', 'benchmark')
            dimension = config.get('dimension', 128)
            metric_type = config.get('metric_type', 'L2')

            client.create_collection(collection_name, dimension, metric_type)

            # Load data
            # Implementation would load actual dataset
            vectors = np.random.rand(10000, dimension).astype(np.float32)
            client.insert_vectors(vectors)

            # Create index
            index_params = config.get('index_params', {})
            client.create_index(index_params)

            # Benchmark search
            query_vectors = np.random.rand(100, dimension).astype(np.float32)
            k = 100

            # Measure search performance
            start_time = time.time()
            ids, distances = client.search(query_vectors, k)
            elapsed = time.time() - start_time

            # Calculate metrics
            qps = len(query_vectors) / elapsed
            avg_latency = (elapsed / len(query_vectors)) * 1000  # ms

            results['metrics'] = {
                'qps': qps,
                'avg_latency_ms': avg_latency,
                'memory_mb': client.get_memory_usage(),
                'recall': None,
                'total_queries': len(query_vectors)
            }

            logger.info(f"Benchmark complete: QPS={qps:.0f}, Latency={avg_latency:.2f}ms")

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise
        finally:
            client.disconnect()

        return results


# Example usage
if __name__ == '__main__':
    config = {
        'host': 'localhost',
        'port': 5678,
        'collection_name': 'vectordbbench_test',
        'dimension': 128,
        'metric_type': 'L2',
        'index_params': {
            'M': 16,
            'efConstruction': 200
        }
    }

    adapter = VectorDBBenchAdapter()
    results = adapter.run_benchmark(config)

    print("Benchmark Results:")
    print(f"  QPS: {results['metrics']['qps']:.0f}")
    print(f"  Latency: {results['metrics']['avg_latency_ms']:.2f} ms")
    print(f"  Memory: {results['metrics']['memory_mb']:.0f} MB")
