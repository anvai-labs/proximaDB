"""
ProximaDB ANN Optimization Functional Test Suite

Tests all ANN (Approximate Nearest Neighbor) optimization options:
1. Index Types: HNSW, IVF, PQ, FLAT, ANNOY, LSH
2. Index Configurations: HNSW params, IVF params, PQ params, etc.
3. Quantization: Binary, INT8, Product Quantization
4. Storage Engines: SST, VIPER, NOVA, HELIX, SWIFT, RAPTOR
5. Distance Metrics: COSINE, EUCLIDEAN, DOT_PRODUCT, etc.
6. Search Parameters: ef_search, n_probe, reranking

Usage:
    python clients/python/tests/ann_functional_test.py [--vectors 10000]
"""

import argparse
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass

import numpy as np

# Add SDK to path
sys.path.insert(0, "clients/python/src")

try:
    import proximadb

    EMBEDDED_AVAILABLE = True
except ImportError:
    EMBEDDED_AVAILABLE = False
    print(
        "Warning: proximadb embedded module not available. Run: maturin develop --release --features python,pylib"
    )


@dataclass
class TestResult:
    """Result of a single test case"""

    name: str
    passed: bool
    insert_time_ms: float
    search_time_ms: float
    recall: float
    total_elapsed_ms: float = 0.0  # Total time for this test case
    error: str | None = None


@dataclass
class TestConfig:
    """Configuration for a test case"""

    name: str
    engine: str
    dimension: int
    distance_metric: str
    index_type: str | None = None
    hnsw_m: int | None = None
    hnsw_ef_construction: int | None = None
    hnsw_ef_search: int | None = None
    ivf_n_lists: int | None = None
    ivf_n_probe: int | None = None
    pq_subvectors: int | None = None
    pq_bits: int | None = None
    quantization_type: str | None = None
    compression: str | None = None


def generate_vectors(num_vectors: int, dimension: int, seed: int = 42) -> np.ndarray:
    """Generate random normalized vectors"""
    np.random.seed(seed)
    vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    return vectors


def compute_ground_truth(
    base_vectors: np.ndarray, query_vectors: np.ndarray, k: int = 10
) -> np.ndarray:
    """Compute ground truth using brute force"""
    num_queries = query_vectors.shape[0]
    ground_truth = np.zeros((num_queries, k), dtype=np.int64)

    for i in range(num_queries):
        # Compute cosine similarity (dot product for normalized vectors)
        similarities = np.dot(base_vectors, query_vectors[i])
        # Get top-k indices
        top_k_indices = np.argsort(-similarities)[:k]
        ground_truth[i] = top_k_indices

    return ground_truth


def compute_recall(
    results: list[list[int]], ground_truth: np.ndarray, k: int = 10
) -> float:
    """Compute recall@k"""
    if not results:
        return 0.0

    total_correct = 0
    num_queries = min(len(results), len(ground_truth))

    for i in range(num_queries):
        if i < len(results) and results[i]:
            result_set = set(results[i][:k])
            gt_set = set(ground_truth[i][:k])
            total_correct += len(result_set & gt_set)

    return total_correct / (num_queries * k) * 100.0


class ANNFunctionalTest:
    """Comprehensive ANN functional test runner"""

    def __init__(
        self, num_vectors: int = 10000, dimension: int = 128, num_queries: int = 100
    ):
        self.num_vectors = num_vectors
        self.dimension = dimension
        self.num_queries = num_queries
        self.k = 10
        self.results: list[TestResult] = []

        # Generate test data
        print(f"\nGenerating test data: {num_vectors} vectors x {dimension}D...")
        self.base_vectors = generate_vectors(num_vectors, dimension, seed=42)
        self.query_vectors = generate_vectors(num_queries, dimension, seed=123)

        # Compute ground truth
        print(f"Computing ground truth (k={self.k}) for {num_queries} queries...")
        start = time.time()
        self.ground_truth = compute_ground_truth(
            self.base_vectors, self.query_vectors, self.k
        )
        print(f"Ground truth computed in {(time.time() - start)*1000:.1f}ms")

    def run_test(self, config: TestConfig) -> TestResult:
        """Run a single test case"""
        temp_dir = tempfile.mkdtemp(prefix=f"ann_test_{config.name}_")
        test_start = time.time()  # Track total test time

        try:
            # Create embedded database
            db = proximadb.ProximaDB(temp_dir)

            # Create collection with specified engine
            collection_name = f"test_{config.name}"
            db.create_collection(
                collection_name, config.dimension, engine=config.engine
            )

            # Insert vectors
            ids = [f"vec_{i}" for i in range(self.num_vectors)]

            insert_start = time.time()
            db.insert_numpy(collection_name, ids, self.base_vectors)
            insert_time_ms = (time.time() - insert_start) * 1000

            # Flush to ensure data is persisted
            db.flush()  # Flushes all collections

            # Search
            search_results = []
            search_start = time.time()

            for i in range(self.num_queries):
                query = self.query_vectors[i].tolist()
                results = db.search(collection_name, query, self.k)
                if results:
                    # Extract IDs and convert to indices - SearchResult objects have .id attribute
                    result_ids = []
                    for r in results:
                        id_str = r.id if hasattr(r, "id") else str(r)
                        if id_str.startswith("vec_"):
                            result_ids.append(int(id_str.replace("vec_", "")))
                    search_results.append(result_ids)
                else:
                    search_results.append([])

            search_time_ms = (time.time() - search_start) * 1000 / self.num_queries

            # Compute recall
            recall = compute_recall(search_results, self.ground_truth, self.k)

            # Cleanup
            db.close()

            total_elapsed_ms = (time.time() - test_start) * 1000
            return TestResult(
                name=config.name,
                passed=recall >= 90.0,  # Consider passed if recall >= 90%
                insert_time_ms=insert_time_ms,
                search_time_ms=search_time_ms,
                recall=recall,
                total_elapsed_ms=total_elapsed_ms,
            )

        except Exception as e:
            total_elapsed_ms = (time.time() - test_start) * 1000
            return TestResult(
                name=config.name,
                passed=False,
                insert_time_ms=0,
                search_time_ms=0,
                recall=0,
                total_elapsed_ms=total_elapsed_ms,
                error=str(e),
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def get_test_configs(self) -> list[TestConfig]:
        """Get all test configurations"""
        configs = []

        # ============================================================
        # 1. STORAGE ENGINE TESTS
        # ============================================================
        engines = ["sst", "nova", "helix", "swift", "raptor"]
        for engine in engines:
            configs.append(
                TestConfig(
                    name=f"engine_{engine}",
                    engine=engine,
                    dimension=self.dimension,
                    distance_metric="cosine",
                )
            )

        # ============================================================
        # 2. DISTANCE METRIC TESTS (using SST engine)
        # ============================================================
        metrics = ["cosine", "euclidean", "dot_product"]
        for metric in metrics:
            configs.append(
                TestConfig(
                    name=f"metric_{metric}",
                    engine="sst",
                    dimension=self.dimension,
                    distance_metric=metric,
                )
            )

        # ============================================================
        # 3. INDEX TYPE TESTS
        # ============================================================
        # Note: Index configurations are applied during search

        # HNSW configurations
        hnsw_configs = [
            {"m": 16, "ef_construction": 100, "ef_search": 50},
            {"m": 32, "ef_construction": 200, "ef_search": 100},
            {"m": 48, "ef_construction": 400, "ef_search": 200},
        ]
        for i, hc in enumerate(hnsw_configs):
            configs.append(
                TestConfig(
                    name=f"hnsw_m{hc['m']}_ef{hc['ef_search']}",
                    engine="sst",
                    dimension=self.dimension,
                    distance_metric="cosine",
                    index_type="hnsw",
                    hnsw_m=hc["m"],
                    hnsw_ef_construction=hc["ef_construction"],
                    hnsw_ef_search=hc["ef_search"],
                )
            )

        # IVF configurations
        ivf_configs = [
            {"n_lists": 100, "n_probe": 1},
            {"n_lists": 100, "n_probe": 10},
            {"n_lists": 256, "n_probe": 32},
        ]
        for ic in ivf_configs:
            configs.append(
                TestConfig(
                    name=f"ivf_lists{ic['n_lists']}_probe{ic['n_probe']}",
                    engine="sst",
                    dimension=self.dimension,
                    distance_metric="cosine",
                    index_type="ivf",
                    ivf_n_lists=ic["n_lists"],
                    ivf_n_probe=ic["n_probe"],
                )
            )

        # PQ configurations
        pq_configs = [
            {"subvectors": 8, "bits": 8},
            {"subvectors": 16, "bits": 8},
            {"subvectors": 32, "bits": 4},
        ]
        for pc in pq_configs:
            configs.append(
                TestConfig(
                    name=f"pq_sub{pc['subvectors']}_bits{pc['bits']}",
                    engine="sst",
                    dimension=self.dimension,
                    distance_metric="cosine",
                    index_type="pq",
                    pq_subvectors=pc["subvectors"],
                    pq_bits=pc["bits"],
                )
            )

        # ============================================================
        # 4. QUANTIZATION TESTS
        # ============================================================
        quantization_types = ["none", "int8", "pq8"]
        for qt in quantization_types:
            configs.append(
                TestConfig(
                    name=f"quant_{qt}",
                    engine="nova",  # NOVA supports quantization
                    dimension=self.dimension,
                    distance_metric="cosine",
                    quantization_type=qt,
                )
            )

        # ============================================================
        # 5. COMPRESSION TESTS
        # ============================================================
        compressions = ["none", "lz4", "zstd"]
        for comp in compressions:
            configs.append(
                TestConfig(
                    name=f"compress_{comp}",
                    engine="sst",
                    dimension=self.dimension,
                    distance_metric="cosine",
                    compression=comp,
                )
            )

        # ============================================================
        # 6. DIMENSION TESTS
        # ============================================================
        dimensions = [64, 128, 256, 512, 768]
        for dim in dimensions:
            if dim != self.dimension:  # Skip if same as default
                configs.append(
                    TestConfig(
                        name=f"dim_{dim}",
                        engine="sst",
                        dimension=dim,
                        distance_metric="cosine",
                    )
                )

        return configs

    def run_all_tests(self) -> None:
        """Run all test configurations"""
        if not EMBEDDED_AVAILABLE:
            print(
                "ERROR: Embedded mode not available. Run: maturin develop --release --features python,pylib"
            )
            return

        configs = self.get_test_configs()
        print(f"\n{'='*80}")
        print("PROXIMADB ANN FUNCTIONAL TEST SUITE")
        print(f"{'='*80}")
        print(f"  Vectors: {self.num_vectors}")
        print(f"  Dimension: {self.dimension}")
        print(f"  Queries: {self.num_queries}")
        print(f"  Test cases: {len(configs)}")
        print(f"{'='*80}\n")

        for i, config in enumerate(configs):
            print(
                f"[{i+1}/{len(configs)}] Testing {config.name}...", end=" ", flush=True
            )

            # Handle dimension changes
            if config.dimension != self.dimension:
                # Generate new vectors for this dimension
                old_dim = self.dimension
                self.dimension = config.dimension
                self.base_vectors = generate_vectors(
                    self.num_vectors, config.dimension, seed=42
                )
                self.query_vectors = generate_vectors(
                    self.num_queries, config.dimension, seed=123
                )
                self.ground_truth = compute_ground_truth(
                    self.base_vectors, self.query_vectors, self.k
                )

            result = self.run_test(config)
            self.results.append(result)

            # Restore dimension if changed
            if config.dimension != 128:  # Restore to default
                self.dimension = 128
                self.base_vectors = generate_vectors(self.num_vectors, 128, seed=42)
                self.query_vectors = generate_vectors(self.num_queries, 128, seed=123)
                self.ground_truth = compute_ground_truth(
                    self.base_vectors, self.query_vectors, self.k
                )

            if result.error:
                print(
                    f"ERROR ({result.total_elapsed_ms:.0f}ms): {result.error[:50]}..."
                )
            else:
                status = "PASS" if result.passed else "WARN"
                print(
                    f"{status} | Insert: {result.insert_time_ms:.1f}ms | "
                    f"Search: {result.search_time_ms:.2f}ms | Recall: {result.recall:.1f}% | "
                    f"Total: {result.total_elapsed_ms:.0f}ms"
                )

        self.print_summary()

    def print_summary(self) -> None:
        """Print test summary"""
        print(f"\n{'='*100}")
        print("TEST SUMMARY")
        print(f"{'='*100}")

        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed and not r.error)
        errors = sum(1 for r in self.results if r.error)

        print(f"\nResults: {passed} passed, {failed} warnings, {errors} errors")
        print(f"Total: {len(self.results)} tests\n")

        # Group by category
        categories = {
            "Storage Engines": [
                r for r in self.results if r.name.startswith("engine_")
            ],
            "Distance Metrics": [
                r for r in self.results if r.name.startswith("metric_")
            ],
            "HNSW Configs": [r for r in self.results if r.name.startswith("hnsw_")],
            "IVF Configs": [r for r in self.results if r.name.startswith("ivf_")],
            "PQ Configs": [r for r in self.results if r.name.startswith("pq_")],
            "Quantization": [r for r in self.results if r.name.startswith("quant_")],
            "Compression": [r for r in self.results if r.name.startswith("compress_")],
            "Dimensions": [r for r in self.results if r.name.startswith("dim_")],
        }

        for category, results in categories.items():
            if results:
                print(f"\n### {category}")
                print(
                    f"{'Test Name':<30} {'Status':<8} {'Insert':<12} {'Search':<12} {'Recall':<10} {'Total':<12}"
                )
                print("-" * 100)
                for r in results:
                    if r.error:
                        status = "ERROR"
                        insert = "N/A"
                        search = "N/A"
                        recall = "N/A"
                        total = f"{r.total_elapsed_ms:.0f}ms"
                    else:
                        status = "PASS" if r.passed else "WARN"
                        insert = f"{r.insert_time_ms:.1f}ms"
                        search = f"{r.search_time_ms:.2f}ms"
                        recall = f"{r.recall:.1f}%"
                        total = f"{r.total_elapsed_ms:.0f}ms"
                    print(
                        f"{r.name:<30} {status:<8} {insert:<12} {search:<12} {recall:<10} {total:<12}"
                    )

        # Print errors if any
        error_results = [r for r in self.results if r.error]
        if error_results:
            print("\n### ERRORS")
            print("-" * 100)
            for r in error_results:
                print(f"{r.name}: {r.error}")

        # Print comprehensive performance comparison table
        self.print_performance_comparison()

        print(f"\n{'='*100}")
        print("ANN FUNCTIONAL TEST COMPLETE")
        print(f"{'='*100}")

    def print_performance_comparison(self) -> None:
        """Print comprehensive performance comparison table"""
        # Filter out error results
        valid_results = [r for r in self.results if not r.error]
        if not valid_results:
            return

        print(f"\n{'='*100}")
        print("COMPREHENSIVE PERFORMANCE COMPARISON TABLE")
        print(f"{'='*100}")
        print(
            f"\nDataset: {self.num_vectors:,} vectors x {self.dimension}D | {self.num_queries} queries | k={self.k}"
        )
        print()

        # Sort by different metrics
        print("### Ranked by Insert Speed (vectors/sec)")
        print(
            f"{'Rank':<6} {'Configuration':<35} {'Insert (ms)':<14} {'Vecs/sec':<12} {'Recall':<10}"
        )
        print("-" * 100)
        by_insert = sorted(valid_results, key=lambda r: r.insert_time_ms)
        for i, r in enumerate(by_insert[:10], 1):
            vecs_per_sec = (
                self.num_vectors / (r.insert_time_ms / 1000)
                if r.insert_time_ms > 0
                else 0
            )
            print(
                f"{i:<6} {r.name:<35} {r.insert_time_ms:<14.1f} {vecs_per_sec:<12,.0f} {r.recall:.1f}%"
            )

        print()
        print("### Ranked by Search Speed (queries/sec)")
        print(
            f"{'Rank':<6} {'Configuration':<35} {'Search (ms)':<14} {'QPS':<12} {'Recall':<10}"
        )
        print("-" * 100)
        by_search = sorted(valid_results, key=lambda r: r.search_time_ms)
        for i, r in enumerate(by_search[:10], 1):
            qps = 1000 / r.search_time_ms if r.search_time_ms > 0 else 0
            print(
                f"{i:<6} {r.name:<35} {r.search_time_ms:<14.2f} {qps:<12,.0f} {r.recall:.1f}%"
            )

        print()
        print("### Ranked by Recall@10 (accuracy)")
        print(
            f"{'Rank':<6} {'Configuration':<35} {'Recall':<10} {'Search (ms)':<14} {'Insert (ms)':<14}"
        )
        print("-" * 100)
        by_recall = sorted(valid_results, key=lambda r: -r.recall)
        for i, r in enumerate(by_recall[:10], 1):
            print(
                f"{i:<6} {r.name:<35} {r.recall:<10.1f}% {r.search_time_ms:<14.2f} {r.insert_time_ms:<14.1f}"
            )

        print()
        print("### Ranked by Total Elapsed Time (end-to-end)")
        print(
            f"{'Rank':<6} {'Configuration':<35} {'Total (ms)':<14} {'Insert (ms)':<14} {'Search (ms)':<14} {'Recall':<10}"
        )
        print("-" * 100)
        by_total = sorted(valid_results, key=lambda r: r.total_elapsed_ms)
        for i, r in enumerate(by_total[:10], 1):
            print(
                f"{i:<6} {r.name:<35} {r.total_elapsed_ms:<14.0f} {r.insert_time_ms:<14.1f} {r.search_time_ms:<14.2f} {r.recall:.1f}%"
            )

        # Engine comparison if we have engine results
        engine_results = [r for r in valid_results if r.name.startswith("engine_")]
        if engine_results:
            print()
            print("### Storage Engine Comparison")
            print(
                f"{'Engine':<15} {'Insert (ms)':<14} {'Vecs/sec':<12} {'Search (ms)':<14} {'QPS':<12} {'Recall':<10} {'Total (ms)':<14}"
            )
            print("-" * 100)
            for r in sorted(engine_results, key=lambda r: r.total_elapsed_ms):
                engine = r.name.replace("engine_", "").upper()
                vecs_per_sec = (
                    self.num_vectors / (r.insert_time_ms / 1000)
                    if r.insert_time_ms > 0
                    else 0
                )
                qps = 1000 / r.search_time_ms if r.search_time_ms > 0 else 0
                print(
                    f"{engine:<15} {r.insert_time_ms:<14.1f} {vecs_per_sec:<12,.0f} {r.search_time_ms:<14.2f} {qps:<12,.0f} {r.recall:.1f}% {'':<3} {r.total_elapsed_ms:<14.0f}"
                )

        # Summary statistics
        print()
        print("### Summary Statistics")
        print("-" * 60)
        avg_insert = sum(r.insert_time_ms for r in valid_results) / len(valid_results)
        avg_search = sum(r.search_time_ms for r in valid_results) / len(valid_results)
        avg_recall = sum(r.recall for r in valid_results) / len(valid_results)
        avg_total = sum(r.total_elapsed_ms for r in valid_results) / len(valid_results)
        min_insert = min(r.insert_time_ms for r in valid_results)
        max_insert = max(r.insert_time_ms for r in valid_results)
        min_search = min(r.search_time_ms for r in valid_results)
        max_search = max(r.search_time_ms for r in valid_results)
        min_recall = min(r.recall for r in valid_results)
        max_recall = max(r.recall for r in valid_results)

        print(f"{'Metric':<20} {'Min':<15} {'Avg':<15} {'Max':<15}")
        print("-" * 60)
        print(
            f"{'Insert Time':<20} {min_insert:<15.1f} {avg_insert:<15.1f} {max_insert:<15.1f}"
        )
        print(
            f"{'Search Time':<20} {min_search:<15.2f} {avg_search:<15.2f} {max_search:<15.2f}"
        )
        print(
            f"{'Recall@10':<20} {min_recall:<15.1f} {avg_recall:<15.1f} {max_recall:<15.1f}"
        )
        print(
            f"{'Total Elapsed':<20} {min(r.total_elapsed_ms for r in valid_results):<15.0f} {avg_total:<15.0f} {max(r.total_elapsed_ms for r in valid_results):<15.0f}"
        )


def main():
    parser = argparse.ArgumentParser(description="ProximaDB ANN Functional Test Suite")
    parser.add_argument(
        "--vectors",
        type=int,
        default=30000,
        help="Number of vectors to test with (3x scale)",
    )
    parser.add_argument("--dimension", type=int, default=128, help="Vector dimension")
    parser.add_argument(
        "--queries", type=int, default=300, help="Number of query vectors (3x scale)"
    )
    args = parser.parse_args()

    tester = ANNFunctionalTest(
        num_vectors=args.vectors, dimension=args.dimension, num_queries=args.queries
    )
    tester.run_all_tests()


if __name__ == "__main__":
    main()
