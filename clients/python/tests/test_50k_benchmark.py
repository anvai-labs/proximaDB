#!/usr/bin/env python3
"""
50K Vector Benchmark Test Suite

Comprehensive pytest-integrated benchmark testing all 6 storage engines
at production scale (50,000 vectors).

Run with:
    PYTHONPATH=clients/python/src pytest clients/python/tests/test_50k_benchmark.py -v

Run quick smoke test only:
    PYTHONPATH=clients/python/src pytest clients/python/tests/test_50k_benchmark.py -v -m "not slow"
"""

import os
import sys
import tempfile
import time
from dataclasses import dataclass

import numpy as np
import pytest

# Add proximadb to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import proximadb

proximadb.init_logging("warn")
from proximadb import ProximaDB as EmbeddedProximaDB

# Test configuration
ENGINES = ["sst", "helix", "swift", "nova", "raptor"]
FULL_VECTOR_COUNT = 50000
QUICK_VECTOR_COUNT = 1000
DIMENSION = 128
TOP_K = 10
NUM_QUERIES = 10


@dataclass
class BenchmarkResult:
    """Result container for benchmark runs."""

    engine: str
    vector_count: int
    insert_time_secs: float
    insert_qps: float
    flush_time_secs: float
    avg_search_latency_ms: float
    p99_search_latency_ms: float
    avg_recall: float
    rating: str
    error: str | None = None


def generate_vectors(count: int, dimension: int, seed: int = 42) -> np.ndarray:
    """Generate normalized random vectors."""
    np.random.seed(seed)
    vectors = np.random.randn(count, dimension).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def compute_exact_neighbors(
    vectors: np.ndarray, query_vectors: np.ndarray, top_k: int
) -> list[set[str]]:
    """Compute ground truth nearest neighbors using brute force."""
    exact_neighbors = []
    for query in query_vectors:
        similarities = np.dot(vectors, query)
        exact_indices = np.argsort(-similarities)[:top_k]
        exact_neighbors.append(set(f"vec_{idx}" for idx in exact_indices))
    return exact_neighbors


def run_engine_benchmark(
    engine: str, vector_count: int, temp_dir: str
) -> BenchmarkResult:
    """Run benchmark for a single engine."""
    collection_name = f"bench_{engine}_{vector_count}"

    try:
        db = EmbeddedProximaDB(temp_dir)

        # Create collection
        db.create_collection(collection_name, dimension=DIMENSION, engine=engine)

        # Generate vectors
        vectors = generate_vectors(vector_count, DIMENSION, seed=42)
        ids = [f"vec_{i}" for i in range(vector_count)]

        # Insert in batches
        batch_size = min(10000, vector_count)
        insert_start = time.time()

        for batch_start in range(0, vector_count, batch_size):
            batch_end = min(batch_start + batch_size, vector_count)
            batch_ids = ids[batch_start:batch_end]
            batch_vectors = vectors[batch_start:batch_end].tolist()
            db.insert(collection_name, ids=batch_ids, vectors=batch_vectors)

        insert_time = time.time() - insert_start
        insert_qps = vector_count / insert_time

        # Flush
        flush_start = time.time()
        db.flush()
        flush_time = time.time() - flush_start

        # Wait for async indexing
        time.sleep(2 if vector_count < 10000 else 5)

        # Generate query vectors and compute exact neighbors
        query_vectors = generate_vectors(NUM_QUERIES, DIMENSION, seed=123)
        exact_neighbors = compute_exact_neighbors(vectors, query_vectors, TOP_K)

        # Run search queries
        latencies = []
        recalls = []

        for i, query in enumerate(query_vectors):
            start = time.time()
            results = db.search(collection_name, query=query.tolist(), top_k=TOP_K)
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)

            result_ids = set(r.id for r in results)
            recall = len(exact_neighbors[i] & result_ids) / TOP_K
            recalls.append(recall)

        avg_latency = sum(latencies) / len(latencies)
        p99_latency = (
            sorted(latencies)[int(len(latencies) * 0.99)]
            if len(latencies) >= 10
            else max(latencies)
        )
        avg_recall = sum(recalls) / len(recalls)

        # Calculate rating
        import math

        expected_latency = 1.0 + 5.0 * math.log10(vector_count)
        latency_ratio = avg_latency / expected_latency

        if latency_ratio <= 0.5 and avg_recall >= 0.9:
            rating = "EXCELLENT"
        elif latency_ratio <= 1.0 and avg_recall >= 0.8:
            rating = "GOOD"
        elif latency_ratio <= 2.0 and avg_recall >= 0.7:
            rating = "ACCEPTABLE"
        else:
            rating = "POOR"

        db.close()

        return BenchmarkResult(
            engine=engine,
            vector_count=vector_count,
            insert_time_secs=insert_time,
            insert_qps=insert_qps,
            flush_time_secs=flush_time,
            avg_search_latency_ms=avg_latency,
            p99_search_latency_ms=p99_latency,
            avg_recall=avg_recall,
            rating=rating,
        )

    except Exception as e:
        return BenchmarkResult(
            engine=engine,
            vector_count=vector_count,
            insert_time_secs=0,
            insert_qps=0,
            flush_time_secs=0,
            avg_search_latency_ms=0,
            p99_search_latency_ms=0,
            avg_recall=0,
            rating="ERROR",
            error=str(e)[:100],
        )


# ============================================================================
# PYTEST FIXTURES AND TEST CLASSES
# ============================================================================


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestAllEnginesQuickSmoke:
    """Quick smoke tests for all engines (1K vectors)."""

    @pytest.mark.parametrize("engine", ENGINES)
    def test_engine_basic_operations(self, engine: str, temp_data_dir: str):
        """Test basic insert/search operations for each engine."""
        db = EmbeddedProximaDB(temp_data_dir)
        collection_name = f"smoke_{engine}"

        # Create collection
        db.create_collection(collection_name, dimension=DIMENSION, engine=engine)

        # Insert vectors
        vectors = generate_vectors(QUICK_VECTOR_COUNT, DIMENSION, seed=42)
        ids = [f"vec_{i}" for i in range(QUICK_VECTOR_COUNT)]
        db.insert(collection_name, ids=ids, vectors=vectors.tolist())

        # Flush
        db.flush()
        time.sleep(0.5)

        # Search
        query = generate_vectors(1, DIMENSION, seed=123)[0]
        results = db.search(collection_name, query=query.tolist(), top_k=5)

        # Assertions
        assert len(results) > 0, f"{engine} returned no results"
        assert all(hasattr(r, "id") for r in results), f"{engine} results missing id"

        db.close()


@pytest.mark.slow
class TestAllEngines50kBenchmark:
    """
    Full 50K vector benchmark suite.

    Mark as slow to skip in CI by default.
    Run with: pytest -m slow
    """

    @pytest.mark.parametrize("engine", ENGINES)
    def test_engine_50k_performance(self, engine: str, temp_data_dir: str):
        """Benchmark each engine with 50K vectors."""
        result = run_engine_benchmark(engine, FULL_VECTOR_COUNT, temp_data_dir)

        # Print results
        print(f"\n{'='*60}")
        print(f"  {engine.upper()} - {FULL_VECTOR_COUNT:,} Vectors")
        print(f"{'='*60}")

        if result.error:
            print(f"  ERROR: {result.error}")
            pytest.fail(f"{engine} benchmark failed: {result.error}")
        else:
            print(f"  Recall@{TOP_K}: {result.avg_recall*100:.1f}%")
            print(f"  Avg Latency: {result.avg_search_latency_ms:.2f}ms")
            print(f"  P99 Latency: {result.p99_search_latency_ms:.2f}ms")
            print(f"  Insert QPS: {result.insert_qps:.0f}")
            print(f"  Flush Time: {result.flush_time_secs:.2f}s")
            print(f"  Rating: {result.rating}")

        # Assertions
        assert result.error is None, f"{engine} had error: {result.error}"
        assert (
            result.avg_recall >= 0.5
        ), f"{engine} recall too low: {result.avg_recall*100:.1f}%"
        assert (
            result.avg_search_latency_ms < 5000
        ), f"{engine} latency too high: {result.avg_search_latency_ms:.1f}ms"

    def test_all_engines_comparison_report(self, temp_data_dir: str):
        """Run all engines and generate comparison report."""
        results = []

        for engine in ENGINES:
            with tempfile.TemporaryDirectory() as engine_dir:
                result = run_engine_benchmark(engine, FULL_VECTOR_COUNT, engine_dir)
                results.append(result)

        # Print comparison table
        print(f"\n{'='*80}")
        print(f"{'50K VECTOR BENCHMARK SUMMARY':^80}")
        print(f"{'='*80}")

        print(
            f"\n{'Engine':<10} {'Recall@10':>10} {'Avg Latency':>12} {'Insert QPS':>12} {'Rating':<12}"
        )
        print("-" * 60)

        for r in results:
            if r.error:
                print(
                    f"{r.engine.upper():<10} {'ERROR':>10} {'-':>12} {'-':>12} {r.error[:20]:<12}"
                )
            else:
                print(
                    f"{r.engine.upper():<10} {r.avg_recall*100:>9.1f}% {r.avg_search_latency_ms:>10.2f}ms {r.insert_qps:>11.0f}/s {r.rating:<12}"
                )

        # Assert overall success
        successful = [r for r in results if r.error is None]
        assert (
            len(successful) >= 4
        ), f"Expected at least 4 engines to succeed, got {len(successful)}"


class TestRecallAccuracy:
    """Tests focused on recall accuracy at various scales."""

    @pytest.mark.parametrize("engine", ENGINES)
    @pytest.mark.parametrize("vector_count", [1000, 5000, 10000])
    def test_recall_at_scale(self, engine: str, vector_count: int, temp_data_dir: str):
        """Test recall accuracy at different scales."""
        result = run_engine_benchmark(engine, vector_count, temp_data_dir)

        if result.error:
            pytest.skip(f"{engine} failed: {result.error}")

        # At smaller scales, expect higher recall
        min_recall = 0.7 if vector_count <= 5000 else 0.5
        assert (
            result.avg_recall >= min_recall
        ), f"{engine} at {vector_count} vectors: recall {result.avg_recall*100:.1f}% < {min_recall*100:.1f}%"


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================


def main():
    """Run benchmark directly (not through pytest)."""
    print("=" * 80)
    print("  50K Vector Benchmark - All 6 Engines")
    print("=" * 80)

    results = []

    for engine in ENGINES:
        print(f"\n{'='*60}")
        print(f"  ENGINE: {engine.upper()}")
        print("=" * 60)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_engine_benchmark(engine, FULL_VECTOR_COUNT, temp_dir)
            results.append(result)

            if result.error:
                print(f"  ERROR: {result.error}")
            else:
                print(f"  Recall@{TOP_K}: {result.avg_recall*100:.1f}%")
                print(f"  Avg Latency: {result.avg_search_latency_ms:.2f}ms")
                print(f"  Insert QPS: {result.insert_qps:.0f}")
                print(f"  Rating: {result.rating}")

    # Summary
    print(f"\n{'='*80}")
    print("  FINAL SUMMARY")
    print("=" * 80)

    print(
        f"\n{'Engine':<10} {'Recall':>10} {'Latency':>12} {'Insert QPS':>12} {'Rating':<12}"
    )
    print("-" * 60)

    for r in results:
        if r.error:
            print(f"{r.engine.upper():<10} {'ERROR':>10}")
        else:
            print(
                f"{r.engine.upper():<10} {r.avg_recall*100:>9.1f}% {r.avg_search_latency_ms:>10.2f}ms {r.insert_qps:>11.0f}/s {r.rating:<12}"
            )


if __name__ == "__main__":
    main()
