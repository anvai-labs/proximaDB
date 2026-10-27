"""
Simplified benchmark test for chunking and embedding strategies

Tests different combinations and measures accuracy and performance.
"""

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pytest

from proximadb_sdk import ProximaDBClient
from proximadb_sdk.chunking import (
    ChunkingConfig,
    ChunkingStrategy,
    chunk_and_embed_text,
)
from proximadb_sdk.embedding_providers import get_embedding_provider
from proximadb_sdk.models import CollectionConfig, DistanceMetric, StorageEngine


@dataclass
class BenchmarkResult:
    """Results for a single benchmark configuration"""

    chunking_strategy: str
    embedding_provider: str
    storage_engine: str
    protocol: str
    num_chunks: int
    processing_time: float
    search_accuracies: list[float] = field(default_factory=list)
    sql_accuracies: list[float] = field(default_factory=list)

    @property
    def avg_search_accuracy(self) -> float:
        return np.mean(self.search_accuracies) if self.search_accuracies else 0.0

    @property
    def avg_sql_accuracy(self) -> float:
        return np.mean(self.sql_accuracies) if self.sql_accuracies else 0.0


def find_proximadb_root() -> str:
    """Find the ProximaDB root directory by looking for key files"""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Try to find the root by looking for characteristic files
    for _ in range(10):  # Max 10 levels up
        if (
            os.path.exists(os.path.join(current_dir, "Cargo.toml"))
            and os.path.exists(os.path.join(current_dir, "README.adoc"))
            and os.path.exists(os.path.join(current_dir, "docs"))
        ):
            return current_dir
        current_dir = os.path.dirname(current_dir)

    # If not found, assume we're in tests/integration, go up 4 levels
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )


def load_test_document() -> str:
    """Load test document using relative paths"""
    proximadb_root = find_proximadb_root()

    # Try multiple possible documentation files
    possible_docs = [
        os.path.join(
            proximadb_root, "docs", "appendices", "appendix_testing_development.adoc"
        ),
        os.path.join(proximadb_root, "docs", "user", "user_guide.adoc"),
        os.path.join(proximadb_root, "docs", "developer", "developer_guide.adoc"),
        os.path.join(proximadb_root, "README.adoc"),
    ]

    for doc_path in possible_docs:
        if os.path.exists(doc_path):
            print(f"Loading document from: {os.path.relpath(doc_path, os.getcwd())}")
            with open(doc_path, encoding="utf-8") as f:
                return f.read()

    # If no files found, create a synthetic document
    print("No documentation files found, using synthetic document")
    return """
        # ProximaDB Testing and Development Guide
        
        ## Introduction
        ProximaDB is a high-performance vector database designed for similarity search and machine learning applications.
        It supports multiple storage engines including SST and VIPER, with advanced indexing algorithms like HNSW.
        
        ## Architecture Overview
        The system uses a modular architecture with pluggable components for chunking, embedding, and storage.
        Text chunking strategies include sliding window, sentence-based, paragraph-based, semantic, and recursive approaches.
        Embedding providers support various models including BERT, sentence-transformers, and custom implementations.
        
        ## Search Capabilities
        ProximaDB offers both vector similarity search and SQL query support. The SQL engine supports complex queries
        with metadata filtering, ORDER BY clauses, and vector similarity functions. HNSW indexing provides fast
        approximate nearest neighbor search with configurable precision-recall tradeoffs.
        
        ## Performance Optimization
        The database employs various optimization techniques including query result caching, parallel processing,
        bloom filters for efficient filtering, and compression algorithms for storage efficiency. The VIPER engine
        uses columnar storage for better compression and query performance.
        
        ## Testing Framework
        Comprehensive testing includes unit tests, integration tests, and performance benchmarks. The test suite
        covers all major components including chunking strategies, embedding providers, storage engines, and query APIs.
        
        """ * 10  # Repeat to get reasonable size

    with open(doc_path) as f:
        return f.read()


def calculate_search_accuracy(results: list[dict], query_terms: list[str]) -> float:
    """Calculate how many query terms appear in top results"""
    if not results:
        return 0.0

    found_terms = set()
    for result in results[:5]:  # Check top 5 results
        text = str(result.get("metadata", {})).lower()
        for term in query_terms:
            if term.lower() in text:
                found_terms.add(term)

    return len(found_terms) / len(query_terms) if query_terms else 0.0


class TestChunkingEmbeddingBenchmark:
    """Benchmark different configurations"""

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_benchmark_configurations(self, rest_client):
        """Run benchmark on different configurations (marked as slow - run with -m benchmark)"""

        # Load document
        document = load_test_document()
        print(f"\nDocument size: {len(document)} characters")

        # Use fixture client
        client = rest_client

        # Test queries
        test_queries = [
            {
                "text": "HNSW indexing algorithm performance",
                "terms": ["HNSW", "index", "algorithm", "performance"],
            },
            {
                "text": "vector database architecture design",
                "terms": ["vector", "database", "architecture", "design"],
            },
            {
                "text": "SQL query metadata filtering",
                "terms": ["SQL", "query", "metadata", "filter"],
            },
        ]

        # Configurations to test
        configs = [
            # (chunking_strategy, embedding_provider, storage_engine, protocol)
            (ChunkingStrategy.SLIDING_WINDOW, "simulated", StorageEngine.SST, "rest"),
            (ChunkingStrategy.SLIDING_WINDOW, "simulated", StorageEngine.VIPER, "grpc"),
            (ChunkingStrategy.PARAGRAPH, "simulated", StorageEngine.SST, "grpc"),
            (ChunkingStrategy.PARAGRAPH, "simulated", StorageEngine.VIPER, "rest"),
            (ChunkingStrategy.SEMANTIC, "simulated", StorageEngine.SST, "rest"),
            (ChunkingStrategy.SEMANTIC, "simulated", StorageEngine.VIPER, "grpc"),
        ]

        # Try to add real embedding provider if available
        try:
            provider = get_embedding_provider("sentence-transformer")
            if provider.is_available():
                configs.extend(
                    [
                        (
                            ChunkingStrategy.SEMANTIC,
                            "sentence-transformer",
                            StorageEngine.VIPER,
                            "grpc",
                        ),
                        (
                            ChunkingStrategy.PARAGRAPH,
                            "sentence-transformer",
                            StorageEngine.SST,
                            "rest",
                        ),
                    ]
                )
                print("✓ Adding sentence-transformer tests")
        except:
            print("✗ Sentence-transformer not available")

        results = []

        for chunking_strategy, embedding_name, storage_engine, protocol in configs:
            print(f"\n{'='*60}")
            print(
                f"Testing: {chunking_strategy.value} + {embedding_name} + {storage_engine.value} + {protocol}"
            )

            try:
                result = self._run_single_benchmark(
                    client,
                    document,
                    chunking_strategy,
                    embedding_name,
                    storage_engine,
                    protocol,
                    test_queries,
                )
                results.append(result)

                print(f"✓ Chunks: {result.num_chunks}")
                print(f"✓ Time: {result.processing_time:.2f}s")
                print(f"✓ Search accuracy: {result.avg_search_accuracy:.2%}")
                if result.sql_accuracies:
                    print(f"✓ SQL accuracy: {result.avg_sql_accuracy:.2%}")

            except Exception as e:
                print(f"✗ Error: {e}")

        # Generate summary
        self._print_summary(results)

    def _run_single_benchmark(
        self,
        client: ProximaDBClient,
        document: str,
        chunking_strategy: ChunkingStrategy,
        embedding_name: str,
        storage_engine: StorageEngine,
        protocol: str,
        test_queries: list[dict],
    ) -> BenchmarkResult:
        """Run benchmark for a single configuration"""

        start_time = time.time()

        # Create unique collection name
        collection_name = f"bench_{int(time.time())}_{chunking_strategy.value[:4]}_{embedding_name[:4]}"

        # Clean up any existing collection
        try:
            client.delete_collection(collection_name)
        except:
            pass

        # Create collection
        client._force_protocol = protocol
        collection = client.create_collection(
            name=collection_name,
            config=CollectionConfig(
                name=collection_name,
                dimension=384,  # Standard dimension
                distance_metric=DistanceMetric.COSINE,
                storage_engine=storage_engine,
            ),
        )

        # Get embedding provider
        embedding_provider = get_embedding_provider(embedding_name)

        # Chunk and embed
        records = chunk_and_embed_text(
            text=document,
            source_id="test_doc",
            embedding_provider=embedding_provider,
            chunking_config=ChunkingConfig(
                strategy=chunking_strategy, chunk_size=500, chunk_overlap=50
            ),
            metadata={"doc_type": "testing_guide"},
        )

        num_chunks = len(records)

        # Insert vectors
        client.insert_vectors(collection_name, records)

        # Wait for indexing
        time.sleep(0.5)

        # Test searches
        search_accuracies = []
        sql_accuracies = []

        for query in test_queries:
            # Generate query embedding
            query_embedding = embedding_provider.embed_text(query["text"])

            # Vector search
            search_results = client.search(
                collection_id=collection_name,
                query_vector=(
                    query_embedding.tolist()
                    if hasattr(query_embedding, "tolist")
                    else query_embedding
                ),
                top_k=5,
            )

            accuracy = calculate_search_accuracy(
                search_results.get("results", []), query["terms"]
            )
            search_accuracies.append(accuracy)

            # SQL search (REST only)
            if protocol == "rest":
                try:
                    vector_str = json.dumps(
                        query_embedding.tolist()
                        if hasattr(query_embedding, "tolist")
                        else query_embedding
                    )
                    sql = f"""
                    SELECT id, metadata
                    FROM {collection_name}
                    ORDER BY VECTOR_SIMILARITY(vector, {vector_str}, 'cosine')
                    LIMIT 5
                    """

                    sql_results = client.execute_sql(sql)
                    sql_accuracy = calculate_search_accuracy(
                        sql_results.get("results", []), query["terms"]
                    )
                    sql_accuracies.append(sql_accuracy)
                except Exception as e:
                    print(f"  SQL error: {e}")

        # Clean up
        client.delete_collection(collection_name)

        processing_time = time.time() - start_time

        return BenchmarkResult(
            chunking_strategy=chunking_strategy.value,
            embedding_provider=embedding_name,
            storage_engine=storage_engine.value,
            protocol=protocol,
            num_chunks=num_chunks,
            processing_time=processing_time,
            search_accuracies=search_accuracies,
            sql_accuracies=sql_accuracies,
        )

    def _print_summary(self, results: list[BenchmarkResult]):
        """Print summary of benchmark results"""
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}\n")

        if not results:
            print("No successful results!")
            return

        # Sort by search accuracy
        results.sort(key=lambda x: x.avg_search_accuracy, reverse=True)

        print("TOP CONFIGURATIONS BY SEARCH ACCURACY:")
        for i, r in enumerate(results[:5], 1):
            print(
                f"{i}. {r.chunking_strategy} + {r.embedding_provider} + {r.storage_engine} + {r.protocol}"
            )
            print(
                f"   Search: {r.avg_search_accuracy:.2%}, Time: {r.processing_time:.2f}s, Chunks: {r.num_chunks}"
            )

        # Compare by chunking strategy
        print("\nBY CHUNKING STRATEGY:")
        strategy_stats = {}
        for r in results:
            if r.chunking_strategy not in strategy_stats:
                strategy_stats[r.chunking_strategy] = []
            strategy_stats[r.chunking_strategy].append(r.avg_search_accuracy)

        for strategy, accuracies in sorted(strategy_stats.items()):
            print(f"  {strategy}: {np.mean(accuracies):.2%} avg accuracy")

        # Compare by storage engine
        print("\nBY STORAGE ENGINE:")
        engine_stats = {}
        for r in results:
            if r.storage_engine not in engine_stats:
                engine_stats[r.storage_engine] = []
            engine_stats[r.storage_engine].append(r.avg_search_accuracy)

        for engine, accuracies in sorted(engine_stats.items()):
            print(f"  {engine}: {np.mean(accuracies):.2%} avg accuracy")

        # Compare protocols
        print("\nBY PROTOCOL:")
        protocol_stats = {}
        for r in results:
            if r.protocol not in protocol_stats:
                protocol_stats[r.protocol] = {"search": [], "time": []}
            protocol_stats[r.protocol]["search"].append(r.avg_search_accuracy)
            protocol_stats[r.protocol]["time"].append(r.processing_time)

        for protocol, stats in protocol_stats.items():
            print(
                f"  {protocol}: {np.mean(stats['search']):.2%} accuracy, {np.mean(stats['time']):.2f}s avg time"
            )

        # SQL vs Vector search
        print("\nSQL vs VECTOR SEARCH (REST only):")
        rest_results = [r for r in results if r.protocol == "rest" and r.sql_accuracies]
        if rest_results:
            for r in rest_results:
                print(
                    f"  {r.chunking_strategy} + {r.embedding_provider}: Vector={r.avg_search_accuracy:.2%}, SQL={r.avg_sql_accuracy:.2%}"
                )


if __name__ == "__main__":
    test = TestChunkingEmbeddingBenchmark()
    test.test_benchmark_configurations()
