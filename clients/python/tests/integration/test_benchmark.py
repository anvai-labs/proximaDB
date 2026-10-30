"""
Comprehensive benchmark test for ProximaDB Python SDK

This test evaluates different combinations of:
- Chunking strategies (sliding_window, sentence, paragraph, semantic, recursive)
- Embedding providers (simulated, sentence-transformer, fastembed, instructor)
- Storage engines (SST, VIPER)
- Protocols (REST, gRPC)
- Search methods (vector search, SQL API)

Uses real documentation content (~100KB) for realistic testing.
"""

import logging
import os
import time
from dataclasses import dataclass

import numpy as np
import pytest

from proximadb_sdk import ProximaDBClient
from proximadb_sdk.chunking import ChunkingConfig, ChunkingStrategy, TextChunker
from proximadb_sdk.embedding_providers import get_provider
from proximadb_sdk.embedding_providers.core import (
    BaseEmbeddingProvider as EmbeddingProvider,
)
from proximadb_sdk.models import (
    CollectionConfig,
    DistanceMetric,
    StorageEngine,
    VectorRecord,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results for a single benchmark run"""

    chunking_strategy: str
    embedding_provider: str
    storage_engine: str
    protocol: str
    num_chunks: int
    chunk_time: float
    embed_time: float
    insert_time: float
    search_time: float
    sql_time: float
    search_accuracy: float
    sql_accuracy: float
    total_time: float
    error: str = None


class TestComprehensiveBenchmark:
    """Comprehensive benchmark test suite"""

    @staticmethod
    def find_proximadb_root() -> str:
        """Find the ProximaDB root directory"""
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Look for root by checking for characteristic files
        for _ in range(10):  # Max 10 levels up
            if all(
                os.path.exists(os.path.join(current_dir, f))
                for f in ["Cargo.toml", "README.adoc", "docs"]
            ):
                return current_dir
            current_dir = os.path.dirname(current_dir)

        # Fallback: assume standard structure
        return os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )

    @pytest.fixture(scope="class")
    def documentation_text(self):
        """Load documentation text from the largest single file"""
        root_dir = self.find_proximadb_root()

        # Try multiple documentation files in order of preference
        doc_candidates = [
            os.path.join(
                root_dir, "docs", "appendices", "appendix_testing_development.adoc"
            ),
            os.path.join(root_dir, "docs", "user", "user_guide.adoc"),
            os.path.join(root_dir, "docs", "developer", "developer_guide.adoc"),
            os.path.join(root_dir, "README.adoc"),
        ]

        for doc_file in doc_candidates:
            if os.path.exists(doc_file):
                with open(doc_file, encoding="utf-8") as f:
                    full_text = f.read()
                logger.info(
                    f"Loaded {len(full_text)} characters from {os.path.relpath(doc_file, os.getcwd())}"
                )
                return full_text

        # If no files found, use synthetic content
        logger.warning("No documentation files found, using synthetic content")
        return self.generate_synthetic_document()

    @staticmethod
    def generate_synthetic_document() -> str:
        """Generate synthetic document for testing when no docs available"""
        return """
        # ProximaDB Comprehensive Documentation
        
        ## Architecture Overview
        ProximaDB is a high-performance vector database with advanced features for similarity search.
        The architecture includes multiple storage engines (SST and VIPER), indexing algorithms (HNSW, IVF),
        and supports both REST and gRPC protocols.
        
        ## Storage Engines
        - SST Engine: Row-based storage optimized for write performance
        - VIPER Engine: Columnar storage with compression for analytics workloads
        
        ## Indexing Algorithms
        - HNSW: Hierarchical Navigable Small World graphs for fast approximate search
        - IVF: Inverted File Index for partition-based search
        - Flat: Brute-force search for exact results
        
        ## Query Capabilities
        ProximaDB supports SQL queries with vector similarity functions:
        SELECT * FROM collection WHERE metadata->>'category' = 'tech'
        ORDER BY VECTOR_SIMILARITY(vector, query_vector, 'cosine') LIMIT 10
        
        ## Text Processing
        Multiple chunking strategies are available:
        - Sliding Window: Fixed-size overlapping chunks
        - Sentence-based: Chunks aligned with sentence boundaries
        - Paragraph-based: Chunks following paragraph structure
        - Semantic: Topic-aware chunking using coherence analysis
        - Recursive: Hierarchical chunking with fallback strategies
        
        ## Embedding Providers
        The SDK supports various embedding models:
        - Sentence Transformers: BERT-based models like all-MiniLM-L6-v2
        - FastEmbed: ONNX-optimized models for speed
        - Instructor: Task-specific embeddings
        - OpenAI Compatible: Any API following OpenAI format
        
        ## Performance Optimization
        - Query result caching
        - Parallel processing
        - Bloom filters for metadata
        - Compression algorithms
        - Connection pooling
        """ * 5  # Repeat to get reasonable size

    @pytest.fixture(scope="class")
    def search_queries(self):
        """Define test search queries with expected terms"""
        return [
            {
                "query": "vector database architecture",
                "expected_terms": [
                    "vector",
                    "database",
                    "architecture",
                    "storage",
                    "engine",
                ],
                "embedding_text": "How does the vector database architecture work?",
            },
            {
                "query": "HNSW indexing algorithm",
                "expected_terms": ["HNSW", "index", "algorithm", "search", "graph"],
                "embedding_text": "Explain the HNSW indexing algorithm",
            },
            {
                "query": "SQL query support",
                "expected_terms": ["SQL", "query", "SELECT", "WHERE", "ORDER BY"],
                "embedding_text": "What SQL query features are supported?",
            },
            {
                "query": "chunking strategies",
                "expected_terms": [
                    "chunk",
                    "strategy",
                    "semantic",
                    "sliding",
                    "window",
                ],
                "embedding_text": "Different text chunking strategies available",
            },
            {
                "query": "embedding providers",
                "expected_terms": ["embedding", "provider", "BERT", "model", "vector"],
                "embedding_text": "Available embedding provider options",
            },
        ]

    @pytest.fixture(scope="class")
    def proximadb_client(self):
        """Get ProximaDB client instance"""
        # Ensure ProximaDB server is running
        url = os.getenv("PROXIMADB_URL", "http://localhost:5678")

        return ProximaDBClient(url=url)

    def get_available_embedding_providers(self) -> list[tuple[str, EmbeddingProvider]]:
        """Get list of available free embedding providers"""
        providers = []

        # Always include simulated
        providers.append(("simulated", get_provider("simulated", dimension=384)))

        # Try other free providers
        for provider_name in ["sentence-transformer", "fastembed", "instructor"]:
            try:
                provider = get_provider(provider_name)
                if provider.is_available():
                    providers.append((provider_name, provider))
                    logger.info(f"✓ {provider_name} available")
                else:
                    logger.info(f"✗ {provider_name} not installed")
            except Exception as e:
                logger.info(f"✗ {provider_name} error: {e}")

        return providers

    def calculate_search_accuracy(
        self, results: list[dict], expected_terms: list[str], top_k: int = 10
    ) -> float:
        """Calculate search accuracy based on expected terms in results"""
        if not results:
            return 0.0

        # Check top results for expected terms
        found_terms = set()
        for i, result in enumerate(results[:top_k]):
            # Handle both dict and SearchResult objects
            if hasattr(result, "model_dump"):  # Pydantic model
                metadata = result.metadata if result.metadata else {}
            elif isinstance(result, dict):  # Regular dict
                metadata = result.get("metadata", {})
            else:
                metadata = {}

            text = metadata.get("text_preview", "").lower()
            for term in expected_terms:
                if term.lower() in text:
                    found_terms.add(term)

        accuracy = len(found_terms) / len(expected_terms) if expected_terms else 0.0
        return accuracy

    def benchmark_combination(
        self,
        client: ProximaDBClient,
        text: str,
        chunking_strategy: ChunkingStrategy,
        embedding_provider: EmbeddingProvider,
        storage_engine: StorageEngine,
        protocol: str,
        search_queries: list[dict],
    ) -> BenchmarkResult:
        """Run benchmark for a specific combination"""

        # Create unique collection name
        collection_name = f"bench_{chunking_strategy.value}_{embedding_provider.model_name.replace('/', '_')}_{storage_engine.value}_{protocol}"[
            :50
        ]

        result = BenchmarkResult(
            chunking_strategy=chunking_strategy.value,
            embedding_provider=embedding_provider.model_name,
            storage_engine=storage_engine.value,
            protocol=protocol,
            num_chunks=0,
            chunk_time=0,
            embed_time=0,
            insert_time=0,
            search_time=0,
            sql_time=0,
            search_accuracy=0,
            sql_accuracy=0,
            total_time=0,
        )

        start_total = time.time()

        try:
            # Clean up any existing collection
            try:
                client.delete_collection(collection_name)
            except:
                pass

            # Configure client protocol
            client._force_protocol = protocol

            # 1. Create collection
            collection = client.create_collection(
                name=collection_name,
                config=CollectionConfig(
                    name=collection_name,
                    dimension=embedding_provider.dimension,
                    distance_metric=DistanceMetric.COSINE,
                    storage_engine=storage_engine,
                    quantization_config=None,
                ),
            )

            # 2. Chunk text
            start_chunk = time.time()
            chunker = TextChunker(
                ChunkingConfig(
                    strategy=chunking_strategy, chunk_size=1000, chunk_overlap=100
                )
            )
            chunks = chunker.chunk_text(text, "benchmark_doc")
            result.num_chunks = len(chunks)
            result.chunk_time = time.time() - start_chunk

            # 3. Generate embeddings
            start_embed = time.time()
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = embedding_provider.embed_texts(chunk_texts)
            result.embed_time = time.time() - start_embed

            # 4. Insert vectors
            start_insert = time.time()
            records = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                record = VectorRecord(
                    id=f"{collection_name}_chunk_{i}",
                    vector=(
                        embedding.tolist()
                        if hasattr(embedding, "tolist")
                        else embedding
                    ),
                    metadata={
                        "chunk_index": i,
                        "text_preview": chunk.text[:200],
                        "chunking_strategy": chunking_strategy.value,
                        "source_id": "benchmark_doc",
                    },
                )
                records.append(record)

            # Batch insert
            client.insert_vectors(collection_name, records)
            result.insert_time = time.time() - start_insert

            # Wait for indexing
            time.sleep(1)

            # 5. Test vector search
            search_accuracies = []
            search_times = []

            for query_info in search_queries:
                # Generate query embedding
                query_embedding = embedding_provider.embed_text(
                    query_info["embedding_text"]
                )

                start_search = time.time()
                search_results = client.search(
                    collection_id=collection_name,
                    vector=(
                        query_embedding.tolist()
                        if hasattr(query_embedding, "tolist")
                        else query_embedding
                    ),  # Changed from query_vector to vector
                    top_k=10,
                )
                search_time = time.time() - start_search
                search_times.append(search_time)

                # Calculate accuracy
                # search_results is already a list, not a dict with 'results' key
                results_list = (
                    search_results
                    if isinstance(search_results, list)
                    else search_results.get("results", [])
                )
                accuracy = self.calculate_search_accuracy(
                    results_list, query_info["expected_terms"]
                )
                search_accuracies.append(accuracy)

            result.search_time = np.mean(search_times)
            result.search_accuracy = np.mean(search_accuracies)

            # 6. Test SQL search (REST only)
            if protocol == "rest":
                sql_accuracies = []
                sql_times = []

                for query_info in search_queries:
                    # Build SQL query
                    query_embedding = embedding_provider.embed_text(
                        query_info["embedding_text"]
                    )
                    vector_str = str(
                        query_embedding.tolist()
                        if hasattr(query_embedding, "tolist")
                        else query_embedding
                    )

                    sql = f"""
                    SELECT id, metadata
                    FROM {collection_name}
                    ORDER BY VECTOR_SIMILARITY(vector, {vector_str}, 'cosine')
                    LIMIT 10
                    """

                    start_sql = time.time()
                    try:
                        sql_results = client.execute_sql(sql)
                        sql_time = time.time() - start_sql
                        sql_times.append(sql_time)

                        # Calculate accuracy
                        if sql_results and "results" in sql_results:
                            accuracy = self.calculate_search_accuracy(
                                sql_results["results"], query_info["expected_terms"]
                            )
                            sql_accuracies.append(accuracy)
                        else:
                            sql_accuracies.append(0.0)
                    except Exception as e:
                        logger.error(f"SQL query failed: {e}")
                        sql_accuracies.append(0.0)
                        sql_times.append(0.0)

                result.sql_time = np.mean(sql_times) if sql_times else 0.0
                result.sql_accuracy = np.mean(sql_accuracies) if sql_accuracies else 0.0

            # Clean up
            client.delete_collection(collection_name)

        except Exception as e:
            result.error = str(e)
            logger.error(f"Benchmark failed: {e}")

        result.total_time = time.time() - start_total
        return result

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_comprehensive_benchmark(
        self, documentation_text, search_queries, proximadb_client
    ):
        """Run comprehensive benchmark test (marked as slow - run with -m benchmark)"""

        # Get available providers
        embedding_providers = self.get_available_embedding_providers()
        if not embedding_providers:
            pytest.skip("No embedding providers available")

        # Test configurations
        chunking_strategies = [
            ChunkingStrategy.SLIDING_WINDOW,
            ChunkingStrategy.SENTENCE,
            ChunkingStrategy.PARAGRAPH,
            ChunkingStrategy.SEMANTIC,
            ChunkingStrategy.RECURSIVE,
        ]

        storage_engines = [StorageEngine.SST, StorageEngine.VIPER]
        protocols = ["rest", "grpc"]

        # Run benchmarks
        results = []
        total_combinations = (
            len(chunking_strategies)
            * len(embedding_providers)
            * len(storage_engines)
            * len(protocols)
        )

        print(f"\nRunning {total_combinations} benchmark combinations...")
        print("=" * 80)

        for chunking_strategy in chunking_strategies:
            for provider_name, embedding_provider in embedding_providers:
                for storage_engine in storage_engines:
                    for protocol in protocols:
                        print(
                            f"\nTesting: {chunking_strategy.value} + {provider_name} + {storage_engine.value} + {protocol}"
                        )

                        result = self.benchmark_combination(
                            proximadb_client,
                            documentation_text,
                            chunking_strategy,
                            embedding_provider,
                            storage_engine,
                            protocol,
                            search_queries,
                        )

                        results.append(result)

                        if result.error:
                            print(f"  ❌ Error: {result.error}")
                        else:
                            print(f"  ✓ Chunks: {result.num_chunks}")
                            print(f"  ✓ Chunk time: {result.chunk_time:.2f}s")
                            print(f"  ✓ Embed time: {result.embed_time:.2f}s")
                            print(f"  ✓ Insert time: {result.insert_time:.2f}s")
                            print(f"  ✓ Search accuracy: {result.search_accuracy:.2%}")
                            if protocol == "rest":
                                print(f"  ✓ SQL accuracy: {result.sql_accuracy:.2%}")

        # Generate summary report
        self.generate_summary_report(results)

    def generate_summary_report(self, results: list[BenchmarkResult]):
        """Generate comprehensive summary report"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE BENCHMARK SUMMARY")
        print("=" * 80)

        # Filter successful results
        successful_results = [r for r in results if not r.error]

        if not successful_results:
            print("No successful benchmark runs!")
            return

        # 1. Best performers by accuracy
        print("\n📊 TOP 5 BY SEARCH ACCURACY:")
        sorted_by_accuracy = sorted(
            successful_results, key=lambda x: x.search_accuracy, reverse=True
        )[:5]
        for i, r in enumerate(sorted_by_accuracy, 1):
            print(
                f"{i}. {r.chunking_strategy} + {r.embedding_provider} + {r.storage_engine} + {r.protocol}: {r.search_accuracy:.2%}"
            )

        # 2. Best performers by speed
        print("\n⚡ TOP 5 BY SPEED (Total Time):")
        sorted_by_speed = sorted(successful_results, key=lambda x: x.total_time)[:5]
        for i, r in enumerate(sorted_by_speed, 1):
            print(
                f"{i}. {r.chunking_strategy} + {r.embedding_provider} + {r.storage_engine} + {r.protocol}: {r.total_time:.2f}s"
            )

        # 3. Chunking strategy comparison
        print("\n📈 CHUNKING STRATEGY AVERAGE ACCURACY:")
        strategy_stats = {}
        for r in successful_results:
            if r.chunking_strategy not in strategy_stats:
                strategy_stats[r.chunking_strategy] = []
            strategy_stats[r.chunking_strategy].append(r.search_accuracy)

        for strategy, accuracies in sorted(strategy_stats.items()):
            avg_accuracy = np.mean(accuracies)
            print(f"  {strategy}: {avg_accuracy:.2%} (n={len(accuracies)})")

        # 4. Embedding provider comparison
        print("\n🧠 EMBEDDING PROVIDER AVERAGE ACCURACY:")
        provider_stats = {}
        for r in successful_results:
            provider_name = r.embedding_provider.split("/")[-1]  # Simplify name
            if provider_name not in provider_stats:
                provider_stats[provider_name] = []
            provider_stats[provider_name].append(r.search_accuracy)

        for provider, accuracies in sorted(provider_stats.items()):
            avg_accuracy = np.mean(accuracies)
            print(f"  {provider}: {avg_accuracy:.2%} (n={len(accuracies)})")

        # 5. Storage engine comparison
        print("\n💾 STORAGE ENGINE PERFORMANCE:")
        engine_stats = {}
        for r in successful_results:
            if r.storage_engine not in engine_stats:
                engine_stats[r.storage_engine] = {
                    "accuracy": [],
                    "insert_time": [],
                    "search_time": [],
                }
            engine_stats[r.storage_engine]["accuracy"].append(r.search_accuracy)
            engine_stats[r.storage_engine]["insert_time"].append(r.insert_time)
            engine_stats[r.storage_engine]["search_time"].append(r.search_time)

        for engine, stats in engine_stats.items():
            print(f"\n  {engine}:")
            print(f"    Avg Accuracy: {np.mean(stats['accuracy']):.2%}")
            print(f"    Avg Insert Time: {np.mean(stats['insert_time']):.2f}s")
            print(f"    Avg Search Time: {np.mean(stats['search_time']):.3f}s")

        # 6. Protocol comparison
        print("\n🌐 PROTOCOL PERFORMANCE:")
        protocol_stats = {}
        for r in successful_results:
            if r.protocol not in protocol_stats:
                protocol_stats[r.protocol] = {"accuracy": [], "total_time": []}
            protocol_stats[r.protocol]["accuracy"].append(r.search_accuracy)
            protocol_stats[r.protocol]["total_time"].append(r.total_time)

        for protocol, stats in protocol_stats.items():
            print(f"\n  {protocol.upper()}:")
            print(f"    Avg Accuracy: {np.mean(stats['accuracy']):.2%}")
            print(f"    Avg Total Time: {np.mean(stats['total_time']):.2f}s")

        # 7. SQL vs Vector Search (REST only)
        print("\n🔍 SQL vs VECTOR SEARCH (REST only):")
        rest_results = [
            r for r in successful_results if r.protocol == "rest" and r.sql_accuracy > 0
        ]
        if rest_results:
            vector_accuracies = [r.search_accuracy for r in rest_results]
            sql_accuracies = [r.sql_accuracy for r in rest_results]
            print(f"  Vector Search Avg: {np.mean(vector_accuracies):.2%}")
            print(f"  SQL Search Avg: {np.mean(sql_accuracies):.2%}")
            print(
                f"  SQL/Vector Ratio: {np.mean(sql_accuracies) / np.mean(vector_accuracies):.2f}x"
            )

        # 8. Optimal combinations
        print("\n🏆 RECOMMENDED CONFIGURATIONS:")
        print("\n  For Accuracy:")
        best_accuracy = sorted_by_accuracy[0]
        print(
            f"    {best_accuracy.chunking_strategy} + {best_accuracy.embedding_provider} + {best_accuracy.storage_engine} + {best_accuracy.protocol}"
        )

        print("\n  For Speed:")
        best_speed = sorted_by_speed[0]
        print(
            f"    {best_speed.chunking_strategy} + {best_speed.embedding_provider} + {best_speed.storage_engine} + {best_speed.protocol}"
        )

        # 9. Overall statistics
        print("\n📊 OVERALL STATISTICS:")
        print(f"  Total runs: {len(results)}")
        print(f"  Successful: {len(successful_results)}")
        print(f"  Failed: {len(results) - len(successful_results)}")
        print(f"  Success rate: {len(successful_results) / len(results) * 100:.1f}%")
        print(
            f"  Avg chunks per document: {np.mean([r.num_chunks for r in successful_results]):.0f}"
        )
        print(f"  Total benchmark time: {sum(r.total_time for r in results):.1f}s")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run the benchmark
    pytest.main([__file__, "-v", "-s"])
