#!/usr/bin/env python3
"""
Embedded Vector Database Benchmark for Victor CLI Distribution
Compares: ChromaDB vs LanceDB vs FAISS vs Qdrant

This benchmark evaluates truly embedded databases for use as embedded vector
storage in Victor CLI. All databases run in-process without external servers.

Note: ProximaDB PyO3 bindings are implemented (src/embedded/mod.rs, python.rs)
but the Python wheel hasn't been built yet. When ready, it will provide
native in-process access to ProximaDB storage engines.

Metrics:
- Indexing speed (docs/sec)
- Search latency (ms)
- Semantic accuracy (relevance of results)
- Batch insert performance

Embedding: Sentence-Transformers (BAAI/bge-small-en-v1.5)
"""

import gc
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Vector databases
import chromadb
import httpx
import lancedb
import numpy as np

# Embedding models
from proximadb_sdk import CollectionConfig, connect_grpc, connect_rest
from proximadb_sdk.embedded import SentenceTransformerModel

# FAISS
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Qdrant
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# gRPC support
try:
    import grpc
    from google.protobuf import struct_pb2

    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False


# =============================================================================
# Code Corpus Collection
# =============================================================================


def collect_code_files(base_path: str, max_files: int = 50) -> list[dict[str, Any]]:
    """Collect code files from the repository."""
    documents = []
    base = Path(base_path)

    extensions = {".py", ".rs", ".ts", ".js", ".go", ".java"}

    for ext in extensions:
        for file_path in base.rglob(f"*{ext}"):
            if len(documents) >= max_files:
                break

            # Skip test files and generated code
            rel_path = str(file_path.relative_to(base))
            if (
                "test" in rel_path.lower()
                or "target" in rel_path
                or "__pycache__" in rel_path
            ):
                continue

            try:
                content = file_path.read_text(errors="ignore")
                if len(content) < 100 or len(content) > 10000:
                    continue

                documents.append(
                    {
                        "id": rel_path.replace("/", "_").replace(".", "_"),
                        "content": content[:2000],
                        "language": ext[1:],
                        "path": rel_path,
                    }
                )
            except Exception:
                pass

        if len(documents) >= max_files:
            break

    return documents


# =============================================================================
# Benchmark Result Class
# =============================================================================


class BenchmarkResult:
    def __init__(self, name: str):
        self.name = name
        self.index_times: list[float] = []
        self.search_times: list[float] = []
        self.search_results: list[list[str]] = []
        self.errors: list[str] = []

    def summary(self) -> dict[str, Any]:
        if not self.index_times:
            return {
                "name": self.name,
                "index_docs_per_sec": 0,
                "search_avg_ms": 0,
                "search_p95_ms": 0,
                "errors": len(self.errors),
            }

        total_index_time = sum(self.index_times)
        return {
            "name": self.name,
            "index_docs_per_sec": (
                len(self.index_times) / total_index_time if total_index_time > 0 else 0
            ),
            "search_avg_ms": (
                statistics.mean(self.search_times) * 1000 if self.search_times else 0
            ),
            "search_p50_ms": (
                statistics.median(self.search_times) * 1000 if self.search_times else 0
            ),
            "search_p95_ms": (
                sorted(self.search_times)[int(len(self.search_times) * 0.95)] * 1000
                if len(self.search_times) > 1
                else 0
            ),
            "errors": len(self.errors),
        }


# =============================================================================
# ChromaDB Benchmark
# =============================================================================


class ChromaDBBenchmark:
    def __init__(self, embedding_model, data_dir: str):
        self.model = embedding_model
        self.data_dir = data_dir
        self.client = None
        self.collection = None

    def setup(self):
        self.client = chromadb.PersistentClient(path=self.data_dir)
        try:
            self.client.delete_collection("benchmark")
        except:
            pass
        self.collection = self.client.create_collection(
            name="benchmark", metadata={"hnsw:space": "cosine"}
        )

    def index(self, documents: list[dict], result: BenchmarkResult):
        for doc in documents:
            start = time.perf_counter()
            embedding = self.model.embed(doc["content"])
            self.collection.add(
                ids=[doc["id"]],
                embeddings=[embedding],
                documents=[doc["content"][:500]],
                metadatas=[{"language": doc["language"], "path": doc["path"]}],
            )
            result.index_times.append(time.perf_counter() - start)

    def search(self, query: str, top_k: int = 5) -> tuple[list[str], float]:
        start = time.perf_counter()
        embedding = self.model.embed(query)
        results = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        elapsed = time.perf_counter() - start
        return results["ids"][0] if results["ids"] else [], elapsed

    def cleanup(self):
        self.client = None
        self.collection = None


# =============================================================================
# LanceDB Benchmark
# =============================================================================


class LanceDBBenchmark:
    def __init__(self, embedding_model, data_dir: str):
        self.model = embedding_model
        self.data_dir = data_dir
        self.db = None
        self.table = None

    def setup(self):
        self.db = lancedb.connect(self.data_dir)
        if "benchmark" in self.db.table_names():
            self.db.drop_table("benchmark")
        self.table = None

    def index(self, documents: list[dict], result: BenchmarkResult):
        data = []
        for doc in documents:
            start = time.perf_counter()
            embedding = self.model.embed(doc["content"])
            data.append(
                {
                    "id": doc["id"],
                    "vector": embedding,
                    "content": doc["content"][:500],
                    "language": doc["language"],
                    "path": doc["path"],
                }
            )
            result.index_times.append(time.perf_counter() - start)

        if self.table is None:
            self.table = self.db.create_table("benchmark", data)
        else:
            self.table.add(data)

    def search(self, query: str, top_k: int = 5) -> tuple[list[str], float]:
        start = time.perf_counter()
        embedding = self.model.embed(query)
        results = self.table.search(embedding).limit(top_k).to_list()
        elapsed = time.perf_counter() - start
        return [r["id"] for r in results], elapsed

    def cleanup(self):
        self.db = None
        self.table = None


# =============================================================================
# FAISS Benchmark
# =============================================================================


class FAISSBenchmark:
    """FAISS (Facebook AI Similarity Search) - CPU version."""

    def __init__(self, embedding_model, data_dir: str):
        self.model = embedding_model
        self.data_dir = data_dir
        self.dimension = embedding_model.get_dimension()
        self.faiss_index = None
        self.id_map = {}  # Map index position to doc id
        self.doc_map = {}  # Map doc id to content

    def setup(self):
        if FAISS_AVAILABLE:
            # Use IndexFlatIP for cosine similarity (normalize vectors first)
            self.faiss_index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ImportError("FAISS not available")

    def index(self, documents: list[dict], result: BenchmarkResult):
        vectors = []
        for i, doc in enumerate(documents):
            start = time.perf_counter()
            embedding = self.model.embed(doc["content"])
            # Normalize for cosine similarity
            embedding = np.array(embedding, dtype=np.float32)
            faiss.normalize_L2(embedding.reshape(1, -1))
            vectors.append(embedding)
            self.id_map[i] = doc["id"]
            self.doc_map[doc["id"]] = doc
            result.index_times.append(time.perf_counter() - start)

        # Batch add to index
        vectors_array = np.vstack(vectors).astype(np.float32)
        self.faiss_index.add(vectors_array)

    def search(self, query: str, top_k: int = 5) -> tuple[list[str], float]:
        start = time.perf_counter()
        embedding = self.model.embed(query)
        query_vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vec)

        distances, indices = self.faiss_index.search(query_vec, top_k)
        elapsed = time.perf_counter() - start

        ids = [self.id_map.get(int(idx), "") for idx in indices[0] if idx >= 0]
        return ids, elapsed

    def cleanup(self):
        self.faiss_index = None
        self.id_map = {}
        self.doc_map = {}


# =============================================================================
# Qdrant Benchmark (Embedded Mode)
# =============================================================================


class QdrantBenchmark:
    """Qdrant embedded mode - in-memory or on-disk."""

    def __init__(self, embedding_model, data_dir: str):
        self.model = embedding_model
        self.data_dir = data_dir
        self.dimension = embedding_model.get_dimension()
        self.client = None
        self.collection_name = "benchmark"

    def setup(self):
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant not available")

        # Use in-memory mode for benchmark
        self.client = QdrantClient(":memory:")

        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
        )

    def index(self, documents: list[dict], result: BenchmarkResult):
        points = []
        for i, doc in enumerate(documents):
            start = time.perf_counter()
            embedding = self.model.embed(doc["content"])
            points.append(
                PointStruct(
                    id=i,
                    vector=embedding,
                    payload={
                        "doc_id": doc["id"],
                        "content": doc["content"][:500],
                        "language": doc["language"],
                        "path": doc["path"],
                    },
                )
            )
            result.index_times.append(time.perf_counter() - start)

        # Batch upsert
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k: int = 5) -> tuple[list[str], float]:
        start = time.perf_counter()
        embedding = self.model.embed(query)

        # Use query method (newer Qdrant API)
        results = self.client.query_points(
            collection_name=self.collection_name, query=embedding, limit=top_k
        )

        elapsed = time.perf_counter() - start
        ids = [r.payload.get("doc_id", "") for r in results.points]
        return ids, elapsed

    def cleanup(self):
        if self.client:
            try:
                self.client.delete_collection(self.collection_name)
            except:
                pass
        self.client = None


# =============================================================================
# ProximaDB Embedded Benchmark (uses gRPC client to local server)
# =============================================================================


class ProximaDBEmbeddedBenchmark:
    """ProximaDB benchmark using gRPC protocol for best performance.

    Uses the ProximaDB gRPC sync client which provides:
    - Connection pooling
    - Binary protocol (faster than REST)
    - Batch operations
    """

    def __init__(self, embedding_model, grpc_url: str = "localhost:5679"):
        self.model = embedding_model
        self.grpc_url = grpc_url
        self.collection_name = f"benchmark_embedded_{int(time.time()*1000)}"
        self.dimension = embedding_model.get_dimension()
        self.client = None

    def setup(self):
        from proximadb_sdk.protocols.grpc_sync import ProximaDBSyncGrpcClient

        self.client = ProximaDBSyncGrpcClient(self.grpc_url)

        # Create collection with cosine distance
        self.client.create_collection(
            self.collection_name, self.dimension, distance_metric=1  # COSINE
        )

    def index(self, documents: list[dict], result: BenchmarkResult):
        """Index documents one by one (like other benchmarks)."""
        for doc in documents:
            start = time.perf_counter()
            embedding = self.model.embed(doc["content"])

            records = [
                {
                    "id": doc["id"],
                    "vector": embedding,
                    "props": {
                        "language": doc["language"],
                        "path": doc["path"],
                        "content": doc["content"][:500],
                    },
                }
            ]
            self.client.insert_records(self.collection_name, records)
            result.index_times.append(time.perf_counter() - start)

    def search(self, query: str, top_k: int = 5) -> tuple[list[str], float]:
        start = time.perf_counter()
        embedding = self.model.embed(query)

        results = self.client.search_vectors(
            self.collection_name, query_vector=embedding, top_k=top_k
        )

        elapsed = time.perf_counter() - start

        # Extract IDs from results
        ids = []
        if hasattr(results, "results"):
            for r in results.results:
                if hasattr(r, "id"):
                    ids.append(r.id)
                elif hasattr(r, "vector_id"):
                    ids.append(r.vector_id)
        elif isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    ids.append(r.get("id", r.get("vector_id", "")))
                elif hasattr(r, "id"):
                    ids.append(r.id)

        return ids, elapsed

    def cleanup(self):
        if self.client:
            try:
                self.client.delete_collection(self.collection_name)
            except:
                pass


# =============================================================================
# ProximaDB REST Benchmark
# =============================================================================


class ProximaDBRestBenchmark:
    def __init__(self, embedding_model, server_url: str = "http://localhost:5678"):
        self.model = embedding_model
        self.server_url = server_url
        self.collection_name = "benchmark_rest"
        self.dimension = embedding_model.get_dimension()
        self.client = None

    def setup(self):
        self.client = connect_rest(self.server_url)
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.client.create_collection(
            self.collection_name,
            CollectionConfig(name=self.collection_name, dimension=self.dimension),
        )

    def index(self, documents: list[dict], result: BenchmarkResult):
        for doc in documents:
            start = time.perf_counter()
            embedding = self.model.embed(doc["content"])

            self.client.insert_records(
                self.collection_name,
                [
                    {
                        "id": doc["id"],
                        "vector": embedding,
                        "props": {
                            "content": doc["content"][:500],
                            "language": doc["language"],
                            "path": doc["path"],
                        },
                    }
                ],
            )
            result.index_times.append(time.perf_counter() - start)

    def search(self, query: str, top_k: int = 5) -> tuple[list[str], float]:
        start = time.perf_counter()
        embedding = self.model.embed(query)

        response = self.client.search_envelope(
            self.collection_name, embedding, top_k=top_k
        )
        elapsed = time.perf_counter() - start
        ids = [item.id for item in response.items]
        return ids, elapsed

    def cleanup(self):
        if self.client:
            try:
                self.client.delete_collection(self.collection_name)
                self.client.close()
                return
            except Exception:
                pass
        try:
            httpx.delete(
                f"{self.server_url}/api/v1/collections/{self.collection_name}",
                timeout=10,
            )
        except:
            pass


# =============================================================================
# ProximaDB gRPC Benchmark
# =============================================================================


class ProximaDBGrpcBenchmark:
    def __init__(self, embedding_model, server_url: str = "localhost:5679"):
        self.model = embedding_model
        self.server_url = server_url
        self.collection_name = "benchmark_grpc"
        self.dimension = embedding_model.get_dimension()
        self.client = None

    def setup(self):
        grpc_url = self.server_url
        if not grpc_url.startswith("grpc://"):
            grpc_url = f"grpc://{grpc_url}"
        self.client = connect_grpc(grpc_url)
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.client.create_collection(
            self.collection_name,
            CollectionConfig(name=self.collection_name, dimension=self.dimension),
        )

    def index(self, documents: list[dict], result: BenchmarkResult):
        for doc in documents:
            start = time.perf_counter()
            embedding = self.model.embed(doc["content"])

            self.client.insert_records(
                self.collection_name,
                [
                    {
                        "id": doc["id"],
                        "vector": embedding,
                        "props": {
                            "content": doc["content"][:500],
                            "language": doc["language"],
                            "path": doc["path"],
                        },
                    }
                ],
            )
            result.index_times.append(time.perf_counter() - start)

    def search(self, query: str, top_k: int = 5) -> tuple[list[str], float]:
        start = time.perf_counter()
        embedding = self.model.embed(query)
        response = self.client.search(self.collection_name, embedding, top_k=top_k)
        elapsed = time.perf_counter() - start
        ids = [item.id for item in response]
        return ids, elapsed

    def cleanup(self):
        if self.client:
            try:
                self.client.delete_collection(self.collection_name)
                self.client.close()
                return
            except Exception:
                pass
        try:
            httpx.delete(
                f"http://localhost:5678/api/v1/collections/{self.collection_name}",
                timeout=10,
            )
        except:
            pass


# =============================================================================
# ProximaDB SQL Benchmark
# =============================================================================


class ProximaDBSqlBenchmark:
    """Uses SQL-like query interface via REST API."""

    def __init__(self, embedding_model, server_url: str = "http://localhost:5678"):
        self.model = embedding_model
        self.server_url = server_url
        self.collection_name = "benchmark_sql"
        self.dimension = embedding_model.get_dimension()
        self.client = None

    def setup(self):
        self.client = connect_rest(self.server_url)
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.client.create_collection(
            self.collection_name,
            CollectionConfig(name=self.collection_name, dimension=self.dimension),
        )

    def index(self, documents: list[dict], result: BenchmarkResult):
        # Batch insert for SQL-style
        batch_size = 10
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            start = time.perf_counter()

            vectors = []
            for doc in batch:
                embedding = self.model.embed(doc["content"])
                vectors.append(
                    {
                        "id": doc["id"],
                        "vector": embedding,
                        "props": {
                            "content": doc["content"][:500],
                            "language": doc["language"],
                            "path": doc["path"],
                        },
                    }
                )

            self.client.insert_records(self.collection_name, vectors)

            elapsed = time.perf_counter() - start
            for _ in batch:
                result.index_times.append(elapsed / len(batch))

    def search(self, query: str, top_k: int = 5) -> tuple[list[str], float]:
        start = time.perf_counter()
        embedding = self.model.embed(query)

        response = self.client.search_envelope(
            self.collection_name, embedding, top_k=top_k
        )
        elapsed = time.perf_counter() - start
        ids = [item.id for item in response.items]
        return ids, elapsed

    def cleanup(self):
        if self.client:
            try:
                self.client.delete_collection(self.collection_name)
                self.client.close()
                return
            except Exception:
                pass
        try:
            httpx.delete(
                f"{self.server_url}/api/v1/collections/{self.collection_name}",
                timeout=10,
            )
        except:
            pass


# =============================================================================
# Benchmark Runner
# =============================================================================


def run_benchmark(
    name: str,
    benchmark_class,
    embedding_model,
    documents: list[dict],
    queries: list[str],
    **kwargs,
) -> BenchmarkResult:
    result = BenchmarkResult(name)

    try:
        benchmark = benchmark_class(embedding_model, **kwargs)
        benchmark.setup()

        print(f"    Indexing {len(documents)} docs...")
        benchmark.index(documents, result)

        print(f"    Running {len(queries)} searches...")
        for query in queries:
            ids, elapsed = benchmark.search(query, top_k=5)
            result.search_times.append(elapsed)
            result.search_results.append(ids)

        benchmark.cleanup()

    except Exception as e:
        result.errors.append(str(e))
        print(f"    ERROR: {e}")

    return result


def run_batch_benchmark(
    embedding_model, temp_dir: str, num_docs: int = 100
) -> dict[str, dict]:
    """Run batch insert benchmark to show batch vs single insert performance."""

    print(f"\n  Running BATCH INSERT benchmark ({num_docs} docs)...")
    results = {}

    # Pre-generate embeddings (to isolate DB performance)
    print("    Pre-generating embeddings...")
    texts = [
        f"Sample document {i} with some content for embedding test"
        for i in range(num_docs)
    ]
    embeddings = embedding_model.embed_batch(texts)

    # Create test data
    documents = [
        {
            "id": f"doc_{i}",
            "content": texts[i],
            "vector": embeddings[i],
            "language": "test",
            "path": f"test/doc_{i}.txt",
        }
        for i in range(num_docs)
    ]

    # ChromaDB batch
    print("    ChromaDB batch insert...")
    try:
        chroma_client = chromadb.Client()  # In-memory
        chroma_coll = chroma_client.create_collection("batch_test")

        start = time.perf_counter()
        chroma_coll.add(
            ids=[d["id"] for d in documents],
            embeddings=[d["vector"] for d in documents],
            documents=[d["content"] for d in documents],
        )
        chroma_time = time.perf_counter() - start
        results["ChromaDB"] = {
            "batch_time_ms": chroma_time * 1000,
            "docs_per_sec": num_docs / chroma_time,
        }
    except Exception as e:
        results["ChromaDB"] = {"error": str(e)}

    # LanceDB batch
    print("    LanceDB batch insert...")
    try:
        lance_db = lancedb.connect(os.path.join(temp_dir, "lance_batch"))
        lance_data = [
            {"id": d["id"], "vector": d["vector"], "content": d["content"]}
            for d in documents
        ]

        start = time.perf_counter()
        lance_db.create_table("batch_test", lance_data)
        lance_time = time.perf_counter() - start
        results["LanceDB"] = {
            "batch_time_ms": lance_time * 1000,
            "docs_per_sec": num_docs / lance_time,
        }
    except Exception as e:
        results["LanceDB"] = {"error": str(e)}

    # FAISS batch
    print("    FAISS batch insert...")
    if FAISS_AVAILABLE:
        try:
            dim = len(embeddings[0])
            faiss_idx = faiss.IndexFlatIP(dim)
            vectors = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(vectors)

            start = time.perf_counter()
            faiss_idx.add(vectors)
            faiss_time = time.perf_counter() - start
            results["FAISS"] = {
                "batch_time_ms": faiss_time * 1000,
                "docs_per_sec": num_docs / faiss_time,
            }
        except Exception as e:
            results["FAISS"] = {"error": str(e)}

    # Qdrant batch
    print("    Qdrant batch insert...")
    if QDRANT_AVAILABLE:
        try:
            qdrant_client = QdrantClient(":memory:")
            qdrant_client.create_collection(
                "batch_test",
                vectors_config=VectorParams(
                    size=len(embeddings[0]), distance=Distance.COSINE
                ),
            )

            points = [
                PointStruct(id=i, vector=d["vector"], payload={"content": d["content"]})
                for i, d in enumerate(documents)
            ]

            start = time.perf_counter()
            qdrant_client.upsert("batch_test", points=points)
            qdrant_time = time.perf_counter() - start
            results["Qdrant"] = {
                "batch_time_ms": qdrant_time * 1000,
                "docs_per_sec": num_docs / qdrant_time,
            }
        except Exception as e:
            results["Qdrant"] = {"error": str(e)}

    return results


def calculate_semantic_accuracy(
    results: dict[str, BenchmarkResult],
    queries: list[str],
    expected: dict[str, list[str]],
) -> dict[str, float]:
    accuracy = {}

    for name, result in results.items():
        if not result.search_results:
            accuracy[name] = 0.0
            continue

        matches = 0
        total = 0

        for i, query in enumerate(queries):
            if query in expected and i < len(result.search_results):
                expected_keywords = expected[query]
                found_ids = result.search_results[i]

                for keyword in expected_keywords:
                    total += 1
                    if any(keyword.lower() in fid.lower() for fid in found_ids):
                        matches += 1

        accuracy[name] = (matches / total * 100) if total > 0 else 0.0

    return accuracy


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 80)
    print("EMBEDDED VECTOR DATABASE BENCHMARK FOR VICTOR CLI")
    print("Comparing: ChromaDB vs LanceDB vs FAISS vs Qdrant vs ProximaDB")
    print("Embedding: Sentence-Transformers (BAAI/bge-small-en-v1.5)")
    print("=" * 80)
    print("\nPurpose: Evaluate best vector DB for Victor CLI distribution")
    print("Note: ProximaDB uses gRPC, others are truly embedded.\n")

    # Collect code corpus
    print("[1/4] Collecting code corpus...")
    base_path = Path(__file__).parent.parent.parent.parent
    documents = collect_code_files(str(base_path), max_files=50)
    print(f"  Found {len(documents)} code files")
    print(f"  Languages: {set(d['language'] for d in documents)}")

    queries = [
        "vector similarity search function",
        "async HTTP request handler",
        "database collection create",
        "error handling exception",
        "configuration settings",
        "graph traversal algorithm",
        "embedding model dimension",
        "batch insert operation",
    ]

    expected_matches = {
        "vector similarity search function": ["search", "vector", "query"],
        "async HTTP request handler": ["http", "api", "handler", "server"],
        "database collection create": ["collection", "create"],
        "error handling exception": ["error", "exception"],
        "configuration settings": ["config", "settings"],
    }

    temp_base = tempfile.mkdtemp(prefix="vectordb_bench_")

    # Initialize embedding models (only sentence-transformers for reliable local inference)
    print("\n[2/4] Initializing embedding models...")
    models = {}

    try:
        print("  Loading BAAI/bge-small-en-v1.5 (Sentence-Transformers)...")
        models["ST-BGE"] = SentenceTransformerModel("BAAI/bge-small-en-v1.5")
        _ = models["ST-BGE"].get_dimension()
        print(f"    Ready: dim={models['ST-BGE'].get_dimension()}")
    except Exception as e:
        print(f"    ERROR: {e}")

    # Note: Ollama removed for benchmark reliability
    # Ollama can have 500 errors with long documents

    if not models:
        print("ERROR: No embedding models available!")
        return

    # Run benchmarks
    print("\n[3/4] Running benchmarks...")
    results = {}

    # Truly embedded databases only (no external servers needed)
    # Note: ProximaDB PyO3 bindings are implemented in Rust but wheel not yet built
    # See src/embedded/mod.rs and src/embedded/python.rs for the implementation
    benchmarks = [
        (
            "ChromaDB",
            ChromaDBBenchmark,
            lambda m, t: {"data_dir": os.path.join(t, f"chroma_{m}")},
        ),
        (
            "LanceDB",
            LanceDBBenchmark,
            lambda m, t: {"data_dir": os.path.join(t, f"lance_{m}")},
        ),
        (
            "FAISS",
            FAISSBenchmark,
            lambda m, t: {"data_dir": os.path.join(t, f"faiss_{m}")},
        ),
        (
            "Qdrant",
            QdrantBenchmark,
            lambda m, t: {"data_dir": os.path.join(t, f"qdrant_{m}")},
        ),
        # ProximaDB embedded bindings ready in Rust, wheel build needed for Python
        # ("ProximaDB", ProximaDBEmbeddedBenchmark, lambda m, t: {"grpc_url": "localhost:5679"}),
    ]

    for model_name, model in models.items():
        print(f"\n  ═══ {model_name} (dim={model.get_dimension()}) ═══")

        for db_name, db_class, kwargs_fn in benchmarks:
            kwargs = kwargs_fn(model_name, temp_base)
            if "data_dir" in kwargs:
                os.makedirs(kwargs["data_dir"], exist_ok=True)

            key = f"{db_name}+{model_name}"
            print(f"\n  {key}:")
            results[key] = run_benchmark(
                key, db_class, model, documents, queries, **kwargs
            )
            gc.collect()

    # Run batch insert benchmark (DB performance only, no embedding overhead)
    if "ST-BGE" in models:
        batch_results = run_batch_benchmark(models["ST-BGE"], temp_base, num_docs=500)

        print("\n  ═══ BATCH INSERT RESULTS (500 docs, pre-embedded) ═══")
        print("  ┌──────────────┬───────────────┬──────────────────┐")
        print("  │ Database     │ Batch Time    │ Throughput       │")
        print("  ├──────────────┼───────────────┼──────────────────┤")
        for db_name, stats in sorted(batch_results.items()):
            if "error" not in stats:
                print(
                    f"  │ {db_name:<12} │ {stats['batch_time_ms']:>10.2f}ms │ {stats['docs_per_sec']:>12.0f} docs/s │"
                )
            else:
                print(f"  │ {db_name:<12} │ ERROR: {stats['error'][:30]} │")
        print("  └──────────────┴───────────────┴──────────────────┘")

    # Calculate accuracy
    accuracy = calculate_semantic_accuracy(results, queries, expected_matches)

    # Print results table
    print("\n" + "=" * 80)
    print("[4/4] BENCHMARK RESULTS")
    print("=" * 80)

    print(
        "\n┌───────────────────────────────┬───────────┬───────────┬───────────┬──────────┐"
    )
    print(
        "│ Database + Embedding          │ Index     │ Search    │ Search    │ Semantic │"
    )
    print(
        "│                               │ (docs/s)  │ Avg (ms)  │ P95 (ms)  │ Accuracy │"
    )
    print(
        "├───────────────────────────────┼───────────┼───────────┼───────────┼──────────┤"
    )

    for name in sorted(results.keys()):
        result = results[name]
        summary = result.summary()
        acc = accuracy.get(name, 0)
        err = " ERR" if summary["errors"] > 0 else ""
        print(
            f"│ {name:<29} │ {summary['index_docs_per_sec']:>9.1f} │ {summary['search_avg_ms']:>9.2f} │ {summary['search_p95_ms']:>9.2f} │ {acc:>6.1f}%{err} │"
        )

    print(
        "└───────────────────────────────┴───────────┴───────────┴───────────┴──────────┘"
    )

    # Summary by Database
    print("\n" + "─" * 80)
    print("SUMMARY BY DATABASE (Embedded Mode)")
    print("─" * 80)

    for db in ["ChromaDB", "LanceDB", "FAISS", "Qdrant", "ProximaDB"]:
        db_results = {k: v for k, v in results.items() if k.startswith(db)}
        if db_results:
            valid_results = [
                r for r in db_results.values() if r.summary()["search_avg_ms"] > 0
            ]
            if valid_results:
                avg_index = statistics.mean(
                    r.summary()["index_docs_per_sec"] for r in valid_results
                )
                avg_search = statistics.mean(
                    r.summary()["search_avg_ms"] for r in valid_results
                )
                avg_acc = statistics.mean(accuracy.get(k, 0) for k in db_results)
                print(
                    f"  {db:<18}: Index={avg_index:>6.1f} docs/s | Search={avg_search:>6.2f}ms | Acc={avg_acc:>5.1f}%"
                )

    # Summary by Embedding
    print("\n" + "─" * 80)
    print("SUMMARY BY EMBEDDING MODEL")
    print("─" * 80)

    for emb in models:
        emb_results = {k: v for k, v in results.items() if emb in k}
        if emb_results:
            valid_results = [
                r for r in emb_results.values() if r.summary()["search_avg_ms"] > 0
            ]
            if valid_results:
                avg_index = statistics.mean(
                    r.summary()["index_docs_per_sec"] for r in valid_results
                )
                avg_search = statistics.mean(
                    r.summary()["search_avg_ms"] for r in valid_results
                )
                avg_acc = statistics.mean(accuracy.get(k, 0) for k in emb_results)
                print(
                    f"  {emb:<18}: Index={avg_index:>6.1f} docs/s | Search={avg_search:>6.2f}ms | Acc={avg_acc:>5.1f}%"
                )

    # Strengths & Weaknesses
    print("\n" + "=" * 80)
    print("STRENGTHS & WEAKNESSES")
    print("=" * 80)

    valid_results = {
        k: v for k, v in results.items() if v.summary()["search_avg_ms"] > 0
    }
    if valid_results:
        best_index = max(
            valid_results.items(), key=lambda x: x[1].summary()["index_docs_per_sec"]
        )
        best_search = min(
            valid_results.items(), key=lambda x: x[1].summary()["search_avg_ms"]
        )
        best_accuracy = max(accuracy.items(), key=lambda x: x[1])

        print(
            f"\n  FASTEST INDEXING:  {best_index[0]} ({best_index[1].summary()['index_docs_per_sec']:.1f} docs/s)"
        )
        print(
            f"  FASTEST SEARCH:    {best_search[0]} ({best_search[1].summary()['search_avg_ms']:.2f} ms)"
        )
        print(f"  BEST ACCURACY:     {best_accuracy[0]} ({best_accuracy[1]:.1f}%)")

    print("\n  EMBEDDED DATABASE ANALYSIS FOR VICTOR CLI:")
    print(
        "  ─────────────────────────────────────────────────────────────────────────────"
    )
    print("""
  ChromaDB:
    ✅ Simple API, Python-native
    ✅ Built-in persistence (SQLite + HNSW)
    ✅ Rich metadata filtering
    ✅ Good documentation
    ⚠️  Slower indexing for large batches
    ⚠️  Higher memory for large datasets
    📦 Deps: chromadb (~50MB)

  LanceDB:
    ✅ Fast disk-based storage (Arrow/Lance format)
    ✅ Scales to billions of vectors
    ✅ Excellent compression (columnar)
    ✅ Zero-copy reads
    ✅ Good batch performance
    ⚠️  Newer project, evolving API
    📦 Deps: lancedb + pyarrow (~100MB)

  FAISS (Facebook AI):
    ✅ Fastest search (highly optimized C++)
    ✅ Multiple index types (Flat, IVF, HNSW, PQ)
    ✅ CPU/GPU support
    ✅ Industry standard
    ❌ No built-in persistence (need to serialize)
    ❌ No metadata filtering (need separate store)
    ❌ Lower-level API
    📦 Deps: faiss-cpu (~15MB)

  Qdrant:
    ✅ Rich filtering on payload (metadata)
    ✅ Good persistence model
    ✅ HNSW + custom quantization
    ✅ Batch upsert
    ⚠️  Larger memory footprint
    ⚠️  More dependencies
    📦 Deps: qdrant-client (~30MB)

  ProximaDB (via gRPC - NOT truly embedded yet):
    ✅ specialized storage engines (SST, HELIX, etc.)
    ✅ Graph database integration (ORION, PULSAR, QUASAR)
    ✅ Rich quantization (Binary, INT8, PQ)
    ✅ Multiple distance metrics (Cosine, Euclidean, DotProduct)
    ✅ gRPC + REST APIs (binary protocol is fast)
    ✅ WAL for durability
    ✅ Native Rust performance
    ⚠️  Currently requires server process (gRPC overhead)
    ⚠️  Not truly embedded (unlike FAISS, LanceDB)
    📦 Deps: proximadb-sdk + grpcio (~20MB)

    📋 ROADMAP: PyO3 Python bindings for true embedded mode
       - Would eliminate server process
       - Zero serialization overhead
       - Same performance as native Rust

  RECOMMENDATION FOR VICTOR CLI:
  ────────────────────────────────
  1. LanceDB - Best for pure embedding storage (fast, scalable)
  2. ProximaDB - Best for code knowledge graphs + vectors
  3. FAISS + metadata store - Fastest search, but needs extra work
  4. ChromaDB - Simplest to integrate, slower at scale
  5. Qdrant - Best filtering, but heavier
""")

    print("  EMBEDDING MODEL ANALYSIS:")
    print(
        "  ─────────────────────────────────────────────────────────────────────────────"
    )
    print("""
  Sentence-Transformers (BGE):
    + Fast local inference (~5-10ms)
    + No API costs
    + Good for code search
    - Smaller context window
    - 384 dimensions

  Ollama (nomic-embed-text):
    + Higher quality embeddings
    + 768 dimensions (more expressive)
    + Larger context (8K tokens)
    - Slower (~50-100ms per embed)
    - Requires Ollama running
""")

    # Cleanup
    try:
        shutil.rmtree(temp_base)
    except:
        pass

    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
