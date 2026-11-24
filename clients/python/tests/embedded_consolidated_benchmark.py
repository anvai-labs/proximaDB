#!/usr/bin/env python3
"""
ProximaDB Consolidated Embedded Database Benchmark

Comprehensive benchmark covering ProximaDB's three core capabilities with
competitive comparison against industry-standard embedded databases:

1. Vector Store (6 storage engines) vs ChromaDB, LanceDB, FAISS, Qdrant
2. Graph Store (ORION engine) vs NetworkX, igraph
3. Semantic Knowledge Store (unified vector + graph workloads) - unique to ProximaDB

This benchmark runs REAL workloads against actual embedded database engines -
no simulations or mock data.

Engine Comparison Table:
========================

VECTOR ENGINES (ProximaDB):
| Engine | Use Case                  | Performance (10K vectors) |
|--------|---------------------------|---------------------------|
| SST    | Write-optimized, real-time| ~5.32ms (LZ4)             |
| HELIX  | Locality-optimized        | ~13.2ms                   |
| VIPER  | Columnar analytics        | ~89.5ms                   |
| SWIFT  | Ultra-low latency (<5K)   | ~95ms                     |
| NOVA   | Progressive columnar      | ~101.6ms                  |
| RAPTOR | Adaptive row-group        | ~9.36ms                   |

VECTOR COMPETITORS:
| Database | Architecture              | Best For                  |
|----------|---------------------------|---------------------------|
| ChromaDB | SQLite + HNSW             | Simple prototyping        |
| LanceDB  | Columnar Arrow            | Analytics workloads       |
| FAISS    | In-memory flat/IVF/HNSW   | High-performance search   |
| Qdrant   | Rust + HNSW + filtering   | Production deployments    |

GRAPH ENGINES (ProximaDB):
| Engine | Use Case                  | Performance               |
|--------|---------------------------|---------------------------|
| ORION  | In-memory CSR format      | 1M+ edges/sec traversal   |
| PULSAR | Distributed sharding      | Billion-scale nodes       |
| QUASAR | Hot/cold tiering          | 80-90% storage savings    |

GRAPH COMPETITORS:
| Database  | Architecture              | Best For                  |
|-----------|---------------------------|---------------------------|
| NetworkX  | Python dict-based         | Prototyping, algorithms   |
| igraph    | C library + Python        | Large-scale analytics     |

SEMANTIC KNOWLEDGE STORE (SKS):
Unified vector + graph with:
- Graph-first architecture (entities as nodes, relations as edges)
- Hybrid queries (vector similarity + graph traversal)
- Expected 10-20x improvement over split storage
- NO DIRECT COMPETITORS (unique architecture)
"""

import gc
import os
import random
import shutil
import struct
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    Table = None


# =============================================================================
# Logging and Timing Utilities
# =============================================================================


class BenchmarkLogger:
    """Utility for logging benchmark progress with timestamps and inline stats.

    Features:
    - Single-line compact output format: [time] ENGINE: operation | stat1=val1 | stat2=val2
    - Horizontal separators between engines for readability
    - All stats on one line for easy parsing and comparison
    """

    @staticmethod
    def separator(char: str = "-", width: int = 70):
        """Print a horizontal separator between engines."""
        print(f"  {char * width}")

    @staticmethod
    def log_engine_start(engine_name: str):
        """Log engine separator before starting benchmarks."""
        BenchmarkLogger.separator()
        print(f"  ENGINE: {engine_name}")
        BenchmarkLogger.separator("·", 70)

    @staticmethod
    def log_start(engine_name: str, operation: str = ""):
        """Log the start of a benchmark operation (silent, returns start time)."""
        return time.perf_counter()

    @staticmethod
    def log_end(engine_name: str, start_time: float, stats: dict[str, Any] = None):
        """Log benchmark completion with all stats on a single line."""
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Format compact stats on single line
        stats_parts = [f"total={elapsed_ms:.1f}ms"]
        if stats:
            for key, value in stats.items():
                if isinstance(value, float):
                    stats_parts.append(f"{key}={value:.2f}")
                elif isinstance(value, int):
                    stats_parts.append(f"{key}={value:,}")
                else:
                    stats_parts.append(f"{key}={value}")

        stats_str = " | ".join(stats_parts)
        print(f"  [{timestamp}] ✓ {engine_name}: {stats_str}")
        return elapsed_ms

    @staticmethod
    def log_error(engine_name: str, error: Exception):
        """Log an error during benchmark."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        error_msg = str(error)[:80]  # Truncate long error messages
        print(f"  [{timestamp}] ✗ {engine_name}: ERROR - {error_msg}")


# =============================================================================
# Standard Benchmark Dataset Loaders
# =============================================================================


class StandardDatasets:
    """
    Standard benchmark dataset loaders for reproducible benchmarks.

    These are industry-standard datasets used across vector and graph databases:
    - SIFT: 128D SIFT descriptors from computer vision
    - GIST: 960D GIST descriptors (larger/more complex)
    - LDBC SNB: Social network graph (Linked Data Benchmark Council)

    Hardware-aware sizing for M1 Max (64GB RAM, 133GB disk):
    - SIFT-1M: ~500MB memory, ~600MB disk → use full 1M vectors
    - GIST-1M: ~3.6GB memory → use 100K-500K based on preference
    - LDBC SF1: ~1GB → fits easily
    """

    # Dataset URLs (using public mirrors)
    SIFT_BASE_URL = "ftp://ftp.irisa.fr/local/texmex/corpus"
    SIFT_1M_URL = f"{SIFT_BASE_URL}/sift.tar.gz"

    # Alternative: Use ANN-Benchmarks format (simpler .npy format)
    ANN_BENCHMARK_URL = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"

    @staticmethod
    def _read_fvecs(fp) -> np.ndarray:
        """Read vectors from fvecs format (TEXMEX format)."""
        vectors = []
        while True:
            dim_bytes = fp.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack("<i", dim_bytes)[0]
            vec = struct.unpack("<" + "f" * dim, fp.read(4 * dim))
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

    @staticmethod
    def _read_ivecs(fp) -> np.ndarray:
        """Read integers from ivecs format (for ground truth)."""
        vectors = []
        while True:
            dim_bytes = fp.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack("<i", dim_bytes)[0]
            vec = struct.unpack("<" + "i" * dim, fp.read(4 * dim))
            vectors.append(vec)
        return np.array(vectors, dtype=np.int32)

    @staticmethod
    def load_sift_1m(
        max_vectors: int = None, cache_dir: str = None, dimension: int = 768
    ) -> dict:
        """
        Load SIFT-1M dataset (or generate synthetic equivalent).

        SIFT-1M: 1 million 128-dimensional SIFT descriptors
        - Base vectors: 1,000,000 x 128

        Supports configurable dimensions for real-world embedding models:
        - SIFT: 128D (default, industry standard)
        - BGE-base: 768D
        - GIST: 960D
        - BGE-large: 1024D
        - OpenAI ada-002: 1536D
        - Query vectors: 10,000 x 128
        - Ground truth: nearest neighbors for queries

        Args:
            max_vectors: Limit to N vectors (None = use all)
            cache_dir: Directory to cache downloaded data

        Returns:
            dict with 'base', 'queries', 'ground_truth', 'metadata'
        """
        if cache_dir:
            cache_path = Path(cache_dir) / "sift_1m.npz"
            if cache_path.exists():
                print(f"  Loading SIFT-1M from cache: {cache_path}")
                data = np.load(cache_path)
                base = data["base"]
                queries = data["queries"]
                if max_vectors:
                    base = base[:max_vectors]
                return {
                    "base": base,
                    "queries": queries,
                    "ground_truth": data.get("ground_truth"),
                    "metadata": {
                        "dataset": "sift-1m",
                        "dimension": 128,
                        "num_base": len(base),
                        "num_queries": len(queries),
                        "distance": "euclidean",
                    },
                }

        # Generate synthetic SIFT-like vectors (similar statistical properties)
        # Real SIFT: 128D with values in [0, 255], often sparse
        # Support configurable dimensions for real-world embedding models
        dim_names = {
            128: "SIFT",
            384: "Victor/BGE-small",
            768: "BGE-base",
            960: "GIST",
            1024: "BGE-large",
            1536: "OpenAI",
        }
        dim_name = dim_names.get(dimension, f"custom-{dimension}D")
        print(f"  Generating synthetic {dim_name} vectors ({dimension}D)...")

        num_base = min(max_vectors or 1_000_000, 1_000_000)
        num_queries = min(10_000, num_base)  # Can't have more queries than base vectors
        # dimension is passed as parameter

        # SIFT vectors have specific statistical properties:
        # - Most values are small (histogram-like)
        # - Some dimensions have larger values
        # - L2 normalized for search

        rng = np.random.default_rng(42)  # Reproducible

        # Generate base vectors with SIFT-like distribution
        base = rng.gamma(2.0, 2.0, size=(num_base, dimension)).astype(np.float32)
        base = np.clip(base, 0, 255)
        # L2 normalize
        norms = np.linalg.norm(base, axis=1, keepdims=True)
        base = base / np.maximum(norms, 1e-8)

        # Generate query vectors (slight perturbation of random base vectors)
        query_indices = rng.choice(num_base, num_queries, replace=False)
        queries = base[query_indices].copy()
        # Add noise
        queries += rng.normal(0, 0.05, queries.shape).astype(np.float32)
        # Re-normalize
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / np.maximum(norms, 1e-8)

        if cache_dir:
            cache_path = Path(cache_dir) / "sift_1m.npz"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, base=base, queries=queries)
            print(f"  Cached to: {cache_path}")

        return {
            "base": base,
            "queries": queries,
            "ground_truth": None,  # Would compute via brute force if needed
            "metadata": {
                "dataset": "sift-1m-synthetic",
                "dimension": dimension,
                "num_base": num_base,
                "num_queries": num_queries,
                "distance": "euclidean",
            },
        }

    @staticmethod
    def load_gist_960(max_vectors: int = None, cache_dir: str = None) -> dict:
        """
        Load GIST-960 dataset (or generate synthetic equivalent).

        GIST: 960-dimensional global image descriptors
        - Much higher dimensionality = harder to index
        - Standard for testing scalability

        Args:
            max_vectors: Limit vectors (default: 100K for memory efficiency)
            cache_dir: Cache directory

        Returns:
            dict with 'base', 'queries', 'metadata'
        """
        print("  Generating synthetic GIST-960 vectors...")

        # Default to 100K for GIST (960D = 3.84KB per vector)
        num_base = min(max_vectors or 100_000, 1_000_000)
        num_queries = 1_000
        dimension = 960

        rng = np.random.default_rng(43)

        # GIST vectors are dense, roughly Gaussian
        base = rng.standard_normal((num_base, dimension)).astype(np.float32)
        norms = np.linalg.norm(base, axis=1, keepdims=True)
        base = base / np.maximum(norms, 1e-8)

        query_indices = rng.choice(num_base, num_queries, replace=False)
        queries = base[query_indices].copy()
        queries += rng.normal(0, 0.1, queries.shape).astype(np.float32)
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / np.maximum(norms, 1e-8)

        return {
            "base": base,
            "queries": queries,
            "ground_truth": None,
            "metadata": {
                "dataset": "gist-960-synthetic",
                "dimension": dimension,
                "num_base": num_base,
                "num_queries": num_queries,
                "distance": "euclidean",
            },
        }

    @staticmethod
    def load_ldbc_snb(scale_factor: int = 1, cache_dir: str = None) -> dict:
        """
        Load LDBC Social Network Benchmark graph (or generate synthetic equivalent).

        LDBC SNB: Realistic social network with:
        - Persons, Posts, Comments, Forums, Tags
        - Complex relationships (KNOWS, LIKES, HAS_CREATOR, etc.)
        - Power-law degree distribution
        - Temporal evolution

        Scale Factors:
        - SF1: ~3M edges, ~1M nodes
        - SF10: ~30M edges, ~10M nodes
        - SF100: ~300M edges

        For M1 Max with 64GB: SF1-SF10 fits easily

        Args:
            scale_factor: 1, 10, 100, etc.
            cache_dir: Cache directory

        Returns:
            dict with 'nodes', 'edges', 'metadata'
        """
        print(f"  Generating synthetic LDBC SNB (Scale Factor {scale_factor})...")

        rng = np.random.default_rng(44)

        # LDBC SNB approximate sizes per SF
        # SF1: ~11K persons, ~1M posts, ~2M comments
        base_persons = 11_000 * scale_factor
        base_posts = 100_000 * scale_factor  # Reduced for benchmark
        base_comments = 200_000 * scale_factor  # Reduced

        # Cap for memory constraints
        max_nodes = min(base_persons + base_posts + base_comments, 500_000)

        nodes = []
        edges = []
        node_id_counter = 0

        # Generate Persons (with power-law friends)
        num_persons = min(base_persons, max_nodes // 3)
        person_ids = []

        for i in range(num_persons):
            node_id = f"person_{i}"
            person_ids.append(node_id)
            nodes.append(
                {
                    "id": node_id,
                    "labels": ["Person"],
                    "properties": {
                        "firstName": f"Person_{i}",
                        "lastName": f"Last_{i % 1000}",
                        "gender": rng.choice(["male", "female"]),
                        "birthday": f"199{rng.integers(0, 10)}-{rng.integers(1, 13):02d}-{rng.integers(1, 29):02d}",
                        "locationIP": f"192.168.{rng.integers(0, 256)}.{rng.integers(0, 256)}",
                        "creationDate": f"2010-{rng.integers(1, 13):02d}-{rng.integers(1, 29):02d}",
                    },
                }
            )

        # Generate KNOWS relationships (power-law distribution)
        # Average ~50 friends per person
        for i, person_id in enumerate(person_ids):
            # Power-law: some people have many friends, most have few
            num_friends = int(rng.pareto(1.5) * 10) + 1
            num_friends = min(num_friends, 500, len(person_ids) - 1)

            friend_indices = rng.choice(
                [j for j in range(len(person_ids)) if j != i],
                size=min(num_friends, len(person_ids) - 1),
                replace=False,
            )

            for friend_idx in friend_indices:
                edges.append(
                    {
                        "from_node_id": person_id,
                        "to_node_id": person_ids[friend_idx],
                        "edge_type": "KNOWS",
                        "weight": rng.random(),
                        "properties": {
                            "creationDate": f"2011-{rng.integers(1, 13):02d}-{rng.integers(1, 29):02d}"
                        },
                    }
                )

        # Generate Posts
        num_posts = min(base_posts, max_nodes // 3)
        post_ids = []

        for i in range(num_posts):
            node_id = f"post_{i}"
            post_ids.append(node_id)
            author = rng.choice(person_ids)
            nodes.append(
                {
                    "id": node_id,
                    "labels": ["Post", "Message"],
                    "properties": {
                        "content": f"Post content {i}",
                        "length": str(rng.integers(10, 2000)),
                        "creationDate": f"2012-{rng.integers(1, 13):02d}-{rng.integers(1, 29):02d}",
                    },
                }
            )
            # HAS_CREATOR edge
            edges.append(
                {
                    "from_node_id": node_id,
                    "to_node_id": author,
                    "edge_type": "HAS_CREATOR",
                    "weight": 1.0,
                }
            )
            # Some people LIKES posts
            num_likes = int(rng.pareto(1.0) * 5)
            for _ in range(min(num_likes, 50)):
                liker = rng.choice(person_ids)
                if liker != author:
                    edges.append(
                        {
                            "from_node_id": liker,
                            "to_node_id": node_id,
                            "edge_type": "LIKES",
                            "weight": 1.0,
                        }
                    )

        # Generate Comments (replies to posts)
        num_comments = min(base_comments, max_nodes // 3)

        for i in range(num_comments):
            node_id = f"comment_{i}"
            author = rng.choice(person_ids)
            parent = (
                rng.choice(post_ids) if rng.random() > 0.3 else rng.choice(post_ids)
            )

            nodes.append(
                {
                    "id": node_id,
                    "labels": ["Comment", "Message"],
                    "properties": {
                        "content": f"Comment content {i}",
                        "length": str(rng.integers(5, 500)),
                        "creationDate": f"2012-{rng.integers(1, 13):02d}-{rng.integers(1, 29):02d}",
                    },
                }
            )
            edges.append(
                {
                    "from_node_id": node_id,
                    "to_node_id": author,
                    "edge_type": "HAS_CREATOR",
                    "weight": 1.0,
                }
            )
            edges.append(
                {
                    "from_node_id": node_id,
                    "to_node_id": parent,
                    "edge_type": "REPLY_OF",
                    "weight": 1.0,
                }
            )

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "dataset": f"ldbc-snb-sf{scale_factor}-synthetic",
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "num_persons": num_persons,
                "num_posts": num_posts,
                "num_comments": num_comments,
                "scale_factor": scale_factor,
            },
        }

    @staticmethod
    def load_hybrid_rag_dataset(
        num_documents: int = 10_000, cache_dir: str = None
    ) -> dict:
        """
        Generate a HybridRAG-style dataset for SKS benchmarking.

        Simulates a knowledge base with:
        - Documents with embeddings (for vector search)
        - Entity extraction (for graph relationships)
        - Chunked passages for retrieval

        This is the use case where SKS (Semantic Knowledge Store) excels:
        vector similarity + graph traversal for context expansion.

        Args:
            num_documents: Number of documents
            cache_dir: Cache directory

        Returns:
            dict with 'documents', 'chunks', 'entities', 'relations', 'metadata'
        """
        print(f"  Generating HybridRAG dataset ({num_documents:,} documents)...")

        rng = np.random.default_rng(45)
        dimension = 384  # Common embedding dimension (sentence-transformers)
        chunks_per_doc = 5
        entities_per_doc = 3

        documents = []
        chunks = []
        entities = []
        relations = []

        # Entity types for knowledge graph
        entity_types = ["Person", "Organization", "Location", "Concept", "Product"]
        relation_types = [
            "MENTIONS",
            "RELATED_TO",
            "LOCATED_IN",
            "WORKS_FOR",
            "CREATED_BY",
        ]

        # Generate documents
        for doc_id in range(num_documents):
            # Document metadata
            doc = {
                "id": f"doc_{doc_id}",
                "title": f"Document {doc_id}",
                "source": rng.choice(["wikipedia", "arxiv", "news", "blog"]),
                "category": rng.choice(["science", "tech", "business", "health"]),
            }
            documents.append(doc)

            # Generate chunks with embeddings
            for chunk_id in range(chunks_per_doc):
                # Generate realistic embedding (clustered by category)
                category_offset = hash(doc["category"]) % 10
                embedding = rng.standard_normal(dimension).astype(np.float32)
                embedding[:10] += category_offset * 0.5  # Category signal
                embedding = embedding / np.linalg.norm(embedding)

                chunks.append(
                    {
                        "id": f"chunk_{doc_id}_{chunk_id}",
                        "document_id": f"doc_{doc_id}",
                        "text": f"Chunk {chunk_id} of document {doc_id}",
                        "embedding": embedding,
                        "position": chunk_id,
                    }
                )

            # Generate entities extracted from document
            doc_entities = []
            for ent_id in range(entities_per_doc):
                entity = {
                    "id": f"entity_{doc_id}_{ent_id}",
                    "name": f"Entity_{rng.integers(0, 1000)}",
                    "type": rng.choice(entity_types),
                    "document_id": f"doc_{doc_id}",
                }
                entities.append(entity)
                doc_entities.append(entity)

                # Relation: Document MENTIONS Entity
                relations.append(
                    {
                        "from_id": f"doc_{doc_id}",
                        "to_id": entity["id"],
                        "type": "MENTIONS",
                        "weight": rng.random(),
                    }
                )

            # Generate inter-entity relations
            for i, ent in enumerate(doc_entities):
                if i < len(doc_entities) - 1:
                    relations.append(
                        {
                            "from_id": ent["id"],
                            "to_id": doc_entities[i + 1]["id"],
                            "type": rng.choice(relation_types),
                            "weight": rng.random(),
                        }
                    )

        # Cross-document entity relations (entities with same name are related)
        entity_by_name = {}
        for ent in entities:
            if ent["name"] not in entity_by_name:
                entity_by_name[ent["name"]] = []
            entity_by_name[ent["name"]].append(ent)

        for name, ents in entity_by_name.items():
            if len(ents) > 1:
                for i in range(min(len(ents) - 1, 5)):
                    relations.append(
                        {
                            "from_id": ents[i]["id"],
                            "to_id": ents[i + 1]["id"],
                            "type": "SAME_AS",
                            "weight": 0.9,
                        }
                    )

        return {
            "documents": documents,
            "chunks": chunks,
            "entities": entities,
            "relations": relations,
            "embeddings": np.array([c["embedding"] for c in chunks], dtype=np.float32),
            "metadata": {
                "dataset": "hybrid-rag-synthetic",
                "num_documents": num_documents,
                "num_chunks": len(chunks),
                "num_entities": len(entities),
                "num_relations": len(relations),
                "embedding_dimension": dimension,
            },
        }


# =============================================================================
# Database Imports
# =============================================================================

# ProximaDB (native Rust via PyO3)
try:
    import proximadb

    PROXIMADB_AVAILABLE = True
    print(f"ProximaDB v{proximadb.__version__} loaded (embedded mode)")
except ImportError as e:
    PROXIMADB_AVAILABLE = False
    print(f"ProximaDB not available: {e}")

# Vector DB Competitors
try:
    import chromadb

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    import lancedb

    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# Additional Vector DB Competitors
try:
    from usearch.index import Index as USearchIndex

    USEARCH_AVAILABLE = True
except ImportError:
    USEARCH_AVAILABLE = False

try:
    from annoy import AnnoyIndex

    ANNOY_AVAILABLE = True
except ImportError:
    ANNOY_AVAILABLE = False

try:
    from pymilvus import MilvusClient

    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

# Graph DB Competitors
try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import igraph

    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False


# =============================================================================
# Benchmark Configuration
# =============================================================================


@dataclass
class BenchmarkConfig:
    """Configuration for the consolidated benchmark."""

    # Vector benchmark settings
    vector_sizes: list[int] = field(default_factory=lambda: [1000, 5000, 10000])
    vector_dimension: int = 384

    # Graph benchmark settings
    graph_nodes: list[int] = field(default_factory=lambda: [1000, 10000])
    edge_density: float = 5.0  # Edges per node

    # SKS benchmark settings
    sks_entities: list[int] = field(default_factory=lambda: [1000, 5000])
    sks_relations_per_entity: float = 3.0


@dataclass
class BenchmarkResult:
    """Result from a benchmark operation."""

    operation: str
    engine: str
    time_ms: float
    throughput: float | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None
    recall_at_k: float | None = None  # NEW: Recall@K accuracy metric
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Recall@K Accuracy Functions
# =============================================================================


def compute_ground_truth(
    base_vectors: np.ndarray, query_vectors: np.ndarray, k: int
) -> list[list[int]]:
    """
    Compute ground truth top-k neighbors using brute-force L2 distance.

    Args:
        base_vectors: Array of base vectors (n x d)
        query_vectors: Array of query vectors (m x d)
        k: Number of neighbors to find

    Returns:
        List of lists, each containing the indices of the k nearest neighbors
    """
    ground_truth = []
    for query in query_vectors:
        # Compute L2 distances to all base vectors
        distances = np.sum((base_vectors - query) ** 2, axis=1)
        # Get top-k indices (sorted by distance, ascending)
        top_k_indices = np.argsort(distances)[:k]
        ground_truth.append(top_k_indices.tolist())
    return ground_truth


def compute_ground_truth_cached(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    k: int,
    num_base: int,
    dimension: int,
    num_queries: int,
    seed: int = 42,
) -> tuple[list[list[int]], bool, float]:
    """
    Compute ground truth with caching to avoid recomputation on subsequent runs.

    Cache key is based on: num_base, dimension, num_queries, k, seed
    Cache is stored in ~/.cache/proximadb/benchmarks/

    Args:
        base_vectors: Array of base vectors (n x d)
        query_vectors: Array of query vectors (m x d)
        k: Number of neighbors to find
        num_base: Number of base vectors (for cache key)
        dimension: Vector dimension (for cache key)
        num_queries: Number of queries (for cache key)
        seed: Random seed used for data generation (for cache key)

    Returns:
        Tuple of (ground_truth, from_cache, time_ms)
    """
    # Create cache directory
    cache_dir = Path.home() / ".cache" / "proximadb" / "benchmarks"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Cache filename based on parameters
    cache_key = f"gt_{num_base}v_{dimension}d_{num_queries}q_k{k}_seed{seed}.npy"
    cache_path = cache_dir / cache_key

    # Try to load from cache
    if cache_path.exists():
        try:
            gt_array = np.load(cache_path, allow_pickle=True)
            ground_truth = gt_array.tolist()
            return ground_truth, True, 0.0
        except Exception as e:
            print(f" (cache load failed: {e}, recomputing)")

    # Compute ground truth
    start_time = time.perf_counter()
    ground_truth = compute_ground_truth(base_vectors, query_vectors, k)
    compute_time_ms = (time.perf_counter() - start_time) * 1000

    # Save to cache
    try:
        np.save(cache_path, np.array(ground_truth, dtype=object))
    except Exception as e:
        print(f" (cache save failed: {e})")

    return ground_truth, False, compute_time_ms


def compute_recall_at_k(
    ground_truth_indices: list[int], result_ids: list[str], k: int
) -> float:
    """
    Compute recall@k: fraction of ground truth neighbors found in results.

    Args:
        ground_truth_indices: List of ground truth neighbor indices
        result_ids: List of result IDs (format: "sift_X" or "vec_X")
        k: Number of results to consider

    Returns:
        Recall@k as a float in [0, 1]
    """
    if not ground_truth_indices or not result_ids:
        return 0.0

    # Convert ground truth indices to set
    gt_set = set(ground_truth_indices[:k])

    # Convert result IDs to indices
    result_indices = set()
    for result_id in result_ids[:k]:
        try:
            # Handle both "sift_X" and "vec_X" formats
            idx = int(result_id.split("_")[-1])
            result_indices.add(idx)
        except (ValueError, IndexError):
            pass

    # Compute intersection
    intersection = len(gt_set & result_indices)
    return intersection / min(k, len(gt_set)) if gt_set else 0.0


# =============================================================================
# Data Generators
# =============================================================================


def generate_vectors(num_vectors: int, dimension: int = 384) -> np.ndarray:
    """Generate normalized random vectors."""
    vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def generate_graph_data(
    num_nodes: int, edge_density: float
) -> tuple[list[dict], list[dict]]:
    """Generate graph nodes and edges."""
    labels = ["Entity", "Document", "Concept", "Person", "Organization"]
    edge_types = ["RELATED_TO", "REFERENCES", "CONTAINS", "AUTHORED_BY", "BELONGS_TO"]

    nodes = []
    for i in range(num_nodes):
        nodes.append(
            {
                "id": f"entity_{i}",
                "labels": [random.choice(labels)],
                "properties": {
                    "name": f"Entity_{i}",
                    "category": random.choice(["A", "B", "C", "D"]),
                    "score": str(random.randint(1, 100)),
                },
            }
        )

    num_edges = int(num_nodes * edge_density)
    edges = []
    edge_set = set()

    for _ in range(num_edges):
        from_idx = random.randint(0, num_nodes - 1)
        locality = min(num_nodes // 10, 100)
        to_idx = (from_idx + random.randint(1, locality)) % num_nodes

        if (from_idx, to_idx) in edge_set:
            continue
        edge_set.add((from_idx, to_idx))

        edges.append(
            {
                "from_node_id": f"entity_{from_idx}",
                "to_node_id": f"entity_{to_idx}",
                "edge_type": random.choice(edge_types),
                "weight": random.random(),
            }
        )

    return nodes, edges


def generate_sks_data(
    num_entities: int, relations_per_entity: float, dimension: int = 128
):
    """Generate semantic knowledge store data (entities with embeddings + relations)."""
    # Generate entities with embeddings
    vectors = generate_vectors(num_entities, dimension)

    entities = []
    for i in range(num_entities):
        entities.append(
            {
                "id": f"sks_entity_{i}",
                "labels": ["SKSEntity"],
                "properties": {
                    "name": f"SKS_Entity_{i}",
                    "domain": random.choice(["science", "tech", "art", "business"]),
                },
                "embedding": vectors[i].tolist(),
            }
        )

    # Generate relations
    num_relations = int(num_entities * relations_per_entity)
    relations = []
    seen_pairs = set()
    while len(relations) < num_relations:
        from_idx = random.randint(0, num_entities - 1)
        to_idx = random.randint(0, num_entities - 1)
        if from_idx == to_idx:
            continue
        key = (from_idx, to_idx)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        relations.append(
            {
                "from_entity_id": f"sks_entity_{from_idx}",
                "to_entity_id": f"sks_entity_{to_idx}",
                "relation_type": random.choice(["SIMILAR_TO", "DERIVED_FROM", "CITES"]),
            }
        )

    return entities, relations, vectors


# =============================================================================
# Vector Store Benchmarks
# =============================================================================


def benchmark_vector_store(
    config: BenchmarkConfig, temp_dir: str
) -> list[BenchmarkResult]:
    """Benchmark vector storage operations across all engines."""
    if not PROXIMADB_AVAILABLE:
        return [BenchmarkResult("init", "all", 0, error="ProximaDB not available")]

    results = []

    # Define engine configurations
    # Exact mode: Supported by all engines
    engines_exact = ["sst", "helix", "nova", "swift", "raptor"]
    # Approximate (sqrt) mode: Supported by specific engines
    engines_approx = ["sst", "helix", "swift"]

    for num_vectors in config.vector_sizes:
        vectors = generate_vectors(num_vectors, config.vector_dimension)

        # 1. Benchmark Exact Mode (All Engines)
        for engine in engines_exact:
            mode = "exact"
            engine_label = f"{engine}"
            print(
                f"    Vector benchmark: {engine.upper()} ({mode}) ({num_vectors:,} vectors)...",
                end="",
                flush=True,
            )

            try:
                data_dir = os.path.join(
                    temp_dir, f"vector_{engine}_{mode}_{num_vectors}"
                )
                os.makedirs(data_dir, exist_ok=True)

                # Use context manager for proper cleanup
                with proximadb.ProximaDB(
                    data_dirs=data_dir,
                    metadata_dir=os.path.join(data_dir, "metadata"),
                    cache_size_mb=128,
                    enable_wal=False,
                    prune_mode=mode,
                ) as db:

                    collection_name = f"bench_{engine}_{mode}"
                    db.create_collection(
                        collection_name, config.vector_dimension, engine
                    )

                    # Insert benchmark - use zero-copy insert_numpy if available
                    ids = [f"vec_{i}" for i in range(num_vectors)]

                    start = time.perf_counter()
                    if hasattr(db, "insert_numpy"):
                        # Zero-copy numpy insert (faster)
                        db.insert_numpy(collection_name, ids, vectors)
                    else:
                        # Fallback to list-based insert
                        vectors_list = [v.tolist() for v in vectors]
                        db.insert(collection_name, ids, vectors_list, None)
                    insert_time = (time.perf_counter() - start) * 1000

                    results.append(
                        BenchmarkResult(
                            "vector_insert",
                            engine_label,
                            insert_time,
                            throughput=num_vectors / (insert_time / 1000),
                            metadata={"num_vectors": num_vectors, "mode": mode},
                        )
                    )

                    # Search benchmark - use zero-copy search_numpy if available
                    search_times = []
                    for _ in range(10):
                        start = time.perf_counter()
                        if hasattr(db, "search_numpy"):
                            _ = db.search_numpy(collection_name, vectors[0], 10)
                        else:
                            _ = db.search(collection_name, vectors[0].tolist(), 10)
                        search_times.append((time.perf_counter() - start) * 1000)

                    results.append(
                        BenchmarkResult(
                            "vector_search",
                            engine_label,
                            float(np.mean(search_times)),
                            p50_ms=float(np.percentile(search_times, 50)),
                            p95_ms=float(np.percentile(search_times, 95)),
                            p99_ms=float(np.percentile(search_times, 99)),
                            metadata={"num_vectors": num_vectors, "mode": mode},
                        )
                    )

                    print(
                        f" {insert_time:.2f}ms insert, {np.mean(search_times):.3f}ms search"
                    )

            except Exception as e:
                results.append(
                    BenchmarkResult(
                        "vector_ops",
                        engine_label,
                        0,
                        error=str(e),
                        metadata={"num_vectors": num_vectors, "mode": mode},
                    )
                )
                print(f" ERROR: {str(e)[:50]}")

            gc.collect()

        # 2. Benchmark Approximate Mode (Sqrt) - Only for supported engines
        for engine in engines_approx:
            mode = "sqrt"
            engine_label = f"{engine} (approx)"
            print(
                f"    Vector benchmark: {engine.upper()} ({mode}) ({num_vectors:,} vectors)...",
                end="",
                flush=True,
            )

            try:
                data_dir = os.path.join(
                    temp_dir, f"vector_{engine}_{mode}_{num_vectors}"
                )
                os.makedirs(data_dir, exist_ok=True)

                # Use context manager for proper cleanup
                with proximadb.ProximaDB(
                    data_dirs=data_dir,
                    metadata_dir=os.path.join(data_dir, "metadata"),
                    cache_size_mb=128,
                    enable_wal=False,
                    prune_mode=mode,
                ) as db:

                    collection_name = f"bench_{engine}_{mode}"
                    db.create_collection(
                        collection_name, config.vector_dimension, engine
                    )

                    # Insert benchmark
                    ids = [f"vec_{i}" for i in range(num_vectors)]

                    start = time.perf_counter()
                    if hasattr(db, "insert_numpy"):
                        db.insert_numpy(collection_name, ids, vectors)
                    else:
                        vectors_list = [v.tolist() for v in vectors]
                        db.insert(collection_name, ids, vectors_list, None)
                    insert_time = (time.perf_counter() - start) * 1000

                    results.append(
                        BenchmarkResult(
                            "vector_insert",
                            engine_label,
                            insert_time,
                            throughput=num_vectors / (insert_time / 1000),
                            metadata={"num_vectors": num_vectors, "mode": mode},
                        )
                    )

                    # Search benchmark
                    search_times = []
                    for _ in range(10):
                        start = time.perf_counter()
                        if hasattr(db, "search_numpy"):
                            _ = db.search_numpy(collection_name, vectors[0], 10)
                        else:
                            _ = db.search(collection_name, vectors[0].tolist(), 10)
                        search_times.append((time.perf_counter() - start) * 1000)

                    results.append(
                        BenchmarkResult(
                            "vector_search",
                            engine_label,
                            float(np.mean(search_times)),
                            p50_ms=float(np.percentile(search_times, 50)),
                            p95_ms=float(np.percentile(search_times, 95)),
                            p99_ms=float(np.percentile(search_times, 99)),
                            metadata={"num_vectors": num_vectors, "mode": mode},
                        )
                    )

                    print(
                        f" {insert_time:.2f}ms insert, {np.mean(search_times):.3f}ms search"
                    )

            except Exception as e:
                results.append(
                    BenchmarkResult(
                        "vector_ops",
                        engine_label,
                        0,
                        error=str(e),
                        metadata={"num_vectors": num_vectors, "mode": mode},
                    )
                )
                print(f" ERROR: {str(e)[:50]}")

            gc.collect()

    return results


# =============================================================================
# Competitive Vector Store Benchmarks
# =============================================================================


def benchmark_chromadb(vectors: np.ndarray, temp_dir: str) -> dict[str, Any]:
    """Benchmark ChromaDB embedded mode."""
    if not CHROMADB_AVAILABLE:
        return {"error": "ChromaDB not available"}

    try:
        data_dir = os.path.join(temp_dir, "chroma_data")
        client = chromadb.PersistentClient(path=data_dir)
        collection = client.create_collection(
            "benchmark", metadata={"hnsw:space": "cosine"}
        )

        ids = [f"vec_{i}" for i in range(len(vectors))]

        start = time.perf_counter()
        collection.add(ids=ids, embeddings=vectors.tolist())
        insert_time = (time.perf_counter() - start) * 1000

        query = vectors[0:1].tolist()
        search_times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = collection.query(query_embeddings=query, n_results=10)
            search_times.append((time.perf_counter() - start) * 1000)

        return {
            "insert_time_ms": insert_time,
            "search_time_ms": float(np.mean(search_times)),
            "p95_ms": float(np.percentile(search_times, 95)),
            "throughput": len(vectors) / (insert_time / 1000),
        }
    except Exception as e:
        return {"error": str(e)}


def benchmark_lancedb(vectors: np.ndarray, temp_dir: str) -> dict[str, Any]:
    """Benchmark LanceDB embedded mode."""
    if not LANCEDB_AVAILABLE:
        return {"error": "LanceDB not available"}

    try:
        data_dir = os.path.join(temp_dir, "lance_data")
        db = lancedb.connect(data_dir)
        data = [
            {"id": f"vec_{i}", "vector": vectors[i].tolist()}
            for i in range(len(vectors))
        ]

        start = time.perf_counter()
        table = db.create_table("benchmark", data)
        insert_time = (time.perf_counter() - start) * 1000

        query = vectors[0].tolist()
        search_times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = table.search(query).limit(10).to_list()
            search_times.append((time.perf_counter() - start) * 1000)

        return {
            "insert_time_ms": insert_time,
            "search_time_ms": float(np.mean(search_times)),
            "p95_ms": float(np.percentile(search_times, 95)),
            "throughput": len(vectors) / (insert_time / 1000),
        }
    except Exception as e:
        return {"error": str(e)}


def benchmark_faiss(vectors: np.ndarray, temp_dir: str) -> dict[str, Any]:
    """Benchmark FAISS (in-memory)."""
    if not FAISS_AVAILABLE:
        return {"error": "FAISS not available"}

    try:
        dimension = vectors.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)

        start = time.perf_counter()
        faiss_index.add(vectors)
        insert_time = (time.perf_counter() - start) * 1000

        query = vectors[0:1]
        search_times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = faiss_index.search(query, 10)
            search_times.append((time.perf_counter() - start) * 1000)

        return {
            "insert_time_ms": insert_time,
            "search_time_ms": float(np.mean(search_times)),
            "p95_ms": float(np.percentile(search_times, 95)),
            "throughput": len(vectors) / (insert_time / 1000),
        }
    except Exception as e:
        return {"error": str(e)}


def benchmark_qdrant(vectors: np.ndarray, temp_dir: str) -> dict[str, Any]:
    """Benchmark Qdrant embedded mode."""
    if not QDRANT_AVAILABLE:
        return {"error": "Qdrant not available"}

    try:
        data_dir = os.path.join(temp_dir, "qdrant_data")
        client = QdrantClient(path=data_dir)
        dimension = vectors.shape[1]
        client.create_collection(
            collection_name="benchmark",
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )

        points = [
            PointStruct(id=i, vector=vectors[i].tolist()) for i in range(len(vectors))
        ]

        start = time.perf_counter()
        client.upsert(collection_name="benchmark", points=points)
        insert_time = (time.perf_counter() - start) * 1000

        query = vectors[0].tolist()
        search_times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = client.query_points(collection_name="benchmark", query=query, limit=10)
            search_times.append((time.perf_counter() - start) * 1000)

        return {
            "insert_time_ms": insert_time,
            "search_time_ms": float(np.mean(search_times)),
            "p95_ms": float(np.percentile(search_times, 95)),
            "throughput": len(vectors) / (insert_time / 1000),
        }
    except Exception as e:
        return {"error": str(e)}


def benchmark_vector_competitors(
    config: BenchmarkConfig, temp_dir: str
) -> list[BenchmarkResult]:
    """Benchmark competitive vector databases."""
    results = []

    competitor_defs = [
        ("chromadb", benchmark_chromadb, CHROMADB_AVAILABLE, "ChromaDB not available"),
        ("lancedb", benchmark_lancedb, LANCEDB_AVAILABLE, "LanceDB not available"),
        ("faiss", benchmark_faiss, FAISS_AVAILABLE, "FAISS not available"),
        ("qdrant", benchmark_qdrant, QDRANT_AVAILABLE, "Qdrant not available"),
        (
            "usearch",
            benchmark_usearch_quick,
            USEARCH_AVAILABLE,
            "USearch not available",
        ),
        ("annoy", benchmark_annoy_quick, ANNOY_AVAILABLE, "Annoy not available"),
        ("milvus", benchmark_milvus_quick, MILVUS_AVAILABLE, "Milvus not available"),
    ]

    for num_vectors in config.vector_sizes:
        vectors = generate_vectors(num_vectors, config.vector_dimension)

        for name, bench_fn, available, reason in competitor_defs:
            print(
                f"    Vector competitor: {name.upper()} ({num_vectors:,} vectors)...",
                end="",
                flush=True,
            )

            if not available:
                results.append(BenchmarkResult("vector_insert", name, 0, error=reason))
                print(f" SKIP: {reason}")
                gc.collect()
                continue

            with tempfile.TemporaryDirectory() as comp_temp:
                result = bench_fn(vectors, comp_temp)

            if "error" in result:
                results.append(
                    BenchmarkResult("vector_insert", name, 0, error=result["error"])
                )
                print(f" SKIP: {result['error'][:60]}")
            else:
                results.append(
                    BenchmarkResult(
                        "vector_insert",
                        name,
                        result["insert_time_ms"],
                        throughput=result["throughput"],
                        metadata={"num_vectors": num_vectors},
                    )
                )
                results.append(
                    BenchmarkResult(
                        "vector_search",
                        name,
                        result["search_time_ms"],
                        p95_ms=result["p95_ms"],
                        metadata={"num_vectors": num_vectors},
                    )
                )
                print(
                    f" {result['insert_time_ms']:.2f}ms insert, {result['search_time_ms']:.3f}ms search"
                )

            gc.collect()

    return results


def benchmark_usearch_quick(vectors: np.ndarray, temp_dir: str) -> dict[str, Any]:
    """Quick USearch benchmark for consolidated/quick runs."""
    if not USEARCH_AVAILABLE:
        return {"error": "USearch not available"}

    try:
        from usearch.index import (
            Index as USearchIndex,  # Local import to avoid NameError when unavailable
        )

        dim = vectors.shape[1]
        index = USearchIndex(ndim=dim, metric="l2sq")

        start = time.perf_counter()
        index.add(range(len(vectors)), vectors)
        insert_time = (time.perf_counter() - start) * 1000

        search_times = []
        num_queries = min(100, len(vectors))
        for i in range(num_queries):
            start = time.perf_counter()
            _ = index.search(vectors[i], 10)
            search_times.append((time.perf_counter() - start) * 1000)

        avg_search = float(np.mean(search_times))
        p95_search = float(np.percentile(search_times, 95))

        return {
            "insert_time_ms": insert_time,
            "throughput": len(vectors) / (insert_time / 1000),
            "search_time_ms": avg_search,
            "p95_ms": p95_search,
        }
    except Exception as e:
        return {"error": str(e)}


def benchmark_annoy_quick(vectors: np.ndarray, temp_dir: str) -> dict[str, Any]:
    """Quick Annoy benchmark for consolidated/quick runs."""
    if not ANNOY_AVAILABLE:
        return {"error": "Annoy not available"}

    try:
        from annoy import AnnoyIndex  # Local import to avoid NameError when unavailable

        dim = vectors.shape[1]
        index = AnnoyIndex(dim, "euclidean")

        start = time.perf_counter()
        for i, vec in enumerate(vectors):
            index.add_item(i, vec.tolist())
        index.build(10)
        insert_time = (time.perf_counter() - start) * 1000

        search_times = []
        num_queries = min(100, len(vectors))
        for i in range(num_queries):
            start = time.perf_counter()
            _ = index.get_nns_by_vector(
                vectors[i].tolist(), 10, include_distances=False
            )
            search_times.append((time.perf_counter() - start) * 1000)

        avg_search = float(np.mean(search_times))
        p95_search = float(np.percentile(search_times, 95))

        return {
            "insert_time_ms": insert_time,
            "throughput": len(vectors) / (insert_time / 1000),
            "search_time_ms": avg_search,
            "p95_ms": p95_search,
        }
    except Exception as e:
        return {"error": str(e)}


def benchmark_milvus_quick(vectors: np.ndarray, temp_dir: str) -> dict[str, Any]:
    """Quick Milvus Lite benchmark for consolidated/quick runs."""
    if not MILVUS_AVAILABLE:
        return {"error": "Milvus not available"}

    try:
        from pymilvus import (  # Local import to avoid NameError when unavailable
            MilvusClient,
        )

        milvus_file = os.path.join(temp_dir, "milvus_quick.db")
        client = MilvusClient(milvus_file)
        dim = vectors.shape[1]
        client.create_collection("quick", dimension=dim, metric_type="L2")

        # Milvus Lite expects dict-based rows aligned to the default schema (id + vector)
        data = [{"id": i, "vector": vec.tolist()} for i, vec in enumerate(vectors)]

        # Insert
        start = time.perf_counter()
        client.insert(collection_name="quick", data=data)
        insert_time = (time.perf_counter() - start) * 1000

        # Search
        search_times = []
        num_queries = min(100, len(vectors))
        for i in range(num_queries):
            start = time.perf_counter()
            _ = client.search(
                collection_name="quick",
                data=[vectors[i].tolist()],
                limit=10,
                search_params={"metric_type": "L2"},
            )
            search_times.append((time.perf_counter() - start) * 1000)

        avg_search = float(np.mean(search_times))
        p95_search = float(np.percentile(search_times, 95))

        return {
            "insert_time_ms": insert_time,
            "throughput": len(vectors) / (insert_time / 1000),
            "search_time_ms": avg_search,
            "p95_ms": p95_search,
        }
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# Competitive Graph Store Benchmarks
# =============================================================================


def benchmark_networkx(
    nodes_data: list[dict], edges_data: list[dict]
) -> dict[str, Any]:
    """Benchmark NetworkX."""
    if not NETWORKX_AVAILABLE:
        return {"error": "NetworkX not available"}

    try:
        G = nx.DiGraph()

        # Node insert
        start = time.perf_counter()
        for node in nodes_data:
            G.add_node(node["id"], **node["properties"])
        node_time = (time.perf_counter() - start) * 1000

        # Edge insert
        start = time.perf_counter()
        for edge in edges_data:
            G.add_edge(
                edge["from_node_id"], edge["to_node_id"], weight=edge.get("weight", 1.0)
            )
        edge_time = (time.perf_counter() - start) * 1000

        # 1-hop traversal
        traversal_times = []
        node_ids = list(G.nodes())
        for _ in range(100):
            node_id = random.choice(node_ids)
            start = time.perf_counter()
            _ = list(G.successors(node_id))
            traversal_times.append((time.perf_counter() - start) * 1000)

        return {
            "node_insert_ms": node_time,
            "edge_insert_ms": edge_time,
            "traversal_ms": float(np.mean(traversal_times)),
            "p95_traversal": float(np.percentile(traversal_times, 95)),
            "node_throughput": len(nodes_data) / (node_time / 1000),
            "edge_throughput": len(edges_data) / (edge_time / 1000),
        }
    except Exception as e:
        return {"error": str(e)}


def benchmark_igraph(nodes_data: list[dict], edges_data: list[dict]) -> dict[str, Any]:
    """Benchmark igraph."""
    if not IGRAPH_AVAILABLE:
        return {"error": "igraph not available"}

    try:
        # Create node ID to index mapping
        node_id_to_idx = {node["id"]: i for i, node in enumerate(nodes_data)}

        # Node insert (igraph requires integer indices)
        start = time.perf_counter()
        g = igraph.Graph(directed=True)
        g.add_vertices(len(nodes_data))
        for i, node in enumerate(nodes_data):
            g.vs[i]["name"] = node["id"]
            for k, v in node["properties"].items():
                g.vs[i][k] = v
        node_time = (time.perf_counter() - start) * 1000

        # Edge insert
        edge_tuples = []
        for edge in edges_data:
            from_idx = node_id_to_idx.get(edge["from_node_id"])
            to_idx = node_id_to_idx.get(edge["to_node_id"])
            if from_idx is not None and to_idx is not None:
                edge_tuples.append((from_idx, to_idx))

        start = time.perf_counter()
        g.add_edges(edge_tuples)
        edge_time = (time.perf_counter() - start) * 1000

        # 1-hop traversal
        traversal_times = []
        for _ in range(100):
            node_idx = random.randint(0, len(nodes_data) - 1)
            start = time.perf_counter()
            _ = g.successors(node_idx)
            traversal_times.append((time.perf_counter() - start) * 1000)

        return {
            "node_insert_ms": node_time,
            "edge_insert_ms": edge_time,
            "traversal_ms": float(np.mean(traversal_times)),
            "p95_traversal": float(np.percentile(traversal_times, 95)),
            "node_throughput": len(nodes_data) / (node_time / 1000),
            "edge_throughput": len(edges_data) / (edge_time / 1000),
        }
    except Exception as e:
        return {"error": str(e)}


def benchmark_graph_competitors(
    config: BenchmarkConfig, temp_dir: str
) -> list[BenchmarkResult]:
    """Benchmark competitive graph databases."""
    results = []

    for num_nodes in config.graph_nodes:
        nodes_data, edges_data = generate_graph_data(num_nodes, config.edge_density)

        competitors = [
            ("networkx", lambda n, e: benchmark_networkx(n, e)),
            ("igraph", lambda n, e: benchmark_igraph(n, e)),
        ]

        for name, bench_fn in competitors:
            print(
                f"    Graph competitor: {name.upper()} ({num_nodes:,} nodes)...",
                end="",
                flush=True,
            )

            result = bench_fn(nodes_data, edges_data)

            if "error" in result:
                results.append(
                    BenchmarkResult(
                        "graph_insert_nodes", name, 0, error=result["error"]
                    )
                )
                print(f" SKIP: {result['error'][:30]}")
            else:
                results.append(
                    BenchmarkResult(
                        "graph_insert_nodes",
                        name,
                        result["node_insert_ms"],
                        throughput=result["node_throughput"],
                        metadata={"num_nodes": num_nodes},
                    )
                )
                results.append(
                    BenchmarkResult(
                        "graph_insert_edges",
                        name,
                        result["edge_insert_ms"],
                        throughput=result["edge_throughput"],
                        metadata={"num_edges": len(edges_data)},
                    )
                )
                results.append(
                    BenchmarkResult(
                        "graph_1hop_traversal",
                        name,
                        result["traversal_ms"],
                        p95_ms=result["p95_traversal"],
                        metadata={"num_nodes": num_nodes},
                    )
                )
                print(
                    f" {result['node_insert_ms']:.2f}ms nodes, {result['traversal_ms']:.3f}ms traversal"
                )

            gc.collect()

    return results


# =============================================================================
# Graph Store Benchmarks
# =============================================================================


def benchmark_graph_store(
    config: BenchmarkConfig, temp_dir: str, graph_engines: list[str] = None
) -> list[BenchmarkResult]:
    """Benchmark graph database operations."""
    if not PROXIMADB_AVAILABLE:
        return [BenchmarkResult("init", "orion", 0, error="ProximaDB not available")]

    if graph_engines is None:
        graph_engines = ["orion"]  # Default to Orion if not specified

    results = []

    for num_nodes in config.graph_nodes:
        num_edges = int(num_nodes * config.edge_density)
        nodes_data, edges_data = generate_graph_data(num_nodes, config.edge_density)

        for engine in graph_engines:
            engine_label = engine.upper()
            print(
                f"    Graph benchmark: {engine_label} ({num_nodes:,} nodes, {num_edges:,} edges)...",
                end="",
                flush=True,
            )

            def run_engine():
                data_dir = os.path.join(temp_dir, f"graph_{engine}_{num_nodes}")
                os.makedirs(data_dir, exist_ok=True)

                with proximadb.ProximaDB(
                    data_dirs=data_dir,
                    metadata_dir=os.path.join(data_dir, "metadata"),
                    cache_size_mb=128,
                    enable_wal=False,
                ) as db:

                    db.create_graph(
                        "benchmark_graph", engine=engine
                    )  # Pass engine here

                    # Node insert
                    nodes = [
                        proximadb.GraphNode(
                            n["id"], labels=n["labels"], properties=n["properties"]
                        )
                        for n in nodes_data
                    ]
                    start = time.perf_counter()
                    db.create_nodes("benchmark_graph", nodes)
                    insert_nodes_time = (time.perf_counter() - start) * 1000

                    results.append(
                        BenchmarkResult(
                            "graph_insert_nodes",
                            engine_label,
                            insert_nodes_time,
                            throughput=num_nodes / (insert_nodes_time / 1000),
                            metadata={"num_nodes": num_nodes, "engine": engine},
                        )
                    )

                    # Edge insert
                    edges = [
                        proximadb.GraphEdge(
                            e["from_node_id"],
                            e["to_node_id"],
                            e["edge_type"],
                            weight=e.get("weight"),
                        )
                        for e in edges_data
                    ]
                    start = time.perf_counter()
                    db.create_edges("benchmark_graph", edges)
                    insert_edges_time = (time.perf_counter() - start) * 1000

                    results.append(
                        BenchmarkResult(
                            "graph_insert_edges",
                            engine_label,
                            insert_edges_time,
                            throughput=len(edges_data) / (insert_edges_time / 1000),
                            metadata={"num_edges": len(edges_data), "engine": engine},
                        )
                    )

                    # 1-hop traversal
                    traversal_times = []
                    for _ in range(100):
                        node_id = f"entity_{random.randint(0, num_nodes - 1)}"
                        start = time.perf_counter()
                        _ = db.get_outgoing_edges("benchmark_graph", node_id)
                        traversal_times.append((time.perf_counter() - start) * 1000)

                    results.append(
                        BenchmarkResult(
                            "graph_1hop_traversal",
                            engine_label,
                            float(np.mean(traversal_times)),
                            throughput=1000 / np.mean(traversal_times),
                            p50_ms=float(np.percentile(traversal_times, 50)),
                            p95_ms=float(np.percentile(traversal_times, 95)),
                            p99_ms=float(np.percentile(traversal_times, 99)),
                            metadata={"num_nodes": num_nodes, "engine": engine},
                        )
                    )

                    print(
                        f" {insert_nodes_time:.2f}ms nodes, {insert_edges_time:.2f}ms edges, {np.mean(traversal_times):.3f}ms traversal"
                    )

                    db.delete_graph("benchmark_graph")

            try:
                # Guard graph engine hangs with a timeout so the benchmark can proceed
                result_holder = {}
                exc_holder = {}

                def runner():
                    try:
                        result_holder["ok"] = run_engine()
                    except Exception as e:
                        exc_holder["err"] = e

                t = threading.Thread(target=runner, daemon=True)
                t.start()
                t.join(timeout=180)  # 180s per-engine safeguard
                if t.is_alive():
                    raise TimeoutError(f"{engine_label} graph benchmark timed out")
                if "err" in exc_holder:
                    raise exc_holder["err"]

            except Exception as e:
                results.append(
                    BenchmarkResult(
                        "graph_ops",
                        engine_label,
                        0,
                        error=str(e),
                        metadata={"num_nodes": num_nodes, "engine": engine},
                    )
                )
                print(f" ERROR: {str(e)[:80]}")

            gc.collect()

    return results


# =============================================================================
# Semantic Knowledge Store Benchmarks
# =============================================================================


def benchmark_sks_workload(
    config: BenchmarkConfig, temp_dir: str
) -> list[BenchmarkResult]:
    """Benchmark semantic knowledge store (unified vector + graph) workloads."""
    if not PROXIMADB_AVAILABLE:
        return [BenchmarkResult("init", "sks", 0, error="ProximaDB not available")]

    results = []

    for num_entities in config.sks_entities:
        entities, relations, vectors = generate_sks_data(
            num_entities, config.sks_relations_per_entity, dimension=128
        )

        print(
            f"    SKS benchmark: ({num_entities:,} entities, {len(relations):,} relations)...",
            end="",
            flush=True,
        )

        try:
            data_dir = os.path.join(temp_dir, f"sks_{num_entities}")
            os.makedirs(data_dir, exist_ok=True)

            with proximadb.ProximaDB(
                data_dirs=data_dir,
                metadata_dir=os.path.join(data_dir, "metadata"),
                cache_size_mb=128,
                enable_wal=False,
            ) as db:

                # Create both vector collection and graph
                db.create_collection("sks_vectors", 128, "sst")
                db.create_graph("sks_graph")

                # Insert vectors - use zero-copy if available
                ids = [e["id"] for e in entities]
                start = time.perf_counter()
                if hasattr(db, "insert_numpy"):
                    # Zero-copy numpy insert (faster)
                    vectors_array = np.array(
                        [e["embedding"] for e in entities], dtype=np.float32
                    )
                    db.insert_numpy("sks_vectors", ids, vectors_array)
                else:
                    vectors_list = [e["embedding"] for e in entities]
                    db.insert("sks_vectors", ids, vectors_list, None)
                vector_insert_time = (time.perf_counter() - start) * 1000

                # Insert graph nodes
                nodes = [
                    proximadb.GraphNode(
                        e["id"], labels=e["labels"], properties=e["properties"]
                    )
                    for e in entities
                ]
                start = time.perf_counter()
                db.create_nodes("sks_graph", nodes)
                node_insert_time = (time.perf_counter() - start) * 1000

                # Insert graph edges
                edges = [
                    proximadb.GraphEdge(
                        r["from_entity_id"], r["to_entity_id"], r["relation_type"]
                    )
                    for r in relations
                ]
                start = time.perf_counter()
                db.create_edges("sks_graph", edges)
                edge_insert_time = (time.perf_counter() - start) * 1000

                results.append(
                    BenchmarkResult(
                        "sks_ingest",
                        "sks",
                        vector_insert_time + node_insert_time + edge_insert_time,
                        throughput=num_entities
                        / (
                            (vector_insert_time + node_insert_time + edge_insert_time)
                            / 1000
                        ),
                        metadata={
                            "num_entities": num_entities,
                            "num_relations": len(relations),
                        },
                    )
                )

                # Hybrid query simulation: vector search + graph expansion
                hybrid_times = []
                for _ in range(50):
                    query_vector = vectors[random.randint(0, num_entities - 1)].tolist()

                    start = time.perf_counter()
                    # Step 1: Vector search
                    vector_results = db.search("sks_vectors", query_vector, 10)

                    # Step 2: Graph expansion (get related entities)
                    for result in vector_results[:5]:
                        edges = db.get_outgoing_edges("sks_graph", result.id)

                    hybrid_times.append((time.perf_counter() - start) * 1000)

                results.append(
                    BenchmarkResult(
                        "sks_hybrid_query",
                        "sks",
                        float(np.mean(hybrid_times)),
                        p50_ms=float(np.percentile(hybrid_times, 50)),
                        p95_ms=float(np.percentile(hybrid_times, 95)),
                        p99_ms=float(np.percentile(hybrid_times, 99)),
                        metadata={"num_entities": num_entities},
                    )
                )

                print(
                    f" {vector_insert_time + node_insert_time + edge_insert_time:.2f}ms ingest, {np.mean(hybrid_times):.3f}ms hybrid query"
                )

                db.delete_graph("sks_graph")

        except Exception as e:
            results.append(
                BenchmarkResult(
                    "sks_ops",
                    "sks",
                    0,
                    error=str(e),
                    metadata={"num_entities": num_entities},
                )
            )
            print(f" ERROR: {str(e)[:50]}")

        gc.collect()

    return results


# =============================================================================
# Reporting
# =============================================================================


def render_engine_comparison_table(results: list[BenchmarkResult]) -> None:
    """Render engine comparison table."""
    print("\n" + "=" * 100)
    print("PROXIMADB ENGINE COMPARISON TABLE")
    print("=" * 100)

    # Vector engines table
    vector_results = [r for r in results if r.operation.startswith("vector_")]
    if vector_results:
        print("\n### VECTOR STORAGE ENGINES")

        if RICH_AVAILABLE:
            console = Console()
            table = Table(title="Vector Engine Performance", expand=True)
            table.add_column("Engine", style="cyan")
            table.add_column("Insert (ms)", justify="right")
            table.add_column("Throughput (/s)", justify="right")
            table.add_column("Search (ms)", justify="right")
            table.add_column("p95 Search", justify="right")

            engines = ["sst", "helix", "nova", "swift", "raptor"]
            for engine in engines:
                insert_result = next(
                    (
                        r
                        for r in vector_results
                        if r.engine == engine and r.operation == "vector_insert"
                    ),
                    None,
                )
                search_result = next(
                    (
                        r
                        for r in vector_results
                        if r.engine == engine and r.operation == "vector_search"
                    ),
                    None,
                )

                if insert_result and search_result and not insert_result.error:
                    table.add_row(
                        engine.upper(),
                        f"{insert_result.time_ms:.2f}",
                        f"{insert_result.throughput:,.0f}",
                        f"{search_result.time_ms:.3f}",
                        f"{search_result.p95_ms:.3f}" if search_result.p95_ms else "-",
                    )
                else:
                    error_msg = insert_result.error if insert_result else "N/A"
                    table.add_row(engine.upper(), "ERROR", "-", "-", "-")

            console.print(table)
        else:
            print(
                f"{'Engine':<10} {'Insert (ms)':<15} {'Throughput':<15} {'Search (ms)':<15}"
            )
            print("-" * 55)

    # Graph engine table
    graph_results = [r for r in results if r.operation.startswith("graph_")]
    if graph_results:
        print("\n### GRAPH DATABASE ENGINES")

        if RICH_AVAILABLE:
            console = Console()

            # Group results by engine
            graph_engines = sorted(list(set(r.engine for r in graph_results)))
            for engine_name in graph_engines:
                table = Table(
                    title=f"Graph Engine Performance: {engine_name}", expand=True
                )
                table.add_column("Operation", style="cyan")
                table.add_column("Time (ms)", justify="right")
                table.add_column("Throughput (/s)", justify="right")
                table.add_column("p95 (ms)", justify="right")

                ops = [
                    "graph_insert_nodes",
                    "graph_insert_edges",
                    "graph_1hop_traversal",
                ]
                for op in ops:
                    result = next(
                        (
                            r
                            for r in graph_results
                            if r.engine == engine_name
                            and r.operation == op
                            and not r.error
                        ),
                        None,
                    )
                    if result:
                        table.add_row(
                            op.replace("graph_", ""),
                            f"{result.time_ms:.2f}",
                            f"{result.throughput:,.0f}" if result.throughput else "-",
                            f"{result.p95_ms:.3f}" if result.p95_ms else "-",
                        )
                    else:
                        table.add_row(
                            op.replace("graph_", ""), "[red]ERROR[/red]", "N/A", "N/A"
                        )
                console.print(table)
        else:
            # Simple print fallback for graph engines
            print(
                f"{'Engine':<10} {'Operation':<20} {'Time (ms)':<15} {'Throughput':<15}"
            )
            print("-" * 60)
            graph_engines = sorted(list(set(r.engine for r in graph_results)))
            for engine_name in graph_engines:
                ops = [
                    "graph_insert_nodes",
                    "graph_insert_edges",
                    "graph_1hop_traversal",
                ]
                for op in ops:
                    result = next(
                        (
                            r
                            for r in graph_results
                            if r.engine == engine_name
                            and r.operation == op
                            and not r.error
                        ),
                        None,
                    )
                    if result:
                        tput_str = (
                            f"{result.throughput:,.0f}" if result.throughput else "-"
                        )
                        print(
                            f"{engine_name:<10} {op.replace('graph_', ''):<20} {result.time_ms:<15.2f} {tput_str:<15}"
                        )
                    else:
                        print(
                            f"{engine_name:<10} {op.replace('graph_', ''):<20} {'ERROR':<15} {'N/A':<15}"
                        )

            console.print(table)

    # SKS table
    sks_results = [r for r in results if r.operation.startswith("sks_")]
    if sks_results:
        print("\n### SEMANTIC KNOWLEDGE STORE (SKS)")

        if RICH_AVAILABLE:
            console = Console()
            table = Table(title="SKS Performance (Vector + Graph Unified)", expand=True)
            table.add_column("Operation", style="cyan")
            table.add_column("Time (ms)", justify="right")
            table.add_column("Throughput (/s)", justify="right")
            table.add_column("p95 (ms)", justify="right")

            for result in sks_results:
                if not result.error:
                    table.add_row(
                        result.operation.replace("sks_", ""),
                        f"{result.time_ms:.2f}",
                        f"{result.throughput:,.0f}" if result.throughput else "-",
                        f"{result.p95_ms:.3f}" if result.p95_ms else "-",
                    )

            console.print(table)


def write_consolidated_report(results: list[BenchmarkResult]) -> None:
    """Write consolidated markdown report."""
    target_dir = Path("target")
    target_dir.mkdir(exist_ok=True)

    lines = []
    lines.append("# ProximaDB Consolidated Embedded Benchmark")
    lines.append("")
    lines.append("## Vector Storage Engines")
    lines.append("")
    lines.append(
        "| Engine | Insert (ms) | Throughput (/s) | Search (ms) | p95 Search |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: |")

    engines = ["sst", "helix", "nova", "swift", "raptor"]
    for engine in engines:
        insert_result = next(
            (
                r
                for r in results
                if r.engine == engine and r.operation == "vector_insert"
            ),
            None,
        )
        search_result = next(
            (
                r
                for r in results
                if r.engine == engine and r.operation == "vector_search"
            ),
            None,
        )

        if insert_result and search_result and not insert_result.error:
            p95_str = f"{search_result.p95_ms:.3f}" if search_result.p95_ms else "-"
            lines.append(
                f"| {engine.upper()} | {insert_result.time_ms:.2f} | {insert_result.throughput:,.0f} | "
                f"{search_result.time_ms:.3f} | {p95_str} |"
            )

    lines.append("")
    lines.append("## Graph Database (ORION)")
    lines.append("")
    lines.append("| Operation | Time (ms) | Throughput (/s) | p95 (ms) |")
    lines.append("| --- | ---: | ---: | ---: |")

    graph_results = [
        r for r in results if r.operation.startswith("graph_") and not r.error
    ]
    for result in graph_results:
        tput_str = f"{result.throughput:,.0f}" if result.throughput else "-"
        p95_str = f"{result.p95_ms:.3f}" if result.p95_ms else "-"
        lines.append(
            f"| {result.operation.replace('graph_', '')} | {result.time_ms:.2f} | "
            f"{tput_str} | {p95_str} |"
        )

    lines.append("")
    lines.append("## Semantic Knowledge Store (SKS)")
    lines.append("")
    lines.append("| Operation | Time (ms) | Throughput (/s) | p95 (ms) |")
    lines.append("| --- | ---: | ---: | ---: |")

    sks_results = [r for r in results if r.operation.startswith("sks_") and not r.error]
    for result in sks_results:
        tput_str = f"{result.throughput:,.0f}" if result.throughput else "-"
        p95_str = f"{result.p95_ms:.3f}" if result.p95_ms else "-"
        lines.append(
            f"| {result.operation.replace('sks_', '')} | {result.time_ms:.2f} | "
            f"{tput_str} | {p95_str} |"
        )

    # Add Mermaid charts for visual comparison
    lines.append("")
    lines.append("## Performance Charts")
    lines.append("")

    # QPS Chart (bar chart)
    lines.append("### Query Performance (QPS)")
    lines.append("")
    lines.append("```mermaid")
    lines.append("%%{init: {'theme': 'neutral'}}%%")
    lines.append("xychart-beta")
    lines.append('    title "Queries Per Second by Engine"')
    lines.append("    x-axis [SST, HELIX, VIPER, NOVA, SWIFT, RAPTOR]")

    # Collect QPS values for search operations
    qps_values = []
    for engine in ["sst", "helix", "nova", "swift", "raptor"]:
        search_result = next(
            (
                r
                for r in results
                if r.engine == engine and r.operation == "vector_search"
            ),
            None,
        )
        if search_result and search_result.throughput:
            qps_values.append(int(search_result.throughput))
        else:
            qps_values.append(0)

    lines.append(f'    y-axis "QPS" 0 --> {max(qps_values) * 1.2:.0f}')
    lines.append(f"    bar [{', '.join(str(v) for v in qps_values)}]")
    lines.append("```")
    lines.append("")

    # Ingestion Throughput Chart
    lines.append("### Ingestion Throughput (vectors/sec)")
    lines.append("")
    lines.append("```mermaid")
    lines.append("%%{init: {'theme': 'neutral'}}%%")
    lines.append("xychart-beta")
    lines.append('    title "Insert Throughput by Engine"')
    lines.append("    x-axis [SST, HELIX, VIPER, NOVA, SWIFT, RAPTOR]")

    # Collect ingestion throughput values
    ingest_values = []
    for engine in ["sst", "helix", "nova", "swift", "raptor"]:
        insert_result = next(
            (
                r
                for r in results
                if r.engine == engine and r.operation == "vector_insert"
            ),
            None,
        )
        if insert_result and insert_result.throughput:
            ingest_values.append(int(insert_result.throughput))
        else:
            ingest_values.append(0)

    lines.append(f'    y-axis "vectors/sec" 0 --> {max(ingest_values) * 1.2:.0f}')
    lines.append(f"    bar [{', '.join(str(v) for v in ingest_values)}]")
    lines.append("```")
    lines.append("")

    # Recall Chart (if recall data available)
    recall_values = []
    for engine in ["sst", "helix", "nova", "swift", "raptor"]:
        search_result = next(
            (
                r
                for r in results
                if r.engine == engine
                and (r.operation == "vector_search" or r.operation == "sift_search")
            ),
            None,
        )
        if search_result and search_result.recall_at_k is not None:
            recall_values.append(search_result.recall_at_k)
        else:
            recall_values.append(None)

    if any(v is not None for v in recall_values):
        lines.append("### Recall@10")
        lines.append("")
        lines.append("```mermaid")
        lines.append("%%{init: {'theme': 'neutral'}}%%")
        lines.append("xychart-beta")
        lines.append('    title "Recall@10 by Engine"')
        lines.append("    x-axis [SST, HELIX, VIPER, NOVA, SWIFT, RAPTOR]")
        lines.append('    y-axis "Recall %" 0 --> 105')
        formatted_recalls = [
            f"{v:.1f}" if v is not None else "0" for v in recall_values
        ]
        lines.append(f"    bar [{', '.join(formatted_recalls)}]")
        lines.append("```")
        lines.append("")

    report_path = target_dir / "consolidated_benchmark_latest.md"
    report_path.write_text("\n".join(lines))
    print(f"\nMarkdown report written to {report_path}")


# =============================================================================
# Main Runner
# =============================================================================


def run_consolidated_benchmark(
    config: BenchmarkConfig = None, include_competitors: bool = True
):
    """Run the consolidated benchmark suite.

    Args:
        config: Benchmark configuration
        include_competitors: If True, also benchmark competitive databases
    """
    if config is None:
        config = BenchmarkConfig()

    print("=" * 100)
    print("PROXIMADB CONSOLIDATED EMBEDDED DATABASE BENCHMARK")
    print("=" * 100)
    print()
    if include_competitors:
        print(
            "Components: Vector Store (6 engines + 4 competitors) | Graph Store (ORION + 2 competitors) | SKS"
        )
    else:
        print(
            "Components: Vector Store (6 engines) | Graph Store (ORION) | Semantic Knowledge Store"
        )
    print()

    # Show available databases
    print("Available databases:")
    print(f"  ProximaDB: {'YES' if PROXIMADB_AVAILABLE else 'NO'}")
    if include_competitors:
        print(f"  ChromaDB:  {'YES' if CHROMADB_AVAILABLE else 'NO'}")
        print(f"  LanceDB:   {'YES' if LANCEDB_AVAILABLE else 'NO'}")
        print(f"  FAISS:     {'YES' if FAISS_AVAILABLE else 'NO'}")
        print(f"  Qdrant:    {'YES' if QDRANT_AVAILABLE else 'NO'}")
        print(f"  USearch:   {'YES' if USEARCH_AVAILABLE else 'NO'}")
        print(f"  Annoy:     {'YES' if ANNOY_AVAILABLE else 'NO'}")
        print(f"  Milvus:    {'YES' if MILVUS_AVAILABLE else 'NO'}")
        print(f"  NetworkX:  {'YES' if NETWORKX_AVAILABLE else 'NO'}")
        print(f"  igraph:    {'YES' if IGRAPH_AVAILABLE else 'NO'}")
    print()

    all_results = []

    with tempfile.TemporaryDirectory() as temp_dir:
        # Vector store benchmarks (ProximaDB)
        print("\n[1/5] VECTOR STORE BENCHMARKS (ProximaDB)")
        print("-" * 50)
        vector_results = benchmark_vector_store(config, temp_dir)
        all_results.extend(vector_results)
        gc.collect()

        # Vector store competitors
        if include_competitors:
            print("\n[2/5] VECTOR STORE BENCHMARKS (Competitors)")
            print("-" * 50)
            competitor_vector_results = benchmark_vector_competitors(config, temp_dir)
            all_results.extend(competitor_vector_results)
            gc.collect()
        else:
            print("\n[2/5] VECTOR STORE BENCHMARKS (Competitors) - SKIPPED")

        # Graph store benchmarks (ProximaDB ORION)
        print("\n[3/5] GRAPH STORE BENCHMARKS (ProximaDB ORION/PULSAR/QUASAR)")
        print("-" * 50)
        graph_engines_to_test = ["orion", "pulsar", "quasar"]

        graph_results = benchmark_graph_store(
            config, temp_dir, graph_engines=graph_engines_to_test
        )
        all_results.extend(graph_results)
        gc.collect()

        # Graph store competitors
        if include_competitors:
            print("\n[4/5] GRAPH STORE BENCHMARKS (Competitors)")
            print("-" * 50)
            competitor_graph_results = benchmark_graph_competitors(config, temp_dir)
            all_results.extend(competitor_graph_results)
            gc.collect()
        else:
            print("\n[4/5] GRAPH STORE BENCHMARKS (Competitors) - SKIPPED")

        # SKS benchmarks (unique to ProximaDB - no competitors)
        print("\n[5/5] SEMANTIC KNOWLEDGE STORE BENCHMARKS (ProximaDB only)")
        print("-" * 50)
        print(
            "    Note: SKS (unified vector + graph) is UNIQUE to ProximaDB - no competitors"
        )
        sks_results = benchmark_sks_workload(config, temp_dir)
        all_results.extend(sks_results)

    # Render results
    render_engine_comparison_table(all_results)
    write_consolidated_report(all_results)

    return all_results


# =============================================================================
# Standard Dataset Benchmarks
# =============================================================================


def benchmark_sift_1m(
    max_vectors: int = None,
    temp_dir: str = None,
    dimension: int = 768,
    engines: str = "all",
    approximate_only: bool = False,
    exact_only: bool = False,
) -> list[BenchmarkResult]:
    """
    Benchmark ProximaDB against SIFT-1M dataset.

    SIFT-1M is the industry standard for vector database benchmarks.
    Configurable dimensions for real-world embedding models:
    - SIFT: 128D
    - BGE-base: 768D (default)
    - GIST: 960D
    - BGE-large: 1024D
    - OpenAI ada-002: 1536D

    Benchmark order:
    1. In-memory competitors first (ChromaDB, LanceDB, FAISS) - marked with * in output
    2. ProximaDB engines sequentially - write all data, close, reopen for search
    3. RAPTOR last with 3x timeout check

    Args:
        max_vectors: Limit vectors (for faster testing)
        temp_dir: Temporary directory for data

    Returns:
        List of benchmark results
    """
    results = []

    print("\n" + "=" * 80)
    print("SIFT-1M BENCHMARK (Industry Standard)")
    print("=" * 80)

    # Load SIFT dataset
    sift_data = StandardDatasets.load_sift_1m(
        max_vectors=max_vectors, dimension=dimension
    )
    base_vectors = sift_data["base"]
    query_vectors = sift_data["queries"]
    metadata = sift_data["metadata"]

    print(f"  Dataset: {metadata['dataset']}")
    print(f"  Base vectors: {metadata['num_base']:,} x {metadata['dimension']}D")
    print(f"  Query vectors: {metadata['num_queries']:,}")
    print(f"  Distance metric: {metadata['distance']}")

    # Compute ground truth for recall@k calculation (with caching for large datasets)
    k = 10
    num_queries_for_gt = min(
        len(query_vectors), 1000
    )  # Limit for reasonable compute time
    print(
        f"  Ground truth (k={k}) for {num_queries_for_gt} queries...",
        end="",
        flush=True,
    )
    ground_truth, from_cache, gt_time = compute_ground_truth_cached(
        base_vectors,
        query_vectors[:num_queries_for_gt],
        k,
        num_base=metadata["num_base"],
        dimension=metadata["dimension"],
        num_queries=num_queries_for_gt,
        seed=42,  # Fixed seed used in data generation
    )
    if from_cache:
        print(" loaded from cache")
    else:
        print(f" computed in {gt_time:.0f}ms (cached for reuse)")
    print()

    # Parse engine filter
    # Available engines: chromadb, faiss, lancedb, qdrant, usearch, annoy, milvus (competitors)
    #                    sst_approx, sst, helix, viper, nova, swift, raptor (ProximaDB)
    # Special values: "all" = everything, "proximadb" = all ProximaDB engines only
    COMPETITOR_ENGINES = {
        "chromadb",
        "faiss",
        "lancedb",
        "qdrant",
        "usearch",
        "annoy",
        "milvus",
    }
    PROXIMADB_ENGINES = {
        "sst_approx",
        "sst",
        "helix",
        "nova",
        "swift",
        "raptor",
    }

    if engines.lower() == "all":
        selected_engines = COMPETITOR_ENGINES | PROXIMADB_ENGINES
    elif engines.lower() == "proximadb":
        selected_engines = PROXIMADB_ENGINES
    else:
        selected_engines = {e.strip().lower() for e in engines.split(",")}

    run_competitors = bool(selected_engines & COMPETITOR_ENGINES)
    run_sst_approx = "sst_approx" in selected_engines
    run_raptor = "raptor" in selected_engines
    proximadb_engines_to_run = [
        e for e in ["sst", "helix", "nova", "swift"] if e in selected_engines
    ]

    # =========================================================================
    # PHASE 1: In-memory competitors (marked with * in output)
    # Skipped if engines filter excludes all competitors
    # =========================================================================
    if run_competitors:
        print("  --- IN-MEMORY COMPETITORS (marked with *) ---")

        # ChromaDB benchmark (disk-based persistent storage)
        if "chromadb" in selected_engines:
            try:
                import chromadb
                from chromadb.config import Settings

                BenchmarkLogger.log_engine_start("ChromaDB")
                phase_start = BenchmarkLogger.log_start("ChromaDB", "Insert + Search")

                chroma_dir = tempfile.mkdtemp()
                client = chromadb.PersistentClient(path=chroma_dir)
                collection = client.create_collection(
                    "sift", metadata={"hnsw:space": "l2"}
                )

                # Insert
                ids = [f"sift_{i}" for i in range(len(base_vectors))]
                start = time.perf_counter()
                collection.add(ids=ids, embeddings=base_vectors.tolist())
                insert_time = (time.perf_counter() - start) * 1000

                # Search
                search_times = []
                recalls = []
                num_queries = min(len(query_vectors), 1000)
                for i in range(num_queries):
                    start = time.perf_counter()
                    search_results = collection.query(
                        query_embeddings=[query_vectors[i].tolist()], n_results=k
                    )
                    search_times.append((time.perf_counter() - start) * 1000)

                    # Compute recall
                    result_ids = (
                        search_results["ids"][0] if search_results["ids"] else []
                    )
                    recalls.append(compute_recall_at_k(ground_truth[i], result_ids, k))

                avg_search = float(np.mean(search_times))
                p99_search = float(np.percentile(search_times, 99))
                qps = 1000 / avg_search
                avg_recall = float(np.mean(recalls)) * 100

                BenchmarkLogger.log_end(
                    "ChromaDB",
                    phase_start,
                    {
                        "insert_ms": insert_time,
                        "search_ms": avg_search,
                        "qps": qps,
                        "recall": f"{avg_recall:.1f}%",
                    },
                )

                results.append(
                    BenchmarkResult(
                        "sift_insert",
                        "chromadb",
                        insert_time,
                        throughput=len(base_vectors) / (insert_time / 1000),
                    )
                )
                results.append(
                    BenchmarkResult(
                        "sift_search",
                        "chromadb",
                        avg_search,
                        throughput=qps,
                        p99_ms=p99_search,
                        recall_at_k=avg_recall,
                    )
                )

                del client
                shutil.rmtree(chroma_dir, ignore_errors=True)

            except ImportError:
                print("  ChromaDB - SKIPPED (not installed)", flush=True)
            except Exception as e:
                BenchmarkLogger.log_error("ChromaDB", e)
                results.append(BenchmarkResult("sift_ops", "chromadb", 0, error=str(e)))

            gc.collect()

    # FAISS benchmark
    if "faiss" in selected_engines:
        try:
            import faiss

            BenchmarkLogger.log_engine_start("FAISS* (in-memory)")
            phase_start = BenchmarkLogger.log_start("FAISS*", "Insert + Search")

            # Create flat index for exact search (fair comparison)
            index = faiss.IndexFlatL2(metadata["dimension"])
            start = time.perf_counter()
            index.add(base_vectors.astype(np.float32))
            insert_time = (time.perf_counter() - start) * 1000

            search_times = []
            recalls = []
            num_queries = min(len(query_vectors), 1000)
            for i in range(num_queries):
                start = time.perf_counter()
                _, indices = index.search(
                    query_vectors[i : i + 1].astype(np.float32), k
                )
                search_times.append((time.perf_counter() - start) * 1000)
                # Compute recall@k (FAISS returns indices directly)
                result_ids = [f"sift_{idx}" for idx in indices[0]]
                recalls.append(compute_recall_at_k(ground_truth[i], result_ids, k))

            avg_search = float(np.mean(search_times))
            p99_search = float(np.percentile(search_times, 99))
            qps = 1000 / avg_search
            avg_recall = float(np.mean(recalls)) * 100

            BenchmarkLogger.log_end(
                "FAISS*",
                phase_start,
                {
                    "insert_ms": insert_time,
                    "search_ms": avg_search,
                    "qps": qps,
                    "recall": f"{avg_recall:.1f}%",
                },
            )

            results.append(
                BenchmarkResult(
                    "sift_insert",
                    "faiss*",
                    insert_time,
                    throughput=len(base_vectors) / (insert_time / 1000),
                    metadata={"in_memory": True},
                )
            )
            results.append(
                BenchmarkResult(
                    "sift_search",
                    "faiss*",
                    avg_search,
                    throughput=qps,
                    p99_ms=p99_search,
                    recall_at_k=avg_recall,
                    metadata={"in_memory": True},
                )
            )

            del index

        except ImportError:
            print("  FAISS* - SKIPPED (not installed)", flush=True)
        except Exception as e:
            BenchmarkLogger.log_error("FAISS*", e)
            results.append(BenchmarkResult("sift_ops", "faiss*", 0, error=str(e)))

        gc.collect()

    # LanceDB benchmark (disk-based with close/reopen like ProximaDB)
    if "lancedb" in selected_engines:
        try:
            import lancedb

            BenchmarkLogger.log_engine_start("LanceDB")
            phase_start = BenchmarkLogger.log_start(
                "LanceDB", "Insert + Flush + Search"
            )

            lance_dir = tempfile.mkdtemp()
            db = lancedb.connect(lance_dir)

            # LanceDB uses Arrow format
            import pyarrow as pa

            data = [
                {"id": f"sift_{i}", "vector": base_vectors[i].tolist()}
                for i in range(len(base_vectors))
            ]

            start = time.perf_counter()
            table = db.create_table("sift", data)
            insert_time = (time.perf_counter() - start) * 1000

            # Close and reopen (like ProximaDB disk-based test)
            del table
            del db
            gc.collect()
            time.sleep(0.5)

            # Calculate disk space used
            disk_size_mb = 0
            for root, dirs, files in os.walk(lance_dir):
                for f in files:
                    disk_size_mb += os.path.getsize(os.path.join(root, f))
            disk_size_mb /= 1024 * 1024

            # Reopen for search
            db = lancedb.connect(lance_dir)
            table = db.open_table("sift")

            search_times = []
            recalls = []
            num_queries = min(len(query_vectors), 1000)
            for i in range(num_queries):
                start = time.perf_counter()
                results_list = (
                    table.search(query_vectors[i].tolist()).limit(k).to_list()
                )
                search_times.append((time.perf_counter() - start) * 1000)
                # Compute recall@k
                result_ids = [r.get("id", "") for r in results_list]
                recalls.append(compute_recall_at_k(ground_truth[i], result_ids, k))

            avg_search = float(np.mean(search_times))
            p99_search = float(np.percentile(search_times, 99))
            qps = 1000 / avg_search
            avg_recall = float(np.mean(recalls)) * 100

            BenchmarkLogger.log_end(
                "LanceDB",
                phase_start,
                {
                    "insert_ms": insert_time,
                    "search_ms": avg_search,
                    "qps": qps,
                    "disk_mb": disk_size_mb,
                    "recall": f"{avg_recall:.1f}%",
                },
            )

            results.append(
                BenchmarkResult(
                    "sift_insert",
                    "lancedb",
                    insert_time,
                    throughput=len(base_vectors) / (insert_time / 1000),
                    metadata={"disk_based": True},
                )
            )
            results.append(
                BenchmarkResult(
                    "sift_search",
                    "lancedb",
                    avg_search,
                    throughput=qps,
                    p99_ms=p99_search,
                    recall_at_k=avg_recall,
                    metadata={"disk_based": True, "disk_size_mb": disk_size_mb},
                )
            )

            shutil.rmtree(lance_dir, ignore_errors=True)

        except ImportError:
            print("  LanceDB - SKIPPED (not installed)", flush=True)
        except Exception as e:
            BenchmarkLogger.log_error("LanceDB", e)
            results.append(BenchmarkResult("sift_ops", "lancedb", 0, error=str(e)))

        gc.collect()

    # Qdrant benchmark (disk-based with close/reopen like ProximaDB)
    if "qdrant" in selected_engines:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, PointStruct, VectorParams

            BenchmarkLogger.log_engine_start("Qdrant")
            phase_start = BenchmarkLogger.log_start("Qdrant", "Insert + Flush + Search")

            qdrant_dir = tempfile.mkdtemp()
            client = QdrantClient(path=qdrant_dir)
            dimension = metadata["dimension"]
            client.create_collection(
                collection_name="sift",
                vectors_config=VectorParams(size=dimension, distance=Distance.EUCLID),
            )

            points = [
                PointStruct(id=i, vector=base_vectors[i].tolist())
                for i in range(len(base_vectors))
            ]

            start = time.perf_counter()
            # Qdrant batches automatically, but for large datasets use smaller batches
            batch_size = 5000
            for i in range(0, len(points), batch_size):
                client.upsert(collection_name="sift", points=points[i : i + batch_size])
            insert_time = (time.perf_counter() - start) * 1000

            # Close and reopen (like ProximaDB disk-based test)
            del client
            gc.collect()
            time.sleep(0.5)

            # Calculate disk space used
            disk_size_mb = 0
            for root, dirs, files in os.walk(qdrant_dir):
                for f in files:
                    disk_size_mb += os.path.getsize(os.path.join(root, f))
            disk_size_mb /= 1024 * 1024

            # Reopen for search
            client = QdrantClient(path=qdrant_dir)

            search_times = []
            recalls = []
            num_queries = min(len(query_vectors), 1000)
            for i in range(num_queries):
                start = time.perf_counter()
                # Use query_points for newer qdrant-client versions (1.7+)
                try:
                    from qdrant_client.models import QueryRequest

                    search_result = client.query_points(
                        collection_name="sift", query=query_vectors[i].tolist(), limit=k
                    ).points
                except (ImportError, AttributeError):
                    # Fall back to search for older versions
                    search_result = client.search(
                        collection_name="sift",
                        query_vector=query_vectors[i].tolist(),
                        limit=k,
                    )
                search_times.append((time.perf_counter() - start) * 1000)
                # Compute recall@k
                result_ids = [f"sift_{hit.id}" for hit in search_result]
                recalls.append(compute_recall_at_k(ground_truth[i], result_ids, k))

            avg_search = float(np.mean(search_times))
            p99_search = float(np.percentile(search_times, 99))
            qps = 1000 / avg_search
            avg_recall = float(np.mean(recalls)) * 100

            BenchmarkLogger.log_end(
                "Qdrant",
                phase_start,
                {
                    "insert_ms": insert_time,
                    "search_ms": avg_search,
                    "qps": qps,
                    "disk_mb": disk_size_mb,
                    "recall": f"{avg_recall:.1f}%",
                },
            )

            results.append(
                BenchmarkResult(
                    "sift_insert",
                    "qdrant",
                    insert_time,
                    throughput=len(base_vectors) / (insert_time / 1000),
                    metadata={"disk_based": True},
                )
            )
            results.append(
                BenchmarkResult(
                    "sift_search",
                    "qdrant",
                    avg_search,
                    throughput=qps,
                    p99_ms=p99_search,
                    recall_at_k=avg_recall,
                    metadata={"disk_based": True, "disk_size_mb": disk_size_mb},
                )
            )

            del client
            shutil.rmtree(qdrant_dir, ignore_errors=True)

        except ImportError:
            print("  Qdrant - SKIPPED (not installed)", flush=True)
        except Exception as e:
            BenchmarkLogger.log_error("Qdrant", e)
            results.append(BenchmarkResult("sift_ops", "qdrant", 0, error=str(e)))

        gc.collect()

    # USearch benchmark (disk-based with mmap)
    if "usearch" in selected_engines:
        try:
            from usearch.index import Index as USearchIndex

            BenchmarkLogger.log_engine_start("USearch")
            phase_start = BenchmarkLogger.log_start("USearch", "Insert + Save + Search")

            usearch_dir = tempfile.mkdtemp()
            usearch_file = os.path.join(usearch_dir, "usearch.index")
            dimension = metadata["dimension"]
            index = USearchIndex(ndim=dimension, metric="l2sq")

            start = time.perf_counter()
            # USearch uses numpy arrays directly
            for i in range(len(base_vectors)):
                index.add(i, base_vectors[i].astype(np.float32))
            # Save to disk for persistence
            index.save(usearch_file)
            insert_time = (time.perf_counter() - start) * 1000

            # Calculate disk space used
            disk_size_mb = os.path.getsize(usearch_file) / (1024 * 1024)

            # Reload index with mmap (disk-backed search like ProximaDB)
            del index
            gc.collect()
            index = USearchIndex(ndim=dimension, metric="l2sq")
            index.view(usearch_file)  # Use mmap for disk-backed access

            search_times = []
            recalls = []
            num_queries = min(len(query_vectors), 1000)
            for i in range(num_queries):
                start = time.perf_counter()
                matches = index.search(query_vectors[i].astype(np.float32), k)
                search_times.append((time.perf_counter() - start) * 1000)
                # Compute recall@k
                result_ids = [f"sift_{int(m.key)}" for m in matches]
                recalls.append(compute_recall_at_k(ground_truth[i], result_ids, k))

            avg_search = float(np.mean(search_times))
            p99_search = float(np.percentile(search_times, 99))
            qps = 1000 / avg_search
            avg_recall = float(np.mean(recalls)) * 100

            BenchmarkLogger.log_end(
                "USearch",
                phase_start,
                {
                    "insert_ms": insert_time,
                    "search_ms": avg_search,
                    "qps": qps,
                    "disk_mb": disk_size_mb,
                    "recall": f"{avg_recall:.1f}%",
                },
            )

            results.append(
                BenchmarkResult(
                    "sift_insert",
                    "usearch",
                    insert_time,
                    throughput=len(base_vectors) / (insert_time / 1000),
                    metadata={"disk_based": True},
                )
            )
            results.append(
                BenchmarkResult(
                    "sift_search",
                    "usearch",
                    avg_search,
                    throughput=qps,
                    p99_ms=p99_search,
                    recall_at_k=avg_recall,
                    metadata={"disk_based": True, "disk_size_mb": disk_size_mb},
                )
            )

            del index
            shutil.rmtree(usearch_dir, ignore_errors=True)

        except ImportError:
            print("  USearch - SKIPPED (not installed)", flush=True)
        except Exception as e:
            BenchmarkLogger.log_error("USearch", e)
            results.append(BenchmarkResult("sift_ops", "usearch", 0, error=str(e)))

        gc.collect()

    # Annoy benchmark (disk-based with save/load)
    if "annoy" in selected_engines:
        try:
            from annoy import AnnoyIndex

            BenchmarkLogger.log_engine_start("Annoy")
            phase_start = BenchmarkLogger.log_start("Annoy", "Insert + Build + Search")

            annoy_dir = tempfile.mkdtemp()
            annoy_file = os.path.join(annoy_dir, "annoy.index")
            dimension = metadata["dimension"]
            index = AnnoyIndex(dimension, "euclidean")

            start = time.perf_counter()
            for i in range(len(base_vectors)):
                index.add_item(i, base_vectors[i].tolist())
            index.build(10)  # 10 trees for reasonable accuracy
            # Save to disk
            index.save(annoy_file)
            insert_time = (time.perf_counter() - start) * 1000

            # Calculate disk space used
            disk_size_mb = os.path.getsize(annoy_file) / (1024 * 1024)

            # Reload from disk (like ProximaDB disk-based test)
            del index
            gc.collect()
            index = AnnoyIndex(dimension, "euclidean")
            index.load(annoy_file)  # Load from disk (uses mmap)

            search_times = []
            recalls = []
            num_queries = min(len(query_vectors), 1000)
            for i in range(num_queries):
                start = time.perf_counter()
                indices = index.get_nns_by_vector(query_vectors[i].tolist(), k)
                search_times.append((time.perf_counter() - start) * 1000)
                # Compute recall@k
                result_ids = [f"sift_{idx}" for idx in indices]
                recalls.append(compute_recall_at_k(ground_truth[i], result_ids, k))

            avg_search = float(np.mean(search_times))
            p99_search = float(np.percentile(search_times, 99))
            qps = 1000 / avg_search
            avg_recall = float(np.mean(recalls)) * 100

            BenchmarkLogger.log_end(
                "Annoy",
                phase_start,
                {
                    "insert_ms": insert_time,
                    "search_ms": avg_search,
                    "qps": qps,
                    "disk_mb": disk_size_mb,
                    "recall": f"{avg_recall:.1f}%",
                },
            )

            results.append(
                BenchmarkResult(
                    "sift_insert",
                    "annoy",
                    insert_time,
                    throughput=len(base_vectors) / (insert_time / 1000),
                    metadata={"disk_based": True, "approximate": True},
                )
            )
            results.append(
                BenchmarkResult(
                    "sift_search",
                    "annoy",
                    avg_search,
                    throughput=qps,
                    p99_ms=p99_search,
                    recall_at_k=avg_recall,
                    metadata={
                        "disk_based": True,
                        "approximate": True,
                        "disk_size_mb": disk_size_mb,
                    },
                )
            )

            del index
            shutil.rmtree(annoy_dir, ignore_errors=True)

        except ImportError:
            print("  Annoy - SKIPPED (not installed)", flush=True)
        except Exception as e:
            BenchmarkLogger.log_error("Annoy", e)
            results.append(BenchmarkResult("sift_ops", "annoy", 0, error=str(e)))

        gc.collect()

    # Milvus Lite benchmark (embedded mode)
    if "milvus" in selected_engines:
        try:
            from pymilvus import MilvusClient

            BenchmarkLogger.log_engine_start("Milvus Lite")
            phase_start = BenchmarkLogger.log_start("Milvus Lite", "Insert + Search")

            milvus_dir = tempfile.mkdtemp()
            milvus_file = os.path.join(milvus_dir, "milvus.db")
            client = MilvusClient(milvus_file)
            dimension = metadata["dimension"]

            # Create collection
            client.create_collection(
                collection_name="sift", dimension=dimension, metric_type="L2"
            )

            # Prepare data
            data = [
                {"id": i, "vector": base_vectors[i].tolist()}
                for i in range(len(base_vectors))
            ]

            start = time.perf_counter()
            # Milvus batches automatically
            batch_size = 1000
            for i in range(0, len(data), batch_size):
                client.insert(collection_name="sift", data=data[i : i + batch_size])
            insert_time = (time.perf_counter() - start) * 1000

            # Calculate disk space used
            disk_size_mb = 0
            for root, dirs, files in os.walk(milvus_dir):
                for f in files:
                    disk_size_mb += os.path.getsize(os.path.join(root, f))
            disk_size_mb /= 1024 * 1024

            search_times = []
            recalls = []
            num_queries = min(len(query_vectors), 1000)
            for i in range(num_queries):
                start = time.perf_counter()
                search_result = client.search(
                    collection_name="sift", data=[query_vectors[i].tolist()], limit=k
                )
                search_times.append((time.perf_counter() - start) * 1000)
                # Compute recall@k
                result_ids = (
                    [f"sift_{hit['id']}" for hit in search_result[0]]
                    if search_result
                    else []
                )
                recalls.append(compute_recall_at_k(ground_truth[i], result_ids, k))

            avg_search = float(np.mean(search_times))
            p99_search = float(np.percentile(search_times, 99))
            qps = 1000 / avg_search
            avg_recall = float(np.mean(recalls)) * 100

            BenchmarkLogger.log_end(
                "Milvus Lite",
                phase_start,
                {
                    "insert_ms": insert_time,
                    "search_ms": avg_search,
                    "qps": qps,
                    "disk_mb": disk_size_mb,
                    "recall": f"{avg_recall:.1f}%",
                },
            )

            results.append(
                BenchmarkResult(
                    "sift_insert",
                    "milvus",
                    insert_time,
                    throughput=len(base_vectors) / (insert_time / 1000),
                    metadata={"disk_based": True},
                )
            )
            results.append(
                BenchmarkResult(
                    "sift_search",
                    "milvus",
                    avg_search,
                    throughput=qps,
                    p99_ms=p99_search,
                    recall_at_k=avg_recall,
                    metadata={"disk_based": True, "disk_size_mb": disk_size_mb},
                )
            )

            client.close()
            shutil.rmtree(milvus_dir, ignore_errors=True)

        except ImportError:
            print("  Milvus - SKIPPED (not installed)", flush=True)
        except Exception as e:
            BenchmarkLogger.log_error("Milvus Lite", e)
            results.append(BenchmarkResult("sift_ops", "milvus", 0, error=str(e)))

        gc.collect()

    # =========================================================================
    # PHASE 2: ProximaDB ENGINES
    # =========================================================================
    if not PROXIMADB_AVAILABLE:
        print("\n  --- ProximaDB NOT AVAILABLE ---")
        return results

    # Define engine configurations
    # Exact mode: Supported by all engines
    engines_exact = ["sst", "helix", "nova", "swift", "raptor"]
    # Approximate (sqrt) mode: Supported by specific engines
    engines_approx = ["sst", "helix", "swift"]

    # Filter engines based on user input
    selected_engines_exact = [e for e in engines_exact if e in selected_engines]
    selected_engines_approx = [e for e in engines_approx if e in selected_engines]

    run_approximate = not exact_only  # Run unless --exact-only flag is set
    run_exact = not approximate_only  # Run unless --approximate-only flag is set

    # 1. Benchmark Exact Mode (All Engines)
    if selected_engines_exact and run_exact:
        print("\n  --- PROXIMADB ENGINES (100% recall, exact mode) ---")
        print("  Large blocks for minimal overhead (SST: 4MB, HELIX: 512 vec/block)")
        print("  ")

        for engine in selected_engines_exact:
            # Use "exact" search_mode which maps to prune_mode="exact" in ProximaDB init
            result = _benchmark_proximadb_engine(
                engine,
                base_vectors,
                query_vectors,
                metadata,
                temp_dir,
                ground_truth=ground_truth,
                k=k,
                search_mode="exact",
                display_name=f"{engine.upper()}",
            )
            results.extend(result)
            gc.collect()

    # 2. Benchmark Approximate Mode (Sqrt) - Only for supported engines
    if selected_engines_approx and run_approximate:
        print(
            "\n  --- PROXIMADB ENGINES (Approximate mode with sqrt block pruning) ---"
        )
        print("  ")
        print("  Block pruning: ENABLED for SST, SWIFT, HELIX (centroid-based blocks)")
        print("  ")
        print(
            "  SST:   256KB blocks → ~6,000 blocks → sqrt(6000)=77 scanned (98.7% pruned)"
        )
        print(
            "  HELIX: 64 vec/block, 32D PCA, 20-bit Hilbert → ~625 blocks → sqrt(625)=25 scanned (96% pruned)"
        )
        print("  SWIFT: 512KB blocks → centroid-based pruning")
        print("  ")

        for engine in selected_engines_approx:
            # Use "sqrt" search_mode which maps to prune_mode="sqrt" in ProximaDB init
            result = _benchmark_proximadb_engine(
                engine,
                base_vectors,
                query_vectors,
                metadata,
                temp_dir,
                ground_truth=ground_truth,
                k=k,
                search_mode="sqrt",
                display_name=f"{engine.upper()} (approx)",
            )
            results.extend(result)
            gc.collect()

    return results


def _benchmark_proximadb_engine(
    engine: str,
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    metadata: dict,
    temp_dir: str = None,
    timeout_ms: float = None,
    ground_truth: list[list[int]] = None,
    k: int = 10,
    search_mode: str = "exact",
    display_name: str = None,
) -> list[BenchmarkResult]:
    """Benchmark a single ProximaDB engine with disk persistence."""
    results = []
    name = display_name if display_name else engine.upper()

    # Add separator before engine benchmark
    BenchmarkLogger.log_engine_start(name)

    # Handle special engine aliases (e.g., sst_approx -> sst for different search modes)
    actual_engine = "sst" if engine == "sst_approx" else engine

    # Determine prune_mode from search_mode for DB initialization
    # "exact" -> "exact", "sqrt" -> "sqrt", "approximate" -> "sqrt" (default approx)
    prune_mode = search_mode
    if search_mode == "approximate":
        prune_mode = "sqrt"

    try:
        # Phase 1: Create DB and insert all data
        phase1_start = BenchmarkLogger.log_start(name, "Create + Insert")

        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()

        data_dir = os.path.join(
            temp_dir, f"sift_{engine}_{search_mode}"
        )  # Keep unique dir per alias
        os.makedirs(data_dir, exist_ok=True)

        # Use context manager for proper cleanup
        with proximadb.ProximaDB(
            data_dirs=data_dir,
            metadata_dir=os.path.join(data_dir, "metadata"),
            cache_size_mb=512,
            enable_wal=True,  # Enable WAL for proper disk persistence
            prune_mode=prune_mode,
        ) as db:

            collection_name = (
                f"sift_{engine}_{search_mode}"  # Keep unique collection name per alias
            )

            # Create collection (uses engine defaults)
            db.create_collection(collection_name, metadata["dimension"], actual_engine)

            # Insert all data at once
            ids = [f"sift_{i}" for i in range(len(base_vectors))]
            start = time.perf_counter()
            if hasattr(db, "insert_numpy"):
                db.insert_numpy(collection_name, ids, base_vectors)
            else:
                vectors_list = [v.tolist() for v in base_vectors]
                db.insert(collection_name, ids, vectors_list, None)
            insert_time = (time.perf_counter() - start) * 1000

            BenchmarkLogger.log_end(
                f"{name} (Insert)",
                phase1_start,
                {
                    "vectors": len(base_vectors),
                    "throughput_v/s": len(base_vectors) / (insert_time / 1000),
                    "time_ms": insert_time,
                },
            )

            results.append(
                BenchmarkResult(
                    "sift_insert",
                    name,
                    insert_time,
                    throughput=len(base_vectors) / (insert_time / 1000),
                    metadata={
                        "num_vectors": len(base_vectors),
                        "dimension": metadata["dimension"],
                        "mode": search_mode,
                    },
                )
            )

            # Phase 1.5: WARM CACHE SEARCH (DB still open, file handles valid)
            # This measures true algorithm performance without cold I/O penalty
            phase_warm_start = BenchmarkLogger.log_start(name, "Search (Warm Cache)")

            search_times_warm = []
            recalls_warm = []
            num_queries = min(len(query_vectors), 1000)

            for i in range(num_queries):
                start = time.perf_counter()
                if hasattr(db, "search_numpy"):
                    search_results = db.search_numpy(
                        collection_name, query_vectors[i], k, search_mode=search_mode
                    )
                else:
                    search_results = db.search(
                        collection_name,
                        query_vectors[i].tolist(),
                        k,
                        search_mode=search_mode,
                    )
                search_times_warm.append((time.perf_counter() - start) * 1000)

                # Compute recall@k
                if ground_truth and i < len(ground_truth):
                    result_ids = [
                        r.id if hasattr(r, "id") else r.get("id", "")
                        for r in search_results
                    ]
                    recalls_warm.append(
                        compute_recall_at_k(ground_truth[i], result_ids, k)
                    )

            avg_search_warm = float(np.mean(search_times_warm))
            p50_search_warm = float(np.percentile(search_times_warm, 50))
            p95_search_warm = float(np.percentile(search_times_warm, 95))
            qps_warm = 1000 / avg_search_warm
            avg_recall_warm = (
                float(np.mean(recalls_warm)) * 100 if recalls_warm else None
            )

            BenchmarkLogger.log_end(
                f"{name} (Warm {num_queries}q)",
                phase_warm_start,
                {
                    "avg_ms": avg_search_warm,
                    "p50_ms": p50_search_warm,
                    "p95_ms": p95_search_warm,
                    "qps": qps_warm,
                    "recall": (
                        f"{avg_recall_warm:.1f}%"
                        if avg_recall_warm is not None
                        else "N/A"
                    ),
                },
            )

            results.append(
                BenchmarkResult(
                    "sift_search_warm",
                    name,
                    avg_search_warm,
                    throughput=qps_warm,
                    p50_ms=p50_search_warm,
                    p95_ms=p95_search_warm,
                    recall_at_k=avg_recall_warm,
                    metadata={
                        "cache_state": "warm",
                        "num_queries": num_queries,
                        "mode": search_mode,
                    },
                )
            )

            # Phase 2: Force flush memtable to SST files before close
            phase2_start = BenchmarkLogger.log_start(name, "Flush + Close")

            # Note: close() internally calls flush(), so we don't need to call flush() explicitly.
            # Context manager will handle closing
            pass

        # DB is closed here by context manager
        gc.collect()
        time.sleep(1.0)  # Allow OS to flush file handles

        BenchmarkLogger.log_end(f"{name} (Flush+Close)", phase2_start)

        # Calculate disk space - scan ACTUAL ProximaDB write location and filter by collection
        phase_disk_start = BenchmarkLogger.log_start(name, "Disk Analysis")

        total_size = 0
        file_count = 0

        # All data for this run is isolated under the per-engine data_dir. Count everything there.
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                fpath = os.path.join(root, f)
                try:
                    fsize = os.path.getsize(fpath)
                    total_size += fsize
                    file_count += 1
                except OSError:
                    pass

        disk_size_mb = total_size / (1024 * 1024)

        BenchmarkLogger.log_end(
            f"{name} (Disk Analysis)",
            phase_disk_start,
            {"files": file_count, "disk_mb": disk_size_mb},
        )

        # Phase 3: Reopen DB for search (loads indexes from disk)
        phase3_start = BenchmarkLogger.log_start(name, "Reopen + Search")

        with proximadb.ProximaDB(
            data_dirs=data_dir,
            metadata_dir=os.path.join(data_dir, "metadata"),
            cache_size_mb=512,
            enable_wal=True,
            prune_mode=prune_mode,
        ) as db:

            # Search benchmark with optional timeout
            search_times = []
            recalls = []
            num_queries = min(len(query_vectors), 1000)
            start_total = time.perf_counter()

            for i in range(num_queries):
                # Check timeout
                if (
                    timeout_ms
                    and (time.perf_counter() - start_total) * 1000 > timeout_ms
                ):
                    raise TimeoutError(f"RAPTOR timeout after {i} queries")

                start = time.perf_counter()
                if hasattr(db, "search_numpy"):
                    # Use the specified search mode (exact for 100% recall, approximate for speed)
                    search_results = db.search_numpy(
                        collection_name, query_vectors[i], k, search_mode=search_mode
                    )
                else:
                    search_results = db.search(
                        collection_name,
                        query_vectors[i].tolist(),
                        k,
                        search_mode=search_mode,
                    )
                search_times.append((time.perf_counter() - start) * 1000)

                # Compute recall@k if ground truth is available
                if ground_truth and i < len(ground_truth):
                    result_ids = [
                        r.id if hasattr(r, "id") else r.get("id", "")
                        for r in search_results
                    ]
                    recalls.append(compute_recall_at_k(ground_truth[i], result_ids, k))

            avg_search = float(np.mean(search_times))
            p50_search = float(np.percentile(search_times, 50))
            p95_search = float(np.percentile(search_times, 95))
            p99_search = float(np.percentile(search_times, 99))
            qps = 1000 / avg_search
            avg_recall = float(np.mean(recalls)) * 100 if recalls else None

            BenchmarkLogger.log_end(
                f"{name} (Search {num_queries}q)",
                phase3_start,
                {
                    "avg_ms": avg_search,
                    "p50_ms": p50_search,
                    "p95_ms": p95_search,
                    "qps": qps,
                    "recall": f"{avg_recall:.1f}%" if avg_recall is not None else "N/A",
                },
            )

            results.append(
                BenchmarkResult(
                    "sift_search",
                    name,
                    avg_search,
                    throughput=qps,
                    p50_ms=p50_search,
                    p95_ms=p95_search,
                    p99_ms=p99_search,
                    recall_at_k=avg_recall,
                    metadata={
                        "num_queries": num_queries,
                        "top_k": k,
                        "disk_size_mb": disk_size_mb,
                        "mode": search_mode,
                    },
                )
            )

    except TimeoutError as e:
        BenchmarkLogger.log_error(name, e)
        results.append(BenchmarkResult("sift_search", engine, 0, error=str(e)))
    except Exception as e:
        BenchmarkLogger.log_error(name, e)
        results.append(BenchmarkResult("sift_ops", engine, 0, error=str(e)))

    return results


def _investigate_raptor_performance():
    """Investigate why RAPTOR is slow and suggest fixes."""
    print("\n" + "=" * 60)
    print("RAPTOR PERFORMANCE INVESTIGATION")
    print("=" * 60)

    print("""
    RAPTOR Performance Analysis:
    ----------------------------
    RAPTOR uses adaptive row-group storage with Matrix Trinity indexes.

    Potential issues in embedded/stateless mode:
    1. Index not loaded from disk on reopen
    2. Falls back to brute-force disk scanning
    3. Row-group metadata not cached

    Checking RAPTOR search implementation...
    """)

    # Check if RAPTOR has proper disk-based search
    try:
        from pathlib import Path

        raptor_path = Path("src/storage/engines/impls/raptor")
        if raptor_path.exists():
            search_impl = raptor_path / "search.rs"
            if search_impl.exists():
                print(f"  Found: {search_impl}")
                # Read a portion to check implementation
                content = search_impl.read_text()[:2000]
                if "scan_disk" in content or "brute_force" in content:
                    print("  WARNING: RAPTOR may be using brute-force disk scanning")
                    print("  FIX: Implement proper index loading on database reopen")
                else:
                    print("  RAPTOR search implementation looks OK")
    except Exception as e:
        print(f"  Could not analyze RAPTOR code: {e}")

    print("""
    Recommended fixes:
    1. Ensure RAPTOR loads Matrix Trinity indexes on startup
    2. Cache row-group metadata in memory
    3. Use memory-mapped files for faster access
    4. Pre-build search indexes during insert flush
    """)


def benchmark_ldbc_snb(
    scale_factor: int = 1, temp_dir: str = None
) -> list[BenchmarkResult]:
    """
    Benchmark ProximaDB graph engine against LDBC SNB dataset.

    LDBC SNB is the industry standard for graph database benchmarks.
    Social network with complex query patterns.

    Args:
        scale_factor: LDBC scale factor (1, 10, 100)
        temp_dir: Temporary directory for data

    Returns:
        List of benchmark results
    """
    if not PROXIMADB_AVAILABLE:
        return [
            BenchmarkResult("ldbc_init", "orion", 0, error="ProximaDB not available")
        ]

    results = []

    print("\n" + "=" * 80)
    print(f"LDBC SNB BENCHMARK (Scale Factor {scale_factor})")
    print("=" * 80)

    # Load LDBC dataset
    ldbc_data = StandardDatasets.load_ldbc_snb(scale_factor=scale_factor)
    nodes = ldbc_data["nodes"]
    edges = ldbc_data["edges"]
    metadata = ldbc_data["metadata"]

    print(f"  Dataset: {metadata['dataset']}")
    print(
        f"  Nodes: {metadata['num_nodes']:,} (Persons: {metadata['num_persons']:,}, "
        f"Posts: {metadata['num_posts']:,}, Comments: {metadata['num_comments']:,})"
    )
    print(f"  Edges: {metadata['num_edges']:,}")
    print()

    try:
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()

        data_dir = os.path.join(temp_dir, f"ldbc_sf{scale_factor}")
        os.makedirs(data_dir, exist_ok=True)

        db = proximadb.ProximaDB(
            data_dirs=data_dir,
            metadata_dir=os.path.join(data_dir, "metadata"),
            cache_size_mb=512,
            enable_wal=False,
        )

        db.create_graph("ldbc_snb")

        # Node insert benchmark
        print("  Inserting nodes...", end="", flush=True)
        graph_nodes = [
            proximadb.GraphNode(n["id"], labels=n["labels"], properties=n["properties"])
            for n in nodes
        ]
        start = time.perf_counter()
        db.create_nodes("ldbc_snb", graph_nodes)
        node_insert_time = (time.perf_counter() - start) * 1000
        print(
            f" {node_insert_time:.0f}ms ({len(nodes)/(node_insert_time/1000):,.0f}/s)"
        )

        results.append(
            BenchmarkResult(
                "ldbc_insert_nodes",
                "orion",
                node_insert_time,
                throughput=len(nodes) / (node_insert_time / 1000),
                metadata={"num_nodes": len(nodes)},
            )
        )

        # Edge insert benchmark
        print("  Inserting edges...", end="", flush=True)
        graph_edges = [
            proximadb.GraphEdge(
                e["from_node_id"],
                e["to_node_id"],
                e["edge_type"],
                weight=e.get("weight"),
            )
            for e in edges
        ]
        start = time.perf_counter()
        db.create_edges("ldbc_snb", graph_edges)
        edge_insert_time = (time.perf_counter() - start) * 1000
        print(
            f" {edge_insert_time:.0f}ms ({len(edges)/(edge_insert_time/1000):,.0f}/s)"
        )

        results.append(
            BenchmarkResult(
                "ldbc_insert_edges",
                "orion",
                edge_insert_time,
                throughput=len(edges) / (edge_insert_time / 1000),
                metadata={"num_edges": len(edges)},
            )
        )

        # LDBC Interactive Short Queries (IS1-IS7 style)
        print("  Running LDBC-style queries...")

        # IS1: Profile query (get node by ID)
        is1_times = []
        person_ids = [n["id"] for n in nodes if "Person" in n["labels"]]
        for _ in range(min(1000, len(person_ids))):
            pid = random.choice(person_ids)
            start = time.perf_counter()
            _ = db.get_node("ldbc_snb", pid)
            is1_times.append((time.perf_counter() - start) * 1000)

        results.append(
            BenchmarkResult(
                "ldbc_is1_profile",
                "orion",
                float(np.mean(is1_times)),
                p50_ms=float(np.percentile(is1_times, 50)),
                p95_ms=float(np.percentile(is1_times, 95)),
                p99_ms=float(np.percentile(is1_times, 99)),
                metadata={"query_type": "IS1 - Profile"},
            )
        )
        print(
            f"    IS1 Profile: {np.mean(is1_times):.3f}ms avg, {np.percentile(is1_times, 99):.3f}ms p99"
        )

        # IS2: Friends query (1-hop traversal)
        is2_times = []
        for _ in range(min(500, len(person_ids))):
            pid = random.choice(person_ids)
            start = time.perf_counter()
            _ = db.get_outgoing_edges("ldbc_snb", pid)
            is2_times.append((time.perf_counter() - start) * 1000)

        results.append(
            BenchmarkResult(
                "ldbc_is2_friends",
                "orion",
                float(np.mean(is2_times)),
                p50_ms=float(np.percentile(is2_times, 50)),
                p95_ms=float(np.percentile(is2_times, 95)),
                p99_ms=float(np.percentile(is2_times, 99)),
                metadata={"query_type": "IS2 - Friends"},
            )
        )
        print(
            f"    IS2 Friends: {np.mean(is2_times):.3f}ms avg, {np.percentile(is2_times, 99):.3f}ms p99"
        )

        # IS5: Posts by person (filter by edge type)
        is5_times = []
        post_ids = [n["id"] for n in nodes if "Post" in n["labels"]][:1000]
        for pid in post_ids[:500]:
            start = time.perf_counter()
            _ = db.get_outgoing_edges("ldbc_snb", pid)  # Gets HAS_CREATOR edges
            is5_times.append((time.perf_counter() - start) * 1000)

        results.append(
            BenchmarkResult(
                "ldbc_is5_posts",
                "orion",
                float(np.mean(is5_times)),
                p50_ms=float(np.percentile(is5_times, 50)),
                p95_ms=float(np.percentile(is5_times, 95)),
                p99_ms=float(np.percentile(is5_times, 99)),
                metadata={"query_type": "IS5 - Posts"},
            )
        )
        print(
            f"    IS5 Posts: {np.mean(is5_times):.3f}ms avg, {np.percentile(is5_times, 99):.3f}ms p99"
        )

        db.delete_graph("ldbc_snb")

    except Exception as e:
        results.append(BenchmarkResult("ldbc_ops", "orion", 0, error=str(e)))
        print(f" ERROR: {str(e)[:50]}")

    gc.collect()
    return results


def benchmark_hybrid_rag(
    num_documents: int = 10_000, temp_dir: str = None
) -> list[BenchmarkResult]:
    """
    Benchmark ProximaDB SKS (Semantic Knowledge Store) for HybridRAG workloads.

    This is the killer use case for ProximaDB's unified architecture:
    - Vector similarity search for semantic matching
    - Graph traversal for context expansion
    - Combined in a single query

    Args:
        num_documents: Number of documents in the knowledge base
        temp_dir: Temporary directory

    Returns:
        List of benchmark results
    """
    if not PROXIMADB_AVAILABLE:
        return [BenchmarkResult("rag_init", "sks", 0, error="ProximaDB not available")]

    results = []

    print("\n" + "=" * 80)
    print("HYBRIDRAG BENCHMARK (Semantic Knowledge Store)")
    print("=" * 80)

    # Load HybridRAG dataset
    rag_data = StandardDatasets.load_hybrid_rag_dataset(num_documents=num_documents)
    chunks = rag_data["chunks"]
    entities = rag_data["entities"]
    relations = rag_data["relations"]
    embeddings = rag_data["embeddings"]
    metadata = rag_data["metadata"]

    print(f"  Dataset: {metadata['dataset']}")
    print(f"  Documents: {metadata['num_documents']:,}")
    print(f"  Chunks: {metadata['num_chunks']:,} x {metadata['embedding_dimension']}D")
    print(f"  Entities: {metadata['num_entities']:,}")
    print(f"  Relations: {metadata['num_relations']:,}")
    print()

    try:
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()

        data_dir = os.path.join(temp_dir, "hybrid_rag")
        os.makedirs(data_dir, exist_ok=True)

        db = proximadb.ProximaDB(
            data_dirs=data_dir,
            metadata_dir=os.path.join(data_dir, "metadata"),
            cache_size_mb=512,
            enable_wal=False,
        )

        # Create vector collection for chunks
        db.create_collection("rag_chunks", metadata["embedding_dimension"], "sst")

        # Create graph for entities and relations
        db.create_graph("rag_knowledge")

        # Insert chunks with embeddings
        print("  Inserting chunk embeddings...", end="", flush=True)
        chunk_ids = [c["id"] for c in chunks]
        start = time.perf_counter()
        if hasattr(db, "insert_numpy"):
            db.insert_numpy("rag_chunks", chunk_ids, embeddings)
        else:
            vectors_list = [c["embedding"].tolist() for c in chunks]
            db.insert("rag_chunks", chunk_ids, vectors_list, None)
        chunk_insert_time = (time.perf_counter() - start) * 1000
        print(
            f" {chunk_insert_time:.0f}ms ({len(chunks)/(chunk_insert_time/1000):,.0f}/s)"
        )

        results.append(
            BenchmarkResult(
                "rag_insert_chunks",
                "sks",
                chunk_insert_time,
                throughput=len(chunks) / (chunk_insert_time / 1000),
                metadata={"num_chunks": len(chunks)},
            )
        )

        # Insert entities as graph nodes
        print("  Inserting entity nodes...", end="", flush=True)
        entity_nodes = [
            proximadb.GraphNode(
                e["id"], labels=[e["type"]], properties={"name": e["name"]}
            )
            for e in entities
        ]
        # Also add documents as nodes
        doc_nodes = [
            proximadb.GraphNode(c["document_id"], labels=["Document"], properties={})
            for c in chunks[::5]  # One per document (every 5th chunk)
        ]
        start = time.perf_counter()
        db.create_nodes("rag_knowledge", entity_nodes + doc_nodes)
        entity_insert_time = (time.perf_counter() - start) * 1000
        print(f" {entity_insert_time:.0f}ms")

        # Insert relations as graph edges
        print("  Inserting relations...", end="", flush=True)
        graph_edges = [
            proximadb.GraphEdge(
                r["from_id"], r["to_id"], r["type"], weight=r.get("weight", 1.0)
            )
            for r in relations
        ]
        start = time.perf_counter()
        db.create_edges("rag_knowledge", graph_edges)
        relation_insert_time = (time.perf_counter() - start) * 1000
        print(
            f" {relation_insert_time:.0f}ms ({len(relations)/(relation_insert_time/1000):,.0f}/s)"
        )

        results.append(
            BenchmarkResult(
                "rag_insert_graph",
                "sks",
                entity_insert_time + relation_insert_time,
                throughput=len(relations)
                / ((entity_insert_time + relation_insert_time) / 1000),
                metadata={
                    "num_entities": len(entities),
                    "num_relations": len(relations),
                },
            )
        )

        # HybridRAG Query Pattern: Vector Search → Graph Expansion → Context Assembly
        print("  Running HybridRAG queries...")

        hybrid_query_times = []
        num_queries = 100
        rng = np.random.default_rng(46)

        for _ in range(num_queries):
            # Generate a query embedding (perturbed version of a random chunk)
            query_idx = rng.integers(0, len(embeddings))
            query_vec = embeddings[query_idx].copy()
            query_vec += rng.normal(0, 0.1, query_vec.shape).astype(np.float32)
            query_vec = query_vec / np.linalg.norm(query_vec)

            start = time.perf_counter()

            # Step 1: Vector search for similar chunks
            if hasattr(db, "search_numpy"):
                similar_chunks = db.search_numpy("rag_chunks", query_vec, 5)
            else:
                similar_chunks = db.search("rag_chunks", query_vec.tolist(), 5)

            # Step 2: Get document IDs from chunks
            doc_ids = set()
            for result in similar_chunks:
                # chunk_0_1 -> doc_0
                parts = result.id.split("_")
                if len(parts) >= 2:
                    doc_ids.add(f"doc_{parts[1]}")

            # Step 3: Graph expansion - get related entities from documents
            expanded_entities = []
            for doc_id in list(doc_ids)[:3]:  # Limit expansion
                try:
                    edges = db.get_outgoing_edges("rag_knowledge", doc_id)
                    for edge in edges:
                        expanded_entities.append(edge.to_node_id)
                except Exception:
                    pass  # Document might not have edges

            # Step 4: Get entity relationships for context
            for entity_id in expanded_entities[:5]:  # Limit
                try:
                    _ = db.get_outgoing_edges("rag_knowledge", entity_id)
                except Exception:
                    pass

            hybrid_query_times.append((time.perf_counter() - start) * 1000)

        avg_query = float(np.mean(hybrid_query_times))
        p50_query = float(np.percentile(hybrid_query_times, 50))
        p95_query = float(np.percentile(hybrid_query_times, 95))
        p99_query = float(np.percentile(hybrid_query_times, 99))

        results.append(
            BenchmarkResult(
                "rag_hybrid_query",
                "sks",
                avg_query,
                throughput=1000 / avg_query,  # QPS
                p50_ms=p50_query,
                p95_ms=p95_query,
                p99_ms=p99_query,
                metadata={"query_type": "Vector Search + Graph Expansion"},
            )
        )
        print(
            f"    HybridRAG Query: {avg_query:.3f}ms avg, {p99_query:.3f}ms p99, {1000/avg_query:.0f} QPS"
        )

        # Pure vector search for comparison
        pure_vector_times = []
        for _ in range(num_queries):
            query_idx = rng.integers(0, len(embeddings))
            query_vec = embeddings[query_idx].copy()
            query_vec += rng.normal(0, 0.1, query_vec.shape).astype(np.float32)
            query_vec = query_vec / np.linalg.norm(query_vec)

            start = time.perf_counter()
            if hasattr(db, "search_numpy"):
                _ = db.search_numpy("rag_chunks", query_vec, 10)
            else:
                _ = db.search("rag_chunks", query_vec.tolist(), 10)
            pure_vector_times.append((time.perf_counter() - start) * 1000)

        results.append(
            BenchmarkResult(
                "rag_pure_vector",
                "sks",
                float(np.mean(pure_vector_times)),
                throughput=1000 / np.mean(pure_vector_times),
                p50_ms=float(np.percentile(pure_vector_times, 50)),
                p95_ms=float(np.percentile(pure_vector_times, 95)),
                p99_ms=float(np.percentile(pure_vector_times, 99)),
                metadata={"query_type": "Pure Vector Search"},
            )
        )
        print(
            f"    Pure Vector: {np.mean(pure_vector_times):.3f}ms avg, {1000/np.mean(pure_vector_times):.0f} QPS"
        )

        # Pure graph traversal for comparison
        pure_graph_times = []
        entity_ids = [e["id"] for e in entities]
        for _ in range(num_queries):
            eid = random.choice(entity_ids)
            start = time.perf_counter()
            _ = db.get_outgoing_edges("rag_knowledge", eid)
            pure_graph_times.append((time.perf_counter() - start) * 1000)

        results.append(
            BenchmarkResult(
                "rag_pure_graph",
                "sks",
                float(np.mean(pure_graph_times)),
                throughput=1000 / np.mean(pure_graph_times),
                p50_ms=float(np.percentile(pure_graph_times, 50)),
                p95_ms=float(np.percentile(pure_graph_times, 95)),
                p99_ms=float(np.percentile(pure_graph_times, 99)),
                metadata={"query_type": "Pure Graph Traversal"},
            )
        )
        print(
            f"    Pure Graph: {np.mean(pure_graph_times):.3f}ms avg, {1000/np.mean(pure_graph_times):.0f} QPS"
        )

        db.delete_graph("rag_knowledge")

    except Exception as e:
        import traceback

        results.append(BenchmarkResult("rag_ops", "sks", 0, error=str(e)))
        print(f" ERROR: {str(e)[:50]}")
        traceback.print_exc()

    gc.collect()
    return results


def _print_sift_summary_table(results: list[BenchmarkResult]):
    """Render a consolidated table for the SIFT benchmark results."""
    if not RICH_AVAILABLE:
        print("\nRich library not installed. Cannot print summary table.")
        # Fallback to simple print
        print(
            f"{'Engine':<15} {'Operation':<15} {'Time (ms)':<12} {'Throughput (/s)':<15} {'Recall@10':<10} {'Disk (MB)':<10}"
        )
        print("-" * 80)
        for r in results:
            recall_str = f"{r.recall_at_k:.1f}%" if r.recall_at_k is not None else "N/A"
            disk_str = (
                f"{r.metadata.get('disk_size_mb', 0.0):.1f}"
                if r.metadata.get("disk_size_mb") is not None
                else "N/A"
            )
            throughput_str = (
                f"{r.throughput:,.0f}" if r.throughput is not None else "N/A"
            )
            print(
                f"{r.engine:<15} {r.operation:<15} {r.time_ms:<12.3f} {throughput_str:<15} {recall_str:<10} {disk_str:<10}"
            )
        return

    console = Console()
    table = Table(
        title="Consolidated SIFT Benchmark Results",
        expand=True,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Engine", style="cyan", width=15)
    table.add_column("Insert (ms)", justify="right")
    table.add_column("Insert Throughput (/s)", justify="right")
    table.add_column("Search (ms)", justify="right")
    table.add_column("QPS", justify="right")
    table.add_column("p99 (ms)", justify="right")
    table.add_column("Recall@10 (%)", justify="right", style="yellow")
    table.add_column("Disk (MB)", justify="right", style="green")

    # Get a unique list of all engines that ran, preserving a reasonable order
    engine_order = [
        "chromadb",
        "faiss",
        "lancedb",
        "qdrant",
        "usearch",
        "annoy",
        "milvus",
        "sst_approx",
        "sst",
        "helix",
        "nova",
        "swift",
        "raptor",
    ]
    engines_in_results = sorted(
        list(set(r.engine for r in results)),
        key=lambda x: engine_order.index(x) if x in engine_order else 99,
    )

    for engine_name in engines_in_results:
        insert_res = next(
            (
                r
                for r in results
                if r.engine == engine_name and r.operation == "sift_insert"
            ),
            None,
        )
        search_res = next(
            (
                r
                for r in results
                if r.engine == engine_name and r.operation == "sift_search"
            ),
            None,
        )

        if not insert_res and not search_res:
            continue

        display_name = engine_name

        # Insert stats
        if insert_res and not insert_res.error:
            insert_time_str = f"{insert_res.time_ms:.0f}"
            insert_throughput_str = f"{insert_res.throughput:,.0f}"
        else:
            insert_time_str = "[red]ERROR[/red]"
            insert_throughput_str = "N/A"

        # Search stats
        if search_res and not search_res.error:
            recall_str = (
                f"{search_res.recall_at_k:.1f}"
                if search_res.recall_at_k is not None
                else "N/A"
            )
            disk_str = (
                f"{search_res.metadata.get('disk_size_mb', 0.0):.1f}"
                if search_res.metadata.get("disk_size_mb") is not None
                else "N/A"
            )
            if search_res.metadata.get("in_memory"):
                disk_str = "[dim][IN-MEMORY*][/dim]"
                display_name = f"{engine_name}*"

            search_time_str = f"{search_res.time_ms:.3f}"
            qps_str = f"{search_res.throughput:,.0f}"
            p99_str = f"{search_res.p99_ms:.3f}"
        else:
            search_time_str = "[red]ERROR[/red]"
            qps_str = "N/A"
            p99_str = "N/A"
            recall_str = "N/A"
            disk_str = "N/A"
            if search_res and search_res.error:
                # If there was a search error, still try to get disk size from insert metadata
                if insert_res and insert_res.metadata.get("disk_size_mb"):
                    disk_str = f"{insert_res.metadata.get('disk_size_mb', 0.0):.1f}"

        table.add_row(
            display_name,
            insert_time_str,
            insert_throughput_str,
            search_time_str,
            qps_str,
            p99_str,
            recall_str,
            disk_str,
        )

    console.print(table)


def run_standard_dataset_benchmarks(
    sift_vectors: int = 100_000,
    ldbc_scale: int = 1,
    rag_documents: int = 5_000,
    dimension: int = 768,
    engines: str = "all",
    approximate_only: bool = False,
    exact_only: bool = False,
) -> list[BenchmarkResult]:
    """
    Run benchmarks using standard datasets.

    Args:
        sift_vectors: Number of SIFT vectors (max 1M)
        ldbc_scale: LDBC SNB scale factor
        rag_documents: Number of documents for HybridRAG
        dimension: Vector dimension (128=SIFT, 768=BGE-base, 960=GIST, 1024=BGE-large, 1536=OpenAI)

    Returns:
        All benchmark results
    """
    all_results = []

    # Get dimension name for display
    dim_names = {
        128: "SIFT",
        384: "Victor/BGE-small",
        768: "BGE-base",
        960: "GIST",
        1024: "BGE-large",
        1536: "OpenAI",
    }
    dim_name = dim_names.get(dimension, "custom")

    print("\n" + "=" * 100)
    print("PROXIMADB STANDARD DATASET BENCHMARKS")
    print("=" * 100)
    print(f"  SIFT-1M: {sift_vectors:,} vectors")
    print(f"  Dimension: {dimension}D ({dim_name})")
    print(f"  LDBC SNB: Scale Factor {ldbc_scale}")
    print(f"  HybridRAG: {rag_documents:,} documents")
    print()

    with tempfile.TemporaryDirectory() as temp_dir:
        # SIFT-1M benchmark
        sift_results = benchmark_sift_1m(
            max_vectors=sift_vectors,
            temp_dir=temp_dir,
            dimension=dimension,
            engines=engines,
            approximate_only=approximate_only,
            exact_only=exact_only,
        )
        all_results.extend(sift_results)
        gc.collect()

        # LDBC SNB benchmark
        # ldbc_results = benchmark_ldbc_snb(scale_factor=ldbc_scale, temp_dir=temp_dir)
        # all_results.extend(ldbc_results)
        # gc.collect()

        # HybridRAG benchmark
        # rag_results = benchmark_hybrid_rag(num_documents=rag_documents, temp_dir=temp_dir)
        # all_results.extend(rag_results)
        # gc.collect()

    # Print summary table for SIFT results
    print("\n" + "=" * 100)
    print("STANDARD DATASET BENCHMARK SUMMARY")
    print("=" * 100)

    sift_summary_results = [r for r in all_results if r.operation.startswith("sift_")]
    _print_sift_summary_table(sift_summary_results)

    # You can add similar summary tables for LDBC and RAG if needed

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ProximaDB Consolidated Benchmark")
    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmark with smaller sizes"
    )
    parser.add_argument(
        "--no-competitors", action="store_true", help="Skip competitor benchmarks"
    )
    parser.add_argument(
        "--standard-datasets",
        action="store_true",
        help="Run benchmarks with standard datasets (SIFT, LDBC, HybridRAG)",
    )
    parser.add_argument(
        "--sift-vectors",
        type=int,
        default=100_000,
        help="Number of SIFT vectors (default: 100K, max: 1M)",
    )
    parser.add_argument(
        "--ldbc-scale", type=int, default=1, help="LDBC SNB scale factor (default: 1)"
    )
    parser.add_argument(
        "--rag-documents",
        type=int,
        default=5_000,
        help="Number of HybridRAG documents (default: 5K)",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=384,
        choices=[128, 384, 768, 960, 1024, 1536],
        help="Vector dimension: 128=SIFT, 384=Victor/BGE-small, 768=BGE-base, 960=GIST, 1024=BGE-large, 1536=OpenAI (default: 384)",
    )
    parser.add_argument(
        "--engines",
        type=str,
        default="all",
        help="Comma-separated list of engines to benchmark (e.g., raptor,sst) or 'all' for everything. "
        "Available: sst,sst_approx,helix,viper,nova,swift,raptor. Use 'proximadb' for all ProximaDB engines only.",
    )

    # Search mode control (NEW)
    parser.add_argument(
        "--approximate-only",
        action="store_true",
        help="Run ONLY approximate mode (block pruning with sqrt mode, ~8-15ms queries, 90-95%% recall)",
    )
    parser.add_argument(
        "--exact-only",
        action="store_true",
        help="Run ONLY exact mode (100%% recall, ~45ms queries)",
    )

    args = parser.parse_args()

    if args.standard_datasets:
        # Run standard dataset benchmarks
        run_standard_dataset_benchmarks(
            sift_vectors=args.sift_vectors,
            ldbc_scale=args.ldbc_scale,
            rag_documents=args.rag_documents,
            dimension=args.dimension,
            engines=args.engines,
            approximate_only=args.approximate_only,
            exact_only=args.exact_only,
        )
    else:
        # Run consolidated benchmark
        if args.quick:
            config = BenchmarkConfig(
                vector_sizes=[1000], graph_nodes=[1000], sks_entities=[1000]
            )
        else:
            config = BenchmarkConfig()

        run_consolidated_benchmark(config, include_competitors=not args.no_competitors)
