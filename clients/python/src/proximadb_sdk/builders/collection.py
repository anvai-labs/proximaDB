"""
Collection Builder

Fluent interface for building collection configurations.
"""

from typing import Any

from ..models import (
    CollectionConfig,
    CompressionType,
    DistanceMetric,
    IndexingAlgorithm,
    StorageEngine,
)


class CollectionBuilder:
    """
    Fluent interface for building collection configurations.

    Examples:
        # Simple collection
        config = (CollectionBuilder("my_collection", 384)
            .cosine_similarity()
            .sst_storage()
            .hnsw_index()
            .build())

        # Advanced collection with optimization
        config = (CollectionBuilder("advanced_collection", 768)
            .euclidean_distance()
            .hybrid_storage()
            .ivf_index()
            .description("High-performance collection for ML embeddings")
            .compression("zstd")
            .enable_bloom_filter()
            .index_param("ef_construction", 200)
            .index_param("m", 16)
            .build())
    """

    def __init__(self, name: str, dimension: int):
        """
        Initialize collection builder

        Args:
            name: Collection name
            dimension: Vector dimension
        """
        self._name = name
        self._dimension = dimension
        self._distance_metric = DistanceMetric.COSINE
        self._storage_engine = StorageEngine.SST
        self._index_type = IndexingAlgorithm.HNSW
        self._description: str | None = None
        self._index_params: dict[str, Any] = {}
        # Compression/bloom moved to server/engine config; SDK leaves defaults

    # Distance metrics
    def cosine_similarity(self) -> "CollectionBuilder":
        """Use cosine similarity (default)"""
        self._distance_metric = DistanceMetric.COSINE
        return self

    def euclidean_distance(self) -> "CollectionBuilder":
        """Use Euclidean (L2) distance"""
        self._distance_metric = DistanceMetric.EUCLIDEAN
        return self

    def dot_product(self) -> "CollectionBuilder":
        """Use dot product similarity"""
        self._distance_metric = DistanceMetric.DOT_PRODUCT
        return self

    def manhattan_distance(self) -> "CollectionBuilder":
        """Use Manhattan (L1) distance"""
        self._distance_metric = DistanceMetric.MANHATTAN
        return self

    def hamming_distance(self) -> "CollectionBuilder":
        """Use Hamming distance (for binary vectors)"""
        self._distance_metric = DistanceMetric.HAMMING
        return self

    def jaccard_similarity(self) -> "CollectionBuilder":
        """Use Jaccard similarity"""
        self._distance_metric = DistanceMetric.JACCARD
        return self

    def distance_metric(self, metric: DistanceMetric) -> "CollectionBuilder":
        """Set distance metric directly"""
        self._distance_metric = metric
        return self

    # Storage engines
    def sst_storage(self) -> "CollectionBuilder":
        """Use SST row-based storage (optimized for writes)"""
        self._storage_engine = StorageEngine.SST
        return self

    def hybrid_storage(self) -> "CollectionBuilder":
        """Use hybrid adaptive storage"""
        self._storage_engine = StorageEngine.HYBRID
        return self

    def storage_engine(self, engine: StorageEngine) -> "CollectionBuilder":
        """Set storage engine directly"""
        self._storage_engine = engine
        return self

    # Index types
    def hnsw_index(self) -> "CollectionBuilder":
        """Use HNSW index (default, best for most use cases)"""
        self._index_type = IndexingAlgorithm.HNSW
        return self

    def ivf_index(self) -> "CollectionBuilder":
        """Use IVF index (good for large datasets)"""
        self._index_type = IndexingAlgorithm.IVF
        return self

    def flat_index(self) -> "CollectionBuilder":
        """Use flat/brute-force index (exact search)"""
        self._index_type = IndexingAlgorithm.FLAT
        return self

    def annoy_index(self) -> "CollectionBuilder":
        """Use Annoy index (memory efficient)"""
        self._index_type = IndexingAlgorithm.ANNOY
        return self

    def lsh_index(self) -> "CollectionBuilder":
        """Use LSH index (fast approximate search)"""
        self._index_type = IndexingAlgorithm.LSH
        return self

    def index_type(self, index: IndexingAlgorithm) -> "CollectionBuilder":
        """Set index type directly"""
        self._index_type = index
        return self

    # Configuration
    def description(self, desc: str) -> "CollectionBuilder":
        """Set collection description"""
        self._description = desc
        return self

    def compression(self, comp_type: str) -> "CollectionBuilder":
        """Set compression type"""
        if isinstance(comp_type, str):
            comp_type = CompressionType(comp_type.lower())
        self._compression = comp_type
        return self

    def no_compression(self) -> "CollectionBuilder":
        """Disable compression"""
        self._compression = CompressionType.NONE
        return self

    def gzip_compression(self) -> "CollectionBuilder":
        """Use gzip compression"""
        self._compression = CompressionType.GZIP
        return self

    def zstd_compression(self) -> "CollectionBuilder":
        """Use zstd compression (default, best balance)"""
        self._compression = CompressionType.ZSTD
        return self

    def lz4_compression(self) -> "CollectionBuilder":
        """Use lz4 compression (fastest)"""
        self._compression = CompressionType.LZ4
        return self

    def enable_bloom_filter(self, enable: bool = True) -> "CollectionBuilder":
        """Enable/disable bloom filter for faster lookups"""
        self._enable_bloom_filter = enable
        return self

    def disable_bloom_filter(self) -> "CollectionBuilder":
        """Disable bloom filter"""
        self._enable_bloom_filter = False
        return self

    # Index parameters
    def index_param(self, name: str, value: Any) -> "CollectionBuilder":
        """Set index-specific parameter"""
        self._index_params[name] = value
        return self

    def hnsw_params(
        self, m: int = 16, ef_construction: int = 200
    ) -> "CollectionBuilder":
        """Set HNSW-specific parameters"""
        self._index_params.update({"m": m, "ef_construction": ef_construction})
        return self

    def ivf_params(self, n_lists: int = 100, n_probes: int = 10) -> "CollectionBuilder":
        """Set IVF-specific parameters"""
        self._index_params.update({"n_lists": n_lists, "n_probes": n_probes})
        return self

    def annoy_params(self, n_trees: int = 10) -> "CollectionBuilder":
        """Set Annoy-specific parameters"""
        self._index_params.update({"n_trees": n_trees})
        return self

    def lsh_params(self, n_tables: int = 10, n_bits: int = 10) -> "CollectionBuilder":
        """Set LSH-specific parameters"""
        self._index_params.update({"n_tables": n_tables, "n_bits": n_bits})
        return self

    # Build
    def build(self) -> CollectionConfig:
        """Build CollectionConfig object"""
        # Build a clean config (index set at call site as IndexConfig)
        return CollectionConfig(
            name=self._name,
            dimension=self._dimension,
            distance_metric=self._distance_metric,
            storage_engine=self._storage_engine,
            description=self._description,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation"""
        config = self.build()
        return {
            "name": config.name,
            "dimension": config.dimension,
            "distance_metric": (
                config.distance_metric.value if config.distance_metric else None
            ),
            "storage_engine": (
                config.storage_engine.value if config.storage_engine else None
            ),
            # Indexing is specified via index_configs in proto; builder omits it here
            "primary_indexing_algorithm": None,
            "description": config.description,
        }


# Convenience functions
def collection(name: str, dimension: int) -> CollectionBuilder:
    """Create a new CollectionBuilder"""
    return CollectionBuilder(name, dimension)


def text_collection(
    name: str, embedding_model: str = "all-mpnet-base-v2"
) -> CollectionBuilder:
    """Create collection optimized for text embeddings"""
    # Common text embedding dimensions
    dimension_map = {
        "all-mpnet-base-v2": 768,
        "all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "openai/text-embedding-ada-002": 1536,
    }

    dimension = dimension_map.get(embedding_model, 768)  # Default to 768

    return (
        CollectionBuilder(name, dimension)
        .cosine_similarity()
        .sst_storage()
        .hnsw_index()
        .zstd_compression()
        .enable_bloom_filter()
        .description(f"Text collection using {embedding_model} embeddings")
    )


def image_collection(name: str, model: str = "clip") -> CollectionBuilder:
    """Create collection optimized for image embeddings"""
    # Common image embedding dimensions
    dimension_map = {
        "clip": 512,
        "resnet": 2048,
        "vit": 768,
    }

    dimension = dimension_map.get(model, 512)

    return (
        CollectionBuilder(name, dimension)
        .cosine_similarity()
        .hybrid_storage()
        .hnsw_index()
        .lz4_compression()  # Images often benefit from faster compression
        .enable_bloom_filter()
        .description(f"Image collection using {model} embeddings")
    )


def high_performance_collection(name: str, dimension: int) -> CollectionBuilder:
    """Create collection optimized for high performance"""
    return (
        CollectionBuilder(name, dimension)
        .cosine_similarity()
        .sst_storage()  # Fast writes
        .hnsw_index()
        .hnsw_params(m=32, ef_construction=400)  # Higher quality index
        .lz4_compression()  # Fast compression
        .enable_bloom_filter()
        .description("High-performance collection with optimized settings")
    )
