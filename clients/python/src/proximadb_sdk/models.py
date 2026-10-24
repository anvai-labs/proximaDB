"""
ProximaDB Python Client - Data Models for REST API

These Pydantic models are designed to work with the ProximaDB REST API.
They align with the server-side REST handlers (not proto definitions).

Copyright 2025 ProximaDB

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Type aliases for convenience
VectorArray = Union[list[list[float]], np.ndarray]
MetadataDict = dict[str, str | int | float | bool | list[str | int | float]]
FilterDict = dict[str, Any]


# ============================================================================
# ENUMS - String values for REST API
# ============================================================================


class DistanceMetric(str, Enum):
    """Distance metrics for REST API (string-based)

    Note: Server supports 13+ distance metrics. Unsupported metrics
    will fallback to COSINE (server behavior, not client validation).

    For gRPC: Use DistanceMetricType enum with integer values instead.
    """

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"  # Fallback: COSINE (server decides)
    HAMMING = "hamming"  # Fallback: COSINE (server decides)
    JACCARD = "jaccard"  # Fallback: COSINE (server decides)

    # Extended metrics supported by server (ProximaDB v1.0+)
    CHEBYSHEV = "chebyshev"
    CANBERRA = "canberra"
    MINKOWSKI = "minkowski"
    ANGULAR = "angular"
    BRAY_CURTIS = "bray_curtis"
    HELLINGER = "hellinger"
    CUSTOM = "custom"


class DistanceMetricType(int, Enum):
    """Distance metric types for gRPC API (integer-based, matches v1 proto)

    Maps to proximadb.v1.types_pb2.DistanceMetric enum.
    Use these constants when working with gRPC to avoid magic numbers.

    Example:
        config = CollectionConfig(distance_metric=DistanceMetricType.COSINE, ...)
    """

    UNSPECIFIED = 0
    COSINE = 1
    EUCLIDEAN = 2
    DOT_PRODUCT = 3
    HAMMING = 4
    MANHATTAN = 5
    JACCARD = 6
    CHEBYSHEV = 7
    CANBERRA = 8
    MINKOWSKI = 9
    ANGULAR = 10
    BRAY_CURTIS = 11
    HELLINGER = 12
    CUSTOM = 13


class StorageEngine(str, Enum):
    """Storage engines for ProximaDB

    Default: SST (row-based, write-optimized). VIPER is deprecated (ADR-093)
    and can no longer be selected; SWIFT/RAPTOR are experimental
    (TD-SDKENG-1).
    """

    VIPER = "viper"  # DEPRECATED (ADR-093) — the server rejects it
    SST = "sst"  # Row-based, write-optimized with three-stage filtering (DEFAULT)
    NOVA = "nova"  # Progressive columnar storage with multi-level quantization
    HELIX = "helix"  # Locality-optimized storage with Hilbert curve clustering
    SWIFT = "swift"  # High-speed row-based with FastLanes encoding
    RAPTOR = "raptor"  # Adaptive row-group management with PxK optimization
    MMAP = "mmap"  # Memory-mapped storage
    HYBRID = "hybrid"  # Hybrid storage approach


class IndexingAlgorithm(str, Enum):
    """Indexing algorithms for REST API (string-based)

    Note: All 6 indexing algorithms are fully supported by the server as of 2025-08.
    No fallbacks required - server handles all algorithms natively.

    For gRPC: Use IndexType enum with integer values instead.
    """

    HNSW = "hnsw"  # Hierarchical Navigable Small World
    IVF = "ivf"  # Inverted File Index
    PQ = "pq"  # Product Quantization
    FLAT = "flat"  # Brute force / exhaustive search
    ANNOY = "annoy"  # Approximate Nearest Neighbors Oh Yeah
    LSH = "lsh"  # Locality Sensitive Hashing (fully supported)


class IndexType(int, Enum):
    """Index types for gRPC API (integer-based, matches v1 proto)

    Maps to proximadb.v1.collection_types_pb2.IndexConfig.algorithm field.
    Use these constants when working with gRPC to avoid magic numbers.

    Example:
        ic = IndexConfig(algorithm=IndexType.HNSW, index_name="primary")
    """

    UNSPECIFIED = 0
    HNSW = 1  # Hierarchical Navigable Small World
    IVF = 2  # Inverted File Index
    PQ = 3  # Product Quantization
    FLAT = 4  # Brute force / exhaustive search
    ANNOY = 5  # Approximate Nearest Neighbors Oh Yeah
    LSH = 6  # Locality Sensitive Hashing


class EmbeddingPrecision(str, Enum):
    """Canonical embedding precision for a collection.

    Mirrors the server's proto ``EmbeddingPrecision`` enum
    (``proto/proximadb/v1/collection_types.proto``) and the Rust SDK's
    ``EmbeddingPrecision`` (``clients/rust/src/collection.rs``). Set
    once at collection-create time via
    ``CollectionConfig.canonical_embedding_precision``; controls the
    on-disk + in-memory scalar type for the embedding column.

    See ``docs/05-concepts/embedding-precision.adoc`` for the operator
    guide. ``FP32`` (the default) preserves legacy behavior — existing
    callers that never set the field continue to produce identical
    serialized payloads.

    Example:
        cfg = CollectionConfig(
            name="docs_fp16",
            dimension=768,
            canonical_embedding_precision=EmbeddingPrecision.FP16,
        )
        # Or pass a string — common aliases all accepted
        cfg = CollectionConfig(
            name="docs_fp16",
            dimension=768,
            canonical_embedding_precision="fp16",  # or "half", "float16", "EMBEDDING_PRECISION_FP16"
        )
    """

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    UINT8 = "uint8"

    @classmethod
    def _normalize(cls, raw):
        """Accept the same string forms the server's
        ``apply_proto_enum_workarounds`` accepts: canonical lowercase,
        proto SCREAMING label (``EMBEDDING_PRECISION_FP16``), and common
        aliases (``half``, ``float16``, ``bfloat16``, ``i8``,
        ``int8_scalar``, ``u8``, ``uint8_scalar``).

        Returns an EmbeddingPrecision or raises ValueError. Used by the
        CollectionConfig field_validator so users can pass either the
        enum or a string and get consistent semantics.
        """
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, str):
            key = raw.strip().lower()
            if key.startswith("embedding_precision_"):
                key = key[len("embedding_precision_") :]
            aliases = {
                "fp32": cls.FP32,
                "f32": cls.FP32,
                "float32": cls.FP32,
                "fp16": cls.FP16,
                "f16": cls.FP16,
                "half": cls.FP16,
                "float16": cls.FP16,
                "bf16": cls.BF16,
                "bfloat16": cls.BF16,
                "int8": cls.INT8,
                "i8": cls.INT8,
                "int8_scalar": cls.INT8,
                "uint8": cls.UINT8,
                "u8": cls.UINT8,
                "uint8_scalar": cls.UINT8,
            }
            if key in aliases:
                return aliases[key]
        raise ValueError(
            f"unrecognised canonical_embedding_precision {raw!r}; "
            "accepted: fp32, fp16, bf16, int8, uint8 (case-insensitive, "
            "with shorthand half/float16/bfloat16/i8/u8/etc.)"
        )


class StorageEngineType(int, Enum):
    """Storage engine types for gRPC API (integer-based, matches v1 proto)

    Maps to proximadb.v1.types_pb2.StorageEngine enum.
    Use these constants when working with gRPC to avoid magic numbers.

    ProximaDB supports 6 specialized storage engines that auto-optimize for different workloads.

    Example:
        config = CollectionConfig(storage_engine=StorageEngineType.VIPER, ...)
    """

    UNSPECIFIED = 0
    VIPER = 1  # Columnar Parquet format with advanced quantization (DEFAULT)
    SST = 2  # Row-based, write-optimized with three-stage filtering
    NOVA = 3  # Progressive columnar storage with multi-level quantization
    HELIX = 4  # Locality-optimized storage with Hilbert curve clustering
    SWIFT = 5  # High-speed row-based with FastLanes encoding
    RAPTOR = 6  # Adaptive row-group management with PxK optimization


class IndexUpdateMode(str, Enum):
    """Index update modes"""

    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    HYBRID_MODE = "hybrid_mode"


class FilterableDataType(str, Enum):
    """Filterable data types"""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    ARRAY_STRING = "array_string"
    ARRAY_INTEGER = "array_integer"
    ARRAY_FLOAT = "array_float"


class CompressionAlgorithm(str, Enum):
    """Compression algorithms for SDK-driven compression"""

    NONE = "none"
    ZSTD = "zstd"
    LZ4 = "lz4"
    SNAPPY = "snappy"


class CompressionLevel(int, Enum):
    """Compression levels (1-9 for ZSTD, ignored for others)"""

    FASTEST = 1
    FAST = 3
    BALANCED = 6
    HIGH = 9


class ServerCapabilities(BaseModel):
    """Server capabilities and fallback behavior for configuration validation"""

    # Fully supported distance metrics (no fallback) - All 13 metrics supported as of 2025-08
    supported_distance_metrics: list[str] = [
        "cosine",
        "euclidean",
        "dot_product",
        "manhattan",
        "hamming",
        "jaccard",
        "chebyshev",
        "canberra",
        "minkowski",
        "angular",
        "bray_curtis",
        "hellinger",
        "custom",
    ]

    # Distance metrics that fallback to cosine (none - all are now supported natively)
    fallback_distance_metrics: list[str] = []

    # Fully supported storage engines (no fallback)
    supported_storage_engines: list[str] = ["viper", "sst"]

    # Storage engines that fallback to viper
    fallback_storage_engines: list[str] = ["mmap", "hybrid"]

    # Fully supported indexing algorithms (no fallback) - All 6 algorithms supported as of 2025-08
    supported_indexing_algorithms: list[str] = [
        "hnsw",
        "ivf",
        "pq",
        "flat",
        "annoy",
        "lsh",
    ]

    # Indexing algorithms that fallback to hnsw (none - all are now supported natively)
    fallback_indexing_algorithms: list[str] = []

    # Quantization types (all supported in VIPER engine)
    supported_quantization_types: list[str] = [
        "none",
        "uniform",
        "pq",
        "scalar",
        "binary",
        "custom",
    ]

    # Server behavior notes
    notes: dict[str, str] = {
        "fallback_policy": "Server uses intelligent fallbacks instead of errors",
        "dimension_limit": "Server default maximum is 65536 dimensions (configurable)",
        "name_validation": "Collection names must be 8+ characters to avoid collision with 7-char base62 IDs",
        "quantization_engine": "Quantization fully supported in VIPER engine only",
        "filterable_columns": "All FilterableDataType values supported",
    }

    @classmethod
    def get_fallback_for(cls, config_type: str, value: str) -> str | None:
        """Get the fallback value for an unsupported configuration"""
        capabilities = cls()

        if config_type == "distance_metric":
            if value in capabilities.fallback_distance_metrics:
                return "cosine"
        elif config_type == "storage_engine":
            if value in capabilities.fallback_storage_engines:
                return "viper"
        elif config_type == "indexing_algorithm":
            if value in capabilities.fallback_indexing_algorithms:
                return "hnsw"

        return None

    @classmethod
    def is_supported(cls, config_type: str, value: str) -> bool:
        """Check if a configuration value is fully supported (no fallback)"""
        capabilities = cls()

        if config_type == "distance_metric":
            return value in capabilities.supported_distance_metrics
        elif config_type == "storage_engine":
            return value in capabilities.supported_storage_engines
        elif config_type == "indexing_algorithm":
            return value in capabilities.supported_indexing_algorithms
        elif config_type == "quantization_type":
            return value in capabilities.supported_quantization_types

        return True  # Default to supported for unknown types


class CollectionOperationType(str, Enum):
    """Collection operation types"""

    CREATE = "create"
    UPDATE = "update"
    GET = "get"
    LIST = "list"
    DELETE = "delete"
    MIGRATE = "migrate"


class VectorOperationType(str, Enum):
    """Vector operation types"""

    INSERT = "insert"
    UPSERT = "upsert"
    UPDATE = "update"
    DELETE = "delete"
    SEARCH = "search"
    GET = "get"


class FilterOperator(str, Enum):
    """Filter operators"""

    AND = "and"
    OR = "or"
    NOT = "not"


class FilterOperation(str, Enum):
    """Filter operations"""

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN = "less_than"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"


# ============================================================================
# QUANTIZATION MODELS
# ============================================================================


class QuantizationType(str, Enum):
    """Quantization types for REST API"""

    NONE = "none"
    UNIFORM = "uniform"
    PRODUCT = "pq"
    SCALAR = "scalar"
    BINARY = "binary"
    CUSTOM = "custom"


class QuantizationLevel(BaseModel):
    """Quantization level configuration"""

    level_type: str  # "none", "uniform", "pq", "scalar", "binary", "custom"
    bits: int | None = None
    scale: float | None = None
    offset: float | None = None
    num_subvectors: int | None = None
    bits_per_code: int | None = None
    codebook_id: str | None = None
    adaptive_subvectors: bool | None = None
    threshold: float | None = None
    sign_based: bool | None = None
    clamp_values: bool | None = None
    type_id: str | None = None
    bits_per_element: int | None = None
    config: dict[str, str] | None = None


class StorageQuantizationConfig(BaseModel):
    """Storage quantization configuration"""

    enabled: bool = False
    level: QuantizationLevel | None = None
    codebook_id: str | None = None
    progressive_quantization: bool = False
    storage_compatibility: str = "VIPER_ONLY"


class IndexQuantizationStrategy(BaseModel):
    """Index quantization strategy"""

    index_name: str
    level: QuantizationLevel
    build_async: bool = False
    codebook_id: str | None = None


class IndexQuantizationConfig(BaseModel):
    """Index quantization configuration"""

    enabled: bool = False
    strategies: list[IndexQuantizationStrategy] = Field(default_factory=list)
    auto_select_strategy: bool = False


class SearchQuantizationConfig(BaseModel):
    """Search quantization configuration"""

    enabled: bool = False
    default_level: QuantizationLevel | None = None
    adaptive_precision: bool = True
    accuracy_threshold: float = 0.95
    candidate_multiplier: int = 3


class QuantizationValidation(BaseModel):
    """Quantization validation configuration"""

    accuracy_threshold: float = 0.95
    validation_sample_size: int = 1000
    enable_quality_monitoring: bool = True
    retraining_threshold: float = 0.90


class CompressionConfig(BaseModel):
    """Unified compression configuration matching proto definition

    This configuration aligns with the proto CompressionConfig message,
    providing a unified interface where engine-specific features are
    automatically applied based on the collection's storage_engine.

    Defaults are optimized for VIPER engine (server default).
    """

    # Unified compression settings (proto fields 1-2)
    algorithm: CompressionAlgorithm = Field(
        default=CompressionAlgorithm.NONE,
        description="Compression algorithm (ZSTD/LZ4/Snappy)",
    )
    level: int | None = Field(
        default=None, description="Compression level (1-22 for ZSTD, 1-9 for others)"
    )

    # Global settings (proto field 3-4)
    adaptive: bool = Field(
        default=False,
        description="Enable adaptive compression based on data characteristics",
    )
    min_ratio: float | None = Field(
        default=None,
        description="Minimum compression ratio (e.g., 1.5 = 50% reduction)",
    )

    # VIPER-specific quantization (proto fields 5-7)
    enable_quantization: bool = Field(
        default=False,
        description="Enable VIPER dual columns (FP32 + quantized). Ignored by SST engine.",
    )
    quantization_type: str | None = Field(
        default=None,
        description="VIPER quantization method: 'int8', 'pq8', 'pq4'. Ignored by SST engine.",
    )
    normalization_method: str | None = Field(
        default=None,
        description="VIPER normalization: 'mean', 'trimmed_mean', 'median'. Ignored by SST engine.",
    )

    # SST-specific block sizing (proto fields 8-9)
    block_size_kb: int | None = Field(
        default=None,
        description="SST block size in KB (256-16384). Ignored by VIPER engine.",
    )
    dynamic_block_sizing: bool = Field(
        default=False,
        description="Auto-adjust SST block size based on vector dimensions. Ignored by VIPER engine.",
    )

    @field_validator("level")
    def validate_compression_level(cls, v):
        """Validate compression level matches server-side validation"""
        if v is not None and (v < 1 or v > 22):
            raise ValueError(
                "Compression level must be between 1-22 (1-9 for most algorithms, 1-22 for ZSTD)"
            )
        return v

    @field_validator("min_ratio")
    def validate_compression_ratio(cls, v):
        """Validate compression ratio matches server-side validation"""
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("Minimum compression ratio must be between 0.0 and 1.0")
        return v

    @field_validator("quantization_type")
    def validate_quantization_type(cls, v):
        """Validate quantization type is supported"""
        if v is not None:
            valid_types = {
                "int8",
                "pq8",
                "pq4",
                "uniform",
                "pq",
                "scalar",
                "binary",
                "none",
            }
            if v not in valid_types:
                raise ValueError(
                    f"Quantization type must be one of: {', '.join(valid_types)}"
                )
        return v

    @field_validator("normalization_method")
    def validate_normalization_method(cls, v):
        """Validate normalization method is supported"""
        if v is not None:
            valid_methods = {"mean", "trimmed_mean", "median", "none"}
            if v not in valid_methods:
                raise ValueError(
                    f"Normalization method must be one of: {', '.join(valid_methods)}"
                )
        return v

    @field_validator("block_size_kb")
    def validate_block_size(cls, v):
        """Validate SST block size is within acceptable range"""
        if v is not None and (v < 256 or v > 16384):
            raise ValueError("SST block size must be between 256-16384 KB")
        return v


class QuantizationConfig(BaseModel):
    """Quantization configuration"""

    enabled: bool = False
    type: QuantizationType = QuantizationType.NONE
    progressive_quantization: bool = False

    # Product quantization params
    bits_per_subvector: int | None = None
    num_subvectors: int | None = None

    # Scalar quantization params
    bits_per_vector: int | None = None

    # Binary quantization params
    threshold: float | None = None

    # Common params
    accuracy_threshold: float | None = 0.95
    compression_ratio_target: float | None = None
    validation_sample_size: int | None = 1000
    retraining_threshold: float | None = 0.90


class ComprehensiveQuantizationConfig(BaseModel):
    """Comprehensive quantization configuration matching proto structure"""

    enabled: bool = False
    storage_quantization: StorageQuantizationConfig | None = None
    index_quantization: IndexQuantizationConfig | None = None
    search_quantization: SearchQuantizationConfig | None = None
    compression_ratio_target: float | None = None
    validation: QuantizationValidation | None = None


# ============================================================================
# STORAGE ENGINE CONFIGURATION MODELS
# ============================================================================


class AccessPattern(str, Enum):
    """Access patterns for storage optimization"""

    UNKNOWN = "unknown"
    WRITE_HEAVY = "write_heavy"
    READ_HEAVY = "read_heavy"
    BALANCED = "balanced"
    ARCHIVE = "archive"


class DataDensity(str, Enum):
    """Data density characteristics"""

    UNKNOWN = "unknown"
    DENSE = "dense"  # >80% non-zero values
    SPARSE = "sparse"  # <20% non-zero values
    MIXED = "mixed"


class ParquetWriterSettings(BaseModel):
    """Parquet writer settings for columnar engines"""

    row_group_size: int | None = None
    page_size: int | None = None
    enable_bloom_filters: bool | None = None
    bloom_filter_fpp: float | None = None
    bloom_filter_columns: list[str] | None = None
    enable_column_statistics: bool | None = None
    enable_page_index: bool | None = None
    enable_column_index: bool | None = None
    enable_offset_index: bool | None = None
    page_index_granularity: int | None = None
    enable_dictionary: bool | None = None
    dictionary_threshold: float | None = None
    enable_delta_encoding: bool | None = None
    enable_byte_stream_split: bool | None = None
    enable_pq_sorting: bool | None = None
    pq_sorting_segments: int | None = None
    pq_sorting_codebook_size: int | None = None
    enable_native_metadata: bool | None = None
    metadata_inference_samples: int | None = None
    write_batch_size: int | None = None
    id_less_storage: bool | None = None


class FooterCacheSettings(BaseModel):
    """Footer cache settings for cloud storage optimization"""

    enable: bool | None = None
    max_entries: int | None = None
    ttl_seconds: int | None = None
    time_to_idle_seconds: int | None = None
    enable_persistence: bool | None = None
    persistence_path: str | None = None
    enable_prefetch: bool | None = None
    prefetch_threshold: int | None = None
    warming_interval_seconds: int | None = None
    enable_compression: bool | None = None
    compression_level: int | None = None


class HybridWriterSettings(BaseModel):
    """Hybrid writer settings for adaptive performance"""

    enable: bool | None = None
    initial_mode: str | None = None  # "streaming", "batch", "adaptive"
    enable_auto_switch: bool | None = None
    mode_switch_threshold: int | None = None
    pattern_window_size: int | None = None
    streaming_threshold: float | None = None
    batch_threshold: int | None = None
    max_buffer_size: int | None = None
    buffer_time_limit_seconds: int | None = None
    enable_concurrent_writes: bool | None = None
    max_concurrent_writers: int | None = None
    optimize_row_group_size: bool | None = None
    min_row_group_size: int | None = None
    max_row_group_size: int | None = None


class SstEngineSettings(BaseModel):
    """SST-specific engine settings"""

    enable_bloom_filters: bool | None = None
    bloom_filter_fpp: float | None = None
    compression: CompressionAlgorithm | None = None
    compression_level: int | None = None
    write_buffer_size: int | None = None
    max_write_buffers: int | None = None
    block_size_kb: int | None = None
    dynamic_block_sizing: bool | None = None


class ViperEngineSettings(BaseModel):
    """VIPER-specific engine settings"""

    inherit_global_settings: bool | None = None
    enable_columnar_compression: bool | None = None
    enable_vector_quantization: bool | None = None
    vector_chunk_size: int | None = None
    enable_lazy_loading: bool | None = None


class NovaEngineSettings(BaseModel):
    """NOVA-specific engine settings"""

    inherit_global_settings: bool | None = None
    enable_real_time_mode: bool | None = None
    streaming_buffer_size: int | None = None
    prefer_low_latency: bool | None = None


# Note: StorageEngineConfig is deprecated, use StorageConfig instead


# ============================================================================
# INDEX CONFIGURATION MODELS
# ============================================================================


class HnswConfig(BaseModel):
    """HNSW index configuration"""

    m: int = 16
    ef_construction: int = 200
    ef_search: int = 50
    max_partition_size: int = 100000
    adaptive_parameters: bool = True
    use_simd: bool = True
    memory_limit_mb: int = 512
    lazy_loading: bool = True
    prune_connections: int = 0
    level_multiplier: float = 0.69


class IvfConfig(BaseModel):
    """IVF index configuration"""

    n_lists: int = 100
    n_probe: int = 1
    quantization_bits: int = 8
    use_pq: bool = False
    pq_subspaces: int = 8
    train_on_insert: bool = False
    min_train_size: int = 1000


class FlatConfig(BaseModel):
    """Flat index configuration"""

    enable_simd: bool = True
    batch_size: int = 1000
    enable_parallel_search: bool = True


class PqConfig(BaseModel):
    """Product Quantization index configuration"""

    subvectors: int = 8
    bits_per_subvector: int = 8
    training_sample_count: int = 10000
    enable_reranking: bool = True


class AnnoyConfig(BaseModel):
    """Annoy index configuration"""

    n_trees: int = 10
    search_k: int = -1
    max_leaf_size: int = 100
    enable_mmap: bool = True


class RandomProjectionType(str, Enum):
    """Random projection types for LSH"""

    GAUSSIAN = "gaussian"
    BINARY = "binary"
    SPARSE = "sparse"


class LshConfig(BaseModel):
    """LSH index configuration"""

    n_hash_tables: int = 10
    n_hash_functions: int = 8
    bucket_width: float = 4.0
    binary_vectors: bool = False
    max_candidates: int = 100
    projection: RandomProjectionType = RandomProjectionType.GAUSSIAN


class IndexConfiguration(BaseModel):
    """Index configuration"""

    index_name: str
    algorithm: IndexingAlgorithm | IndexType
    update_mode: IndexUpdateMode = IndexUpdateMode.SYNCHRONOUS
    async_update_timeout_ms: int | None = None
    async_update_batch_size: int | None = None
    enable_background_optimization: bool | None = None
    hnsw_config: HnswConfig | None = None
    ivf_config: IvfConfig | None = None
    flat_config: FlatConfig | None = None
    pq_config: PqConfig | None = None
    annoy_config: AnnoyConfig | None = None
    lsh_config: LshConfig | None = None
    build_concurrency: int | None = None
    memory_limit_mb: int | None = None
    checkpoint_interval_ms: int | None = None
    is_primary: bool | None = None
    use_cases: list[str] | None = None
    selectivity_threshold: float | None = None


# ============================================================================
# COLLECTION MODELS
# ============================================================================


class FilterableColumn(BaseModel):
    """Filterable column specification"""

    name: str
    data_type: FilterableDataType
    indexed: bool = True
    supports_range: bool = False
    estimated_cardinality: int | None = None


class CollectionConfig(BaseModel):
    """Collection configuration aligned with proto CollectionConfig"""

    model_config = ConfigDict(populate_by_name=True)

    # CORE CONFIGURATION (Required)
    name: str = Field(
        min_length=1
    )  # Non-empty only — the server resolves collections by name OR id and accepts
    # short names (relational-DDL tables: part/orders/region). The former 8-char
    # minimum (to avoid colliding with 7-char base62 IDs) was a vector-collection-
    # era constraint the server relaxed; the SDK must not be stricter than the
    # server. See ADR-068 / TD-SDK-1 S2a.
    dimension: int = Field(
        ge=1, le=65536
    )  # Server default maximum is 65536 (configurable)
    distance_metric: DistanceMetric | None = (
        DistanceMetric.COSINE
    )  # Default to most common metric

    # STORAGE CONFIGURATION
    storage_engine: StorageEngine | None = (
        StorageEngine.SST
    )  # Default to SST (fast, production-ready)
    storage_config: Optional["StorageConfig"] = None  # Complete storage configuration
    compression: Optional["CompressionConfig"] = (
        None  # Optional compression configuration (SDK convenience)
    )

    # INDEX CONFIGURATION
    index_configs: list[IndexConfiguration] | None = None
    primary_index: str | None = None  # Primary index name
    auto_index_selection: bool | None = None  # Auto-select best index

    # SCHEMA CONFIGURATION
    filterable_columns: list[FilterableColumn] | None = None
    quantization_config: QuantizationConfig | None = Field(
        None, alias="quantization"
    )  # Vector quantization configuration
    primary_indexing_algorithm: IndexingAlgorithm | None = (
        None  # Primary indexing algorithm
    )

    @property
    def quantization(self):
        """Alias property for backward compatibility"""
        return self.quantization_config

    # METADATA
    description: str | None = None
    tags: list[str] | None = None
    owner: str | None = None

    # Additional Python SDK fields
    metadata_schema: dict[str, Any] | None = None
    filterable_metadata_fields: list[str] | None = None

    # Per-collection canonical embedding precision (fp16/bf16/int8/uint8).
    # Default `None` means "use the server's fp32 default" — preserves the
    # wire payload byte-identical with pre-precision-rollout SDK requests.
    # See docs/05-concepts/embedding-precision.adoc for the operator guide.
    canonical_embedding_precision: EmbeddingPrecision | None = None

    @field_validator("canonical_embedding_precision", mode="before")
    @classmethod
    def normalize_canonical_embedding_precision(cls, v):
        """Accept enum, canonical string, proto SCREAMING label, or
        common shorthand aliases (half / float16 / bfloat16 / i8 / etc.).
        Same dispatch as the server's apply_proto_enum_workarounds so
        REST / gRPC / SQL DDL clients see consistent semantics."""
        if v is None:
            return None
        return EmbeddingPrecision._normalize(v)

    @field_validator("name")
    def validate_name_length(cls, v):
        """Collection name must be non-empty.

        The former >=8-char floor (to avoid colliding with 7-char base62 IDs)
        was a vector-collection-era constraint the server relaxed for relational
        DDL (tables like `part`/`orders`/`region`). The server resolves
        collections by name OR id, so the SDK must not be stricter than the
        server. See ADR-068 / TD-SDK-1 S2a.
        """
        if not v or not v.strip():
            raise ValueError("Collection name cannot be empty")
        return v.strip()

    def model_post_init(self, __context):
        """Post-initialization validation to align compression config with storage engine"""
        (
            super().model_post_init(__context)
            if hasattr(super(), "model_post_init")
            else None
        )

        # Apply engine-specific compression defaults and validation
        if self.compression:
            self._validate_compression_for_engine()

    def _validate_compression_for_engine(self):
        """Validate compression configuration aligns with storage engine capabilities"""
        if not self.compression:
            return

        # TD-SDKENG-1: SST is the default; VIPER is deprecated (ADR-093).
        engine = self.storage_engine or StorageEngine.SST

        if engine == StorageEngine.VIPER:
            # VIPER engine: quantization features are valid
            if self.compression.block_size_kb is not None:
                import warnings

                warnings.warn(
                    "block_size_kb is ignored by VIPER engine (only applies to SST engine)",
                    UserWarning,
                    stacklevel=2,
                )

            # Apply VIPER-optimized defaults
            if (
                self.compression.quantization_type
                and not self.compression.enable_quantization
            ):
                # Auto-enable quantization if type is specified
                self.compression.enable_quantization = True

        elif engine == StorageEngine.SST:
            # SST engine: quantization features are ignored
            if self.compression.enable_quantization:
                import warnings

                warnings.warn(
                    "enable_quantization is ignored by SST engine (only applies to VIPER engine)",
                    UserWarning,
                    stacklevel=2,
                )

            if self.compression.quantization_type:
                import warnings

                warnings.warn(
                    "quantization_type is ignored by SST engine (only applies to VIPER engine)",
                    UserWarning,
                    stacklevel=2,
                )

            # Apply SST-optimized defaults
            if self.compression.block_size_kb is None:
                self.compression.block_size_kb = 8192  # Reasonable default for SST

        # Validate algorithm-level compatibility
        if self.compression.level is not None:
            algo = self.compression.algorithm
            if algo == CompressionAlgorithm.ZSTD and self.compression.level > 22:
                raise ValueError("ZSTD compression level must be between 1-22")
            elif algo != CompressionAlgorithm.ZSTD and self.compression.level > 9:
                raise ValueError(f"{algo.value} compression level must be between 1-9")

    @property
    def index_config(self):
        """Backward compatibility property for singular index_config access"""
        if self.index_configs and len(self.index_configs) > 0:
            return self.index_configs[0]
        return None


class CollectionStats(BaseModel):
    """Collection statistics"""

    #: Number of vectors, or ``None`` when the server does not know.
    #:
    #: ``None`` and ``0`` mean different things: ``0`` is a collection the
    #: server knows to be empty, ``None`` is a collection whose counter was
    #: never maintained (the write path does not call ``update_stats``).
    #: These used to be indistinguishable, and reading the fabricated ``0``
    #: as "empty" produced a false data-loss report against the server.
    vector_count: int | None = None
    index_size_bytes: int = 0
    data_size_bytes: int = 0


class CollectionInfo(BaseModel):
    """Collection info for list response"""

    id: str
    name: str
    dimension: int
    metric: str
    created_at_ms: int  # Milliseconds since epoch (signed int64)
    updated_at_ms: int  # Milliseconds since epoch (signed int64)
    vector_count: int | None = None
    indexed: bool = False

    # Backward compatibility properties
    @property
    def created_at(self) -> int:
        """Backward compatibility: created_at in seconds"""
        return self.created_at_ms // 1000

    @created_at.setter
    def created_at(self, value: int):
        """Backward compatibility: created_at in seconds"""
        self.created_at_ms = value * 1000

    @property
    def updated_at(self) -> int:
        """Backward compatibility: updated_at in seconds"""
        return self.updated_at_ms // 1000

    @updated_at.setter
    def updated_at(self, value: int):
        """Backward compatibility: updated_at in seconds"""
        self.updated_at_ms = value * 1000


class Collection(BaseModel):
    """Collection information"""

    id: str
    config: CollectionConfig
    stats: CollectionStats = Field(
        default_factory=CollectionStats
    )  # Made required to match REST API
    created_at_ms: int = Field(
        default_factory=lambda: int(__import__("time").time() * 1000)
    )  # Milliseconds since epoch (signed int64)
    updated_at_ms: int = Field(
        default_factory=lambda: int(__import__("time").time() * 1000)
    )  # Milliseconds since epoch (signed int64)

    @property
    def name(self) -> str:
        """Backward compatibility property for collection name"""
        return self.config.name

    @property
    def timestamp(self) -> int:
        """Backward compatibility property for timestamp (seconds)"""
        return self.created_at_ms // 1000

    @property
    def created_at(self) -> int:
        """Backward compatibility: created_at in seconds"""
        return self.created_at_ms // 1000

    @created_at.setter
    def created_at(self, value: int):
        """Backward compatibility: created_at in seconds"""
        self.created_at_ms = value * 1000

    @property
    def updated_at(self) -> int:
        """Backward compatibility: updated_at in seconds"""
        return self.updated_at_ms // 1000

    @updated_at.setter
    def updated_at(self, value: int):
        """Backward compatibility: updated_at in seconds"""
        self.updated_at_ms = value * 1000

    @property
    def dimension(self) -> int:
        """Backward compatibility property for dimension"""
        return self.config.dimension

    @property
    def distance_metric(self):
        """Backward compatibility property for distance metric"""
        return self.config.distance_metric

    @property
    def storage_engine(self):
        """Backward compatibility property for storage engine"""
        return self.config.storage_engine

    @property
    def vector_count(self) -> int | None:
        """Backward compatibility property for vector count.

        ``None`` means the server did not report a count, which is distinct
        from a known-zero collection.
        """
        return self.stats.vector_count


# ============================================================================
# VECTOR MODELS
# ============================================================================


class VectorRecord(BaseModel):
    """Vector record for REST API"""

    id: str | None = None
    vector: list[float]
    metadata: dict[str, str | int | float | bool | list[str | int | float]] = Field(
        default_factory=dict
    )
    timestamp_ms: int = Field(
        default_factory=lambda: int(__import__("time").time() * 1000)
    )  # Required - milliseconds since epoch (signed int64)
    updated_at_ms: int | None = (
        None  # Only set if different from timestamp_ms (saves bytes)
    )
    expires_at_ms: int | None = (
        None  # TTL support (milliseconds since epoch, signed int64)
    )
    version: int | None = 0  # Optional to save bytes, use small positive values
    source: str | None = (
        None  # Original content that generated this vector (e.g., chunk text for RAG)
    )

    @field_validator("vector")
    def validate_vector(cls, v):
        if not v:
            raise ValueError("Vector cannot be empty")
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Vector must contain only numeric values")
        if not all(math.isfinite(float(x)) for x in v):
            raise ValueError("Vector must contain only finite numeric values")
        return v

    def __getitem__(self, key: str) -> Any:
        """Support legacy dict-style access for SDK callers/tests."""
        return self.model_dump()[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Support legacy dict.get(...) access for SDK callers/tests."""
        return self.model_dump().get(key, default)

    # Backward compatibility properties
    @property
    def timestamp(self) -> int:
        """Backward compatibility: timestamp in seconds"""
        return self.timestamp_ms // 1000

    @timestamp.setter
    def timestamp(self, value: int):
        """Backward compatibility: timestamp in seconds"""
        self.timestamp_ms = value * 1000

    @property
    def updated_at(self) -> int | None:
        """Backward compatibility: updated_at in seconds"""
        return self.updated_at_ms // 1000 if self.updated_at_ms else None

    @updated_at.setter
    def updated_at(self, value: int | None):
        """Backward compatibility: updated_at in seconds"""
        self.updated_at_ms = value * 1000 if value is not None else None

    @property
    def expires_at(self) -> int | None:
        """Backward compatibility: expires_at in seconds"""
        return self.expires_at_ms // 1000 if self.expires_at_ms else None

    @expires_at.setter
    def expires_at(self, value: int | None):
        """Backward compatibility: expires_at in seconds"""
        self.expires_at_ms = value * 1000 if value is not None else None


# ============================================================================
# SEARCH MODELS
# ============================================================================


class FilterCondition(BaseModel):
    """Filter condition"""

    field_name: str
    operation: FilterOperation
    value: Any


class MetadataFilter(BaseModel):
    """Metadata filter"""

    conditions: list[FilterCondition]
    operator: FilterOperator = FilterOperator.AND


class SearchQuery(BaseModel):
    """Search query"""

    vector: list[float]
    filters: dict[str, Any] = (
        {}
    )  # Simple equality filters (proto map field - always include)
    id: str | None = None
    metadata_filter: MetadataFilter | None = None


class SearchParameters(BaseModel):
    """Search parameters"""

    ef_search: int | None = None
    max_connections: int | None = None
    n_probe: int | None = None
    enable_reranking: bool | None = None
    batch_size: int | None = None
    timeout_ms: int | None = None
    accuracy_threshold: float | None = None
    enable_parallel_search: bool | None = None
    thread_count: int | None = None


class IncludeFields(BaseModel):
    """Fields to include in search results"""

    vector: bool = False
    metadata: bool = True
    score: bool = True
    rank: bool = True


class QuantizationHint(BaseModel):
    """Quantization hint for search"""

    hint_type: str  # "none", "binary", "scalar", "product", "uniform"
    parameters: dict[str, Any] | None = None


class SearchOptimization(BaseModel):
    """Search optimization hints including compression-aware options"""

    top_k: int | None = None
    filters: dict[str, Any] | None = None
    accuracy_threshold: float | None = None
    include_expired: bool | None = None
    timeout_ms: int | None = None
    enable_two_stage: bool | None = None
    quantization_hint: QuantizationHint | None = None
    enable_clustering_hint: bool | None = None
    enable_metadata_filtering_hint: bool | None = None

    # Compression-aware search hints
    prefer_compressed_search: bool | None = Field(
        default=None, description="Prefer searching compressed data when available"
    )
    decompression_budget_ms: int | None = Field(
        default=None, description="Maximum time budget for decompression operations"
    )
    use_decompression_cache: bool | None = Field(
        default=True, description="Use decompression cache for repeated searches"
    )
    compression_aware_routing: bool | None = Field(
        default=None, description="Enable compression-aware query routing"
    )

    custom_hints: dict[str, Any] | None = None


class SearchResult(BaseModel):
    """Search result - aligned with SearchVectorRecord proto"""

    id: str
    score: float
    vector: list[float] | None = None
    metadata: dict[str, Any] | None = None
    rank: int | None = None
    # Additional SearchVectorRecord fields (proto field 5-13)
    version: int | None = None  # Proto field 5
    similarity: float | None = None  # Proto field 6
    timestamp: int | None = None  # Proto field 7 (milliseconds)
    source: str | None = None  # Proto field 8 (original content for RAG)
    expanded_context: list[str] | None = None  # Proto field 9
    semantic_similarity: float | None = None  # Proto field 10
    quantization_info: str | None = None  # Proto field 11
    engine_stats: dict[str, str] | None = None  # Proto field 12
    index_path: str | None = None  # Proto field 13

    # Backward compatibility - map timestamp to timestamp_ms
    @property
    def timestamp_ms(self) -> int | None:
        """Alias for timestamp field"""
        return self.timestamp


class SearchProgress(BaseModel):
    """Progress state for progressive search"""

    stage: int
    stages: int
    complete: bool


class SearchEnvelope(BaseModel):
    """Envelope for paginated/progressive SKS search results"""

    items: list[SearchResult]
    total: int | None = None
    cursor: str | None = None
    has_more: bool = False
    progress: SearchProgress | None = None


class VectorGetResponse(BaseModel):
    """Vector get response"""

    id: str
    collection_id: str
    vector: list[float] | None = None
    metadata: dict[str, Any] | None = None
    score: float | None = None
    rank: int | None = None


class ListCollectionsResponse(BaseModel):
    """List collections response"""

    collections: list[CollectionInfo]
    total_count: int


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class CollectionOperationRequest(BaseModel):
    """Collection operation request"""

    operation: CollectionOperationType
    collection_id: str | None = None
    collection_name: str | None = None
    config: CollectionConfig | None = None
    query_params: dict[str, str] | None = None
    options: dict[str, bool] | None = None


class CollectionResponse(BaseModel):
    """Collection operation response"""

    success: bool
    operation: str
    collection: Collection | None = None
    collections: list[Collection] | None = None
    affected_count: int = 0
    total_count: int | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    error_message: str | None = None
    error_code: str | None = None
    processing_time_us: int = 0


class VectorBatchRequest(BaseModel):
    """Vector batch operation request - aligned with REST API"""

    collection_id: str
    vectors: list[VectorRecord]  # Changed from 'records' to match REST API
    batch_timeout_ms: int | None = None
    request_id: str | None = None


class VectorSearchRequest(BaseModel):
    """Vector search request"""

    collection_id: str
    queries: list[SearchQuery]
    top_k: int = 10
    distance_metric_override: str | None = None
    search_parameters: SearchParameters | None = None  # Fixed field name
    include_fields: IncludeFields | None = None
    search_optimization: SearchOptimization | None = None


class OperationMetrics(BaseModel):
    """Operation metrics"""

    total_processed: int = 0
    successful_count: int = 0
    failed_count: int = 0
    updated_count: int = 0
    processing_time_us: int = 0
    wal_write_time_us: int = 0
    index_update_time_us: int = 0


class DeleteResult(BaseModel):
    """Delete operation result"""

    deleted_count: int = 0
    success: bool = True
    message: str | None = None
    # Per-id failure messages; populated by batch delete_vectors when individual
    # deletes fail. (Previously absent, so delete_vectors silently dropped the
    # errors it tried to set — a real bug this field closes.)
    errors: list[str] = Field(default_factory=list)
    metrics: OperationMetrics | None = None


class BatchResult(BaseModel):
    """Batch operation result"""

    total: int = 0
    success: int = 0
    failed: int = 0
    errors: list[str] = Field(default_factory=list)
    duration_ms: float = 0.0
    metrics: OperationMetrics = Field(default_factory=OperationMetrics)


class VectorOperationResponse(BaseModel):
    """Vector operation response"""

    success: bool | int
    operation: str
    metrics: OperationMetrics
    results: list[SearchResult] | None = None
    vector_ids: list[str] = Field(default_factory=list)
    error_message: str | None = None
    error_code: str | None = None

    @property
    def count(self) -> int:
        """Backward compatibility: return successful count from metrics"""
        return self.metrics.successful_count if self.metrics else 0


class ApiError(BaseModel):
    """API error details"""

    code: str
    message: str
    details: Any | None = None


class ApiResponse(BaseModel):
    """Generic API response wrapper"""

    success: bool
    data: Any | None = None
    error: ApiError | None = None
    message: str | None = None

    model_config = ConfigDict(extra="allow")


# ============================================================================
# HEALTH AND MONITORING
# ============================================================================


class CompressionType(str, Enum):
    """Compression types"""

    NONE = "none"
    LZ4 = "lz4"
    ZSTD = "zstd"
    SNAPPY = "snappy"
    # `builders/collection.py` already references `CompressionType.GZIP`
    # and the `test_collection_builder_fluent_methods_and_helpers`
    # unit test expects `"gzip"` to be a valid value. Add the variant
    # so both call sites work.
    GZIP = "gzip"


class StorageConfig(BaseModel):
    """Complete storage configuration matching proto StorageConfig"""

    # Storage location and persistence
    storage_location: str | None = None  # Override default storage path
    persistent: bool | None = True  # Whether data persists after restart

    # Compression configuration
    compression: CompressionConfig | None = None

    # Optimization hints
    access_pattern: AccessPattern | None = None
    data_density: DataDensity | None = None
    frequent_updates: bool | None = None
    expected_size_gb: int | None = None
    read_write_ratio: float | None = None

    # Quick presets
    preset: str | None = (
        None  # "maximum_performance", "balanced", "memory_constrained", "cloud_optimized", "real_time", "archive"
    )

    # Master optimization control
    enable_all_optimizations: bool | None = True  # Default enabled

    # Specific configuration overrides
    parquet_writer: ParquetWriterSettings | None = None
    footer_cache: FooterCacheSettings | None = None
    hybrid_writer: HybridWriterSettings | None = None

    # Engine-specific settings
    sst_settings: SstEngineSettings | None = None
    viper_settings: ViperEngineSettings | None = None
    nova_settings: NovaEngineSettings | None = None


class FlushConfig(BaseModel):
    """Flush configuration"""

    force_flush: bool = False
    timeout_ms: int = 5000
    include_secondary_indexes: bool = True
    include_metadata: bool = True
    max_wal_size_mb: float | None = None


class HealthStatus(BaseModel):
    """Health check response"""

    status: str
    version: str
    uptime_seconds: int
    services: dict[str, str]
    timestamp_ms: int  # Milliseconds since epoch (signed int64)

    # Backward compatibility property
    @property
    def timestamp(self) -> int:
        """Backward compatibility: timestamp in seconds"""
        return self.timestamp_ms // 1000

    @timestamp.setter
    def timestamp(self, value: int):
        """Backward compatibility: timestamp in seconds"""
        self.timestamp_ms = value * 1000


class ProbeResponse(BaseModel):
    """Kubernetes liveness/readiness probe response (OpenAPI ProbeResponse)."""

    model_config = ConfigDict(extra="allow")
    status: str


class ColumnDefinition(BaseModel):
    """Column definition for a collection schema (OpenAPI ColumnDefinition)."""

    model_config = ConfigDict(extra="allow")
    name: str
    data_type: str
    nullable: bool | None = None
    indexed: bool | None = None
    filterable: bool | None = None
    max_length: int | None = None
    precision: int | None = None
    scale: int | None = None
    vector_dimension: int | None = None


class SchemaDefinition(BaseModel):
    """Schema definition (OpenAPI SchemaDefinition)."""

    model_config = ConfigDict(extra="allow")
    columns: list[ColumnDefinition]
    enforcement: str | None = None
    allow_additional_fields: bool | None = None


class SchemaResponse(BaseModel):
    """Response from GET /api/v2/collections/{id}/schema (OpenAPI SchemaResponse)."""

    model_config = ConfigDict(extra="allow")
    schema_id: str
    schema_version: str
    collection_id: str
    schema_: SchemaDefinition = Field(alias="schema")
    created_at: str
    updated_at: str | None = None
    parent_schema_id: str | None = None


class UpdateSchemaResponse(BaseModel):
    """Response from PUT /api/v2/collections/{id}/schema (OpenAPI UpdateSchemaResponse)."""

    model_config = ConfigDict(extra="allow")
    schema_id: str
    schema_version: str
    previous_schema_id: str
    changes: list[dict[str, Any]]
    warnings: list[str]
    updated_at: str


# Simple alias
Vector = VectorRecord


# TD-SDKENG-1: `create_collection(name, **kwargs)` used to build
# `CollectionConfig(name=name, **kwargs)`, and pydantic IGNORES unknown fields
# — `engine=` (the REST API's spelling) and `index_params=` were silently
# discarded, so a user asking for HELIX got the SST default.
_COLLECTION_KWARG_ALIASES: dict[str, str] = {
    "engine": "storage_engine",
    "index_params": "index_configs",
    "index_specs": "index_configs",
}

# Keyword arguments the TRANSPORT layer consumes (not CollectionConfig
# fields — they are forwarded past the config, never into it):
# rest_sync.create_collection(schema=..., initial_capacity=...) builds the
# v2 body; grpc_sync.create_collection(indexing_algorithm=...) tunes the
# primary index. Round-1 review: rejecting these broke the SDK's only route
# to CreateCollectionRequest.schema.
_COLLECTION_PASSTHROUGH_KWARGS = frozenset(
    {"schema", "initial_capacity", "indexing_algorithm"}
)


def normalize_collection_config_kwargs(
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split create-style kwargs into (config fields, transport pass-through).

    Applies the REST-spelling aliases, REJECTS unknown keywords with a clear
    error (a wrong keyword is a caller bug — silently dropping it created a
    different collection than asked for), and keeps the transport-consumed
    keywords separate so they are forwarded to the transport, never into the
    config. The `quantization=` spelling needs no alias here: pydantic's own
    field alias already accepts it.
    """
    config_kwargs: dict[str, Any] = {}
    passthrough: dict[str, Any] = {}
    source_of: dict[str, str] = {}
    for key, value in kwargs.items():
        if key in _COLLECTION_PASSTHROUGH_KWARGS:
            passthrough[key] = value
            continue
        target = _COLLECTION_KWARG_ALIASES.get(key, key)
        if target in source_of and source_of[target] != key:
            raise TypeError(
                f"conflicting keyword arguments: both "
                f"{source_of[target]}= and {key}= given"
            )
        source_of[target] = key
        config_kwargs[target] = value
    # Field names AND pydantic's own field aliases (e.g. `quantization` is
    # CollectionConfig.quantization_config's alias) are accepted spellings.
    accepted = set(CollectionConfig.model_fields) | {
        field.alias for field in CollectionConfig.model_fields.values() if field.alias
    }
    unknown = sorted(set(config_kwargs) - accepted)
    if unknown:
        alias_hint = ", ".join(
            f"{alias}={target}" for alias, target in _COLLECTION_KWARG_ALIASES.items()
        )
        raise TypeError(
            f"create_collection() got unexpected keyword argument(s): "
            f"{', '.join(unknown)}. Accepted: CollectionConfig fields plus "
            f"{sorted(_COLLECTION_PASSTHROUGH_KWARGS)} "
            f"(aliases: {alias_hint})"
        )
    return config_kwargs, passthrough
