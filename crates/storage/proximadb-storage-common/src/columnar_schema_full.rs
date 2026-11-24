//! Unified Columnar Schema Generation
use arrow_schema::DataType;
// This module provides automatic quantization-aware schema generation for NOVA and NOVA engines.
// When QuantizationConfig is detected, schemas automatically include quantized columns with
// optimized mixed compression strategies per column type.

use anyhow::Result;
use arrow_schema::{Field, Schema, TimeUnit};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, trace};

use crate::columnar_constants::{DEFAULT_PAGE_SIZE, DEFAULT_ROW_GROUP_SIZE};
use proximadb_compression::CompressionAlgorithm;
use proximadb_proto::proximadb_v1::QuantizationConfig;
use proximadb_proto::proximadb_v1::{FilterableColumnSpec, FilterableDataType};

/// Convert proto FilterableColumnSpec to internal ColumnarFilterableSpec
pub fn convert_filterable_spec(spec: &FilterableColumnSpec) -> ColumnarFilterableSpec {
    let data_type = match FilterableDataType::try_from(spec.data_type) {
        Ok(FilterableDataType::FilterableString)
        | Ok(FilterableDataType::FilterableArrayString) => FilterableData::String,
        Ok(FilterableDataType::FilterableInteger)
        | Ok(FilterableDataType::FilterableArrayInteger) => FilterableData::Integer,
        Ok(FilterableDataType::FilterableFloat) | Ok(FilterableDataType::FilterableArrayFloat) => {
            FilterableData::Float
        }
        Ok(FilterableDataType::FilterableBoolean) => FilterableData::Boolean,
        Ok(FilterableDataType::FilterableDatetime) => FilterableData::Datetime,
        _ => FilterableData::String, // Default fallback
    };

    ColumnarFilterableSpec {
        name: spec.name.clone(),
        data_type,
        nullable: true,
        indexed: spec.indexed,
        estimated_cardinality: spec.estimated_cardinality.map(|c| c as usize),
    }
}

/// Create Arrow schema for Parquet files from proto specs
/// Used by both NOVA and NOVA engines for consistent schema generation
pub fn create_parquet_schema_from_specs(
    dimension: usize,
    filterable_specs: &[FilterableColumnSpec],
    include_quantization: bool,
) -> Arc<Schema> {
    let mut fields = vec![
        // Core fields
        Field::new("id", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeBinary((dimension * 4) as i32), // FP32 vectors
            false,
        ),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("version", DataType::Int64, true),
    ];

    // Add quantization fields if enabled
    if include_quantization {
        // Binary quantization
        fields.push(Field::new(
            "vector_binary",
            DataType::FixedSizeBinary(dimension.div_ceil(8) as i32),
            true,
        ));

        // INT8 quantization
        fields.push(Field::new(
            "vector_int8",
            DataType::FixedSizeBinary(dimension as i32),
            true,
        ));
        fields.push(Field::new("int8_scale", DataType::Float32, true));
        fields.push(Field::new("int8_zero_point", DataType::Int8, true));

        // Product Quantization (default 32 segments)
        fields.push(Field::new("vector_pq", DataType::FixedSizeBinary(32), true));
    }

    // Add filterable metadata columns with proper types
    for spec in filterable_specs {
        let arrow_data_type = match FilterableDataType::try_from(spec.data_type) {
            Ok(FilterableDataType::FilterableString)
            | Ok(FilterableDataType::FilterableArrayString) => DataType::Utf8,
            Ok(FilterableDataType::FilterableInteger)
            | Ok(FilterableDataType::FilterableArrayInteger) => DataType::Int64,
            Ok(FilterableDataType::FilterableFloat)
            | Ok(FilterableDataType::FilterableArrayFloat) => DataType::Float64,
            Ok(FilterableDataType::FilterableBoolean) => DataType::Boolean,
            Ok(FilterableDataType::FilterableDatetime) => {
                DataType::Timestamp(TimeUnit::Millisecond, None)
            }
            _ => DataType::Utf8, // Default to string for unknown types
        };

        fields.push(Field::new(&spec.name, arrow_data_type, true));
    }

    // Add extra_metadata field for non-filterable data
    fields.push(Field::new("extra_metadata", DataType::Utf8, true));

    Arc::new(Schema::new(fields))
}

/// Schema configuration with quantization awareness
#[derive(Debug, Clone)]
pub struct ColumnarSchemaConfig {
    /// Vector dimension
    pub dimension: usize,

    /// Quantization configuration from collection
    pub quantization: Option<QuantizationConfig>,

    /// Filterable columns specification
    pub filterable_columns: Vec<ColumnarFilterableSpec>,

    /// Schema optimization settings
    pub optimization: SchemaOptimization,

    /// Compression strategy per column type
    pub compression_strategy: CompressionStrategy,
}

/// Columnar-specific filterable column specification
/// Different from proto::proximadb_v1::FilterableColumnSpec to avoid confusion
/// This is used internally for columnar storage schema generation
#[derive(Debug, Clone)]
pub struct ColumnarFilterableSpec {
    pub name: String,
    pub data_type: FilterableData,
    pub nullable: bool,
    pub indexed: bool,
    pub estimated_cardinality: Option<usize>,
}

/// Supported filterable data types
///
/// Extended with rich types for ProximaRecord to support OLAP/OLTP workloads.
/// Each type maps to a specific Arrow DataType for columnar storage.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterableData {
    // Basic types (original)
    String,
    Integer,
    Float,
    Boolean,
    Datetime,
    Array(Box<FilterableData>),
    Json,
    Map,

    // Rich text types (NEW for ProximaRecord)
    /// Large text with dedicated columnar storage (LargeUtf8)
    Text(TextColumnOptions),
    /// Very large text with sidecar storage
    TextLarge,

    // Precision numeric types (NEW)
    /// Decimal128 for financial precision (38,18 default)
    Decimal {
        precision: u8,
        scale: u8,
    },

    // Temporal types (NEW)
    /// Timestamp with timezone (TimestampTz)
    TimestampTz {
        timezone: std::string::String,
    },
    /// Date only (Date32)
    Date,
    /// Time only (Time64)
    Time,

    // Identifier types (NEW)
    /// RFC 4122 UUID (FixedSizeBinary(16))
    Uuid,

    // Binary types (NEW)
    /// Raw bytes (Binary)
    Binary,

    // Geospatial types (NEW)
    /// Geographic point (lat, lon)
    GeoPoint,
    /// Geographic polygon (list of points)
    GeoPolygon,
}

/// TEXT column-specific options for storage strategy
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TextColumnOptions {
    /// Storage strategy
    pub storage_strategy: TextStorageStrategy,
    /// Enable fulltext index (Tantivy)
    pub enable_fulltext_index: bool,
    /// Enable n-gram bloom filter for CONTAINS queries
    pub enable_ngram_bloom: bool,
    /// N-gram size (default: 3)
    pub ngram_size: u32,
}

/// TEXT storage strategy for columnar storage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextStorageStrategy {
    /// Store inline in main Parquet/Arrow column (<4KB)
    Inline,
    /// Split into chunks with embeddings (4KB-1MB)
    Chunked,
    /// Store in separate sidecar file (>1MB)
    Sidecar,
    /// Auto-select based on actual size (default)
    #[default]
    Adaptive,
}

impl TextStorageStrategy {
    /// Size thresholds for adaptive strategy
    pub const INLINE_MAX_SIZE: usize = 4 * 1024; // 4KB
    pub const CHUNKED_MAX_SIZE: usize = 1024 * 1024; // 1MB

    /// Determine strategy based on content size
    pub fn for_size(size: usize) -> Self {
        if size <= Self::INLINE_MAX_SIZE {
            Self::Inline
        } else if size <= Self::CHUNKED_MAX_SIZE {
            Self::Chunked
        } else {
            Self::Sidecar
        }
    }
}

/// Schema optimization settings
#[derive(Debug, Clone)]
pub struct SchemaOptimization {
    /// Enable dictionary encoding for low-cardinality strings
    pub enable_dictionary_encoding: bool,

    /// Enable nullable optimization (reduce null overhead)
    pub optimize_nullability: bool,

    /// Enable timestamp precision optimization
    pub optimize_timestamp_precision: bool,

    /// Enable fixed-size binary optimization for vectors
    pub enable_fixed_size_binary: bool,

    /// Target row group size for optimal I/O
    pub target_row_group_size: usize,
}

/// Compression strategy per column type
#[derive(Debug, Clone)]
pub struct CompressionStrategy {
    /// Compression for FP32 vector columns
    pub fp32_vectors: CompressionAlgorithm,

    /// Compression for quantized vector columns
    pub quantized_vectors: CompressionAlgorithm,

    /// Compression for metadata columns
    pub metadata_columns: CompressionAlgorithm,

    /// Compression for ID columns
    pub id_columns: CompressionAlgorithm,

    /// No compression for binary sketches (already compact)
    pub binary_sketches: Option<CompressionAlgorithm>,
}

impl Default for CompressionStrategy {
    fn default() -> Self {
        Self {
            // ZSTD for FP32 vectors - best compression ratio
            fp32_vectors: CompressionAlgorithm::Zstd,

            // LZ4 for quantized vectors - fast decompression, already compact
            quantized_vectors: CompressionAlgorithm::Lz4,

            // ZSTD for metadata - excellent text compression
            metadata_columns: CompressionAlgorithm::Zstd,

            // Dictionary + ZSTD for IDs - handles repeated patterns
            id_columns: CompressionAlgorithm::Zstd,

            // No compression for binary sketches - they're already bit-packed
            binary_sketches: None,
        }
    }
}

impl Default for SchemaOptimization {
    fn default() -> Self {
        Self {
            enable_dictionary_encoding: true,
            optimize_nullability: true,
            optimize_timestamp_precision: true,
            enable_fixed_size_binary: true,
            target_row_group_size: 50_000,
        }
    }
}

/// Unified schema builder for NOVA and NOVA engines
pub struct ColumnarSchemaBuilder {
    /// Schema cache to avoid regenerating identical schemas
    schema_cache: Arc<RwLock<HashMap<String, CachedSchema>>>,

    /// Default optimization settings
    #[allow(dead_code)]
    default_optimization: SchemaOptimization,

    /// Default compression strategy
    #[allow(dead_code)]
    default_compression: CompressionStrategy,
}

/// Cached schema with expiration
#[derive(Debug, Clone)]
struct CachedSchema {
    schema: Arc<Schema>,
    compression_metadata: ColumnarSchemaCompressionMetadata,
    timestamp: std::time::Instant,
    ttl: std::time::Duration,
}

/// Backwards-compat alias for [`ColumnarSchemaCompressionMetadata`].
pub type CompressionMetadata = ColumnarSchemaCompressionMetadata;

/// Compression metadata for schema
#[derive(Debug, Clone)]
pub struct ColumnarSchemaCompressionMetadata {
    /// Compression settings per column
    pub column_compression: HashMap<String, CompressionAlgorithm>,

    /// Expected compression ratios
    pub compression_ratios: HashMap<String, f32>,

    /// Parquet writer properties
    pub writer_properties: WriterPropertiesConfig,
}

/// Parquet writer properties configuration
#[derive(Debug, Clone)]
pub struct WriterPropertiesConfig {
    pub row_group_size: usize,
    pub page_size: usize,
    pub dictionary_enabled: bool,
    pub statistics_enabled: bool,
    pub bloom_filter_enabled: bool,
}

impl Default for WriterPropertiesConfig {
    fn default() -> Self {
        Self {
            row_group_size: DEFAULT_ROW_GROUP_SIZE,
            page_size: DEFAULT_PAGE_SIZE,
            dictionary_enabled: true,
            statistics_enabled: true,
            bloom_filter_enabled: true,
        }
    }
}

impl ColumnarSchemaBuilder {
    /// Create new schema builder
    pub fn new() -> Self {
        Self {
            schema_cache: Arc::new(RwLock::new(HashMap::new())),
            default_optimization: SchemaOptimization::default(),
            default_compression: CompressionStrategy::default(),
        }
    }

    /// Create schema builder with custom defaults
    pub fn with_defaults(
        optimization: SchemaOptimization,
        compression: CompressionStrategy,
    ) -> Self {
        Self {
            schema_cache: Arc::new(RwLock::new(HashMap::new())),
            default_optimization: optimization,
            default_compression: compression,
        }
    }

    /// Build optimized schema for collection with automatic quantization detection
    pub async fn build_schema(
        &self,
        collection_id: &str,
        config: &ColumnarSchemaConfig,
    ) -> Result<(Arc<Schema>, ColumnarSchemaCompressionMetadata)> {
        // Generate cache key based on configuration
        let cache_key = self.generate_cache_key(collection_id, config);

        // Check cache first
        if let Some(cached) = self.get_cached_schema(&cache_key).await
            && !cached.is_expired()
        {
            debug!("Schema cache hit for collection: {}", collection_id);
            return Ok((cached.schema, cached.compression_metadata));
        }

        info!(
            "Building quantization-aware schema for collection: {} (dim: {})",
            collection_id, config.dimension
        );

        let (schema, compression_metadata) = self.build_schema_internal(config)?;

        // Cache the result
        self.cache_schema(cache_key, schema.clone(), compression_metadata.clone())
            .await;

        info!(
            "Schema built with {} fields, {} quantized columns",
            schema.fields().len(),
            self.count_quantized_columns(&schema)
        );

        Ok((schema, compression_metadata))
    }

    /// Build schema with automatic quantization column generation
    fn build_schema_internal(
        &self,
        config: &ColumnarSchemaConfig,
    ) -> Result<(Arc<Schema>, ColumnarSchemaCompressionMetadata)> {
        let mut fields = Vec::new();
        let mut column_compression = HashMap::new();
        let mut compression_ratios = HashMap::new();

        // Core fields - always present
        self.add_core_fields(
            &mut fields,
            &mut column_compression,
            &mut compression_ratios,
            config,
        )?;

        // Vector field - FP32 baseline
        self.add_vector_field(
            &mut fields,
            &mut column_compression,
            &mut compression_ratios,
            config,
        )?;

        // Quantized vector fields - automatic based on QuantizationConfig
        if let Some(ref quant_config) = config.quantization {
            self.add_quantized_fields(
                &mut fields,
                &mut column_compression,
                &mut compression_ratios,
                config,
                quant_config,
            )?;
        }

        // Filterable metadata columns
        self.add_filterable_fields(
            &mut fields,
            &mut column_compression,
            &mut compression_ratios,
            config,
        )?;

        // Extra metadata field for non-filterable data
        self.add_extra_metadata_field(
            &mut fields,
            &mut column_compression,
            &mut compression_ratios,
            config,
        )?;

        let schema = Arc::new(Schema::new(fields));
        let compression_metadata = ColumnarSchemaCompressionMetadata {
            column_compression,
            compression_ratios,
            writer_properties: WriterPropertiesConfig::default(),
        };

        Ok((schema, compression_metadata))
    }

    /// Add core fields (ID, timestamp, version, etc.)
    fn add_core_fields(
        &self,
        fields: &mut Vec<Field>,
        column_compression: &mut HashMap<String, CompressionAlgorithm>,
        compression_ratios: &mut HashMap<String, f32>,
        config: &ColumnarSchemaConfig,
    ) -> Result<()> {
        // ID field - required for customer APIs
        fields.push(Field::new("id", DataType::Utf8, false));
        column_compression.insert("id".to_string(), config.compression_strategy.id_columns);
        compression_ratios.insert("id".to_string(), 2.0); // Good compression for UUIDs

        // Timestamp fields with optimized precision
        let timestamp_type = if config.optimization.optimize_timestamp_precision {
            DataType::Timestamp(TimeUnit::Millisecond, None) // Sufficient for most use cases
        } else {
            DataType::Timestamp(TimeUnit::Microsecond, None)
        };

        fields.push(Field::new("timestamp", timestamp_type.clone(), false));
        fields.push(Field::new("created_at", timestamp_type.clone(), true));
        fields.push(Field::new("updated_at", timestamp_type, true));

        // No compression for timestamps - they're already compact
        for field in ["timestamp", "created_at", "updated_at"] {
            compression_ratios.insert(field.to_string(), 1.0);
        }

        // Version field for MVCC
        fields.push(Field::new("version", DataType::Int64, true));
        compression_ratios.insert("version".to_string(), 1.5);

        // Expiration field for TTL
        fields.push(Field::new(
            "expires_at",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            true,
        ));
        compression_ratios.insert("expires_at".to_string(), 1.0);

        debug!("Added {} core fields", fields.len());
        Ok(())
    }

    /// Add FP32 vector field with optimal configuration
    fn add_vector_field(
        &self,
        fields: &mut Vec<Field>,
        column_compression: &mut HashMap<String, CompressionAlgorithm>,
        compression_ratios: &mut HashMap<String, f32>,
        config: &ColumnarSchemaConfig,
    ) -> Result<()> {
        let vector_type = if config.optimization.enable_fixed_size_binary {
            // Fixed-size binary for better performance and compression
            DataType::FixedSizeBinary(config.dimension as i32 * 4)
        } else {
            // FixedSizeList for known dimension vectors
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                config.dimension as i32,
            )
        };

        fields.push(Field::new("vector", vector_type, false));
        column_compression.insert(
            "vector".to_string(),
            config.compression_strategy.fp32_vectors,
        );

        // ZSTD typically achieves 2-4x compression on float vectors
        compression_ratios.insert("vector".to_string(), 3.0);

        trace!("Added FP32 vector field (dim: {})", config.dimension);
        Ok(())
    }

    /// Add quantized vector fields based on QuantizationConfig
    fn add_quantized_fields(
        &self,
        fields: &mut Vec<Field>,
        column_compression: &mut HashMap<String, CompressionAlgorithm>,
        compression_ratios: &mut HashMap<String, f32>,
        config: &ColumnarSchemaConfig,
        quant_config: &QuantizationConfig,
    ) -> Result<()> {
        let mut quantized_count = 0;

        // Binary quantization - ultra-fast filtering
        if quant_config.enable_binary.unwrap_or(false) {
            let binary_size = config.dimension.div_ceil(8); // Bits to bytes
            fields.push(Field::new(
                "vector_binary",
                DataType::FixedSizeBinary(binary_size as i32),
                true, // Nullable for progressive rollout
            ));

            // Binary sketches are already maximally compressed
            if let Some(compression) = config.compression_strategy.binary_sketches {
                column_compression.insert("vector_binary".to_string(), compression);
                compression_ratios.insert("vector_binary".to_string(), 1.2);
            } else {
                compression_ratios.insert("vector_binary".to_string(), 1.0);
            }

            quantized_count += 1;
            trace!("Added binary quantization field ({} bytes)", binary_size);
        }

        // INT8 quantization - good balance of speed and quality
        if quant_config.enable_int8.unwrap_or(false) {
            fields.push(Field::new(
                "vector_int8",
                DataType::FixedSizeBinary(config.dimension as i32),
                true,
            ));
            fields.push(Field::new("int8_scale", DataType::Float32, true));
            fields.push(Field::new("int8_zero_point", DataType::Int8, true));

            column_compression.insert(
                "vector_int8".to_string(),
                config.compression_strategy.quantized_vectors,
            );
            compression_ratios.insert("vector_int8".to_string(), 1.5); // INT8 compresses moderately
            compression_ratios.insert("int8_scale".to_string(), 1.0);
            compression_ratios.insert("int8_zero_point".to_string(), 1.0);

            quantized_count += 1;
            trace!("Added INT8 quantization fields");
        }

        // Product Quantization - configurable precision
        if quant_config.enable_pq.unwrap_or(false) {
            let pq_size = quant_config.pq_segments.unwrap_or(1) as i32;
            fields.push(Field::new(
                "vector_pq",
                DataType::FixedSizeBinary(pq_size),
                true,
            ));

            // PQ codes benefit from dictionary encoding
            column_compression.insert(
                "vector_pq".to_string(),
                config.compression_strategy.quantized_vectors,
            );
            compression_ratios.insert("vector_pq".to_string(), 2.0); // Good compression for PQ codes

            quantized_count += 1;
            trace!(
                "Added PQ quantization field ({:?} segments)",
                quant_config.pq_segments
            );
        }

        info!("Added {} quantized vector columns", quantized_count);
        Ok(())
    }

    /// Add filterable metadata columns
    fn add_filterable_fields(
        &self,
        fields: &mut Vec<Field>,
        column_compression: &mut HashMap<String, CompressionAlgorithm>,
        compression_ratios: &mut HashMap<String, f32>,
        config: &ColumnarSchemaConfig,
    ) -> Result<()> {
        for filterable in &config.filterable_columns {
            // Convert our local FilterableData enum to Arrow DataType
            let data_type = filterable.data_type.to_arrow_type();

            // Enable dictionary encoding for low-cardinality strings
            let field = if config.optimization.enable_dictionary_encoding
                && filterable.data_type == FilterableData::String
                && filterable.estimated_cardinality.is_some_and(|c| c < 10000)
            {
                Field::new(&filterable.name, data_type, filterable.nullable).with_metadata(
                    HashMap::from([("encoding".to_string(), "dictionary".to_string())]),
                )
            } else {
                Field::new(&filterable.name, data_type, filterable.nullable)
            };

            fields.push(field);
            column_compression.insert(
                filterable.name.clone(),
                config.compression_strategy.metadata_columns,
            );

            // Estimate compression ratio based on data type
            let ratio = match &filterable.data_type {
                FilterableData::String => 3.0,         // Text compresses well
                FilterableData::Json => 4.0,           // JSON compresses very well
                FilterableData::Array(_) => 2.5,       // Arrays have moderate compression
                FilterableData::Text(_) => 4.0,        // Large text compresses very well
                FilterableData::TextLarge => 4.5,      // Very large text compresses excellently
                FilterableData::Decimal { .. } => 1.5, // Decimals have moderate compression
                FilterableData::Uuid => 1.2,           // UUIDs are already dense
                FilterableData::Binary => 1.0,         // Binary is opaque
                FilterableData::GeoPoint => 1.3,       // Geo data has some compression
                FilterableData::GeoPolygon => 2.0,     // Polygon lists compress better
                _ => 1.5,                              // Numbers compress moderately
            };
            compression_ratios.insert(filterable.name.clone(), ratio);
        }

        debug!(
            "Added {} filterable columns",
            config.filterable_columns.len()
        );
        Ok(())
    }

    /// Add extra metadata field for non-filterable data
    fn add_extra_metadata_field(
        &self,
        fields: &mut Vec<Field>,
        column_compression: &mut HashMap<String, CompressionAlgorithm>,
        compression_ratios: &mut HashMap<String, f32>,
        config: &ColumnarSchemaConfig,
    ) -> Result<()> {
        // Key-value pairs for arbitrary metadata
        let kv_struct = DataType::Struct(
            vec![
                Field::new("key", DataType::Utf8, false),
                Field::new("value", DataType::Utf8, false),
            ]
            .into(),
        );

        fields.push(Field::new(
            "extra_metadata_info",
            DataType::List(Arc::new(Field::new("item", kv_struct, true))),
            true,
        ));

        column_compression.insert(
            "extra_metadata_info".to_string(),
            config.compression_strategy.metadata_columns,
        );
        compression_ratios.insert("extra_metadata_info".to_string(), 4.0); // JSON-like data compresses well

        trace!("Added extra metadata field");
        Ok(())
    }

    /// Convert filterable data type to Arrow data type
    #[allow(dead_code)]
    fn convert_filterable_type(&self, data_type: &FilterableData) -> Result<DataType> {
        // Use the to_arrow_type method which handles all types
        Ok(data_type.to_arrow_type())
    }

    /// Generate cache key for schema configuration
    fn generate_cache_key(&self, collection_id: &str, config: &ColumnarSchemaConfig) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        collection_id.hash(&mut hasher);
        config.dimension.hash(&mut hasher);

        if let Some(ref quant) = config.quantization {
            quant.enable_binary.hash(&mut hasher);
            quant.enable_int8.hash(&mut hasher);
            quant.enable_pq.hash(&mut hasher);
            quant.pq_segments.hash(&mut hasher);
        }

        for col in &config.filterable_columns {
            col.name.hash(&mut hasher);
        }

        format!("schema_{}_{:x}", collection_id, hasher.finish())
    }

    /// Get cached schema if valid
    async fn get_cached_schema(&self, cache_key: &str) -> Option<CachedSchema> {
        let cache = self.schema_cache.read().await;
        cache.get(cache_key).cloned()
    }

    /// Cache schema with TTL
    async fn cache_schema(
        &self,
        cache_key: String,
        schema: Arc<Schema>,
        compression_metadata: ColumnarSchemaCompressionMetadata,
    ) {
        let cached = CachedSchema {
            schema,
            compression_metadata,
            timestamp: std::time::Instant::now(),
            ttl: std::time::Duration::from_secs(3600), // 1 hour TTL
        };

        let mut cache = self.schema_cache.write().await;
        cache.insert(cache_key, cached);
    }

    /// Count quantized columns in schema
    fn count_quantized_columns(&self, schema: &Schema) -> usize {
        schema
            .fields()
            .iter()
            .filter(|field| field.name().starts_with("vector_"))
            .count()
    }

    /// Clear cache for collection
    pub async fn clear_cache(&self, collection_id: &str) {
        let mut cache = self.schema_cache.write().await;
        cache.retain(|key, _| !key.contains(collection_id));
        debug!("Cleared schema cache for collection: {}", collection_id);
    }

    /// Get cache statistics
    pub async fn cache_stats(&self) -> (usize, usize) {
        let cache = self.schema_cache.read().await;
        let total = cache.len();
        let expired = cache.values().filter(|cached| cached.is_expired()).count();
        (total, expired)
    }
}

impl CachedSchema {
    /// Check if cached schema has expired
    fn is_expired(&self) -> bool {
        self.timestamp.elapsed() > self.ttl
    }
}

impl ColumnarFilterableSpec {
    /// Convert from proto FilterableColumnSpec to our internal type
    pub fn from_proto(proto: &proximadb_proto::proximadb_v1::FilterableColumnSpec) -> Self {
        let data_type = match proto.data_type {
            x if x
                == proximadb_proto::proximadb_v1::FilterableDataType::FilterableString as i32 =>
            {
                FilterableData::String
            }
            x if x
                == proximadb_proto::proximadb_v1::FilterableDataType::FilterableInteger as i32 =>
            {
                FilterableData::Integer
            }
            x if x == proximadb_proto::proximadb_v1::FilterableDataType::FilterableFloat as i32 => {
                FilterableData::Float
            }
            x if x
                == proximadb_proto::proximadb_v1::FilterableDataType::FilterableBoolean as i32 =>
            {
                FilterableData::Boolean
            }
            x if x
                == proximadb_proto::proximadb_v1::FilterableDataType::FilterableDatetime as i32 =>
            {
                FilterableData::Datetime
            }
            _ => FilterableData::String, // Default to string
        };

        Self {
            name: proto.name.clone(),
            data_type,
            nullable: true, // Default to nullable - any column could be nullable
            indexed: proto.indexed,
            estimated_cardinality: proto.estimated_cardinality.map(|c| c as usize),
        }
    }

    /// Convert a list of proto specs to internal specs
    pub fn from_proto_vec(
        protos: &[proximadb_proto::proximadb_v1::FilterableColumnSpec],
    ) -> Vec<Self> {
        protos.iter().map(Self::from_proto).collect()
    }
}

impl FilterableData {
    /// Convert to Arrow DataType
    pub fn to_arrow_type(&self) -> DataType {
        match self {
            // Basic types (original)
            FilterableData::String => DataType::Utf8,
            FilterableData::Integer => DataType::Int64,
            FilterableData::Float => DataType::Float64,
            FilterableData::Boolean => DataType::Boolean,
            FilterableData::Datetime => {
                DataType::Timestamp(arrow_schema::TimeUnit::Millisecond, None)
            }
            FilterableData::Array(inner) => {
                let inner_type = inner.to_arrow_type();
                DataType::List(Arc::new(Field::new("item", inner_type, true)))
            }
            FilterableData::Json => DataType::Utf8, // Store JSON as string
            FilterableData::Map => {
                // Native Parquet Map<String, String>
                DataType::Map(
                    Arc::new(Field::new(
                        "entries",
                        DataType::Struct(
                            vec![
                                Field::new("key", DataType::Utf8, false),
                                Field::new("value", DataType::Utf8, true),
                            ]
                            .into(),
                        ),
                        false,
                    )),
                    false,
                )
            }

            // Rich text types (NEW)
            FilterableData::Text(_) => DataType::LargeUtf8,
            FilterableData::TextLarge => DataType::LargeUtf8,

            // Precision numeric types (NEW)
            FilterableData::Decimal { precision, scale } => {
                DataType::Decimal128(*precision, *scale as i8)
            }

            // Temporal types (NEW)
            FilterableData::TimestampTz { timezone } => {
                DataType::Timestamp(TimeUnit::Microsecond, Some(timezone.clone().into()))
            }
            FilterableData::Date => DataType::Date32,
            FilterableData::Time => DataType::Time64(TimeUnit::Microsecond),

            // Identifier types (NEW)
            FilterableData::Uuid => DataType::FixedSizeBinary(16),

            // Binary types (NEW)
            FilterableData::Binary => DataType::Binary,

            // Geospatial types (NEW)
            FilterableData::GeoPoint => DataType::Struct(
                vec![
                    Field::new("latitude", DataType::Float64, false),
                    Field::new("longitude", DataType::Float64, false),
                    Field::new("altitude", DataType::Float64, true),
                ]
                .into(),
            ),
            FilterableData::GeoPolygon => DataType::List(Arc::new(Field::new(
                "point",
                DataType::Struct(
                    vec![
                        Field::new("latitude", DataType::Float64, false),
                        Field::new("longitude", DataType::Float64, false),
                    ]
                    .into(),
                ),
                false,
            ))),
        }
    }

    /// Check if this type is a TEXT variant
    pub fn is_text(&self) -> bool {
        matches!(self, FilterableData::Text(_) | FilterableData::TextLarge)
    }

    /// Check if this type supports range queries
    pub fn supports_range(&self) -> bool {
        matches!(
            self,
            FilterableData::Integer
                | FilterableData::Float
                | FilterableData::Decimal { .. }
                | FilterableData::Datetime
                | FilterableData::TimestampTz { .. }
                | FilterableData::Date
                | FilterableData::Time
        )
    }

    /// Check if this type supports fulltext search
    pub fn supports_fulltext(&self) -> bool {
        matches!(self, FilterableData::Text(_) | FilterableData::TextLarge)
    }

    /// Get storage size hint in bytes (0 for variable size)
    pub fn storage_size_hint(&self) -> usize {
        match self {
            FilterableData::Integer => 8,
            FilterableData::Float => 8,
            FilterableData::Decimal { .. } => 16,
            FilterableData::Boolean => 1,
            FilterableData::Datetime => 8,
            FilterableData::TimestampTz { .. } => 8,
            FilterableData::Date => 4,
            FilterableData::Time => 8,
            FilterableData::Uuid => 16,
            FilterableData::GeoPoint => 24,
            _ => 0, // Variable size
        }
    }
}

impl Default for ColumnarSchemaBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to create schema from collection metadata
pub async fn create_schema_from_collection(
    collection_id: &str,
    dimension: usize,
    quantization: Option<&QuantizationConfig>,
    filterable_columns: &[ColumnarFilterableSpec],
) -> Result<(Arc<Schema>, ColumnarSchemaCompressionMetadata)> {
    let builder = ColumnarSchemaBuilder::new();

    let config = ColumnarSchemaConfig {
        dimension,
        quantization: quantization.cloned(),
        filterable_columns: filterable_columns.to_vec(),
        optimization: SchemaOptimization::default(),
        compression_strategy: CompressionStrategy::default(),
    };

    builder.build_schema(collection_id, &config).await
}

/// Validate schema compatibility with quantization config
pub fn validate_schema_compatibility(
    schema: &Schema,
    quantization: &QuantizationConfig,
) -> Result<()> {
    // Check that required quantized columns exist
    if quantization.enable_binary.unwrap_or(false)
        && schema.field_with_name("vector_binary").is_err()
    {
        return Err(anyhow::anyhow!(
            "Binary quantization enabled but vector_binary column missing"
        ));
    }

    if quantization.enable_int8.unwrap_or(false) && schema.field_with_name("vector_int8").is_err() {
        return Err(anyhow::anyhow!(
            "INT8 quantization enabled but vector_int8 column missing"
        ));
    }

    if quantization.enable_pq.unwrap_or(false) && schema.field_with_name("vector_pq").is_err() {
        return Err(anyhow::anyhow!(
            "PQ quantization enabled but vector_pq column missing"
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_schema_generation_without_quantization() {
        let builder = ColumnarSchemaBuilder::new();

        let config = ColumnarSchemaConfig {
            dimension: 768,
            quantization: None,
            filterable_columns: vec![ColumnarFilterableSpec {
                name: "category".to_string(),
                data_type: FilterableData::String,
                nullable: true,
                indexed: false,
                estimated_cardinality: Some(100),
            }],
            optimization: SchemaOptimization::default(),
            compression_strategy: CompressionStrategy::default(),
        };

        let (schema, metadata) = builder
            .build_schema("test_collection", &config)
            .await
            .unwrap();

        // Should have core fields + vector + filterable + extra_metadata
        assert!(schema.fields().len() >= 8);

        // Check core fields
        assert!(schema.field_with_name("id").is_ok());
        assert!(schema.field_with_name("vector").is_ok());
        assert!(schema.field_with_name("timestamp").is_ok());
        assert!(schema.field_with_name("category").is_ok());

        // Should not have quantized fields
        assert!(schema.field_with_name("vector_binary").is_err());
        assert!(schema.field_with_name("vector_int8").is_err());
        assert!(schema.field_with_name("vector_pq").is_err());

        // Check compression metadata
        assert!(metadata.column_compression.contains_key("vector"));
        assert!(metadata.compression_ratios.contains_key("vector"));
    }

    #[tokio::test]
    async fn test_schema_generation_with_quantization() {
        let builder = ColumnarSchemaBuilder::new();

        let quant_config = QuantizationConfig {
            enable_binary: Some(true),
            enable_int8: Some(true),
            enable_pq: Some(true),
            pq_segments: Some(16),
            ..Default::default()
        };

        let config = ColumnarSchemaConfig {
            dimension: 768,
            quantization: Some(quant_config),
            filterable_columns: vec![],
            optimization: SchemaOptimization::default(),
            compression_strategy: CompressionStrategy::default(),
        };

        let (schema, metadata) = builder
            .build_schema("test_collection", &config)
            .await
            .unwrap();

        // Should have quantized fields
        assert!(schema.field_with_name("vector_binary").is_ok());
        assert!(schema.field_with_name("vector_int8").is_ok());
        assert!(schema.field_with_name("int8_scale").is_ok());
        assert!(schema.field_with_name("int8_zero_point").is_ok());
        assert!(schema.field_with_name("vector_pq").is_ok());

        // Check compression strategies for quantized columns
        assert!(metadata.column_compression.contains_key("vector_int8"));
        assert!(metadata.column_compression.contains_key("vector_pq"));

        // Binary field might not have compression
        assert!(metadata.compression_ratios.contains_key("vector_binary"));
    }

    #[tokio::test]
    async fn test_schema_caching() {
        let builder = ColumnarSchemaBuilder::new();

        let config = ColumnarSchemaConfig {
            dimension: 512,
            quantization: None,
            filterable_columns: vec![],
            optimization: SchemaOptimization::default(),
            compression_strategy: CompressionStrategy::default(),
        };

        // First call - should build schema
        let (schema1, _) = builder.build_schema("cache_test", &config).await.unwrap();

        // Second call - should hit cache
        let (schema2, _) = builder.build_schema("cache_test", &config).await.unwrap();

        // Should be the same Arc instance (cached)
        assert!(Arc::ptr_eq(&schema1, &schema2));

        // Check cache stats
        let (total, expired) = builder.cache_stats().await;
        assert_eq!(total, 1);
        assert_eq!(expired, 0);
    }

    #[test]
    fn test_filterable_type_conversion() {
        let builder = ColumnarSchemaBuilder::new();

        assert!(matches!(
            builder
                .convert_filterable_type(&FilterableData::String)
                .unwrap(),
            DataType::Utf8
        ));

        assert!(matches!(
            builder
                .convert_filterable_type(&FilterableData::Integer)
                .unwrap(),
            DataType::Int64
        ));

        assert!(matches!(
            builder
                .convert_filterable_type(&FilterableData::Array(Box::new(FilterableData::String)))
                .unwrap(),
            DataType::List(_)
        ));
    }

    #[test]
    fn test_schema_validation() {
        let fields = vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("vector", DataType::FixedSizeBinary(768 * 4), false),
            Field::new("vector_binary", DataType::FixedSizeBinary(96), true),
            Field::new("vector_int8", DataType::FixedSizeBinary(768), true),
            Field::new("vector_pq", DataType::FixedSizeBinary(16), true),
        ];

        let schema = Schema::new(fields);

        let quant_config = QuantizationConfig {
            enable_binary: Some(true),
            enable_int8: Some(true),
            enable_pq: Some(true),
            ..Default::default()
        };

        // Should validate successfully
        validate_schema_compatibility(&schema, &quant_config).unwrap();

        // Test missing field
        let quant_config_missing = QuantizationConfig {
            enable_binary: Some(true),
            enable_int8: Some(true),
            enable_pq: Some(false),
            ..Default::default()
        };

        validate_schema_compatibility(&schema, &quant_config_missing).unwrap();
    }
}
