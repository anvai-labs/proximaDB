//! # ProximaDB TableProvider Trait
//!
//! Defines the ProximaTableProvider trait for engine-agnostic table access.
//! This trait extends DataFusion's TableProvider with ProximaDB-specific
//! capabilities for multi-engine vector database support.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                        PROXIMA TABLE PROVIDER TRAIT                          │
//! │  ┌───────────────────────────────────────────────────────────────────────┐  │
//! │  │  ProximaTableProvider (extends DataFusion TableProvider)              │  │
//! │  │  - engine_type(): EngineType                                          │  │
//! │  │  - collection_info(): DataFusionCollectionInfo                                  │  │
//! │  │  - get_splits(): Vec<FileSplit>                                       │  │
//! │  │  - pruning_stats(): PruningStatistics                                 │  │
//! │  └───────────────────────────────────────────────────────────────────────┘  │
//! │                                      │                                       │
//! │                                      ▼                                       │
//! │  ┌───────────────────────────────────────────────────────────────────────┐  │
//! │  │                      ENGINE IMPLEMENTATIONS                            │  │
//! │  │  SST │ HELIX │ SWIFT │ NOVA │ VIPER │ RAPTOR                          │  │
//! │  └───────────────────────────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## SOLID Principles
//!
//! - **S (Single Responsibility)**: TableProvider handles schema/metadata, SplitReader handles I/O
//! - **O (Open/Closed)**: New engines add implementations without modifying core
//! - **L (Liskov Substitution)**: All engines implement same ProximaTableProvider trait
//! - **I (Interface Segregation)**: Separate traits for table metadata and split reading
//! - **D (Dependency Inversion)**: ExecutionPlan depends on abstract SplitReader trait

use std::fmt::Debug;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::Result as DFResult;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;
use serde::{Deserialize, Serialize};

use crate::compute::quantization::QuantizationStrategy;
use crate::storage::formats::FileSplit;
use crate::storage::traits::StorageFormatStrategy;

// ============================================================================
// Engine Type Enumeration
// ============================================================================

/// Storage engine type enumeration.
///
/// Maps to ProximaDB's storage engine strategies for type-safe dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EngineType {
    /// SST: Sorted String Table - write-optimized, real-time queries
    Sst,
    /// HELIX: Hilbert curve locality-optimized storage
    Helix,
    /// SWIFT: Ultra-low latency with hierarchical superblocks
    Swift,
    /// NOVA: Progressive columnar with zone maps
    Nova,
    /// VIPER: Columnar Parquet for analytics
    Viper,
    /// RAPTOR: Adaptive row-group with dynamic tiering
    Raptor,
}

impl From<StorageFormatStrategy> for EngineType {
    fn from(strategy: StorageFormatStrategy) -> Self {
        match strategy {
            StorageFormatStrategy::Sst => EngineType::Sst,
            StorageFormatStrategy::Helix => EngineType::Helix,
            StorageFormatStrategy::Swift => EngineType::Swift,
            StorageFormatStrategy::Nova => EngineType::Nova,
            StorageFormatStrategy::Viper => EngineType::Viper,
            StorageFormatStrategy::Raptor => EngineType::Raptor,
            StorageFormatStrategy::Hybrid => EngineType::Viper, // Default to VIPER for hybrid
            // Specialized engines have no dedicated DataFusion reader yet; route them to
            // the columnar/time-series readers that best match their on-disk layout.
            StorageFormatStrategy::TimeSeries => EngineType::Helix, // HELIX is time-series
            StorageFormatStrategy::Chrono => EngineType::Helix,     // temporal -> time-series
            StorageFormatStrategy::Cedar => EngineType::Viper,      // multimodal -> columnar
        }
    }
}

impl From<EngineType> for StorageFormatStrategy {
    fn from(engine: EngineType) -> Self {
        match engine {
            EngineType::Sst => StorageFormatStrategy::Sst,
            EngineType::Helix => StorageFormatStrategy::Helix,
            EngineType::Swift => StorageFormatStrategy::Swift,
            EngineType::Nova => StorageFormatStrategy::Nova,
            EngineType::Viper => StorageFormatStrategy::Viper,
            EngineType::Raptor => StorageFormatStrategy::Raptor,
        }
    }
}

impl std::fmt::Display for EngineType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineType::Sst => write!(f, "SST"),
            EngineType::Helix => write!(f, "HELIX"),
            EngineType::Swift => write!(f, "SWIFT"),
            EngineType::Nova => write!(f, "NOVA"),
            EngineType::Viper => write!(f, "VIPER"),
            EngineType::Raptor => write!(f, "RAPTOR"),
        }
    }
}

// ============================================================================
// Collection Information
// ============================================================================

/// Collection metadata for table provider operations.
///
/// Contains essential information about a ProximaDB collection
/// needed for query planning and execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFusionCollectionInfo {
    /// Collection name/identifier
    pub name: String,
    /// Vector dimension
    pub dimension: usize,
    /// Total vector count (approximate)
    pub vector_count: u64,
    /// Storage engine type
    pub engine_type: EngineType,
    /// Quantization type if enabled
    pub quantization: Option<QuantizationStrategy>,
    /// Total storage size in bytes
    pub storage_size_bytes: u64,
    /// Number of data files
    pub file_count: usize,
    /// Base storage path
    pub base_path: String,
    /// Creation timestamp (millis since epoch)
    pub created_at_ms: i64,
    /// Last modified timestamp (millis since epoch)
    pub updated_at_ms: i64,
}

/// Backwards-compat alias: adapters and `mod.rs` refer to this type as
/// `CollectionInfo`. The canonical name is [`DataFusionCollectionInfo`].
pub type CollectionInfo = DataFusionCollectionInfo;

impl DataFusionCollectionInfo {
    /// Create a new DataFusionCollectionInfo with required fields.
    pub fn new(name: String, dimension: usize, engine_type: EngineType) -> Self {
        let now = chrono::Utc::now().timestamp_millis();
        Self {
            name,
            dimension,
            vector_count: 0,
            engine_type,
            quantization: None,
            storage_size_bytes: 0,
            file_count: 0,
            base_path: String::new(),
            created_at_ms: now,
            updated_at_ms: now,
        }
    }

    /// Builder method to set vector count.
    pub fn with_vector_count(mut self, count: u64) -> Self {
        self.vector_count = count;
        self
    }

    /// Builder method to set quantization type.
    pub fn with_quantization(mut self, quant: QuantizationStrategy) -> Self {
        self.quantization = Some(quant);
        self
    }

    /// Builder method to set storage size.
    pub fn with_storage_size(mut self, size: u64) -> Self {
        self.storage_size_bytes = size;
        self
    }

    /// Builder method to set file count.
    pub fn with_file_count(mut self, count: usize) -> Self {
        self.file_count = count;
        self
    }

    /// Builder method to set base path.
    pub fn with_base_path(mut self, path: String) -> Self {
        self.base_path = path;
        self
    }

    /// Calculate average vector size in bytes.
    pub fn avg_vector_size_bytes(&self) -> usize {
        // Assume float32 vectors: 4 bytes per dimension
        self.dimension * 4
    }

    /// Estimate total vector data size (excluding metadata).
    pub fn estimated_vector_data_size(&self) -> u64 {
        self.vector_count * self.avg_vector_size_bytes() as u64
    }
}

// ============================================================================
// Pruning Statistics
// ============================================================================

/// Statistics for split/partition pruning during query planning.
///
/// These statistics enable the query optimizer to skip splits
/// that cannot possibly match the query predicates.
#[derive(Debug, Clone, Default)]
pub struct PruningStatistics {
    /// Total number of splits
    pub total_splits: usize,
    /// Splits with min/max statistics
    pub splits_with_stats: usize,
    /// Splits with bloom filters
    pub splits_with_bloom: usize,
    /// Splits with centroid information (for vector pruning)
    pub splits_with_centroid: usize,
    /// Column-level statistics availability
    pub columns_with_stats: Vec<String>,
    /// Estimated pruning effectiveness (0.0 - 1.0)
    pub estimated_pruning_ratio: f32,
}

impl PruningStatistics {
    /// Create empty pruning statistics.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Create pruning statistics from splits.
    pub fn from_splits(splits: &[FileSplit]) -> Self {
        let total_splits = splits.len();
        let mut splits_with_stats = 0;
        let mut splits_with_bloom = 0;
        let mut splits_with_centroid = 0;
        let mut columns_with_stats = std::collections::HashSet::new();

        for split in splits {
            // Check for row group metadata
            if split.statistics.row_count.is_some() {
                splits_with_stats += 1;
            }

            // Check for bloom filter
            if split.statistics.bloom_filter.is_some() {
                splits_with_bloom += 1;
            }

            // Check for centroid
            if split.statistics.centroid.is_some() {
                splits_with_centroid += 1;
            }

            // Collect columns with stats
            for col_name in split.statistics.column_stats.keys() {
                columns_with_stats.insert(col_name.clone());
            }
        }

        // Estimate pruning effectiveness based on available statistics
        let stats_coverage = if total_splits > 0 {
            splits_with_stats as f32 / total_splits as f32
        } else {
            0.0
        };

        let bloom_coverage = if total_splits > 0 {
            splits_with_bloom as f32 / total_splits as f32
        } else {
            0.0
        };

        // Weighted average: stats are more valuable than bloom filters
        let estimated_pruning_ratio = (stats_coverage * 0.6 + bloom_coverage * 0.4).min(1.0);

        Self {
            total_splits,
            splits_with_stats,
            splits_with_bloom,
            splits_with_centroid,
            columns_with_stats: columns_with_stats.into_iter().collect(),
            estimated_pruning_ratio,
        }
    }

    /// Check if pruning is likely to be effective.
    pub fn is_pruning_effective(&self) -> bool {
        self.estimated_pruning_ratio > 0.1 && self.splits_with_stats > 0
    }
}

// ============================================================================
// ProximaTableProvider Trait
// ============================================================================

/// ProximaDB-specific table provider trait.
///
/// Extends DataFusion's TableProvider with ProximaDB-specific capabilities
/// for multi-engine vector database support.
///
/// ## Implementation Guide
///
/// Each storage engine should implement this trait to enable SQL queries:
///
/// ```rust,ignore
/// impl ProximaTableProvider for SstTableProvider {
///     fn engine_type(&self) -> EngineType { EngineType::Sst }
///     fn collection_info(&self) -> &DataFusionCollectionInfo { &self.info }
///     async fn get_splits(&self, filters: &[Expr]) -> Result<Vec<FileSplit>> {
///         // Engine-specific split generation with filter pushdown
///     }
/// }
/// ```
#[async_trait]
pub trait ProximaTableProvider: TableProvider + Send + Sync + Debug {
    /// Get the storage engine type.
    ///
    /// Used for engine-specific optimizations and statistics collection.
    fn engine_type(&self) -> EngineType;

    /// Get collection metadata.
    ///
    /// Returns essential information about the collection for query planning.
    fn collection_info(&self) -> &DataFusionCollectionInfo;

    /// Get file splits for distributed execution.
    ///
    /// Splits are independent units of work that can be processed in parallel.
    /// The filters parameter enables predicate pushdown to reduce splits.
    ///
    /// # Arguments
    /// * `filters` - DataFusion expressions that can be used for pruning
    ///
    /// # Returns
    /// * Vector of FileSplit objects for parallel reading
    async fn get_splits(&self, filters: &[Expr]) -> DFResult<Vec<FileSplit>>;

    /// Get pruning statistics for query optimization.
    ///
    /// Returns information about the availability and effectiveness
    /// of various pruning techniques.
    fn pruning_stats(&self) -> Option<PruningStatistics>;

    /// Check if vector search is supported.
    ///
    /// Returns true if this provider supports vector similarity search.
    fn supports_vector_search(&self) -> bool {
        self.collection_info().dimension > 0
    }

    /// Get the vector column name (if applicable).
    ///
    /// Returns the name of the vector column for vector search operations.
    fn vector_column_name(&self) -> Option<&str> {
        if self.supports_vector_search() {
            Some("vector")
        } else {
            None
        }
    }

    /// Get engine-specific capabilities.
    ///
    /// Returns a map of capability names to boolean values.
    fn capabilities(&self) -> std::collections::HashMap<String, bool> {
        let mut caps = std::collections::HashMap::new();
        caps.insert("vector_search".to_string(), self.supports_vector_search());
        caps.insert(
            "predicate_pushdown".to_string(),
            self.pruning_stats()
                .map(|s| s.splits_with_stats > 0)
                .unwrap_or(false),
        );
        caps.insert(
            "bloom_filter".to_string(),
            self.pruning_stats()
                .map(|s| s.splits_with_bloom > 0)
                .unwrap_or(false),
        );
        caps
    }
}

// ============================================================================
// Null Implementation for Testing
// ============================================================================

/// Null implementation of ProximaTableProvider for testing.
#[derive(Debug)]
pub struct NullProximaTableProvider {
    schema: SchemaRef,
    info: DataFusionCollectionInfo,
}

impl NullProximaTableProvider {
    /// Create a new null provider for testing.
    pub fn new(schema: SchemaRef, info: DataFusionCollectionInfo) -> Self {
        Self { schema, info }
    }
}

#[async_trait]
impl TableProvider for NullProximaTableProvider {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        _projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        // Return empty execution plan for testing
        Err(datafusion::error::DataFusionError::NotImplemented(
            "NullProximaTableProvider does not support scanning".to_string(),
        ))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        Ok(vec![
            TableProviderFilterPushDown::Unsupported;
            filters.len()
        ])
    }
}

#[async_trait]
impl ProximaTableProvider for NullProximaTableProvider {
    fn engine_type(&self) -> EngineType {
        self.info.engine_type
    }

    fn collection_info(&self) -> &DataFusionCollectionInfo {
        &self.info
    }

    async fn get_splits(&self, _filters: &[Expr]) -> DFResult<Vec<FileSplit>> {
        Ok(vec![])
    }

    fn pruning_stats(&self) -> Option<PruningStatistics> {
        Some(PruningStatistics::empty())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field, Schema};

    fn test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("vector", DataType::FixedSizeBinary(512), false),
            Field::new("metadata", DataType::Utf8, true),
        ]))
    }

    #[test]
    fn test_engine_type_conversion() {
        assert_eq!(
            EngineType::from(StorageFormatStrategy::Sst),
            EngineType::Sst
        );
        assert_eq!(
            EngineType::from(StorageFormatStrategy::Viper),
            EngineType::Viper
        );
        assert_eq!(
            StorageFormatStrategy::from(EngineType::Nova),
            StorageFormatStrategy::Nova
        );
    }

    #[test]
    fn test_engine_type_display() {
        assert_eq!(format!("{}", EngineType::Sst), "SST");
        assert_eq!(format!("{}", EngineType::Helix), "HELIX");
        assert_eq!(format!("{}", EngineType::Viper), "VIPER");
    }

    #[test]
    fn test_collection_info_creation() {
        let info =
            DataFusionCollectionInfo::new("test_collection".to_string(), 128, EngineType::Sst)
                .with_vector_count(1000)
                .with_storage_size(1024 * 1024)
                .with_file_count(5)
                .with_base_path("/data/test".to_string());

        assert_eq!(info.name, "test_collection");
        assert_eq!(info.dimension, 128);
        assert_eq!(info.vector_count, 1000);
        assert_eq!(info.engine_type, EngineType::Sst);
        assert_eq!(info.storage_size_bytes, 1024 * 1024);
        assert_eq!(info.file_count, 5);
        assert_eq!(info.base_path, "/data/test");
    }

    #[test]
    fn test_collection_info_vector_size() {
        let info = DataFusionCollectionInfo::new("test".to_string(), 768, EngineType::Viper);
        assert_eq!(info.avg_vector_size_bytes(), 768 * 4);
    }

    #[test]
    fn test_collection_info_estimated_size() {
        let info = DataFusionCollectionInfo::new("test".to_string(), 128, EngineType::Nova)
            .with_vector_count(1000);
        assert_eq!(info.estimated_vector_data_size(), 1000 * 128 * 4);
    }

    #[test]
    fn test_pruning_statistics_empty() {
        let stats = PruningStatistics::empty();
        assert_eq!(stats.total_splits, 0);
        assert!(!stats.is_pruning_effective());
    }

    #[test]
    fn test_pruning_statistics_from_splits() {
        use crate::storage::formats::splits::{FileSplit, SplitStatistics};

        let mut split = FileSplit::new_block("/data/file.sst".to_string(), 0, 0, 1024, 100);
        split.statistics = SplitStatistics {
            row_count: Some(100),
            bloom_filter: Some(vec![0u8; 32]),
            ..Default::default()
        };

        let stats = PruningStatistics::from_splits(&[split]);
        assert_eq!(stats.total_splits, 1);
        assert_eq!(stats.splits_with_stats, 1);
        assert_eq!(stats.splits_with_bloom, 1);
        assert!(stats.is_pruning_effective());
    }

    #[tokio::test]
    async fn test_null_provider() {
        let schema = test_schema();
        let info = DataFusionCollectionInfo::new("test".to_string(), 128, EngineType::Sst);
        let provider = NullProximaTableProvider::new(schema.clone(), info);

        assert_eq!(provider.engine_type(), EngineType::Sst);
        assert_eq!(provider.collection_info().name, "test");
        assert_eq!(provider.collection_info().dimension, 128);

        let splits = provider.get_splits(&[]).await.unwrap();
        assert!(splits.is_empty());

        let stats = provider.pruning_stats().unwrap();
        assert_eq!(stats.total_splits, 0);
    }

    #[test]
    fn test_provider_capabilities() {
        let schema = test_schema();
        let info = DataFusionCollectionInfo::new("test".to_string(), 128, EngineType::Sst);
        let provider = NullProximaTableProvider::new(schema, info);

        let caps = provider.capabilities();
        assert!(caps.contains_key("vector_search"));
        assert!(caps.contains_key("predicate_pushdown"));
        assert!(caps.contains_key("bloom_filter"));
    }

    #[test]
    fn test_supports_vector_search() {
        let schema = test_schema();

        // Provider with dimension > 0 should support vector search
        let info = DataFusionCollectionInfo::new("test".to_string(), 128, EngineType::Sst);
        let provider = NullProximaTableProvider::new(schema.clone(), info);
        assert!(provider.supports_vector_search());

        // Provider with dimension = 0 should not support vector search
        let info_no_vector = DataFusionCollectionInfo::new("test".to_string(), 0, EngineType::Sst);
        let provider_no_vector = NullProximaTableProvider::new(schema, info_no_vector);
        assert!(!provider_no_vector.supports_vector_search());
    }

    #[test]
    fn test_vector_column_name() {
        let schema = test_schema();
        let info = DataFusionCollectionInfo::new("test".to_string(), 128, EngineType::Sst);
        let provider = NullProximaTableProvider::new(schema, info);

        assert_eq!(provider.vector_column_name(), Some("vector"));
    }
}
