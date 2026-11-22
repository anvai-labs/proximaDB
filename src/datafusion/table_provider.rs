//! # ProximaDB TableProvider Implementation
//!
//! Implements DataFusion's TableProvider trait for ProximaDB collections,
//! enabling SQL queries over vector and metadata columns.
//!
//! ## Two Provider Types
//!
//! 1. **ProximaDBTableProvider**: Original format-based provider using InternalFormat
//! 2. **ProximaDataFusionTable**: New split-based provider using SplitReader
//!
//! ## Usage
//!
//! ```rust,ignore
//! // Format-based provider (existing)
//! let provider = ProximaDBTableProvider::new(collection, schema, format, path);
//!
//! // Split-based provider (new)
//! let provider = ProximaDataFusionTable::new(collection, schema, reader);
//! ```

use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::Session;
use datafusion::common::Statistics;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::Result as DFResult;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;
use tracing::{debug, info, trace};

use crate::storage::formats::FileSplit;
use crate::storage::formats::InternalFormat;
use crate::storage::schema::ProximaSchema;

use super::predicate_pushdown::convert_expr_to_filter;
use super::proxima_scan_exec::{ProximaScanExec, SplitReader};
use super::proxima_table_provider::{
    CollectionInfo, EngineType, ProximaTableProvider, PruningStatistics,
};
use super::scan_executor::ProximaDBScanExec;

/// Configuration for ProximaDBTableProvider.
#[derive(Debug, Clone)]
pub struct ProximaDBTableProviderConfig {
    /// Collection name
    pub collection_name: String,
    /// Maximum partitions for parallel scan
    pub max_partitions: usize,
    /// Enable predicate pushdown to storage
    pub enable_filter_pushdown: bool,
    /// Enable projection pushdown
    pub enable_projection_pushdown: bool,
    /// Batch size for reading
    pub batch_size: usize,
}

impl Default for ProximaDBTableProviderConfig {
    fn default() -> Self {
        Self {
            collection_name: String::new(),
            max_partitions: num_cpus::get(),
            enable_filter_pushdown: true,
            enable_projection_pushdown: true,
            batch_size: 8192,
        }
    }
}

/// DataFusion TableProvider for ProximaDB collections.
///
/// Provides SQL access to vector collections stored in VIPER, NOVA, or other engines.
pub struct ProximaDBTableProvider {
    /// Collection name
    collection_name: String,
    /// Arrow schema for this collection
    schema: SchemaRef,
    /// ProximaDB schema with additional metadata
    proxima_schema: Arc<ProximaSchema>,
    /// Storage format for reading data
    storage_format: Arc<dyn InternalFormat>,
    /// Cached statistics
    statistics: Option<Statistics>,
    /// Configuration
    config: ProximaDBTableProviderConfig,
    /// Base path for data files
    base_path: String,
}

impl ProximaDBTableProvider {
    /// Create a new TableProvider for a ProximaDB collection.
    pub fn new(
        collection_name: String,
        proxima_schema: Arc<ProximaSchema>,
        storage_format: Arc<dyn InternalFormat>,
        base_path: String,
    ) -> Self {
        // to_arrow_schema() already returns Arc<ArrowSchema>
        let schema = proxima_schema.to_arrow_schema();
        Self {
            collection_name: collection_name.clone(),
            schema,
            proxima_schema,
            storage_format,
            statistics: None,
            config: ProximaDBTableProviderConfig {
                collection_name,
                ..Default::default()
            },
            base_path,
        }
    }

    /// Create with custom configuration.
    pub fn with_config(
        collection_name: String,
        proxima_schema: Arc<ProximaSchema>,
        storage_format: Arc<dyn InternalFormat>,
        base_path: String,
        config: ProximaDBTableProviderConfig,
    ) -> Self {
        // to_arrow_schema() already returns Arc<ArrowSchema>
        let schema = proxima_schema.to_arrow_schema();
        Self {
            collection_name,
            schema,
            proxima_schema,
            storage_format,
            statistics: None,
            config,
            base_path,
        }
    }

    /// Set statistics for query optimization.
    pub fn with_statistics(mut self, stats: Statistics) -> Self {
        self.statistics = Some(stats);
        self
    }

    /// Get the collection name.
    pub fn collection_name(&self) -> &str {
        &self.collection_name
    }

    /// Get the ProximaDB schema.
    pub fn proxima_schema(&self) -> &ProximaSchema {
        &self.proxima_schema
    }
}

impl std::fmt::Debug for ProximaDBTableProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProximaDBTableProvider")
            .field("collection_name", &self.collection_name)
            .field("schema", &self.schema)
            .field("base_path", &self.base_path)
            .finish()
    }
}

#[async_trait]
impl TableProvider for ProximaDBTableProvider {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        if !self.config.enable_filter_pushdown {
            return Ok(vec![
                TableProviderFilterPushDown::Unsupported;
                filters.len()
            ]);
        }

        let results = filters
            .iter()
            .map(
                |expr| match convert_expr_to_filter(expr, &self.proxima_schema) {
                    Ok(_) => TableProviderFilterPushDown::Exact,
                    Err(_) => TableProviderFilterPushDown::Unsupported,
                },
            )
            .collect();

        Ok(results)
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&[usize]>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        debug!(
            "Creating scan for collection '{}' with projection {:?}, {} filters, limit {:?}",
            self.collection_name,
            projection,
            filters.len(),
            limit
        );

        // Convert filters to ProximaDB FilterExpression
        let storage_filters = if self.config.enable_filter_pushdown {
            filters
                .iter()
                .filter_map(|expr| convert_expr_to_filter(expr, &self.proxima_schema).ok())
                .collect::<Vec<_>>()
        } else {
            vec![]
        };

        // Combine filters with AND
        let combined_filter = if storage_filters.is_empty() {
            None
        } else if storage_filters.len() == 1 {
            storage_filters.into_iter().next()
        } else {
            // Combine multiple filters with AND
            Some(crate::storage::formats::FilterExpression::And(
                storage_filters,
            ))
        };

        // Create scan executor
        let scan_exec = ProximaDBScanExec::try_new(
            self.collection_name.clone(),
            self.schema.clone(),
            self.storage_format.clone(),
            self.base_path.clone(),
            projection.map(<[usize]>::to_vec),
            combined_filter,
            limit,
            self.config.max_partitions,
            self.config.batch_size,
        )?;

        info!(
            "Created scan executor for '{}' with {} partitions",
            self.collection_name,
            scan_exec.partition_count()
        );

        Ok(Arc::new(scan_exec))
    }

    fn statistics(&self) -> Option<Statistics> {
        self.statistics.clone()
    }
}

// ============================================================================
// Factory Functions
// ============================================================================

/// Create a TableProvider for a VIPER collection.
pub fn create_viper_table_provider(
    collection_name: String,
    proxima_schema: Arc<ProximaSchema>,
    storage_format: Arc<dyn InternalFormat>,
    base_path: String,
) -> ProximaDBTableProvider {
    ProximaDBTableProvider::new(collection_name, proxima_schema, storage_format, base_path)
}

/// Create a TableProvider for a NOVA collection.
pub fn create_nova_table_provider(
    collection_name: String,
    proxima_schema: Arc<ProximaSchema>,
    storage_format: Arc<dyn InternalFormat>,
    base_path: String,
) -> ProximaDBTableProvider {
    ProximaDBTableProvider::new(collection_name, proxima_schema, storage_format, base_path)
}

// ============================================================================
// ProximaDataFusionTable - New Split-Based Provider
// ============================================================================

/// Configuration for ProximaDataFusionTable.
#[derive(Debug, Clone)]
pub struct ProximaDataFusionTableConfig {
    /// Target number of partitions for parallel execution
    pub target_partitions: usize,
    /// Batch size for reading
    pub batch_size: usize,
    /// Enable filter pushdown
    pub enable_filter_pushdown: bool,
    /// Enable projection pushdown
    pub enable_projection_pushdown: bool,
}

impl Default for ProximaDataFusionTableConfig {
    fn default() -> Self {
        Self {
            target_partitions: num_cpus::get(),
            batch_size: 8192,
            enable_filter_pushdown: true,
            enable_projection_pushdown: true,
        }
    }
}

/// DataFusion table provider backed by ProximaDB's split-based reading.
///
/// This is the new, preferred way to integrate ProximaDB collections
/// with DataFusion, using the SplitReader abstraction for efficient
/// parallel scanning.
///
/// ## Features
///
/// - Automatic split-based partitioning for parallel execution
/// - Filter pushdown to reduce I/O
/// - Projection pushdown to read only required columns
/// - Integration with ProximaTableProvider trait
///
/// ## Example
///
/// ```rust,ignore
/// use crate::storage::formats::table_provider::CollectionInfo;
/// use crate::storage::formats::execution_plan::NullSplitReader;
///
/// let info = CollectionInfo::new("vectors".to_string(), 768, EngineType::Sst);
/// let reader = Arc::new(NullSplitReader::new(schema.clone(), EngineType::Sst));
///
/// let table = ProximaDataFusionTable::new(
///     "vectors".to_string(),
///     info,
///     schema,
///     reader,
/// );
///
/// // Register with DataFusion
/// ctx.register_table("vectors", Arc::new(table))?;
/// ```
pub struct ProximaDataFusionTable {
    /// Collection name
    collection_name: String,
    /// Collection metadata
    collection_info: CollectionInfo,
    /// Arrow schema
    schema: SchemaRef,
    /// Split reader for this engine
    reader: Arc<dyn SplitReader>,
    /// Cached splits (populated lazily)
    cached_splits: parking_lot::RwLock<Option<Vec<FileSplit>>>,
    /// Configuration
    config: ProximaDataFusionTableConfig,
    /// Pruning statistics
    pruning_stats: parking_lot::RwLock<Option<PruningStatistics>>,
    /// Cached DataFusion statistics
    statistics: Option<Statistics>,
}

impl ProximaDataFusionTable {
    /// Create a new ProximaDataFusionTable.
    pub fn new(
        collection_name: String,
        collection_info: CollectionInfo,
        schema: SchemaRef,
        reader: Arc<dyn SplitReader>,
    ) -> Self {
        Self {
            collection_name,
            collection_info,
            schema,
            reader,
            cached_splits: parking_lot::RwLock::new(None),
            config: ProximaDataFusionTableConfig::default(),
            pruning_stats: parking_lot::RwLock::new(None),
            statistics: None,
        }
    }

    /// Create with custom configuration.
    pub fn with_config(
        collection_name: String,
        collection_info: CollectionInfo,
        schema: SchemaRef,
        reader: Arc<dyn SplitReader>,
        config: ProximaDataFusionTableConfig,
    ) -> Self {
        Self {
            collection_name,
            collection_info,
            schema,
            reader,
            cached_splits: parking_lot::RwLock::new(None),
            config,
            pruning_stats: parking_lot::RwLock::new(None),
            statistics: None,
        }
    }

    /// Set pre-computed splits.
    pub fn with_splits(self, splits: Vec<FileSplit>) -> Self {
        let stats = PruningStatistics::from_splits(&splits);
        *self.cached_splits.write() = Some(splits);
        *self.pruning_stats.write() = Some(stats);
        self
    }

    /// Set statistics.
    pub fn with_statistics(mut self, stats: Statistics) -> Self {
        self.statistics = Some(stats);
        self
    }

    /// Get collection info.
    pub fn collection_info(&self) -> &CollectionInfo {
        &self.collection_info
    }

    /// Get the engine type.
    pub fn engine_type(&self) -> EngineType {
        self.collection_info.engine_type
    }

    /// Prune splits based on filters.
    fn prune_splits(&self, splits: Vec<FileSplit>, _filters: &[Expr]) -> Vec<FileSplit> {
        // For now, return all splits
        // Future: implement filter-based pruning
        trace!(
            "Pruning {} splits with {} filters",
            splits.len(),
            _filters.len()
        );
        splits
    }
}

impl Debug for ProximaDataFusionTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProximaDataFusionTable")
            .field("collection_name", &self.collection_name)
            .field("engine", &self.collection_info.engine_type)
            .field("dimension", &self.collection_info.dimension)
            .field("vector_count", &self.collection_info.vector_count)
            .finish()
    }
}

#[async_trait]
impl TableProvider for ProximaDataFusionTable {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        if !self.config.enable_filter_pushdown {
            return Ok(vec![
                TableProviderFilterPushDown::Unsupported;
                filters.len()
            ]);
        }

        // For now, mark all filters as Inexact (we'll apply them but can't guarantee pruning)
        // Future: analyze filters and determine exact vs inexact pushdown
        Ok(filters
            .iter()
            .map(|_| TableProviderFilterPushDown::Inexact)
            .collect())
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&[usize]>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        debug!(
            "ProximaDataFusionTable::scan collection='{}' projection={:?} filters={} limit={:?}",
            self.collection_name,
            projection,
            filters.len(),
            limit
        );

        // Get or create splits
        let splits = {
            let cached = self.cached_splits.read();
            if let Some(ref splits) = *cached {
                splits.clone()
            } else {
                // Create default empty splits - real implementation would discover files
                vec![]
            }
        };

        // Prune splits based on filters
        let pruned_splits = self.prune_splits(splits, filters);

        info!(
            "ProximaDataFusionTable: {} splits for collection '{}'",
            pruned_splits.len(),
            self.collection_name
        );

        // Build the execution plan
        let exec = ProximaScanExec::builder()
            .schema(self.schema.clone())
            .splits(pruned_splits)
            .projection(projection.map(<[usize]>::to_vec))
            .filters(filters.to_vec())
            .limit(limit)
            .reader(self.reader.clone())
            .batch_size(self.config.batch_size)
            .collection_name(self.collection_name.clone())
            .target_partitions(self.config.target_partitions)
            .build()?;

        Ok(Arc::new(exec))
    }

    fn statistics(&self) -> Option<Statistics> {
        self.statistics.clone()
    }
}

#[async_trait]
impl ProximaTableProvider for ProximaDataFusionTable {
    fn engine_type(&self) -> EngineType {
        self.collection_info.engine_type
    }

    fn collection_info(&self) -> &CollectionInfo {
        &self.collection_info
    }

    async fn get_splits(&self, _filters: &[Expr]) -> DFResult<Vec<FileSplit>> {
        let cached = self.cached_splits.read();
        Ok(cached.clone().unwrap_or_default())
    }

    fn pruning_stats(&self) -> Option<PruningStatistics> {
        self.pruning_stats.read().clone()
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
    fn test_default_config() {
        let config = ProximaDBTableProviderConfig::default();
        assert!(config.enable_filter_pushdown);
        assert!(config.enable_projection_pushdown);
        assert_eq!(config.batch_size, 8192);
    }

    #[test]
    fn test_datafusion_table_config() {
        let config = ProximaDataFusionTableConfig::default();
        assert!(config.enable_filter_pushdown);
        assert!(config.enable_projection_pushdown);
        assert_eq!(config.batch_size, 8192);
        assert!(config.target_partitions > 0);
    }

    #[test]
    fn test_datafusion_table_creation() {
        use crate::datafusion::proxima_scan_exec::NullSplitReader;

        let schema = test_schema();
        let info = CollectionInfo::new("test_collection".to_string(), 128, EngineType::Sst)
            .with_vector_count(1000);
        let reader = Arc::new(NullSplitReader::new(schema.clone(), EngineType::Sst));

        let table = ProximaDataFusionTable::new(
            "test_collection".to_string(),
            info,
            schema.clone(),
            reader,
        );

        assert_eq!(table.collection_name, "test_collection");
        assert_eq!(table.engine_type(), EngineType::Sst);
        assert_eq!(table.collection_info().dimension, 128);
        assert_eq!(table.collection_info().vector_count, 1000);
    }

    #[test]
    fn test_datafusion_table_with_splits() {
        use crate::datafusion::proxima_scan_exec::NullSplitReader;

        let schema = test_schema();
        let info = CollectionInfo::new("test".to_string(), 128, EngineType::Viper);
        let reader = Arc::new(NullSplitReader::new(schema.clone(), EngineType::Viper));

        let splits = vec![
            FileSplit::new_row_group("/data/file1.parquet".to_string(), 0, 0, 65536, 10000),
            FileSplit::new_row_group("/data/file1.parquet".to_string(), 1, 65536, 65536, 10000),
        ];

        let table = ProximaDataFusionTable::new("test".to_string(), info, schema, reader)
            .with_splits(splits);

        let stats = table.pruning_stats().unwrap();
        assert_eq!(stats.total_splits, 2);
    }

    #[test]
    fn test_datafusion_table_debug() {
        use crate::datafusion::proxima_scan_exec::NullSplitReader;

        let schema = test_schema();
        let info = CollectionInfo::new("my_vectors".to_string(), 768, EngineType::Nova)
            .with_vector_count(50000);
        let reader = Arc::new(NullSplitReader::new(schema.clone(), EngineType::Nova));

        let table = ProximaDataFusionTable::new("my_vectors".to_string(), info, schema, reader);

        let debug_str = format!("{:?}", table);
        assert!(debug_str.contains("my_vectors"));
        assert!(debug_str.contains("Nova"));
    }

    #[tokio::test]
    async fn test_datafusion_table_get_splits() {
        use crate::datafusion::proxima_scan_exec::NullSplitReader;

        let schema = test_schema();
        let info = CollectionInfo::new("test".to_string(), 128, EngineType::Swift);
        let reader = Arc::new(NullSplitReader::new(schema.clone(), EngineType::Swift));

        let splits = vec![FileSplit::new_block(
            "/data/file.sst".to_string(),
            0,
            0,
            1024,
            100,
        )];

        let table = ProximaDataFusionTable::new("test".to_string(), info, schema, reader)
            .with_splits(splits);

        let retrieved_splits = table.get_splits(&[]).await.unwrap();
        assert_eq!(retrieved_splits.len(), 1);
    }
}
