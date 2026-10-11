#![allow(dead_code, unused_imports, unused_variables)]
//! Test Collection Builder
//!
//! Provides a fluent builder interface for creating test collections
//! with sensible defaults and easy customization.
//!
//! This eliminates the ~50 lines of boilerplate collection setup code
//! found in 10+ test files.

use std::collections::HashMap;
use std::path::PathBuf;
use tempfile::TempDir;

use proximadb::proto::proximadb_v1::{
    Collection, CollectionConfig, CompressionAlgorithm, DistanceMetric, FilterableColumnSpec,
    FilterableDataType, IndexingAlgorithm, StorageConfig, StorageEngine as StorageEngineType,
};

/// Fluent builder for test collections with sensible defaults
///
/// # Examples
///
/// ```
/// use tests::common::collection_builder::TestCollectionBuilder;
///
/// // Simple SST collection
/// let (collection, _temp_dir) = TestCollectionBuilder::new()
///     .with_id("my_test")
///     .with_dimension(256)
///     .build();
///
/// // VIPER collection with compression
/// let (collection, _temp_dir) = TestCollectionBuilder::new()
///     .with_engine(StorageEngine::Viper)
///     .with_compression(CompressionAlgorithm::Lz4)
///     .with_dimension(512)
///     .build();
///
/// // Collection with filterable metadata
/// let (collection, _temp_dir) = TestCollectionBuilder::new()
///     .with_filterable_column("category", "STRING")
///     .with_filterable_column("price", "FLOAT")
///     .build();
/// ```
pub struct TestCollectionBuilder {
    id: Option<String>,
    name: Option<String>,
    dimension: u32,
    storage_engine: Option<StorageEngineType>,
    indexing_algorithm: IndexingAlgorithm,
    distance_metric: DistanceMetric,
    compression: Option<CompressionAlgorithm>,
    filterable_columns: Vec<FilterableColumnSpec>,
    temp_dir: Option<TempDir>,
    use_temp_dir: bool,
}

impl Default for TestCollectionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TestCollectionBuilder {
    /// Create a new builder with sensible defaults
    pub fn new() -> Self {
        Self {
            id: None,
            name: None,
            dimension: 128,
            storage_engine: Some(StorageEngineType::Sst),
            indexing_algorithm: IndexingAlgorithm::Hnsw,
            distance_metric: DistanceMetric::Euclidean,
            compression: None,
            filterable_columns: Vec::new(),
            temp_dir: None,
            use_temp_dir: true,
        }
    }

    /// Set the collection ID (defaults to auto-generated UUID)
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set the collection name (defaults to same as ID)
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the vector dimension (default: 128)
    pub fn with_dimension(mut self, dim: u32) -> Self {
        self.dimension = dim;
        self
    }

    /// Set the storage engine (default: SST)
    pub fn with_engine(mut self, engine: StorageEngineType) -> Self {
        self.storage_engine = Some(engine);
        self
    }

    /// Set the indexing algorithm (default: HNSW)
    pub fn with_indexing_algorithm(mut self, algorithm: IndexingAlgorithm) -> Self {
        self.indexing_algorithm = algorithm;
        self
    }

    /// Set the distance metric (default: Euclidean)
    pub fn with_distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Enable compression with specified algorithm
    pub fn with_compression(mut self, compression: CompressionAlgorithm) -> Self {
        self.compression = Some(compression);
        self
    }

    /// Add a filterable column for metadata filtering
    pub fn with_filterable_column(
        mut self,
        name: impl Into<String>,
        data_type: FilterableDataType,
    ) -> Self {
        self.filterable_columns.push(FilterableColumnSpec {
            name: name.into(),
            data_type: data_type as i32,
            indexed: true,
            supports_range: true,
            estimated_cardinality: None,
        });
        self
    }

    /// Don't create a temp directory (use provided path instead)
    pub fn without_temp_dir(mut self) -> Self {
        self.use_temp_dir = false;
        self
    }

    /// Build the collection with a temporary directory
    ///
    /// Returns (Collection, TempDir) - keep TempDir alive to prevent cleanup
    pub fn build(mut self) -> (Collection, TempDir) {
        let temp_dir = if self.use_temp_dir {
            tempfile::tempdir().expect("Failed to create temp directory")
        } else {
            self.temp_dir
                .take()
                .expect("No temp_dir provided when use_temp_dir=false")
        };

        let id = self.id.clone().unwrap_or_else(|| {
            use uuid::Uuid;
            format!("test_{}", Uuid::new_v4().simple())
        });

        let name = self.name.unwrap_or_else(|| id.clone());

        let collection = Collection {
            id,
            config: Some(CollectionConfig {
                name,
                embedding_config: None,
                dimension: self.dimension,
                distance_metric: Some(self.distance_metric as i32),
                storage_engine: self.storage_engine.map(|e| e as i32),
                tags: vec![],
                description: None,
                filterable_columns: self.filterable_columns,
                index_configs: vec![],
                quantization: None,
                storage_config: Some(StorageConfig {
                    storage_path: None,
                    data_paths: vec![],
                    compression: self.compression.map(|c| c as i32),
                    max_file_size_mb: None,
                    enable_caching: None,
                }),
                primary_index: None,
                auto_index_selection: None,
                owner: None,
                embedding_models: vec![],
                record_schema: None,
                enable_proxima_record: None,
                text_columns: vec![],
                text_storage_configs: vec![],
                enable_dual_use_embeddings: None,
                canonical_embedding_precision: None,
                permitted_principals: vec![],
                index_policy: None,
                pax_vector_quant: None,
            }),
            stats: None,
            created_at: 0,
            updated_at: 0,
            storage_assignment: Some(proximadb::proto::proximadb_v1::StorageAssignment {
                primary_path: temp_dir.path().to_str().unwrap().to_string(),
                backup_paths: vec![],
                engine: self.storage_engine.unwrap_or(StorageEngineType::Sst) as i32,
                engine_config: HashMap::new(),
                base_location: temp_dir.path().to_str().unwrap().to_string(),
                assigned_at: 0,
                ..Default::default()
            }),
        };

        (collection, temp_dir)
    }

    /// Build with a specific path (no temp directory created)
    pub fn build_with_path(self, path: PathBuf) -> Collection {
        let id = self.id.clone().unwrap_or_else(|| {
            use uuid::Uuid;
            format!("test_{}", Uuid::new_v4().simple())
        });

        let name = self.name.unwrap_or_else(|| id.clone());

        Collection {
            id,
            config: Some(CollectionConfig {
                name,
                embedding_config: None,
                dimension: self.dimension,
                distance_metric: Some(self.distance_metric as i32),
                storage_engine: self.storage_engine.map(|e| e as i32),
                tags: vec![],
                description: None,
                filterable_columns: self.filterable_columns,
                index_configs: vec![],
                quantization: None,
                storage_config: Some(StorageConfig {
                    storage_path: None,
                    data_paths: vec![],
                    compression: self.compression.map(|c| c as i32),
                    max_file_size_mb: None,
                    enable_caching: None,
                }),
                primary_index: None,
                auto_index_selection: None,
                owner: None,
                embedding_models: vec![],
                record_schema: None,
                enable_proxima_record: None,
                text_columns: vec![],
                text_storage_configs: vec![],
                enable_dual_use_embeddings: None,
                canonical_embedding_precision: None,
                permitted_principals: vec![],
                index_policy: None,
                pax_vector_quant: None,
            }),
            stats: None,
            created_at: 0,
            updated_at: 0,
            storage_assignment: Some(proximadb::proto::proximadb_v1::StorageAssignment {
                primary_path: path.to_str().unwrap().to_string(),
                backup_paths: vec![],
                engine: self.storage_engine.unwrap_or(StorageEngineType::Sst) as i32,
                engine_config: HashMap::new(),
                base_location: path.to_str().unwrap().to_string(),
                assigned_at: 0,
                ..Default::default()
            }),
        }
    }
}

/// Preset builders for common test scenarios
pub mod presets {
    use super::*;

    /// SST collection optimized for OLTP workloads
    pub fn sst_oltp() -> TestCollectionBuilder {
        TestCollectionBuilder::new()
            .with_engine(StorageEngineType::Sst)
            .with_compression(CompressionAlgorithm::CompressionLz4)
            .with_dimension(128)
    }

    /// VIPER collection optimized for analytics
    pub fn viper_analytics() -> TestCollectionBuilder {
        TestCollectionBuilder::new()
            .with_engine(StorageEngineType::Viper)
            .with_compression(CompressionAlgorithm::CompressionZstd)
            .with_dimension(512)
    }

    /// SWIFT collection for low-latency operations
    pub fn swift_low_latency() -> TestCollectionBuilder {
        TestCollectionBuilder::new()
            .with_engine(StorageEngineType::Swift)
            .with_dimension(128)
    }

    /// HELIX collection for spatial data
    pub fn helix_spatial() -> TestCollectionBuilder {
        TestCollectionBuilder::new()
            .with_engine(StorageEngineType::Helix)
            .with_compression(CompressionAlgorithm::CompressionZstd)
            .with_dimension(2048)
    }

    /// Collection with comprehensive metadata filtering
    pub fn with_metadata_filtering() -> TestCollectionBuilder {
        TestCollectionBuilder::new()
            .with_filterable_column("category", FilterableDataType::FilterableString)
            .with_filterable_column("price", FilterableDataType::FilterableFloat)
            .with_filterable_column("in_stock", FilterableDataType::FilterableBoolean)
            .with_filterable_column("created_at", FilterableDataType::FilterableDatetime)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_defaults() {
        let (collection, _temp_dir) = TestCollectionBuilder::new().build();

        let config = collection.config.as_ref().unwrap();
        assert_eq!(config.dimension, 128);
        assert_eq!(config.storage_engine, Some(StorageEngineType::Sst as i32));
        assert_eq!(
            config.distance_metric,
            Some(DistanceMetric::Euclidean as i32)
        );
        assert!(collection.storage_assignment.is_some());
        let storage = collection.storage_assignment.as_ref().unwrap();
        assert!(!storage.primary_path.is_empty());
    }

    #[test]
    fn test_builder_customization() {
        let (collection, _temp_dir) = TestCollectionBuilder::new()
            .with_id("custom_test")
            .with_dimension(256)
            .with_engine(StorageEngineType::Viper)
            .with_compression(CompressionAlgorithm::CompressionLz4)
            .build();

        assert_eq!(collection.id, "custom_test");
        let config = collection.config.as_ref().unwrap();
        assert_eq!(config.dimension, 256);
        assert_eq!(config.storage_engine, Some(StorageEngineType::Viper as i32));
        let storage_config = config.storage_config.as_ref().unwrap();
        assert_eq!(
            storage_config.compression,
            Some(CompressionAlgorithm::CompressionLz4 as i32)
        );
    }

    #[test]
    fn test_filterable_columns() {
        let (collection, _temp_dir) = TestCollectionBuilder::new()
            .with_filterable_column("category", FilterableDataType::FilterableString)
            .with_filterable_column("price", FilterableDataType::FilterableFloat)
            .build();

        let config = collection.config.as_ref().unwrap();
        assert_eq!(config.filterable_columns.len(), 2);
        assert_eq!(config.filterable_columns[0].name, "category");
        assert_eq!(config.filterable_columns[1].name, "price");
    }

    #[test]
    fn test_presets() {
        let (sst, _) = presets::sst_oltp().build();
        let sst_config = sst.config.as_ref().unwrap();
        assert_eq!(
            sst_config.storage_engine,
            Some(StorageEngineType::Sst as i32)
        );
        let sst_storage = sst_config.storage_config.as_ref().unwrap();
        assert_eq!(
            sst_storage.compression,
            Some(CompressionAlgorithm::CompressionLz4 as i32)
        );

        let (viper, _) = presets::viper_analytics().build();
        let viper_config = viper.config.as_ref().unwrap();
        assert_eq!(
            viper_config.storage_engine,
            Some(StorageEngineType::Viper as i32)
        );
        assert_eq!(viper_config.dimension, 512);

        let (metadata_coll, _) = presets::with_metadata_filtering().build();
        assert_eq!(
            metadata_coll
                .config
                .as_ref()
                .unwrap()
                .filterable_columns
                .len(),
            4
        );
    }
}
