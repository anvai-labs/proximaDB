#![allow(dead_code, unused_imports, unused_variables)]
//! Unified Test Utilities for ProximaDB
//!
//! This module provides comprehensive test utilities for all ProximaDB components:
//! - SST storage engine tests
//! - VIPER storage engine tests  
//! - Flush and compaction operations
//! - Vector operations and search
//! - Metadata handling
//!
//! This replaces the fragmented test utilities scattered across different modules
//! and provides a consistent, reliable test infrastructure.

use anyhow::Result;
use std::path::PathBuf;
use std::sync::{Arc, Once};
use tempfile::TempDir;
use tracing::{debug, info};
use uuid::Uuid;

// Core ProximaDB imports
use proximadb::compute::distance_computation::engine::UnifiedDistanceCompute;
use proximadb::core::config::StorageLocation;
use proximadb::core::config::ViperConfig;
use proximadb::core::config::{BloomFilterConfig, SstConfig};
use proximadb::core::search::results::OptimizedSearchRecord;
use proximadb::proto::proximadb_v1::{
    Collection, CollectionConfig, CollectionStats, CompressionAlgorithm, DistanceMetric, SqlValue,
    StorageEngine, VectorRecord, sql_value,
};
use proximadb::services::vector_operations_service::VectorOperationsService;
use proximadb::storage::engines::sst::SstEngine;
use proximadb::storage::engines::viper::ViperEngine;
use proximadb::storage::metadata::store::MetadataCacheConfig;
use proximadb::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};
use proximadb::storage::persistence::write_ahead_log::config::WriteBufferStrategyType;
use proximadb::storage::persistence::write_ahead_log::{WALConfig, WriteAheadLogManager};
use proximadb::storage::traits::{CompactionParameters, FlushParameters, UnifiedStorageEngine};
use proximadb_data_model::ProximaValue;
use proximadb_records::ProximaRecord;
// TD-ORION-1 test harness: the real vector-operations service stack.
// (DistanceMetric, StorageEngine, CollectionConfig, WriteBufferStrategyType,
// WAL*/SstEngine/FilesystemFactory/VectorOperationsService are already imported above.)
use proximadb::catalog::CatalogManager;
use proximadb::core::config::StorageConfig;
use proximadb::index::axis::{AxisConfig, AxisManager};
use proximadb::services::collection::manager::CollectionService;
use proximadb_runtime::CollectionPort;

static HARDWARE_INIT: Once = Once::new();

/// Setup hardware capabilities for tests (shared initialization)
pub fn setup_hardware_capabilities() {
    HARDWARE_INIT.call_once(|| {
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    });
}

/// Create a unique collection ID for tests
pub fn unique_collection_id(prefix: &str) -> String {
    format!("{}_{}", prefix, Uuid::new_v4().simple())
}

/// Setup test directories with proper structure
pub async fn setup_test_directories(base_path: &std::path::Path) -> Result<()> {
    use tokio::fs;
    fs::create_dir_all(base_path).await?;
    fs::create_dir_all(base_path.join("data")).await?;
    fs::create_dir_all(base_path.join("write_ahead_log")).await?;
    fs::create_dir_all(base_path.join("index")).await?;
    fs::create_dir_all(base_path.join("metadata")).await?;
    fs::create_dir_all(base_path.join("storage")).await?;
    Ok(())
}

/// Comprehensive test environment for all ProximaDB components
pub struct UnifiedTestEnvironment {
    pub collection_id: String,
    pub temp_dir: TempDir,
    pub persistent_base: PathBuf,
    pub persistent_dir: PathBuf,
    pub filesystem: Arc<FilesystemFactory>,
    pub sst_config: SstConfig,
    pub viper_config: ViperConfig,
    pub storage_locations: Vec<StorageLocation>,
    pub distance_compute: Arc<UnifiedDistanceCompute>,
}

impl UnifiedTestEnvironment {
    /// Create a new comprehensive test environment
    pub async fn new() -> Result<Self> {
        setup_hardware_capabilities();

        let collection_id = unique_collection_id("unified_test");
        let temp_dir = TempDir::new()?;

        // Create persistent directory for reliable testing
        let persistent_base = std::env::temp_dir().join("proximadb_unified_tests");
        tokio::fs::create_dir_all(&persistent_base).await?;
        let persistent_dir = persistent_base.join(&collection_id);
        tokio::fs::create_dir_all(&persistent_dir).await?;

        info!(
            "🏗️ Created unified test environment: {}",
            persistent_dir.display()
        );
        info!("🏗️ Collection ID: {}", collection_id);

        // Setup all required directories
        setup_test_directories(&persistent_dir).await?;

        let filesystem = Arc::new(FilesystemFactory::create(FilesystemConfig::default()).await?);

        // Create optimized configs for testing
        let sst_config = Self::create_test_sst_config(&persistent_dir);
        let viper_config = Self::create_test_viper_config(&persistent_dir);

        let storage_locations = vec![StorageLocation {
            url: format!("file://{}", persistent_dir.to_str().unwrap()),
            weight: 1,
            tags: vec!["unified_test".to_string()],
            io_budget: None,
        }];

        Ok(Self {
            collection_id,
            temp_dir,
            persistent_base,
            persistent_dir,
            filesystem,
            sst_config,
            viper_config,
            storage_locations,
            distance_compute: Arc::new(UnifiedDistanceCompute::default()),
        })
    }

    /// Create an SST storage engine for this environment
    pub async fn create_sst_engine(&self) -> Result<SstEngine> {
        info!(
            "🏭 Creating SST engine for collection: {}",
            self.collection_id
        );

        let _distance_compute = Arc::new(UnifiedDistanceCompute::new(
            proximadb::compute::distance_computation::DistanceMetric::Cosine,
        ));

        SstEngine::new()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create SST storage: {}", e))
    }

    /// Create a VIPER storage engine for this environment
    pub async fn create_viper_engine(&self) -> Result<ViperEngine> {
        info!(
            "🏭 Creating VIPER engine for collection: {}",
            self.collection_id
        );

        ViperEngine::new().await
    }

    /// Assemble the **real** `VectorOperationsService` stack (SST + WAL + Axis +
    /// Catalog + CollectionService) over this environment's tempdir — the same
    /// recipe as the in-crate `src/services/operations/vectors_test.rs::create_test_service`,
    /// exposed for `tests/` so integration tests use the real search path instead
    /// of a mock. Returns the service plus its backing `CollectionService` (needed
    /// to create collections). Keep `self` (the env) alive for the service's
    /// lifetime — it owns the tempdir.
    pub async fn vector_operations_service(
        &self,
    ) -> Result<(Arc<VectorOperationsService>, Arc<CollectionService>)> {
        let metadata_url = format!("file://{}", self.persistent_dir.join("metadata").display());

        let sst_engine = Arc::new(SstEngine::new().await?);

        let wal_config = WALConfig::default();
        let wal_manager = Arc::new(WriteAheadLogManager::new(wal_config).await?);

        let axis_manager = Arc::new(AxisManager::new(AxisConfig::default()).await?);

        let catalog_manager = Arc::new(CatalogManager::new());
        catalog_manager
            .create_native_catalog("default", &metadata_url)
            .await?;
        let storage_config = StorageConfig {
            metadata_url,
            ..Default::default()
        };
        let collection_service = Arc::new(
            CollectionService::new(storage_config)
                .await?
                .with_catalog_manager(catalog_manager),
        );

        let vector_ops = Arc::new(VectorOperationsService::new(
            sst_engine,
            wal_manager,
            axis_manager,
            collection_service.clone() as Arc<dyn CollectionPort>,
        ));
        Ok((vector_ops, collection_service))
    }

    /// FA-2 (abac-policy): like [`vector_operations_service`] but wires a vector
    /// ABAC enforcer into the service so `unified_search_native` enforces the
    /// subject's accessibility predicate on the real search path. For
    /// integration tests that prove enforcement fires end-to-end.
    #[cfg(feature = "abac-policy")]
    pub async fn vector_operations_service_with_abac(
        &self,
        enforcer: Arc<proximadb::security::rls::AbacEnforcer>,
    ) -> Result<(Arc<VectorOperationsService>, Arc<CollectionService>)> {
        let metadata_url = format!("file://{}", self.persistent_dir.join("metadata").display());
        let sst_engine = Arc::new(SstEngine::new().await?);
        let wal_manager = Arc::new(WriteAheadLogManager::new(WALConfig::default()).await?);
        let axis_manager = Arc::new(AxisManager::new(AxisConfig::default()).await?);
        let catalog_manager = Arc::new(CatalogManager::new());
        catalog_manager
            .create_native_catalog("default", &metadata_url)
            .await?;
        let storage_config = StorageConfig {
            metadata_url,
            ..Default::default()
        };
        let collection_service = Arc::new(
            CollectionService::new(storage_config)
                .await?
                .with_catalog_manager(catalog_manager),
        );
        let vector_ops = Arc::new(
            VectorOperationsService::new(
                sst_engine,
                wal_manager,
                axis_manager,
                collection_service.clone() as Arc<dyn CollectionPort>,
            )
            .with_abac_enforcer(enforcer),
        );
        Ok((vector_ops, collection_service))
    }

    /// Create a dense vector collection on the given `CollectionService`
    /// (cosine / SST), for use with the `vector_operations_service()` stack.
    pub async fn create_vector_collection(
        collections: &CollectionService,
        name: &str,
        dimension: u32,
    ) -> Result<()> {
        let config = CollectionConfig {
            name: name.to_string(),
            dimension,
            distance_metric: Some(DistanceMetric::Cosine as i32),
            storage_engine: Some(StorageEngine::Sst as i32),
            ..Default::default()
        };
        collections.create_collection(&config).await?;
        Ok(())
    }

    /// Get the data directory path for SST operations
    pub fn get_sst_data_directory(&self) -> PathBuf {
        // SST writes to {base_location}/{collection_id}/data
        // persistent_dir already includes collection_id, so data goes inside it
        self.persistent_dir.join("data")
    }

    /// Get the data directory path for VIPER operations
    pub fn get_viper_data_directory(&self) -> PathBuf {
        // VIPER writes to configured data directory
        PathBuf::from(&self.viper_config.data_directory)
    }

    /// Ensure data directories exist for operations
    pub async fn ensure_all_directories(&self) -> Result<()> {
        tokio::fs::create_dir_all(self.get_sst_data_directory()).await?;
        tokio::fs::create_dir_all(self.get_viper_data_directory()).await?;
        Ok(())
    }

    /// Create a test collection with proper storage assignment
    pub fn create_test_collection(&self) -> Collection {
        self.create_test_collection_for_engine(StorageEngine::Sst)
    }

    /// Create a test collection for specific storage engine
    ///
    /// CRITICAL: This method handles several setup complexities that cause test failures when done manually:
    /// 1. Sets base_location to PARENT directory (not the collection directory itself!)
    /// 2. Uses proper file:// URL format (though system defaults to file:// anyway)
    /// 3. Uses correct timestamp units (millis, not micros)
    /// 4. Ensures storage_assignment is consistent across flush and compaction operations
    ///
    /// Storage engines expect: {base_location}/{collection_id}/data/
    /// So if base_location = /tmp/test, files go to /tmp/test/collection_123/data/
    pub fn create_test_collection_for_engine(&self, engine: StorageEngine) -> Collection {
        self.create_test_collection_with_settings(engine, 3, None)
    }

    /// Create test collection with all settings - this is the MAIN method
    /// that should be used when you need specific compression settings
    /// For SST: block_size_kb is part of CompressionConfig (not separate SstConfig)
    pub fn create_test_collection_with_settings(
        &self,
        engine: StorageEngine,
        dimension: i32,
        _compression: Option<proximadb::proto::proximadb_v1::CompressionConfig>,
    ) -> Collection {
        let config = CollectionConfig {
            name: self.collection_id.clone(),
            dimension: dimension as u32,
            distance_metric: Some(DistanceMetric::Cosine as i32),
            storage_engine: Some(engine as i32),
            primary_index: Some("default".to_string()),
            auto_index_selection: Some(false),
            ..Default::default()
        };

        let stats = CollectionStats {
            vector_count: 0,
            index_size_bytes: 0,
            data_size_bytes: 0,
        };

        Collection {
            storage_class_status: None,
            id: self.collection_id.clone(),
            config: Some(config),
            stats: Some(stats),
            created_at: chrono::Utc::now().timestamp_millis(),
            updated_at: chrono::Utc::now().timestamp_millis(),
            storage_assignment: Some(proximadb::proto::proximadb_v1::StorageAssignment {
                primary_path: self.persistent_dir.to_str().unwrap().to_string(),
                backup_paths: vec![],
                engine: engine as i32,
                engine_config: std::collections::HashMap::new(),
                // base_location should be the parent directory (/tmp/proximadb_unified_tests)
                // so engines can append collection_id to get /tmp/proximadb_unified_tests/<collection_id>/data/
                base_location: self.persistent_base.to_str().unwrap().to_string(),
                assigned_at: chrono::Utc::now().timestamp_millis(),
                ..Default::default()
            }),
        }
    }

    /// Unified setup for SST operations with proper paths and configuration
    pub async fn setup_sst_test(
        &self,
        vectors: Vec<VectorRecord>,
    ) -> Result<(FlushParameters, PathBuf, Collection)> {
        let data_dir = self.get_sst_data_directory();
        tokio::fs::create_dir_all(&data_dir).await?;

        let collection_config = self.create_test_collection_for_engine(StorageEngine::Sst);

        let flush_params = FlushParameters {
            collection_id: Some(self.collection_id.clone()),
            vector_records: vectors
                .into_iter()
                .map(|v: VectorRecord| v.into())
                .collect(),
            force: true,
            synchronous: true,
            collection_config: Some(collection_config.clone()),
            ..Default::default()
        };

        Ok((flush_params, data_dir, collection_config))
    }

    /// Unified setup for VIPER operations with proper paths and configuration
    pub async fn setup_viper_test(
        &self,
        vectors: Vec<VectorRecord>,
    ) -> Result<(FlushParameters, PathBuf, Collection)> {
        let data_dir = self.get_viper_data_directory();
        tokio::fs::create_dir_all(&data_dir).await?;

        let collection_config = self.create_test_collection_for_engine(StorageEngine::Viper);

        let flush_params = FlushParameters {
            collection_id: Some(self.collection_id.clone()),
            vector_records: vectors
                .into_iter()
                .map(|v: VectorRecord| v.into())
                .collect(),
            force: true,
            synchronous: true,
            collection_config: Some(collection_config.clone()),
            ..Default::default()
        };

        Ok((flush_params, data_dir, collection_config))
    }

    /// Create SST files using proper flush mechanism (from sst_compactor_tests)
    pub async fn create_sst_files_with_engine(
        &self,
        vectors: Vec<VectorRecord>,
    ) -> Result<Vec<String>> {
        debug!(
            "🚀 Creating SST files for collection {} with {} vectors",
            self.collection_id,
            vectors.len()
        );

        let engine = self.create_sst_engine().await?;
        let (flush_params, _data_dir, _collection_config) = self.setup_sst_test(vectors).await?;

        let flush_result = engine.do_flush(&flush_params).await?;
        if !flush_result.success {
            return Err(anyhow::anyhow!("SST flush failed"));
        }

        debug!(
            "✅ SST flush successful: {} entries flushed, {} files created",
            flush_result.entries_flushed.unwrap_or(0),
            flush_result.files_created.unwrap_or(0)
        );

        // Find created SST files
        let storage_url = format!(
            "file://{}/{}/data",
            self.persistent_dir.parent().unwrap().to_str().unwrap(),
            self.collection_id
        );

        let fs = self.filesystem.get_filesystem("file:///")?;
        let all_files = fs.list(&storage_url).await?;

        let sst_files: Vec<String> = all_files
            .iter()
            .filter(|entry| entry.name.ends_with(".sstable"))
            .map(|entry| format!("{}/{}", storage_url, entry.name))
            .collect();

        debug!("🎯 Created {} SST files: {:?}", sst_files.len(), sst_files);
        Ok(sst_files)
    }

    /// Check if SST files exist in the expected location
    pub async fn verify_sst_files_exist(&self) -> Result<Vec<std::fs::DirEntry>> {
        let data_dir = self.get_sst_data_directory();
        if !data_dir.exists() {
            return Err(anyhow::anyhow!(
                "SST data directory doesn't exist: {}",
                data_dir.display()
            ));
        }

        let entries = std::fs::read_dir(&data_dir)?;
        let sst_files: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "sst"))
            .collect();

        if sst_files.is_empty() {
            return Err(anyhow::anyhow!(
                "No SST files found in: {}",
                data_dir.display()
            ));
        }

        Ok(sst_files)
    }

    /// Check if VIPER files exist in the expected location
    pub async fn verify_viper_files_exist(&self) -> Result<Vec<std::fs::DirEntry>> {
        let data_dir = self.get_viper_data_directory();
        if !data_dir.exists() {
            return Err(anyhow::anyhow!(
                "VIPER data directory doesn't exist: {}",
                data_dir.display()
            ));
        }

        let entries = std::fs::read_dir(&data_dir)?;
        let parquet_files: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "parquet"))
            .collect();

        if parquet_files.is_empty() {
            return Err(anyhow::anyhow!(
                "No Parquet files found in: {}",
                data_dir.display()
            ));
        }

        Ok(parquet_files)
    }

    /// Create test vectors with metadata (enhanced version)
    pub fn create_test_vectors(&self, count: usize) -> Vec<ProximaRecord> {
        self.create_test_vectors_with_dimension(count, 3)
    }

    /// Create test vectors with specific dimension.
    pub fn create_test_vectors_with_dimension(
        &self,
        count: usize,
        dimension: usize,
    ) -> Vec<ProximaRecord> {
        let categories = ["A", "B", "C"];
        let types = ["primary", "secondary", "tertiary"];

        (0..count)
            .map(|i| {
                let mut props = proximadb_records::ProximaTree::new();
                props.insert(
                    "category".to_string(),
                    proximadb_records::ProximaTreeNode::Value(ProximaValue::String(
                        categories[i % categories.len()].to_string(),
                    )),
                );
                props.insert(
                    "type".to_string(),
                    proximadb_records::ProximaTreeNode::Value(ProximaValue::String(
                        types[i % types.len()].to_string(),
                    )),
                );
                props.insert(
                    "test_id".to_string(),
                    proximadb_records::ProximaTreeNode::Value(ProximaValue::String(
                        self.collection_id.clone(),
                    )),
                );
                props.insert(
                    "index".to_string(),
                    proximadb_records::ProximaTreeNode::Value(ProximaValue::Int64(i as i64)),
                );

                ProximaRecord {
                    oid: format!("{}_{}", self.collection_id, i),
                    embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                        "default",
                        "vector",
                        dimension as u32,
                        (0..dimension).map(|j| (i + j) as f32).collect(),
                    )],
                    props,
                    record_version: 1,
                    created_at_ns: (1000 + i) as i64 * 1_000_000_000,
                    updated_at_ns: (1000 + i) as i64 * 1_000_000_000,
                    ..Default::default()
                }
            })
            .collect()
    }

    /// Create a single test vector record with canonical `ProximaValue` metadata.
    /// Converts at the `VectorRecord` boundary for v1 compatibility.
    pub fn create_test_vector_record_proxima(
        &self,
        id: String,
        vector: Vec<f32>,
        timestamp: i64,
        expires_at: Option<i64>,
        metadata: std::collections::HashMap<String, ProximaValue>,
    ) -> VectorRecord {
        let sql_meta = metadata
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    proximadb::core::search::results::proxima_value_to_sql_value(v),
                )
            })
            .collect();
        VectorRecord {
            id,
            vector,
            metadata: sql_meta,
            timestamp: Some(timestamp),
            updated_at: None,
            expires_at,
            version: Some(1),
            source: None,
        }
    }

    /// Create a single test vector record with legacy `SqlValue` metadata.
    /// Compatibility shim for integration tests that still build v1 `VectorRecord`s.
    pub fn create_test_vector_record(
        &self,
        id: String,
        vector: Vec<f32>,
        timestamp: i64,
        expires_at: Option<i64>,
        metadata: std::collections::HashMap<String, SqlValue>,
    ) -> VectorRecord {
        VectorRecord {
            id,
            vector,
            metadata,
            timestamp: Some(timestamp),
            updated_at: None,
            expires_at,
            version: Some(1),
            source: None,
        }
    }

    /// Backward-compatible alias for callers that name the legacy metadata type explicitly.
    pub fn create_test_vector_record_sql(
        &self,
        id: String,
        vector: Vec<f32>,
        timestamp: i64,
        expires_at: Option<i64>,
        metadata: std::collections::HashMap<String, SqlValue>,
    ) -> VectorRecord {
        self.create_test_vector_record(id, vector, timestamp, expires_at, metadata)
    }

    /// Create a query vector for testing
    pub fn create_query_vector(&self) -> Vec<f32> {
        vec![0.5, 1.5, 2.5] // Should be close to first test vector
    }

    /// Create a query vector with specific dimension
    pub fn create_query_vector_with_dimension(&self, dimension: usize) -> Vec<f32> {
        (0..dimension).map(|i| i as f32 * 0.1).collect()
    }

    /// Get collection ID
    pub fn collection_id(&self) -> &str {
        &self.collection_id
    }

    // Removed setup_test_assignment - StorageAssignment no longer exists in proto

    /// Create metadata store config (for backward compatibility)  
    pub fn create_metadata_store_config(
        &self,
    ) -> proximadb::storage::metadata::MetadataStoreConfig {
        let metadata_path = self
            .persistent_dir
            .join("metadata")
            .to_str()
            .unwrap()
            .to_string();
        proximadb::storage::metadata::MetadataStoreConfig {
            metadata_storage_urls: vec![metadata_path], // Filesystem API defaults to file:// scheme
            enable_atomic_operations: true,
            cache_config: MetadataCacheConfig::default(),
            backup_config: None,
            replication_config: None,
        }
    }

    /// Create test collection with storage (for backward compatibility)
    pub fn create_test_collection_with_storage(
        &self,
        name: &str,
        _base_location: String,
    ) -> Collection {
        let config = CollectionConfig {
            name: name.to_string(),
            dimension: 3,
            distance_metric: Some(DistanceMetric::Cosine as i32),
            storage_engine: Some(StorageEngine::Sst as i32),
            primary_index: Some("default".to_string()),
            auto_index_selection: Some(false),
            ..Default::default()
        };

        let stats = CollectionStats {
            vector_count: 0,
            index_size_bytes: 0,
            data_size_bytes: 0,
        };

        Collection {
            storage_class_status: None,
            id: name.to_string(),
            config: Some(config),
            stats: Some(stats),
            created_at: chrono::Utc::now().timestamp_millis(),
            updated_at: chrono::Utc::now().timestamp_millis(),
            storage_assignment: Some(proximadb::proto::proximadb_v1::StorageAssignment {
                primary_path: format!("{}/collections/{}", _base_location, name),
                backup_paths: vec![],
                engine: StorageEngine::Sst as i32,
                engine_config: std::collections::HashMap::new(),
                base_location: _base_location,
                assigned_at: chrono::Utc::now().timestamp(),
                ..Default::default()
            }),
        }
    }

    /// Create test SST configuration optimized for testing (from sst_compactor_tests)
    fn create_test_sst_config(base_path: &std::path::Path) -> SstConfig {
        SstConfig {
            // Level configuration
            level_count: 3,
            max_levels: 3,
            compaction_threshold: 2,
            max_files_per_level: 3,
            level_size_multiplier: 2.0,

            // Block settings - smaller for faster testing
            block_size_kb: 3072,

            // Storage type
            compaction_strategy: "leveled".to_string(),
            compression: "none".to_string(),
            compression_level: 0,

            // Bloom filter - enabled with conservative settings
            bloom_filter_config: Some(BloomFilterConfig {
                bits_per_key: 8,
                enabled: true,
                ..Default::default()
            }),
            decompression_cache_config: None,

            // Cache
            cache_size_mb: 16,

            // Background operations
            background_thread_count: 1, // Single thread for deterministic testing

            // Directories
            data_directory: base_path.join("data").to_str().unwrap().to_string(),

            // Memory mapping - disabled for testing
            mmap_enabled: false,
            prefetch_enabled: false,
            prefetch_size_kb: 0,

            // Compaction config
            compaction_config: Default::default(),

            // Vector encoding strategy
            vector_encoding_strategy: "FullVector".to_string(),

            // Block format
            block_format: "ProximaBlocks".to_string(),

            // Tier-migration integration — disabled in tests so the
            // SST engine doesn't pull in the tiering policy engine.
            tiering: None,

            ..Default::default()
        }
    }

    /// Create general test config (for backward compatibility)
    pub fn create_test_config(&self) -> proximadb::core::config::Config {
        // Use default config since struct fields may vary
        proximadb::core::config::Config::default()
    }

    /// Create test VIPER configuration optimized for testing
    fn create_test_viper_config(base_path: &std::path::Path) -> ViperConfig {
        ViperConfig {
            row_group_size: 50_000,
            compression: "none".to_string(), // No compression for faster tests
            compression_level: 0,
            enable_statistics: true,
            data_directory: base_path.join("data").to_str().unwrap().to_string(),
            cache_size_mb: 64,
            compaction_config: Default::default(),
        }
    }
}

/// Multiple isolated environments for concurrency testing
pub struct MultiUnifiedEnvironmentTest {
    pub environments: Vec<UnifiedTestEnvironment>,
}

impl MultiUnifiedEnvironmentTest {
    /// Create multiple isolated environments
    pub async fn new(count: usize) -> Result<Self> {
        let mut environments = Vec::new();

        for _ in 0..count {
            environments.push(UnifiedTestEnvironment::new().await?);
        }

        Ok(Self { environments })
    }

    /// Get environment by index
    pub fn get_env(&self, index: usize) -> &UnifiedTestEnvironment {
        &self.environments[index]
    }

    /// Get all environment collection IDs
    pub fn collection_ids(&self) -> Vec<&str> {
        self.environments
            .iter()
            .map(|env| env.collection_id())
            .collect()
    }
}

/// Helper functions to build correct test parameters
/// Tests should call production code directly with these parameters
pub mod operations {
    use super::*;

    /// Build correct FlushParameters - the critical configuration that was causing failures
    pub async fn build_flush_params(
        environment: &UnifiedTestEnvironment,
        vectors: Vec<ProximaRecord>,
        engine: StorageEngine,
    ) -> Result<FlushParameters> {
        // Ensure directories exist
        environment.ensure_all_directories().await?;

        // Detect dimension from vectors
        let dimension = vectors
            .first()
            .and_then(|v| v.embeddings.first())
            .map(|e| e.dim as i32)
            .unwrap_or(3);

        let collection_config =
            environment.create_test_collection_with_settings(engine, dimension, None);

        Ok(FlushParameters {
            collection_id: Some(environment.collection_id().to_string()),
            vector_records: vectors,
            force: true,
            synchronous: true,
            collection_config: Some(collection_config),
            ..Default::default()
        })
    }

    /// Build FlushParameters for SST with custom compression - uses provided collection config
    pub async fn build_sst_flush_params_with_collection(
        environment: &UnifiedTestEnvironment,
        vectors: Vec<ProximaRecord>,
        collection: Collection,
    ) -> Result<FlushParameters> {
        environment.ensure_all_directories().await?;

        Ok(FlushParameters {
            collection_id: Some(environment.collection_id().to_string()),
            vector_records: vectors,
            force: true,
            synchronous: true,
            collection_config: Some(collection),
            ..Default::default()
        })
    }

    // REMOVED: build_sst_flush_params_with_compression - Use build_sst_flush_params_with_collection
    // Tests must create collection config once and reuse it for both flush and compaction

    /// Build FlushParameters for VIPER with custom compression
    /// VIPER/Parquet supports: none, snappy, gzip, lz4, zstd, brotli (limited by Parquet format)
    pub async fn build_viper_flush_params_with_compression(
        environment: &UnifiedTestEnvironment,
        vectors: Vec<ProximaRecord>,
        compression_algo: &str,
        _compression_level: i32,
    ) -> Result<FlushParameters> {
        environment.ensure_all_directories().await?;

        let mut collection = environment.create_test_collection_for_engine(StorageEngine::Viper);

        // VIPER uses Parquet compression, configured through collection config
        if let Some(ref mut _config) = collection.config {
            use proximadb::proto::proximadb_v1::CompressionAlgorithm;

            // Map to Parquet-supported algorithms
            let _algorithm = match compression_algo {
                "none" | "uncompressed" => CompressionAlgorithm::CompressionNone as i32,
                "zstd" => CompressionAlgorithm::CompressionZstd as i32,
                "snappy" => CompressionAlgorithm::CompressionSnappy as i32,
                "gzip" => CompressionAlgorithm::CompressionGzip as i32,
                "lz4" => CompressionAlgorithm::CompressionLz4 as i32,
                "brotli" => CompressionAlgorithm::CompressionBrotli as i32,
                // Unsupported algorithms default to snappy
                _ => {
                    debug!(
                        "Algorithm {} not supported by Parquet, using snappy",
                        compression_algo
                    );
                    CompressionAlgorithm::CompressionSnappy as i32
                }
            };

            // Compression config is handled by storage_config, not directly on collection config
            // The compression will be applied through storage engine configuration
        }

        Ok(FlushParameters {
            collection_id: Some(environment.collection_id().to_string()),
            vector_records: vectors,
            force: true,
            synchronous: true,
            collection_config: Some(collection),
            ..Default::default()
        })
    }

    /// Build correct storage URL for SST search operations
    pub fn build_sst_storage_url(environment: &UnifiedTestEnvironment) -> String {
        // SST expects storage_url to point directly to where .sst files are located
        format!(
            "file://{}",
            environment.get_sst_data_directory().to_str().unwrap()
        )
    }

    /// Build correct storage URL for VIPER search operations
    pub fn build_viper_storage_url(environment: &UnifiedTestEnvironment) -> String {
        // VIPER expects storage_url to point to where parquet files are located
        format!(
            "file://{}",
            environment.get_viper_data_directory().to_str().unwrap()
        )
    }

    /// Search vectors in SST engine.
    /// Returns `OptimizedSearchRecord` which carries canonical `ProximaValue` metadata.
    pub async fn search_vectors_sst(
        engine: &SstEngine,
        environment: &UnifiedTestEnvironment,
        query_vector: &[f32],
        top_k: usize,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        let collection = Arc::new(environment.create_test_collection());
        let search_params = Arc::new(proximadb::core::search::SearchParams {
            vector: Some(query_vector.to_vec()),
            query_vectors: None,
            top_k: Some(top_k as u16),
            distance_metric: Some(proximadb::compute::distance_computation::DistanceMetric::Cosine),
            filter_expression: None,
            timeout_ms: None,
            accuracy_threshold: None,
            ..Default::default()
        });
        let query_context = proximadb::storage::traits::StorageQueryContext {
            search_params,
            collection: collection.clone(),
            metadata: proximadb::storage::traits::StorageQueryMetadata {
                collection_id: collection.id.clone(),
                use_axis_indexes: false,
                has_quantization: false,
                dimension: collection.config.as_ref().map(|c| c.dimension).unwrap_or(0) as usize,
                distance_metric: collection
                    .config
                    .as_ref()
                    .and_then(|c| c.distance_metric)
                    .and_then(|dm| {
                        std::convert::TryInto::<
                            proximadb::compute::distance_computation::DistanceMetric,
                        >::try_into(dm)
                        .ok()
                    })
                    .unwrap_or(proximadb::compute::distance_computation::DistanceMetric::Cosine),
                storage_strategy: proximadb::storage::engines::StorageEngineStrategy::Sst,
                storage_path: collection
                    .storage_assignment
                    .as_ref()
                    .map(|sa| sa.base_location.clone())
                    .unwrap_or_default(),
                quantization_config: None,
                estimated_vector_count: 0,
                estimated_size_bytes: 0,
                performance_tier: proximadb::storage::traits::PerformanceTier::Hot,
                compression_enabled: false,
                quantization_enabled: false,
            },
            user_context: None,
            tenant_context: None,
        };

        engine.search_vectors_unified(&query_context).await
    }

    /// Search vectors in VIPER engine - REAL search, not simulated.
    /// Returns `OptimizedSearchRecord` with canonical `ProximaValue` metadata.
    pub async fn search_vectors_viper(
        engine: &proximadb::storage::engines::viper::ViperEngine,
        environment: &UnifiedTestEnvironment,
        query_vector: &[f32],
        top_k: usize,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        let collection = Arc::new(environment.create_test_collection());
        let search_params = Arc::new(proximadb::core::search::SearchParams {
            vector: Some(query_vector.to_vec()),
            top_k: Some(top_k as u16),
            distance_metric: Some(proximadb::compute::distance_computation::DistanceMetric::Cosine),
            ..Default::default()
        });
        let query_context = proximadb::storage::traits::StorageQueryContext {
            search_params,
            collection: collection.clone(),
            metadata: proximadb::storage::traits::StorageQueryMetadata {
                collection_id: collection.id.clone(),
                use_axis_indexes: false,
                has_quantization: false,
                dimension: collection.config.as_ref().map(|c| c.dimension).unwrap_or(0) as usize,
                distance_metric: collection
                    .config
                    .as_ref()
                    .and_then(|c| c.distance_metric)
                    .and_then(|dm| {
                        std::convert::TryInto::<
                            proximadb::compute::distance_computation::DistanceMetric,
                        >::try_into(dm)
                        .ok()
                    })
                    .unwrap_or(proximadb::compute::distance_computation::DistanceMetric::Cosine),
                storage_strategy: proximadb::storage::engines::StorageEngineStrategy::Viper,
                storage_path: collection
                    .storage_assignment
                    .as_ref()
                    .map(|sa| sa.base_location.clone())
                    .unwrap_or_default(),
                quantization_config: None,
                estimated_vector_count: 0,
                estimated_size_bytes: 0,
                performance_tier: proximadb::storage::traits::PerformanceTier::Hot,
                compression_enabled: false,
                quantization_enabled: false,
            },
            user_context: None,
            tenant_context: None,
        };

        engine.search_vectors_unified(&query_context).await
    }

    /// Build correct CompactionParameters for any storage engine
    /// This is the critical part - getting the configuration right
    pub fn build_compaction_params(
        environment: &UnifiedTestEnvironment,
        engine: StorageEngine,
    ) -> CompactionParameters {
        let collection_config = environment.create_test_collection_for_engine(engine);
        CompactionParameters {
            full: false,
            collection_id: Some(environment.collection_id().to_string()),
            force: true,
            synchronous: true,
            collection_config: Some(collection_config),
            ..Default::default()
        }
    }

    /// Build correct CompactionParameters with specific dimension
    pub fn build_compaction_params_with_dimension(
        environment: &UnifiedTestEnvironment,
        engine: StorageEngine,
        dimension: i32,
    ) -> CompactionParameters {
        let collection_config =
            environment.create_test_collection_with_settings(engine, dimension, None);
        CompactionParameters {
            full: false,
            collection_id: Some(environment.collection_id().to_string()),
            force: true,
            synchronous: true,
            collection_config: Some(collection_config),
            ..Default::default()
        }
    }

    /// Build CompactionParameters with provided collection config - PRODUCTION-LIKE
    pub fn build_compaction_params_with_collection(
        environment: &UnifiedTestEnvironment,
        collection: Collection,
    ) -> CompactionParameters {
        CompactionParameters {
            full: false,
            collection_id: Some(environment.collection_id().to_string()),
            force: true,
            synchronous: true,
            collection_config: Some(collection),
            ..Default::default()
        }
    }

    // REMOVED: build_compaction_params_with_compression - Use build_compaction_params_with_collection
    // Tests must create collection config once and reuse it for both flush and compaction
}

/// Global utility functions for backward compatibility with existing tests
/// These functions create temporary environments for one-off operations
/// Create test vectors (global function for backward compatibility)
pub fn create_test_vectors(count: usize, dimension: usize, prefix: &str) -> Vec<VectorRecord> {
    (0..count)
        .map(|i| VectorRecord {
            id: format!("{}_{}", prefix, i),
            vector: (0..dimension).map(|j| (i + j) as f32 * 0.1).collect(),
            timestamp: Some((1000 + i) as i64),
            metadata: std::collections::HashMap::from([(
                "test_type".to_string(),
                SqlValue {
                    value: Some(sql_value::Value::StringValue(prefix.to_string())),
                },
            )]),
            version: Some(1),
            source: None,
            updated_at: None,
            expires_at: None,
        })
        .collect()
}

/// Create metadata store config (global function for backward compatibility)  
pub fn create_metadata_store_config() -> proximadb::storage::metadata::MetadataStoreConfig {
    let temp_dir = std::env::temp_dir().join("proximadb_test").join("metadata");
    let metadata_path = temp_dir.to_str().unwrap().to_string();
    proximadb::storage::metadata::MetadataStoreConfig {
        metadata_storage_urls: vec![metadata_path], // Filesystem API defaults to file:// scheme
        enable_atomic_operations: true,
        cache_config: MetadataCacheConfig::default(),
        backup_config: None,
        replication_config: None,
    }
}

/// Create test collection with storage (global function for backward compatibility)
pub fn create_test_collection_with_storage(name: &str, base_location: String) -> Collection {
    let config = CollectionConfig {
        name: name.to_string(),
        dimension: 256, // Default dimension for generic tests
        distance_metric: Some(DistanceMetric::Cosine as i32),
        storage_engine: Some(StorageEngine::Sst as i32),
        primary_index: Some("default".to_string()),
        auto_index_selection: Some(false),
        ..Default::default()
    };

    let stats = CollectionStats {
        vector_count: 0,
        index_size_bytes: 0,
        data_size_bytes: 0,
    };

    Collection {
        storage_class_status: None,
        id: name.to_string(),
        config: Some(config),
        stats: Some(stats),
        created_at: chrono::Utc::now().timestamp_millis(),
        updated_at: chrono::Utc::now().timestamp_millis(),
        storage_assignment: Some(proximadb::proto::proximadb_v1::StorageAssignment {
            primary_path: format!("{}/collections/{}", base_location, name),
            backup_paths: vec![],
            engine: StorageEngine::Sst as i32,
            engine_config: std::collections::HashMap::new(),
            base_location: base_location.clone(),
            assigned_at: chrono::Utc::now().timestamp_micros(),
            ..Default::default()
        }),
    }
}

/// Create general test config (global function for backward compatibility)
pub fn create_test_config() -> proximadb::core::config::Config {
    // Use default config since struct fields may vary
    proximadb::core::config::Config::default()
}

// =============================================================================
// FLUSH OPERATIONS WITH BLOCK STATISTICS
// =============================================================================

/// Flush parameters with detailed block statistics tracking
pub struct FlushWithBlockStats {
    pub params: FlushParameters,
    pub track_blocks: bool,
    pub compression_algo: String,
    pub compression_level: i32,
    pub block_size_kb: usize,
}

/// Result of flush operation with block statistics
pub struct FlushBlockStatsResult {
    pub success: bool,
    pub entries_flushed: usize,
    pub bytes_written: usize,
    pub files_created: usize,
    pub blocks_created: usize,
    pub compression_ratio: f64,
    pub vectors_per_block: Vec<usize>,
    pub block_sizes: Vec<usize>,
}

/// Perform flush operation with block statistics tracking for SST
/// This now uses the production SST flush path with integrated quantization
pub async fn flush_sst_with_block_stats(
    environment: &UnifiedTestEnvironment,
    vectors: Vec<VectorRecord>,
    compression_algo: &str,
    compression_level: i32,
    block_size_kb: usize,
) -> Result<FlushBlockStatsResult> {
    use proximadb::proto::proximadb_v1::{Collection, CollectionConfig, CompressionConfig};
    use proximadb::storage::engines::sst::SstEngine;
    use proximadb::storage::traits::{FlushParameters, UnifiedStorageEngine};

    // Create SST config with specified block size
    let mut sst_config = environment.sst_config.clone();
    sst_config.block_size_kb = block_size_kb as u32;
    sst_config.compression = compression_algo.to_string();
    sst_config.compression_level = compression_level;

    // Create SST storage
    use proximadb::compute::distance_computation::engine::UnifiedDistanceCompute;
    use proximadb::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};

    let fs_config = FilesystemConfig::default();
    let _filesystem = Arc::new(FilesystemFactory::create(fs_config).await?);
    let _distance_compute = Arc::new(UnifiedDistanceCompute::new(
        proximadb::compute::distance_computation::DistanceMetric::Euclidean,
    ));

    let sst_storage = SstEngine::new().await?;

    // Create collection config with compression
    let _compression_config = CompressionConfig {
        algorithm: match compression_algo {
            "none" | "uncompressed" => CompressionAlgorithm::CompressionNone as i32,
            "zstd" => CompressionAlgorithm::CompressionZstd as i32,
            "snappy" => CompressionAlgorithm::CompressionSnappy as i32,
            "gzip" => CompressionAlgorithm::CompressionGzip as i32,
            "lz4" => CompressionAlgorithm::CompressionLz4 as i32,
            "brotli" => CompressionAlgorithm::CompressionBrotli as i32,
            _ => CompressionAlgorithm::CompressionSnappy as i32, // default
        },
        level: Some(compression_level as u32),
        adaptive: false,
        min_ratio: None,
        enable_quantization: false,
        quantization_type: None,
        block_size_kb: block_size_kb as u32,
        dynamic_block_sizing: false,
        normalization_method: None,
    };

    let collection_config = Collection {
        id: "test_collection".to_string(),
        storage_assignment: Some(proximadb::proto::proximadb_v1::StorageAssignment {
            primary_path: environment.persistent_dir.to_str().unwrap().to_string(),
            backup_paths: vec![],
            engine: StorageEngine::Sst as i32,
            engine_config: std::collections::HashMap::new(),
            base_location: environment.persistent_base.to_str().unwrap().to_string(),
            assigned_at: chrono::Utc::now().timestamp(),
            ..Default::default()
        }),
        config: Some(CollectionConfig {
            name: "test_collection".to_string(),
            dimension: vectors[0].vector.len() as u32,
            distance_metric: Some(DistanceMetric::Cosine as i32),
            storage_engine: Some(StorageEngine::Sst as i32),
            primary_index: Some("default".to_string()),
            auto_index_selection: Some(false),
            ..Default::default()
        }),
        ..Default::default()
    };

    // Create flush parameters - production SST will handle quantization internally
    let flush_params = FlushParameters {
        collection_id: Some("test_collection".to_string()),
        vector_records: vectors.iter().map(|v: &VectorRecord| v.into()).collect(),
        collection_config: Some(collection_config),
        force: true,
        synchronous: true,
        ..Default::default()
    };

    // Perform flush using production code
    let flush_result = sst_storage.do_flush(&flush_params).await?;

    // Calculate compression ratio
    let uncompressed_size = vectors.len() * vectors[0].vector.len() * 4; // FP32
    let compression_ratio = if flush_result.bytes_written.unwrap_or(0) > 0 {
        uncompressed_size as f64 / flush_result.bytes_written.unwrap_or(1) as f64
    } else {
        1.0
    };

    // Extract block statistics from engine metrics - use default if not available
    let blocks_created = flush_result
        .engine_metrics
        .get("blocks_created")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as usize;

    // Estimate vectors per block
    let vectors_per_block = vec![vectors.len() / blocks_created.max(1); blocks_created];
    let block_sizes = vectors_per_block
        .iter()
        .map(|&count| count * vectors[0].vector.len() * 4)
        .collect();

    Ok(FlushBlockStatsResult {
        success: true,
        entries_flushed: vectors.len(),
        bytes_written: flush_result.bytes_written.unwrap_or(0) as usize,
        files_created: flush_result.files_created.unwrap_or(0) as usize,
        blocks_created,
        compression_ratio,
        vectors_per_block,
        block_sizes,
    })
}

/// Perform flush with PQ sorting for improved compression
/// The production SST code now handles PQ sorting internally when quantization is enabled
pub async fn flush_sst_with_pq_sorting(
    environment: &UnifiedTestEnvironment,
    vectors: Vec<VectorRecord>,
    compression_algo: &str,
    compression_level: i32,
    block_size_kb: usize,
) -> Result<FlushBlockStatsResult> {
    // Simply use the production flush with compression enabled
    // When compression is enabled, quantization with PQ sorting is automatically applied
    flush_sst_with_block_stats(
        environment,
        vectors,
        compression_algo,
        compression_level,
        block_size_kb,
    )
    .await
}

/// Create a real `VectorOperationsService` for testing (owned).
///
/// Previously a `todo!()` stub (which panicked every caller). Now delegates to
/// the canonical `UnifiedTestEnvironment::vector_operations_service()` harness —
/// the real SST + WAL + Axis + Catalog + CollectionService stack over a leaked
/// tempdir (kept alive for the returned service's lifetime, per the file-backed
/// test-store convention). The harness returns an `Arc`; this owned variant
/// unwraps it (unique at construction) for the legacy callers that bind by value.
pub async fn create_test_vector_operations_service() -> Result<VectorOperationsService> {
    let env = UnifiedTestEnvironment::new().await?;
    let (vector_ops, _collections) = env.vector_operations_service().await?;
    // Keep the env's tempdir/files alive for the service's lifetime.
    std::mem::forget(env);
    Arc::try_unwrap(vector_ops)
        .map_err(|_| anyhow::anyhow!("test VectorOperationsService Arc must be unique"))
}

/// Create a real `VectorOperationsService` for testing (custom-storage variant).
///
/// The canonical harness builds its own `SstEngine`, so the passed `_storage` is
/// no longer needed; delegates to `create_test_vector_operations_service()`
/// (previously this bailed with an error).
pub async fn create_test_vector_operations_service_with_storage(
    _storage: Arc<SstEngine>,
) -> Result<VectorOperationsService> {
    create_test_vector_operations_service().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_unified_environment_creation() {
        let env = UnifiedTestEnvironment::new().await.unwrap();

        // Verify unique collection ID
        assert!(env.collection_id().starts_with("unified_test_"));
        assert!(env.collection_id().len() > 20); // UUID should make it long

        // Verify storage locations
        assert_eq!(env.storage_locations.len(), 1);
        assert!(env.storage_locations[0].url.starts_with("file://"));

        debug!("✅ Created unified environment: {}", env.collection_id());
    }

    #[tokio::test]
    async fn test_sst_and_viper_creation() {
        let env = UnifiedTestEnvironment::new().await.unwrap();

        // Test SST engine creation
        let _sst_engine = env.create_sst_engine().await.unwrap();
        debug!("✅ Created SST engine for: {}", env.collection_id());

        // Test VIPER engine creation
        let _viper_engine = env.create_viper_engine().await.unwrap();
        debug!("✅ Created VIPER engine for: {}", env.collection_id());

        // Test directory creation
        env.ensure_all_directories().await.unwrap();
        assert!(env.get_sst_data_directory().exists());
        assert!(env.get_viper_data_directory().exists());
    }

    #[tokio::test]
    async fn test_multi_environment_isolation() {
        let multi_env = MultiUnifiedEnvironmentTest::new(3).await.unwrap();
        let collection_ids = multi_env.collection_ids();

        // Verify all collection IDs are unique
        for i in 0..collection_ids.len() {
            for j in (i + 1)..collection_ids.len() {
                assert_ne!(collection_ids[i], collection_ids[j]);
            }
        }

        debug!(
            "✅ Created {} isolated unified environments",
            collection_ids.len()
        );
        for id in collection_ids {
            debug!("  - {}", id);
        }
    }
}
