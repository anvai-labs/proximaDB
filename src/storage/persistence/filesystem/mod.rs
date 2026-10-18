/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! # Filesystem Abstraction Layer - Unified Storage Interface
//!
//! This module provides ProximaDB's cloud-native filesystem abstraction that enables
//! seamless operation across local and cloud storage systems. It implements the Strategy
//! Pattern with Abstract Factory for automatic backend selection based on URL schemes.
//!
//! ## Role in ProximaDB Architecture
//!
//! The filesystem layer provides storage-agnostic operations:
//! ```text
//! Storage Engines → Filesystem API → Backend Selection
//!                                           ↓
//!                    ┌──────────────────────┴──────────────────┐
//!                    │         Storage Backends                 │
//!                    ├───────────────────────────────────────────┤
//!                    │ Local │ S3 │ Azure │ GCS │ HDFS │       │
//!                    └───────────────────────────────────────────┘
//!                                           ↓
//!                          Zero-Copy I/O + Caching Layer
//! ```
//!
//! ## Supported Storage Backends
//!
//! | Scheme | Backend | Features | Use Case |
//! |--------|---------|----------|----------|
//! | `file://` | Local filesystem | Direct I/O, memory mapping | Development, single-node |
//! | `s3://` | Amazon S3 | IAM roles, STS, multipart | AWS deployments |
//! | `adls://` | Azure Data Lake | Managed identity, SAS | Azure deployments |
//! | `gcs://` | Google Cloud Storage | Service accounts, ADC | GCP deployments |
//! | `hdfs://` | Hadoop HDFS | Kerberos, HA namenode | Big data clusters |
//!
//! ## Key Features
//!
//! ### 1. **Transparent Backend Selection**
//! Automatic routing based on URL scheme:
//! ```rust,ignore
//! use std::sync::Arc;
//! // Create a factory and get a filesystem by URL
//! let factory = Arc::new(FilesystemFactory::create_default().await?);
//! let fs = factory.get_filesystem("s3://bucket/path")?; // S3 backend
//! let local = factory.get_filesystem("file:///data/vectors")?; // Local backend
//! ```
//!
//! ### 2. **Atomic Operations**
//! Configurable strategies for data consistency:
//! - **DirectWrite**: For filesystems with native atomicity
//! - **SameDirectory**: Temp files in `___temp/` subdirectory
//! - **ConfiguredTemp**: User-specified temp location
//! - **SystemTemp**: Fallback to `/tmp` for development
//!
//! ### 3. **Zero-Copy I/O System**
//! High-performance I/O with intelligent caching:
//! - Memory-mapped files for local storage
//! - Bandwidth optimization for cloud
//! - Prefetching and read-ahead
//! - LRU cache with TTL support
//!
//! ### 4. **Cloud-Native Authentication**
//! Automatic credential management:
//! - **AWS**: IAM roles, instance profiles, STS
//! - **Azure**: Managed identity, service principals
//! - **GCS**: Service accounts, application default
//! - **HDFS**: Kerberos, simple auth
//!
//! ## Performance Characteristics
//!
//! - **Local I/O**: < 1ms latency, GB/s throughput
//! - **S3 Operations**: 10-50ms latency, 100MB/s throughput
//! - **Cache Hit Rate**: 80-95% for hot data
//! - **Memory Mapping**: Zero-copy for local files
//! - **Multipart Upload**: Parallel chunks for large files
//!
//! ## Configuration
//!
//! ```toml
//! [storage.filesystem]
//! # Default filesystem URL
//! default_url = "file:///data"
//!
//! # Atomic write strategy
//! temp_strategy = "same_directory"
//!
//! # Zero-copy configuration
//! enable_mmap = true
//! cache_size_mb = 1024
//! prefetch_size_kb = 256
//!
//! # S3 specific
//! [storage.filesystem.s3]
//! region = "us-west-2"
//! max_connections = 100
//! multipart_threshold_mb = 64
//!
//! # Azure specific
//! [storage.filesystem.azure]
//! account = "myaccount"
//! use_managed_identity = true
//! ```
//!
//! ## Module Organization
//!
//! - **`local.rs`**: Local filesystem implementation
//! - **`s3.rs`**: Amazon S3 backend
//! - **`azure.rs`**: Azure Data Lake Storage
//! - **`gcs.rs`**: Google Cloud Storage
//! - **`hdfs.rs`**: Hadoop HDFS support
//! - **`zero_copy_filesystem.rs`**: High-performance I/O layer
//! - **`atomic_strategy.rs`**: Atomic write implementations
//! - **`manager.rs`**: Filesystem factory and routing
//! - **`auth/`**: Authentication providers
//!
//! ## Usage Examples
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use proximadb::storage::persistence::filesystem::{FilesystemFactory, FileOptions};
//!
//! // Create factory and filesystem
//! let factory = Arc::new(FilesystemFactory::create_default().await?);
//! let fs = factory.get_filesystem("s3://my-bucket/vectors")?;
//!
//! // Write with atomic guarantees
//! fs.write_atomic(
//!     "collection/segment.parquet",
//!     data,
//!     FileOptions::default()
//! ).await?;
//!
//! // Read with caching
//! let content = fs.read("collection/segment.parquet").await?;
//!
//! // List directory
//! let entries = fs.list("collection/").await?;
//! ```
//!
//! ## Error Handling
//!
//! The module provides detailed error types:
//! - `FilesystemError::Io`: Low-level I/O failures
//! - `FilesystemError::Auth`: Authentication issues
//! - `FilesystemError::Network`: Connection problems
//! - `FilesystemError::NotFound`: Missing files/paths
//!
//! ## Cloud Cost Optimization
//!
//! Built-in features to minimize cloud storage costs:
//! - Intelligent caching reduces API calls
//! - Batch operations for list/delete
//! - Storage class transitions (S3 IA, Glacier)
//! - Bandwidth optimization with compression

use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error, info, trace};
use url::Url;

pub mod atomic_strategy;
// intelligent_filesystem removed - using UnifiedCachingFilesystem instead
pub mod local;
// Cloud object-store backends, built on the official SDKs (which handle signing,
// custom endpoints, and path-style — incl. MinIO/Azurite/fake-gcs emulators) and
// gated behind their features. These replace the deleted legacy s3.rs/azure.rs/
// gcs.rs/auth.rs/manager.rs, which were dead, incomplete, hand-rolled clients.
#[cfg(feature = "aws")]
pub mod aws_s3;
#[cfg(feature = "azure")]
pub mod azure_blob;
#[cfg(feature = "gcp")]
pub mod gcs_store;
pub mod scheme_validation;
pub mod write_strategy;
// zero_copy_filesystem removed - functionality integrated into UnifiedCachingFilesystem

// Unified filesystem modules
pub mod access_tracker;
pub mod cache_config;
pub mod cache_metrics;
pub mod caching_filesystem;
pub mod disk_cache;
pub mod metadata_cache;
pub mod metadata_traits;
pub mod orchestrator_integration;
pub mod prefetch_engine;
pub mod range_optimizer;
pub mod smart_io;

// Filesystem implementations
pub use local::LocalFileSystem;

// Unified caching filesystem
pub use caching_filesystem::UnifiedCachingFilesystem;

// Re-export centralized scheme validation functions
pub use scheme_validation::{
    FilesystemScheme, extract_scheme, is_supported_scheme, normalize_url, validate_url,
};

// The filesystem abstraction trait + value types live in the leaf crate
// `proximadb-storage-filesystem-types` (extracted to break the
// encryption/schema/common/metadata → root coupling). Re-exported here so every
// existing `crate::storage::persistence::filesystem::{FileSystem, ...}` path
// keeps resolving unchanged.
pub use proximadb_storage_filesystem_types::*;

/// Filesystem factory configuration
#[derive(Debug, Clone)]
pub struct FilesystemConfig {
    /// Default filesystem URL for unqualified paths
    pub default_fs: Option<String>,

    /// Local filesystem configuration
    pub local: Option<local::LocalConfig>,

    /// Global filesystem options
    pub global_options: FileOptions,

    /// Authentication configuration
    pub auth_config: Option<FilesystemAuthConfig>,

    /// Performance optimization settings
    pub performance_config: FilesystemPerformanceConfig,

    /// Scheme mapping for URL scheme overrides (e.g., "gs" -> "gcs")
    pub scheme_mapping: HashMap<String, String>,
}

impl Default for FilesystemConfig {
    fn default() -> Self {
        Self {
            default_fs: Some("file://".to_string()),
            local: Some(local::LocalConfig::default()),
            global_options: FileOptions::default(),
            auth_config: None,
            performance_config: FilesystemPerformanceConfig::default(),
            scheme_mapping: {
                let mut mapping = HashMap::new();
                mapping.insert("gs".to_string(), "gcs".to_string()); // Support Google Cloud gs:// scheme
                mapping
            },
        }
    }
}

/// Abstract factory for creating filesystem instances
pub struct FilesystemFactory {
    config: FilesystemConfig,
    filesystems: HashMap<String, Arc<dyn FileSystem>>,
    tier_mapping: HashMap<FileStorageTier, String>,
}

impl std::fmt::Debug for FilesystemFactory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilesystemFactory")
            .field("config", &self.config)
            .field("filesystems_count", &self.filesystems.len())
            .finish()
    }
}

/// Record a successful whole-object read at the physical filesystem boundary.
/// This is always-on query evidence; `PROXIMADB_COUNT_FS_IO` controls only the
/// separate process-global benchmark counters.
pub(super) fn record_physical_full_read(bytes: u64) {
    crate::observability::io_trace::record_op(crate::observability::io_trace::IoOp::Get);
    crate::observability::io_trace::record_bytes_read(bytes);
}

/// Record a successful ranged read at the physical filesystem boundary.
/// `get_ops` is the total physical GET count; `range_gets` is its ranged subset.
pub(super) fn record_physical_range_read(bytes: u64) {
    crate::observability::io_trace::record_op(crate::observability::io_trace::IoOp::Get);
    crate::observability::io_trace::record_range_gets(1);
    crate::observability::io_trace::record_bytes_read(bytes);
}

/// Records `ProximaObjectStore` reads into the per-query `IoTrace`.
///
/// `ProximaObjectStore` goes straight to upstream `object_store`, bypassing the
/// root `FileSystem` leaf backends and their always-on helpers above. This
/// recorder is therefore the sole source for that stack and records each
/// physical counter in full.
#[derive(Debug)]
pub struct IoTraceObjectStoreRecorder;

impl proximadb_storage_filesystem_types::counting::IoRecorder for IoTraceObjectStoreRecorder {
    fn record_full_read(&self, bytes: u64) {
        crate::observability::io_trace::record_op(crate::observability::io_trace::IoOp::Get);
        if bytes > 0 {
            crate::observability::io_trace::record_bytes_read(bytes);
        }
    }

    fn record_range_read(&self, bytes: u64) {
        crate::observability::io_trace::record_op(crate::observability::io_trace::IoOp::Get);
        crate::observability::io_trace::record_range_gets(1);
        if bytes > 0 {
            crate::observability::io_trace::record_bytes_read(bytes);
        }
    }

    fn record_batched_ranges(&self, physical_gets: u64, bytes: u64) {
        // `physical_gets` here is an UPPER BOUND, not a measurement: upstream
        // `object_store::get_ranges` coalesces internally at a hard-coded 1 MiB
        // gap and never reports how many HTTP requests it issued. Over-stating
        // requests is the safe direction — it can never hide cost — but this
        // number must not be compared naively against the FileSystem stack's
        // measured counts.
        for _ in 0..physical_gets {
            crate::observability::io_trace::record_op(crate::observability::io_trace::IoOp::Get);
        }
        crate::observability::io_trace::record_range_gets(physical_gets);
        if bytes > 0 {
            crate::observability::io_trace::record_bytes_read(bytes);
        }
    }
}

impl FilesystemFactory {
    fn maybe_wrap_with_encryption(
        &self,
        filesystem: Arc<dyn FileSystem>,
    ) -> FsResult<Arc<dyn FileSystem>> {
        let Some(env_var) = self.config.global_options.encryption.as_deref() else {
            return Ok(filesystem);
        };

        let env_var = env_var.trim();
        if env_var.is_empty() {
            return Err(FilesystemError::Config(
                "Filesystem encryption env var name cannot be empty".to_string(),
            ));
        }

        let key_manager =
            crate::storage::encryption::KeyManager::from_env(env_var).map_err(|e| {
                FilesystemError::Config(format!(
                    "Failed to load filesystem encryption key from {}: {}",
                    env_var, e
                ))
            })?;
        let version_manager = Arc::new(crate::storage::encryption::KeyVersionManager::new(
            Arc::new(key_manager),
        ));

        Ok(Arc::new(
            crate::storage::encryption::EncryptedFilesystem::new(filesystem, version_manager, true),
        ))
    }

    /// Create a fully initialized filesystem factory with default configuration
    ///
    /// This is the preferred way to create a FilesystemFactory. It initializes all
    /// filesystem backends (local, cloud) based on the default configuration.
    ///
    /// # Examples
    /// ```ignore
    /// let factory = FilesystemFactory::create_default().await?;
    /// let fs = factory.get_filesystem("file:///tmp/data")?;
    /// ```text
    pub async fn create_default() -> FsResult<Self> {
        Self::create(FilesystemConfig::default()).await
    }

    /// Create a fully initialized filesystem factory with custom configuration
    ///
    /// This static factory method creates and initializes a FilesystemFactory with
    /// all configured filesystem backends registered and ready to use.
    ///
    /// # Arguments
    /// * `config` - Filesystem configuration specifying which backends to enable
    ///
    /// # Examples
    /// ```ignore
    /// let config = FilesystemConfig {
    ///     local: Some(LocalConfig::default()),
    ///     ..Default::default()
    /// };
    /// let factory = FilesystemFactory::create(config).await?;
    /// ```text
    pub async fn create(config: FilesystemConfig) -> FsResult<Self> {
        let mut factory = Self {
            config,
            filesystems: HashMap::new(),
            tier_mapping: HashMap::new(),
        };

        // Pre-initialize configured filesystems
        factory.initialize_filesystems().await?;

        // Build tier mapping from configuration
        factory.initialize_tier_mapping();

        Ok(factory)
    }

    /// Initialize all configured filesystem backends
    async fn initialize_filesystems(&mut self) -> FsResult<()> {
        // Initialize local filesystem with root directory resolution
        if let Some(local_config) = &self.config.local {
            let local_fs: Arc<dyn FileSystem> =
                Arc::new(LocalFileSystem::new(local_config.clone()).await?);
            self.filesystems.insert(
                "file".to_string(),
                self.maybe_wrap_with_encryption(local_fs)?,
            );
        } else {
            // Create default local filesystem without root restriction
            let default_config = local::LocalConfig::default();
            let local_fs: Arc<dyn FileSystem> =
                Arc::new(LocalFileSystem::new(default_config).await?);
            self.filesystems.insert(
                "file".to_string(),
                self.maybe_wrap_with_encryption(local_fs)?,
            );
        }

        // Cloud object-store backends (real official-SDK FileSystem impls). Each
        // is registered by scheme when its feature is compiled in; config comes
        // from env (standard cloud env vars + PROXIMADB_* overrides for custom
        // endpoints like MinIO/Azurite/fake-gcs). Registration is best-effort —
        // a misconfigured cloud backend logs a warning and is skipped rather than
        // failing factory init. The default build (no cloud feature) registers
        // only "file". `get_filesystem(url)` then dispatches s3://, gs://, adls://
        // to these without further changes.
        #[cfg(feature = "aws")]
        match aws_s3::AwsS3FileSystem::new(Self::aws_s3_config_from_env()).await {
            Ok(backend) => {
                let fs =
                    self.maybe_wrap_with_encryption(Arc::new(backend) as Arc<dyn FileSystem>)?;
                self.filesystems.insert("s3".to_string(), fs);
            }
            Err(e) => tracing::warn!("S3 FileSystem not registered: {e}"),
        }
        #[cfg(feature = "azure")]
        match azure_blob::AzureBlobFileSystem::new(Self::azure_config_from_env()).await {
            Ok(backend) => {
                let fs =
                    self.maybe_wrap_with_encryption(Arc::new(backend) as Arc<dyn FileSystem>)?;
                // Scheme honesty (ADR-036): all four schemes resolve to the SAME
                // Azure backend, which talks to the **Blob endpoint**
                // (`*.blob.core.windows.net`, flat object keys) via `object_store`'s
                // `MicrosoftAzureBuilder`. `az`/`azure` are the canonical schemes.
                // `adls`/`abfs` are accepted **aliases** for ergonomics — but they
                // do NOT engage the ADLS Gen2 DFS endpoint, the ABFS Hadoop driver,
                // or Hierarchical Namespace; they are a Blob-endpoint write under a
                // familiar name. We deliberately run flat Blob (HNS-off) + access-tier
                // as the cost lever; HNS buys nothing for our flat-key, immutable,
                // ranged-read workload. The one-time log makes the aliasing explicit
                // so an operator never assumes DFS/HNS semantics from the scheme.
                for scheme in ["az", "azure", "adls", "abfs"] {
                    self.filesystems.insert(scheme.to_string(), fs.clone());
                }
                tracing::info!(
                    canonical = "az://, azure://",
                    aliases = "adls://, abfs://",
                    endpoint = "blob.core.windows.net (flat, HNS-off)",
                    "Azure FileSystem registered: all schemes route to the Blob \
                     endpoint; adls/abfs are aliases and do NOT use the DFS/ABFS \
                     endpoint or Hierarchical Namespace. Cost lever = access tier."
                );
            }
            Err(e) => tracing::warn!("Azure FileSystem not registered: {e}"),
        }
        #[cfg(feature = "gcp")]
        match gcs_store::GcsFileSystem::new(Self::gcs_config_from_env()).await {
            Ok(backend) => {
                let fs =
                    self.maybe_wrap_with_encryption(Arc::new(backend) as Arc<dyn FileSystem>)?;
                for scheme in ["gcs", "gs"] {
                    self.filesystems.insert(scheme.to_string(), fs.clone());
                }
            }
            Err(e) => tracing::warn!("GCS FileSystem not registered: {e}"),
        }

        Ok(())
    }

    /// Build the S3 backend config from env (`AWS_*` + `PROXIMADB_S3_*` overrides
    /// for custom endpoints / path-style, e.g. MinIO).
    #[cfg(feature = "aws")]
    fn aws_s3_config_from_env() -> aws_s3::AwsS3Config {
        aws_s3::AwsS3Config {
            region: std::env::var("PROXIMADB_S3_REGION")
                .or_else(|_| std::env::var("AWS_REGION"))
                .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
                .unwrap_or_else(|_| "us-east-1".to_string()),
            endpoint_url: std::env::var("PROXIMADB_S3_ENDPOINT").ok(),
            force_path_style: std::env::var("PROXIMADB_S3_FORCE_PATH_STYLE")
                .map(|v| matches!(v.as_str(), "1" | "true" | "yes"))
                .unwrap_or(false),
            access_key_id: std::env::var("AWS_ACCESS_KEY_ID").ok(),
            secret_access_key: std::env::var("AWS_SECRET_ACCESS_KEY").ok(),
            session_token: std::env::var("AWS_SESSION_TOKEN").ok(),
        }
    }

    /// Build the Azure backend config from env (`AZURE_STORAGE_*`;
    /// `PROXIMADB_AZURE_EMULATOR=1` for Azurite).
    #[cfg(feature = "azure")]
    fn azure_config_from_env() -> azure_blob::AzureBlobConfig {
        azure_blob::AzureBlobConfig {
            account: std::env::var("AZURE_STORAGE_ACCOUNT")
                .unwrap_or_else(|_| "devstoreaccount1".to_string()),
            access_key: std::env::var("AZURE_STORAGE_KEY").ok(),
            use_emulator: std::env::var("PROXIMADB_AZURE_EMULATOR")
                .map(|v| matches!(v.as_str(), "1" | "true" | "yes"))
                .unwrap_or(false),
            endpoint: std::env::var("AZURE_STORAGE_ENDPOINT").ok(),
            // Secret-less auth: the AKS workload-identity webhook projects these
            // three env vars into the pod, so an MVP deployment with a federated
            // identity authenticates to ADLS with no storage key in config.
            // `AZURE_CLIENT_ID` alone (no tenant/token) selects a user-assigned
            // Managed Identity; absent entirely, the system-assigned MI is used.
            client_id: std::env::var("AZURE_CLIENT_ID").ok(),
            tenant_id: std::env::var("AZURE_TENANT_ID").ok(),
            federated_token_file: std::env::var("AZURE_FEDERATED_TOKEN_FILE").ok(),
        }
    }

    /// Build the GCS backend config from env (`PROXIMADB_GCS_*`;
    /// `PROXIMADB_GCS_ANONYMOUS=1` + endpoint for fake-gcs-server).
    #[cfg(feature = "gcp")]
    fn gcs_config_from_env() -> gcs_store::GcsConfig {
        gcs_store::GcsConfig {
            endpoint_url: std::env::var("PROXIMADB_GCS_ENDPOINT").ok(),
            anonymous: std::env::var("PROXIMADB_GCS_ANONYMOUS")
                .map(|v| matches!(v.as_str(), "1" | "true" | "yes"))
                .unwrap_or(false),
            project_id: std::env::var("GCP_PROJECT")
                .or_else(|_| std::env::var("GOOGLE_CLOUD_PROJECT"))
                .ok(),
        }
    }

    /// Get filesystem instance for URL scheme (cached instances)
    /// Get filesystem instance for URL scheme (returns Arc for safe sharing).
    ///
    /// Use this when you need raw filesystem access without caching.
    pub fn get_filesystem(&self, url: &str) -> FsResult<Arc<dyn FileSystem>> {
        // Use centralized scheme extraction and validation
        let scheme = extract_scheme(url)?;
        let scheme_str = scheme.as_str();

        let fs = self
            .filesystems
            .get(scheme_str)
            .cloned()
            .ok_or_else(|| FilesystemError::UnsupportedScheme(scheme_str.to_string()))?;

        // TD-096 S2: the env gate enables process-global benchmark counters.
        // Per-query IoTrace accounting is an independent production invariant
        // recorded by each physical leaf backend, so routing evidence cannot
        // silently disappear when this diagnostic gate is OFF.
        // TD-COMPACT-13 TDD: the env-gated delete-fault wrapper sits INSIDE the
        // counting wrapper so retirement tests observe the fault while the
        // counters keep counting the inner backend. Unset = byte-identical.
        let fs: Arc<dyn FileSystem> = if std::env::var_os("PROXIMADB_TEST_FS_DELETE_FAIL_FIRST")
            .is_some()
            || proximadb_storage_filesystem_types::faults::read_fault_requested()
        {
            Arc::new(proximadb_storage_filesystem_types::faults::FaultInjectingFileSystem::new(fs))
        } else {
            fs
        };
        if std::env::var_os("PROXIMADB_COUNT_FS_IO").is_some() {
            Ok(Arc::new(
                proximadb_storage_filesystem_types::counting::CountingFileSystem::new(
                    fs,
                    proximadb_storage_filesystem_types::counting::global_counters(),
                ),
            ))
        } else {
            Ok(fs)
        }
    }

    /// Create filesystem with unified caching
    ///
    /// # Example
    /// ```text
    /// let cached_fs = factory.get_unified_caching_filesystem(
    ///     "s3://bucket/collection",
    ///     "collection_123".to_string(),
    ///     "sst".to_string(),
    /// )?;
    /// ```text
    pub fn get_unified_caching_filesystem(
        &self,
        url: &str,
        collection_id: String,
        engine_type: String,
    ) -> FsResult<Arc<dyn FileSystem>> {
        // Get the appropriate filesystem for this URL
        let fs = self.get_filesystem(url)?;

        // Get the metadata serializer for this engine type
        let metadata_serializer: Arc<dyn crate::storage::persistence::filesystem::metadata_traits::EngineMetadataSerializer> =
            match engine_type.as_str() {
                "sst" => Arc::new(crate::storage::engines::core::sst_format_serializer::SstUnifiedMetadataSerializer::new()),
                "viper" => Arc::new(crate::storage::engines::core::parquet_format_serializer::ViperMetadataSerializer::new()),
                "raptor" => Arc::new(crate::storage::engines::core::matrix_trinity_serializer::RaptorUnifiedMetadataSerializer::new()),
                "nova" => Arc::new(crate::storage::engines::core::columnar_format_serializer::NovaUnifiedMetadataSerializer::new()),
                "swift" => Arc::new(crate::storage::engines::core::proximablocks_compact_serializer::SwiftUnifiedMetadataSerializer::new()),
                "helix" => Arc::new(crate::storage::engines::core::proximablocks_format_serializer::HelixUnifiedMetadataSerializer::new()),
                _ => {
                    // Default serializer for other engines
                    #[derive(Debug)]
                    struct DefaultSerializer;
                    impl crate::storage::persistence::filesystem::metadata_traits::EngineMetadataSerializer for DefaultSerializer {
                        fn serialize(&self, _metadata: &dyn std::any::Any) -> anyhow::Result<bytes::Bytes> {
                            Ok(bytes::Bytes::new())
                        }
                        fn deserialize(&self, _bytes: &[u8]) -> anyhow::Result<Box<dyn std::any::Any + Send + Sync>> {
                            Ok(Box::new(()))
                        }
                        fn engine_type(&self) -> &str { "default" }
                        fn extract_cacheable_component(&self, _data: &[u8], _file_path: &str) -> Option<bytes::Bytes> {
                            None
                        }
                        fn should_cache_metadata(&self, _file_path: &str) -> bool { false }
                    }
                    Arc::new(DefaultSerializer)
                }
            };

        // Wrap it with UnifiedCachingFilesystem for caching
        let unified_fs = crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem::with_serializer(
            fs,
            collection_id.to_string(),
            engine_type.to_string(),
            metadata_serializer,
        );

        Ok(Arc::new(unified_fs))
    }

    /// Cross-storage atomic operations - handles full URLs for source and destination
    pub async fn copy_atomic(&self, from_url: &str, to_url: &str) -> FsResult<()> {
        trace!("📋 [] copy_atomic START");
        trace!("from_url: {}", from_url);
        trace!("to_url: {}", to_url);
        debug!("📋 [DEBUG] copy_atomic START");
        debug!("    from_url: {}", from_url);
        debug!("    to_url: {}", to_url);

        let from_fs = self.get_filesystem(from_url)?;
        let to_fs = self.get_filesystem(to_url)?;

        // Extract paths from URLs
        let from_path = Self::resolve_path(from_url)?;
        let to_path = Self::resolve_path(to_url)?;

        trace!("from_path: {}", from_path);
        trace!("to_path: {}", to_path);
        debug!("    [DEBUG] from_path resolved: {}", from_path);
        debug!("    [DEBUG] to_path resolved: {}", to_path);

        // Same-backend fast path (TD-OBJSTORE-4 S1 review): comparing scheme
        // STRINGS classifies az://→adls:// as cross-filesystem, and even same
        // schemes fall into the streaming path below — which cloud backends do
        // not support ("streaming open_file is not supported on the Azure
        // backend"). Compare CANONICAL scheme groups (az/azure/adls/abfs are one
        // backend; gs/gcs likewise) rather than Arc identity: per-call wrappers
        // (the TD-096 PROXIMADB_COUNT_FS_IO CountingFileSystem) mint a fresh Arc
        // per lookup and would silently defeat a ptr_eq check. Only cloud groups
        // take this path — their whole-object copy() avoids the unsupported
        // streaming API (today's backend copy() is client-side read+write;
        // server-side copy is a follow-up optimization).
        let same_cloud_backend = match (extract_scheme(from_url), extract_scheme(to_url)) {
            (Ok(from_scheme), Ok(to_scheme)) => {
                use crate::storage::persistence::filesystem::scheme_validation::FilesystemScheme;
                let group = |s: FilesystemScheme| match s {
                    FilesystemScheme::AzureBlobStorage | FilesystemScheme::AzureDataLakeStorage => {
                        Some("azure")
                    }
                    FilesystemScheme::S3 => Some("s3"),
                    FilesystemScheme::GoogleCloudStorage => Some("gs"),
                    _ => None, // file/hdfs keep the existing streaming path
                };
                match (group(from_scheme), group(to_scheme)) {
                    (Some(a), Some(b)) => a == b,
                    _ => false,
                }
            }
            _ => false,
        };
        if same_cloud_backend || Arc::ptr_eq(&from_fs, &to_fs) {
            from_fs.copy(&from_path, &to_path).await?;
            trace!("Same-backend native copy complete");
            debug!("    ✅ [DEBUG] Same-backend native copy complete");
            trace!("📋 [] copy_atomic COMPLETE");
            debug!("📋 [DEBUG] copy_atomic COMPLETE");
            return Ok(());
        }

        if from_fs.filesystem_type() == "encrypted" || to_fs.filesystem_type() == "encrypted" {
            let data = from_fs.read(&from_path).await?;
            to_fs.write_atomic(&to_path, &data, None).await?;
            trace!("Whole-file copy complete");
            debug!("    ✅ [DEBUG] Whole-file copy complete");
            trace!("📋 [] copy_atomic COMPLETE");
            debug!("📋 [DEBUG] copy_atomic COMPLETE");
            return Ok(());
        }

        // Open source and destination files for streaming
        trace!("Opening source file for streaming...");
        let mut source_file = from_fs.open_file(&from_path, false).await?;
        trace!("Opening destination file for streaming...");
        let mut dest_file = to_fs.open_file(&to_path, true).await?;

        // Stream data in chunks
        let mut buffer = vec![0; 8 * 1024 * 1024]; // 8MB buffer
        loop {
            let bytes_read = source_file.read(&mut buffer).await?;
            if bytes_read == 0 {
                break;
            }
            dest_file.write(&buffer[..bytes_read]).await?;
        }

        // Flush and sync destination file
        dest_file.flush().await?;
        dest_file.sync_all().await?;

        trace!("Streaming copy complete");
        debug!("    ✅ [DEBUG] Streaming copy complete");

        trace!("📋 [] copy_atomic COMPLETE");
        debug!("📋 [DEBUG] copy_atomic COMPLETE");
        Ok(())
    }

    /// Move operation with atomic cross-storage support
    pub async fn move_atomic(&self, from_url: &str, to_url: &str) -> FsResult<()> {
        trace!("🚚 [] move_atomic START");
        trace!("from_url: {}", from_url);
        trace!("to_url: {}", to_url);
        debug!("🚚 [DEBUG] move_atomic called:");
        debug!("    from_url: {}", from_url);
        debug!("    to_url: {}", to_url);

        // Copy first using streaming copy
        trace!("Copying file atomically (streaming)...");
        debug!("    📋 [DEBUG] Copying file atomically (streaming)...");
        self.copy_atomic(from_url, to_url).await?;
        debug!("Streaming copy successful");
        debug!("    ✅ [DEBUG] Streaming copy successful");

        // Delete source after successful copy
        trace!("Deleting source file...");
        debug!("    🗑️ [DEBUG] Deleting source file...");
        let from_fs = self.get_filesystem(from_url)?;
        let from_path = Self::resolve_path(from_url)?;
        trace!("from_path extracted: {}", from_path);
        debug!("    [DEBUG] from_path extracted: {}", from_path);
        from_fs.delete(&from_path).await?;
        trace!("Delete successful");
        debug!("    ✅ [DEBUG] Delete successful");

        trace!("🚚 [] move_atomic COMPLETE");
        debug!("🚚 [DEBUG] move_atomic COMPLETE");
        Ok(())
    }

    /// Extract bucket/container name from URL
    pub fn extract_bucket_from_url(&self, url: &str) -> FsResult<Option<String>> {
        // Handle URLs without schemes by prepending file://
        let normalized_url = if !url.contains("://") {
            format!("file://{}", url)
        } else {
            url.to_string()
        };

        // For file:// URLs, there's no bucket
        if normalized_url.starts_with("file://") {
            return Ok(None);
        }

        let parsed_url = Url::parse(&normalized_url)?;

        match parsed_url.scheme() {
            // Flat object stores (and the canonical Azure Blob schemes): the
            // container/bucket is the URL host. `az://container/blob` matches the
            // backend's flat parse — the Blob endpoint, no account/HNS segment
            // (ADR-036). `adls`/`abfs` keep their legacy account-in-path/host@account
            // shapes below for backward compatibility.
            "s3" | "gcs" | "gs" | "az" | "azure" => {
                // Bucket/container is the hostname
                Ok(parsed_url.host_str().map(|s| s.to_string()))
            }
            "adls" => {
                // Container is the second path segment
                let path_parts: Vec<&str> = parsed_url
                    .path()
                    .trim_start_matches('/')
                    .split('/')
                    .collect();
                if path_parts.len() >= 2 {
                    Ok(Some(path_parts[1].to_string()))
                } else {
                    Ok(None)
                }
            }
            "abfs" => {
                // Container is before @ in hostname
                if let Some(host) = parsed_url.host_str()
                    && let Some(at_pos) = host.find('@')
                {
                    return Ok(Some(host[..at_pos].to_string()));
                }
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    /// Extract account name from URL (for Azure)
    pub fn extract_account_from_url(&self, url: &str) -> FsResult<Option<String>> {
        // Handle URLs without schemes by prepending file://
        let normalized_url = if !url.contains("://") {
            format!("file://{}", url)
        } else {
            url.to_string()
        };

        // For file:// URLs, there's no account
        if normalized_url.starts_with("file://") {
            return Ok(None);
        }

        let parsed_url = Url::parse(&normalized_url)?;

        match parsed_url.scheme() {
            "adls" => {
                // Account is the first path segment
                let path_parts: Vec<&str> = parsed_url
                    .path()
                    .trim_start_matches('/')
                    .split('/')
                    .collect();
                if !path_parts.is_empty() && !path_parts[0].is_empty() {
                    Ok(Some(path_parts[0].to_string()))
                } else {
                    Ok(None)
                }
            }
            "abfs" => {
                // Account is after @ in hostname
                if let Some(host) = parsed_url.host_str()
                    && let Some(at_pos) = host.find('@')
                {
                    return Ok(Some(host[at_pos + 1..].to_string()));
                }
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    /// Extract relative path from URL (removes base path configured for the storage)
    pub fn extract_relative_path(&self, url: &str) -> FsResult<String> {
        info!("🔍 resolve_path: {}", url);

        // Handle URLs without schemes by prepending file://
        let normalized_url = if !url.contains("://") {
            format!("file://{}", url)
        } else {
            url.to_string()
        };

        // Log the normalized URL for debugging
        info!("    normalized_url: {}", normalized_url);

        // Handle file:// URLs specially to avoid URL parsing issues
        if let Some(after_file) = normalized_url.strip_prefix("file://") {
            let path = if let Some(after_slashes) = normalized_url.strip_prefix("file:///") {
                // Absolute path
                format!("/{}", after_slashes)
            } else {
                // Relative or other path
                after_file.to_string()
            };
            info!("    scheme: file, returning path: {}", path);
            return Ok(path);
        }

        // For non-file URLs, use URL parsing
        let parsed_url = match Url::parse(&normalized_url) {
            Ok(url) => url,
            Err(e) => {
                error!("Failed to parse URL '{}': {}", normalized_url, e);
                return Err(FilesystemError::UrlParse(e));
            }
        };
        let path = parsed_url.path();
        info!("    parsed path: {}", path);

        match parsed_url.scheme() {
            "s3" | "gcs" | "gs" => {
                // For object stores, remove the bucket from path
                let path_without_bucket = path.trim_start_matches('/');

                // Skip the bucket name (first path segment)
                if let Some(slash_pos) = path_without_bucket.find('/') {
                    Ok(path_without_bucket[slash_pos + 1..].to_string())
                } else {
                    // No path after bucket, return empty string
                    Ok(String::new())
                }
            }
            "adls" | "abfs" => {
                // For Azure, remove account/container from path
                let path_parts: Vec<&str> = path.trim_start_matches('/').split('/').collect();
                if path_parts.len() > 1 {
                    Ok(path_parts[1..].join("/"))
                } else {
                    Ok(String::new())
                }
            }
            "hdfs" => {
                // For HDFS, return the full path
                Ok(path.to_string())
            }
            _ => {
                // For unknown schemes, return path as-is
                Ok(path.to_string())
            }
        }
    }

    /// Extract scheme from URL, handling paths without schemes
    fn extract_scheme(&self, url: &str) -> FsResult<String> {
        debug!("extract_scheme called with URL: {}", url);
        if let Some(scheme_end) = url.find("://") {
            // Extract just the scheme part without parsing the full URL
            let raw_scheme = &url[..scheme_end];
            debug!("extracted scheme: {}", raw_scheme);

            // Check for scheme mapping (e.g., gs -> gcs)
            let mapped_scheme = self.config.scheme_mapping.get(raw_scheme);

            let result = mapped_scheme
                .cloned()
                .unwrap_or_else(|| raw_scheme.to_string());
            debug!("returning scheme: {}", result);
            Ok(result)
        } else {
            // No scheme present - assume local file
            debug!("no scheme found, returning 'file'");
            Ok("file".to_string())
        }
    }

    /// Centralized URL path extraction utility (handles relative paths correctly)
    /// This method should be used throughout the filesystem layer for consistent URL parsing
    /// Unified path extraction from URLs with consistent behavior
    /// This is the SINGLE method that should be used throughout the filesystem layer
    pub fn resolve_path(url: &str) -> FsResult<String> {
        trace!("🔍 [FILESYSTEM] resolve_path: Input URL = '{}'", url);

        // Case 1: No scheme present - return as-is (this preserves relative paths)
        if !url.contains("://") {
            trace!(
                "🔍 [FILESYSTEM] resolve_path: No scheme, returning as-is: '{}'",
                url
            );
            return Ok(url.to_string());
        }

        // Case 2: Handle file:// URLs specially to avoid URL parsing issues
        if let Some(after_file) = url.strip_prefix("file://") {
            // CRITICAL: Avoid URL parsing for file:// to prevent international domain name errors
            // The URL parser can fail on paths that look like domain names
            if url.starts_with("file://./") {
                // Explicit relative path: file://./path/to/file
                let relative_path = after_file; // Keep the "./"
                trace!(
                    "🔍 [FILESYSTEM] resolve_path: Explicit relative path: '{}'",
                    relative_path
                );
                Ok(relative_path.to_string())
            } else if let Some(absolute_path) = url.strip_prefix("file:///") {
                // Absolute path: file:///absolute/path
                // Remove "file:///" prefix
                trace!(
                    "🔍 [FILESYSTEM] resolve_path: Absolute path: '/{}'",
                    absolute_path
                );
                Ok(format!("/{}", absolute_path))
            } else {
                // Implicit relative path: file://relative/path (treat as relative)
                trace!(
                    "🔍 [FILESYSTEM] resolve_path: Implicit relative path: '{}'",
                    after_file
                );
                Ok(after_file.to_string())
            }
        } else {
            // Case 3: Cloud schemes (s3://, az://, adls://, abfs://, gs://, ...).
            // The cloud backends parse the FULL url themselves to recover the
            // container/bucket + blob key — e.g. `AzureBlobFileSystem::parse`
            // strips the `az://` scheme and split_once('/')s the container off
            // the blob. Returning only `parsed_url.path()` here drops the host
            // (the container), so the backend rejects it: TD-FLUSH-6 root cause
            // was the staging-dir `list` failing with
            // `Invalid path: not an azure path: /1/data/__flush` (container
            // `proximadb-bench` stripped). Pass the url through verbatim and let
            // the backend parse it; only `file://` (Case 2 above) and scheme-less
            // paths (Case 1) are reduced to a local path. Url::parse is kept as
            // a fail-fast validity check; its `.path()` is intentionally unused.
            let _parsed = Url::parse(url)
                .map_err(|e| FilesystemError::InvalidPath(format!("Invalid URL: {}", e)))?;
            trace!(
                "🔍 [FILESYSTEM] resolve_path: cloud scheme — full url passed to backend: '{}'",
                url
            );
            Ok(url.to_string())
        }
    }

    /// Initialize tier-to-URL mapping from configuration
    fn initialize_tier_mapping(&mut self) {
        for tier_config in &self.config.performance_config.tier_configs {
            self.tier_mapping
                .insert(tier_config.tier, tier_config.base_url.clone());
        }

        // Add default mappings if not configured
        self.tier_mapping
            .entry(FileStorageTier::Memory)
            .or_insert_with(|| "memory://".to_string());
    }

    /// Get filesystem URL for a specific storage tier
    pub fn get_tier_url(&self, tier: FileStorageTier, relative_path: &str) -> FsResult<String> {
        let base_url = self.tier_mapping.get(&tier).ok_or_else(|| {
            FilesystemError::Config(format!("No filesystem configured for tier {:?}", tier))
        })?;

        // Construct full URL
        if base_url.ends_with('/') {
            Ok(format!("{}{}", base_url, relative_path))
        } else {
            Ok(format!("{}/{}", base_url, relative_path))
        }
    }

    /// Promote data from one tier to another
    pub async fn promote_data(
        &self,
        from_tier: FileStorageTier,
        to_tier: FileStorageTier,
        relative_path: &str,
    ) -> FsResult<()> {
        if !to_tier.is_faster_than(&from_tier) {
            return Err(FilesystemError::InvalidOperation(format!(
                "Cannot promote from {:?} to {:?} (target not faster)",
                from_tier, to_tier
            )));
        }

        let from_url = self.get_tier_url(from_tier, relative_path)?;
        let to_url = self.get_tier_url(to_tier, relative_path)?;

        info!(
            "Promoting data from {:?} to {:?}: {}",
            from_tier, to_tier, relative_path
        );
        self.move_atomic(&from_url, &to_url).await
    }

    /// Demote data from one tier to another
    pub async fn demote_data(
        &self,
        from_tier: FileStorageTier,
        to_tier: FileStorageTier,
        relative_path: &str,
    ) -> FsResult<()> {
        if from_tier.is_faster_than(&to_tier) {
            let from_url = self.get_tier_url(from_tier, relative_path)?;
            let to_url = self.get_tier_url(to_tier, relative_path)?;

            info!(
                "Demoting data from {:?} to {:?}: {}",
                from_tier, to_tier, relative_path
            );
            self.move_atomic(&from_url, &to_url).await
        } else {
            Err(FilesystemError::InvalidOperation(format!(
                "Cannot demote from {:?} to {:?} (target not slower)",
                from_tier, to_tier
            )))
        }
    }

    /// Get optimal tier for data based on access patterns
    pub fn suggest_tier(&self, access_frequency: f64, data_size_bytes: u64) -> FileStorageTier {
        // Simple heuristic: hot data → fast tiers, cold data → slow tiers
        if access_frequency > 100.0 {
            // Very hot: >100 accesses per hour
            if data_size_bytes < 100 * 1024 * 1024 {
                FileStorageTier::Memory
            } else {
                FileStorageTier::NVMe
            }
        } else if access_frequency > 10.0 {
            // Warm: 10-100 accesses per hour
            FileStorageTier::SSD
        } else if access_frequency > 1.0 {
            // Cool: 1-10 accesses per hour
            FileStorageTier::HDD
        } else {
            // Cold: <1 access per hour
            FileStorageTier::S3Standard
        }
    }

    /// List all available filesystem types
    pub fn available_filesystems(&self) -> Vec<&str> {
        self.filesystems.keys().map(|s| s.as_str()).collect()
    }

    /// Create a zero-copy filesystem wrapper for intelligent caching and optimization
    // create_zero_copy_filesystem removed - functionality integrated into get_unified_caching_filesystem
    /// Unified filesystem operations - automatically route to correct backend
    pub async fn read(&self, url: &str) -> FsResult<Vec<u8>> {
        tracing::debug!("🔍 FilesystemFactory::read() - URL: {}", url);
        let fs = self.get_filesystem(url)?;
        let path = Self::resolve_path(url)?;
        tracing::debug!(
            "📖 Routing to {} filesystem for path: {}",
            fs.filesystem_type(),
            path
        );
        let result = fs.read(&path).await;

        match &result {
            Ok(data) => tracing::debug!("✅ Read {} bytes successfully from {}", data.len(), url),
            Err(e) => tracing::error!("❌ Read failed from {}: {}", url, e),
        }

        result
    }

    /// Read a single byte range `[offset, offset+length)` from `url`. Convenience
    /// over `get_filesystem(url).read_range(...)` mirroring [`Self::read`] — used by
    /// the PAX-native ranged reader to fetch a segment's tail index and individual
    /// surviving blocks without pulling the whole object (TD-DOC-PUSHDOWN-1).
    pub async fn read_range(&self, url: &str, offset: u64, length: u64) -> FsResult<Vec<u8>> {
        let fs = self.get_filesystem(url)?;
        let path = Self::resolve_path(url)?;
        fs.read_range(&path, offset, length).await
    }

    /// Read multiple byte ranges from `url` in one batched call. Mirrors
    /// [`Self::read`]; the ranged PAX reader uses this to fetch all surviving
    /// blocks of a pruned scan together.
    ///
    /// Whether those logical ranges become fewer physical requests depends on
    /// `PROXIMADB_FS_READ_RANGES_COALESCE_GAP`. **Unset — the default — means
    /// one physical GET per range**, and object stores bill per request.
    ///
    /// When armed, this method plans the merge itself rather than delegating,
    /// because this is the layer that knows the backend: `IopsBudget::for_path`
    /// needs the URL, and the filesystem-types crate is a leaf that must not
    /// depend upward to reach it. Callers still receive exactly one buffer per
    /// input range, in input order, sliced exactly.
    ///
    /// An earlier version of this comment asserted the backend coalesced. It
    /// did not, and no backend overrode `read_ranges`, so callers trusting it
    /// silently paid one billed request per range.
    pub async fn read_ranges(
        &self,
        url: &str,
        ranges: Vec<std::ops::Range<u64>>,
    ) -> FsResult<Vec<Vec<u8>>> {
        let fs = self.get_filesystem(url)?;
        let path = Self::resolve_path(url)?;
        let Some(policy) = Self::range_coalesce_policy_for(url) else {
            return fs.read_ranges(&path, ranges).await;
        };

        // Coalesce HERE rather than inside the filesystem, because this is the
        // layer that knows the backend: `IopsBudget::for_path` needs the URL,
        // and `proximadb-storage-filesystem-types` is a leaf crate that must
        // not depend up into `storage-common` to reach it.
        //
        // Scope is exactly right by construction, not by luck:
        //   * the SST vector path issues singular `read_range` calls after
        //     planning its own coalescing, so it cannot double-merge here;
        //   * `SmartIo` coalesces internally and calls `FileSystem::read_ranges`
        //     directly, bypassing this method, so it cannot double-merge either;
        //   * the DataFusion PAX adapter is the one production caller, and it
        //     currently pays one billed GET per surviving block.
        let plan =
            proximadb_storage_filesystem_types::read_ranges_plan::coalesce_ranges_with_mapping(
                &ranges,
                Some(policy),
            )?;
        let mut buffers = Vec::with_capacity(plan.physical.len());
        for physical in &plan.physical {
            buffers.push(
                fs.read_range(&path, physical.start, physical.end - physical.start)
                    .await?,
            );
        }
        Ok(plan
            .mapping
            .iter()
            .map(|slice| match slice.physical {
                Some(idx) => {
                    proximadb_storage_filesystem_types::read_ranges_plan::slice_from_physical(
                        &buffers[idx],
                        *slice,
                    )
                }
                None => Vec::new(),
            })
            .collect())
    }

    /// Range-merging policy for `url`, or `None` to issue one request per range.
    ///
    /// Gate semantics: **unset is OFF**. An explicit `0` gap is a legitimate
    /// setting — "merge only exactly-adjacent ranges" — so absence, not zero,
    /// has to mean disabled. The byte ceiling defaults to the backend's own
    /// `IopsBudget` maximum and is clamped to it, so this knob can tighten the
    /// over-read bound but never loosen it past what the backend profile
    /// already permits. Raising that profile is TD-SEARCH-3's call, not this
    /// gate's.
    fn range_coalesce_policy_for(
        url: &str,
    ) -> Option<proximadb_storage_filesystem_types::RangeCoalescePolicy> {
        let budget = proximadb_storage_common::iops_budget::IopsBudget::for_path(url);
        resolve_range_coalesce_policy(
            std::env::var("PROXIMADB_FS_READ_RANGES_COALESCE_GAP")
                .ok()
                .as_deref(),
            std::env::var("PROXIMADB_FS_READ_RANGES_COALESCE_MAX")
                .ok()
                .as_deref(),
            budget.max,
        )
    }

    pub async fn write(
        &self,
        url: &str,
        data: &[u8],
        options: Option<FileOptions>,
    ) -> FsResult<()> {
        tracing::debug!(
            "📝 FilesystemFactory::write() - URL: {} ({} bytes)",
            url,
            data.len()
        );
        let fs = self.get_filesystem(url)?;
        let path = Self::resolve_path(url)?;
        tracing::debug!(
            "💾 Routing to {} filesystem for path: {}",
            fs.filesystem_type(),
            path
        );
        let result = fs.write(&path, data, options).await;

        match &result {
            Ok(_) => tracing::debug!("✅ Wrote {} bytes successfully to {}", data.len(), url),
            Err(e) => tracing::error!("❌ Write failed to {}: {}", url, e),
        }

        result
    }

    pub async fn append(&self, url: &str, data: &[u8]) -> FsResult<()> {
        tracing::debug!(
            "➕ FilesystemFactory::append() - URL: {} ({} bytes)",
            url,
            data.len()
        );
        let fs = self.get_filesystem(url)?;
        let path = Self::resolve_path(url)?;
        tracing::debug!(
            "📎 Routing to {} filesystem for path: {}",
            fs.filesystem_type(),
            path
        );
        let result = fs.append(&path, data).await;

        match &result {
            Ok(_) => tracing::debug!("✅ Appended {} bytes successfully to {}", data.len(), url),
            Err(e) => tracing::error!("❌ Append failed to {}: {}", url, e),
        }

        result
    }

    pub async fn delete(&self, url: &str) -> FsResult<()> {
        tracing::debug!("🗑️ FilesystemFactory::delete() - URL: {}", url);
        let fs = self.get_filesystem(url)?;
        let path = Self::resolve_path(url)?;
        tracing::debug!(
            "🚮 Routing to {} filesystem for path: {}",
            fs.filesystem_type(),
            path
        );
        let result = fs.delete(&path).await;

        match &result {
            Ok(_) => tracing::debug!("✅ Deleted successfully: {}", url),
            Err(e) => tracing::error!("❌ Delete failed for {}: {}", url, e),
        }

        result
    }

    pub async fn exists(&self, url: &str) -> FsResult<bool> {
        tracing::trace!("🔍 FilesystemFactory::exists() - URL: {}", url);
        let fs = self.get_filesystem(url)?;
        let path = Self::resolve_path(url)?;
        let result = fs.exists(&path).await;

        match &result {
            Ok(exists) => tracing::trace!("✅ Exists check for {}: {}", url, exists),
            Err(e) => tracing::error!("❌ Exists check failed for {}: {}", url, e),
        }

        result
    }

    pub async fn metadata(&self, url: &str) -> FsResult<FsFileMetadata> {
        let fs = self.get_filesystem(url)?;
        let path = Self::resolve_path(url)?;
        fs.metadata(&path).await
    }

    pub async fn list(&self, url: &str) -> FsResult<Vec<DirEntry>> {
        let fs = self.get_filesystem(url)?;
        let path = Self::resolve_path(url)?;
        fs.list(&path).await
    }

    pub async fn create_dir(&self, url: &str) -> FsResult<()> {
        let fs = self.get_filesystem(url)?;
        let path = Self::resolve_path(url)?;
        fs.create_dir(&path).await
    }

    pub async fn create_dir_all(&self, url: &str) -> FsResult<()> {
        let fs = self.get_filesystem(url)?;
        let path = Self::resolve_path(url)?;
        fs.create_dir_all(&path).await
    }

    pub async fn copy(&self, from_url: &str, to_url: &str) -> FsResult<()> {
        // Handle cross-filesystem copies
        let from_scheme = self.extract_scheme(from_url)?;
        let to_scheme = self.extract_scheme(to_url)?;

        if from_scheme == to_scheme {
            // Same filesystem - use native copy
            let fs = self.get_filesystem(from_url)?;
            let from_path = Self::resolve_path(from_url)?;
            let to_path = Self::resolve_path(to_url)?;
            fs.copy(&from_path, &to_path).await
        } else {
            // Cross-filesystem copy - read from source, write to destination
            let data = self.read(from_url).await?;
            self.write(to_url, &data, None).await
        }
    }

    pub async fn move_file(&self, from_url: &str, to_url: &str) -> FsResult<()> {
        // Handle cross-filesystem moves
        let from_scheme = self.extract_scheme(from_url)?;
        let to_scheme = self.extract_scheme(to_url)?;

        if from_scheme == to_scheme {
            // Same filesystem - use native move
            let fs = self.get_filesystem(from_url)?;
            let from_path = Self::resolve_path(from_url)?;
            let to_path = Self::resolve_path(to_url)?;
            fs.move_file(&from_path, &to_path).await
        } else {
            // Cross-filesystem move - copy then delete
            self.copy(from_url, to_url).await?;
            self.delete(from_url).await
        }
    }

    pub async fn sync(&self, url: &str) -> FsResult<()> {
        let fs = self.get_filesystem(url)?;
        fs.sync().await
    }

    /// Create an Arc-wrapped filesystem factory (safe helper)
    ///
    /// This is a convenience helper for creating Arc<FilesystemFactory> with proper error handling.
    /// Use this to avoid panic-prone call sites when creating filesystem factories.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use anyhow::Context;
    ///
    /// // Before (panics on error):
    /// let factory = Arc::new(FilesystemFactory::create(config).await?);
    ///
    /// // After (proper error handling):
    /// let factory = FilesystemFactory::create_arc(config).await
    ///     .context("Failed to create filesystem factory")?;
    /// ```
    pub async fn create_arc(config: FilesystemConfig) -> FsResult<Arc<Self>> {
        let factory = Self::create(config).await?;
        Ok(Arc::new(factory))
    }

    /// Create an Arc-wrapped filesystem factory with default config (safe helper)
    ///
    /// This is a convenience helper for creating Arc<FilesystemFactory> with proper error handling.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use anyhow::Context;
    ///
    /// // Before (panics on error):
    /// let factory = Arc::new(FilesystemFactory::create_default().await?);
    ///
    /// // After (proper error handling):
    /// let factory = FilesystemFactory::create_default_arc().await
    ///     .context("Failed to create filesystem factory")?;
    /// ```
    pub async fn create_default_arc() -> FsResult<Arc<Self>> {
        let factory = Self::create_default().await?;
        Ok(Arc::new(factory))
    }
}

#[cfg(test)]
mod inline_tests {
    use super::*;
    use std::ffi::OsString;
    use tempfile::TempDir;

    struct EnvVarGuard {
        key: String,
        previous: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &str, value: &str) -> Self {
            let previous = std::env::var_os(key);
            unsafe {
                std::env::set_var(key, value);
            }
            Self {
                key: key.to_string(),
                previous,
            }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            unsafe {
                if let Some(previous) = &self.previous {
                    std::env::set_var(&self.key, previous);
                } else {
                    std::env::remove_var(&self.key);
                }
            }
        }
    }

    /// Physical read accounting is a production invariant, not a benchmark
    /// capability.  The diagnostic counting wrapper may be disabled, but the
    /// route-cost ledger must still see both whole-object and ranged GETs.
    #[tokio::test]
    async fn local_leaf_records_all_physical_reads_without_counting_wrapper() {
        use crate::observability::io_trace;
        use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

        let temp = TempDir::new().unwrap();
        let file_path = temp.path().join("physical-reads.bin");
        let payload = vec![7u8; 4096];
        std::fs::write(&file_path, &payload).unwrap();
        let url = format!("file://{}", file_path.display());
        let fs = LocalFileSystem::new(LocalConfig::default()).await.unwrap();

        let snap = io_trace::scope(async {
            assert_eq!(fs.read(&url).await.unwrap().len(), 4096);
            assert_eq!(fs.read_range(&url, 1024, 512).await.unwrap().len(), 512);
            io_trace::snapshot().expect("io_trace scope active")
        })
        .await;

        assert_eq!(snap.get_ops, 2, "both physical reads are GET operations");
        assert_eq!(snap.range_gets, 1, "only the ranged read is a range GET");
        assert_eq!(snap.bytes_read, 4096 + 512);
    }

    /// Regression for io-trace double-counting under the diagnostic wrapper.
    /// `CountingFileSystem` (active under `PROXIMADB_COUNT_FS_IO`) wraps the leaf
    /// `LocalFileSystem`; both used to call `record_range_gets(1)` +
    /// `record_bytes_read` on every ranged GET, so io-trace reported 2× the real
    /// GET count and 2× the real bytes whenever the counting wrapper was on (`avg_get_bytes`
    /// self-corrected by cancelling both halves). The wrapper must now record only
    /// only owns process-global counters; all per-query evidence comes once from
    /// the physical backend.
    #[tokio::test]
    async fn counting_wrapper_records_ranged_gets_once_not_twice() {
        use crate::observability::io_trace;
        use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
        use proximadb_storage_filesystem_types::counting::{CountingFileSystem, global_counters};

        let temp = TempDir::new().unwrap();
        let file_path = temp.path().join("seg.bin");
        let payload = vec![7u8; 4096];
        std::fs::write(&file_path, &payload).unwrap();
        let url = format!("file://{}", file_path.display());

        let inner = LocalFileSystem::new(LocalConfig::default()).await.unwrap();
        let fs: Arc<dyn FileSystem> =
            Arc::new(CountingFileSystem::new(Arc::new(inner), global_counters()));

        let snap = io_trace::scope(async {
            let a = fs.read_range(&url, 0, 1024).await.unwrap();
            let b = fs.read_range(&url, 1024, 2048).await.unwrap();
            assert_eq!(a.len(), 1024);
            assert_eq!(b.len(), 2048);
            io_trace::snapshot().expect("io_trace scope active")
        })
        .await;

        assert_eq!(
            snap.range_gets, 2,
            "each ranged GET must count once (leaf-backend source), not twice"
        );
        assert_eq!(
            snap.bytes_read,
            1024 + 2048,
            "bytes must count once, not twice"
        );
        assert_eq!(
            snap.get_ops, 2,
            "each physical leaf read records its GET exactly once"
        );
    }

    #[tokio::test]
    async fn test_filesystem_factory_creation() {
        let config = FilesystemConfig::default();
        let factory = FilesystemFactory::create(config)
            .await
            .expect("Failed to create filesystem factory with default config");

        // Should have local filesystem by default
        assert!(factory.available_filesystems().contains(&"file"));
    }

    #[tokio::test]
    async fn test_url_scheme_extraction() {
        let config = FilesystemConfig::default();
        let factory = FilesystemFactory::create(config)
            .await
            .expect("Failed to create filesystem factory for URL scheme extraction test");

        assert_eq!(
            factory
                .extract_scheme("file:///tmp/test.txt")
                .expect("Failed to extract scheme from file:// URL"),
            "file"
        );
        assert_eq!(
            factory
                .extract_scheme("s3://bucket/key")
                .expect("Failed to extract scheme from s3:// URL"),
            "s3"
        );
        assert_eq!(
            factory
                .extract_scheme("adls://account/container/path")
                .expect("Failed to extract scheme from adls:// URL"),
            "adls"
        );
        assert_eq!(
            factory
                .extract_scheme("abfs://container@account/path")
                .expect("Failed to extract scheme from abfs:// URL"),
            "abfs"
        );
        assert_eq!(
            factory
                .extract_scheme("gcs://bucket/object")
                .expect("Failed to extract scheme from gcs:// URL"),
            "gcs"
        );
        // Test gs:// scheme mapping to gcs
        assert_eq!(
            factory
                .extract_scheme("gs://bucket/object")
                .expect("Failed to extract scheme from gs:// URL"),
            "gcs"
        );
        assert_eq!(
            factory
                .extract_scheme("hdfs://namenode:9000/path")
                .expect("Failed to extract scheme from hdfs:// URL"),
            "hdfs"
        );
    }

    #[tokio::test]
    async fn test_path_extraction() {
        let _config = FilesystemConfig::default();
        let _factory = FilesystemFactory::create(_config)
            .await
            .expect("Failed to create filesystem factory for path extraction test");

        assert_eq!(
            FilesystemFactory::resolve_path("file:///tmp/test.txt")
                .expect("Failed to resolve path from file:// URL"),
            "/tmp/test.txt"
        );
        assert_eq!(
            FilesystemFactory::resolve_path("/local/path").expect("Failed to resolve local path"),
            "/local/path"
        );
    }

    /// TD-FLUSH-6 regression: cloud-scheme URLs must be passed through VERBATIM
    /// to the backend. The cloud backends (`AzureBlobFileSystem::parse`,
    /// `AwsS3FileSystem`, ...) recover the container/bucket + blob key from the
    /// full url themselves; stripping to `Url::path()` here dropped the host
    /// (container) and made `list`/`copy`/`delete` fail with
    /// `Invalid path: not an azure path: /1/data/__flush` on a clean object
    /// store, blocking flush. `file://` (above) and scheme-less paths are still
    /// reduced to a local path.
    #[tokio::test]
    async fn test_resolve_path_cloud_urls_pass_through_verbatim() {
        // exact shape that broke the SST flush staging-dir list on Azurite
        assert_eq!(
            FilesystemFactory::resolve_path("az://proximadb-bench/1/data/__flush")
                .expect("az:// staging url must resolve"),
            "az://proximadb-bench/1/data/__flush"
        );
        // s3 / gs likewise — backend parses container+blob from the full url
        assert_eq!(
            FilesystemFactory::resolve_path("s3://bucket/key").expect("s3:// url must resolve"),
            "s3://bucket/key"
        );
        assert_eq!(
            FilesystemFactory::resolve_path("gs://bucket/a/b/c.pax")
                .expect("gs:// url must resolve"),
            "gs://bucket/a/b/c.pax"
        );
        // malformed url is still rejected (Url::parse validity check retained)
        assert!(
            FilesystemFactory::resolve_path("az://[bad").is_err(),
            "malformed cloud url must be rejected"
        );
    }

    #[tokio::test]
    async fn test_filesystem_factory_wraps_local_fs_with_encryption() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let _env = EnvVarGuard::set(
            "TEST_PROXIMADB_FS_FACTORY_KEY",
            "factory-test-master-key-32-bytes!!",
        );

        let config = FilesystemConfig {
            local: Some(local::LocalConfig {
                root_dir: Some(temp_dir.path().to_path_buf()),
                ..Default::default()
            }),
            global_options: FileOptions {
                encryption: Some("TEST_PROXIMADB_FS_FACTORY_KEY".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };

        let factory = FilesystemFactory::create(config)
            .await
            .expect("failed to create encrypted filesystem factory");
        let fs = factory
            .get_filesystem("file:///tmp/proximadb-encrypted")
            .expect("failed to get encrypted filesystem");

        assert_eq!(fs.filesystem_type(), "encrypted");

        let logical_path = temp_dir.path().join("wrapped.bin");
        let logical_path = logical_path.to_string_lossy().to_string();
        let encrypted_path = format!("{}.enc", logical_path);

        fs.write(&logical_path, b"factory-secret", None)
            .await
            .expect("failed to write encrypted file");

        assert!(std::path::Path::new(&encrypted_path).exists());
        assert_eq!(
            fs.read(&logical_path)
                .await
                .expect("failed to read encrypted file"),
            b"factory-secret"
        );
    }

    #[tokio::test]
    async fn test_copy_atomic_falls_back_for_encrypted_filesystems() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let _env = EnvVarGuard::set(
            "TEST_PROXIMADB_FS_COPY_KEY",
            "copy-test-master-key-32-bytes!!!!!",
        );

        let config = FilesystemConfig {
            local: Some(local::LocalConfig {
                root_dir: Some(temp_dir.path().to_path_buf()),
                ..Default::default()
            }),
            global_options: FileOptions {
                encryption: Some("TEST_PROXIMADB_FS_COPY_KEY".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };

        let factory = FilesystemFactory::create(config)
            .await
            .expect("failed to create encrypted filesystem factory");
        let fs = factory
            .get_filesystem("file:///tmp/proximadb-encrypted")
            .expect("failed to get encrypted filesystem");

        let src_url = format!("file://{}", temp_dir.path().join("src.bin").display());
        let dst_url = format!("file://{}", temp_dir.path().join("dst.bin").display());
        let src_path = FilesystemFactory::resolve_path(&src_url).expect("failed to resolve src");
        let dst_path = FilesystemFactory::resolve_path(&dst_url).expect("failed to resolve dst");

        fs.write(&src_path, b"copy-secret", None)
            .await
            .expect("failed to write source file");

        factory
            .copy_atomic(&src_url, &dst_url)
            .await
            .expect("copy_atomic should succeed for encrypted filesystems");

        assert_eq!(
            fs.read(&dst_path)
                .await
                .expect("failed to read copied file"),
            b"copy-secret"
        );
        assert!(std::path::Path::new(&format!("{}.enc", dst_path)).exists());
    }
}

#[cfg(test)]
mod comprehensive_tests;

// ============================================================================
// FilesystemPort impl — Slice D port-inversion (gap 5/6 enabler)
// ============================================================================
// Exposes the root-local `FilesystemFactory` behind the `FilesystemPort` trait
// (defined in `proximadb-storage-ports`) so engine leaves can depend on
// `Arc<dyn FilesystemPort>` instead of this concrete type, enabling their
// extraction to crates. Pure delegation to the inherent methods above; the
// composition root injects `FilesystemFactory` (as `Arc<dyn FilesystemPort>`)
// into engines. See `EngineFilesystemAccess` in `src/storage/traits/mod.rs` and
// `ROOT_CRATE_DECOMPOSITION_ENGINES_EXTRACTION_2026_07_12.adoc`.
#[async_trait::async_trait]
impl proximadb_storage_ports::FilesystemPort for FilesystemFactory {
    fn get_filesystem(&self, url: &str) -> FsResult<std::sync::Arc<dyn FileSystem>> {
        FilesystemFactory::get_filesystem(self, url)
    }

    async fn create_dir_all(&self, url: &str) -> FsResult<()> {
        FilesystemFactory::create_dir_all(self, url).await
    }

    async fn write(&self, url: &str, data: &[u8], options: Option<FileOptions>) -> FsResult<()> {
        FilesystemFactory::write(self, url, data, options).await
    }

    async fn move_atomic(&self, from_url: &str, to_url: &str) -> FsResult<()> {
        FilesystemFactory::move_atomic(self, from_url, to_url).await
    }

    async fn delete(&self, url: &str) -> FsResult<()> {
        FilesystemFactory::delete(self, url).await
    }

    async fn read(&self, url: &str) -> FsResult<Vec<u8>> {
        FilesystemFactory::read(self, url).await
    }

    async fn list(&self, url: &str) -> FsResult<Vec<DirEntry>> {
        FilesystemFactory::list(self, url).await
    }
}

/// Pure resolution of the range-coalescing gate, split out so its semantics are
/// testable without mutating process environment (`set_var`/`remove_var` are
/// unsafe in edition 2024 precisely because they race across threads).
///
/// `None` means "issue one physical request per logical range" — today's
/// behaviour, and what an unset gate must produce.
fn resolve_range_coalesce_policy(
    gap_raw: Option<&str>,
    max_raw: Option<&str>,
    ceiling: u64,
) -> Option<proximadb_storage_filesystem_types::RangeCoalescePolicy> {
    // Absence is OFF. A malformed value is also OFF rather than defaulting to
    // some merging: silently coalescing because someone typo'd a gap would
    // change billed request counts with no signal anywhere.
    let max_gap_bytes = gap_raw?.trim().parse::<u64>().ok()?;
    let max_merged_bytes = max_raw
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(ceiling)
        .min(ceiling)
        .max(1);
    Some(proximadb_storage_filesystem_types::RangeCoalescePolicy {
        max_gap_bytes,
        max_merged_bytes,
    })
}

#[cfg(test)]
mod range_coalesce_gate_tests {
    use super::resolve_range_coalesce_policy;

    const AZURE_MAX: u64 = 4 * 1024 * 1024;

    /// Unset must be OFF, so an un-gated deployment issues exactly the requests
    /// it issues today.
    #[test]
    fn unset_gap_disables_coalescing() {
        assert!(resolve_range_coalesce_policy(None, None, AZURE_MAX).is_none());
        // ...and a max without a gap is still off — the gap is the arming knob.
        assert!(resolve_range_coalesce_policy(None, Some("1048576"), AZURE_MAX).is_none());
    }

    /// Zero is a REAL setting ("merge only exactly-adjacent ranges"), which is
    /// why OFF has to be encoded as absence rather than as zero.
    #[test]
    fn explicit_zero_gap_is_armed_not_disabled() {
        let policy = resolve_range_coalesce_policy(Some("0"), None, AZURE_MAX)
            .expect("explicit 0 must arm the policy");
        assert_eq!(policy.max_gap_bytes, 0);
        assert_eq!(
            policy.max_merged_bytes, AZURE_MAX,
            "defaults to the backend ceiling"
        );
    }

    /// A typo must not silently change billed request counts.
    #[test]
    fn malformed_gap_is_off_rather_than_guessed() {
        for bad in ["", "1MiB", "-1", "0x10", "1.5"] {
            assert!(
                resolve_range_coalesce_policy(Some(bad), None, AZURE_MAX).is_none(),
                "malformed gap {bad:?} must disable, not guess"
            );
        }
    }

    /// The ceiling can be tightened but never loosened. Raising a backend's
    /// range profile is TD-SEARCH-3's decision, gated on its own measurement —
    /// this knob must not be a back door to it.
    #[test]
    fn max_is_clamped_to_the_backend_ceiling() {
        let tighter = resolve_range_coalesce_policy(Some("65536"), Some("1048576"), AZURE_MAX)
            .expect("armed");
        assert_eq!(
            tighter.max_merged_bytes,
            1024 * 1024,
            "tightening is honoured"
        );

        let looser = resolve_range_coalesce_policy(Some("65536"), Some("25165824"), AZURE_MAX)
            .expect("armed");
        assert_eq!(
            looser.max_merged_bytes, AZURE_MAX,
            "24 MiB must clamp to the backend ceiling, not widen it"
        );

        let zero =
            resolve_range_coalesce_policy(Some("65536"), Some("0"), AZURE_MAX).expect("armed");
        assert_eq!(
            zero.max_merged_bytes, 1,
            "a zero ceiling would merge nothing at all"
        );
    }
}
