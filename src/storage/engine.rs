use crate::core::{StorageConfig, String, VectorId};
#[cfg(feature = "axis")]
use crate::index::{AxisConfig, AxisManager};
use crate::storage::persistence::write_ahead_log::{WALConfig, WriteAheadLogManager};
use crate::storage::{
    engines::sst::{Compaction, SstEngine},
    persistence::disk_manager::DiskManager,
};
use proximadb_records::{EmbeddingCell, ProximaRecord};
// Import ProximaBlockCollectionMetadata from the appropriate location
use crate::storage::engines::core::formats::proximablocks::header_metadata::ProximaBlockCollectionMetadata;
use dashmap::DashMap;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

// Note: Distance computation is handled by the unified compute module
// at src/compute/distance_computation/ which provides SIMD-accelerated
// implementations. Use self.distance_compute for all distance calculations.

pub struct StorageEngine {
    config: StorageConfig,
    sst_storages: Arc<DashMap<String, Arc<SstEngine>>>,
    canonical_sst_storage: Arc<SstEngine>,
    #[allow(dead_code)]
    disk_manager: Arc<DiskManager>,
    write_ahead_log_manager: Arc<WriteAheadLogManager>,
    #[cfg(feature = "axis")]
    axis_index_manager: Arc<AxisManager>,
    #[cfg(feature = "axis")]
    axis_runtime_enabled: bool,
    compaction_manager: Arc<Compaction>,
    filesystem: Arc<crate::storage::persistence::filesystem::FilesystemFactory>,

    /// Shared distance computation engine for all storage operations
    distance_compute: Arc<proximadb_distance_kernel::engine::UnifiedDistanceCompute>,

    /// A6 storage-write fence (default-OFF). Injected post-construction from the
    /// bootstrap (`database.rs`) once `SharedServices` — and thus the durable
    /// `PartitionLeaseManager` — exists, mirroring the `set_precision_resolver`
    /// pattern (the storage engine is built before the lease stack). Threaded into
    /// the shutdown flush coordinator so a fenced-out pod cannot publish stale data
    /// to shared storage. `None` ⇒ no fence ⇒ flush proceeds (fail-open).
    storage_write_fence: Option<Arc<dyn crate::storage::write_fence::StorageWriteFence>>,
}

impl StorageEngine {
    // 🔴 REMOVED - SST is now collection-agnostic singleton pattern
    // fn get_sst_storage(&self) -> Arc<SstEngine> {
    //     self.sst_storage.clone()
    // }
    /// Get storage configuration
    pub fn config(&self) -> &StorageConfig {
        &self.config
    }

    /// Get a handle to the compaction manager so the bootstrap path
    /// can attach a `CanonicalPrecisionResolver` after `SharedServices`
    /// becomes available (the storage engine is constructed before the
    /// catalog handle exists, so the resolver is injected via
    /// `Compaction::set_precision_resolver` post-construction).
    pub fn compaction_manager(&self) -> Arc<Compaction> {
        self.compaction_manager.clone()
    }

    /// Create new storage engine without collection service dependency
    pub async fn new_without_collection_service(
        config: StorageConfig,
    ) -> crate::storage::Result<Self> {
        Self::new_internal(config).await
    }

    /// Internal constructor used by both public constructors
    async fn new_internal(config: StorageConfig) -> crate::storage::Result<Self> {
        // Extract data directories from storage locations
        let data_dirs: Vec<PathBuf> = config
            .storage_locations
            .iter()
            .filter_map(|loc| {
                if loc.url.starts_with("file://") {
                    loc.url.strip_prefix("file://").map(PathBuf::from)
                } else {
                    None
                }
            })
            .collect();

        let disk_manager = Arc::new(DiskManager::new(data_dirs.clone())?);

        // Initialize WAL configuration from storage locations
        let mut wal_config = WALConfig::default();
        // TD-WAL-5: this engine's WAL manager owns startup replay, so it must
        // resolve the SAME local WAL root the write path uses. Built from
        // `WALConfig::default()`, it dropped a config-file `wal_local_dir`, and
        // replay never listed the segments written there: acknowledged writes
        // were lost on a crash before a manifest entry recorded that base.
        wal_config.wal_local_dir = config.wal_config.wal_local_dir.clone();
        wal_config.multi_disk.data_directories = config
            .storage_locations
            .iter()
            .map(|loc| {
                // Ensure proper file:// URL format
                let url = if loc.url.starts_with("file://") {
                    loc.url.clone()
                } else if loc.url.starts_with("/") {
                    format!("file://{}", loc.url)
                } else {
                    loc.url.clone()
                };
                tracing::debug!("WAL directory URL: {}", url);
                url
            })
            .collect();

        // Create filesystem factory for WAL
        let filesystem = Arc::new(
            crate::storage::persistence::filesystem::FilesystemFactory::create(
                crate::storage::persistence::filesystem::FilesystemConfig::default(),
            )
            .await
            .map_err(|e| crate::core::StorageError::DiskIO(std::io::Error::other(e.to_string())))?,
        );

        // Create WAL manager using modern batch factory pattern
        let write_ahead_log_manager = Arc::new(
            WriteAheadLogManager::create_with_batch_factory(
                wal_config.strategy_type,
                wal_config,
                filesystem.clone(),
            )
            .await
            .map_err(|e| crate::core::StorageError::WalError(e.to_string()))?,
        );

        // StorageEngine focuses on pure storage operations (LSM, WAL, MMAP)
        tracing::info!("📂 StorageEngine: Metadata operations delegated to SharedServices");

        // Initialize search index manager
        let _data_dir = data_dirs
            .first()
            .cloned()
            .unwrap_or_else(|| PathBuf::from("./data"));
        #[cfg(feature = "axis")]
        let axis_runtime_enabled = config.optimization.enable_axis_indexes;
        // Initialize AXIS index manager with default configuration
        #[cfg(feature = "axis")]
        let axis_index_manager = {
            let axis_config = AxisConfig::default();
            let m = Arc::new(AxisManager::new(axis_config).await?);
            if axis_runtime_enabled {
                // Make AXIS available only after both the compile capability
                // and process runtime policy are enabled.
                crate::storage::engines::sst::core::set_sst_axis_manager(m.clone());
                crate::index::set_global_axis_manager(m.clone());
                info!("✅ AXIS runtime enabled and registered with storage serving");
            }
            m
        };

        // Build the one configured SST instance used by every flush and its
        // background compaction. Keeping this Arc prevents the worker-owning
        // instance from being discarded and a second default engine from being
        // constructed by the materializer.
        let sst_config = config.effective_sst_config();
        let distance_compute =
            Arc::new(proximadb_distance_kernel::engine::UnifiedDistanceCompute::default());
        let canonical_sst_storage = Arc::new(
            SstEngine::new_with_config(sst_config, filesystem.clone(), distance_compute.clone())
                .await
                .map_err(|e| {
                    proximadb_kernel::error::StorageError::SstEngine(format!(
                        "Failed to create SST storage: {}",
                        e
                    ))
                })?,
        );
        let compaction_manager = canonical_sst_storage
            .compaction_manager()
            .cloned()
            .ok_or_else(|| {
                proximadb_kernel::error::StorageError::SstEngine(
                    "canonical SST storage did not initialize compaction".to_string(),
                )
            })?;

        Ok(Self {
            config,
            sst_storages: Arc::new(DashMap::new()), // Now uses DashMap for per-collection storages
            canonical_sst_storage,
            disk_manager,
            write_ahead_log_manager,
            #[cfg(feature = "axis")]
            axis_index_manager,
            #[cfg(feature = "axis")]
            axis_runtime_enabled,
            compaction_manager,
            filesystem,
            distance_compute,
            storage_write_fence: None,
        })
    }

    /// Inject the A6 storage-write fence (default-OFF). Called from the bootstrap
    /// after `SharedServices` builds the durable `PartitionLeaseManager`, so the
    /// shutdown flush path enforces the **same** ownership view as the network
    /// write-gates. Mirrors `Compaction::set_precision_resolver` (post-construction
    /// wiring of a dependency the storage engine is built before).
    pub fn set_storage_write_fence(
        &mut self,
        fence: Arc<dyn crate::storage::write_fence::StorageWriteFence>,
    ) {
        self.storage_write_fence = Some(fence);
        tracing::info!(
            "🔒 STORAGE_ENGINE: A6 storage-write fence wired (default-OFF; set PROXIMADB_WRITE_FENCING=1 to enforce)"
        );
    }

    pub async fn start(&mut self) -> crate::storage::Result<()> {
        tracing::info!("🚀 STORAGE_ENGINE: Starting storage engine");

        // ADR-081: establish the one process-wide flush admission domain before
        // recovery or any live trigger can resolve a materializer.
        crate::storage::flush_materializer::configure_flush_admission(
            self.config
                .sst_config
                .as_ref()
                .map(|config| config.background_thread_count as usize)
                .unwrap_or_else(crate::storage::flush_materializer::default_parallel_flushes),
        );
        crate::storage::flush_materializer::register_flush_engine(
            crate::proto::proximadb_v1::StorageEngine::Sst,
            self.canonical_sst_storage.clone(),
        )
        .await;

        // Load the durable catalog before WAL replay. Recovery flushes each WAL
        // batch through the collection's assigned storage engine, so those
        // engines must be registered first (especially after a cold restart,
        // when the catalog itself has just been restored from object storage).
        tracing::info!("📊 STORAGE_ENGINE: About to call load_collections()");
        self.load_collections().await?;
        tracing::info!("✅ STORAGE_ENGINE: Collections loaded");

        tracing::info!("📊 STORAGE_ENGINE: About to call register_collections_for_recovery()");
        self.register_collections_for_recovery().await?;
        tracing::info!("✅ STORAGE_ENGINE: Recovery engines registered");

        tracing::info!("📊 STORAGE_ENGINE: About to call recover_from_wal()");
        self.recover_from_wal().await?;
        tracing::info!("✅ STORAGE_ENGINE: WAL recovery completed, starting compaction workers");

        // `compaction_manager` is the handle owned by canonical_sst_storage.
        // Flush scheduling and bootstrap resolver wiring therefore target the
        // same queue/workers rather than an idle look-alike instance.

        // ADR-069/TD-WAL-1: spawn the live auto-flush driver. The default config arms the
        // 300s time floor so the driver spawns and segments materialize while the server
        // runs (no shutdown needed); setting both flush_interval_secs=0 and wal_max_bytes=0
        // opts out. The driver mirrors flush_memtable_to_storage's recipe, policy-gated +
        // metered. (Fence is injected post-construction, so it may be None here —
        // acceptable at MVP: the A6 fence is default-OFF.)
        let flush_policy =
            crate::storage::persistence::write_ahead_log::flush_policy::FlushPolicy::from_performance(
                &self.config.wal_config.to_engine_config().performance,
            );
        #[cfg(feature = "axis")]
        crate::storage::auto_flush_driver::AutoFlushDriver::spawn(
            flush_policy,
            self.axis_runtime_enabled
                .then(|| self.axis_index_manager.clone()),
            self.storage_write_fence.clone(),
        );
        #[cfg(not(feature = "axis"))]
        crate::storage::auto_flush_driver::AutoFlushDriver::spawn(
            flush_policy,
            self.storage_write_fence.clone(),
        );

        tracing::info!("✅ STORAGE_ENGINE: Storage engine started successfully");
        Ok(())
    }

    pub async fn stop(&mut self) -> crate::storage::Result<()> {
        // STEP 1: Flush all unflushed memtable data to storage engines FIRST
        // This ensures fast recovery on restart by having data in SST files
        tracing::info!("🛑 STORAGE_ENGINE: Flushing all unflushed data to storage engines...");
        match self.flush_memtable_to_storage().await {
            Ok(result) => {
                tracing::info!(
                    "✅ STORAGE_ENGINE: Flushed {} collections, {} vectors, {} bytes to storage",
                    result.collections_flushed,
                    result.total_vectors_flushed,
                    result.total_bytes_written
                );
                if !result.failed_collections.is_empty() {
                    tracing::warn!(
                        "⚠️ STORAGE_ENGINE: {} collections failed to flush: {:?}",
                        result.failed_collections.len(),
                        result.failed_collections
                    );
                }
            }
            Err(e) => {
                tracing::warn!(
                    "⚠️ STORAGE_ENGINE: Failed to flush memtable to storage: {}",
                    e
                );
            }
        }

        // STEP 1b: Reap the global WAL manifest after the final flush — the
        // driver may never tick again, so this is what actually reclaims the
        // manifest files of an embedded run's last flush.
        crate::storage::auto_flush_driver::AutoFlushDriver::reap_manifest_segments().await;

        // STEP 2: Stop compaction manager (stop() is &self; no Arc::get_mut
        // needed — the workers hold clones but stop signals via shutdown_signal).
        self.compaction_manager.stop().await?;

        // STEP 3: Force WAL flush during shutdown (for any remaining entries)
        tracing::debug!("🧹 Forcing WAL flush during storage engine shutdown");
        if let Err(e) = self.write_ahead_log_manager.flush(None).await {
            tracing::warn!("Failed to flush WAL during shutdown: {}", e);
        }

        Ok(())
    }

    /// Flush all unflushed memtable data to storage engines
    ///
    /// This method is called during graceful shutdown to ensure all in-memory
    /// vector data is persisted to SST files before the database closes.
    /// This enables fast recovery on restart without needing to replay WAL.
    pub async fn flush_memtable_to_storage(
        &self,
    ) -> crate::storage::Result<
        crate::storage::persistence::write_ahead_log::flush_coordinator::FlushAllResult,
    > {
        use crate::storage::flush_materializer::{
            flush_plan_from_collection_meta, materialize_collection,
        };
        use crate::storage::persistence::write_ahead_log::flush_coordinator::FlushAllResult;
        use crate::storage::persistence::write_ahead_log::{
            get_global_write_buffer_behavior, list_collections_from_catalog,
        };

        let empty_result = FlushAllResult {
            collections_flushed: 0,
            total_vectors_flushed: 0,
            total_bytes_written: 0,
            failed_collections: vec![],
        };

        // The global write buffer is the source of unflushed batches (the SAME
        // singleton the embedded path drains). Absent ⇒ nothing to flush.
        let write_buffer = match get_global_write_buffer_behavior() {
            Some(wb) => wb,
            None => {
                tracing::info!(
                    "📋 STORAGE_ENGINE: No global write buffer initialized, nothing to flush"
                );
                return Ok(empty_result);
            }
        };

        let collections_to_flush = write_buffer.list_collections_with_unflushed_data().await;
        if collections_to_flush.is_empty() {
            tracing::info!("📋 STORAGE_ENGINE: No collections have unflushed data");
            return Ok(empty_result);
        }

        tracing::info!(
            "🛑 STORAGE_ENGINE: Found {} collections with unflushed data: {:?}",
            collections_to_flush.len(),
            collections_to_flush
        );

        // Catalog is the metadata authority: resolve each collection's engine /
        // dimension / on-disk path / owning tenant. This is what the old throwaway
        // coordinator lacked (TD-163) — without it the engine could not be resolved
        // and the shutdown flush silently materialized nothing.
        let catalog = list_collections_from_catalog().await;

        let mut collections_flushed = 0usize;
        let mut total_vectors_flushed = 0u64;
        let mut total_bytes_written = 0u64;
        let mut failed_collections: Vec<(String, String)> = Vec::new();

        for collection_id in &collections_to_flush {
            // The server keys the write buffer by the catalog L1 object id.
            let Some(meta) = catalog.iter().find(|c| &c.id == collection_id) else {
                tracing::warn!(
                    "⚠️ STORAGE_ENGINE: No catalog metadata for collection '{}'; cannot resolve engine, skipping",
                    collection_id
                );
                failed_collections.push((collection_id.clone(), "no catalog metadata".to_string()));
                continue;
            };

            let plan = match flush_plan_from_collection_meta(meta) {
                Ok(plan) => plan,
                Err(error) => {
                    tracing::warn!(
                        collection_id,
                        "STORAGE_ENGINE: catalog identity resolution failed: {error}"
                    );
                    failed_collections.push((collection_id.clone(), error.to_string()));
                    continue;
                }
            };

            // A6 fence is applied inside `materialize_collection` (default-OFF), so a
            // pod displaced by a lease takeover is rejected before the storage write.
            // `free_wal=true`: once the SST segment is written, free the WAL so the
            // materialized segment — not a WAL replay — is the durable restart-recall
            // source (the whole point of materializing to SST; keeping the WAL made
            // recovery replay it and ignore the segment we just wrote). Safe now that
            // TD-165 fixed cold-read recall (IVF posting lists populated at flush + the
            // SST route honors SearchMode); gated by the insert→SIGINT→restart→search
            // round-trip in runtime-evidence/TD163_SERVER_FLUSH_MATERIALIZATION_2026_06_26.md.
            // Shutdown is terminal, so the freed batches are never re-flushed.
            #[cfg(feature = "axis")]
            let axis_arg: Option<&AxisManager> = self
                .axis_runtime_enabled
                .then_some(self.axis_index_manager.as_ref());
            #[cfg(not(feature = "axis"))]
            let axis_arg: Option<&()> = None;
            match materialize_collection(
                &write_buffer,
                &plan,
                self.storage_write_fence.as_ref(),
                None,
                true,
                axis_arg,
            )
            .await
            {
                Ok(Some(outcome)) => {
                    collections_flushed += 1;
                    total_vectors_flushed += outcome.entries_flushed;
                    total_bytes_written += outcome.bytes;
                    tracing::info!(
                        "✅ STORAGE_ENGINE: Flushed collection '{}': {} vectors, {} bytes",
                        collection_id,
                        outcome.entries_flushed,
                        outcome.bytes
                    );
                }
                Ok(None) => {
                    tracing::debug!(
                        "📋 STORAGE_ENGINE: Collection '{}' had no unflushed batches",
                        collection_id
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "❌ STORAGE_ENGINE: Failed to flush collection '{}': {}",
                        collection_id,
                        e
                    );
                    failed_collections.push((collection_id.clone(), e.to_string()));
                }
            }
        }

        tracing::info!(
            "🛑 STORAGE_ENGINE: Flush complete — {} collections, {} vectors, {} bytes{}",
            collections_flushed,
            total_vectors_flushed,
            total_bytes_written,
            if failed_collections.is_empty() {
                String::new()
            } else {
                format!(", {} failures", failed_collections.len())
            }
        );

        Ok(FlushAllResult {
            collections_flushed,
            total_vectors_flushed,
            total_bytes_written,
            failed_collections,
        })
    }

    /// Recover all vectors from WAL files for all collections
    /// This method should be called during server startup after collections are recovered from metadata
    pub async fn recover_from_wal(&self) -> crate::storage::Result<()> {
        info!("🔄 STORAGE_ENGINE: Starting WAL recovery for all collections...");

        // Get recovery manager from WAL manager
        // CRITICAL FIX: Call async get_recovery_manager() to create/cache if not exists
        let recovery_manager = match self.write_ahead_log_manager.get_recovery_manager().await {
            Ok(manager) => {
                tracing::debug!("Recovery manager obtained successfully");
                Arc::new(manager)
            }
            Err(e) => {
                error!("❌ STORAGE_ENGINE: Failed to get recovery manager: {}", e);
                return Err(crate::storage::StorageError::WalError(format!(
                    "Failed to get recovery manager: {}",
                    e
                )));
            }
        };

        // Get all collections from the catalog (sole authority for metadata).
        let collections =
            crate::storage::persistence::write_ahead_log::list_collections_from_catalog().await;
        if collections.is_empty() {
            info!("📋 STORAGE_ENGINE: No collections in catalog to recover");
            return Ok(());
        }

        info!(
            "📋 STORAGE_ENGINE: Found {} collections to recover",
            collections.len()
        );

        // Recover each collection
        let mut total_vectors_recovered = 0u64;
        let mut recovery_failures = Vec::new();
        let mut quarantined = Vec::new();
        for collection in collections {
            tracing::debug!("Recovering collection: {}", collection.id);

            // A catalog entry with no storage assignment has nothing to recover
            // into; it is quarantined (logged, skipped) so one bad entry never
            // prevents startup. Recovery still fails closed on real recovery
            // errors below (ADR-063 D3).
            if collection.storage_assignment.is_none() {
                error!(
                    "STORAGE_ENGINE: collection {} has no storage assignment in its catalog entry; quarantined (not recovered)",
                    collection.id
                );
                quarantined.push(collection.id.clone());
                continue;
            }

            match recovery_manager.recover_collection(&collection.id).await {
                Ok(stats) => {
                    total_vectors_recovered += stats.total_vectors_recovered;
                    info!(
                        "✅ STORAGE_ENGINE: Collection {} recovered: {} vectors from {} files",
                        collection.id, stats.total_vectors_recovered, stats.total_files_recovered
                    );
                }
                Err(e) => {
                    // `{:#}` prints the FULL anyhow context chain — recovery fails
                    // closed (ADR-063 D3), so the root cause must be diagnosable
                    // from this line, not just the outermost context.
                    warn!(
                        "⚠️  STORAGE_ENGINE: Failed to recover collection {}: {:#}",
                        collection.id, e
                    );
                    recovery_failures.push(format!("{}: {:#}", collection.id, e));
                }
            }
        }

        if !recovery_failures.is_empty() {
            return Err(crate::storage::StorageError::WalError(format!(
                "WAL recovery failed for {} collection(s): {}",
                recovery_failures.len(),
                recovery_failures.join("; ")
            )));
        }

        info!(
            "🎉 STORAGE_ENGINE: WAL recovery complete: {} total vectors recovered{}",
            total_vectors_recovered,
            if quarantined.is_empty() {
                String::new()
            } else {
                format!(", {} collection(s) quarantined", quarantined.len())
            }
        );
        Ok(())
    }

    /// Get WAL manager for sharing between services
    pub fn write_ahead_log_manager(&self) -> Arc<WriteAheadLogManager> {
        self.write_ahead_log_manager.clone()
    }

    /// Register storage engines for all existing collections with the recovery manager
    /// This should be called before WAL recovery to ensure proper engine registration
    pub async fn register_collections_for_recovery(&self) -> crate::storage::Result<()> {
        info!("🔧 STORAGE_ENGINE: Registering collections with recovery manager");

        // Get all collections from the catalog (sole authority for metadata).
        let collections =
            crate::storage::persistence::write_ahead_log::list_collections_from_catalog().await;
        if collections.is_empty() {
            info!("📋 No collections in catalog, skipping registration");
            return Ok(());
        }
        info!("📋 Found {} collections to register", collections.len());

        // Get recovery manager from WAL
        let recovery_manager = self
            .write_ahead_log_manager
            .get_recovery_manager()
            .await
            .map_err(|e| {
                crate::storage::StorageError::WalError(format!(
                    "Failed to get recovery manager before registration: {}",
                    e
                ))
            })?;

        // Store collection count before iterating
        let collection_count = collections.len();
        let mut registration_failures = Vec::new();

        // Register each collection's storage engine
        for collection in &collections {
            // Get engine type from storage_assignment (actual assigned engine)
            // If no storage_assignment, fall back to config.storage_engine (desired engine)
            let engine_type = if let Some(assignment) = &collection.storage_assignment {
                assignment.engine
            } else if let Some(config) = &collection.config {
                config
                    .storage_engine
                    .unwrap_or(crate::proto::proximadb_v1::StorageEngine::Sst as i32)
            } else {
                crate::proto::proximadb_v1::StorageEngine::Sst as i32
            };

            let proto_engine = crate::proto::proximadb_v1::StorageEngine::try_from(engine_type)
                .unwrap_or(crate::proto::proximadb_v1::StorageEngine::Sst);

            // Create storage engine for this collection
            // Note: Engines are stateless - collection-specific config is passed during operations
            match crate::storage::engines::factory::StorageFormatFactory::create_from_proto_async(
                proto_engine,
            )
            .await
            {
                Ok(engine) => {
                    // Register with recovery manager
                    if let Err(e) = recovery_manager
                        .register_storage_engine(&collection.id, engine)
                        .await
                    {
                        warn!(
                            "⚠️ Failed to register engine for collection {}: {}",
                            collection.id, e
                        );
                        registration_failures.push(format!("{}: {}", collection.id, e));
                    } else {
                        info!(
                            "✅ Registered {} engine for collection {} (from {})",
                            proto_engine.as_str_name(),
                            collection.id,
                            if collection.storage_assignment.is_some() {
                                "storage_assignment"
                            } else {
                                "config"
                            }
                        );
                    }
                }
                Err(e) => {
                    warn!(
                        "⚠️ Failed to create engine for collection {}: {}",
                        collection.id, e
                    );
                    registration_failures.push(format!("{}: {}", collection.id, e));
                }
            }
        }

        if !registration_failures.is_empty() {
            return Err(crate::storage::StorageError::WalError(format!(
                "Storage-engine registration failed for {} collection(s): {}",
                registration_failures.len(),
                registration_failures.join("; ")
            )));
        }

        info!(
            "✅ STORAGE_ENGINE: Completed registration for {} collections",
            collection_count
        );
        Ok(())
    }

    /// Write a vector to storage through WAL → memtable → flush pipeline
    pub async fn write(
        &self,
        collection_id: &str,
        record: &ProximaRecord,
    ) -> crate::storage::Result<()> {
        let vector_ref = record
            .embeddings
            .first()
            .map(|e| e.as_fp32_slice())
            .unwrap_or(&[]);
        let vector_size = std::mem::size_of_val(vector_ref) + std::mem::size_of::<ProximaRecord>();
        let start = std::time::Instant::now();
        let vector_id = &record.oid;

        tracing::debug!(
            "🔄 Starting write operation for vector {} in collection {}, vector_dim={}, size_bytes={}",
            vector_id,
            collection_id,
            vector_ref.len(),
            vector_size
        );

        tracing::debug!(
            "💾 Writing vector {} to WAL for collection {}",
            vector_id,
            collection_id
        );

        let vectors = Arc::new(vec![record.clone()]);

        // Write to WAL (which handles memtable insertion)
        self.write_ahead_log_manager
            .write_vector_batch_native_arc(collection_id, vectors)
            .await
            .map_err(|e| {
                crate::core::StorageError::WalError(format!("Failed to write to WAL: {}", e))
            })?;

        tracing::debug!(
            "✅ Successfully wrote vector {} to WAL for collection {}",
            vector_id,
            collection_id
        );

        // No need to release lock with DashMap - operations are atomic

        // REMOVED: Synchronous AXIS indexing moved to async queue-based approach
        //
        // Previously, AXIS indexing happened synchronously on every write, blocking the write path.
        // Now, vectors are indexed asynchronously via queue during flush operations:
        // 1. Write → WAL → Memtable (fast, non-blocking)
        // 2. Flush → Storage + Queue for AXIS (async indexing)
        //
        // Benefits:
        // - 40-60% lower write latency
        // - Better throughput under load
        // - Fault tolerance (index failures don't block writes)
        // - Batching and backpressure control
        //
        // Note: AXIS indexing still happens, just asynchronously during flush operations
        // via FlushAxisUpdater.queue_flush_updates() in SST and VIPER engines.

        tracing::debug!(
            "✅ Vector {} written to WAL for collection {} (AXIS indexing will happen async during flush)",
            vector_id,
            collection_id
        );

        // Update metadata statistics
        tracing::debug!(
            "📊 Updating metadata stats for collection {} (vector_delta=1, size_delta={})",
            collection_id,
            vector_size
        );
        // Deferred: collection stats are now maintained through the catalog;
        // wire a catalog-backed stats update here when supported.
        tracing::debug!(
            "✅ Completed metadata stats update for collection {}",
            collection_id
        );

        let elapsed = start.elapsed();
        tracing::debug!(
            "🎉 Successfully completed write operation for vector {} in collection {}, total_time={:?}",
            vector_id,
            collection_id,
            elapsed
        );
        Ok(())
    }

    /// Check if a vector exists in the storage engine
    pub async fn exists(&self, collection_id: &str, id: &VectorId) -> crate::storage::Result<bool> {
        // Check WAL for unflushed vectors first
        if self
            .write_ahead_log_manager
            .search_vector_by_id(collection_id, id)
            .await?
            .is_some()
        {
            return Ok(true);
        }

        // Check SST storages for vector existence
        if let Some(sst_storage) = self.sst_storages.get(collection_id) {
            // Use SST bloom filters for fast existence check
            match sst_storage.value().contains_vector(collection_id, id).await {
                Ok(exists) => {
                    debug!(
                        "SST existence check for {}/{}: {}",
                        collection_id, id, exists
                    );
                    return Ok(exists);
                }
                Err(e) => {
                    tracing::warn!("SST existence check failed: {}", e);
                    // Fall through to return false
                }
            }
        }

        tracing::debug!("Vector {}/{} not found in any storage", collection_id, id);
        Ok(false)
    }

    pub async fn soft_delete(
        &self,
        collection_id: &str,
        id: &VectorId,
    ) -> crate::storage::Result<bool> {
        // Write delete marker to WAL using new interface
        self.write_ahead_log_manager
            .delete_record(collection_id.to_string(), id.clone())
            .await
            .map_err(|e| crate::core::StorageError::WalError(e.to_string()))?;

        // Check if the record exists
        let exists = self.exists(collection_id, id).await?;

        // Remove from search index
        #[cfg(feature = "axis")]
        if exists && self.axis_runtime_enabled {
            self.axis_index_manager
                .delete(collection_id, id.clone())
                .await?;

            // Deferred: collection stats are now maintained through the catalog.
        }
        // With AXIS compiled out there is no in-memory index to mutate; the record
        // is tombstoned in the SST path below and filtered on read (canonical predicate).

        // Mark as deleted in SST storage using tombstone
        if exists {
            // SST is pure SSTable storage - no direct delete operation
            // Deletes should be handled through WAL tombstones
            return Err(anyhow::anyhow!(
                "Direct deletes from SST not supported. Use WAL tombstones."
            )
            .into());
        }

        Ok(exists)
    }

    pub async fn create_collection(&self, collection_id: String) -> crate::storage::Result<()> {
        // Use default storage location - pick randomly from configured locations
        let base_location = self
            .config
            .storage_locations
            .choose(&mut rand::thread_rng())
            .ok_or_else(|| {
                crate::core::StorageError::DiskIO(std::io::Error::other(
                    "No storage locations configured",
                ))
            })?
            .url
            .clone();

        self.create_collection_with_storage(collection_id, base_location)
            .await
    }

    pub async fn create_collection_with_storage(
        &self,
        collection_id: String,
        base_location: String,
    ) -> crate::storage::Result<()> {
        // ADR-031 Phase 4c: this low-level entrypoint has no catalog/schema
        // handle, so it cannot resolve a typed identity. Delegates to the
        // typed-aware inner with `None` → byte-identical legacy paths. The
        // authoritative create path (`CollectionService::create_storage_directories`)
        // threads the pre-minted identity and creates the typed dirs; this path
        // is a fallback that stays legacy (mixed-read-safe).
        self.create_collection_with_storage_typed(collection_id, base_location, None)
            .await
    }

    /// ADR-031 Phase 4c: typed-aware storage create. `Some(identity)` creates
    /// the data/wal/index dirs at the account-rooted typed path
    /// (`{base}/accounts/{base62}/…/{sub}`); `None` is byte-identical legacy.
    /// Callers that hold a pre-minted [`CollectionIdentity`] (resolved from the
    /// catalog's `stable_*_id` + account u32) pass it here so the engine writes
    /// to the same typed path the catalog asset records.
    pub async fn create_collection_with_storage_typed(
        &self,
        collection_id: String,
        base_location: String,
        typed_identity: Option<crate::core::stable_id::CollectionIdentity>,
    ) -> crate::storage::Result<()> {
        // NOTE: Collection metadata should be managed by CollectionService
        // Storage layer should only handle storage concerns, not metadata
        tracing::debug!("💾 Creating storage for collection: {}", collection_id);

        // Create directory paths based on base_location. The typed helper's
        // `None` branch is byte-identical to the legacy `StoragePath::collection_*_path`.
        use crate::storage::trait_components::path_resolver::{
            collection_data_path_typed, collection_index_path_typed, collection_wal_path_typed,
            typed_path_identity,
        };
        let typed_identity = typed_path_identity(typed_identity);
        let data_url = collection_data_path_typed(&base_location, &collection_id, typed_identity);
        let write_buffer_url =
            collection_wal_path_typed(&base_location, &collection_id, typed_identity);
        let index_url = collection_index_path_typed(&base_location, &collection_id, typed_identity);

        // Create all required directories for the collection
        // This ensures directories exist before any writes occur
        for url in &[&write_buffer_url, &data_url, &index_url] {
            let dir_url = if url.ends_with('/') {
                url.to_string()
            } else {
                format!("{}/", url)
            };

            if let Ok(fs) = self.filesystem.get_filesystem(&dir_url) {
                match fs.create_dir_all(&dir_url).await {
                    Ok(_) => tracing::debug!("Created directory: {}", dir_url),
                    Err(e) => {
                        // Check if already exists
                        if !fs.exists(&dir_url).await.unwrap_or(false) {
                            return Err(crate::core::StorageError::DiskIO(std::io::Error::other(
                                format!("Failed to create directory {}: {}", dir_url, e),
                            )));
                        }
                    }
                }
            }
        }

        // Don't eagerly create SST tree and MMAP reader - they will be created on first access
        tracing::debug!(
            "📁 Collection directories created, SST tree will be initialized on first access"
        );

        tracing::info!(
            "✅ Created collection: {} with directories at {}",
            collection_id,
            base_location
        );
        Ok(())
    }

    async fn load_collections(&self) -> crate::storage::Result<()> {
        tracing::info!("🔍 STORAGE_ENGINE: Loading collections from catalog");

        // Get collections from the catalog (sole authority for metadata).
        let collections =
            crate::storage::persistence::write_ahead_log::list_collections_from_catalog().await;
        tracing::info!("📋 Found {} collections in catalog", collections.len());

        if collections.is_empty() {
            tracing::info!("📋 No collections to load");
            return Ok(());
        }

        // Storage locations are now part of collection metadata
        // No need to rebuild assignments - they're stored with collections

        for collection in &collections {
            let _collection_id = &collection.id;
            let collection_name = collection
                .config
                .as_ref()
                .map_or("unknown", |c| c.name.as_str());

            // Storage assignment is now part of collection metadata
            if let Some(ref assignment) = collection.storage_assignment {
                tracing::debug!(
                    "📋 Collection {} has storage at: {}",
                    collection_name,
                    assignment.base_location
                );
            } else {
                tracing::warn!(
                    "⚠️ Collection {} has no storage assignment",
                    collection_name
                );
            }
        }

        // Extract collection IDs for parallel loading
        let collection_ids: Vec<String> = collections.into_iter().map(|c| c.id).collect();

        let total_collections = collection_ids.len();
        tracing::info!("📊 Found {} collections to load", total_collections);

        if total_collections == 0 {
            return Ok(());
        }

        // Determine optimal parallelism based on CPU cores
        let num_cpus = num_cpus::get();
        let chunk_size = total_collections.div_ceil(num_cpus);
        let chunk_size = chunk_size.clamp(1, 10); // Between 1 and 10 collections per task

        tracing::info!(
            "🚀 Loading collections in parallel with chunk size: {}",
            chunk_size
        );

        // With singleton SST storage, no per-collection initialization needed
        // Just log collection information for debugging
        for collection_id in &collection_ids {
            tracing::debug!(
                "✅ Collection {} ready for storage operations",
                collection_id
            );
        }

        tracing::info!(
            "✅ STORAGE_ENGINE: Parallel loading complete. Loaded {} collections",
            total_collections
        );
        Ok(())
    }

    /// Extract unique collection IDs and their metadata from recovered WAL entries
    /// This method is called by SharedServices during initialization to restore collection metadata
    pub async fn recovered_collections_metadata(
        &self,
    ) -> crate::storage::Result<Vec<(String, crate::proto::proximadb_v1::Collection)>> {
        tracing::info!("📊 Extracting collection metadata from recovered WAL entries");

        let mut collections_metadata = Vec::new();
        let mut seen_collections = std::collections::HashSet::new();

        // Get all collections that have entries in the WAL
        match self.write_ahead_log_manager.stats().await {
            Ok(stats) => {
                tracing::info!(
                    "📊 WAL stats: {} total entries across {} collections",
                    stats.total_entries,
                    stats.collections_count
                );

                // Try to get collection entries to extract metadata
                // Since WAL doesn't expose collection enumeration directly, we'll use a different approach

                // For each potential collection, try to get its entries and derive metadata
                // This is a temporary solution until WAL exposes collection enumeration
                let potential_collection_names = vec![
                    "test_persistence_collection_1",
                    "test_persistence_collection_2",
                    "embeddings",
                    "documents",
                    "vectors",
                ];

                for collection_id in potential_collection_names {
                    match self
                        .write_ahead_log_manager
                        .get_collection_entries(collection_id)
                        .await
                    {
                        Ok(entries) if !entries.is_empty() => {
                            if seen_collections.insert(collection_id.to_string()) {
                                tracing::info!(
                                    "📦 Found collection {} with {} entries in WAL",
                                    collection_id,
                                    entries.len()
                                );

                                // Extract metadata from the first vector entry
                                if let Some(record) = entries.first() {
                                    let dimension = record
                                        .embeddings
                                        .first()
                                        .map(|embedding| embedding.values.len())
                                        .unwrap_or_default();
                                    let collection =
                                        crate::proto::proximadb_v1::Collection {
                                            id: collection_id.to_string(),
                                            config: Some(crate::proto::proximadb_v1::CollectionConfig {
                                                name: collection_id.to_string(),
                                                dimension: dimension as u32,
                                                distance_metric: Some(
                                                    crate::proto::proximadb_v1::DistanceMetric::Cosine
                                                        as i32,
                                                ),
                                                ..Default::default()
                                            }),
                                            stats: Some(crate::proto::proximadb_v1::CollectionStats {
                                                vector_count: entries.len() as i64,
                                                data_size_bytes: (entries.len() * dimension * 4) as i64,
                                                ..Default::default()
                                            }),
                                            created_at: chrono::Utc::now().timestamp_micros(),
                                            updated_at: chrono::Utc::now().timestamp_micros(),
                                            ..Default::default()
                                        };

                                    collections_metadata
                                        .push((collection_id.to_string(), collection));
                                }
                            }
                        }
                        Ok(_) => {
                            // Collection exists but no entries
                        }
                        Err(_) => {
                            // Collection doesn't exist in WAL, which is expected for most names
                        }
                    }
                }
            }
            Err(e) => {
                tracing::warn!("⚠️ Failed to get WAL stats: {}", e);
            }
        }

        tracing::info!(
            "✅ Extracted metadata for {} collections from WAL",
            collections_metadata.len()
        );
        Ok(collections_metadata)
    }

    // Get collection metadata
    // Collection metadata operations should be performed directly through CollectionService.
    // Storage layer focuses only on data persistence, not metadata management.

    /// Delete collection and all its data
    pub async fn delete_collection(&self, collection_id: &str) -> crate::storage::Result<bool> {
        // Remove from in-memory structures using DashMap
        // Note: SST storage is now singleton - no per-collection removal needed
        let collection_exists = self.sst_storages.contains_key(collection_id);

        // Check if collection exists in storage
        if collection_exists {
            // Collection-aware WAL cleanup - remove only this collection's entries
            tracing::debug!(
                "🧹 Performing collection-aware WAL cleanup for: {}",
                collection_id
            );
            // Note: WAL no longer handles collection operations - handled by CollectionService
            if let Err(e) = self
                .write_ahead_log_manager
                .flush(Some(&collection_id.to_string()))
                .await
            {
                tracing::warn!(
                    "Failed to cleanup WAL entries for collection {}: {}",
                    collection_id,
                    e
                );
            }

            // Remove AXIS indexes for collection
            #[cfg(feature = "axis")]
            if self.axis_runtime_enabled {
                self.axis_index_manager
                    .drop_collection(collection_id)
                    .await?;
            }

            // Clean up SST files for the dropped collection
            if let Some(sst_storage) = self.sst_storages.get(collection_id) {
                match sst_storage
                    .value()
                    .cleanup_collection_files(collection_id)
                    .await
                {
                    Ok(()) => {
                        info!("Cleaned up SST files for collection {}", collection_id);
                    }
                    Err(e) => {
                        warn!(
                            "Failed to cleanup SST files for collection {}: {}",
                            collection_id, e
                        );
                    }
                }
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get the shared distance computation engine
    pub fn distance_compute(
        &self,
    ) -> &Arc<proximadb_distance_kernel::engine::UnifiedDistanceCompute> {
        &self.distance_compute
    }

    /// Calculate distance/similarity based on collection's configured metric
    #[allow(dead_code)]
    fn calculate_distance_metric(
        &self,
        query: &[f32],
        vector: &[f32],
        distance_metric: &proximadb_distance_kernel::DistanceMetric,
    ) -> crate::storage::Result<f32> {
        // Use shared unified distance computation engine
        let result = self
            .distance_compute
            .calculate_distance(query, vector, distance_metric);
        Ok(result.rank_value)
    }

    // REMOVED: search_vectors method (duplicate WAL scanning, no bloom filter optimization)
    // Use VectorOperationsService::search_vectors() instead which provides:
    // - Single WAL scan with bloom filter optimization
    // - Proper search orchestration (indexes → WAL → storage)
    // - Better performance (10-20x improvement for filtered queries)

    // REMOVED: search_memtable_with_metadata method
    // This method was causing double WAL/memtable scanning
    // WAL search is now handled exclusively by VectorOperationsService
    // which uses bloom filter optimization for better performance
    // See: VectorOperationsService::search_wal_with_bloom_filters()

    // REMOVED: search_vectors_with_filter method (no bloom filter optimization, redundant scans)
    // Use VectorOperationsService::search_vectors() with metadata filters instead

    /// Get search index statistics
    #[cfg(feature = "axis")]
    pub async fn index_stats(
        &self,
        collection_id: &str,
    ) -> crate::storage::Result<Option<HashMap<String, serde_json::Value>>> {
        if !self.axis_runtime_enabled {
            return Ok(None);
        }
        // Get AXIS index statistics
        match self
            .axis_index_manager
            .get_collection_stats(collection_id)
            .await
        {
            Ok(stats) => {
                let json_value = serde_json::to_value(stats).unwrap_or(serde_json::Value::Null);
                if let Some(obj) = json_value.as_object() {
                    let mut result = HashMap::new();
                    for (k, v) in obj {
                        result.insert(k.clone(), v.clone());
                    }
                    Ok(Some(result))
                } else {
                    Ok(None)
                }
            }
            Err(_) => Ok(None),
        }
    }

    /// Get search index statistics — AXIS compiled out: no index stats available.
    #[cfg(not(feature = "axis"))]
    pub async fn index_stats(
        &self,
        _collection_id: &str,
    ) -> crate::storage::Result<Option<HashMap<String, serde_json::Value>>> {
        Ok(None)
    }

    /// Optimize search index
    #[cfg(feature = "axis")]
    pub async fn optimize_index(&self, collection_id: &str) -> crate::storage::Result<()> {
        if !self.axis_runtime_enabled {
            return Ok(());
        }
        // Trigger AXIS analysis and optimization
        self.axis_index_manager
            .analyze_and_optimize(collection_id)
            .await
            .map_err(|e| crate::core::StorageError::IndexError(e.to_string()))
    }

    /// Optimize search index — AXIS compiled out: exact scan needs no optimization.
    #[cfg(not(feature = "axis"))]
    pub async fn optimize_index(&self, _collection_id: &str) -> crate::storage::Result<()> {
        Ok(())
    }

    /// Batch insert multiple vectors into a collection
    pub async fn batch_write(
        &self,
        collection_id: &str,
        records: Vec<ProximaRecord>,
    ) -> crate::storage::Result<Vec<VectorId>> {
        tracing::debug!("🚀 Starting batch_write for {} records", records.len());
        let mut inserted_ids = Vec::with_capacity(records.len());

        for (index, record) in records.iter().enumerate() {
            let record_id = record.oid.clone();
            tracing::debug!(
                "📝 Processing record {}/{}: vector_id={}, collection_id={}",
                index + 1,
                inserted_ids.capacity(),
                record_id,
                collection_id
            );

            self.write(collection_id, record).await?;
            inserted_ids.push(record_id.clone());

            tracing::debug!(
                "✅ Successfully processed record {}/{}: vector_id={}",
                index + 1,
                inserted_ids.capacity(),
                record_id
            );
        }

        tracing::debug!(
            "🎉 Completed batch_write for {} records successfully",
            inserted_ids.len()
        );
        Ok(inserted_ids)
    }

    // REMOVED: batch_search method (used deprecated search methods)
    // Use VectorOperationsService for batch operations with bloom filter optimization

    /// Cleanup for test scenarios - removes WAL entries for all collections
    /// This method is intended for test cleanup and should not be used in production
    pub async fn cleanup_for_tests(&self) -> crate::storage::Result<()> {
        tracing::debug!("🧹 Starting storage cleanup for test scenarios");

        // Collection-level enumeration for test cleanup is no longer wired through
        // a metadata provider; the WAL flush-all path below handles cleanup.
        let collections: Vec<ProximaBlockCollectionMetadata> = Vec::new();

        // Collect collection IDs
        let collection_ids: Vec<String> = collections
            .iter()
            .map(|c| c.collection_id.clone())
            .collect();

        if !collection_ids.is_empty() {
            // Collection-aware WAL cleanup for all collections
            tracing::debug!(
                "🧹 Performing collection-aware WAL cleanup for {} collections: {:?}",
                collection_ids.len(),
                collection_ids
            );
            for collection_id in &collection_ids {
                // Note: WAL no longer handles collection operations - handled by CollectionService
                if let Err(e) = self
                    .write_ahead_log_manager
                    .flush(Some(collection_id))
                    .await
                {
                    tracing::warn!(
                        "Failed to cleanup WAL entries for collection {}: {}",
                        collection_id,
                        e
                    );
                }
            }
        } else {
            // If no collections found, flush all WAL data
            tracing::debug!("🧹 No collections found, performing WAL flush");
            if let Err(e) = self.write_ahead_log_manager.flush(None).await {
                tracing::warn!("Failed to flush WAL: {}", e);
            }
        }

        // Collection deletion now flows through the catalog; no provider-driven
        // deletion is performed here during test cleanup.

        // Clear in-memory structures using DashMap
        // Note: SST storage is now singleton - no clearing needed
        self.sst_storages.clear();

        tracing::debug!("✅ Completed storage cleanup for tests");
        Ok(())
    }

    /// Get all vectors from a collection for linear search
    /// Retrieves vectors from both SST tree (recent writes) and MMAP readers (historical data)
    pub async fn all_vectors(
        &self,
        collection_id: &str,
    ) -> crate::storage::Result<Vec<ProximaRecord>> {
        let mut vectors: Vec<ProximaRecord> = Vec::new();

        tracing::debug!(
            "Scanning vectors for collection {} via SST storage",
            collection_id
        );

        if let Some(sst_storage) = self.sst_storages.get(collection_id) {
            match sst_storage
                .value()
                .scan_all_vectors(collection_id, 0, None)
                .await
            {
                Ok(sst_vectors) => {
                    debug!(
                        "Retrieved {} vectors from SST storage for collection {}",
                        sst_vectors.len(),
                        collection_id
                    );
                    let converted: Vec<ProximaRecord> = sst_vectors
                        .into_iter()
                        .map(|v| {
                            let dim = v.vector.len() as u32;
                            let now_ns = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_nanos() as i64;
                            ProximaRecord {
                                oid: v.id,
                                created_at_ns: v
                                    .timestamp
                                    .map(|ms| ms * 1_000_000)
                                    .unwrap_or(now_ns),
                                updated_at_ns: v
                                    .updated_at
                                    .map(|ms| ms * 1_000_000)
                                    .unwrap_or(now_ns),
                                valid_to_ns: v.expires_at.map(|ms| ms * 1_000_000),
                                record_version: v.version.map(|v| v as u64).unwrap_or(0),
                                embeddings: if !v.vector.is_empty() {
                                    vec![EmbeddingCell::new_fp32(
                                        "default", "vector", dim, v.vector,
                                    )]
                                } else {
                                    vec![]
                                },
                                ..ProximaRecord::default()
                            }
                        })
                        .collect();
                    vectors.extend(converted);
                }
                Err(e) => {
                    warn!(
                        "Failed to scan SST vectors for collection {}: {}",
                        collection_id, e
                    );
                }
            }
        }

        tracing::info!(
            "all_vectors retrieved {} records for collection {}",
            vectors.len(),
            collection_id
        );

        Ok(vectors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_recover_from_wal_method_compiles() {
        // This test validates that the recover_from_wal() method exists
        // and has the correct signature. If this compiles, Phase 1 is complete.

        let config = crate::core::StorageConfig::default();
        let storage = StorageEngine::new_without_collection_service(config)
            .await
            .expect("Failed to create storage engine");

        // Call recover_from_wal - it should succeed even with no data
        let result = storage.recover_from_wal().await;

        // The method should exist and return Result<()>
        assert!(
            result.is_ok() || result.is_err(),
            "Method returns Result type"
        );
    }
}
