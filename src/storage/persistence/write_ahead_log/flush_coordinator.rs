// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Common WAL Flush Coordination Logic
//!
//! This module provides shared flush coordination logic that can be used by
//! both AvroWAL and BincodeWAL implementations to manage flush state tracking,
//! cleanup of memory structures, and coordination between memory/disk WAL modes.

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use proximadb_storage_ports::CollectionMetadataPort;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use super::config::SyncMode;

use super::enhanced_flush_result::EnhancedFlushResult;
use super::flush_result_optimization::OptimizedFlushCoordinator;
use crate::storage::background_flush_context::BackgroundFlushContext;
use crate::storage::traits::{FlushParameters, FlushResult, UnifiedStorageFormat};

/// Flush state tracking for coordinated WAL cleanup
#[derive(Debug, Clone)]
pub struct FlushState {
    /// Pending flushes waiting for storage acknowledgment
    pub pending_flushes: HashMap<u64, PendingFlush>,
    /// Last successfully flushed sequence number
    pub last_flushed_sequence: u64,
    /// Whether this collection uses disk WAL (vs memory-only)
    pub uses_disk_wal: bool,
}

impl Default for FlushState {
    fn default() -> Self {
        Self {
            pending_flushes: HashMap::new(),
            last_flushed_sequence: 0,
            uses_disk_wal: true, // Default to disk WAL for durability
        }
    }
}

/// Information about a pending flush operation
#[derive(Debug, Clone)]
pub struct PendingFlush {
    /// Unique flush identifier
    pub flush_id: u64,
    /// Sequences being flushed
    pub sequences: Vec<u64>,
    /// When the flush was initiated
    pub initiated_at: DateTime<Utc>,
    /// Data source for this flush
    pub data_source: FlushDataSource,
}

/// Where flush data comes from (memory vs disk WAL files)
#[derive(Debug, Clone)]
pub enum FlushDataSource {
    /// Flush from memory structures (memory-only durability mode)
    Memory,
    /// Flush from disk WAL files (disk durability mode)
    DiskWalFiles(Vec<String>),
    /// Flush from pre-extracted canonical records (optimized path)
    VectorRecords(Vec<proximadb_records::ProximaRecord>),
}

/// Common flush coordination logic shared between WAL strategies
#[derive(Clone)]
pub struct WALFlushCoordinator {
    /// Per-collection flush state
    flush_states: Arc<RwLock<HashMap<String, FlushState>>>,
    /// Global flush ID counter
    next_flush_id: Arc<tokio::sync::Mutex<u64>>,
    /// Storage engine registry for polymorphic flush delegation
    storage_engines: Arc<RwLock<HashMap<String, Arc<dyn UnifiedStorageFormat>>>>,
    /// Optimized flush coordinator for high-performance flushing
    optimized_coordinator: Option<Arc<OptimizedFlushCoordinator>>,
    /// Collection service for fetching metadata
    collection_service: Option<Arc<dyn CollectionMetadataPort>>,
    /// Metrics updater for tracking flush operations
    metrics_updater: Option<Arc<dyn crate::metrics::InternalMetricsUpdater>>,
    /// Memtable manager for cleanup after flush
    memtable_manager: Option<Arc<super::memtable_manager::MemtableManager>>,
    /// A6 storage-write fence (default-OFF). When wired (from `shared_services`
    /// via the same `PartitionLeaseManager` the network write-gates use), a flush
    /// from a pod that has been fenced out of `(tenant, collection)` is rejected
    /// before any storage write. `None` ⇒ no fence wired ⇒ fail-open.
    storage_write_fence: Option<Arc<dyn crate::storage::write_fence::StorageWriteFence>>,
}

impl WALFlushCoordinator {
    /// Create new flush coordinator
    pub fn new() -> Self {
        Self {
            flush_states: Arc::new(RwLock::new(HashMap::new())),
            next_flush_id: Arc::new(tokio::sync::Mutex::new(1)),
            storage_engines: Arc::new(RwLock::new(HashMap::new())),
            optimized_coordinator: None,
            collection_service: None,
            metrics_updater: None,
            memtable_manager: None,
            storage_write_fence: None,
        }
    }

    /// Wire the A6 storage-write fence (default-OFF; enforced only when
    /// `PROXIMADB_WRITE_FENCING=1`). Injected from `shared_services` so the flush
    /// path shares the **same** `PartitionLeaseManager` instance as the network
    /// write-gates — a consistent, durable view of `(tenant, collection)`
    /// ownership across the pod. Absent ⇒ the boundary check fails open.
    pub fn set_storage_write_fence(
        &mut self,
        fence: Arc<dyn crate::storage::write_fence::StorageWriteFence>,
    ) {
        self.storage_write_fence = Some(fence);
        info!(
            "🔒 FlushCoordinator: A6 storage-write fence wired (default-OFF; set PROXIMADB_WRITE_FENCING=1 to enforce)"
        );
    }

    /// Resolve the owning tenant for a flush of `collection_id` (A6 fence input).
    ///
    /// Precedence (each is at most one amortized lookup per flush, never per row —
    /// D3 co-design): (1) the precomputed `BackgroundFlushContext.tenant_id`;
    /// (2) the collection service, reusing the canonical tag/owner resolver the
    /// network gates use. `None` ⇒ tenant unknown ⇒ the fence fails open. The
    /// shutdown coordinator has no collection service today (see the A6 TD /
    /// dependent materialization TD), so this returns `None` on that path and the
    /// fence stays dark there until that wiring lands.
    async fn resolve_flush_tenant(
        &self,
        collection_id: &str,
        flush_context: Option<&BackgroundFlushContext>,
    ) -> Option<String> {
        if let Some(ctx) = flush_context
            && let Some(tenant) = ctx.tenant_id.as_ref()
            && !tenant.is_empty()
        {
            return Some(tenant.clone());
        }
        if let Some(service) = &self.collection_service
            && let Ok(Some(collection)) = service.collection(collection_id).await
        {
            return proximadb_tenant::tenant_id_of(&collection);
        }
        None
    }

    /// Set memtable manager for cleanup after flush
    pub fn set_memtable_manager(&mut self, manager: Arc<super::memtable_manager::MemtableManager>) {
        self.memtable_manager = Some(manager);
        info!("🔗 FlushCoordinator: Memtable manager registered for post-flush cleanup");
    }

    /// Set collection service for metadata fetching
    pub fn set_collection_service(&mut self, service: Arc<dyn CollectionMetadataPort>) {
        self.collection_service = Some(service);
    }

    /// Set metrics updater for tracking flush operations
    pub fn set_metrics_updater(
        &mut self,
        updater: Arc<dyn crate::metrics::InternalMetricsUpdater>,
    ) {
        self.metrics_updater = Some(updater);
        info!("🔗 FlushCoordinator: Metrics updater registered for flush operation tracking");
    }

    /// Enable optimized flush processing
    pub fn enable_optimized_flush(
        &mut self,
        batch_size: usize,
        worker_count: usize,
        dimension: usize,
    ) {
        self.optimized_coordinator = Some(Arc::new(OptimizedFlushCoordinator::new(
            batch_size,
            worker_count,
            dimension,
        )));
        info!(
            "🚀 FlushCoordinator: Optimized flush enabled with batch_size={}, workers={}",
            batch_size, worker_count
        );
    }

    /// Initialize flush state for a collection
    /// Register a storage engine for polymorphic flush delegation
    pub async fn register_storage_engine(
        &self,
        engine_type: &str,
        engine: Arc<dyn UnifiedStorageFormat>,
    ) {
        let mut engines = self.storage_engines.write().await;
        engines.insert(engine_type.to_string(), engine);
        info!(
            "🏭 Registered {} storage engine with FlushCoordinator",
            engine_type
        );
    }

    /// Clean up flush coordinator state for a deleted collection
    pub async fn cleanup_collection(&self, collection_id: &str) {
        let mut flush_states = self.flush_states.write().await;
        if flush_states.remove(collection_id).is_some() {
            info!(
                "🧹 Cleaned up flush coordinator state for collection: {}",
                collection_id
            );
        }
    }

    /// Execute coordinated flush: WAL → Storage Engine → WAL Cleanup (ATOMIC)
    /// 🚀 OPTIMIZED: Now accepts BackgroundFlushContext to eliminate service calls
    pub async fn execute_coordinated_flush(
        &self,
        collection_id: &str,
        flush_data: FlushDataSource,
        preferred_engine: Option<&str>,
        flush_context: Option<&BackgroundFlushContext>,
    ) -> Result<EnhancedFlushResult> {
        info!(
            "🚀 Coordinator: Starting ATOMIC coordinated flush for collection {}",
            collection_id
        );

        let _flush_id = proximadb_kernel::uuid::Uuid::new_v4().to_string();

        // Step 0: A6 storage-write fence (default-OFF; `PROXIMADB_WRITE_FENCING=1`).
        //
        // Checked *before* any record extraction, metadata resolution, or storage
        // write: a pod displaced by a lease takeover must not publish its buffered
        // (stale) data to shared storage, or it would durably resurrect
        // deleted/tombstoned records (CLAUDE.md #16; ADR-025 deletion vectors are a
        // dependent of A6). This is the boundary backstop behind the routing-layer
        // lease-on-write gate (#337); the decision delegates to the same durable
        // `PartitionLeaseManager` (#346 `is_fenced_out`) the gates consult.
        //
        // Co-design (Dim-1): at most one durable lease read per flush (amortized
        // over the whole batch), never per row. With fencing enabled, unknown
        // tenant / no fence wired / lease-read errors fail closed.
        {
            use crate::storage::write_fence::{
                FenceDecision, evaluate_fence, write_fencing_enabled,
            };
            let enabled = write_fencing_enabled();
            if enabled {
                let tenant = self
                    .resolve_flush_tenant(collection_id, flush_context)
                    .await;
                let now_ms = chrono::Utc::now().timestamp_millis();
                if evaluate_fence(
                    enabled,
                    self.storage_write_fence.as_ref(),
                    tenant.as_deref(),
                    collection_id,
                    now_ms,
                )
                .await
                    == FenceDecision::Fenced
                {
                    warn!(
                        target: "proximadb.fence",
                        tenant_id = tenant.as_deref().unwrap_or("<unknown>"),
                        collection_id = %collection_id,
                        "🛡️ A6 fence: this pod is fenced out of (tenant, collection); rejecting stale-pod flush before storage write"
                    );
                    return Err(anyhow::anyhow!(
                        "A6 storage-write fence: pod is fenced out of collection '{}' (tenant '{}') — a live lease is held by another pod; refusing to publish stale data",
                        collection_id,
                        tenant.as_deref().unwrap_or("<unknown>")
                    ));
                }
            }
        }

        // Step 1: Extract vector records from FlushDataSource + Mark for cleanup
        let vector_records = match &flush_data {
            FlushDataSource::Memory => {
                // Memory flush is handled by VectorOperationsService in the optimized architecture
                warn!(
                    "📋 Coordinator: Memory flush source used - should be handled by VectorOperationsService with context"
                );
                Vec::new()
            }
            FlushDataSource::DiskWalFiles(files) => {
                info!(
                    "📋 Coordinator: Extracting vector records from {} disk WAL files",
                    files.len()
                );
                // Implement comprehensive disk WAL file reading and recovery
                self.extract_vectors_from_disk_files(files)
                    .await
                    .unwrap_or_else(|e| {
                        warn!(
                            "📋 Coordinator: Failed to extract vectors from disk files: {}",
                            e
                        );
                        Vec::new()
                    })
            }
            FlushDataSource::VectorRecords(records) => {
                info!(
                    "📋 Coordinator: Using pre-extracted {} vector records",
                    records.len()
                );
                records.clone()
            }
        };

        if vector_records.is_empty() {
            info!(
                "📋 Coordinator: No vector records to flush, completing without storage operation"
            );
            return Ok(EnhancedFlushResult::new(
                FlushResult {
                    success: true,
                    collections_affected: vec![collection_id.to_string()],
                    entries_flushed: Some(0),
                    bytes_written: Some(0),
                    files_created: Some(0),
                    file_paths: vec![],
                    duration_ms: Some(0),
                    completed_at: chrono::Utc::now(),
                    engine_metrics: std::collections::HashMap::new(),
                    compaction_triggered: false,
                    compaction_error: None,
                    flushed_batch_ids: vec![],
                },
                Vec::new(),
            ));
        }

        info!(
            "📋 Coordinator: Prepared {} vector records for flush to storage",
            vector_records.len()
        );

        // Step 2: 🚀 OPTIMIZATION: Use pre-computed context metadata when available (eliminates service calls)
        // This includes: storage engine type, compression settings, storage assignment, etc.
        let collection_metadata = if let Some(context) = flush_context {
            info!(
                "✅ CONTEXT_OPTIMIZED: Using pre-computed metadata for collection {}",
                collection_id
            );
            // Use centralized helper method for consistent collection proto creation
            Some(context.to_collection_proto())
        } else if let Some(ref collection_service) = self.collection_service {
            // Fallback: Use collection service (legacy path)
            warn!("⚠️ FALLBACK: Using collection service - context not provided");
            match collection_service.collection(collection_id).await {
                Ok(Some(collection)) => {
                    info!(
                        "📋 Coordinator: Fetched collection metadata for '{}' - engine: {:?}, compression: {:?}",
                        collection_id,
                        collection.config.as_ref().map(|c| c.storage_engine),
                        collection
                            .config
                            .as_ref()
                            .and_then(|c| c.quantization.as_ref())
                    );
                    Some(collection)
                }
                Ok(None) => {
                    warn!(
                        "⚠️ Coordinator: Collection '{}' not found in metadata_info",
                        collection_id
                    );
                    None
                }
                Err(e) => {
                    warn!("⚠️ Coordinator: Failed to fetch collection metadata: {}", e);
                    None
                }
            }
        } else {
            warn!(
                "⚠️ Coordinator: No collection service available, proceeding without metadata_info"
            );
            None
        };

        // 🚀 OPTIMIZATION: Determine storage engine - use context directly when available
        let engine_type = if let Some(context) = flush_context {
            // Direct context optimization - no metadata parsing needed!
            info!(
                "✅ ENGINE_OPTIMIZED: Using pre-computed engine {} for collection {}",
                context.engine_name(),
                collection_id
            );
            context.engine_name()
        } else if let Some(ref metadata) = collection_metadata {
            // Legacy path: Parse from metadata
            if let Some(ref config) = metadata.config {
                // Map proto storage engine enum to string
                use crate::proto::proximadb_v1::StorageEngine;
                match StorageEngine::try_from(config.storage_engine.unwrap_or(0)) {
                    Ok(StorageEngine::Viper) => "viper", // wire label; engine removed (ADR-093)
                    Ok(StorageEngine::Sst) => "sst",
                    Ok(StorageEngine::Tst) => "tst",
                    _ => preferred_engine.unwrap_or("sst"),
                }
            } else {
                preferred_engine.unwrap_or("sst")
            }
        } else {
            preferred_engine.ok_or_else(|| {
                anyhow::anyhow!(
                    "No storage engine specified for collection {} and no metadata available",
                    collection_id
                )
            })?
        };

        info!(
            "🔍 Coordinator: Using {} storage engine for collection {}",
            engine_type, collection_id
        );

        let engine = {
            let engines = self.storage_engines.read().await;
            info!(
                "🔍 Coordinator: Available engines: {:?}",
                engines.keys().collect::<Vec<_>>()
            );
            engines
                .get(engine_type)
                .ok_or_else(|| anyhow::anyhow!("Storage engine {} not registered", engine_type))?
                .clone()
        };

        info!(
            "🔄 Coordinator: Using {} engine for ATOMIC flush with metadata_info",
            engine_type
        );

        // Step 3: Create flush parameters with actual vector data + BatchId coordination
        let batch_ids = Vec::new(); // No cycle data needed in this simplified flow

        // Clone canonical records for projection indexing before moving into flush params.
        let vector_records_for_indexing = vector_records.clone();

        // Check if optimized flush is enabled and use it
        let storage_result = if let Some(optimized) = &self.optimized_coordinator {
            info!("🚀 Coordinator: Using optimized flush path");

            // Execute optimized flush
            let optimized_result = optimized
                .execute_optimized_flush(collection_id, vector_records)
                .await?;

            // Convert to standard flush result
            let mut base_result = optimized_result.base.clone();
            base_result.flushed_batch_ids = batch_ids.clone();

            // Store optimized records for later AXIS indexing.
            base_result
        } else {
            // Regular flush path with collection metadata
            let flush_params = FlushParameters {
                collection_id: Some(collection_id.to_string()),
                force: true,
                synchronous: true,
                vector_records,
                batch_ids, // ✅ Include BatchIds for coordination
                collection_config: collection_metadata.clone(), // ✅ Pass metadata to avoid duplicate fetches
                ..Default::default()
            };

            // Step 4: Execute polymorphic flush via the single trait `flush()`
            // funnel (D1 convergence). `flush()` wraps the engine-specific
            // `do_flush` with the common pre/post processing (validation, timing,
            // optional post-flush compaction), so the coordinator path and the
            // embedded path (`storage_engine.flush()`) now go through ONE funnel
            // rather than the coordinator reaching past it to `do_flush`. The bytes
            // written are identical (`flush()` delegates to the same `do_flush`);
            // `trigger_compaction` is unset here, so no extra compaction is run.
            engine.flush(flush_params).await?
        };

        info!(
            "✅ Coordinator: Storage flush completed - {} entries, {} bytes, {} files",
            storage_result.entries_flushed.unwrap_or(0),
            storage_result.bytes_written.unwrap_or(0),
            storage_result.files_created.unwrap_or(0)
        );

        // Step 5: ATOMIC WAL CLEANUP using BatchId coordination - Only if storage flush succeeded
        if storage_result.success && storage_result.entries_flushed.unwrap_or(0) > 0 {
            info!(
                "🧹 Coordinator: Starting BatchId-coordinated cleanup for {} flushed entries, {} batch IDs",
                storage_result.entries_flushed.unwrap_or(0),
                storage_result.flushed_batch_ids.len()
            );

            // WAL cleanup is handled by VectorOperationsService in the optimized architecture
            // The context-based approach ensures proper coordination between flush and cleanup
            info!(
                "📋 Coordinator: WAL cleanup handled by VectorOperationsService with context optimization"
            );

            // Cleanup memtable using BatchIds - remove flushed data from memory
            if let Some(ref memtable_manager) = self.memtable_manager {
                if !storage_result.flushed_batch_ids.is_empty() {
                    match memtable_manager
                        .remove_flushed_batches(collection_id, &storage_result.flushed_batch_ids)
                        .await
                    {
                        Ok(()) => {
                            info!(
                                "🧹 Coordinator: Successfully cleaned up {} batches from memtable for collection {}",
                                storage_result.flushed_batch_ids.len(),
                                collection_id
                            );
                        }
                        Err(e) => {
                            warn!(
                                "⚠️ Coordinator: Failed to cleanup memtable for collection {}: {:?}",
                                collection_id, e
                            );
                            // Continue despite cleanup failure - data is already persisted
                        }
                    }
                }
            } else {
                debug!(
                    "📋 Coordinator: Memtable cleanup skipped (no manager registered) for {} batches",
                    storage_result.flushed_batch_ids.len()
                );
            }
        } else {
            info!("📋 Coordinator: Skipping cleanup (no entries flushed or storage failed)");
        }

        // NOTE: AXIS indexing notification is handled by BackgroundManager
        // after the complete flush-compaction cycle to ensure proper sequential execution:
        // 1. FLUSH (materialized data)
        // 2. COMPACTION (if needed)
        // 3. INDEXING (final optimized layout)
        info!(
            "📋 Coordinator: Flush completed - indexing will be handled by BackgroundManager after compaction cycle"
        );

        info!(
            "🎯 Coordinator: ATOMIC coordinated flush COMPLETE for collection {}",
            collection_id
        );

        // 📊 METRICS: Record flush operation metrics (non-blocking)
        if let Some(ref metrics) = self.metrics_updater {
            let engine_type_str = if let Some(context) = flush_context {
                context.engine_name().to_uppercase()
            } else {
                engine_type.to_uppercase()
            };

            let _ = metrics
                .record_flush(
                    collection_id,
                    crate::metrics::FlushMetricsUpdate {
                        vectors_flushed: storage_result.entries_flushed.unwrap_or(0) as i64,
                        bytes_written: storage_result.bytes_written.unwrap_or(0) as i64,
                        duration_ms: storage_result.duration_ms.unwrap_or(0) as i64,
                        files_created: storage_result.files_created.unwrap_or(0) as i32,
                        engine_type: engine_type_str,
                        timestamp: chrono::Utc::now().timestamp_millis(),
                    },
                )
                .await;
            debug!("📊 Recorded flush metrics for collection {}", collection_id);
        }

        // Return enhanced result with canonical record data for projection indexing.
        Ok(EnhancedFlushResult::new(
            storage_result,
            vector_records_for_indexing,
        ))
    }

    pub async fn initialize_flush_state(&self, collection_id: &str) -> Result<()> {
        let mut flush_states = self.flush_states.write().await;
        if !flush_states.contains_key(collection_id) {
            flush_states.insert(collection_id.to_string(), FlushState::default());
            debug!(
                "🔄 Initialized flush state for collection: {}",
                collection_id
            );
        }
        Ok(())
    }

    /// Initiate a flush operation and return the data source
    /// This method determines whether to flush from memory or disk based on configuration
    pub async fn initiate_flush(
        &self,
        collection_id: &str,
        sequences: Vec<u64>,
        sync_mode: &SyncMode,
    ) -> Result<FlushDataSource> {
        let flush_id = {
            let mut next_id = self.next_flush_id.lock().await;
            let id = *next_id;
            *next_id += 1;
            id
        };

        let mut flush_states = self.flush_states.write().await;
        let flush_state = flush_states
            .entry(collection_id.to_string())
            .or_insert_with(FlushState::default);

        // Determine data source based on sync mode and configuration
        let data_source = match sync_mode {
            SyncMode::MemoryOnly => {
                flush_state.uses_disk_wal = false;
                FlushDataSource::Memory
            }
            _ => {
                flush_state.uses_disk_wal = true;
                // Get WAL files for these sequences (placeholder - to be implemented by strategy)
                let wal_files = self
                    .get_wal_files_for_sequences(collection_id, &sequences)
                    .await?;
                FlushDataSource::DiskWalFiles(wal_files)
            }
        };

        // Track pending flush
        let pending_flush = PendingFlush {
            flush_id,
            sequences: sequences.clone(),
            initiated_at: Utc::now(),
            data_source: data_source.clone(),
        };

        flush_state.pending_flushes.insert(flush_id, pending_flush);

        info!(
            "🚀 Initiated flush {} for collection {} with {} sequences from {:?}",
            flush_id,
            collection_id,
            sequences.len(),
            data_source
        );

        Ok(data_source)
    }

    /// Acknowledge a successful flush and clean up corresponding WAL data
    /// This is called by the storage engine after successful flush
    pub async fn acknowledge_flush(
        &self,
        collection_id: &str,
        flush_id: u64,
        flushed_sequences: Vec<u64>,
    ) -> Result<CleanupInstructions> {
        let mut flush_states = self.flush_states.write().await;
        let flush_state = flush_states
            .get_mut(collection_id)
            .ok_or_else(|| anyhow::anyhow!("No flush state for collection: {}", collection_id))?;

        let pending_flush = flush_state
            .pending_flushes
            .remove(&flush_id)
            .ok_or_else(|| anyhow::anyhow!("No pending flush with ID: {}", flush_id))?;

        // Update last flushed sequence
        if let Some(&max_seq) = flushed_sequences.iter().max() {
            flush_state.last_flushed_sequence = flush_state.last_flushed_sequence.max(max_seq);
        }

        // Determine cleanup instructions based on data source
        let cleanup_instructions = match pending_flush.data_source {
            FlushDataSource::Memory => CleanupInstructions {
                cleanup_memory: true,
                cleanup_disk_files: Vec::new(),
                sequences_to_cleanup: flushed_sequences.clone(),
            },
            FlushDataSource::DiskWalFiles(wal_files) => {
                // Only cleanup disk files that are fully flushed
                let files_to_cleanup = self
                    .filter_fully_flushed_files(collection_id, &wal_files, &flushed_sequences)
                    .await?;

                CleanupInstructions {
                    cleanup_memory: true, // Always cleanup memory after successful flush
                    cleanup_disk_files: files_to_cleanup,
                    sequences_to_cleanup: flushed_sequences.clone(),
                }
            }
            FlushDataSource::VectorRecords(_) => CleanupInstructions {
                cleanup_memory: true, // Cleanup memory after successful flush
                cleanup_disk_files: Vec::new(),
                sequences_to_cleanup: flushed_sequences.clone(),
            },
        };

        info!(
            "✅ Acknowledged flush {} for collection {} - {} sequences flushed, cleanup: memory={}, disk_files={}",
            flush_id,
            collection_id,
            flushed_sequences.len(),
            cleanup_instructions.cleanup_memory,
            cleanup_instructions.cleanup_disk_files.len()
        );

        Ok(cleanup_instructions)
    }

    /// Get flush state for a collection
    pub async fn get_flush_state(&self, collection_id: &str) -> Option<FlushState> {
        let flush_states = self.flush_states.read().await;
        flush_states.get(collection_id).cloned()
    }

    /// Check if a collection uses disk WAL
    pub async fn uses_disk_wal(&self, collection_id: &str) -> bool {
        let flush_states = self.flush_states.read().await;
        flush_states
            .get(collection_id)
            .is_none_or(|state| state.uses_disk_wal) // Default to disk WAL
    }

    /// Get pending flushes for a collection
    pub async fn get_pending_flushes(&self, collection_id: &str) -> Vec<PendingFlush> {
        let flush_states = self.flush_states.read().await;
        flush_states
            .get(collection_id)
            .map_or_else(Vec::new, |state| {
                state.pending_flushes.values().cloned().collect()
            })
    }

    /// Cancel a pending flush (in case of errors)
    pub async fn cancel_flush(&self, collection_id: &str, flush_id: u64) -> Result<()> {
        let mut flush_states = self.flush_states.write().await;
        if let Some(flush_state) = flush_states.get_mut(collection_id) {
            flush_state.pending_flushes.remove(&flush_id);
            warn!(
                "❌ Cancelled flush {} for collection {}",
                flush_id, collection_id
            );
        }
        Ok(())
    }

    /// Drop all flush state for a collection
    pub async fn drop_collection(&self, collection_id: &str) -> Result<()> {
        let mut flush_states = self.flush_states.write().await;
        flush_states.remove(collection_id);
        info!("🗑️ Dropped flush state for collection: {}", collection_id);
        Ok(())
    }

    // Private helper methods (to be implemented by specific WAL strategies)

    /// Get WAL files containing the specified sequences
    /// This is a placeholder - actual implementation should be provided by the WAL strategy
    async fn get_wal_files_for_sequences(
        &self,
        collection_id: &str,
        sequences: &[u64],
    ) -> Result<Vec<String>> {
        // Placeholder implementation - to be overridden by strategy-specific logic
        warn!(
            "📁 get_wal_files_for_sequences not implemented for collection {} (sequences: {:?})",
            collection_id, sequences
        );
        Ok(Vec::new())
    }

    /// Filter WAL files that are fully flushed and can be safely deleted
    /// After a successful flush, all WAL files containing the flushed sequences
    /// can be safely deleted since their data is now persisted in SST files
    async fn filter_fully_flushed_files(
        &self,
        collection_id: &str,
        wal_files: &[String],
        flushed_sequences: &[u64],
    ) -> Result<Vec<String>> {
        // If no sequences were flushed or no WAL files provided, nothing to cleanup
        if flushed_sequences.is_empty() || wal_files.is_empty() {
            debug!(
                "🔍 No WAL files to cleanup for collection {} (sequences: {}, files: {})",
                collection_id,
                flushed_sequences.len(),
                wal_files.len()
            );
            return Ok(Vec::new());
        }

        // After a successful flush, all WAL files that were part of the flush operation
        // can be safely deleted since their data is now durably stored in SST files.
        // The flush operation ensures atomicity: either all data is persisted or none.
        info!(
            "🧹 Marking {} WAL files for deletion after successful flush of {} sequences for collection {}",
            wal_files.len(),
            flushed_sequences.len(),
            collection_id
        );

        Ok(wal_files.to_vec())
    }

    /// Extract vectors from disk WAL files for recovery and flushing
    async fn extract_vectors_from_disk_files(
        &self,
        wal_files: &[String],
    ) -> Result<Vec<proximadb_records::ProximaRecord>> {
        debug!(
            "📋 Coordinator: Extracting vectors from {} disk WAL files",
            wal_files.len()
        );

        let mut all_vectors = Vec::new();
        let mut files_processed = 0;
        let mut total_vectors_extracted = 0;

        for file_path in wal_files {
            // Determine serialization format from file extension
            let format = if file_path.ends_with(".pbwal") {
                crate::storage::persistence::write_ahead_log::serialization::SerializationFormat::ProtocolBuffers
            } else if file_path.ends_with(".avwal") {
                crate::storage::persistence::write_ahead_log::serialization::SerializationFormat::Avro
            } else if file_path.ends_with(".bcwal") {
                crate::storage::persistence::write_ahead_log::serialization::SerializationFormat::Bincode
            } else {
                warn!("Unknown WAL file format for file: {}", file_path);
                continue;
            };

            // Read and deserialize canonical WAL records.
            match self.read_wal_file_vectors(file_path, format).await {
                Ok(file_vectors) => {
                    let count = file_vectors.len();
                    all_vectors.extend(file_vectors);
                    total_vectors_extracted += count;
                    files_processed += 1;

                    debug!(
                        "📋 Coordinator: Extracted {} vectors from {}",
                        count, file_path
                    );
                }
                Err(e) => {
                    warn!(
                        "📋 Coordinator: Failed to extract vectors from {}: {}",
                        file_path, e
                    );
                    continue;
                }
            }
        }

        info!(
            "📋 Coordinator: Extracted {} vectors from {}/{} WAL files",
            total_vectors_extracted,
            files_processed,
            wal_files.len()
        );

        Ok(all_vectors)
    }

    /// Read canonical records from a single WAL file.
    async fn read_wal_file_vectors(
        &self,
        file_path: &str,
        format: crate::storage::persistence::write_ahead_log::serialization::SerializationFormat,
    ) -> Result<Vec<proximadb_records::ProximaRecord>> {
        use crate::storage::persistence::write_ahead_log::serialization::SerializerFactory;

        // Create filesystem interface to read the file
        // Note: This assumes we have access to a filesystem factory
        // In production, this would be injected as a dependency
        let filesystem_config =
            crate::storage::persistence::filesystem::FilesystemConfig::default();
        let filesystem_factory =
            crate::storage::persistence::filesystem::FilesystemFactory::create(filesystem_config)
                .await?;

        let filesystem = filesystem_factory.get_filesystem(file_path)?;

        // Read the file data
        let data = filesystem
            .read(file_path)
            .await
            .context("Failed to read WAL file")?;

        if data.is_empty() {
            warn!("Empty WAL file encountered: {}", file_path);
            return Ok(Vec::new());
        }

        // Create serializer for the detected format
        let serializer = SerializerFactory::create(format);

        // Deserialize the batch
        let vectors = serializer
            .deserialize_batch(&data)
            .context("Failed to deserialize WAL file")?;

        Ok(vectors)
    }

    /// Flush all collections with unflushed data to their respective storage engines
    ///
    /// This method is called during graceful shutdown to ensure all memtable data
    /// is persisted to storage engines before the database closes. It:
    /// 1. Gets list of collections with unflushed data from the global write buffer
    /// 2. For each collection, retrieves unflushed batches
    /// 3. Routes flush to the appropriate storage engine via the registered engines
    ///
    /// # Returns
    /// - `Ok(FlushAllResult)` with summary of flushed collections and vectors
    /// - `Err` if critical flush failures occur
    pub async fn flush_all_collections(&self) -> Result<FlushAllResult> {
        info!("🛑 FlushCoordinator: Starting graceful shutdown flush for all collections");

        // Get the global write buffer behavior singleton
        let write_buffer = match super::get_global_write_buffer_behavior() {
            Some(wb) => wb,
            None => {
                info!("📋 FlushCoordinator: No global write buffer initialized, nothing to flush");
                return Ok(FlushAllResult {
                    collections_flushed: 0,
                    total_vectors_flushed: 0,
                    total_bytes_written: 0,
                    failed_collections: vec![],
                });
            }
        };

        // Get list of collections with unflushed data
        let collections_to_flush = write_buffer.list_collections_with_unflushed_data().await;

        if collections_to_flush.is_empty() {
            info!(
                "📋 FlushCoordinator: No collections have unflushed data, shutdown flush complete"
            );
            return Ok(FlushAllResult {
                collections_flushed: 0,
                total_vectors_flushed: 0,
                total_bytes_written: 0,
                failed_collections: vec![],
            });
        }

        info!(
            "🔄 FlushCoordinator: Found {} collections with unflushed data: {:?}",
            collections_to_flush.len(),
            collections_to_flush
        );

        let mut total_vectors_flushed = 0u64;
        let mut total_bytes_written = 0u64;
        let mut collections_flushed = 0usize;
        let mut failed_collections = Vec::new();

        // Flush each collection
        for collection_id in &collections_to_flush {
            info!(
                "🔄 FlushCoordinator: Flushing collection '{}'",
                collection_id
            );

            // Get unflushed batches for this collection
            match write_buffer.get_unflushed_batches(collection_id).await {
                Ok(batches) => {
                    if batches.is_empty() {
                        debug!(
                            "📋 FlushCoordinator: Collection '{}' has no unflushed batches",
                            collection_id
                        );
                        continue;
                    }

                    // Combine all canonical records from unflushed batches.
                    let vector_records: Vec<proximadb_records::ProximaRecord> = batches
                        .iter()
                        .flat_map(|batch| batch.vector_records.iter().cloned())
                        .collect();

                    let vector_count = vector_records.len();
                    info!(
                        "📋 FlushCoordinator: Collection '{}' has {} vectors to flush from {} batches",
                        collection_id,
                        vector_count,
                        batches.len()
                    );

                    // Execute coordinated flush via storage engine
                    match self
                        .execute_coordinated_flush(
                            collection_id,
                            FlushDataSource::VectorRecords(vector_records),
                            None, // Let coordinator determine engine from collection metadata
                            None, // No flush context during shutdown
                        )
                        .await
                    {
                        Ok(result) => {
                            let entries = result.base.entries_flushed.unwrap_or(0);
                            let bytes = result.base.bytes_written.unwrap_or(0);

                            total_vectors_flushed += entries;
                            total_bytes_written += bytes;
                            collections_flushed += 1;

                            // Mark batches as flushed and clear from memtable
                            if let Err(e) = write_buffer.clear_flushed(collection_id).await {
                                warn!(
                                    "⚠️ FlushCoordinator: Failed to clear flushed batches for '{}': {}",
                                    collection_id, e
                                );
                            }

                            info!(
                                "✅ FlushCoordinator: Flushed collection '{}': {} vectors, {} bytes",
                                collection_id, entries, bytes
                            );
                        }
                        Err(e) => {
                            warn!(
                                "❌ FlushCoordinator: Failed to flush collection '{}': {}",
                                collection_id, e
                            );
                            failed_collections.push((collection_id.clone(), e.to_string()));
                        }
                    }
                }
                Err(e) => {
                    warn!(
                        "❌ FlushCoordinator: Failed to get unflushed batches for '{}': {}",
                        collection_id, e
                    );
                    failed_collections.push((collection_id.clone(), e.to_string()));
                }
            }
        }

        info!(
            "🛑 FlushCoordinator: Graceful shutdown flush complete - {} collections, {} vectors, {} bytes{}",
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
}

impl Default for WALFlushCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of flushing all collections during graceful shutdown
#[derive(Debug, Clone)]
pub struct FlushAllResult {
    /// Number of collections successfully flushed
    pub collections_flushed: usize,
    /// Total number of vectors flushed across all collections
    pub total_vectors_flushed: u64,
    /// Total bytes written to storage
    pub total_bytes_written: u64,
    /// Collections that failed to flush with error messages
    pub failed_collections: Vec<(String, String)>,
}

/// Instructions for cleaning up WAL data after successful flush
#[derive(Debug, Clone)]
pub struct CleanupInstructions {
    /// Whether to cleanup memory structures (ArtMap, HashMap, etc.)
    pub cleanup_memory: bool,
    /// Disk WAL files to delete
    pub cleanup_disk_files: Vec<String>,
    /// Specific sequences to cleanup from memory
    pub sequences_to_cleanup: Vec<u64>,
}

/// Trait for WAL strategies to implement flush coordination callbacks
#[async_trait]
pub trait FlushCoordinatorCallbacks {
    /// Get WAL files containing the specified sequences
    async fn get_wal_files_for_sequences(
        &self,
        collection_id: &str,
        sequences: &[u64],
    ) -> Result<Vec<String>>;

    /// Check if a WAL file is fully flushed and can be safely deleted
    async fn is_wal_file_fully_flushed(
        &self,
        collection_id: &str,
        wal_file: &str,
        flushed_sequences: &[u64],
    ) -> Result<bool>;
}

#[cfg(test)]
mod fence_wiring_tests {
    use super::*;
    use crate::storage::background_flush_context::{BackgroundFlushContext, StorageEngineType};

    /// `resolve_flush_tenant` prefers the tenant carried on the precomputed
    /// `BackgroundFlushContext` (D3: amortized, no extra lookup).
    #[tokio::test]
    async fn resolve_flush_tenant_prefers_context_tenant() {
        let coordinator = WALFlushCoordinator::new();
        let mut ctx = BackgroundFlushContext::for_testing("c", StorageEngineType::Sst);
        ctx.tenant_id = Some("acme".to_string());
        assert_eq!(
            coordinator
                .resolve_flush_tenant("c", Some(&ctx))
                .await
                .as_deref(),
            Some("acme")
        );
    }

    /// Fail-open: with no context tenant and no collection service wired (the
    /// shutdown-coordinator shape today), tenant resolution yields `None`, so the
    /// A6 fence stays dark on that path.
    #[tokio::test]
    async fn resolve_flush_tenant_is_none_without_context_or_service() {
        let coordinator = WALFlushCoordinator::new();
        assert_eq!(coordinator.resolve_flush_tenant("c", None).await, None);

        let ctx = BackgroundFlushContext::for_testing("c", StorageEngineType::Sst);
        assert_eq!(
            coordinator.resolve_flush_tenant("c", Some(&ctx)).await,
            None
        );
    }

    /// The fence setter wires an `Arc<dyn StorageWriteFence>` without consuming the
    /// coordinator (so the shutdown path can inject the shared lease-backed fence).
    #[tokio::test]
    async fn set_storage_write_fence_wires_a_fence() {
        use crate::storage::write_fence::StorageWriteFence;
        struct Never;
        #[async_trait::async_trait]
        impl StorageWriteFence for Never {
            async fn is_fenced_out(&self, _t: &str, _c: &str, _n: i64) -> anyhow::Result<bool> {
                Ok(false)
            }
        }
        let mut coordinator = WALFlushCoordinator::new();
        assert!(coordinator.storage_write_fence.is_none());
        coordinator.set_storage_write_fence(Arc::new(Never));
        assert!(coordinator.storage_write_fence.is_some());
    }
}
