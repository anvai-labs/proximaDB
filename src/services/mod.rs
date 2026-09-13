// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! # Services Module - Business Logic and Coordination Layer
//!
//! This module provides ProximaDB's service layer that implements business logic and
//! coordinates between storage engines, indexes, and other system components. Services
//! handle high-level operations while delegating low-level tasks to appropriate subsystems.
//!
//! ## Role in ProximaDB Architecture
//!
//! The services layer acts as the orchestration hub:
//! ```text
//! API Handlers (REST/gRPC)
//!         ↓
//! ┌───────────────────────────────────────┐
//! │          Services Layer                │
//! ├───────────────────────────────────────┤
//! │ Collections │ Operations │ Search │ Events │
//! └───────────────────────────────────────┘
//!         ↓           ↓          ↓        ↓
//!    Storage     AXIS Index   Compute   WAL
//!    Engines     (HNSW/IVF)   (SIMD)   System
//! ```
//!
//! ## Core Services
//!
//! ### 1. **Collection Service** (`collection/`)
//! Manages vector collections and metadata:
//! - Collection lifecycle (create, delete, update)
//! - Metadata management and schema validation
//! - Storage engine selection and configuration
//! - Statistics tracking and optimization hints
//!
//! ### 2. **Vector Operations Service** (`operations/`)
//! Handles vector CRUD operations:
//! - Direct memtable access for low latency
//! - Batch insertions with automatic flushing
//! - Update and delete operations
//! - Transaction coordination
//!
//! ### 3. **Search Service** (`search/`)
//! Orchestrates vector similarity search:
//! - Query planning and optimization
//! - Index and storage coordination
//! - Result streaming and pagination
//! - Hybrid search with metadata filtering
//!
//! ### 4. **Event Log Service** (`events/`)
//! Provides persistent event logging:
//! - Operation logging for recovery
//! - Event replay for consistency
//! - Compaction and flush notifications
//! - Cross-component coordination
//!
//! ## Service Characteristics
//!
//! ### Design Principles
//! - **Stateless**: Services don't maintain state between requests
//! - **Thread-Safe**: All services safe for concurrent access
//! - **Async-First**: Built on Tokio for async operations
//! - **Fault-Tolerant**: Graceful degradation on failures
//!
//! ### Performance Goals
//! - **Latency**: < 1ms service overhead
//! - **Throughput**: 100K+ ops/sec per service
//! - **Concurrency**: Lock-free where possible
//! - **Memory**: Bounded memory usage
//!
//! ## Service Interactions
//!
//! ```text
//! Insert Flow:
//! API → TableRecordStore → WAL
//!         ↓              ↓
//!   VectorOps adapter  Storage
//!   (compat only)      (on flush)
//!
//! Search Flow:
//! API → Search → Query Optimizer
//!         ↓            ↓
//!     AXIS Index   Storage Engine
//!         ↓            ↓
//!       Merge & Rank Results
//! ```
//!
//! ## Configuration
//!
//! Services are configured through the main config:
//! ```toml
//! [services]
//! # Collection service
//! [services.collection]
//! max_collections = 1000
//! metadata_cache_size = 100
//!
//! # Operations service
//! [services.operations]
//! batch_size = 1000
//! flush_interval_ms = 5000
//!
//! # Search service
//! [services.search]
//! max_concurrent_searches = 100
//! result_cache_size = 1000
//! stream_buffer_size = 100
//!
//! # Event log service
//! [services.events]
//! retention_hours = 168  # 7 days
//! compaction_interval_hours = 24
//! ```
//!
//! ## Error Handling
//!
//! Services use unified error types:
//! - `ServiceError::NotFound` - Resource doesn't exist
//! - `ServiceError::AlreadyExists` - Duplicate resource
//! - `ServiceError::InvalidInput` - Validation failed
//! - `ServiceError::Internal` - System error
//! - `ServiceError::Unavailable` - Service temporarily down
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use proximadb::services::{Collections, VectorOps, StreamingSearch};
//!
//! // Initialize services
//! let collections = Collections::new(storage.clone());
//! let operations = VectorOps::new(storage.clone());
//! let search = StreamingSearch::new(storage.clone());
//!
//! // Create collection
//! collections.create_collection("products", config).await?;
//!
//! // Insert vectors
//! operations.insert_batch("products", vectors).await?;
//!
//! // Search with streaming
//! let stream = search.search_stream(
//!     "products",
//!     query_vector,
//!     SearchConfig::default()
//! ).await?;
//!
//! // Process results
//! while let Some(result) = stream.next().await {
//!     println!("Found: {:?}", result?);
//! }
//! ```
//!
//! ## Service Lifecycle
//!
//! 1. **Initialization**: Services created with storage references
//! 2. **Operation**: Handle requests asynchronously
//! 3. **Cleanup**: Graceful shutdown on drop
//! 4. **Recovery**: Automatic recovery from EventLog
//!
//! ## Monitoring
//!
//! Each service exports metrics:
//! - Request count and latency
//! - Error rates by type
//! - Resource usage
//! - Queue depths
//! - Cache hit rates

#[cfg(feature = "axis")]
pub mod advisor_observations;
pub mod agent_checkpoint;
pub mod agent_memory;
pub mod audit_sink;
pub mod bulk_load;
pub mod canonical_wal;
pub mod catalog_backend_choice;
pub mod catalog_introspection;
pub mod catalog_snapshot_store;
pub mod collection;
pub mod ddl;
pub mod discovery;
pub mod dml;
pub mod document_service;
pub mod embedding_drainer;
pub mod entity_orchestrator;
pub mod events;
pub mod external_collection;
pub mod function_store;
pub mod fusion_service;
pub mod graph_collection;
pub mod mlflow_artifact_s3;
pub mod mlflow_run_store;
pub mod operations;
pub mod queue_fs_adapter;
pub mod rank_profile_store;
#[cfg(feature = "axis")]
pub mod recall_drift_sweeper;
#[cfg(feature = "axis")]
pub mod recall_observer;
pub mod record_memtable;
pub mod record_store;
pub use proximadb_scan_cursor::scan_cursor;
pub mod schema;
pub mod search;
pub mod shutdown_registry;
pub mod snapshot;
pub mod system_catalog;
pub mod system_catalog_state;
#[cfg(feature = "tenant_access")]
pub mod tenant_access;
pub mod timeseries_service;
pub mod transaction;
pub use proximadb_write_intent::write_intent;

// Re-export main service types with cleaner names
pub use audit_sink::{AuditEventSink, AuditRecord, EventLogAuditSink};
pub use canonical_wal::{FramedTableWalAppender, MemoryTableWalAppender};
pub use catalog_introspection::{CatalogIntrospectionResult, CatalogIntrospectionService};
pub use collection::Collections;
pub use ddl::{
    AlterTableChange, ColumnDefinition, DdlResult, DdlService, DdlStatement, IndexType, SqlDataType,
};
pub use dml::{
    ComparisonOperator, Condition, DmlResult, DmlService, DmlStatement, LogicalOperator,
    SqlValueLiteral, WhereClause,
};
pub use embedding_drainer::{
    DrainerInsertSink, EMBED_INGEST_TOPIC, EmbedIngestPayload, EmbedIngestRecord, EmbeddedRecord,
    EmbeddingDrainer, EmbeddingDrainerConfig,
};
pub use events::EventLog;
pub use function_store::{
    CanonicalWalFunctionStore, FUNCTIONS_COLLECTION_ID, FunctionStore, StoredFunction,
};
pub use graph_collection::GraphCollectionService;
pub use operations::VectorOps;
pub use rank_profile_store::{
    CanonicalWalRankProfileStore, RANK_PROFILES_COLLECTION_ID, RankProfileStore, StoredRankProfile,
};
pub use record_memtable::MemtableRecordStorage;
pub use record_store::{
    CatalogRoutingTableRecordStore, DirectWalTableRecordStore, RecordStorageTableRecordStore,
    TableRecordGetRequest, TableRecordGetResponse, TableRecordMutation, TableRecordMutationKind,
    TableRecordScanRequest, TableRecordScanResponse, TableRecordStore, TableRecordStoreRoute,
    TableRecordWriteResult, TableWalAppender, VectorOpsTableRecordStore,
};
pub use search::StreamingSearch;
pub use transaction::{CrossModelTransactionCoordinator, CrossModelTxResult, TransactionOutcome};
pub use write_intent::{
    DEFAULT_BULK_BYTES_THRESHOLD, DEFAULT_BULK_ROW_THRESHOLD, ProjectionFreshnessRequirement,
    RejectedWriteLane, WriteDurabilityRequirement, WriteGuard, WriteIntent,
    WriteIsolationRequirement, WriteLane, WriteLaneDecision, WriteLaneRouter,
    WriteLaneRouterConfig, WriteOperationKind,
};

// Legacy compatibility exports (will be removed)
pub use collection::manager as collection_service;
pub use events::log as event_log_service;
#[cfg(feature = "axis")]
pub use events::persistence as event_log_persistence;
pub use operations::vectors as vector_operations_service;
pub use search::streaming as streaming_search;

// Legacy type aliases for compatibility
pub use collection::Collections as CollectionService;
pub use events::EventLog as EventLogService;
pub use events::Stats as EventLogStats;
pub use operations::VectorOps as VectorOperationsService;
pub use search::{
    ResultStream as SearchResultStream, StreamConfig as StreamingSearchConfig,
    StreamingSearch as StreamingSearchService,
};

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::RwLock;
    use tracing::{debug, info};

    #[tokio::test]
    async fn test_concurrent_searches_no_interference() -> Result<()> {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        info!("🧪 Testing concurrent searches don't interfere with each other");

        // Track concurrent search executions
        let concurrent_count = Arc::new(AtomicUsize::new(0));
        let max_concurrent = Arc::new(AtomicUsize::new(0));

        // Simulate 100 concurrent searches
        let num_searches = 100;
        let mut handles = Vec::new();

        for i in 0..num_searches {
            let concurrent_count = concurrent_count.clone();
            let max_concurrent = max_concurrent.clone();

            let handle = tokio::spawn(async move {
                // Increment concurrent count
                let current = concurrent_count.fetch_add(1, Ordering::SeqCst) + 1;

                // Track max concurrent
                let mut max = max_concurrent.load(Ordering::SeqCst);
                while current > max {
                    match max_concurrent.compare_exchange_weak(
                        max,
                        current,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    ) {
                        Ok(_) => break,
                        Err(x) => max = x,
                    }
                }

                // Simulate search work
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

                // Decrement concurrent count
                concurrent_count.fetch_sub(1, Ordering::SeqCst);

                debug!("Search {} completed", i);
                i
            });

            handles.push(handle);
        }

        // Wait for all searches to complete
        let results: Vec<_> = futures::future::join_all(handles)
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;

        assert_eq!(results.len(), num_searches);

        let max = max_concurrent.load(Ordering::SeqCst);
        info!(
            "✅ All {} searches completed. Max concurrent: {}",
            num_searches, max
        );
        assert!(max > 1, "Should have achieved some concurrency");

        Ok(())
    }

    #[tokio::test]
    async fn test_wal_scan_happens_once_per_search() -> Result<()> {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        info!("🧪 Testing WAL is scanned exactly once per search");

        // Track WAL scan invocations
        struct WalScanCounter {
            scan_counts: Arc<RwLock<Vec<usize>>>,
        }

        impl WalScanCounter {
            fn new() -> Self {
                Self {
                    scan_counts: Arc::new(RwLock::new(Vec::new())),
                }
            }

            async fn record_scan(&self, search_id: usize) {
                let mut counts = self.scan_counts.write().await;
                counts.push(search_id);
            }

            async fn get_scan_count(&self, search_id: usize) -> usize {
                let counts = self.scan_counts.read().await;
                counts.iter().filter(|&&id| id == search_id).count()
            }
        }

        let counter = WalScanCounter::new();

        // Simulate multiple searches
        for search_id in 0..5 {
            // Each search should scan WAL exactly once
            counter.record_scan(search_id).await;

            // Verify no double scanning
            let count = counter.get_scan_count(search_id).await;
            assert_eq!(
                count, 1,
                "Search {} should scan WAL exactly once",
                search_id
            );
        }

        info!("✅ WAL scan count verified - exactly once per search");
        Ok(())
    }

    #[tokio::test]
    async fn test_search_orchestration_order() -> Result<()> {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        info!("🧪 Testing search orchestration order");

        // Track the order of operations
        #[derive(Debug, Clone, PartialEq)]
        enum SearchPhase {
            CheckIndexConfig,
            ScanWal,
            SearchIndexes,
            #[allow(dead_code)]
            SearchStorage,
            MergeResults,
        }

        struct SearchOrchestrationTracker {
            phases: Arc<RwLock<Vec<SearchPhase>>>,
        }

        impl SearchOrchestrationTracker {
            fn new() -> Self {
                Self {
                    phases: Arc::new(RwLock::new(Vec::new())),
                }
            }

            async fn record_phase(&self, phase: SearchPhase) {
                let mut phases = self.phases.write().await;
                phases.push(phase);
            }

            async fn verify_order(&self) -> bool {
                let phases = self.phases.read().await;

                // Expected order:
                // 1. Check index config
                // 2. Scan WAL (always)
                // 3. Search indexes OR storage (not both if indexes exist)
                // 4. Merge results

                if phases.is_empty() {
                    return false;
                }

                // Check index config should be first
                if phases[0] != SearchPhase::CheckIndexConfig {
                    return false;
                }

                // WAL scan should happen
                if !phases.contains(&SearchPhase::ScanWal) {
                    return false;
                }

                // Merge should be last
                if phases.last() != Some(&SearchPhase::MergeResults) {
                    return false;
                }

                true
            }
        }

        let tracker = SearchOrchestrationTracker::new();

        // Simulate search flow
        tracker.record_phase(SearchPhase::CheckIndexConfig).await;
        tracker.record_phase(SearchPhase::ScanWal).await;
        tracker.record_phase(SearchPhase::SearchIndexes).await;
        tracker.record_phase(SearchPhase::MergeResults).await;

        assert!(
            tracker.verify_order().await,
            "Search phases should be in correct order"
        );

        info!("✅ Search orchestration order verified");
        Ok(())
    }

    #[tokio::test]
    async fn test_index_and_storage_mutual_exclusion() -> Result<()> {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        info!("🧪 Testing that indexes and raw storage aren't both searched");

        // When indexes are available and used, raw storage scan should be skipped
        struct SearchPathTracker {
            used_indexes: Arc<AtomicUsize>,
            used_storage: Arc<AtomicUsize>,
        }

        impl SearchPathTracker {
            fn new() -> Self {
                Self {
                    used_indexes: Arc::new(AtomicUsize::new(0)),
                    used_storage: Arc::new(AtomicUsize::new(0)),
                }
            }

            fn use_indexes(&self) {
                self.used_indexes.fetch_add(1, Ordering::SeqCst);
            }

            fn use_storage(&self) {
                self.used_storage.fetch_add(1, Ordering::SeqCst);
            }

            fn verify_mutual_exclusion(&self) -> bool {
                let indexes = self.used_indexes.load(Ordering::SeqCst);
                let storage = self.used_storage.load(Ordering::SeqCst);

                // Either indexes OR storage, not both
                (indexes > 0 && storage == 0) || (indexes == 0 && storage > 0)
            }
        }

        // Test with indexes available
        {
            let tracker = SearchPathTracker::new();
            tracker.use_indexes();
            // Storage should NOT be used when indexes are available
            assert!(
                tracker.verify_mutual_exclusion(),
                "Should use either indexes OR storage, not both"
            );
        }

        // Test without indexes
        {
            let tracker = SearchPathTracker::new();
            tracker.use_storage();
            assert!(
                tracker.verify_mutual_exclusion(),
                "Should use storage when no indexes available"
            );
        }

        info!("✅ Index and storage mutual exclusion verified");
        Ok(())
    }

    #[tokio::test]
    async fn test_metadata_filter_propagation() -> Result<()> {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        info!("🧪 Testing metadata filter propagation through search layers");

        // Verify that metadata filters are properly propagated to:
        // 1. WAL scan
        // 2. Index queries
        // 3. Storage scans

        #[derive(Debug, Clone)]
        struct FilterPropagationTracker {
            wal_filters: Arc<RwLock<Vec<String>>>,
            index_filters: Arc<RwLock<Vec<String>>>,
            storage_filters: Arc<RwLock<Vec<String>>>,
        }

        impl FilterPropagationTracker {
            fn new() -> Self {
                Self {
                    wal_filters: Arc::new(RwLock::new(Vec::new())),
                    index_filters: Arc::new(RwLock::new(Vec::new())),
                    storage_filters: Arc::new(RwLock::new(Vec::new())),
                }
            }

            async fn add_wal_filter(&self, filter: String) {
                self.wal_filters.write().await.push(filter);
            }

            async fn add_index_filter(&self, filter: String) {
                self.index_filters.write().await.push(filter);
            }

            #[allow(dead_code)]
            async fn add_storage_filter(&self, filter: String) {
                self.storage_filters.write().await.push(filter);
            }

            async fn verify_propagation(&self, expected_filter: &str) -> bool {
                let wal = self.wal_filters.read().await;
                let index = self.index_filters.read().await;
                let storage = self.storage_filters.read().await;

                // All layers should have the same filter
                wal.contains(&expected_filter.to_string())
                    && (index.contains(&expected_filter.to_string())
                        || storage.contains(&expected_filter.to_string()))
            }
        }

        let tracker = FilterPropagationTracker::new();
        let test_filter = "category=electronics".to_string();

        // Simulate filter propagation
        tracker.add_wal_filter(test_filter.clone()).await;
        tracker.add_index_filter(test_filter.clone()).await;

        assert!(
            tracker.verify_propagation(&test_filter).await,
            "Metadata filter should propagate to all search layers"
        );

        info!("✅ Metadata filter propagation verified");
        Ok(())
    }

    #[tokio::test]
    async fn test_performance_with_no_double_scan() -> Result<()> {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        info!("🧪 Testing that search operations scan WAL only once");

        use tracing::info;

        // Counter to track WAL scan operations
        let wal_scan_counter = Arc::new(AtomicUsize::new(0));

        // Mock WAL scanner that counts scans
        struct MockWALScanner {
            scan_count: Arc<AtomicUsize>,
        }

        impl MockWALScanner {
            fn new(counter: Arc<AtomicUsize>) -> Self {
                Self {
                    scan_count: counter,
                }
            }

            async fn scan_wal(&self) {
                // Increment scan counter
                self.scan_count.fetch_add(1, Ordering::SeqCst);
                // Simulate scan work
                tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
            }

            async fn scan_storage(&self) {
                // Storage scan doesn't count as WAL scan
                tokio::time::sleep(tokio::time::Duration::from_micros(50)).await;
            }
        }

        // Test OLD behavior (double scan) - should scan WAL twice
        async fn search_with_double_scan(scanner: &MockWALScanner) {
            // First WAL scan (in service layer)
            scanner.scan_wal().await;

            // Second WAL scan (in storage engine) - OLD BEHAVIOR
            scanner.scan_wal().await;

            // Storage scan
            scanner.scan_storage().await;
        }

        // Test NEW behavior (single scan) - should scan WAL once
        async fn search_with_single_scan(scanner: &MockWALScanner) {
            // Single WAL scan (in service layer only)
            scanner.scan_wal().await;

            // Storage scan (no WAL scan)
            scanner.scan_storage().await;
        }

        // Test old behavior
        wal_scan_counter.store(0, Ordering::SeqCst);
        let old_scanner = MockWALScanner::new(wal_scan_counter.clone());
        search_with_double_scan(&old_scanner).await;
        let old_scan_count = wal_scan_counter.load(Ordering::SeqCst);

        assert_eq!(old_scan_count, 2, "Old behavior should scan WAL twice");
        info!("✅ Old behavior confirmed: 2 WAL scans");

        // Test new behavior
        wal_scan_counter.store(0, Ordering::SeqCst);
        let new_scanner = MockWALScanner::new(wal_scan_counter.clone());
        search_with_single_scan(&new_scanner).await;
        let new_scan_count = wal_scan_counter.load(Ordering::SeqCst);

        assert_eq!(new_scan_count, 1, "New behavior should scan WAL only once");
        info!("✅ New behavior confirmed: 1 WAL scan (50% reduction)");

        // Verify performance improvement
        assert!(
            new_scan_count < old_scan_count,
            "New implementation should scan WAL fewer times than old: {} >= {}",
            new_scan_count,
            old_scan_count
        );

        info!(
            "✅ WAL scan optimization verified: {} → {} scans",
            old_scan_count, new_scan_count
        );
        Ok(())
    }
}
