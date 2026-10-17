//! Clean Background Maintenance Manager with Context-Based Operations
//!
//! This is the optimized version that eliminates redundant collection service calls
//! by using pre-computed BackgroundFlushContext.

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::{info, warn};

use super::WALConfig;
use crate::metrics::InternalMetricsUpdater;
use crate::storage::background_flush_context::BackgroundFlushContext;
use crate::storage::traits::UnifiedStorageFormat;

/// Background task status enumeration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackgroundTaskStatus {
    Idle,
    Flushing,
    Compacting,
    FlushAndCompact,
}

/// Background maintenance statistics
#[derive(Debug, Clone)]
pub struct BackgroundMaintenanceStats {
    pub flush_operations_completed: u64,
    pub flush_operations_skipped: u64,
    pub compaction_operations_completed: u64,
    pub compaction_operations_failed: u64,
    pub model_training_skipped_small: u64,
    pub average_flush_duration_ms: f64,
    pub average_compaction_duration_ms: f64,
}

impl Default for BackgroundMaintenanceStats {
    fn default() -> Self {
        Self {
            flush_operations_completed: 0,
            flush_operations_skipped: 0,
            compaction_operations_completed: 0,
            compaction_operations_failed: 0,
            model_training_skipped_small: 0,
            average_flush_duration_ms: 0.0,
            average_compaction_duration_ms: 0.0,
        }
    }
}

/// Clean Background Maintenance Manager - Optimized with Context-Based Operations
pub struct BackgroundMaintenanceManager {
    #[allow(dead_code)]
    config: Arc<WALConfig>,
    collection_status: Arc<RwLock<HashMap<String, BackgroundTaskStatus>>>,
    stats: Arc<Mutex<BackgroundMaintenanceStats>>,
    storage_engines: Arc<RwLock<HashMap<String, Arc<dyn UnifiedStorageFormat>>>>,
    /// Metrics updater for tracking compaction operations
    metrics_updater: Option<Arc<dyn InternalMetricsUpdater>>,
}

impl BackgroundMaintenanceManager {
    /// Create new background maintenance manager
    pub fn new(config: Arc<WALConfig>) -> Self {
        Self {
            config,
            collection_status: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(Mutex::new(BackgroundMaintenanceStats::default())),
            storage_engines: Arc::new(RwLock::new(HashMap::new())),
            metrics_updater: None, // Set via set_metrics_updater for dependency injection
        }
    }

    /// Register a storage engine for compaction delegation
    pub async fn register_storage_engine(
        &self,
        engine_name: &str,
        engine: Arc<dyn UnifiedStorageFormat>,
    ) -> Result<()> {
        let mut engines = self.storage_engines.write().await;
        engines.insert(engine_name.to_string(), engine);
        info!(
            "🏭 BackgroundManager: Registered {} storage engine for compaction delegation",
            engine_name
        );
        Ok(())
    }

    /// Set metrics updater for tracking compaction operations
    pub fn set_metrics_updater(&mut self, updater: Arc<dyn InternalMetricsUpdater>) {
        self.metrics_updater = Some(updater);
        info!("🔗 BackgroundManager: Metrics updater registered for compaction tracking");
    }

    /// 🚀 OPTIMIZED: Context-based compaction that eliminates collection service calls
    pub async fn execute_compaction_with_context(
        storage_engines: &Arc<RwLock<HashMap<String, Arc<dyn UnifiedStorageFormat>>>>,
        context: &BackgroundFlushContext,
        metrics_updater: Option<&Arc<dyn InternalMetricsUpdater>>,
    ) -> Result<Vec<String>> {
        info!(
            "🔄 [COMPACTION] Starting compaction for collection {} using {} engine (context-optimized)",
            context.collection_id,
            context.engine_name()
        );

        // 🚀 OPTIMIZATION: Use pre-computed engine name (no service calls needed!)
        let engine_name = context.engine_name();

        info!(
            "✅ CONTEXT_OPTIMIZED: Using pre-computed engine {} for collection {}",
            engine_name, context.collection_id
        );

        // Get storage engine for delegation using pre-computed engine type
        let engines = storage_engines.read().await;

        // Use the engine type from context instead of defaulting to VIPER
        let engine = if let Some(engine) = engines.get(engine_name) {
            info!(
                "🏭 [COMPACTION] Using {} storage engine for collection {}",
                engine_name, context.collection_id
            );
            engine.clone()
        } else {
            // Fallback to VIPER if the requested engine isn't available
            if let Some(viper_engine) = engines.get("VIPER") {
                warn!(
                    "⚠️ [COMPACTION] Requested engine '{}' not found, falling back to VIPER for collection {}",
                    engine_name, context.collection_id
                );
                viper_engine.clone()
            } else if let Some(sst_engine) = engines.get("SST") {
                warn!(
                    "⚠️ [COMPACTION] Neither '{}' nor 'viper' found, falling back to SST for collection {}",
                    engine_name, context.collection_id
                );
                sst_engine.clone()
            } else {
                warn!(
                    "⚠️ [COMPACTION] No storage engines registered, cannot perform compaction_info"
                );
                return Err(anyhow::anyhow!(
                    "No storage engines available for compaction_info"
                ));
            }
        };

        drop(engines); // Release the read lock

        // 🚀 OPTIMIZATION: Create compaction parameters with context metadata (no service calls!)
        // Use centralized helper method for consistent collection proto creation
        let collection_config = Some(context.to_collection_proto());

        let compaction_params = crate::storage::traits::CompactionParameters {
            collection_id: Some(context.collection_id.clone()),
            force: false,      // Background compaction is not forced
            synchronous: true, // Wait for completion
            // TD-READFID-2 review: tell the engine whether `collection_config` is
            // the real catalog meta (tags + `pax_vector_quant`) or a synthesized
            // stand-in, so it never resolves an output encoding from a guess.
            hints: std::collections::HashMap::from([(
                crate::storage::engines::sst::COLLECTION_META_COMPLETE_HINT.to_string(),
                serde_json::Value::Bool(context.collection_meta.is_some()),
            )]),
            timeout_ms: context.timeout_ms.or(Some(300_000)), // Use context timeout or 5 minute default
            estimated_input_size: 0,                          // Will be calculated by compaction
            priority: match context.priority {
                crate::storage::background_flush_context::OperationPriority::Low => {
                    crate::storage::traits::OperationPriority::Low
                }
                crate::storage::background_flush_context::OperationPriority::Normal => {
                    crate::storage::traits::OperationPriority::Medium
                }
                crate::storage::background_flush_context::OperationPriority::High => {
                    crate::storage::traits::OperationPriority::High
                }
                crate::storage::background_flush_context::OperationPriority::Critical => {
                    crate::storage::traits::OperationPriority::High
                } // Map to High since Critical may not exist
            },
            collection_config, // Provide collection config with storage assignment
        };

        info!(
            "📋 [COMPACTION] Delegating to {} engine: do_compact({})",
            engine.format_name(),
            context.collection_id
        );

        // Execute compaction via storage engine
        match engine.do_compact(&compaction_params).await {
            Ok(result) => {
                if result.success {
                    info!(
                        "✅ [COMPACTION] {} compaction completed for collection {}: {} entries processed, {} files {} → {}",
                        engine.format_name(),
                        context.collection_id,
                        result.entries_processed.unwrap_or(0),
                        result.input_files.unwrap_or(0),
                        result.output_files.unwrap_or(0),
                        result.duration_ms.unwrap_or(0)
                    );

                    // 📊 METRICS: Record compaction operation metrics (non-blocking)
                    if let Some(metrics) = metrics_updater {
                        let _ = metrics
                            .record_compaction(
                                &context.collection_id,
                                crate::metrics::CompactionMetricsUpdate {
                                    files_before: result.input_files.unwrap_or(0) as i32,
                                    files_after: result.output_files.unwrap_or(0) as i32,
                                    bytes_before: result.bytes_read.unwrap_or(0) as i64, // Use bytes_read instead
                                    bytes_after: result.bytes_written.unwrap_or(0) as i64, // Use bytes_written instead
                                    duration_ms: result.duration_ms.unwrap_or(0) as i64,
                                    timestamp: chrono::Utc::now().timestamp_millis(),
                                },
                            )
                            .await;
                        info!(
                            "📊 Recorded compaction metrics for collection {}",
                            context.collection_id
                        );
                    }

                    // Return file list for compatibility - for VIPER this would be the compacted files
                    // Since the UnifiedStorageFormat doesn't return file paths, we'll return a placeholder
                    Ok(vec![format!(
                        "compacted_collection_{}_{}files",
                        context.collection_id,
                        result.output_files.unwrap_or(0)
                    )])
                } else {
                    warn!(
                        "❌ [COMPACTION] {} compaction failed for collection {}",
                        engine.format_name(),
                        context.collection_id
                    );
                    Err(anyhow::anyhow!("Storage engine compaction failed"))
                }
            }
            Err(e) => {
                warn!(
                    "❌ [COMPACTION] {} compaction error for collection {}: {}",
                    engine.format_name(),
                    context.collection_id,
                    e
                );
                Err(e)
            }
        }
    }

    /// Get collection status
    pub async fn get_collection_status(&self, collection_id: &str) -> BackgroundTaskStatus {
        let status_map = self.collection_status.read().await;
        status_map
            .get(collection_id)
            .cloned()
            .unwrap_or(BackgroundTaskStatus::Idle)
    }

    /// Get background maintenance statistics
    pub async fn get_stats(&self) -> BackgroundMaintenanceStats {
        let stats = self.stats.lock().await;
        stats.clone()
    }

    /// Check if there are any active background operations
    pub async fn has_active_operations(&self) -> bool {
        let status_map = self.collection_status.read().await;
        status_map
            .values()
            .any(|status| *status != BackgroundTaskStatus::Idle)
    }
}
