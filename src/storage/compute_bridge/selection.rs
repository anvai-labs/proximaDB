//! Quantization Engine Selection Logic
//!
//! Shared logic for intelligently selecting between persistent StorageQuantizationEngine
//! and stateless UnifiedQuantizationEngine based on operation context.

use crate::storage::traits::FlushParameters;
use tracing::debug;

/// Smart quantization engine selection logic
///
/// This module provides the shared decision logic used by all storage engines
/// to determine whether to use persistent collection-based quantization or
/// stateless ad-hoc quantization.
pub struct QuantizationSelector;

impl QuantizationSelector {
    /// Check if we should use persistent quantization for this operation (simple version)
    ///
    /// Returns true for operations that benefit from persistent quantization:
    /// - Large collections (>1000 vectors)
    /// - Flush operations (writes to persistent storage)
    /// - Frequent operations that benefit from cached codebooks
    pub fn should_use_persistent_quantization_simple(
        operation_context: &str,
        collection_size: Option<usize>,
    ) -> bool {
        match operation_context {
            "flush" | "compact" => true, // Write operations benefit from persistent codebooks
            "search" | "query" => {
                // Large collections benefit from persistent codebooks
                collection_size.is_some_and(|size| size > 1000)
            }
            _ => false, // Default to stateless for unknown operations
        }
    }

    /// Check if we should use persistent quantization for this operation
    ///
    /// Returns true when ALL conditions are met:
    /// - Operation has collection configuration
    /// - Quantization is explicitly enabled
    /// - Collection ID is provided
    ///
    /// This indicates a persistent collection-based operation that benefits
    /// from cached codebooks and collection-partitioned storage.
    pub fn should_use_persistent_quantization(params: &FlushParameters, engine_name: &str) -> bool {
        let has_collection_config = params.collection_config.is_some();
        // Check if quantization is enabled in collection config
        let quantization_enabled = params
            .collection_config
            .as_ref()
            .and_then(|collection| collection.config.as_ref())
            .and_then(|config| config.quantization.as_ref())
            .map(|q| q.enabled);
        let has_collection_id = params.collection_id.is_some();

        if has_collection_config
            && quantization_enabled.flatten().unwrap_or(false)
            && has_collection_id
        {
            debug!(
                "🎯 {}: Using persistent StorageQuantizationEngine for collection: {:?}",
                engine_name, params.collection_id
            );
            true
        } else {
            debug!(
                "🔄 {}: Using stateless UnifiedQuantizationEngine for ad-hoc query",
                engine_name
            );
            false
        }
    }
}
