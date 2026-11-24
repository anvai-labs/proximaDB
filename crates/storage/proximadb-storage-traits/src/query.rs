//! Query context types for storage engine operations.
//!
//! This module provides types used by storage engines during query execution,
//! including row-level security predicates, search parameters, and metadata.

use proximadb_proto::proximadb_v1::{Collection, CollectionConfig};
pub use proximadb_quantization_types::QuantizationType;
use proximadb_search_types::block_prune::BlockPruneMode;
use proximadb_storage_ports::{
    collection_data_path_typed, typed_identity_from_storage_assignment, typed_path_identity,
};
use proximadb_tenant::{RbacTenantContext as TenantContext, UnifiedUserContext};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::{PerformanceTier, StorageFormatStrategy};

/// Whether a collection explicitly declares an AXIS index.
///
/// This is the single collection-level serving predicate shared by query,
/// insert, and flush paths. Storage-format and quantization tags affect the
/// representation written to PAX; they are not index declarations.
pub fn collection_declares_axis_index(config: Option<&CollectionConfig>) -> bool {
    config.is_some_and(|config| !config.index_configs.is_empty())
}

/// Predicate evaluated by the storage engine at scan time for row-level security.
///
/// Engines that implement `rls_record_filter()` apply this predicate inside their
/// scan iterator — no application-layer tenant filtering required.
/// See MULTIMODAL_OVERHAUL_SPEC_2026_05_08.adoc §8.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RlsRecordPredicate {
    /// If `Some`, only records whose `tenant_id` matches are returned.
    pub required_tenant_id: Option<String>,
    /// If `Some`, only records that contain this principal in `permitted_principals`
    /// (or have an empty `permitted_principals` list) are returned.
    pub required_principal: Option<String>,
}

impl RlsRecordPredicate {
    /// Return `true` when no filtering is needed (both fields are `None`).
    pub fn is_passthrough(&self) -> bool {
        self.required_tenant_id.is_none() && self.required_principal.is_none()
    }

    /// Evaluate this predicate against a canonical record.
    ///
    /// Engines should use this helper inside scan/search iterators so REST,
    /// gRPC, Arrow Flight, pgwire, and internal query paths share the same
    /// engine-level RLS semantics.
    pub fn matches_record(&self, record: &proximadb_records::ProximaRecord) -> bool {
        let tenant_matches = self
            .required_tenant_id
            .as_deref()
            .is_none_or(|tenant_id| record.matches_tenant(tenant_id));

        let principal_matches = self
            .required_principal
            .as_deref()
            .is_none_or(|principal| record.is_accessible_by(principal));

        tenant_matches && principal_matches
    }
}

/// Search context for STORAGE ENGINES - bundles immutable references to search parameters
/// and collection configuration for zero-copy access during search operations.
///
/// IMPORTANT: This is the STORAGE LAYER context. Do not confuse with:
/// - `core::search::SearchPlan` - Used for search planning/optimization
/// - `core::service_types::SearchRequest` - Used for API request representation
///
/// Used by: Storage engines (SST, NOVA, NOVA, SWIFT, RAPTOR)
/// Created by: VectorOperationsService.execute_search_internal()
///
/// Design principles:
/// - Immutable: All references are read-only during search
/// - Zero-copy: Uses Arc for shared ownership without cloning
/// - Cache-friendly: Collection comes directly from cache as Arc
/// - Extensible: Additional context can be added as needed
#[derive(Debug, Clone)]
pub struct StorageQueryContext {
    /// Original search parameters (immutable reference)
    pub search_params: Arc<proximadb_search_types::search_params::SearchParams>,

    /// Collection configuration from cache (immutable reference)
    /// Contains storage_assignment with storage URL
    pub collection: Arc<Collection>,

    /// Additional context that might be needed during search
    /// (can be extended without breaking existing code)
    pub metadata: StorageQueryMetadata,

    /// User context for RBAC authorization checks
    /// Optional for backward compatibility with existing code
    pub user_context: Option<UnifiedUserContext>,

    /// Tenant context for multi-tenant operations
    /// Optional for backward compatibility with existing code
    pub tenant_context: Option<TenantContext>,
}

/// Additional metadata for storage query context.
/// Contains all information storage engines need - no additional cache lookups required.
#[derive(Debug, Clone, Default)]
pub struct StorageQueryMetadata {
    /// Collection ID extracted for convenience
    pub collection_id: String,

    /// Whether this search should use AXIS indexes
    pub use_axis_indexes: bool,

    /// Whether progressive quantization is available
    pub has_quantization: bool,

    /// Dimension of vectors in this collection
    pub dimension: usize,

    /// Distance metric for the collection
    pub distance_metric: proximadb_distance_kernel::DistanceMetric,

    /// Storage engine strategy for this collection
    pub storage_strategy: StorageFormatStrategy,

    /// Base storage path for this collection (extracted from storage_assignment)
    pub storage_path: String,

    /// Parsed quantization configuration for progressive search
    pub quantization_config: Option<ParsedQuantizationConfig>,

    /// Collection size estimates for strategy selection
    pub estimated_vector_count: u64,
    pub estimated_size_bytes: u64,

    /// Performance hints for engines
    pub performance_tier: PerformanceTier,
    pub compression_enabled: bool,
    pub quantization_enabled: bool,
}

/// Parsed quantization configuration for efficient progressive search.
#[derive(Debug, Clone)]
pub struct ParsedQuantizationConfig {
    /// Strategy being used (SmartDefaults, CustomLevels, etc.)
    pub strategy: proximadb_proto::proximadb_v1::quantization_config::Strategy,

    /// Whether progressive search is enabled
    pub progressive_search_enabled: bool,

    /// Ordered quantization levels for progressive refinement
    pub progressive_levels: Vec<QuantizationLevel>,

    /// Search stage selectivity thresholds
    pub binary_filter_selectivity: f32,
    pub int8_ranking_selectivity: f32,
    pub pq_ranking_selectivity: f32,

    /// Quality and performance settings
    pub quality_threshold: f32,
    pub training_sample_size: i32,
    pub enable_simd_acceleration: bool,
    pub optimize_for_storage: bool,
    pub optimize_for_memory: bool,
}

// Re-export foundation quantization types for backward compatibility
// TODO: Migrate all uses to FoundationQuantizationType (Phase 2.2)

/// Individual quantization level for progressive search.
#[derive(Debug, Clone)]
pub struct QuantizationLevel {
    /// Level identifier (e.g., "binary", "int8", "pq8")
    pub level_id: String,

    /// Quantization type (using foundation type)
    pub quantization_type: QuantizationType,

    /// Bits per element
    pub bits: i32,

    /// Search priority (0 = first filter)
    pub search_priority: i32,

    /// PQ-specific settings
    pub num_subvectors: Option<i32>,

    /// Minimum recall for this level
    pub min_recall: f32,
}

// NOTE: QuantizationType is now re-exported from the proximadb-quantization-types
// foundation crate. The local QuantizationLevel struct remains the parsed
// progressive-search level shape used by storage engines.
//
// Legacy migration:
// - QuantizationType::Uniform → QuantizationType::None
// - All other variants map directly to foundation types

impl StorageQueryContext {
    /// Parse quantization config into ready-to-use format for progressive search.
    ///
    /// Note: this storage-traits layer only *parses* the supplied config. Smart
    /// default generation (which depends on the vector modality crate) is owned
    /// by the higher layer that constructs the collection config — see
    /// `services/collection/manager.rs`, which populates `custom_levels` at
    /// collection-create time via the smart-defaults generator. When a config
    /// reaches here with no `custom_levels`, there are simply no progressive
    /// levels configured (the consumer never second-guesses the producer).
    fn parse_quantization_config(
        quant_config: &proximadb_proto::proximadb_v1::QuantizationConfig,
        _dimension: usize,
    ) -> Option<ParsedQuantizationConfig> {
        if !quant_config.enabled.unwrap_or(false) {
            return None;
        }

        // Parse the progressive levels supplied by the config producer.
        // Empty custom_levels ⇒ no progressive levels (no modality fallback here).
        let progressive_levels = Self::parse_proto_levels(&quant_config.custom_levels);

        Some(ParsedQuantizationConfig {
            strategy: quant_config.strategy(),
            progressive_search_enabled: quant_config.enable_progressive_search.unwrap_or(false),
            progressive_levels,
            binary_filter_selectivity: quant_config.binary_filter_selectivity.unwrap_or(0.3),
            int8_ranking_selectivity: quant_config.int8_ranking_selectivity.unwrap_or(0.1),
            pq_ranking_selectivity: quant_config.pq_ranking_selectivity.unwrap_or(0.05),
            quality_threshold: quant_config.quality_threshold.unwrap_or(0.95),
            training_sample_size: quant_config.training_sample_size.unwrap_or(10000) as i32,
            enable_simd_acceleration: quant_config.enable_simd_acceleration.unwrap_or(true),
            optimize_for_storage: quant_config.optimize_for_storage.unwrap_or(false),
            optimize_for_memory: quant_config.optimize_for_memory.unwrap_or(false),
        })
    }

    /// Parse proto levels into internal format.
    fn parse_proto_levels(
        proto_levels: &[proximadb_proto::proximadb_v1::QuantizationLevel],
    ) -> Vec<QuantizationLevel> {
        use proximadb_proto::proximadb_v1::quantization_level::QuantizationType as ProtoQuantType;

        let mut levels: Vec<_> = proto_levels
            .iter()
            .enumerate()
            .map(|(idx, level)| {
                let quantization_type = match level.r#type() {
                    ProtoQuantType::Binary => QuantizationType::Binary,
                    ProtoQuantType::Scalar => QuantizationType::Scalar,
                    ProtoQuantType::Product => QuantizationType::Product,
                    ProtoQuantType::Uniform => QuantizationType::None,
                    ProtoQuantType::None => QuantizationType::None,
                };

                QuantizationLevel {
                    level_id: level.level_id.clone(),
                    quantization_type,
                    bits: level.bits as i32,
                    search_priority: idx as i32,
                    num_subvectors: Some(level.num_subvectors as i32),
                    min_recall: 0.9,
                }
            })
            .collect();

        levels.sort_by_key(|l| l.search_priority);
        levels
    }

    /// Create a new search context from cached components.
    pub fn new(
        search_params: Arc<proximadb_search_types::search_params::SearchParams>,
        collection: Arc<Collection>,
    ) -> Self {
        let config = collection.config.as_ref();
        let storage_assignment = collection.storage_assignment.as_ref();

        let storage_strategy = config
            .and_then(|c| c.storage_engine)
            .and_then(|e| proximadb_proto::proximadb_v1::StorageEngine::try_from(e).ok())
            .map_or(StorageFormatStrategy::Sst, |engine| match engine {
                proximadb_proto::proximadb_v1::StorageEngine::Viper => StorageFormatStrategy::Viper,
                proximadb_proto::proximadb_v1::StorageEngine::Sst => StorageFormatStrategy::Sst,
                proximadb_proto::proximadb_v1::StorageEngine::Nova => StorageFormatStrategy::Nova,
                proximadb_proto::proximadb_v1::StorageEngine::Helix => StorageFormatStrategy::Helix,
                proximadb_proto::proximadb_v1::StorageEngine::Swift => StorageFormatStrategy::Swift,
                proximadb_proto::proximadb_v1::StorageEngine::Raptor => {
                    StorageFormatStrategy::Raptor
                }
                proximadb_proto::proximadb_v1::StorageEngine::Tst => {
                    StorageFormatStrategy::TimeSeries
                }
                _ => StorageFormatStrategy::Sst,
            });

        let mut adjusted_params = (*search_params).clone();
        if matches!(
            storage_strategy,
            StorageFormatStrategy::Nova | StorageFormatStrategy::Raptor
        ) {
            adjusted_params.block_prune.force_exact = true;
            adjusted_params.block_prune.mode = BlockPruneMode::Sqrt;
            adjusted_params.block_prune.ratio = 0.0;
            adjusted_params.block_prune.min_keep = 0;
            adjusted_params.block_prune.max_keep = 0;
        }

        let metadata = StorageQueryMetadata {
            collection_id: collection.id.clone(),
            // Index declarations are authoritative. A storage-format or
            // quantization tag changes representation, not serving policy.
            use_axis_indexes: collection_declares_axis_index(config),
            has_quantization: config.and_then(|c| c.quantization.as_ref()).is_some(),
            dimension: config.map_or(0, |c| c.dimension as usize),
            distance_metric: config
                .and_then(|c| c.distance_metric)
                .and_then(|metric| {
                    proximadb_proto::proximadb_v1::DistanceMetric::try_from(metric).ok()
                })
                .map_or(
                    proximadb_distance_kernel::DistanceMetric::Cosine,
                    |metric| match metric {
                        proximadb_proto::proximadb_v1::DistanceMetric::Unspecified
                        | proximadb_proto::proximadb_v1::DistanceMetric::Cosine => {
                            proximadb_distance_kernel::DistanceMetric::Cosine
                        }
                        proximadb_proto::proximadb_v1::DistanceMetric::Euclidean => {
                            proximadb_distance_kernel::DistanceMetric::Euclidean
                        }
                        proximadb_proto::proximadb_v1::DistanceMetric::DotProduct => {
                            proximadb_distance_kernel::DistanceMetric::DotProduct
                        }
                        proximadb_proto::proximadb_v1::DistanceMetric::Hamming => {
                            proximadb_distance_kernel::DistanceMetric::Hamming
                        }
                        proximadb_proto::proximadb_v1::DistanceMetric::Manhattan => {
                            proximadb_distance_kernel::DistanceMetric::Manhattan
                        }
                        proximadb_proto::proximadb_v1::DistanceMetric::Jaccard => {
                            proximadb_distance_kernel::DistanceMetric::Jaccard
                        }
                        proximadb_proto::proximadb_v1::DistanceMetric::Angular => {
                            proximadb_distance_kernel::DistanceMetric::Angular
                        }
                        proximadb_proto::proximadb_v1::DistanceMetric::Chebyshev => {
                            proximadb_distance_kernel::DistanceMetric::Chebyshev
                        }
                        proximadb_proto::proximadb_v1::DistanceMetric::Canberra => {
                            proximadb_distance_kernel::DistanceMetric::Canberra
                        }
                        proximadb_proto::proximadb_v1::DistanceMetric::Minkowski => {
                            proximadb_distance_kernel::DistanceMetric::Minkowski
                        }
                        proximadb_proto::proximadb_v1::DistanceMetric::BrayCurtis => {
                            proximadb_distance_kernel::DistanceMetric::BrayCurtis
                        }
                        proximadb_proto::proximadb_v1::DistanceMetric::Hellinger => {
                            proximadb_distance_kernel::DistanceMetric::Hellinger
                        }
                        proximadb_proto::proximadb_v1::DistanceMetric::Custom => {
                            proximadb_distance_kernel::DistanceMetric::Custom
                        }
                    },
                ),
            storage_strategy,
            storage_path: storage_assignment
                .map_or_else(|| "./data".to_string(), |sa| sa.base_location.clone()),
            estimated_vector_count: 0,
            estimated_size_bytes: 0,
            performance_tier: PerformanceTier::Warm,
            compression_enabled: config
                .and_then(|c| c.storage_config.as_ref())
                .is_some_and(|s| s.compression.unwrap_or(0) != 0),
            quantization_enabled: config
                .and_then(|c| c.quantization.as_ref())
                .is_some_and(|_| true),
            quantization_config: config.and_then(|c| c.quantization.as_ref()).and_then(|qc| {
                config
                    .map(|c| c.dimension)
                    .and_then(|dim| Self::parse_quantization_config(qc, dim as usize))
            }),
        };

        Self {
            search_params: Arc::new(adjusted_params),
            collection,
            metadata,
            user_context: None,
            tenant_context: None,
        }
    }

    /// TD-CACHE-3 S1: attach the requesting tenant so engine-side consumers
    /// (per-tenant cache fair-share, RLS, billing labels) receive the real
    /// tenant instead of `None`. Builder-style; absent ⇒ legacy behavior.
    pub fn with_tenant_context(mut self, tenant_context: Option<TenantContext>) -> Self {
        self.tenant_context = tenant_context;
        self
    }

    /// Get the query vector (convenience method).
    pub fn query_vector(&self) -> Option<&[f32]> {
        if let Some(ref vector) = self.search_params.vector {
            return Some(vector.as_slice());
        }

        self.search_params
            .query_vectors
            .as_ref()
            .and_then(|vecs| vecs.first())
            .map(|v| v.as_slice())
    }

    /// Get top_k value with fallback to default.
    pub fn top_k(&self) -> usize {
        self.search_params.top_k.unwrap_or(10) as usize
    }

    /// Get distance metric (pre-computed from collection config).
    pub fn distance_metric(&self) -> proximadb_distance_kernel::DistanceMetric {
        self.search_params
            .distance_metric
            .unwrap_or(self.metadata.distance_metric)
    }

    /// Get dimension from metadata (pre-computed).
    pub fn dimension(&self) -> usize {
        self.metadata.dimension
    }

    /// Check if progressive search is enabled.
    pub fn is_progressive_search_enabled(&self) -> bool {
        self.metadata
            .quantization_config
            .as_ref()
            .is_some_and(|qc| qc.progressive_search_enabled)
    }

    /// Get progressive quantization levels ordered by search priority.
    pub fn get_progressive_levels(&self) -> Option<&[QuantizationLevel]> {
        self.metadata
            .quantization_config
            .as_ref()
            .map(|qc| qc.progressive_levels.as_slice())
    }

    /// Get binary filter selectivity for progressive search.
    pub fn binary_filter_selectivity(&self) -> f32 {
        self.metadata
            .quantization_config
            .as_ref()
            .map_or(0.1, |qc| qc.binary_filter_selectivity)
    }

    /// Check if SIMD acceleration should be used.
    pub fn use_simd_acceleration(&self) -> bool {
        self.metadata
            .quantization_config
            .as_ref()
            .is_none_or(|qc| qc.enable_simd_acceleration)
    }

    /// Get the parsed quantization config.
    pub fn quantization_config(&self) -> Option<&ParsedQuantizationConfig> {
        self.metadata.quantization_config.as_ref()
    }

    /// Check if quantization is enabled (pre-computed).
    pub fn has_quantization(&self) -> bool {
        self.metadata.has_quantization
    }

    /// Get storage path (pre-computed from storage assignment).
    pub fn storage_path(&self) -> &str {
        &self.metadata.storage_path
    }

    /// Get storage strategy (pre-computed).
    pub fn storage_strategy(&self) -> StorageFormatStrategy {
        self.metadata.storage_strategy
    }

    /// Get performance tier hint (pre-computed).
    pub fn performance_tier(&self) -> PerformanceTier {
        self.metadata.performance_tier.clone()
    }

    /// Get collection size estimates (pre-computed).
    pub fn estimated_vector_count(&self) -> u64 {
        self.metadata.estimated_vector_count
    }

    /// Get estimated collection size in bytes (pre-computed).
    pub fn estimated_size_bytes(&self) -> u64 {
        self.metadata.estimated_size_bytes
    }

    /// Check if compression is enabled (pre-computed).
    pub fn compression_enabled(&self) -> bool {
        self.metadata.compression_enabled
    }

    /// Check if quantization is enabled (pre-computed).
    pub fn quantization_enabled(&self) -> bool {
        self.metadata.quantization_enabled
    }

    /// Get collection ID from the collection object directly.
    pub fn collection_id(&self) -> &str {
        &self.collection.id
    }

    /// Get storage URL from collection's storage assignment.
    pub fn storage_url(&self) -> Option<&str> {
        self.collection
            .storage_assignment
            .as_ref()
            .map(|sa| sa.base_location.as_str())
    }

    /// Get collection-specific storage path (ADR-031 Phase 4d: typed when the
    /// collection carries a typed identity triple on its storage_assignment;
    /// legacy `{base}/{coll_id}/data` otherwise — mixed-read-safe).
    pub fn collection_storage_path(&self) -> Option<String> {
        self.storage_url().map(|base| {
            let identity =
                typed_identity_from_storage_assignment(self.collection.storage_assignment.as_ref());
            collection_data_path_typed(base, self.collection_id(), typed_path_identity(identity))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rls_predicate_default_is_passthrough() {
        let pred = RlsRecordPredicate::default();
        assert!(
            pred.is_passthrough(),
            "default predicate must not filter anything"
        );
    }

    #[test]
    fn test_rls_predicate_tenant_id_not_passthrough() {
        let pred = RlsRecordPredicate {
            required_tenant_id: Some("acme".to_string()),
            required_principal: None,
        };
        assert!(!pred.is_passthrough());
        assert_eq!(pred.required_tenant_id.as_deref(), Some("acme"));
    }

    #[test]
    fn test_rls_predicate_principal_not_passthrough() {
        let pred = RlsRecordPredicate {
            required_tenant_id: None,
            required_principal: Some("alice".to_string()),
        };
        assert!(!pred.is_passthrough());
    }

    #[test]
    fn test_rls_predicate_both_set() {
        let pred = RlsRecordPredicate {
            required_tenant_id: Some("acme".to_string()),
            required_principal: Some("alice".to_string()),
        };
        assert!(!pred.is_passthrough());
        assert_eq!(pred.required_tenant_id.as_deref(), Some("acme"));
        assert_eq!(pred.required_principal.as_deref(), Some("alice"));
    }

    #[test]
    fn test_rls_predicate_matches_canonical_record() {
        let mut record = proximadb_records::ProximaRecord {
            tenant_id: "acme".to_string(),
            permitted_principals: vec!["alice".to_string()],
            ..Default::default()
        };

        let pred = RlsRecordPredicate {
            required_tenant_id: Some("acme".to_string()),
            required_principal: Some("alice".to_string()),
        };
        assert!(pred.matches_record(&record));

        record.tenant_id = "other".to_string();
        assert!(!pred.matches_record(&record));
    }

    #[test]
    fn test_rls_predicate_open_principals_match_any_user() {
        let record = proximadb_records::ProximaRecord {
            tenant_id: "acme".to_string(),
            permitted_principals: Vec::new(),
            ..Default::default()
        };
        let pred = RlsRecordPredicate {
            required_tenant_id: Some("acme".to_string()),
            required_principal: Some("bob".to_string()),
        };

        assert!(pred.matches_record(&record));
    }

    #[test]
    fn test_storage_query_metadata_default() {
        let metadata = StorageQueryMetadata::default();
        assert_eq!(metadata.collection_id, "");
        assert!(!metadata.use_axis_indexes);
        assert!(!metadata.has_quantization);
    }

    #[test]
    fn query_context_requires_an_explicit_index_config_for_axis() {
        use proximadb_proto::proximadb_v1::{CollectionConfig, IndexConfig};

        let format_only = Arc::new(Collection {
            config: Some(CollectionConfig {
                tags: vec!["pax_vector_format:off".to_string()],
                ..Default::default()
            }),
            ..Default::default()
        });
        let format_only_ctx = StorageQueryContext::new(
            Arc::new(proximadb_search_types::search_params::SearchParams::default()),
            format_only,
        );
        assert!(!format_only_ctx.metadata.use_axis_indexes);

        let indexed = Arc::new(Collection {
            config: Some(CollectionConfig {
                index_configs: vec![IndexConfig::default()],
                ..Default::default()
            }),
            ..Default::default()
        });
        let indexed_ctx = StorageQueryContext::new(
            Arc::new(proximadb_search_types::search_params::SearchParams::default()),
            indexed,
        );
        assert!(indexed_ctx.metadata.use_axis_indexes);
    }

    #[test]
    fn collection_axis_declaration_fails_closed() {
        assert!(!collection_declares_axis_index(None));
        assert!(!collection_declares_axis_index(Some(
            &CollectionConfig::default()
        )));
        assert!(collection_declares_axis_index(Some(&CollectionConfig {
            index_configs: vec![proximadb_proto::proximadb_v1::IndexConfig::default()],
            ..Default::default()
        })));
    }
}
