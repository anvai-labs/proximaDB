/*
 * Copyright 2025 ProximaDB
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

//! # Capability Registry Implementation
//!
//! This module provides the core capability registry types and implementations.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, RwLock};

/// Represents a single capability that a storage engine or query processor may support.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Capability {
    // Query Types
    VectorSearch,
    GraphQuery,
    DocumentQuery,
    LogQuery,
    MetricsQuery,
    TimeSeriesQuery,
    EventSourcingQuery,

    // Operations
    Scan,
    Filter,
    Project,
    Join,
    Aggregate,
    Sort,
    Limit,

    // Features
    PredicatePushdown,
    Quantization,
    WALRecovery,
    Replication,
    Sharding,
    MultiRegion,
    ColumnarAnalytics,
    RowGroupPruning,
    BloomFilter,
    CachedQueries,

    // Index Types
    HNSWIndex,
    IVFIndex,
    AnnoyIndex,
    LSHIndex,
    DiskANNIndex,
    SparseVectorIndex,

    // Vector Operations
    CosineDistance,
    EuclideanDistance,
    DotProduct,
    HybridSearch,

    // Graph Operations
    GraphTraversal,
    PatternMatching,
    CypherFunctions,
    GraphAggregation,

    // Document Operations
    FullTextSearch,
    JSONPathQueries,
    DocumentAggregation,

    // Observability Operations
    PromQLQuery,
    LogAggregation,
    MetricAggregation,
    SIEMIntegration,

    // Distributed Operations
    FederatedQuery,
    CrossModelJoin,
    DistributedTransaction,
    ConsensusProtocol,

    // Streaming Operations
    ChangeDataCapture,
    RealTimeStreaming,
    WebSocketStreaming,
}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Capability::VectorSearch => write!(f, "VectorSearch"),
            Capability::GraphQuery => write!(f, "GraphQuery"),
            Capability::DocumentQuery => write!(f, "DocumentQuery"),
            Capability::LogQuery => write!(f, "LogQuery"),
            Capability::MetricsQuery => write!(f, "MetricsQuery"),
            Capability::TimeSeriesQuery => write!(f, "TimeSeriesQuery"),
            Capability::EventSourcingQuery => write!(f, "EventSourcingQuery"),
            Capability::Scan => write!(f, "Scan"),
            Capability::Filter => write!(f, "Filter"),
            Capability::Project => write!(f, "Project"),
            Capability::Join => write!(f, "Join"),
            Capability::Aggregate => write!(f, "Aggregate"),
            Capability::Sort => write!(f, "Sort"),
            Capability::Limit => write!(f, "Limit"),
            Capability::PredicatePushdown => write!(f, "PredicatePushdown"),
            Capability::Quantization => write!(f, "Quantization"),
            Capability::WALRecovery => write!(f, "WALRecovery"),
            Capability::Replication => write!(f, "Replication"),
            Capability::Sharding => write!(f, "Sharding"),
            Capability::MultiRegion => write!(f, "MultiRegion"),
            Capability::ColumnarAnalytics => write!(f, "ColumnarAnalytics"),
            Capability::RowGroupPruning => write!(f, "RowGroupPruning"),
            Capability::BloomFilter => write!(f, "BloomFilter"),
            Capability::CachedQueries => write!(f, "CachedQueries"),
            Capability::HNSWIndex => write!(f, "HNSWIndex"),
            Capability::IVFIndex => write!(f, "IVFIndex"),
            Capability::AnnoyIndex => write!(f, "AnnoyIndex"),
            Capability::LSHIndex => write!(f, "LSHIndex"),
            Capability::DiskANNIndex => write!(f, "DiskANNIndex"),
            Capability::SparseVectorIndex => write!(f, "SparseVectorIndex"),
            Capability::CosineDistance => write!(f, "CosineDistance"),
            Capability::EuclideanDistance => write!(f, "EuclideanDistance"),
            Capability::DotProduct => write!(f, "DotProduct"),
            Capability::HybridSearch => write!(f, "HybridSearch"),
            Capability::GraphTraversal => write!(f, "GraphTraversal"),
            Capability::PatternMatching => write!(f, "PatternMatching"),
            Capability::CypherFunctions => write!(f, "CypherFunctions"),
            Capability::GraphAggregation => write!(f, "GraphAggregation"),
            Capability::FullTextSearch => write!(f, "FullTextSearch"),
            Capability::JSONPathQueries => write!(f, "JSONPathQueries"),
            Capability::DocumentAggregation => write!(f, "DocumentAggregation"),
            Capability::PromQLQuery => write!(f, "PromQLQuery"),
            Capability::LogAggregation => write!(f, "LogAggregation"),
            Capability::MetricAggregation => write!(f, "MetricAggregation"),
            Capability::SIEMIntegration => write!(f, "SIEMIntegration"),
            Capability::FederatedQuery => write!(f, "FederatedQuery"),
            Capability::CrossModelJoin => write!(f, "CrossModelJoin"),
            Capability::DistributedTransaction => write!(f, "DistributedTransaction"),
            Capability::ConsensusProtocol => write!(f, "ConsensusProtocol"),
            Capability::ChangeDataCapture => write!(f, "ChangeDataCapture"),
            Capability::RealTimeStreaming => write!(f, "RealTimeStreaming"),
            Capability::WebSocketStreaming => write!(f, "WebSocketStreaming"),
        }
    }
}

/// A set of capabilities with efficient set operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilitySet {
    capabilities: HashSet<Capability>,
}

impl CapabilitySet {
    /// Create a new empty capability set.
    pub fn new() -> Self {
        Self {
            capabilities: HashSet::new(),
        }
    }

    /// Create a capability set from an array of capabilities.
    pub fn from_capabilities(capabilities: &[Capability]) -> Self {
        Self {
            capabilities: capabilities.iter().cloned().collect(),
        }
    }

    /// Add a capability to the set.
    pub fn add(&mut self, capability: Capability) {
        self.capabilities.insert(capability);
    }

    /// Check if the set contains all capabilities from another set.
    pub fn contains(&self, other: &CapabilitySet) -> bool {
        other.capabilities.is_subset(&self.capabilities)
    }

    /// Check if the set contains a specific capability.
    pub fn contains_capability(&self, capability: &Capability) -> bool {
        self.capabilities.contains(capability)
    }

    /// Check if the set intersects with another set.
    pub fn intersects(&self, other: &CapabilitySet) -> bool {
        !self.capabilities.is_disjoint(&other.capabilities)
    }

    /// Return the union of two capability sets.
    pub fn union(&self, other: &CapabilitySet) -> CapabilitySet {
        CapabilitySet {
            capabilities: self
                .capabilities
                .union(&other.capabilities)
                .cloned()
                .collect(),
        }
    }

    /// Return the difference of two capability sets.
    pub fn difference(&self, other: &CapabilitySet) -> CapabilitySet {
        CapabilitySet {
            capabilities: self
                .capabilities
                .difference(&other.capabilities)
                .cloned()
                .collect(),
        }
    }

    /// Return the intersection of two capability sets.
    pub fn intersection(&self, other: &CapabilitySet) -> CapabilitySet {
        CapabilitySet {
            capabilities: self
                .capabilities
                .intersection(&other.capabilities)
                .cloned()
                .collect(),
        }
    }

    /// Get the number of capabilities in the set.
    pub fn len(&self) -> usize {
        self.capabilities.len()
    }

    /// Check if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.capabilities.is_empty()
    }

    /// Get an iterator over the capabilities.
    pub fn iter(&self) -> impl Iterator<Item = &Capability> {
        self.capabilities.iter()
    }

    /// Convert to a vector for serialization.
    pub fn to_vec(&self) -> Vec<Capability> {
        self.capabilities.iter().cloned().collect()
    }
}

impl Default for CapabilitySet {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CapabilitySet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let caps: Vec<String> = self.iter().map(|c| c.to_string()).collect();
        write!(f, "[{}]", caps.join(", "))
    }
}

/// Error type for capability checking failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityCheckError {
    /// A required capability is not supported.
    UnsupportedCapability {
        capability: String,
        available_alternatives: Vec<String>,
    },

    /// Multiple required capabilities are missing.
    MultipleUnsupportedCapabilities {
        missing_capabilities: Vec<String>,
        available_alternatives: Vec<String>,
    },
}

impl fmt::Display for CapabilityCheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CapabilityCheckError::UnsupportedCapability {
                capability,
                available_alternatives,
            } => {
                write!(f, "Unsupported capability: {}", capability)?;
                if !available_alternatives.is_empty() {
                    write!(
                        f,
                        ". Available alternatives: {}",
                        available_alternatives.join(", ")
                    )?;
                }
                Ok(())
            }
            CapabilityCheckError::MultipleUnsupportedCapabilities {
                missing_capabilities,
                available_alternatives,
            } => {
                write!(
                    f,
                    "Missing capabilities: {}",
                    missing_capabilities.join(", ")
                )?;
                if !available_alternatives.is_empty() {
                    write!(
                        f,
                        ". Available alternatives: {}",
                        available_alternatives.join(", ")
                    )?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for CapabilityCheckError {}

/// Global capability registry for tracking engine capabilities.
#[derive(Clone)]
pub struct CapabilityRegistry {
    /// Map from engine/provider name to their capabilities.
    capabilities: Arc<RwLock<HashMap<String, CapabilitySet>>>,
}

impl CapabilityRegistry {
    /// Create a new empty capability registry.
    pub fn new() -> Self {
        Self {
            capabilities: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register capabilities for an engine or provider.
    #[expect(
        clippy::expect_used,
        reason = "RwLock poisoning is unrecoverable; propagating would hide the root cause"
    )]
    pub fn register_capabilities(&self, name: &str, capabilities: CapabilitySet) {
        let mut caps = self
            .capabilities
            .write()
            .expect("RwLock write lock poisoned - capability registry unavailable");
        caps.insert(name.to_string(), capabilities);
    }

    /// Get capabilities for an engine or provider.
    #[expect(
        clippy::expect_used,
        reason = "RwLock poisoning is unrecoverable; propagating would hide the root cause"
    )]
    pub fn get_capabilities(&self, name: &str) -> Option<CapabilitySet> {
        let caps = self
            .capabilities
            .read()
            .expect("RwLock read lock poisoned - capability registry unavailable");
        caps.get(name).cloned()
    }

    /// Check if required capabilities are supported.
    pub fn check_support(
        &self,
        engine_name: &str,
        required: &CapabilitySet,
    ) -> Result<(), CapabilityCheckError> {
        let available = self.get_capabilities(engine_name).ok_or_else(|| {
            CapabilityCheckError::UnsupportedCapability {
                capability: engine_name.to_string(),
                available_alternatives: vec![],
            }
        })?;

        if available.contains(required) {
            Ok(())
        } else {
            let missing = required.difference(&available);
            let missing_caps: Vec<String> = missing.iter().map(|c| c.to_string()).collect();
            let available_caps: Vec<String> = available.iter().map(|c| c.to_string()).collect();

            if missing_caps.len() == 1 {
                Err(CapabilityCheckError::UnsupportedCapability {
                    capability: missing_caps[0].clone(),
                    available_alternatives: available_caps,
                })
            } else {
                Err(CapabilityCheckError::MultipleUnsupportedCapabilities {
                    missing_capabilities: missing_caps,
                    available_alternatives: available_caps,
                })
            }
        }
    }

    /// Get all registered engine names.
    #[expect(
        clippy::expect_used,
        reason = "RwLock poisoning is unrecoverable; propagating would hide the root cause"
    )]
    pub fn registered_engines(&self) -> Vec<String> {
        let caps = self
            .capabilities
            .read()
            .expect("RwLock read lock poisoned - capability registry unavailable");
        caps.keys().cloned().collect()
    }

    /// List all registered engine names (alias for registered_engines).
    pub fn list_registered_engines(&self) -> Vec<String> {
        self.registered_engines()
    }

    /// Find engines that support all the given capabilities.
    ///
    /// Returns a list of engine names that have all the specified capabilities.
    #[expect(
        clippy::expect_used,
        reason = "RwLock poisoning is unrecoverable; propagating would hide the root cause"
    )]
    pub fn find_engines_with_capabilities(&self, required: &CapabilitySet) -> Vec<String> {
        let caps = self
            .capabilities
            .read()
            .expect("RwLock read lock poisoned - capability registry unavailable");
        caps.iter()
            .filter(|(_, engine_caps)| engine_caps.contains(required))
            .map(|(name, _)| name.clone())
            .collect()
    }
}

impl Default for CapabilityRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_capabilities() -> Vec<(Capability, &'static str)> {
        vec![
            (Capability::VectorSearch, "VectorSearch"),
            (Capability::GraphQuery, "GraphQuery"),
            (Capability::DocumentQuery, "DocumentQuery"),
            (Capability::LogQuery, "LogQuery"),
            (Capability::MetricsQuery, "MetricsQuery"),
            (Capability::TimeSeriesQuery, "TimeSeriesQuery"),
            (Capability::EventSourcingQuery, "EventSourcingQuery"),
            (Capability::Scan, "Scan"),
            (Capability::Filter, "Filter"),
            (Capability::Project, "Project"),
            (Capability::Join, "Join"),
            (Capability::Aggregate, "Aggregate"),
            (Capability::Sort, "Sort"),
            (Capability::Limit, "Limit"),
            (Capability::PredicatePushdown, "PredicatePushdown"),
            (Capability::Quantization, "Quantization"),
            (Capability::WALRecovery, "WALRecovery"),
            (Capability::Replication, "Replication"),
            (Capability::Sharding, "Sharding"),
            (Capability::MultiRegion, "MultiRegion"),
            (Capability::ColumnarAnalytics, "ColumnarAnalytics"),
            (Capability::RowGroupPruning, "RowGroupPruning"),
            (Capability::BloomFilter, "BloomFilter"),
            (Capability::CachedQueries, "CachedQueries"),
            (Capability::HNSWIndex, "HNSWIndex"),
            (Capability::IVFIndex, "IVFIndex"),
            (Capability::AnnoyIndex, "AnnoyIndex"),
            (Capability::LSHIndex, "LSHIndex"),
            (Capability::DiskANNIndex, "DiskANNIndex"),
            (Capability::SparseVectorIndex, "SparseVectorIndex"),
            (Capability::CosineDistance, "CosineDistance"),
            (Capability::EuclideanDistance, "EuclideanDistance"),
            (Capability::DotProduct, "DotProduct"),
            (Capability::HybridSearch, "HybridSearch"),
            (Capability::GraphTraversal, "GraphTraversal"),
            (Capability::PatternMatching, "PatternMatching"),
            (Capability::CypherFunctions, "CypherFunctions"),
            (Capability::GraphAggregation, "GraphAggregation"),
            (Capability::FullTextSearch, "FullTextSearch"),
            (Capability::JSONPathQueries, "JSONPathQueries"),
            (Capability::DocumentAggregation, "DocumentAggregation"),
            (Capability::PromQLQuery, "PromQLQuery"),
            (Capability::LogAggregation, "LogAggregation"),
            (Capability::MetricAggregation, "MetricAggregation"),
            (Capability::SIEMIntegration, "SIEMIntegration"),
            (Capability::FederatedQuery, "FederatedQuery"),
            (Capability::CrossModelJoin, "CrossModelJoin"),
            (Capability::DistributedTransaction, "DistributedTransaction"),
            (Capability::ConsensusProtocol, "ConsensusProtocol"),
            (Capability::ChangeDataCapture, "ChangeDataCapture"),
            (Capability::RealTimeStreaming, "RealTimeStreaming"),
            (Capability::WebSocketStreaming, "WebSocketStreaming"),
        ]
    }

    #[test]
    fn capability_display_covers_every_declared_variant() {
        for (capability, rendered) in all_capabilities() {
            assert_eq!(capability.to_string(), rendered);
        }
    }

    #[test]
    fn test_capability_set_contains() {
        let set1 =
            CapabilitySet::from_capabilities(&[Capability::VectorSearch, Capability::Filter]);
        let set2 = CapabilitySet::from_capabilities(&[Capability::VectorSearch]);

        assert!(set1.contains(&set2));
        assert!(!set2.contains(&set1));
    }

    #[test]
    fn capability_set_helpers_cover_empty_add_iter_vec_and_intersection() {
        let mut set = CapabilitySet::default();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);

        set.add(Capability::Scan);
        set.add(Capability::Filter);
        set.add(Capability::Filter);

        assert_eq!(set.len(), 2);
        assert!(!set.is_empty());
        assert!(set.contains_capability(&Capability::Scan));
        assert!(!set.contains_capability(&Capability::Join));

        let other = CapabilitySet::from_capabilities(&[Capability::Filter, Capability::Project]);
        assert!(set.intersects(&other));
        let intersection = set.intersection(&other);
        assert_eq!(intersection.len(), 1);
        assert!(intersection.contains_capability(&Capability::Filter));

        let disjoint = CapabilitySet::from_capabilities(&[Capability::GraphQuery]);
        assert!(!set.intersects(&disjoint));

        let mut as_vec = set.to_vec();
        as_vec.sort_by_key(|capability| capability.to_string());
        assert_eq!(as_vec, vec![Capability::Filter, Capability::Scan]);

        let rendered = set.to_string();
        assert!(rendered.starts_with('['));
        assert!(rendered.ends_with(']'));
        assert!(rendered.contains("Scan"));
        assert!(rendered.contains("Filter"));
    }

    #[test]
    fn test_capability_set_union() {
        let set1 = CapabilitySet::from_capabilities(&[Capability::VectorSearch]);
        let set2 = CapabilitySet::from_capabilities(&[Capability::Filter]);
        let union = set1.union(&set2);

        assert!(union.contains(&CapabilitySet::from_capabilities(&[
            Capability::VectorSearch,
            Capability::Filter,
        ])));
    }

    #[test]
    fn test_capability_set_difference() {
        let set1 =
            CapabilitySet::from_capabilities(&[Capability::VectorSearch, Capability::Filter]);
        let set2 = CapabilitySet::from_capabilities(&[Capability::VectorSearch]);
        let diff = set1.difference(&set2);

        assert!(diff.contains(&CapabilitySet::from_capabilities(&[Capability::Filter,])));
    }

    #[test]
    fn test_capability_registry() {
        let registry = CapabilityRegistry::new();
        let caps =
            CapabilitySet::from_capabilities(&[Capability::VectorSearch, Capability::Filter]);

        registry.register_capabilities("SST", caps.clone());
        let retrieved = registry.get_capabilities("SST").unwrap();

        assert!(retrieved.contains(&caps));
    }

    #[test]
    fn test_capability_check_support() {
        let registry = CapabilityRegistry::new();
        let caps =
            CapabilitySet::from_capabilities(&[Capability::VectorSearch, Capability::Filter]);

        registry.register_capabilities("SST", caps);

        let required = CapabilitySet::from_capabilities(&[Capability::VectorSearch]);
        assert!(registry.check_support("SST", &required).is_ok());

        let missing = CapabilitySet::from_capabilities(&[Capability::GraphQuery]);
        assert!(registry.check_support("SST", &missing).is_err());
    }

    #[test]
    fn capability_check_errors_render_single_multiple_and_unknown_engine_cases() {
        let registry = CapabilityRegistry::default();
        let caps =
            CapabilitySet::from_capabilities(&[Capability::VectorSearch, Capability::Filter]);
        registry.register_capabilities("SST", caps);

        let unknown = registry
            .check_support(
                "missing",
                &CapabilitySet::from_capabilities(&[Capability::VectorSearch]),
            )
            .unwrap_err();
        assert_eq!(unknown.to_string(), "Unsupported capability: missing");

        let single_missing = registry
            .check_support(
                "SST",
                &CapabilitySet::from_capabilities(&[
                    Capability::VectorSearch,
                    Capability::GraphQuery,
                ]),
            )
            .unwrap_err();
        let single_message = single_missing.to_string();
        assert!(single_message.starts_with("Unsupported capability: GraphQuery"));
        assert!(single_message.contains("Available alternatives:"));
        assert!(single_message.contains("VectorSearch"));
        assert!(single_message.contains("Filter"));

        let multiple_missing = registry
            .check_support(
                "SST",
                &CapabilitySet::from_capabilities(&[
                    Capability::GraphQuery,
                    Capability::DocumentQuery,
                ]),
            )
            .unwrap_err();
        let multiple_message = multiple_missing.to_string();
        assert!(multiple_message.starts_with("Missing capabilities:"));
        assert!(multiple_message.contains("GraphQuery"));
        assert!(multiple_message.contains("DocumentQuery"));
        assert!(multiple_message.contains("Available alternatives:"));

        let explicit_no_alternatives = CapabilityCheckError::MultipleUnsupportedCapabilities {
            missing_capabilities: vec!["A".to_string(), "B".to_string()],
            available_alternatives: vec![],
        };
        assert_eq!(
            explicit_no_alternatives.to_string(),
            "Missing capabilities: A, B"
        );
    }

    #[test]
    fn registry_listing_and_capability_search_return_matching_engines() {
        let registry = CapabilityRegistry::new();
        registry.register_capabilities(
            "vector",
            CapabilitySet::from_capabilities(&[
                Capability::VectorSearch,
                Capability::Filter,
                Capability::PredicatePushdown,
            ]),
        );
        registry.register_capabilities(
            "graph",
            CapabilitySet::from_capabilities(&[Capability::GraphQuery, Capability::Filter]),
        );

        let mut engines = registry.registered_engines();
        engines.sort();
        assert_eq!(engines, vec!["graph".to_string(), "vector".to_string()]);

        let mut listed = registry.list_registered_engines();
        listed.sort();
        assert_eq!(listed, engines);

        let mut filter_engines =
            registry.find_engines_with_capabilities(&CapabilitySet::from_capabilities(&[
                Capability::Filter,
            ]));
        filter_engines.sort();
        assert_eq!(
            filter_engines,
            vec!["graph".to_string(), "vector".to_string()]
        );

        let exact = registry.find_engines_with_capabilities(&CapabilitySet::from_capabilities(&[
            Capability::VectorSearch,
            Capability::PredicatePushdown,
        ]));
        assert_eq!(exact, vec!["vector".to_string()]);

        let none = registry.find_engines_with_capabilities(&CapabilitySet::from_capabilities(&[
            Capability::DistributedTransaction,
        ]));
        assert!(none.is_empty());
    }
}

/// Convert to the unified wire error envelope (ADR-094). Lives next to the
/// type's definition because orphan rules require the `From` source to be
/// local once `ApiError` moved to the foundation tier.
impl From<CapabilityCheckError> for proximadb_api_error::ApiError {
    fn from(err: CapabilityCheckError) -> Self {
        use proximadb_api_error::ApiError;
        match err {
            CapabilityCheckError::UnsupportedCapability {
                capability,
                available_alternatives,
            } => {
                let msg = if available_alternatives.is_empty() {
                    format!(
                        "The requested capability '{}' is not supported by the selected storage engine.",
                        capability
                    )
                } else {
                    format!(
                        "The requested capability '{}' is not supported. Available alternatives: {}",
                        capability,
                        available_alternatives.join(", ")
                    )
                };
                ApiError::UnsupportedCapability(msg)
            }
            CapabilityCheckError::MultipleUnsupportedCapabilities {
                missing_capabilities,
                available_alternatives,
            } => {
                let msg = if available_alternatives.is_empty() {
                    format!(
                        "Multiple capabilities are not supported: {}. Please check the storage engine capabilities.",
                        missing_capabilities.join(", ")
                    )
                } else {
                    format!(
                        "Multiple capabilities are not supported: {}. Available alternatives: {}",
                        missing_capabilities.join(", "),
                        available_alternatives.join(", ")
                    )
                };
                ApiError::UnsupportedCapability(msg)
            }
        }
    }
}
