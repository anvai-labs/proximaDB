//! Multi-Tier Vector Deduplication for ProximaDB
//!
//! Handles ID-based deduplication across storage tiers:
//! 1. Unflushed WAL data (memtable) - highest priority
//! 2. Flushed SST files - medium priority
//! 3. Compacted storage - lowest priority
//!
//! The merge strategy ensures that for records with the same OID:
//! - Unflushed data overrides flushed data
//! - Flushed data overrides compacted data
//! - Records without OIDs are included without deduplication

use crate::core::search::mvcc_resolution::MvccResolver;
use crate::core::search::sql_value_filter::proxima_tree_to_json_map;
use crate::core::{MetadataQuery, MetadataQueryEngine};
use chrono::{DateTime, Utc};
use proximadb_records::ProximaRecord;
use serde_json::Value as JsonValue;
use std::collections::HashMap;

/// A search candidate from a specific storage tier awaiting deduplication.
#[derive(Debug, Clone)]
pub struct TieredSearchCandidate {
    /// The matched record.
    pub record: ProximaRecord,
    /// Similarity score to the query vector.
    pub similarity: f32,
    /// Data freshness tier this result came from.
    pub tier: DataFreshnessTier,
    /// Storage engine that produced this result.
    pub engine: DeduplicationStorageEngine,
    /// Timestamp when this candidate was produced.
    pub timestamp: DateTime<Utc>,
    /// Write sequence number for ordering within a tier.
    pub sequence: u64,
    /// File path for flushed/compacted results (SST or Parquet).
    pub file_path: Option<String>,
}

/// Data freshness tier hierarchy for deduplication priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DataFreshnessTier {
    /// Lowest priority — final compacted storage.
    Compacted = 0,
    /// Medium priority — flushed but not yet compacted.
    Flushed = 1,
    /// Highest priority — WAL data still in memtable.
    Unflushed = 2,
}

/// Storage engine type for search result context (includes WAL for unflushed data).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeduplicationStorageEngine {
    SST,
    VIPER,
    WAL,
}

/// Simple key→JSON equality filters for backward compatibility.
pub type MetadataFilter = HashMap<String, JsonValue>;

/// Multi-tier deduplication manager with metadata filtering support.
pub struct MultiTierDeduplicator {
    id_to_latest: HashMap<String, TieredSearchCandidate>,
    results_without_id: Vec<TieredSearchCandidate>,
    metadata_filters: Option<MetadataFilter>,
    metadata_query: Option<MetadataQuery>,
    query_engine: MetadataQueryEngine,
    target_k: Option<usize>,
    early_termination_possible: bool,
    requires_ordering: bool,
}

impl MultiTierDeduplicator {
    pub fn new() -> Self {
        Self {
            id_to_latest: HashMap::new(),
            results_without_id: Vec::new(),
            metadata_filters: None,
            metadata_query: None,
            query_engine: MetadataQueryEngine::new(),
            target_k: None,
            early_termination_possible: false,
            requires_ordering: true,
        }
    }

    pub fn with_k(k: usize) -> Self {
        Self {
            target_k: Some(k),
            ..Self::new()
        }
    }

    pub fn with_filters(metadata_filters: MetadataFilter) -> Self {
        Self {
            metadata_filters: Some(metadata_filters),
            ..Self::new()
        }
    }

    pub fn with_query(metadata_query: MetadataQuery) -> Self {
        Self {
            metadata_query: Some(metadata_query),
            ..Self::new()
        }
    }

    pub fn with_filters_and_query(
        metadata_filters: MetadataFilter,
        metadata_query: MetadataQuery,
    ) -> Self {
        Self {
            metadata_filters: Some(metadata_filters),
            metadata_query: Some(metadata_query),
            ..Self::new()
        }
    }

    pub fn set_target_k(&mut self, k: usize) {
        self.target_k = Some(k);
    }

    pub fn set_requires_ordering(&mut self, requires_ordering: bool) {
        self.requires_ordering = requires_ordering;
        if requires_ordering {
            self.early_termination_possible = false;
        }
    }

    pub fn can_terminate_early(&self) -> bool {
        !self.requires_ordering && self.target_k.is_some()
    }

    pub fn is_early_terminated(&self) -> bool {
        self.early_termination_possible
    }

    /// Check whether a record passes the configured metadata filters / query.
    fn matches_filters(&mut self, record: &ProximaRecord) -> bool {
        let json_meta = proxima_tree_to_json_map(&record.props);

        // Logical metadata query takes precedence
        if let Some(ref query) = self.metadata_query {
            match self.query_engine.evaluate(query, &json_meta) {
                Ok(result) => {
                    if !result {
                        tracing::debug!(
                            "Query filter: record '{}' did not match logical query",
                            if record.oid.is_empty() {
                                "<no-id>"
                            } else {
                                &record.oid
                            }
                        );
                    }
                    return result;
                }
                Err(e) => {
                    tracing::warn!(
                        "Query evaluation error for record '{}': {}",
                        if record.oid.is_empty() {
                            "<no-id>"
                        } else {
                            &record.oid
                        },
                        e
                    );
                    return false;
                }
            }
        }

        // Simple equality filters
        match &self.metadata_filters {
            None => true,
            Some(filters) => {
                for (key, expected) in filters {
                    match json_meta.get(key) {
                        Some(actual) if actual == expected => {}
                        Some(actual) => {
                            tracing::debug!(
                                "Simple filter mismatch: {} expected {:?}, got {:?}",
                                key,
                                expected,
                                actual
                            );
                            return false;
                        }
                        None => {
                            tracing::debug!("Simple filter: key '{}' not in props", key);
                            return false;
                        }
                    }
                }
                true
            }
        }
    }

    /// Add search results from a specific storage tier.
    pub fn add_tier_results(&mut self, results: Vec<TieredSearchCandidate>) {
        if self.early_termination_possible && !self.requires_ordering {
            tracing::debug!(
                "Early termination: already have {} results, skipping {} new",
                self.target_k.unwrap_or(0),
                results.len()
            );
            return;
        }

        for result in results {
            if !self.matches_filters(&result.record) {
                tracing::debug!(
                    "Filter: skipping record '{}' due to metadata mismatch",
                    if result.record.oid.is_empty() {
                        "<no-id>"
                    } else {
                        &result.record.oid
                    }
                );
                continue;
            }

            if result.record.oid.is_empty() {
                self.results_without_id.push(result);
            } else {
                let oid = result.record.oid.clone();
                let should_replace = match self.id_to_latest.get(&oid) {
                    Some(existing) => {
                        if result.tier > existing.tier {
                            true
                        } else if result.tier < existing.tier {
                            false
                        } else if result.sequence > existing.sequence {
                            true
                        } else if result.sequence < existing.sequence {
                            false
                        } else {
                            let resolver = MvccResolver::new();
                            resolver.compare_records(&result.record, &existing.record)
                        }
                    }
                    None => true,
                };

                if should_replace {
                    if let Some(existing) = self.id_to_latest.get(&oid) {
                        tracing::debug!(
                            "Dedup: replacing '{}' {:?}/{:?} (seq:{}, v:{}) with {:?}/{:?} (seq:{}, v:{})",
                            oid,
                            existing.tier,
                            existing.engine,
                            existing.sequence,
                            existing.record.record_version,
                            result.tier,
                            result.engine,
                            result.sequence,
                            result.record.record_version,
                        );
                    }
                    self.id_to_latest.insert(oid.clone(), result);
                }
            }

            if !self.requires_ordering
                && let Some(k) = self.target_k
                && self.id_to_latest.len() + self.results_without_id.len() >= k
            {
                self.early_termination_possible = true;
                tracing::info!(
                    "Early termination: reached {} unique results (target k={})",
                    self.id_to_latest.len() + self.results_without_id.len(),
                    k
                );
                return;
            }
        }
    }

    /// Get final deduplicated results sorted by similarity (descending), limited to `k`.
    pub fn get_final_results(self, k: usize) -> Vec<TieredSearchCandidate> {
        let current_time_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);

        let unique_ids_count = self.id_to_latest.len();
        let without_id_count = self.results_without_id.len();
        let mut final_results = Vec::new();
        let mut tombstones_filtered = 0;

        for candidate in self.id_to_latest.into_values() {
            // Tombstone: no embeddings + valid_to_ns in the past
            let is_tombstone = candidate.record.embeddings.is_empty()
                && candidate
                    .record
                    .valid_to_ns
                    .is_some_and(|vt| vt <= current_time_ns);
            if is_tombstone {
                tombstones_filtered += 1;
                tracing::debug!("Filtering tombstone: '{}'", candidate.record.oid);
            } else {
                final_results.push(candidate);
            }
        }

        for candidate in self.results_without_id {
            let is_tombstone = candidate.record.embeddings.is_empty()
                && candidate
                    .record
                    .valid_to_ns
                    .is_some_and(|vt| vt <= current_time_ns);
            if is_tombstone {
                tombstones_filtered += 1;
            } else {
                final_results.push(candidate);
            }
        }

        final_results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        final_results.truncate(k);

        tracing::info!(
            "Multi-tier dedup: {} unique IDs, {} without ID, {} tombstones filtered, {} final",
            unique_ids_count,
            without_id_count,
            tombstones_filtered,
            final_results.len()
        );
        final_results
    }

    pub fn get_stats(&self) -> DeduplicationStats {
        DeduplicationStats {
            unique_ids: self.id_to_latest.len(),
            records_without_id: self.results_without_id.len(),
            total_records: self.id_to_latest.len() + self.results_without_id.len(),
        }
    }
}

/// Statistics for the deduplication process.
#[derive(Debug, Clone)]
pub struct DeduplicationStats {
    pub unique_ids: usize,
    pub records_without_id: usize,
    pub total_records: usize,
}

impl Default for MultiTierDeduplicator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    use proximadb_data_model::ProximaValue;
    use proximadb_records::{EmbeddingCell, LabelSet, ProximaTree, ProximaTreeNode};

    fn make_record(oid: &str, version: u64, values: Vec<f32>) -> ProximaRecord {
        let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
        ProximaRecord {
            oid: oid.to_string(),
            local_id: None,
            tid: None,
            variation_id: None,
            record_version: version,
            spec_version: 1,
            tenant_id: String::new(),
            permitted_principals: Vec::new(),
            rls_policy_id: None,
            created_at_ns: now_ns,
            updated_at_ns: now_ns,
            valid_from_ns: None,
            valid_to_ns: None,
            origin: None,
            actor: None,
            method: None,
            memory_type: None,
            props: ProximaTree::new(),
            refs: Vec::new(),
            edge: None,
            embeddings: if values.is_empty() {
                vec![]
            } else {
                vec![EmbeddingCell::new_fp32(
                    "default",
                    "dense_vector",
                    values.len() as u32,
                    values,
                )]
            },
            sequence: None,
            labels: LabelSet::new(),
            ..Default::default()
        }
    }

    fn make_record_with_props(
        oid: &str,
        version: u64,
        values: Vec<f32>,
        props: ProximaTree,
    ) -> ProximaRecord {
        let mut r = make_record(oid, version, values);
        r.props = props;
        r
    }

    fn candidate(
        record: ProximaRecord,
        similarity: f32,
        tier: DataFreshnessTier,
        engine: DeduplicationStorageEngine,
        seq: u64,
    ) -> TieredSearchCandidate {
        TieredSearchCandidate {
            record,
            similarity,
            tier,
            engine,
            timestamp: Utc::now(),
            sequence: seq,
            file_path: None,
        }
    }

    // -----------------------------------------------------------------------
    // Tier priority
    // -----------------------------------------------------------------------

    #[test]
    fn test_storage_tier_ordering() {
        assert!(DataFreshnessTier::Unflushed > DataFreshnessTier::Flushed);
        assert!(DataFreshnessTier::Flushed > DataFreshnessTier::Compacted);
        assert_eq!(DataFreshnessTier::Compacted as u8, 0);
        assert_eq!(DataFreshnessTier::Flushed as u8, 1);
        assert_eq!(DataFreshnessTier::Unflushed as u8, 2);
    }

    // -----------------------------------------------------------------------
    // Basic deduplication
    // -----------------------------------------------------------------------

    #[test]
    fn test_multi_tier_deduplication() {
        let mut dedup = MultiTierDeduplicator::new();
        let now = Utc::now();

        let compacted = TieredSearchCandidate {
            record: make_record("vector_1", 1, vec![1.0, 2.0, 3.0]),
            similarity: 0.5,
            tier: DataFreshnessTier::Compacted,
            engine: DeduplicationStorageEngine::VIPER,
            timestamp: now,
            sequence: 100,
            file_path: Some("/data/compacted.parquet".to_string()),
        };

        let flushed = TieredSearchCandidate {
            record: make_record("vector_1", 2, vec![1.1, 2.1, 3.1]),
            similarity: 0.4,
            tier: DataFreshnessTier::Flushed,
            engine: DeduplicationStorageEngine::SST,
            timestamp: now,
            sequence: 200,
            file_path: Some("/data/flushed.sstable".to_string()),
        };

        let unflushed = TieredSearchCandidate {
            record: make_record("vector_1", 3, vec![1.2, 2.2, 3.2]),
            similarity: 0.3,
            tier: DataFreshnessTier::Unflushed,
            engine: DeduplicationStorageEngine::WAL,
            timestamp: now,
            sequence: 300,
            file_path: None,
        };

        dedup.add_tier_results(vec![compacted]);
        dedup.add_tier_results(vec![flushed]);
        dedup.add_tier_results(vec![unflushed]);

        let results = dedup.get_final_results(10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].tier, DataFreshnessTier::Unflushed);
        assert_eq!(results[0].record.record_version, 3);
        assert_eq!(results[0].similarity, 0.3);
    }

    #[test]
    fn test_same_tier_ordering() {
        let mut dedup = MultiTierDeduplicator::new();

        // Same sequence, different versions — higher version wins
        let v1 = TieredSearchCandidate {
            record: make_record("vector_1", 1, vec![1.0, 2.0, 3.0]),
            similarity: 0.5,
            tier: DataFreshnessTier::Unflushed,
            engine: DeduplicationStorageEngine::WAL,
            timestamp: Utc::now(),
            sequence: 100,
            file_path: None,
        };
        let v2 = TieredSearchCandidate {
            record: make_record("vector_1", 2, vec![1.1, 2.1, 3.1]),
            similarity: 0.4,
            tier: DataFreshnessTier::Unflushed,
            engine: DeduplicationStorageEngine::WAL,
            timestamp: Utc::now(),
            sequence: 100,
            file_path: None,
        };

        dedup.add_tier_results(vec![v1]);
        dedup.add_tier_results(vec![v2]);

        let results = dedup.get_final_results(10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].record.record_version, 2);
    }

    #[test]
    fn test_basic_deduplication() {
        let mut dedup = MultiTierDeduplicator::new();

        let results = vec![
            TieredSearchCandidate {
                record: make_record("vec1", 1, vec![1.0, 0.0, 0.0]),
                similarity: 0.8,
                tier: DataFreshnessTier::Compacted,
                engine: DeduplicationStorageEngine::SST,
                timestamp: Utc::now() - Duration::hours(2),
                sequence: 100,
                file_path: Some("/data/compacted.db".to_string()),
            },
            TieredSearchCandidate {
                record: make_record("vec1", 2, vec![1.0, 0.0, 0.0]),
                similarity: 0.85,
                tier: DataFreshnessTier::Flushed,
                engine: DeduplicationStorageEngine::SST,
                timestamp: Utc::now() - Duration::hours(1),
                sequence: 200,
                file_path: Some("/data/flushed.db".to_string()),
            },
        ];

        dedup.add_tier_results(results);
        let merged = dedup.get_final_results(10);

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].record.record_version, 2);
        assert_eq!(merged[0].similarity, 0.85);
    }

    #[test]
    fn test_deduplication_without_ids() {
        let _ = proximadb_hardware::hardware_capabilities();
        let mut dedup = MultiTierDeduplicator::new();

        let results = vec![
            TieredSearchCandidate {
                record: make_record("", 1, vec![1.0, 0.0, 0.0]),
                similarity: 0.9,
                tier: DataFreshnessTier::Flushed,
                engine: DeduplicationStorageEngine::VIPER,
                timestamp: Utc::now(),
                sequence: 100,
                file_path: Some("/data/viper/vectors.parquet".to_string()),
            },
            TieredSearchCandidate {
                record: make_record("", 1, vec![0.0, 1.0, 0.0]),
                similarity: 0.85,
                tier: DataFreshnessTier::Flushed,
                engine: DeduplicationStorageEngine::VIPER,
                timestamp: Utc::now(),
                sequence: 101,
                file_path: Some("/data/viper/vectors.parquet".to_string()),
            },
        ];

        dedup.add_tier_results(results);
        let merged = dedup.get_final_results(10);

        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].similarity, 0.9);
        assert_eq!(merged[1].similarity, 0.85);
    }

    // -----------------------------------------------------------------------
    // Metadata filtering
    // -----------------------------------------------------------------------

    fn props_from(pairs: &[(&str, &str)]) -> ProximaTree {
        pairs
            .iter()
            .map(|(k, v)| {
                (
                    k.to_string(),
                    ProximaTreeNode::Value(ProximaValue::String(v.to_string())),
                )
            })
            .collect()
    }

    #[test]
    fn test_metadata_filtering() {
        let mut filters = MetadataFilter::new();
        filters.insert("category".to_string(), serde_json::json!("science"));
        filters.insert("published".to_string(), serde_json::json!("true"));

        let mut dedup = MultiTierDeduplicator::with_filters(filters);

        let results: Vec<TieredSearchCandidate> = vec![
            // doc1: matches
            candidate(
                make_record_with_props(
                    "doc1",
                    1,
                    vec![1.0, 0.0],
                    props_from(&[("category", "science"), ("published", "true")]),
                ),
                0.9,
                DataFreshnessTier::Flushed,
                DeduplicationStorageEngine::SST,
                0,
            ),
            // doc2: wrong category
            candidate(
                make_record_with_props(
                    "doc2",
                    1,
                    vec![0.0, 1.0],
                    props_from(&[("category", "history"), ("published", "true")]),
                ),
                0.8,
                DataFreshnessTier::Flushed,
                DeduplicationStorageEngine::SST,
                1,
            ),
        ];

        dedup.add_tier_results(results);
        let merged = dedup.get_final_results(10);

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].record.oid, "doc1");
    }

    #[test]
    fn test_simple_metadata_query() {
        let mut filters = MetadataFilter::new();
        filters.insert("language".to_string(), serde_json::json!("en"));

        let mut dedup = MultiTierDeduplicator::with_filters(filters);

        let results: Vec<TieredSearchCandidate> = vec![
            candidate(
                make_record_with_props(
                    "doc1",
                    1,
                    vec![1.0, 0.0],
                    props_from(&[("language", "en"), ("category", "tech")]),
                ),
                0.9,
                DataFreshnessTier::Flushed,
                DeduplicationStorageEngine::VIPER,
                0,
            ),
            candidate(
                make_record_with_props(
                    "doc2",
                    1,
                    vec![0.0, 1.0],
                    props_from(&[("language", "fr"), ("category", "tech")]),
                ),
                0.8,
                DataFreshnessTier::Flushed,
                DeduplicationStorageEngine::VIPER,
                1,
            ),
        ];

        dedup.add_tier_results(results);
        let merged = dedup.get_final_results(10);

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].record.oid, "doc1");
    }

    // -----------------------------------------------------------------------
    // Mixed engines and complex scenarios
    // -----------------------------------------------------------------------

    #[test]
    fn test_mixed_engine_deduplication() {
        let mut dedup = MultiTierDeduplicator::new();

        let results = vec![
            candidate(
                make_record("vec1", 1, vec![1.0, 0.0, 0.0]),
                0.8,
                DataFreshnessTier::Compacted,
                DeduplicationStorageEngine::SST,
                100,
            ),
            candidate(
                make_record("vec1", 2, vec![1.0, 0.0, 0.0]),
                0.85,
                DataFreshnessTier::Compacted,
                DeduplicationStorageEngine::VIPER,
                200,
            ),
            candidate(
                make_record("vec1", 3, vec![1.0, 0.0, 0.0]),
                0.9,
                DataFreshnessTier::Unflushed,
                DeduplicationStorageEngine::WAL,
                300,
            ),
        ];

        dedup.add_tier_results(results);
        let merged = dedup.get_final_results(10);

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].record.record_version, 3);
        assert_eq!(merged[0].similarity, 0.9);
    }

    #[test]
    fn test_k_limit_enforcement() {
        let mut dedup = MultiTierDeduplicator::new();

        let results: Vec<TieredSearchCandidate> = (0..20)
            .map(|i| {
                candidate(
                    make_record(&format!("vec{}", i), 1, vec![i as f32, 0.0, 0.0]),
                    i as f32 * 0.01,
                    DataFreshnessTier::Flushed,
                    DeduplicationStorageEngine::SST,
                    i as u64,
                )
            })
            .collect();

        dedup.add_tier_results(results);
        let merged = dedup.get_final_results(10);

        assert_eq!(merged.len(), 10);
        // Sorted descending by score — vec19 (0.19) should be first
        assert_eq!(merged[0].record.oid, "vec19");
        assert_eq!(merged[merged.len() - 1].record.oid, "vec10");
    }

    #[test]
    fn test_complex_deduplication_scenario() {
        let mut dedup = MultiTierDeduplicator::new();
        let mut results = Vec::new();

        // vecA: in all tiers
        for (ver, tier, engine, seq) in [
            (
                1u64,
                DataFreshnessTier::Compacted,
                DeduplicationStorageEngine::SST,
                100u64,
            ),
            (
                2,
                DataFreshnessTier::Flushed,
                DeduplicationStorageEngine::SST,
                200,
            ),
            (
                3,
                DataFreshnessTier::Unflushed,
                DeduplicationStorageEngine::WAL,
                300,
            ),
        ] {
            results.push(candidate(
                make_record_with_props(
                    "vecA",
                    ver,
                    vec![1.0, 0.0, 0.0],
                    props_from(&[("version", &ver.to_string())]),
                ),
                0.95,
                tier,
                engine,
                seq,
            ));
        }

        // vecB: compacted + flushed only
        for (ver, tier, engine, seq) in [
            (
                1u64,
                DataFreshnessTier::Compacted,
                DeduplicationStorageEngine::VIPER,
                150u64,
            ),
            (
                2,
                DataFreshnessTier::Flushed,
                DeduplicationStorageEngine::VIPER,
                250,
            ),
        ] {
            results.push(candidate(
                make_record("vecB", ver, vec![0.0, 1.0, 0.0]),
                0.90,
                tier,
                engine,
                seq,
            ));
        }

        // vecC: no ID (immutable)
        results.push(TieredSearchCandidate {
            record: make_record("", 1, vec![0.0, 0.0, 1.0]),
            similarity: 0.85,
            tier: DataFreshnessTier::Flushed,
            engine: DeduplicationStorageEngine::VIPER,
            timestamp: Utc::now() - Duration::hours(4),
            sequence: 1000,
            file_path: Some("/data/immutable.parquet".to_string()),
        });

        dedup.add_tier_results(results);
        let merged = dedup.get_final_results(10);

        assert_eq!(merged.len(), 3);

        let vec_a = merged.iter().find(|r| r.record.oid == "vecA").unwrap();
        assert_eq!(vec_a.record.record_version, 3);

        let vec_b = merged.iter().find(|r| r.record.oid == "vecB").unwrap();
        assert_eq!(vec_b.record.record_version, 2);

        assert!(merged.iter().any(|r| r.record.oid.is_empty()));
    }

    // -----------------------------------------------------------------------
    // Early termination
    // -----------------------------------------------------------------------

    #[test]
    fn test_early_termination_without_ordering() {
        let mut dedup = MultiTierDeduplicator::with_k(2);
        dedup.set_requires_ordering(false);

        let candidates: Vec<TieredSearchCandidate> = (0..5)
            .map(|i| {
                candidate(
                    make_record(&format!("vector_{}", i), 1, vec![i as f32, 0.0, 0.0]),
                    i as f32,
                    DataFreshnessTier::Unflushed,
                    DeduplicationStorageEngine::WAL,
                    100 + i as u64,
                )
            })
            .collect();

        dedup.add_tier_results(candidates[0..2].to_vec());
        assert!(dedup.early_termination_possible);

        dedup.add_tier_results(candidates[2..5].to_vec());

        let results = dedup.get_final_results(2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_no_early_termination_with_ordering() {
        let mut dedup = MultiTierDeduplicator::with_k(2);
        dedup.set_requires_ordering(true);

        let candidates: Vec<TieredSearchCandidate> = (0..5)
            .map(|i| {
                candidate(
                    make_record(&format!("vector_{}", i), 1, vec![i as f32, 0.0, 0.0]),
                    (5 - i) as f32,
                    DataFreshnessTier::Unflushed,
                    DeduplicationStorageEngine::WAL,
                    100 + i as u64,
                )
            })
            .collect();

        dedup.add_tier_results(candidates[0..2].to_vec());
        assert!(!dedup.early_termination_possible);

        dedup.add_tier_results(candidates[2..5].to_vec());
        let results = dedup.get_final_results(2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].similarity, 5.0);
        assert_eq!(results[1].similarity, 4.0);
    }

    #[test]
    fn test_early_termination_logic() {
        // With ordering — no early termination even at k=2
        {
            let mut dedup = MultiTierDeduplicator::with_k(2);
            dedup.set_requires_ordering(true);

            let candidates: Vec<TieredSearchCandidate> = (0..5)
                .map(|i| {
                    candidate(
                        make_record(&format!("vec_{}", i), 1, vec![i as f32]),
                        (5 - i) as f32,
                        DataFreshnessTier::Unflushed,
                        DeduplicationStorageEngine::WAL,
                        i as u64,
                    )
                })
                .collect();

            dedup.add_tier_results(candidates);
            let results = dedup.get_final_results(2);
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].similarity, 5.0);
            assert_eq!(results[1].similarity, 4.0);
        }

        // Without ordering — early termination at k=2
        {
            let mut dedup = MultiTierDeduplicator::with_k(2);
            dedup.set_requires_ordering(false);

            let candidates: Vec<TieredSearchCandidate> = (0..5)
                .map(|i| {
                    candidate(
                        make_record(&format!("vec_{}", i), 1, vec![i as f32]),
                        i as f32,
                        DataFreshnessTier::Unflushed,
                        DeduplicationStorageEngine::WAL,
                        i as u64,
                    )
                })
                .collect();

            dedup.add_tier_results(candidates[0..2].to_vec());
            assert!(dedup.is_early_terminated());

            dedup.add_tier_results(candidates[2..5].to_vec());
            let results = dedup.get_final_results(10);
            assert_eq!(results.len(), 2);
        }
    }

    #[test]
    fn test_sql_query_behavior() {
        {
            let mut dedup = MultiTierDeduplicator::with_k(10);
            dedup.set_requires_ordering(true);
            assert!(!dedup.can_terminate_early());
        }
        {
            let mut dedup = MultiTierDeduplicator::with_k(10);
            dedup.set_requires_ordering(false);
            assert!(dedup.can_terminate_early());
        }
    }

    #[test]
    fn test_grpc_rest_always_ordered() {
        let mut dedup = MultiTierDeduplicator::with_k(25);
        dedup.set_requires_ordering(true);
        assert!(!dedup.can_terminate_early());
    }
}
