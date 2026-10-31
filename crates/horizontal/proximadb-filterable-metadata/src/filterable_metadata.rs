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

//! Filterable Metadata for HNSW Nodes (TD-064)
//!
//! This module defines a compact, cacheable metadata schema that lives inside
//! HNSW nodes to enable predicate-aware ANN search. Only filterable attributes
//! are stored; heavy metadata remains in SST storage.
//!
//! ## Design Principles
//!
//! 1. **Minimal Memory Footprint**: Target <50 bytes per record overhead
//! 2. **Filter-Centric**: Only attributes used for WHERE/HAVING clauses
//! 3. **Fast Comparison**: Zero-copy or simple primitive comparisons
//! 4. **Serializable**: Must persist with HNSW index checkpoints
//!
//! ## What Gets Cached vs What Doesn't
//!
//! ✅ **Cached in HNSW** (this module):
//! - tenant_id: String (tenant isolation)
//! - rls_tags: Vec<String> (row-level security)
//! - created_at_ns: i64 (time range predicates)
//! - expires_at_ns: Option<i64> (TTL filtering)
//! - typed_attrs: Compact primitive attributes (int, float, bool, string)
//!
//! ❌ **NOT Cached** (stays in SST):
//! - Large JSON blobs
//! - Text content chunks
//! - Vector embeddings (already separate)
//! - Rarely-filtered fields

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use proximadb_data_model::ProximaValue;
use proximadb_records::{ProximaRecord, ProximaTreeNode};

/// Filterable metadata cached in HNSW nodes for predicate-aware search
///
/// Memory overhead target: <50 bytes per record (excluding typed_attrs)
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FilterableHnswMetadata {
    /// Tenant identifier for multi-tenancy isolation
    pub tenant_id: Option<String>,

    /// Row-level security tags for authorization predicates
    pub rls_tags: Vec<String>,

    /// Creation timestamp (nanoseconds since epoch) for time-range queries
    pub created_at_ns: i64,

    /// Expiration timestamp for TTL predicates (None = no expiration)
    pub expires_at_ns: Option<i64>,

    /// Compact typed attributes for equality/range predicates
    /// Only stores primitives; complex types remain in SST
    pub typed_attrs: TypedAttributes,
}

impl FilterableHnswMetadata {
    /// Estimate memory footprint in bytes
    pub fn estimated_size_bytes(&self) -> usize {
        let mut size = 8 + // created_at_ns
            self.rls_tags.len() * 8 + // rough estimate per tag
            8; // expires_at_ns option

        if let Some(ref tenant) = self.tenant_id {
            size += tenant.len();
        }

        size += self.typed_attrs.estimated_size_bytes();
        size
    }

    /// Check if this metadata matches a tenant predicate
    pub fn matches_tenant(&self, tenant_id: &str) -> bool {
        self.tenant_id.as_deref() == Some(tenant_id)
    }

    /// Check if this metadata matches an RLS tag predicate
    pub fn matches_rls_tag(&self, required_tag: &str) -> bool {
        self.rls_tags.iter().any(|tag| tag == required_tag)
    }

    /// Check if this metadata is within a time range
    pub fn matches_time_range(&self, start_ns: i64, end_ns: i64) -> bool {
        self.created_at_ns >= start_ns && self.created_at_ns <= end_ns
    }

    /// Check if this record has not expired (TTL check)
    pub fn is_valid_at_time(&self, query_time_ns: i64) -> bool {
        self.expires_at_ns
            .map(|exp| exp > query_time_ns)
            .unwrap_or(true)
    }
}

/// Compact typed attributes stored in HNSW for fast predicate evaluation
///
/// Only primitive types are supported to keep memory overhead low.
/// Complex attributes must be fetched from SST storage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TypedAttributes {
    /// Integer attributes (name -> value)
    pub int_attrs: HashMap<String, i64>,

    /// Float attributes (name -> value)
    /// Note: Stored as f64 but compared with epsilon tolerance
    pub float_attrs: HashMap<String, OrderedFloat>,

    /// Boolean attributes (name -> value)
    pub bool_attrs: HashMap<String, bool>,

    /// String attributes (name -> value)
    /// Only short strings (<64 bytes) should be cached
    pub string_attrs: HashMap<String, String>,
}

impl TypedAttributes {
    pub fn estimated_size_bytes(&self) -> usize {
        let int_size = self.int_attrs.len() * (8 + 16); // value + key overhead
        let float_size = self.float_attrs.len() * (8 + 16);
        let bool_size = self.bool_attrs.len() * (1 + 16);
        let string_size: usize = self
            .string_attrs
            .iter()
            .map(|(k, v)| k.len() + v.len())
            .sum();

        int_size + float_size + bool_size + string_size
    }

    /// Get integer attribute
    pub fn get_int(&self, name: &str) -> Option<i64> {
        self.int_attrs.get(name).copied()
    }

    /// Get float attribute
    pub fn get_float(&self, name: &str) -> Option<f64> {
        self.float_attrs.get(name).map(|f| f.0)
    }

    /// Get boolean attribute
    pub fn get_bool(&self, name: &str) -> Option<bool> {
        self.bool_attrs.get(name).copied()
    }

    /// Get string attribute
    pub fn get_string(&self, name: &str) -> Option<&str> {
        self.string_attrs.get(name).map(|s| s.as_str())
    }

    /// Set integer attribute
    pub fn set_int(&mut self, name: String, value: i64) {
        self.int_attrs.insert(name, value);
    }

    /// Set float attribute
    pub fn set_float(&mut self, name: String, value: f64) {
        self.float_attrs.insert(name, OrderedFloat(value));
    }

    /// Set boolean attribute
    pub fn set_bool(&mut self, name: String, value: bool) {
        self.bool_attrs.insert(name, value);
    }

    /// Set string attribute (returns error if string too long)
    pub fn set_string(&mut self, name: String, value: String) -> Result<(), MetadataError> {
        if value.len() > 64 {
            return Err(MetadataError::StringTooLong(value.len()));
        }
        self.string_attrs.insert(name, value);
        Ok(())
    }
}

/// Wrapper for f64 that implements Eq and Hash
pub use proximadb_records::BitKeyF64 as OrderedFloat;

/// Metadata extraction errors
#[derive(Debug, thiserror::Error)]
pub enum MetadataError {
    #[error("String attribute too long: {0} bytes (max 64)")]
    StringTooLong(usize),

    #[error("Attribute not found: {0}")]
    AttributeNotFound(String),

    #[error("Type mismatch for attribute: {0}")]
    TypeMismatch(String),
}

/// Extract filterable metadata from a ProximaRecord
///
/// This function extracts only the attributes that are useful for predicate
/// evaluation during HNSW traversal. Heavy metadata is intentionally skipped.
pub fn extract_filterable_metadata(
    record: &ProximaRecord,
    filterable_fields: &FilterableFieldsConfig,
) -> FilterableHnswMetadata {
    let mut metadata = FilterableHnswMetadata {
        tenant_id: if !record.tenant_id.is_empty() {
            Some(record.tenant_id.clone())
        } else {
            None
        },
        created_at_ns: record.created_at_ns,
        expires_at_ns: record
            .props
            .get("expires_at_ns")
            .and_then(extract_i64_from_node),
        ..FilterableHnswMetadata::default()
    };

    // Extract RLS tags from props if present
    if let Some(rls_value) = record.props.get("rls_tags")
        && let Some(tags) = extract_string_array(rls_value)
    {
        metadata.rls_tags = tags;
    }

    // Extract typed attributes based on config
    for field in &filterable_fields.int_fields {
        if let Some(value) = record.props.get(field)
            && let Some(int_val) = extract_i64_from_node(value)
        {
            metadata.typed_attrs.set_int(field.clone(), int_val);
        }
    }

    for field in &filterable_fields.float_fields {
        if let Some(value) = record.props.get(field)
            && let Some(float_val) = extract_f64_from_node(value)
        {
            metadata.typed_attrs.set_float(field.clone(), float_val);
        }
    }

    for field in &filterable_fields.bool_fields {
        if let Some(value) = record.props.get(field)
            && let Some(bool_val) = extract_bool_from_node(value)
        {
            metadata.typed_attrs.set_bool(field.clone(), bool_val);
        }
    }

    for field in &filterable_fields.string_fields {
        if let Some(value) = record.props.get(field)
            && let Some(string_val) = extract_string_from_node(value)
            && string_val.len() <= 64
        {
            let _ = metadata.typed_attrs.set_string(field.clone(), string_val);
        }
    }

    metadata
}

/// Configuration for which fields should be extracted as filterable metadata
#[derive(Debug, Clone, Default)]
pub struct FilterableFieldsConfig {
    /// Integer field names to extract
    pub int_fields: Vec<String>,

    /// Float field names to extract
    pub float_fields: Vec<String>,

    /// Boolean field names to extract
    pub bool_fields: Vec<String>,

    /// String field names to extract (max 64 bytes each)
    pub string_fields: Vec<String>,
}

/// Helper to extract i64 from ProximaTreeNode
fn extract_i64_from_node(node: &ProximaTreeNode) -> Option<i64> {
    match node {
        ProximaTreeNode::Value(ProximaValue::Int64(i)) => Some(*i),
        ProximaTreeNode::Value(ProximaValue::Int32(i)) => Some(*i as i64),
        ProximaTreeNode::Value(ProximaValue::Int16(i)) => Some(*i as i64),
        ProximaTreeNode::Value(ProximaValue::Int8(i)) => Some(*i as i64),
        ProximaTreeNode::Value(ProximaValue::UInt64(u)) => Some(*u as i64),
        ProximaTreeNode::Value(ProximaValue::UInt32(u)) => Some(*u as i64),
        ProximaTreeNode::Value(ProximaValue::UInt16(u)) => Some(*u as i64),
        ProximaTreeNode::Value(ProximaValue::UInt8(u)) => Some(*u as i64),
        ProximaTreeNode::Value(ProximaValue::String(s)) => s.parse().ok(),
        _ => None,
    }
}

/// Helper to extract f64 from ProximaTreeNode
fn extract_f64_from_node(node: &ProximaTreeNode) -> Option<f64> {
    match node {
        ProximaTreeNode::Value(ProximaValue::Float64(d)) => Some(*d),
        ProximaTreeNode::Value(ProximaValue::Float32(f)) => Some(*f as f64),
        ProximaTreeNode::Value(ProximaValue::Float16(f)) => Some(*f as f64),
        ProximaTreeNode::Value(ProximaValue::Int64(i)) => Some(*i as f64),
        ProximaTreeNode::Value(ProximaValue::UInt64(u)) => Some(*u as f64),
        ProximaTreeNode::Value(ProximaValue::String(s)) => s.parse().ok(),
        _ => None,
    }
}

/// Helper to extract bool from ProximaTreeNode
fn extract_bool_from_node(node: &ProximaTreeNode) -> Option<bool> {
    match node {
        ProximaTreeNode::Value(ProximaValue::Boolean(b)) => Some(*b),
        ProximaTreeNode::Value(ProximaValue::String(s)) => s.parse().ok(),
        _ => None,
    }
}

/// Helper to extract String from ProximaTreeNode
fn extract_string_from_node(node: &ProximaTreeNode) -> Option<String> {
    match node {
        ProximaTreeNode::Value(ProximaValue::String(s)) => Some(s.clone()),
        ProximaTreeNode::Value(ProximaValue::Int64(i)) => Some(i.to_string()),
        ProximaTreeNode::Value(ProximaValue::UInt64(u)) => Some(u.to_string()),
        ProximaTreeNode::Value(ProximaValue::Float64(d)) => Some(d.to_string()),
        ProximaTreeNode::Value(ProximaValue::Float32(f)) => Some(f.to_string()),
        ProximaTreeNode::Value(ProximaValue::Boolean(b)) => Some(b.to_string()),
        _ => None,
    }
}

/// Helper to extract String array from ProximaTreeNode
fn extract_string_array(node: &ProximaTreeNode) -> Option<Vec<String>> {
    match node {
        ProximaTreeNode::Value(ProximaValue::String(s)) => {
            // Comma-separated string as array
            Some(s.split(',').map(|s| s.trim().to_string()).collect())
        }
        _ => None,
    }
}

/// Shared cache used by every AXIS index (HNSW, IVF, Annoy, LSH) to hold
/// filterable metadata. Indexes compose this rather than maintaining their
/// own DashMap so the AXIS layer can evolve caching, eviction, and
/// projection-freshness semantics in one place.
///
/// Concurrency: lock-free reads via DashMap; the optional
/// `FilterableFieldsConfig` is guarded by a parking RwLock since config
/// updates are rare.
#[derive(Debug, Clone, Default)]
pub struct FilterableMetadataCache {
    metadata: Arc<DashMap<String, FilterableHnswMetadata>>,
    fields_config: Arc<RwLock<Option<FilterableFieldsConfig>>>,
}

impl FilterableMetadataCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert/replace metadata for a record id.
    pub fn insert(&self, id: String, metadata: FilterableHnswMetadata) {
        self.metadata.insert(id, metadata);
    }

    /// Drop metadata for a record id (on remove). Idempotent.
    pub fn remove(&self, id: &str) {
        self.metadata.remove(id);
    }

    /// True when any metadata has been cached. Used by indexes to decide
    /// whether `supports_predicate_search()` should return true.
    pub fn is_empty(&self) -> bool {
        self.metadata.is_empty()
    }

    /// Number of cached entries (for stats/EXPLAIN).
    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    /// Replace the filterable-fields configuration.
    pub fn configure_fields(&self, config: &FilterableFieldsConfig) {
        let mut guard = self
            .fields_config
            .write()
            .unwrap_or_else(|p| p.into_inner());
        *guard = Some(config.clone());
    }

    /// Snapshot the current fields configuration (clones for caller).
    pub fn fields_config_snapshot(&self) -> Option<FilterableFieldsConfig> {
        self.fields_config
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Build a `Fn(&str) -> bool` predicate that evaluates the standard
    /// TD-064 predicate set (tenant / RLS / time range / TTL) against
    /// cached metadata.
    ///
    /// Policy on missing metadata: **fail closed** — when the caller has
    /// supplied any policy-bearing predicate (tenant, RLS, time), records
    /// without cached metadata are excluded. With no predicate args the
    /// closure admits everything.
    pub fn build_predicate(
        &self,
        tenant_id: Option<&str>,
        time_range_ns: Option<(i64, i64)>,
        rls_tags: Option<&[String]>,
    ) -> impl Fn(&str) -> bool + Send + Sync + 'static {
        let metadata_cache = Arc::clone(&self.metadata);
        let tenant_owned = tenant_id.map(|s| s.to_string());
        let rls_owned: Option<Vec<String>> = rls_tags.map(|t| t.to_vec());
        let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(i64::MAX);

        move |id: &str| -> bool {
            let Some(meta) = metadata_cache.get(id) else {
                return tenant_owned.is_none() && rls_owned.is_none() && time_range_ns.is_none();
            };

            if let Some(ref t) = tenant_owned
                && !meta.matches_tenant(t)
            {
                return false;
            }

            if let Some((start, end)) = time_range_ns
                && !meta.matches_time_range(start, end)
            {
                return false;
            }

            if !meta.is_valid_at_time(now_ns) {
                return false;
            }

            if let Some(ref tags) = rls_owned
                && !tags.is_empty()
                && !tags.iter().any(|tag| meta.matches_rls_tag(tag))
            {
                return false;
            }

            true
        }
    }

    /// Snapshot the cache into a sorted vector (for serialization).
    pub fn snapshot(&self) -> Vec<(String, FilterableHnswMetadata)> {
        self.metadata
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// Restore cache from a snapshot (for deserialization).
    pub fn restore(&self, entries: Vec<(String, FilterableHnswMetadata)>) {
        for (id, meta) in entries {
            self.metadata.insert(id, meta);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_records::ProximaRecord;

    #[test]
    fn test_extract_filterable_metadata() {
        let mut record = ProximaRecord::default();
        record.tenant_id = "tenant_123".to_string();
        record.created_at_ns = 1_700_000_000_000_000_000;
        record.props.insert(
            "status".to_string(),
            ProximaTreeNode::Value(ProximaValue::String("active".to_string())),
        );
        record.props.insert(
            "score".to_string(),
            ProximaTreeNode::Value(ProximaValue::Float64(0.95)),
        );
        record.props.insert(
            "verified".to_string(),
            ProximaTreeNode::Value(ProximaValue::Boolean(true)),
        );
        record.props.insert(
            "category".to_string(),
            ProximaTreeNode::Value(ProximaValue::String("electronics".to_string())),
        );

        let config = FilterableFieldsConfig {
            int_fields: vec![],
            float_fields: vec!["score".to_string()],
            bool_fields: vec!["verified".to_string()],
            string_fields: vec!["status".to_string(), "category".to_string()],
        };

        let metadata = extract_filterable_metadata(&record, &config);

        assert_eq!(metadata.tenant_id, Some("tenant_123".to_string()));
        assert_eq!(metadata.created_at_ns, 1_700_000_000_000_000_000);
        assert_eq!(metadata.typed_attrs.get_string("status"), Some("active"));
        assert_eq!(metadata.typed_attrs.get_float("score"), Some(0.95));
        assert_eq!(metadata.typed_attrs.get_bool("verified"), Some(true));
        assert_eq!(
            metadata.typed_attrs.get_string("category"),
            Some("electronics")
        );
    }

    #[test]
    fn test_matches_tenant() {
        let mut metadata = FilterableHnswMetadata::default();
        metadata.tenant_id = Some("tenant_123".to_string());

        assert!(metadata.matches_tenant("tenant_123"));
        assert!(!metadata.matches_tenant("tenant_456"));
        assert!(!metadata.matches_tenant("")); // No tenant_id doesn't match empty
    }

    #[test]
    fn test_matches_time_range() {
        let mut metadata = FilterableHnswMetadata::default();
        metadata.created_at_ns = 1000;

        assert!(metadata.matches_time_range(500, 1500));
        assert!(!metadata.matches_time_range(1500, 2000)); // Before range
        assert!(!metadata.matches_time_range(500, 800)); // After range
    }

    #[test]
    fn test_ttl_validation() {
        let mut metadata = FilterableHnswMetadata::default();
        metadata.expires_at_ns = Some(1000);

        assert!(metadata.is_valid_at_time(500)); // Before expiration
        assert!(!metadata.is_valid_at_time(1500)); // After expiration

        metadata.expires_at_ns = None;
        assert!(metadata.is_valid_at_time(9999)); // No expiration = always valid
    }

    #[test]
    fn test_rls_tag_matching() {
        let mut metadata = FilterableHnswMetadata::default();
        metadata.rls_tags = vec!["admin".to_string(), "user".to_string()];

        assert!(metadata.matches_rls_tag("admin"));
        assert!(metadata.matches_rls_tag("user"));
        assert!(!metadata.matches_rls_tag("guest"));
    }

    #[test]
    fn test_string_too_long() {
        let mut attrs = TypedAttributes::default();
        let long_string = "a".repeat(100);

        let result = attrs.set_string("field".to_string(), long_string);
        assert!(matches!(result, Err(MetadataError::StringTooLong(100))));
    }

    #[test]
    fn test_estimated_size() {
        let mut metadata = FilterableHnswMetadata::default();
        metadata.tenant_id = Some("tenant_123".to_string());
        metadata.rls_tags = vec!["tag1".to_string(), "tag2".to_string()];
        metadata.typed_attrs.set_int("count".to_string(), 42);

        let size = metadata.estimated_size_bytes();
        // Should be relatively small (<100 bytes for this data)
        assert!(size < 200, "Metadata size {} exceeds threshold", size);
    }

    fn cached_metadata(tenant: &str, tags: Vec<&str>) -> FilterableHnswMetadata {
        let mut m = FilterableHnswMetadata::default();
        m.tenant_id = Some(tenant.to_string());
        m.rls_tags = tags.into_iter().map(String::from).collect();
        m
    }

    #[test]
    fn cache_predicate_admits_matching_tenant() {
        let cache = FilterableMetadataCache::new();
        cache.insert("v1".to_string(), cached_metadata("acme", vec![]));
        let pred = cache.build_predicate(Some("acme"), None, None);
        assert!(pred("v1"));
    }

    #[test]
    fn cache_predicate_excludes_other_tenant() {
        let cache = FilterableMetadataCache::new();
        cache.insert("v1".to_string(), cached_metadata("acme", vec![]));
        let pred = cache.build_predicate(Some("other"), None, None);
        assert!(!pred("v1"));
    }

    #[test]
    fn cache_predicate_fails_closed_on_missing_metadata_with_tenant_set() {
        let cache = FilterableMetadataCache::new();
        // Note: no insert for "missing-id"
        let pred = cache.build_predicate(Some("acme"), None, None);
        assert!(!pred("missing-id"));
    }

    #[test]
    fn cache_predicate_admits_missing_metadata_with_no_predicate_args() {
        let cache = FilterableMetadataCache::new();
        let pred = cache.build_predicate(None, None, None);
        assert!(pred("missing-id"));
    }

    #[test]
    fn cache_predicate_enforces_rls_tag_match() {
        let cache = FilterableMetadataCache::new();
        cache.insert("v1".to_string(), cached_metadata("acme", vec!["admin"]));
        cache.insert("v2".to_string(), cached_metadata("acme", vec!["user"]));
        let required = vec!["admin".to_string()];
        let pred = cache.build_predicate(Some("acme"), None, Some(&required));
        assert!(pred("v1"));
        assert!(!pred("v2"));
    }

    #[test]
    fn cache_snapshot_restore_roundtrip_preserves_entries() {
        let cache_a = FilterableMetadataCache::new();
        cache_a.insert("v1".to_string(), cached_metadata("acme", vec!["admin"]));
        cache_a.insert("v2".to_string(), cached_metadata("acme", vec![]));
        let snap = cache_a.snapshot();
        assert_eq!(snap.len(), 2);

        let cache_b = FilterableMetadataCache::new();
        cache_b.restore(snap);
        assert_eq!(cache_b.len(), 2);
        let pred = cache_b.build_predicate(Some("acme"), None, None);
        assert!(pred("v1"));
        assert!(pred("v2"));
    }

    #[test]
    fn cache_remove_drops_entry() {
        let cache = FilterableMetadataCache::new();
        cache.insert("v1".to_string(), cached_metadata("acme", vec![]));
        assert_eq!(cache.len(), 1);
        cache.remove("v1");
        assert!(cache.is_empty());
    }
}
