//! Search result types

use proximadb_data_model::ProximaValue;
use proximadb_distance_kernel::engine::SimilarityResult;
use proximadb_proto::proximadb_v1::{SearchVectorRecord, SourceContent};
pub use proximadb_records::is_record_dead;
// Canonical ranking score types — see roadmap/RANKING_FRAMEWORK_SPEC_2026_05_23.md (R-0).
pub use proximadb_kernel::{PhaseId, ScoreComponent, ScoreVector};
use std::collections::HashMap;
use std::sync::Arc;

/// Convert a v1 SqlValue to the canonical ProximaValue.
/// Magic prefix for JSONB-tagged BytesValue. Two bytes not valid in UTF-8
/// so they will never appear at the start of raw binary data from users.
const JSONB_MAGIC: &[u8] = proximadb_data_model::JSONB_LEGACY_MAGIC;

/// Used at WAL/gRPC boundary deserialization so the rest of the system
/// works entirely with ProximaValue.
pub fn sql_value_to_proxima_value(v: proximadb_proto::proximadb_v1::SqlValue) -> ProximaValue {
    use proximadb_proto::proximadb_v1::sql_value::Value;
    match v.value {
        Some(Value::StringValue(s)) => ProximaValue::String(s),
        Some(Value::NumberValue(f)) => ProximaValue::Float64(f),
        Some(Value::Int64Value(i)) => ProximaValue::Int64(i),
        Some(Value::BoolValue(b)) => ProximaValue::Boolean(b),
        Some(Value::BytesValue(b)) => {
            if b.starts_with(JSONB_MAGIC) {
                // JSONB-tagged bytes — decode the JSON payload after the magic prefix
                let json_bytes = &b[JSONB_MAGIC.len()..];
                match serde_json::from_slice(json_bytes) {
                    Ok(j) => ProximaValue::Jsonb(j),
                    Err(_) => ProximaValue::Binary(b),
                }
            } else {
                ProximaValue::Binary(b)
            }
        }
        // types.proto tag 9: JSONB by declaration — canonical decode-or-binary
        // via the shared helper so a value compares equal across read paths.
        Some(Value::JsonbValue(b)) => ProximaValue::from_jsonb_or_binary(&b),
        Some(Value::ObjectValue(obj)) => ProximaValue::Map(
            obj.fields
                .into_iter()
                .map(|(k, v)| (k, sql_value_to_proxima_value(v)))
                .collect(),
        ),
        Some(Value::ArrayValue(arr)) => ProximaValue::Array(
            arr.values
                .into_iter()
                .map(sql_value_to_proxima_value)
                .collect(),
        ),
        Some(Value::NullValue(_)) | None => ProximaValue::Null,
    }
}

/// Convert a canonical ProximaValue back to v1 SqlValue for WAL/gRPC writes.
pub fn proxima_value_to_sql_value(v: ProximaValue) -> proximadb_proto::proximadb_v1::SqlValue {
    use proximadb_proto::proximadb_v1::{SqlArray, SqlObject, SqlValue, sql_value::Value};
    let inner = match v {
        ProximaValue::String(s) | ProximaValue::Symbol(s) => Value::StringValue(s),
        ProximaValue::Float32(f) => Value::NumberValue(f as f64),
        ProximaValue::Float64(f) => Value::NumberValue(f),
        ProximaValue::Int8(i) => Value::Int64Value(i as i64),
        ProximaValue::Int16(i) => Value::Int64Value(i as i64),
        ProximaValue::Int32(i) => Value::Int64Value(i as i64),
        ProximaValue::Int64(i) => Value::Int64Value(i),
        ProximaValue::UInt8(i) => Value::Int64Value(i as i64),
        ProximaValue::UInt16(i) => Value::Int64Value(i as i64),
        ProximaValue::UInt32(i) => Value::Int64Value(i as i64),
        ProximaValue::UInt64(i) => Value::Int64Value(i as i64),
        ProximaValue::Boolean(b) => Value::BoolValue(b),
        ProximaValue::Binary(b) => Value::BytesValue(b),
        ProximaValue::Map(m) => Value::ObjectValue(SqlObject {
            fields: m
                .into_iter()
                .map(|(k, v)| (k, proxima_value_to_sql_value(v)))
                .collect(),
        }),
        ProximaValue::Struct(m) => Value::ObjectValue(SqlObject {
            fields: m
                .into_iter()
                .map(|(k, v)| (k, proxima_value_to_sql_value(v)))
                .collect(),
        }),
        ProximaValue::Array(arr) => Value::ArrayValue(SqlArray {
            values: arr.into_iter().map(proxima_value_to_sql_value).collect(),
        }),
        ProximaValue::Json(json) => {
            // JSON stored as StringValue — readable and queryable at the gRPC layer
            Value::StringValue(json.to_string())
        }
        ProximaValue::Jsonb(json) => {
            // TD-PROTO-2 round 4: emit the declared tag-9 variant with the
            // canonical MessagePack payload — the legacy tag-8 magic+JSON-text
            // encoding made the canonical decode, structured filters, and every
            // JsonbValue render arm unreachable for engine search responses.
            // The tag-8 magic form is still DECODED (below) for results
            // serialized before this change.
            match ProximaValue::to_jsonb_vec(&json) {
                Ok(bytes) => Value::JsonbValue(bytes),
                Err(_) => {
                    // Preserve the established fallback encoding under its
                    // legacy tag. Putting magic+JSON bytes under tag 9 would
                    // violate tag 9's MessagePack contract and decode as
                    // opaque binary.
                    let mut bytes = JSONB_MAGIC.to_vec();
                    bytes.extend_from_slice(json.to_string().as_bytes());
                    Value::BytesValue(bytes)
                }
            }
        }
        ProximaValue::Null => {
            return SqlValue { value: None };
        }
        // Temporal / UUID / vector types: fall back to string representation
        other => Value::StringValue(format!("{:?}", other)),
    };
    SqlValue { value: Some(inner) }
}

/// Convert a SqlValue metadata map to a ProximaValue metadata map.
pub fn sql_map_to_proxima(
    map: HashMap<String, proximadb_proto::proximadb_v1::SqlValue>,
) -> HashMap<String, ProximaValue> {
    map.into_iter()
        .map(|(k, v)| (k, sql_value_to_proxima_value(v)))
        .collect()
}

/// Convert a ProximaValue metadata map back to SqlValue for protocol edges.
pub fn proxima_map_to_sql(
    map: HashMap<String, ProximaValue>,
) -> HashMap<String, proximadb_proto::proximadb_v1::SqlValue> {
    map.into_iter()
        .map(|(k, v)| (k, proxima_value_to_sql_value(v)))
        .collect()
}

// ---------------------------------------------------------------------------
// Multi-model record type support
// ---------------------------------------------------------------------------

/// Discriminant for the modality a search result represents.
///
/// Allows callers to interpret modality-specific fields in
/// [`OptimizedSearchRecord`] without probing the `metadata` map.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum RecordType {
    #[default]
    Vector,
    Document,
    Graph,
    Observability,
    TimeSeries,
}

/// Backwards-compat alias for [`SearchGraphEdge`].
pub type GraphEdge = SearchGraphEdge;

/// Directed or undirected edge in a graph result.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SearchGraphEdge {
    /// Target (or other endpoint) node identifier.
    pub neighbor_id: String,
    /// Edge label / relation type.
    pub edge_type: Option<String>,
    /// Optional edge weight (similarity, distance, or user-assigned).
    pub weight: Option<f32>,
    /// Direction: true = outgoing, false = incoming, None = undirected.
    pub outgoing: Option<bool>,
}

/// Observability context attached to a search result from a log/trace collection.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ObservabilityContext {
    pub trace_id: Option<String>,
    pub span_id: Option<String>,
    pub parent_span_id: Option<String>,
    pub service_name: Option<String>,
    pub log_level: Option<String>,
    /// Structured key-value tags from the originating observability system.
    pub tags: std::collections::HashMap<String, String>,
}

// MIGRATION COMPLETE: InternalSearchResult eliminated entirely
// All functionality moved to OptimizedSearchRecord for better performance

// Type definitions needed by OptimizedSearchRecord
/// Debug information attached to individual search results
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SearchDebugInfo {
    /// Storage engine that served this result
    pub engine_used: String,
    /// Time to retrieve this result in milliseconds
    pub search_time_ms: f64,
    /// Number of candidate vectors evaluated
    pub candidates_evaluated: usize,
}

/// Quantization information for a search result
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct QuantizationInfo {
    /// Quantization method used (e.g., "pq", "binary", "int8")
    pub quantization_type: String,
    /// Compression ratio achieved
    pub compression_ratio: f32,
}

/// Backwards-compat alias for [`SearchEngineStats`].
pub type EngineStats = SearchEngineStats;

/// Per-engine performance statistics for a search result
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SearchEngineStats {
    /// Number of vectors scanned
    pub vectors_scanned: usize,
    /// Number of cache hits during search
    pub cache_hits: usize,
    /// Number of I/O operations performed
    pub io_operations: usize,
}

/// Optimized search record — the unified result envelope for all ProximaDB modalities.
///
/// `record_type` indicates which modality-specific extension fields are populated.
/// The base fields (id, score, metadata, temporal) are common across all modalities.
/// Modality-specific groups:
/// - **Vector** – `vector`, `semantic_similarity`, `quantization_info`
/// - **Graph**   – `graph_edges`, `graph_degree`
/// - **Document** – `parent_doc_id`, `content_type`, `chunk_ordinal`
/// - **Observability** – `observability`
/// - **TimeSeries** – `series_id`, `time_bucket_ns`
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OptimizedSearchRecord {
    // --- Identity ---
    /// Vector/document/node identifier
    pub id: String,
    /// Alternative identifier field for compatibility
    pub vector_id: Option<String>,
    /// Modality discriminant — drives interpretation of extension fields.
    pub record_type: RecordType,

    // --- Ranking ---
    /// Similarity score (higher = more similar). When a rank profile is
    /// attached this mirrors `score_vector.primary`; readers that only need
    /// a scalar score keep working unchanged.
    pub score: f32,
    /// Distance value (lower = more similar, if different from score)
    pub similarity: Option<f32>,
    /// Multi-component score from the multi-phase ranking pipeline. `None`
    /// when no rank profile is attached — this keeps the no-profile path
    /// zero-cost on the wire (NFR-9 in the ranking framework spec).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_vector: Option<ScoreVector>,

    // --- Vector modality ---
    /// Original vector data (Arc avoids clone on fan-out)
    #[serde(skip)]
    pub vector: Option<Arc<Vec<f32>>>,
    /// Semantic distance/similarity breakdown
    pub semantic_similarity: Option<SimilarityResult>,
    /// Quantization information (PQ, binary, int8 …)
    pub quantization_info: Option<QuantizationInfo>,

    // --- Graph modality ---
    /// Adjacent edges/neighbors for graph result records.
    pub graph_edges: Option<Vec<SearchGraphEdge>>,
    /// In-degree + out-degree of the matched node.
    pub graph_degree: Option<u32>,

    // --- Document modality ---
    /// Parent document identifier for chunk-level results.
    pub parent_doc_id: Option<String>,
    /// MIME type or content-type label (e.g. "text/plain", "application/pdf").
    pub content_type: Option<String>,
    /// 0-based chunk ordinal within the parent document.
    pub chunk_ordinal: Option<u32>,

    // --- Observability modality ---
    /// Trace/span/service context for log and trace collection results.
    pub observability: Option<ObservabilityContext>,

    // --- TimeSeries modality ---
    /// Logical series / metric name this record belongs to.
    pub series_id: Option<String>,
    /// Aligned time bucket (nanoseconds since Unix epoch).
    pub time_bucket_ns: Option<i64>,

    // --- Shared structured metadata ---
    /// Canonical ProximaValue metadata map — replaces legacy SqlValue.
    /// At protocol edges (gRPC/REST) use `sql_map_to_proxima` / `proxima_map_to_sql`
    /// to convert between SqlValue (wire) and ProximaValue (internal).
    pub metadata: HashMap<String, ProximaValue>,

    // --- Temporal ---
    /// Version for MVCC
    pub version: Option<u32>,
    /// Record creation timestamp (ms since epoch)
    pub timestamp: Option<i64>,
    /// Last-update timestamp (ms since epoch)
    pub updated_at: Option<i64>,
    /// TTL expiration timestamp (ms since epoch). DISPLAY ONLY — unit-muddled
    /// (ms on the proto wire, secs when derived in the WAL path). Correctness
    /// (dead-record filtering) MUST use [`Self::valid_to_ns`] via
    /// [`Self::is_dead`], never this field.
    pub expires_at: Option<i64>,
    /// Validity end in nanoseconds (mirrors `ProximaRecord.valid_to_ns`).
    /// `Some(0)` = tombstone; `Some(v)` in the past = TTL-expired. This is
    /// the canonical field for dead-record filtering.
    pub valid_to_ns: Option<i64>,

    // --- Source / context ---
    /// Original source content (skipped from serde — too large for wire)
    #[serde(skip)]
    pub source: Option<SourceContent>,
    /// Expanded context chunks for RAG
    #[serde(skip)]
    pub expanded_context: Vec<SourceContent>,

    // --- Diagnostics ---
    /// Debug information for result
    pub debug_info: Option<SearchDebugInfo>,
    /// Engine-specific optimization stats
    pub engine_stats: Option<SearchEngineStats>,
    /// Index path for result tracking
    pub index_path: Option<String>,
}

// Serde derives handle serialization/deserialization automatically
// No custom implementations needed

impl OptimizedSearchRecord {
    /// Create a new OptimizedSearchRecord with just ID and score
    pub fn new(id: String, score: f32) -> Self {
        Self {
            id,
            score,
            ..Default::default()
        }
    }

    /// Returns true when this record must NOT appear in a read at `now_ns`
    /// (tombstone `valid_to_ns == Some(0)` or TTL-expired). Uses the
    /// canonical [`is_record_dead`] on `valid_to_ns` (ns) — never the
    /// unit-muddled `expires_at`.
    pub fn is_dead(&self, now_ns: i64) -> bool {
        is_record_dead(self.valid_to_ns, now_ns)
    }

    /// Standardized distance-to-similarity conversion for consistent ranking across all metrics
    /// This is a static method that can be used without an instance
    pub fn standardized_distance_to_similarity(
        distance: f32,
        metric: &proximadb_distance_kernel::DistanceMetric,
    ) -> f32 {
        use proximadb_distance_kernel::DistanceMetric;
        match metric {
            DistanceMetric::Cosine => {
                // Cosine distance is in [0, 2], similarity = 1 - distance/2 for normalized range
                if distance.is_infinite() {
                    0.0 // Zero vectors get worst similarity score
                } else {
                    1.0 - (distance / 2.0).clamp(0.0, 1.0)
                }
            }
            DistanceMetric::Euclidean => {
                // Euclidean distance is in [0, ∞), convert using 1/(1+d) for [0,1] range
                1.0 / (1.0 + distance)
            }
            DistanceMetric::DotProduct => {
                // Dot product similarity is already in similarity form (higher = better)
                // Just ensure it's in [0,1] range
                ((distance + 1.0) / 2.0).clamp(0.0, 1.0)
            }
            DistanceMetric::Manhattan => {
                // Manhattan distance is in [0, ∞), use exponential decay
                (-distance / 10.0).exp()
            }
            _ => {
                // Default conversion for other metrics
                1.0 / (1.0 + distance)
            }
        }
    }

    /// Create with ID, score and vector
    pub fn with_vector(id: String, score: f32, vector: Vec<f32>) -> Self {
        Self {
            id,
            score,
            vector: Some(Arc::new(vector)),
            ..Default::default()
        }
    }

    /// Create with ID, score and shared vector (Arc)
    pub fn with_arc_vector(id: String, score: f32, vector: Arc<Vec<f32>>) -> Self {
        Self {
            id,
            score,
            vector: Some(vector),
            ..Default::default()
        }
    }

    /// Builder method to add vector
    pub fn add_vector(mut self, vector: Vec<f32>) -> Self {
        self.vector = Some(Arc::new(vector));
        self
    }

    /// Builder: set metadata from legacy v1 SqlValue map (converts to ProximaValue internally).
    /// All existing call sites pass `HashMap<String, SqlValue>` — conversion happens here.
    pub fn with_metadata(
        mut self,
        metadata: std::collections::HashMap<String, proximadb_proto::proximadb_v1::SqlValue>,
    ) -> Self {
        self.metadata = sql_map_to_proxima(metadata);
        self
    }

    /// Builder: set metadata directly from canonical ProximaValue map (preferred for new code).
    pub fn with_proxima_metadata(mut self, metadata: HashMap<String, ProximaValue>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Builder method to add similarity
    pub fn with_similarity(mut self, similarity: f32) -> Self {
        self.similarity = Some(similarity);
        self
    }

    /// Builder method to add version info
    pub fn with_version_info(mut self, version: u32, timestamp: i64) -> Self {
        self.version = Some(version);
        self.timestamp = Some(timestamp);
        self
    }

    /// Builder method to add source content
    pub fn with_source(mut self, source: SourceContent) -> Self {
        self.source = Some(source);
        self
    }

    /// Builder method to attach a multi-component score from the ranking pipeline.
    /// Also mirrors `score_vector.primary` into the scalar `score` field so
    /// sort order is consistent regardless of which the reader inspects.
    pub fn with_score_vector(mut self, sv: ScoreVector) -> Self {
        self.score = sv.primary;
        self.score_vector = Some(sv);
        self
    }

    // REMOVED: from_internal and to_internal methods - InternalSearchResult eliminated
}

/// Collection of search results with metadata
/// Using Arc<[OptimizedSearchRecord]> for immutable, zero-copy sharing of results
#[derive(Debug, Clone)]
pub struct SearchResultSet {
    /// Individual search results - immutable for performance
    pub results: Arc<[OptimizedSearchRecord]>,
    /// Total number of matching documents (before pagination)
    pub total_count: u64,
    /// Query that generated these results
    pub query_id: Option<String>,
    /// Processing time for entire query (microseconds)
    pub processing_time_us: u64,
    /// Algorithm used for search
    pub algorithm: String,
    /// Additional query metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl SearchResultSet {
    /// Create a SearchResultSet from a Vec<OptimizedSearchRecord>
    pub fn from_vec(
        results: Vec<OptimizedSearchRecord>,
        total_count: u64,
        query_id: Option<String>,
        processing_time_us: u64,
        algorithm: String,
        metadata: HashMap<String, serde_json::Value>,
    ) -> Self {
        Self {
            results: Arc::from(results.into_boxed_slice()),
            total_count,
            query_id,
            processing_time_us,
            algorithm,
            metadata,
        }
    }

    /// Create an empty SearchResultSet
    pub fn empty(algorithm: String) -> Self {
        Self {
            results: Arc::from(Vec::new().into_boxed_slice()),
            total_count: 0,
            query_id: None,
            processing_time_us: 0,
            algorithm,
            metadata: HashMap::new(),
        }
    }
}

/// Helper module for serializing Arc<[T]>
mod arc_slice_serde {
    use super::OptimizedSearchRecord;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::sync::Arc;

    #[allow(dead_code)]
    pub fn serialize<S>(
        results: &Arc<[OptimizedSearchRecord]>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        results.as_ref().serialize(serializer)
    }

    #[allow(dead_code)]
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Arc<[OptimizedSearchRecord]>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let vec = Vec::<OptimizedSearchRecord>::deserialize(deserializer)?;
        Ok(Arc::from(vec.into_boxed_slice()))
    }
}

// Manual trait implementations for ordering (HashMap doesn't implement Ord)

// Native-to-proto conversion: lives here (not in the root `core::conversions`)
// because `OptimizedSearchRecord` is local to this crate — the orphan rule
// forbids `impl From<OptimizedSearchRecord> for SearchVectorRecord` in any crate
// where both types are foreign. Moved during the search-types extraction.
impl From<OptimizedSearchRecord> for SearchVectorRecord {
    fn from(native: OptimizedSearchRecord) -> Self {
        SearchVectorRecord {
            id: native.id,
            score: native.score as f64,
            vector: native
                .vector
                .as_ref()
                .map(|v| (**v).clone())
                .unwrap_or_default(),
            metadata: proxima_map_to_sql(native.metadata),
            version: native.version,
            similarity: native.similarity,
            timestamp: native.timestamp,
            source: native.source.as_ref().map(|sc| match &sc.data {
                Some(proximadb_proto::proximadb_v1::source_content::Data::TextContent(text)) => {
                    text.clone()
                }
                Some(proximadb_proto::proximadb_v1::source_content::Data::ExternalReference(
                    url,
                )) => url.clone(),
                Some(proximadb_proto::proximadb_v1::source_content::Data::BinaryContent(_)) => {
                    "[Binary Content]".to_string()
                }
                None => "[Empty Content]".to_string(),
            }),
            expanded_context: native
                .expanded_context
                .iter()
                .map(|sc| match &sc.data {
                    Some(proximadb_proto::proximadb_v1::source_content::Data::TextContent(
                        text,
                    )) => text.clone(),
                    Some(
                        proximadb_proto::proximadb_v1::source_content::Data::ExternalReference(url),
                    ) => url.clone(),
                    Some(proximadb_proto::proximadb_v1::source_content::Data::BinaryContent(_)) => {
                        "[Binary Content]".to_string()
                    }
                    None => "[Empty Content]".to_string(),
                })
                .collect(),
            semantic_similarity: native
                .semantic_similarity
                .as_ref()
                .map(|s| s.similarity_score),
            quantization_info: native
                .quantization_info
                .as_ref()
                .map(|q| format!("{:?}", q)),
            engine_stats: native
                .engine_stats
                .as_ref()
                .map(|stats| {
                    std::collections::HashMap::from_iter([
                        (
                            "vectors_scanned".to_string(),
                            stats.vectors_scanned.to_string(),
                        ),
                        ("io_operations".to_string(), stats.io_operations.to_string()),
                        ("cache_hits".to_string(), stats.cache_hits.to_string()),
                    ])
                })
                .unwrap_or_default(),
            index_path: None,
        }
    }
}

impl From<&OptimizedSearchRecord> for SearchVectorRecord {
    fn from(native: &OptimizedSearchRecord) -> Self {
        SearchVectorRecord {
            id: native.id.clone(),
            score: native.score as f64,
            vector: native
                .vector
                .as_ref()
                .map(|v| (**v).clone())
                .unwrap_or_default(),
            metadata: proxima_map_to_sql(native.metadata.clone()),
            version: native.version,
            similarity: native.similarity,
            timestamp: native.timestamp,
            source: native.source.as_ref().map(|sc| match &sc.data {
                Some(proximadb_proto::proximadb_v1::source_content::Data::TextContent(text)) => {
                    text.clone()
                }
                Some(proximadb_proto::proximadb_v1::source_content::Data::ExternalReference(
                    url,
                )) => url.clone(),
                Some(proximadb_proto::proximadb_v1::source_content::Data::BinaryContent(_)) => {
                    "[Binary Content]".to_string()
                }
                None => "[Empty Content]".to_string(),
            }),
            expanded_context: native
                .expanded_context
                .iter()
                .map(|sc| match &sc.data {
                    Some(proximadb_proto::proximadb_v1::source_content::Data::TextContent(
                        text,
                    )) => text.clone(),
                    Some(
                        proximadb_proto::proximadb_v1::source_content::Data::ExternalReference(url),
                    ) => url.clone(),
                    Some(proximadb_proto::proximadb_v1::source_content::Data::BinaryContent(_)) => {
                        "[Binary Content]".to_string()
                    }
                    None => "[Empty Content]".to_string(),
                })
                .collect(),
            semantic_similarity: native
                .semantic_similarity
                .as_ref()
                .map(|s| s.similarity_score),
            quantization_info: native
                .quantization_info
                .as_ref()
                .map(|q| format!("{:?}", q)),
            engine_stats: native
                .engine_stats
                .as_ref()
                .map(|stats| {
                    std::collections::HashMap::from_iter([
                        (
                            "vectors_scanned".to_string(),
                            stats.vectors_scanned.to_string(),
                        ),
                        ("io_operations".to_string(), stats.io_operations.to_string()),
                        ("cache_hits".to_string(), stats.cache_hits.to_string()),
                    ])
                })
                .unwrap_or_default(),
            index_path: None,
        }
    }
}

#[cfg(test)]
mod score_vector_tests {
    //! R-0 tests for the ScoreVector promotion into OptimizedSearchRecord.
    //! See roadmap/RANKING_FRAMEWORK_SPEC_2026_05_23.md §6.1.
    use super::*;

    #[test]
    fn jsonb_writes_the_declared_tag9_variant_on_search_results() {
        let document = serde_json::json!({"kind": "fact", "active": true});

        let sql = proxima_value_to_sql_value(ProximaValue::Jsonb(document.clone()));

        let Some(proximadb_proto::proximadb_v1::sql_value::Value::JsonbValue(bytes)) = sql.value
        else {
            panic!("JSONB must use SqlValue.jsonb_value (tag 9)");
        };
        assert_eq!(
            ProximaValue::from_jsonb_or_binary(&bytes),
            ProximaValue::Jsonb(document)
        );
    }

    #[test]
    fn record_default_has_no_score_vector() {
        let r = OptimizedSearchRecord::default();
        assert!(r.score_vector.is_none());
    }

    #[test]
    fn record_serde_omits_score_vector_when_none() {
        // NFR-9: zero-cost when no profile attached — payload must not carry
        // a `score_vector` key for records without one.
        let r = OptimizedSearchRecord::new("doc1".to_string(), 0.5);
        let j = serde_json::to_string(&r).unwrap();
        assert!(
            !j.contains("score_vector"),
            "score_vector must be omitted when None: {j}"
        );
    }

    #[test]
    fn record_with_score_vector_round_trips() {
        let sv = ScoreVector::new(
            0.87,
            PhaseId::GLOBAL,
            vec![
                ScoreComponent {
                    name: "bm25(title)".into(),
                    value: 12.4,
                    weight: 0.4,
                    contribution: 4.96,
                },
                ScoreComponent {
                    name: "model(rerank-v3)".into(),
                    value: 0.87,
                    weight: 1.0,
                    contribution: 0.87,
                },
            ],
        );
        let r = OptimizedSearchRecord::new("doc1".into(), 0.0).with_score_vector(sv.clone());
        let j = serde_json::to_string(&r).unwrap();
        let back: OptimizedSearchRecord = serde_json::from_str(&j).unwrap();
        assert_eq!(back.score_vector.as_ref(), Some(&sv));
    }

    #[test]
    fn with_score_vector_mirrors_primary_into_score() {
        let sv = ScoreVector::from_primary(0.42, PhaseId::SECOND);
        let r = OptimizedSearchRecord::new("doc1".into(), 0.0).with_score_vector(sv);
        // Sort-order field must equal the score vector's primary.
        assert_eq!(r.score, 0.42);
        assert_eq!(r.score_vector.as_ref().unwrap().primary, 0.42);
    }

    #[test]
    fn score_component_re_export_is_usable() {
        // Confirms the re-export path src::core::search::results::ScoreComponent
        // resolves to the same type as proximadb_kernel::ScoreComponent.
        let c: ScoreComponent = proximadb_kernel::ScoreComponent {
            name: "x".into(),
            value: 1.0,
            weight: 1.0,
            contribution: 1.0,
        };
        assert_eq!(c.name, "x");
    }

    /// TD-SEARCH-1 score-scale contract, as an executable assertion.
    ///
    /// The cross-stage merges (`merge_delta_with_directory_results`, the
    /// two-stage dedup sort) rank `OptimizedSearchRecord.score` values produced
    /// by DIFFERENT stages: the WAL delta scan + memtable Stage-1 score via
    /// `SimilarityResult::new(d, metric).normalized_score` (distance-kernel),
    /// while the SST PAX cascade scores via
    /// `standardized_distance_to_similarity(d, metric)` (this crate). Those two
    /// normalizations MUST be the same function of (distance, metric), or
    /// records from one stage systematically displace the other's in the merged
    /// top-k regardless of true distance — the TD-RECALL-1 failure mode (recall
    /// collapsed to the memtable fraction when the delta re-score ran on a
    /// different scale). If either formula changes, change the other AND every
    /// producer, or better: collapse them into one shared function.
    #[test]
    fn cross_stage_score_normalizations_agree_per_metric() {
        use proximadb_distance_kernel::DistanceMetric;
        use proximadb_distance_kernel::engine::SimilarityResult;

        // Canonical-distance samples per metric, spanning exact-match to far.
        let cases: &[(DistanceMetric, &[f32])] = &[
            (DistanceMetric::Euclidean, &[0.0, 0.5, 1.8, 137.0, 5000.0]),
            (DistanceMetric::Cosine, &[0.0, 0.05, 0.5, 1.0, 2.0]),
            (DistanceMetric::Manhattan, &[0.0, 1.0, 10.0, 300.0]),
        ];
        for (metric, distances) in cases {
            for &d in *distances {
                let kernel = SimilarityResult::new(d, *metric).normalized_score;
                let search_types =
                    OptimizedSearchRecord::standardized_distance_to_similarity(d, metric);
                assert!(
                    (kernel - search_types).abs() < 1e-6,
                    "score-scale contract violated for {metric:?} at d={d}: \
                     kernel SimilarityResult={kernel} vs \
                     standardized_distance_to_similarity={search_types}"
                );
            }
        }
    }

    /// Same contract, monotonicity leg: within one metric the normalized score
    /// must be strictly decreasing in canonical distance (higher = closer)
    /// across that metric's valid distance domain, so a descending-score sort
    /// ranks by true distance for records from ANY stage that honors the
    /// contract. (Cosine distance lives in [0, 2] — the conversion clamps
    /// beyond that, so its samples stay inside the domain.)
    #[test]
    fn normalized_score_is_monotonic_in_distance() {
        use proximadb_distance_kernel::DistanceMetric;
        let cases: &[(DistanceMetric, &[f32])] = &[
            (
                DistanceMetric::Euclidean,
                &[0.0, 0.1, 0.5, 1.0, 1.9, 10.0, 100.0],
            ),
            (DistanceMetric::Cosine, &[0.0, 0.1, 0.5, 1.0, 1.5, 1.9]),
            (
                DistanceMetric::Manhattan,
                &[0.0, 0.1, 0.5, 1.0, 1.9, 10.0, 100.0],
            ),
        ];
        for (metric, distances) in cases {
            let mut prev = f32::INFINITY;
            for &d in *distances {
                let s = OptimizedSearchRecord::standardized_distance_to_similarity(d, metric);
                assert!(
                    s < prev,
                    "{metric:?}: score not strictly decreasing at d={d} (s={s}, prev={prev})"
                );
                prev = s;
            }
        }
    }
}
