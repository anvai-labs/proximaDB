// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! # Statistics Substrate — the agent-facing catalog producer (ADR-037)
//!
//! The [`StatisticsEnvelope`] is the **boundary object** between the OSS engine
//! (sole producer) and AnvaiOps (consumer). It is a versioned, modality-neutral
//! summary aggregated from PAX zone maps + streaming sketches at the
//! flush/compaction boundary (TD-174) — the engine never scans the corpus to
//! produce it, and an agent reads the tiny envelope instead of the data.
//!
//! ## What this type MUST NOT carry
//!
//! Units and distributions only. **No meaning and no money:** no descriptions,
//! glossary, data-quality grade, PII classification, per-account scope, or `$`.
//! Those are AnvaiOps policy (AnvaiOps ADR-0021). The engine attests a freshness
//! *fact* ([`Freshness`]) — a watermark — never an SLA. Keeping this invariant is
//! what lets AnvaiOps consume the envelope without the open-core boundary
//! (ADR-027/030 "OSS = mechanism; pricing/semantics = AnvaiOps policy") forking.
//!
//! The canonical wire contract is frozen at
//! `docs/12-design/contracts/statistics-envelope.v1.schema.json`; the golden
//! example is round-tripped by the test below and pinned by the AnvaiOps consumer.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub mod registry;
// `sketches` now lives in `proximadb-kernel` (TD-DECOMP-18); re-exported.
pub use proximadb_kernel::sketches;
mod summary;

pub use registry::{StatisticsRegistry, global as statistics_registry};

/// The statistics-registry key of a relational table as a tenant sees it:
/// `{tenant}/{table}` (absent/blank tenant = the default tenant). Two tenants'
/// same-named tables never share statistics. (Vector collections are keyed by
/// their globally unique collection id.)
pub fn table_statistics_key(tenant: Option<&str>, table: &str) -> String {
    format!(
        "{}/{}",
        proximadb_tenant::resolve_request_tenant(tenant),
        table
    )
}
pub use summary::{FieldAccumulator, StatisticsSummary};

/// The semver this build of the engine emits. Consumers pin the MAJOR
/// (ADR-0016 version-pin discipline) and reject anything outside it.
pub const ENVELOPE_VERSION: &str = "1.0.0";

/// Versioned, modality-neutral statistics for one collection (ADR-037 Decision 2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatisticsEnvelope {
    /// semver; engine-owned. See [`ENVELOPE_VERSION`].
    pub envelope_version: String,
    pub collection_id: String,
    pub freshness: Freshness,
    pub record_count: u64,
    pub storage_size_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index_size_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fields: Vec<FieldStatistics>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub modality: Vec<ModalityStatistics>,
}

/// Snapshot watermark — a FACT, not an SLA (advanced at flush/compaction).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Freshness {
    /// RFC3339 timestamp of the snapshot.
    pub as_of: String,
    /// `"flush"` | `"compaction"`.
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segment_watermark: Option<u64>,
}

/// Per-field distribution statistics aggregated from zone maps + sketches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldStatistics {
    pub name: String,
    /// Canonical [`proximadb_data_model::ProximaType`] (ADR-024) in serde form:
    /// a string for unit variants (`"String"`, `"Int64"`, …) or an object for
    /// parametrized variants (e.g. `{"DenseVector":{"element":"Float32","dim":768}}`).
    /// Held as [`Value`] at the wire-DTO layer; the extractor (TD-174) supplies
    /// `serde_json::to_value(&proxima_type)`.
    pub data_type: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub null_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distinct_estimate: Option<u64>,
    /// `"hll"` (approximate) | `"exact"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distinct_method: Option<String>,
    /// Zone-map minimum (typed value).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min: Option<Value>,
    /// Zone-map maximum (typed value).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantiles: Option<Quantiles>,
    /// Reservoir sample of typed values.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub indexed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filterable: Option<bool>,
    /// True if any statistic for this field is sketch-derived.
    #[serde(default)]
    pub approximate: bool,
}

/// Quantile/histogram summary for a field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Quantiles {
    /// `"tdigest"` | `"histogram"`.
    pub method: String,
    /// quantile -> value (e.g. `{"0.5": .., "0.9": ..}`).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub q: BTreeMap<String, f64>,
    /// Histogram buckets when `method == "histogram"` (engine-defined shape).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub buckets: Vec<Value>,
}

/// Per-modality statistics block — tagged union discriminated by `kind`
/// (ADR-037 Decision 3). Each block is an aggregation over an existing modality
/// engine; consumers that understand only [`StatisticsEnvelope::fields`] ignore it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModalityStatistics {
    Document(DocumentStatistics),
    Vector(VectorStatistics),
    Graph(GraphStatistics),
    Trace(TraceStatistics),
    Timeseries(TimeseriesStatistics),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocumentStatistics {
    pub doc_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_doc_length: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unique_terms_estimate: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub top_terms: Vec<TermStat>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TermStat {
    pub term: String,
    pub doc_frequency: u64,
    pub idf: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorStatistics {
    pub dimension: u32,
    pub distance_metric: String,
    /// Mean distance to centroid.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spread: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub centroid: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cluster_occupancy: Vec<u64>,
    /// Per-cluster centroid vectors, aligned **by index** with `cluster_occupancy`
    /// (TD-175, #494). Lets a consumer seed `fusion-search` once per cluster to build
    /// vector-cluster × graph-label co-statistics (the cluster partition the overall
    /// `centroid` cannot provide). Populated by the same clustering pass that fills
    /// `cluster_occupancy`. NOTE: `cluster_occupancy` is not yet fed from a clustering
    /// pass (the AXIS k-means in `src/index/axis/clustering.rs` is decoupled from this
    /// summary), so both stay empty until that wiring lands — this is the additive,
    /// back-compat contract slot.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cluster_centroids: Vec<Vec<f64>>,
    #[serde(default)]
    pub approximate: bool,
}

/// Aligns with the engine's `GraphStats` (`src/graph/model.rs`) for reuse (TD-175).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphStatistics {
    pub total_nodes: u64,
    pub total_edges: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub average_degree: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_degree: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub connected_components: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub label_counts: Vec<LabelCount>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub edge_type_counts: Vec<EdgeTypeCount>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabelCount {
    pub label: String,
    pub count: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdgeTypeCount {
    pub edge_type: String,
    pub count: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceStatistics {
    pub span_count: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub service_distribution: Vec<ServiceCount>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_depth: Option<DepthStat>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServiceCount {
    pub service: String,
    pub count: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DepthStat {
    pub p50: f64,
    pub p90: f64,
    pub max: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeseriesStatistics {
    pub point_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_range: Option<TimeRange>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub downsample_state: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: String,
    pub end: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frozen v1 golden example — the same artifact AnvaiOps pins as its
    /// consumer fixture (one source of truth across the open-core seam).
    const GOLDEN: &str =
        include_str!("../../../docs/12-design/contracts/statistics-envelope.v1.example.json");

    #[test]
    fn golden_example_matches_frozen_v1_contract() {
        let env: StatisticsEnvelope = serde_json::from_str(GOLDEN)
            .expect("golden example must parse into StatisticsEnvelope");

        // The example pins the frozen major.
        assert_eq!(
            env.envelope_version, ENVELOPE_VERSION,
            "golden example must pin v{ENVELOPE_VERSION}"
        );

        // Serde round-trip is stable at the struct level (independent of JSON
        // key ordering / optional omission).
        let value = serde_json::to_value(&env).expect("serialize");
        let env2: StatisticsEnvelope = serde_json::from_value(value).expect("re-parse");
        assert_eq!(env, env2, "serde round-trip must be stable");

        // The modality discriminator round-trips (graph block is present).
        assert!(
            env.modality
                .iter()
                .any(|m| matches!(m, ModalityStatistics::Graph(_))),
            "graph modality block must deserialize via the `kind` discriminator"
        );
    }

    #[test]
    fn vector_cluster_centroids_is_additive_and_back_compat() {
        // #494 / TD-175: per-cluster centroids are an additive, back-compat slot —
        // aligned by index with cluster_occupancy, omitted when empty, defaulted when
        // absent from older payloads (same gating as cluster_occupancy).
        let mut vs = VectorStatistics {
            dimension: 2,
            distance_metric: "cosine".into(),
            spread: None,
            centroid: vec![0.5, 0.5],
            cluster_occupancy: vec![3, 1],
            cluster_centroids: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            approximate: true,
        };

        // Round-trips, aligned by index with occupancy.
        let json = serde_json::to_string(&vs).expect("serialize");
        let back: VectorStatistics = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.cluster_centroids, vs.cluster_centroids);
        assert_eq!(back.cluster_centroids.len(), back.cluster_occupancy.len());

        // Empty ⇒ omitted from the wire (skip_serializing_if), like cluster_occupancy.
        vs.cluster_centroids = Vec::new();
        let json_empty = serde_json::to_string(&vs).expect("serialize");
        assert!(
            !json_empty.contains("cluster_centroids"),
            "empty cluster_centroids must be omitted"
        );

        // Absent from an older payload ⇒ defaults to empty (back-compat).
        let older = r#"{"dimension":2,"distance_metric":"cosine","approximate":false}"#;
        let parsed: VectorStatistics =
            serde_json::from_str(older).expect("older payload without the field must parse");
        assert!(
            parsed.cluster_centroids.is_empty(),
            "absent cluster_centroids defaults to empty"
        );
    }

    #[test]
    fn envelope_carries_units_not_meaning_or_money() {
        // Boundary-object invariant (ADR-037): the envelope is units only.
        // Semantics/pricing live in AnvaiOps (ADR-0021), never here.
        assert!(
            !GOLDEN.contains("\"description\""),
            "envelope must not carry semantic descriptions (consumer-owned, not the engine's concern)"
        );
        assert!(
            !GOLDEN.contains("\"price"),
            "envelope must not carry pricing (consumer-owned, not the engine's concern)"
        );
    }
}
