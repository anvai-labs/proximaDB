// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! TD-TRACE-2 S3 — the durable io_trace record envelope (ADR-066 D1).
//!
//! A trace record is a homogeneous **header** + a modality-discriminated
//! **payload** (a tagged union), *not* a flat superset of every field:
//!
//! ```text
//!  TraceEnvelope = { schema_version, writer_uuid, sequence, header, payload }
//!    header  (EVERY modality — the homogeneous co-design cost spine, billing-class):
//!            query_id, tenant, event_time, route(shape,backend), storage_profile,
//!            get/put/list/delete_ops, bytes_read/written, range_gets,
//!            footer_hits/misses, compute_ms{engine→ms}, setup/open/plan/emit/session_ms,
//!            table_open_hits/misses, egress_bytes
//!    payload (tagged by `modality`, internally homogeneous):
//!            Relational { plan geometry + exec[] + split/runtime-filter }
//!            VectorAnn  { centroid prune + logical striped read }
//!            Embedding  { KEU token counts }
//!            Generic    (no distinguishing modality signal)
//! ```
//!
//! **Billing is untouched.** The billing observer reads the in-memory
//! [`IoTraceSnapshot`] directly (`compute_ms` / `bytes_read` / `range_gets`), which
//! this refactor does NOT change — the envelope is only the *serialization view* at
//! the durable-sink boundary, so meter behavior stays byte-identical (ADR-027,
//! ADR-066 D1). The header simply re-exposes those same values for the warehouse.
//!
//! Forward compatibility: every field is `#[serde(default)]` (additive), and an
//! unrecognized `modality` tag from a newer writer deserializes to
//! [`TracePayload::Other`] so a consumer can skip it rather than fail.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::io_trace::{ExecOpTrace, IoTraceSnapshot, VectorAccessTrace};

/// Current envelope schema version. Bumped from the flat S1–S2b record (`1`) to the
/// header+payload envelope (`2`). Explicit — never elided by `serde(default)`.
pub const TRACE_ENVELOPE_SCHEMA_VERSION: u16 = 2;

/// One durable trace record: ingestion identity + homogeneous header + payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceEnvelope {
    /// Envelope schema version (additive-compatible; explicit, never defaulted away).
    pub schema_version: u16,
    /// Writer incarnation UUID — with `sequence`, the ingestion-event identity that
    /// de-duplicates retries in the warehouse (ADR-066 D1). Distinct from
    /// `header.query_id`, which is the analytical join key.
    pub writer_uuid: String,
    /// Monotonic per-incarnation record sequence.
    pub sequence: u64,
    /// The homogeneous header every modality emits (billing-class fields).
    pub header: TraceHeader,
    /// The modality-discriminated payload.
    pub payload: TracePayload,
}

/// The homogeneous header (ADR-066 D1). EVERY record carries it; billing reads only
/// header-class fields (`compute_ms`, `bytes_read`, `range_gets`, `egress_bytes`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceHeader {
    /// Stable per-query id (UUID v4) — the warehouse analytical join key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_id: Option<String>,
    /// Owning tenant (sink-supplied; the snapshot itself is tenant-agnostic).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    /// Record event time — unix epoch milliseconds (sink-supplied at serialize).
    pub event_time_unix_ms: u64,
    /// The served route `(shape_class, backend_label)` — the top dispatch dimension.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route: Option<(String, String)>,
    /// Resolved per-collection storage profile (`append_bulk`/`churn`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage_profile: Option<String>,
    #[serde(default)]
    pub get_ops: u64,
    #[serde(default)]
    pub put_ops: u64,
    #[serde(default)]
    pub list_ops: u64,
    #[serde(default)]
    pub delete_ops: u64,
    #[serde(default)]
    pub bytes_read: u64,
    #[serde(default)]
    pub bytes_written: u64,
    #[serde(default)]
    pub range_gets: u64,
    #[serde(default)]
    pub footer_hits: u64,
    #[serde(default)]
    pub footer_misses: u64,
    /// In-process DRAM cache outcomes. Additive and mixed-read-safe.
    #[serde(default)]
    pub survivor_l1_hits: u64,
    #[serde(default)]
    pub survivor_l1_misses: u64,
    /// Persistent local-disk L2 cache probe outcomes (ADR-085 / TD-IOTRACE-4).
    /// `serde(default)` keeps pre-existing envelopes readable (mixed-read-safe).
    #[serde(default)]
    pub l2_hits: u64,
    #[serde(default)]
    pub l2_misses: u64,
    /// Compute wall ms attributed per engine (`datafusion`/`native`/`volcano`).
    /// A billing-class field — read verbatim by the KRU meter.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub compute_ms: BTreeMap<String, u64>,
    #[serde(default)]
    pub setup_ms: u64,
    #[serde(default)]
    pub open_ms: u64,
    #[serde(default)]
    pub plan_ms: u64,
    #[serde(default)]
    pub emit_ms: u64,
    #[serde(default)]
    pub session_ms: u64,
    #[serde(default)]
    pub table_open_hits: u64,
    #[serde(default)]
    pub table_open_misses: u64,
    /// Chargeable egress bytes (KOU) — a billing-class field.
    #[serde(default)]
    pub egress_bytes: u64,
}

/// The modality-discriminated payload (ADR-066 D1 tagged union). Internally
/// homogeneous per variant, so each maps to one warehouse satellite table.
/// `#[serde(other)]` makes an unrecognized `modality` tag (a newer writer's
/// modality) deserialize to [`TracePayload::Other`] — skippable, not an error.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "modality", rename_all = "snake_case")]
pub enum TracePayload {
    /// OLAP / pgwire relational query: plan geometry, per-op exec actuals, and
    /// split-pruning / runtime-filter outcomes.
    Relational(RelationalPayload),
    /// Vector-ANN search over the PAX format: centroid block-prune + logical
    /// striped-read projection.
    VectorAnn(VectorAnnPayload),
    /// Embedding op: KEU token counts (KEU itself is metered from provider usage).
    Embedding(EmbeddingPayload),
    /// No distinguishing modality signal (point lookup, DDL, empty snapshot).
    Generic,
    /// Forward-compat: an unrecognized `modality` from a newer writer. Deserialize
    /// only — consumers skip it.
    #[serde(other)]
    Other,
}

/// Relational (OLAP/pgwire) modality payload — plan geometry (TD-EXEC-2), per-op
/// exec actuals (TD-TRACE-1 S2), and OLAP scan pruning outcomes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationalPayload {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub pinned_binds:
        BTreeMap<crate::io_trace::PinnedSourceFormat, crate::io_trace::PinnedBindTrace>,
    #[serde(default)]
    pub plan_depth: u64,
    #[serde(default)]
    pub plan_nodes: u64,
    #[serde(default)]
    pub plan_leaves: u64,
    #[serde(default)]
    pub plan_fanout: u64,
    #[serde(default)]
    pub plan_blocking: u64,
    /// Per-operator-kind histogram — bounded operator keywords, never SQL text.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub plan_ops: BTreeMap<String, u64>,
    #[serde(default)]
    pub stack_hwm_bytes: u64,
    /// Per-operator execution actuals in pre-order (EXPLAIN ANALYZE only). Each
    /// `ExecOpTrace.op` is a bounded operator keyword, never raw data/query text.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exec: Vec<ExecOpTrace>,
    #[serde(default)]
    pub splits_total: u64,
    #[serde(default)]
    pub splits_pruned: u64,
    #[serde(default)]
    pub runtime_filter_arrived: u64,
    #[serde(default)]
    pub runtime_filter_timed_out: u64,
    #[serde(default)]
    pub runtime_filter_wait_ms: u64,
}

/// Vector-ANN modality payload — PAX centroid block-prune (TD-RDSTRAT-5), the
/// logical striped-read projection (ADR-057 / TD-RDSTRAT-3), and the two-level
/// IVF coarse-probe outcome (TD-RDSTRAT-8). Each field defaults to 0 so an older
/// reader deserializes a newer writer's envelope losslessly.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorAnnPayload {
    /// One row per successfully completed vector operator is projected into the
    /// warehouse's normalized vector-access satellite. Defaulted for envelopes
    /// written before TD-XMODAL-4 S2.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub access_paths: Vec<VectorAccessTrace>,
    #[serde(default)]
    pub centroid_total_blocks: u64,
    #[serde(default)]
    pub centroid_pruned_blocks: u64,
    #[serde(default)]
    pub logical_striped_bytes: u64,
    #[serde(default)]
    pub logical_striped_gets: u64,
    /// TD-RDSTRAT-8 two-level IVF coarse probe: of `ivf_cells_total` persisted
    /// coarse centroids, `ivf_cells_probed` were ranked in RAM and
    /// `ivf_probed_rows` rows read across `ivf_fetch_rounds` coalesced Region-A
    /// ranged-reads. `ivf_whole_region_fallback` counts segments where the armed
    /// probe missed and the read fell back to the whole Region-A scan (the GET
    /// budget the probe exists to avoid). The dominant cloud cost term is GET
    /// round-trips — these make the probe's cut observable per-query.
    #[serde(default)]
    pub ivf_cells_total: u64,
    #[serde(default)]
    pub ivf_cells_probed: u64,
    #[serde(default)]
    pub ivf_probed_rows: u64,
    #[serde(default)]
    pub ivf_fetch_rounds: u64,
    #[serde(default)]
    pub ivf_whole_region_fallback: u64,
    /// Physical PAX Region-A (RaBitQ) / Region-B (SQ8) bytes fetched by the
    /// coarse probe (TD-RDSTRAT-8 PR-C1). Cache hits contribute zero.
    #[serde(default)]
    pub ivf_region_a_bytes: u64,
    #[serde(default)]
    pub ivf_region_b_bytes: u64,
}

/// Embedding modality payload — KEU token counts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingPayload {
    #[serde(default)]
    pub embedding_calls: u64,
    #[serde(default)]
    pub embedding_input_tokens: u64,
    #[serde(default)]
    pub embedding_output_tokens: u64,
}

impl TraceHeader {
    /// Extract the homogeneous header from a snapshot, injecting the sink-supplied
    /// `tenant` + `event_time_unix_ms` (the snapshot is tenant/time-agnostic). The
    /// billing-class fields (`compute_ms`, `bytes_read`, `range_gets`, `egress_bytes`)
    /// are copied verbatim — see `header_carries_billing_inputs_verbatim`.
    pub fn from_snapshot(
        snap: &IoTraceSnapshot,
        tenant: Option<&str>,
        event_time_unix_ms: u64,
    ) -> Self {
        Self {
            query_id: snap.query_id.clone(),
            tenant: tenant.map(str::to_string),
            event_time_unix_ms,
            route: snap.route.clone(),
            storage_profile: snap.storage_profile.clone(),
            get_ops: snap.get_ops,
            put_ops: snap.put_ops,
            list_ops: snap.list_ops,
            delete_ops: snap.delete_ops,
            bytes_read: snap.bytes_read,
            bytes_written: snap.bytes_written,
            range_gets: snap.range_gets,
            footer_hits: snap.footer_hits,
            footer_misses: snap.footer_misses,
            survivor_l1_hits: snap.survivor_l1_hits,
            survivor_l1_misses: snap.survivor_l1_misses,
            l2_hits: snap.l2_hits,
            l2_misses: snap.l2_misses,
            compute_ms: snap.compute_ms.clone(),
            setup_ms: snap.setup_ms,
            open_ms: snap.open_ms,
            plan_ms: snap.plan_ms,
            emit_ms: snap.emit_ms,
            session_ms: snap.session_ms,
            table_open_hits: snap.table_open_hits,
            table_open_misses: snap.table_open_misses,
            egress_bytes: snap.egress_bytes,
        }
    }
}

impl TracePayload {
    /// Classify the snapshot's dominant modality and pack its payload. ANN signals
    /// (PAX centroid/striped) take precedence over relational plan geometry — a
    /// plain relational query never touches PAX centroid prune, so their presence is
    /// the distinctive vector-engine marker; embedding is the last-resort signal.
    /// Observe-only best-effort: a cross-modal query resolves to one dominant tag
    /// (the header still carries every universal counter regardless).
    pub fn classify(snap: &IoTraceSnapshot) -> Self {
        if snap.centroid_total_blocks > 0
            || snap.logical_striped_gets > 0
            || snap.logical_striped_bytes > 0
            || snap.ivf_cells_total > 0
            || snap.ivf_whole_region_fallback > 0
            || snap.ivf_region_a_bytes > 0
            || snap.ivf_region_b_bytes > 0
            || !snap.vector_accesses.is_empty()
        {
            TracePayload::VectorAnn(VectorAnnPayload {
                access_paths: snap.vector_accesses.clone(),
                centroid_total_blocks: snap.centroid_total_blocks,
                centroid_pruned_blocks: snap.centroid_pruned_blocks,
                logical_striped_bytes: snap.logical_striped_bytes,
                logical_striped_gets: snap.logical_striped_gets,
                ivf_cells_total: snap.ivf_cells_total,
                ivf_cells_probed: snap.ivf_cells_probed,
                ivf_probed_rows: snap.ivf_probed_rows,
                ivf_fetch_rounds: snap.ivf_fetch_rounds,
                ivf_whole_region_fallback: snap.ivf_whole_region_fallback,
                ivf_region_a_bytes: snap.ivf_region_a_bytes,
                ivf_region_b_bytes: snap.ivf_region_b_bytes,
            })
        } else if snap.plan_nodes > 0
            || snap.plan_depth > 0
            || !snap.exec_ops.is_empty()
            || snap.splits_total > 0
            || snap.plan_ms > 0
            || !snap.pinned_binds.is_empty()
        {
            TracePayload::Relational(RelationalPayload {
                pinned_binds: snap.pinned_binds.clone(),
                plan_depth: snap.plan_depth,
                plan_nodes: snap.plan_nodes,
                plan_leaves: snap.plan_leaves,
                plan_fanout: snap.plan_fanout,
                plan_blocking: snap.plan_blocking,
                plan_ops: snap.plan_ops.clone(),
                stack_hwm_bytes: snap.stack_hwm_bytes,
                exec: snap.exec_ops.clone(),
                splits_total: snap.splits_total,
                splits_pruned: snap.splits_pruned,
                runtime_filter_arrived: snap.runtime_filter_arrived,
                runtime_filter_timed_out: snap.runtime_filter_timed_out,
                runtime_filter_wait_ms: snap.runtime_filter_wait_ms,
            })
        } else if snap.embedding_calls > 0
            || snap.embedding_input_tokens > 0
            || snap.embedding_output_tokens > 0
        {
            TracePayload::Embedding(EmbeddingPayload {
                embedding_calls: snap.embedding_calls,
                embedding_input_tokens: snap.embedding_input_tokens,
                embedding_output_tokens: snap.embedding_output_tokens,
            })
        } else {
            TracePayload::Generic
        }
    }

    /// The `modality` discriminant string (matches the serde tag).
    pub fn modality(&self) -> &'static str {
        match self {
            TracePayload::Relational(_) => "relational",
            TracePayload::VectorAnn(_) => "vector_ann",
            TracePayload::Embedding(_) => "embedding",
            TracePayload::Generic => "generic",
            TracePayload::Other => "other",
        }
    }
}

impl TraceEnvelope {
    /// Build the full envelope from a snapshot + the sink's per-record identity.
    pub fn from_snapshot(
        snap: &IoTraceSnapshot,
        tenant: Option<&str>,
        writer_uuid: &str,
        sequence: u64,
        event_time_unix_ms: u64,
    ) -> Self {
        Self {
            schema_version: TRACE_ENVELOPE_SCHEMA_VERSION,
            writer_uuid: writer_uuid.to_string(),
            sequence,
            header: TraceHeader::from_snapshot(snap, tenant, event_time_unix_ms),
            payload: TracePayload::classify(snap),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TD-IOTRACE-4: the header carries the persistent-L2 probe counters, and
    /// an envelope written BEFORE the fields existed still deserializes with
    /// them defaulted to zero (mixed-read-safe — never a flag-day).
    #[test]
    fn header_carries_l2_probes_and_legacy_json_defaults_to_zero() {
        let snap = IoTraceSnapshot {
            survivor_l1_hits: 7,
            survivor_l1_misses: 3,
            l2_hits: 5,
            l2_misses: 2,
            ..Default::default()
        };
        let h = TraceHeader::from_snapshot(&snap, Some("t"), 1);
        assert_eq!((h.survivor_l1_hits, h.survivor_l1_misses), (7, 3));
        assert_eq!((h.l2_hits, h.l2_misses), (5, 2));

        let legacy: TraceHeader = serde_json::from_str(r#"{"event_time_unix_ms":7}"#).unwrap();
        assert_eq!((legacy.survivor_l1_hits, legacy.survivor_l1_misses), (0, 0));
        assert_eq!((legacy.l2_hits, legacy.l2_misses), (0, 0));
    }

    fn relational_snapshot() -> IoTraceSnapshot {
        IoTraceSnapshot {
            query_id: Some("q-rel".to_string()),
            bytes_read: 4096,
            range_gets: 3,
            egress_bytes: 512,
            compute_ms: BTreeMap::from([("datafusion".to_string(), 12)]),
            plan_nodes: 7,
            plan_depth: 4,
            plan_ops: BTreeMap::from([("HashJoin".to_string(), 1)]),
            splits_total: 10,
            splits_pruned: 6,
            exec_ops: vec![ExecOpTrace {
                op: "HashJoinExec".to_string(),
                rows_in: 100,
                rows_out: 40,
                ms_self: 5,
                bytes: Some(2048),
                spill: false,
            }],
            ..Default::default()
        }
    }

    /// Billing input parity: the header re-exposes the billing-class fields
    /// (`compute_ms`, `bytes_read`, `range_gets`, `egress_bytes`) byte-identically.
    #[test]
    fn header_carries_billing_inputs_verbatim() {
        let snap = relational_snapshot();
        let h = TraceHeader::from_snapshot(&snap, Some("acme"), 1_700_000_000_000);
        assert_eq!(h.compute_ms, snap.compute_ms, "compute_ms verbatim (KRU)");
        assert_eq!(h.bytes_read, snap.bytes_read, "bytes_read verbatim");
        assert_eq!(h.range_gets, snap.range_gets, "range_gets verbatim");
        assert_eq!(
            h.egress_bytes, snap.egress_bytes,
            "egress_bytes verbatim (KOU)"
        );
        assert_eq!(h.tenant.as_deref(), Some("acme"));
        assert_eq!(h.query_id.as_deref(), Some("q-rel"));
    }

    #[test]
    fn classify_relational() {
        assert!(matches!(
            TracePayload::classify(&relational_snapshot()),
            TracePayload::Relational(_)
        ));
    }

    #[test]
    fn classify_vector_ann() {
        let snap = IoTraceSnapshot {
            centroid_total_blocks: 64,
            centroid_pruned_blocks: 50,
            ..Default::default()
        };
        match TracePayload::classify(&snap) {
            TracePayload::VectorAnn(p) => {
                assert_eq!(p.centroid_total_blocks, 64);
                assert_eq!(p.centroid_pruned_blocks, 50);
            }
            other => panic!("expected VectorAnn, got {other:?}"),
        }
    }

    #[test]
    fn vector_access_alone_classifies_and_round_trips_durably() {
        use crate::io_trace::{VectorAccessPath, VectorCachePolicy, VectorSearchIntent};

        let access = VectorAccessTrace {
            engine: "sst".to_string(),
            dimensions: 384,
            top_k: 10,
            has_filter: true,
            requested_mode: VectorSearchIntent::Adaptive,
            actual_path: VectorAccessPath::Exact,
            storage_scope: crate::io_trace::VectorStorageScope::Local,
            cache_policy: VectorCachePolicy::Disabled,
        };
        let snap = IoTraceSnapshot {
            vector_accesses: vec![access.clone()],
            ..Default::default()
        };
        let payload = TracePayload::classify(&snap);
        let json = serde_json::to_string(&payload).unwrap();
        let round_trip: TracePayload = serde_json::from_str(&json).unwrap();
        match round_trip {
            TracePayload::VectorAnn(vector) => assert_eq!(vector.access_paths, vec![access]),
            other => panic!("expected VectorAnn, got {other:?}"),
        }
    }

    /// ANN signals win over co-present relational plan geometry (vector-SQL hybrid).
    #[test]
    fn classify_ann_precedence_over_relational() {
        let mut snap = relational_snapshot();
        snap.centroid_total_blocks = 32; // add ANN signal alongside the plan
        assert!(matches!(
            TracePayload::classify(&snap),
            TracePayload::VectorAnn(_)
        ));
    }

    #[test]
    fn classify_embedding() {
        let snap = IoTraceSnapshot {
            embedding_calls: 2,
            embedding_input_tokens: 500,
            ..Default::default()
        };
        assert!(matches!(
            TracePayload::classify(&snap),
            TracePayload::Embedding(_)
        ));
    }

    #[test]
    fn classify_packs_ivf_coarse_probe_into_vector_ann() {
        // TD-RDSTRAT-8: an IVF coarse-probe outcome (no centroid/striped signal)
        // must classify as VectorAnn and carry all five durable fields, so the
        // warehouse satellite captures the probe's GET cut per query.
        let snap = IoTraceSnapshot {
            ivf_cells_total: 64,
            ivf_cells_probed: 8,
            ivf_probed_rows: 4096,
            ivf_fetch_rounds: 3,
            ivf_whole_region_fallback: 1,
            ivf_region_a_bytes: 200_000,
            ivf_region_b_bytes: 800_000,
            ..Default::default()
        };
        match TracePayload::classify(&snap) {
            TracePayload::VectorAnn(v) => {
                assert_eq!(v.ivf_cells_total, 64);
                assert_eq!(v.ivf_cells_probed, 8);
                assert_eq!(v.ivf_probed_rows, 4096);
                assert_eq!(v.ivf_fetch_rounds, 3);
                assert_eq!(v.ivf_whole_region_fallback, 1);
                assert_eq!(v.ivf_region_a_bytes, 200_000);
                assert_eq!(v.ivf_region_b_bytes, 800_000);
            }
            other => panic!("expected VectorAnn, got {other:?}"),
        }
    }

    #[test]
    fn classify_generic_for_empty() {
        assert!(matches!(
            TracePayload::classify(&IoTraceSnapshot::default()),
            TracePayload::Generic
        ));
    }

    /// Round-trip: an envelope serializes to JSON and deserializes back identically.
    #[test]
    fn envelope_round_trips() {
        let snap = relational_snapshot();
        let env =
            TraceEnvelope::from_snapshot(&snap, Some("acme"), "writer-1", 7, 1_700_000_000_000);
        let json = serde_json::to_string(&env).unwrap();
        let back: TraceEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(env, back);
        assert_eq!(back.schema_version, TRACE_ENVELOPE_SCHEMA_VERSION);
        assert_eq!(back.payload.modality(), "relational");
    }

    /// The tagged payload serializes with a `modality` discriminant + inner fields.
    #[test]
    fn payload_serializes_with_modality_tag() {
        let json = serde_json::to_value(TracePayload::classify(&relational_snapshot())).unwrap();
        assert_eq!(json["modality"], "relational");
        assert_eq!(json["plan_nodes"], 7);
        assert_eq!(json["splits_pruned"], 6);
    }

    /// Forward compatibility: an unknown `modality` tag from a newer writer
    /// deserializes to `Other` rather than erroring (skippable payload).
    #[test]
    fn unknown_modality_deserializes_to_other() {
        let payload: TracePayload =
            serde_json::from_str(r#"{"modality":"rank_fusion","top_k":5}"#).unwrap();
        assert!(matches!(payload, TracePayload::Other));
        // And a full envelope carrying an unknown payload still parses.
        let env_json = r#"{"schema_version":9,"writer_uuid":"w","sequence":0,
            "header":{"event_time_unix_ms":1},"payload":{"modality":"future","x":1}}"#;
        let env: TraceEnvelope = serde_json::from_str(env_json).unwrap();
        assert!(matches!(env.payload, TracePayload::Other));
    }

    /// Additive back-compat: a header from an older/newer writer missing optional
    /// fields still deserializes (every field is `serde(default)`).
    #[test]
    fn header_tolerates_missing_fields() {
        let h: TraceHeader = serde_json::from_str(r#"{"event_time_unix_ms":42}"#).unwrap();
        assert_eq!(h.event_time_unix_ms, 42);
        assert_eq!(h.bytes_read, 0);
        assert!(h.query_id.is_none());
    }

    /// No raw data/query/vector: the envelope's top-level shape is exactly the
    /// allowlist — a structural guard against a field addition leaking user data.
    #[test]
    fn envelope_shape_is_bounded() {
        let snap = relational_snapshot();
        let env = TraceEnvelope::from_snapshot(&snap, Some("acme"), "w", 0, 1);
        let v = serde_json::to_value(&env).unwrap();
        let obj = v.as_object().unwrap();
        let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
        keys.sort();
        assert_eq!(
            keys,
            vec![
                "header",
                "payload",
                "schema_version",
                "sequence",
                "writer_uuid"
            ]
        );
    }
}
