//! WAL v2 wire shapes for [`ProximaRecord`] and [`EmbeddingCell`].
//!
//! The legacy v1 wire path uses the custom `Serialize`/`Deserialize`
//! impls on [`EmbeddingCell`] (in `lib.rs`), which deliberately hard-
//! error on any non-`Fp32` `EmbeddingValues` variant. That keeps fp32-
//! only consumers safe across the rollout — see the docstring at
//! `impl Serialize for EmbeddingCell` for the full rationale.
//!
//! The v2 wire path needs to actually carry fp16 / bf16 / int8 records
//! to disk. This module defines `*V2` shadow structs that use natural
//! derived `Serialize`/`Deserialize`, so:
//!
//! ```text
//! ProximaRecordV2.embeddings: Vec<EmbeddingCellV2>
//! EmbeddingCellV2.values:     EmbeddingValues  // natural enum-aware serde
//! ```
//!
//! `EmbeddingValues` itself has a natural derived `Serialize`/`Deserialize`
//! (kebab-case enum) and the `half` crate is built with its `serde`
//! feature, so `Fp16(Vec<half::f16>)` and `Bf16(Vec<half::bf16>)`
//! serialize/deserialize bit-faithfully via bincode without going
//! through the v1 custom impl.
//!
//! The WAL bincode strategy switches between v1 and v2 wire shapes
//! at the `serialize_batch_with_v2_segment_header` boundary, gated on
//! `EmbeddingPrecisionConfig::cached().schema_v2_enabled`. v2 readers
//! peek the `PWAL` magic and parse the segment header before decoding
//! payloads as `Vec<ProximaRecordV2>`.

use serde::{Deserialize, Serialize};

use crate::{
    EdgeShape, EmbeddingCell, EmbeddingScalarType, EmbeddingValues, LabelSet, MemoryType,
    ProximaRecord, ProximaTree, TokenSequence, TypedRef,
};

/// v2 wire shape for [`EmbeddingCell`]. Mirrors the legacy layout
/// field-for-field but uses the natural derived `Serialize` /
/// `Deserialize` so `values` round-trips through bincode as
/// `EmbeddingValues` (enum-aware) instead of `Vec<f32>` (the v1
/// custom impl's hard-coded shape).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingCellV2 {
    pub model_id: String,
    pub modality: String,
    pub values: EmbeddingValues,
    pub dim: u32,
    pub precision: EmbeddingScalarType,
    pub precision_epoch: Option<u64>,
}

impl From<&EmbeddingCell> for EmbeddingCellV2 {
    fn from(c: &EmbeddingCell) -> Self {
        Self {
            model_id: c.model_id.clone(),
            modality: c.modality.clone(),
            values: c.values.clone(),
            dim: c.dim,
            precision: c.precision,
            precision_epoch: c.precision_epoch,
        }
    }
}

impl From<EmbeddingCellV2> for EmbeddingCell {
    fn from(c: EmbeddingCellV2) -> Self {
        Self {
            model_id: c.model_id,
            modality: c.modality,
            values: c.values,
            dim: c.dim,
            precision: c.precision,
            precision_epoch: c.precision_epoch,
        }
    }
}

/// v2 wire shape for [`ProximaRecord`]. Differs from the legacy
/// derived serde only in `embeddings: Vec<EmbeddingCellV2>`. Every
/// other field is identical in layout, so a v1 record and its v2
/// shadow round-trip bit-faithfully when the embeddings are fp32.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProximaRecordV2 {
    // === Identity ===
    pub oid: String,
    pub local_id: Option<String>,
    pub tid: Option<u64>,

    // === Catalog / version ===
    pub variation_id: Option<String>,
    pub record_version: u64,
    pub spec_version: u16,

    // === Tenancy + RLS ===
    pub tenant_id: String,
    pub permitted_principals: Vec<String>,
    pub rls_policy_id: Option<String>,

    // === Temporal ===
    pub created_at_ns: i64,
    pub updated_at_ns: i64,
    pub valid_from_ns: Option<i64>,
    pub valid_to_ns: Option<i64>,

    // === Provenance ===
    pub origin: Option<String>,
    pub actor: Option<String>,
    pub method: Option<String>,

    // === Agentic Memory ===
    pub memory_type: Option<MemoryType>,

    // === Properties + refs + edge ===
    pub props: ProximaTree,
    pub refs: Vec<TypedRef>,
    pub edge: Option<EdgeShape>,

    // === Embeddings (the only field whose serde shape differs from v1) ===
    pub embeddings: Vec<EmbeddingCellV2>,

    // === Token sequence + labels ===
    pub sequence: Option<TokenSequence>,
    pub labels: LabelSet,

    // === Branch identity (ADR-012, T3.1 Slice 2 — 2026-05-26) ===
    /// Branch identifier this record belongs to. Carried on the V2 wire so
    /// the future WAL filter (T3.1 Slice 3) can scope by branch. The
    /// runtime `ProximaRecord.branch_id` is `#[serde(skip)]` so only the
    /// V2 path persists it; V1 frames continue to load with `None`.
    pub branch_id: Option<String>,
}

impl From<&ProximaRecord> for ProximaRecordV2 {
    fn from(r: &ProximaRecord) -> Self {
        Self {
            oid: r.oid.clone(),
            local_id: r.local_id.clone(),
            tid: r.tid,
            variation_id: r.variation_id.clone(),
            record_version: r.record_version,
            spec_version: r.spec_version,
            tenant_id: r.tenant_id.clone(),
            permitted_principals: r.permitted_principals.clone(),
            rls_policy_id: r.rls_policy_id.clone(),
            created_at_ns: r.created_at_ns,
            updated_at_ns: r.updated_at_ns,
            valid_from_ns: r.valid_from_ns,
            valid_to_ns: r.valid_to_ns,
            origin: r.origin.clone(),
            actor: r.actor.clone(),
            method: r.method.clone(),
            memory_type: r.memory_type,
            props: r.props.clone(),
            refs: r.refs.clone(),
            edge: r.edge.clone(),
            embeddings: r.embeddings.iter().map(EmbeddingCellV2::from).collect(),
            sequence: r.sequence.clone(),
            labels: r.labels.clone(),
            branch_id: r.branch_id.clone(),
        }
    }
}

impl From<ProximaRecordV2> for ProximaRecord {
    fn from(r: ProximaRecordV2) -> Self {
        Self {
            // schema_version is `serde(skip)` on ProximaRecord — the
            // v2 reader stamps it post-decode via
            // `WalSerializerStrategy::deserialize_batch_with_schema_version`.
            schema_version: crate::schema_version::default_schema_version(),
            oid: r.oid,
            local_id: r.local_id,
            tid: r.tid,
            variation_id: r.variation_id,
            record_version: r.record_version,
            spec_version: r.spec_version,
            tenant_id: r.tenant_id,
            permitted_principals: r.permitted_principals,
            rls_policy_id: r.rls_policy_id,
            created_at_ns: r.created_at_ns,
            updated_at_ns: r.updated_at_ns,
            valid_from_ns: r.valid_from_ns,
            valid_to_ns: r.valid_to_ns,
            origin: r.origin,
            actor: r.actor,
            method: r.method,
            memory_type: r.memory_type,
            props: r.props,
            refs: r.refs,
            edge: r.edge,
            embeddings: r.embeddings.into_iter().map(EmbeddingCell::from).collect(),
            sequence: r.sequence,
            labels: r.labels,
            branch_id: r.branch_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp16_cell(values: &[f32]) -> EmbeddingCell {
        let f16s: Vec<half::f16> = values.iter().map(|&x| half::f16::from_f32(x)).collect();
        EmbeddingCell {
            model_id: "test".to_string(),
            modality: "dense_vector".to_string(),
            dim: f16s.len() as u32,
            values: EmbeddingValues::Fp16(f16s),
            precision: EmbeddingScalarType::Fp16,
            ..Default::default()
        }
    }

    #[test]
    fn fp16_cell_round_trips_through_v2_bincode_bit_exact() {
        let original = fp16_cell(&[1.0, -2.5, 65504.0, 0.0001]);
        let v2: EmbeddingCellV2 = (&original).into();

        // v2 bincode uses derived serde — natural enum-aware encoding.
        let bytes = bincode::serialize(&v2).expect("v2 bincode serialize");
        let decoded: EmbeddingCellV2 =
            bincode::deserialize(&bytes).expect("v2 bincode deserialize");

        let recovered: EmbeddingCell = decoded.into();
        match (&original.values, &recovered.values) {
            (EmbeddingValues::Fp16(orig), EmbeddingValues::Fp16(got)) => {
                assert_eq!(orig, got, "fp16 bit-exact round-trip");
            }
            (a, b) => panic!(
                "variants must match: orig={:?} got={:?}",
                a.scalar_type(),
                b.scalar_type()
            ),
        }
        assert_eq!(recovered.precision, EmbeddingScalarType::Fp16);
    }

    #[test]
    fn fp32_cell_round_trips_through_v2_bincode_bit_exact() {
        let original = EmbeddingCell::new_fp32("test", "dense_vector", 4, vec![1.0, 2.0, 3.0, 4.0]);
        let v2: EmbeddingCellV2 = (&original).into();
        let bytes = bincode::serialize(&v2).expect("v2 bincode serialize");
        let decoded: EmbeddingCellV2 = bincode::deserialize(&bytes).expect("decode");
        let recovered: EmbeddingCell = decoded.into();
        assert_eq!(original.values, recovered.values);
    }

    #[test]
    fn full_record_v2_round_trip_preserves_all_fields() {
        let mut original = ProximaRecord {
            oid: "rec-1".to_string(),
            local_id: Some("local-1".to_string()),
            tenant_id: "tenant-a".to_string(),
            record_version: 7,
            created_at_ns: 1_000_000,
            updated_at_ns: 2_000_000,
            origin: Some("test".to_string()),
            embeddings: vec![fp16_cell(&[1.0, 2.0, 3.0])],
            ..ProximaRecord::default()
        };
        // Set schema_version so we can verify it's STAMPED to V2 (the
        // `serde(skip)` field always defaults on decode; the WAL reader
        // is responsible for stamping the real version).
        original.schema_version = crate::schema_version::V2;

        let v2: ProximaRecordV2 = (&original).into();
        let bytes = bincode::serialize(&v2).expect("serialize");
        let decoded: ProximaRecordV2 = bincode::deserialize(&bytes).expect("deserialize");
        let recovered: ProximaRecord = decoded.into();

        // Field-by-field equality except schema_version (serde-skipped,
        // always defaults on decode — caller restamps).
        assert_eq!(recovered.oid, original.oid);
        assert_eq!(recovered.local_id, original.local_id);
        assert_eq!(recovered.tenant_id, original.tenant_id);
        assert_eq!(recovered.record_version, original.record_version);
        assert_eq!(recovered.created_at_ns, original.created_at_ns);
        assert_eq!(recovered.origin, original.origin);
        assert_eq!(recovered.embeddings.len(), 1);
        assert!(matches!(
            recovered.embeddings[0].values,
            EmbeddingValues::Fp16(_)
        ));
    }

    #[test]
    fn batch_of_mixed_precision_records_round_trips_each_variant() {
        let recs = [
            ProximaRecord {
                oid: "fp32-rec".to_string(),
                embeddings: vec![EmbeddingCell::new_fp32("", "", 2, vec![1.0, 2.0])],
                ..ProximaRecord::default()
            },
            ProximaRecord {
                oid: "fp16-rec".to_string(),
                embeddings: vec![fp16_cell(&[3.0, 4.0])],
                ..ProximaRecord::default()
            },
        ];
        let v2: Vec<ProximaRecordV2> = recs.iter().map(Into::into).collect();
        let bytes = bincode::serialize(&v2).expect("serialize batch");
        let decoded: Vec<ProximaRecordV2> =
            bincode::deserialize(&bytes).expect("deserialize batch");
        let recovered: Vec<ProximaRecord> = decoded.into_iter().map(Into::into).collect();

        assert_eq!(recovered.len(), 2);
        assert!(matches!(
            recovered[0].embeddings[0].values,
            EmbeddingValues::Fp32(_)
        ));
        assert!(matches!(
            recovered[1].embeddings[0].values,
            EmbeddingValues::Fp16(_)
        ));
    }

    // ── T3.1 Slice 2 — branch_id roundtrip tests ───────────────────────────

    #[test]
    fn branch_id_round_trips_through_v2_wire() {
        let original = ProximaRecord {
            oid: "node-1".to_string(),
            branch_id: Some("feature-x".to_string()),
            ..ProximaRecord::default()
        };
        let v2 = ProximaRecordV2::from(&original);
        let bytes = bincode::serialize(&v2).expect("serialize v2");
        let decoded: ProximaRecordV2 = bincode::deserialize(&bytes).expect("deserialize v2");
        let recovered: ProximaRecord = decoded.into();
        assert_eq!(recovered.branch_id.as_deref(), Some("feature-x"));
    }

    #[test]
    fn branch_id_survives_v1_bincode_roundtrip() {
        // TD-072 (2026-05-27): branch_id was originally `#[serde(skip)]` to
        // preserve the V1 bincode wire — but that silently dropped the
        // field through the canonical WAL (rmp_serde named-field) too,
        // structurally breaking the merge endpoint. The fix flipped the
        // attribute to `#[serde(default)]`. This test pins the new
        // contract: V1 bincode now carries branch_id end-to-end. No
        // production code bincode-serializes ProximaRecord directly, so
        // there's no on-disk artifact to migrate.
        let original = ProximaRecord {
            oid: "node-2".to_string(),
            branch_id: Some("dev".to_string()),
            ..ProximaRecord::default()
        };
        let bytes = bincode::serialize(&original).expect("serialize v1");
        let recovered: ProximaRecord = bincode::deserialize(&bytes).expect("deserialize v1");
        assert_eq!(recovered.branch_id.as_deref(), Some("dev"));
    }

    #[test]
    fn branch_id_none_default_for_v2_wire() {
        // Records without a branch_id should serialize with `branch_id = None`
        // and roundtrip cleanly.
        let original = ProximaRecord {
            oid: "node-3".to_string(),
            ..ProximaRecord::default()
        };
        let v2 = ProximaRecordV2::from(&original);
        let bytes = bincode::serialize(&v2).expect("serialize v2");
        let decoded: ProximaRecordV2 = bincode::deserialize(&bytes).expect("deserialize v2");
        let recovered: ProximaRecord = decoded.into();
        assert_eq!(recovered.branch_id, None);
    }
}
