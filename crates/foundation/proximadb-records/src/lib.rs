/*
 * Copyright 2025 Vijaykumar Singh
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

//! # ProximaRecord — Unified Record Envelope (Phase B, TD-054)
//!
//! Authoritative Rust definition of the `ProximaRecord` envelope as specified
//! in MULTIMODAL_OVERHAUL_SPEC_2026_05_08.adoc §3.
//!
//! Every record stored in ProximaDB — vector, graph node/edge, document, relational
//! row, time-series sample, event, or observability span — projects onto this single
//! envelope. Modality-specific fields live in `props` (NF² tree), `edge`, and
//! `embeddings`. Cross-cutting concerns (identity, tenancy, temporal, provenance,
//! labels) are top-level fields so security and query layers can reach them without
//! deserialising opaque blobs.

use std::collections::HashMap;

use proximadb_data_model::MemoryType;
use serde::{Deserialize, Serialize};

/// Re-export so consumers can write `proximadb_records::ProximaValue`
/// without taking a direct dependency on `proximadb-data-model`.
/// Matches the existing re-export pattern this crate uses for other
/// data-model types.
pub use proximadb_data_model::ProximaValue;

pub mod conversions;
pub mod proto_v2;
pub mod store;
pub mod wire_v2;

pub use store::{
    RecordKey, RecordRecoveryOperation, RecordRecoverySummary, RecordScan, RecordScanOptions,
    RecordScanPredicate, RecordScanView, RecordStorage, RecordStore, RecordStoreResult,
    RecordWriteResult, replay_record_recovery_operations,
};

// ---------------------------------------------------------------------------
// NF² Property Tree
// ---------------------------------------------------------------------------

/// A node in a nested property tree (NF² JSONB analogue).
///
/// Each key maps to either a leaf `ProximaValue` or a nested sub-tree, allowing
/// arbitrary depth without a fixed schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProximaTreeNode {
    Value(ProximaValue),
    Object(ProximaTree),
}

/// Nested property map — the `props` field of [`ProximaRecord`].
pub type ProximaTree = HashMap<String, ProximaTreeNode>;

/// Look up a dot-separated path in a [`ProximaTree`].
///
/// Returns `None` if any segment is missing or the path traverses a leaf.
pub fn tree_get<'a>(tree: &'a ProximaTree, path: &str) -> Option<&'a ProximaValue> {
    let mut segments = path.splitn(2, '.');
    let head = segments.next()?;
    let tail = segments.next();

    match tree.get(head)? {
        ProximaTreeNode::Value(v) => {
            if tail.is_none() {
                Some(v)
            } else {
                None // path continues but this is a leaf
            }
        }
        ProximaTreeNode::Object(subtree) => match tail {
            Some(rest) => tree_get(subtree, rest),
            None => None, // path ends at an object node
        },
    }
}

// ---------------------------------------------------------------------------
// Supporting record components
// ---------------------------------------------------------------------------

/// Typed inter-record reference (spec §3 — refs field).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypedRef {
    /// Relational foreign-key reference.
    ForeignKey { table: String, id: String },
    /// Graph directed edge reference.
    GraphEdge {
        edge_id: String,
        direction: EdgeDirection,
    },
    /// Reference to an embedding model + vector index.
    Embedding { model_id: String },
}

/// Direction of a graph edge reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeDirection {
    Outgoing,
    Incoming,
}

/// Graph edge topology fields (present only when the record IS an edge).
///
/// Corresponds to the topology layer of the three-layer storage model (spec §5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdgeShape {
    pub source_id: String,
    pub target_id: String,
    pub edge_type: String,
    pub weight: Option<f64>,
}

/// Scalar storage type for embedding values. See
/// `docs/12-design/EMBEDDING_PRECISION_LLD_2026_05_22.adoc` for the canonical
/// byte tags and serialization shape.
///
/// Wire encoding:
/// * **Binary formats (bincode, etc.)** — emit the `#[repr(u8)]` value as a
///   single byte. Saves 3 bytes per cell vs. serde's default u32 variant
///   index. The LLD-locked tags (0x01..0x05) are the canonical wire IDs.
/// * **Human-readable formats (JSON, YAML, etc.)** — emit the snake_case
///   variant name. Same shape catalog rows + dashboards already consume.
///
/// The custom impls below replace the previous `derive(Serialize,
/// Deserialize)` which emitted a u32 (4 bytes) on the wire regardless of
/// `#[repr(u8)]`. INT-2.5b pre-step: lock the 1-byte tag before flipping
/// `EmbeddingCell.values` to `EmbeddingValues` so the cell's bincode shape
/// stays minimal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum EmbeddingScalarType {
    /// IEEE-754 f32, 4 bytes per element. Today's universal default.
    #[default]
    Fp32 = 0x01,
    /// IEEE-754 f16 (half-precision), 2 bytes per element.
    Fp16 = 0x02,
    /// Brain float16, 2 bytes per element. Reserved for Phase 6 hardware paths.
    Bf16 = 0x03,
    /// Signed 8-bit scalar with per-cell scale + zero-point. Lossy.
    Int8Scalar = 0x04,
    /// Unsigned 8-bit scalar with per-cell scale + zero-point. Lossy.
    UInt8Scalar = 0x05,
}

impl Serialize for EmbeddingScalarType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if serializer.is_human_readable() {
            serializer.serialize_str(match self {
                Self::Fp32 => "fp32",
                Self::Fp16 => "fp16",
                Self::Bf16 => "bf16",
                Self::Int8Scalar => "int8_scalar",
                Self::UInt8Scalar => "uint8_scalar",
            })
        } else {
            serializer.serialize_u8(*self as u8)
        }
    }
}

impl<'de> Deserialize<'de> for EmbeddingScalarType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            let s = <&str>::deserialize(deserializer)?;
            match s {
                "fp32" => Ok(Self::Fp32),
                "fp16" => Ok(Self::Fp16),
                "bf16" => Ok(Self::Bf16),
                "int8_scalar" => Ok(Self::Int8Scalar),
                "uint8_scalar" => Ok(Self::UInt8Scalar),
                other => Err(serde::de::Error::custom(format!(
                    "unknown EmbeddingScalarType: {other:?}"
                ))),
            }
        } else {
            let b = u8::deserialize(deserializer)?;
            match b {
                0x01 => Ok(Self::Fp32),
                0x02 => Ok(Self::Fp16),
                0x03 => Ok(Self::Bf16),
                0x04 => Ok(Self::Int8Scalar),
                0x05 => Ok(Self::UInt8Scalar),
                other => Err(serde::de::Error::custom(format!(
                    "unknown EmbeddingScalarType tag: 0x{other:02x}"
                ))),
            }
        }
    }
}

impl EmbeddingScalarType {
    /// Bytes-per-element for this scalar type (excluding per-cell metadata).
    pub fn bytes_per_element(self) -> usize {
        match self {
            Self::Fp32 => 4,
            Self::Fp16 | Self::Bf16 => 2,
            Self::Int8Scalar | Self::UInt8Scalar => 1,
        }
    }

    /// Whether this precision is lossy relative to the f32 source.
    pub fn is_lossy(self) -> bool {
        matches!(self, Self::Int8Scalar | Self::UInt8Scalar)
    }
}

/// Typed embedding payload. PR 1 of the precision rollout adds the enum
/// variants; only `Fp32` is exercised by existing call sites today. PR 3
/// switches durable storage to this enum.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EmbeddingValues {
    Fp32(Vec<f32>),
    Fp16(Vec<half::f16>),
    Bf16(Vec<half::bf16>),
    Int8Scalar {
        values: Vec<i8>,
        scale: f32,
        zero_point: i8,
    },
    UInt8Scalar {
        values: Vec<u8>,
        scale: f32,
        zero_point: u8,
    },
}

impl Default for EmbeddingValues {
    /// `Fp32(Vec::new())` so an empty default matches what `Vec<f32>`'s
    /// own default produced before INT-2.5b's field flip. Lets
    /// `#[derive(Default)]` continue to work on `EmbeddingCell` + lets
    /// callers spread `..Default::default()` over the struct without
    /// having to spell out an explicit variant.
    fn default() -> Self {
        Self::Fp32(Vec::new())
    }
}

impl EmbeddingValues {
    pub fn len(&self) -> usize {
        match self {
            Self::Fp32(v) => v.len(),
            Self::Fp16(v) => v.len(),
            Self::Bf16(v) => v.len(),
            Self::Int8Scalar { values, .. } => values.len(),
            Self::UInt8Scalar { values, .. } => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn scalar_type(&self) -> EmbeddingScalarType {
        match self {
            Self::Fp32(_) => EmbeddingScalarType::Fp32,
            Self::Fp16(_) => EmbeddingScalarType::Fp16,
            Self::Bf16(_) => EmbeddingScalarType::Bf16,
            Self::Int8Scalar { .. } => EmbeddingScalarType::Int8Scalar,
            Self::UInt8Scalar { .. } => EmbeddingScalarType::UInt8Scalar,
        }
    }

    /// On-disk byte size of the values payload (excluding the surrounding
    /// EmbeddingCell metadata). Drives storage budget enforcement.
    pub fn byte_size(&self) -> usize {
        let bytes = self.scalar_type().bytes_per_element();
        let scale_meta = match self {
            Self::Int8Scalar { .. } | Self::UInt8Scalar { .. } => 5, // 4B scale + 1B zero_point
            _ => 0,
        };
        self.len() * bytes + scale_meta
    }

    pub fn as_fp32_slice(&self) -> Option<&[f32]> {
        match self {
            Self::Fp32(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    pub fn to_fp32_owned(&self) -> Vec<f32> {
        match self {
            Self::Fp32(v) => v.clone(),
            Self::Fp16(v) => v.iter().map(|x| x.to_f32()).collect(),
            Self::Bf16(v) => v.iter().map(|x| x.to_f32()).collect(),
            Self::Int8Scalar {
                values,
                scale,
                zero_point,
            } => values
                .iter()
                .map(|x| (i32::from(*x) - i32::from(*zero_point)) as f32 * scale)
                .collect(),
            Self::UInt8Scalar {
                values,
                scale,
                zero_point,
            } => values
                .iter()
                .map(|x| (i32::from(*x) - i32::from(*zero_point)) as f32 * scale)
                .collect(),
        }
    }

    /// Downconvert an `&[f32]` to the requested scalar type. Symmetric for
    /// Int8Scalar (per-cell scale, zero_point=0), zero-point-aware for UInt8.
    pub fn from_fp32_lossy(src: &[f32], target: EmbeddingScalarType) -> Self {
        match target {
            EmbeddingScalarType::Fp32 => Self::Fp32(src.to_vec()),
            EmbeddingScalarType::Fp16 => {
                Self::Fp16(src.iter().map(|&x| half::f16::from_f32(x)).collect())
            }
            EmbeddingScalarType::Bf16 => {
                Self::Bf16(src.iter().map(|&x| half::bf16::from_f32(x)).collect())
            }
            EmbeddingScalarType::Int8Scalar => {
                let abs_max = src
                    .iter()
                    .fold(0.0_f32, |acc, &x| acc.max(x.abs()))
                    .max(f32::EPSILON);
                let scale = abs_max / 127.0;
                let values: Vec<i8> = src
                    .iter()
                    .map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8)
                    .collect();
                Self::Int8Scalar {
                    values,
                    scale,
                    zero_point: 0,
                }
            }
            EmbeddingScalarType::UInt8Scalar => {
                let min = src.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = src.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let range = (max - min).max(f32::EPSILON);
                let scale = range / 255.0;
                let zero_point: u8 = (-min / scale).round().clamp(0.0, 255.0) as u8;
                let values: Vec<u8> = src
                    .iter()
                    .map(|&x| ((x / scale).round() + zero_point as f32).clamp(0.0, 255.0) as u8)
                    .collect();
                Self::UInt8Scalar {
                    values,
                    scale,
                    zero_point,
                }
            }
        }
    }
}

/// A single embedding stored alongside a record (spec §3 — embeddings field).
///
/// One record can carry multiple `EmbeddingCell`s — one per model or modality.
/// The `values: Vec<f32>` field remains the durable canonical storage for PR 1;
/// `precision` and `precision_epoch` are advisory until PR 3 switches durable
/// storage to the `EmbeddingValues` enum.
///
/// `Default` derive: enables existing struct literals to spread
/// `..Default::default()` for the new precision fields without per-site code
/// migration. Defaults are `Fp32` and `None` — back-compat with PR 0 records.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EmbeddingCell {
    /// Identifier of the model that produced this embedding.
    pub model_id: String,
    /// Modality tag (e.g. "text", "image", "audio").
    pub modality: String,
    /// Typed vector payload. INT-2.5b flipped this from `Vec<f32>` to
    /// `EmbeddingValues` so non-fp32 precisions (fp16, bf16, int8) can
    /// flow through the ingest → WAL → memtable → PAX chain without
    /// being downconverted to fp32. The PR 1 advisory `precision` tag
    /// is now structurally enforced: `cell.precision ==
    /// cell.values.scalar_type()` is the invariant the custom serde
    /// impls + the v1 ingest validator (PR 3b) maintain together.
    pub values: EmbeddingValues,
    /// Declared dimensionality (must equal `values.len()` for dense vectors).
    pub dim: u32,
    /// Scalar storage precision. INT-2.5b: structurally implied by the
    /// active `values` variant; kept on the struct for ergonomics (some
    /// readers want the tag without matching the enum) and for the
    /// custom Serialize impl's wire layout.
    pub precision: EmbeddingScalarType,
    /// Precision epoch this cell was written under, when known.
    pub precision_epoch: Option<u64>,
}

// INT-2.5b step 2: custom Serialize/Deserialize impls. Today's behavior
// (values as raw Vec<f32>) is preserved byte-for-byte by the
// `embedding_cell_bincode_fixture_locks_pre_2_5b_layout` test in this
// crate's tests module. The custom impls replace the derived ones so
// that step 3 (field type flip Vec<f32> → EmbeddingValues) can change
// the in-memory shape without changing the on-disk shape.
//
// Field order locked by the fixture: model_id, modality, values, dim,
// precision, precision_epoch. Skip-serializing-if-None on
// precision_epoch is honored via the optional .skip_field branch.
impl Serialize for EmbeddingCell {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        // INT-2.5b Q1 (accepted recommendation): hard-error on non-Fp32
        // variants for the v1 bincode/JSON wire shape. The PR 3b ingest
        // validator makes this unreachable in production; converting the
        // bug into a loud error here means recovery / compaction /
        // cross-region replication paths that bypass the validator can't
        // silently downconvert + write fp32 bytes for a record the
        // catalog claims is fp16.
        //
        // The future v2 wire path (PWAL-prefixed segments from INT-2b)
        // will use a separate serializer that emits the natural enum
        // shape — that path's code lives in a different method on the
        // WAL serializer trait and bypasses this impl.
        let values_fp32: &[f32] = match &self.values {
            EmbeddingValues::Fp32(v) => v.as_slice(),
            other => {
                return Err(serde::ser::Error::custom(format!(
                    "v1 EmbeddingCell serializer refuses non-Fp32 variant {:?}: \
                     PR 3b ingest validator should have rejected this record. \
                     If this fires from a recovery/compaction/replication path, \
                     wire that path through the v2 serializer instead.",
                    other.scalar_type()
                )));
            }
        };

        // `precision_epoch` is the trailing field. bincode (and every other
        // non-self-describing format) is positional with no field tags, so the
        // decoder reads a FIXED field count. Skipping this field when `None`
        // desyncs the positional decoder: it reads the next record's bytes as
        // this cell's `precision_epoch` and then runs off the end of the buffer
        // ("unexpected end of file"), and it makes `Vec<EmbeddingCell>`
        // boundaries ambiguous. So for binary formats we ALWAYS emit
        // `precision_epoch` — this matches the design in
        // EMBEDDING_PRECISION_BRIDGE_BINCODE_MEMO §Step 2 (a `None` is a single
        // `0x00` Option-tag byte). For human-readable formats (JSON) we keep
        // skipping it so the key is omitted rather than rendered as `null`.
        let skip_epoch = serializer.is_human_readable() && self.precision_epoch.is_none();
        let field_count = if skip_epoch { 5 } else { 6 };
        let mut state = serializer.serialize_struct("EmbeddingCell", field_count)?;
        state.serialize_field("model_id", &self.model_id)?;
        state.serialize_field("modality", &self.modality)?;
        // Emit as a length-prefixed fp32 sequence — byte-identical to
        // what the pre-INT-2.5b `Vec<f32>` field's derived Serialize
        // produced. The fixture test locks this.
        state.serialize_field("values", values_fp32)?;
        state.serialize_field("dim", &self.dim)?;
        state.serialize_field("precision", &self.precision)?;
        if skip_epoch {
            state.skip_field("precision_epoch")?;
        } else {
            // Serialize the full `Option<u64>` (not the unwrapped value) so the
            // bincode bytes are symmetric with the `Option<u64>` field the
            // deserialize shadow reads back.
            state.serialize_field("precision_epoch", &self.precision_epoch)?;
        }
        state.end()
    }
}

impl<'de> Deserialize<'de> for EmbeddingCell {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Shadow struct mirrors the v1 wire shape exactly. `values` is
        // read as `Vec<f32>` (the legacy field type) and immediately
        // wrapped in `EmbeddingValues::Fp32(v)` because v1 wire format
        // always carries fp32.
        #[derive(Deserialize)]
        #[serde(rename = "EmbeddingCell")]
        struct Shadow {
            model_id: String,
            modality: String,
            values: Vec<f32>,
            dim: u32,
            #[serde(default)]
            precision: EmbeddingScalarType,
            #[serde(default)]
            precision_epoch: Option<u64>,
        }
        let s = Shadow::deserialize(deserializer)?;
        // INT-2.5b Q2 (accepted recommendation): stamp precision = Fp32
        // unconditionally on v1 deserialize. Enforces the invariant
        // `cell.precision == cell.values.scalar_type()` for every cell
        // returned by the v1 reader. The on-disk `precision` tag is
        // discarded — it was always advisory and the actual bytes are
        // always fp32 on the v1 wire. (No log: the records crate has no
        // tracing dep; if writers ever stamp non-Fp32 on a v1 record,
        // the bug will surface via the inevitable downstream mismatch.)
        let _ignored_on_disk_precision = s.precision;
        Ok(EmbeddingCell {
            model_id: s.model_id,
            modality: s.modality,
            values: EmbeddingValues::Fp32(s.values),
            dim: s.dim,
            precision: EmbeddingScalarType::Fp32,
            precision_epoch: s.precision_epoch,
        })
    }
}

impl EmbeddingCell {
    /// Construct an fp32 embedding cell with default precision metadata.
    pub fn new_fp32(
        model_id: impl Into<String>,
        modality: impl Into<String>,
        dim: u32,
        values: Vec<f32>,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            modality: modality.into(),
            values: EmbeddingValues::Fp32(values),
            dim,
            precision: EmbeddingScalarType::Fp32,
            precision_epoch: None,
        }
    }

    /// Construct a cell from already-typed values. The `precision` field
    /// is set to match the variant so the
    /// `cell.precision == cell.values.scalar_type()` invariant holds.
    pub fn new_typed(
        model_id: impl Into<String>,
        modality: impl Into<String>,
        dim: u32,
        values: EmbeddingValues,
    ) -> Self {
        let precision = values.scalar_type();
        Self {
            model_id: model_id.into(),
            modality: modality.into(),
            values,
            dim,
            precision,
            precision_epoch: None,
        }
    }

    /// Borrowed fp32 view. Returns `Some(&[f32])` only when the underlying
    /// variant is `Fp32`. For non-fp32 variants, callers must use
    /// `as_fp32_cow()` (one-shot promote) or `as_embedding_values()`
    /// (typed view).
    pub fn as_fp32_slice(&self) -> &[f32] {
        match &self.values {
            EmbeddingValues::Fp32(v) => v.as_slice(),
            // Non-fp32 path returns empty rather than panic — callers
            // that need the actual bytes should use as_fp32_cow().
            _ => &[],
        }
    }

    /// Borrowed-or-owned fp32 view of the cell's values.
    ///
    /// `Fp32` → `Cow::Borrowed` (zero copy). All other variants promote
    /// to fp32 once and return `Cow::Owned`. The 145+ read sites that
    /// just want fp32 bytes use this as their migration target.
    pub fn as_fp32_cow(&self) -> std::borrow::Cow<'_, [f32]> {
        match &self.values {
            EmbeddingValues::Fp32(v) => std::borrow::Cow::Borrowed(v.as_slice()),
            other => std::borrow::Cow::Owned(other.to_fp32_owned()),
        }
    }

    /// Total bytes the cell's values occupy in canonical storage.
    /// Uses the variant's `byte_size()` directly so it's correct for
    /// every precision (4 bytes/elem for fp32, 2 for fp16/bf16, etc.).
    pub fn values_byte_size(&self) -> usize {
        self.values.byte_size()
    }

    /// Clone the typed values. Cheap when the underlying Vec is the
    /// natural representation; semantically identical to `self.values.clone()`.
    pub fn as_embedding_values(&self) -> EmbeddingValues {
        self.values.clone()
    }

    /// Coerce this cell's values to `target` precision in place. No-op when
    /// `self.values.scalar_type() == target`. Goes through `fp32` as a
    /// pivot for non-trivial precision changes, which is bit-exact for the
    /// `fp16 → fp32 → fp16` round-trip (fp32 strictly dominates fp16, and
    /// `half::f16` ↔ `f32` is deterministic in both directions).
    ///
    /// The `precision` metadata is updated to match the new variant so the
    /// `cell.precision == cell.values.scalar_type()` invariant continues
    /// to hold.
    ///
    /// Intended for compaction read+rewrite paths where the in-flight
    /// intermediate representation went through fp32 (e.g. the legacy
    /// `VectorRecord` shape) and the writer needs to re-apply the
    /// collection's canonical precision before flushing the rewritten
    /// block. The `target` typically comes from
    /// `CanonicalPrecisionResolver::resolve` (PR INT-3-followup-b).
    pub fn coerce_to_precision(&mut self, target: EmbeddingScalarType) {
        if self.values.scalar_type() == target {
            return;
        }
        let fp32_owned = self.values.to_fp32_owned();
        self.values = EmbeddingValues::from_fp32_lossy(&fp32_owned, target);
        self.precision = target;
    }
}

/// Token sequence for LLM / event-stream records (spec §3 — sequence field).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenSequence {
    pub tokens: Vec<u32>,
    pub model_id: String,
    pub offset: u64,
}

/// Searchable label set (spec §3 — labels field).
///
/// Labels are stored as a `Vec<String>` and deduplicated on insert.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LabelSet(Vec<String>);

impl LabelSet {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// Insert a label. No-op if already present.
    pub fn insert(&mut self, label: impl Into<String>) {
        let label = label.into();
        if !self.0.contains(&label) {
            self.0.push(label);
        }
    }

    pub fn contains(&self, label: &str) -> bool {
        self.0.iter().any(|l| l == label)
    }

    pub fn iter(&self) -> impl Iterator<Item = &String> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<Vec<String>> for LabelSet {
    fn from(v: Vec<String>) -> Self {
        let mut set = LabelSet::new();
        for label in v {
            set.insert(label);
        }
        set
    }
}

// ---------------------------------------------------------------------------
// ProximaRecord — the unified envelope
// ---------------------------------------------------------------------------

/// Embedding-precision schema versions (Q17, LLD §schema-version-dispatch).
///
/// The WAL/PAX/record path tags each `ProximaRecord` with the schema version
/// under which it was written so readers can dispatch to the correct payload
/// decoder. `V1` is the legacy fp32-only shape; `V2` is the precision-aware
/// shape introduced by the embedding-precision rollout.
pub mod schema_version {
    /// Legacy schema: every `EmbeddingCell.values` is read as `Vec<f32>`.
    pub const V1: u8 = 1;
    /// Precision-aware schema: cells carry an explicit precision discriminant.
    pub const V2: u8 = 2;
    /// Current default for newly constructed records (PR 2: still V1; PR 3 will
    /// flip via `PROXIMADB_EMBED_PRECISION_SCHEMA_V2`).
    pub const CURRENT: u8 = V1;

    /// `serde(default = ...)` callback for `ProximaRecord.schema_version`. Old
    /// JSON payloads and freshly-defaulted records resolve to `V1` for
    /// backwards compatibility with PR 0 readers.
    pub fn default_schema_version() -> u8 {
        V1
    }
}

/// PR 3 §"Feature Flag and Rolling Deploy" — reject records that cannot be
/// represented in the schema-v1 wire shape.
///
/// Callers (WAL writer, API ingress handler) invoke this before serializing a
/// batch when the cluster is still on V1 (the default, see
/// `EmbeddingPrecisionConfig.schema_v2_enabled`). Any embedding cell whose
/// `precision` is not `Fp32` would silently lose data through the legacy fp32
/// path, so the check fails fast with the LLD-locked error tag.
///
/// Returns `Ok(())` if every embedding in every record is fp32-shaped (matches
/// what schema-v1 can durably represent without loss).
pub fn validate_records_for_schema_v1<'a, I>(records: I) -> Result<(), SchemaV1ValidationError>
where
    I: IntoIterator<Item = &'a ProximaRecord>,
{
    for record in records {
        for cell in &record.embeddings {
            if cell.precision != EmbeddingScalarType::Fp32 {
                return Err(SchemaV1ValidationError {
                    record_oid: record.oid.clone(),
                    model_id: cell.model_id.clone(),
                    found_precision: cell.precision,
                });
            }
        }
    }
    Ok(())
}

/// Failure variant for [`validate_records_for_schema_v1`].
///
/// The `Display` text starts with the LLD-locked tag
/// `unsupported_precision_schema_v1_only` so operators and CI can grep for it
/// across logs and SDK error responses.
#[derive(Debug, Clone, PartialEq)]
pub struct SchemaV1ValidationError {
    pub record_oid: String,
    pub model_id: String,
    pub found_precision: EmbeddingScalarType,
}

impl std::fmt::Display for SchemaV1ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unsupported_precision_schema_v1_only: record {} embedding (model={}) \
             declared precision {:?}, but PROXIMADB_EMBED_PRECISION_SCHEMA_V2 is off",
            self.record_oid, self.model_id, self.found_precision,
        )
    }
}

impl std::error::Error for SchemaV1ValidationError {}

/// The unified record envelope for all ProximaDB modalities.
///
/// Implements spec §3 of MULTIMODAL_OVERHAUL_SPEC_2026_05_08.adoc. Every stored
/// record — regardless of modality — projects onto this shape. Modality-specific
/// properties live in `props`. The `edge` field populates topology storage only
/// for graph edge records.
///
/// # RLS contract (spec §8)
///
/// `tenant_id` and `permitted_principals` are **record fields**, not context
/// fields. Engine-level scan iterators use them for row-level security without
/// referencing request context beyond the predicate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProximaRecord {
    // === Embedding-precision schema version (Q17, LLD §schema-version-dispatch) ===
    /// Embedding-precision schema that wrote this record.
    ///
    /// * `schema_version::V1` — legacy: every `EmbeddingCell.values` is read as
    ///   `Vec<f32>` and `EmbeddingCell.precision = Fp32`.
    /// * `schema_version::V2` — precision-aware: cells carry an explicit precision
    ///   discriminant and (in PR 3+) a durable typed payload.
    ///
    /// Not serialized: bincode WAL frames are positional and on-disk format does
    /// not change in PR 2. The WAL reader stamps this field after deserialization
    /// based on the segment header (PR 4) or the legacy default `V1`. JSON and
    /// proto bridges populate it from external metadata.
    #[serde(skip, default = "schema_version::default_schema_version")]
    pub schema_version: u8,

    // === Identity ===
    /// Globally unique object identifier (UUID or ULID string).
    pub oid: String,
    /// Caller-supplied local identifier (e.g. user-visible slug).
    pub local_id: Option<String>,
    /// Storage-layer tuple identifier for fast physical lookups.
    pub tid: Option<u64>,

    // === Catalog / version ===
    /// Schema variation identifier (U-Schema structural variant tag).
    pub variation_id: Option<String>,
    /// Monotonically increasing record version for OCC.
    pub record_version: u64,
    /// Spec/format version for forward compatibility.
    pub spec_version: u16,

    // === Tenancy + RLS (engine-level — spec §8) ===
    /// Owning tenant. Empty string = single-tenant / no isolation.
    pub tenant_id: String,
    /// ACL: principals allowed to read this record. Empty = unrestricted.
    pub permitted_principals: Vec<String>,
    /// ID of the RLS policy to evaluate at scan time, if any.
    pub rls_policy_id: Option<String>,

    // === Temporal ===
    /// Creation time (nanoseconds since Unix epoch).
    pub created_at_ns: i64,
    /// Last-update time (nanoseconds since Unix epoch).
    pub updated_at_ns: i64,
    /// Bi-temporal valid-from (nanoseconds since Unix epoch).
    pub valid_from_ns: Option<i64>,
    /// Bi-temporal valid-to (nanoseconds since Unix epoch).
    pub valid_to_ns: Option<i64>,

    // === Provenance ===
    /// Source system or connector that produced this record.
    pub origin: Option<String>,
    /// Principal that authored this record.
    pub actor: Option<String>,
    /// Ingestion method (e.g. "api", "cdc", "migration").
    pub method: Option<String>,

    // === Agentic Memory (Memanto — TD-055) ===
    /// High-fidelity memory category.
    pub memory_type: Option<MemoryType>,

    // === Properties (NF² tree) ===
    /// Modality-specific and user-defined properties.
    pub props: ProximaTree,

    // === Cross-modal references ===
    pub refs: Vec<TypedRef>,

    // === Graph topology (None for non-edge records) ===
    pub edge: Option<EdgeShape>,

    // === Embeddings (per-model, per-modality) ===
    pub embeddings: Vec<EmbeddingCell>,

    // === Token sequence (LLM / event streams) ===
    pub sequence: Option<TokenSequence>,

    // === Searchable labels ===
    pub labels: LabelSet,

    // === Branch identity (ADR-012 graph branch merge) ===
    /// Branch identifier this record belongs to. `None` = main / default branch.
    ///
    /// **Wire visibility (TD-072 fix, 2026-05-27)**: this field IS serialized
    /// by every serde encoder. Originally Slice 2 (`1dca6ea7b`) set
    /// `#[serde(skip, default)]` to protect the V1 bincode wire — but that
    /// silently dropped `branch_id` from the canonical WAL too (which uses
    /// `rmp_serde::to_vec_named`), structurally breaking the REST endpoint
    /// `POST /api/v1/collections/:col/branches/:b/merge`. TD-072 flipped the
    /// attribute to `#[serde(default)]`: rmp_serde named-field is
    /// schema-tolerant so old readers ignore the field, and no production
    /// code bincode-serializes `ProximaRecord` directly. See
    /// `docs/10-quality/TECHNICAL_DEBT.adoc#TD-072` and
    /// `tests/graph_branch_merge_integration_test.rs`.
    #[serde(default)]
    pub branch_id: Option<String>,
}

impl Default for ProximaRecord {
    fn default() -> Self {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as i64;

        Self {
            schema_version: schema_version::default_schema_version(),
            oid: String::new(),
            local_id: None,
            tid: None,
            variation_id: None,
            record_version: 0,
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
            props: HashMap::new(),
            refs: Vec::new(),
            edge: None,
            embeddings: Vec::new(),
            sequence: None,
            labels: LabelSet::new(),
            branch_id: None,
        }
    }
}

/// Canonical "is this record dead" rule on `valid_to_ns` (nanoseconds) —
/// the single source of truth for dead-record filtering. Operates on ns
/// directly so it is immune to the `expires_at` unit confusion (ms on the
/// proto wire vs secs in the WAL path). Used by types that carry only
/// `valid_to_ns` (e.g. `OptimizedSearchRecord`); `ProximaRecord` prefers
/// the richer [`ProximaRecord::is_dead`] (which also honors `valid_from`).
///
/// * `Some(0)` → tombstone (delete marker).
/// * `Some(v)` where `0 < v < now_ns` → TTL-expired.
/// * `None` or `v >= now_ns` → alive.
pub fn is_record_dead(valid_to_ns: Option<i64>, now_ns: i64) -> bool {
    match valid_to_ns {
        Some(0) => true,
        Some(v) if v > 0 && v < now_ns => true,
        _ => false,
    }
}

impl ProximaRecord {
    /// Returns true when this record is a tombstone marker at `snapshot_time_ns`.
    ///
    /// Legacy vector paths encode delete tombstones as records with no embeddings
    /// and `valid_to_ns <= snapshot_time_ns`. Relational paths should converge on
    /// the same visibility rule while storage compaction eventually closes older
    /// versions and removes obsolete tombstones after retention.
    pub fn is_tombstone_at(&self, snapshot_time_ns: i64) -> bool {
        self.valid_to_ns
            .is_some_and(|valid_to_ns| valid_to_ns <= snapshot_time_ns)
            && self.embeddings.is_empty()
            && self.origin.as_deref() == Some("delete")
    }

    /// Returns true when this record is visible at `snapshot_time_ns`.
    ///
    /// `valid_to_ns` is exclusive: a row with `valid_to_ns == snapshot_time_ns`
    /// has already been closed for that snapshot.
    pub fn is_visible_at(&self, snapshot_time_ns: i64) -> bool {
        if self.is_tombstone_at(snapshot_time_ns) {
            return false;
        }

        let valid_from_ok = self
            .valid_from_ns
            .is_none_or(|valid_from_ns| valid_from_ns <= snapshot_time_ns);
        let valid_to_ok = self
            .valid_to_ns
            .is_none_or(|valid_to_ns| snapshot_time_ns < valid_to_ns);
        valid_from_ok && valid_to_ok
    }

    /// Returns true when this record must NOT appear in a read at `now_ns`
    /// — i.e. it is deleted (tombstone) or TTL-expired or not-yet-valid.
    /// Convenience inverse of [`Self::is_visible_at`]; the single source of
    /// truth for read-path dead-record filtering on `ProximaRecord`.
    pub fn is_dead(&self, now_ns: i64) -> bool {
        !self.is_visible_at(now_ns)
    }

    /// Construct a canonical tombstone marker for a logical record id.
    pub fn tombstone(oid: impl Into<String>, timestamp_ns: i64) -> Self {
        Self {
            oid: oid.into(),
            created_at_ns: timestamp_ns,
            updated_at_ns: timestamp_ns,
            valid_to_ns: Some(0),
            origin: Some("delete".to_string()),
            method: Some("tombstone".to_string()),
            ..Self::default()
        }
    }

    /// Check whether a given principal is permitted to access this record.
    ///
    /// Returns `true` if `permitted_principals` is empty (open access) OR if
    /// `principal` appears in the list.
    pub fn is_accessible_by(&self, principal: &str) -> bool {
        self.permitted_principals.is_empty()
            || self.permitted_principals.iter().any(|p| p == principal)
    }

    /// Check whether this record belongs to the given tenant.
    ///
    /// An empty `tenant_id` on the record matches any tenant (single-tenant mode).
    pub fn matches_tenant(&self, tenant_id: &str) -> bool {
        self.tenant_id.is_empty() || self.tenant_id == tenant_id
    }

    /// Resolve a conflict between two records using Last-Write-Wins (LWW).
    ///
    /// If `self.updated_at_ns >= other.updated_at_ns`, `self` is returned.
    /// Otherwise, `other` is returned.
    pub fn resolve_conflict(&self, other: &Self) -> Self {
        if self.updated_at_ns >= other.updated_at_ns {
            self.clone()
        } else {
            other.clone()
        }
    }
}

// ---------------------------------------------------------------------------
// Search result types (spec §1490)
// ---------------------------------------------------------------------------

/// Canonical search result carrier per MULTIMODAL_OVERHAUL_SPEC §1490.
///
/// Replaces the legacy proto v1 `SearchVectorRecord` type everywhere outside
/// of wire-format protocol adapters. The triple `(record, score, rank)` is
/// the authoritative internal shape for search results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredRecord {
    pub record: ProximaRecord,
    /// Similarity/relevance score (higher is more relevant).
    pub score: f32,
    /// 1-based rank in the result set.
    pub rank: u32,
}

/// Top-level search response envelope (v2).
///
/// Returned by query execution, search services, and hybrid search orchestrators.
/// Protocol adapters (gRPC/REST/Arrow) serialize this into wire shapes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResponse {
    pub records: Vec<ScoredRecord>,
    pub total_found: i64,
    pub collection_id: String,
    /// Query execution time in microseconds.
    pub query_time_us: u64,
}

// ---------------------------------------------------------------------------
// Tests (TDD)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_data_model::{MemoryType, ProximaValue};

    #[test]
    fn test_proxima_record_defaults() {
        let r = ProximaRecord::default();
        assert!(r.created_at_ns > 0, "created_at_ns must be positive");
        assert_eq!(r.tenant_id, "");
        assert!(r.permitted_principals.is_empty());
        assert!(r.props.is_empty());
        assert_eq!(r.record_version, 0);
        assert_eq!(r.spec_version, 1);
        assert!(r.memory_type.is_none());
    }

    #[test]
    fn test_resolve_conflict_lww() {
        let r1 = ProximaRecord {
            updated_at_ns: 100,
            origin: Some("r1".to_string()),
            ..ProximaRecord::default()
        };

        let r2 = ProximaRecord {
            updated_at_ns: 200,
            origin: Some("r2".to_string()),
            ..ProximaRecord::default()
        };

        let resolved = r1.resolve_conflict(&r2);
        assert_eq!(resolved.origin.unwrap(), "r2");

        let resolved_inverse = r2.resolve_conflict(&r1);
        assert_eq!(resolved_inverse.origin.unwrap(), "r2");
    }

    #[test]
    fn test_memory_type_field() {
        let r = ProximaRecord {
            memory_type: Some(MemoryType::Decision),
            ..ProximaRecord::default()
        };
        assert_eq!(r.memory_type, Some(MemoryType::Decision));
    }

    #[test]
    fn test_tenant_id_is_record_field() {
        let r = ProximaRecord {
            tenant_id: "acme".to_string(),
            ..ProximaRecord::default()
        };
        assert_eq!(r.tenant_id, "acme");
        // Accessible without unwrapping an Option<UserContext>
    }

    #[test]
    fn test_permitted_principals_empty_means_open_access() {
        let r = ProximaRecord::default();
        assert!(r.is_accessible_by("alice"));
        assert!(r.is_accessible_by("bob"));
    }

    #[test]
    fn test_permitted_principals_restricts_access() {
        let r = ProximaRecord {
            permitted_principals: vec!["alice".to_string()],
            ..ProximaRecord::default()
        };
        assert!(r.is_accessible_by("alice"));
        assert!(!r.is_accessible_by("bob"));
    }

    #[test]
    fn test_matches_tenant() {
        let mut r = ProximaRecord::default();
        assert!(r.matches_tenant("acme"), "empty tenant_id matches any");

        r.tenant_id = "acme".to_string();
        assert!(r.matches_tenant("acme"));
        assert!(!r.matches_tenant("other"));
    }

    #[test]
    fn test_mvcc_visibility_uses_exclusive_valid_to() {
        let r = ProximaRecord {
            valid_from_ns: Some(100),
            valid_to_ns: Some(200),
            ..ProximaRecord::default()
        };

        assert!(!r.is_visible_at(99));
        assert!(r.is_visible_at(100));
        assert!(r.is_visible_at(199));
        assert!(!r.is_visible_at(200));
    }

    #[test]
    fn test_tombstone_is_not_visible() {
        let r = ProximaRecord::tombstone("row-1", 500);
        assert_eq!(r.oid, "row-1");
        assert!(r.is_tombstone_at(500));
        assert!(!r.is_visible_at(500));
        assert!(r.is_dead(500), "tombstone is_dead");
    }

    #[test]
    fn test_is_record_dead_canonical_rule() {
        let now = 1_000_000_000i64;
        // Tombstone marker.
        assert!(is_record_dead(Some(0), now));
        // TTL-expired (valid_to in the past).
        assert!(is_record_dead(Some(999_999_999), now));
        // Alive: no expiry.
        assert!(!is_record_dead(None, now));
        // Alive: expiry in the future.
        assert!(!is_record_dead(Some(now + 1_000_000_000), now));
        // Boundary: valid_to == now is exclusive (still alive at exactly now).
        assert!(!is_record_dead(Some(now), now));
    }

    #[test]
    fn test_edge_shape_present_for_graph() {
        let r = ProximaRecord {
            edge: Some(EdgeShape {
                source_id: "node_a".to_string(),
                target_id: "node_b".to_string(),
                edge_type: "KNOWS".to_string(),
                weight: Some(0.9),
            }),
            ..ProximaRecord::default()
        };
        let edge = r.edge.as_ref().unwrap();
        assert_eq!(edge.source_id, "node_a");
        assert_eq!(edge.edge_type, "KNOWS");
        assert!((edge.weight.unwrap() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_embedding_cell_fields() {
        let cell =
            EmbeddingCell::new_fp32("text-embedding-3-small", "text", 3, vec![0.1, 0.2, 0.3]);
        assert_eq!(cell.dim, 3);
        assert_eq!(cell.values.len(), 3);

        let mut r = ProximaRecord::default();
        r.embeddings.push(cell);
        assert_eq!(r.embeddings[0].modality, "text");
    }

    #[test]
    fn test_typed_ref_variants_exist() {
        let fk = TypedRef::ForeignKey {
            table: "orders".to_string(),
            id: "ord_1".to_string(),
        };
        let ge = TypedRef::GraphEdge {
            edge_id: "e_1".to_string(),
            direction: EdgeDirection::Outgoing,
        };
        let emb = TypedRef::Embedding {
            model_id: "ada-002".to_string(),
        };
        assert!(matches!(fk, TypedRef::ForeignKey { .. }));
        assert!(matches!(ge, TypedRef::GraphEdge { .. }));
        assert!(matches!(emb, TypedRef::Embedding { .. }));
    }

    #[test]
    fn test_label_set_contains() {
        let mut labels = LabelSet::new();
        labels.insert("vector");
        labels.insert("text");
        assert!(labels.contains("vector"));
        assert!(!labels.contains("graph"));
    }

    #[test]
    fn test_label_set_deduplication() {
        let mut labels = LabelSet::new();
        labels.insert("rag");
        labels.insert("rag");
        assert_eq!(labels.len(), 1);
    }

    #[test]
    fn test_proxima_tree_flat_lookup() {
        let mut tree = ProximaTree::new();
        tree.insert(
            "category".to_string(),
            ProximaTreeNode::Value(ProximaValue::String("technology".to_string())),
        );
        let val = tree_get(&tree, "category").unwrap();
        assert!(matches!(val, ProximaValue::String(s) if s == "technology"));
    }

    #[test]
    fn test_proxima_tree_nested_lookup() {
        let mut inner = ProximaTree::new();
        inner.insert(
            "city".to_string(),
            ProximaTreeNode::Value(ProximaValue::String("NYC".to_string())),
        );
        let mut tree = ProximaTree::new();
        tree.insert("address".to_string(), ProximaTreeNode::Object(inner));

        let val = tree_get(&tree, "address.city").unwrap();
        assert!(matches!(val, ProximaValue::String(s) if s == "NYC"));
    }

    #[test]
    fn test_proxima_tree_missing_path_returns_none() {
        let tree = ProximaTree::new();
        assert!(tree_get(&tree, "nonexistent").is_none());
        assert!(tree_get(&tree, "a.b.c").is_none());
    }

    #[test]
    fn test_record_serde_roundtrip() {
        let mut r = ProximaRecord {
            oid: "rec_abc".to_string(),
            tenant_id: "acme".to_string(),
            ..ProximaRecord::default()
        };
        r.labels.insert("vector");
        r.props.insert(
            "score".to_string(),
            ProximaTreeNode::Value(ProximaValue::Float64(0.95)),
        );

        let json = serde_json::to_string(&r).expect("serialize");
        let back: ProximaRecord = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.oid, "rec_abc");
        assert_eq!(back.tenant_id, "acme");
        assert!(back.labels.contains("vector"));
    }

    #[test]
    fn test_search_response_carries_scored_records() {
        let record = ProximaRecord {
            oid: "rec_search".to_string(),
            ..ProximaRecord::default()
        };
        let response = SearchResponse {
            records: vec![ScoredRecord {
                record,
                score: 0.98,
                rank: 1,
            }],
            total_found: 1,
            collection_id: "docs".to_string(),
            query_time_us: 42,
        };

        assert_eq!(response.records[0].record.oid, "rec_search");
        assert_eq!(response.records[0].rank, 1);
        assert_eq!(response.total_found, 1);
        assert_eq!(response.collection_id, "docs");
    }

    // ---- EmbeddingScalarType + EmbeddingValues (PR 1, precision rollout) ----

    #[test]
    fn scalar_type_bytes_per_element_match_lld_table() {
        assert_eq!(EmbeddingScalarType::Fp32.bytes_per_element(), 4);
        assert_eq!(EmbeddingScalarType::Fp16.bytes_per_element(), 2);
        assert_eq!(EmbeddingScalarType::Bf16.bytes_per_element(), 2);
        assert_eq!(EmbeddingScalarType::Int8Scalar.bytes_per_element(), 1);
        assert_eq!(EmbeddingScalarType::UInt8Scalar.bytes_per_element(), 1);
    }

    #[test]
    fn scalar_type_lossiness_flags_int8_paths() {
        assert!(!EmbeddingScalarType::Fp32.is_lossy());
        assert!(!EmbeddingScalarType::Fp16.is_lossy());
        assert!(!EmbeddingScalarType::Bf16.is_lossy());
        assert!(EmbeddingScalarType::Int8Scalar.is_lossy());
        assert!(EmbeddingScalarType::UInt8Scalar.is_lossy());
    }

    #[test]
    fn scalar_type_default_is_fp32_for_backward_compat() {
        assert_eq!(EmbeddingScalarType::default(), EmbeddingScalarType::Fp32);
    }

    #[test]
    fn embedding_values_byte_size_matches_layout() {
        let dim = 1024usize;
        assert_eq!(EmbeddingValues::Fp32(vec![0.0; dim]).byte_size(), 4 * dim);
        assert_eq!(
            EmbeddingValues::Fp16(vec![half::f16::from_f32(0.0); dim]).byte_size(),
            2 * dim
        );
        assert_eq!(
            EmbeddingValues::Bf16(vec![half::bf16::from_f32(0.0); dim]).byte_size(),
            2 * dim
        );
        assert_eq!(
            EmbeddingValues::Int8Scalar {
                values: vec![0i8; dim],
                scale: 1.0,
                zero_point: 0,
            }
            .byte_size(),
            dim + 5
        );
        assert_eq!(
            EmbeddingValues::UInt8Scalar {
                values: vec![0u8; dim],
                scale: 1.0,
                zero_point: 128,
            }
            .byte_size(),
            dim + 5
        );
    }

    #[test]
    fn embedding_values_len_matches_inner_vec() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let ev = EmbeddingValues::Fp32(v.clone());
        assert_eq!(ev.len(), v.len());
        assert!(!ev.is_empty());
        assert!(EmbeddingValues::Fp32(vec![]).is_empty());
    }

    #[test]
    fn coerce_to_precision_is_noop_when_target_matches() {
        let original = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut cell = EmbeddingCell::new_fp32("m", "dense_vector", 4, original.clone());
        cell.coerce_to_precision(EmbeddingScalarType::Fp32);
        assert_eq!(cell.values, EmbeddingValues::Fp32(original));
        assert_eq!(cell.precision, EmbeddingScalarType::Fp32);
    }

    #[test]
    fn coerce_fp16_to_fp16_via_fp32_pivot_is_bit_exact() {
        // The fp32-pivot round-trip is bit-exact for fp16 because fp32
        // strictly dominates fp16. This is the compaction read+rewrite
        // case: source block is fp16, intermediate flattens to fp32,
        // writer coerces back to fp16 — must match the original bits.
        let src_fp32 = [1.0_f32, -2.5, 65504.0, 0.0001, 100.25, -0.5, 42.0, 7.125];
        let original_fp16: Vec<half::f16> =
            src_fp32.iter().map(|&x| half::f16::from_f32(x)).collect();

        // Simulate the round-trip: cell stores fp32 (after the
        // VectorRecord intermediate), then gets coerced back to fp16.
        let promoted_fp32: Vec<f32> = original_fp16.iter().map(|x| x.to_f32()).collect();
        let mut cell = EmbeddingCell::new_fp32("m", "dense_vector", 8, promoted_fp32);
        cell.coerce_to_precision(EmbeddingScalarType::Fp16);

        match cell.values {
            EmbeddingValues::Fp16(got) => assert_eq!(
                got, original_fp16,
                "bit-exact fp16 → fp32 → fp16 round-trip"
            ),
            other => panic!("expected Fp16, got {:?}", other.scalar_type()),
        }
        assert_eq!(cell.precision, EmbeddingScalarType::Fp16);
    }

    #[test]
    fn coerce_fp32_to_fp16_updates_metadata() {
        let mut cell = EmbeddingCell::new_fp32("m", "dense_vector", 3, vec![1.0, 2.0, 3.0]);
        cell.coerce_to_precision(EmbeddingScalarType::Fp16);
        assert_eq!(cell.values.scalar_type(), EmbeddingScalarType::Fp16);
        assert_eq!(cell.precision, EmbeddingScalarType::Fp16);
        assert_eq!(cell.values.len(), 3);
    }

    #[test]
    fn coerce_fp32_to_bf16_then_back_preserves_precision_tag() {
        let mut cell = EmbeddingCell::new_fp32("m", "dense_vector", 4, vec![1.0, 2.0, 3.0, 4.0]);
        cell.coerce_to_precision(EmbeddingScalarType::Bf16);
        assert_eq!(cell.values.scalar_type(), EmbeddingScalarType::Bf16);
        assert_eq!(cell.precision, EmbeddingScalarType::Bf16);
        // bf16 has same range as fp32 but lower mantissa; round-trip
        // back to fp32 then forward to bf16 must be stable.
        cell.coerce_to_precision(EmbeddingScalarType::Bf16);
        assert_eq!(cell.precision, EmbeddingScalarType::Bf16);
    }

    #[test]
    fn embedding_values_scalar_type_round_trip() {
        for st in [
            EmbeddingScalarType::Fp32,
            EmbeddingScalarType::Fp16,
            EmbeddingScalarType::Bf16,
            EmbeddingScalarType::Int8Scalar,
            EmbeddingScalarType::UInt8Scalar,
        ] {
            let ev = EmbeddingValues::from_fp32_lossy(&[1.0, 2.0, 3.0], st);
            assert_eq!(ev.scalar_type(), st);
        }
    }

    #[test]
    fn embedding_values_as_fp32_slice_only_for_fp32() {
        let v = vec![1.0, 2.0, 3.0];
        assert_eq!(
            EmbeddingValues::Fp32(v.clone()).as_fp32_slice(),
            Some(v.as_slice())
        );
        assert!(
            EmbeddingValues::from_fp32_lossy(&v, EmbeddingScalarType::Fp16)
                .as_fp32_slice()
                .is_none()
        );
    }

    #[test]
    fn embedding_values_to_fp32_owned_round_trips_fp16_within_tolerance() {
        let src: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
        let ev = EmbeddingValues::from_fp32_lossy(&src, EmbeddingScalarType::Fp16);
        let back = ev.to_fp32_owned();
        assert_eq!(back.len(), src.len());
        for (a, b) in src.iter().zip(back.iter()) {
            assert!((a - b).abs() < 0.01, "fp16 RT tolerance fail: {a} vs {b}");
        }
    }

    #[test]
    fn embedding_values_int8_lossy_round_trip_within_per_cell_scale() {
        let src: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
        let ev = EmbeddingValues::from_fp32_lossy(&src, EmbeddingScalarType::Int8Scalar);
        let back = ev.to_fp32_owned();
        let quantum = match &ev {
            EmbeddingValues::Int8Scalar { scale, .. } => *scale,
            _ => unreachable!(),
        };
        for (a, b) in src.iter().zip(back.iter()) {
            assert!(
                (a - b).abs() <= quantum * 1.05,
                "int8 RT beyond one quantum: {a} vs {b} q={quantum}"
            );
        }
    }

    #[test]
    fn embedding_values_serde_round_trips_fp32() {
        let ev = EmbeddingValues::Fp32(vec![1.0, 2.0, 3.0]);
        let json = serde_json::to_string(&ev).unwrap();
        let back: EmbeddingValues = serde_json::from_str(&json).unwrap();
        assert_eq!(ev, back);
    }

    #[test]
    fn embedding_values_serde_round_trips_fp16() {
        let ev = EmbeddingValues::from_fp32_lossy(&[0.5, 1.0, 1.5], EmbeddingScalarType::Fp16);
        let json = serde_json::to_string(&ev).unwrap();
        let back: EmbeddingValues = serde_json::from_str(&json).unwrap();
        assert_eq!(ev, back);
    }

    // ---- EmbeddingCell migration to new schema (PR 1) ----

    #[test]
    fn new_fp32_constructor_sets_default_precision_metadata() {
        let cell = EmbeddingCell::new_fp32("bge-small", "text", 4, vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(cell.model_id, "bge-small");
        assert_eq!(cell.modality, "text");
        assert_eq!(cell.dim, 4);
        assert_eq!(cell.as_fp32_slice(), &[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(cell.precision, EmbeddingScalarType::Fp32);
        assert_eq!(cell.precision_epoch, None);
    }

    #[test]
    fn embedding_cell_as_fp32_slice_borrows_values() {
        let cell = EmbeddingCell::new_fp32("m", "text", 3, vec![1.0, 2.0, 3.0]);
        assert_eq!(cell.as_fp32_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn embedding_cell_as_fp32_cow_borrows_when_storage_is_fp32() {
        // INT-2.5a: with today's Vec<f32> storage, the Cow must be
        // Borrowed (zero-copy). After INT-2.5b's field flip, this
        // invariant holds for the Fp32 variant; non-fp32 variants
        // return Owned.
        let cell = EmbeddingCell::new_fp32("m", "text", 3, vec![1.0, 2.0, 3.0]);
        let cow = cell.as_fp32_cow();
        assert!(matches!(cow, std::borrow::Cow::Borrowed(_)));
        assert_eq!(&*cow, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn embedding_cell_as_fp32_cow_matches_as_fp32_slice() {
        // The 145 read sites that currently do `&cell.values` or
        // `cell.values.as_slice()` get equivalent bytes via as_fp32_cow.
        // This lets us migrate sites incrementally without behavior
        // change today.
        let cell = EmbeddingCell::new_fp32("m", "text", 4, vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(&*cell.as_fp32_cow(), cell.as_fp32_slice());
    }

    #[test]
    fn embedding_cell_byte_size_reflects_active_variant() {
        // INT-2.5b: values are now authoritative for byte size — the
        // precision tag must match the variant (the
        // cell.precision == cell.values.scalar_type() invariant).
        // Constructing via new_fp32 builds an Fp32 variant (4 bytes/elem).
        let cell = EmbeddingCell::new_fp32("m", "text", 1024, vec![0.0; 1024]);
        assert_eq!(cell.values_byte_size(), 4096);
        // Switching to a fp16 cell requires building the variant explicitly;
        // mutating the precision tag alone (the pre-INT-2.5b pattern) is no
        // longer enough because byte_size() reads from the variant directly.
        let cell16 = EmbeddingCell::new_typed(
            "m",
            "text",
            1024,
            EmbeddingValues::Fp16(vec![half::f16::from_f32(0.0); 1024]),
        );
        assert_eq!(cell16.values_byte_size(), 2048);
        assert_eq!(cell16.precision, EmbeddingScalarType::Fp16);
    }

    /// INT-2.5b fixture: locks the on-disk bincode bytes for a known
    /// EmbeddingCell BEFORE the field-type flip. After 2.5b's custom
    /// Serialize/Deserialize impls land, this same byte string must
    /// round-trip identically — that's the irreversible-format
    /// insurance the bridge memo calls out.
    ///
    /// To update this fixture (only when bincode shape is intentionally
    /// changing): copy the bytes from the assertion failure into
    /// `EXPECTED_BYTES` below.
    #[test]
    fn embedding_cell_bincode_fixture_locks_pre_2_5b_layout() {
        let cell = EmbeddingCell::new_fp32("fixture-model", "text", 4, vec![0.1, 0.2, 0.3, 0.4]);
        let bytes = bincode::serialize(&cell).unwrap();
        let hex: String = bytes.iter().map(|b| format!("\\x{:02x}", b)).collect();

        // INT-2.5b lock: bytes captured 2026-05-23 on the
        // pre-field-flip layout. Field order:
        //   model_id (len:u64 || utf8)
        //   modality (len:u64 || utf8)
        //   values   (len:u64 || elements:Vec<f32>)
        //   dim      (u32 LE)
        //   precision (u8 — EmbeddingScalarType discriminant)
        //   precision_epoch (Option<u64>) is ALWAYS emitted on the binary wire
        //     (see EmbeddingCell::serialize). When None it is a single 0x00
        //     Option-tag byte — NOT skipped — so the positional decoder stays
        //     aligned and Vec<EmbeddingCell> boundaries are unambiguous.
        const EXPECTED_HEX: &str = "\
\\x0d\\x00\\x00\\x00\\x00\\x00\\x00\\x00\
fixture-model\\x04\\x00\\x00\\x00\\x00\\x00\\x00\\x00\
text\\x04\\x00\\x00\\x00\\x00\\x00\\x00\\x00\
\\xcd\\xcc\\xcc\\x3d\\xcd\\xcc\\x4c\\x3e\\x9a\\x99\\x99\\x3e\\xcd\\xcc\\xcc\\x3e\
\\x04\\x00\\x00\\x00\
\\x01\
\\x00";

        // Strip the ASCII embeds for the comparison — they're there so
        // the literal is human-readable. Build the expected hex string
        // by mixing escape sequences and raw ASCII the same way bincode
        // emits string bytes.
        let expected = build_expected_fixture_bytes();
        assert_eq!(
            bytes, expected,
            "v1 bincode byte layout drifted!\n\
             got:      {hex}\n\
             expected: see build_expected_fixture_bytes() in tests.\n\
             If this drift is intentional (e.g. you're INT-2.5b's field flip + custom serde \
             and the test now reflects the v2 layout), update build_expected_fixture_bytes() \
             AND the EMBEDDING_PRECISION_BRIDGE_BINCODE_MEMO.adoc decision log."
        );
        // Echo the const for grep-ability in CI logs.
        let _ = EXPECTED_HEX;
    }

    /// Build the bincode bytes the v1 layout MUST emit for the fixture.
    /// Each field is appended in declaration order using bincode's
    /// little-endian length-prefixed encoding.
    ///
    /// Subtleties learned the hard way 2026-05-23 while writing this
    /// fixture (these were the actual surprises vs the naive layout):
    /// * `#[repr(u8)]` on `EmbeddingScalarType` does NOT change the serde
    ///   serialization. Serde emits the variant index as `u32` (4 bytes)
    ///   regardless of the repr. Fp32 is variant 0 → `00 00 00 00`, not
    ///   `0x01` like the repr suggests.
    /// * `precision_epoch: Option<u64>` is ALWAYS written on the binary wire
    ///   (the custom Serialize skips it only for human-readable formats).
    ///   `None` is a single `0x00` Option-tag byte — this keeps the positional
    ///   bincode decoder aligned across a `Vec<EmbeddingCell>`.
    /// * `dim: u32` serializes as 4 LE bytes (bincode 1.x fixint).
    /// * String + Vec lengths use `u64` (8 bytes LE), not varint.
    fn build_expected_fixture_bytes() -> Vec<u8> {
        let mut b = Vec::new();
        // model_id: "fixture-model" (13 bytes)
        b.extend_from_slice(&13u64.to_le_bytes());
        b.extend_from_slice(b"fixture-model");
        // modality: "text" (4 bytes)
        b.extend_from_slice(&4u64.to_le_bytes());
        b.extend_from_slice(b"text");
        // values: 4 fp32 elements, len-prefixed
        b.extend_from_slice(&4u64.to_le_bytes());
        for v in [0.1f32, 0.2, 0.3, 0.4] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        // dim: u32 LE
        b.extend_from_slice(&4u32.to_le_bytes());
        // precision: single byte = #[repr(u8)] discriminant value
        // (Fp32 = 0x01). Saves 3 bytes per cell vs the default u32
        // variant index that serde-bincode would emit without the
        // custom Serialize impl on EmbeddingScalarType.
        b.push(EmbeddingScalarType::Fp32 as u8);
        // precision_epoch: None → single 0x00 Option-tag byte (always emitted
        // on the binary wire to keep the positional decoder aligned).
        b.push(0x00);
        b
    }

    #[test]
    fn embedding_cell_serde_back_compat_round_trip() {
        // Old-shape JSON without precision/precision_epoch should deserialize
        // into a cell whose precision defaults to Fp32 and epoch is None.
        let old_json = r#"{
            "model_id": "legacy",
            "modality": "text",
            "values": [0.1, 0.2, 0.3],
            "dim": 3
        }"#;
        let cell: EmbeddingCell = serde_json::from_str(old_json).unwrap();
        assert_eq!(cell.precision, EmbeddingScalarType::Fp32);
        assert_eq!(cell.precision_epoch, None);
        assert_eq!(cell.as_fp32_slice(), &[0.1, 0.2, 0.3]);
    }

    #[test]
    fn embedding_cell_as_embedding_values_returns_active_variant() {
        // INT-2.5b: as_embedding_values now returns a clone of the
        // stored variant directly — the pre-INT-2.5b behavior of
        // "promote-then-tag-from-precision-field" no longer applies
        // because the variant IS the source of truth. A cell built
        // with new_typed(Fp16(...)) returns Fp16; one built with
        // new_fp32 returns Fp32.
        let cell_fp32 = EmbeddingCell::new_fp32("m", "text", 3, vec![0.0, 1.0, 2.0]);
        let ev_fp32 = cell_fp32.as_embedding_values();
        assert_eq!(ev_fp32.scalar_type(), EmbeddingScalarType::Fp32);
        assert_eq!(ev_fp32.len(), 3);

        let cell_fp16 = EmbeddingCell::new_typed(
            "m",
            "text",
            3,
            EmbeddingValues::Fp16(vec![
                half::f16::from_f32(0.0),
                half::f16::from_f32(1.0),
                half::f16::from_f32(2.0),
            ]),
        );
        let ev_fp16 = cell_fp16.as_embedding_values();
        assert_eq!(ev_fp16.scalar_type(), EmbeddingScalarType::Fp16);
        assert_eq!(ev_fp16.len(), 3);
    }

    // === PR 2: ProximaRecord.schema_version ===

    #[test]
    fn schema_version_constants_match_lld() {
        assert_eq!(schema_version::V1, 1);
        assert_eq!(schema_version::V2, 2);
        // PR 2 keeps writers on V1; PR 3 flips this via the feature flag.
        assert_eq!(schema_version::CURRENT, schema_version::V1);
    }

    #[test]
    fn default_record_has_schema_version_v1() {
        let r = ProximaRecord::default();
        assert_eq!(r.schema_version, schema_version::V1);
    }

    #[test]
    fn schema_version_is_skipped_by_serde() {
        // serde(skip) means a v2-tagged record serialized to JSON omits the
        // schema_version field entirely. Old readers see the same bytes they
        // always did — back-compat with PR 0 wire formats.
        let mut r = ProximaRecord::default();
        r.schema_version = schema_version::V2;
        let json = serde_json::to_string(&r).unwrap();
        assert!(
            !json.contains("schema_version"),
            "schema_version must not appear on the wire (got {json})"
        );
    }

    #[test]
    fn schema_version_round_trip_through_json_resets_to_v1() {
        // serde(skip) on the way IN means deserialization re-applies the
        // default; the WAL/segment-header reader is responsible for stamping
        // the correct value (PR 4).
        let mut r = ProximaRecord::default();
        r.schema_version = schema_version::V2;
        r.oid = "oid-1".into();
        let json = serde_json::to_string(&r).unwrap();
        let parsed: ProximaRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.schema_version, schema_version::V1);
        assert_eq!(parsed.oid, "oid-1");
    }

    #[test]
    fn schema_version_round_trip_through_bincode_resets_to_v1() {
        // Same contract for bincode: positional encoder skips the field.
        let mut r = ProximaRecord::default();
        r.schema_version = schema_version::V2;
        r.oid = "oid-bincode".into();
        let bytes = bincode::serialize(&r).unwrap();
        let parsed: ProximaRecord = bincode::deserialize(&bytes).unwrap();
        assert_eq!(parsed.schema_version, schema_version::V1);
        assert_eq!(parsed.oid, "oid-bincode");
    }

    // === TD-072: branch_id survives canonical-WAL roundtrip ===
    //
    // Regression tests for the bug Slice 8 surfaced: the canonical WAL
    // (rmp_serde::to_vec_named) was dropping `branch_id` because of the
    // prior `#[serde(skip)]` attribute, structurally breaking the merge
    // endpoint. The fix flipped the attribute to `#[serde(default)]`; these
    // tests pin that behavior so it can't silently regress.

    #[test]
    fn branch_id_survives_rmp_serde_named_roundtrip() {
        let mut record = ProximaRecord::default();
        record.oid = "oid-rmp".into();
        record.branch_id = Some("dev".to_string());
        let bytes = rmp_serde::to_vec_named(&record).expect("rmp encode");
        let decoded: ProximaRecord = rmp_serde::from_slice(&bytes).expect("rmp decode");
        assert_eq!(decoded.branch_id.as_deref(), Some("dev"));
        assert_eq!(decoded.oid, "oid-rmp");
    }

    #[test]
    fn branch_id_survives_bincode_roundtrip() {
        let mut record = ProximaRecord::default();
        record.oid = "oid-bincode".into();
        record.branch_id = Some("feature-x".to_string());
        let bytes = bincode::serialize(&record).expect("bincode encode");
        let decoded: ProximaRecord = bincode::deserialize(&bytes).expect("bincode decode");
        assert_eq!(decoded.branch_id.as_deref(), Some("feature-x"));
        assert_eq!(decoded.oid, "oid-bincode");
    }

    #[test]
    fn branch_id_defaults_to_none_when_absent_in_rmp_named() {
        // The #[serde(default)] half of the attribute: a record encoded
        // without setting branch_id (or by an older writer that lacked the
        // field entirely) must decode to None, not error.
        let mut record = ProximaRecord::default();
        record.oid = "oid-no-branch".into();
        let bytes = rmp_serde::to_vec_named(&record).expect("rmp encode");
        let decoded: ProximaRecord = rmp_serde::from_slice(&bytes).expect("rmp decode");
        assert!(decoded.branch_id.is_none());
        assert_eq!(decoded.oid, "oid-no-branch");
    }

    // === PR 3: validate_records_for_schema_v1 ===

    #[test]
    fn validate_records_for_schema_v1_accepts_all_fp32() {
        let r = ProximaRecord {
            embeddings: vec![EmbeddingCell::new_fp32("m", "text", 3, vec![0.1, 0.2, 0.3])],
            ..ProximaRecord::default()
        };
        assert!(validate_records_for_schema_v1(std::iter::once(&r)).is_ok());
    }

    #[test]
    fn validate_records_for_schema_v1_accepts_empty_embeddings() {
        // Non-embedding modalities (graph nodes, log lines, relational rows)
        // must pass the schema-v1 gate unchanged.
        let r = ProximaRecord {
            oid: "graph-node-1".into(),
            ..ProximaRecord::default()
        };
        assert!(validate_records_for_schema_v1(std::iter::once(&r)).is_ok());
    }

    #[test]
    fn validate_records_for_schema_v1_rejects_fp16_with_locked_error_tag() {
        let mut r = ProximaRecord {
            oid: "fp16-rec".into(),
            embeddings: vec![EmbeddingCell::new_fp32(
                "bge-large",
                "text",
                3,
                vec![0.1, 0.2, 0.3],
            )],
            ..ProximaRecord::default()
        };
        r.embeddings[0].precision = EmbeddingScalarType::Fp16;

        let err = validate_records_for_schema_v1(std::iter::once(&r)).unwrap_err();
        assert_eq!(err.record_oid, "fp16-rec");
        assert_eq!(err.model_id, "bge-large");
        assert_eq!(err.found_precision, EmbeddingScalarType::Fp16);
        let msg = err.to_string();
        assert!(
            msg.starts_with("unsupported_precision_schema_v1_only:"),
            "error tag must match LLD: {msg}"
        );
    }

    #[test]
    fn validate_records_for_schema_v1_stops_at_first_offender() {
        // Two records, second one bad — error must reference the second.
        let good = ProximaRecord {
            oid: "good".into(),
            embeddings: vec![EmbeddingCell::new_fp32("m", "text", 1, vec![1.0])],
            ..ProximaRecord::default()
        };
        let mut bad = ProximaRecord {
            oid: "bad".into(),
            embeddings: vec![EmbeddingCell::new_fp32("m", "text", 1, vec![1.0])],
            ..ProximaRecord::default()
        };
        bad.embeddings[0].precision = EmbeddingScalarType::Int8Scalar;

        let err = validate_records_for_schema_v1([&good, &bad]).unwrap_err();
        assert_eq!(err.record_oid, "bad");
        assert_eq!(err.found_precision, EmbeddingScalarType::Int8Scalar);
    }

    #[test]
    fn validate_records_for_schema_v1_inspects_all_cells_per_record() {
        // One record with two embeddings — second cell is non-fp32.
        let mut r = ProximaRecord {
            oid: "multi-cell".into(),
            embeddings: vec![
                EmbeddingCell::new_fp32("text-model", "text", 1, vec![1.0]),
                EmbeddingCell::new_fp32("image-model", "image", 1, vec![1.0]),
            ],
            ..ProximaRecord::default()
        };
        r.embeddings[1].precision = EmbeddingScalarType::Bf16;

        let err = validate_records_for_schema_v1(std::iter::once(&r)).unwrap_err();
        assert_eq!(err.model_id, "image-model");
        assert_eq!(err.found_precision, EmbeddingScalarType::Bf16);
    }

    #[test]
    fn schema_version_old_json_without_field_deserializes_to_v1() {
        // Serialize a v2 record, parse the JSON, confirm it contains no
        // schema_version key (so old/new readers see the same bytes), then
        // deserialize to verify the default kicks in.
        let mut r = ProximaRecord::default();
        r.schema_version = schema_version::V2;
        r.oid = "legacy-record".into();
        let json = serde_json::to_string(&r).unwrap();
        let parsed_value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(
            parsed_value.get("schema_version").is_none(),
            "serialized record must not contain a schema_version field"
        );
        let parsed: ProximaRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.schema_version, schema_version::V1);
        assert_eq!(parsed.oid, "legacy-record");
    }
}
