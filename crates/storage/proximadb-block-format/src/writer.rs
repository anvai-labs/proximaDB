// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! PAX block writer.
//!
//! `PaxBlockWriter` accumulates `ProximaRecord` rows in memory and serialises
//! them to a PAX block when flushed. The block body layout is:
//!
//! ```text
//! [BlockHeader: 64B]
//! [RowDirectory: N_rows * 32B]        ← OLTP/PAX only
//! [ColumnStripes: concatenated]       ← OLAP/PAX only
//! [ColumnFooter: N_cols * 64B]
//! [BlockFooter: 32B]
//! ```
//!
//! Call `add_record()` to buffer rows, then `flush()` to serialise.
//! `flush()` returns a `Vec<u8>` suitable for direct I/O; the caller owns
//! the bytes and decides where to write them (WAL shard, Parquet data file,
//! memory-mapped region, etc.).

use anyhow::Result;
use crc32fast::Hasher as Crc32;
use proximadb_codec::{
    AccessTemperature, CompressionProfile, DataAnalysis, DataDomain, DictionaryScope,
    ProximaScheme, SelectionContext, StrategyRegistry, TypeId, WorkloadProfile, functions,
};
use proximadb_records::ProximaRecord;
use std::collections::{HashMap, HashSet};

use crate::{
    header::{BlockCompression, BlockHeader, BlockMode, HEADER_SIZE, flags, fnv1a_hash},
    record::{FlatRow, col_id, encode_str_col, update_i64_bounds},
    row_dir::{RowDirectory, RowEntry},
    rowgroup::{RowGroupBlock, RowGroupEntry, f64_bounds, i64_bounds},
    stripe::{COLUMN_META_SIZE, ColumnMeta, ColumnRole, ColumnStripe},
    vparam::{
        QUANT_FP16, QUANT_RABITQ, QUANT_RAW_F32, QUANT_SQ8, RaBitQColumn,
        TRANSFORM_CLUSTERED_FOR_U8, VectorParamBlock, VectorParamEntry, VectorTransformColumn,
    },
};

/// Size of the trailing `BlockFooter` in bytes.
pub const BLOCK_FOOTER_SIZE: usize = 32;

/// One P-Shred column: which prop key goes to which typed stripe, and — when the
/// caller can supply one — the EXACT declared type (TD-USUB-6).
///
/// `declared_type` is `Option` on purpose, because the two production paths that
/// shred are not equally informed:
///
/// * The **relational** path resolves each key against `CatalogTableSchema` and
///   therefore has a full [`ProximaType`] (`Int32`, `Timestamp(Micros)`, …).
/// * The **vector/SST flush** path builds its spec from the v1
///   `FilterableColumnSpec`, whose `FilterableDataType` is **coarse** —
///   `FilterableInteger` carries no width, `FilterableDatetime` no `TimeUnit`.
///   It cannot distinguish `Int32` from `Int64`, so it supplies `None`.
///
/// That distinction is load-bearing rather than cosmetic: a typed stripe stores
/// one of three physical classes (i64 / f64 / string), so reconstructing the
/// ORIGINAL `ProximaValue` variant is only possible where an exact declared type
/// says what to rebuild. Where it is absent, the msgpack `PROPS` tail must stay
/// complete and the stripe stays an advisory pruning index — which is why
/// "declared columns authoritative" (ADR-094 spec §2.1) can only ever apply to
/// the paths that declare exactly.
#[derive(Debug, Clone, PartialEq)]
pub struct ShredColumn {
    /// Prop key promoted into the stripe.
    pub key: String,
    /// Target stripe id (>= `col_id::USER_BASE`).
    pub column_id: i32,
    /// Exact declared type, when the caller has one.
    pub declared_type: Option<proximadb_data_model::ProximaType>,
}

impl ShredColumn {
    /// A column whose declared type is unknown or only coarsely known.
    pub fn untyped(key: impl Into<String>, column_id: i32) -> Self {
        Self {
            key: key.into(),
            column_id,
            declared_type: None,
        }
    }

    /// A column with an exact declared type from the catalog schema.
    pub fn typed(
        key: impl Into<String>,
        column_id: i32,
        declared_type: proximadb_data_model::ProximaType,
    ) -> Self {
        Self {
            key: key.into(),
            column_id,
            declared_type: Some(declared_type),
        }
    }
}

/// One entry of the on-disk shred directory (`col_id::SHRED_DIRECTORY`).
///
/// Serialized as msgpack. Readers that fail to parse it fall back to
/// caller-supplied key lists, so extending this shape stays mixed-read-safe.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ShredDirectoryEntry {
    pub column_id: i32,
    pub key: String,
    /// `None` where the writer had no exact declared type — the signal that this
    /// stripe can never be authoritative and the tail must stay complete.
    #[serde(default)]
    pub declared_type: Option<proximadb_data_model::ProximaType>,
}

/// Trailing block footer: encodes offsets for the column meta, row directory,
/// and the v2 footer-resident side regions (vector params + row-group index).
///
/// Layout (little-endian, 32 bytes):
/// ```text
/// [0..4]   col_footer_offset  u32  byte offset from block start
/// [4..8]   row_dir_offset     u32  byte offset from block start (0 = no dir)
/// [8..12]  stripe_start       u32  byte offset from block start
/// [12..16] n_columns          u32
/// [16..20] n_rows             u32
/// [20..24] vparam_offset      u32  byte offset of VectorParamBlock (0 = none)
/// [24..28] vparam_len         u32  byte length of VectorParamBlock (0 = none)
/// [28..32] rgdir_offset       u32  byte offset of RowGroupBlock (0 = none)
/// ```
///
/// `vparam_*` and `rgdir_offset` reclaim what were 12 reserved bytes in v1.
/// A reader fetches the trailing 32 bytes first, then range-reads the column
/// footer, the VectorParamBlock, and the RowGroupBlock from these offsets —
/// the footer-first object-store read path.
#[derive(Debug, Clone, Copy, Default)]
pub struct BlockFooter {
    pub col_footer_offset: u32,
    pub row_dir_offset: u32,
    pub stripe_start: u32,
    pub n_columns: u32,
    pub n_rows: u32,
    /// Byte offset of the [`VectorParamBlock`] from block start (0 = none).
    pub vparam_offset: u32,
    /// Byte length of the [`VectorParamBlock`] (0 = none).
    pub vparam_len: u32,
    /// Byte offset of the row-group sub-index (`RowGroupBlock`) from block
    /// start (0 = none). Its length is derivable from its own header.
    pub rgdir_offset: u32,
}

impl BlockFooter {
    pub fn to_bytes(self) -> [u8; BLOCK_FOOTER_SIZE] {
        let mut b = [0u8; BLOCK_FOOTER_SIZE];
        b[0..4].copy_from_slice(&self.col_footer_offset.to_le_bytes());
        b[4..8].copy_from_slice(&self.row_dir_offset.to_le_bytes());
        b[8..12].copy_from_slice(&self.stripe_start.to_le_bytes());
        b[12..16].copy_from_slice(&self.n_columns.to_le_bytes());
        b[16..20].copy_from_slice(&self.n_rows.to_le_bytes());
        b[20..24].copy_from_slice(&self.vparam_offset.to_le_bytes());
        b[24..28].copy_from_slice(&self.vparam_len.to_le_bytes());
        b[28..32].copy_from_slice(&self.rgdir_offset.to_le_bytes());
        b
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        use anyhow::bail;
        if b.len() < BLOCK_FOOTER_SIZE {
            bail!("BlockFooter slice too short");
        }
        Ok(Self {
            col_footer_offset: u32::from_le_bytes(b[0..4].try_into()?),
            row_dir_offset: u32::from_le_bytes(b[4..8].try_into()?),
            stripe_start: u32::from_le_bytes(b[8..12].try_into()?),
            n_columns: u32::from_le_bytes(b[12..16].try_into()?),
            n_rows: u32::from_le_bytes(b[16..20].try_into()?),
            vparam_offset: u32::from_le_bytes(b[20..24].try_into()?),
            vparam_len: u32::from_le_bytes(b[24..28].try_into()?),
            rgdir_offset: u32::from_le_bytes(b[28..32].try_into()?),
        })
    }
}

/// PAX block writer — buffers records and serialises to PAX block bytes.
/// Per-segment vector quantization selection (P3 Phase D — caller/collection-controlled,
/// replacing the process-global env flags). `Auto` preserves the legacy env behavior
/// (`PROXIMADB_VECTOR_RABITQ_ENABLE` / `PROXIMADB_VECTOR_SQ8_DISABLE`) so existing callers are
/// unchanged; the flush passes an explicit strategy from the collection's config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VectorQuant {
    /// Decide from env (legacy default).
    #[default]
    Auto,
    /// Raw fixed-stride f32 (no quantization).
    RawF32,
    /// SQ8 (4×, lossy within scale/2).
    Sq8,
    /// RaBitQ (~30×, lossy 1-bit) — requires a decoupled f32 rerank tier for recall.
    RaBitQ,
    /// FP16 (2×, near-lossless). Suitable as a tier-2 rerank column when higher
    /// fidelity than SQ8 is needed (recall ~0.999 vs ~0.98).
    Fp16,
}

pub struct PaxBlockWriter {
    mode: BlockMode,
    compression: BlockCompression,
    collection_id_hash: u64,
    schema_fingerprint: u64,
    /// Number of embedding columns expected (from collection schema).
    embedding_count: usize,
    /// Vector quantization strategy for this writer (P3 Phase D; `Auto` = env-driven).
    quant: VectorQuant,
    /// Drop the per-row `tenant_id` stripe (catalog-resolution): the segment is
    /// path-isolated to one tenant, so the reader stamps tenant from the
    /// catalog/path context. The block-header `tenant_id_hash` (the RLS skip) is
    /// retained. Default from `PROXIMADB_PAX_DROP_TENANT_COL` (off) so the write
    /// format only changes once readers that stamp tenant are deployed.
    drop_tenant_col: bool,
    /// Opt-in exact-f32 tier (P3 Phase D): when true, the writer ALSO emits the
    /// raw f32 embedding at `col_id::F32_TIER_BASE + i` (one
    /// stripe per embedding column). Read LAZILY — only an exact final rerank or
    /// `include_vectors` decodes it; id+score queries never touch it (zero scan /
    /// egress cost when unused). Default OFF (set via `with_f32_tier`).
    include_f32_tier: bool,
    /// Quantization strategy for the co-located rerank column (RERANK_BASE).
    /// Default `Sq8` (the validated tier-2); `Fp16` for near-lossless rerank;
    /// `RawF32` for exact (equivalent to the f32 tier but at RERANK_BASE).
    /// Only used when tier 1 is RaBitQ (the rerank column is emitted alongside
    /// RaBitQ; non-RaBitQ tier 1 doesn't need a separate rerank column).
    rerank_quant: VectorQuant,
    /// Apply the exact clustered FOR/bit-pack transform to SQ8 code bytes when
    /// its realized payload is smaller. Default OFF; readers key off the
    /// VectorParamBlock transform trailer rather than inspecting payload magic.
    clustered_sq8_lossless: bool,
    /// Apply exact all-null elision and shared LZ4 to scalar stripes when smaller.
    lossless_scalar: bool,
    /// Emit the additive canonical MVCC `record_version` stripe. Default OFF
    /// until mixed-version PAX read/write evidence is baked; the stripe's
    /// presence is its marker and older blocks decode as legacy version 1.
    record_version_stripe: bool,
    /// Row offsets at which a new producer-defined cluster starts. Row zero is
    /// implicit. These boundaries are block-local and reset by [`Self::clear`].
    cluster_run_starts: Vec<usize>,
    /// ADR-065 Region B: when true, the tier-1 vector stripe (`EMBED_BASE`, e.g.
    /// SQ8) + its co-located rerank column are **hoisted into segment-level
    /// regions** (A: RaBitQ, B: SQ8), so the block emits neither — it is pure row
    /// data (Region D). The f32 exact tier (`F32_TIER`) is NOT hoisted here; it
    /// stays in the block until PR4 moves it to Region C. Set by the segment
    /// writer in coalesced mode so survivor rerank fetches read dense Region B
    /// instead of dragging the block's full row payload.
    hoist_vector_tier: bool,
    /// TD-PAXRG-1: suppress the in-block RowGroupBlock sub-index (`rgdir`). Set
    /// for v4 row-group Region D blocks, where the RG **is** the granule — its
    /// stats live in the segment footer's per-entry MinMax payload, so an
    /// in-block sub-index would be redundant framing. `BlockFooter.rgdir_offset`
    /// serializes 0 (readers already treat 0 as "none").
    suppress_rgdir: bool,

    // Accumulated column data
    oids: Vec<String>,
    tenant_ids: Vec<String>,
    created_at: Vec<i64>,
    updated_at: Vec<i64>,
    record_versions: Vec<u64>,
    valid_from: Vec<Option<i64>>,
    valid_to: Vec<Option<i64>>,
    actors: Vec<Option<String>>,
    origins: Vec<Option<String>>,
    props_bytes: Vec<Option<Vec<u8>>>,
    labels_bytes: Vec<Option<Vec<u8>>>,
    edge_src: Vec<Option<String>>,
    edge_tgt: Vec<Option<String>>,
    edge_type: Vec<Option<String>>,
    edge_weight: Vec<Option<f64>>,
    /// One `Vec<Vec<f32>>` per embedding position.
    embeddings: Vec<Vec<Option<Vec<f32>>>>,

    /// P-Shred (ADR-055): declared/hot props keys to shred into typed user-columns,
    /// as `(prop_key, user_col_id ≥ USER_BASE)`. Empty ⇒ no shredding (byte-for-byte
    /// today's behavior). These columns are a DUPLICATED pruning/projection index;
    /// the msgpack `PROPS` tail stays the source of truth (the reader reconstructs
    /// from the tail and ignores `USER_BASE+` columns), so this is additive and
    /// mixed-read-safe with no format-version bump.
    shred_spec: Vec<ShredColumn>,
    /// TD-USUB-6 2b: drop losslessly-shreddable values from the msgpack tail.
    /// Default OFF — with it off the tail stays complete and output is
    /// byte-for-byte today's.
    residual_tail: bool,
    /// Set once any row actually had a value removed from its tail, which is
    /// what makes the block v3: readers MUST reconstruct from the stripes.
    tail_is_residual: bool,
    /// One buffer per `shred_spec` entry: the CLONED prop value per row (never
    /// removed from `props` — the tail must remain complete).
    user_col_buffers: Vec<Vec<Option<proximadb_data_model::ProximaValue>>>,

    // MVCC metadata for row directory
    tenant_id_hash_set: u64,
    min_ts: i64,
    max_ts: i64,
    /// ACTUAL accumulated metadata bytes (oids + props + labels + timestamps —
    /// excludes embeddings, which are quantized at flush and predicted separately
    /// by PaxSegmentWriter). Used by PaxSegmentWriter for an accurate block-flush
    /// threshold (replaces the flat per-row estimate).
    accumulated_metadata_bytes: usize,
}

impl PaxBlockWriter {
    pub fn new(
        mode: BlockMode,
        compression: BlockCompression,
        collection_id: &str,
        schema_fingerprint: u64,
        embedding_count: usize,
    ) -> Self {
        Self {
            mode,
            compression,
            collection_id_hash: fnv1a_hash(collection_id),
            schema_fingerprint,
            embedding_count,
            quant: VectorQuant::Auto,
            drop_tenant_col: drop_tenant_col_enabled(),
            include_f32_tier: false,
            rerank_quant: VectorQuant::Sq8,
            clustered_sq8_lossless: false,
            lossless_scalar: false,
            record_version_stripe: record_version_stripe_enabled(),
            cluster_run_starts: Vec::new(),
            hoist_vector_tier: false,
            suppress_rgdir: false,
            oids: Vec::new(),
            tenant_ids: Vec::new(),
            created_at: Vec::new(),
            updated_at: Vec::new(),
            record_versions: Vec::new(),
            valid_from: Vec::new(),
            valid_to: Vec::new(),
            actors: Vec::new(),
            origins: Vec::new(),
            props_bytes: Vec::new(),
            labels_bytes: Vec::new(),
            edge_src: Vec::new(),
            edge_tgt: Vec::new(),
            edge_type: Vec::new(),
            edge_weight: Vec::new(),
            embeddings: vec![Vec::new(); embedding_count],
            shred_spec: Vec::new(),
            residual_tail: residual_tail_enabled(),
            tail_is_residual: false,
            user_col_buffers: Vec::new(),
            tenant_id_hash_set: 0,
            min_ts: i64::MAX,
            max_ts: i64::MIN,
            accumulated_metadata_bytes: 0,
        }
    }

    /// Set the vector quantization strategy (P3 Phase D). Builder form so existing
    /// `new(..)` call sites are unchanged; `Auto` (the default) keeps env behavior.
    pub fn with_quant(mut self, quant: VectorQuant) -> Self {
        self.quant = quant;
        self
    }

    /// Enable (or disable) dropping the per-row `tenant_id` stripe
    /// (catalog-resolution). Builder form mirroring [`with_quant`]; the default
    /// comes from `PROXIMADB_PAX_DROP_TENANT_COL`.
    pub fn with_drop_tenant_col(mut self, enabled: bool) -> Self {
        self.drop_tenant_col = enabled;
        self
    }

    /// Enable (or disable) the optional exact-f32 tier (P3 Phase D). When true,
    /// each embedding also gets a co-located raw-f32 stripe at
    /// `col_id::F32_TIER_BASE + i` for an exact final rerank / `include_vectors`.
    /// Default OFF; the flush path enables it from the `pax_f32_tier` tag / env.
    pub fn with_f32_tier(mut self, enabled: bool) -> Self {
        self.include_f32_tier = enabled;
        self
    }

    /// Set the quantization strategy for the co-located rerank column
    /// (RERANK_BASE). Default `Sq8`; `Fp16` for near-lossless; `RawF32` for
    /// exact. Only used when tier 1 is RaBitQ.
    pub fn with_rerank_quant(mut self, quant: VectorQuant) -> Self {
        self.rerank_quant = quant;
        self
    }

    /// Enable the exact clustered transform for SQ8 code bytes. The writer
    /// still stores flat SQ8 unless the complete encoded payload is smaller.
    pub fn with_clustered_sq8_lossless(mut self, enabled: bool) -> Self {
        self.clustered_sq8_lossless = enabled;
        self
    }

    /// Enable exact Parquet-style all-null pages and post-codec scalar LZ4.
    pub fn with_lossless_scalar(mut self, enabled: bool) -> Self {
        self.lossless_scalar = enabled;
        self
    }

    /// Enable the additive fixed-width `u64` MVCC version stripe.
    pub fn with_record_version(mut self, enabled: bool) -> Self {
        self.record_version_stripe = enabled;
        self
    }

    /// Whether this writer emits the additive MVCC version stripe.
    pub fn writes_record_version(&self) -> bool {
        self.record_version_stripe
    }

    /// Mark the next appended row as the start of a new cluster run.
    ///
    /// The clustering producer calls this at ordering boundaries. Calls before
    /// the first row or repeated at the same boundary are harmless.
    pub fn start_cluster_run(&mut self) {
        let start = self.row_count();
        if start > 0 && self.cluster_run_starts.last().copied() != Some(start) {
            self.cluster_run_starts.push(start);
        }
    }

    /// ADR-065 Region B: when true, the tier-1 vector (`EMBED_BASE`) + its rerank
    /// column are hoisted into segment-level regions (A/B) — this block emits
    /// neither (pure row data, Region D); the f32 tier still emits when set via
    /// `with_f32_tier`. Set by the segment writer in coalesced mode.
    pub fn with_hoist_vector_tier(mut self, enabled: bool) -> Self {
        self.hoist_vector_tier = enabled;
        self
    }

    /// TD-PAXRG-1: suppress the in-block RowGroupBlock sub-index for v4
    /// row-group Region D blocks (the RG is the granule; its stats live in the
    /// segment footer). `BlockFooter.rgdir_offset` serializes 0.
    pub fn with_suppress_rgdir(mut self, enabled: bool) -> Self {
        self.suppress_rgdir = enabled;
        self
    }

    /// P-Shred (ADR-055): shred the given props keys into typed user-columns
    /// (`USER_BASE`+) for future zone-map/bloom pruning + projection pushdown, while
    /// keeping the full msgpack `PROPS` tail intact. `spec` is `(prop_key, col_id)`;
    /// empty ⇒ no shredding. Builder form mirroring [`with_quant`].
    pub fn with_shred_spec(self, spec: Vec<(String, i32)>) -> Self {
        // Shim for callers whose declared type is unknown or only coarsely known
        // (the SST flush path's `FilterableDataType`). Such columns can never be
        // authoritative, so they carry `declared_type: None`.
        self.with_shred_columns(
            spec.into_iter()
                .map(|(key, id)| ShredColumn::untyped(key, id))
                .collect(),
        )
    }

    /// Set the residual-tail mode explicitly, overriding the process env gate.
    ///
    /// The gate ([`RESIDUAL_TAIL_ENV`]) is read through a `OnceLock`, i.e. ONCE
    /// per process — which is correct for a server that reads configuration at
    /// startup, but makes the flag untestable and unmeasurable from a single
    /// process that wants to compare both modes. The write-amplification reporter
    /// needs exactly that comparison, and without this builder it silently
    /// measured the first-initialised value twice.
    pub fn with_residual_tail(mut self, residual: bool) -> Self {
        self.residual_tail = residual;
        self
    }

    /// Declare the shred columns with their exact types where known (TD-USUB-6).
    pub fn with_shred_columns(mut self, spec: Vec<ShredColumn>) -> Self {
        self.user_col_buffers = vec![Vec::new(); spec.len()];
        self.shred_spec = spec;
        self
    }

    pub fn row_count(&self) -> usize {
        self.oids.len()
    }

    /// Actual accumulated metadata bytes (oids + props + labels + timestamps —
    /// excludes embeddings, which are quantized at flush). Used by PaxSegmentWriter
    /// for an accurate block-flush threshold.
    pub fn accumulated_metadata_bytes(&self) -> usize {
        self.accumulated_metadata_bytes
    }

    pub fn is_empty(&self) -> bool {
        self.oids.is_empty()
    }

    /// Minimum `created_at_ns` seen across all buffered records. Returns 0 if empty.
    pub fn min_ts(&self) -> i64 {
        if self.min_ts == i64::MAX {
            0
        } else {
            self.min_ts
        }
    }

    /// Maximum `created_at_ns` seen across all buffered records. Returns 0 if empty.
    pub fn max_ts(&self) -> i64 {
        if self.max_ts == i64::MIN {
            0
        } else {
            self.max_ts
        }
    }

    /// Buffer one `ProximaRecord` for inclusion in the next `flush()`.
    pub fn add_record(&mut self, record: &ProximaRecord) -> Result<()> {
        // TD-USUB-6 2b: with residual mode on, a prop whose typed stripe can give
        // it back EXACTLY is dropped from the msgpack tail — that removal is the
        // write-amplification win. Anything the stripe cannot reproduce
        // identically stays, so losslessness holds by construction.
        let residual_keys: Vec<&str> = if self.residual_tail {
            self.shred_spec
                .iter()
                .filter_map(|c| {
                    let t = c.declared_type.as_ref()?;
                    match record.props.get(&c.key) {
                        Some(proximadb_records::ProximaTreeNode::Value(v))
                            if shreds_losslessly(v, t) =>
                        {
                            Some(c.key.as_str())
                        }
                        _ => None,
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        let flat = if residual_keys.is_empty() {
            FlatRow::from_record(record)?
        } else {
            FlatRow::from_record_with_residual(record, &residual_keys)?
        };
        if !residual_keys.is_empty() {
            // The tail is now a residual, so the block can only be read by a
            // reader that reconstructs from the stripes. Mark it.
            self.tail_is_residual = true;
        }

        // Update block-level stats
        let ts = flat.created_at_ns;
        if ts < self.min_ts {
            self.min_ts = ts;
        }
        if ts > self.max_ts {
            self.max_ts = ts;
        }

        // Track tenant hash — if all rows share one tenant, store it;
        // otherwise use 0 (multi-tenant block, cannot prune).
        let th = fnv1a_hash(&flat.tenant_id);
        if self.oids.is_empty() {
            self.tenant_id_hash_set = th;
        } else if self.tenant_id_hash_set != th {
            self.tenant_id_hash_set = 0; // mixed tenants
        }

        // Track ACTUAL metadata bytes (oids + props + labels + timestamps — NOT embeddings,
        // which are quantized at flush and predicted separately by PaxSegmentWriter).
        // Computed BEFORE the fields are moved into the column buffers below.
        self.accumulated_metadata_bytes += flat.oid.len() + flat.tenant_id.len()
            + 8 + 8 // created_at_ns + updated_at_ns (i64 each)
            + usize::from(self.record_version_stripe) * 8
            + flat.valid_from_ns.map_or(0, |_| 8)
            + flat.valid_to_ns.map_or(0, |_| 8)
            + flat.actor.as_ref().map_or(0, |s| s.len())
            + flat.origin.as_ref().map_or(0, |s| s.len())
            + flat.props_bytes.as_ref().map_or(0, |b| b.len())
            + flat.labels_bytes.as_ref().map_or(0, |b| b.len())
            + flat.edge_src.as_ref().map_or(0, |s| s.len())
            + flat.edge_tgt.as_ref().map_or(0, |s| s.len())
            + flat.edge_type.as_ref().map_or(0, |s| s.len())
            + flat.edge_weight.map_or(0, |_| 8);

        self.oids.push(flat.oid);
        self.tenant_ids.push(flat.tenant_id);
        self.created_at.push(flat.created_at_ns);
        self.updated_at.push(flat.updated_at_ns);
        if self.record_version_stripe {
            self.record_versions.push(flat.record_version);
        }
        self.valid_from.push(flat.valid_from_ns);
        self.valid_to.push(flat.valid_to_ns);
        self.actors.push(flat.actor);
        self.origins.push(flat.origin);
        self.props_bytes.push(flat.props_bytes);
        self.labels_bytes.push(flat.labels_bytes);
        self.edge_src.push(flat.edge_src);
        self.edge_tgt.push(flat.edge_tgt);
        self.edge_type.push(flat.edge_type);
        self.edge_weight.push(flat.edge_weight);

        // Pad or truncate embeddings to match expected count
        for i in 0..self.embedding_count {
            let v = flat.embeddings.get(i).cloned();
            self.embeddings[i].push(v);
        }

        // P-Shred: buffer the shredded prop values BY CLONE. `FlatRow::from_record`
        // above kept the FULL props tree in `props_bytes` (the source of truth), so
        // reading each key here with `.get(..).cloned()` — never `.remove(..)` —
        // guarantees the shredded column equals the tail value and loses nothing.
        for (i, col) in self.shred_spec.iter().enumerate() {
            let key = &col.key;
            let v = match record.props.get(key) {
                Some(proximadb_records::ProximaTreeNode::Value(v)) => Some(v.clone()),
                _ => None,
            };
            self.user_col_buffers[i].push(v);
        }

        Ok(())
    }

    /// Serialise all buffered records to a PAX block byte vector.
    ///
    /// After flush, the writer is NOT reset — call `clear()` if reuse is intended.
    pub fn flush(&mut self) -> Result<Vec<u8>> {
        let n = self.oids.len();
        let mode = self.mode;

        // ---- Build row directory (OLTP/PAX) ----
        let mut row_dir = RowDirectory::new();
        if mode.has_row_directory() {
            for (i, oid) in self.oids.iter().enumerate() {
                let hash = fnv1a_hash(oid);
                let from_ns = self.valid_from[i].unwrap_or(0);
                row_dir.push(RowEntry::new(hash, from_ns, i as u32));
            }
        }
        let row_dir_bytes = if mode.has_row_directory() {
            row_dir.to_bytes()
        } else {
            Vec::new()
        };

        // ---- Build column stripes (OLAP/PAX) ----
        let mut stripes: Vec<ColumnStripe> = Vec::new();
        // One entry per vector column, serialized into the footer's VectorParamBlock.
        let mut vparam_entries: Vec<VectorParamEntry> = Vec::new();
        // RaBitQ side data (centroid + seed) for any binary-quantized columns.
        let mut vparam_rabitq: Vec<RaBitQColumn> = Vec::new();
        // Exact post-quantization transforms selected by realized byte size.
        let mut vparam_transforms: Vec<VectorTransformColumn> = Vec::new();

        if mode.has_column_stripes() {
            stripes.push(
                self.build_str_stripe(
                    col_id::OID,
                    "id",
                    ColumnRole::Identity,
                    false,
                    &self
                        .oids
                        .iter()
                        .map(|s| Some(s.as_str()))
                        .collect::<Vec<_>>(),
                ),
            );
            // Catalog-resolution: a segment is path-isolated to one tenant
            // (DrPathBuilder data/{tenant}/{namespace}/...), so the per-row tenant
            // is redundant — the reader stamps it from the catalog/path context.
            // When the flag is set, drop the stripe entirely; the block-header
            // tenant_id_hash (computed in `add_record`) still carries the RLS skip.
            if !self.drop_tenant_col {
                stripes.push(
                    self.build_str_stripe(
                        col_id::TENANT_ID,
                        "tenant_id",
                        ColumnRole::Tenant,
                        false,
                        &self
                            .tenant_ids
                            .iter()
                            .map(|s| Some(s.as_str()))
                            .collect::<Vec<_>>(),
                    ),
                );
            }
            stripes.push(self.build_i64_stripe(
                col_id::CREATED_AT,
                ColumnRole::Timestamp,
                false,
                &self.created_at.iter().map(|&v| Some(v)).collect::<Vec<_>>(),
            )?);
            stripes.push(self.build_i64_stripe(
                col_id::UPDATED_AT,
                ColumnRole::Timestamp,
                false,
                &self.updated_at.iter().map(|&v| Some(v)).collect::<Vec<_>>(),
            )?);
            if self.record_version_stripe {
                stripes.push(self.build_u64_stripe(
                    col_id::RECORD_VERSION,
                    ColumnRole::Temporal,
                    &self.record_versions,
                ));
            }
            stripes.push(self.build_i64_stripe(
                col_id::VALID_FROM,
                ColumnRole::Temporal,
                true,
                &self.valid_from,
            )?);
            stripes.push(self.build_i64_stripe(
                col_id::VALID_TO,
                ColumnRole::Temporal,
                true,
                &self.valid_to,
            )?);

            for (actor_opt, origin_opt) in self.actors.iter().zip(self.origins.iter()) {
                let _ = actor_opt;
                let _ = origin_opt; // will encode below
            }
            stripes.push(self.build_str_opt_stripe(
                col_id::ACTOR,
                ColumnRole::Provenance,
                &self.actors,
            ));
            stripes.push(self.build_str_opt_stripe(
                col_id::ORIGIN,
                ColumnRole::Provenance,
                &self.origins,
            ));
            stripes.push(self.build_bytes_stripe(
                col_id::PROPS,
                ColumnRole::Props,
                &self.props_bytes,
            ));
            stripes.push(self.build_bytes_stripe(
                col_id::LABELS,
                ColumnRole::Props,
                &self.labels_bytes,
            ));
            stripes.push(self.build_str_opt_stripe(
                col_id::EDGE_SRC,
                ColumnRole::Edge,
                &self.edge_src,
            ));
            stripes.push(self.build_str_opt_stripe(
                col_id::EDGE_TGT,
                ColumnRole::Edge,
                &self.edge_tgt,
            ));
            stripes.push(self.build_str_opt_stripe(
                col_id::EDGE_TYPE,
                ColumnRole::Edge,
                &self.edge_type,
            ));
            stripes.push(self.build_f64_stripe(
                col_id::EDGE_WEIGHT,
                ColumnRole::Edge,
                true,
                &self.edge_weight,
            )?);

            for (i, col) in self.embeddings.iter().enumerate() {
                let refs: Vec<Option<&[f32]>> = col.iter().map(|v| v.as_deref()).collect();
                if !self.hoist_vector_tier {
                    let (stripe, entry, rabitq_col, transform) =
                        self.build_f32_vec_stripe(col_id::EMBED_BASE + i as i32, &refs)?;
                    let is_rabitq = rabitq_col.is_some();
                    stripes.push(stripe);
                    vparam_entries.push(entry);
                    if let Some(rc) = rabitq_col {
                        vparam_rabitq.push(rc);
                    }
                    if let Some(transform) = transform {
                        vparam_transforms.push(transform);
                    }
                    // P3 cascade: every RaBitQ-coded embedding gets a co-located rerank
                    // column at `RERANK_BASE + i`. The rerank quant strategy is
                    // configurable (default SQ8; Fp16 for near-lossless; RawF32 for
                    // exact). RaBitQ codes drive the cheap candidate scan; the rerank
                    // pool is scored against this co-located copy before the final top-k.
                    if is_rabitq {
                        let (rerank_stripe, rerank_entry, rerank_transform) = match self
                            .rerank_quant
                        {
                            VectorQuant::Fp16 => {
                                let (stripe, entry) = self
                                    .build_fp16_vec_stripe(col_id::RERANK_BASE + i as i32, &refs)?;
                                (stripe, entry, None)
                            }
                            VectorQuant::RawF32 => {
                                let (stripe, entry) = self.build_raw_f32_vec_stripe(
                                    col_id::RERANK_BASE + i as i32,
                                    &refs,
                                )?;
                                (stripe, entry, None)
                            }
                            // Default: SQ8 (the validated tier-2).
                            _ => {
                                self.build_sq8_vec_stripe(col_id::RERANK_BASE + i as i32, &refs)?
                            }
                        };
                        stripes.push(rerank_stripe);
                        vparam_entries.push(rerank_entry);
                        if let Some(transform) = rerank_transform {
                            vparam_transforms.push(transform);
                        }
                    }
                } // end hoist_vector_tier — f32 tier below stays in the block (Region D)
                // P3 Phase D f32 tier (opt-in): an exact-f32 copy of the embedding
                // at `F32_TIER_BASE + i`, for an exact final rerank (→ recall ≈ 1.0)
                // and exact `include_vectors`. The stripe is read LAZILY — id+score
                // queries never decode it — so the only always-paid cost is the
                // storage bytes (the egress/scan cost is paid only when used).
                if self.include_f32_tier {
                    let (f32_stripe, f32_entry) =
                        self.build_raw_f32_vec_stripe(col_id::F32_TIER_BASE + i as i32, &refs)?;
                    stripes.push(f32_stripe);
                    vparam_entries.push(f32_entry);
                }
            }

            // P-Shred (ADR-055): one typed user-column stripe per shredded prop key,
            // at its catalog-assigned `col_id` (≥ USER_BASE). A DUPLICATED pruning/
            // projection index — the `PROPS` tail above is authoritative, so the
            // reader ignores these on reconstruction (mixed-read-safe). Empty spec ⇒
            // this loop is a no-op ⇒ byte-for-byte today's output.
            for (i, col) in self.shred_spec.iter().enumerate() {
                stripes.push(self.build_shred_stripe(
                    col.column_id,
                    &self.user_col_buffers[i],
                    col.declared_type.as_ref(),
                )?);
            }
        }

        // ---- Assign stripe offsets ----
        let row_dir_offset = HEADER_SIZE as u32;
        let stripe_start = row_dir_offset + row_dir_bytes.len() as u32;
        let mut cursor = stripe_start;
        for s in &mut stripes {
            s.meta.stripe_offset = cursor;
            cursor += s.meta.stripe_len;
        }

        // ---- P-Shred self-describing directory (TD-USUB-6 slice 2a) ----
        // Emit the block's OWN (col_id -> prop_key) map so reconstruction is
        // driven by what was WRITTEN, not by a caller-supplied positional key
        // list that can drift with schema evolution. Carries no stripe
        // (`stripe_len == 0`); the payload rides the footer-extras region via the
        // existing pointer pair, so this is additive in v2.
        let mut shred_dir_payload: Vec<u8> = Vec::new();
        if !self.shred_spec.is_empty() {
            let dir: Vec<ShredDirectoryEntry> = self
                .shred_spec
                .iter()
                .map(|c| ShredDirectoryEntry {
                    column_id: c.column_id,
                    key: c.key.clone(),
                    declared_type: c.declared_type.clone(),
                })
                .collect();
            let payload = rmp_serde::to_vec(&dir)?;
            let meta = ColumnMeta {
                column_id: col_id::SHRED_DIRECTORY,
                // Opaque msgpack, like PROPS — deliberately NOT a new role byte,
                // which an older build's `ColumnRole::from_u8` would reject and
                // so would break mixed reads of v2 blocks.
                role: ColumnRole::Props,
                data_type_id: 0xff,
                encoding_id: 0,
                nullable: false,
                has_bloom: false,
                is_sorted: false,
                is_lz4_compressed: false,
                stripe_offset: 0,
                stripe_len: 0,
                null_count: 0,
                distinct_hint: 0,
                min_val: [0u8; 16],
                max_val: [0u8; 16],
                bloom_offset: 0,
                bloom_len: 0,
            };
            // Pointer fields are filled after the bloom loop below so that
            // `has_bloom` stays FALSE — this entry has no bloom, it only borrows
            // the pointer pair.
            stripes.push(ColumnStripe::new(meta, Vec::new()));
            shred_dir_payload = payload;
        }

        // ---- Build column footer and footer-resident pruning payloads ----
        let col_footer_offset = cursor;
        let mut footer_extra_bytes = Vec::new();
        let bloom_base_offset = stripes.len() * COLUMN_META_SIZE;
        for s in &mut stripes {
            if !s.bloom.is_empty() {
                s.meta.has_bloom = true;
                s.meta.bloom_offset = (bloom_base_offset + footer_extra_bytes.len()) as u32;
                s.meta.bloom_len = s.bloom.len() as u32;
                footer_extra_bytes.extend_from_slice(&s.bloom);
            }
        }

        // Place the shred directory in the same footer-extras region, addressed
        // by the same pointer pair but WITHOUT claiming a bloom filter.
        if !shred_dir_payload.is_empty()
            && let Some(dir_stripe) = stripes
                .iter_mut()
                .find(|s| s.meta.column_id == col_id::SHRED_DIRECTORY)
        {
            dir_stripe.meta.bloom_offset = (bloom_base_offset + footer_extra_bytes.len()) as u32;
            dir_stripe.meta.bloom_len = shred_dir_payload.len() as u32;
            footer_extra_bytes.extend_from_slice(&shred_dir_payload);
        }

        let mut col_footer_bytes =
            Vec::with_capacity(stripes.len() * COLUMN_META_SIZE + footer_extra_bytes.len());
        for s in &stripes {
            col_footer_bytes.extend_from_slice(&s.meta.to_bytes());
        }
        col_footer_bytes.extend_from_slice(&footer_extra_bytes);

        // ---- Build the VectorParamBlock side region (after the column footer) ----
        let (vparam_bytes, vparam_offset, vparam_len) = if vparam_entries.is_empty() {
            (Vec::new(), 0u32, 0u32)
        } else {
            let block = VectorParamBlock {
                entries: vparam_entries,
                rabitq: vparam_rabitq,
                transforms: vparam_transforms,
            };
            let bytes = block.to_bytes();
            let offset = col_footer_offset + col_footer_bytes.len() as u32;
            let len = bytes.len() as u32;
            (bytes, offset, len)
        };

        // ---- Build the RowGroupBlock side region (after the vector params) ----
        // TD-PAXRG-1: suppressed entirely for v4 row-group Region D blocks —
        // the RG is the granule and its stats live in the segment footer.
        let rg_index = if self.suppress_rgdir {
            RowGroupBlock::default()
        } else {
            build_row_group_index(
                n,
                &self.created_at,
                &self.updated_at,
                &self.valid_from,
                &self.valid_to,
                &self.edge_weight,
            )
        };
        let vparam_end =
            col_footer_offset + col_footer_bytes.len() as u32 + vparam_bytes.len() as u32;
        let (rgdir_bytes, rgdir_offset) = if rg_index.is_empty() {
            (Vec::new(), 0u32)
        } else {
            (rg_index.to_bytes(), vparam_end)
        };

        // ---- Build block footer ----
        let block_footer_offset = vparam_end + rgdir_bytes.len() as u32;
        let block_footer = BlockFooter {
            col_footer_offset,
            row_dir_offset,
            stripe_start,
            n_columns: stripes.len() as u32,
            n_rows: n as u32,
            vparam_offset,
            vparam_len,
            rgdir_offset,
        };

        let total_size = block_footer_offset + BLOCK_FOOTER_SIZE as u32;

        // ---- Assemble body (everything after header) ----
        let mut body = Vec::with_capacity(total_size as usize - HEADER_SIZE);
        body.extend_from_slice(&row_dir_bytes);
        for s in &stripes {
            body.extend_from_slice(&s.data);
        }
        body.extend_from_slice(&col_footer_bytes);
        body.extend_from_slice(&vparam_bytes);
        body.extend_from_slice(&rgdir_bytes);
        body.extend_from_slice(&block_footer.to_bytes());

        // ---- Compute checksum ----
        let mut hasher = Crc32::new();
        hasher.update(&body);
        let checksum = hasher.finalize();

        // ---- Build header ----
        let block_flags = {
            let mut f = 0u8;
            if stripes.iter().any(|s| s.meta.has_bloom) {
                f |= flags::HAS_BLOOM;
            }
            if !self.embeddings.is_empty() {
                f |= flags::HAS_VECTOR;
            }
            if self.edge_src.iter().any(|v| v.is_some()) {
                f |= flags::HAS_EDGE;
            }
            if mode.has_row_directory() {
                f |= flags::HAS_MVCC;
            }
            if mode.has_row_directory() {
                f |= flags::DIR_SORTED;
            }
            f
        };
        let header = BlockHeader {
            format_version: if self.tail_is_residual {
                // A residual tail cannot be read by a v2 reader (it would return
                // records missing the shredded props), so the block declares v3.
                crate::header::FORMAT_VERSION_V3
            } else {
                BlockHeader::current_version()
            },
            block_mode: mode,
            compression: self.compression,
            flags: block_flags,
            column_count: stripes.len() as u16,
            row_count: n as u32,
            block_size: total_size,
            checksum,
            collection_id_hash: self.collection_id_hash,
            schema_fingerprint: self.schema_fingerprint,
            min_timestamp_ns: if n == 0 { 0 } else { self.min_ts },
            max_timestamp_ns: if n == 0 { 0 } else { self.max_ts },
            tenant_id_hash: self.tenant_id_hash_set,
        };

        // ---- Assemble full block ----
        let mut block = Vec::with_capacity(total_size as usize);
        block.extend_from_slice(&header.to_bytes());
        block.extend_from_slice(&body);

        Ok(block)
    }

    pub fn clear(&mut self) {
        self.oids.clear();
        self.tenant_ids.clear();
        self.created_at.clear();
        self.updated_at.clear();
        self.record_versions.clear();
        self.valid_from.clear();
        self.valid_to.clear();
        self.actors.clear();
        self.origins.clear();
        self.props_bytes.clear();
        self.labels_bytes.clear();
        self.edge_src.clear();
        self.edge_tgt.clear();
        self.edge_type.clear();
        self.edge_weight.clear();
        for col in &mut self.embeddings {
            col.clear();
        }
        self.cluster_run_starts.clear();
        self.tenant_id_hash_set = 0;
        self.min_ts = i64::MAX;
        self.max_ts = i64::MIN;
        self.accumulated_metadata_bytes = 0;
    }

    // ---- Private helpers ----

    fn build_str_stripe(
        &self,
        id: i32,
        _name: &str,
        role: ColumnRole,
        nullable: bool,
        vals: &[Option<&str>],
    ) -> ColumnStripe {
        let scheme = select_str_scheme(role, nullable, vals);
        let (data, null_count) = encode_str_with_scheme(vals, &scheme);
        let (data, is_lz4_compressed) =
            self.encode_lossless_scalar(data, null_count as usize, vals.len());
        let stats = string_stats(vals);
        let meta = ColumnMeta {
            column_id: id,
            role,
            data_type_id: 0xff,
            encoding_id: scheme.to_marker(),
            nullable,
            has_bloom: false,
            is_sorted: false,
            is_lz4_compressed,
            stripe_offset: 0,
            stripe_len: data.len() as u32,
            null_count,
            distinct_hint: stats.distinct_hint,
            min_val: stats.min_hash_bytes,
            max_val: stats.max_hash_bytes,
            bloom_offset: 0,
            bloom_len: 0,
        };
        ColumnStripe::new(meta, data).with_bloom(stats.bloom)
    }

    /// Build an f32 vector stripe (v2 fixed-stride layout) plus its
    /// [`VectorParamEntry`].
    ///
    /// The stripe is `[validity bitmap: ceil(n/8)][fixed-stride payload]`. By
    /// default the payload is SQ8-quantized (1 byte/value, 4× smaller); set
    /// `PROXIMADB_VECTOR_SQ8_DISABLE` to fall back to raw fixed-stride f32. The
    /// per-row dimension prefix of v1 is gone — `dim` and the SQ8 params live
    /// once in the returned entry. Null rows still occupy a full (zeroed) row
    /// slot so any row is seekable by `offset + i * stride` (row-group reads).
    fn build_f32_vec_stripe(
        &self,
        id: i32,
        vals: &[Option<&[f32]>],
    ) -> Result<(
        ColumnStripe,
        VectorParamEntry,
        Option<RaBitQColumn>,
        Option<VectorTransformColumn>,
    )> {
        let dim = vector_column_dim(vals)?;
        let flat: Vec<f32> = vals
            .iter()
            .flatten()
            .flat_map(|s| s.iter().copied())
            .collect();
        // Always derive exact bounds (vmin/vmax) for the entry, even for raw.
        let params = functions::sq8::fit_params(&flat);

        let has_data = dim > 0 && !flat.is_empty();
        // P3 Phase D: explicit per-collection strategy wins; Auto falls back to env.
        let (use_rabitq, use_sq8) = match self.quant {
            VectorQuant::RaBitQ => (has_data, false),
            VectorQuant::Sq8 => (false, has_data),
            VectorQuant::RawF32 | VectorQuant::Fp16 => (false, false),
            VectorQuant::Auto => {
                let r = has_data && rabitq_enabled();
                (r, has_data && !r && !sq8_disabled())
            }
        };
        let (data, scheme, quant_kind, rabitq_col, transformed) = if use_rabitq {
            let (bytes, col) = encode_f32_vec_rabitq(vals, dim, id);
            (bytes, ProximaScheme::RaBitQ, QUANT_RABITQ, Some(col), false)
        } else if use_sq8 {
            let (bytes, transformed) = self.encode_sq8(vals, dim, &params)?;
            (bytes, ProximaScheme::Sq8, QUANT_SQ8, None, transformed)
        } else {
            (
                encode_f32_vec_raw_v2(vals, dim),
                ProximaScheme::Raw,
                QUANT_RAW_F32,
                None,
                false,
            )
        };

        let null_count = vals.iter().filter(|v| v.is_none()).count() as u32;
        let meta = ColumnMeta {
            column_id: id,
            role: ColumnRole::Vector,
            data_type_id: 0x01, // F32
            encoding_id: scheme.to_marker(),
            nullable: true,
            has_bloom: false,
            is_sorted: false,
            is_lz4_compressed: false,
            stripe_offset: 0,
            stripe_len: data.len() as u32,
            null_count,
            distinct_hint: vals
                .iter()
                .filter(|value| value.is_some())
                .count()
                .min(u32::MAX as usize) as u32,
            min_val: [0u8; 16],
            max_val: [0u8; 16],
            bloom_offset: 0,
            bloom_len: 0,
        };
        let entry = VectorParamEntry {
            column_id: id,
            dim,
            quant_kind,
            params,
        };
        let transform = transformed.then_some(VectorTransformColumn {
            column_id: id,
            transform_kind: TRANSFORM_CLUSTERED_FOR_U8,
            transform_version: 1,
        });
        Ok((ColumnStripe::new(meta, data), entry, rabitq_col, transform))
    }

    /// Build an SQ8-quantized vector stripe unconditionally (ignoring the
    /// writer's configured `quant`). This backs the co-located rerank column
    /// emitted alongside each RaBitQ embedding: the candidate pool from the
    /// RaBitQ scan is reranked against full-stride SQ8 here, so the column must
    /// always be SQ8 regardless of the primary quant strategy.
    fn build_sq8_vec_stripe(
        &self,
        id: i32,
        vals: &[Option<&[f32]>],
    ) -> Result<(
        ColumnStripe,
        VectorParamEntry,
        Option<VectorTransformColumn>,
    )> {
        let dim = vector_column_dim(vals)?;
        let flat: Vec<f32> = vals
            .iter()
            .flatten()
            .flat_map(|s| s.iter().copied())
            .collect();
        let params = functions::sq8::fit_params(&flat);
        let (data, transformed) = self.encode_sq8(vals, dim, &params)?;
        let null_count = vals.iter().filter(|v| v.is_none()).count() as u32;
        let meta = ColumnMeta {
            column_id: id,
            role: ColumnRole::Vector,
            data_type_id: 0x01, // F32
            encoding_id: ProximaScheme::Sq8.to_marker(),
            nullable: true,
            has_bloom: false,
            is_sorted: false,
            is_lz4_compressed: false,
            stripe_offset: 0,
            stripe_len: data.len() as u32,
            null_count,
            distinct_hint: vals
                .iter()
                .filter(|value| value.is_some())
                .count()
                .min(u32::MAX as usize) as u32,
            min_val: [0u8; 16],
            max_val: [0u8; 16],
            bloom_offset: 0,
            bloom_len: 0,
        };
        let entry = VectorParamEntry {
            column_id: id,
            dim,
            quant_kind: QUANT_SQ8,
            params,
        };
        let transform = transformed.then_some(VectorTransformColumn {
            column_id: id,
            transform_kind: TRANSFORM_CLUSTERED_FOR_U8,
            transform_version: 1,
        });
        Ok((ColumnStripe::new(meta, data), entry, transform))
    }

    fn encode_sq8(
        &self,
        vals: &[Option<&[f32]>],
        dim: u32,
        params: &functions::Sq8Params,
    ) -> Result<(Vec<u8>, bool)> {
        let (bitmap, codes) = encode_f32_vec_sq8_parts(vals, dim, params);
        if !self.clustered_sq8_lossless || codes.is_empty() {
            return Ok((join_vector_payload(bitmap, codes), false));
        }

        let runs = cluster_runs(vals.len(), &self.cluster_run_starts)?;
        let encoded = functions::clustered_for_bitpack::encode_u8_rows(
            &codes,
            dim as usize,
            &runs,
            functions::clustered_for_bitpack::ClusteredU8Config::default(),
        )?;
        if encoded.as_bytes().len() >= codes.len() {
            return Ok((join_vector_payload(bitmap, codes), false));
        }
        Ok((join_vector_payload(bitmap, encoded.into_bytes()), true))
    }

    /// Build a raw-f32 vector stripe unconditionally (the optional exact-f32
    /// tier). Mirrors [`build_sq8_vec_stripe`] but emits `QUANT_RAW_F32` (no
    /// quantization) so the cascade can do an exact final rerank or
    /// `include_vectors` can return exact vectors. Emitted only when the writer's
    /// `include_f32_tier` flag is set (one stripe per embedding column).
    fn build_raw_f32_vec_stripe(
        &self,
        id: i32,
        vals: &[Option<&[f32]>],
    ) -> Result<(ColumnStripe, VectorParamEntry)> {
        let dim = vector_column_dim(vals)?;
        let flat: Vec<f32> = vals
            .iter()
            .flatten()
            .flat_map(|s| s.iter().copied())
            .collect();
        let params = functions::sq8::fit_params(&flat);
        let data = encode_f32_vec_raw_v2(vals, dim);
        let null_count = vals.iter().filter(|v| v.is_none()).count() as u32;
        let meta = ColumnMeta {
            column_id: id,
            role: ColumnRole::Vector,
            data_type_id: 0x01, // F32
            encoding_id: ProximaScheme::Raw.to_marker(),
            nullable: true,
            has_bloom: false,
            is_sorted: false,
            is_lz4_compressed: false,
            stripe_offset: 0,
            stripe_len: data.len() as u32,
            null_count,
            distinct_hint: vals
                .iter()
                .filter(|value| value.is_some())
                .count()
                .min(u32::MAX as usize) as u32,
            min_val: [0u8; 16],
            max_val: [0u8; 16],
            bloom_offset: 0,
            bloom_len: 0,
        };
        let entry = VectorParamEntry {
            column_id: id,
            dim,
            quant_kind: QUANT_RAW_F32,
            params,
        };
        Ok((ColumnStripe::new(meta, data), entry))
    }

    /// Build an FP16-quantized vector stripe (2 bytes/value, near-lossless).
    /// Used as a configurable tier-2 rerank column when higher fidelity than SQ8
    /// is needed (recall ~0.999 vs ~0.98).
    fn build_fp16_vec_stripe(
        &self,
        id: i32,
        vals: &[Option<&[f32]>],
    ) -> Result<(ColumnStripe, VectorParamEntry)> {
        use half::f16;
        let dim = vector_column_dim(vals)?;
        let flat: Vec<f32> = vals
            .iter()
            .flatten()
            .flat_map(|s| s.iter().copied())
            .collect();
        let params = functions::sq8::fit_params(&flat);
        let mut data = vector_validity_bitmap(vals);
        let stride = dim as usize * 2; // 2 bytes per f16 value
        data.reserve(vals.len() * stride);
        for v in vals {
            match v {
                Some(vec) => {
                    for &x in vec.iter() {
                        data.extend_from_slice(&f16::from_f32(x).to_le_bytes());
                    }
                }
                None => {
                    // Null: reserve stride bytes (will be skipped by the bitmap).
                    data.extend(std::iter::repeat_n(0u8, stride));
                }
            }
        }
        let null_count = vals.iter().filter(|v| v.is_none()).count() as u32;
        let meta = ColumnMeta {
            column_id: id,
            role: ColumnRole::Vector,
            data_type_id: 0x01, // F32 (source type; encoding = FP16)
            encoding_id: ProximaScheme::Raw.to_marker(),
            nullable: true,
            has_bloom: false,
            is_sorted: false,
            is_lz4_compressed: false,
            stripe_offset: 0,
            stripe_len: data.len() as u32,
            null_count,
            distinct_hint: vals
                .iter()
                .filter(|value| value.is_some())
                .count()
                .min(u32::MAX as usize) as u32,
            min_val: [0u8; 16],
            max_val: [0u8; 16],
            bloom_offset: 0,
            bloom_len: 0,
        };
        let entry = VectorParamEntry {
            column_id: id,
            dim,
            quant_kind: QUANT_FP16,
            params,
        };
        Ok((ColumnStripe::new(meta, data), entry))
    }

    fn build_f64_stripe(
        &self,
        id: i32,
        role: ColumnRole,
        nullable: bool,
        vals: &[Option<f64>],
    ) -> Result<ColumnStripe> {
        let (raw_values, null_count) = flatten_f64_values(vals);
        let scheme = select_f64_scheme(role, nullable, vals);
        let data = encode_f64_with_scheme(&raw_values, &scheme)?;
        let (data, is_lz4_compressed) =
            self.encode_lossless_scalar(data, null_count as usize, vals.len());
        let mut meta = ColumnMeta {
            column_id: id,
            role,
            data_type_id: 0x07, // F64
            encoding_id: scheme.to_marker(),
            nullable,
            has_bloom: false,
            is_sorted: false,
            is_lz4_compressed,
            stripe_offset: 0,
            stripe_len: data.len() as u32,
            null_count,
            distinct_hint: distinct_f64_hint(vals),
            min_val: [0u8; 16],
            max_val: [0u8; 16],
            bloom_offset: 0,
            bloom_len: 0,
        };
        // Populate the f64 zone map (min/max) so range predicates can prune the
        // block. NaN is skipped: a range predicate never matches NaN, so leaving
        // it out of the bounds causes no false skips. ±inf IS kept so that open
        // `>`/`<` bounds stay correct.
        let mut bounds: Option<(f64, f64)> = None;
        for &v in vals.iter().flatten() {
            if v.is_nan() {
                continue;
            }
            bounds = Some(match bounds {
                Some((mn, mx)) => (mn.min(v), mx.max(v)),
                None => (v, v),
            });
        }
        if let Some((mn, mx)) = bounds {
            meta.min_val[0..8].copy_from_slice(&mn.to_le_bytes());
            meta.max_val[0..8].copy_from_slice(&mx.to_le_bytes());
        }
        Ok(ColumnStripe::new(meta, data))
    }

    fn build_str_opt_stripe(
        &self,
        id: i32,
        role: ColumnRole,
        vals: &[Option<String>],
    ) -> ColumnStripe {
        let refs: Vec<Option<&str>> = vals.iter().map(|v| v.as_deref()).collect();
        self.build_str_stripe(id, "", role, true, &refs)
    }

    /// P-Shred: build ONE typed user-column stripe (`ColumnRole::UserDefined`) from the
    /// buffered `ProximaValue`s for a shredded prop key. The physical type is inferred from
    /// the first non-null value (int-family → i64, float-family → f64, else → string), and
    /// every row is coerced to it (a value that doesn't fit the chosen type is `None`). This
    /// is a best-effort PRUNING/PROJECTION index — imperfect coercion is safe because the
    /// msgpack `PROPS` tail is the source of truth for reconstruction.
    fn build_shred_stripe(
        &self,
        id: i32,
        vals: &[Option<proximadb_data_model::ProximaValue>],
        declared: Option<&proximadb_data_model::ProximaType>,
    ) -> Result<ColumnStripe> {
        // TD-USUB-6: a DECLARED type fixes the physical class up front. Only an
        // undeclared column still infers it from the first non-null value — the
        // behaviour that made the class depend on row order.
        let class = declared
            .and_then(declared_shred_class)
            .or_else(|| vals.iter().flatten().next().map(shred_class));
        match class {
            Some(ShredClass::Int) => {
                let col: Vec<Option<i64>> = vals
                    .iter()
                    .map(|o| o.as_ref().and_then(pv_to_i64))
                    .collect();
                self.build_i64_stripe(id, ColumnRole::UserDefined, true, &col)
            }
            Some(ShredClass::Float) => {
                let col: Vec<Option<f64>> = vals
                    .iter()
                    .map(|o| o.as_ref().and_then(pv_to_f64))
                    .collect();
                self.build_f64_stripe(id, ColumnRole::UserDefined, true, &col)
            }
            // String class, or an all-null column (default to a nullable string stripe).
            _ => {
                let col: Vec<Option<String>> = vals
                    .iter()
                    .map(|o| o.as_ref().and_then(pv_to_str))
                    .collect();
                Ok(self.build_str_opt_stripe(id, ColumnRole::UserDefined, &col))
            }
        }
    }

    fn build_i64_stripe(
        &self,
        id: i32,
        role: ColumnRole,
        nullable: bool,
        vals: &[Option<i64>],
    ) -> Result<ColumnStripe> {
        let (raw_values, null_count) = flatten_i64_values(vals);
        let scheme = select_i64_scheme(role, nullable, vals);
        let data = encode_i64_with_scheme(&raw_values, &scheme)?;
        let (data, is_lz4_compressed) =
            self.encode_lossless_scalar(data, null_count as usize, vals.len());
        let mut meta = ColumnMeta {
            column_id: id,
            role,
            data_type_id: 0x03,
            encoding_id: scheme.to_marker(),
            nullable,
            has_bloom: false,
            is_sorted: is_i64_sorted(vals),
            is_lz4_compressed,
            stripe_offset: 0,
            stripe_len: data.len() as u32,
            null_count,
            distinct_hint: distinct_i64_hint(vals),
            min_val: [0u8; 16],
            max_val: [0u8; 16],
            bloom_offset: 0,
            bloom_len: 0,
        };
        for v in vals.iter().flatten() {
            update_i64_bounds(&mut meta, *v);
        }
        Ok(ColumnStripe::new(meta, data))
    }

    fn build_u64_stripe(&self, id: i32, role: ColumnRole, vals: &[u64]) -> ColumnStripe {
        let mut data = Vec::with_capacity(vals.len().saturating_mul(std::mem::size_of::<u64>()));
        for value in vals {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let (data, is_lz4_compressed) = self.encode_lossless_scalar(data, 0, vals.len());
        let mut min_val = [0u8; 16];
        let mut max_val = [0u8; 16];
        if let Some(min) = vals.iter().min() {
            min_val[..8].copy_from_slice(&min.to_le_bytes());
        }
        if let Some(max) = vals.iter().max() {
            max_val[..8].copy_from_slice(&max.to_le_bytes());
        }
        let distinct_hint = vals
            .iter()
            .copied()
            .collect::<HashSet<_>>()
            .len()
            .min(u32::MAX as usize) as u32;
        ColumnStripe::new(
            ColumnMeta {
                column_id: id,
                role,
                data_type_id: TypeId::U64.to_u8(),
                encoding_id: ProximaScheme::Raw.to_marker(),
                nullable: false,
                has_bloom: false,
                is_sorted: vals.windows(2).all(|pair| pair[0] <= pair[1]),
                is_lz4_compressed,
                stripe_offset: 0,
                stripe_len: data.len() as u32,
                null_count: 0,
                distinct_hint,
                min_val,
                max_val,
                bloom_offset: 0,
                bloom_len: 0,
            },
            data,
        )
    }

    fn build_bytes_stripe(
        &self,
        id: i32,
        role: ColumnRole,
        vals: &[Option<Vec<u8>>],
    ) -> ColumnStripe {
        let refs: Vec<Option<&str>> = vals
            .iter()
            .map(|v| {
                // Treat bytes as raw; encode length-prefixed
                v.as_ref().map(|_| "")
            })
            .collect();
        // Encode as raw bytes with 4B length prefix
        let mut data = Vec::new();
        let mut null_count = 0u32;
        for v in vals {
            match v {
                Some(b) => {
                    data.extend_from_slice(&(b.len() as u32).to_le_bytes());
                    data.extend_from_slice(b);
                }
                None => {
                    data.extend_from_slice(&u32::MAX.to_le_bytes());
                    null_count += 1;
                }
            }
        }
        let (data, is_lz4_compressed) =
            self.encode_lossless_scalar(data, null_count as usize, vals.len());
        let _ = refs; // suppress warning
        let meta = ColumnMeta {
            column_id: id,
            role,
            data_type_id: 0xff,
            encoding_id: ProximaScheme::Raw.to_marker(),
            nullable: true,
            has_bloom: false,
            is_sorted: false,
            is_lz4_compressed,
            stripe_offset: 0,
            stripe_len: data.len() as u32,
            null_count,
            distinct_hint: distinct_bytes_hint(vals),
            min_val: [0u8; 16],
            max_val: [0u8; 16],
            bloom_offset: 0,
            bloom_len: 0,
        };
        ColumnStripe::new(meta, data)
    }

    /// Apply the exact post-codec storage layer selected for this block.
    /// All-null pages use the existing `null_count` as their definition level
    /// and need no value payload. Other pages retain the encoded bytes unless
    /// shared LZ4 clears the realized-size threshold.
    fn encode_lossless_scalar(
        &self,
        data: Vec<u8>,
        null_count: usize,
        row_count: usize,
    ) -> (Vec<u8>, bool) {
        if !self.lossless_scalar {
            return (data, false);
        }
        if row_count > 0 && null_count == row_count {
            return (Vec::new(), false);
        }
        match functions::lossless_compression::compress_lz4_if_smaller(&data, 8) {
            Ok(Some(compressed)) => (compressed, true),
            Ok(None) | Err(_) => (data, false),
        }
    }
}

/// Physical type class chosen for a shredded user-column (P-Shred). Derived from
/// the EXACT declared type when the column has one (TD-USUB-6), and otherwise
/// inferred from the first non-null `ProximaValue`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShredClass {
    Int,
    Float,
    Str,
}

/// Crate-visible alias so the record reconstruction path can name the class.
pub(crate) use ShredClass as ShredClassPub;

/// Crate-visible wrapper over [`declared_shred_class`].
pub(crate) fn declared_shred_class_pub(
    t: &proximadb_data_model::ProximaType,
) -> Option<ShredClass> {
    declared_shred_class(t)
}

/// Crate-visible wrapper over [`reconstruct_declared`].
pub(crate) fn reconstruct_declared_pub(
    class: ShredClass,
    i: Option<i64>,
    f: Option<f64>,
    s: Option<String>,
    t: &proximadb_data_model::ProximaType,
) -> Option<proximadb_data_model::ProximaValue> {
    reconstruct_declared(class, i, f, s, t)
}

/// Classify a `ProximaValue` into the shred physical type. Integer-like and temporal
/// values map to i64, float-like to f64, everything scalar-textual to string; complex
/// values (Array/Map/Struct/Binary/vectors/etc.) fall through to string, where their
/// coercion returns `None` (not shreddable) — the `PROPS` tail still holds them.
fn shred_class(v: &proximadb_data_model::ProximaValue) -> ShredClass {
    use proximadb_data_model::ProximaValue as PV;
    match v {
        PV::Boolean(_)
        | PV::Int8(_)
        | PV::Int16(_)
        | PV::Int32(_)
        | PV::Int64(_)
        | PV::UInt8(_)
        | PV::UInt16(_)
        | PV::UInt32(_)
        | PV::UInt64(_)
        | PV::Date(_)
        | PV::Time(..)
        | PV::Timestamp(..)
        | PV::TimestampTz(..) => ShredClass::Int,
        PV::Float16(_) | PV::Float32(_) | PV::Float64(_) => ShredClass::Float,
        _ => ShredClass::Str,
    }
}

/// Gate for the residual tail (TD-USUB-6 2b). Unset ⇒ OFF, so the msgpack
/// `PROPS` tail stays complete and block bytes are unchanged.
pub const RESIDUAL_TAIL_ENV: &str = "PROXIMADB_PAX_RESIDUAL_TAIL";

fn residual_tail_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(RESIDUAL_TAIL_ENV)
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Physical stripe class implied by an EXACT declared type (TD-USUB-6 2b).
///
/// For a typed column this replaces first-value inference entirely: the class is
/// a property of the schema, not of whichever row happened to be non-null first.
/// That is what makes the per-row residual decision below answerable at
/// `add_record` time rather than only after every row is buffered.
///
/// `None` for types no physical class represents (Array/Map/Struct/Binary/…) —
/// those are never shredded authoritatively and stay in the tail.
fn declared_shred_class(t: &proximadb_data_model::ProximaType) -> Option<ShredClass> {
    use proximadb_data_model::ProximaType as PT;
    match t {
        PT::Boolean
        | PT::Int8
        | PT::Int16
        | PT::Int32
        | PT::Int64
        | PT::UInt8
        | PT::UInt16
        | PT::UInt32
        | PT::UInt64
        | PT::Date
        | PT::Time(_)
        | PT::Timestamp(_)
        | PT::TimestampTz(_) => Some(ShredClass::Int),
        PT::Float16 | PT::Float32 | PT::Float64 => Some(ShredClass::Float),
        PT::String | PT::Symbol => Some(ShredClass::Str),
        _ => None,
    }
}

/// Rebuild the declared variant from the physical i64 a typed stripe stores.
///
/// This is the inverse the residual tail depends on: a stripe holds i64, so
/// `Int32` and `Timestamp(Micros)` are indistinguishable in it — only the
/// declared type says which to rebuild. Returns `None` when the physical value
/// does not fit the declared type (e.g. an i64 too large for `Int32`), which the
/// caller treats as "not losslessly shreddable, keep it in the tail".
fn i64_to_declared(
    x: i64,
    t: &proximadb_data_model::ProximaType,
) -> Option<proximadb_data_model::ProximaValue> {
    use proximadb_data_model::ProximaType as PT;
    use proximadb_data_model::ProximaValue as PV;
    Some(match t {
        PT::Boolean => match x {
            0 => PV::Boolean(false),
            1 => PV::Boolean(true),
            _ => return None,
        },
        PT::Int8 => PV::Int8(i8::try_from(x).ok()?),
        PT::Int16 => PV::Int16(i16::try_from(x).ok()?),
        PT::Int32 => PV::Int32(i32::try_from(x).ok()?),
        PT::Int64 => PV::Int64(x),
        PT::UInt8 => PV::UInt8(u8::try_from(x).ok()?),
        PT::UInt16 => PV::UInt16(u16::try_from(x).ok()?),
        PT::UInt32 => PV::UInt32(u32::try_from(x).ok()?),
        PT::UInt64 => PV::UInt64(u64::try_from(x).ok()?),
        PT::Date => PV::Date(i32::try_from(x).ok()?),
        PT::Time(u) => PV::Time(x, *u),
        PT::Timestamp(u) => PV::Timestamp(x, *u),
        PT::TimestampTz(u) => PV::TimestampTz(x, *u),
        _ => return None,
    })
}

/// Rebuild the declared variant from the physical f64 a typed stripe stores.
fn f64_to_declared(
    x: f64,
    t: &proximadb_data_model::ProximaType,
) -> Option<proximadb_data_model::ProximaValue> {
    use proximadb_data_model::ProximaType as PT;
    use proximadb_data_model::ProximaValue as PV;
    Some(match t {
        PT::Float16 => PV::Float16(x as f32),
        PT::Float32 => PV::Float32(x as f32),
        PT::Float64 => PV::Float64(x),
        _ => return None,
    })
}

/// Rebuild the declared variant from the physical string a typed stripe stores.
fn str_to_declared(
    x: String,
    t: &proximadb_data_model::ProximaType,
) -> Option<proximadb_data_model::ProximaValue> {
    use proximadb_data_model::ProximaType as PT;
    use proximadb_data_model::ProximaValue as PV;
    Some(match t {
        PT::String => PV::String(x),
        PT::Symbol => PV::Symbol(x),
        _ => return None,
    })
}

/// Reconstruct a declared value from what the stripe physically stored.
pub(crate) fn reconstruct_declared(
    class: ShredClass,
    i: Option<i64>,
    f: Option<f64>,
    s: Option<String>,
    t: &proximadb_data_model::ProximaType,
) -> Option<proximadb_data_model::ProximaValue> {
    match class {
        ShredClass::Int => i64_to_declared(i?, t),
        ShredClass::Float => f64_to_declared(f?, t),
        ShredClass::Str => str_to_declared(s?, t),
    }
}

/// True iff the typed stripe stores `v` such that reconstruction under `t`
/// returns `v` **exactly** — same variant, same payload.
///
/// This is the rule the residual tail is built on (TD-USUB-6): a value is
/// dropped from the msgpack tail ONLY when the stripe can give it back
/// identically. Anything else — a variant that does not match the declared type,
/// a value that does not fit it, a type no physical class represents — stays in
/// the tail. Losslessness is then a property of the encoding rather than a
/// requirement on the data, so heterogeneous prop keys degrade instead of
/// failing the write (which is what the filed "non-conformance is a write error"
/// rule would have done to a schemaless props tree).
fn shreds_losslessly(
    v: &proximadb_data_model::ProximaValue,
    t: &proximadb_data_model::ProximaType,
) -> bool {
    let Some(class) = declared_shred_class(t) else {
        return false;
    };
    let rebuilt = match class {
        ShredClass::Int => pv_to_i64(v).and_then(|x| i64_to_declared(x, t)),
        ShredClass::Float => pv_to_f64(v).and_then(|x| f64_to_declared(x, t)),
        ShredClass::Str => pv_to_str(v).and_then(|x| str_to_declared(x, t)),
    };
    rebuilt.as_ref() == Some(v)
}

fn pv_to_i64(v: &proximadb_data_model::ProximaValue) -> Option<i64> {
    use proximadb_data_model::ProximaValue as PV;
    match v {
        PV::Boolean(b) => Some(*b as i64),
        PV::Int8(x) => Some(*x as i64),
        PV::Int16(x) => Some(*x as i64),
        PV::Int32(x) => Some(*x as i64),
        PV::Int64(x) => Some(*x),
        PV::UInt8(x) => Some(*x as i64),
        PV::UInt16(x) => Some(*x as i64),
        PV::UInt32(x) => Some(*x as i64),
        PV::UInt64(x) => i64::try_from(*x).ok(),
        PV::Date(d) => Some(*d as i64),
        PV::Time(t, _) | PV::Timestamp(t, _) | PV::TimestampTz(t, _) => Some(*t),
        _ => None,
    }
}

fn pv_to_f64(v: &proximadb_data_model::ProximaValue) -> Option<f64> {
    use proximadb_data_model::ProximaValue as PV;
    match v {
        PV::Float16(x) | PV::Float32(x) => Some(*x as f64),
        PV::Float64(x) => Some(*x),
        _ => None,
    }
}

fn pv_to_str(v: &proximadb_data_model::ProximaValue) -> Option<String> {
    use proximadb_data_model::ProximaValue as PV;
    match v {
        PV::String(s) | PV::Symbol(s) | PV::Decimal(s) => Some(s.clone()),
        _ => None,
    }
}

fn flatten_i64_values(values: &[Option<i64>]) -> (Vec<i64>, u32) {
    let mut null_count = 0u32;
    let raw_values = values
        .iter()
        .map(|value| match value {
            Some(value) => *value,
            None => {
                null_count += 1;
                i64::MIN
            }
        })
        .collect();
    (raw_values, null_count)
}

fn flatten_f64_values(values: &[Option<f64>]) -> (Vec<f64>, u32) {
    let mut null_count = 0u32;
    let raw_values = values
        .iter()
        .map(|value| match value {
            Some(value) => *value,
            None => {
                null_count += 1;
                f64::NAN
            }
        })
        .collect();
    (raw_values, null_count)
}

fn is_i64_sorted(values: &[Option<i64>]) -> bool {
    let mut previous = None;
    for value in values.iter().flatten() {
        if let Some(previous) = previous
            && *value < previous
        {
            return false;
        }
        previous = Some(*value);
    }
    true
}

fn distinct_i64_hint(values: &[Option<i64>]) -> u32 {
    values
        .iter()
        .flatten()
        .copied()
        .collect::<HashSet<_>>()
        .len()
        .min(u32::MAX as usize) as u32
}

fn distinct_f64_hint(values: &[Option<f64>]) -> u32 {
    values
        .iter()
        .flatten()
        .map(|v| v.to_bits())
        .collect::<HashSet<_>>()
        .len()
        .min(u32::MAX as usize) as u32
}

fn distinct_bytes_hint(values: &[Option<Vec<u8>>]) -> u32 {
    values
        .iter()
        .flatten()
        .map(|value| value.as_slice())
        .collect::<HashSet<_>>()
        .len()
        .min(u32::MAX as usize) as u32
}

struct StringStats {
    distinct_hint: u32,
    min_hash_bytes: [u8; 16],
    max_hash_bytes: [u8; 16],
    bloom: Vec<u8>,
}

fn string_stats(values: &[Option<&str>]) -> StringStats {
    let mut distinct = HashSet::new();
    let mut min_hash = u64::MAX;
    let mut max_hash = 0u64;
    let mut bloom = vec![0u8; PAX_STRING_BLOOM_BYTES];

    for value in values.iter().flatten() {
        distinct.insert(*value);
        let hash = fnv1a_hash(value);
        min_hash = min_hash.min(hash);
        max_hash = max_hash.max(hash);
        bloom_insert_hash(&mut bloom, hash);
    }

    if distinct.is_empty() {
        bloom.clear();
        return StringStats {
            distinct_hint: 0,
            min_hash_bytes: [0u8; 16],
            max_hash_bytes: [0u8; 16],
            bloom,
        };
    }

    StringStats {
        distinct_hint: distinct.len().min(u32::MAX as usize) as u32,
        min_hash_bytes: hash_bound_bytes(min_hash),
        max_hash_bytes: hash_bound_bytes(max_hash),
        bloom,
    }
}

const PAX_STRING_BLOOM_BYTES: usize = 32;
const PAX_BLOOM_SALTS: [u64; 3] = [
    0x9e37_79b9_7f4a_7c15,
    0xbf58_476d_1ce4_e5b9,
    0x94d0_49bb_1331_11eb,
];

fn hash_bound_bytes(hash: u64) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[0..8].copy_from_slice(&hash.to_le_bytes());
    bytes
}

fn bloom_insert_hash(bloom: &mut [u8], hash: u64) {
    if bloom.is_empty() {
        return;
    }
    let bit_count = bloom.len() * 8;
    for salt in PAX_BLOOM_SALTS {
        let bit = (mix_hash64(hash ^ salt) as usize) % bit_count;
        bloom[bit / 8] |= 1 << (bit % 8);
    }
}

fn mix_hash64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn select_i64_scheme(role: ColumnRole, nullable: bool, values: &[Option<i64>]) -> ProximaScheme {
    let non_null_values: Vec<i64> = values.iter().flatten().copied().collect();
    if non_null_values.is_empty() {
        return ProximaScheme::Raw;
    }

    let analysis = DataAnalysis::from_i64_values(&non_null_values);
    let domain = match role {
        ColumnRole::Timestamp | ColumnRole::Temporal => DataDomain::TimeSeries,
        _ => DataDomain::General,
    };
    let mut context = SelectionContext::for_pax_stripe(TypeId::I64, domain);
    context.target_compression = None;
    context.is_sorted = is_i64_sorted(values);

    let mut profile = CompressionProfile::from_selection_context(&context);
    profile.target_compression_ratio = None;
    profile.hotness = AccessTemperature::Warm;
    profile.workload_profile = WorkloadProfile::Htap;

    let mut hints = context.layout_hints();
    if nullable {
        hints.dictionary_scope = DictionaryScope::Block;
    }

    StrategyRegistry::default()
        .select_decision(&analysis, &context, &profile, &hints)
        .scheme
}

fn select_f64_scheme(role: ColumnRole, _nullable: bool, values: &[Option<f64>]) -> ProximaScheme {
    let non_null_values: Vec<f64> = values.iter().flatten().copied().collect();
    if non_null_values.is_empty() {
        return ProximaScheme::Raw;
    }

    let analysis = DataAnalysis::from_f64_values(&non_null_values);
    let domain = match role {
        ColumnRole::Edge => DataDomain::General,
        _ => DataDomain::TimeSeries,
    };
    let mut context = SelectionContext::for_pax_stripe(TypeId::F64, domain);
    context.target_compression = None;

    let mut profile = CompressionProfile::from_selection_context(&context);
    profile.target_compression_ratio = None;
    profile.hotness = AccessTemperature::Warm;
    profile.workload_profile = WorkloadProfile::Htap;

    let hints = context.layout_hints();
    // `encode_f64_with_scheme` / `decode_f64_with_encoding` only support Raw and
    // Gorilla. Constrain the registry's decision to that set (mirrors
    // `select_str_scheme`). Anything else the strategy registry picks — e.g.
    // RunLength or Dictionary, which it may favor for run-heavy / low-cardinality
    // f64 columns — falls back to Raw: losslessly correct, just uncompressed.
    // Returning an unencodable scheme here previously made `flush()` bail with
    // "unsupported PAX f64 scheme: RunLength".
    match StrategyRegistry::default()
        .select_decision(&analysis, &context, &profile, &hints)
        .scheme
    {
        ProximaScheme::Gorilla => ProximaScheme::Gorilla,
        _ => ProximaScheme::Raw,
    }
}

fn select_str_scheme(role: ColumnRole, nullable: bool, values: &[Option<&str>]) -> ProximaScheme {
    let codes = string_category_codes(values);
    if codes.is_empty() {
        return ProximaScheme::Raw;
    }

    let analysis = DataAnalysis::from_i64_values(&codes);
    let domain = match role {
        ColumnRole::Identity | ColumnRole::Tenant | ColumnRole::Provenance | ColumnRole::Edge => {
            DataDomain::Metadata
        }
        _ => DataDomain::General,
    };
    let mut context = SelectionContext::for_pax_stripe(TypeId::I64, domain);
    context.target_compression = None;

    let mut profile = CompressionProfile::from_selection_context(&context);
    profile.target_compression_ratio = None;
    profile.hotness = AccessTemperature::Warm;
    profile.workload_profile = WorkloadProfile::Htap;

    let mut hints = context.layout_hints();
    if nullable
        || matches!(
            role,
            ColumnRole::Identity | ColumnRole::Tenant | ColumnRole::Provenance | ColumnRole::Edge
        )
    {
        hints.dictionary_scope = DictionaryScope::Block;
    }

    match StrategyRegistry::default()
        .select_decision(&analysis, &context, &profile, &hints)
        .scheme
    {
        ProximaScheme::Dictionary | ProximaScheme::RunLength => ProximaScheme::Dictionary,
        _ => ProximaScheme::Raw,
    }
}

fn string_category_codes(values: &[Option<&str>]) -> Vec<i64> {
    let mut value_to_code: HashMap<&str, i64> = HashMap::new();
    let mut codes = Vec::new();

    for value in values.iter().flatten() {
        let code = if let Some(code) = value_to_code.get(*value) {
            *code
        } else {
            let code = value_to_code.len() as i64;
            value_to_code.insert(*value, code);
            code
        };
        codes.push(code);
    }

    codes
}

fn encode_str_with_scheme(values: &[Option<&str>], scheme: &ProximaScheme) -> (Vec<u8>, u32) {
    match scheme {
        ProximaScheme::Dictionary => encode_str_dictionary_col(values),
        _ => encode_str_col(values),
    }
}

fn encode_str_dictionary_col(values: &[Option<&str>]) -> (Vec<u8>, u32) {
    let mut dictionary = Vec::new();
    let mut value_to_code: HashMap<&str, u32> = HashMap::new();

    for value in values.iter().flatten() {
        if !value_to_code.contains_key(*value) {
            let code = dictionary.len() as u32;
            value_to_code.insert(*value, code);
            dictionary.push(*value);
        }
    }

    let mut buf = Vec::new();
    buf.extend_from_slice(&(dictionary.len() as u32).to_le_bytes());
    for value in &dictionary {
        let bytes = value.as_bytes();
        buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(bytes);
    }

    let mut null_count = 0u32;
    for value in values {
        match value {
            Some(value) => {
                let code = value_to_code[value];
                buf.extend_from_slice(&code.to_le_bytes());
            }
            None => {
                buf.extend_from_slice(&u32::MAX.to_le_bytes());
                null_count += 1;
            }
        }
    }

    (buf, null_count)
}

/// `(column id, per-row i64 accessor)` pair for the prunable timestamp columns.
type I64ColAccessor<'a> = (i32, &'a dyn Fn(usize) -> Option<i64>);

/// Build the row-group sub-index from the writer's in-memory scalar columns.
///
/// For each row group, records per-column `[min, max]` for the prunable i64
/// columns (created/updated/valid timestamps) and the f64 edge weight. Columns
/// whose group has no usable values are omitted (then that group can't be pruned
/// on that column — conservative). Vector byte ranges are NOT stored: they are
/// derivable from the fixed stride + [`crate::vparam`].
fn build_row_group_index(
    n: usize,
    created_at: &[i64],
    updated_at: &[i64],
    valid_from: &[Option<i64>],
    valid_to: &[Option<i64>],
    edge_weight: &[Option<f64>],
) -> RowGroupBlock {
    let rgs = RowGroupBlock::group_count(n as u32);
    let size = crate::rowgroup::ROW_GROUP_SIZE;
    let mut entries: Vec<RowGroupEntry> = Vec::new();

    let i64_cols: [I64ColAccessor; 4] = [
        (col_id::CREATED_AT, &|i| created_at.get(i).copied()),
        (col_id::UPDATED_AT, &|i| updated_at.get(i).copied()),
        (col_id::VALID_FROM, &|i| {
            valid_from.get(i).copied().flatten()
        }),
        (col_id::VALID_TO, &|i| valid_to.get(i).copied().flatten()),
    ];

    for rg in 0..rgs {
        let start = (rg * size) as usize;
        let end = ((rg + 1) * size).min(n as u32) as usize;
        if start >= end {
            continue;
        }

        for (col, get) in i64_cols.iter() {
            let mut lo = i64::MAX;
            let mut hi = i64::MIN;
            let mut any = false;
            for i in start..end {
                if let Some(v) = get(i) {
                    any = true;
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            if any {
                let (min_val, max_val) = i64_bounds(lo, hi);
                entries.push(RowGroupEntry {
                    column_id: *col,
                    rg_index: rg,
                    data_type_id: 0x03,
                    min_val,
                    max_val,
                });
            }
        }

        // f64 edge weight (skip None + NaN).
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        let mut any = false;
        for v in edge_weight.iter().take(end).skip(start).flatten() {
            if v.is_finite() {
                any = true;
                lo = lo.min(*v);
                hi = hi.max(*v);
            }
        }
        if any {
            let (min_val, max_val) = f64_bounds(lo, hi);
            entries.push(RowGroupEntry {
                column_id: col_id::EDGE_WEIGHT,
                rg_index: rg,
                data_type_id: 0x07,
                min_val,
                max_val,
            });
        }
    }

    RowGroupBlock {
        n_row_groups: rgs,
        row_group_size: size,
        entries,
    }
}

/// Kill-switch: when set, vector stripes fall back to raw fixed-stride f32
/// instead of SQ8. Mirrors the project's other `PROXIMADB_*_DISABLE` toggles.
fn sq8_disabled() -> bool {
    std::env::var_os("PROXIMADB_VECTOR_SQ8_DISABLE").is_some()
}

/// Opt-in: when set, vector stripes use RaBitQ binary quantization (1 bit/dim)
/// instead of SQ8 — ~30× smaller, search-by-estimator + rerank. Off by default
/// (SQ8 reconstructs more faithfully without a rerank tier).
fn rabitq_enabled() -> bool {
    std::env::var_os("PROXIMADB_VECTOR_RABITQ_ENABLE").is_some()
}

/// Default for dropping the per-row tenant stripe (catalog-resolution). Off
/// unless `PROXIMADB_PAX_DROP_TENANT_COL` is set, so the on-disk format only
/// changes once readers that stamp tenant from context are deployed.
fn drop_tenant_col_enabled() -> bool {
    std::env::var_os("PROXIMADB_PAX_DROP_TENANT_COL").is_some()
}

/// Default-off write gate for the additive MVCC version stripe. Readers detect
/// the stripe by column ID and remain compatible with blocks written either way.
fn record_version_stripe_enabled() -> bool {
    std::env::var_os("PROXIMADB_PAX_RECORD_VERSION").is_some()
}

/// Encode a RaBitQ vector stripe: validity bitmap + per row
/// `[dist_to_centroid f32][inv_factor f32][sign bits ceil(dim/8)]` (fixed
/// stride, null rows zeroed). Returns the stripe bytes and the per-column
/// [`RaBitQColumn`] side data (centroid + rotation seed) for the
/// `VectorParamBlock` trailer.
pub(crate) fn encode_f32_vec_rabitq(
    vals: &[Option<&[f32]>],
    dim: u32,
    column_id: i32,
) -> (Vec<u8>, RaBitQColumn) {
    let dim_us = dim as usize;
    let refs: Vec<&[f32]> = vals.iter().filter_map(|o| *o).collect();
    // Deterministic per-column seed so decode reproduces the same rotation.
    let seed = 0x9E37_79B9_7F4A_7C15u64 ^ (column_id as u64);
    let params = functions::rabitq::fit_params(&refs, dim_us, seed);
    let rotation = functions::rabitq::build_rotation(dim_us, seed);
    let bits_len = dim_us.div_ceil(8);
    let stride = 8 + bits_len; // dist(4) + inv_factor(4) + bits

    let mut buf = vector_validity_bitmap(vals);
    buf.reserve(vals.len() * stride);
    for v in vals {
        match v {
            Some(vec) => {
                let code = functions::rabitq::encode(vec, &params, &rotation);
                buf.extend_from_slice(&code.dist_to_centroid.to_le_bytes());
                buf.extend_from_slice(&code.inv_factor.to_le_bytes());
                buf.extend_from_slice(&code.bits);
            }
            None => buf.extend(std::iter::repeat_n(0u8, stride)),
        }
    }
    (
        buf,
        RaBitQColumn {
            column_id,
            seed,
            centroid: params.centroid,
        },
    )
}

/// Fixed dimensionality of a vector column = the length of its first non-null
/// row. Returns 0 when every row is null. Errors if two non-null rows disagree
/// (embedding columns are fixed-width by construction).
fn vector_column_dim(vals: &[Option<&[f32]>]) -> Result<u32> {
    let mut dim: Option<usize> = None;
    for v in vals.iter().flatten() {
        match dim {
            None => dim = Some(v.len()),
            Some(d) if d != v.len() => {
                anyhow::bail!("inconsistent vector dimension: {d} vs {}", v.len())
            }
            _ => {}
        }
    }
    Ok(dim.unwrap_or(0) as u32)
}

/// Validity bitmap prefix: bit `i` set ⇒ row `i` is present (non-null).
fn vector_validity_bitmap(vals: &[Option<&[f32]>]) -> Vec<u8> {
    let mut bm = vec![0u8; vals.len().div_ceil(8)];
    for (i, v) in vals.iter().enumerate() {
        if v.is_some() {
            bm[i / 8] |= 1u8 << (i % 8);
        }
    }
    bm
}

/// Encode the two logical parts of an SQ8 vector stripe. Null rows occupy
/// `dim` zero bytes so the decoded code matrix remains fixed-stride.
fn encode_f32_vec_sq8_parts(
    vals: &[Option<&[f32]>],
    dim: u32,
    params: &functions::Sq8Params,
) -> (Vec<u8>, Vec<u8>) {
    let dim = dim as usize;
    let bitmap = vector_validity_bitmap(vals);
    let mut codes = Vec::with_capacity(vals.len() * dim);
    for v in vals {
        match v {
            Some(floats) => {
                for &f in *floats {
                    codes.push(functions::sq8::quantize_one(f, params));
                }
            }
            None => codes.extend(std::iter::repeat_n(0u8, dim)),
        }
    }
    (bitmap, codes)
}

fn join_vector_payload(mut bitmap: Vec<u8>, payload: Vec<u8>) -> Vec<u8> {
    bitmap.reserve(payload.len());
    bitmap.extend_from_slice(&payload);
    bitmap
}

fn cluster_runs(
    row_count: usize,
    explicit_starts: &[usize],
) -> Result<Vec<functions::clustered_for_bitpack::ClusterRun>> {
    if row_count == 0 {
        return Ok(Vec::new());
    }
    let mut starts = Vec::with_capacity(explicit_starts.len() + 1);
    starts.push(0usize);
    for &start in explicit_starts {
        if start == 0 || start >= row_count {
            anyhow::bail!("cluster run start {start} outside row count {row_count}");
        }
        if starts
            .last()
            .copied()
            .is_some_and(|previous| start <= previous)
        {
            anyhow::bail!("cluster run starts must be strictly increasing");
        }
        starts.push(start);
    }
    let mut runs = Vec::with_capacity(starts.len());
    for (index, &start) in starts.iter().enumerate() {
        let end = starts.get(index + 1).copied().unwrap_or(row_count);
        runs.push(functions::clustered_for_bitpack::ClusterRun::new(
            start,
            end - start,
        ));
    }
    Ok(runs)
}

/// Encode a raw fixed-stride f32 vector stripe: validity bitmap + `n_rows *
/// dim * 4` little-endian f32 bytes. Null rows occupy a zeroed row slot.
pub(crate) fn encode_f32_vec_raw_v2(vals: &[Option<&[f32]>], dim: u32) -> Vec<u8> {
    let dim = dim as usize;
    let mut buf = vector_validity_bitmap(vals);
    buf.reserve(vals.len() * dim * 4);
    for v in vals {
        match v {
            Some(floats) => {
                for &f in *floats {
                    buf.extend_from_slice(&f.to_le_bytes());
                }
            }
            None => buf.extend(std::iter::repeat_n(0u8, dim * 4)),
        }
    }
    buf
}

fn encode_i64_with_scheme(values: &[i64], scheme: &ProximaScheme) -> Result<Vec<u8>> {
    match scheme {
        ProximaScheme::Raw => functions::raw::encode_i64(values),
        ProximaScheme::Delta { base } => functions::delta::encode_i64(values, *base),
        ProximaScheme::BitPacked { bits } => functions::bitpack::encode_i64(values, *bits),
        ProximaScheme::FrameOfReference { reference, .. } => {
            functions::frame_of_ref::encode_i64(values, *reference)
        }
        ProximaScheme::PForDelta { base, .. } => functions::pfor_delta::encode_i64(values, *base),
        ProximaScheme::Zigzag { bits } => functions::zigzag::encode_i64(values, *bits),
        ProximaScheme::Simple8b => functions::simple8b::encode_i64(values),
        ProximaScheme::VByte => functions::vbyte::encode_i64(values),
        ProximaScheme::DoubleDelta { .. } => functions::double_delta::encode_i64(values),
        ProximaScheme::PForDoubleDelta { base, .. } => {
            functions::pfor_double_delta::encode_i64(values, *base)
        }
        ProximaScheme::Gorilla => functions::gorilla::encode_i64(values),
        ProximaScheme::SparseBitmap => functions::sparse_bitmap::encode_i64(values),
        ProximaScheme::SparseCOO => functions::sparse_coo::encode_i64(values),
        ProximaScheme::Dictionary => functions::dictionary::encode_i64(values),
        ProximaScheme::RunLength => functions::run_length::encode_i64(values),
        ProximaScheme::Sq8 | ProximaScheme::RaBitQ => {
            anyhow::bail!("quantized vector scheme not valid for i64 columns")
        }
        ProximaScheme::Adaptive => functions::adaptive::encode_i64(values),
    }
}

fn encode_f64_with_scheme(values: &[f64], scheme: &ProximaScheme) -> Result<Vec<u8>> {
    match scheme {
        ProximaScheme::Raw => functions::raw::encode_f64(values),
        ProximaScheme::Gorilla => functions::gorilla::encode_f64(values),
        other => anyhow::bail!("unsupported PAX f64 scheme: {}", other.name()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_records::ProximaRecord;

    fn make_record(oid: &str, tenant: &str, ts: i64) -> ProximaRecord {
        ProximaRecord {
            oid: oid.into(),
            tenant_id: tenant.into(),
            created_at_ns: ts,
            updated_at_ns: ts,
            ..Default::default()
        }
    }

    /// Regression: `select_f64_scheme` used to return `RunLength` for run-heavy /
    /// low-cardinality f64 columns, which `encode_f64_with_scheme` cannot encode
    /// (flush bailed "unsupported PAX f64 scheme: RunLength"). Whatever the
    /// strategy registry favors, the chosen scheme MUST be encodable.
    #[test]
    fn select_f64_scheme_only_returns_encodable_schemes() {
        let cases: Vec<Vec<Option<f64>>> = vec![
            vec![Some(1.0); 64],                               // constant run
            (0..64).map(|i| Some((i % 4) as f64)).collect(),   // low cardinality / runs
            (0..64).map(|i| Some(i as f64 * 0.5)).collect(),   // monotone
            vec![Some(0.5), None, Some(0.5), None, Some(0.5)], // nulls + repeats
        ];
        // Edge → General domain; UserDefined → TimeSeries domain (the path that
        // previously selected RunLength for an f64 user column).
        for vals in &cases {
            for role in [ColumnRole::Edge, ColumnRole::UserDefined] {
                let scheme = select_f64_scheme(role, true, vals);
                let raw: Vec<f64> = vals.iter().flatten().copied().collect();
                assert!(
                    encode_f64_with_scheme(&raw, &scheme).is_ok(),
                    "select_f64_scheme({role:?}) returned non-encodable {scheme:?}"
                );
            }
        }
    }

    #[test]
    fn pax_write_flush_roundtrip_header() {
        let mut writer = PaxBlockWriter::new(
            BlockMode::Pax,
            BlockCompression::None,
            "test_collection",
            0x1234,
            0,
        );
        writer
            .add_record(&make_record("r1", "tenant_a", 1000))
            .unwrap();
        writer
            .add_record(&make_record("r2", "tenant_a", 2000))
            .unwrap();

        let block = writer.flush().unwrap();
        assert!(block.len() > HEADER_SIZE);

        let header = BlockHeader::from_bytes(&block).unwrap();
        assert_eq!(header.row_count, 2);
        assert_eq!(header.block_mode as u8, BlockMode::Pax as u8);
        assert_eq!(header.min_timestamp_ns, 1000);
        assert_eq!(header.max_timestamp_ns, 2000);
        assert!(header.tenant_matches(fnv1a_hash("tenant_a")));
        assert!(!header.tenant_matches(fnv1a_hash("tenant_b")));
    }

    #[test]
    fn block_footer_carries_vparam_and_rgdir_offsets() {
        // v2 BlockFooter reclaims the old 12 reserved bytes for the
        // VectorParamBlock + RowGroupBlock pointers; they must round-trip.
        let f = BlockFooter {
            col_footer_offset: 1024,
            row_dir_offset: 64,
            stripe_start: 128,
            n_columns: 5,
            n_rows: 42,
            vparam_offset: 2048,
            vparam_len: 96,
            rgdir_offset: 2144,
        };
        let f2 = BlockFooter::from_bytes(&f.to_bytes()).unwrap();
        assert_eq!(f2.vparam_offset, 2048);
        assert_eq!(f2.vparam_len, 96);
        assert_eq!(f2.rgdir_offset, 2144);
        assert_eq!(f2.col_footer_offset, 1024);
        assert_eq!(f2.n_rows, 42);
    }

    #[test]
    fn olap_write_no_row_directory() {
        let mut writer = PaxBlockWriter::new(BlockMode::Olap, BlockCompression::None, "col", 0, 0);
        writer.add_record(&make_record("r1", "t", 1)).unwrap();
        let block = writer.flush().unwrap();
        let header = BlockHeader::from_bytes(&block).unwrap();
        // OLAP has no MVCC or DIR_SORTED flags
        assert_eq!(header.flags & flags::HAS_MVCC, 0);
    }

    // ---- P-Shred (ADR-055): hybrid props shredding ----

    fn record_with_props(oid: &str, ts: i64) -> ProximaRecord {
        use proximadb_data_model::ProximaValue as PV;
        use proximadb_records::ProximaTreeNode as Node;
        let mut rec = make_record(oid, "tenant-A", ts);
        rec.props
            .insert("status".into(), Node::Value(PV::String("active".into())));
        rec.props.insert("age".into(), Node::Value(PV::Int64(42)));
        rec.props
            .insert("score".into(), Node::Value(PV::Float64(0.75)));
        rec.props.insert(
            "note".into(),
            Node::Value(PV::String("keep me only in the tail".into())),
        );
        rec
    }

    /// (a) Losslessness ratchet: shredding declared keys must NOT lose or mutate any prop —
    /// the reader rebuilds ProximaRecord from the msgpack tail (ignoring user columns), so every
    /// prop (shredded or not) must come back identical. This is the load-bearing clone-not-remove
    /// guarantee.
    #[test]
    fn shredding_preserves_the_full_props_tail() {
        use crate::reader::PaxBlockReader;
        let rec = record_with_props("d1", 100);
        let original_props = rec.props.clone();

        let spec = vec![
            ("status".to_string(), col_id::USER_BASE),
            ("age".to_string(), col_id::USER_BASE + 1),
        ];
        let mut w = PaxBlockWriter::new(BlockMode::Pax, BlockCompression::None, "c", 0, 0)
            .with_shred_spec(spec);
        w.add_record(&rec).unwrap();
        let block = w.flush().unwrap();

        let reader = PaxBlockReader::open(&block).unwrap();
        // Reconstruct with NO user_column_keys — every prop must be served by the tail alone.
        let flat = FlatRow::from_block_reader(&reader).unwrap().remove(0);
        let record = flat.into_record(&[], &[], None).unwrap();
        assert_eq!(
            record.props, original_props,
            "shredding must preserve the full props tail byte-for-byte (clone-not-remove)"
        );
    }

    /// TD-USUB-6 slice 2a: the block names its OWN shredded columns.
    ///
    /// Without this, `FlatRow::into_record` names user columns from the CALLER's
    /// `user_column_keys`, positionally — so a short, stale or reordered caller
    /// list silently drops values. That is harmless while the msgpack tail is
    /// complete, and becomes silent data loss the moment the tail is reduced to a
    /// residual (slice 2b). This test proves the naming can come from the block
    /// instead, which is the precondition for reducing the tail at all.
    #[test]
    fn block_carries_its_own_shred_directory() {
        use crate::reader::PaxBlockReader;
        let rec = record_with_props("d1", 100);

        let spec = vec![
            ("status".to_string(), col_id::USER_BASE),
            ("age".to_string(), col_id::USER_BASE + 1),
        ];
        let mut w = PaxBlockWriter::new(BlockMode::Pax, BlockCompression::None, "c", 0, 0)
            .with_shred_spec(spec.clone());
        w.add_record(&rec).unwrap();
        let block = w.flush().unwrap();

        let reader = PaxBlockReader::open(&block).unwrap();
        let dir = reader
            .shred_directory()
            .expect("a block written with a shred spec must carry its directory");

        // The directory is (col_id, prop_key) and must mirror the spec exactly —
        // including the ids, since position is precisely what must stop mattering.
        let expected: Vec<(i32, String)> = spec.iter().map(|(k, c)| (*c, k.clone())).collect();
        let actual: Vec<(i32, String)> = dir.iter().map(|e| (e.column_id, e.key.clone())).collect();
        assert_eq!(actual, expected);

        // Naming is now answerable WITHOUT any caller-supplied key list — the
        // property that makes a residual tail safe.
        let name_of = |col: i32| {
            dir.iter()
                .find(|e| e.column_id == col)
                .map(|e| e.key.as_str())
        };
        assert_eq!(name_of(col_id::USER_BASE), Some("status"));
        assert_eq!(name_of(col_id::USER_BASE + 1), Some("age"));
    }

    /// TD-USUB-6: the directory records the EXACT declared type when the caller
    /// has one, and records its ABSENCE when the caller does not.
    ///
    /// That distinction is the whole point. A typed stripe stores one of three
    /// physical classes (i64/f64/string), so `Int32` and `Int64` land in the same
    /// stripe and are indistinguishable on read. Only a column with an exact
    /// declared type can be reconstructed back to its original variant — and only
    /// such a column can ever become authoritative. Columns without one must keep
    /// a complete `PROPS` tail.
    #[test]
    fn directory_records_the_declared_type_when_the_caller_has_one() {
        use crate::reader::PaxBlockReader;
        use proximadb_data_model::ProximaType;

        let rec = record_with_props("d1", 100);
        let mut w = PaxBlockWriter::new(BlockMode::Pax, BlockCompression::None, "c", 0, 0)
            .with_shred_columns(vec![
                ShredColumn::typed("age", col_id::USER_BASE, ProximaType::Int32),
                ShredColumn::untyped("status", col_id::USER_BASE + 1),
            ]);
        w.add_record(&rec).unwrap();
        let block = w.flush().unwrap();

        let reader = PaxBlockReader::open(&block).unwrap();
        let dir = reader.shred_directory().expect("directory present");
        assert_eq!(dir.len(), 2);

        let age = dir.iter().find(|e| e.key == "age").expect("age entry");
        assert_eq!(age.column_id, col_id::USER_BASE);
        assert_eq!(
            age.declared_type,
            Some(ProximaType::Int32),
            "an exact declared type must round-trip — Int32 must NOT come back as Int64"
        );

        let status = dir
            .iter()
            .find(|e| e.key == "status")
            .expect("status entry");
        assert_eq!(
            status.declared_type, None,
            "absence must be recorded, not guessed: this stripe can never be authoritative"
        );
    }

    /// The `with_shred_spec` shim (the SST flush path, whose only declared type is
    /// the coarse v1 `FilterableDataType`) yields untyped columns — so that path
    /// can never be mistaken for one that declares exactly.
    #[test]
    fn untyped_shim_records_no_declared_type() {
        use crate::reader::PaxBlockReader;
        let rec = record_with_props("d1", 100);
        let mut w = PaxBlockWriter::new(BlockMode::Pax, BlockCompression::None, "c", 0, 0)
            .with_shred_spec(vec![("status".to_string(), col_id::USER_BASE)]);
        w.add_record(&rec).unwrap();
        let block = w.flush().unwrap();

        let reader = PaxBlockReader::open(&block).unwrap();
        let dir = reader.shred_directory().expect("directory present");
        assert_eq!(dir.len(), 1);
        assert_eq!(dir[0].key, "status");
        assert_eq!(dir[0].declared_type, None);
    }

    /// **The losslessness ratchet (TD-USUB-6 2b).** With the residual tail ON, a
    /// record must round-trip EXACTLY — same variants, same payloads — even
    /// though the shredded values no longer live in the msgpack tail.
    ///
    /// Covers the three cases that matter:
    /// * an exactly-typed value that IS dropped from the tail and rebuilt from
    ///   its stripe (`Int32` must come back `Int32`, not `Int64`);
    /// * a value whose variant does NOT match its declared type, which must stay
    ///   in the tail rather than be silently coerced;
    /// * an untyped column, which can never be authoritative and keeps its tail.
    #[test]
    fn residual_tail_round_trips_every_prop_exactly() {
        use crate::reader::PaxBlockReader;
        use proximadb_data_model::{ProximaType as PT, ProximaValue as PV};
        use proximadb_records::ProximaTreeNode as Node;

        let mut rec = ProximaRecord {
            oid: "r1".into(),
            ..Default::default()
        };
        // Exactly typed → droppable from the tail.
        rec.props.insert("age".into(), Node::Value(PV::Int32(41)));
        rec.props
            .insert("score".into(), Node::Value(PV::Float32(0.5)));
        rec.props
            .insert("name".into(), Node::Value(PV::String("ada".into())));
        // Declared Int32 but holds a String → NOT losslessly shreddable.
        rec.props
            .insert("mixed".into(), Node::Value(PV::String("not-an-int".into())));
        // Untyped column → never authoritative.
        rec.props
            .insert("tag".into(), Node::Value(PV::String("x".into())));
        let original = rec.props.clone();

        let mut w = PaxBlockWriter::new(BlockMode::Pax, BlockCompression::None, "c", 0, 0)
            .with_shred_columns(vec![
                ShredColumn::typed("age", col_id::USER_BASE, PT::Int32),
                ShredColumn::typed("score", col_id::USER_BASE + 1, PT::Float32),
                ShredColumn::typed("name", col_id::USER_BASE + 2, PT::String),
                ShredColumn::typed("mixed", col_id::USER_BASE + 3, PT::Int32),
                ShredColumn::untyped("tag", col_id::USER_BASE + 4),
            ]);
        w.residual_tail = true; // gate forced on for the test
        w.add_record(&rec).unwrap();
        let block = w.flush().unwrap();

        let reader = PaxBlockReader::open(&block).unwrap();
        assert!(
            reader.header().is_v3(),
            "a block whose tail became a residual must declare v3"
        );

        let flat = FlatRow::from_block_reader(&reader).unwrap().remove(0);
        let back = flat.into_record(&[], &[], None).unwrap();
        assert_eq!(
            back.props, original,
            "every prop must round-trip exactly, including the variant"
        );
        // Spell out the one that motivated the whole design.
        assert!(
            matches!(back.props.get("age"), Some(Node::Value(PV::Int32(41)))),
            "Int32 must not come back as Int64"
        );
    }

    /// The residual actually SHRINKS the tail — the write-amplification win this
    /// TD is justified on. Same record, gate off vs on.
    #[test]
    fn residual_tail_is_smaller_than_the_complete_tail() {
        use proximadb_data_model::{ProximaType as PT, ProximaValue as PV};
        use proximadb_records::ProximaTreeNode as Node;

        let build = |residual: bool| {
            let mut rec = ProximaRecord {
                oid: "r1".into(),
                ..Default::default()
            };
            rec.props
                .insert("name".into(), Node::Value(PV::String("a".repeat(256))));
            let mut w = PaxBlockWriter::new(BlockMode::Pax, BlockCompression::None, "c", 0, 0)
                .with_shred_columns(vec![ShredColumn::typed(
                    "name",
                    col_id::USER_BASE,
                    PT::String,
                )]);
            w.residual_tail = residual;
            w.add_record(&rec).unwrap();
            w.flush().unwrap().len()
        };

        let complete = build(false);
        let residual = build(true);
        assert!(
            residual < complete,
            "residual tail must be smaller: {residual} !< {complete}"
        );
    }

    /// Gate OFF (the shipped default) keeps the tail complete and the block v2 —
    /// enabling nothing changes nothing.
    #[test]
    fn residual_gate_off_keeps_a_complete_v2_tail() {
        use crate::reader::PaxBlockReader;
        use proximadb_data_model::{ProximaType as PT, ProximaValue as PV};
        use proximadb_records::ProximaTreeNode as Node;

        let mut rec = ProximaRecord {
            oid: "r1".into(),
            ..Default::default()
        };
        rec.props.insert("age".into(), Node::Value(PV::Int32(7)));
        let original = rec.props.clone();

        let mut w = PaxBlockWriter::new(BlockMode::Pax, BlockCompression::None, "c", 0, 0)
            .with_shred_columns(vec![ShredColumn::typed(
                "age",
                col_id::USER_BASE,
                PT::Int32,
            )]);
        w.residual_tail = false;
        w.add_record(&rec).unwrap();
        let block = w.flush().unwrap();

        let reader = PaxBlockReader::open(&block).unwrap();
        assert_eq!(
            reader.header().format_version,
            crate::FORMAT_VERSION,
            "gate off ⇒ still v2"
        );
        let flat = FlatRow::from_block_reader(&reader).unwrap().remove(0);
        assert_eq!(flat.into_record(&[], &[], None).unwrap().props, original);
    }

    /// A block written with NO shred spec carries no directory, and asking for one
    /// is a clean `None` rather than an error — every v2 block on disk today is
    /// this shape, so the caller-key fallback must stay reachable.
    #[test]
    fn no_shred_spec_means_no_directory_and_no_error() {
        use crate::reader::PaxBlockReader;
        let rec = record_with_props("d1", 100);
        let mut w = PaxBlockWriter::new(BlockMode::Pax, BlockCompression::None, "c", 0, 0);
        w.add_record(&rec).unwrap();
        let block = w.flush().unwrap();

        let reader = PaxBlockReader::open(&block).unwrap();
        assert!(reader.shred_directory().is_none());
    }

    /// The directory must not disturb what was already there: the full props tail
    /// still round-trips, and the shredded stripes still decode. It is additive in
    /// v2 — no version bump, no footer growth, no new `ColumnRole` byte.
    #[test]
    fn shred_directory_is_additive_and_disturbs_nothing() {
        use crate::reader::PaxBlockReader;
        let rec = record_with_props("d1", 100);
        let original_props = rec.props.clone();

        let spec = vec![("status".to_string(), col_id::USER_BASE)];
        let mut w = PaxBlockWriter::new(BlockMode::Pax, BlockCompression::None, "c", 0, 0)
            .with_shred_spec(spec);
        w.add_record(&rec).unwrap();
        let block = w.flush().unwrap();

        let reader = PaxBlockReader::open(&block).unwrap();
        assert_eq!(
            reader.header().format_version,
            crate::FORMAT_VERSION,
            "the directory must NOT require a format bump"
        );
        assert!(reader.shred_directory().is_some());

        // The tail is still complete and byte-identical.
        let flat = FlatRow::from_block_reader(&reader).unwrap().remove(0);
        let record = flat.into_record(&[], &[], None).unwrap();
        assert_eq!(record.props, original_props);

        // The directory entry itself claims no bloom and no stripe.
        let meta = reader
            .column_metas()
            .iter()
            .find(|m| m.column_id == col_id::SHRED_DIRECTORY)
            .expect("directory meta present");
        assert_eq!(
            meta.stripe_len, 0,
            "the directory carries no per-row stripe"
        );
        assert!(
            !meta.has_bloom,
            "it borrows the pointer pair, it is not a bloom"
        );
    }

    /// (b) The shredded columns are present at their assigned ids and their decoded values equal
    /// the tail values (the pruning/projection index is correct).
    #[test]
    fn shredded_columns_present_and_equal_tail() {
        use crate::reader::PaxBlockReader;
        let rec = record_with_props("d1", 100);
        let spec = vec![
            ("status".to_string(), col_id::USER_BASE),
            ("age".to_string(), col_id::USER_BASE + 1),
        ];
        let mut w = PaxBlockWriter::new(BlockMode::Pax, BlockCompression::None, "c", 0, 0)
            .with_shred_spec(spec);
        w.add_record(&rec).unwrap();
        let block = w.flush().unwrap();
        let reader = PaxBlockReader::open(&block).unwrap();

        assert!(
            reader
                .column_metas()
                .iter()
                .any(|m| m.column_id == col_id::USER_BASE),
            "string shred column present at USER_BASE"
        );
        assert!(
            reader
                .column_metas()
                .iter()
                .any(|m| m.column_id == col_id::USER_BASE + 1),
            "i64 shred column present at USER_BASE+1"
        );
        assert_eq!(
            reader.decode_str_stripe(col_id::USER_BASE),
            Some(vec![Some("active".to_string())]),
            "shredded string column equals the tail value"
        );
        assert_eq!(
            reader.decode_i64_stripe(col_id::USER_BASE + 1),
            Some(vec![Some(42)]),
            "shredded i64 column equals the tail value"
        );
    }

    /// (c) An empty shred spec is byte-for-byte today's behavior: no user columns, props intact.
    #[test]
    fn empty_shred_spec_is_unchanged() {
        use crate::reader::PaxBlockReader;
        let rec = record_with_props("d1", 100);
        let original_props = rec.props.clone();

        let mut w = PaxBlockWriter::new(BlockMode::Pax, BlockCompression::None, "c", 0, 0)
            .with_shred_spec(vec![]);
        w.add_record(&rec).unwrap();
        let block = w.flush().unwrap();
        let reader = PaxBlockReader::open(&block).unwrap();

        assert!(
            !reader
                .column_metas()
                .iter()
                .any(|m| m.column_id >= col_id::USER_BASE && m.column_id < col_id::RERANK_BASE),
            "empty shred spec must emit no user columns"
        );
        let flat = FlatRow::from_block_reader(&reader).unwrap().remove(0);
        let record = flat.into_record(&[], &[], None).unwrap();
        assert_eq!(record.props, original_props);
    }
}
