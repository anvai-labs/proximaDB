// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! PAX segment layer: file-backed multi-block storage with predicate pruning.
//!
//! A **segment file** (`.pax`) contains one or more PAX blocks written
//! sequentially. Each block is self-describing (header + footer) so readers
//! can scan block-by-block and apply block-level predicate pruning without
//! reading the full file.
//!
//! ## File layout
//!
//! ```text
//! [Block_0: N_0 bytes]
//! [Block_1: N_1 bytes]
//! ...
//! [Block_k: N_k bytes]
//! [SegmentIndex: block_count u32 + (offset u64 + size u32) × block_count + crc32 u32]
//! [SegmentMagic: 8B "PAXSEG01"]
//! ```
//!
//! The index at the tail allows random-access block lookup without scanning
//! the whole file. Writers append the index after all blocks are flushed.
//!
//! ## Scan predicate pushdown
//!
//! `PaxSegmentScanner` applies three levels of skipping before decoding:
//! 1. Tenant hash mismatch → skip block entirely
//! 2. Time range outside block min/max → skip block
//! 3. Column stats exclude predicate value → skip stripe within block
//!
//! ## Iceberg manifest integration
//!
//! `PaxSegmentWriter::finish()` returns `SegmentMeta` containing per-block
//! `BlockStats`. These map directly to Iceberg `DataFile` entries in the
//! `iceberg_rest_service.rs` manifest generator.

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use proximadb_block_format::coalesced_rabitq::{
    RABITQ_SEED_BASE, encode_region, region_header_len as rabitq_region_header_len,
};
use proximadb_block_format::coalesced_sq8::{
    codes_offset as sq8_codes_offset, encode_region as encode_sq8_region,
};
use proximadb_block_format::{
    BlockCompression, BlockMode, BlockStats, BlockZoneSource, ColumnMeta, FlatRow, PaxBlockReader,
    PaxBlockWriter, RowGroupBlock, VectorQuant, col_id, header::fnv1a_hash,
};
use proximadb_records::ProximaRecord;
use serde::{Deserialize, Serialize};

use crate::coarse_directory::{CoarseCellEntry, CoarseDirectory, CoarseModel};
use crate::engine_constants::{
    DEFAULT_BLOCK_METADATA_OVERHEAD_BYTES, DEFAULT_TARGET_BLOCK_SIZE_BYTES, MAX_BLOCK_ROWS,
    MAX_TARGET_BLOCK_SIZE_BYTES,
};
use crate::segment_layout::{
    BlockTierAssignment, EXTERNAL_CANONICAL_SOURCE_ID, FooterBlockEntry, FooterRowGroupStats,
    LosslessCompressionTag, LosslessTransformTag, ParameterScope, SEG_HEADER_PREFIX_LEN,
    SEG_LAYOUT_VERSION, SegmentFooterIndex, SegmentHeaderPrefix, SourceFidelity, SourceRole,
    StatsKind, StripeEncodingDescriptor, TierRole, VectorTransform, compression_flags,
    is_coalesced_segment,
};
use crate::spill_regions::DiskVectorSpool;

/// File extension for PAX segment files.
pub const PAX_SEGMENT_EXT: &str = ".pax";

/// TD-PAXRG-1: floor for the v4 row-group Region D block target. Below this,
/// per-RG framing (64 B header + ~1 KB ColumnMeta frame + 32 B footer) stops
/// amortizing — measured: 400 B targets produced ~400 micro-RGs whose pruned
/// fetch totalled ~1.25 MB for a 38 KB-Parquet-equivalent dataset. 256 KiB
/// keeps fixed cost ≤ ~1% of payload.
pub const RG_TARGET_MIN_BYTES: usize = 256 * 1024;

/// Magic bytes at the tail of a segment file (after the index).
pub const SEGMENT_MAGIC: &[u8; 8] = b"PAXSEG01";

// ── Segment index ─────────────────────────────────────────────────────────────

/// Trailer marker for a **v2** segment index carrying per-block zone-map
/// summaries (TD-167 follow-up / ADR-034 P1). v1 indexes lack it, so the reader
/// detects the format by whether the bytes before [`SEGMENT_MAGIC`] end with this
/// marker — keeping v1 byte-identical (old readers/segments unaffected).
pub const ZONE_INDEX_MARKER: &[u8; 4] = b"PAXZ";

// Bit positions in [`BlockZoneSummary::present`] — only the numeric canonical
// columns predicate pushdown prunes on without decoding rows.
pub(crate) const ZONE_CREATED_AT: u8 = 1 << 0;
pub(crate) const ZONE_UPDATED_AT: u8 = 1 << 1;
pub(crate) const ZONE_VALID_FROM: u8 = 1 << 2;
pub(crate) const ZONE_VALID_TO: u8 = 1 << 3;
pub(crate) const ZONE_EDGE_WEIGHT: u8 = 1 << 4;

/// Serialized fixed width of a [`BlockZoneSummary`]: row_count(4) + present(1) +
/// 4×(i64,i64)=64 + (f64,f64)=16.
pub(crate) const ZONE_SUMMARY_BYTES: usize = 4 + 1 + 64 + 16;
/// v2 per-block index entry width: offset(8) + size(4) + zone summary.
const V2_ENTRY_BYTES: usize = 8 + 4 + ZONE_SUMMARY_BYTES;

/// Fixed-width per-block zone-map summary embedded in a v2 segment index, so the
/// reader can prune a block from the *cached index* with **zero** per-block
/// footer/metadata GETs (cold-path depth 4→2). min/max are copied verbatim from
/// the block's `ColumnMeta` (the same bounds the full-layout prune uses), so v2
/// pruning is byte-for-byte identical to v1 for these predicates; columns absent
/// from the summary fall through to "may match" (conservative — never skips a
/// block wrongly).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlockZoneSummary {
    pub row_count: u32,
    /// Bitmask of which canonical columns carry valid bounds.
    pub present: u8,
    pub created_at: (i64, i64),
    pub updated_at: (i64, i64),
    pub valid_from: (i64, i64),
    pub valid_to: (i64, i64),
    pub edge_weight: (f64, f64),
}

impl BlockZoneSummary {
    /// An empty summary (no bounds) — used for blocks whose zone is unknown when
    /// a v2 index is written.
    pub fn empty(row_count: u32) -> Self {
        Self {
            row_count,
            present: 0,
            created_at: (0, 0),
            updated_at: (0, 0),
            valid_from: (0, 0),
            valid_to: (0, 0),
            edge_weight: (0.0, 0.0),
        }
    }

    /// Extract the canonical-column bounds from a flushed block's column metadata.
    /// A column with `distinct_hint == 0` (no usable bounds) is left out of
    /// `present`.
    pub fn from_column_metas(row_count: u32, metas: &[ColumnMeta]) -> Self {
        let mut s = Self::empty(row_count);
        for m in metas {
            if m.distinct_hint == 0 {
                continue;
            }
            let i64_lo = i64::from_le_bytes(m.min_val[0..8].try_into().unwrap_or([0; 8]));
            let i64_hi = i64::from_le_bytes(m.max_val[0..8].try_into().unwrap_or([0; 8]));
            match m.column_id {
                col_id::CREATED_AT => {
                    s.created_at = (i64_lo, i64_hi);
                    s.present |= ZONE_CREATED_AT;
                }
                col_id::UPDATED_AT => {
                    s.updated_at = (i64_lo, i64_hi);
                    s.present |= ZONE_UPDATED_AT;
                }
                col_id::VALID_FROM => {
                    s.valid_from = (i64_lo, i64_hi);
                    s.present |= ZONE_VALID_FROM;
                }
                col_id::VALID_TO => {
                    s.valid_to = (i64_lo, i64_hi);
                    s.present |= ZONE_VALID_TO;
                }
                col_id::EDGE_WEIGHT => {
                    s.edge_weight = (
                        f64::from_le_bytes(m.min_val[0..8].try_into().unwrap_or([0; 8])),
                        f64::from_le_bytes(m.max_val[0..8].try_into().unwrap_or([0; 8])),
                    );
                    s.present |= ZONE_EDGE_WEIGHT;
                }
                _ => {}
            }
        }
        s
    }

    pub(crate) fn write_to(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.row_count.to_le_bytes());
        buf.push(self.present);
        for (lo, hi) in [
            self.created_at,
            self.updated_at,
            self.valid_from,
            self.valid_to,
        ] {
            buf.extend_from_slice(&lo.to_le_bytes());
            buf.extend_from_slice(&hi.to_le_bytes());
        }
        buf.extend_from_slice(&self.edge_weight.0.to_le_bytes());
        buf.extend_from_slice(&self.edge_weight.1.to_le_bytes());
    }

    pub(crate) fn read_from(data: &[u8]) -> Result<Self> {
        if data.len() < ZONE_SUMMARY_BYTES {
            bail!("zone summary truncated");
        }
        let row_count = u32::from_le_bytes(data[0..4].try_into()?);
        let present = data[4];
        let rd = |off: usize| -> Result<(i64, i64)> {
            Ok((
                i64::from_le_bytes(data[off..off + 8].try_into()?),
                i64::from_le_bytes(data[off + 8..off + 16].try_into()?),
            ))
        };
        let created_at = rd(5)?;
        let updated_at = rd(21)?;
        let valid_from = rd(37)?;
        let valid_to = rd(53)?;
        let edge_weight = (
            f64::from_le_bytes(data[69..77].try_into()?),
            f64::from_le_bytes(data[77..85].try_into()?),
        );
        Ok(Self {
            row_count,
            present,
            created_at,
            updated_at,
            valid_from,
            valid_to,
            edge_weight,
        })
    }

    fn i64_bounds(&self, column_id: i32) -> Option<(i64, i64)> {
        match column_id {
            col_id::CREATED_AT if self.present & ZONE_CREATED_AT != 0 => Some(self.created_at),
            col_id::UPDATED_AT if self.present & ZONE_UPDATED_AT != 0 => Some(self.updated_at),
            col_id::VALID_FROM if self.present & ZONE_VALID_FROM != 0 => Some(self.valid_from),
            col_id::VALID_TO if self.present & ZONE_VALID_TO != 0 => Some(self.valid_to),
            _ => None,
        }
    }

    fn f64_bounds(&self, column_id: i32) -> Option<(f64, f64)> {
        match column_id {
            col_id::EDGE_WEIGHT if self.present & ZONE_EDGE_WEIGHT != 0 => Some(self.edge_weight),
            _ => None,
        }
    }
}

/// Shared empty row-group sub-index for summary-based pruning (the summary is
/// block-level only — no per-row-group bounds).
static EMPTY_ROW_GROUP: LazyLock<RowGroupBlock> = LazyLock::new(RowGroupBlock::default);

// block-format wire `data_type_id`s (private in prune.rs) — stable on-disk codec
// markers, mirrored here so the summary can advertise the right predicate path.
const DT_I64: u8 = 0x03;
const DT_F64: u8 = 0x07;

/// Prune a block from its compact zone-map summary alone — the reader path that
/// needs NO per-block footer/metadata GET. Soundness matches the full-layout
/// prune: bounds are copied verbatim from the block's `ColumnMeta`, and any column
/// not in the summary advertises `None` (→ `evaluate_leaf` returns `MayMatch`), so
/// a block is skipped only when provably empty for the summarized predicate.
impl BlockZoneSource for BlockZoneSummary {
    fn column_meta_type(&self, column_id: i32) -> Option<u8> {
        if self.i64_bounds(column_id).is_some() {
            Some(DT_I64)
        } else if self.f64_bounds(column_id).is_some() {
            Some(DT_F64)
        } else {
            None
        }
    }

    fn may_contain_i64(&self, column_id: i32, value: i64) -> bool {
        self.i64_bounds(column_id)
            .is_none_or(|(lo, hi)| value >= lo && value <= hi)
    }

    fn range_overlaps_i64(&self, column_id: i32, lo: i64, hi: i64) -> bool {
        self.i64_bounds(column_id)
            .is_none_or(|(min, max)| lo <= max && hi >= min)
    }

    fn range_overlaps_f64(&self, column_id: i32, lo: f64, hi: f64) -> bool {
        self.f64_bounds(column_id).is_none_or(|(min, max)| {
            if min.is_nan() || max.is_nan() {
                return true;
            }
            lo <= max && hi >= min
        })
    }

    fn may_contain_str(&self, _column_id: i32, _value: &str) -> bool {
        true // the summary carries no string bounds; unreachable (column_meta_type → None)
    }

    fn row_group_index(&self) -> &RowGroupBlock {
        &EMPTY_ROW_GROUP
    }

    fn row_count_hint(&self) -> u32 {
        self.row_count
    }
}

/// Per-block entry in the segment index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockIndexEntry {
    /// Byte offset from the start of the segment file.
    pub offset: u64,
    /// Block byte length (matches the block's header + body + footer).
    pub size: u32,
    /// v2 zone-map summary (`None` in a legacy v1 index — prune falls back to a
    /// per-block metadata read).
    #[serde(default)]
    pub zone: Option<BlockZoneSummary>,
    /// TD-FPRUNE-1 P2: serialized `FooterBlockStats` for this block's
    /// bounded columns, captured at flush (when the block's `ColumnMeta` is in
    /// hand) and lifted into the segment footer so a filtered query prunes the
    /// block without a body GET. `None` (default) when footer stats are gated OFF.
    #[serde(default)]
    pub footer_stats: Option<Vec<u8>>,
}

impl BlockIndexEntry {}

/// TD-FPRUNE-1 P2 (default-OFF): `PROXIMADB_PAX_FOOTER_STATS=1|on|true|yes`
/// makes the writer lift per-block bounded-column `ColumnMeta` into the segment
/// footer, so a filtered query can prune a block without a body GET. OFF keeps
/// the footer byte-identical to today (mixed-read-safe: the reader treats an
/// absent payload as "may match").
fn footer_stats_enabled() -> bool {
    matches!(
        std::env::var("PROXIMADB_PAX_FOOTER_STATS")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("1" | "on" | "true" | "yes")
    )
}

/// Index appended at the tail of a segment file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentIndex {
    pub blocks: Vec<BlockIndexEntry>,
}

impl SegmentIndex {
    /// Serialise to bytes. **Writes are always v2** (TD-167 / ADR-034 P1):
    /// `[n u32] [offset, size, zone(85B)] × n [crc32 u32] [body_len u32] [PAXZ]`.
    /// A block with no computable bounds gets an **empty** zone summary, so EVERY
    /// new segment prunes from the cached index with **zero per-block metadata
    /// GETs** — the legacy v1 layout (`[n][offset,size]×n[crc]`) forced a per-block
    /// footer read for pruning (the depth the co-design exists to collapse).
    ///
    /// The v1 layout is **deprecated for writes** and retained for READS only
    /// (`from_bytes` detects it via the absent `PAXZ` trailer), so existing v1
    /// segments stay readable — mixed-read-safe, no flag-day migration. Per-block
    /// empty-zone overhead (73 B/block) is read once and cached: a worthwhile trade
    /// for unconditional depth-collapse (`DEPTH ≫ BYTES`).
    pub fn to_bytes(&self) -> Vec<u8> {
        self.to_bytes_v2()
    }

    /// Legacy v1 segment-index encoding — **write-deprecated** (kept only to build
    /// v1 fixtures for read mixed-read-safety tests; production writes v2 via
    /// [`Self::to_bytes`]). Real v1 segments on disk are read by `from_bytes`.
    #[cfg(test)]
    fn to_bytes_v1(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(4 + self.blocks.len() * 12 + 4);
        buf.extend_from_slice(&(self.blocks.len() as u32).to_le_bytes());
        for e in &self.blocks {
            buf.extend_from_slice(&e.offset.to_le_bytes());
            buf.extend_from_slice(&e.size.to_le_bytes());
        }
        let crc = crc32fast::hash(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        buf
    }

    fn to_bytes_v2(&self) -> Vec<u8> {
        let mut body = Vec::with_capacity(4 + self.blocks.len() * V2_ENTRY_BYTES + 12);
        body.extend_from_slice(&(self.blocks.len() as u32).to_le_bytes());
        for e in &self.blocks {
            body.extend_from_slice(&e.offset.to_le_bytes());
            body.extend_from_slice(&e.size.to_le_bytes());
            e.zone
                .clone()
                .unwrap_or_else(|| BlockZoneSummary::empty(0))
                .write_to(&mut body);
        }
        let crc = crc32fast::hash(&body);
        body.extend_from_slice(&crc.to_le_bytes());
        // Self-describing trailer so the reader needs no brute-force for v2.
        let body_len = body.len() as u32;
        body.extend_from_slice(&body_len.to_le_bytes());
        body.extend_from_slice(ZONE_INDEX_MARKER);
        body
    }

    /// Deserialise a **v1** index from the last N bytes of a segment file.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            bail!("segment index too small");
        }
        let n = u32::from_le_bytes(data[0..4].try_into()?) as usize;
        let body_len = 4 + n * 12;
        if data.len() < body_len + 4 {
            bail!("segment index truncated: expected {} + 4 bytes", body_len);
        }
        let stored_crc = u32::from_le_bytes(data[body_len..body_len + 4].try_into()?);
        let computed = crc32fast::hash(&data[..body_len]);
        if stored_crc != computed {
            bail!("segment index CRC mismatch");
        }
        let mut blocks = Vec::with_capacity(n);
        for i in 0..n {
            let off = 4 + i * 12;
            let offset = u64::from_le_bytes(data[off..off + 8].try_into()?);
            let size = u32::from_le_bytes(data[off + 8..off + 12].try_into()?);
            blocks.push(BlockIndexEntry {
                offset,
                size,
                zone: None,
                footer_stats: None,
            });
        }
        Ok(Self { blocks })
    }

    /// Deserialise a **v2** index body (`[n][offset,size,zone]×n[crc]`, without the
    /// `[body_len][PAXZ]` trailer).
    fn from_bytes_v2(body: &[u8]) -> Result<Self> {
        if body.len() < 8 {
            bail!("v2 segment index too small");
        }
        let n = u32::from_le_bytes(body[0..4].try_into()?) as usize;
        let crc_pos = 4 + n * V2_ENTRY_BYTES;
        if body.len() < crc_pos + 4 {
            bail!("v2 segment index truncated");
        }
        let stored_crc = u32::from_le_bytes(body[crc_pos..crc_pos + 4].try_into()?);
        if crc32fast::hash(&body[..crc_pos]) != stored_crc {
            bail!("v2 segment index CRC mismatch");
        }
        let mut blocks = Vec::with_capacity(n);
        for i in 0..n {
            let off = 4 + i * V2_ENTRY_BYTES;
            let offset = u64::from_le_bytes(body[off..off + 8].try_into()?);
            let size = u32::from_le_bytes(body[off + 8..off + 12].try_into()?);
            let zone = BlockZoneSummary::read_from(&body[off + 12..off + 12 + ZONE_SUMMARY_BYTES])?;
            blocks.push(BlockIndexEntry {
                offset,
                size,
                zone: Some(zone),
                footer_stats: None,
            });
        }
        Ok(Self { blocks })
    }

    /// Locate and parse the index at the tail of `before_magic` (the segment bytes
    /// with the trailing [`SEGMENT_MAGIC`] removed). v2 is found directly via the
    /// self-describing `[body_len][PAXZ]` trailer; v1's length is not stored, so
    /// candidate counts are tried until the embedded count + CRC validate. Shared
    /// by the whole-file scanner and the ranged reader.
    pub fn locate(before_magic: &[u8]) -> Result<Self> {
        if before_magic.len() < 8 {
            bail!("no room for segment index");
        }
        // v2: trailing [body_len u32][PAXZ].
        if &before_magic[before_magic.len() - 4..] == ZONE_INDEX_MARKER {
            let blen_pos = before_magic.len() - 8;
            let body_len =
                u32::from_le_bytes(before_magic[blen_pos..blen_pos + 4].try_into()?) as usize;
            if before_magic.len() < 8 + body_len {
                bail!("v2 index marker present but body does not fit");
            }
            let body = &before_magic[before_magic.len() - 8 - body_len..before_magic.len() - 8];
            return Self::from_bytes_v2(body);
        }
        // v1 brute-force.
        for candidate_n in 0usize..=(before_magic.len().saturating_sub(8) / 12) {
            let index_len = 4 + candidate_n * 12 + 4;
            if index_len > before_magic.len() {
                break;
            }
            let idx_start = before_magic.len() - index_len;
            let n_in_data = u32::from_le_bytes(
                before_magic[idx_start..idx_start + 4]
                    .try_into()
                    .unwrap_or([0; 4]),
            ) as usize;
            if n_in_data == candidate_n
                && let Ok(idx) = SegmentIndex::from_bytes(&before_magic[idx_start..])
            {
                return Ok(idx);
            }
        }
        bail!("could not locate valid segment index");
    }

    /// Locate and parse the index from a file **suffix** that must contain the
    /// trailing [`SEGMENT_MAGIC`] and the full index. Returns `Ok(None)` when the
    /// suffix is too small to hold the whole index (the caller should re-read a
    /// larger suffix), and `Err` only on a corrupt/invalid tail.
    pub fn locate_in_suffix(suffix: &[u8]) -> Result<Option<Self>> {
        if suffix.len() < 8 || &suffix[suffix.len() - 8..] != SEGMENT_MAGIC {
            bail!("segment suffix missing magic (not a PAX segment tail)");
        }
        let before_magic = &suffix[..suffix.len() - 8];
        // A v2 index needs its whole body in the suffix; signal a re-read otherwise.
        if before_magic.len() >= 8 && &before_magic[before_magic.len() - 4..] == ZONE_INDEX_MARKER {
            let blen_pos = before_magic.len() - 8;
            let body_len =
                u32::from_le_bytes(before_magic[blen_pos..blen_pos + 4].try_into()?) as usize;
            if before_magic.len() < 8 + body_len {
                return Ok(None);
            }
        }
        match Self::locate(before_magic) {
            Ok(idx) => Ok(Some(idx)),
            Err(_) => Ok(None),
        }
    }
}

// ── Segment metadata (returned from finish()) ──────────────────────────────────

// `SegmentMeta` now lives in `proximadb-block-format` (next to the per-block
// `BlockStats` it aggregates) so segment-level consumers below the storage
// layer — e.g. the catalog's segment registry — can reach it without a
// `catalog -> storage-common` cycle. Re-exported here so the historical
// `proximadb_storage_common::pax_block::SegmentMeta` path keeps working.
pub use proximadb_block_format::SegmentMeta;

/// Immutable PAX regions retained from the writer for post-publication cache
/// population. This is rebuildable acceleration data, not commit authority.
#[derive(Debug, Clone)]
pub struct PaxCacheSeed {
    pub header_bytes: Vec<u8>,
    pub a0_bytes: Option<std::sync::Arc<[u8]>>,
    pub rabitq_bytes: std::sync::Arc<[u8]>,
    pub rabitq_header_bytes: std::sync::Arc<[u8]>,
    pub footer_bytes: Vec<u8>,
    pub sq8_off: u64,
    pub sq8_bytes: Option<std::sync::Arc<[u8]>>,
}

/// Writer result used by staged flush/compaction. Callers must install the
/// optional seed only after their atomic publication succeeds.
#[derive(Debug)]
pub struct PaxSegmentWrite {
    pub meta: SegmentMeta,
    pub cache_seed: Option<PaxCacheSeed>,
}

// ── Writer ────────────────────────────────────────────────────────────────────

/// Writes `ProximaRecord` rows to a PAX segment file.
///
/// Records are buffered in a `PaxBlockWriter`; when the estimated block size
/// reaches `block_size_threshold`, the block is flushed and a new one begins.
/// TD-RDSTRAT-8 writer-side bookkeeping for the two-level (v3) layout.
struct TwoLevelState {
    /// The trained coarse model (validated at finish; fail-closed).
    model: CoarseModel,
    /// Cumulative cell-end row boundaries (prefix sums of `model.cell_rows`,
    /// len `k_c`). Rows past the last boundary are the no-embedding tail.
    boundaries: Vec<u64>,
    /// Next boundary index awaiting its flush.
    next_boundary: usize,
    /// Block-ordinal at each crossed cell end (1:1 with `boundaries` once all
    /// rows are fed) — the exact Region D `[d_block_begin, d_block_end)` data.
    cell_end_blocks: Vec<u32>,
}

/// Local backing for a bounded coalesced-segment build. Scratch is never
/// authoritative and is reclaimed with the writer on every exit path.
struct LocalSpillBacking {
    task_directory: tempfile::TempDir,
    blocks_path: PathBuf,
    blocks: BufWriter<std::fs::File>,
    blocks_len: u64,
    oid_entries_path: PathBuf,
    oid_entries: BufWriter<std::fs::File>,
    oid_count: u32,
    vectors: Option<DiskVectorSpool>,
}

impl LocalSpillBacking {
    fn new(scratch_root: &Path) -> Result<Self> {
        let task_directory = tempfile::Builder::new()
            .prefix("proximadb-pax-writer-")
            .tempdir_in(scratch_root)?;
        let blocks_path = task_directory.path().join("region-d.blocks");
        let oid_entries_path = task_directory.path().join("oid-entries.bin");
        let blocks = BufWriter::new(
            std::fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&blocks_path)?,
        );
        let oid_entries = BufWriter::new(
            std::fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&oid_entries_path)?,
        );
        let vectors = DiskVectorSpool::new(task_directory.path())?;
        Ok(Self {
            task_directory,
            blocks_path,
            blocks,
            blocks_len: 0,
            oid_entries_path,
            oid_entries,
            oid_count: 0,
            vectors: Some(vectors),
        })
    }

    fn push_oid(&mut self, oid: &str) -> Result<()> {
        if self.oid_count == u32::MAX {
            bail!("PAX OID resolver row count exceeds u32");
        }
        let length =
            u32::try_from(oid.len()).map_err(|_| anyhow::anyhow!("PAX OID length exceeds u32"))?;
        self.oid_entries.write_all(&length.to_le_bytes())?;
        self.oid_entries.write_all(oid.as_bytes())?;
        self.oid_count = self.oid_count.saturating_add(1);
        Ok(())
    }

    fn write_block(&mut self, bytes: &[u8]) -> Result<u64> {
        let offset = self.blocks_len;
        self.blocks.write_all(bytes)?;
        self.blocks_len = self
            .blocks_len
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| anyhow::anyhow!("PAX Region D length exceeds u64"))?;
        Ok(offset)
    }

    fn flush_inputs(&mut self) -> Result<()> {
        self.blocks.flush()?;
        self.blocks.get_ref().sync_data()?;
        self.oid_entries.flush()?;
        self.oid_entries.get_ref().sync_data()?;
        Ok(())
    }

    fn finish_oid_resolver(&mut self, enabled: bool) -> Result<Option<(PathBuf, u64)>> {
        if !enabled || self.oid_count == 0 {
            return Ok(None);
        }
        self.oid_entries.flush()?;
        let output = self.task_directory.path().join("oid-resolver.orp");
        let mut writer = BufWriter::new(
            std::fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&output)?,
        );
        let mut checksum = crc32fast::Hasher::new();
        let magic = b"ORP1";
        writer.write_all(magic)?;
        checksum.update(magic);
        let count = self.oid_count.to_le_bytes();
        writer.write_all(&count)?;
        checksum.update(&count);
        let mut reader = BufReader::new(std::fs::File::open(&self.oid_entries_path)?);
        let mut buffer = vec![0u8; 1024 * 1024];
        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            writer.write_all(&buffer[..read])?;
            checksum.update(&buffer[..read]);
        }
        writer.write_all(&checksum.finalize().to_le_bytes())?;
        writer.flush()?;
        writer.get_ref().sync_data()?;
        let length = writer.get_ref().metadata()?.len();
        Ok(Some((output, length)))
    }
}

/// Cluster-ordered embedding-0 vectors retained for the coalesced A/B regions.
///
/// One contiguous allocation replaces `Vec<Option<Vec<f32>>>`, which created
/// one allocation per record and left a multi-gigabyte allocator RSS tail after
/// million-row compactions. Missing rows occupy a zero-filled fixed-width slot;
/// the compact validity bitmap determines whether encoders observe that slot.
#[derive(Default)]
struct CoalescedVectorBuffer {
    dim: usize,
    expected_rows: usize,
    values: Vec<f32>,
    valid: Vec<bool>,
}

impl CoalescedVectorBuffer {
    fn push(&mut self, values: Option<&[f32]>) -> Result<usize> {
        if self.valid.is_empty() && self.expected_rows > 0 {
            self.valid
                .try_reserve_exact(self.expected_rows)
                .map_err(|error| anyhow::anyhow!("reserve coalesced validity bitmap: {error}"))?;
        }
        match values {
            Some(values) if !values.is_empty() => {
                if self.dim == 0 {
                    self.dim = values.len();
                    let prior_values = self.valid.len().checked_mul(self.dim).ok_or_else(|| {
                        anyhow::anyhow!("coalesced vector buffer size exceeds usize")
                    })?;
                    let observed_rows = self.valid.len().checked_add(1).ok_or_else(|| {
                        anyhow::anyhow!("coalesced vector row count exceeds usize")
                    })?;
                    let expected_values = self
                        .expected_rows
                        .max(observed_rows)
                        .checked_mul(self.dim)
                        .ok_or_else(|| {
                            anyhow::anyhow!("coalesced vector reservation exceeds usize")
                        })?;
                    self.values
                        .try_reserve_exact(expected_values)
                        .map_err(|error| {
                            anyhow::anyhow!("reserve coalesced vector buffer: {error}")
                        })?;
                    self.values.resize(prior_values, 0.0);
                }
                if values.len() != self.dim {
                    bail!(
                        "coalesced vector dim {} != established dim {}",
                        values.len(),
                        self.dim
                    );
                }
                self.values.extend_from_slice(values);
                self.valid.push(true);
                values
                    .len()
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| anyhow::anyhow!("coalesced vector byte size exceeds usize"))
            }
            _ => {
                if self.dim > 0 {
                    let next_len = self.values.len().checked_add(self.dim).ok_or_else(|| {
                        anyhow::anyhow!("coalesced vector buffer size exceeds usize")
                    })?;
                    self.values.resize(next_len, 0.0);
                }
                self.valid.push(false);
                Ok(0)
            }
        }
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn with_expected_rows(mut self, expected_rows: usize) -> Self {
        self.expected_rows = expected_rows;
        self
    }

    fn refs(&self) -> Vec<Option<&[f32]>> {
        if self.dim == 0 {
            return Vec::new();
        }
        self.values
            .chunks_exact(self.dim)
            .zip(&self.valid)
            .map(|(values, valid)| valid.then_some(values))
            .collect()
    }
}

/// TD-PAXRG-1 Region C (ADR-065's exact tier, hoisted): raw f32 rows in ordinal
/// order — `[n u32][dim u32][validity bitmap ceil(n/8)][n·dim·f32]`, invalid
/// rows zero-filled. Built from the SAME coalesced buffer as Regions A/B before
/// its drop, so hoisting costs no extra buffering pass.
pub(crate) fn exact_region_header_len(n_rows: usize) -> u64 {
    8 + n_rows.div_ceil(8) as u64
}

fn encode_exact_region(refs: &[Option<&[f32]>], dim: usize) -> Result<Vec<u8>> {
    let n = refs.len();
    let mut buf = Vec::with_capacity(exact_region_header_len(n) as usize + n * dim * 4);
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    buf.extend_from_slice(&(dim as u32).to_le_bytes());
    let mut bitmap = vec![0u8; n.div_ceil(8)];
    for (i, r) in refs.iter().enumerate() {
        if r.is_some() {
            bitmap[i / 8] |= 1 << (i % 8);
        }
    }
    buf.extend_from_slice(&bitmap);
    for r in refs {
        match r {
            Some(v) if v.len() == dim => {
                for f in v.iter() {
                    buf.extend_from_slice(&f.to_le_bytes());
                }
            }
            _ => buf.extend_from_slice(&vec![0u8; dim * 4]),
        }
    }
    Ok(buf)
}

/// Decoder for the Region C wire form (see [`encode_exact_region`]).
struct ExactRegion<'a> {
    n_rows: usize,
    dim: u32,
    data: &'a [u8],
}

impl<'a> ExactRegion<'a> {
    fn from_bytes(region: &'a [u8]) -> Result<Self> {
        if region.len() < 8 {
            bail!("exact region truncated");
        }
        let n = u32::from_le_bytes(region[0..4].try_into()?) as usize;
        let dim = u32::from_le_bytes(region[4..8].try_into()?);
        let expected = exact_region_header_len(n) as usize + n * dim as usize * 4;
        if region.len() < expected {
            bail!(
                "exact region payload truncated: {} < {expected}",
                region.len()
            );
        }
        Ok(Self {
            n_rows: n,
            dim,
            data: region,
        })
    }

    /// Exact f32 row `i`, or `None` when the validity bitmap marks it absent.
    fn decode_row(&self, row: usize) -> Option<Vec<f32>> {
        if row >= self.n_rows {
            return None;
        }
        let bitmap_base = 8usize;
        let valid = self.data.get(bitmap_base + row / 8)? & (1 << (row % 8)) != 0;
        if !valid {
            return None;
        }
        let payload_base = bitmap_base + self.n_rows.div_ceil(8);
        let start = payload_base + row * self.dim as usize * 4;
        let end = start + self.dim as usize * 4;
        let bytes = self.data.get(start..end)?;
        Some(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        )
    }
}

/// TD-COMPACT-1 S2: cumulative writer sub-phase timers, allocated only when
/// `PROXIMADB_TRACE_PAX_WRITE` is set at writer construction. Buckets cover the
/// per-record `add_record` work plus the finish-time coalesced region encodes;
/// one summary line is emitted from [`PaxSegmentWriter::finish`]. Near-zero cost
/// when the env is unset: a single `Option` discriminant check per span.
#[derive(Default)]
struct WriteTraceStats {
    /// Coalesced Region A: per-record f32 buffering for `rabitq_vectors` +
    /// the finish-time `encode_region` (fit + rotation + per-vector encode).
    rabitq_encode: Duration,
    /// Coalesced Region B: the finish-time `encode_sq8_region` (SQ8 rerank tier).
    rerank_encode: Duration,
    /// `PaxBlockWriter::add_record` row buffering (FlatRow extraction, msgpack
    /// props, embedding clones, column pushes).
    raw_buffer: Duration,
    /// Every `flush_current_block` body: block serialize (stripes + codecs),
    /// re-open for stats/zone summaries, and `file_buf` append.
    block_cut_compress: Duration,
    /// Centroid/radius accumulation (`accumulate_centroid`).
    cluster_bookkeeping: Duration,
    /// `add_record` remainder (size estimate, two-level boundary checks, ...).
    other: Duration,
    /// Blocks cut so far (size-triggered + cell-boundary + final flushes).
    blocks_cut: u64,
    /// Running f32 bytes buffered in `rabitq_vectors` for the coalesced regions.
    rabitq_buf_bytes: usize,
    /// Peak of `file_buf` + coalesced vector buffer observed at block cuts.
    peak_buffered_bytes: usize,
}

pub struct PaxSegmentWriter {
    path: PathBuf,
    mode: BlockMode,
    compression: BlockCompression,
    collection_id: String,
    schema_fingerprint: u64,
    embedding_count: usize,
    block_size_threshold: usize,
    /// Vector quantization strategy for every block in this segment (P3 Phase D).
    quant: VectorQuant,
    /// Opt-in exact-f32 tier (P3 Phase D) — re-applied to every block writer so
    /// each block carries the f32 stripe when enabled.
    f32_tier: bool,
    /// Tier-2 rerank quant (default Sq8) — re-applied to every block writer.
    rerank_quant: VectorQuant,
    /// Exact clustered transform over SQ8 code bytes. Default OFF and selected
    /// independently per stripe by realized byte size.
    lossless_clustered: bool,
    /// Exact scalar all-null elision and post-codec LZ4. Default OFF.
    lossless_scalar: bool,
    /// Additive canonical MVCC `record_version` stripe. Default OFF; the local
    /// spill path enables it because spill itself is an opt-in write format.
    record_version_stripe: bool,
    /// P-Shred (ADR-055): `(prop_key, user_col_id)` to shred into typed user-columns —
    /// re-applied to every block writer so all blocks in the segment shred uniformly.
    shred_spec: Vec<proximadb_block_format::ShredColumn>,

    current_writer: PaxBlockWriter,
    index: SegmentIndex,
    block_stats: Vec<BlockStats>,
    file_buf: Vec<u8>,
    /// Optional bounded local backing for Regions A/B/D and the OID resolver.
    /// `None` preserves the canonical in-memory writer byte-for-byte.
    local_spill: Option<LocalSpillBacking>,
    row_count: u64,

    /// TD-RDSTRAT-5 S1: when true, accumulate each block's centroid (mean of its
    /// embedding-0 f32 vectors) into `block_centroids` as blocks flush. Off by
    /// default — zero cost for callers that don't opt in.
    compute_centroids: bool,
    /// Running sum of the current (not-yet-flushed) block's embedding-0 vectors.
    cur_centroid_sum: Vec<f64>,
    /// Count of vectors summed into `cur_centroid_sum` for the current block.
    cur_centroid_n: u64,
    /// TD-RDSTRAT-5 lever-3: running sum of ‖x‖² over the current block's
    /// embedding-0 vectors. With `cur_centroid_sum` this yields the block's RMS
    /// spread in ONE pass: `radius² = mean(‖x‖²) − ‖centroid‖²` (trace of the
    /// covariance). The read-side prune ranks blocks by the distance lower bound
    /// `d(q,centroid) − k·radius`, so spread-aware blocks aren't wrongly pruned.
    cur_centroid_sumsq: f64,
    /// Finalised per-block centroids, one per flushed block (emission order).
    block_centroids: Vec<Vec<f32>>,
    /// Finalised per-block RMS radius (spread), 1:1 with `block_centroids`
    /// (`0.0` for a block with no Fp32 vector). Empty unless centroids opted in.
    block_radii: Vec<f32>,
    /// Exact SQ8 transforms selected in each emitted block, by physical column
    /// ID. Kept 1:1 with `index.blocks` for footer-v2 assignments.
    block_transformed_sq8_columns: Vec<Vec<i32>>,

    /// ADR-062 / TD-RDSTRAT-6: emit the coalesced-RaBitQ layout — a file-level
    /// header region (cluster-ordered, single segment centroid) + self-describing
    /// footer-index — so the read path scans ALL RaBitQ codes in one GET
    /// (keep=100%, ~0.99 recall) and reranks survivors via coalesced block GETs.
    /// Default OFF; the flush path opts in for RaBitQ-quantized collections. When
    /// on, data blocks are written as `VectorQuant::Sq8` (the survivor-rerank
    /// data) — the unchanged SQ8 decode path reconstructs + reranks them.
    coalesced_rabitq: bool,
    /// TD-PAXRG-1: write the v4 **row-group Region D** layout — Region D
    /// granules are Parquet-style row groups (Olap-framed blocks with the
    /// RowDirectory and in-block rgdir suppressed), per-RG stats ride the
    /// footer's MinMax payload, and the block-size target is floored at
    /// [`RG_TARGET_MIN_BYTES`] so sub-granule framing amplification is
    /// structurally impossible. Default OFF (`PROXIMADB_PAX_WRITE_RG_LAYOUT`).
    rg_layout: bool,
    /// Per-flushed-RG OID-chunk descriptors `(rel_off, len, encoding_id, lz4)`
    /// for the OID stripe inside each RG's block bytes, 1:1 with
    /// `index.blocks`. Populated only when `rg_layout` is on; serialized in the
    /// footer's per-entry MinMax stats payload for the ANN top-k chunk fetch.
    rg_oid_chunks: Vec<(u32, u32, u8, bool)>,
    /// Embedding-0 f32 vectors in cluster (add) order, buffered contiguously for
    /// the segment-level RaBitQ/SQ8 regions. Populated only in coalesced mode.
    rabitq_vectors: CoalescedVectorBuffer,
    /// TD-RDSTRAT-8: the two-level-IVF coarse model + boundary bookkeeping.
    /// Present ⇒ emit the v3 layout (`[prefix][A0][A][B][D][footer]`): the
    /// current block is force-flushed at every coarse-cell boundary (blocks
    /// never straddle a cell) and Region A0 records per-cell byte extents.
    two_level: Option<TwoLevelState>,
    /// Encoding-aware per-row byte estimate (computed from the first record's
    /// dim + the quant/f32_tier/embedding_count config). Replaces the flat 1024
    /// that overestimated SQ8 blocks by ~4.5×. 0 = not yet computed.
    per_row_estimate: usize,
    /// TD-DELVEC-1 WI-2b: when true, capture each record's canonical oid in
    /// add_record (stream) order into `oids`, and expose it via
    /// [`PaxSegmentWriter::oid_resolver_bytes`] as a serialized
    /// `OidPositionResolver` (WI-2c persists it as a footer region inside the segment).
    /// Default OFF — zero cost; the default flush path is byte-for-byte unchanged.
    compute_oid_resolver: bool,
    /// Captured canonical oids in add_record (stream) order, one per record,
    /// populated only when `compute_oid_resolver` is on. Position `i` here is the
    /// segment's footer-block position `i` (verified: `add_record` call-order is
    /// the on-disk order; TD-DELVEC-1 §9.2).
    oids: Vec<String>,
    /// TD-COMPACT-1 S2: `Some` iff `PROXIMADB_TRACE_PAX_WRITE` was set when the
    /// writer was constructed (read ONCE, here) — cumulative sub-phase timers.
    write_trace: Option<Box<WriteTraceStats>>,
}

impl PaxSegmentWriter {
    /// Create a new segment writer. The segment file is written to `path`.
    ///
    /// `block_size_threshold` (bytes) controls when the current block is
    /// flushed and a new one begins. Defaults to `DEFAULT_TARGET_BLOCK_SIZE_BYTES`.
    pub fn new(
        path: impl AsRef<Path>,
        mode: BlockMode,
        compression: BlockCompression,
        collection_id: impl Into<String>,
        schema_fingerprint: u64,
        embedding_count: usize,
        block_size_threshold: Option<usize>,
    ) -> Self {
        let collection_id = collection_id.into();
        let threshold = block_size_threshold
            .unwrap_or(DEFAULT_TARGET_BLOCK_SIZE_BYTES)
            .min(MAX_TARGET_BLOCK_SIZE_BYTES);

        let writer = PaxBlockWriter::new(
            mode,
            compression,
            &collection_id,
            schema_fingerprint,
            embedding_count,
        );

        let record_version_stripe = writer.writes_record_version();
        Self {
            path: path.as_ref().to_path_buf(),
            mode,
            compression,
            collection_id,
            schema_fingerprint,
            embedding_count,
            block_size_threshold: threshold,
            rg_layout: false,
            rg_oid_chunks: Vec::new(),
            quant: VectorQuant::Auto,
            f32_tier: false,
            rerank_quant: VectorQuant::Sq8,
            lossless_clustered: false,
            lossless_scalar: false,
            record_version_stripe,
            shred_spec: Vec::new(),
            current_writer: writer,
            index: SegmentIndex { blocks: Vec::new() },
            block_stats: Vec::new(),
            file_buf: Vec::new(),
            local_spill: None,
            row_count: 0,
            compute_centroids: false,
            cur_centroid_sum: Vec::new(),
            cur_centroid_n: 0,
            cur_centroid_sumsq: 0.0,
            block_centroids: Vec::new(),
            block_radii: Vec::new(),
            block_transformed_sq8_columns: Vec::new(),
            coalesced_rabitq: false,
            rabitq_vectors: CoalescedVectorBuffer::default(),
            two_level: None,
            per_row_estimate: 0,
            compute_oid_resolver: false,
            oids: Vec::new(),
            write_trace: std::env::var_os("PROXIMADB_TRACE_PAX_WRITE")
                .is_some()
                .then(|| Box::new(WriteTraceStats::default())),
        }
    }

    /// TD-RDSTRAT-5 S1: opt in to per-block **centroid** computation. When on, the
    /// writer accumulates each block's embedding-0 f32 mean and returns them in
    /// [`SegmentMeta::block_centroids`] — the vector zone-map the Vector Object
    /// Economy directory prunes on. Default OFF (zero cost otherwise). Builder
    /// form mirroring [`with_quant`]; no block-writer rebuild needed (centroids
    /// are accumulated in the segment writer, orthogonal to block encoding).
    pub fn with_block_centroids(mut self, enabled: bool) -> Self {
        self.compute_centroids = enabled;
        self
    }

    /// TD-DELVEC-1 WI-2b: opt in to capturing the segment's oid→footer-block
    /// position resolver. When on, `add_record` records each canonical oid in
    /// stream order; [`PaxSegmentWriter::oid_resolver_bytes`] then serializes a
    /// CRC32'd `ORP1` resolver (WI-2c persists it as a footer region inside
    /// the segment). Default OFF — pre-existing callers pay nothing. Builder form
    /// mirroring [`with_block_centroids`]; no block-writer rebuild needed.
    pub fn with_oid_resolver(mut self, enabled: bool) -> Self {
        self.compute_oid_resolver = enabled;
        self
    }

    /// TD-DELVEC-1 WI-2b: serialize the captured oid→footer-block-position
    /// resolver (CRC32'd `ORP1`), or `Ok(None)` if `with_oid_resolver` was not
    /// enabled. The oids are complete once every record is added, so call this
    /// BEFORE [`finish`](Self::finish) consumes the writer; the flush path
    /// (WI-2c) persists the bytes as a footer region inside the segment.
    pub fn oid_resolver_bytes(&self) -> Result<Option<Vec<u8>>, crate::bitmap::BitmapError> {
        if !self.compute_oid_resolver {
            return Ok(None);
        }
        if self.local_spill.is_some() {
            return Err(crate::bitmap::BitmapError::SerializationError(
                "disk-backed OID resolver is finalized only with the segment".to_string(),
            ));
        }
        let bytes =
            crate::oid_position_resolver::OidPositionResolver::from_stream_order(self.oids.clone())
                .serialize()?;
        Ok(Some(bytes))
    }

    /// Set the vector quantization strategy for this segment (P3 Phase D). Builder form
    /// so existing `new(..)` callers are unchanged; rebuilds the (still-empty) current
    /// block writer so the strategy applies from the first record. `Auto` = env default.
    pub fn with_quant(mut self, quant: VectorQuant) -> Self {
        self.quant = quant;
        self.current_writer = self.fresh_block_writer();
        self
    }

    /// Enable (or disable) the optional exact-f32 tier (P3 Phase D) for every
    /// block in this segment. Builder form mirroring [`with_quant`]; rebuilds the
    /// (still-empty) current block writer so it applies from the first record.
    /// Default OFF; the flush path enables it from the `pax_f32_tier` tag / env.
    pub fn with_f32_tier(mut self, enabled: bool) -> Self {
        self.f32_tier = enabled;
        self.current_writer = self.fresh_block_writer();
        self
    }

    /// Set the tier-2 rerank quantization strategy for every block in this
    /// segment. Default `Sq8` (the validated tier-2); `Fp16` for near-lossless;
    /// `RawF32` for exact. Only used when tier 1 is RaBitQ.
    pub fn with_rerank_quant(mut self, quant: VectorQuant) -> Self {
        self.rerank_quant = quant;
        self.current_writer = self.fresh_block_writer();
        self
    }

    /// Enable the lossless clustered SQ8 transform for every block.
    pub fn with_lossless_clustered(mut self, enabled: bool) -> Self {
        self.lossless_clustered = enabled;
        self.current_writer = self.fresh_block_writer();
        self
    }

    /// Enable exact scalar all-null elision and post-codec LZ4 for every block.
    pub fn with_lossless_scalar(mut self, enabled: bool) -> Self {
        self.lossless_scalar = enabled;
        self.current_writer = self.fresh_block_writer();
        self
    }

    /// Enable the additive canonical MVCC `record_version` stripe in every
    /// block emitted by this segment.
    pub fn with_record_version(mut self, enabled: bool) -> Self {
        self.record_version_stripe = enabled;
        self.current_writer = self.fresh_block_writer();
        self
    }

    /// Mark the next appended record as the start of a producer-defined cluster.
    pub fn start_cluster_run(&mut self) {
        self.current_writer.start_cluster_run();
    }

    /// P-Shred (ADR-055): shred the given props keys `(prop_key, user_col_id)` into
    /// typed user-columns for every block in this segment. Builder form mirroring
    /// [`with_quant`]; empty ⇒ no shredding (byte-for-byte today's output).
    pub fn with_shred_spec(self, spec: Vec<(String, i32)>) -> Self {
        // Coarse/unknown declared types (SST flush path) — see `ShredColumn`.
        self.with_shred_columns(
            spec.into_iter()
                .map(|(key, id)| proximadb_block_format::ShredColumn::untyped(key, id))
                .collect(),
        )
    }

    /// Declare shred columns with their exact types where the caller has them
    /// (TD-USUB-6) — the relational path resolves them from `CatalogTableSchema`.
    pub fn with_shred_columns(mut self, spec: Vec<proximadb_block_format::ShredColumn>) -> Self {
        self.shred_spec = spec;
        self.current_writer = self.fresh_block_writer();
        self
    }

    /// ADR-062 / TD-RDSTRAT-6: emit the coalesced-RaBitQ layout. When on, the
    /// RaBitQ binary tier is hoisted into a coalesced file-level header region
    /// (single segment centroid) and data blocks are written as SQ8 (the
    /// survivor-rerank data); the read path scans the region in one GET and
    /// reranks survivors via coalesced block GETs. Builder form mirroring
    /// [`with_quant`]; rebuilds the (still-empty) current block writer so the SQ8
    /// block encoding applies from the first record.
    pub fn with_coalesced_rabitq(mut self, enabled: bool) -> Self {
        self.coalesced_rabitq = enabled;
        self.current_writer = self.fresh_block_writer();
        self
    }

    /// TD-PAXRG-1: write the v4 row-group Region D layout (see the `rg_layout`
    /// field doc). Builder form mirroring [`with_coalesced_rabitq`]; rebuilds
    /// the (still-empty) current block writer so the Olap framing +
    /// rgdir suppression apply from the first record. Also floors the block
    /// target at [`RG_TARGET_MIN_BYTES`] — sub-granule targets would recreate
    /// the per-RG framing amplification the layout exists to kill.
    pub fn with_rg_layout(mut self, enabled: bool) -> Self {
        self.rg_layout = enabled;
        if enabled {
            self.block_size_threshold = self.block_size_threshold.max(RG_TARGET_MIN_BYTES);
            // TD-PAXRG-1 (post-flip invariant): row-group Region D suppresses
            // the RowDirectory, so the OID resolver region IS the point-lookup
            // authority — a format invariant of the layout, auto-enabled here
            // rather than a caller obligation.
            self.compute_oid_resolver = true;
            self.current_writer = self.fresh_block_writer();
        }
        self
    }

    /// Pre-size corpus-wide coalesced bookkeeping from the already-admitted
    /// record count. Reservation itself is fallible and occurs on first append,
    /// so allocation failure is returned through [`Self::add_record`].
    pub fn with_expected_rows(mut self, expected_rows: usize) -> Self {
        self.rabitq_vectors =
            std::mem::take(&mut self.rabitq_vectors).with_expected_rows(expected_rows);
        self
    }

    /// Build coalesced regions and data blocks through bounded local scratch.
    /// The caller must still opt into coalesced RaBitQ and feed final row order.
    /// Construction fails before the first record when scratch cannot be opened.
    pub fn with_local_spill(mut self, scratch_root: &Path) -> Result<Self> {
        if self.row_count != 0 || !self.current_writer.is_empty() {
            bail!("local spill must be configured before adding PAX records");
        }
        self.local_spill = Some(LocalSpillBacking::new(scratch_root)?);
        Ok(self)
    }

    /// TD-RDSTRAT-8: emit the **persisted-IVF-probe (v3)** layout. The caller has
    /// already reordered records by `(cell rank, PC1, index)` (see
    /// `cluster_plan_ivf_probe`); `model.cell_rows` tells the writer where the
    /// cell boundaries fall so it can pad/flush blocks there and record exact
    /// per-cell byte extents into Region A0. Only meaningful together with
    /// [`Self::with_coalesced_rabitq`] — without it the model is ignored
    /// (fail-safe: a non-coalesced segment has no regions to address).
    pub fn with_two_level(mut self, model: CoarseModel) -> Self {
        let mut boundaries = Vec::with_capacity(model.cell_rows.len());
        let mut acc = 0u64;
        for &rows in &model.cell_rows {
            acc += rows;
            boundaries.push(acc);
        }
        let mut state = TwoLevelState {
            model,
            boundaries,
            next_boundary: 0,
            cell_end_blocks: Vec::new(),
        };
        // Leading empty cells end at row 0 — `add_record` never sees row 0, so
        // consume them up front (their block range is empty at ordinal 0).
        while state
            .boundaries
            .get(state.next_boundary)
            .is_some_and(|&b| b == 0)
        {
            state.cell_end_blocks.push(0);
            state.next_boundary += 1;
        }
        self.two_level = Some(state);
        self
    }

    /// Build a fresh (empty) block writer carrying ALL of this segment's accumulated
    /// settings. Single source of truth so every builder AND every mid-segment block
    /// rotation re-applies the same config (quant, f32 tier, rerank quant, shred spec)
    /// — no per-builder drift. In coalesced mode the block's tier-1 encoding is forced
    /// to SQ8 (the RaBitQ binary tier lives in the header region, not the block).
    fn fresh_block_writer(&self) -> PaxBlockWriter {
        let block_quant = if self.coalesced_rabitq {
            VectorQuant::Sq8
        } else {
            self.quant
        };
        // TD-PAXRG-1: v4 row-group Region D granules are Olap-framed — no
        // RowDirectory (point lookups go through the OID resolver), no in-block
        // rgdir (the RG's stats live in the segment footer).
        let mode = if self.rg_layout {
            BlockMode::Olap
        } else {
            self.mode
        };
        PaxBlockWriter::new(
            mode,
            self.compression,
            &self.collection_id,
            self.schema_fingerprint,
            self.embedding_count,
        )
        .with_quant(block_quant)
        // TD-PAXRG-1: under v4 the exact tier is HOISTED to Region C, so the
        // block stops emitting F32_TIER stripes.
        .with_f32_tier(self.f32_tier && !self.rg_layout)
        .with_rerank_quant(self.rerank_quant)
        .with_clustered_sq8_lossless(self.lossless_clustered)
        .with_lossless_scalar(self.lossless_scalar)
        .with_record_version(self.record_version_stripe)
        .with_hoist_vector_tier(self.coalesced_rabitq)
        .with_suppress_rgdir(self.rg_layout)
        .with_shred_columns(self.shred_spec.clone())
    }

    /// Encoding-aware per-row byte estimate (uncompressed). Controls the block
    /// COUNT (how many blocks → how many survivor GETs). Compression is a
    /// separate, orthogonal lever (reduces on-disk bytes/GET, not GET count).
    /// `DEFAULT_BLOCK_METADATA_OVERHEAD_BYTES` covers OID + timestamps + props.
    fn estimate_per_row_bytes(&self, dim: usize) -> usize {
        let vector_bytes = if self.coalesced_rabitq {
            // RaBitQ (Region A) + SQ8 (Region B) are hoisted out of blocks; the
            // block carries only row data + the optional f32 exact tier. Under
            // v4 (rg_layout) the exact tier is ALSO hoisted (Region C), so the
            // block carries no vector bytes at all.
            let hoist_exact = self.f32_tier && !self.rg_layout;
            let f32 = if hoist_exact { dim * 4 } else { 0 };
            f32 * self.embedding_count.max(1)
        } else {
            // Non-coalesced: EMBED_BASE carries the tier-1 encoding directly.
            let tier1 = match self.quant {
                VectorQuant::RawF32 => dim * 4,
                VectorQuant::Fp16 => dim * 2,
                VectorQuant::RaBitQ => dim.div_ceil(8) + 8 + dim, // code + SQ8 rerank
                _ => dim,                                         // SQ8 / Auto
            };
            let f32 = if self.f32_tier { dim * 4 } else { 0 };
            (tier1 + f32) * self.embedding_count.max(1)
        };
        vector_bytes + DEFAULT_BLOCK_METADATA_OVERHEAD_BYTES
    }

    /// Append a record to the current block.
    ///
    /// Flushes the block automatically when it exceeds `block_size_threshold`.
    pub fn add_record(&mut self, record: &ProximaRecord) -> Result<()> {
        // TD-COMPACT-1 S2: per-span timers, active only when the writer was
        // constructed with `PROXIMADB_TRACE_PAX_WRITE` set (plain bool check +
        // no `Instant::now` otherwise). Spans are accumulated into locals and
        // merged into `write_trace` once at the end of this call.
        let trace_on = self.write_trace.is_some();
        let t_total = trace_on.then(Instant::now);
        let flush_before = self
            .write_trace
            .as_deref()
            .map(|tr| tr.block_cut_compress)
            .unwrap_or_default();
        let mut d_raw = Duration::ZERO;
        let mut d_cluster = Duration::ZERO;
        let mut d_rabitq = Duration::ZERO;
        let mut rabitq_bytes = 0usize;

        let t = trace_on.then(Instant::now);
        self.current_writer.add_record(record)?;
        self.row_count += 1;
        if let Some(t) = t {
            d_raw = t.elapsed();
        }
        if self.compute_centroids {
            let t = trace_on.then(Instant::now);
            self.accumulate_centroid(record);
            if let Some(t) = t {
                d_cluster = t.elapsed();
            }
        }
        // TD-DELVEC-1 WI-2b: capture the canonical oid in add_record (stream)
        // order — the only point that sees post-cluster-reorder footer-block
        // order (verified: call-order == on-disk position; §9.2). Default off;
        // mirrors the centroid opt-in above.
        if self.compute_oid_resolver {
            if let Some(spill) = self.local_spill.as_mut() {
                spill.push_oid(&record.oid)?;
            } else {
                self.oids.push(record.oid.clone());
            }
        }
        // ADR-062: buffer the embedding-0 f32 vector (in cluster/add order) for
        // the segment-level RaBitQ region. The caller has already reordered
        // records by `cluster_order_pca_ivf`, so this preserves survivor locality.
        if self.coalesced_rabitq {
            // Dequantize ANY EmbeddingValues variant (Fp32/Fp16/Bf16/Int8/UInt8) to
            // f32 for the RaBitQ+SQ8 segment-level quantization. The old Fp32-only
            // extraction silently dropped non-Fp32 embeddings (e.g. after ingest-time
            // canonical-precision coercion) → garbage SQ8 params → 0.35% recall.
            let t = trace_on.then(Instant::now);
            let values = record.embeddings.first().map(|cell| cell.as_fp32_cow());
            if let Some(spill) = self.local_spill.as_mut() {
                let vectors = spill
                    .vectors
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("PAX spill vector spool already finalized"))?;
                vectors.push(values.as_deref())?;
                rabitq_bytes = values.as_deref().map_or(0, |vector| {
                    vector.len().saturating_mul(std::mem::size_of::<f32>())
                });
            } else {
                rabitq_bytes = self.rabitq_vectors.push(values.as_deref())?;
            }
            if let Some(t) = t {
                d_rabitq = t.elapsed();
            }
        }

        // Encoding-aware size estimate: compute the per-row byte cost from the
        // encoding config (quant, coalesced, f32_tier) + the vector dim once
        // (from the first record). This replaces the flat 1024 B/row estimate
        // that overestimated SQ8 blocks by ~4.5× (actual ~228 B/row for 128d).
        if self.per_row_estimate == 0
            && let Some(dim) = record.embeddings.first().map(|e| e.dim as usize)
        {
            self.per_row_estimate = self.estimate_per_row_bytes(dim);
        }
        // PR2: accurate block-flush threshold — actual metadata bytes (tracked from the
        // records, includes real text/props) + predicted vector bytes (deterministic from
        // the encoding config). Replaces the flat per-row estimate.
        let per_row = if self.per_row_estimate > 0 {
            self.per_row_estimate
        } else {
            1024 // safe fallback for non-vector records
        };
        let vector_bytes = self.current_writer.row_count() * per_row;
        let metadata_bytes = self.current_writer.accumulated_metadata_bytes();
        let block_overhead = 1024; // header(64B) + footer(32B) + column footer — amortized
        let total = vector_bytes + metadata_bytes + block_overhead;
        // TD-COMPACT-2: ALSO cut on a hard row bound. The byte cut works off the
        // modeled `per_row` estimate; when it under-counts (coalesced mode hoists
        // vectors out of blocks → ~200 B/row modeled), a giant two-level cell
        // would otherwise become one giant block whose cut-time stripe assembly
        // is the untracked memory/latency spike. Cell alignment is preserved: a
        // mid-cell cut just makes the cell span more whole blocks, which the A0
        // directory's `d_block_begin..d_block_end` range already represents.
        if total >= self.block_size_threshold || self.current_writer.row_count() >= MAX_BLOCK_ROWS {
            self.flush_current_block()?;
        }
        // TD-RDSTRAT-8: pad/flush at every coarse-cell boundary so blocks never
        // straddle a cell (per-cell Region D ranges stay exact). Empty cells
        // share the boundary row, hence the loop. A size-triggered flush at the
        // same row is harmless (the second flush is a no-op).
        loop {
            let at_boundary = self
                .two_level
                .as_ref()
                .is_some_and(|tl| tl.boundaries.get(tl.next_boundary) == Some(&self.row_count));
            if !at_boundary {
                break;
            }
            self.flush_current_block()?;
            let block_count = u32::try_from(self.index.blocks.len())
                .map_err(|_| anyhow::anyhow!("block ordinal exceeds u32"))?;
            if let Some(tl) = &mut self.two_level {
                tl.cell_end_blocks.push(block_count);
                tl.next_boundary += 1;
            }
        }
        // TD-COMPACT-1 S2: merge this call's spans. Block flushes triggered above
        // accounted themselves into `block_cut_compress` (see
        // `flush_current_block`), so `other` = total − spans − flush delta:
        // the size-estimate arithmetic + boundary checks + anything unattributed.
        if let (Some(t0), Some(tr)) = (t_total, self.write_trace.as_deref_mut()) {
            tr.raw_buffer += d_raw;
            tr.cluster_bookkeeping += d_cluster;
            tr.rabitq_encode += d_rabitq;
            tr.rabitq_buf_bytes += rabitq_bytes;
            let flush_delta = tr.block_cut_compress.saturating_sub(flush_before);
            tr.other += t0
                .elapsed()
                .saturating_sub(d_raw + d_cluster + d_rabitq + flush_delta);
        }
        Ok(())
    }

    /// TD-RDSTRAT-5 S1: fold `record`'s embedding-0 f32 vector into the current
    /// block's running centroid sum. Only the canonical Fp32 write-time
    /// representation contributes; other variants are skipped (the block's
    /// centroid is the mean over its Fp32 rows).
    fn accumulate_centroid(&mut self, record: &ProximaRecord) {
        // Dequantize ANY variant to f32 for the centroid sum (was Fp32-only).
        let Some(cell) = record.embeddings.first() else {
            return;
        };
        let v = cell.as_fp32_cow();
        if v.is_empty() {
            return;
        }
        if self.cur_centroid_sum.is_empty() {
            self.cur_centroid_sum = vec![0f64; v.len()];
        }
        if self.cur_centroid_sum.len() == v.len() {
            let mut norm_sq = 0f64;
            for (s, &x) in self.cur_centroid_sum.iter_mut().zip(v.iter()) {
                let x = x as f64;
                *s += x;
                norm_sq += x * x;
            }
            self.cur_centroid_sumsq += norm_sq;
            self.cur_centroid_n += 1;
        }
    }

    /// Finalise the current block's centroid (mean = sum / n) into
    /// `block_centroids` and reset the accumulator. Pushes an empty centroid when
    /// the block carried no Fp32 vector, so `block_centroids` stays 1:1 with
    /// blocks. No-op unless centroid computation was opted in.
    fn finalize_block_centroid(&mut self) {
        if !self.compute_centroids {
            return;
        }
        let (centroid, radius) = if self.cur_centroid_n > 0 {
            let n = self.cur_centroid_n as f64;
            let centroid: Vec<f32> = self
                .cur_centroid_sum
                .iter()
                .map(|&s| (s / n) as f32)
                .collect();
            // RMS spread in one pass: radius² = mean(‖x‖²) − ‖centroid‖². Clamp at
            // 0 to absorb float error when all rows are identical.
            let centroid_norm_sq: f64 = self
                .cur_centroid_sum
                .iter()
                .map(|&s| (s / n) * (s / n))
                .sum();
            let var = (self.cur_centroid_sumsq / n - centroid_norm_sq).max(0.0);
            (centroid, var.sqrt() as f32)
        } else {
            (Vec::new(), 0.0)
        };
        self.block_centroids.push(centroid);
        self.block_radii.push(radius);
        self.cur_centroid_sum.clear();
        self.cur_centroid_n = 0;
        self.cur_centroid_sumsq = 0.0;
    }

    /// Force-flush any buffered records as the final (possibly partial) block.
    fn flush_current_block(&mut self) -> Result<()> {
        if self.current_writer.is_empty() {
            return Ok(());
        }
        // TD-COMPACT-1 S2: everything in this body (block serialize + codecs +
        // stats/zone re-open + file_buf append) is the `block_cut_compress` bucket.
        let t_flush = self.write_trace.is_some().then(Instant::now);
        // Capture timestamp bounds before flush (flush does not reset internal state).
        let min_ts = self.current_writer.min_ts();
        let max_ts = self.current_writer.max_ts();
        let row_count = self.current_writer.row_count() as u32;

        let block_bytes = self.current_writer.flush()?;
        let block_size = block_bytes.len() as u32;
        let offset = match self.local_spill.as_mut() {
            Some(spill) => spill.write_block(&block_bytes)?,
            None => self.file_buf.len() as u64,
        };

        let reader = PaxBlockReader::open(&block_bytes)?;
        self.block_transformed_sq8_columns.push(
            reader
                .vector_params()
                .transforms
                .iter()
                .map(|transform| transform.column_id)
                .collect(),
        );
        let stats =
            BlockStats::from_metas(row_count, block_size, min_ts, max_ts, reader.column_metas());

        // TD-167 / ADR-034 P1: capture each block's canonical-column bounds (already
        // in hand via the just-opened reader) into the **v2** segment index, so the
        // reader prunes from the cached index with NO per-block metadata GET. Always
        // emitted now — v1 (no zone summaries → per-block footer read to prune) is
        // write-deprecated. A block with no canonical bounds gets an empty summary
        // (it simply doesn't prune — never a per-block read), so the depth-collapse
        // is unconditional.
        let zone = Some(BlockZoneSummary::from_column_metas(
            row_count,
            reader.column_metas(),
        ));
        // TD-PAXRG-1: capture this RG's OID-chunk extent (the OID stripe's
        // offset/len within the RG's block bytes) so the footer's MinMax
        // payload can address it for the ANN top-k chunk fetch. 1:1 with the
        // index entry pushed right below.
        if self.rg_layout {
            let oid_chunk = reader
                .column_metas()
                .iter()
                .find(|m| m.column_id == col_id::OID)
                .map(|m| {
                    (
                        m.stripe_offset,
                        m.stripe_len,
                        m.encoding_id,
                        m.is_lz4_compressed,
                    )
                })
                .unwrap_or((0u32, 0u32, 0u8, false));
            self.rg_oid_chunks.push(oid_chunk);
        }
        // TD-FPRUNE-1 P2 (default-OFF): lift this block's bounded columns'
        // ColumnMeta into a footer stats payload while the reader is open (the
        // only point the metas are in hand). Region-D coalesced blocks carry no
        // vector stripes, so these are scalar/metadata columns only. Populated
        // for ALL columns with usable bounds — the filter simply ignores
        // columns it doesn't query, and this avoids threading the filterable-
        // column set through the writer.
        let footer_stats = if footer_stats_enabled() {
            let keep: Vec<i32> = reader.column_metas().iter().map(|m| m.column_id).collect();
            let stats = proximadb_block_format::FooterBlockStats::from_column_metas(
                row_count,
                reader.column_metas(),
                &keep,
            );
            (!stats.is_empty()).then(|| stats.to_bytes())
        } else {
            None
        };
        self.index.blocks.push(BlockIndexEntry {
            offset,
            size: block_size,
            zone,
            footer_stats,
        });
        if self.local_spill.is_none() {
            self.file_buf.extend_from_slice(&block_bytes);
        }
        self.block_stats.push(stats);
        // Finalise this block's centroid (1:1 with the index entry just pushed).
        self.finalize_block_centroid();

        // Reset writer for the next block (preserving the segment's quant + f32-tier +
        // rerank + shred-spec strategy — see `fresh_block_writer`).
        self.current_writer = self.fresh_block_writer();
        if let Some(t) = t_flush {
            let file_len = self
                .local_spill
                .as_ref()
                .map_or(self.file_buf.len() as u64, |spill| spill.blocks_len)
                .min(usize::MAX as u64) as usize;
            if let Some(tr) = self.write_trace.as_deref_mut() {
                tr.block_cut_compress += t.elapsed();
                tr.blocks_cut += 1;
                let buffered = file_len + tr.rabitq_buf_bytes;
                if buffered > tr.peak_buffered_bytes {
                    tr.peak_buffered_bytes = buffered;
                }
            }
        }
        Ok(())
    }

    /// Finalise the segment: flush remaining records, then either append the
    /// legacy `[blocks][index][magic]` tail or — when coalesced-RaBitQ is on —
    /// assemble the ADR-062 layout `[header][RaBitQ region][blocks][footer][tail]`.
    /// Persists to `self.path`; returns `SegmentMeta` for Iceberg manifest use.
    pub fn finish(self) -> Result<SegmentMeta> {
        Ok(self.finish_internal(None)?.meta)
    }

    /// Finish while retaining writer-owned regions for cache-on-write.
    /// `include_sq8=false` captures invariant/control data only.
    pub fn finish_with_cache_seed(self, include_sq8: bool) -> Result<PaxSegmentWrite> {
        self.finish_internal(Some(include_sq8))
    }

    fn finish_internal(mut self, capture_sq8: Option<bool>) -> Result<PaxSegmentWrite> {
        // Flush any remaining rows as the last block
        self.flush_current_block()?;

        if self.index.blocks.is_empty() {
            bail!("segment is empty — nothing to write");
        }

        // Coalesced-RaBitQ requires embedding-0 f32 vectors to build the region.
        // If there are none (a non-vector or malformed batch), fall through to the
        // legacy layout rather than failing the flush — mixed-read-safe.
        let coalesced_dim = self.local_spill.as_ref().map_or_else(
            || self.rabitq_vectors.dim() as u32,
            |spill| {
                spill
                    .vectors
                    .as_ref()
                    .map_or(0, |vectors| vectors.dim() as u32)
            },
        );

        let result = if self.coalesced_rabitq && coalesced_dim > 0 && self.local_spill.is_some() {
            self.finish_coalesced_spill(coalesced_dim, capture_sq8)
        } else if self.coalesced_rabitq && coalesced_dim > 0 {
            self.finish_coalesced(coalesced_dim, capture_sq8)
        } else if self.local_spill.is_some() {
            bail!("local-spill PAX writer requires a non-empty coalesced vector column")
        } else {
            self.finish_legacy().map(|meta| PaxSegmentWrite {
                meta,
                cache_seed: None,
            })
        };
        self.emit_write_trace();
        result
    }

    /// TD-COMPACT-1 S2: emit ONE cumulative sub-phase summary line for this
    /// segment write. No-op unless `PROXIMADB_TRACE_PAX_WRITE` was set when the
    /// writer was constructed. Mirrored to stderr so it interleaves with the
    /// S1 `[PAX write]` phase timers emitted by the flush/compaction entry.
    fn emit_write_trace(&mut self) {
        let row_count = self.row_count;
        let file_len = self
            .local_spill
            .as_ref()
            .map_or(self.file_buf.len() as u64, |spill| spill.blocks_len)
            .min(usize::MAX as u64) as usize;
        let Some(tr) = self.write_trace.as_deref_mut() else {
            return;
        };
        let buffered = file_len + tr.rabitq_buf_bytes;
        if buffered > tr.peak_buffered_bytes {
            tr.peak_buffered_bytes = buffered;
        }
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let line = format!(
            "[PAX write detail] records={} blocks_cut={} rabitq={:.0} ms rerank={:.0} ms raw_buffer={:.0} ms block={:.0} ms cluster={:.0} ms other={:.0} ms peak_buffered_bytes={}",
            row_count,
            tr.blocks_cut,
            ms(tr.rabitq_encode),
            ms(tr.rerank_encode),
            ms(tr.raw_buffer),
            ms(tr.block_cut_compress),
            ms(tr.cluster_bookkeeping),
            ms(tr.other),
            tr.peak_buffered_bytes,
        );
        tracing::info!("{line}");
        eprintln!("{line}");
    }

    /// Bounded counterpart of `finish_coalesced`: vector reductions, Region-A/B
    /// encodes, Region-D blocks, and the OID resolver are all local files. The
    /// final PAX bytes and footer semantics remain identical.
    fn finish_coalesced_spill(
        &mut self,
        dim: u32,
        _capture_sq8: Option<bool>,
    ) -> Result<PaxSegmentWrite> {
        let mut spill = self
            .local_spill
            .take()
            .ok_or_else(|| anyhow::anyhow!("PAX local spill backing is missing"))?;
        // TD-PAXRG-1: Region C on the spill twin — the spool's raw-f32 file IS
        // the Region C payload, so the emit is a finish-time header + copy
        // (mirror of the memory twin), gated on row-group layout ∧ f32 tier.
        let emit_exact = self.rg_layout && self.f32_tier;
        spill.flush_inputs()?;
        if self.compute_oid_resolver && spill.oid_count as u64 != self.row_count {
            bail!(
                "PAX spill OID resolver rows {} != segment rows {}",
                spill.oid_count,
                self.row_count
            );
        }
        let vectors = spill
            .vectors
            .take()
            .ok_or_else(|| anyhow::anyhow!("PAX spill vector spool is missing"))?;
        let seed = RABITQ_SEED_BASE ^ (col_id::EMBED_BASE as u64);
        let regions = vectors.finish(seed, emit_exact)?;
        if regions.row_count as u64 != self.row_count || regions.dim != dim {
            bail!(
                "PAX spill vector rows/dim ({}/{}) != segment ({}/{dim})",
                regions.row_count,
                regions.dim,
                self.row_count
            );
        }
        let resolver = spill.finish_oid_resolver(self.compute_oid_resolver)?;

        let two_level = self.two_level.take();
        // TD-PAXRG-1 collapse: all writes use the single current header; the
        // row-group switch controls Region D framing rather than header shape.
        let (layout_version, header_len, a0_len) = match &two_level {
            Some(tl) => {
                tl.model.validate()?;
                if tl.model.dim != dim {
                    bail!(
                        "two-level model dim {} != segment embedding dim {dim}",
                        tl.model.dim
                    );
                }
                if tl.next_boundary != tl.boundaries.len()
                    || tl.cell_end_blocks.len() != tl.boundaries.len()
                {
                    bail!(
                        "two-level cell boundaries not all reached ({}/{} — plan rows {} vs fed rows {})",
                        tl.next_boundary,
                        tl.boundaries.len(),
                        tl.model.rows_covered(),
                        self.row_count
                    );
                }
                let length = CoarseDirectory::serialized_len(
                    tl.model.k_c(),
                    dim as usize,
                    tl.model.n_comp as usize,
                ) as u64;
                (SEG_LAYOUT_VERSION, SEG_HEADER_PREFIX_LEN as u64, length)
            }
            None => (SEG_LAYOUT_VERSION, SEG_HEADER_PREFIX_LEN as u64, 0),
        };
        let a0_off = if two_level.is_some() { header_len } else { 0 };
        let rabitq_off = header_len + a0_len;
        let rabitq_len = regions.rabitq_len;
        let sq8_off = rabitq_off + rabitq_len;
        let sq8_len = regions.sq8_len;
        // TD-PAXRG-1: Region C extent (zero-when-absent — twin invariant).
        let c_len = regions.exact().map_or(0u64, |(_, len)| len);
        let c_off = if c_len > 0 { sq8_off + sq8_len } else { 0 };
        let resolver_len = resolver.as_ref().map_or(0, |(_, length)| *length);
        let opr_off = sq8_off + sq8_len + c_len;
        let opr_len = resolver_len;
        let data_offset = opr_off + opr_len;

        let a0_bytes = match two_level {
            Some(tl) => {
                let n_rows = self.row_count as usize;
                let stride_a = (8 + (dim as usize).div_ceil(8)) as u64;
                let codes_base_a =
                    rabitq_off + rabitq_region_header_len(dim) as u64 + n_rows.div_ceil(8) as u64;
                let codes_base_b = sq8_off + sq8_codes_offset(n_rows) as u64;
                // TD-PAXRG-1: Region C per-cell extents (mirror of the memory
                // twin) so selective probing can address C the same way it
                // addresses A/B.
                let stride_c = dim as u64 * 4;
                let codes_base_c = c_off + exact_region_header_len(n_rows);
                let mut cells = Vec::with_capacity(tl.model.k_c());
                let mut row = 0u64;
                for (index, &rows) in tl.model.cell_rows.iter().enumerate() {
                    let d_block_begin = if index == 0 {
                        0
                    } else {
                        tl.cell_end_blocks[index - 1]
                    };
                    let (cell_c_off, cell_c_len) = if c_len > 0 {
                        (codes_base_c + row * stride_c, rows * stride_c)
                    } else {
                        (0, 0)
                    };
                    cells.push(CoarseCellEntry {
                        row_begin: row,
                        row_end: row + rows,
                        a_off: codes_base_a + row * stride_a,
                        a_len: rows * stride_a,
                        b_off: codes_base_b + row * dim as u64,
                        b_len: rows * dim as u64,
                        c_off: cell_c_off,
                        c_len: cell_c_len,
                        d_block_begin,
                        d_block_end: tl.cell_end_blocks[index],
                    });
                    row += rows;
                }
                if row > self.row_count {
                    bail!(
                        "two-level cell rows {row} exceed segment rows {}",
                        self.row_count
                    );
                }
                let bytes = CoarseDirectory {
                    model: tl.model,
                    cells,
                }
                .to_bytes()?;
                if bytes.len() as u64 != a0_len {
                    bail!(
                        "coarse directory serialized {} bytes != planned {a0_len}",
                        bytes.len()
                    );
                }
                Some(bytes)
            }
            None => None,
        };

        let blocks = self
            .index
            .blocks
            .iter()
            .map(|block| FooterBlockEntry {
                offset: data_offset + block.offset,
                size: block.size,
                row_count: block.zone.as_ref().map_or(0, |zone| zone.row_count),
                stats_kind: StatsKind::None,
            })
            .collect::<Vec<_>>();
        // TD-FPRUNE-1 P2: the filterable stats ride the sparse
        // SECTION_BLOCK_FILTERABLE_STATS footer section (NOT the per-entry
        // frame, which carries the typed row-group stats).
        let block_filterable_stats: Vec<(u32, Vec<u8>)> = self
            .index
            .blocks
            .iter()
            .enumerate()
            .filter_map(|(i, block)| block.footer_stats.clone().map(|p| (i as u32, p)))
            .collect();
        let (encoding_map, block_tier_assignments) = self.footer_encoding_map()?;
        // TD-PAXRG-1: per-RG stats ride the footer's opaque MinMax payload —
        // the zone summary for the DataFusion ranged prune + the OID-chunk
        // extent for the ANN top-k chunk fetch. Empty for v1/v3 (footers stay
        // byte-identical when the gate is off).
        let rg_block_stats: Vec<Option<FooterRowGroupStats>> = if self.rg_layout {
            self.index
                .blocks
                .iter()
                .zip(
                    self.rg_oid_chunks
                        .iter()
                        .chain(std::iter::repeat(&(0u32, 0u32, 0u8, false))),
                )
                .map(|(entry, &(chunk_off, chunk_len, chunk_enc, chunk_lz4))| {
                    entry.zone.as_ref().map(|zone| FooterRowGroupStats {
                        zone: zone.clone(),
                        oid_chunk_rel_off: chunk_off,
                        oid_chunk_len: chunk_len,
                        oid_encoding_id: chunk_enc,
                        oid_is_lz4: chunk_lz4,
                    })
                })
                .collect()
        } else {
            Vec::new()
        };
        let footer = SegmentFooterIndex {
            row_count: self.row_count,
            rabitq_off,
            rabitq_len,
            sq8_off,
            sq8_len,
            sq8_min: regions.sq8_params.offset,
            sq8_scale: regions.sq8_params.scale,
            embed_dim: dim,
            embed_count: self.embedding_count as u32,
            embed_quant_tag: 1,
            has_f32_tier: self.f32_tier,
            blocks,
            block_stats: rg_block_stats,
            block_filterable_stats,
            encoding_map,
            block_tier_assignments,
            a0_off,
            a0_len,
            opr_off,
            opr_len,
            c_off,
            c_len,
            // TD-FPRUNE-1 P2: self-describe the shredded user columns so a
            // filtered read resolves a user-tag field name → col-id with no
            // catalog lookup. Gated with the footer stats (useless without them).
            shred_field_map: if footer_stats_enabled() {
                // The SEGMENT footer's map is (name -> col_id) only; the exact
                // declared type lives in the per-BLOCK shred directory, which is
                // ungated and self-contained.
                self.shred_spec
                    .iter()
                    .map(|c| (c.key.clone(), c.column_id))
                    .collect()
            } else {
                Vec::new()
            },
        };
        let footer_body = footer.to_bytes()?;
        let footer_off = data_offset + spill.blocks_len;
        let footer_len = footer_body.len() as u64;
        let header = SegmentHeaderPrefix {
            layout_version,
            rabitq_off,
            rabitq_len,
            sq8_off,
            sq8_len,
            footer_off,
            footer_len,
            a0_off,
            a0_len,
            c_off,
            c_len,
        };
        let header_bytes = header.to_bytes();
        let mut tail = Vec::with_capacity(16);
        tail.extend_from_slice(&footer_len.to_le_bytes());
        tail.extend_from_slice(SEGMENT_MAGIC);

        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut output = BufWriter::new(std::fs::File::create(&self.path)?);
        let mut total = 0u64;
        total = write_counted(&mut output, &header_bytes, total)?;
        if let Some(a0) = a0_bytes.as_deref() {
            total = write_counted(&mut output, a0, total)?;
        }
        total = copy_counted(&mut output, regions.rabitq_path(), total)?;
        total = copy_counted(&mut output, regions.sq8_path(), total)?;
        if let Some((exact_path, _)) = regions.exact() {
            total = copy_counted(&mut output, exact_path, total)?;
        }
        if let Some((path, _)) = &resolver {
            total = copy_counted(&mut output, path, total)?;
        }
        total = copy_counted(&mut output, &spill.blocks_path, total)?;
        total = write_counted(&mut output, &footer_body, total)?;
        total = write_counted(&mut output, &tail, total)?;
        output.flush()?;
        output.get_ref().sync_data()?;

        let planned_total = footer_off
            .checked_add(footer_len)
            .and_then(|bytes| bytes.checked_add(tail.len() as u64))
            .ok_or_else(|| anyhow::anyhow!("PAX spill output length exceeds u64"))?;
        if total != planned_total {
            bail!("PAX spill assembled {total} bytes but layout planned {planned_total}");
        }
        let meta = SegmentMeta {
            path: self.path.clone(),
            size_bytes: total,
            block_count: self.index.blocks.len() as u32,
            row_count: self.row_count,
            block_stats: std::mem::take(&mut self.block_stats),
            block_centroids: std::mem::take(&mut self.block_centroids),
            block_radii: std::mem::take(&mut self.block_radii),
            rabitq_off,
            rabitq_len,
            sq8_off,
            sq8_len,
        };
        Ok(PaxSegmentWrite {
            meta,
            // Never re-materialize multi-gigabyte A/B merely for cache-on-write.
            cache_seed: None,
        })
    }

    /// Legacy `[blocks][SegmentIndex][SEGMENT_MAGIC]` layout (readability-preserving
    /// fallback for non-coalesced writes — the reader detects it via the `PBLK`
    /// head + `PAXSEG01` tail).
    fn finish_legacy(&mut self) -> Result<SegmentMeta> {
        // Append segment index
        let index_bytes = self.index.to_bytes();
        self.file_buf.extend_from_slice(&index_bytes);
        // Append magic
        self.file_buf.extend_from_slice(SEGMENT_MAGIC);

        let total_bytes = self.write_file(&self.file_buf)?;
        Ok(SegmentMeta {
            path: self.path.clone(),
            size_bytes: total_bytes,
            block_count: self.index.blocks.len() as u32,
            row_count: self.row_count,
            block_stats: std::mem::take(&mut self.block_stats),
            block_centroids: std::mem::take(&mut self.block_centroids),
            block_radii: std::mem::take(&mut self.block_radii),
            rabitq_off: 0,
            rabitq_len: 0,
            sq8_off: 0,
            sq8_len: 0,
        })
    }

    fn footer_encoding_map(
        &self,
    ) -> Result<(Vec<StripeEncodingDescriptor>, Vec<BlockTierAssignment>)> {
        if !self
            .block_transformed_sq8_columns
            .iter()
            .any(|columns| !columns.is_empty())
        {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut descriptors = Vec::new();
        let mut assignments = Vec::new();
        let mut next_id = 1u16;
        for embedding in 0..self.embedding_count {
            let logical_field_id = col_id::EMBED_BASE + embedding as i32;
            let physical_column_id = logical_field_id;
            // TD-PAXRG-1: under v4 the exact tier lives in Region C — no
            // in-block F32_TIER columns exist, so no block-local Exact
            // descriptor is emitted (has_f32_tier still advertises the tier).
            let in_block_exact = self.f32_tier && !self.rg_layout;
            let canonical_id = if in_block_exact {
                let id = take_descriptor_id(&mut next_id)?;
                let exact_column = col_id::F32_TIER_BASE + embedding as i32;
                descriptors.push(StripeEncodingDescriptor {
                    descriptor_id: id,
                    logical_field_id,
                    physical_column_id: exact_column,
                    tier_role: TierRole::Exact,
                    value_codec_tag: 0x01,
                    value_codec_version: 1,
                    transform_tag: LosslessTransformTag::None,
                    transform_version: 0,
                    compression_tag: LosslessCompressionTag::None,
                    compression_version: 0,
                    compression_flags: compression_flags::LOSSLESS,
                    parameter_scope: ParameterScope::Block,
                    vector_transform: VectorTransform::None,
                    auxiliary_flags: 0,
                    source_role: SourceRole::Canonical,
                    source_fidelity: SourceFidelity::ExactBitwise,
                    rebuild_source_id: 0,
                    projection_generation: 0,
                });
                for block_ordinal in 0..self.index.blocks.len() {
                    assignments.push(BlockTierAssignment {
                        block_ordinal: u32::try_from(block_ordinal)
                            .map_err(|_| anyhow::anyhow!("block ordinal exceeds u32"))?,
                        physical_column_id: exact_column,
                        tier_role: TierRole::Exact,
                        descriptor_id: id,
                    });
                }
                id
            } else {
                EXTERNAL_CANONICAL_SOURCE_ID
            };

            if embedding == 0 {
                descriptors.push(StripeEncodingDescriptor {
                    descriptor_id: take_descriptor_id(&mut next_id)?,
                    logical_field_id,
                    physical_column_id,
                    tier_role: TierRole::Index,
                    value_codec_tag: 0x71,
                    value_codec_version: 1,
                    transform_tag: LosslessTransformTag::None,
                    transform_version: 0,
                    compression_tag: LosslessCompressionTag::None,
                    compression_version: 0,
                    compression_flags: compression_flags::LOSSLESS,
                    parameter_scope: ParameterScope::Segment,
                    vector_transform: VectorTransform::CenteredRotated,
                    auxiliary_flags: 0,
                    source_role: SourceRole::IndexProjection,
                    source_fidelity: SourceFidelity::Lossy,
                    rebuild_source_id: canonical_id,
                    projection_generation: 1,
                });
            }

            let flat_id = take_descriptor_id(&mut next_id)?;
            descriptors.push(StripeEncodingDescriptor {
                descriptor_id: flat_id,
                logical_field_id,
                physical_column_id,
                tier_role: TierRole::Rerank,
                value_codec_tag: 0x05,
                value_codec_version: 1,
                transform_tag: LosslessTransformTag::None,
                transform_version: 0,
                compression_tag: LosslessCompressionTag::None,
                compression_version: 0,
                compression_flags: compression_flags::LOSSLESS,
                parameter_scope: ParameterScope::Block,
                vector_transform: VectorTransform::None,
                auxiliary_flags: 0,
                source_role: SourceRole::RerankProjection,
                source_fidelity: SourceFidelity::Lossy,
                rebuild_source_id: canonical_id,
                projection_generation: 1,
            });

            let has_transformed = self
                .block_transformed_sq8_columns
                .iter()
                .any(|columns| columns.contains(&physical_column_id));
            let transformed_id = if has_transformed {
                let id = take_descriptor_id(&mut next_id)?;
                descriptors.push(StripeEncodingDescriptor {
                    descriptor_id: id,
                    logical_field_id,
                    physical_column_id,
                    tier_role: TierRole::Rerank,
                    value_codec_tag: 0x05,
                    value_codec_version: 1,
                    transform_tag: LosslessTransformTag::ClusteredForBitpackU8,
                    transform_version: 1,
                    compression_tag: LosslessCompressionTag::None,
                    compression_version: 0,
                    compression_flags: compression_flags::LOSSLESS,
                    parameter_scope: ParameterScope::MicroChunk,
                    vector_transform: VectorTransform::None,
                    auxiliary_flags: 0,
                    source_role: SourceRole::RerankProjection,
                    source_fidelity: SourceFidelity::Lossy,
                    rebuild_source_id: canonical_id,
                    projection_generation: 1,
                });
                Some(id)
            } else {
                None
            };

            for (block_ordinal, transformed_columns) in
                self.block_transformed_sq8_columns.iter().enumerate()
            {
                let descriptor_id = if transformed_columns.contains(&physical_column_id) {
                    transformed_id.ok_or_else(|| {
                        anyhow::anyhow!("transformed block has no encoding descriptor")
                    })?
                } else {
                    flat_id
                };
                assignments.push(BlockTierAssignment {
                    block_ordinal: u32::try_from(block_ordinal)
                        .map_err(|_| anyhow::anyhow!("block ordinal exceeds u32"))?,
                    physical_column_id,
                    tier_role: TierRole::Rerank,
                    descriptor_id,
                });
            }
        }
        Ok((descriptors, assignments))
    }

    /// ADR-062 / TD-RDSTRAT-6 coalesced layout: `[HEADER-PREFIX][RaBitQ region]
    /// [blocks][FOOTER-INDEX][footer_len][SEGMENT_MAGIC]`. The RaBitQ region is
    /// one ranged GET (keep=100% scan); the footer-index block table carries
    /// absolute offsets so the read path maps survivors → blocks → coalesced GETs.
    ///
    /// TD-RDSTRAT-8 (v3, when [`Self::with_two_level`] armed): the same
    /// assembly with Region A0 (the coarse directory) between the prefix and
    /// Region A, `layout_version = 3`, and the A0 extent mirrored in the
    /// footer. Regions A/B/D are byte-identical in format — only the row
    /// order (coarse-cell-major) and the block padding differ.
    fn finish_coalesced(&mut self, dim: u32, capture_sq8: Option<bool>) -> Result<PaxSegmentWrite> {
        // 1. Build the coalesced RaBitQ region (single segment centroid) over the
        //    cluster-ordered embedding-0 vectors. `self.file_buf` already holds the
        //    blocks at 0-based offsets (relative to the blocks region).
        let refs = self.rabitq_vectors.refs();
        let seed = RABITQ_SEED_BASE ^ (col_id::EMBED_BASE as u64);
        // TD-COMPACT-1 S2: the finish-time Region A encode (fit + rotation +
        // per-vector RaBitQ) accumulates into the `rabitq` bucket, Region B
        // (segment-level SQ8) into `rerank`.
        let t = self.write_trace.is_some().then(Instant::now);
        let (region_bytes, _centroid) = encode_region(&refs, dim, seed)?;
        if let (Some(t), Some(tr)) = (t, self.write_trace.as_deref_mut()) {
            tr.rabitq_encode += t.elapsed();
        }
        // ADR-065 Region B: the SQ8 rerank tier, hoisted out of blocks so survivor
        // rerank fetches read pure dense SQ8 (one segment-level Sq8Params fit).
        let t = self.write_trace.is_some().then(Instant::now);
        let (sq8_region_bytes, sq8_params) = encode_sq8_region(&refs, dim)?;
        if let (Some(t), Some(tr)) = (t, self.write_trace.as_deref_mut()) {
            tr.rerank_encode += t.elapsed();
        }
        if let Some(tr) = self.write_trace.as_deref_mut() {
            let live_buffers = self
                .file_buf
                .capacity()
                .saturating_add(
                    self.rabitq_vectors
                        .values
                        .capacity()
                        .saturating_mul(std::mem::size_of::<f32>()),
                )
                .saturating_add(region_bytes.capacity())
                .saturating_add(sq8_region_bytes.capacity());
            tr.peak_buffered_bytes = tr.peak_buffered_bytes.max(live_buffers);
        }
        // TD-PAXRG-1 Region C (ADR-065's exact tier, hoisted): v4 ∧ f32_tier
        // emits the raw exact-f32 rows as their own region — built from the
        // SAME coalesced buffer as A/B before its drop (no extra pass), and the
        // block writers stop emitting F32_TIER stripes (fresh_block_writer).
        let exact_region_bytes = if self.rg_layout && self.f32_tier {
            Some(encode_exact_region(&refs, dim as usize)?)
        } else {
            None
        };
        // Both coalesced tiers are now encoded. Release the full f32 corpus
        // before footer assembly, file publication, and optional cache seeding.
        drop(refs);
        self.rabitq_vectors = CoalescedVectorBuffer::default();

        // TD-RDSTRAT-8: geometry first — A0's byte length is deterministic from
        // (k_c, dim, n_comp), so every downstream offset is known BEFORE A0's
        // contents (which embed absolute per-cell extents) are serialized. One
        // pass, no placeholder rewrites.
        let two_level = self.two_level.take();
        // TD-PAXRG-1 collapse: all writes use the single current header; the
        // row-group switch controls Region D framing rather than header shape.
        // Mirror of the spill twin — geometry assembly MUST stay identical.
        let (layout_version, header_len, a0_len) = match &two_level {
            Some(tl) => {
                tl.model.validate()?;
                if tl.model.dim != dim {
                    bail!(
                        "two-level model dim {} != segment embedding dim {dim}",
                        tl.model.dim
                    );
                }
                if tl.next_boundary != tl.boundaries.len()
                    || tl.cell_end_blocks.len() != tl.boundaries.len()
                {
                    bail!(
                        "two-level cell boundaries not all reached ({}/{} — plan rows {} vs fed rows {})",
                        tl.next_boundary,
                        tl.boundaries.len(),
                        tl.model.rows_covered(),
                        self.row_count
                    );
                }
                let a0_len = CoarseDirectory::serialized_len(
                    tl.model.k_c(),
                    dim as usize,
                    tl.model.n_comp as usize,
                ) as u64;
                (SEG_LAYOUT_VERSION, SEG_HEADER_PREFIX_LEN as u64, a0_len)
            }
            None => (SEG_LAYOUT_VERSION, SEG_HEADER_PREFIX_LEN as u64, 0),
        };
        let a0_off = if two_level.is_some() { header_len } else { 0 };
        let rabitq_off = header_len + a0_len;
        let rabitq_len = region_bytes.len() as u64;
        let sq8_off = rabitq_off + rabitq_len;
        let sq8_len = sq8_region_bytes.len() as u64;
        // TD-PAXRG-1 Region C sits between Region B and the resolver: its extent
        // is declared in the header-prefix AND mirrored as a footer section.
        // Rendered fields are ZERO-WHEN-ABSENT (twin-symmetry: the spill path
        // hardcodes 0/0); the LAYOUT offset chain below always uses the real
        // offset regardless.
        let c_len = exact_region_bytes.as_ref().map_or(0u64, |b| b.len() as u64);
        let c_off = if c_len > 0 { sq8_off + sq8_len } else { 0 };
        // Resolver region (TD-DELVEC-1 WI-2c): the OID→position resolver is an
        // immutable write-time component → a region *inside* the segment (atomic
        // with it on the single PUT), not a sidecar. It sits between Region B/C
        // and the blocks, so the block table's absolute offsets
        // (data_offset + b.offset below) account for its space. Absent
        // (opr_off/opr_len = 0) when `with_oid_resolver` is off or no oids were
        // captured — the segment is then byte-identical to the pre-WI-2c form.
        let opr_bytes = self.oid_resolver_bytes()?;
        // Blocks (Region D) begin after header [+ A0] + Region A + Region B
        // [+ C] [+ resolver].
        let opr_off = sq8_off + sq8_len + c_len;
        let opr_len = opr_bytes.as_ref().map_or(0u64, |b| b.len() as u64);
        let data_offset = opr_off + opr_len;

        // Serialize A0 (v3+). Per-cell extents are ABSOLUTE file offsets
        // into the fixed-stride code payloads of Regions A/B (the writer owns
        // the layout, so the reader never derives offsets at query time) plus
        // the padded Region D block ranges. TD-PAXRG-1: Region C per-cell
        // extents are populated when the exact tier is hoisted, so selective
        // probing can address C the same way it addresses A/B.
        let a0_bytes: Option<Vec<u8>> = match two_level {
            Some(tl) => {
                let n_rows = self.row_count as usize;
                let stride_a = (8 + (dim as usize).div_ceil(8)) as u64;
                let codes_base_a =
                    rabitq_off + rabitq_region_header_len(dim) as u64 + n_rows.div_ceil(8) as u64;
                let codes_base_b = sq8_off + sq8_codes_offset(n_rows) as u64;
                let exact_present = exact_region_bytes.is_some();
                let stride_c = dim as u64 * 4;
                let codes_base_c = c_off + exact_region_header_len(n_rows);
                let mut cells = Vec::with_capacity(tl.model.k_c());
                let mut row = 0u64;
                for (i, &rows) in tl.model.cell_rows.iter().enumerate() {
                    let d_block_begin = if i == 0 { 0 } else { tl.cell_end_blocks[i - 1] };
                    let (cell_c_off, cell_c_len) = if exact_present {
                        (codes_base_c + row * stride_c, rows * stride_c)
                    } else {
                        (0, 0)
                    };
                    cells.push(CoarseCellEntry {
                        row_begin: row,
                        row_end: row + rows,
                        a_off: codes_base_a + row * stride_a,
                        a_len: rows * stride_a,
                        b_off: codes_base_b + row * dim as u64,
                        b_len: rows * dim as u64,
                        c_off: cell_c_off,
                        c_len: cell_c_len,
                        d_block_begin,
                        d_block_end: tl.cell_end_blocks[i],
                    });
                    row += rows;
                }
                if row > self.row_count {
                    bail!(
                        "two-level cell rows {row} exceed segment rows {}",
                        self.row_count
                    );
                }
                let bytes = CoarseDirectory {
                    model: tl.model,
                    cells,
                }
                .to_bytes()?;
                if bytes.len() as u64 != a0_len {
                    bail!(
                        "coarse directory serialized {} bytes != planned {a0_len}",
                        bytes.len()
                    );
                }
                Some(bytes)
            }
            None => None,
        };

        // 2. Footer block table: absolute offsets = data_offset + block's 0-based
        //    offset; row_count from the block's zone summary (1:1 with the index).
        let blocks: Vec<FooterBlockEntry> = self
            .index
            .blocks
            .iter()
            .map(|b| FooterBlockEntry {
                offset: data_offset + b.offset,
                size: b.size,
                row_count: b.zone.as_ref().map(|z| z.row_count).unwrap_or(0),
                stats_kind: StatsKind::None,
            })
            .collect();
        // TD-FPRUNE-1 P2: filterable stats as the sparse footer section (spill
        // twin — identical geometry assembly is a pinned invariant).
        let block_filterable_stats: Vec<(u32, Vec<u8>)> = self
            .index
            .blocks
            .iter()
            .enumerate()
            .filter_map(|(i, b)| b.footer_stats.clone().map(|p| (i as u32, p)))
            .collect();

        // 3. Footer-index body + header-prefix offsets. The footer sits after the
        //    blocks; its length is known once serialized.
        let (encoding_map, block_tier_assignments) = self.footer_encoding_map()?;
        // TD-PAXRG-1: per-RG stats — mirror of the spill twin (identical
        // geometry assembly is a pinned invariant).
        let rg_block_stats: Vec<Option<FooterRowGroupStats>> = if self.rg_layout {
            self.index
                .blocks
                .iter()
                .zip(
                    self.rg_oid_chunks
                        .iter()
                        .chain(std::iter::repeat(&(0u32, 0u32, 0u8, false))),
                )
                .map(|(entry, &(chunk_off, chunk_len, chunk_enc, chunk_lz4))| {
                    entry.zone.as_ref().map(|zone| FooterRowGroupStats {
                        zone: zone.clone(),
                        oid_chunk_rel_off: chunk_off,
                        oid_chunk_len: chunk_len,
                        oid_encoding_id: chunk_enc,
                        oid_is_lz4: chunk_lz4,
                    })
                })
                .collect()
        } else {
            Vec::new()
        };
        let footer = SegmentFooterIndex {
            row_count: self.row_count,
            rabitq_off,
            rabitq_len,
            sq8_off,
            sq8_len,
            // Cache-co-design: mirror the SQ8 dequant key (min + scale) into the
            // footer so the read path decodes survivors without a separate 24 B
            // Region-B-header GET. (offset == vmin == min; vmax recoverable.)
            sq8_min: sq8_params.offset,
            sq8_scale: sq8_params.scale,
            embed_dim: dim,
            embed_count: self.embedding_count as u32,
            embed_quant_tag: 1, // SQ8 rerank tier — now Region B (hoisted out of blocks)
            has_f32_tier: self.f32_tier,
            blocks,
            block_stats: rg_block_stats,
            block_filterable_stats,
            encoding_map,
            block_tier_assignments,
            a0_off,
            a0_len,
            opr_off,
            opr_len,
            c_off,
            c_len,
            // TD-FPRUNE-1 P2: self-describe shredded user columns (see the other
            // footer builder). Gated with the footer stats.
            shred_field_map: if footer_stats_enabled() {
                // The SEGMENT footer's map is (name -> col_id) only; the exact
                // declared type lives in the per-BLOCK shred directory, which is
                // ungated and self-contained.
                self.shred_spec
                    .iter()
                    .map(|c| (c.key.clone(), c.column_id))
                    .collect()
            } else {
                Vec::new()
            },
        };
        let footer_body = footer.to_bytes()?;
        let footer_off = data_offset + self.file_buf.len() as u64;
        let footer_len = footer_body.len() as u64;

        let header = SegmentHeaderPrefix {
            layout_version,
            rabitq_off,
            rabitq_len,
            sq8_off,
            sq8_len,
            footer_off,
            footer_len,
            a0_off,
            a0_len,
            c_off,
            c_len,
        };

        // 4. Stream the already-ordered regions into the local staging file.
        // The former assembly allocated a second whole-segment `Vec` and copied
        // Region D into it while `file_buf` remained live. Sequential writes are
        // byte-identical and remove that output-size peak from compaction RSS.
        let header_bytes = header.to_bytes();
        let mut trailer = Vec::with_capacity(16);
        trailer.extend_from_slice(&(footer_body.len() as u64).to_le_bytes());
        trailer.extend_from_slice(SEGMENT_MAGIC);
        let total_bytes = self.write_coalesced_file(
            &header_bytes,
            a0_bytes.as_deref(),
            &region_bytes,
            &sq8_region_bytes,
            exact_region_bytes.as_deref(),
            opr_bytes.as_deref(),
            &footer_body,
            &trailer,
        )?;
        // Region D has reached the local staging file and is no longer needed.
        // Releasing it here keeps cache-on-write Arc construction from
        // overlapping a segment-sized compressed block buffer.
        self.file_buf = Vec::new();
        let cache_seed = capture_sq8.map(|include_sq8| {
            let rabitq_header_len = rabitq_region_header_len(dim).min(region_bytes.len());
            PaxCacheSeed {
                header_bytes,
                a0_bytes: a0_bytes.map(std::sync::Arc::from),
                rabitq_header_bytes: std::sync::Arc::from(&region_bytes[..rabitq_header_len]),
                rabitq_bytes: std::sync::Arc::from(region_bytes),
                footer_bytes: footer_body,
                sq8_off,
                sq8_bytes: include_sq8.then(|| std::sync::Arc::from(sq8_region_bytes)),
            }
        });
        let meta = SegmentMeta {
            path: self.path.clone(),
            size_bytes: total_bytes,
            block_count: self.index.blocks.len() as u32,
            row_count: self.row_count,
            block_stats: std::mem::take(&mut self.block_stats),
            block_centroids: std::mem::take(&mut self.block_centroids),
            block_radii: std::mem::take(&mut self.block_radii),
            rabitq_off,
            rabitq_len,
            sq8_off,
            sq8_len,
        };
        Ok(PaxSegmentWrite { meta, cache_seed })
    }

    /// Create the parent dir + write `buf` to `self.path`; returns bytes written.
    fn write_file(&self, buf: &[u8]) -> Result<u64> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut f = std::fs::File::create(&self.path)?;
        f.write_all(buf)?;
        Ok(buf.len() as u64)
    }

    #[allow(clippy::too_many_arguments)]
    fn write_coalesced_file(
        &self,
        header: &[u8],
        a0: Option<&[u8]>,
        rabitq: &[u8],
        sq8: &[u8],
        exact_region: Option<&[u8]>,
        oid_resolver: Option<&[u8]>,
        footer: &[u8],
        tail: &[u8],
    ) -> Result<u64> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::File::create(&self.path)?;
        let mut total = 0u64;
        for part in [
            Some(header),
            a0,
            Some(rabitq),
            Some(sq8),
            exact_region,
            oid_resolver,
            Some(&self.file_buf),
            Some(footer),
            Some(tail),
        ]
        .into_iter()
        .flatten()
        {
            file.write_all(part)?;
            total = total
                .checked_add(part.len() as u64)
                .ok_or_else(|| anyhow::anyhow!("PAX segment length exceeds u64"))?;
        }
        Ok(total)
    }

    /// Test-only: finish writing a **legacy v1** segment (segment index without
    /// inline zone-map summaries). Production always writes v2 ([`Self::finish`]);
    /// this fixture exercises the mixed-read-safe legacy path — a v1 segment on
    /// disk must still read correctly and prune via per-block footer reads.
    #[cfg(test)]
    pub fn finish_v1(mut self) -> Result<SegmentMeta> {
        self.flush_current_block()?;
        if self.index.blocks.is_empty() {
            bail!("segment is empty — nothing to write");
        }
        let index_bytes = self.index.to_bytes_v1();
        self.file_buf.extend_from_slice(&index_bytes);
        self.file_buf.extend_from_slice(SEGMENT_MAGIC);
        let total_bytes = self.file_buf.len() as u64;
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut f = std::fs::File::create(&self.path)?;
        f.write_all(&self.file_buf)?;
        Ok(SegmentMeta {
            path: self.path,
            size_bytes: total_bytes,
            block_count: self.index.blocks.len() as u32,
            row_count: self.row_count,
            block_stats: self.block_stats,
            block_centroids: self.block_centroids,
            block_radii: self.block_radii,
            rabitq_off: 0,
            rabitq_len: 0,
            sq8_off: 0,
            sq8_len: 0,
        })
    }
}

fn write_counted(output: &mut BufWriter<std::fs::File>, bytes: &[u8], total: u64) -> Result<u64> {
    output.write_all(bytes)?;
    total
        .checked_add(bytes.len() as u64)
        .ok_or_else(|| anyhow::anyhow!("PAX spill output length exceeds u64"))
}

fn copy_counted(output: &mut BufWriter<std::fs::File>, path: &Path, total: u64) -> Result<u64> {
    let mut input = BufReader::new(std::fs::File::open(path)?);
    let copied = std::io::copy(&mut input, output)?;
    total
        .checked_add(copied)
        .ok_or_else(|| anyhow::anyhow!("PAX spill output length exceeds u64"))
}

fn take_descriptor_id(next_id: &mut u16) -> Result<u16> {
    if *next_id == EXTERNAL_CANONICAL_SOURCE_ID {
        bail!("footer encoding descriptor id space exhausted");
    }
    let id = *next_id;
    *next_id = next_id
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("footer encoding descriptor id overflow"))?;
    Ok(id)
}

// ── Scanner ───────────────────────────────────────────────────────────────────

/// Predicate context for block-level pruning during segment scans.
#[derive(Debug, Clone, Default)]
pub struct ScanPredicate {
    /// If set, skip blocks whose tenant hash doesn't match.
    pub tenant_hash: Option<u64>,
    /// If set, skip blocks with no overlap with `[from_ns, to_ns]`.
    pub time_range: Option<(i64, i64)>,
}

impl ScanPredicate {
    pub fn for_tenant(tenant_id: &str) -> Self {
        Self {
            tenant_hash: Some(fnv1a_hash(tenant_id)),
            ..Default::default()
        }
    }

    pub fn for_time_range(from_ns: i64, to_ns: i64) -> Self {
        Self {
            time_range: Some((from_ns, to_ns)),
            ..Default::default()
        }
    }

    pub fn with_tenant(mut self, tenant_id: &str) -> Self {
        self.tenant_hash = Some(fnv1a_hash(tenant_id));
        self
    }

    pub fn with_time_range(mut self, from_ns: i64, to_ns: i64) -> Self {
        self.time_range = Some((from_ns, to_ns));
        self
    }
}

/// Iterator over raw PAX blocks in a segment file with block-level pruning.
///
/// Reads the segment index from the file tail, then yields the byte slice of
/// each block that passes the predicate. Callers decode individual stripes via
/// `PaxBlockReader`.
pub struct PaxSegmentScanner {
    data: Vec<u8>,
    index: SegmentIndex,
    predicate: ScanPredicate,
    cursor: usize,
}

impl PaxSegmentScanner {
    /// Open a segment file and parse its index.
    pub fn open(path: impl AsRef<Path>, predicate: ScanPredicate) -> Result<Self> {
        let data = std::fs::read(path.as_ref())?;
        Self::from_bytes(data, predicate)
    }

    /// Parse from an in-memory byte slice (useful for WAL replay / testing).
    ///
    /// Mixed-read-safe (ADR-061 amendment): a coalesced-RaBitQ segment
    /// (`SEG_HEADER_MAGIC` head) is parsed from its self-describing footer-index
    /// (the block table there carries absolute offsets); a legacy segment
    /// (`PBLK` head) keeps using `SegmentIndex::locate`. Both feed `next_block`
    /// the same `SegmentIndex` shape.
    pub fn from_bytes(data: Vec<u8>, predicate: ScanPredicate) -> Result<Self> {
        // Validate magic (both layouts tail with SEGMENT_MAGIC).
        if data.len() < 8 || &data[data.len() - 8..] != SEGMENT_MAGIC {
            bail!("not a valid PAX segment file (bad magic)");
        }
        let index = if is_coalesced_segment(&data) {
            Self::parse_coalesced_index(&data)?
        } else {
            let magic_start = data.len() - 8;
            Self::parse_index(&data[..magic_start])?
        };

        Ok(Self {
            data,
            index,
            predicate,
            cursor: 0,
        })
    }

    fn parse_index(before_magic: &[u8]) -> Result<SegmentIndex> {
        SegmentIndex::locate(before_magic)
    }

    /// Build the block list for a coalesced-RaBitQ segment from its footer-index.
    /// The footer block table carries absolute offsets + row counts; we mirror
    /// them into a `SegmentIndex` (with a row-count-only zone summary) so the
    /// shared `next_block` / `read_records` paths work unchanged.
    fn parse_coalesced_index(data: &[u8]) -> Result<SegmentIndex> {
        let footer = SegmentFooterIndex::locate_in_segment(data)?
            .ok_or_else(|| anyhow::anyhow!("coalesced segment missing footer-index"))?;
        let blocks = footer
            .blocks
            .iter()
            .map(|b| BlockIndexEntry {
                offset: b.offset,
                size: b.size,
                zone: Some(BlockZoneSummary::empty(b.row_count)),
                footer_stats: None,
            })
            .collect();
        Ok(SegmentIndex { blocks })
    }

    /// Yield the next block that passes predicate pruning.
    pub fn next_block(&mut self) -> Option<PaxBlockReader<'_>> {
        while self.cursor < self.index.blocks.len() {
            let entry = &self.index.blocks[self.cursor];
            self.cursor += 1;

            let start = entry.offset as usize;
            let end = start + entry.size as usize;
            if end > self.data.len() {
                continue; // corrupted index entry — skip
            }

            let block_data = &self.data[start..end];
            let reader = match PaxBlockReader::open(block_data) {
                Ok(r) => r,
                Err(_) => continue,
            };

            // Block-level predicate pruning
            if let Some(th) = self.predicate.tenant_hash
                && !reader.tenant_matches(th)
            {
                continue;
            }
            if let Some((from, to)) = self.predicate.time_range
                && !reader.time_overlaps(from, to)
            {
                continue;
            }

            // SAFETY: we extend the lifetime here because `self.data` owns the
            // bytes and outlives the returned reader. The borrow checker cannot
            // see through the index indirection, so we use a raw pointer.
            // INVARIANT: `block_data` is a sub-slice of `self.data`; as long as
            // `self.data` is not modified (which PaxSegmentScanner does not do
            // after construction), this is sound.
            let reader: PaxBlockReader<'_> = unsafe {
                let ptr = block_data.as_ptr();
                let len = block_data.len();
                let static_slice: &'static [u8] = std::slice::from_raw_parts(ptr, len);
                PaxBlockReader::open(static_slice).ok()?
            };

            return Some(reader);
        }
        None
    }

    pub fn block_count(&self) -> usize {
        self.index.blocks.len()
    }

    /// Reconstruct every row of the segment into full [`ProximaRecord`]s — the
    /// canonical read-side inverse of `PaxSegmentWriter::add_record`. Iterates
    /// the (predicate-pruned) blocks, rebuilds each row via
    /// [`FlatRow::from_block_reader`], and materializes records through
    /// [`FlatRow::into_record`].
    ///
    /// `embedding_model_ids` / `user_column_keys` come from the collection schema
    /// (the segment stores embeddings positionally and does not persist model ids
    /// or promoted-column names). Pass empty slices for best-effort defaults
    /// (`model_0`, `model_1`, …).
    /// `tenant_ctx` is the segment's owning tenant (from the catalog/path); it is
    /// stamped onto rows whose tenant column was dropped (catalog-resolution) and
    /// ignored when the column is still present. Pass `None` to keep stored values.
    pub fn read_records(
        &mut self,
        embedding_model_ids: &[String],
        user_column_keys: &[String],
        tenant_ctx: Option<&str>,
    ) -> Result<Vec<ProximaRecord>> {
        let mut records = Vec::new();
        while let Some(block) = self.next_block() {
            for flat in FlatRow::from_block_reader(&block)? {
                records.push(flat.into_record(
                    embedding_model_ids,
                    user_column_keys,
                    tenant_ctx,
                )?);
            }
        }
        Ok(records)
    }
}

// ── Mixed-format PAX read (the Pax branch of the segment read router) ──────────

/// Decode a PAX vector segment (the columnar format) back to records — the Pax
/// arm of the root-level mixed-format read router (`read_segment_records`).
///
/// Pure PAX: no format detection and no legacy path (the router has already
/// dispatched here via
/// [`crate::segment_layout::SegmentFormat::detect`]). Records are reconstructed
/// via the canonical inverse ([`PaxSegmentScanner::read_records`]); for a
/// coalesced segment with an SQ8 Region B (ADR-065), SQ8 rerank vectors fill only
/// rows that lack an authoritative block-local f32 embedding. Block row order ==
/// Region B cluster order. Factored out of the root crate so it is unit-testable
/// without linking the root; byte-for-byte identical to the former in-root path.
pub fn read_pax_segment_records(
    bytes: &[u8],
    embedding_model_ids: &[String],
    user_column_keys: &[String],
    tenant_ctx: Option<&str>,
) -> Result<Vec<ProximaRecord>> {
    let mut scanner = PaxSegmentScanner::from_bytes(bytes.to_vec(), ScanPredicate::default())?;
    let mut recs = scanner.read_records(embedding_model_ids, user_column_keys, tenant_ctx)?;
    // TD-PAXRG-1 Region C: v4 segments carry the exact tier as a hoisted
    // region — fill from it FIRST (exact authority) when present.
    if is_coalesced_segment(bytes)
        && let Ok(h) = SegmentHeaderPrefix::parse(bytes)
        && h.c_len > 0
    {
        let region = &bytes[h.c_off as usize..(h.c_off + h.c_len) as usize];
        if let Ok(exact) = ExactRegion::from_bytes(region) {
            for (i, rec) in recs.iter_mut().enumerate() {
                if rec.embeddings.is_empty()
                    && let Some(v) = exact.decode_row(i)
                {
                    rec.embeddings.push(proximadb_records::EmbeddingCell {
                        modality: "dense".into(),
                        dim: exact.dim,
                        values: proximadb_records::EmbeddingValues::Fp32(v),
                        ..Default::default()
                    });
                }
            }
        }
    }
    // ADR-065 Region B: blocks without an exact tier are pure row data, so fill
    // their missing vector from SQ8 by row index. When an exact tier is present,
    // `FlatRow::from_block_reader` has already materialized it and remains the
    // canonical value.
    if is_coalesced_segment(bytes)
        && let Ok(h) = SegmentHeaderPrefix::parse(bytes)
        && h.sq8_len > 0
    {
        let region = &bytes[h.sq8_off as usize..(h.sq8_off + h.sq8_len) as usize];
        if let Ok(sq8) = proximadb_block_format::coalesced_sq8::Sq8Region::from_bytes(region) {
            let dim = sq8.header.dim;
            for (i, rec) in recs.iter_mut().enumerate() {
                if rec.embeddings.is_empty()
                    && let Some(v) = sq8.decode_row(i)
                {
                    rec.embeddings.push(proximadb_records::EmbeddingCell {
                        modality: "dense".into(),
                        dim,
                        values: proximadb_records::EmbeddingValues::Fp32(v),
                        ..Default::default()
                    });
                }
            }
        }
    }
    Ok(recs)
}

// ── Compaction (TD-114) ─────────────────────────────────────────────────────────

/// Statistics from a PAX segment compaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionStats {
    /// Number of input segment files merged.
    pub inputs: usize,
    /// Total records read across all inputs.
    pub records_in: u64,
    /// Records written to the output (survivors, after dropping tombstones).
    pub records_out: u64,
    /// Records dropped because they were tombstones at `now_ns` (merge-on-read deletes).
    pub tombstones_dropped: u64,
}

/// Merge several L0 PAX segments into one L1 segment, dropping records that are
/// tombstones as of `now_ns` (merge-on-read deletes; TD-114).
///
/// `embedding_model_ids` and `user_column_keys` are the collection's schema keys
/// used to reconstruct records (see [`PaxSegmentScanner::read_records`]). Inputs are
/// read in order; surviving records are written to `output` via a fresh
/// [`PaxSegmentWriter`]. This is a pure, engine-agnostic primitive: the caller owns
/// L0 discovery (the input paths) and manifest registration of the output.
#[allow(clippy::too_many_arguments)]
pub fn compact_pax_segments(
    inputs: &[PathBuf],
    output: &Path,
    mode: BlockMode,
    compression: BlockCompression,
    collection_id: &str,
    schema_fingerprint: u64,
    embedding_count: usize,
    embedding_model_ids: &[String],
    user_column_keys: &[String],
    tenant_ctx: Option<&str>,
    now_ns: i64,
) -> Result<CompactionStats> {
    let preserve_exact = !inputs.is_empty()
        && inputs.iter().all(|input| {
            let Ok(mut scanner) = PaxSegmentScanner::open(input, ScanPredicate::default()) else {
                return false;
            };
            let mut saw_block = false;
            while let Some(block) = scanner.next_block() {
                saw_block = true;
                if !block.has_exact_vector_authority() {
                    return false;
                }
            }
            saw_block
        });
    let mut writer = PaxSegmentWriter::new(
        output,
        mode,
        compression,
        collection_id,
        schema_fingerprint,
        embedding_count,
        None,
    )
    .with_f32_tier(preserve_exact);
    let mut records_in = 0u64;
    let mut records_out = 0u64;
    let mut tombstones_dropped = 0u64;

    for input in inputs {
        let mut scanner = PaxSegmentScanner::open(input, ScanPredicate::default())?;
        for record in scanner.read_records(embedding_model_ids, user_column_keys, tenant_ctx)? {
            records_in += 1;
            if record.is_tombstone_at(now_ns) {
                tombstones_dropped += 1;
                continue;
            }
            writer.add_record(&record)?;
            records_out += 1;
        }
    }
    writer.finish()?;

    Ok(CompactionStats {
        inputs: inputs.len(),
        records_in,
        records_out,
        tombstones_dropped,
    })
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_records::ProximaRecord;

    #[test]
    fn coalesced_vector_buffer_is_contiguous_and_preserves_null_rows() -> Result<()> {
        let mut buffer = CoalescedVectorBuffer::default();
        buffer.push(None)?;
        buffer.push(Some(&[1.0, 2.0, 3.0]))?;
        buffer.push(None)?;
        buffer.push(Some(&[4.0, 5.0, 6.0]))?;

        assert_eq!(buffer.dim(), 3);
        assert_eq!(buffer.values.len(), 12);
        assert_eq!(
            buffer.refs(),
            vec![
                None,
                Some(&[1.0, 2.0, 3.0][..]),
                None,
                Some(&[4.0, 5.0, 6.0][..])
            ]
        );
        assert!(buffer.push(Some(&[7.0, 8.0])).is_err());
        Ok(())
    }

    fn make_record(oid: &str, tenant: &str, ts: i64) -> ProximaRecord {
        ProximaRecord {
            oid: oid.into(),
            tenant_id: tenant.into(),
            created_at_ns: ts,
            updated_at_ns: ts,
            ..Default::default()
        }
    }

    #[test]
    fn local_spill_writer_is_byte_identical_and_reclaims_scratch() -> Result<()> {
        use proximadb_records::{EmbeddingCell, EmbeddingValues};

        const DIM: usize = 8;
        let output = tempfile::tempdir()?;
        let scratch = tempfile::tempdir()?;
        let memory_path = output.path().join("memory.pax");
        let spill_path = output.path().join("spill.pax");
        let model = CoarseModel {
            dim: DIM as u32,
            n_comp: 1,
            pca_mean: vec![0.0; DIM],
            pca_components: vec![0.1; DIM],
            centroids: vec![-1.0, 1.0],
            radii: vec![1.0, 1.0],
            cell_rows: vec![8, 8],
            seed: 7,
            trained_on: 16,
        };
        let writer = |path: &Path| {
            PaxSegmentWriter::new(
                path,
                BlockMode::Pax,
                BlockCompression::Lz4,
                "collection",
                0,
                1,
                Some(512),
            )
            .with_quant(VectorQuant::RaBitQ)
            .with_rerank_quant(VectorQuant::Sq8)
            .with_coalesced_rabitq(true)
            .with_oid_resolver(true)
            .with_two_level(model.clone())
        };
        let mut memory = writer(&memory_path);
        let mut spilled = writer(&spill_path).with_local_spill(scratch.path())?;
        for row in 0..16usize {
            let mut record = make_record(&format!("oid-{row:03}"), "tenant", row as i64);
            record.embeddings.push(EmbeddingCell {
                model_id: "model_0".to_string(),
                modality: "dense".to_string(),
                dim: DIM as u32,
                values: EmbeddingValues::Fp32(
                    (0..DIM)
                        .map(|dimension| (row * 13 + dimension * 5) as f32 * 0.01)
                        .collect(),
                ),
                ..EmbeddingCell::default()
            });
            memory.add_record(&record)?;
            spilled.add_record(&record)?;
        }
        memory.finish()?;
        let spill_write = spilled.finish_with_cache_seed(true)?;

        assert!(spill_write.cache_seed.is_none());
        assert_eq!(std::fs::read(&spill_path)?, std::fs::read(&memory_path)?);
        let segment = std::fs::read(&spill_path)?;
        let footer = SegmentFooterIndex::locate_in_segment(&segment)?
            .ok_or_else(|| anyhow::anyhow!("spill output footer missing"))?;
        let resolver = crate::oid_position_resolver::OidPositionResolver::deserialize(
            &segment[footer.opr_off as usize..(footer.opr_off + footer.opr_len) as usize],
        )?;
        assert_eq!(resolver.position_of("oid-000"), Some(0));
        assert_eq!(resolver.position_of("oid-015"), Some(15));
        assert_eq!(std::fs::read_dir(scratch.path())?.count(), 0);
        Ok(())
    }

    #[test]
    fn cache_seed_is_byte_identical_to_published_regions() -> Result<()> {
        use proximadb_records::{EmbeddingCell, EmbeddingValues};

        const DIM: usize = 8;
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("cache-seed.pax");
        let model = CoarseModel {
            dim: DIM as u32,
            n_comp: 1,
            pca_mean: vec![0.0; DIM],
            pca_components: vec![0.1; DIM],
            centroids: vec![-1.0, 1.0],
            radii: vec![1.0, 1.0],
            cell_rows: vec![2, 2],
            seed: 7,
            trained_on: 4,
        };
        let mut writer = PaxSegmentWriter::new(
            &path,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            1,
            None,
        )
        .with_quant(VectorQuant::RaBitQ)
        .with_coalesced_rabitq(true)
        .with_two_level(model);
        for row in 0..4usize {
            let mut record = make_record(&format!("r{row}"), "t", row as i64);
            record.embeddings.push(EmbeddingCell {
                modality: "dense".into(),
                dim: DIM as u32,
                values: EmbeddingValues::Fp32(vec![row as f32; DIM]),
                ..Default::default()
            });
            writer.add_record(&record)?;
        }

        let write = writer.finish_with_cache_seed(true)?;
        let seed = write
            .cache_seed
            .ok_or_else(|| anyhow::anyhow!("capturing writer must return a seed"))?;
        let segment = std::fs::read(&path)?;
        let header = SegmentHeaderPrefix::parse(&segment)?;

        assert_eq!(
            seed.header_bytes.as_slice(),
            &segment[..seed.header_bytes.len()]
        );
        assert_eq!(
            seed.a0_bytes.as_deref(),
            Some(&segment[header.a0_off as usize..(header.a0_off + header.a0_len) as usize])
        );
        assert_eq!(
            seed.rabitq_bytes.as_ref(),
            &segment[header.rabitq_off as usize..(header.rabitq_off + header.rabitq_len) as usize]
        );
        assert_eq!(
            seed.rabitq_header_bytes.as_ref(),
            &segment[header.rabitq_off as usize
                ..header.rabitq_off as usize + seed.rabitq_header_bytes.len()]
        );
        assert_eq!(seed.sq8_off, header.sq8_off);
        assert_eq!(
            seed.sq8_bytes.as_deref(),
            Some(&segment[header.sq8_off as usize..(header.sq8_off + header.sq8_len) as usize])
        );
        assert_eq!(
            seed.footer_bytes.as_slice(),
            &segment[header.footer_off as usize..(header.footer_off + header.footer_len) as usize]
        );
        Ok(())
    }

    #[test]
    fn coalesced_footer_does_not_advertise_hoisted_sq8_as_block_transform() -> Result<()> {
        use proximadb_records::{EmbeddingCell, EmbeddingValues};

        const ROWS_PER_CLUSTER: usize = 256;
        const DIM: usize = 32;
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("clustered-footer.pax");
        let mut writer = PaxSegmentWriter::new(
            &path,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            1,
            Some(ROWS_PER_CLUSTER * 2 * 1024),
        )
        .with_quant(VectorQuant::RaBitQ)
        .with_coalesced_rabitq(true)
        .with_lossless_clustered(true);

        for row in 0..(ROWS_PER_CLUSTER * 2) {
            if row == ROWS_PER_CLUSTER {
                writer.start_cluster_run();
            }
            let center = if row < ROWS_PER_CLUSTER { -10.0 } else { 10.0 };
            let values: Vec<f32> = (0..DIM)
                .map(|lane| center + ((row + lane) % 3) as f32 * 0.01)
                .collect();
            let mut record = make_record(&format!("r{row}"), "t", 1_000 + row as i64);
            record.embeddings.push(EmbeddingCell {
                modality: "dense".into(),
                dim: DIM as u32,
                values: EmbeddingValues::Fp32(values),
                ..Default::default()
            });
            writer.add_record(&record)?;
        }
        writer.finish()?;

        let segment = std::fs::read(&path)?;
        let footer = SegmentFooterIndex::locate_in_segment(&segment)?
            .ok_or_else(|| anyhow::anyhow!("coalesced footer missing"))?;
        assert!(
            footer.encoding_map.iter().all(|descriptor| {
                descriptor.physical_column_id != col_id::EMBED_BASE
                    || descriptor.transform_tag != LosslessTransformTag::ClusteredForBitpackU8
            }),
            "coalesced SQ8 lives in Region B, so Region D must not advertise a block transform"
        );
        assert!(
            footer
                .block_tier_assignments
                .iter()
                .all(|assignment| assignment.physical_column_id != col_id::EMBED_BASE),
            "hoisted Region B has no per-block SQ8 assignment"
        );
        Ok(())
    }

    /// TD-DELVEC-1 WI-2c: a coalesced segment written with `with_oid_resolver(true)`
    /// embeds the OID→position resolver as a **footer region** — locatable via
    /// `SegmentFooterIndex::locate_in_segment`, fetchable by `(opr_off, opr_len)`,
    /// and round-tripping through `OidPositionResolver::deserialize` with the oids
    /// in add_record (stream) order. With the opt-in OFF, no resolver region is
    /// emitted (`opr_len == 0`) — the segment is byte-for-byte the pre-WI-2c form.
    #[test]
    fn coalesced_segment_embeds_oid_resolver_footer_region() -> Result<()> {
        use proximadb_records::{EmbeddingCell, EmbeddingValues};

        const DIM: usize = 32;
        let oids: Vec<String> = (0..8).map(|i| format!("oid-{i}")).collect();

        let write_segment = |path: &std::path::Path, resolver: bool| -> Result<()> {
            let mut writer = PaxSegmentWriter::new(
                path,
                BlockMode::Pax,
                BlockCompression::None,
                "col",
                0,
                1,
                Some(8 * 2 * 1024),
            )
            .with_quant(VectorQuant::RaBitQ)
            .with_coalesced_rabitq(true);
            if resolver {
                writer = writer.with_oid_resolver(true);
            }
            for oid in &oids {
                let mut record = make_record(oid, "t", 1);
                record.embeddings.push(EmbeddingCell {
                    modality: "dense".into(),
                    dim: DIM as u32,
                    values: EmbeddingValues::Fp32(vec![0.5; DIM]),
                    ..Default::default()
                });
                writer.add_record(&record)?;
            }
            writer.finish()?;
            Ok(())
        };

        let dir = tempfile::tempdir()?;

        // Opt-in ON → resolver region present, locatable, and round-trips with the
        // oids in add_record (stream) order.
        let on_path = dir.path().join("resolver-on.pax");
        write_segment(&on_path, true)?;
        let segment = std::fs::read(&on_path)?;
        let footer = SegmentFooterIndex::locate_in_segment(&segment)?
            .ok_or_else(|| anyhow::anyhow!("coalesced footer missing"))?;
        assert!(
            footer.opr_len > 0,
            "resolver region must be present when opted in"
        );
        let region = &segment[footer.opr_off as usize..(footer.opr_off + footer.opr_len) as usize];
        let resolver = crate::oid_position_resolver::OidPositionResolver::deserialize(region)?;
        assert_eq!(resolver.len(), oids.len() as u32);
        for (i, oid) in oids.iter().enumerate() {
            assert_eq!(
                resolver.position_of(oid),
                Some(i as u32),
                "oid must map to its add_record (stream) position"
            );
        }

        // Opt-in OFF → no resolver region (opr_off/opr_len == 0); segment parses.
        let off_path = dir.path().join("resolver-off.pax");
        write_segment(&off_path, false)?;
        let footer_off = SegmentFooterIndex::locate_in_segment(&std::fs::read(&off_path)?)?
            .ok_or_else(|| anyhow::anyhow!("coalesced footer missing"))?;
        assert_eq!(footer_off.opr_off, 0);
        assert_eq!(
            footer_off.opr_len, 0,
            "no resolver region when the opt-in is off"
        );
        Ok(())
    }

    // ── TD-PAXRG-1 Phase B: v4 row-group Region D writer ─────────────────────

    /// Shared RG fixture: `rows` tiny records across two created_at windows so
    /// per-RG zone summaries carry real bounds.
    fn write_rg_segment(
        path: &std::path::Path,
        rows: usize,
        resolver: bool,
        spill: Option<&std::path::Path>,
    ) -> Result<()> {
        write_rg_segment_f32(path, rows, resolver, false, spill)
    }

    /// RG fixture with an explicit f32-tier switch (Region C hoist coverage).
    fn write_rg_segment_f32(
        path: &std::path::Path,
        rows: usize,
        resolver: bool,
        f32_tier: bool,
        spill: Option<&std::path::Path>,
    ) -> Result<()> {
        use proximadb_records::{EmbeddingCell, EmbeddingValues};
        const DIM: usize = 32;
        let mut writer = PaxSegmentWriter::new(
            path,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            1,
            Some(RG_TARGET_MIN_BYTES),
        )
        .with_quant(VectorQuant::RaBitQ)
        .with_coalesced_rabitq(true)
        .with_f32_tier(f32_tier)
        .with_rg_layout(true);
        if resolver {
            writer = writer.with_oid_resolver(true);
        }
        if let Some(scratch) = spill {
            writer = writer.with_local_spill(scratch)?;
        }
        for row in 0..rows {
            let mut record = make_record(&format!("oid-{row}"), "t", 1_000 + row as i64);
            record.embeddings.push(EmbeddingCell {
                modality: "dense".into(),
                dim: DIM as u32,
                values: EmbeddingValues::Fp32(vec![0.25 + row as f32 * 0.001; DIM]),
                ..Default::default()
            });
            writer.add_record(&record)?;
        }
        writer.finish()?;
        Ok(())
    }

    /// TD-PAXRG-1 Phase B: the RG writer emits the v4 header (88 B, Region C
    /// extent present-but-absent) and a footer whose block table IS the row-group
    /// table — per-entry MinMax stats carrying the RG zone summary + an OID-chunk
    /// extent, with row counts summing to the segment total (ordinal alignment
    /// with Regions A/B is the cumulative-count invariant).
    #[test]
    fn rg_layout_writer_emits_v4_footer_and_ordinal_aligned_regions() -> Result<()> {
        const ROWS: usize = 512;
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("rg-v4.pax");
        write_rg_segment(&path, ROWS, true, None)?;

        let segment = std::fs::read(&path)?;
        let header = SegmentHeaderPrefix::parse(&segment[..SEG_HEADER_PREFIX_LEN])?;
        assert_eq!(
            header.layout_version, SEG_LAYOUT_VERSION,
            "rg_layout writes the v4 layout"
        );
        assert_eq!(header.c_len, 0, "Region C not emitted in Phase B");
        let footer = SegmentFooterIndex::locate_in_segment(&segment)?
            .ok_or_else(|| anyhow::anyhow!("coalesced footer missing"))?;
        assert!(!footer.blocks.is_empty(), "at least one row group");
        assert_eq!(
            footer.block_stats.len(),
            footer.blocks.len(),
            "stats are index-aligned with the RG table"
        );
        let mut cumulative = 0u64;
        for (entry, stats) in footer.blocks.iter().zip(&footer.block_stats) {
            let stats = stats
                .as_ref()
                .expect("every v4 RG carries a MinMax stats payload");
            assert_eq!(stats.zone.row_count, entry.row_count);
            assert!(
                stats.zone.present != 0,
                "zone summary carries real created_at bounds"
            );
            assert!(
                stats.oid_chunk_len > 0,
                "OID chunk extent captured for the top-k fetch"
            );
            cumulative += entry.row_count as u64;
        }
        assert_eq!(
            cumulative, footer.row_count,
            "RG row counts are ordinal-aligned with Regions A/B"
        );
        assert_eq!(cumulative, ROWS as u64);
        Ok(())
    }

    /// TD-PAXRG-1 fail-safe: without the OID resolver there is no point-lookup
    /// authority once the RowDirectory is suppressed — the finish must bail.
    /// With the resolver, RGs decode via the ordinary PaxBlockReader, carry no
    /// row directory, and decode their stripes.
    #[test]
    fn rg_writer_row_directory_absent_and_resolver_auto_enabled() -> Result<()> {
        use proximadb_block_format::PaxBlockReader;

        let dir = tempfile::tempdir()?;

        // Post-flip invariant: rg AUTO-ENABLES the OID resolver (the layout's
        // point-lookup authority) — no explicit opt-in, no bail path.
        let path = dir.path().join("rg-resolver.pax");
        write_rg_segment(&path, 8, false, None)?;
        let segment = std::fs::read(&path)?;
        let footer = SegmentFooterIndex::locate_in_segment(&segment)?
            .ok_or_else(|| anyhow::anyhow!("coalesced footer missing"))?;
        let first = footer.blocks.first().expect("at least one RG");
        let rg_bytes = &segment[first.offset as usize..(first.offset + first.size as u64) as usize];
        let reader = PaxBlockReader::open(rg_bytes)?;
        assert_eq!(
            reader.header().block_mode,
            proximadb_block_format::BlockMode::Olap,
            "v4 RGs are Olap-framed"
        );
        assert!(
            reader.row_directory()?.is_none(),
            "RowDirectory suppressed in v4 RGs"
        );
        let oids = reader
            .decode_str_stripe(col_id::OID)
            .expect("OID stripe decodes");
        assert_eq!(oids.len() as u64, first.row_count as u64);
        // Auto-enabled resolver region present even without an explicit opt-in.
        assert!(
            footer.opr_len > 0,
            "rg_layout must auto-enable the OID resolver (point-lookup authority)"
        );
        Ok(())
    }

    /// TD-PAXRG-1 Phase B (the 400 B regression): the RG floor clamps the
    /// pathological target so fixed framing stays amortized — RG count stays
    /// bounded and every non-final RG is at least ~78% of the floor payload
    /// (fixed cost ≤ ~1%).
    #[test]
    fn rg_target_floor_clamps_pathological_block_size() -> Result<()> {
        use proximadb_records::{EmbeddingCell, EmbeddingValues};
        const ROWS: usize = 2_000;
        const DIM: usize = 32;
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("rg-floor.pax");
        // Deliberately pathological 400-byte target — the pre-v4 amplifier.
        let mut writer = PaxSegmentWriter::new(
            &path,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            1,
            Some(400),
        )
        .with_quant(VectorQuant::RaBitQ)
        .with_coalesced_rabitq(true)
        .with_rg_layout(true)
        .with_oid_resolver(true);
        let _ = &mut writer;
        for row in 0..ROWS {
            let mut record = make_record(&format!("oid-{row}"), "t", 1_000 + row as i64);
            record.embeddings.push(EmbeddingCell {
                modality: "dense".into(),
                dim: DIM as u32,
                values: EmbeddingValues::Fp32(vec![0.5; DIM]),
                ..Default::default()
            });
            writer.add_record(&record)?;
        }
        writer.finish()?;

        let segment = std::fs::read(&path)?;
        let footer = SegmentFooterIndex::locate_in_segment(&segment)?
            .ok_or_else(|| anyhow::anyhow!("coalesced footer missing"))?;
        assert!(
            footer.blocks.len() <= ROWS / 100,
            "floor must prevent micro-RGs: {} RGs for {ROWS} rows",
            footer.blocks.len()
        );
        // The invariant that matters: framing (~1.2 KB/RG) stays a rounding
        // error. The byte-floor clamps the size trigger, but the row-cut also
        // fires on REAL accumulated metadata bytes, which can cut a non-final
        // RG below the floor — as long as it stays ≥ 64 KiB, fixed cost is
        // ≤ ~2% and the micro-RG pathology (~800 B RGs at 400 B targets) is
        // structurally impossible.
        for entry in footer
            .blocks
            .iter()
            .take(footer.blocks.len().saturating_sub(1))
        {
            assert!(
                entry.size as usize >= 64 * 1024,
                "non-final RG {} bytes must amortize framing (fixed ~1.2KB)",
                entry.size
            );
        }
        Ok(())
    }

    /// TD-PAXRG-1 Phase B: the spill twin and the in-memory twin produce
    /// IDENTICAL geometry for the same ordered input — footer RG table, stats
    /// payloads, and region extents must agree byte-for-byte where they are
    /// offsets-independent (block offsets share `data_offset` in both).
    #[test]
    fn spill_and_memory_rg_writers_produce_identical_footers() -> Result<()> {
        const ROWS: usize = 256;
        let dir = tempfile::tempdir()?;
        let mem_path = dir.path().join("rg-mem.pax");
        write_rg_segment(&mem_path, ROWS, true, None)?;

        let scratch = dir.path().join("scratch");
        std::fs::create_dir_all(&scratch)?;
        let spill_path = dir.path().join("rg-spill.pax");
        write_rg_segment(&spill_path, ROWS, true, Some(&scratch))?;

        let footer_of = |p: &std::path::Path| -> Result<SegmentFooterIndex> {
            SegmentFooterIndex::locate_in_segment(&std::fs::read(p)?)?
                .ok_or_else(|| anyhow::anyhow!("coalesced footer missing"))
        };
        let mem = footer_of(&mem_path)?;
        let spill = footer_of(&spill_path)?;
        assert_eq!(
            mem.blocks, spill.blocks,
            "RG tables must be identical (offsets share data_offset)"
        );
        assert_eq!(
            mem.block_stats, spill.block_stats,
            "RG stats payloads must be identical"
        );
        assert_eq!(
            (mem.rabitq_off, mem.rabitq_len, mem.sq8_off, mem.sq8_len),
            (
                spill.rabitq_off,
                spill.rabitq_len,
                spill.sq8_off,
                spill.sq8_len
            ),
            "region extents must be identical"
        );
        Ok(())
    }

    /// TD-PAXRG-1 Phase B (mixed-read guarantee): with the gate OFF the writer
    /// emits NO v4 artifacts — layout version stays v1/v3 and the footer's
    /// stats column is empty (zero-length payloads, byte-identical footers).
    #[test]
    fn rg_layout_gate_default_off_writes_v1_v3_bytes() -> Result<()> {
        use proximadb_records::{EmbeddingCell, EmbeddingValues};
        const DIM: usize = 32;
        unsafe { std::env::set_var("PROXIMADB_PAX_WRITE_RG_LAYOUT", "0") };
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("rg-off.pax");
        let mut writer = PaxSegmentWriter::new(
            &path,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            1,
            Some(64 * 1024),
        )
        .with_quant(VectorQuant::RaBitQ)
        .with_coalesced_rabitq(true)
        .with_oid_resolver(true);
        for row in 0..64 {
            let mut record = make_record(&format!("oid-{row}"), "t", 1_000 + row as i64);
            record.embeddings.push(EmbeddingCell {
                modality: "dense".into(),
                dim: DIM as u32,
                values: EmbeddingValues::Fp32(vec![0.5; DIM]),
                ..Default::default()
            });
            writer.add_record(&record)?;
        }
        writer.finish()?;

        let segment = std::fs::read(&path)?;
        // Post-flip the invariant is about REGION ARTIFACTS under the
        // kill-switch: no RG stats payloads, no Region C.
        let footer = SegmentFooterIndex::locate_in_segment(&segment)?
            .ok_or_else(|| anyhow::anyhow!("coalesced footer missing"))?;
        assert!(
            footer.block_stats.iter().all(|s| s.is_none()),
            "kill-switch footers carry no RG stats payloads"
        );
        assert_eq!(footer.c_len, 0, "kill-switch emits no Region C");
        Ok(())
    }

    // ── TD-PAXRG-1 Phase C: Region C (exact tier) hoist ──────────────────────

    /// TD-PAXRG-1 Phase C: v4 ∧ f32_tier hoists the exact tier into Region C —
    /// the header + footer declare its extent, and the RG blocks carry NO
    /// in-block F32_TIER stripes anymore.
    #[test]
    fn v4_f32_tier_writes_region_c_not_block_stripes() -> Result<()> {
        use proximadb_block_format::PaxBlockReader;
        use proximadb_block_format::col_id::F32_TIER_BASE;

        let dir = tempfile::tempdir()?;
        let path = dir.path().join("rg-c.pax");
        write_rg_segment_f32(&path, 256, true, true, None)?;

        let segment = std::fs::read(&path)?;
        let header = SegmentHeaderPrefix::parse(&segment[..SEG_HEADER_PREFIX_LEN])?;
        assert_eq!(header.layout_version, SEG_LAYOUT_VERSION);
        assert!(
            header.c_len > 0,
            "Region C must be present under v4 ∧ f32_tier"
        );

        let footer = SegmentFooterIndex::locate_in_segment(&segment)?
            .ok_or_else(|| anyhow::anyhow!("coalesced footer missing"))?;
        assert_eq!(
            footer.c_off, header.c_off,
            "footer mirrors the header extent"
        );
        assert_eq!(footer.c_len, header.c_len);
        assert!(
            footer.has_f32_tier,
            "exact-tier capability stays advertised"
        );

        // The RG itself carries no F32_TIER stripe.
        let first = footer.blocks.first().expect("at least one RG");
        let rg_bytes = &segment[first.offset as usize..(first.offset + first.size as u64) as usize];
        let reader = PaxBlockReader::open(rg_bytes)?;
        assert!(
            reader
                .column_metas()
                .iter()
                .all(|m| m.column_id < F32_TIER_BASE),
            "v4 RGs must not carry in-block exact-tier stripes"
        );
        Ok(())
    }

    /// TD-PAXRG-1 Phase C (parity oracle): the SAME records reconstructed
    /// through (a) the v3 layout with in-block f32 stripes and (b) the v4
    /// layout with Region C must yield BITWISE-identical embedding values —
    /// the hoist moves bytes, never fidelity.
    #[test]
    fn region_c_reconstruction_matches_block_f32_tier_reconstruction() -> Result<()> {
        use proximadb_records::{EmbeddingCell, EmbeddingValues};
        const DIM: usize = 32;
        const ROWS: usize = 64;
        let dir = tempfile::tempdir()?;

        let build = |path: &std::path::Path, rg: bool| -> Result<()> {
            let mut writer = PaxSegmentWriter::new(
                path,
                BlockMode::Pax,
                BlockCompression::None,
                "col",
                0,
                1,
                Some(RG_TARGET_MIN_BYTES),
            )
            .with_quant(VectorQuant::RaBitQ)
            .with_coalesced_rabitq(true)
            .with_f32_tier(true)
            .with_oid_resolver(true);
            if rg {
                writer = writer.with_rg_layout(true);
            }
            for row in 0..ROWS {
                let mut record = make_record(&format!("oid-{row}"), "t", 1_000 + row as i64);
                record.embeddings.push(EmbeddingCell {
                    modality: "dense".into(),
                    dim: DIM as u32,
                    values: EmbeddingValues::Fp32(vec![0.25 + row as f32 * 0.001; DIM]),
                    ..Default::default()
                });
                writer.add_record(&record)?;
            }
            writer.finish()?;
            Ok(())
        };

        let v3_path = dir.path().join("parity-v3.pax");
        build(&v3_path, false)?;
        let v4_path = dir.path().join("parity-v4.pax");
        build(&v4_path, true)?;

        let v3_recs =
            crate::pax_block::read_pax_segment_records(&std::fs::read(&v3_path)?, &[], &[], None)?;
        let v4_recs =
            crate::pax_block::read_pax_segment_records(&std::fs::read(&v4_path)?, &[], &[], None)?;
        assert_eq!(v3_recs.len(), ROWS);
        assert_eq!(v4_recs.len(), ROWS);
        for (v3, v4) in v3_recs.iter().zip(&v4_recs) {
            let a = v3.embeddings.first().expect("v3 embedding");
            let b = v4.embeddings.first().expect("v4 embedding");
            let (
                proximadb_records::EmbeddingValues::Fp32(a),
                proximadb_records::EmbeddingValues::Fp32(b),
            ) = (&a.values, &b.values)
            else {
                panic!("both sides must decode to Fp32");
            };
            assert_eq!(a, b, "Region C reconstruction must be bitwise-identical");
        }
        Ok(())
    }

    /// TD-RDSTRAT-5 S1: with `with_block_centroids(true)`, the segment writer
    /// returns one centroid per block = the exact mean of that block's embedding-0
    /// f32 vectors. This fixture cuts an explicit block boundary so its
    /// two-block shape is independent of the encoding-aware byte estimator.
    #[test]
    fn block_centroids_are_exact_per_block_means() {
        use proximadb_records::{EmbeddingCell, EmbeddingValues};
        let rec = |oid: &str, v: Vec<f32>| {
            let dim = v.len() as u32;
            let mut r = ProximaRecord {
                oid: oid.into(),
                tenant_id: "t".into(),
                created_at_ns: 1,
                updated_at_ns: 1,
                ..Default::default()
            };
            r.embeddings.push(EmbeddingCell {
                modality: "dense".into(),
                dim,
                values: EmbeddingValues::Fp32(v),
                ..Default::default()
            });
            r
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seg.pax");
        let mut w = PaxSegmentWriter::new(
            &path,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            1,
            Some(2 * 1024), // 2 rows/block
        )
        .with_quant(VectorQuant::RawF32)
        .with_block_centroids(true);
        for (row, r) in [
            rec("a", vec![0.0, 0.0]),
            rec("b", vec![2.0, 2.0]),
            rec("c", vec![10.0, 10.0]),
            rec("d", vec![12.0, 12.0]),
        ]
        .into_iter()
        .enumerate()
        {
            if row == 2 {
                w.flush_current_block().unwrap();
            }
            w.add_record(&r).unwrap();
        }
        let meta = w.finish().unwrap();
        assert_eq!(meta.block_count, 2, "2 rows/block ⇒ 2 blocks");
        assert_eq!(
            meta.block_centroids.len(),
            meta.block_count as usize,
            "one centroid per block"
        );
        assert_eq!(
            meta.block_centroids[0],
            vec![1.0, 1.0],
            "mean of [0,0],[2,2]"
        );
        assert_eq!(
            meta.block_centroids[1],
            vec![11.0, 11.0],
            "mean of [10,10],[12,12]"
        );
    }

    /// TD-DELVEC-1 WI-2b: with `with_oid_resolver(true)`, the writer captures
    /// each record's canonical oid in add_record (stream) order and exposes a
    /// CRC32'd `ORP1` resolver whose positions match that order. Default OFF ⇒
    /// `oid_resolver_bytes()` is `Ok(None)` (the default flush path is unchanged).
    /// The footer-block-order property (capture order == on-disk scan order) is
    /// verified by code read (§9.2) + a WI-2c scan integration test.
    #[test]
    fn oid_resolver_captures_stream_order_and_is_off_by_default() {
        use proximadb_records::{EmbeddingCell, EmbeddingValues};
        let rec = |oid: &str| {
            let mut r = ProximaRecord {
                oid: oid.into(),
                tenant_id: "t".into(),
                created_at_ns: 1,
                updated_at_ns: 1,
                ..Default::default()
            };
            r.embeddings.push(EmbeddingCell {
                modality: "dense".into(),
                dim: 1,
                values: EmbeddingValues::Fp32(vec![0.0]),
                ..Default::default()
            });
            r
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seg.pax");

        // Default OFF: no resolver captured.
        let mut w0 = PaxSegmentWriter::new(
            &path,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            1,
            None,
        )
        .with_quant(VectorQuant::RawF32);
        w0.add_record(&rec("x")).unwrap();
        assert!(
            w0.oid_resolver_bytes().unwrap().is_none(),
            "default off ⇒ no resolver bytes"
        );

        // Opt in: capture oids in add_record order, round-trip through ORP1+CRC.
        let mut w = PaxSegmentWriter::new(
            &path,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            1,
            None,
        )
        .with_quant(VectorQuant::RawF32)
        .with_oid_resolver(true);
        for oid in ["a", "b", "", "d"] {
            w.add_record(&rec(oid)).unwrap();
        }
        let bytes = w
            .oid_resolver_bytes()
            .unwrap()
            .expect("captured when enabled");
        let resolver =
            crate::oid_position_resolver::OidPositionResolver::deserialize(&bytes).unwrap();
        assert_eq!(resolver.len(), 4);
        assert_eq!(resolver.position_of("a"), Some(0));
        assert_eq!(resolver.position_of("b"), Some(1));
        assert_eq!(
            resolver.position_of(""),
            Some(2),
            "an empty oid is kept at its position (append-only rows are non-addressable, harmless)"
        );
        assert_eq!(resolver.position_of("d"), Some(3));
        assert_eq!(resolver.oid_at(1), Some("b"));
    }

    /// TD-WLP-3 (TD-RDSTRAT-5 lever-3): the writer emits one RMS radius per
    /// block, 1:1 with `block_centroids`, computed in ONE pass as
    /// `radius² = mean(‖x‖²) − ‖centroid‖²`. Block [0,0],[2,2]: centroid
    /// [1,1], mean‖x‖² = (0+8)/2 = 4, ‖c‖² = 2 ⇒ radius = √2 (both blocks by
    /// symmetry). A block of identical rows must clamp to exactly 0.0.
    #[test]
    fn block_radii_are_exact_rms_spread() {
        use proximadb_records::{EmbeddingCell, EmbeddingValues};
        let rec = |oid: &str, v: Vec<f32>| {
            let dim = v.len() as u32;
            let mut r = ProximaRecord {
                oid: oid.into(),
                tenant_id: "t".into(),
                created_at_ns: 1,
                updated_at_ns: 1,
                ..Default::default()
            };
            r.embeddings.push(EmbeddingCell {
                modality: "dense".into(),
                dim,
                values: EmbeddingValues::Fp32(v),
                ..Default::default()
            });
            r
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seg_radii.pax");
        let mut w = PaxSegmentWriter::new(
            &path,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            1,
            Some(2 * 1024), // 2 rows/block
        )
        .with_quant(VectorQuant::RawF32)
        .with_block_centroids(true);
        for (row, r) in [
            rec("a", vec![0.0, 0.0]),
            rec("b", vec![2.0, 2.0]),
            rec("c", vec![7.0, 7.0]), // identical pair: zero spread
            rec("d", vec![7.0, 7.0]),
        ]
        .into_iter()
        .enumerate()
        {
            if row == 2 {
                w.flush_current_block().unwrap();
            }
            w.add_record(&r).unwrap();
        }
        let meta = w.finish().unwrap();
        assert_eq!(meta.block_count, 2, "2 rows/block ⇒ 2 blocks");
        assert_eq!(
            meta.block_radii.len(),
            meta.block_count as usize,
            "one radius per block, 1:1 with block_centroids"
        );
        let expected = 2f32.sqrt();
        assert!(
            (meta.block_radii[0] - expected).abs() < 1e-6,
            "RMS spread of [0,0],[2,2] must be √2, got {}",
            meta.block_radii[0]
        );
        assert_eq!(
            meta.block_radii[1], 0.0,
            "a block of identical rows must clamp to exactly 0.0"
        );

        // Not opted in ⇒ no radii (parity with block_centroids).
        let path2 = dir.path().join("seg_no_radii.pax");
        let mut w2 = PaxSegmentWriter::new(
            &path2,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            1,
            Some(2 * 1024),
        )
        .with_quant(VectorQuant::RawF32);
        w2.add_record(&rec("a", vec![0.0, 0.0])).unwrap();
        w2.add_record(&rec("b", vec![2.0, 2.0])).unwrap();
        let meta2 = w2.finish().unwrap();
        assert!(
            meta2.block_radii.is_empty(),
            "radii must be empty unless centroids are opted in"
        );
    }

    /// TD-RDSTRAT-5 S1: exact per-block means at a realistic **16-dim** width
    /// (2-byte sign-key territory) — 4 rows, 2 rows/block ⇒ 2 blocks, each
    /// centroid the elementwise mean of its block's 16-dim vectors.
    #[test]
    fn block_centroids_16dim_exact_means() {
        use proximadb_records::{EmbeddingCell, EmbeddingValues};
        let dim = 16usize;
        let rec = |oid: &str, fill: f32| {
            let mut r = ProximaRecord {
                oid: oid.into(),
                tenant_id: "t".into(),
                created_at_ns: 1,
                updated_at_ns: 1,
                ..Default::default()
            };
            r.embeddings.push(EmbeddingCell {
                modality: "dense".into(),
                dim: dim as u32,
                values: EmbeddingValues::Fp32(vec![fill; dim]),
                ..Default::default()
            });
            r
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seg.pax");
        let mut w = PaxSegmentWriter::new(
            &path,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            1,
            Some(2 * 1024), // 2 rows/block
        )
        .with_quant(VectorQuant::RawF32)
        .with_block_centroids(true);
        // block 0: fills 0 & 4 → mean 2; block 1: fills 10 & 20 → mean 15.
        for (row, r) in [rec("a", 0.0), rec("b", 4.0), rec("c", 10.0), rec("d", 20.0)]
            .into_iter()
            .enumerate()
        {
            if row == 2 {
                w.flush_current_block().unwrap();
            }
            w.add_record(&r).unwrap();
        }
        let meta = w.finish().unwrap();
        assert_eq!(meta.block_count, 2);
        assert_eq!(meta.block_centroids.len(), 2);
        assert_eq!(meta.block_centroids[0], vec![2.0f32; dim]);
        assert_eq!(meta.block_centroids[1], vec![15.0f32; dim]);
    }

    /// Default off: a writer built WITHOUT `with_block_centroids` returns no
    /// centroids (zero cost for pre-existing callers).
    #[test]
    fn block_centroids_empty_when_not_opted_in() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seg.pax");
        use proximadb_records::{EmbeddingCell, EmbeddingValues};
        let mut r = make_record("a", "t", 1);
        r.embeddings.push(EmbeddingCell {
            modality: "dense".into(),
            dim: 2,
            values: EmbeddingValues::Fp32(vec![1.0, 2.0]),
            ..Default::default()
        });
        let mut w = PaxSegmentWriter::new(
            &path,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            1,
            None,
        )
        .with_quant(VectorQuant::RawF32);
        w.add_record(&r).unwrap();
        let meta = w.finish().unwrap();
        assert!(meta.block_centroids.is_empty());
    }

    #[test]
    fn segment_index_round_trip_is_always_v2_even_without_zones() {
        // Zone-less blocks: writes are now ALWAYS v2 (PAXZ trailer) so pruning
        // needs zero per-block metadata GETs; the blocks get empty zone summaries.
        let idx = SegmentIndex {
            blocks: vec![
                BlockIndexEntry {
                    offset: 0,
                    size: 4096,
                    zone: None,
                    footer_stats: None,
                },
                BlockIndexEntry {
                    offset: 4096,
                    size: 8192,
                    zone: None,
                    footer_stats: None,
                },
            ],
        };
        let bytes = idx.to_bytes();
        assert_eq!(
            &bytes[bytes.len() - 4..],
            ZONE_INDEX_MARKER,
            "writes are always v2 (PAXZ trailer), even with no zones"
        );
        // `locate` is the production read path — it dispatches v1/v2 via the
        // self-describing trailer (the v1 `from_bytes` would mis-read v2 bytes).
        let idx2 = SegmentIndex::locate(&bytes).unwrap();
        assert_eq!(idx2.blocks.len(), 2);
        assert_eq!(idx2.blocks[0].size, 4096);
        assert_eq!(idx2.blocks[1].offset, 4096);
        // Every block has a zone (empty) ⇒ the reader never falls back to a
        // per-block footer read on a v2 segment.
        assert!(idx2.blocks.iter().all(|b| b.zone.is_some()));
    }

    #[test]
    fn segment_index_v1_legacy_still_reads() {
        // Mixed-read-safety: v1 segments already on disk (no PAXZ trailer) must
        // still read correctly even though writes are now v2-only.
        let idx = SegmentIndex {
            blocks: vec![
                BlockIndexEntry {
                    offset: 0,
                    size: 4096,
                    zone: None,
                    footer_stats: None,
                },
                BlockIndexEntry {
                    offset: 4096,
                    size: 8192,
                    zone: None,
                    footer_stats: None,
                },
            ],
        };
        let v1 = idx.to_bytes_v1(); // legacy fixture (write-deprecated)
        assert_ne!(
            &v1[v1.len() - 4..],
            ZONE_INDEX_MARKER,
            "v1 fixture has no PAXZ trailer"
        );
        let read = SegmentIndex::from_bytes(&v1).unwrap();
        assert_eq!(read.blocks.len(), 2);
        assert_eq!(read.blocks[1].offset, 4096);
        // v1 has no zones ⇒ reader prunes via the per-block footer fallback.
        assert!(read.blocks.iter().all(|b| b.zone.is_none()));
    }

    #[test]
    fn segment_index_v2_zonemap_round_trips_and_detects_via_locate() {
        let mut zone = BlockZoneSummary::empty(10);
        zone.created_at = (1000, 2000);
        zone.valid_to = (0, 5000);
        zone.present = ZONE_CREATED_AT | ZONE_VALID_TO;
        let idx = SegmentIndex {
            blocks: vec![
                BlockIndexEntry {
                    offset: 0,
                    size: 4096,
                    zone: Some(zone.clone()),
                    footer_stats: None,
                },
                BlockIndexEntry {
                    offset: 4096,
                    size: 8192,
                    zone: Some(BlockZoneSummary::empty(3)),
                    footer_stats: None,
                },
            ],
        };
        // to_bytes picks v2 (a zone is present) and appends the PAXZ trailer.
        let mut bytes = idx.to_bytes();
        assert_eq!(
            &bytes[bytes.len() - 4..],
            ZONE_INDEX_MARKER,
            "v2 trailer marker"
        );
        // The reader path strips SEGMENT_MAGIC, then locate() detects v2 by PAXZ.
        bytes.extend_from_slice(SEGMENT_MAGIC);
        let located = SegmentIndex::locate(&bytes[..bytes.len() - 8]).unwrap();
        assert_eq!(located.blocks.len(), 2);
        let z0 = located.blocks[0].zone.as_ref().expect("v2 zone preserved");
        assert_eq!(z0.created_at, (1000, 2000));
        assert_eq!(z0.valid_to, (0, 5000));
        assert_eq!(z0.present, ZONE_CREATED_AT | ZONE_VALID_TO);
        assert_eq!(located.blocks[1].size, 8192);
    }

    #[test]
    fn compact_drops_tombstones_and_merges() {
        let dir = tempfile::tempdir().unwrap();
        let seg0 = dir.path().join("L0_0.pax");
        let seg1 = dir.path().join("L0_1.pax");
        let out = dir.path().join("L1.pax");

        // L0 segment 0: two live records.
        let mut w0 = PaxSegmentWriter::new(
            &seg0,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            0,
            Some(1),
        );
        w0.add_record(&make_record("a", "t", 100)).unwrap();
        w0.add_record(&make_record("b", "t", 200)).unwrap();
        w0.finish().unwrap();

        // L0 segment 1: one live record + a tombstone (deleted at ts=500).
        let mut w1 = PaxSegmentWriter::new(
            &seg1,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            0,
            Some(1),
        );
        w1.add_record(&make_record("c", "t", 300)).unwrap();
        let mut tombstone = make_record("a", "t", 500);
        tombstone.valid_to_ns = Some(500);
        tombstone.origin = Some("delete".to_string());
        w1.add_record(&tombstone).unwrap();
        w1.finish().unwrap();

        // Compact as of now=1000 (after the delete) → the tombstone is dropped.
        let stats = compact_pax_segments(
            &[seg0, seg1],
            &out,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            0,
            &[],
            &[],
            None,
            1000,
        )
        .unwrap();

        assert_eq!(stats.inputs, 2);
        assert_eq!(stats.records_in, 4);
        assert_eq!(stats.tombstones_dropped, 1);
        assert_eq!(stats.records_out, 3);

        // The merged L1 segment holds exactly the 3 survivors; no tombstone remains.
        let mut scanner = PaxSegmentScanner::open(&out, ScanPredicate::default()).unwrap();
        let survivors = scanner.read_records(&[], &[], None).unwrap();
        assert_eq!(survivors.len(), 3);
        assert!(
            survivors
                .iter()
                .all(|r| r.origin.as_deref() != Some("delete")),
            "the delete tombstone must not survive compaction"
        );
    }

    #[test]
    fn compaction_preserves_exact_only_when_every_input_has_authority() {
        use proximadb_records::{EmbeddingCell, EmbeddingValues};

        let vector_record = |oid: &str, values: Vec<f32>| {
            let mut record = make_record(oid, "t", 1);
            record.embeddings.push(EmbeddingCell {
                modality: "dense".into(),
                dim: values.len() as u32,
                values: EmbeddingValues::Fp32(values),
                ..Default::default()
            });
            record
        };
        let dir = tempfile::tempdir().unwrap();
        let exact_input = dir.path().join("exact.pax");
        let lossy_input = dir.path().join("lossy.pax");
        let exact_output = dir.path().join("exact-output.pax");
        let mixed_output = dir.path().join("mixed-output.pax");
        let exact_records = vec![
            vector_record("exact-a", vec![-3.25, 0.123_456_7, 8.75]),
            vector_record("exact-b", vec![2.5, 4.765_432, -1.125]),
        ];
        let lossy_records = vec![
            vector_record("lossy-a", vec![11.0, -7.123_456, 0.375]),
            vector_record("lossy-b", vec![6.25, 9.765_432, -4.5]),
        ];

        let mut exact_writer = PaxSegmentWriter::new(
            &exact_input,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            1,
            None,
        )
        .with_quant(VectorQuant::RaBitQ)
        .with_f32_tier(true);
        for record in &exact_records {
            exact_writer.add_record(record).unwrap();
        }
        exact_writer.finish().unwrap();

        let mut lossy_writer = PaxSegmentWriter::new(
            &lossy_input,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            1,
            None,
        )
        .with_quant(VectorQuant::RaBitQ);
        for record in &lossy_records {
            lossy_writer.add_record(record).unwrap();
        }
        lossy_writer.finish().unwrap();

        compact_pax_segments(
            std::slice::from_ref(&exact_input),
            &exact_output,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            1,
            &[],
            &[],
            None,
            i64::MAX,
        )
        .unwrap();
        {
            let mut exact_scanner =
                PaxSegmentScanner::open(&exact_output, ScanPredicate::default()).unwrap();
            let exact_block = exact_scanner.next_block().expect("exact compacted block");
            assert!(exact_block.has_exact_vector_authority());
        }
        let mut exact_scanner =
            PaxSegmentScanner::open(&exact_output, ScanPredicate::default()).unwrap();
        let exact_back = exact_scanner.read_records(&[], &[], None).unwrap();
        for got in &exact_back {
            let want = exact_records
                .iter()
                .find(|record| record.oid == got.oid)
                .unwrap();
            assert_eq!(
                got.embeddings[0].as_fp32_slice(),
                want.embeddings[0].as_fp32_slice()
            );
        }

        compact_pax_segments(
            &[exact_input, lossy_input],
            &mixed_output,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            1,
            &[],
            &[],
            None,
            i64::MAX,
        )
        .unwrap();
        let mut mixed_scanner =
            PaxSegmentScanner::open(&mixed_output, ScanPredicate::default()).unwrap();
        let mixed_block = mixed_scanner.next_block().expect("mixed compacted block");
        assert!(
            !mixed_block.has_exact_vector_authority(),
            "a lossy sibling must prevent exact output authority"
        );
    }

    #[test]
    fn segment_write_read_scan() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.pax");

        let mut writer = PaxSegmentWriter::new(
            &path,
            BlockMode::Pax,
            BlockCompression::None,
            "col_a",
            0,
            0,
            Some(1), // 1-byte threshold → each record gets its own block
        );

        writer
            .add_record(&make_record("r1", "tenant_a", 1000))
            .unwrap();
        writer
            .add_record(&make_record("r2", "tenant_a", 2000))
            .unwrap();
        writer
            .add_record(&make_record("r3", "tenant_b", 3000))
            .unwrap();

        let meta = writer.finish().unwrap();
        assert_eq!(meta.row_count, 3);
        assert!(meta.size_bytes > 0);
        assert_eq!(meta.block_stats.len(), meta.block_count as usize);
        assert!(
            meta.block_stats[0]
                .distinct_counts
                .contains_key(&proximadb_block_format::col_id::TENANT_ID)
        );
        assert!(
            meta.block_stats[0]
                .bloom_filter_bytes
                .contains_key(&proximadb_block_format::col_id::TENANT_ID)
        );
        assert_eq!(
            meta.block_stats[0]
                .lower_bounds
                .get(&proximadb_block_format::col_id::CREATED_AT),
            Some(&1000)
        );

        // Scan with tenant_a predicate
        let mut scanner =
            PaxSegmentScanner::open(&path, ScanPredicate::for_tenant("tenant_a")).unwrap();

        let mut matched_blocks = 0usize;
        while scanner.next_block().is_some() {
            matched_blocks += 1;
        }
        // tenant_b block(s) should be pruned
        assert!(matched_blocks < meta.block_count as usize);
    }

    /// `read_records` is the canonical inverse of `add_record`: props, labels,
    /// timestamps, and the dense embedding all round-trip (not just oid+vector).
    #[test]
    fn segment_read_records_round_trips_full_fidelity() {
        use proximadb_records::{EmbeddingCell, EmbeddingValues, ProximaTreeNode, ProximaValue};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("full.pax");

        let mut rich = make_record("r1", "tenant_a", 1_700_000_000_000_000_000);
        rich.props.insert(
            "category".into(),
            ProximaTreeNode::Value(ProximaValue::String("books".into())),
        );
        rich.props
            .insert("qty".into(), ProximaTreeNode::Value(ProximaValue::Int64(7)));
        rich.labels = vec!["a".to_string(), "b".to_string()].into();
        rich.embeddings.push(EmbeddingCell {
            modality: "dense".into(),
            dim: 3,
            values: EmbeddingValues::Fp32(vec![1.0, 2.0, 3.0]),
            ..Default::default()
        });

        let mut writer = PaxSegmentWriter::new(
            &path,
            BlockMode::Pax,
            BlockCompression::None,
            "col_full",
            0,
            1, // embedding_count
            None,
        );
        writer.add_record(&rich).unwrap();
        writer
            .add_record(&make_record("r2", "tenant_a", 1_700_000_000_000_000_001))
            .unwrap();
        writer.finish().unwrap();

        let mut scanner = PaxSegmentScanner::open(&path, ScanPredicate::default()).unwrap();
        let records = scanner.read_records(&[], &[], None).unwrap();

        assert_eq!(records.len(), 2);
        let r1 = records.iter().find(|r| r.oid == "r1").expect("r1 present");
        assert_eq!(r1.tenant_id, "tenant_a");
        assert_eq!(r1.created_at_ns, 1_700_000_000_000_000_000);
        assert_eq!(
            r1.props.get("category"),
            Some(&ProximaTreeNode::Value(ProximaValue::String(
                "books".into()
            )))
        );
        assert_eq!(
            r1.props.get("qty"),
            Some(&ProximaTreeNode::Value(ProximaValue::Int64(7)))
        );
        let mut labels: Vec<String> = r1.labels.iter().map(|s| s.to_string()).collect();
        labels.sort();
        assert_eq!(labels, vec!["a".to_string(), "b".to_string()]);
        // Vectors are SQ8-quantized in PAX v2 (lossy, 4× smaller): assert the
        // embedding reconstructs within the per-column quantization error rather
        // than bit-exactly. For [1,2,3] the step is (3-1)/255 ≈ 0.0078, so the
        // bound is ~0.004 — well under 0.01.
        let recon = r1
            .embeddings
            .first()
            .map(|e| e.values.to_fp32_owned())
            .expect("embedding present");
        let expected = [1.0f32, 2.0, 3.0];
        assert_eq!(recon.len(), expected.len());
        for (got, exp) in recon.iter().zip(expected.iter()) {
            assert!(
                (got - exp).abs() <= 0.01,
                "SQ8 embedding {got} not within 0.01 of {exp}"
            );
        }
    }

    #[test]
    fn segment_scan_time_range() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("time.pax");

        let mut writer = PaxSegmentWriter::new(
            &path,
            BlockMode::Olap,
            BlockCompression::None,
            "col_b",
            0,
            0,
            Some(1), // one record per block
        );
        writer.add_record(&make_record("a", "t", 500)).unwrap();
        writer.add_record(&make_record("b", "t", 5000)).unwrap();
        writer.add_record(&make_record("c", "t", 9999)).unwrap();
        let meta = writer.finish().unwrap();

        // Only the [5000..9999] blocks should survive a [4000..6000] scan
        let mut scanner =
            PaxSegmentScanner::open(&path, ScanPredicate::for_time_range(4000, 6000)).unwrap();
        let mut hits = 0usize;
        while scanner.next_block().is_some() {
            hits += 1;
        }
        assert!(hits < meta.block_count as usize);
    }

    /// TD-RDSTRAT-8 PR-A: the two-level (v3) writer emits
    /// `[prefix v3][A0][A][B][D][footer]` with exact per-cell extents, pads
    /// blocks at coarse-cell boundaries (blocks never straddle a cell), keeps
    /// the footer-driven scanner working (mixed-read), and is deterministic
    /// (identical input ⇒ identical segment bytes).
    #[test]
    fn two_level_writer_emits_v3_with_exact_cell_extents() -> Result<()> {
        use crate::segment_layout::SEG_HEADER_PREFIX_LEN;
        use proximadb_records::{EmbeddingCell, EmbeddingValues};

        const DIM: usize = 16;
        // Three coarse cells (middle one empty) + 5 tail rows without embeddings.
        let cell_rows = vec![100u64, 0, 60];
        let n_covered = 160usize;
        let n_rows = n_covered + 5;

        let model = || CoarseModel {
            dim: DIM as u32,
            n_comp: 2,
            pca_mean: vec![0.0; DIM],
            pca_components: vec![0.1; 2 * DIM],
            centroids: vec![0.5; 3 * 2],
            radii: vec![1.0, 0.0, 2.0],
            cell_rows: cell_rows.clone(),
            seed: 42,
            trained_on: n_covered as u64,
        };
        let write = |path: &Path| -> Result<SegmentMeta> {
            let mut writer = PaxSegmentWriter::new(
                path,
                BlockMode::Pax,
                BlockCompression::None,
                "col",
                0,
                1,
                Some(64 * 1024),
            )
            .with_quant(VectorQuant::RaBitQ)
            .with_coalesced_rabitq(true)
            .with_two_level(model());
            for row in 0..n_covered {
                let mut r = make_record(&format!("r{row:04}"), "t", 1);
                let v: Vec<f32> = (0..DIM)
                    .map(|d| ((row * 7 + d) % 13) as f32 - 6.0)
                    .collect();
                r.embeddings.push(EmbeddingCell {
                    modality: "dense".into(),
                    dim: DIM as u32,
                    values: EmbeddingValues::Fp32(v),
                    ..Default::default()
                });
                writer.add_record(&r)?;
            }
            for t in 0..(n_rows - n_covered) {
                writer.add_record(&make_record(&format!("tail{t}"), "t", 1))?;
            }
            writer.finish()
        };

        let dir = tempfile::tempdir()?;
        let p1 = dir.path().join("one.pax");
        let p2 = dir.path().join("two.pax");
        write(&p1)?;
        write(&p2)?;
        let bytes = std::fs::read(&p1)?;
        assert_eq!(
            bytes,
            std::fs::read(&p2)?,
            "v3 write must be deterministic (fixed input ⇒ identical bytes)"
        );

        // Header-prefix: version 3, A0 right after the prefix, A after A0.
        let h = SegmentHeaderPrefix::parse(&bytes)?;
        assert_eq!(h.layout_version, SEG_LAYOUT_VERSION);
        assert!(h.a0_len > 0, "two-level segments declare A0 by extent");
        assert_eq!(h.a0_off, SEG_HEADER_PREFIX_LEN as u64);
        assert!(h.a0_len > 0);
        assert_eq!(h.rabitq_off, h.a0_off + h.a0_len);
        assert_eq!(h.sq8_off, h.rabitq_off + h.rabitq_len);

        // Footer mirrors the A0 extent; row count covers cells + tail.
        let footer = SegmentFooterIndex::locate_in_segment(&bytes)?
            .ok_or_else(|| anyhow::anyhow!("v3 segment must carry a footer-index"))?;
        assert_eq!(footer.row_count, n_rows as u64);
        assert_eq!(footer.a0_off, h.a0_off);
        assert_eq!(footer.a0_len, h.a0_len);

        // A0 round-trips from the segment; extents address the exact
        // fixed-stride code runs of Regions A and B.
        let a0 = CoarseDirectory::parse(&bytes[h.a0_off as usize..(h.a0_off + h.a0_len) as usize])?;
        assert_eq!(a0.model.cell_rows, cell_rows);
        assert_eq!(a0.model.rows_covered(), n_covered as u64);
        let stride_a = 8 + DIM.div_ceil(8);
        let codes_base_a =
            h.rabitq_off as usize + rabitq_region_header_len(DIM as u32) + n_rows.div_ceil(8);
        let codes_base_b = h.sq8_off as usize + sq8_codes_offset(n_rows);
        let mut row = 0u64;
        for (i, cell) in a0.cells.iter().enumerate() {
            assert_eq!(cell.row_begin, row, "cell {i} row_begin");
            assert_eq!(cell.row_end - cell.row_begin, cell_rows[i], "cell {i} rows");
            assert_eq!(
                cell.a_off as usize,
                codes_base_a + row as usize * stride_a,
                "cell {i} a_off"
            );
            assert_eq!(cell.a_len as usize, cell_rows[i] as usize * stride_a);
            assert_eq!(
                cell.b_off as usize,
                codes_base_b + row as usize * DIM,
                "cell {i} b_off"
            );
            assert_eq!(cell.b_len as usize, cell_rows[i] as usize * DIM);
            assert_eq!((cell.c_off, cell.c_len), (0, 0), "no Region C yet");
            row = cell.row_end;
        }

        // Blocks never straddle a cell: each cell's Region D block range holds
        // exactly its rows, and starts exactly at its row_begin.
        for (i, cell) in a0.cells.iter().enumerate() {
            let (d0, d1) = (cell.d_block_begin as usize, cell.d_block_end as usize);
            assert!(d0 <= d1 && d1 <= footer.blocks.len(), "cell {i} d-range");
            let rows_before: u64 = footer.blocks[..d0].iter().map(|b| b.row_count as u64).sum();
            let rows_in: u64 = footer.blocks[d0..d1]
                .iter()
                .map(|b| b.row_count as u64)
                .sum();
            assert_eq!(
                rows_before, cell.row_begin,
                "cell {i} starts on a block edge"
            );
            assert_eq!(rows_in, cell.row_end - cell.row_begin, "cell {i} padded");
        }

        // A probed cell's Region A byte slice is exactly the full-region code
        // run for its rows (what PR-B's ranged sub-reads will fetch).
        let cell0 = &a0.cells[0];
        assert_eq!(
            &bytes[cell0.a_off as usize..(cell0.a_off + cell0.a_len) as usize],
            &bytes[codes_base_a..codes_base_a + cell_rows[0] as usize * stride_a],
        );

        // Mixed-read: the footer-driven scanner reconstructs every row of a v3
        // segment unchanged (compaction/recovery read path).
        let mut scanner = PaxSegmentScanner::from_bytes(bytes.clone(), ScanPredicate::default())?;
        let recs = scanner.read_records(&[], &[], None)?;
        assert_eq!(recs.len(), n_rows, "scanner must read all v3 rows");

        // Flag-off control: the same records without `with_two_level` produce a
        // v1 segment (no A0, version byte 1) — the two-level layout is strictly
        // opt-in at the writer.
        let p3 = dir.path().join("v1.pax");
        let mut w1 = PaxSegmentWriter::new(
            &p3,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            1,
            Some(64 * 1024),
        )
        .with_quant(VectorQuant::RaBitQ)
        .with_coalesced_rabitq(true);
        for row in 0..64usize {
            let mut r = make_record(&format!("r{row:04}"), "t", 1);
            let v: Vec<f32> = (0..DIM)
                .map(|d| ((row * 7 + d) % 13) as f32 - 6.0)
                .collect();
            r.embeddings.push(EmbeddingCell {
                modality: "dense".into(),
                dim: DIM as u32,
                values: EmbeddingValues::Fp32(v),
                ..Default::default()
            });
            w1.add_record(&r)?;
        }
        w1.finish()?;
        let v1_bytes = std::fs::read(&p3)?;
        let h1 = SegmentHeaderPrefix::parse(&v1_bytes)?;
        assert_eq!(h1.layout_version, SEG_LAYOUT_VERSION);
        assert_eq!((h1.a0_off, h1.a0_len), (0, 0));
        Ok(())
    }

    /// TD-COMPACT-2: a two-level cell far past the hard row bound must NOT
    /// become one giant block. The writer cuts at `MAX_BLOCK_ROWS` mid-cell;
    /// blocks stay aligned to cell boundaries (a cell = whole D-blocks, now
    /// possibly several); the persisted probe directory maps every cell to its
    /// exact row/byte/block ranges; and the persisted layout round-trips every
    /// record's id + vector.
    #[test]
    fn two_level_row_bound_splits_giant_cells() -> Result<()> {
        use proximadb_records::{EmbeddingCell, EmbeddingValues};

        const DIM: usize = 8;
        // One cell well past the row bound + a small trailing cell.
        let cell_rows = vec![MAX_BLOCK_ROWS as u64 + 700, 300];
        let n_rows = MAX_BLOCK_ROWS + 700 + 300;

        let model = CoarseModel {
            dim: DIM as u32,
            n_comp: 1,
            pca_mean: vec![0.0; DIM],
            pca_components: vec![0.1; DIM],
            centroids: vec![0.5; 2],
            radii: vec![1.0, 1.0],
            cell_rows: cell_rows.clone(),
            seed: 7,
            trained_on: n_rows as u64,
        };
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("giant.pax");
        let mut writer = PaxSegmentWriter::new(
            &path,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            1,
            // Largest allowed byte target: the byte cut alone would admit ~19k
            // rows at the coalesced ~200 B/row model, so only the row bound can
            // split the giant cell below.
            Some(MAX_TARGET_BLOCK_SIZE_BYTES),
        )
        .with_quant(VectorQuant::RaBitQ)
        .with_coalesced_rabitq(true)
        .with_two_level(model);
        let vec_for = |row: usize| -> Vec<f32> {
            (0..DIM)
                .map(|d| ((row * 7 + d) % 13) as f32 - 6.0)
                .collect()
        };
        for row in 0..n_rows {
            let mut r = make_record(&format!("r{row:05}"), "t", 1);
            r.embeddings.push(EmbeddingCell {
                modality: "dense".into(),
                dim: DIM as u32,
                values: EmbeddingValues::Fp32(vec_for(row)),
                ..Default::default()
            });
            writer.add_record(&r)?;
        }
        let meta = writer.finish()?;
        let bytes = std::fs::read(&path)?;

        // (a) Every block's row count respects the bound; the giant cell split.
        let footer = SegmentFooterIndex::locate_in_segment(&bytes)?
            .ok_or_else(|| anyhow::anyhow!("v3 segment must carry a footer-index"))?;
        assert_eq!(footer.row_count, n_rows as u64);
        assert!(
            meta.block_count >= 3,
            "giant cell must split into multiple blocks (got {})",
            meta.block_count
        );
        for (i, b) in footer.blocks.iter().enumerate() {
            assert!(
                (b.row_count as usize) <= MAX_BLOCK_ROWS,
                "block {i} rows {} exceed MAX_BLOCK_ROWS {MAX_BLOCK_ROWS}",
                b.row_count
            );
        }

        // (b)+(d) The probe directory still maps every cell to exact row/byte
        // extents and WHOLE-block ranges; the over-bound cell spans several.
        let h = SegmentHeaderPrefix::parse(&bytes)?;
        let a0 = CoarseDirectory::parse(&bytes[h.a0_off as usize..(h.a0_off + h.a0_len) as usize])?;
        assert_eq!(a0.model.cell_rows, cell_rows);
        let stride_a = 8 + DIM.div_ceil(8);
        let codes_base_a =
            h.rabitq_off as usize + rabitq_region_header_len(DIM as u32) + n_rows.div_ceil(8);
        let codes_base_b = h.sq8_off as usize + sq8_codes_offset(n_rows);
        let mut row = 0u64;
        for (i, cell) in a0.cells.iter().enumerate() {
            assert_eq!(cell.row_begin, row, "cell {i} row_begin");
            assert_eq!(cell.row_end - cell.row_begin, cell_rows[i], "cell {i} rows");
            assert_eq!(cell.a_off as usize, codes_base_a + row as usize * stride_a);
            assert_eq!(cell.a_len as usize, cell_rows[i] as usize * stride_a);
            assert_eq!(cell.b_off as usize, codes_base_b + row as usize * DIM);
            assert_eq!(cell.b_len as usize, cell_rows[i] as usize * DIM);
            let (d0, d1) = (cell.d_block_begin as usize, cell.d_block_end as usize);
            assert!(d0 < d1 && d1 <= footer.blocks.len(), "cell {i} d-range");
            let rows_before: u64 = footer.blocks[..d0].iter().map(|b| b.row_count as u64).sum();
            let rows_in: u64 = footer.blocks[d0..d1]
                .iter()
                .map(|b| b.row_count as u64)
                .sum();
            assert_eq!(
                rows_before, cell.row_begin,
                "cell {i} starts on a block edge"
            );
            assert_eq!(rows_in, cell_rows[i], "cell {i} = whole blocks");
            row = cell.row_end;
        }
        let big = &a0.cells[0];
        assert!(
            big.d_block_end - big.d_block_begin >= 2,
            "the over-bound cell must span multiple whole blocks"
        );

        // (c) Round-trip: block rows preserve add (cluster) order and ids, and
        // Region B row i is row i's vector (the overlay contract the production
        // inverse `read_segment_records` relies on), within SQ8 tolerance.
        let mut scanner = PaxSegmentScanner::from_bytes(bytes.clone(), ScanPredicate::default())?;
        let recs = scanner.read_records(&[], &[], None)?;
        assert_eq!(recs.len(), n_rows, "scanner must read every row");
        let sq8 = proximadb_block_format::coalesced_sq8::Sq8Region::from_bytes(
            &bytes[h.sq8_off as usize..(h.sq8_off + h.sq8_len) as usize],
        )?;
        for (i, rec) in recs.iter().enumerate() {
            assert_eq!(rec.oid, format!("r{i:05}"), "row {i} id");
            let v = sq8
                .decode_row(i)
                .ok_or_else(|| anyhow::anyhow!("row {i} missing from Region B"))?;
            for (got, exp) in v.iter().zip(vec_for(i).iter()) {
                assert!(
                    (got - exp).abs() <= 0.05,
                    "row {i}: SQ8 vector {got} not within 0.05 of {exp}"
                );
            }
        }
        Ok(())
    }

    /// A two-level plan whose row boundaries are never reached (caller fed
    /// fewer rows than the model claims) must fail closed at finish, not write
    /// a mis-addressed segment.
    #[test]
    fn two_level_writer_fails_closed_on_row_mismatch() -> Result<()> {
        use proximadb_records::{EmbeddingCell, EmbeddingValues};
        const DIM: usize = 8;
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("mismatch.pax");
        let model = CoarseModel {
            dim: DIM as u32,
            n_comp: 1,
            pca_mean: vec![0.0; DIM],
            pca_components: vec![0.1; DIM],
            centroids: vec![0.5; 2],
            radii: vec![1.0, 1.0],
            cell_rows: vec![50, 50], // claims 100 rows; we feed 10
            seed: 7,
            trained_on: 10,
        };
        let mut writer = PaxSegmentWriter::new(
            &path,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            1,
            None,
        )
        .with_quant(VectorQuant::RaBitQ)
        .with_coalesced_rabitq(true)
        .with_two_level(model);
        for row in 0..10usize {
            let mut r = make_record(&format!("r{row}"), "t", 1);
            r.embeddings.push(EmbeddingCell {
                modality: "dense".into(),
                dim: DIM as u32,
                values: EmbeddingValues::Fp32(vec![row as f32; DIM]),
                ..Default::default()
            });
            writer.add_record(&r)?;
        }
        let err = writer.finish().unwrap_err();
        assert!(
            err.to_string().contains("boundaries not all reached"),
            "{err}"
        );
        Ok(())
    }
    /// TD-PAXRG-1 Phase F: coarse cells never straddle row groups — under
    /// rg_layout the writer force-cuts at every coarse-cell boundary, so a
    /// two-cell segment yields exactly one whole RG per cell (A0's
    /// `d_block_begin/end` then hold whole-RG ordinals by construction).
    #[test]
    fn v4_cells_never_straddle_row_groups() -> Result<()> {
        use crate::coarse_directory::CoarseModel;
        use proximadb_records::{EmbeddingCell, EmbeddingValues};

        const DIM: usize = 8;
        let model = CoarseModel {
            dim: DIM as u32,
            n_comp: 1,
            pca_mean: vec![0.0; DIM],
            pca_components: vec![0.1; DIM],
            centroids: vec![-1.0, 1.0],
            radii: vec![1.0, 1.0],
            cell_rows: vec![5, 5],
            seed: 7,
            trained_on: 10,
        };
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("cells-rg.pax");
        // Huge target: WITHOUT the cell-boundary cut this would be one RG.
        let mut writer = PaxSegmentWriter::new(
            &path,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            1,
            Some(RG_TARGET_MIN_BYTES),
        )
        .with_quant(VectorQuant::RaBitQ)
        .with_coalesced_rabitq(true)
        .with_rg_layout(true)
        .with_oid_resolver(true)
        .with_two_level(model);
        for row in 0..10usize {
            let mut record = make_record(&format!("oid-{row}"), "t", 1_000 + row as i64);
            record.embeddings.push(EmbeddingCell {
                modality: "dense".into(),
                dim: DIM as u32,
                values: EmbeddingValues::Fp32(
                    (0..DIM).map(|d| (row * 7 + d) as f32 * 0.01).collect(),
                ),
                ..Default::default()
            });
            writer.add_record(&record)?;
        }
        writer.finish()?;

        let segment = std::fs::read(&path)?;
        let footer = SegmentFooterIndex::locate_in_segment(&segment)?
            .ok_or_else(|| anyhow::anyhow!("coalesced footer missing"))?;
        assert_eq!(
            footer.blocks.len(),
            2,
            "two cells ⇒ two whole RGs (the cell boundary forces an RG cut)"
        );
        let rows: Vec<u32> = footer.blocks.iter().map(|b| b.row_count).collect();
        assert_eq!(rows, vec![5, 5], "one RG per coarse cell");
        Ok(())
    }
    /// TD-PAXRG-1 Phase G (at-rest evidence): for the SAME corpus, the
    /// row-group layout is SMALLER than the legacy block layout — v4 deletes
    /// the 32 B/row RowDirectory and the in-block rgdir from every granule and
    /// adds only the ~95 B/RG footer stats payload. This is the honest
    /// at-rest comparison (v4 vs v3, identical data) — the Parquet-total
    /// comparison on narrow data is a documented boundary, not a claim.
    #[test]
    fn rg_layout_at_rest_is_not_larger_than_legacy_blocks() -> Result<()> {
        use proximadb_records::{EmbeddingCell, EmbeddingValues};
        const DIM: usize = 32;
        const ROWS: usize = 4_000;
        let dir = tempfile::tempdir()?;
        let build = |path: &std::path::Path, rg: bool| -> Result<u64> {
            let mut writer = PaxSegmentWriter::new(
                path,
                BlockMode::Pax,
                BlockCompression::None,
                "col",
                0,
                1,
                Some(RG_TARGET_MIN_BYTES),
            )
            .with_quant(VectorQuant::RaBitQ)
            .with_coalesced_rabitq(true)
            .with_oid_resolver(true);
            if rg {
                writer = writer.with_rg_layout(true);
            }
            for row in 0..ROWS {
                let ts = (row + 1) as i64 * 1000;
                let mut record = make_record(&format!("oid-{row}"), "t", ts);
                record.created_at_ns = ts;
                record.updated_at_ns = ts;
                record.embeddings.push(EmbeddingCell {
                    modality: "dense".into(),
                    dim: DIM as u32,
                    values: EmbeddingValues::Fp32(vec![0.25; DIM]),
                    ..Default::default()
                });
                writer.add_record(&record)?;
            }
            writer.finish()?;
            Ok(std::fs::metadata(path)?.len())
        };

        let v3_len = build(&dir.path().join("size-v3.pax"), false)?;
        let v4_len = build(&dir.path().join("size-v4.pax"), true)?;
        eprintln!(
            "[size] legacy={v3_len} rg_layout={v4_len} ({:.1}% of legacy)",
            v4_len as f64 / v3_len as f64 * 100.0
        );
        assert!(
            v4_len <= v3_len,
            "row-group layout must not be larger at rest: v3={v3_len} v4={v4_len}              (RowDirectory removal should make it smaller)"
        );
        Ok(())
    }
    /// TD-PAXRG-1 closeout: the spill twin with rg + f32 tier emits Region C
    /// (previously a bail), stays byte-identical to the in-memory twin, and
    /// round-trips EXACT embeddings through Region C.
    #[test]
    fn rg_spill_with_f32_region_c_round_trips_and_matches_memory_twin() -> Result<()> {
        use proximadb_records::{EmbeddingCell, EmbeddingValues};
        const DIM: usize = 16;
        const ROWS: usize = 64;
        let dir = tempfile::tempdir()?;
        let scratch = dir.path().join("scratch");
        std::fs::create_dir_all(&scratch)?;

        let build = |path: &std::path::Path, spill: Option<&std::path::Path>| -> Result<()> {
            let mut writer = PaxSegmentWriter::new(
                path,
                BlockMode::Pax,
                BlockCompression::None,
                "col",
                0,
                1,
                Some(RG_TARGET_MIN_BYTES),
            )
            .with_quant(VectorQuant::RaBitQ)
            .with_coalesced_rabitq(true)
            .with_f32_tier(true)
            .with_rg_layout(true)
            .with_oid_resolver(true);
            if let Some(spill_root) = spill {
                writer = writer.with_local_spill(spill_root)?;
            }
            for row in 0..ROWS {
                let mut record = make_record(&format!("oid-{row}"), "t", 1_000 + row as i64);
                record.embeddings.push(EmbeddingCell {
                    modality: "dense".into(),
                    dim: DIM as u32,
                    values: EmbeddingValues::Fp32(
                        (0..DIM)
                            .map(|d| (row * 11 + d * 3) as f32 * 0.017)
                            .collect(),
                    ),
                    ..Default::default()
                });
                writer.add_record(&record)?;
            }
            writer.finish()?;
            Ok(())
        };

        let mem_path = dir.path().join("mem.pax");
        build(&mem_path, None)?;
        let spill_path = dir.path().join("spill.pax");
        build(&spill_path, Some(&scratch))?;

        // Byte-identical twins INCLUDING Region C.
        assert_eq!(
            std::fs::metadata(&mem_path)?.len(),
            std::fs::metadata(&spill_path)?.len(),
            "twins must agree on total length (C present in both)"
        );
        let mem_bytes = std::fs::read(&mem_path)?;
        let spill_bytes = std::fs::read(&spill_path)?;
        assert_eq!(mem_bytes, spill_bytes, "twins must be byte-identical");

        // Region C declared + exact authority holds.
        let header = SegmentHeaderPrefix::parse(&mem_bytes[..SEG_HEADER_PREFIX_LEN])?;
        assert!(header.c_len > 0, "Region C present under rg + f32 tier");
        let footer = SegmentFooterIndex::locate_in_segment(&mem_bytes)?
            .ok_or_else(|| anyhow::anyhow!("footer missing"))?;
        assert_eq!(footer.c_len, header.c_len);
        assert!(footer.has_f32_tier);

        // Exact round-trip: every embedding reconstructs bitwise from Region C.
        let recs = crate::pax_block::read_pax_segment_records(&mem_bytes, &[], &[], None)?;
        assert_eq!(recs.len(), ROWS);
        for (row, rec) in recs.iter().enumerate() {
            let cell = rec.embeddings.first().expect("embedding present");
            let proximadb_records::EmbeddingValues::Fp32(values) = &cell.values else {
                panic!("must decode to Fp32");
            };
            let expected: Vec<f32> = (0..DIM)
                .map(|d| (row * 11 + d * 3) as f32 * 0.017)
                .collect();
            assert_eq!(values, &expected, "row {row} exact via Region C");
        }
        Ok(())
    }
}
