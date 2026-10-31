//! Collection-global PAX centroid directory (TD-PAXDIR-1, slice D1).
//!
//! Every coalesced PAX segment carries its own Region A0 coarse directory
//! ([`crate::coarse_directory`]): a PCA model, `k_c` centroids and per-cell
//! byte extents. A query therefore pays each segment's header and A0 before it
//! can probe anything, and `nprobe` is chosen per segment. This module is the
//! **collection-level** view of those A0s: one small object, published by
//! compaction beside the segments, that lists every covered segment's cells so
//! a query ranks cells across the whole collection in RAM and probes only the
//! segments that own the winners (design:
//! `docs/12-design/P5B_PAX_CENTROID_DIRECTORY_DESIGN.adoc`).
//!
//! The directory is an S7 projection: derived only from segment bytes,
//! rebuildable at any time, never authoritative. Its **stamp** is the segment
//! table — the exact `(name, size)` set it examined. A reader trusts an entry
//! only for a segment that is still live with the same size; everything else
//! falls back to the per-segment cascade, so a stale or missing directory can
//! change a query's cost, never its correctness.
//!
//! ## Object naming
//!
//! `centroids-<generation:016x>.pxcd` in the collection data directory. The
//! extension is deliberately not `.pax` (discovery treats every `*.pax` as a
//! segment); the LIST that discovery already issues returns the directory too.
//!
//! ## Byte layout (fixed, little-endian, deterministic)
//!
//! ```text
//! [magic "PXCD" 4][version u8][flags u8][reserved u16]
//! [generation u64][dim u32][n_models u32][n_segments u32][n_cells u32]
//! models:   n_models × [model_id u64][n_comp u16]
//!                       [pca_mean dim × f32][pca_components n_comp × dim × f32]
//! segments: n_segments × [name_len u32][name utf-8][size_bytes u64]
//!                         [rows_covered u64][model_idx u32 (u32::MAX = uncovered)]
//!                         [has_key u8][decode key, when has_key = 1]   (v2 only)
//! cells:    n_cells × [segment_idx u32][a0_cell u32]
//!                      [centroid n_comp × f32][radius f32][CoarseCellEntry 72 B]
//! [checksum u64]   (FNV-1a over all preceding bytes, as Region A0)
//!
//! decode key (TD-PAXDIR-1 D6):
//!   [header prefix 88 B][a0_seed u64][a0_trained_on u64]
//!   [rabitq_header_len u32][Region-A header bytes]
//!   [row_count u64][sq8_min f32][sq8_scale f32][embed_dim u32][embed_count u32]
//!   [embed_quant_tag u8][has_f32_tier u8]
//!   [n_blocks u32] n_blocks × footer block entry (the footer's own wire form,
//!                              row-group OID-chunk stats included)
//!   [etag_len u32 (u32::MAX = none)][object ETag utf-8]
//! ```
//!
//! Segments are sorted by name and cells by `(segment, a0_cell)`, so identical
//! inputs always serialize to identical bytes regardless of discovery order.
//!
//! ## Decode keys (v2)
//!
//! A covered segment may carry a **decode key**: the per-segment metadata a
//! query reads before its first cell (header prefix, Region-A header, SQ8
//! dequant key, footer block table). Together with the segment's model and
//! cells already in the directory, it lets [`PaxCentroidDirectory::segment_decode`]
//! rebuild the header, Region A0 and the ANN-relevant footer without any
//! segment read, so a covered segment's first read is its probed cells. The
//! segment keeps its self-describing copy; the key is a projection like the
//! rest of the directory. A directory with no key is written as v1, byte for
//! byte the D1 format, and `parse` accepts both versions.
//!
//! Because a key replaces the segment's own metadata reads, it is bound to
//! the exact object it was built from: beyond the `(name, size)` stamp, it
//! records the object's ETag when the backend lists one, and readers use the
//! key only when the current listing reports the same ETag.
//!
//! ## Departure from the TD-PAXDIR-1 §3.1 wire draft
//!
//! The D1 format drops the draft's `built_at_unix_ms` field and the per-table
//! `metric u8`: staleness is carried by the `(name, size)` stamp plus the
//! generation (until TD-PAXMETA-1 adds a snapshot id), and the metric is a
//! collection-level property the caller already holds. `flags` and `reserved`
//! are therefore written as zero and `parse` REJECTS non-zero values there, so
//! a future revision that gives them meaning must bump [`DIRECTORY_VERSION`]
//! rather than silently reinterpret v1 bytes.

#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};

use crate::coarse_directory::{
    CoarseCellEntry, CoarseDirectory, CoarseModel, fnv1a64, project_with_model, read_f32s,
    read_u16, read_u32, read_u64,
};
use crate::segment_layout::{
    FooterBlockEntry, FooterRowGroupStats, SEG_HEADER_PREFIX_LEN, SegmentFooterIndex,
    SegmentHeaderPrefix, read_blocks, write_block_entry,
};

/// Directory magic — first 4 bytes of the serialized object.
pub const DIRECTORY_MAGIC: &[u8; 4] = b"PXCD";

/// Directory serde version of a directory that carries decode keys (D6).
pub const DIRECTORY_VERSION: u8 = 2;

/// The D1 format (no decode keys), still written for a keyless directory so
/// a reader that predates D6 keeps using it.
const DIRECTORY_VERSION_V1: u8 = 1;

/// `etag_len` of a key whose segment's listing reported no ETag.
const NO_ETAG: u32 = u32::MAX;

/// Object-name extension. Not `.pax`: discovery lists `*.pax` as segments.
pub const DIRECTORY_EXT: &str = ".pxcd";

const OBJECT_PREFIX: &str = "centroids-";

/// `model_idx` of a segment the directory examined but cannot address (no A0,
/// a non-coalesced layout, or a dimension mismatch). Readers scan such
/// segments with the per-segment path.
pub const NO_MODEL: u32 = u32::MAX;

/// Fixed head: magic(4) + version(1) + flags(1) + reserved(2) + generation(8)
/// + dim(4) + n_models(4) + n_segments(4) + n_cells(4).
const FIXED_HEAD_LEN: usize = 32;

/// Serialized [`CoarseCellEntry`] length (8 × u64 + 2 × u32).
const CELL_ENTRY_LEN: usize = 8 * 8 + 4 * 2;

const CHECKSUM_LEN: usize = 8;

/// Object name for a directory generation.
pub fn object_name(generation: u64) -> String {
    format!("{OBJECT_PREFIX}{generation:016x}{DIRECTORY_EXT}")
}

/// Generation encoded in a directory object name, or `None` if `name` is not
/// one (segments, sidecars and foreign objects all return `None`).
pub fn parse_object_name(name: &str) -> Option<u64> {
    let hex = name
        .strip_prefix(OBJECT_PREFIX)?
        .strip_suffix(DIRECTORY_EXT)?;
    if hex.len() != 16 {
        return None;
    }
    u64::from_str_radix(hex, 16).ok()
}

/// One PCA model shared by the segments that were clustered with it.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectoryModel {
    /// FNV-1a of the model's serialized `(n_comp, pca_mean, pca_components)`;
    /// segments written with the same model share it, and therefore a cell
    /// space.
    pub model_id: u64,
    pub n_comp: u16,
    pub pca_mean: Vec<f32>,
    pub pca_components: Vec<f32>,
}

impl DirectoryModel {
    fn from_a0(a0: &CoarseDirectory) -> Self {
        let m = &a0.model;
        let mut bytes = Vec::with_capacity(2 + 4 * (m.pca_mean.len() + m.pca_components.len()));
        bytes.extend_from_slice(&m.n_comp.to_le_bytes());
        for v in m.pca_mean.iter().chain(&m.pca_components) {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Self {
            model_id: fnv1a64(&bytes),
            n_comp: m.n_comp,
            pca_mean: m.pca_mean.clone(),
            pca_components: m.pca_components.clone(),
        }
    }
}

/// One segment the directory examined — the stamp.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectorySegment {
    /// Object name relative to the collection data directory.
    pub name: String,
    /// Object size when examined; a size change means the entry is stale.
    pub size_bytes: u64,
    /// Rows covered by the segment's cells (`0` when uncovered).
    pub rows_covered: u64,
    /// Index into the model table, or [`NO_MODEL`].
    pub model_idx: u32,
    /// TD-PAXDIR-1 D6: the segment's query-path metadata, so a covered
    /// segment needs no header, A0 or footer read. Only covered segments
    /// carry one; `None` keeps the per-segment metadata reads.
    pub decode_key: Option<SegmentDecodeKey>,
}

/// TD-PAXDIR-1 D6: everything a covered segment's ANN query reads before its
/// first cell, beyond what the directory already holds (model, centroids,
/// radii, cell extents). Copied from the segment's own bytes at publish.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentDecodeKey {
    /// The segment's header prefix, verbatim ([`SEG_HEADER_PREFIX_LEN`] bytes).
    pub header_prefix: Vec<u8>,
    /// Region A0 model fields the directory does not otherwise store.
    pub a0_seed: u64,
    pub a0_trained_on: u64,
    /// The Region-A header (RaBitQ params and centroid), verbatim.
    pub rabitq_header: Vec<u8>,
    /// Footer facts the ANN path reads: row count, SQ8 dequant key, schema.
    pub row_count: u64,
    pub sq8_min: f32,
    pub sq8_scale: f32,
    pub embed_dim: u32,
    pub embed_count: u32,
    pub embed_quant_tag: u8,
    pub has_f32_tier: bool,
    /// The footer's block (row-group) table with its OID-chunk stats.
    pub blocks: Vec<FooterBlockEntry>,
    pub block_stats: Vec<Option<FooterRowGroupStats>>,
    /// The segment object's ETag as listed when the key was built (`None`
    /// when the backend lists none). See [`Self::matches_object`].
    pub object_etag: Option<String>,
}

impl SegmentDecodeKey {
    /// Build from a segment's header prefix, Region-A header bytes, parsed A0
    /// and footer, and its listed ETag. Fails when they disagree.
    pub fn from_segment(
        header_prefix: &[u8],
        a0: &CoarseDirectory,
        rabitq_header: &[u8],
        footer: &SegmentFooterIndex,
        object_etag: Option<&str>,
    ) -> Result<Self> {
        let key = Self {
            header_prefix: header_prefix
                .get(..SEG_HEADER_PREFIX_LEN)
                .ok_or_else(|| anyhow::anyhow!("decode key: header prefix truncated"))?
                .to_vec(),
            a0_seed: a0.model.seed,
            a0_trained_on: a0.model.trained_on,
            rabitq_header: rabitq_header.to_vec(),
            row_count: footer.row_count,
            sq8_min: footer.sq8_min,
            sq8_scale: footer.sq8_scale,
            embed_dim: footer.embed_dim,
            embed_count: footer.embed_count,
            embed_quant_tag: footer.embed_quant_tag,
            has_f32_tier: footer.has_f32_tier,
            blocks: footer.blocks.clone(),
            block_stats: footer.block_stats.clone(),
            object_etag: object_etag.map(str::to_owned),
        };
        key.check(a0.model.dim, a0.model.rows_covered())?;
        Ok(key)
    }

    /// Whether this key describes the object a listing now reports: the
    /// ETags must agree, and a key built with one never matches a listing
    /// without one. (`None` on both sides — a backend that lists no ETag —
    /// leaves the `(name, size)` stamp as the only binding.)
    pub fn matches_object(&self, listed_etag: Option<&str>) -> bool {
        self.object_etag.as_deref() == listed_etag
    }

    /// Internal consistency against the segment's directory entry. Any
    /// disagreement means the key cannot describe the segment: fail closed.
    fn check(&self, dim: u32, rows_covered: u64) -> Result<()> {
        let header = SegmentHeaderPrefix::parse(&self.header_prefix)?;
        if header.a0_len == 0 {
            bail!("decode key: segment has no Region A0");
        }
        if self.embed_dim != dim {
            bail!("decode key: dim {} differs from {dim}", self.embed_dim);
        }
        if self.rabitq_header.len() != proximadb_block_format::region_header_len(dim) {
            bail!("decode key: Region-A header length mismatch");
        }
        let rq = proximadb_block_format::CoalescedRaBitQHeader::parse(&self.rabitq_header)?;
        if rq.dim != dim || u64::from(rq.n_rows) != self.row_count {
            bail!("decode key: Region-A header disagrees with the segment");
        }
        if !self.sq8_min.is_finite() || !self.sq8_scale.is_finite() {
            bail!("decode key: non-finite SQ8 dequant key");
        }
        let block_rows: u64 = self.blocks.iter().map(|b| u64::from(b.row_count)).sum();
        if self.row_count != rows_covered || block_rows != self.row_count {
            bail!(
                "decode key: rows disagree (footer {}, blocks {block_rows}, A0 {rows_covered})",
                self.row_count
            );
        }
        if self.block_stats.len() != self.blocks.len() {
            bail!("decode key: block stats misaligned with blocks");
        }
        // Readers plan coalesced block fetches by block index and slice each
        // block out of its fetch: offsets must ascend without overlap.
        let mut end = 0u64;
        for block in &self.blocks {
            if block.offset < end {
                bail!("decode key: block table offsets overlap or descend");
            }
            end = block
                .offset
                .checked_add(u64::from(block.size))
                .ok_or_else(|| anyhow::anyhow!("decode key: block extent overflows"))?;
        }
        Ok(())
    }

    fn write_to(&self, buf: &mut Vec<u8>) -> Result<()> {
        buf.extend_from_slice(&self.header_prefix);
        buf.extend_from_slice(&self.a0_seed.to_le_bytes());
        buf.extend_from_slice(&self.a0_trained_on.to_le_bytes());
        let hdr_len = u32::try_from(self.rabitq_header.len())
            .map_err(|_| anyhow::anyhow!("decode key Region-A header exceeds u32"))?;
        buf.extend_from_slice(&hdr_len.to_le_bytes());
        buf.extend_from_slice(&self.rabitq_header);
        buf.extend_from_slice(&self.row_count.to_le_bytes());
        buf.extend_from_slice(&self.sq8_min.to_le_bytes());
        buf.extend_from_slice(&self.sq8_scale.to_le_bytes());
        buf.extend_from_slice(&self.embed_dim.to_le_bytes());
        buf.extend_from_slice(&self.embed_count.to_le_bytes());
        buf.push(self.embed_quant_tag);
        buf.push(u8::from(self.has_f32_tier));
        let n_blocks = u32::try_from(self.blocks.len())
            .map_err(|_| anyhow::anyhow!("decode key block table exceeds u32"))?;
        buf.extend_from_slice(&n_blocks.to_le_bytes());
        for (block, stats) in self.blocks.iter().zip(&self.block_stats) {
            write_block_entry(buf, block, stats.as_ref());
        }
        match &self.object_etag {
            Some(etag) => {
                let len = u32::try_from(etag.len())
                    .ok()
                    .filter(|len| *len != NO_ETAG)
                    .ok_or_else(|| anyhow::anyhow!("decode key ETag too long"))?;
                buf.extend_from_slice(&len.to_le_bytes());
                buf.extend_from_slice(etag.as_bytes());
            }
            None => buf.extend_from_slice(&NO_ETAG.to_le_bytes()),
        }
        Ok(())
    }

    fn read_from(body: &[u8], p: &mut usize) -> Result<Self> {
        let take = |p: &mut usize, len: usize| -> Result<Vec<u8>> {
            let end = p
                .checked_add(len)
                .filter(|end| *end <= body.len())
                .ok_or_else(|| anyhow::anyhow!("centroid directory truncated (decode key)"))?;
            let bytes = body[*p..end].to_vec();
            *p = end;
            Ok(bytes)
        };
        let header_prefix = take(p, SEG_HEADER_PREFIX_LEN)?;
        let a0_seed = read_u64(body, p)?;
        let a0_trained_on = read_u64(body, p)?;
        let hdr_len = read_u32(body, p)? as usize;
        let rabitq_header = take(p, hdr_len)?;
        let row_count = read_u64(body, p)?;
        let sq8_min = f32::from_bits(read_u32(body, p)?);
        let sq8_scale = f32::from_bits(read_u32(body, p)?);
        let embed_dim = read_u32(body, p)?;
        let embed_count = read_u32(body, p)?;
        let flags = take(p, 2)?;
        let has_f32_tier = match flags[1] {
            0 => false,
            1 => true,
            other => bail!("decode key has_f32_tier byte {other}"),
        };
        let (blocks, block_stats) = read_blocks(body, p)?;
        let object_etag = match read_u32(body, p)? {
            NO_ETAG => None,
            len => Some(
                String::from_utf8(take(p, len as usize)?)
                    .map_err(|_| anyhow::anyhow!("decode key ETag is not UTF-8"))?,
            ),
        };
        Ok(Self {
            header_prefix,
            a0_seed,
            a0_trained_on,
            rabitq_header,
            row_count,
            sq8_min,
            sq8_scale,
            embed_dim,
            embed_count,
            embed_quant_tag: flags[0],
            has_f32_tier,
            blocks,
            block_stats,
            object_etag,
        })
    }
}

/// A covered segment's query-path metadata rebuilt from the directory
/// ([`PaxCentroidDirectory::segment_decode`]), in the forms the segment
/// itself stores: the header prefix, Region A0 bytes, the Region-A header,
/// and a footer carrying the ANN path's fields. The footer is **not** the
/// segment's footer (it omits the encoding map, filter stats and shred map),
/// so it must only serve the unfiltered ANN cascade and never be cached as
/// the segment's metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentDecodeBytes {
    pub header_prefix: Vec<u8>,
    pub a0: Vec<u8>,
    pub rabitq_header: Vec<u8>,
    pub ann_footer: Vec<u8>,
}

impl DirectorySegment {
    /// Whether the directory addresses this segment's cells.
    pub fn is_covered(&self) -> bool {
        self.model_idx != NO_MODEL
    }
}

/// One coarse cell of one segment.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectoryCell {
    pub segment_idx: u32,
    /// The cell's index in the segment's own Region A0.
    pub a0_cell: u32,
    /// Centroid in the segment model's PCA space (`n_comp` values).
    pub centroid: Vec<f32>,
    /// Max member→centroid distance in PCA space.
    pub radius: f32,
    /// Row range and absolute byte extents, copied verbatim from A0.
    pub entry: CoarseCellEntry,
}

impl DirectoryCell {
    fn rows(&self) -> u64 {
        self.entry.row_end.saturating_sub(self.entry.row_begin)
    }
}

/// A cell and its global ranking score (squared L2 distance from the query to
/// the centroid's PCA reconstruction in the original space).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredCell {
    pub cell: usize,
    pub score: f32,
}

/// A query's use of the directory: for every usable segment (by object
/// name), the Region A0 cells to probe, in rank order. An empty list means the
/// segment holds none of the globally nearest cells and is skipped. Segments
/// absent from the map are not usable and keep the per-segment path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbePlan {
    pub cells_by_segment: HashMap<String, Vec<u32>>,
    /// Distinct usable non-empty cells `(cell space, a0_cell)` the ranking
    /// chose from (segments sharing a cell space share their cells).
    pub usable_cells: usize,
}

/// TD-PAXDIR-1 D3: the collection's shared cell space, as a compaction uses
/// it: the model to classify with and the rows the space already covers (the
/// growth trigger sizes cells on collection rows, not on one output).
#[derive(Debug, Clone, PartialEq)]
pub struct SharedCellSpace {
    pub model: CoarseModel,
    pub covered_rows: u64,
}

/// The parsed collection directory.
#[derive(Debug, Clone, PartialEq)]
pub struct PaxCentroidDirectory {
    pub generation: u64,
    pub dim: u32,
    pub models: Vec<DirectoryModel>,
    pub segments: Vec<DirectorySegment>,
    pub cells: Vec<DirectoryCell>,
    /// Per-segment cell-space ids, computed once on first use (a directory is
    /// loaded once per generation and planned against on every query).
    /// Derived from the fields above; ignored by equality.
    pub space_ids: SpaceIds,
}

/// Lazily computed per-segment cell-space ids of a [`PaxCentroidDirectory`]
/// (see `cell_spaces`). Derived data: always equal, so it never affects
/// directory equality.
#[derive(Debug, Clone, Default)]
pub struct SpaceIds(std::sync::OnceLock<Vec<u64>>);

impl PartialEq for SpaceIds {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl PaxCentroidDirectory {
    /// Serialize (deterministic).
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let dim = self.dim as usize;
        let mut buf = Vec::new();
        buf.extend_from_slice(DIRECTORY_MAGIC);
        let keyed = self.segments.iter().any(|s| s.decode_key.is_some());
        buf.push(if keyed {
            DIRECTORY_VERSION
        } else {
            DIRECTORY_VERSION_V1
        });
        buf.push(0); // flags (reserved; TD-PAXMETA-1 snapshot id)
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&self.generation.to_le_bytes());
        buf.extend_from_slice(&self.dim.to_le_bytes());
        for count in [self.models.len(), self.segments.len(), self.cells.len()] {
            let count = u32::try_from(count)
                .map_err(|_| anyhow::anyhow!("centroid directory table exceeds u32"))?;
            buf.extend_from_slice(&count.to_le_bytes());
        }
        for model in &self.models {
            buf.extend_from_slice(&model.model_id.to_le_bytes());
            buf.extend_from_slice(&model.n_comp.to_le_bytes());
            debug_assert_eq!(model.pca_mean.len(), dim);
            push_f32s(&mut buf, &model.pca_mean);
            push_f32s(&mut buf, &model.pca_components);
        }
        for segment in &self.segments {
            let name_len = u32::try_from(segment.name.len())
                .map_err(|_| anyhow::anyhow!("segment name exceeds u32"))?;
            buf.extend_from_slice(&name_len.to_le_bytes());
            buf.extend_from_slice(segment.name.as_bytes());
            buf.extend_from_slice(&segment.size_bytes.to_le_bytes());
            buf.extend_from_slice(&segment.rows_covered.to_le_bytes());
            buf.extend_from_slice(&segment.model_idx.to_le_bytes());
            if keyed {
                match &segment.decode_key {
                    Some(key) => {
                        buf.push(1);
                        key.write_to(&mut buf)?;
                    }
                    None => buf.push(0),
                }
            }
        }
        for cell in &self.cells {
            buf.extend_from_slice(&cell.segment_idx.to_le_bytes());
            buf.extend_from_slice(&cell.a0_cell.to_le_bytes());
            push_f32s(&mut buf, &cell.centroid);
            buf.extend_from_slice(&cell.radius.to_le_bytes());
            cell.entry.write_to(&mut buf);
        }
        let checksum = fnv1a64(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        Ok(buf)
    }

    /// Parse and verify. Fail-closed on magic, version, truncation, checksum
    /// or any structural inconsistency; the caller treats an error as "no
    /// directory" and uses the per-segment path.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < FIXED_HEAD_LEN + CHECKSUM_LEN {
            bail!("centroid directory too short: {}", bytes.len());
        }
        if &bytes[..4] != DIRECTORY_MAGIC {
            bail!("not a centroid directory (bad magic)");
        }
        let version = bytes[4];
        if version != DIRECTORY_VERSION && version != DIRECTORY_VERSION_V1 {
            bail!("unsupported centroid directory version {version}");
        }
        // v1 and v2 assign no meaning to flags or reserved: reject non-zero
        // values so a later revision must bump the version instead of
        // silently reinterpreting these bytes.
        if bytes[5] != 0 {
            bail!(
                "unknown centroid directory flags 0x{:02x} (version {})",
                bytes[5],
                bytes[4]
            );
        }
        if bytes[6] != 0 || bytes[7] != 0 {
            bail!("centroid directory reserved bytes must be zero");
        }
        let body_len = bytes.len() - CHECKSUM_LEN;
        let stored = u64::from_le_bytes(bytes[body_len..].try_into()?);
        if stored != fnv1a64(&bytes[..body_len]) {
            bail!("centroid directory checksum mismatch");
        }
        let body = &bytes[..body_len];
        let mut p = 8usize; // magic + version + flags + reserved
        let generation = read_u64(body, &mut p)?;
        let dim = read_u32(body, &mut p)?;
        let n_models = read_u32(body, &mut p)? as usize;
        let n_segments = read_u32(body, &mut p)? as usize;
        let n_cells = read_u32(body, &mut p)? as usize;
        let d = dim as usize;

        // Every count is bounded by the remaining bytes before allocating.
        let mut models = Vec::with_capacity(n_models.min(body.len() / 10));
        for _ in 0..n_models {
            let model_id = read_u64(body, &mut p)?;
            let n_comp = read_u16(body, &mut p)?;
            let pca_mean = read_f32s(body, &mut p, d)?;
            let pca_components = read_f32s(body, &mut p, n_comp as usize * d)?;
            models.push(DirectoryModel {
                model_id,
                n_comp,
                pca_mean,
                pca_components,
            });
        }
        let mut segments = Vec::with_capacity(n_segments.min(body.len() / 24));
        for _ in 0..n_segments {
            let name_len = read_u32(body, &mut p)? as usize;
            let end = p
                .checked_add(name_len)
                .filter(|end| *end <= body.len())
                .ok_or_else(|| anyhow::anyhow!("centroid directory truncated (segment name)"))?;
            let name = std::str::from_utf8(&body[p..end])
                .map_err(|_| anyhow::anyhow!("centroid directory segment name is not UTF-8"))?
                .to_owned();
            p = end;
            let size_bytes = read_u64(body, &mut p)?;
            let rows_covered = read_u64(body, &mut p)?;
            let model_idx = read_u32(body, &mut p)?;
            let decode_key = if version == DIRECTORY_VERSION_V1 {
                None
            } else {
                let has_key = *body
                    .get(p)
                    .ok_or_else(|| anyhow::anyhow!("centroid directory truncated (key flag)"))?;
                p += 1;
                match has_key {
                    0 => None,
                    1 => Some(SegmentDecodeKey::read_from(body, &mut p)?),
                    other => bail!("centroid directory decode-key flag {other}"),
                }
            };
            segments.push(DirectorySegment {
                name,
                size_bytes,
                rows_covered,
                model_idx,
                decode_key,
            });
        }
        let mut cells = Vec::with_capacity(n_cells.min(body.len() / (8 + CELL_ENTRY_LEN)));
        for _ in 0..n_cells {
            let segment_idx = read_u32(body, &mut p)?;
            let a0_cell = read_u32(body, &mut p)?;
            let n_comp = segments
                .get(segment_idx as usize)
                .and_then(|s| models.get(s.model_idx as usize))
                .map(|m| m.n_comp as usize)
                .ok_or_else(|| anyhow::anyhow!("centroid directory cell has no model"))?;
            let centroid = read_f32s(body, &mut p, n_comp)?;
            let radius = f32::from_le_bytes(read_u32(body, &mut p)?.to_le_bytes());
            let entry = CoarseCellEntry::read_from(body, &mut p)?;
            cells.push(DirectoryCell {
                segment_idx,
                a0_cell,
                centroid,
                radius,
                entry,
            });
        }
        if p != body.len() {
            bail!("centroid directory has {} trailing bytes", body.len() - p);
        }
        let dir = Self {
            generation,
            dim,
            models,
            segments,
            cells,
            space_ids: SpaceIds::default(),
        };
        dir.validate()?;
        Ok(dir)
    }

    /// Structural invariants shared by serialize and parse.
    fn validate(&self) -> Result<()> {
        let d = self.dim as usize;
        for model in &self.models {
            if model.n_comp == 0 || u32::from(model.n_comp) > self.dim {
                bail!(
                    "centroid directory model has invalid n_comp {}",
                    model.n_comp
                );
            }
            if model.pca_mean.len() != d || model.pca_components.len() != model.n_comp as usize * d
            {
                bail!("centroid directory model length mismatch");
            }
            // A single non-finite model value makes every cell score of the
            // model non-finite, which would let one poisoned segment occupy
            // all nprobe slots (fail closed instead).
            if model.pca_mean.iter().any(|v| !v.is_finite())
                || model.pca_components.iter().any(|v| !v.is_finite())
            {
                bail!("centroid directory model has a non-finite value");
            }
        }
        let mut names = HashSet::with_capacity(self.segments.len());
        for segment in &self.segments {
            if !names.insert(segment.name.as_str()) {
                bail!("centroid directory lists segment {} twice", segment.name);
            }
            if segment.is_covered() && segment.model_idx as usize >= self.models.len() {
                bail!("centroid directory segment {} has no model", segment.name);
            }
            if let Some(key) = &segment.decode_key {
                if !segment.is_covered() {
                    bail!(
                        "centroid directory uncovered segment {} has a decode key",
                        segment.name
                    );
                }
                key.check(self.dim, segment.rows_covered).map_err(|e| {
                    anyhow::anyhow!("centroid directory segment {}: {e}", segment.name)
                })?;
            }
        }
        for cell in &self.cells {
            let segment = self
                .segments
                .get(cell.segment_idx as usize)
                .filter(|s| s.is_covered())
                .ok_or_else(|| anyhow::anyhow!("centroid directory cell on uncovered segment"))?;
            if cell.centroid.len() != self.models[segment.model_idx as usize].n_comp as usize {
                bail!("centroid directory cell centroid length mismatch");
            }
            if cell.centroid.iter().any(|v| !v.is_finite()) || !cell.radius.is_finite() {
                bail!("centroid directory cell has a non-finite value");
            }
            if cell.entry.row_end < cell.entry.row_begin {
                bail!("centroid directory cell has an inverted row range");
            }
        }
        Ok(())
    }

    /// Covered segments by name — the entries a reader may use, provided the
    /// live object still has the recorded size.
    pub fn covered_segments(&self) -> impl Iterator<Item = (usize, &DirectorySegment)> {
        self.segments
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_covered())
    }

    /// TD-PAXDIR-1 D6: rebuild covered segment `segment_idx`'s query-path
    /// metadata from its decode key, model and cells — no segment read. The
    /// Region A0 bytes equal the segment's own (the publisher verifies this
    /// before it publishes a key). Errors when the segment has no key or the
    /// rebuilt A0 does not have the length the header records.
    pub fn segment_decode(&self, segment_idx: usize) -> Result<SegmentDecodeBytes> {
        let segment = self
            .segments
            .get(segment_idx)
            .filter(|s| s.is_covered())
            .ok_or_else(|| anyhow::anyhow!("segment {segment_idx} is not covered"))?;
        let key = segment
            .decode_key
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("segment {} has no decode key", segment.name))?;
        let header = SegmentHeaderPrefix::parse(&key.header_prefix)?;
        let (model, cells) = self.segment_model(segment_idx, key.a0_seed, key.a0_trained_on)?;
        let a0 = CoarseDirectory { model, cells }.to_bytes()?;
        if a0.len() as u64 != header.a0_len {
            bail!(
                "segment {}: rebuilt A0 is {} bytes, header records {}",
                segment.name,
                a0.len(),
                header.a0_len
            );
        }
        let ann_footer = SegmentFooterIndex {
            row_count: key.row_count,
            rabitq_off: header.rabitq_off,
            rabitq_len: header.rabitq_len,
            sq8_off: header.sq8_off,
            sq8_len: header.sq8_len,
            sq8_min: key.sq8_min,
            sq8_scale: key.sq8_scale,
            embed_dim: key.embed_dim,
            embed_count: key.embed_count,
            embed_quant_tag: key.embed_quant_tag,
            has_f32_tier: key.has_f32_tier,
            blocks: key.blocks.clone(),
            block_stats: key.block_stats.clone(),
            encoding_map: Vec::new(),
            block_tier_assignments: Vec::new(),
            a0_off: header.a0_off,
            a0_len: header.a0_len,
            opr_off: 0,
            opr_len: 0,
            c_off: header.c_off,
            c_len: header.c_len,
            block_filterable_stats: Vec::new(),
            shred_field_map: Vec::new(),
            filter_region: None,
        }
        .to_bytes()?;
        Ok(SegmentDecodeBytes {
            header_prefix: key.header_prefix.clone(),
            a0,
            rabitq_header: key.rabitq_header.clone(),
            ann_footer,
        })
    }

    /// Covered segment `segment_idx`'s coarse model and A0 cell table as the
    /// directory holds them: its PCA model, and its cells in A0 order, which
    /// must be exactly `0..k_c` (A0 lists every cell, empty ones included).
    /// `seed` and `trained_on` are diagnostics the cell table does not carry.
    fn segment_model(
        &self,
        segment_idx: usize,
        seed: u64,
        trained_on: u64,
    ) -> Result<(CoarseModel, Vec<CoarseCellEntry>)> {
        let model = &self.models[self.segments[segment_idx].model_idx as usize];
        let cells: Vec<&DirectoryCell> = self
            .cells
            .iter()
            .filter(|c| c.segment_idx as usize == segment_idx)
            .collect();
        if cells.is_empty()
            || cells
                .iter()
                .enumerate()
                .any(|(i, c)| c.a0_cell as usize != i)
        {
            bail!("segment {segment_idx}: directory cells are not a complete A0 table");
        }
        let out = CoarseModel {
            dim: self.dim,
            n_comp: model.n_comp,
            pca_mean: model.pca_mean.clone(),
            pca_components: model.pca_components.clone(),
            centroids: cells
                .iter()
                .flat_map(|c| c.centroid.iter().copied())
                .collect(),
            radii: cells.iter().map(|c| c.radius).collect(),
            cell_rows: cells.iter().map(|c| c.rows()).collect(),
            seed,
            trained_on,
        };
        out.validate()?;
        Ok((out, cells.iter().map(|c| c.entry).collect()))
    }

    /// Rank every cell across every model by the squared distance from
    /// `query` to the centroid's PCA reconstruction `μ + Pᵀc`:
    ///
    /// `‖P(q − μ) − c‖² + (‖q − μ‖² − ‖P(q − μ)‖²)`
    ///
    /// The first term is the in-segment A0 distance; the second is one value
    /// per model, so within one model the order is exactly A0's order, and
    /// across models the scores are commensurable. Ties break on cell index.
    /// `None` when the query's dimension does not match.
    pub fn rank_cells(&self, query: &[f32]) -> Option<Vec<ScoredCell>> {
        if query.len() != self.dim as usize {
            return None;
        }
        let projected: Vec<(Vec<f32>, f32)> = self
            .models
            .iter()
            .map(|m| {
                let proj =
                    project_with_model(&m.pca_mean, &m.pca_components, m.n_comp as usize, query);
                let centered: f64 = query
                    .iter()
                    .zip(&m.pca_mean)
                    .map(|(q, mu)| {
                        let v = *q as f64 - *mu as f64;
                        v * v
                    })
                    .sum();
                let explained: f64 = proj.iter().map(|v| (*v as f64) * (*v as f64)).sum();
                (proj, (centered - explained).max(0.0) as f32)
            })
            .collect();
        let mut scored: Vec<ScoredCell> = self
            .cells
            .iter()
            .enumerate()
            .map(|(cell, c)| {
                let model = self.segments[c.segment_idx as usize].model_idx as usize;
                let (proj, residual) = &projected[model];
                let d: f32 = proj
                    .iter()
                    .zip(&c.centroid)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                // A non-finite score must never win a nprobe slot: total_cmp
                // ranks -NaN first, so clamp it to +inf (sorts last) instead.
                // v1 files are rejected at parse for non-finite values; this
                // guards directories written before that check existed.
                let score = d + residual;
                ScoredCell {
                    cell,
                    score: if score.is_finite() {
                        score
                    } else {
                        f32::INFINITY
                    },
                }
            })
            .collect();
        scored.sort_by(|a, b| a.score.total_cmp(&b.score).then(a.cell.cmp(&b.cell)));
        Some(scored)
    }

    /// Nearest non-empty cells, in rank order, until at least `nprobe` cells
    /// **and** at least `min_rows` rows are selected (the survivor-pool floor
    /// that keeps recall when cells are small). `None` on a dimension
    /// mismatch.
    pub fn select_cells(&self, query: &[f32], nprobe: usize, min_rows: u64) -> Option<Vec<usize>> {
        self.select_usable_cells(query, nprobe, min_rows, |_| true)
    }

    /// [`Self::select_cells`] restricted to cells of segments `usable` accepts,
    /// counted over **distinct cells** (TD-PAXDIR-1 D3): segments written with
    /// one shared model hold the same cell — same cell space, same A0 index —
    /// once each, so a selected cell is probed in every usable segment that
    /// holds it and counts once toward `nprobe`. With one model per segment
    /// every cell is distinct and this is the D2 behaviour.
    fn select_usable_cells(
        &self,
        query: &[f32],
        nprobe: usize,
        min_rows: u64,
        usable: impl Fn(u32) -> bool,
    ) -> Option<Vec<usize>> {
        let ranked = self.rank_cells(query)?;
        let spaces = self.cell_spaces();
        let group = |c: &DirectoryCell| (spaces[c.segment_idx as usize], c.a0_cell);
        let eligible = |c: &DirectoryCell| c.rows() > 0 && usable(c.segment_idx);
        let mut group_rows: HashMap<(u64, u32), u64> = HashMap::new();
        for cell in self.cells.iter().filter(|c| eligible(c)) {
            *group_rows.entry(group(cell)).or_default() += cell.rows();
        }
        let mut chosen: HashSet<(u64, u32)> = HashSet::new();
        let mut rows = 0u64;
        for scored in &ranked {
            if chosen.len() >= nprobe && rows >= min_rows {
                break;
            }
            let cell = &self.cells[scored.cell];
            if eligible(cell) && chosen.insert(group(cell)) {
                rows += group_rows[&group(cell)];
            }
        }
        Some(
            ranked
                .iter()
                .map(|s| &self.cells[s.cell])
                .zip(ranked.iter().map(|s| s.cell))
                .filter(|(c, _)| eligible(c) && chosen.contains(&group(c)))
                .map(|(_, idx)| idx)
                .collect(),
        )
    }

    /// Per segment, the id of its **cell space**: its model plus its exact
    /// centroid table. Two segments share cells only when both match — equal
    /// PCA bytes alone do not make equal cells (the same rows clustered with
    /// a different `k`, or a partitioned plan, share the PCA but not the
    /// centroids). Uncovered segments get 0, never used for grouping.
    fn cell_spaces(&self) -> &[u64] {
        self.space_ids.0.get_or_init(|| self.compute_cell_spaces())
    }

    fn compute_cell_spaces(&self) -> Vec<u64> {
        let mut bytes: Vec<Vec<u8>> = self
            .segments
            .iter()
            .map(|s| match s.is_covered() {
                true => self.models[s.model_idx as usize]
                    .model_id
                    .to_le_bytes()
                    .to_vec(),
                false => Vec::new(),
            })
            .collect();
        for cell in &self.cells {
            let buf = &mut bytes[cell.segment_idx as usize];
            buf.extend_from_slice(&cell.a0_cell.to_le_bytes());
            for v in &cell.centroid {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        bytes
            .iter()
            .map(|b| if b.is_empty() { 0 } else { fnv1a64(b) })
            .collect()
    }

    /// TD-PAXDIR-1 D3: the collection's shared cell space — the one covering
    /// the most rows — as a [`CoarseModel`] a compaction can classify with,
    /// plus the rows it covers. Centroids and radii come from the space's
    /// largest segment (A0 lists every cell, empty ones included). Radii are
    /// that one segment's, not a max over segments, so an accepted output's
    /// outliers cannot ratchet the drift test open. `None` when nothing is
    /// covered or the template's cells are incomplete.
    pub fn collection_model(&self) -> Option<SharedCellSpace> {
        let spaces = self.cell_spaces();
        let mut rows_by_space: HashMap<u64, u64> = HashMap::new();
        for (idx, segment) in self.covered_segments() {
            *rows_by_space.entry(spaces[idx]).or_default() += segment.rows_covered;
        }
        let (space, covered_rows) = rows_by_space
            .into_iter()
            .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))?;
        let (template, segment) = self
            .covered_segments()
            .filter(|(idx, _)| spaces[*idx] == space)
            .max_by(|a, b| a.1.rows_covered.cmp(&b.1.rows_covered).then(b.0.cmp(&a.0)))?;
        let (model, _) = self.segment_model(template, 0, segment.rows_covered).ok()?;
        Some(SharedCellSpace {
            model,
            covered_rows,
        })
    }

    /// Plan a query against the live segment set (TD-PAXDIR-1 D2).
    ///
    /// Only entries for segments that are live **with the recorded size** are
    /// usable; every other live segment is absent from the plan and the
    /// caller scans it per segment. `nprobe` and `min_rows` are computed from
    /// the usable totals (cells, covered rows), so a partly stale directory
    /// probes proportionally. `None` on a dimension mismatch or when nothing
    /// is usable.
    pub fn plan_probe(
        &self,
        query: &[f32],
        live: &[LiveSegment],
        nprobe_for_cells: impl Fn(usize) -> usize,
        min_rows_for_rows: impl Fn(u64) -> u64,
    ) -> Option<ProbePlan> {
        let live_sizes: HashMap<&str, u64> = live
            .iter()
            .map(|s| (s.name.as_str(), s.size_bytes))
            .collect();
        let usable: Vec<bool> = self
            .segments
            .iter()
            .map(|s| s.is_covered() && live_sizes.get(s.name.as_str()) == Some(&s.size_bytes))
            .collect();
        let usable_cells = self
            .cells
            .iter()
            .filter(|c| usable[c.segment_idx as usize] && c.rows() > 0)
            .map(|c| (self.cell_spaces()[c.segment_idx as usize], c.a0_cell))
            .collect::<HashSet<_>>()
            .len();
        if usable_cells == 0 {
            return None;
        }
        let usable_rows: u64 = self
            .segments
            .iter()
            .zip(&usable)
            .filter(|(_, u)| **u)
            .map(|(s, _)| s.rows_covered)
            .sum();
        let selected = self.select_usable_cells(
            query,
            nprobe_for_cells(usable_cells),
            min_rows_for_rows(usable_rows),
            |segment_idx| usable[segment_idx as usize],
        )?;
        let mut cells_by_segment: HashMap<String, Vec<u32>> = self
            .segments
            .iter()
            .zip(&usable)
            .filter(|(_, u)| **u)
            .map(|(s, _)| (s.name.clone(), Vec::new()))
            .collect();
        for cell in selected {
            let cell = &self.cells[cell];
            if let Some(cells) =
                cells_by_segment.get_mut(&self.segments[cell.segment_idx as usize].name)
            {
                cells.push(cell.a0_cell);
            }
        }
        Some(ProbePlan {
            cells_by_segment,
            usable_cells,
        })
    }

    /// Start a builder that keeps the segments `keep` accepts, carrying their
    /// models and cells over without re-reading any segment.
    pub fn retain_into_builder(
        &self,
        mut keep: impl FnMut(&DirectorySegment) -> bool,
    ) -> DirectoryBuilder {
        let mut builder = DirectoryBuilder::default();
        let mut kept: HashMap<u32, PendingSegment> = HashMap::new();
        for (idx, segment) in self.segments.iter().enumerate() {
            if !keep(segment) {
                continue;
            }
            let model = segment
                .is_covered()
                .then(|| self.models[segment.model_idx as usize].clone());
            kept.insert(
                idx as u32,
                PendingSegment {
                    name: segment.name.clone(),
                    size_bytes: segment.size_bytes,
                    rows_covered: segment.rows_covered,
                    model,
                    cells: Vec::new(),
                    decode_key: segment.decode_key.clone(),
                },
            );
        }
        for cell in &self.cells {
            if let Some(pending) = kept.get_mut(&cell.segment_idx) {
                pending.cells.push(cell.clone());
            }
        }
        // Inherit the base dimension only from a kept segment that is actually
        // covered (carries a model). A base directory of uncovered segments
        // legitimately has dim 0; propagating that 0 would make every later
        // add_covered bail on "dim differs from 0" and silently turn coverage
        // off for the rest of the collection's life.
        if builder.dim.is_none() && kept.values().any(|p| p.model.is_some()) {
            builder.dim = Some(self.dim);
        }
        for (_, pending) in kept {
            builder.pending.insert(pending.name.clone(), pending);
        }
        builder
    }
}

fn push_f32s(buf: &mut Vec<u8>, values: &[f32]) {
    for v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

#[derive(Debug, Clone)]
struct PendingSegment {
    name: String,
    size_bytes: u64,
    rows_covered: u64,
    model: Option<DirectoryModel>,
    cells: Vec<DirectoryCell>,
    decode_key: Option<SegmentDecodeKey>,
}

/// Accumulates segments into a directory. Order-independent: [`Self::build`]
/// sorts segments by name and deduplicates models by `model_id`.
#[derive(Debug, Clone, Default)]
pub struct DirectoryBuilder {
    dim: Option<u32>,
    pending: HashMap<String, PendingSegment>,
}

impl DirectoryBuilder {
    /// Add a segment whose Region A0 the directory will address. Fails on a
    /// dimension that disagrees with the segments already added; the caller
    /// then records the segment with [`Self::add_uncovered`].
    pub fn add_covered(&mut self, name: &str, size_bytes: u64, a0: &CoarseDirectory) -> Result<()> {
        let dim = a0.model.dim;
        if let Some(expected) = self.dim
            && expected != dim
        {
            bail!("segment {name}: dim {dim} differs from the directory's {expected}");
        }
        if a0.cells.len() != a0.model.k_c() {
            bail!("segment {name}: A0 cell table disagrees with k_c");
        }
        let n_comp = a0.model.n_comp as usize;
        let cells = a0
            .cells
            .iter()
            .enumerate()
            .map(|(c, entry)| DirectoryCell {
                segment_idx: 0, // assigned in build()
                a0_cell: c as u32,
                centroid: a0.model.centroids[c * n_comp..(c + 1) * n_comp].to_vec(),
                radius: a0.model.radii[c],
                entry: *entry,
            })
            .collect();
        self.dim = Some(dim);
        self.pending.insert(
            name.to_owned(),
            PendingSegment {
                name: name.to_owned(),
                size_bytes,
                rows_covered: a0.model.rows_covered(),
                model: Some(DirectoryModel::from_a0(a0)),
                cells,
                decode_key: None,
            },
        );
        Ok(())
    }

    /// Record a segment the directory examined but cannot address, so later
    /// publishes do not re-read it and readers know to scan it per segment.
    pub fn add_uncovered(&mut self, name: &str, size_bytes: u64) {
        self.pending.insert(
            name.to_owned(),
            PendingSegment {
                name: name.to_owned(),
                size_bytes,
                rows_covered: 0,
                model: None,
                cells: Vec::new(),
                decode_key: None,
            },
        );
    }

    /// TD-PAXDIR-1 D6: attach a decode key to a segment added with
    /// [`Self::add_covered`]. Fails (leaving the segment keyless) when the
    /// segment is not covered or the key disagrees with its A0.
    pub fn set_decode_key(&mut self, name: &str, key: SegmentDecodeKey) -> Result<()> {
        let dim = self.dim.unwrap_or(0);
        let pending = self
            .pending
            .get_mut(name)
            .filter(|p| p.model.is_some())
            .ok_or_else(|| anyhow::anyhow!("segment {name} is not covered"))?;
        let rows: u64 = pending.cells.iter().map(DirectoryCell::rows).sum();
        key.check(dim, rows)?;
        pending.decode_key = Some(key);
        Ok(())
    }

    /// Whether any segment has been added.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Assemble the directory for `generation`.
    pub fn build(self, generation: u64) -> Result<PaxCentroidDirectory> {
        let mut pending: Vec<PendingSegment> = self.pending.into_values().collect();
        pending.sort_by(|a, b| a.name.cmp(&b.name));
        let mut models: Vec<DirectoryModel> = Vec::new();
        let mut model_index: HashMap<u64, u32> = HashMap::new();
        let mut segments = Vec::with_capacity(pending.len());
        let mut cells = Vec::new();
        for (segment_idx, segment) in pending.into_iter().enumerate() {
            let model_idx = match segment.model {
                Some(model) => *model_index.entry(model.model_id).or_insert_with(|| {
                    models.push(model);
                    (models.len() - 1) as u32
                }),
                None => NO_MODEL,
            };
            let mut segment_cells = segment.cells;
            segment_cells.sort_by_key(|c| c.a0_cell);
            for mut cell in segment_cells {
                cell.segment_idx = segment_idx as u32;
                cells.push(cell);
            }
            segments.push(DirectorySegment {
                name: segment.name,
                size_bytes: segment.size_bytes,
                rows_covered: segment.rows_covered,
                model_idx,
                decode_key: segment.decode_key,
            });
        }
        let dir = PaxCentroidDirectory {
            generation,
            dim: self.dim.unwrap_or(0),
            models,
            segments,
            cells,
            space_ids: SpaceIds::default(),
        };
        dir.validate()?;
        Ok(dir)
    }
}

/// A live segment as discovery sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveSegment {
    pub name: String,
    pub size_bytes: u64,
}

/// What a publish must do to bring a directory up to date with the live set.
#[derive(Debug)]
pub struct ReconcilePlan {
    /// Base entries still valid, carried over.
    pub builder: DirectoryBuilder,
    /// Live segments the base does not describe (new, or changed size); the
    /// publisher reads their A0 and adds them.
    pub to_load: Vec<LiveSegment>,
    /// Whether the result differs from the base (a new generation is due).
    pub changed: bool,
}

/// Plan an incremental rebuild: keep base entries for segments that are still
/// live with the same size, load the rest. With no base this is a full build.
pub fn plan_reconcile(base: Option<&PaxCentroidDirectory>, live: &[LiveSegment]) -> ReconcilePlan {
    plan_reconcile_keeping(base, live, |_| true)
}

/// [`plan_reconcile`], also reloading every live base entry `still_valid`
/// rejects (TD-PAXDIR-1 D6: a decode key whose recorded ETag no longer
/// matches the listing is evidence the object was rewritten in place, so its
/// cells are stale too).
pub fn plan_reconcile_keeping(
    base: Option<&PaxCentroidDirectory>,
    live: &[LiveSegment],
    still_valid: impl Fn(&DirectorySegment) -> bool,
) -> ReconcilePlan {
    let live_sizes: HashMap<&str, u64> = live
        .iter()
        .map(|s| (s.name.as_str(), s.size_bytes))
        .collect();
    let Some(base) = base else {
        return ReconcilePlan {
            builder: DirectoryBuilder::default(),
            to_load: live.to_vec(),
            changed: true,
        };
    };
    let mut kept = HashSet::new();
    let mut dropped = false;
    let builder = base.retain_into_builder(|segment| {
        let keep = live_sizes.get(segment.name.as_str()) == Some(&segment.size_bytes)
            && still_valid(segment);
        if keep {
            kept.insert(segment.name.clone());
        } else {
            dropped = true;
        }
        keep
    });
    let to_load: Vec<LiveSegment> = live
        .iter()
        .filter(|s| !kept.contains(&s.name))
        .cloned()
        .collect();
    let changed = dropped || !to_load.is_empty();
    ReconcilePlan {
        builder,
        to_load,
        changed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coarse_directory::CoarseModel;

    /// A0 for a segment whose PCA keeps the first `n_comp` axes of `dim`
    /// (orthonormal components) with `mean`, and cells at `centroids`.
    fn a0(
        dim: usize,
        n_comp: usize,
        mean: f32,
        centroids: &[&[f32]],
        rows: &[u64],
    ) -> CoarseDirectory {
        let mut components = vec![0f32; n_comp * dim];
        for c in 0..n_comp {
            components[c * dim + c] = 1.0;
        }
        let k_c = centroids.len();
        let mut cells = Vec::with_capacity(k_c);
        let mut row = 0u64;
        for (c, &n) in rows.iter().enumerate() {
            cells.push(CoarseCellEntry {
                row_begin: row,
                row_end: row + n,
                a_off: 1000 + c as u64 * 64,
                a_len: n * 8,
                b_off: 9000 + c as u64 * 64,
                b_len: n * dim as u64,
                c_off: 0,
                c_len: 0,
                d_block_begin: c as u32,
                d_block_end: c as u32 + 1,
            });
            row += n;
        }
        CoarseDirectory {
            model: CoarseModel {
                dim: dim as u32,
                n_comp: n_comp as u16,
                pca_mean: vec![mean; dim],
                pca_components: components,
                centroids: centroids.iter().flat_map(|c| c.iter().copied()).collect(),
                radii: vec![0.5; k_c],
                cell_rows: rows.to_vec(),
                seed: 7,
                trained_on: rows.iter().sum(),
            },
            cells,
        }
    }

    fn two_model_directory() -> PaxCentroidDirectory {
        let mut b = DirectoryBuilder::default();
        b.add_covered(
            "b.pax",
            200,
            &a0(4, 2, 0.0, &[&[0.0, 0.0], &[5.0, 5.0]], &[10, 20]),
        )
        .unwrap();
        b.add_covered(
            "a.pax",
            100,
            &a0(
                4,
                2,
                1.0,
                &[&[1.0, -1.0], &[-3.0, 2.0], &[0.0, 0.0]],
                &[5, 0, 7],
            ),
        )
        .unwrap();
        b.add_uncovered("l0.pax", 50);
        b.build(3).unwrap()
    }

    #[test]
    fn object_names_round_trip_and_reject_other_objects() {
        assert_eq!(object_name(0x2a), "centroids-000000000000002a.pxcd");
        assert_eq!(parse_object_name(&object_name(u64::MAX)), Some(u64::MAX));
        for other in [
            "centroids-2a.pxcd",
            "centroids-000000000000002a.pax",
            "seg-000000000000002a.pxcd",
            "retire-pending.json",
            "centroids-00000000000000zz.pxcd",
        ] {
            assert_eq!(parse_object_name(other), None, "{other}");
        }
    }

    #[test]
    fn serialization_round_trips_and_is_order_independent() {
        let dir = two_model_directory();
        let bytes = dir.to_bytes().unwrap();
        assert_eq!(PaxCentroidDirectory::parse(&bytes).unwrap(), dir);

        let mut b = DirectoryBuilder::default();
        b.add_uncovered("l0.pax", 50);
        b.add_covered(
            "a.pax",
            100,
            &a0(
                4,
                2,
                1.0,
                &[&[1.0, -1.0], &[-3.0, 2.0], &[0.0, 0.0]],
                &[5, 0, 7],
            ),
        )
        .unwrap();
        b.add_covered(
            "b.pax",
            200,
            &a0(4, 2, 0.0, &[&[0.0, 0.0], &[5.0, 5.0]], &[10, 20]),
        )
        .unwrap();
        assert_eq!(b.build(3).unwrap().to_bytes().unwrap(), bytes);

        assert_eq!(
            dir.segments
                .iter()
                .map(|s| s.name.as_str())
                .collect::<Vec<_>>(),
            ["a.pax", "b.pax", "l0.pax"]
        );
        assert_eq!(dir.models.len(), 2);
        assert!(!dir.segments[2].is_covered());
        assert_eq!(dir.cells.len(), 5);
        assert_eq!(dir.segments[0].rows_covered, 12);
    }

    #[test]
    fn segments_sharing_a_model_share_one_model_entry() {
        let mut b = DirectoryBuilder::default();
        b.add_covered("a.pax", 1, &a0(4, 2, 0.0, &[&[0.0, 0.0]], &[3]))
            .unwrap();
        b.add_covered("b.pax", 1, &a0(4, 2, 0.0, &[&[1.0, 1.0]], &[4]))
            .unwrap();
        let dir = b.build(1).unwrap();
        assert_eq!(dir.models.len(), 1);
        assert!(dir.segments.iter().all(|s| s.model_idx == 0));
    }

    #[test]
    fn parse_fails_closed_on_corruption() {
        let bytes = two_model_directory().to_bytes().unwrap();
        for i in [0usize, 4, 9, bytes.len() / 2, bytes.len() - 1] {
            let mut bad = bytes.clone();
            bad[i] ^= 0x40;
            assert!(PaxCentroidDirectory::parse(&bad).is_err(), "flip at {i}");
        }
        for len in [0, 10, bytes.len() - 1] {
            assert!(
                PaxCentroidDirectory::parse(&bytes[..len]).is_err(),
                "len {len}"
            );
        }
    }

    #[test]
    fn mismatched_dimension_is_rejected_for_coverage() {
        let mut b = DirectoryBuilder::default();
        b.add_covered("a.pax", 1, &a0(4, 2, 0.0, &[&[0.0, 0.0]], &[3]))
            .unwrap();
        assert!(
            b.add_covered("b.pax", 1, &a0(6, 2, 0.0, &[&[0.0, 0.0]], &[3]))
                .is_err()
        );
    }

    /// The score is the squared distance to the centroid's reconstruction.
    #[test]
    fn score_is_the_full_space_distance_to_the_reconstructed_centroid() {
        let dir = two_model_directory();
        let query = [0.3f32, -2.0, 4.0, 1.5];
        for scored in dir.rank_cells(&query).unwrap() {
            let cell = &dir.cells[scored.cell];
            let model = &dir.models[dir.segments[cell.segment_idx as usize].model_idx as usize];
            // μ + Pᵀc with P = the first n_comp axes.
            let mut recon = model.pca_mean.clone();
            for (c, v) in cell.centroid.iter().enumerate() {
                recon[c] += v;
            }
            let expected: f32 = query
                .iter()
                .zip(&recon)
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            assert!(
                (scored.score - expected).abs() < 1e-4,
                "{scored:?} vs {expected}"
            );
        }
    }

    #[test]
    fn within_one_model_rank_order_equals_a0_order() {
        let seg = a0(
            4,
            2,
            0.5,
            &[&[0.0, 0.0], &[2.0, 2.0], &[-1.0, 3.0], &[4.0, -4.0]],
            &[1, 1, 1, 1],
        );
        let mut b = DirectoryBuilder::default();
        b.add_covered("s.pax", 1, &seg).unwrap();
        let dir = b.build(1).unwrap();
        let query = [1.7f32, 0.2, -9.0, 3.0];
        let q = seg.project(&query);
        let mut a0_order: Vec<usize> = (0..4).collect();
        let a0_dist = |c: usize| -> f32 {
            q.iter()
                .zip(&seg.model.centroids[c * 2..c * 2 + 2])
                .map(|(a, b)| (a - b) * (a - b))
                .sum()
        };
        a0_order.sort_by(|a, b| a0_dist(*a).total_cmp(&a0_dist(*b)));
        let ranked: Vec<usize> = dir
            .rank_cells(&query)
            .unwrap()
            .iter()
            .map(|s| dir.cells[s.cell].a0_cell as usize)
            .collect();
        assert_eq!(ranked, a0_order);
    }

    #[test]
    fn selection_honours_nprobe_and_the_row_floor_and_skips_empty_cells() {
        let dir = two_model_directory();
        let query = [1.0f32, -1.0, 1.0, 1.0];
        assert_eq!(dir.rank_cells(&[0.0; 3]), None);
        let one = dir.select_cells(&query, 1, 0).unwrap();
        assert_eq!(one.len(), 1);
        let all = dir.select_cells(&query, 1, u64::MAX).unwrap();
        // Every non-empty cell, never the empty one.
        assert_eq!(all.len(), 4);
        assert!(all.iter().all(|&c| dir.cells[c].rows() > 0));
        let floor = dir.select_cells(&query, 1, 13).unwrap();
        assert!(floor.iter().map(|&c| dir.cells[c].rows()).sum::<u64>() >= 13);
    }

    #[test]
    fn reconcile_keeps_unchanged_segments_and_reloads_the_rest() {
        let base = two_model_directory();
        let live = |items: &[(&str, u64)]| -> Vec<LiveSegment> {
            items
                .iter()
                .map(|(n, s)| LiveSegment {
                    name: (*n).to_owned(),
                    size_bytes: *s,
                })
                .collect()
        };

        let same = plan_reconcile(
            Some(&base),
            &live(&[("a.pax", 100), ("b.pax", 200), ("l0.pax", 50)]),
        );
        assert!(!same.changed);
        assert!(same.to_load.is_empty());
        assert_eq!(same.builder.build(base.generation).unwrap(), base);

        // b.pax retired, a.pax rewritten in place (size change), c.pax new.
        let plan = plan_reconcile(
            Some(&base),
            &live(&[("a.pax", 101), ("c.pax", 9), ("l0.pax", 50)]),
        );
        assert!(plan.changed);
        let names: Vec<&str> = plan.to_load.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["a.pax", "c.pax"]);
        let rebuilt = plan.builder.build(4).unwrap();
        assert_eq!(
            rebuilt
                .segments
                .iter()
                .map(|s| s.name.as_str())
                .collect::<Vec<_>>(),
            ["l0.pax"]
        );

        // Retirement alone is a change.
        let shrink = plan_reconcile(Some(&base), &live(&[("a.pax", 100)]));
        assert!(shrink.changed && shrink.to_load.is_empty());
        let full = plan_reconcile(None, &live(&[("a.pax", 100)]));
        assert!(full.changed && full.to_load.len() == 1);
    }

    /// A base entry the caller rejects is reloaded even when its stamp
    /// matches, and the plan counts as a change.
    #[test]
    fn rejected_base_entries_are_reloaded() {
        let base = two_model_directory();
        let live: Vec<LiveSegment> = [("a.pax", 100), ("b.pax", 200), ("l0.pax", 50)]
            .iter()
            .map(|(n, s)| LiveSegment {
                name: (*n).into(),
                size_bytes: *s,
            })
            .collect();
        let plan = plan_reconcile_keeping(Some(&base), &live, |s| s.name != "b.pax");
        assert!(plan.changed);
        let names: Vec<&str> = plan.to_load.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["b.pax"]);
    }

    /// Incremental reconcile produces exactly the directory a full build does.
    #[test]
    fn incremental_build_equals_full_build() {
        let seg_c = a0(4, 3, 0.0, &[&[1.0, 2.0, 3.0]], &[9]);
        let base = two_model_directory();
        let live = vec![
            LiveSegment {
                name: "a.pax".into(),
                size_bytes: 100,
            },
            LiveSegment {
                name: "c.pax".into(),
                size_bytes: 9,
            },
        ];
        let mut plan = plan_reconcile(Some(&base), &live);
        assert_eq!(plan.to_load.len(), 1);
        plan.builder.add_covered("c.pax", 9, &seg_c).unwrap();
        let incremental = plan.builder.build(4).unwrap();

        let mut full = DirectoryBuilder::default();
        // `a.pax` in the base was built from this A0's first model/cells.
        full.add_covered(
            "a.pax",
            100,
            &a0(
                4,
                2,
                1.0,
                &[&[1.0, -1.0], &[-3.0, 2.0], &[0.0, 0.0]],
                &[5, 0, 7],
            ),
        )
        .unwrap();
        full.add_covered("c.pax", 9, &seg_c).unwrap();
        assert_eq!(
            incremental.to_bytes().unwrap(),
            full.build(4).unwrap().to_bytes().unwrap()
        );
    }

    #[test]
    fn retain_from_uncovered_base_leaves_dim_unset() {
        // A base directory of uncovered-only segments has dim 0; retaining it
        // must not pin the builder's dim to 0 (which would make every later
        // add_covered bail and silently turn coverage off for good).
        let mut base_builder = DirectoryBuilder::default();
        base_builder.add_uncovered("l0.pax", 50);
        let base = base_builder.build(1).unwrap();
        assert_eq!(base.dim, 0);
        let mut builder = base.retain_into_builder(|_| true);
        assert!(
            builder.dim.is_none(),
            "uncovered-only base must not set the builder's dim"
        );
        builder
            .add_covered(
                "c.pax",
                9,
                &a0(4, 2, 0.0, &[&[0.0, 0.0], &[5.0, 5.0]], &[10, 20]),
            )
            .expect("covered segment after uncovered-only base must be accepted");
        let rebuilt = builder.build(4).unwrap();
        assert_eq!(rebuilt.dim, 4);
        assert_eq!(rebuilt.covered_segments().count(), 1);
    }

    #[test]
    fn parse_rejects_non_finite_model_values() {
        let dir = two_model_directory();
        let mut bytes = dir.to_bytes().unwrap();
        // Flip one pca_mean f32 inside the first model to a bit pattern for
        // NaN, then re-stamp the checksum so ONLY the finite check can catch
        // it (exercises a valid checksum over structurally bad values).
        let model_mean_off = FIXED_HEAD_LEN
            + 8 /* model_id */
            + 2 /* n_comp */;
        let f = f32::NAN.to_le_bytes();
        for (i, b) in f.iter().enumerate() {
            bytes[model_mean_off + i] = *b;
        }
        let body_len = bytes.len() - CHECKSUM_LEN;
        let sum = fnv1a64(&bytes[..body_len]);
        bytes[body_len..].copy_from_slice(&sum.to_le_bytes());
        let err = PaxCentroidDirectory::parse(&bytes).unwrap_err().to_string();
        assert!(
            err.contains("non-finite"),
            "expected the finite check to fire, got: {err}"
        );
    }

    #[test]
    fn parse_rejects_unknown_flags_and_nonzero_reserved() {
        let bytes = two_model_directory().to_bytes().unwrap();

        let mut flagged = bytes.clone();
        flagged[5] = 0b0000_0001;
        let body_len = flagged.len() - CHECKSUM_LEN;
        let sum = fnv1a64(&flagged[..body_len]);
        flagged[body_len..].copy_from_slice(&sum.to_le_bytes());
        let err = PaxCentroidDirectory::parse(&flagged)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("unknown centroid directory flags"),
            "expected the flags check to fire, got: {err}"
        );

        let mut reserved = bytes;
        reserved[6] = 0x7f;
        let sum = fnv1a64(&reserved[..body_len]);
        reserved[body_len..].copy_from_slice(&sum.to_le_bytes());
        let err = PaxCentroidDirectory::parse(&reserved)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("reserved bytes must be zero"),
            "expected the reserved check to fire, got: {err}"
        );
    }

    #[test]
    fn rank_cells_clamps_non_finite_scores_to_infinity() {
        // A directory written before the parse-time finite check (or poisoned
        // in memory) must not let its NaN-scored cells win nprobe slots:
        // total_cmp ranks -NaN FIRST, so the score is clamped to +inf.
        // Assembled as a struct literal because build()/parse() now reject
        // non-finite values outright.
        let identity_2x4 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let models = vec![
            DirectoryModel {
                model_id: 1,
                n_comp: 2,
                pca_mean: vec![f32::NAN, 0.0, 0.0, 0.0],
                pca_components: identity_2x4.clone(),
            },
            DirectoryModel {
                model_id: 2,
                n_comp: 2,
                pca_mean: vec![0.0; 4],
                pca_components: identity_2x4,
            },
        ];
        let segments = vec![
            DirectorySegment {
                name: "bad.pax".to_string(),
                size_bytes: 100,
                rows_covered: 10,
                model_idx: 0,
                decode_key: None,
            },
            DirectorySegment {
                name: "good.pax".to_string(),
                size_bytes: 100,
                rows_covered: 10,
                model_idx: 1,
                decode_key: None,
            },
        ];
        let entry = CoarseCellEntry {
            row_begin: 0,
            row_end: 10,
            a_off: 0,
            a_len: 80,
            b_off: 0,
            b_len: 320,
            c_off: 0,
            c_len: 0,
            d_block_begin: 0,
            d_block_end: 1,
        };
        let cells = vec![
            DirectoryCell {
                segment_idx: 0,
                a0_cell: 0,
                centroid: vec![0.0, 0.0],
                radius: 0.5,
                entry,
            },
            DirectoryCell {
                segment_idx: 1,
                a0_cell: 0,
                centroid: vec![0.0, 0.0],
                radius: 0.5,
                entry,
            },
        ];
        let poisoned = PaxCentroidDirectory {
            generation: 7,
            dim: 4,
            models,
            segments,
            cells,
            space_ids: SpaceIds::default(),
        };

        let ranked = poisoned
            .rank_cells(&[0.0, 0.0, 0.0, 0.0])
            .expect("dimension matches");
        assert_eq!(
            ranked.last().expect("cells ranked").score,
            f32::INFINITY,
            "the NaN-model cell must be clamped to +inf"
        );
        assert!(
            ranked[..ranked.len() - 1]
                .iter()
                .all(|s| s.score.is_finite()),
            "only the poisoned cell may sit at +inf"
        );
    }

    #[test]
    fn a_probe_plan_uses_only_live_unchanged_segments() {
        let dir = two_model_directory();
        let query = [1.0f32, -1.0, 1.0, 1.0];
        let live = |items: &[(&str, u64)]| -> Vec<LiveSegment> {
            items
                .iter()
                .map(|(n, s)| LiveSegment {
                    name: (*n).to_owned(),
                    size_bytes: *s,
                })
                .collect()
        };
        // Every usable segment appears; the uncovered one never does.
        let all = dir
            .plan_probe(
                &query,
                &live(&[("a.pax", 100), ("b.pax", 200), ("l0.pax", 50)]),
                |_| 99,
                |_| 0,
            )
            .unwrap();
        assert_eq!(all.usable_cells, 4);
        assert_eq!(all.cells_by_segment.len(), 2);
        assert_eq!(
            all.cells_by_segment["a.pax"].len(),
            2,
            "the empty A0 cell is never probed"
        );
        assert_eq!(all.cells_by_segment["b.pax"].len(), 2);

        // One probe: one segment gets the nearest cell, the other is skipped.
        let one = dir
            .plan_probe(
                &query,
                &live(&[("a.pax", 100), ("b.pax", 200)]),
                |_| 1,
                |_| 0,
            )
            .unwrap();
        let mut sizes: Vec<usize> = one.cells_by_segment.values().map(Vec::len).collect();
        sizes.sort();
        assert_eq!(sizes, [0, 1]);

        // A rewritten (size changed) segment is unusable; a retired one too.
        let stale = dir.plan_probe(
            &query,
            &live(&[("a.pax", 101), ("c.pax", 9)]),
            |_| 99,
            |_| 0,
        );
        assert_eq!(stale, None, "nothing usable");
        let partial = dir
            .plan_probe(
                &query,
                &live(&[("a.pax", 101), ("b.pax", 200)]),
                |n| n,
                |_| 0,
            )
            .unwrap();
        assert_eq!(partial.usable_cells, 2);
        assert!(!partial.cells_by_segment.contains_key("a.pax"));
        assert_eq!(
            dir.plan_probe(&[0.0; 3], &live(&[("b.pax", 200)]), |n| n, |_| 0),
            None
        );
    }

    /// Segments written with one shared model hold the same cells: a
    /// selected cell is probed in every segment holding it and counts once.
    #[test]
    fn shared_model_cells_count_once_and_expand_to_every_segment() {
        let cells: &[&[f32]] = &[&[0.0, 0.0], &[5.0, 5.0], &[-5.0, 5.0]];
        let mut b = DirectoryBuilder::default();
        b.add_covered("s1.pax", 10, &a0(4, 2, 0.0, cells, &[4, 4, 4]))
            .unwrap();
        b.add_covered("s2.pax", 20, &a0(4, 2, 0.0, cells, &[3, 0, 9]))
            .unwrap();
        let dir = b.build(1).unwrap();
        assert_eq!(dir.models.len(), 1, "one shared model");
        let live = [
            LiveSegment {
                name: "s1.pax".into(),
                size_bytes: 10,
            },
            LiveSegment {
                name: "s2.pax".into(),
                size_bytes: 20,
            },
        ];
        let query = [0.1f32, 0.1, 0.0, 0.0];
        let plan = dir.plan_probe(&query, &live, |n| n, |_| 0).unwrap();
        assert_eq!(plan.usable_cells, 3, "distinct cells, not 5 entries");
        let one = dir.plan_probe(&query, &live, |_| 1, |_| 0).unwrap();
        assert_eq!(one.cells_by_segment["s1.pax"], [0]);
        assert_eq!(
            one.cells_by_segment["s2.pax"],
            [0],
            "same cell in both segments"
        );
        // The row floor counts a cell's rows across all its segments: cell 0
        // holds 4 + 3 rows, so a floor of 8 needs a second cell.
        let floor = dir.plan_probe(&query, &live, |_| 1, |_| 8).unwrap();
        assert_eq!(floor.cells_by_segment["s1.pax"].len(), 2);
    }

    #[test]
    fn the_collection_model_is_the_model_covering_most_rows() {
        let shared: &[&[f32]] = &[&[0.0, 0.0], &[5.0, 5.0], &[-5.0, 5.0]];
        let mut b = DirectoryBuilder::default();
        let big = a0(4, 2, 0.0, shared, &[40, 0, 9]);
        b.add_covered("s1.pax", 10, &big).unwrap();
        let mut s2 = a0(4, 2, 0.0, shared, &[1, 2, 3]);
        s2.model.radii = vec![0.25, 2.0, 0.75];
        b.add_covered("s2.pax", 20, &s2).unwrap();
        b.add_covered("other.pax", 30, &a0(4, 2, 1.0, &[&[1.0, 1.0]], &[30]))
            .unwrap();
        b.add_uncovered("l0.pax", 5);
        let dir = b.build(1).unwrap();
        let shared_space = dir.collection_model().unwrap();
        assert_eq!(
            shared_space.covered_rows, 55,
            "s1 + s2 share one cell space"
        );
        let model = shared_space.model;
        assert_eq!(model.k_c(), 3);
        assert_eq!(model.pca_mean, big.model.pca_mean);
        assert_eq!(model.pca_components, big.model.pca_components);
        assert_eq!(model.centroids, big.model.centroids);
        assert_eq!(
            model.radii, big.model.radii,
            "radii are the largest segment's, never a max that ratchets"
        );
        assert_eq!(model.cell_rows, [40, 0, 9]);
        assert!(
            PaxCentroidDirectory {
                generation: 1,
                dim: 4,
                models: Vec::new(),
                segments: Vec::new(),
                cells: Vec::new(),
                space_ids: SpaceIds::default(),
            }
            .collection_model()
            .is_none()
        );
    }

    /// A decode key consistent with `seg` (the `a0` helper's layout), with
    /// one block per cell and an OID chunk on every block.
    fn key_for(seg: &CoarseDirectory) -> SegmentDecodeKey {
        let dim = seg.model.dim;
        let rows = seg.model.rows_covered();
        let a0_len = seg.to_bytes().unwrap().len() as u64;
        let header = SegmentHeaderPrefix {
            layout_version: crate::segment_layout::SEG_LAYOUT_VERSION,
            rabitq_off: 88 + a0_len,
            rabitq_len: 500,
            sq8_off: 2000,
            sq8_len: 900,
            footer_off: 5000,
            footer_len: 300,
            a0_off: 88,
            a0_len,
            c_off: 0,
            c_len: 0,
        };
        let mut rabitq_header = Vec::new();
        rabitq_header.extend_from_slice(&(rows as u32).to_le_bytes());
        rabitq_header.extend_from_slice(&dim.to_le_bytes());
        rabitq_header.extend_from_slice(&42u64.to_le_bytes());
        for d in 0..dim {
            rabitq_header.extend_from_slice(&(d as f32 * 0.5).to_le_bytes());
        }
        let blocks: Vec<FooterBlockEntry> = seg
            .model
            .cell_rows
            .iter()
            .enumerate()
            .map(|(c, &n)| FooterBlockEntry {
                offset: 3000 + c as u64 * 100,
                size: 100,
                row_count: n as u32,
                stats_kind: crate::segment_layout::StatsKind::MinMax,
            })
            .collect();
        let block_stats = blocks
            .iter()
            .enumerate()
            .map(|(c, b)| {
                Some(FooterRowGroupStats {
                    zone: crate::pax_block::BlockZoneSummary::empty(b.row_count),
                    oid_chunk_rel_off: 10 + c as u32,
                    oid_chunk_len: 20,
                    oid_encoding_id: 3,
                    oid_is_lz4: c % 2 == 0,
                })
            })
            .collect();
        SegmentDecodeKey {
            header_prefix: header.to_bytes(),
            a0_seed: seg.model.seed,
            a0_trained_on: seg.model.trained_on,
            rabitq_header,
            row_count: rows,
            sq8_min: -1.5,
            sq8_scale: 0.01,
            embed_dim: dim,
            embed_count: 1,
            embed_quant_tag: 1,
            has_f32_tier: false,
            blocks,
            block_stats,
            object_etag: Some("\"etag-1\"".to_owned()),
        }
    }

    fn keyed_directory() -> (PaxCentroidDirectory, CoarseDirectory) {
        let seg = a0(
            4,
            2,
            0.5,
            &[&[0.0, 0.0], &[5.0, 5.0], &[1.0, 2.0]],
            &[10, 0, 7],
        );
        let mut b = DirectoryBuilder::default();
        b.add_covered("k.pax", 300, &seg).unwrap();
        b.set_decode_key("k.pax", key_for(&seg)).unwrap();
        b.add_covered("plain.pax", 100, &a0(4, 2, 1.0, &[&[1.0, 1.0]], &[4]))
            .unwrap();
        b.add_uncovered("l0.pax", 50);
        (b.build(9).unwrap(), seg)
    }

    /// TD-PAXDIR-1 D6: a decode key round-trips, and rebuilds the segment's
    /// Region A0 byte for byte plus a footer with the ANN path's fields.
    #[test]
    fn decode_keys_round_trip_and_rebuild_the_segment_metadata() {
        let (dir, seg) = keyed_directory();
        let bytes = dir.to_bytes().unwrap();
        assert_eq!(bytes[4], DIRECTORY_VERSION);
        let parsed = PaxCentroidDirectory::parse(&bytes).unwrap();
        assert_eq!(parsed, dir);
        assert_eq!(parsed.to_bytes().unwrap(), bytes, "deterministic");

        let decoded = parsed.segment_decode(0).unwrap();
        assert_eq!(decoded.a0, seg.to_bytes().unwrap(), "A0 rebuilt exactly");
        let key = parsed.segments[0].decode_key.as_ref().unwrap();
        assert_eq!(decoded.header_prefix, key.header_prefix);
        assert_eq!(decoded.rabitq_header, key.rabitq_header);
        let footer = SegmentFooterIndex::parse(&decoded.ann_footer).unwrap();
        assert_eq!(footer.row_count, 17);
        assert_eq!((footer.sq8_min, footer.sq8_scale), (-1.5, 0.01));
        assert_eq!(footer.embed_dim, 4);
        assert_eq!(footer.blocks, key.blocks);
        assert_eq!(footer.block_stats, key.block_stats);
        let header = SegmentHeaderPrefix::parse(&decoded.header_prefix).unwrap();
        assert_eq!(
            (footer.sq8_off, footer.sq8_len),
            (header.sq8_off, header.sq8_len)
        );

        assert!(key.matches_object(Some("\"etag-1\"")));
        assert!(!key.matches_object(Some("\"etag-2\"")), "rewritten object");
        assert!(!key.matches_object(None), "a listing without an ETag");

        // Keyless and uncovered segments have nothing to decode.
        assert!(parsed.segment_decode(1).is_err());
        assert!(parsed.segment_decode(2).is_err());
        assert!(parsed.segment_decode(99).is_err());
    }

    /// Keys survive an incremental rebuild that keeps the segment, and
    /// leave with it.
    #[test]
    fn decode_keys_are_carried_by_reconcile() {
        let (dir, _) = keyed_directory();
        let live = |names: &[(&str, u64)]| -> Vec<LiveSegment> {
            names
                .iter()
                .map(|(n, s)| LiveSegment {
                    name: (*n).into(),
                    size_bytes: *s,
                })
                .collect()
        };
        let kept = plan_reconcile(Some(&dir), &live(&[("k.pax", 300), ("plain.pax", 100)]));
        let rebuilt = kept.builder.build(10).unwrap();
        assert!(rebuilt.segments[0].decode_key.is_some());
        assert_eq!(
            rebuilt.segment_decode(0).unwrap(),
            dir.segment_decode(0).unwrap()
        );
        let rewritten = plan_reconcile(Some(&dir), &live(&[("k.pax", 301)]));
        assert!(rewritten.builder.build(10).unwrap().segments.is_empty());
    }

    /// A key that disagrees with its segment is refused at attach and at
    /// serialize (fail closed), and a v1 object (no key bytes) still parses.
    #[test]
    fn inconsistent_keys_are_rejected_and_v1_still_parses() {
        let seg = a0(4, 2, 0.5, &[&[0.0, 0.0], &[5.0, 5.0]], &[10, 7]);
        let mut b = DirectoryBuilder::default();
        b.add_covered("k.pax", 300, &seg).unwrap();
        b.add_uncovered("l0.pax", 50);
        let mut wrong_rows = key_for(&seg);
        wrong_rows.blocks[0].row_count += 1;
        assert!(b.set_decode_key("k.pax", wrong_rows).is_err());
        let mut wrong_header = key_for(&seg);
        wrong_header.rabitq_header.truncate(16);
        assert!(b.set_decode_key("k.pax", wrong_header).is_err());
        let mut wrong_dim = key_for(&seg);
        wrong_dim.embed_dim = 5;
        assert!(b.set_decode_key("k.pax", wrong_dim).is_err());
        let mut overlapping = key_for(&seg);
        overlapping.blocks[1].offset = overlapping.blocks[0].offset + 1;
        assert!(b.set_decode_key("k.pax", overlapping).is_err());
        assert!(b.set_decode_key("l0.pax", key_for(&seg)).is_err());
        assert!(b.set_decode_key("absent.pax", key_for(&seg)).is_err());

        let mut dir = b.clone().build(1).unwrap();
        let mut bad = key_for(&seg);
        bad.row_count = 99;
        dir.segments[0].decode_key = Some(bad);
        assert!(dir.to_bytes().is_err(), "serialize validates keys too");

        // A keyless directory is written as v1 — the D1 layout, with no
        // per-segment key flag — and parses back unchanged.
        let plain = b.build(1).unwrap();
        let v1 = plain.to_bytes().unwrap();
        assert_eq!(v1[4], DIRECTORY_VERSION_V1);
        let segments_end =
            v1.len() - CHECKSUM_LEN - plain.cells.len() * (8 + 2 * 4 + 4 + CELL_ENTRY_LEN);
        let segments_len: usize = plain
            .segments
            .iter()
            .map(|s| 4 + s.name.len() + 8 + 8 + 4)
            .sum();
        let names_at = segments_end - segments_len;
        assert_eq!(
            &v1[names_at + 4..names_at + 4 + plain.segments[0].name.len()],
            plain.segments[0].name.as_bytes(),
            "segments carry no key flag byte in v1"
        );
        let parsed = PaxCentroidDirectory::parse(&v1).unwrap();
        assert_eq!(parsed, plain);
        assert!(parsed.segments.iter().all(|s| s.decode_key.is_none()));
    }

    /// Equal PCA bytes do not make equal cells: two segments with the same
    /// projection but different centroids are distinct cell spaces, so their
    /// cells never merge (the D2 read behaviour) and the shared model is one
    /// of them, never a blend.
    #[test]
    fn same_pca_with_different_centroids_are_distinct_cell_spaces() {
        let mut b = DirectoryBuilder::default();
        b.add_covered(
            "a.pax",
            1,
            &a0(4, 2, 0.0, &[&[0.0, 0.0], &[3.0, 3.0]], &[5, 5]),
        )
        .unwrap();
        b.add_covered("b.pax", 1, &a0(4, 2, 0.0, &[&[9.0, 9.0]], &[30]))
            .unwrap();
        let dir = b.build(1).unwrap();
        assert_eq!(dir.models.len(), 1, "one PCA entry");
        let live = [
            LiveSegment {
                name: "a.pax".into(),
                size_bytes: 1,
            },
            LiveSegment {
                name: "b.pax".into(),
                size_bytes: 1,
            },
        ];
        let plan = dir.plan_probe(&[0.0; 4], &live, |n| n, |_| 0).unwrap();
        assert_eq!(plan.usable_cells, 3, "cell 0 of a and cell 0 of b differ");
        let one = dir.plan_probe(&[0.0; 4], &live, |_| 1, |_| 0).unwrap();
        assert_eq!(one.cells_by_segment["a.pax"], [0]);
        assert!(one.cells_by_segment["b.pax"].is_empty());
        let shared = dir.collection_model().unwrap();
        assert_eq!(shared.model.k_c(), 1, "b's space covers more rows");
        assert_eq!(shared.covered_rows, 30);
    }
}
