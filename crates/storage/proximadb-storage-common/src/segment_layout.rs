//! Coalesced-RaBitQ segment layout — header-prefix + self-describing footer-index
//! (ADR-062 / TD-RDSTRAT-6 PR1).
//!
//! The new PAX segment layout is Parquet-style: a tiny header-prefix pinned at
//! offset 0 (so the RaBitQ hot-scan is one ranged GET with zero metadata
//! dependency), the data blocks, then a self-describing **footer-index** at the
//! tail (schema snapshot + block table + per-stripe encoding map + extensible
//! per-block stats), located via `[footer_len u64][SEGMENT_MAGIC]`.
//!
//! ```text
//! [HEADER-PREFIX 56B]                 ◄ one GET coalesces header + RaBitQ scan
//! [Region A: coalesced RaBitQ region] ◄ HOT SCAN, keep=100%
//! [Region B: coalesced SQ8 region]    ◄ rerank tier (ADR-065); pure dense SQ8
//! [Block0 … BlockK]                   ◄ row data (OID/props/…) + optional f32; NO SQ8 stripe
//! [FOOTER-INDEX]                      ◄ tail GET, cached
//! [footer_len u64][SEGMENT_MAGIC 8B]
//! ```
//!
//! Same segment magic (`PAXSEG01`) as the legacy `[blocks][SegmentIndex][magic]`
//! layout; a distinct `SEG_HEADER_MAGIC` at offset 0 is the **presence-field**
//! that selects the scan-then-rerank read path (new) vs the legacy in-block
//! RaBitQ path — mixed-read-safe (ADR-061 pre-GA amendment). Readers default to
//! the legacy path when the marker is absent, so already-written segments stay
//! readable.

#![forbid(unsafe_code)]

use anyhow::{Result, bail};
use std::collections::HashMap;

use crate::pax_block::SEGMENT_MAGIC;
use proximadb_block_format::BLOCK_MAGIC;

/// Segment-header magic — the presence-field for the coalesced-RaBitQ layout.
/// Chosen disjoint from the block magic `PBLK` (`0x50…`) so detection is
/// unambiguous: a legacy segment starts with `PBLK` (its first block); a
/// coalesced segment starts with `PXH1`.
pub const SEG_HEADER_MAGIC: &[u8; 4] = b"PXH1";

/// Apache Parquet file magic (`PAR1`), present at BOTH ends of a Parquet file
/// (offset 0 and the last four bytes, after the footer length). Disjoint from
/// `PBLK`, `PXH1` and the legacy version/compression byte range `0x01..=0x0E`,
/// so the four recognised heads never collide.
///
/// TD-USUB-5: before this arm existed, a Parquet object reaching
/// [`SegmentFormat::detect`] fell through to the legacy `ProximaBlocks` default
/// and was handed to a decoder that cannot read it — the exact class of
/// silently-wrong routing mandate #1 forbids. Naming the format makes the
/// outcome honest: either it decodes as Parquet, or it errors as Parquet.
pub const PARQUET_MAGIC: &[u8; 4] = b"PAR1";

/// Header layout version — THE coalesced segment layout (TD-PAXRG-1 collapse,
/// 2026-08): one version, region presence declared by the serialized extents.
///
/// `[prefix][A0?][A][B][C?][OPR?][Region D: row groups][footer]`. The 88 B
/// prefix ALWAYS carries the full region-extent table (`a0_off/len`,
/// `c_off/len` — zero = region absent); the version byte is therefore
/// vestigial today and RESERVED for the first post-GA successor layout
/// (pre-GA there is no serialized legacy data to stay read-compatible with,
/// so the pre-collapse v1/v3/v4 sprawl was deleted rather than maintained —
/// ADR-065's own invariant: region presence is declared by the header/footer,
/// never by a version or flag).
pub const SEG_LAYOUT_VERSION: u8 = 1;

/// v1: `[magic 4][version 1][pad 3][rabitq_off 8][rabitq_len 8][sq8_off 8]
/// [sq8_len 8][footer_off 8][footer_len 8][a0_off 8][a0_len 8][c_off 8]
/// [c_len 8]`. Readers fetch this length (clamped to file size) so one GET
/// covers the whole prefix.
pub const SEG_HEADER_PREFIX_LEN: usize = 88;

/// True iff `bytes` carries the coalesced-RaBitQ header-prefix at offset 0 — the
/// presence-field that selects the scan-then-rerank read path (mixed-read).
pub fn is_coalesced_segment(bytes: &[u8]) -> bool {
    bytes.len() >= SEG_HEADER_MAGIC.len() && &bytes[..SEG_HEADER_MAGIC.len()] == SEG_HEADER_MAGIC
}

/// On-disk format of a persisted vector segment.
///
/// PAX segments (columnar; carry SQ8 / RaBitQ codes for quantized ANN) are
/// recognised by magic; anything else is the legacy row-based `ProximaDataBlock`
/// (raw f32). Detection is unambiguous — the PAX head (`PBLK` or the coalesced
/// header `PXH1`) is disjoint from the legacy version/compression byte — so the
/// legacy path is never mis-routed.
///
/// This is the format-layer detection primitive, factored out of the root
/// `segment_format` module so it is unit-testable without linking the root crate;
/// the root re-exports it and its mixed-format read router builds on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum SegmentFormat {
    /// Legacy row-based block (raw f32). The default for segments written before P3
    /// and for any manifest entry that predates the format field.
    #[default]
    ProximaBlocks,
    /// Columnar PAX segment (carries SQ8 / RaBitQ codes; enables quantized ANN).
    Pax,
    /// Apache Parquet file — the relational/warehouse landing format under
    /// ADR-094 "format follows modality", and the Iceberg interop seam.
    /// Recognised by `PAR1` at both ends (TD-USUB-5).
    Parquet,
}

impl SegmentFormat {
    /// Detect a persisted segment's format from its raw bytes, mixed-read-safe.
    ///
    /// Precedence is PAX → Parquet → legacy, and the three recognisers are
    /// **disjoint by construction**: PAX requires a `PBLK`/`PXH1` head AND a
    /// `PAXSEG01` tail; Parquet requires a `PAR1` head AND a `PAR1` tail; the
    /// legacy format starts with the version/compression byte `0x01..=0x0E`.
    /// Because each recogniser constrains the head, order cannot change any
    /// verdict — the ordering below is documentation, not logic.
    ///
    /// Returns [`SegmentFormat::ProximaBlocks`] for anything not recognisably PAX
    /// or Parquet, so the legacy path is never mis-routed on truncated or unknown
    /// input. Note this default is a *routing* choice, not a claim of validity: a
    /// hybrid such as a PAX head with a `PAR1` tail satisfies neither recogniser
    /// and reaches the legacy decoder, which rejects it (the head is not a legal
    /// version byte) — an honest error rather than a decode of the wrong format.
    pub fn detect(bytes: &[u8]) -> Self {
        if is_pax_segment(bytes) {
            SegmentFormat::Pax
        } else if is_parquet_file(bytes) {
            SegmentFormat::Parquet
        } else {
            SegmentFormat::ProximaBlocks
        }
    }
}

/// True iff `bytes` is a PAX segment: a recognised PAX head AND the trailing
/// segment magic. Both ends are checked so a stray prefix or suffix alone can't
/// false-positive. The head is EITHER the legacy block magic `PBLK` (block 0 at
/// offset 0) OR the coalesced-RaBitQ header magic (ADR-062) — both tail with
/// [`SEGMENT_MAGIC`], so the trailing check is the common anchor.
fn is_pax_segment(bytes: &[u8]) -> bool {
    bytes.len() >= BLOCK_MAGIC.len() + SEGMENT_MAGIC.len()
        && (bytes.starts_with(&BLOCK_MAGIC) || is_coalesced_segment(bytes))
        && bytes.ends_with(SEGMENT_MAGIC)
}

/// True iff `bytes` is an Apache Parquet file: `PAR1` at offset 0 AND `PAR1` as
/// the trailing four bytes. Both ends are required, and the length bound rejects
/// a lone four-byte `PAR1` (which would otherwise satisfy `starts_with` and
/// `ends_with` simultaneously) — the same trap [`is_pax_segment`] guards.
///
/// Detection is not validation: a file passing this check may still be truncated
/// or corrupt, and the Parquet reader reports that as a Parquet error.
pub fn is_parquet_file(bytes: &[u8]) -> bool {
    bytes.len() >= 2 * PARQUET_MAGIC.len()
        && bytes.starts_with(PARQUET_MAGIC)
        && bytes.ends_with(PARQUET_MAGIC)
}

/// The fixed header-prefix at offset 0 (88 B, single form). The RaBitQ scan
/// GET reads `[0, rabitq_off + rabitq_len]`, so the prefix coalesces into that
/// one GET (`[0, a0_off + a0_len]` fetches prefix + coarse directory when A0
/// is present).
#[derive(Debug, Clone)]
pub struct SegmentHeaderPrefix {
    pub layout_version: u8,
    /// Byte offset of the coalesced RaBitQ region (Region A; v1: right after
    /// the prefix, v3: after Region A0).
    pub rabitq_off: u64,
    /// Byte length of the coalesced RaBitQ region (Region A).
    pub rabitq_len: u64,
    /// Byte offset of the coalesced SQ8 region (Region B, ADR-065) — the rerank
    /// tier, hoisted out of blocks so survivor fetches read pure dense SQ8.
    pub sq8_off: u64,
    /// Byte length of the coalesced SQ8 region (Region B).
    pub sq8_len: u64,
    /// Byte offset of the footer-index.
    pub footer_off: u64,
    /// Byte length of the footer-index.
    pub footer_len: u64,
    /// Byte offset of Region A0 (the TD-RDSTRAT-8 coarse directory). `0` =
    /// no coarse level.
    pub a0_off: u64,
    /// Byte length of Region A0. `0` = absent.
    pub a0_len: u64,
    /// Byte offset of Region C (the ADR-065 exact-f32 tier, hoisted when the
    /// collection opted into `pax_f32_tier`). `0` = absent.
    pub c_off: u64,
    /// Byte length of Region C. `0` = absent.
    pub c_len: u64,
}

impl SegmentHeaderPrefix {
    /// Serialize — the single 88 B form; region extents are always present
    /// (zero = region absent).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = vec![0u8; SEG_HEADER_PREFIX_LEN];
        buf[..4].copy_from_slice(SEG_HEADER_MAGIC);
        buf[4] = self.layout_version;
        // buf[5..8] reserved (zero).
        buf[8..16].copy_from_slice(&self.rabitq_off.to_le_bytes());
        buf[16..24].copy_from_slice(&self.rabitq_len.to_le_bytes());
        buf[24..32].copy_from_slice(&self.sq8_off.to_le_bytes());
        buf[32..40].copy_from_slice(&self.sq8_len.to_le_bytes());
        buf[40..48].copy_from_slice(&self.footer_off.to_le_bytes());
        buf[48..56].copy_from_slice(&self.footer_len.to_le_bytes());
        buf[56..64].copy_from_slice(&self.a0_off.to_le_bytes());
        buf[64..72].copy_from_slice(&self.a0_len.to_le_bytes());
        buf[72..80].copy_from_slice(&self.c_off.to_le_bytes());
        buf[80..88].copy_from_slice(&self.c_len.to_le_bytes());
        buf
    }

    /// Parse + validate (magic + version). Fail-closed on a wrong magic, an
    /// unknown version, or truncation — never mis-reads a legacy segment as
    /// coalesced, and a version-1-only binary rejects a v3 segment here (the
    /// TD-RDSTRAT-8 mixed-read contract) rather than mis-probing it.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < SEG_HEADER_PREFIX_LEN {
            bail!("coalesced segment header too short: {}", bytes.len());
        }
        if &bytes[..4] != SEG_HEADER_MAGIC {
            bail!("not a coalesced-RaBitQ segment (bad header magic)");
        }
        let layout_version = bytes[4];
        if layout_version != SEG_LAYOUT_VERSION {
            bail!(
                "unsupported coalesced segment layout version {layout_version} (expected {SEG_LAYOUT_VERSION})"
            );
        }
        let required = SEG_HEADER_PREFIX_LEN;
        if bytes.len() < required {
            bail!(
                "coalesced segment header too short for layout version {layout_version}: {}",
                bytes.len()
            );
        }
        let a0_off = u64::from_le_bytes(bytes[56..64].try_into()?);
        let a0_len = u64::from_le_bytes(bytes[64..72].try_into()?);
        let c_off = u64::from_le_bytes(bytes[72..80].try_into()?);
        let c_len = u64::from_le_bytes(bytes[80..88].try_into()?);
        Ok(Self {
            layout_version,
            rabitq_off: u64::from_le_bytes(bytes[8..16].try_into()?),
            rabitq_len: u64::from_le_bytes(bytes[16..24].try_into()?),
            sq8_off: u64::from_le_bytes(bytes[24..32].try_into()?),
            sq8_len: u64::from_le_bytes(bytes[32..40].try_into()?),
            footer_off: u64::from_le_bytes(bytes[40..48].try_into()?),
            footer_len: u64::from_le_bytes(bytes[48..56].try_into()?),
            a0_off,
            a0_len,
            c_off,
            c_len,
        })
    }
}

/// Extensible per-block per-column stats kind (Parquet-style `Statistics`
/// control plane). Magic/format-family-independent: adding a kind is
/// forward-compatible (unknown kinds are skipped by the reader), never a magic
/// bump. v1 emits [`StatsKind::None`] (the OID-bloom fast path is PR3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum StatsKind {
    /// No per-block stats payload (the PR1 default).
    None = 0,
    /// Per-block OID bloom (no false negatives → `vector_by_id` skips to the one
    /// positive block). Populated in PR3; the structure is emitted now.
    BloomOid = 1,
    /// MinMax zone map (scalar + vector predicate pushdown). Deferred.
    MinMax = 2,
    /// Combined OID bloom + MinMax.
    BloomOidMinMax = 3,
}

impl StatsKind {
    /// Decode from a tag byte; unknown tags degrade to [`StatsKind::None`]
    /// (forward-compatible — a newer writer's stats are ignored, never a
    /// hard error).
    pub fn from_tag(tag: u8) -> Self {
        match tag {
            1 => Self::BloomOid,
            2 => Self::MinMax,
            3 => Self::BloomOidMinMax,
            _ => Self::None,
        }
    }
}

/// Per-row-group stats payload carried in a v4 footer block-table entry's
/// opaque stats field (`StatsKind::MinMax` — TD-PAXRG-1). Old parsers skip
/// stats payloads wholesale, and v1/v3 writers emit zero-length ones, so this
/// structure is additive on the wire in both directions.
#[derive(Debug, Clone, PartialEq)]
pub struct FooterRowGroupStats {
    /// The row group's canonical-column zone map — the same 85 B summary the
    /// v2 SegmentIndex carries per block, so `BlockZoneSource` pruning works
    /// identically at RG granularity (DataFusion ranged arm, TD-OLAP-1).
    pub zone: crate::pax_block::BlockZoneSummary,
    /// RG-relative byte offset of this row group's OID string chunk within its
    /// RG extent. The ANN top-k OID fetch reads `[chunk]` instead of the whole
    /// RG (Phase D) — the Region-D analogue of Region B's row-range planner.
    pub oid_chunk_rel_off: u32,
    /// Byte length of the OID chunk.
    pub oid_chunk_len: u32,
    /// The OID stripe's `encoding_id` (PAX string codec) — the chunk decodes
    /// standalone via `decode_str_chunk`.
    pub oid_encoding_id: u8,
    /// The OID stripe's LZ4 flag.
    pub oid_is_lz4: bool,
}

/// `zone (85 B) + [rel_off u32][len u32][encoding_id u8][lz4 u8]`.
pub const RG_STATS_MIN_BYTES: usize = 95;

impl FooterRowGroupStats {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(RG_STATS_MIN_BYTES);
        self.zone.write_to(&mut buf);
        buf.extend_from_slice(&self.oid_chunk_rel_off.to_le_bytes());
        buf.extend_from_slice(&self.oid_chunk_len.to_le_bytes());
        buf.push(self.oid_encoding_id);
        buf.push(self.oid_is_lz4 as u8);
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < RG_STATS_MIN_BYTES {
            bail!("row-group stats payload truncated: {}", data.len());
        }
        Ok(Self {
            zone: crate::pax_block::BlockZoneSummary::read_from(data)?,
            oid_chunk_rel_off: u32::from_le_bytes(data[85..89].try_into()?),
            oid_chunk_len: u32::from_le_bytes(data[89..93].try_into()?),
            oid_encoding_id: data[93],
            oid_is_lz4: data[94] != 0,
        })
    }
}

/// One block-table entry in the footer-index: the byte extent + row count of a
/// data block. The read path maps survivor rows → blocks via cumulative row
/// counts, then plans coalesced ranged GETs over these extents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FooterBlockEntry {
    /// Absolute byte offset of the block in the segment.
    pub offset: u64,
    /// Block byte length.
    pub size: u32,
    /// Number of rows in this block (for global row → block mapping).
    pub row_count: u32,
    /// The block's stats kind. `None` (legacy) carries no payload; `MinMax`
    /// carries the typed `FooterRowGroupStats` payload (TD-PAXRG-1), decoded
    /// by `read_blocks`. TD-FPRUNE-1 P2's filterable-column stats do NOT ride
    /// this per-entry frame — they live in the optional
    /// `SECTION_BLOCK_FILTERABLE_STATS` footer section so both payloads coexist.
    pub stats_kind: StatsKind,
}

macro_rules! footer_tag_enum {
    ($name:ident { $($variant:ident = $value:expr),+ $(,)? }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        #[repr(u8)]
        pub enum $name {
            $($variant = $value),+
        }

        impl $name {
            fn from_u8(value: u8) -> Result<Self> {
                match value {
                    $($value => Ok(Self::$variant),)+
                    _ => bail!("unknown {} tag 0x{value:02x}", stringify!($name)),
                }
            }
        }
    };
}

footer_tag_enum!(TierRole {
    Index = 0,
    Rerank = 1,
    Exact = 2,
    Metadata = 3,
});

footer_tag_enum!(LosslessTransformTag {
    None = 0,
    ClusteredForBitpackU8 = 1,
    ExactBaseXor = 2,
});

footer_tag_enum!(LosslessCompressionTag {
    None = 0,
    Lz4 = 1,
    Zstd = 2,
    Snappy = 3,
});

footer_tag_enum!(ParameterScope {
    None = 0,
    Segment = 1,
    Block = 2,
    ClusterRun = 3,
    MicroChunk = 4,
});

footer_tag_enum!(VectorTransform {
    None = 0,
    L2Normalized = 1,
    CenteredRotated = 2,
});

footer_tag_enum!(SourceRole {
    Canonical = 0,
    IndexProjection = 1,
    RerankProjection = 2,
});

footer_tag_enum!(SourceFidelity {
    ExactBitwise = 0,
    Lossy = 1,
});

/// Descriptor flags for the byte-compression layer.
pub mod compression_flags {
    /// Compressor inverse reproduces the transform bytes exactly.
    pub const LOSSLESS: u8 = 0b0000_0001;
}

/// Reserved rebuild-source ID for a projection-only object whose durable
/// canonical authority is cataloged outside this segment (for example the WAL
/// lineage). Such a file cannot independently serve exact reconstruction.
pub const EXTERNAL_CANONICAL_SOURCE_ID: u16 = u16::MAX;

/// One typed-footer physical stripe descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StripeEncodingDescriptor {
    pub descriptor_id: u16,
    pub logical_field_id: i32,
    pub physical_column_id: i32,
    pub tier_role: TierRole,
    pub value_codec_tag: u8,
    pub value_codec_version: u8,
    pub transform_tag: LosslessTransformTag,
    pub transform_version: u8,
    pub compression_tag: LosslessCompressionTag,
    pub compression_version: u8,
    pub compression_flags: u8,
    pub parameter_scope: ParameterScope,
    pub vector_transform: VectorTransform,
    pub auxiliary_flags: u16,
    pub source_role: SourceRole,
    pub source_fidelity: SourceFidelity,
    pub rebuild_source_id: u16,
    pub projection_generation: u16,
}

/// Selects one descriptor for one block-local physical tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockTierAssignment {
    pub block_ordinal: u32,
    pub physical_column_id: i32,
    pub tier_role: TierRole,
    pub descriptor_id: u16,
}

const DESCRIPTOR_SIZE: usize = 28;
const ASSIGNMENT_SIZE: usize = 11;
const SECTION_STRIPE_ENCODING_MAP: u8 = 2;
/// Optional footer section mirroring the Region C (exact fp32) extent
/// (TD-PAXRG-1). Same additive-section contract as the A0 mirror: old parsers
/// skip the unknown tag; footer stays single-version.
const SECTION_EXACT_REGION: u8 = 6;
/// `[c_off u64][c_len u64]`.
const EXACT_REGION_SECTION_LEN: usize = 16;
/// Optional footer section mirroring the Region A0 (coarse directory) extent
/// (TD-RDSTRAT-8). Additive: parsers that predate it skip unknown section tags
/// (pinned by `typed_footer_unknown_optional_section_is_skipped`), so the footer
/// stays single-version — layout versioning lives in the header-prefix.
const SECTION_COARSE_DIRECTORY: u8 = 3;
const SECTION_VERSION_V1: u8 = 1;
const FOOTER_V1: u8 = 1;
/// `[a0_off u64][a0_len u64]`.
const COARSE_DIRECTORY_SECTION_LEN: usize = 16;
/// Optional footer section mirroring the OID→position resolver region extent
/// (TD-DELVEC-1 WI-2c). The resolver is an immutable write-time component, so it
/// is persisted as a region *inside* the segment (atomic with it), not a sidecar.
/// Additive: parsers that predate it skip unknown section tags. `opr_len == 0`
/// ⇒ no resolver region ⇒ WI-3 tombstone fallback. The region payload is a
/// self-CRC'd `ORP1` blob (`OidPositionResolver::serialize`).
const SECTION_OID_RESOLVER: u8 = 4;
/// `[opr_off u64][opr_len u64]`.
const OID_RESOLVER_SECTION_LEN: usize = 16;
/// Optional footer section carrying TD-FPRUNE-1 P2's per-block
/// filterable-column stats (`proximadb_block_format::FooterBlockStats`
/// bytes), so a filtered query prunes blocks from the FETCH plan without a
/// body GET. Deliberately a SECTION, not the per-entry `stats_len` frame —
/// that frame carries the typed `FooterRowGroupStats` (TD-PAXRG-1), and the
/// two payloads must coexist. Sparse + index-tagged:
/// `[n u32] per entry: [block_idx u32][len u32][stats bytes]`. Additive:
/// parsers that predate it skip unknown section tags (same contract as the
/// A0/OID-resolver sections); absent ⇒ conservative fetch-per-block.
const SECTION_BLOCK_FILTERABLE_STATS: u8 = 7;
/// Optional footer section mapping each shredded user column's PROP NAME to its
/// physical PAX column id (TD-FPRUNE-1 P2 engagement). Written from the writer's
/// P-Shred spec so a filtered read resolves a user-tag field name (e.g.
/// `partition`) → col-id WITHOUT a catalog lookup — the segment self-describes
/// its shredded columns. Additive: predating parsers skip the unknown tag;
/// absent ⇒ empty map ⇒ user-tag footer-pruning is a no-op (canonical fields
/// still resolve via `pax_field_to_col`). Payload:
/// `[n u16] per entry: [name_len u16][name bytes][col_id i32]`.
const SECTION_SHRED_FIELD_MAP: u8 = 5;

/// The self-describing footer-index (Parquet-style metadata hub): block table +
/// schema snapshot + per-stripe encoding map + RaBitQ mirror + row count. Located
/// via `[footer_index][footer_len u64][SEGMENT_MAGIC]` at the segment tail.
///
/// The block table **subsumes** the legacy `SegmentIndex` for the new layout: the
/// reader builds the scanner's block list from it so `next_block` / `read_records`
/// work unchanged (mixed-read: legacy segments still use `SegmentIndex::locate`).
#[derive(Debug, Clone)]
pub struct SegmentFooterIndex {
    /// Total rows in the segment (== RaBitQ region `n_rows`). **u64 on the
    /// wire** (4-byte prefixes truncate) — hand-built footer bodies must
    /// serialize this as 8 bytes LE.
    pub row_count: u64,
    /// RaBitQ region extent mirror (also in the header-prefix).
    pub rabitq_off: u64,
    pub rabitq_len: u64,
    /// Coalesced SQ8 region (Region B, ADR-065) extent mirror (also in the
    /// header-prefix). The rerank tier hoisted out of blocks.
    pub sq8_off: u64,
    pub sq8_len: u64,
    /// Region B SQ8 dequant key mirror (ADR-065 cache-co-design). SQ8 maps each
    /// f32 dim → 1 byte via `dequant(code) = min + code·scale`, so the read path
    /// needs only these two. Mirrored in the footer (which the read path already
    /// reads) so it decodes survivor SQ8 **without the separate 24 B Region-B
    /// header GET**. Named `sq8_min` (not `sq8_offset`) to avoid collision with
    /// `sq8_off` (the Region B file offset). The codec's `vmin == min` (redundant)
    /// and `vmax = min + 255·scale` (recoverable) are NOT stored — only the
    /// dequant-essential pair. `(0.0, 0.0)` when there is no Region B.
    pub sq8_min: f32,
    pub sq8_scale: f32,
    // -- Schema snapshot (minimal; full evolution deferred to TD-TBL-4 / ADR-047) --
    /// Embedding dimensionality (per embedding column; homogeneous in v1).
    pub embed_dim: u32,
    /// Number of embedding columns.
    pub embed_count: u32,
    // -- Per-stripe encoding map (minimal; future clustered-SQ8/fp32 registers here) --
    /// Tier-1 (block) vector encoding tag: 0 = RawF32, 1 = SQ8, 3 = FP16.
    pub embed_quant_tag: u8,
    /// Whether the opt-in exact-f32 tier is present in the blocks.
    pub has_f32_tier: bool,
    /// The block table (block 0..K in emission/cluster order). On v4 segments
    /// each entry IS a row group (TD-PAXRG-1) — same wire shape, so every
    /// existing cumulative-row-count consumer works unmodified.
    pub blocks: Vec<FooterBlockEntry>,
    /// Per-entry row-group stats, index-aligned with [`Self::blocks`]. `None`
    /// (or an empty Vec) = no stats for that entry; v1/v3 footers serialize
    /// zero-length payloads and stay byte-identical (TD-PAXRG-1 Phase A).
    pub block_stats: Vec<Option<FooterRowGroupStats>>,
    /// Typed-footer physical encoding descriptors. Empty means emit the
    /// sectionless legacy form.
    pub encoding_map: Vec<StripeEncodingDescriptor>,
    /// Per-block descriptor selections for the typed encoding-map section.
    pub block_tier_assignments: Vec<BlockTierAssignment>,
    /// Region A0 (coarse directory) extent mirror (TD-RDSTRAT-8; also in the
    /// v3 header-prefix). `0/0` = no coarse level (v1 segments). Serialized as
    /// an optional trailing section, so v1 footers are byte-unchanged.
    pub a0_off: u64,
    pub a0_len: u64,
    /// OID→position resolver region extent (TD-DELVEC-1 WI-2c). The resolver is
    /// an immutable write-time component persisted as a region *inside* the
    /// segment (atomic with it on the single PUT), not a sidecar. `0/0` = no
    /// resolver (legacy/non-coalesced segments, or `with_oid_resolver` off) ⇒
    /// WI-3 tombstone fallback. Serialized as an optional trailing section
    /// (forward-compatible — old parsers skip the unknown tag).
    pub opr_off: u64,
    pub opr_len: u64,
    /// Region C (exact fp32) extent mirror (TD-PAXRG-1; also in the v4
    /// header-prefix). `0/0` = absent (v1/v3 segments, or v4 without the
    /// `pax_f32_tier` opt-in). Serialized as an optional trailing section.
    pub c_off: u64,
    pub c_len: u64,
    /// TD-FPRUNE-1 P2: per-block filterable-column stats payloads
    /// (`proximadb_block_format::FooterBlockStats` bytes), tagged by block
    /// index. Deliberately OUTSIDE the per-entry `stats_len` frame — that frame
    /// carries the typed `FooterRowGroupStats` (TD-PAXRG-1) — and serialized as
    /// an optional trailing section (`SECTION_BLOCK_FILTERABLE_STATS`), so both
    /// stats payloads coexist and predating parsers skip the unknown tag.
    /// Empty = no section = conservative fetch-per-block (the legacy baseline).
    pub block_filterable_stats: Vec<(u32, Vec<u8>)>,
    /// TD-FPRUNE-1 P2 engagement: shredded user columns' `(prop_name, col_id)`
    /// pairs, so a filtered read resolves a user-tag field name → PAX col-id
    /// without a catalog lookup (the segment self-describes its shredded
    /// columns). Empty (legacy/no-shred segments) ⇒ user-tag footer-pruning is a
    /// no-op. Serialized as an optional trailing section (absent ⇒ byte-unchanged).
    pub shred_field_map: Vec<(String, i32)>,
}

/// `[footer_len u64][SEGMENT_MAGIC 8B]` — the 16 B tail that locates the footer.
pub const SEG_TAIL_LEN: usize = 8 + 8;

impl StripeEncodingDescriptor {
    fn write_to(&self, output: &mut Vec<u8>) {
        output.extend_from_slice(&self.descriptor_id.to_le_bytes());
        output.extend_from_slice(&self.logical_field_id.to_le_bytes());
        output.extend_from_slice(&self.physical_column_id.to_le_bytes());
        output.push(self.tier_role as u8);
        output.push(self.value_codec_tag);
        output.push(self.value_codec_version);
        output.push(self.transform_tag as u8);
        output.push(self.transform_version);
        output.push(self.compression_tag as u8);
        output.push(self.compression_version);
        output.push(self.compression_flags);
        output.push(self.parameter_scope as u8);
        output.push(self.vector_transform as u8);
        output.extend_from_slice(&self.auxiliary_flags.to_le_bytes());
        output.push(self.source_role as u8);
        output.push(self.source_fidelity as u8);
        output.extend_from_slice(&self.rebuild_source_id.to_le_bytes());
        output.extend_from_slice(&self.projection_generation.to_le_bytes());
    }

    fn read_from(input: &[u8], position: &mut usize) -> Result<Self> {
        let descriptor = Self {
            descriptor_id: read_u16(input, position)?,
            logical_field_id: read_i32(input, position)?,
            physical_column_id: read_i32(input, position)?,
            tier_role: TierRole::from_u8(read_u8(input, position)?)?,
            value_codec_tag: read_u8(input, position)?,
            value_codec_version: read_u8(input, position)?,
            transform_tag: LosslessTransformTag::from_u8(read_u8(input, position)?)?,
            transform_version: read_u8(input, position)?,
            compression_tag: LosslessCompressionTag::from_u8(read_u8(input, position)?)?,
            compression_version: read_u8(input, position)?,
            compression_flags: read_u8(input, position)?,
            parameter_scope: ParameterScope::from_u8(read_u8(input, position)?)?,
            vector_transform: VectorTransform::from_u8(read_u8(input, position)?)?,
            auxiliary_flags: read_u16(input, position)?,
            source_role: SourceRole::from_u8(read_u8(input, position)?)?,
            source_fidelity: SourceFidelity::from_u8(read_u8(input, position)?)?,
            rebuild_source_id: read_u16(input, position)?,
            projection_generation: read_u16(input, position)?,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    fn validate(&self) -> Result<()> {
        if self.descriptor_id == 0 {
            bail!("footer encoding descriptor id 0 is reserved");
        }
        if !matches!(self.value_codec_tag, 0x01 | 0x03 | 0x05 | 0x71) {
            bail!(
                "unknown required value codec tag 0x{:02x}",
                self.value_codec_tag
            );
        }
        if self.value_codec_version != 1 {
            bail!(
                "unsupported value codec version {}",
                self.value_codec_version
            );
        }
        let expected_transform_version = match self.transform_tag {
            LosslessTransformTag::None => 0,
            LosslessTransformTag::ClusteredForBitpackU8 | LosslessTransformTag::ExactBaseXor => 1,
        };
        if self.transform_version != expected_transform_version {
            bail!("unsupported lossless transform version");
        }
        let expected_compression_version = match self.compression_tag {
            LosslessCompressionTag::None => 0,
            LosslessCompressionTag::Lz4
            | LosslessCompressionTag::Zstd
            | LosslessCompressionTag::Snappy => 1,
        };
        if self.compression_version != expected_compression_version {
            bail!("unsupported lossless compression version");
        }
        if self.compression_flags & compression_flags::LOSSLESS == 0 {
            bail!("durable stripe compressor must be declared lossless");
        }
        if self.source_role == SourceRole::Canonical {
            if self.source_fidelity != SourceFidelity::ExactBitwise {
                bail!("canonical stripe must be exact-bitwise");
            }
            if self.value_codec_tag != 0x01 {
                bail!("canonical dense-vector stripe must use raw-f32 value codec");
            }
        }
        match self.transform_tag {
            LosslessTransformTag::ClusteredForBitpackU8 if self.value_codec_tag != 0x05 => {
                bail!("clustered u8 transform requires the SQ8 value codec")
            }
            LosslessTransformTag::ExactBaseXor if self.value_codec_tag != 0x01 => {
                bail!("exact base-XOR transform requires the raw-f32 value codec")
            }
            _ => {}
        }
        Ok(())
    }
}

impl BlockTierAssignment {
    fn write_to(&self, output: &mut Vec<u8>) {
        output.extend_from_slice(&self.block_ordinal.to_le_bytes());
        output.extend_from_slice(&self.physical_column_id.to_le_bytes());
        output.push(self.tier_role as u8);
        output.extend_from_slice(&self.descriptor_id.to_le_bytes());
    }

    fn read_from(input: &[u8], position: &mut usize) -> Result<Self> {
        Ok(Self {
            block_ordinal: read_u32(input, position)?,
            physical_column_id: read_i32(input, position)?,
            tier_role: TierRole::from_u8(read_u8(input, position)?)?,
            descriptor_id: read_u16(input, position)?,
        })
    }
}

fn write_block_entry(
    output: &mut Vec<u8>,
    block: &FooterBlockEntry,
    stats: Option<&FooterRowGroupStats>,
) {
    output.extend_from_slice(&block.offset.to_le_bytes());
    output.extend_from_slice(&block.size.to_le_bytes());
    output.extend_from_slice(&block.row_count.to_le_bytes());
    match stats {
        Some(s) => {
            output.push(StatsKind::MinMax as u8);
            let payload = s.to_bytes();
            let len = u32::try_from(payload.len()).expect("row-group stats payload fits u32");
            output.extend_from_slice(&len.to_le_bytes());
            output.extend_from_slice(&payload);
        }
        None => {
            output.push(block.stats_kind as u8);
            output.extend_from_slice(&0u32.to_le_bytes());
        }
    }
}

/// Returns the per-entry stats payloads (index-aligned with the entries).
/// Unknown or zero-length stats tags decode to `None` — forward- and
/// backward-compatible with v1/v3 footers and any future stats kind.
fn read_blocks(
    input: &[u8],
    position: &mut usize,
) -> Result<(Vec<FooterBlockEntry>, Vec<Option<FooterRowGroupStats>>)> {
    let block_count = read_u32(input, position)? as usize;
    if block_count > input.len().saturating_sub(*position) / 21 {
        bail!("footer-index block count exceeds remaining bytes");
    }
    let mut blocks = Vec::with_capacity(block_count);
    let mut stats = Vec::with_capacity(block_count);
    for _ in 0..block_count {
        let offset = read_u64(input, position)?;
        let size = read_u32(input, position)?;
        let row_count = read_u32(input, position)?;
        let stats_tag = read_u8(input, position)?;
        let stats_len = read_u32(input, position)? as usize;
        ensure_remaining(
            input,
            *position,
            stats_len,
            "footer-index stats payload overruns body",
        )?;
        let payload = &input[*position..*position + stats_len];
        *position += stats_len;
        // Only the known MinMax kind decodes; other/unknown kinds keep their
        // payload opaque (skipped, never a hard error).
        let decoded = if stats_tag == StatsKind::MinMax as u8 && stats_len >= RG_STATS_MIN_BYTES {
            FooterRowGroupStats::from_bytes(payload).ok()
        } else {
            None
        };
        stats.push(decoded);
        blocks.push(FooterBlockEntry {
            offset,
            size,
            row_count,
            stats_kind: StatsKind::from_tag(stats_tag),
        });
    }
    Ok((blocks, stats))
}

fn parse_encoding_map_payload(
    payload: &[u8],
) -> Result<(Vec<StripeEncodingDescriptor>, Vec<BlockTierAssignment>)> {
    let mut position = 0usize;
    let descriptor_count = read_u16(payload, &mut position)? as usize;
    if descriptor_count > payload.len().saturating_sub(position) / DESCRIPTOR_SIZE {
        bail!("stripe encoding-map descriptor count exceeds section bytes");
    }
    let mut descriptors = Vec::with_capacity(descriptor_count);
    for _ in 0..descriptor_count {
        descriptors.push(StripeEncodingDescriptor::read_from(payload, &mut position)?);
    }
    let assignment_count = read_u32(payload, &mut position)? as usize;
    if assignment_count > payload.len().saturating_sub(position) / ASSIGNMENT_SIZE {
        bail!("stripe encoding-map assignment count exceeds section bytes");
    }
    let mut assignments = Vec::with_capacity(assignment_count);
    for _ in 0..assignment_count {
        assignments.push(BlockTierAssignment::read_from(payload, &mut position)?);
    }
    if position != payload.len() {
        bail!("stripe encoding-map payload has trailing bytes");
    }
    Ok((descriptors, assignments))
}

fn ensure_remaining(input: &[u8], position: usize, len: usize, message: &str) -> Result<()> {
    let end = position
        .checked_add(len)
        .ok_or_else(|| anyhow::anyhow!("{message}: length overflow"))?;
    if end > input.len() {
        bail!("{message}");
    }
    Ok(())
}

fn read_u8(input: &[u8], position: &mut usize) -> Result<u8> {
    ensure_remaining(input, *position, 1, "footer-index truncated at u8")?;
    let value = input[*position];
    *position += 1;
    Ok(value)
}

fn read_u16(input: &[u8], position: &mut usize) -> Result<u16> {
    ensure_remaining(input, *position, 2, "footer-index truncated at u16")?;
    let end = *position + 2;
    let value = u16::from_le_bytes(input[*position..end].try_into()?);
    *position = end;
    Ok(value)
}

fn read_u32(input: &[u8], position: &mut usize) -> Result<u32> {
    ensure_remaining(input, *position, 4, "footer-index truncated at u32")?;
    let end = *position + 4;
    let value = u32::from_le_bytes(input[*position..end].try_into()?);
    *position = end;
    Ok(value)
}

fn read_f32(input: &[u8], position: &mut usize) -> Result<f32> {
    ensure_remaining(input, *position, 4, "footer-index truncated at f32")?;
    let end = *position + 4;
    let value = f32::from_le_bytes(input[*position..end].try_into()?);
    *position = end;
    Ok(value)
}

fn read_i32(input: &[u8], position: &mut usize) -> Result<i32> {
    ensure_remaining(input, *position, 4, "footer-index truncated at i32")?;
    let end = *position + 4;
    let value = i32::from_le_bytes(input[*position..end].try_into()?);
    *position = end;
    Ok(value)
}

fn read_u64(input: &[u8], position: &mut usize) -> Result<u64> {
    ensure_remaining(input, *position, 8, "footer-index truncated at u64")?;
    let end = *position + 8;
    let value = u64::from_le_bytes(input[*position..end].try_into()?);
    *position = end;
    Ok(value)
}

#[cfg(test)]
fn typed_footer_first_descriptor_offset(bytes: &[u8]) -> Result<usize> {
    if bytes.first().copied() != Some(FOOTER_V1) {
        bail!("not a footer v1 payload");
    }
    let mut position = 1 + 8 * 5 + 4 * 4 + 2;
    let block_count = read_u32(bytes, &mut position)? as usize;
    for _ in 0..block_count {
        let _ = read_u64(bytes, &mut position)?;
        let _ = read_u32(bytes, &mut position)?;
        let _ = read_u32(bytes, &mut position)?;
        let _ = read_u8(bytes, &mut position)?;
        let stats_len = read_u32(bytes, &mut position)? as usize;
        ensure_remaining(bytes, position, stats_len, "footer test stats overrun")?;
        position += stats_len;
    }
    let section_count = read_u16(bytes, &mut position)?;
    if section_count == 0 {
        bail!("typed footer has no sections");
    }
    let section_tag = read_u8(bytes, &mut position)?;
    let _ = read_u8(bytes, &mut position)?;
    let _ = read_u32(bytes, &mut position)?;
    if section_tag != SECTION_STRIPE_ENCODING_MAP {
        bail!("first typed-footer section is not the encoding map");
    }
    let descriptor_count = read_u16(bytes, &mut position)?;
    if descriptor_count == 0 {
        bail!("typed-footer encoding map has no descriptors");
    }
    Ok(position)
}

#[cfg(test)]
fn typed_footer_section_count_offset(bytes: &[u8]) -> Result<usize> {
    if bytes.first().copied() != Some(FOOTER_V1) {
        bail!("not a footer v1 payload");
    }
    let mut position = 1 + 8 * 5 + 4 * 4 + 2;
    let block_count = read_u32(bytes, &mut position)? as usize;
    for _ in 0..block_count {
        let _ = read_u64(bytes, &mut position)?;
        let _ = read_u32(bytes, &mut position)?;
        let _ = read_u32(bytes, &mut position)?;
        let _ = read_u8(bytes, &mut position)?;
        let stats_len = read_u32(bytes, &mut position)? as usize;
        ensure_remaining(bytes, position, stats_len, "footer test stats overrun")?;
        position += stats_len;
    }
    Ok(position)
}

impl SegmentFooterIndex {
    /// Serialize the footer-index body (no tail). Layout (in-place evolution —
    /// pre-release, no versioned files on disk, so the version byte is a constant
    /// that always matches the current code; versioning re-engages at GA):
    /// `[footer_version u8][row_count u64][rabitq_off u64][rabitq_len u64]`
    /// `[sq8_off u64][sq8_len u64]`                                     (ADR-065 Region B)
    /// `[sq8_min f32][sq8_scale f32]`                                   (cache-co-design: dequant key)
    /// `[embed_dim u32][embed_count u32][embed_quant_tag u8][has_f32_tier u8]`
    /// `[n_blocks u32] per block: [off u64][size u32][row_count u32][stats_tag u8][stats_len u32][stats bytes]`.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut buf = Vec::with_capacity(72 + self.blocks.len() * 24);
        buf.push(FOOTER_V1);
        buf.extend_from_slice(&self.row_count.to_le_bytes());
        buf.extend_from_slice(&self.rabitq_off.to_le_bytes());
        buf.extend_from_slice(&self.rabitq_len.to_le_bytes());
        buf.extend_from_slice(&self.sq8_off.to_le_bytes());
        buf.extend_from_slice(&self.sq8_len.to_le_bytes());
        buf.extend_from_slice(&self.sq8_min.to_le_bytes());
        buf.extend_from_slice(&self.sq8_scale.to_le_bytes());
        buf.extend_from_slice(&self.embed_dim.to_le_bytes());
        buf.extend_from_slice(&self.embed_count.to_le_bytes());
        buf.push(self.embed_quant_tag);
        buf.push(if self.has_f32_tier { 1 } else { 0 });
        let block_count = u32::try_from(self.blocks.len())
            .map_err(|_| anyhow::anyhow!("footer-index block count exceeds u32"))?;
        buf.extend_from_slice(&block_count.to_le_bytes());
        for (i, b) in self.blocks.iter().enumerate() {
            write_block_entry(
                &mut buf,
                b,
                self.block_stats.get(i).and_then(|s| s.as_ref()),
            );
        }
        // Optional trailing sections (single version-1 footer layout —
        // pre-release: no versioned files on disk; versioning re-engages at
        // GA). Emitted only when at least one section has content, so a
        // sectionless footer is byte-identical to the historical form:
        //  - stripe encoding map: per-stripe lossless-transform metadata
        //    (clustered-SQ8 / scalar LZ4, TD-RDSTRAT-7 / ADR-065 Region B);
        //  - coarse directory extent: Region A0 mirror (TD-RDSTRAT-8).
        let mut sections: Vec<(u8, Vec<u8>)> = Vec::new();
        if !self.encoding_map.is_empty() {
            self.validate_encoding_map()?;
            sections.push((SECTION_STRIPE_ENCODING_MAP, self.encoding_map_payload()?));
        }
        if self.a0_len > 0 {
            let mut payload = Vec::with_capacity(COARSE_DIRECTORY_SECTION_LEN);
            payload.extend_from_slice(&self.a0_off.to_le_bytes());
            payload.extend_from_slice(&self.a0_len.to_le_bytes());
            sections.push((SECTION_COARSE_DIRECTORY, payload));
        }
        if self.opr_len > 0 {
            let mut payload = Vec::with_capacity(OID_RESOLVER_SECTION_LEN);
            payload.extend_from_slice(&self.opr_off.to_le_bytes());
            payload.extend_from_slice(&self.opr_len.to_le_bytes());
            sections.push((SECTION_OID_RESOLVER, payload));
        }
        if self.c_len > 0 {
            let mut payload = Vec::with_capacity(EXACT_REGION_SECTION_LEN);
            payload.extend_from_slice(&self.c_off.to_le_bytes());
            payload.extend_from_slice(&self.c_len.to_le_bytes());
            sections.push((SECTION_EXACT_REGION, payload));
        }
        // TD-FPRUNE-1 P2: sparse per-block filterable-column stats. Only the
        // blocks whose footer stats were populated carry an entry.
        if !self.block_filterable_stats.is_empty() {
            let count = u32::try_from(self.block_filterable_stats.len())
                .map_err(|_| anyhow::anyhow!("filterable-stats count exceeds u32"))?;
            let mut payload = Vec::with_capacity(4 + self.block_filterable_stats.len() * 8);
            payload.extend_from_slice(&count.to_le_bytes());
            for (block_idx, stats) in &self.block_filterable_stats {
                let len =
                    u32::try_from(stats.len()).map_err(|_| anyhow::anyhow!("stats exceeds u32"))?;
                payload.extend_from_slice(&block_idx.to_le_bytes());
                payload.extend_from_slice(&len.to_le_bytes());
                payload.extend_from_slice(stats);
            }
            sections.push((SECTION_BLOCK_FILTERABLE_STATS, payload));
        }
        if !self.shred_field_map.is_empty() {
            sections.push((SECTION_SHRED_FIELD_MAP, self.shred_field_map_payload()?));
        }
        if !sections.is_empty() {
            let section_count = u16::try_from(sections.len())
                .map_err(|_| anyhow::anyhow!("footer section count exceeds u16"))?;
            buf.extend_from_slice(&section_count.to_le_bytes());
            for (tag, payload) in sections {
                buf.push(tag);
                buf.push(SECTION_VERSION_V1);
                let section_len = u32::try_from(payload.len())
                    .map_err(|_| anyhow::anyhow!("footer section exceeds u32"))?;
                buf.extend_from_slice(&section_len.to_le_bytes());
                buf.extend_from_slice(&payload);
            }
        }
        Ok(buf)
    }

    /// TD-FPRUNE-1 P2: `[n u16] per entry: [name_len u16][name bytes][col_id i32]`.
    fn shred_field_map_payload(&self) -> Result<Vec<u8>> {
        let n = u16::try_from(self.shred_field_map.len())
            .map_err(|_| anyhow::anyhow!("shred field-map count exceeds u16"))?;
        let mut payload = Vec::with_capacity(2 + self.shred_field_map.len() * 16);
        payload.extend_from_slice(&n.to_le_bytes());
        for (name, col_id) in &self.shred_field_map {
            let name_len = u16::try_from(name.len())
                .map_err(|_| anyhow::anyhow!("shred field name exceeds u16 bytes"))?;
            payload.extend_from_slice(&name_len.to_le_bytes());
            payload.extend_from_slice(name.as_bytes());
            payload.extend_from_slice(&col_id.to_le_bytes());
        }
        Ok(payload)
    }

    fn parse_shred_field_map(section: &[u8]) -> Result<Vec<(String, i32)>> {
        let mut p = 0usize;
        let n = read_u16(section, &mut p)? as usize;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let name_len = read_u16(section, &mut p)? as usize;
            ensure_remaining(
                section,
                p,
                name_len,
                "shred field-map name overruns section",
            )?;
            let name = String::from_utf8(section[p..p + name_len].to_vec())
                .map_err(|_| anyhow::anyhow!("shred field-map name is not UTF-8"))?;
            p += name_len;
            let col_id = read_i32(section, &mut p)?;
            out.push((name, col_id));
        }
        Ok(out)
    }

    fn encoding_map_payload(&self) -> Result<Vec<u8>> {
        let descriptor_count = u16::try_from(self.encoding_map.len())
            .map_err(|_| anyhow::anyhow!("footer encoding descriptor count exceeds u16"))?;
        let assignment_count = u32::try_from(self.block_tier_assignments.len())
            .map_err(|_| anyhow::anyhow!("footer tier assignment count exceeds u32"))?;
        let mut payload = Vec::with_capacity(
            2 + self.encoding_map.len() * DESCRIPTOR_SIZE
                + 4
                + self.block_tier_assignments.len() * ASSIGNMENT_SIZE,
        );
        payload.extend_from_slice(&descriptor_count.to_le_bytes());
        for descriptor in &self.encoding_map {
            descriptor.write_to(&mut payload);
        }
        payload.extend_from_slice(&assignment_count.to_le_bytes());
        for assignment in &self.block_tier_assignments {
            assignment.write_to(&mut payload);
        }
        Ok(payload)
    }

    fn validate_encoding_map(&self) -> Result<()> {
        if self.encoding_map.is_empty() {
            bail!("typed footer requires a non-empty stripe encoding map");
        }
        let mut descriptors = HashMap::with_capacity(self.encoding_map.len());
        for descriptor in &self.encoding_map {
            descriptor.validate()?;
            if descriptors
                .insert(descriptor.descriptor_id, descriptor)
                .is_some()
            {
                bail!(
                    "duplicate footer encoding descriptor id {}",
                    descriptor.descriptor_id
                );
            }
        }
        for descriptor in &self.encoding_map {
            match descriptor.source_role {
                SourceRole::Canonical => {
                    if descriptor.rebuild_source_id != 0 {
                        bail!("canonical descriptor must not name a rebuild source");
                    }
                }
                SourceRole::IndexProjection | SourceRole::RerankProjection => {
                    if descriptor.rebuild_source_id == EXTERNAL_CANONICAL_SOURCE_ID {
                        continue;
                    }
                    let source =
                        descriptors
                            .get(&descriptor.rebuild_source_id)
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                                    "projection descriptor {} has no canonical rebuild source {}",
                                    descriptor.descriptor_id,
                                    descriptor.rebuild_source_id
                                )
                            })?;
                    if source.source_role != SourceRole::Canonical
                        || source.source_fidelity != SourceFidelity::ExactBitwise
                    {
                        bail!("projection rebuild source is not exact canonical data");
                    }
                }
            }
        }
        for assignment in &self.block_tier_assignments {
            let block_ordinal = assignment.block_ordinal as usize;
            if block_ordinal >= self.blocks.len() {
                bail!("footer tier assignment references unknown block");
            }
            let descriptor = descriptors.get(&assignment.descriptor_id).ok_or_else(|| {
                anyhow::anyhow!("footer tier assignment references unknown descriptor")
            })?;
            if descriptor.physical_column_id != assignment.physical_column_id
                || descriptor.tier_role != assignment.tier_role
            {
                bail!("footer tier assignment disagrees with its descriptor");
            }
        }
        Ok(())
    }

    /// Parse the footer-index body (the bytes between the data blocks and the
    /// `[footer_len][magic]` tail). Fail-closed on truncation / bad version.
    /// Forward-compatible: an unknown future `FOOTER_VERSION` is an error (a new
    /// footer version is a deliberate, gated change), but unknown `StatsKind`
    /// tags degrade to `None`.
    pub fn parse(body: &[u8]) -> Result<Self> {
        if body.is_empty() {
            bail!("empty footer-index body");
        }
        match body[0] {
            FOOTER_V1 => Self::parse_v1(body),
            version => bail!("unsupported footer-index version {version}"),
        }
    }

    fn parse_v1(body: &[u8]) -> Result<Self> {
        let mut p = 1usize;
        let row_count = read_u64(body, &mut p)?;
        let rabitq_off = read_u64(body, &mut p)?;
        let rabitq_len = read_u64(body, &mut p)?;
        let sq8_off = read_u64(body, &mut p)?;
        let sq8_len = read_u64(body, &mut p)?;
        let sq8_min = read_f32(body, &mut p)?;
        let sq8_scale = read_f32(body, &mut p)?;
        let embed_dim = read_u32(body, &mut p)?;
        let embed_count = read_u32(body, &mut p)?;
        if p + 2 > body.len() {
            bail!("footer-index truncated at encoding map");
        }
        let embed_quant_tag = body[p];
        let has_f32_tier = body[p + 1] != 0;
        p += 2;
        let (blocks, block_stats) = read_blocks(body, &mut p)?;
        // Optional trailing sections (encoding map, coarse-directory extent).
        // Absent ⇒ defaults. Unknown tags are skipped (forward-compatible).
        let mut encoding_map = Vec::new();
        let mut block_tier_assignments = Vec::new();
        let mut a0_off = 0u64;
        let mut a0_len = 0u64;
        let mut opr_off = 0u64;
        let mut opr_len = 0u64;
        let mut c_off = 0u64;
        let mut c_len = 0u64;
        let mut block_filterable_stats: Vec<(u32, Vec<u8>)> = Vec::new();
        let mut shred_field_map = Vec::new();
        if p != body.len() {
            let section_count = read_u16(body, &mut p)? as usize;
            if section_count > body.len().saturating_sub(p) / 6 {
                bail!("footer-index section count exceeds remaining bytes");
            }
            for _ in 0..section_count {
                let tag = read_u8(body, &mut p)?;
                let version = read_u8(body, &mut p)?;
                let section_len = read_u32(body, &mut p)? as usize;
                ensure_remaining(body, p, section_len, "footer-index section overruns body")?;
                let section = &body[p..p + section_len];
                p += section_len;
                if tag == SECTION_STRIPE_ENCODING_MAP {
                    if version != SECTION_VERSION_V1 {
                        bail!("unsupported stripe encoding-map section version {version}");
                    }
                    let (descriptors, assignments) = parse_encoding_map_payload(section)?;
                    encoding_map = descriptors;
                    block_tier_assignments = assignments;
                } else if tag == SECTION_COARSE_DIRECTORY {
                    if version != SECTION_VERSION_V1 {
                        bail!("unsupported coarse-directory section version {version}");
                    }
                    if section.len() != COARSE_DIRECTORY_SECTION_LEN {
                        bail!("coarse-directory section has wrong length");
                    }
                    a0_off = u64::from_le_bytes(section[..8].try_into()?);
                    a0_len = u64::from_le_bytes(section[8..16].try_into()?);
                } else if tag == SECTION_OID_RESOLVER {
                    if version != SECTION_VERSION_V1 {
                        bail!("unsupported oid-resolver section version {version}");
                    }
                    if section.len() != OID_RESOLVER_SECTION_LEN {
                        bail!("oid-resolver section has wrong length");
                    }
                    opr_off = u64::from_le_bytes(section[..8].try_into()?);
                    opr_len = u64::from_le_bytes(section[8..16].try_into()?);
                } else if tag == SECTION_EXACT_REGION {
                    if version != SECTION_VERSION_V1 {
                        bail!("unsupported exact-region section version {version}");
                    }
                    if section.len() != EXACT_REGION_SECTION_LEN {
                        bail!("exact-region section has wrong length");
                    }
                    c_off = u64::from_le_bytes(section[..8].try_into()?);
                    c_len = u64::from_le_bytes(section[8..16].try_into()?);
                } else if tag == SECTION_BLOCK_FILTERABLE_STATS {
                    if version != SECTION_VERSION_V1 {
                        bail!("unsupported filterable-stats section version {version}");
                    }
                    let mut q = 0usize;
                    let count = read_u32(section, &mut q)? as usize;
                    if count > section.len() / 8 {
                        bail!("filterable-stats count exceeds section bytes");
                    }
                    block_filterable_stats = Vec::with_capacity(count);
                    for _ in 0..count {
                        let block_idx = read_u32(section, &mut q)?;
                        let len = read_u32(section, &mut q)? as usize;
                        ensure_remaining(section, q, len, "filterable-stats entry overruns")?;
                        block_filterable_stats.push((block_idx, section[q..q + len].to_vec()));
                        q += len;
                    }
                } else if tag == SECTION_SHRED_FIELD_MAP {
                    if version != SECTION_VERSION_V1 {
                        bail!("unsupported shred field-map section version {version}");
                    }
                    shred_field_map = Self::parse_shred_field_map(section)?;
                }
            }
            if p != body.len() {
                bail!("footer-index has trailing bytes");
            }
        }
        Ok(Self {
            row_count,
            rabitq_off,
            rabitq_len,
            sq8_off,
            sq8_len,
            sq8_min,
            sq8_scale,
            embed_dim,
            embed_count,
            embed_quant_tag,
            has_f32_tier,
            blocks,
            block_stats,
            encoding_map,
            block_tier_assignments,
            a0_off,
            a0_len,
            opr_off,
            opr_len,
            c_off,
            c_len,
            block_filterable_stats,
            shred_field_map,
        })
    }

    /// Locate + parse the footer-index from a full segment buffer (the tail
    /// `[footer_len u64][SEGMENT_MAGIC 8B]` gives the extent). Returns
    /// `Ok(None)` when the buffer is not a coalesced segment (no valid tail) —
    /// the caller then uses the legacy `SegmentIndex::locate` path.
    pub fn locate_in_segment(segment: &[u8]) -> Result<Option<Self>> {
        if segment.len() < SEG_TAIL_LEN {
            return Ok(None);
        }
        let tail = &segment[segment.len() - SEG_TAIL_LEN..];
        if &tail[8..] != SEGMENT_MAGIC {
            return Ok(None);
        }
        let footer_len = u64::from_le_bytes(tail[..8].try_into()?) as usize;
        if SEG_TAIL_LEN + footer_len > segment.len() {
            bail!("coalesced footer-index length overruns segment");
        }
        let body =
            &segment[segment.len() - SEG_TAIL_LEN - footer_len..segment.len() - SEG_TAIL_LEN];
        Ok(Some(Self::parse(body)?))
    }
}

/// Build the segment tail bytes: `[footer-index body][footer_len u64][SEGMENT_MAGIC]`.
pub fn segment_tail(footer_body: &[u8]) -> Vec<u8> {
    let mut tail = Vec::with_capacity(footer_body.len() + SEG_TAIL_LEN);
    tail.extend_from_slice(footer_body);
    tail.extend_from_slice(&(footer_body.len() as u64).to_le_bytes());
    tail.extend_from_slice(SEGMENT_MAGIC);
    tail
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mixed-format segment detection must be unambiguous by magic and never
    /// mis-route the legacy path. Golden table ported from the former root
    /// `segment_format` tests — locks the disjointness of BLOCK_MAGIC /
    /// SEG_HEADER_MAGIC / SEGMENT_MAGIC now that detection lives in this crate
    /// (and proves the moved logic is unit-testable WITHOUT linking the root).
    #[test]
    fn segment_format_detect_golden_table() {
        // Empty / short / unknown bytes -> legacy (never mis-route).
        assert_eq!(SegmentFormat::detect(&[]), SegmentFormat::ProximaBlocks);
        assert_eq!(
            SegmentFormat::detect(&[0x01, 0, 0]),
            SegmentFormat::ProximaBlocks
        );
        assert_eq!(
            SegmentFormat::detect(&[0x05, 0, 0]),
            SegmentFormat::ProximaBlocks
        );

        // PAX head (BLOCK_MAGIC) + trailing SEGMENT_MAGIC -> Pax.
        let mut pax = BLOCK_MAGIC.to_vec();
        pax.extend_from_slice(SEGMENT_MAGIC);
        assert_eq!(SegmentFormat::detect(&pax), SegmentFormat::Pax);

        // Coalesced head (SEG_HEADER_MAGIC) + trailing SEGMENT_MAGIC -> Pax.
        let mut coalesced = SEG_HEADER_MAGIC.to_vec();
        coalesced.extend_from_slice(SEGMENT_MAGIC);
        assert_eq!(SegmentFormat::detect(&coalesced), SegmentFormat::Pax);

        // Head magic alone (no trailing SEGMENT_MAGIC) -> NOT Pax: a stray prefix
        // alone must never false-positive.
        assert_eq!(
            SegmentFormat::detect(&BLOCK_MAGIC[..]),
            SegmentFormat::ProximaBlocks
        );
        assert_eq!(
            SegmentFormat::detect(&SEG_HEADER_MAGIC[..]),
            SegmentFormat::ProximaBlocks
        );

        // Default (e.g. a manifest entry predating the format field) is legacy.
        assert_eq!(SegmentFormat::default(), SegmentFormat::ProximaBlocks);
    }

    /// TD-USUB-5 detect precedence across ALL FOUR recognised heads — `PBLK`,
    /// `PXH1`, `PAR1`, and the legacy version byte — including the adversarial
    /// hybrids where one format's head coexists with another's tail. The
    /// property under test is that the four recognisers are **disjoint**: each
    /// constrains the head, so no ordering of the `detect` arms can change a
    /// verdict, and every hybrid falls to the legacy default rather than
    /// decoding as the wrong format.
    #[test]
    fn segment_format_detect_precedence_all_four_magics() {
        let pax_tail = SEGMENT_MAGIC;

        // --- The four heads, each with its own correct tail. ---
        let mut pblk = BLOCK_MAGIC.to_vec();
        pblk.extend_from_slice(pax_tail);
        assert_eq!(SegmentFormat::detect(&pblk), SegmentFormat::Pax);

        let mut pxh1 = SEG_HEADER_MAGIC.to_vec();
        pxh1.extend_from_slice(pax_tail);
        assert_eq!(SegmentFormat::detect(&pxh1), SegmentFormat::Pax);

        let mut par1 = PARQUET_MAGIC.to_vec();
        par1.extend_from_slice(&[0u8; 4]); // footer length
        par1.extend_from_slice(PARQUET_MAGIC);
        assert_eq!(SegmentFormat::detect(&par1), SegmentFormat::Parquet);

        for version_byte in 0x01u8..=0x0E {
            assert_eq!(
                SegmentFormat::detect(&[version_byte, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                SegmentFormat::ProximaBlocks,
                "legacy version byte {version_byte:#04x} must stay legacy"
            );
        }

        // --- Adversarial: a PAR1 tail coexisting with a PAX head. Neither
        // recogniser is satisfied (PAX wants its own tail; Parquet wants its own
        // head), so this must NOT be claimed as either format. ---
        for head in [&BLOCK_MAGIC[..], &SEG_HEADER_MAGIC[..]] {
            let mut hybrid = head.to_vec();
            hybrid.extend_from_slice(&[0u8; 4]);
            hybrid.extend_from_slice(PARQUET_MAGIC);
            assert_eq!(
                SegmentFormat::detect(&hybrid),
                SegmentFormat::ProximaBlocks,
                "PAX head + PAR1 tail must not be claimed as Pax or Parquet"
            );
        }

        // --- Adversarial: a PAR1 head with a PAX segment tail. Same reasoning
        // from the other side. ---
        let mut hybrid = PARQUET_MAGIC.to_vec();
        hybrid.extend_from_slice(pax_tail);
        assert_eq!(
            SegmentFormat::detect(&hybrid),
            SegmentFormat::ProximaBlocks,
            "PAR1 head + PAXSEG01 tail must not be claimed as Parquet or Pax"
        );

        // --- A lone four-byte PAR1 satisfies starts_with AND ends_with; the
        // length bound must reject it. ---
        assert_eq!(
            SegmentFormat::detect(&PARQUET_MAGIC[..]),
            SegmentFormat::ProximaBlocks
        );
        assert!(!is_parquet_file(&PARQUET_MAGIC[..]));

        // --- Disjointness of the head magics themselves, stated directly: this
        // is the invariant that makes arm ordering irrelevant. ---
        assert_ne!(&BLOCK_MAGIC[..], &PARQUET_MAGIC[..]);
        assert_ne!(&SEG_HEADER_MAGIC[..], &PARQUET_MAGIC[..]);
        assert!(!(0x01u8..=0x0E).contains(&PARQUET_MAGIC[0]));
    }

    fn sample_footer() -> SegmentFooterIndex {
        SegmentFooterIndex {
            c_off: 0,
            c_len: 0,
            row_count: 1000,
            rabitq_off: 56,
            rabitq_len: 24_000,
            sq8_off: 24_056,
            sq8_len: 128_000,
            sq8_min: -1.0,
            sq8_scale: 0.05,
            embed_dim: 128,
            embed_count: 1,
            embed_quant_tag: 1, // SQ8 (Region B)
            has_f32_tier: false,
            encoding_map: Vec::new(),
            block_tier_assignments: Vec::new(),
            a0_off: 0,
            a0_len: 0,
            opr_off: 0,
            opr_len: 0,
            block_filterable_stats: Vec::new(),
            blocks: vec![
                FooterBlockEntry {
                    offset: 152_056,
                    size: 8_000,
                    row_count: 128,
                    stats_kind: StatsKind::None,
                },
                FooterBlockEntry {
                    offset: 160_056,
                    size: 7_900,
                    row_count: 127,
                    stats_kind: StatsKind::None,
                },
            ],
            block_stats: vec![None, None],
            // TD-FPRUNE-1 P2: two shredded user columns, to exercise the
            // self-describing footer field-map section round-trip.
            shred_field_map: vec![("partition".to_string(), 100), ("lang".to_string(), 101)],
        }
    }

    fn sample_typed_footer() -> SegmentFooterIndex {
        let mut footer = sample_footer();
        footer.encoding_map = vec![
            StripeEncodingDescriptor {
                descriptor_id: 1,
                logical_field_id: 20,
                physical_column_id: 20,
                tier_role: TierRole::Index,
                value_codec_tag: 0x71,
                value_codec_version: 1,
                transform_tag: LosslessTransformTag::None,
                transform_version: 0,
                compression_tag: LosslessCompressionTag::None,
                compression_version: 0,
                compression_flags: compression_flags::LOSSLESS,
                parameter_scope: ParameterScope::Block,
                vector_transform: VectorTransform::CenteredRotated,
                auxiliary_flags: 0,
                source_role: SourceRole::IndexProjection,
                source_fidelity: SourceFidelity::Lossy,
                rebuild_source_id: 3,
                projection_generation: 1,
            },
            StripeEncodingDescriptor {
                descriptor_id: 2,
                logical_field_id: 20,
                physical_column_id: 30,
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
                rebuild_source_id: 3,
                projection_generation: 1,
            },
            StripeEncodingDescriptor {
                descriptor_id: 3,
                logical_field_id: 20,
                physical_column_id: 40,
                tier_role: TierRole::Exact,
                value_codec_tag: 0x01,
                value_codec_version: 1,
                transform_tag: LosslessTransformTag::ExactBaseXor,
                transform_version: 1,
                compression_tag: LosslessCompressionTag::Zstd,
                compression_version: 1,
                compression_flags: compression_flags::LOSSLESS,
                parameter_scope: ParameterScope::ClusterRun,
                vector_transform: VectorTransform::None,
                auxiliary_flags: 0,
                source_role: SourceRole::Canonical,
                source_fidelity: SourceFidelity::ExactBitwise,
                rebuild_source_id: 0,
                projection_generation: 0,
            },
        ];
        footer.block_tier_assignments = vec![BlockTierAssignment {
            block_ordinal: 0,
            physical_column_id: 30,
            tier_role: TierRole::Rerank,
            descriptor_id: 2,
        }];
        footer
    }

    #[test]
    fn header_prefix_round_trips() {
        let h = SegmentHeaderPrefix {
            layout_version: SEG_LAYOUT_VERSION,
            rabitq_off: 56,
            rabitq_len: 24_000,
            sq8_off: 24_056,
            sq8_len: 128_000,
            footer_off: 200_000,
            footer_len: 512,
            a0_off: 0,
            a0_len: 0,
            c_off: 0,
            c_len: 0,
        };
        let bytes = h.to_bytes();
        assert_eq!(bytes.len(), SEG_HEADER_PREFIX_LEN);
        let parsed = SegmentHeaderPrefix::parse(&bytes).unwrap();
        assert_eq!(parsed.rabitq_off, 56);
        assert_eq!(parsed.rabitq_len, 24_000);
        assert_eq!(parsed.sq8_off, 24_056);
        assert_eq!(parsed.sq8_len, 128_000);
        assert_eq!(parsed.footer_off, 200_000);
        assert_eq!(parsed.footer_len, 512);
    }

    #[test]
    fn header_parse_rejects_bad_magic_and_version() {
        // Wrong magic → err (a legacy PBLK segment is never mis-detected).
        let mut bad = SegmentHeaderPrefix {
            layout_version: SEG_LAYOUT_VERSION,
            rabitq_off: 0,
            rabitq_len: 0,
            sq8_off: 0,
            sq8_len: 0,
            footer_off: 0,
            footer_len: 0,
            a0_off: 0,
            a0_len: 0,
            c_off: 0,
            c_len: 0,
        }
        .to_bytes();
        bad[0..4].copy_from_slice(b"PBLK");
        assert!(SegmentHeaderPrefix::parse(&bad).is_err());
        // Wrong version → err.
        let mut v = SegmentHeaderPrefix {
            layout_version: SEG_LAYOUT_VERSION,
            rabitq_off: 0,
            rabitq_len: 0,
            sq8_off: 0,
            sq8_len: 0,
            footer_off: 0,
            footer_len: 0,
            a0_off: 0,
            a0_len: 0,
            c_off: 0,
            c_len: 0,
        }
        .to_bytes();
        v[4] = 99;
        assert!(SegmentHeaderPrefix::parse(&v).is_err());
    }

    /// TD-PAXRG-1 Phase A: the v4 (row-group Region D) prefix serializes 88 B
    /// and round-trips its Region C extent. `0/0` c-extent = C absent.
    #[test]
    fn header_prefix_v4_round_trips_with_c_extent() {
        let h = SegmentHeaderPrefix {
            layout_version: SEG_LAYOUT_VERSION,
            rabitq_off: 88,
            rabitq_len: 24_000,
            sq8_off: 24_088,
            sq8_len: 128_000,
            footer_off: 400_000,
            footer_len: 512,
            a0_off: 0,
            a0_len: 0,
            c_off: 152_088,
            c_len: 768_000,
        };
        let bytes = h.to_bytes();
        assert_eq!(bytes.len(), SEG_HEADER_PREFIX_LEN);
        let parsed = SegmentHeaderPrefix::parse(&bytes).expect("v4 prefix parses");
        assert_eq!(parsed.layout_version, SEG_LAYOUT_VERSION);
        assert_eq!(parsed.c_off, 152_088);
        assert_eq!(parsed.c_len, 768_000);
        assert_eq!(parsed.rabitq_off, 88);
        assert_eq!(parsed.footer_off, 400_000);
    }

    /// TD-PAXRG-1 Phase A (mixed-read contract): a pre-v4 binary — whose parse
    /// accepted only v1/v3 — must REJECT a v4 prefix fail-closed, never
    /// mis-probe it as a v3 segment (mirror of the v1/v3 freeze test).

    /// TD-PAXRG-1 Phase C: the Region C extent round-trips as the
    /// `SECTION_EXACT_REGION` footer mirror, and a footer without it (c_len 0)
    /// stays byte-identical to the sectionless form.
    #[test]
    fn footer_exact_region_section_round_trips_and_absent_when_c_len_zero() {
        let base = SegmentFooterIndex {
            row_count: 128,
            rabitq_off: 88,
            rabitq_len: 1_000,
            sq8_off: 1_088,
            sq8_len: 2_000,
            sq8_min: 0.0,
            sq8_scale: 1.0,
            embed_dim: 8,
            embed_count: 1,
            embed_quant_tag: 1,
            has_f32_tier: true,
            blocks: vec![FooterBlockEntry {
                offset: 5_000,
                size: 4_000,
                row_count: 128,
                stats_kind: StatsKind::None,
            }],
            block_stats: Vec::new(),
            encoding_map: Vec::new(),
            block_tier_assignments: Vec::new(),
            a0_off: 0,
            a0_len: 0,
            opr_off: 0,
            opr_len: 0,
            c_off: 3_088,
            c_len: 16_384,
            block_filterable_stats: Vec::new(),
            shred_field_map: Vec::new(),
        };
        let parsed = SegmentFooterIndex::parse(&base.to_bytes().expect("serialize"))
            .expect("parse footer with exact-region section");
        assert_eq!(parsed.c_off, 3_088);
        assert_eq!(parsed.c_len, 16_384);

        // c_len == 0 ⇒ no section: byte-identical to the pre-PAXRG form.
        let mut absent = base.clone();
        absent.c_off = 0;
        absent.c_len = 0;
        let bytes = absent.to_bytes().expect("serialize sectionless");
        assert!(!bytes.is_empty());
        let reparsed = SegmentFooterIndex::parse(&bytes).expect("parse sectionless footer");
        assert_eq!(reparsed.c_off, 0);
        assert_eq!(reparsed.c_len, 0);
    }

    /// TD-PAXRG-1 Phase A: the MinMax stats payload round-trips through the
    /// footer block table, and unknown/short stats tags decode to `None`
    /// (forward-compatible) without disturbing the entry fields.
    #[test]
    fn footer_block_entry_minmax_stats_round_trip_and_unknown_kind_skipped() {
        let mut zone = crate::pax_block::BlockZoneSummary::empty(128);
        zone.created_at = (1_000, 9_000);
        zone.present |= crate::pax_block::ZONE_CREATED_AT;
        let stats = FooterRowGroupStats {
            zone,
            oid_chunk_rel_off: 512,
            oid_chunk_len: 1_024,
            oid_encoding_id: 1,
            oid_is_lz4: false,
        };
        let footer = SegmentFooterIndex {
            row_count: 255,
            rabitq_off: 0,
            rabitq_len: 0,
            sq8_off: 0,
            sq8_len: 0,
            sq8_min: 0.0,
            sq8_scale: 0.0,
            embed_dim: 8,
            embed_count: 1,
            embed_quant_tag: 1,
            has_f32_tier: false,
            c_off: 0,
            c_len: 0,
            block_filterable_stats: Vec::new(),
            shred_field_map: Vec::new(),
            blocks: vec![
                FooterBlockEntry {
                    offset: 1_000,
                    size: 4_096,
                    row_count: 128,
                    stats_kind: StatsKind::MinMax,
                },
                FooterBlockEntry {
                    offset: 5_096,
                    size: 4_000,
                    row_count: 127,
                    stats_kind: StatsKind::None,
                },
            ],
            block_stats: vec![Some(stats.clone()), None],
            encoding_map: Vec::new(),
            block_tier_assignments: Vec::new(),
            a0_off: 0,
            a0_len: 0,
            opr_off: 0,
            opr_len: 0,
        };
        let parsed = SegmentFooterIndex::parse(&footer.to_bytes().expect("serialize"))
            .expect("parse footer with stats");
        assert_eq!(parsed.blocks.len(), 2);
        assert_eq!(parsed.block_stats.len(), 2);
        let got = parsed.block_stats[0]
            .as_ref()
            .expect("first entry carries stats");
        assert_eq!(got, &stats, "stats payload must round-trip exactly");
        assert!(
            parsed.block_stats[1].is_none(),
            "entry without stats stays None"
        );

        // Forward-compat: an UNKNOWN stats tag with a payload is skipped, the
        // entry still parses, kind degrades to None.
        let mut raw = Vec::new();
        raw.extend_from_slice(&7u64.to_le_bytes());
        raw.extend_from_slice(&100u32.to_le_bytes());
        raw.extend_from_slice(&10u32.to_le_bytes());
        raw.push(0xF0); // unknown stats tag
        raw.extend_from_slice(&4u32.to_le_bytes());
        raw.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
        let mut body = Vec::new();
        body.push(FOOTER_V1);
        body.extend_from_slice(&1u64.to_le_bytes()); // row_count (u64 in FOOTER_V1)
        body.extend_from_slice(&0u64.to_le_bytes()); // rabitq_off/len
        body.extend_from_slice(&0u64.to_le_bytes());
        body.extend_from_slice(&0u64.to_le_bytes()); // sq8_off/len
        body.extend_from_slice(&0u64.to_le_bytes());
        body.extend_from_slice(&0f32.to_le_bytes()); // sq8_min/scale
        body.extend_from_slice(&0f32.to_le_bytes());
        body.extend_from_slice(&8u32.to_le_bytes()); // embed_dim
        body.extend_from_slice(&1u32.to_le_bytes()); // embed_count
        body.push(1); // embed_quant_tag
        body.push(0); // has_f32_tier
        body.extend_from_slice(&1u32.to_le_bytes()); // n_blocks = 1
        body.extend_from_slice(&raw);
        let parsed_unknown = SegmentFooterIndex::parse(&body).expect("unknown stats skipped");
        assert_eq!(parsed_unknown.blocks[0].stats_kind, StatsKind::None);
        assert!(parsed_unknown.block_stats[0].is_none());
    }

    #[test]
    fn is_coalesced_segment_presence_field() {
        let h = SegmentHeaderPrefix {
            layout_version: SEG_LAYOUT_VERSION,
            rabitq_off: 56,
            rabitq_len: 1,
            sq8_off: 57,
            sq8_len: 1,
            footer_off: 58,
            footer_len: 1,
            a0_off: 0,
            a0_len: 0,
            c_off: 0,
            c_len: 0,
        }
        .to_bytes();
        assert!(is_coalesced_segment(&h));
        // A legacy block magic prefix is NOT coalesced.
        assert!(!is_coalesced_segment(b"PBLK\x00\x00"));
        assert!(!is_coalesced_segment(&[]));
    }

    /// TD-RDSTRAT-8 mixed-read contract: a **version-1-only reader** (any binary
    /// that predates the two-level layout) must reject a v3 segment cleanly.
    /// Old binaries are frozen, so this pins the two facts their rejection
    /// depends on: the on-disk version byte is 3 (not 1), and a strict
    /// `version == 1` check therefore fails.

    #[test]
    fn footer_coarse_directory_section_round_trips() {
        let mut footer = sample_footer();
        footer.a0_off = 72;
        footer.a0_len = 123_456;
        let bytes = footer.to_bytes().unwrap();
        let parsed = SegmentFooterIndex::parse(&bytes).unwrap();
        assert_eq!(parsed.a0_off, 72);
        assert_eq!(parsed.a0_len, 123_456);
        assert_eq!(parsed.blocks.len(), 2);
        // Coexists with the encoding-map section (two sections).
        let mut typed = sample_typed_footer();
        typed.a0_off = 72;
        typed.a0_len = 99;
        let bytes2 = typed.to_bytes().unwrap();
        let parsed2 = SegmentFooterIndex::parse(&bytes2).unwrap();
        assert_eq!(parsed2.encoding_map, typed.encoding_map);
        assert_eq!(parsed2.a0_len, 99);
        // Absent A0 (a0_len == 0) ⇒ byte-identical to the sectionless footer
        // (v1 segments unchanged on disk by this feature).
        let plain = sample_footer();
        assert_eq!(plain.to_bytes().unwrap(), {
            let mut same = sample_footer();
            same.a0_off = 999; // off without len must NOT emit a section
            same.a0_len = 0;
            same.to_bytes().unwrap()
        });
        let reparsed = SegmentFooterIndex::parse(&plain.to_bytes().unwrap()).unwrap();
        assert_eq!(reparsed.a0_off, 0);
        assert_eq!(reparsed.a0_len, 0);
    }

    #[test]
    fn footer_oid_resolver_section_round_trips() {
        // TD-DELVEC-1 WI-2c: the OID→position resolver region extent is an
        // optional footer section (tag 4), mirroring the coarse-directory
        // section. Round-trips, coexists with other sections, and is absent
        // (byte-identical to the sectionless footer) when opr_len == 0.
        let mut footer = sample_footer();
        footer.opr_off = 500_000;
        footer.opr_len = 7_777;
        let bytes = footer.to_bytes().unwrap();
        let parsed = SegmentFooterIndex::parse(&bytes).unwrap();
        assert_eq!(parsed.opr_off, 500_000);
        assert_eq!(parsed.opr_len, 7_777);
        assert_eq!(parsed.blocks.len(), 2);
        // Coexists with the encoding-map + coarse-directory sections (three).
        let mut typed = sample_typed_footer();
        typed.a0_off = 72;
        typed.a0_len = 99;
        typed.opr_off = 8_000;
        typed.opr_len = 4_000;
        let parsed2 = SegmentFooterIndex::parse(&typed.to_bytes().unwrap()).unwrap();
        assert_eq!(parsed2.encoding_map, typed.encoding_map);
        assert_eq!(parsed2.a0_len, 99);
        assert_eq!(parsed2.opr_off, 8_000);
        assert_eq!(parsed2.opr_len, 4_000);
        // Absent resolver (opr_len == 0) ⇒ byte-identical to the sectionless
        // footer (v1 / non-coalesced segments unchanged on disk by this feature).
        let plain = sample_footer();
        assert_eq!(plain.to_bytes().unwrap(), {
            let mut same = sample_footer();
            same.opr_off = 999; // off without len must NOT emit a section
            same.opr_len = 0;
            same.to_bytes().unwrap()
        });
        let reparsed = SegmentFooterIndex::parse(&plain.to_bytes().unwrap()).unwrap();
        assert_eq!(reparsed.opr_off, 0);
        assert_eq!(reparsed.opr_len, 0);
    }

    #[test]
    fn footer_index_round_trips() {
        let mut f = sample_footer();
        // TD-FPRUNE-1 P2: a sparse filterable-stats section entry for block 1
        // (block 0 stays absent — the mixed-read baseline).
        f.block_filterable_stats = vec![(1u32, vec![1, 0, 42, 7, 255])];
        let bytes = f.to_bytes().unwrap();
        let parsed = SegmentFooterIndex::parse(&bytes).unwrap();
        assert_eq!(parsed.row_count, 1000);
        assert_eq!(parsed.rabitq_off, 56);
        assert_eq!(parsed.sq8_off, 24_056);
        assert_eq!(parsed.sq8_len, 128_000);
        assert_eq!(parsed.embed_dim, 128);
        assert_eq!(parsed.embed_count, 1);
        assert_eq!(parsed.embed_quant_tag, 1);
        assert_eq!(parsed.blocks.len(), 2);
        assert_eq!(parsed.blocks[0].offset, 152_056);
        assert_eq!(parsed.blocks[1].row_count, 127);
        // TD-FPRUNE-1 P2: the sparse filterable-stats section survives the
        // round-trip, and an absent entry stays absent (mixed-read baseline).
        assert_eq!(parsed.block_filterable_stats.len(), 1);
        assert_eq!(parsed.block_filterable_stats[0].0, 1u32);
        assert_eq!(
            parsed.block_filterable_stats[0].1.as_slice(),
            &[1, 0, 42, 7, 255][..]
        );
        // TD-FPRUNE-1 P2: the self-describing shred field-map survives round-trip.
        assert_eq!(
            parsed.shred_field_map,
            vec![("partition".to_string(), 100), ("lang".to_string(), 101)]
        );
    }

    #[test]
    fn footer_locate_in_segment_round_trips() {
        let f = sample_footer();
        let body = f.to_bytes().unwrap();
        // Pretend segment = [some data][footer body][footer_len][magic].
        let mut segment = vec![0xAAu8; 1000];
        segment.extend(segment_tail(&body));
        let parsed = SegmentFooterIndex::locate_in_segment(&segment)
            .unwrap()
            .expect("footer located");
        assert_eq!(parsed.blocks.len(), 2);
        assert_eq!(parsed.row_count, 1000);
    }

    #[test]
    fn footer_locate_returns_none_for_non_coalesced_tail() {
        // A legacy `[..][SEGMENT_MAGIC]` tail (no footer_len before magic) — the
        // 8 bytes before magic would be parsed as footer_len, but the resulting
        // body slice is not a valid footer (wrong version byte) → err → None-ish.
        // Here we assert a too-short buffer returns None cleanly.
        assert!(
            SegmentFooterIndex::locate_in_segment(&[0u8; 5])
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn footer_parse_rejects_bad_version() {
        let mut bytes = sample_footer().to_bytes().unwrap();
        bytes[0] = 2; // unsupported version (current is 1)
        assert!(SegmentFooterIndex::parse(&bytes).is_err());
    }

    #[test]
    fn stats_kind_unknown_tag_degrades_to_none() {
        assert_eq!(StatsKind::from_tag(99), StatsKind::None);
        assert_eq!(StatsKind::from_tag(1), StatsKind::BloomOid);
    }

    #[test]
    fn typed_footer_encoding_map_round_trips() -> Result<()> {
        let footer = sample_typed_footer();
        let bytes = footer.to_bytes()?;
        assert_eq!(bytes[0], 1);

        let parsed = SegmentFooterIndex::parse(&bytes)?;
        assert_eq!(parsed.encoding_map, footer.encoding_map);
        assert_eq!(parsed.block_tier_assignments, footer.block_tier_assignments);
        assert_eq!(
            parsed.encoding_map[1].transform_tag,
            LosslessTransformTag::ClusteredForBitpackU8
        );
        Ok(())
    }

    #[test]
    fn typed_footer_rejects_lossy_canonical_descriptor() {
        let mut footer = sample_typed_footer();
        footer.encoding_map[2].source_fidelity = SourceFidelity::Lossy;
        assert!(footer.to_bytes().is_err());
    }

    #[test]
    fn typed_footer_unknown_required_transform_fails_closed() -> Result<()> {
        let footer = sample_typed_footer();
        let mut bytes = footer.to_bytes()?;
        let transform_offset = typed_footer_first_descriptor_offset(&bytes)? + 13;
        bytes[transform_offset] = 0xfe;
        assert!(SegmentFooterIndex::parse(&bytes).is_err());
        Ok(())
    }

    #[test]
    fn typed_footer_unknown_optional_section_is_skipped() -> Result<()> {
        let footer = sample_typed_footer();
        let mut bytes = footer.to_bytes()?;
        let section_count_offset = typed_footer_section_count_offset(&bytes)?;
        // Compute the count (don't hardcode) — the footer's known-section set
        // grows over time (A0, OID resolver, exact region, shred field map,
        // filterable stats); the invariant under test is that ONE MORE,
        // unknown-tagged section is skipped wholesale.
        let original_count =
            u16::from_le_bytes([bytes[section_count_offset], bytes[section_count_offset + 1]]);
        bytes[section_count_offset..section_count_offset + 2]
            .copy_from_slice(&(original_count + 1).to_le_bytes());
        bytes.push(0xfe);
        bytes.push(1);
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&[1, 2, 3]);

        let parsed = SegmentFooterIndex::parse(&bytes)?;
        assert_eq!(parsed.encoding_map, footer.encoding_map);
        Ok(())
    }
}
