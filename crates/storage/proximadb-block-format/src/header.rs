// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Block header: 64-byte fixed prefix for every ProximaDB block file.
//!
//! The header encodes mode (OLTP/OLAP/PAX), schema fingerprint, time-range
//! statistics (for block-level time pruning), and tenant isolation hash (for
//! RLS skip without deserialising row data).

use anyhow::{Result, bail};

/// Magic bytes for ProximaDB PAX blocks.
pub const BLOCK_MAGIC: [u8; 4] = *b"PBLK";
/// Current format version.
///
/// v2 (this build) is a clean break from v1: SQ8-quantized vector stripes, a
/// per-column dimension carried in the [`crate::writer::VectorParamBlock`]
/// side region (no more per-row dim prefix), and a row-group sub-index. v1
/// blocks are rejected outright by [`BlockHeader::from_bytes`].
///
/// **Mixed-read-safety (mandate #8).** The v1→v2 clean break is safe *because v1
/// was a pre-release development format that never persisted in any released
/// build* — there are no v1 blocks on disk to migrate, so rejecting them loses
/// no data. This exception applies only to that pre-1.0 transition. From v2
/// onward this version byte is the compatibility gate: any future `v3` MUST be
/// mixed-read-safe — readers dispatch on the byte and keep decoding v2 rather
/// than flag-day rejecting it. See `docs/12-design/adr/ADR-010-pax-block-format.adoc`.
pub const FORMAT_VERSION: u8 = 2;

/// PAX v3 — declared columns authoritative, msgpack `PROPS` reduced to a residual
/// (ADR-094 spec §2.1 / TD-USUB-6).
///
/// **No writer emits this yet.** It is defined so the reader can *dispatch* on the
/// version byte instead of equality-rejecting it, which is the precondition the
/// paragraph above states for any v3 to exist. Accepting the byte is separable
/// from — and strictly precedes — implementing the semantics.
pub const FORMAT_VERSION_V3: u8 = 3;

/// Every block format version this build can decode, oldest first.
///
/// [`BlockHeader::from_bytes`] dispatches on membership here rather than
/// `== FORMAT_VERSION`, so a v3 block is readable rather than rejected outright.
/// That is mandate #8's mixed-read rule; the v1→v2 clean break was an explicit
/// pre-release exception to it, not a precedent.
pub const SUPPORTED_FORMAT_VERSIONS: &[u8] = &[FORMAT_VERSION, FORMAT_VERSION_V3];

/// Fixed header size in bytes.
pub const HEADER_SIZE: usize = 64;

/// Physical storage mode of the block.
///
/// * `Oltp`  — row-store: slot directory with per-row offsets + msgpack row blobs.
///   Enables O(1) row lookup by index; suitable for ≤ 1 GB collections.
/// * `Olap`  — pure column stripes, no row directory; bulk-scan optimised.
///   External engines (Spark, Trino, DuckDB) consume this via Iceberg REST.
/// * `Pax`   — Partition Attributes Across: row directory PLUS column stripes.
///   Default for ProximaDB internal storage. Supports both row-level
///   MVCC access (OLTP path) and vectorised column scan (OLAP path).
///   Vector + filter columns are co-located in leading stripes for
///   predicate-aware HNSW (ADR-007, spec §6.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BlockMode {
    Oltp = 1,
    Olap = 2,
    Pax = 3,
}

impl BlockMode {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            1 => Ok(Self::Oltp),
            2 => Ok(Self::Olap),
            3 => Ok(Self::Pax),
            _ => bail!("unknown BlockMode byte 0x{v:02x}"),
        }
    }

    /// True when the block contains a row directory (OLTP point-lookup path).
    pub fn has_row_directory(self) -> bool {
        matches!(self, Self::Oltp | Self::Pax)
    }

    /// True when the block contains column stripes (OLAP scan path).
    pub fn has_column_stripes(self) -> bool {
        matches!(self, Self::Olap | Self::Pax)
    }
}

/// Compression codec applied to column stripes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum BlockCompression {
    #[default]
    None = 0,
    Lz4 = 1,
    Zstd = 2,
    Snappy = 3,
}

impl BlockCompression {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(Self::None),
            1 => Ok(Self::Lz4),
            2 => Ok(Self::Zstd),
            3 => Ok(Self::Snappy),
            _ => bail!("unknown BlockCompression byte 0x{v:02x}"),
        }
    }
}

/// Block-level capability flags (u8 bitfield in header byte [7]).
pub mod flags {
    /// Block contains Bloom filter data in footer.
    pub const HAS_BLOOM: u8 = 0b0000_0001;
    /// Block contains graph edge columns (edge_source_id, edge_target_id, …).
    pub const HAS_EDGE: u8 = 0b0000_0010;
    /// Block contains at least one embedding column.
    pub const HAS_VECTOR: u8 = 0b0000_0100;
    /// Block uses MVCC version chain (valid_from_ns / valid_to_ns).
    pub const HAS_MVCC: u8 = 0b0000_1000;
    /// Row directory is sorted by row_id_hash (enables binary search).
    pub const DIR_SORTED: u8 = 0b0001_0000;
}

/// Fixed 64-byte block header, written at offset 0 of every block file.
///
/// Layout (little-endian):
/// ```text
/// [0..4]   magic              b"PBLK"
/// [4]      format_version     1
/// [5]      block_mode         BlockMode as u8
/// [6]      compression        BlockCompression as u8
/// [7]      flags              capability bitfield
/// [8..10]  column_count       u16
/// [10..14] row_count          u32
/// [14..18] block_size         u32  (total bytes including header)
/// [18..22] checksum           u32  crc32 (IEEE) of bytes [64..block_size]
/// [22..24] _pad               u16
/// [24..32] collection_id_hash u64  xxhash64(collection_id)
/// [32..40] schema_fingerprint u64  schema version fingerprint
/// [40..48] min_timestamp_ns   i64  min(created_at_ns) in block
/// [48..56] max_timestamp_ns   i64  max(created_at_ns) in block
/// [56..64] tenant_id_hash     u64  xxhash64(tenant_id) — RLS skip
/// ```
#[derive(Debug, Clone, Copy)]
pub struct BlockHeader {
    pub block_mode: BlockMode,
    pub compression: BlockCompression,
    pub flags: u8,
    pub column_count: u16,
    pub row_count: u32,
    /// Total block size in bytes (including this 64-byte header).
    pub block_size: u32,
    /// crc32 (IEEE) checksum of bytes [HEADER_SIZE..block_size] (the block body,
    /// i.e. everything after the 64-byte header). Verified on read by
    /// `PaxBlockReader::open`.
    pub checksum: u32,
    /// xxhash64 of the UTF-8 collection identifier.
    pub collection_id_hash: u64,
    /// Schema fingerprint from `CatalogTableSchema.fingerprint` or equivalent.
    pub schema_fingerprint: u64,
    /// Minimum `created_at_ns` of any row in the block (for time-range pruning).
    pub min_timestamp_ns: i64,
    /// Maximum `created_at_ns` of any row in the block.
    pub max_timestamp_ns: i64,
    /// xxhash64 of the tenant_id string (for RLS block-skip without row decode).
    pub tenant_id_hash: u64,
    /// On-disk format version of THIS block, as read from byte 4.
    ///
    /// Previously the version byte was validated and then **discarded**, so a
    /// reader had no way to dispatch on it even once v3 was accepted — the
    /// information simply was not carried. Retaining it is what makes
    /// version-dependent decode possible (TD-USUB-6).
    ///
    /// Construct with [`BlockHeader::current_version`] to stamp the version this
    /// build writes; `to_bytes` serializes this field rather than a constant, so
    /// a header round-trips its own version instead of silently being relabelled.
    pub format_version: u8,
}

impl BlockHeader {
    /// The format version this build writes. Use when constructing a header for
    /// a new block so the intent ("current") is explicit at the call site.
    pub const fn current_version() -> u8 {
        FORMAT_VERSION
    }

    /// True iff this block uses the v3 layout (declared columns authoritative).
    ///
    /// No writer emits v3 yet, so this is `false` for every block on disk today;
    /// it exists so the decode paths can branch on the block rather than on a
    /// caller-supplied assumption.
    pub fn is_v3(&self) -> bool {
        self.format_version == FORMAT_VERSION_V3
    }
}

impl BlockHeader {
    /// Serialize to a 64-byte array (little-endian).
    pub fn to_bytes(self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&BLOCK_MAGIC);
        // Serialize the header's OWN version, not a constant: a header parsed
        // from a v3 block must not be silently relabelled v2 on re-serialize.
        buf[4] = self.format_version;
        buf[5] = self.block_mode as u8;
        buf[6] = self.compression as u8;
        buf[7] = self.flags;
        buf[8..10].copy_from_slice(&self.column_count.to_le_bytes());
        buf[10..14].copy_from_slice(&self.row_count.to_le_bytes());
        buf[14..18].copy_from_slice(&self.block_size.to_le_bytes());
        buf[18..22].copy_from_slice(&self.checksum.to_le_bytes());
        // [22..24] pad — zeros
        buf[24..32].copy_from_slice(&self.collection_id_hash.to_le_bytes());
        buf[32..40].copy_from_slice(&self.schema_fingerprint.to_le_bytes());
        buf[40..48].copy_from_slice(&self.min_timestamp_ns.to_le_bytes());
        buf[48..56].copy_from_slice(&self.max_timestamp_ns.to_le_bytes());
        buf[56..64].copy_from_slice(&self.tenant_id_hash.to_le_bytes());
        buf
    }

    /// Deserialize from a 64-byte slice.
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() < HEADER_SIZE {
            bail!("block header too short: {} < {HEADER_SIZE}", buf.len());
        }
        if buf[0..4] != BLOCK_MAGIC {
            bail!("invalid block magic: {:02x?}", &buf[0..4]);
        }
        // Mixed-read (mandate #8): dispatch on the version byte instead of
        // equality-rejecting anything that is not the version THIS build writes.
        // The error names what is supported so an operator can tell "too old"
        // from "written by a newer peer" without reading the source.
        let format_version = buf[4];
        if !SUPPORTED_FORMAT_VERSIONS.contains(&format_version) {
            bail!(
                "unsupported block format version {format_version}: this build decodes \
                 {SUPPORTED_FORMAT_VERSIONS:?}"
            );
        }

        Ok(Self {
            format_version,
            block_mode: BlockMode::from_u8(buf[5])?,
            compression: BlockCompression::from_u8(buf[6])?,
            flags: buf[7],
            column_count: u16::from_le_bytes(buf[8..10].try_into()?),
            row_count: u32::from_le_bytes(buf[10..14].try_into()?),
            block_size: u32::from_le_bytes(buf[14..18].try_into()?),
            checksum: u32::from_le_bytes(buf[18..22].try_into()?),
            collection_id_hash: u64::from_le_bytes(buf[24..32].try_into()?),
            schema_fingerprint: u64::from_le_bytes(buf[32..40].try_into()?),
            min_timestamp_ns: i64::from_le_bytes(buf[40..48].try_into()?),
            max_timestamp_ns: i64::from_le_bytes(buf[48..56].try_into()?),
            tenant_id_hash: u64::from_le_bytes(buf[56..64].try_into()?),
        })
    }

    /// True if this block may contain rows for the given `tenant_id_hash`.
    ///
    /// This is a block-level **pruning hint**, not the tenant-isolation
    /// boundary: it lets a scan skip blocks that provably hold no rows for the
    /// querying tenant. A `tenant_id_hash` of 0 means the block is
    /// **mixed-tenant** — the writer stamps 0 whenever a single block accretes
    /// records from more than one tenant (see `PaxBlockWriter`) — so such a
    /// block cannot be pruned by tenant and is conservatively scanned. Tenant
    /// isolation for mixed blocks is enforced downstream by **row-level**
    /// tenant filtering, never by this method alone; returning `true` here is
    /// therefore correct (a superset), not a fail-open leak.
    pub fn tenant_matches(&self, tenant_hash: u64) -> bool {
        self.tenant_id_hash == 0 || self.tenant_id_hash == tenant_hash
    }

    /// True if this block's time range overlaps `[from_ns, to_ns]`.
    pub fn time_overlaps(&self, from_ns: i64, to_ns: i64) -> bool {
        self.max_timestamp_ns >= from_ns && self.min_timestamp_ns <= to_ns
    }
}

/// Simple non-cryptographic hash for tenant_id and collection_id routing.
/// Uses FNV-1a for no-dependency implementation; replace with xxhash64 at
/// engine layer when the dependency is acceptable.
pub fn fnv1a_hash(s: &str) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    s.bytes()
        .fold(OFFSET, |h, b| (h ^ b as u64).wrapping_mul(PRIME))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build header bytes carrying an arbitrary on-disk version byte.
    fn header_bytes_with_version(version: u8) -> [u8; HEADER_SIZE] {
        let mut h = BlockHeader {
            format_version: BlockHeader::current_version(),
            block_mode: BlockMode::Pax,
            compression: BlockCompression::None,
            flags: 0,
            column_count: 1,
            row_count: 1,
            block_size: HEADER_SIZE as u32,
            checksum: 0,
            collection_id_hash: 0,
            schema_fingerprint: 0,
            min_timestamp_ns: 0,
            max_timestamp_ns: 0,
            tenant_id_hash: 0,
        };
        h.format_version = version;
        h.to_bytes()
    }

    /// TD-USUB-6: the reader must **dispatch** on the version byte, not
    /// equality-reject it. Before this, `from_bytes` failed anything that was not
    /// exactly `FORMAT_VERSION`, so a v3 block could never be read — making the
    /// mixed-read rule this module's own doc states impossible to honour.
    #[test]
    fn v3_header_is_accepted_and_its_version_retained() {
        let parsed = BlockHeader::from_bytes(&header_bytes_with_version(FORMAT_VERSION_V3))
            .expect("a v3 block must be decodable, not rejected outright");
        assert_eq!(parsed.format_version, FORMAT_VERSION_V3);
        assert!(
            parsed.is_v3(),
            "decode paths must be able to branch on this"
        );
    }

    /// v2 stays the default and is not mistaken for v3.
    #[test]
    fn v2_header_round_trips_as_v2() {
        let parsed = BlockHeader::from_bytes(&header_bytes_with_version(FORMAT_VERSION))
            .expect("v2 must keep decoding");
        assert_eq!(parsed.format_version, FORMAT_VERSION);
        assert!(!parsed.is_v3());
    }

    /// A header parsed from disk must re-serialize with ITS OWN version, not be
    /// silently relabelled as whatever this build writes. `to_bytes` previously
    /// stamped the `FORMAT_VERSION` constant unconditionally, which would rewrite
    /// a v3 block's header as v2 on any read-modify-write path.
    #[test]
    fn to_bytes_preserves_the_parsed_version_rather_than_relabelling() {
        let v3 = BlockHeader::from_bytes(&header_bytes_with_version(FORMAT_VERSION_V3)).unwrap();
        assert_eq!(
            v3.to_bytes()[4],
            FORMAT_VERSION_V3,
            "re-serializing a v3 header must not downgrade the version byte"
        );
    }

    /// Unknown versions still fail closed, and the error names what IS supported
    /// so "too old" is distinguishable from "written by a newer peer".
    #[test]
    fn unknown_versions_fail_closed_with_an_actionable_message() {
        for bad in [0u8, 1, 4, 255] {
            let err = BlockHeader::from_bytes(&header_bytes_with_version(bad))
                .expect_err("unknown version must be rejected");
            let msg = err.to_string();
            assert!(
                msg.contains(&bad.to_string()) && msg.contains("unsupported block format version"),
                "error must name the offending version, got: {msg}"
            );
        }
    }

    /// The dispatch table must not drift from the versions the constants name.
    #[test]
    fn supported_versions_match_the_named_constants() {
        assert!(SUPPORTED_FORMAT_VERSIONS.contains(&FORMAT_VERSION));
        assert!(SUPPORTED_FORMAT_VERSIONS.contains(&FORMAT_VERSION_V3));
        assert_eq!(BlockHeader::current_version(), FORMAT_VERSION);
    }

    #[test]
    fn header_round_trip() {
        let h = BlockHeader {
            format_version: BlockHeader::current_version(),
            block_mode: BlockMode::Pax,
            compression: BlockCompression::Lz4,
            flags: flags::HAS_VECTOR | flags::HAS_MVCC,
            column_count: 12,
            row_count: 1024,
            block_size: 4 * 1024 * 1024,
            checksum: 0xdeadbeef,
            collection_id_hash: fnv1a_hash("my_collection"),
            schema_fingerprint: 0x1234_5678_9abc_def0,
            min_timestamp_ns: 1_000_000_000,
            max_timestamp_ns: 2_000_000_000,
            tenant_id_hash: fnv1a_hash("tenant_a"),
        };
        let bytes = h.to_bytes();
        assert_eq!(bytes.len(), HEADER_SIZE);
        let h2 = BlockHeader::from_bytes(&bytes).unwrap();
        assert_eq!(h2.block_mode as u8, BlockMode::Pax as u8);
        assert_eq!(h2.row_count, 1024);
        assert_eq!(h2.column_count, 12);
        assert_eq!(h2.tenant_id_hash, h.tenant_id_hash);
        assert!(h2.tenant_matches(fnv1a_hash("tenant_a")));
        assert!(!h2.tenant_matches(fnv1a_hash("tenant_b")));
    }

    #[test]
    fn header_rejects_v1() {
        // A well-formed v1 header (correct magic, version byte = 1) must be
        // rejected outright — v2 is a clean break with no migration path.
        let mut bytes = BlockHeader {
            format_version: BlockHeader::current_version(),
            block_mode: BlockMode::Pax,
            compression: BlockCompression::None,
            flags: 0,
            column_count: 1,
            row_count: 1,
            block_size: 64,
            checksum: 0,
            collection_id_hash: 0,
            schema_fingerprint: 0,
            min_timestamp_ns: 0,
            max_timestamp_ns: 0,
            tenant_id_hash: 0,
        }
        .to_bytes();
        bytes[4] = 1; // downgrade the version byte to v1
        let err = BlockHeader::from_bytes(&bytes).unwrap_err();
        assert!(
            err.to_string().contains("unsupported block format version"),
            "expected version rejection, got: {err}"
        );
    }

    #[test]
    fn header_v2_round_trips() {
        // The current build stamps FORMAT_VERSION = 2; a round-trip must hold.
        assert_eq!(FORMAT_VERSION, 2);
        let h = BlockHeader {
            format_version: BlockHeader::current_version(),
            block_mode: BlockMode::Pax,
            compression: BlockCompression::Zstd,
            flags: flags::HAS_VECTOR,
            column_count: 3,
            row_count: 10,
            block_size: 4096,
            checksum: 0xabad_1dea,
            collection_id_hash: 7,
            schema_fingerprint: 9,
            min_timestamp_ns: 1,
            max_timestamp_ns: 2,
            tenant_id_hash: 0,
        };
        let h2 = BlockHeader::from_bytes(&h.to_bytes()).unwrap();
        assert_eq!(h2.row_count, 10);
        assert_eq!(h2.column_count, 3);
    }

    #[test]
    fn time_overlap() {
        let h = BlockHeader {
            format_version: BlockHeader::current_version(),
            min_timestamp_ns: 100,
            max_timestamp_ns: 200,
            block_mode: BlockMode::Pax,
            compression: BlockCompression::None,
            flags: 0,
            column_count: 1,
            row_count: 1,
            block_size: 128,
            checksum: 0,
            collection_id_hash: 0,
            schema_fingerprint: 0,
            tenant_id_hash: 0,
        };
        assert!(h.time_overlaps(50, 150));
        assert!(h.time_overlaps(150, 250));
        assert!(!h.time_overlaps(201, 300));
        assert!(!h.time_overlaps(0, 99));
    }
}
