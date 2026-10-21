//! PAX block per-column alignment header — embedding-precision rollout PR 5.
//!
//! Extends the PAX block header with a per-column alignment table so a mixed-
//! precision block (e.g. text embedding column at fp16, image embedding column
//! at fp32) can lay out each column's payload at its native scalar alignment.
//! Memmap'd reads against an aligned column produce a slice that's directly
//! castable to its native type without an intermediate copy.
//!
//! Layout (all multi-byte fields little-endian, enforced cluster-wide by the
//! PR 2a big-endian compile_error):
//!
//! ```text
//! | magic (4 B = b"PXBL")
//! | version (2 B = 2)
//! | num_columns (2 B = u16)
//! | column_table (num_columns entries of 8 B each):
//!     | column_id (2 B = u16)
//!     | scalar_type (1 B = EmbeddingScalarType discriminant)
//!     | alignment_log2 (1 B = log2(alignment bytes))
//!     | offset (4 B = u32, from block start)
//! | row_count (4 B = u32)
//! | reserved (4 B = zero, future use)
//! | per-column payloads — each padded so column.offset is aligned
//! ```
//!
//! `alignment_log2` is a single byte that encodes alignment as `log2(bytes)`,
//! so values 0..=4 cover 1, 2, 4, 8, 16-byte alignments. The mapping locked
//! by LLD §"PAX block per-column alignment (Q7)":
//!
//! | scalar_type | bytes/elem | alignment_log2 |
//! |-------------|------------|----------------|
//! | Fp32        | 4          | 2              |
//! | Fp16        | 2          | 1              |
//! | Bf16        | 2          | 1              |
//! | Int8Scalar  | 1          | 0              |
//! | UInt8Scalar | 1          | 0              |
//!
//! Spec: `docs/12-design/EMBEDDING_PRECISION_LLD_2026_05_22.adoc` §"PAX block
//! per-column alignment (Q7)" and §"PR 5 — PAX per-column alignment header".

use anyhow::{Result, bail};
use proximadb_records::{EmbeddingScalarType, EmbeddingValues, ProximaRecord};

/// File-format magic for a PAX precision-aware block.
pub const PAX_BLOCK_MAGIC: &[u8; 4] = b"PXBL";

/// Version emitted by the precision-aware PAX writer.
pub const PAX_BLOCK_VERSION_V2: u16 = 2;

/// Pre-precision PAX block version. Old blocks have a single fixed alignment
/// marker (4 B) instead of the per-column table; readers detect them via this
/// version and route to the legacy path.
pub const PAX_BLOCK_VERSION_V1: u16 = 1;

/// Bytes for the fixed prefix preceding the column_table.
/// `magic(4) + version(2) + num_columns(2) = 8`.
pub const PAX_BLOCK_HEADER_PREFIX_LEN: usize = 4 + 2 + 2;

/// Bytes per `ColumnTableEntry` on the wire.
pub const COLUMN_TABLE_ENTRY_LEN: usize = 2 + 1 + 1 + 4;

/// Bytes for the fixed suffix following the column_table.
/// `row_count(4) + reserved(4) = 8`.
pub const PAX_BLOCK_HEADER_SUFFIX_LEN: usize = 4 + 4;

/// One row of the PAX block's per-column alignment table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnTableEntry {
    /// Caller-assigned column identifier (typically catalog column id).
    pub column_id: u16,
    /// Scalar type of values in this column.
    pub scalar_type: EmbeddingScalarType,
    /// Alignment of the column's payload, encoded as `log2(bytes)`.
    /// 0 = 1-byte, 1 = 2-byte, 2 = 4-byte, 3 = 8-byte, 4 = 16-byte.
    pub alignment_log2: u8,
    /// Byte offset of the column's payload from the block start.
    /// Guaranteed by [`PaxBlockHeaderV2::encode`] to be aligned to
    /// `1 << alignment_log2`.
    pub offset: u32,
}

/// Parsed PAX block header v2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaxBlockHeaderV2 {
    pub columns: Vec<ColumnTableEntry>,
    pub row_count: u32,
}

/// Result of peeking the magic + version at the start of a PAX block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeekedPaxBlockVersion {
    /// Legacy fp32-only block — caller dispatches to the v1 reader.
    V1,
    /// Precision-aware block — caller parses the v2 header.
    V2,
}

/// LLD-locked alignment for each scalar type, expressed as `log2(bytes)`.
///
/// Reader callers can use this to validate the alignment field in a parsed
/// column table; writer callers can use it to populate `alignment_log2` from
/// just the scalar type without re-deriving the table each time.
pub const fn alignment_log2_for(scalar_type: EmbeddingScalarType) -> u8 {
    match scalar_type {
        EmbeddingScalarType::Fp32 => 2,                             // 4-byte
        EmbeddingScalarType::Fp16 | EmbeddingScalarType::Bf16 => 1, // 2-byte
        EmbeddingScalarType::Int8Scalar | EmbeddingScalarType::UInt8Scalar => 0, // 1-byte
    }
}

/// Round `offset` up to the next multiple of `1 << alignment_log2`.
///
/// Returns the padded offset; pad bytes between the previous payload and the
/// returned offset are the caller's responsibility to zero-fill.
pub fn pad_offset_to_alignment(offset: u32, alignment_log2: u8) -> u32 {
    let alignment = 1u32 << alignment_log2;
    let mask = alignment - 1;
    (offset + mask) & !mask
}

/// One column the writer wants to encode: its id, scalar type, and payload
/// length in bytes. The writer derives offsets + alignment_log2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnLayoutRequest {
    pub column_id: u16,
    pub scalar_type: EmbeddingScalarType,
    pub payload_len: u32,
}

/// Inspect the magic + version of a PAX block and return whether it's the
/// legacy v1 fp32-only layout or the precision-aware v2 layout.
pub fn peek_pax_block_version(data: &[u8]) -> Result<PeekedPaxBlockVersion> {
    if data.len() < PAX_BLOCK_HEADER_PREFIX_LEN {
        bail!(
            "PAX block too short for header peek: need {} bytes, got {}",
            PAX_BLOCK_HEADER_PREFIX_LEN,
            data.len()
        );
    }
    if &data[..4] != PAX_BLOCK_MAGIC {
        bail!(
            "PAX block magic mismatch: expected {:?}, got {:?}",
            PAX_BLOCK_MAGIC,
            &data[..4]
        );
    }
    let version = u16::from_le_bytes([data[4], data[5]]);
    match version {
        PAX_BLOCK_VERSION_V1 => Ok(PeekedPaxBlockVersion::V1),
        PAX_BLOCK_VERSION_V2 => Ok(PeekedPaxBlockVersion::V2),
        other => bail!(
            "unsupported PAX block header version: {} (expected {} or {})",
            other,
            PAX_BLOCK_VERSION_V1,
            PAX_BLOCK_VERSION_V2
        ),
    }
}

impl PaxBlockHeaderV2 {
    /// Build a header + payload byte stream from a list of column layout
    /// requests. Each column's payload starts at its declared alignment;
    /// `payloads` must be supplied in the same order as `requests` and is
    /// concatenated into the block body (the caller writes the actual bytes
    /// into the returned offsets).
    ///
    /// Returns `(header_bytes, column_table)` where the column_table's
    /// offsets are absolute from the block start (i.e. they include
    /// `header_bytes.len()`). The block payload region begins at
    /// `header_bytes.len()` and the caller is expected to write each column's
    /// payload — with appropriate pre-padding — at the returned offsets.
    ///
    /// This is the planning step; actual payload concatenation is left to
    /// the caller (typically a streaming writer that doesn't want to
    /// materialize all column buffers in memory at once).
    pub fn plan_layout(
        requests: &[ColumnLayoutRequest],
        row_count: u32,
    ) -> Result<(Vec<u8>, Vec<ColumnTableEntry>)> {
        if requests.len() > u16::MAX as usize {
            bail!("too many columns: {} exceeds u16 max", requests.len());
        }
        let num_columns = requests.len() as u16;

        // Header length is fixed for a given num_columns.
        let header_len = PAX_BLOCK_HEADER_PREFIX_LEN
            + (num_columns as usize) * COLUMN_TABLE_ENTRY_LEN
            + PAX_BLOCK_HEADER_SUFFIX_LEN;

        // First pass: compute each column's aligned offset (absolute from
        // block start). The block body starts at `header_len`.
        let mut next_offset = header_len as u32;
        let mut table = Vec::with_capacity(requests.len());
        for req in requests {
            let alignment_log2 = alignment_log2_for(req.scalar_type);
            let aligned = pad_offset_to_alignment(next_offset, alignment_log2);
            table.push(ColumnTableEntry {
                column_id: req.column_id,
                scalar_type: req.scalar_type,
                alignment_log2,
                offset: aligned,
            });
            next_offset = aligned
                .checked_add(req.payload_len)
                .ok_or_else(|| anyhow::anyhow!("column offset overflow u32"))?;
        }

        // Second pass: serialize header bytes with the computed offsets.
        let mut header = Vec::with_capacity(header_len);
        header.extend_from_slice(PAX_BLOCK_MAGIC);
        header.extend_from_slice(&PAX_BLOCK_VERSION_V2.to_le_bytes());
        header.extend_from_slice(&num_columns.to_le_bytes());
        for entry in &table {
            header.extend_from_slice(&entry.column_id.to_le_bytes());
            header.push(entry.scalar_type as u8);
            header.push(entry.alignment_log2);
            header.extend_from_slice(&entry.offset.to_le_bytes());
        }
        header.extend_from_slice(&row_count.to_le_bytes());
        header.extend_from_slice(&[0u8; 4]); // reserved

        debug_assert_eq!(header.len(), header_len, "plan_layout header_len drift");
        Ok((header, table))
    }

    /// Parse a v2 PAX block header from the start of `data`.
    ///
    /// Returns the parsed header and the byte length consumed.
    ///
    /// The `try_into()` conversions below are infallible: each takes an
    /// exact-4-byte slice of `data` and converts it to `[u8; 4]`. The
    /// preceding `data.len() < header_len` checks ensure every slice is
    /// in bounds and exactly 4 bytes long.
    #[allow(clippy::unwrap_used)]
    pub fn decode(data: &[u8]) -> Result<(Self, usize)> {
        match peek_pax_block_version(data)? {
            PeekedPaxBlockVersion::V1 => {
                bail!(
                    "PaxBlockHeaderV2::decode called on a v1 block; route via peek_pax_block_version first"
                );
            }
            PeekedPaxBlockVersion::V2 => {}
        }

        if data.len() < PAX_BLOCK_HEADER_PREFIX_LEN {
            bail!("PAX v2 header missing prefix");
        }
        let num_columns = u16::from_le_bytes([data[6], data[7]]) as usize;

        let header_len = PAX_BLOCK_HEADER_PREFIX_LEN
            + num_columns * COLUMN_TABLE_ENTRY_LEN
            + PAX_BLOCK_HEADER_SUFFIX_LEN;
        if data.len() < header_len {
            bail!(
                "PAX v2 header truncated: declared {} bytes (num_columns={}), got {}",
                header_len,
                num_columns,
                data.len()
            );
        }

        let mut off = PAX_BLOCK_HEADER_PREFIX_LEN;
        let mut columns = Vec::with_capacity(num_columns);
        for _ in 0..num_columns {
            let column_id = u16::from_le_bytes([data[off], data[off + 1]]);
            let scalar_byte = data[off + 2];
            let scalar_type = match scalar_byte {
                0x01 => EmbeddingScalarType::Fp32,
                0x02 => EmbeddingScalarType::Fp16,
                0x03 => EmbeddingScalarType::Bf16,
                0x04 => EmbeddingScalarType::Int8Scalar,
                0x05 => EmbeddingScalarType::UInt8Scalar,
                other => bail!(
                    "unknown scalar_type discriminant 0x{:02x} for column {}",
                    other,
                    column_id
                ),
            };
            let alignment_log2 = data[off + 3];
            if alignment_log2 > 4 {
                bail!(
                    "alignment_log2={} for column {} exceeds maximum supported (4 = 16-byte)",
                    alignment_log2,
                    column_id
                );
            }
            let offset = u32::from_le_bytes(data[off + 4..off + 8].try_into().unwrap());
            // Validate the offset is actually aligned (writers must respect this).
            let mask = (1u32 << alignment_log2) - 1;
            if offset & mask != 0 {
                bail!(
                    "column {} offset {} is not aligned to 2^{}={} bytes",
                    column_id,
                    offset,
                    alignment_log2,
                    1u32 << alignment_log2
                );
            }
            columns.push(ColumnTableEntry {
                column_id,
                scalar_type,
                alignment_log2,
                offset,
            });
            off += COLUMN_TABLE_ENTRY_LEN;
        }

        let row_count = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
        off += 4;
        // reserved (4 bytes) — skipped
        off += 4;

        debug_assert_eq!(off, header_len, "decode consumed length mismatch");

        Ok((Self { columns, row_count }, header_len))
    }
}

/// INT-3 dispatch lever: should this batch of records be written as a v2
/// PAX block? Returns true iff at least one [`EmbeddingCell`] carries a
/// non-Fp32 [`EmbeddingValues`] variant. fp32-only batches stay on the
/// existing v1 path so the format risk is bounded to collections that
/// explicitly opt into a non-fp32 canonical precision (LLD risk knob
/// "gate writer choice on column types FIRST, then on the feature flag").
pub fn should_use_v2_for_records(records: &[ProximaRecord]) -> bool {
    records.iter().any(|r| {
        r.embeddings
            .iter()
            .any(|cell| cell.values.scalar_type() != EmbeddingScalarType::Fp32)
    })
}

/// Flatten an [`EmbeddingValues`] payload to little-endian native bytes
/// (no header, no scale/zero_point metadata — those live elsewhere).
///
/// Returns `Err` for Int8Scalar / UInt8Scalar because the v2 PAX column
/// layout has no slot for per-cell scale/zero_point — int8 column
/// payload format is Phase 3 (`EMBEDDING_PRECISION_LLD_2026_05_22.adoc`
/// §"Phase 3 — int8 canonical").
fn write_native_bytes_into(values: &EmbeddingValues, buf: &mut Vec<u8>) -> Result<()> {
    match values {
        EmbeddingValues::Fp32(vs) => {
            buf.reserve(vs.len() * 4);
            for &v in vs {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        EmbeddingValues::Fp16(vs) => {
            buf.reserve(vs.len() * 2);
            for &v in vs {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        EmbeddingValues::Bf16(vs) => {
            buf.reserve(vs.len() * 2);
            for &v in vs {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        EmbeddingValues::Int8Scalar { .. } | EmbeddingValues::UInt8Scalar { .. } => {
            bail!(
                "int8/uint8 columns are not yet supported by v2 PAX block encoding (Phase 3 work)"
            );
        }
    }
    Ok(())
}

/// Encode a batch of records as a single v2 PAX block.
///
/// Column model: each `record.embeddings[c]` is column `c`. All records
/// must agree on (1) the number of embedding cells and (2) the scalar
/// type of every cell at each slot — these are per-collection invariants
/// (one canonical precision per collection, fixed embedding-cell schema).
///
/// Returns `(block_bytes, column_table)`. The column_table mirrors the
/// header's offsets so callers don't have to re-decode to find a column.
/// Pad bytes between columns are zero-filled.
///
/// Errors:
/// - records is empty (callers should skip the v2 path for empty batches)
/// - mixed scalar_type for the same column slot across records
/// - mixed embedding cell count across records
/// - any column carries Int8/UInt8 (Phase 3 work)
pub fn encode_pax_v2_block(records: &[ProximaRecord]) -> Result<(Vec<u8>, Vec<ColumnTableEntry>)> {
    if records.is_empty() {
        bail!("cannot encode empty record batch as v2 PAX block");
    }
    let num_columns = records[0].embeddings.len();
    if num_columns == 0 {
        bail!("v2 PAX block requires at least one embedding column");
    }

    // First pass: validate per-column scalar_type consistency + build
    // per-column native-bytes payloads.
    let mut column_scalar_types: Vec<EmbeddingScalarType> = Vec::with_capacity(num_columns);
    for c in 0..num_columns {
        column_scalar_types.push(records[0].embeddings[c].values.scalar_type());
    }

    let mut column_payloads: Vec<Vec<u8>> = vec![Vec::new(); num_columns];
    for (r_idx, record) in records.iter().enumerate() {
        if record.embeddings.len() != num_columns {
            bail!(
                "record {} has {} embedding cells; expected {} (first record)",
                r_idx,
                record.embeddings.len(),
                num_columns
            );
        }
        for (c_idx, cell) in record.embeddings.iter().enumerate() {
            let st = cell.values.scalar_type();
            if st != column_scalar_types[c_idx] {
                bail!(
                    "record {} column {} scalar_type {:?} mismatches column scalar_type {:?}",
                    r_idx,
                    c_idx,
                    st,
                    column_scalar_types[c_idx]
                );
            }
            write_native_bytes_into(&cell.values, &mut column_payloads[c_idx])?;
        }
    }

    // Build layout requests + delegate to plan_layout for the header.
    let requests: Vec<ColumnLayoutRequest> = column_scalar_types
        .iter()
        .zip(column_payloads.iter())
        .enumerate()
        .map(|(c_idx, (&scalar_type, payload))| {
            // Column id == slot index; callers that want catalog column
            // ids can post-process the returned ColumnTableEntry list.
            ColumnLayoutRequest {
                column_id: c_idx as u16,
                scalar_type,
                payload_len: payload.len() as u32,
            }
        })
        .collect();
    let row_count: u32 = records.len() as u32;
    let (header, table) = PaxBlockHeaderV2::plan_layout(&requests, row_count)?;

    // Stitch: header || pad to col[0].offset || col[0].payload || pad ||
    // col[1].payload || ... Pad bytes are zero (alignment slack only).
    let total_len = table
        .last()
        .map(|t| t.offset as usize + column_payloads.last().map(|p| p.len()).unwrap_or(0))
        .unwrap_or(header.len());
    let mut out = Vec::with_capacity(total_len);
    out.extend_from_slice(&header);
    for (entry, payload) in table.iter().zip(column_payloads.iter()) {
        let target = entry.offset as usize;
        if out.len() < target {
            out.resize(target, 0u8); // zero-fill alignment pad
        }
        debug_assert_eq!(
            out.len(),
            target,
            "writer offset drift at column {}",
            entry.column_id
        );
        out.extend_from_slice(payload);
    }
    Ok((out, table))
}

/// Decode a v2 PAX block's header + per-column raw byte slices.
///
/// Returns the parsed header and a `Vec<&[u8]>` of column-payload slices
/// in the same order as the header's column_table. Each slice spans
/// `[col[i].offset, col[i+1].offset)` for non-final columns and
/// `[col[last].offset, data.len())` for the last column, which means
/// the slice **may include trailing alignment-pad bytes** that satisfy
/// the next column's alignment. The caller is responsible for chopping
/// the slice to its actual payload length (typically
/// `row_count * scalar_type.bytes_per_element()`) before casting to a
/// typed slice (`&[f16]`, `&[bf16]`, etc.).
///
/// The slice start is guaranteed by [`PaxBlockHeaderV2::decode`] to be
/// aligned to `1 << alignment_log2`, so the typed cast is safe.
pub fn decode_pax_v2_block(data: &[u8]) -> Result<(PaxBlockHeaderV2, Vec<&[u8]>)> {
    let (header, _header_len) = PaxBlockHeaderV2::decode(data)?;
    let mut slices = Vec::with_capacity(header.columns.len());
    for (i, entry) in header.columns.iter().enumerate() {
        let start = entry.offset as usize;
        let end = if i + 1 < header.columns.len() {
            header.columns[i + 1].offset as usize
        } else {
            data.len()
        };
        if start > end || end > data.len() {
            bail!(
                "column {} bounds [{}, {}) outside block (len={})",
                entry.column_id,
                start,
                end,
                data.len()
            );
        }
        slices.push(&data[start..end]);
    }
    Ok((header, slices))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magic_constant_is_pxbl_ascii() {
        assert_eq!(PAX_BLOCK_MAGIC, b"PXBL");
    }

    #[test]
    fn version_constants_match_lld() {
        assert_eq!(PAX_BLOCK_VERSION_V1, 1);
        assert_eq!(PAX_BLOCK_VERSION_V2, 2);
    }

    #[test]
    fn alignment_log2_for_locks_lld_mapping() {
        // The LLD §"PAX block per-column alignment (Q7)" table.
        assert_eq!(alignment_log2_for(EmbeddingScalarType::Fp32), 2);
        assert_eq!(alignment_log2_for(EmbeddingScalarType::Fp16), 1);
        assert_eq!(alignment_log2_for(EmbeddingScalarType::Bf16), 1);
        assert_eq!(alignment_log2_for(EmbeddingScalarType::Int8Scalar), 0);
        assert_eq!(alignment_log2_for(EmbeddingScalarType::UInt8Scalar), 0);
    }

    #[test]
    fn pad_offset_rounds_up_to_alignment_boundary() {
        // 1-byte: always aligned.
        for o in [0, 1, 7, 123] {
            assert_eq!(pad_offset_to_alignment(o, 0), o);
        }
        // 2-byte alignment.
        assert_eq!(pad_offset_to_alignment(0, 1), 0);
        assert_eq!(pad_offset_to_alignment(1, 1), 2);
        assert_eq!(pad_offset_to_alignment(2, 1), 2);
        assert_eq!(pad_offset_to_alignment(3, 1), 4);
        // 4-byte alignment.
        assert_eq!(pad_offset_to_alignment(0, 2), 0);
        assert_eq!(pad_offset_to_alignment(1, 2), 4);
        assert_eq!(pad_offset_to_alignment(5, 2), 8);
        // 16-byte alignment.
        assert_eq!(pad_offset_to_alignment(17, 4), 32);
    }

    #[test]
    fn peek_dispatches_v1_v2_and_rejects_garbage() {
        // v1
        let mut buf = Vec::new();
        buf.extend_from_slice(PAX_BLOCK_MAGIC);
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&[0u8; 2]); // num_columns placeholder
        assert_eq!(
            peek_pax_block_version(&buf).unwrap(),
            PeekedPaxBlockVersion::V1
        );
        // v2
        buf[4..6].copy_from_slice(&2u16.to_le_bytes());
        assert_eq!(
            peek_pax_block_version(&buf).unwrap(),
            PeekedPaxBlockVersion::V2
        );
        // garbage version
        buf[4..6].copy_from_slice(&99u16.to_le_bytes());
        assert!(peek_pax_block_version(&buf).is_err());
        // bad magic
        buf[..4].copy_from_slice(b"XXXX");
        assert!(peek_pax_block_version(&buf).is_err());
        // too short
        assert!(peek_pax_block_version(&buf[..5]).is_err());
    }

    #[test]
    fn plan_layout_round_trips_single_fp32_column() {
        let req = ColumnLayoutRequest {
            column_id: 7,
            scalar_type: EmbeddingScalarType::Fp32,
            payload_len: 1024 * 4,
        };
        let (bytes, table) = PaxBlockHeaderV2::plan_layout(&[req], 256).unwrap();
        assert_eq!(table.len(), 1);
        assert_eq!(table[0].column_id, 7);
        assert_eq!(table[0].scalar_type, EmbeddingScalarType::Fp32);
        assert_eq!(table[0].alignment_log2, 2);
        // Header is fixed-size for 1 column: 8 prefix + 8 entry + 8 suffix = 24.
        assert_eq!(bytes.len(), 24);
        // The fp32 payload's aligned offset should be 24 (already 4-aligned).
        assert_eq!(table[0].offset, 24);
    }

    #[test]
    fn plan_layout_pads_fp16_after_fp32_to_2byte_boundary() {
        // Layout: fp32 column (12 elements * 4B = 48B) then fp16 column.
        // After fp32 ends at offset 24 + 48 = 72, fp16 alignment is 2, so its
        // offset stays at 72 (already 2-aligned). Easy case.
        let reqs = [
            ColumnLayoutRequest {
                column_id: 1,
                scalar_type: EmbeddingScalarType::Fp32,
                payload_len: 48,
            },
            ColumnLayoutRequest {
                column_id: 2,
                scalar_type: EmbeddingScalarType::Fp16,
                payload_len: 24,
            },
        ];
        let (header_bytes, table) = PaxBlockHeaderV2::plan_layout(&reqs, 12).unwrap();
        // Header: 8 prefix + 2*8 entries + 8 suffix = 32.
        assert_eq!(header_bytes.len(), 32);
        assert_eq!(table[0].offset, 32, "fp32 starts right after header");
        assert_eq!(
            table[1].offset,
            32 + 48,
            "fp16 abuts fp32 (already 2-aligned)"
        );
        // Both offsets are aligned.
        assert_eq!(table[0].offset % 4, 0);
        assert_eq!(table[1].offset % 2, 0);
    }

    #[test]
    fn plan_layout_pads_when_fp16_follows_int8_unaligned() {
        // int8 column with odd-length payload forces the fp16 column to skip
        // 1 padding byte to land on a 2-byte boundary.
        let reqs = [
            ColumnLayoutRequest {
                column_id: 1,
                scalar_type: EmbeddingScalarType::Int8Scalar,
                payload_len: 5, // odd → next offset is odd
            },
            ColumnLayoutRequest {
                column_id: 2,
                scalar_type: EmbeddingScalarType::Fp16,
                payload_len: 8,
            },
        ];
        let (_, table) = PaxBlockHeaderV2::plan_layout(&reqs, 4).unwrap();
        let int8_end = table[0].offset + 5;
        assert_eq!(int8_end % 2, 1, "int8 ends on odd offset");
        assert_eq!(
            table[1].offset,
            int8_end + 1,
            "fp16 column gets 1 byte of padding"
        );
        assert_eq!(table[1].offset % 2, 0);
    }

    #[test]
    fn plan_layout_pads_fp32_after_int8_to_4byte_boundary() {
        let reqs = [
            ColumnLayoutRequest {
                column_id: 1,
                scalar_type: EmbeddingScalarType::Int8Scalar,
                payload_len: 7,
            },
            ColumnLayoutRequest {
                column_id: 2,
                scalar_type: EmbeddingScalarType::Fp32,
                payload_len: 16,
            },
        ];
        let (_, table) = PaxBlockHeaderV2::plan_layout(&reqs, 4).unwrap();
        let int8_end = table[0].offset + 7;
        let expected = (int8_end + 3) & !3u32;
        assert_eq!(table[1].offset, expected);
        assert_eq!(table[1].offset % 4, 0);
    }

    #[test]
    fn header_round_trips_mixed_precision_block() {
        // Mirror the LLD test goal: a block with both fp32 and fp16 columns.
        let reqs = [
            ColumnLayoutRequest {
                column_id: 100,
                scalar_type: EmbeddingScalarType::Fp32,
                payload_len: 64,
            },
            ColumnLayoutRequest {
                column_id: 200,
                scalar_type: EmbeddingScalarType::Fp16,
                payload_len: 32,
            },
            ColumnLayoutRequest {
                column_id: 300,
                scalar_type: EmbeddingScalarType::Int8Scalar,
                payload_len: 12,
            },
        ];
        let (bytes, table) = PaxBlockHeaderV2::plan_layout(&reqs, 16).unwrap();
        let (parsed, consumed) = PaxBlockHeaderV2::decode(&bytes).unwrap();
        assert_eq!(consumed, bytes.len());
        assert_eq!(parsed.row_count, 16);
        assert_eq!(parsed.columns.len(), 3);
        assert_eq!(parsed.columns, table);
    }

    #[test]
    fn header_round_trips_zero_columns() {
        // Edge case: a block with no embedding columns (e.g. graph topology
        // only). Header should still be valid.
        let (bytes, table) = PaxBlockHeaderV2::plan_layout(&[], 0).unwrap();
        assert!(table.is_empty());
        // 8 prefix + 0 entries + 8 suffix = 16.
        assert_eq!(bytes.len(), 16);
        let (parsed, consumed) = PaxBlockHeaderV2::decode(&bytes).unwrap();
        assert_eq!(consumed, 16);
        assert!(parsed.columns.is_empty());
        assert_eq!(parsed.row_count, 0);
    }

    #[test]
    fn decode_rejects_v1_block() {
        // v1 block: magic + version=1 + some bytes.
        let mut buf = Vec::new();
        buf.extend_from_slice(PAX_BLOCK_MAGIC);
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&[0u8; 32]);
        let err = PaxBlockHeaderV2::decode(&buf).unwrap_err().to_string();
        assert!(err.contains("v1 block"), "got: {err}");
    }

    #[test]
    fn decode_rejects_unknown_scalar_discriminant() {
        let reqs = [ColumnLayoutRequest {
            column_id: 1,
            scalar_type: EmbeddingScalarType::Fp32,
            payload_len: 4,
        }];
        let (mut bytes, _) = PaxBlockHeaderV2::plan_layout(&reqs, 1).unwrap();
        // scalar_type byte is at: prefix(8) + column_id(2) = offset 10.
        bytes[10] = 0xFE;
        let err = PaxBlockHeaderV2::decode(&bytes).unwrap_err().to_string();
        assert!(err.contains("0xfe"), "got: {err}");
    }

    #[test]
    fn decode_rejects_alignment_log2_above_max() {
        let reqs = [ColumnLayoutRequest {
            column_id: 1,
            scalar_type: EmbeddingScalarType::Fp32,
            payload_len: 4,
        }];
        let (mut bytes, _) = PaxBlockHeaderV2::plan_layout(&reqs, 1).unwrap();
        // alignment_log2 byte is at: prefix(8) + column_id(2) + scalar(1) = 11.
        bytes[11] = 5; // 32-byte alignment is not supported.
        let err = PaxBlockHeaderV2::decode(&bytes).unwrap_err().to_string();
        assert!(err.contains("alignment_log2=5"), "got: {err}");
    }

    #[test]
    fn decode_rejects_misaligned_offset() {
        // Craft an entry by hand: column_id=1, fp32, alignment_log2=2 (4B),
        // but offset=21 which is NOT divisible by 4 → invalid.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(PAX_BLOCK_MAGIC);
        bytes.extend_from_slice(&PAX_BLOCK_VERSION_V2.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes()); // num_columns
        bytes.extend_from_slice(&1u16.to_le_bytes()); // column_id
        bytes.push(EmbeddingScalarType::Fp32 as u8);
        bytes.push(2); // alignment_log2 (4-byte)
        bytes.extend_from_slice(&21u32.to_le_bytes()); // offset (not 4-aligned)
        bytes.extend_from_slice(&1u32.to_le_bytes()); // row_count
        bytes.extend_from_slice(&[0u8; 4]); // reserved
        let err = PaxBlockHeaderV2::decode(&bytes).unwrap_err().to_string();
        assert!(err.contains("not aligned"), "got: {err}");
    }

    #[test]
    fn decode_rejects_truncated_header() {
        let reqs = [ColumnLayoutRequest {
            column_id: 1,
            scalar_type: EmbeddingScalarType::Fp32,
            payload_len: 4,
        }];
        let (bytes, _) = PaxBlockHeaderV2::plan_layout(&reqs, 1).unwrap();
        for cut in 0..bytes.len() {
            assert!(
                PaxBlockHeaderV2::decode(&bytes[..cut]).is_err(),
                "decode succeeded on truncated buffer cut at {cut}"
            );
        }
    }

    #[test]
    fn encoded_layout_matches_lld_byte_offsets() {
        let reqs = [ColumnLayoutRequest {
            column_id: 0xABCD,
            scalar_type: EmbeddingScalarType::Fp16,
            payload_len: 4,
        }];
        let (bytes, _) = PaxBlockHeaderV2::plan_layout(&reqs, 17).unwrap();
        // magic
        assert_eq!(&bytes[0..4], PAX_BLOCK_MAGIC);
        // version
        assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), 2);
        // num_columns
        assert_eq!(u16::from_le_bytes([bytes[6], bytes[7]]), 1);
        // column_id at offset 8
        assert_eq!(u16::from_le_bytes([bytes[8], bytes[9]]), 0xABCD);
        // scalar_type at offset 10
        assert_eq!(bytes[10], EmbeddingScalarType::Fp16 as u8);
        // alignment_log2 at offset 11 — fp16 → 1
        assert_eq!(bytes[11], 1);
        // offset at 12..16 — fp16 column lands at 16 (header_len = 24? no — 1 col → 24; but fp16 alignment requires 2-byte, 24 is aligned).
        let col_offset = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        // header_len = 8 + 8 + 8 = 24, fp16 alignment-2, 24 is already aligned.
        assert_eq!(col_offset, 24);
        // row_count at offset 16 (right after the single 8-byte entry → 8 + 8 = 16).
        assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 17);
        // reserved zeroed
        assert_eq!(&bytes[20..24], &[0u8; 4]);
    }

    #[test]
    fn fp16_payload_offset_is_2byte_aligned_for_cast() {
        // The whole point of PR 5: fp16 columns must land on a 2-byte boundary
        // so memmap'd reads cast to &[u16] safely.
        let reqs = [
            ColumnLayoutRequest {
                column_id: 1,
                scalar_type: EmbeddingScalarType::Int8Scalar,
                payload_len: 3, // odd
            },
            ColumnLayoutRequest {
                column_id: 2,
                scalar_type: EmbeddingScalarType::Fp16,
                payload_len: 8,
            },
            ColumnLayoutRequest {
                column_id: 3,
                scalar_type: EmbeddingScalarType::Bf16,
                payload_len: 8,
            },
        ];
        let (_, table) = PaxBlockHeaderV2::plan_layout(&reqs, 4).unwrap();
        for entry in &table {
            let bytes_aligned = 1u32 << entry.alignment_log2;
            assert_eq!(
                entry.offset % bytes_aligned,
                0,
                "column {} at offset {} not aligned to {}",
                entry.column_id,
                entry.offset,
                bytes_aligned
            );
        }
    }

    // ---- INT-3 writer/reader/dispatch tests --------------------------------

    use proximadb_records::{EmbeddingCell, EmbeddingValues, ProximaRecord};

    fn record_with_cell(oid: &str, values: EmbeddingValues) -> ProximaRecord {
        ProximaRecord {
            oid: oid.to_string(),
            embeddings: vec![EmbeddingCell::new_typed(
                "test",
                "dense_vector",
                values.len() as u32,
                values,
            )],
            ..Default::default()
        }
    }

    #[test]
    fn should_use_v2_for_records_is_false_for_all_fp32() {
        let recs = vec![
            record_with_cell("a", EmbeddingValues::Fp32(vec![1.0, 2.0])),
            record_with_cell("b", EmbeddingValues::Fp32(vec![3.0, 4.0])),
        ];
        assert!(!should_use_v2_for_records(&recs));
    }

    #[test]
    fn should_use_v2_for_records_is_true_when_any_record_is_fp16() {
        let recs = vec![
            record_with_cell("a", EmbeddingValues::Fp32(vec![1.0, 2.0])),
            record_with_cell(
                "b",
                EmbeddingValues::Fp16(vec![half::f16::from_f32(3.0), half::f16::from_f32(4.0)]),
            ),
        ];
        assert!(should_use_v2_for_records(&recs));
    }

    #[test]
    fn should_use_v2_for_records_is_false_for_empty_input() {
        let recs: Vec<ProximaRecord> = vec![];
        assert!(!should_use_v2_for_records(&recs));
    }

    #[test]
    fn encode_pax_v2_block_round_trips_fp16_column() {
        // 3 records, 1 column, 4 fp16 elements each = 3*4*2 = 24 payload bytes.
        let mkv = |xs: &[f32]| {
            EmbeddingValues::Fp16(xs.iter().map(|&x| half::f16::from_f32(x)).collect())
        };
        let recs = vec![
            record_with_cell("r0", mkv(&[1.0, 2.0, 3.0, 4.0])),
            record_with_cell("r1", mkv(&[5.0, 6.0, 7.0, 8.0])),
            record_with_cell("r2", mkv(&[9.0, 10.0, 11.0, 12.0])),
        ];
        let (bytes, table) = encode_pax_v2_block(&recs).unwrap();

        // Header: 8 prefix + 1*8 entry + 8 suffix = 24 bytes.
        // fp16 alignment_log2 = 1 (2-byte) → offset 24 is already 2-aligned.
        // Payload: 3 rows * 4 elts * 2 B = 24 bytes.
        assert_eq!(bytes.len(), 24 + 24);
        assert_eq!(table.len(), 1);
        assert_eq!(table[0].column_id, 0);
        assert_eq!(table[0].scalar_type, EmbeddingScalarType::Fp16);
        assert_eq!(table[0].alignment_log2, 1);
        assert_eq!(table[0].offset, 24);

        // Decode + verify the column slice round-trips the fp16 elements.
        let (parsed, slices) = decode_pax_v2_block(&bytes).unwrap();
        assert_eq!(parsed.row_count, 3);
        assert_eq!(parsed.columns, table);
        assert_eq!(slices.len(), 1);
        assert_eq!(slices[0].len(), 24);

        let mut got = Vec::with_capacity(12);
        for chunk in slices[0].as_chunks::<2>().0 {
            got.push(half::f16::from_le_bytes(*chunk).to_f32());
        }
        assert_eq!(
            got,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0
            ]
        );
    }

    #[test]
    fn encode_pax_v2_block_pads_fp32_column_after_int_alignment_boundary() {
        // Mixed: a small fp32 column followed by an fp16 column.
        // Test that the offset table places each column at its native
        // alignment and that decode_pax_v2_block hands back the right bytes.
        let recs = vec![
            ProximaRecord {
                oid: "r0".into(),
                embeddings: vec![
                    EmbeddingCell::new_fp32("fp32-col", "v", 2, vec![1.0, 2.0]),
                    EmbeddingCell::new_typed(
                        "fp16-col",
                        "v",
                        1,
                        EmbeddingValues::Fp16(vec![half::f16::from_f32(7.0)]),
                    ),
                ],
                ..Default::default()
            },
            ProximaRecord {
                oid: "r1".into(),
                embeddings: vec![
                    EmbeddingCell::new_fp32("fp32-col", "v", 2, vec![3.0, 4.0]),
                    EmbeddingCell::new_typed(
                        "fp16-col",
                        "v",
                        1,
                        EmbeddingValues::Fp16(vec![half::f16::from_f32(8.0)]),
                    ),
                ],
                ..Default::default()
            },
        ];
        let (bytes, table) = encode_pax_v2_block(&recs).unwrap();
        assert_eq!(table.len(), 2);
        assert_eq!(table[0].scalar_type, EmbeddingScalarType::Fp32);
        assert_eq!(table[1].scalar_type, EmbeddingScalarType::Fp16);
        // Header is 8 prefix + 2*8 entries + 8 suffix = 32 bytes.
        // col[0] starts at 32 (already 4-aligned). 4 fp32 * 4 B = 16 B. Ends at 48.
        // col[1] is fp16 (2-aligned), 48 is already 2-aligned → no pad.
        assert_eq!(table[0].offset, 32);
        assert_eq!(table[1].offset, 48);

        let (_, slices) = decode_pax_v2_block(&bytes).unwrap();
        // Column 0: 4 fp32 elements
        assert_eq!(slices[0].len(), 16);
        let fp32_got: Vec<f32> = slices[0]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert_eq!(fp32_got, vec![1.0, 2.0, 3.0, 4.0]);
        // Column 1: 2 fp16 elements
        assert!(slices[1].len() >= 4);
        let fp16_got: Vec<f32> = slices[1][..4]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::f16::from_le_bytes(*c).to_f32())
            .collect();
        assert_eq!(fp16_got, vec![7.0, 8.0]);
    }

    #[test]
    fn encode_pax_v2_block_rejects_per_column_scalar_type_mismatch() {
        let recs = vec![
            record_with_cell("r0", EmbeddingValues::Fp16(vec![half::f16::from_f32(1.0)])),
            record_with_cell("r1", EmbeddingValues::Fp32(vec![2.0])),
        ];
        let err = encode_pax_v2_block(&recs).unwrap_err().to_string();
        assert!(err.contains("scalar_type"), "got: {err}");
    }

    #[test]
    fn encode_pax_v2_block_rejects_int8_column_with_phase3_message() {
        let recs = vec![record_with_cell(
            "r0",
            EmbeddingValues::Int8Scalar {
                values: vec![1, 2, 3],
                scale: 1.0,
                zero_point: 0,
            },
        )];
        let err = encode_pax_v2_block(&recs).unwrap_err().to_string();
        assert!(err.contains("Phase 3"), "got: {err}");
    }

    #[test]
    fn encode_pax_v2_block_rejects_empty_input() {
        assert!(encode_pax_v2_block(&[]).is_err());
    }

    #[test]
    fn encode_pax_v2_block_rejects_records_with_zero_embedding_cells() {
        let rec = ProximaRecord {
            oid: "r0".into(),
            embeddings: vec![],
            ..Default::default()
        };
        let err = encode_pax_v2_block(&[rec]).unwrap_err().to_string();
        assert!(err.contains("at least one"), "got: {err}");
    }
}
