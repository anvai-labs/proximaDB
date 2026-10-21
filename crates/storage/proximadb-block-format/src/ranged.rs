// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Footer-first ranged-read planning — the object-storage read path.
//!
//! This module is **pure**: it computes the exact byte ranges a reader must
//! fetch and decodes the slices it is handed. It performs no I/O and depends on
//! no filesystem, so it is trivially mockable (see the byte-counter tests) and
//! has no dependency on the root crate's async `FileSystem`. The async adapter
//! (root crate) reads the planned ranges via `FileSystem::read_range`/
//! `read_ranges` and hands the bytes back here to decode — dependency inversion,
//! the same separation Parquet draws between file metadata and its async reader.
//!
//! Read order for one block on object storage:
//! 1. `head`/size → [`footer_tail_range`] → fetch the trailing 32 B → `BlockFooter`.
//! 2. [`metadata_ranges`] → fetch column footer + vector params + row-group index.
//! 3. [`BlockLayout::assemble`] → an in-memory metadata view (no stripe bytes).
//! 4. Prune (zone maps / row groups) → [`BlockLayout::column_stripe_range`] (and
//!    [`BlockLayout::vector_row_group_range`]) → fetch only the surviving slabs.
//! 5. Decode via [`BlockLayout::decode_i64_column`] / `decode_f32_vec_column`.

use std::ops::Range;

use anyhow::{Result, bail};

use crate::{
    reader::{
        RankMetric, decode_f32_vec_v2, decode_i64_with_encoding, decode_str_with_encoding,
        parse_rabitq_codes, score_rerank_rows_from_stripe,
    },
    record::col_id,
    rowgroup::RowGroupBlock,
    stripe::{COLUMN_META_SIZE, ColumnMeta},
    vparam::{QUANT_FP16, QUANT_RABITQ, QUANT_SQ8, VectorParamBlock},
    writer::{BLOCK_FOOTER_SIZE, BlockFooter},
};
use proximadb_codec::functions::rabitq;
use proximadb_codec::{RaBitQCode, RaBitQParams, functions};

/// Byte range of the trailing [`BlockFooter`] for an object of `object_size`.
pub fn footer_tail_range(object_size: u64) -> Result<Range<u64>> {
    if object_size < BLOCK_FOOTER_SIZE as u64 {
        bail!("object too small for a block footer: {object_size} bytes");
    }
    Ok((object_size - BLOCK_FOOTER_SIZE as u64)..object_size)
}

/// The metadata ranges to fetch after the footer is known: the column footer
/// (always), the vector-param block, and the row-group index (when present).
pub struct MetadataRanges {
    pub col_meta: Range<u64>,
    pub vparam: Option<Range<u64>>,
    pub rgdir: Option<Range<u64>>,
}

/// Compute the metadata ranges from a parsed footer. `object_size` bounds the
/// row-group index, whose length runs up to the block footer.
pub fn metadata_ranges(footer: &BlockFooter, object_size: u64) -> MetadataRanges {
    let col_start = footer.col_footer_offset as u64;
    let col_end = col_start + footer.n_columns as u64 * COLUMN_META_SIZE as u64;
    let vparam = if footer.vparam_offset != 0 && footer.vparam_len != 0 {
        Some(footer.vparam_offset as u64..(footer.vparam_offset as u64 + footer.vparam_len as u64))
    } else {
        None
    };
    let footer_start = object_size - BLOCK_FOOTER_SIZE as u64;
    let rgdir = if footer.rgdir_offset != 0 {
        Some(footer.rgdir_offset as u64..footer_start)
    } else {
        None
    };
    MetadataRanges {
        col_meta: col_start..col_end,
        vparam,
        rgdir,
    }
}

/// In-memory metadata view of a block — everything needed to plan stripe reads,
/// without any stripe bytes.
pub struct BlockLayout {
    footer: BlockFooter,
    columns: Vec<ColumnMeta>,
    vparams: VectorParamBlock,
    rowgroups: RowGroupBlock,
}

impl BlockLayout {
    /// Build the layout from the footer + the bytes of the metadata ranges
    /// returned by [`metadata_ranges`].
    pub fn assemble(
        footer: BlockFooter,
        col_meta_bytes: &[u8],
        vparam_bytes: Option<&[u8]>,
        rgdir_bytes: Option<&[u8]>,
    ) -> Result<Self> {
        let n = footer.n_columns as usize;
        if col_meta_bytes.len() < n * COLUMN_META_SIZE {
            bail!(
                "column footer bytes too short: {} < {}",
                col_meta_bytes.len(),
                n * COLUMN_META_SIZE
            );
        }
        let mut columns = Vec::with_capacity(n);
        for i in 0..n {
            columns.push(ColumnMeta::from_bytes(
                &col_meta_bytes[i * COLUMN_META_SIZE..],
            )?);
        }
        let vparams = match vparam_bytes {
            Some(b) => VectorParamBlock::from_bytes(b)?,
            None => VectorParamBlock::default(),
        };
        let rowgroups = match rgdir_bytes {
            Some(b) => RowGroupBlock::from_bytes(b)?,
            None => RowGroupBlock::default(),
        };
        Ok(Self {
            footer,
            columns,
            vparams,
            rowgroups,
        })
    }

    pub fn row_count(&self) -> u32 {
        self.footer.n_rows
    }

    /// Approximate in-memory byte size of this metadata view — used as the
    /// weight for a byte-budgeted footer cache. Counts the column footer, the
    /// vector-param block (incl. RaBitQ centroids), and the row-group index.
    pub fn approx_bytes(&self) -> usize {
        let cols = self.columns.len() * COLUMN_META_SIZE;
        let vparam: usize = self.vparams.entries.len() * crate::vparam::ENTRY_SIZE
            + self
                .vparams
                .rabitq
                .iter()
                .map(|r| 16 + r.centroid.len() * 4)
                .sum::<usize>();
        let rg = 12 + self.rowgroups.entries.len() * crate::rowgroup::ENTRY_SIZE;
        BLOCK_FOOTER_SIZE + cols + vparam + rg
    }

    pub fn column_metas(&self) -> &[ColumnMeta] {
        &self.columns
    }

    pub fn vector_params(&self) -> &VectorParamBlock {
        &self.vparams
    }

    pub fn row_groups(&self) -> &RowGroupBlock {
        &self.rowgroups
    }

    fn meta(&self, column_id: i32) -> Option<&ColumnMeta> {
        self.columns.iter().find(|m| m.column_id == column_id)
    }

    /// Byte range of a column's full stripe (the whole-column fetch).
    pub fn column_stripe_range(&self, column_id: i32) -> Option<Range<u64>> {
        self.meta(column_id).map(|m| {
            let start = m.stripe_offset as u64;
            start..(start + m.stripe_len as u64)
        })
    }

    /// Byte sub-range covering rows `[start_row, end_row)` of a **fixed-stride
    /// vector** column — the partial-fetch that makes row-group reads pay off.
    /// Returns `None` for non-vector columns or out-of-range rows. The caller
    /// must ALSO fetch the stripe's validity bitmap prefix (the first
    /// `ceil(n_rows/8)` bytes of the stripe) to know which rows are null.
    pub fn vector_row_range(
        &self,
        column_id: i32,
        start_row: u32,
        end_row: u32,
    ) -> Option<Range<u64>> {
        let entry = self.vparams.get(column_id)?;
        if self.vparams.transform(column_id).is_some() {
            // Transformed stripes are microchunk-addressable rather than
            // fixed-stride. Until the planner consumes the codec directory,
            // callers must fetch the full (smaller) stripe.
            return None;
        }
        let m = self.meta(column_id)?;
        let n = self.footer.n_rows;
        if start_row > end_row || end_row > n {
            return None;
        }
        let stride = match entry.quant_kind {
            QUANT_SQ8 => entry.dim as u64,
            _ => entry.dim as u64 * 4,
        };
        let bitmap_len = (n as u64).div_ceil(8);
        let payload_base = m.stripe_offset as u64 + bitmap_len;
        Some(payload_base + start_row as u64 * stride..payload_base + end_row as u64 * stride)
    }

    /// Decode an i64 column from its (range-fetched) stripe bytes.
    pub fn decode_i64_column(&self, column_id: i32, stripe_bytes: &[u8]) -> Result<Vec<i64>> {
        let m = self
            .meta(column_id)
            .ok_or_else(|| anyhow::anyhow!("column {column_id} not in block"))?;
        let row_count = self.footer.n_rows as usize;
        if m.stripe_len == 0 && m.null_count as usize == row_count {
            return Ok(vec![i64::MIN; row_count]);
        }
        let payload = if m.is_lz4_compressed {
            functions::lossless_compression::decompress_lz4(stripe_bytes)?
        } else {
            stripe_bytes.to_vec()
        };
        decode_i64_with_encoding(&payload, m.encoding_id, row_count)
    }

    /// Decode an f32 vector column from its (range-fetched) stripe bytes.
    pub fn decode_f32_vec_column(
        &self,
        column_id: i32,
        stripe_bytes: &[u8],
    ) -> Result<Vec<Option<Vec<f32>>>> {
        let entry = self
            .vparams
            .get(column_id)
            .ok_or_else(|| anyhow::anyhow!("no vector params for column {column_id}"))?;
        decode_f32_vec_v2(
            stripe_bytes,
            self.footer.n_rows as usize,
            entry,
            self.vparams.transform(column_id),
        )
    }

    /// Decode a string column (e.g. `col_id::OID`) from its range-fetched stripe
    /// bytes — the ranged analogue of [`crate::reader::PaxBlockReader::decode_str_stripe`],
    /// used to attach ids to the ranked candidate rows.
    pub fn decode_str_column(
        &self,
        column_id: i32,
        stripe_bytes: &[u8],
    ) -> Option<Vec<Option<String>>> {
        let meta = self.meta(column_id)?;
        let row_count = self.footer.n_rows as usize;
        if meta.stripe_len == 0 && meta.null_count as usize == row_count {
            return Some(vec![None; row_count]);
        }
        let payload = if meta.is_lz4_compressed {
            functions::lossless_compression::decompress_lz4(stripe_bytes).ok()?
        } else {
            stripe_bytes.to_vec()
        };
        decode_str_with_encoding(&payload, meta.encoding_id, row_count).ok()
    }

    // ── Ranged RaBitQ cascade (ADR-057 / TD-RDSTRAT-3 S1b) ──────────────────────
    // Stage-1 rank + Stage-2 rerank on RANGE-FETCHED stripe bytes, mirroring
    // `PaxBlockReader::{rabitq_rank,rerank_rows}` (which operate on the whole
    // in-memory block) so the selective striped read returns byte-identical
    // candidates + reranked order. Parity is asserted against the whole-block
    // reader in the tests.

    /// Decode this block's RaBitQ codes from the range-fetched `EMBED_BASE` codes
    /// stripe. Returns `None` when the column isn't RaBitQ-quantized. Mirrors
    /// [`crate::reader::PaxBlockReader::decode_rabitq_codes`] but takes the fetched
    /// stripe bytes (no whole-block read).
    pub fn decode_rabitq_codes(
        &self,
        column_id: i32,
        codes_stripe: &[u8],
    ) -> Option<(RaBitQParams, Vec<Option<RaBitQCode>>)> {
        let entry = self.vparams.get(column_id)?;
        if entry.quant_kind != QUANT_RABITQ {
            return None;
        }
        let col = self.vparams.rabitq_column(column_id)?;
        let n = self.footer.n_rows as usize;
        let codes = parse_rabitq_codes(codes_stripe, n, entry.dim as usize).ok()?;
        let params = RaBitQParams {
            dim: entry.dim as usize,
            seed: col.seed,
            centroid: col.centroid.clone(),
        };
        Some((params, codes))
    }

    /// Stage-1 RaBitQ candidate ranking on the range-fetched `EMBED_BASE` codes
    /// stripe: decode codes, rotate the query once, return up to `pool` row indices
    /// nearest-first. `None` if the column isn't RaBitQ-quantized. Byte-identical to
    /// [`crate::reader::PaxBlockReader::rabitq_rank`].
    pub fn rabitq_rank(
        &self,
        query: &[f32],
        pool: usize,
        metric: RankMetric,
        codes_stripe: &[u8],
    ) -> Option<Vec<usize>> {
        let (params, codes) = self.decode_rabitq_codes(col_id::EMBED_BASE, codes_stripe)?;
        let rotation = rabitq::build_rotation_cached(params.dim, params.seed);
        let q_rotated = rabitq::rotate_query(query, &params, &rotation);
        Some(match metric {
            RankMetric::L2 => rabitq::rank_candidates(&q_rotated, &codes, pool),
            RankMetric::Cosine => rabitq::rank_candidates_ip(&q_rotated, &codes, pool),
            // DotProduct ranks by the same IP proxy as Cosine (max inner product).
            RankMetric::DotProduct => rabitq::rank_candidates_ip(&q_rotated, &codes, pool),
        })
    }

    /// Stage-2 rerank of RaBitQ candidate `rows` against the range-fetched rerank
    /// column stripe (SQ8 or FP16), returning `(row, distance)` nearest-first. The
    /// caller fetches only the needed slice of the stripe (the bitmap prefix + the
    /// candidate rows via [`Self::vector_row_range`]); here `stripe_bytes` is the
    /// full rerank stripe (bitmap + payload) for a whole-stripe fetch, matching
    /// [`crate::reader::PaxBlockReader::rerank_rows`] byte-for-byte. `None` if the
    /// column is absent or not SQ8/FP16 (caller falls back to the RaBitQ order).
    pub fn rerank_candidate_rows(
        &self,
        column_id: i32,
        query: &[f32],
        rows: &[usize],
        metric: RankMetric,
        stripe_bytes: &[u8],
    ) -> Option<Vec<(usize, f32)>> {
        let entry = self.vparams.get(column_id)?;
        if entry.quant_kind != QUANT_SQ8 && entry.quant_kind != QUANT_FP16 {
            return None;
        }
        score_rerank_rows_from_stripe(
            entry,
            self.vparams.transform(column_id),
            stripe_bytes,
            self.footer.n_rows as usize,
            query,
            rows,
            metric,
        )
        .ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::{BlockCompression, BlockMode};
    use crate::reader::PaxBlockReader;
    use crate::record::col_id;
    use crate::writer::PaxBlockWriter;
    use proximadb_records::{EmbeddingCell, EmbeddingValues, ProximaRecord};
    use std::cell::Cell;

    /// A mock object that serves byte ranges from an in-memory block and counts
    /// how many bytes were read — the stand-in for an S3 ranged GET.
    struct CountingSource {
        data: Vec<u8>,
        bytes_read: Cell<u64>,
    }
    impl CountingSource {
        fn new(data: Vec<u8>) -> Self {
            Self {
                data,
                bytes_read: Cell::new(0),
            }
        }
        fn size(&self) -> u64 {
            self.data.len() as u64
        }
        fn read(&self, r: Range<u64>) -> Vec<u8> {
            self.bytes_read
                .set(self.bytes_read.get() + (r.end - r.start));
            self.data[r.start as usize..r.end as usize].to_vec()
        }
    }

    fn rec_with_vec(oid: &str, ts: i64, v: Vec<f32>) -> ProximaRecord {
        let dim = v.len() as u32;
        ProximaRecord {
            oid: oid.into(),
            tenant_id: "t".into(),
            created_at_ns: ts,
            updated_at_ns: ts,
            embeddings: vec![EmbeddingCell::new_fp32("m", "dense", dim, v)],
            ..Default::default()
        }
    }

    fn build_block(n: usize, dim: usize) -> Vec<u8> {
        let mut w = PaxBlockWriter::new(BlockMode::Pax, BlockCompression::None, "c", 0, 1);
        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|d| (i * dim + d) as f32 * 0.001).collect();
            w.add_record(&rec_with_vec(&format!("r{i}"), 1000 + i as i64, v))
                .unwrap();
        }
        w.flush().unwrap()
    }

    /// Build a RaBitQ (+ SQ8 rerank) PAX block: `EMBED_BASE` carries RaBitQ codes,
    /// `RERANK_BASE` the co-located SQ8 rerank column — the cascade's two columns.
    fn build_rabitq_block(n: usize, dim: usize) -> Vec<u8> {
        let mut w = PaxBlockWriter::new(BlockMode::Pax, BlockCompression::None, "c", 0, 1)
            .with_quant(crate::writer::VectorQuant::RaBitQ);
        for i in 0..n {
            let v: Vec<f32> = (0..dim)
                .map(|d| ((i * 7 + d * 3) % 97) as f32 * 0.01)
                .collect();
            w.add_record(&rec_with_vec(&format!("r{i}"), 1000 + i as i64, v))
                .unwrap();
        }
        w.flush().unwrap()
    }

    /// Assemble a `BlockLayout` from a whole in-memory block (test helper): the
    /// footer-first ranged plan, with the metadata regions sliced from `block`.
    fn layout_of(block: &[u8]) -> BlockLayout {
        let sz = block.len() as u64;
        let fr = footer_tail_range(sz).unwrap();
        let footer = BlockFooter::from_bytes(&block[fr.start as usize..fr.end as usize]).unwrap();
        let mr = metadata_ranges(&footer, sz);
        let col_meta = block[mr.col_meta.start as usize..mr.col_meta.end as usize].to_vec();
        let vparam = mr
            .vparam
            .as_ref()
            .map(|r| block[r.start as usize..r.end as usize].to_vec());
        let rgdir = mr
            .rgdir
            .as_ref()
            .map(|r| block[r.start as usize..r.end as usize].to_vec());
        BlockLayout::assemble(footer, &col_meta, vparam.as_deref(), rgdir.as_deref()).unwrap()
    }

    /// TD-RDSTRAT-3 S1b Slice A parity: the ranged `BlockLayout` cascade
    /// (`rabitq_rank` + `rerank_candidate_rows` on RANGE-FETCHED stripes) must
    /// return byte-identical candidates + reranked order to the whole-block
    /// `PaxBlockReader` — the correctness contract of the selective striped read.
    #[test]
    fn ranged_rabitq_cascade_matches_whole_block() {
        use crate::reader::RankMetric;
        let block = build_rabitq_block(300, 64);
        let reader = PaxBlockReader::open(&block).unwrap();
        let layout = layout_of(&block);

        // Whole-stripe fetches (byte-identical to what the reader reads internally).
        let cr = layout.column_stripe_range(col_id::EMBED_BASE).unwrap();
        let codes = block[cr.start as usize..cr.end as usize].to_vec();
        let rr = layout.column_stripe_range(col_id::RERANK_BASE).unwrap();
        let rerank = block[rr.start as usize..rr.end as usize].to_vec();

        for metric in [RankMetric::L2, RankMetric::Cosine] {
            let query: Vec<f32> = (0..64).map(|d| (d as f32 * 0.03).sin()).collect();
            // whole-block reference
            let cand_w = reader.rabitq_rank(&query, 64, metric).unwrap();
            let rer_w = reader.rerank_rows(0, &query, &cand_w, metric).unwrap();
            // ranged
            let cand_r = layout.rabitq_rank(&query, 64, metric, &codes).unwrap();
            let rer_r = layout
                .rerank_candidate_rows(col_id::RERANK_BASE, &query, &cand_r, metric, &rerank)
                .unwrap();

            assert_eq!(cand_r, cand_w, "ranged rabitq_rank diverged ({metric:?})");
            assert_eq!(rer_r, rer_w, "ranged rerank diverged ({metric:?})");
        }
    }

    /// Footer-first ranged open + decode of one column equals the whole-block
    /// read, while fetching far fewer bytes than the block size.
    #[test]
    fn ranged_open_equals_whole_read() {
        let block = build_block(256, 128);
        let src = CountingSource::new(block.clone());

        // 1. footer tail.
        let footer_r = footer_tail_range(src.size()).unwrap();
        let footer = BlockFooter::from_bytes(&src.read(footer_r)).unwrap();
        // 2. metadata.
        let mr = metadata_ranges(&footer, src.size());
        let col_meta = src.read(mr.col_meta);
        let vparam = mr.vparam.map(|r| src.read(r));
        let rgdir = mr.rgdir.map(|r| src.read(r));
        let layout =
            BlockLayout::assemble(footer, &col_meta, vparam.as_deref(), rgdir.as_deref()).unwrap();

        // 3. fetch ONLY the embedding stripe + decode.
        let stripe_r = layout.column_stripe_range(col_id::EMBED_BASE).unwrap();
        let stripe = src.read(stripe_r);
        let ranged = layout
            .decode_f32_vec_column(col_id::EMBED_BASE, &stripe)
            .unwrap();

        // Whole-block reference.
        let whole = PaxBlockReader::open(&block).unwrap();
        let expected = whole.decode_f32_vec_stripe(col_id::EMBED_BASE).unwrap();
        assert_eq!(ranged, expected);

        // We read strictly less than the whole block (footer + metadata + one
        // SQ8 stripe ≈ 256*128 + small, vs the whole f32-input block).
        assert!(
            src.bytes_read.get() < block.len() as u64,
            "ranged read {} not < block {}",
            src.bytes_read.get(),
            block.len()
        );
    }

    /// Reading a single scalar column does not pull the (much larger) vector
    /// stripe — bytes read are dominated by just that column + metadata.
    #[test]
    fn ranged_reads_only_requested_column() {
        let block = build_block(256, 256); // big vector stripe
        let src = CountingSource::new(block.clone());

        let footer =
            BlockFooter::from_bytes(&src.read(footer_tail_range(src.size()).unwrap())).unwrap();
        let mr = metadata_ranges(&footer, src.size());
        let col_meta = src.read(mr.col_meta);
        let vparam = mr.vparam.map(|r| src.read(r));
        let rgdir = mr.rgdir.map(|r| src.read(r));
        let layout =
            BlockLayout::assemble(footer, &col_meta, vparam.as_deref(), rgdir.as_deref()).unwrap();

        let stripe_r = layout.column_stripe_range(col_id::CREATED_AT).unwrap();
        let stripe = src.read(stripe_r);
        let created = layout
            .decode_i64_column(col_id::CREATED_AT, &stripe)
            .unwrap();
        assert_eq!(created.len(), 256);
        assert_eq!(created[0], 1000);

        // The vector stripe alone is 256*256 = 65536 bytes; reading only the
        // i64 column + metadata must be a small fraction of the whole block.
        assert!(
            src.bytes_read.get() < (block.len() as u64) / 4,
            "scalar-only ranged read {} not << block {}",
            src.bytes_read.get(),
            block.len()
        );
    }
}
