// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! TD-RDSTRAT-3 S1a — pure segment-level *striped-read* planner (ADR-057).
//!
//! Plans the exact segment-absolute byte ranges a selective ("striped") read of a
//! PAX segment would fetch for the RaBitQ cascade's two-stage access pattern —
//! **Stage 1** (a small rank/codes column stripe for *all* rows) + **Stage 2** (a
//! large rerank column for *only* the candidate rows) — instead of the whole
//! segment the cascade reads today (`search/mod.rs:271`).
//!
//! This module is **pure** (no I/O): it composes the tail [`SegmentIndex`]
//! (random-access block directory) with the per-block, footer-first ranged reader
//! in `proximadb-block-format::ranged`, shifting each block-relative range to
//! segment-absolute by the block's `offset`. It exists to (a) *measure the
//! headroom* (whole-segment vs striped bytes — the TD-RDSTRAT-3 S0 number) and
//! (b) drive the S1b async reader, which fetches exactly `ranges` via
//! `FileSystem::read_ranges`.
//!
//! It changes no on-disk format and no behaviour — the whole-segment read stays
//! the default until S1b/S2 wire the flag + chooser.

use std::ops::Range;

use anyhow::{Result, bail};
use proximadb_block_format::{BlockFooter, BlockLayout, footer_tail_range, metadata_ranges};

use crate::pax_block::{SEGMENT_MAGIC, SegmentIndex};

/// A planned selective ("striped") read of one PAX segment, as segment-absolute
/// byte ranges plus a byte-cost **bracket** vs a whole-segment read.
///
/// The bracket is deliberate: the real Stage-2 cost depends on how scattered the
/// ranked candidate rows are (best case they are contiguous / coalesce to one
/// span; worst case they span the whole rerank stripe) and on whether the segment
/// carries a heavy exact-f32 tier the rerank skips. This pure plan therefore
/// reports both ends; the **flip decision gates on the S1b *physical* io_trace**,
/// never on this estimate (TD-RDSTRAT-3 / ADR-052 observe→act).
#[derive(Debug, Clone)]
pub struct SegmentStripedPlan {
    /// Every byte range the **best-case** striped read fetches, **segment-absolute**
    /// and ready to hand to `FileSystem::read_ranges` (which coalesces adjacent).
    pub ranges: Vec<Range<u64>>,
    /// Best-case striped cost: Σ metadata + rank stripes + the *contiguous*
    /// candidate rows + the tail-directory read. Optimistic (assumes clustered
    /// candidates).
    pub striped_bytes_best: u64,
    /// Worst-case striped cost: as best, but Stage 2 reads the **whole rerank
    /// stripe** of every block with any candidate (scattered candidates coalescing
    /// to the full stripe). This is the conservative bound.
    pub striped_bytes_worst: u64,
    /// The whole-segment read's byte cost (the segment file size) — the baseline
    /// the striped read replaces.
    pub whole_bytes: u64,
    /// Number of blocks the plan touched (after the segment directory is read).
    pub blocks: usize,
}

impl SegmentStripedPlan {
    /// Best-case fractional bytes-moved reduction vs the whole-segment read (an
    /// upper bound on the achievable saving; the flip gates on the S1b trace).
    pub fn reduction_best(&self) -> f64 {
        Self::ratio(self.striped_bytes_best, self.whole_bytes)
    }

    /// Worst-case (conservative) fractional bytes-moved reduction — Stage 2 reads
    /// the whole rerank stripe. A robust headroom clears the gate even here.
    pub fn reduction_worst(&self) -> f64 {
        Self::ratio(self.striped_bytes_worst, self.whole_bytes)
    }

    fn ratio(striped: u64, whole: u64) -> f64 {
        if whole == 0 {
            0.0
        } else {
            1.0 - (striped as f64 / whole as f64)
        }
    }
}

fn push(ranges: &mut Vec<Range<u64>>, total: &mut u64, start: u64, end: u64) {
    debug_assert!(end >= start);
    *total += end - start;
    ranges.push(start..end);
}

/// Plan a two-stage striped read of `segment` (the whole segment bytes).
///
/// - `rank_col`   — the small column whose full stripe Stage 1 ranks over (the
///   RaBitQ codes column in production; any small column here).
/// - `rerank_col` — the large vector column Stage 2 reads for **only** the first
///   `candidates_per_block` rows (a stand-in for the ranked candidate set — the
///   real candidate ids come from Stage 1 in S1b).
///
/// Returns the segment-absolute ranges + byte totals. Pure; performs no I/O.
/// Falls back cleanly (skips a stage) when a column is absent — the caller then
/// reads the whole segment (mixed-read-safe), never a wrong subset.
pub fn plan_segment_striped_read(
    segment: &[u8],
    rank_col: i32,
    rerank_col: i32,
    candidates_per_block: usize,
) -> Result<SegmentStripedPlan> {
    let whole_bytes = segment.len() as u64;
    if segment.len() < SEGMENT_MAGIC.len() {
        bail!("buffer too small to be a PAX segment");
    }
    let before_magic = &segment[..segment.len() - SEGMENT_MAGIC.len()];
    let index = SegmentIndex::locate(before_magic)?;

    // A block-relative range must lie within the block; a corrupt footer offset
    // must fail-closed (bail → the caller falls back to the whole-segment read),
    // never panic (repo NO-panic mandate). Mirrors `PaxBlockReader::open`'s guards.
    let checked = |r: &Range<u64>, size: u64| -> Result<Range<u64>> {
        if r.start > r.end || r.end > size {
            bail!(
                "striped-read plan: range {r:?} outside a block of {size} bytes (corrupt footer)"
            );
        }
        Ok(r.clone())
    };

    let mut ranges: Vec<Range<u64>> = Vec::new();
    // `common` = tail directory + per-block metadata + the Stage-1 rank stripe
    // (paid by both best and worst). Stage 2 differs: best = the contiguous
    // candidate rows; worst = the whole rerank stripe (scattered candidates).
    let mut common_bytes: u64 = 0;
    let mut best_stage2: u64 = 0;
    let mut worst_stage2: u64 = 0;

    // Stage 0: the tail directory — one small GET yields every block's offset/size
    // + zone summary (v2), so blocks prune with no per-block metadata reads.
    common_bytes += (index.to_bytes().len() + SEGMENT_MAGIC.len()) as u64;

    let blocks = index.blocks.len();
    for entry in &index.blocks {
        let off = entry.offset;
        let size = entry.size as u64;
        let (bstart, bend) = (off as usize, off as usize + size as usize);
        if bend > segment.len() {
            bail!("segment index block [{off}..+{size}] runs past the segment");
        }
        let block = &segment[bstart..bend];

        // Block footer + metadata (footer-first ranged open). Ranges are
        // block-relative; shift by `off` for segment-absolute fetch. Every
        // footer-derived range is bounds-checked before slicing.
        let fr = footer_tail_range(size)?;
        let footer = BlockFooter::from_bytes(&block[fr.start as usize..fr.end as usize])?;
        push(&mut ranges, &mut common_bytes, off + fr.start, off + fr.end);

        let mr = metadata_ranges(&footer, size);
        let cm = checked(&mr.col_meta, size)?;
        let col_meta = block[cm.start as usize..cm.end as usize].to_vec();
        push(&mut ranges, &mut common_bytes, off + cm.start, off + cm.end);
        let vparam = match mr.vparam.as_ref() {
            Some(r) => {
                let r = checked(r, size)?;
                push(&mut ranges, &mut common_bytes, off + r.start, off + r.end);
                Some(block[r.start as usize..r.end as usize].to_vec())
            }
            None => None,
        };
        let rgdir = match mr.rgdir.as_ref() {
            Some(r) => {
                let r = checked(r, size)?;
                push(&mut ranges, &mut common_bytes, off + r.start, off + r.end);
                Some(block[r.start as usize..r.end as usize].to_vec())
            }
            None => None,
        };

        let layout = BlockLayout::assemble(footer, &col_meta, vparam.as_deref(), rgdir.as_deref())?;

        // Stage 1: the rank column's full stripe (all rows).
        if let Some(sr) = layout.column_stripe_range(rank_col) {
            let sr = checked(&sr, size)?;
            push(&mut ranges, &mut common_bytes, off + sr.start, off + sr.end);
        }

        // Stage 2: the rerank column. best = the first `candidates_per_block` rows
        // (+ the validity-bitmap prefix the ranged reader needs); worst = the whole
        // rerank stripe (scattered candidates coalesce to the full stripe).
        let rerank_stripe = layout
            .column_stripe_range(rerank_col)
            .map(|r| checked(&r, size))
            .transpose()?;
        let rows = (candidates_per_block as u32).min(layout.row_count());
        if rows > 0
            && let Some(stripe) = &rerank_stripe
            && let Some(rr) = layout.vector_row_range(rerank_col, 0, rows)
        {
            let rr = checked(&rr, size)?;
            let bitmap_len = layout.row_count().div_ceil(8) as u64;
            // Best-case ranges = what S1b fetches for clustered candidates.
            push(
                &mut ranges,
                &mut best_stage2,
                off + stripe.start,
                off + stripe.start + bitmap_len,
            );
            push(&mut ranges, &mut best_stage2, off + rr.start, off + rr.end);
            // Worst-case = the whole rerank stripe (accounting only; not a range).
            worst_stage2 += stripe.end - stripe.start;
        }
    }

    let striped_bytes_best = common_bytes + best_stage2;
    let striped_bytes_worst = common_bytes + worst_stage2.max(best_stage2);

    Ok(SegmentStripedPlan {
        ranges,
        striped_bytes_best,
        striped_bytes_worst,
        whole_bytes,
        blocks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pax_block::PaxSegmentWriter;
    use proximadb_block_format::record::col_id;
    use proximadb_block_format::{BlockCompression, BlockMode};
    use proximadb_records::{EmbeddingCell, EmbeddingValues, ProximaRecord};

    const DIM: usize = 128;

    fn rec(i: usize) -> ProximaRecord {
        let v: Vec<f32> = (0..DIM).map(|d| (i * DIM + d) as f32 * 0.001).collect();
        ProximaRecord {
            oid: format!("r{i:06}"),
            tenant_id: "t".into(),
            created_at_ns: 1_000 + i as i64,
            updated_at_ns: 1_000 + i as i64,
            embeddings: vec![EmbeddingCell::new_fp32("m", "dense", DIM as u32, v)],
            ..Default::default()
        }
    }

    /// Build a multi-block PAX segment; a small block threshold forces several
    /// blocks so the plan exercises the segment directory + per-block ranged open.
    fn build_multiblock_segment() -> Vec<u8> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.pax");
        // ~64 KiB threshold with 512-byte f32 vectors ⇒ ~120 rows/block, several blocks.
        let mut w = PaxSegmentWriter::new(
            &path,
            BlockMode::Pax,
            BlockCompression::None,
            "c",
            0,
            1,
            Some(64 * 1024),
        );
        for i in 0..1_000 {
            w.add_record(&rec(i)).unwrap();
        }
        w.finish().unwrap();
        std::fs::read(&path).unwrap()
    }

    #[test]
    fn striped_plan_moves_far_fewer_bytes_than_whole_segment() {
        let seg = build_multiblock_segment();

        // Sanity: it really is a multi-block segment (else the headroom is trivial).
        let idx = SegmentIndex::locate(&seg[..seg.len() - SEGMENT_MAGIC.len()]).unwrap();
        assert!(
            idx.blocks.len() >= 2,
            "expected a multi-block segment, got {} block(s)",
            idx.blocks.len()
        );

        // Stage 1 ranks over the small CREATED_AT (i64) column for all rows;
        // Stage 2 rereads the large EMBED_BASE vector column for 10 candidates/block.
        let plan = plan_segment_striped_read(&seg, col_id::CREATED_AT, col_id::EMBED_BASE, 10)
            .expect("plan");

        // Report the BRACKET, not a point estimate. The best case (clustered
        // candidates) is an optimistic upper bound; the worst case (scattered
        // candidates → whole rerank stripe) is the conservative floor. The real
        // headroom — which depends on candidate scatter, segment size vs the
        // RaBitQ pool, and whether a heavy exact-f32 tier is present/skipped —
        // is decided by the S1b PHYSICAL io_trace, never by this pure plan.
        eprintln!(
            "[S1a headroom bracket] best {:.1}% / worst {:.1}% (best {} / worst {} / whole {} bytes, {} blocks). \
             NOTE: pure-plan estimate with CREATED_AT rank proxy (real RaBitQ codes ≈3× larger) — \
             flip gates on the S1b trace.",
            100.0 * plan.reduction_best(),
            100.0 * plan.reduction_worst(),
            plan.striped_bytes_best,
            plan.striped_bytes_worst,
            plan.whole_bytes,
            plan.blocks,
        );

        assert_eq!(plan.blocks, idx.blocks.len());
        assert!(plan.striped_bytes_worst >= plan.striped_bytes_best);
        // Weak, robust invariant only: even the best case cannot exceed the whole
        // read. We deliberately do NOT assert a headroom threshold here — the
        // magnitude is config-dependent and is measured for real in S1b.
        assert!(
            plan.striped_bytes_best < plan.whole_bytes,
            "best-case striped {} must be < whole {}",
            plan.striped_bytes_best,
            plan.whole_bytes
        );
        // Every planned (best-case) range is inside the segment.
        for r in &plan.ranges {
            assert!(r.end <= plan.whole_bytes, "range {r:?} past segment");
        }
    }

    #[test]
    fn plan_rejects_undersized_buffer() {
        assert!(
            plan_segment_striped_read(b"tiny", col_id::CREATED_AT, col_id::EMBED_BASE, 4).is_err()
        );
    }
}
