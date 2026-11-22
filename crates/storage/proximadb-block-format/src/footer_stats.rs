// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Footer-resident per-block column statistics (ADR-089 / TD-FPRUNE-1 P2).
//!
//! P1's filtered-cascade Stage F fetches every Region-D block body to prune it
//! (via the shared [`evaluate_block`](crate::evaluate_block) kernel over each
//! block's in-body [`ColumnMeta`]). P2 lifts those same min/max bounds — for the
//! collection's *filterable* columns only — into the segment footer, so a
//! filtered query can drop a block from the footer the cascade already reads,
//! **without fetching the block body at all** (the "parquet footer" win).
//!
//! The bounds are byte-identical to the in-body ones (same [`ColumnMeta`]), so
//! footer-stats pruning is exactly as sound as P1's body-stats pruning — a block
//! is skipped only when provably empty. The stats reuse the existing
//! [`BlockZoneSource`] contract, so no new pruning logic is introduced: the same
//! `evaluate_block` walk runs against a footer-backed source.
//!
//! On-disk shape (little-endian): `[n_columns u16] [ColumnMeta × n_columns]`,
//! each `ColumnMeta` its canonical fixed [`COLUMN_META_SIZE`] bytes. An absent
//! payload (footer `stats_len == 0`, the legacy default) decodes to an empty
//! stats set → "may match" for every predicate (conservative), so old segments
//! and old readers are unaffected — mixed-read-safe by construction.

use crate::prune::BlockZoneSource;
use crate::rowgroup::RowGroupBlock;
use crate::stripe::{COLUMN_META_SIZE, ColumnMeta};

/// The filterable columns' [`ColumnMeta`] for one block, lifted into the footer.
#[derive(Debug, Clone, Default)]
pub struct FooterBlockStats {
    columns: Vec<ColumnMeta>,
    row_count: u32,
    // A footer source prunes at BLOCK granularity only; the per-row-group
    // sub-index stays in the block body. An empty index makes
    // `evaluate_row_groups` fall back to "all groups" (block-level pruning still
    // applies) — but Stage F calls `evaluate_block`, which never consults it.
    empty_row_groups: RowGroupBlock,
}

impl FooterBlockStats {
    /// Build from a block's full column-meta set, keeping only the columns whose
    /// id is in `keep` (the collection's filterable columns resolved to PAX
    /// column ids). Columns with no usable bounds (`distinct_hint == 0`) are
    /// dropped — they would only ever say "may match", so storing them wastes
    /// footer bytes.
    pub fn from_column_metas(row_count: u32, metas: &[ColumnMeta], keep: &[i32]) -> Self {
        let columns = metas
            .iter()
            .filter(|m| keep.contains(&m.column_id) && m.distinct_hint != 0)
            .cloned()
            .collect();
        Self {
            columns,
            row_count,
            empty_row_groups: RowGroupBlock::default(),
        }
    }

    /// True when no column bounds are carried (nothing to prune on).
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// The filterable columns' bounds carried in this block's footer stats.
    pub fn columns(&self) -> &[ColumnMeta] {
        &self.columns
    }

    /// Serialize to the footer stats payload (`[n u16][ColumnMeta × n]`).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(2 + self.columns.len() * COLUMN_META_SIZE);
        out.extend_from_slice(&(self.columns.len() as u16).to_le_bytes());
        for meta in &self.columns {
            out.extend_from_slice(&meta.to_bytes());
        }
        out
    }

    /// Parse a footer stats payload. `row_count` is the block's row count (from
    /// the footer entry) — carried so the source can answer `row_count_hint`.
    /// An empty / malformed-length slice yields an empty (all-"may-match") set,
    /// never an error: unknown/short stats degrade conservatively.
    pub fn from_bytes(payload: &[u8], row_count: u32) -> Self {
        if payload.len() < 2 {
            return Self::empty(row_count);
        }
        let n = u16::from_le_bytes([payload[0], payload[1]]) as usize;
        let mut columns = Vec::with_capacity(n);
        let mut pos = 2usize;
        for _ in 0..n {
            let Some(chunk) = payload.get(pos..pos + COLUMN_META_SIZE) else {
                break; // truncated → keep what parsed (conservative)
            };
            match ColumnMeta::from_bytes(chunk) {
                Ok(meta) => columns.push(meta),
                Err(_) => break,
            }
            pos += COLUMN_META_SIZE;
        }
        Self {
            columns,
            row_count,
            empty_row_groups: RowGroupBlock::default(),
        }
    }

    fn empty(row_count: u32) -> Self {
        Self {
            columns: Vec::new(),
            row_count,
            empty_row_groups: RowGroupBlock::default(),
        }
    }

    fn column(&self, column_id: i32) -> Option<&ColumnMeta> {
        self.columns.iter().find(|m| m.column_id == column_id)
    }
}

// Same block-level zone-map semantics as `BlockLayout` (the metadata-only ranged
// path): find the column by id, delegate to its `ColumnMeta` bounds; an unknown
// column conservatively "may match". `evaluate_block` uses only these methods.
impl BlockZoneSource for FooterBlockStats {
    fn column_meta_type(&self, column_id: i32) -> Option<u8> {
        self.column(column_id).map(|m| m.data_type_id)
    }
    fn may_contain_i64(&self, column_id: i32, value: i64) -> bool {
        self.column(column_id)
            .map(|m| m.i64_in_range(value))
            .unwrap_or(true)
    }
    fn range_overlaps_i64(&self, column_id: i32, lo: i64, hi: i64) -> bool {
        self.column(column_id)
            .map(|m| m.i64_range_overlaps(lo, hi))
            .unwrap_or(true)
    }
    fn range_overlaps_f64(&self, column_id: i32, lo: f64, hi: f64) -> bool {
        self.column(column_id)
            .map(|m| m.f64_range_overlaps(lo, hi))
            .unwrap_or(true)
    }
    fn may_contain_str(&self, column_id: i32, value: &str) -> bool {
        // Footer stats carry hash bounds (like the ranged path), not the block's
        // string bloom — so string equality prunes by the min/max hash window.
        self.column(column_id)
            .map(|m| m.hash64_in_range(crate::header::fnv1a_hash(value)))
            .unwrap_or(true)
    }
    fn row_group_index(&self) -> &RowGroupBlock {
        &self.empty_row_groups
    }
    fn row_count_hint(&self) -> u32 {
        self.row_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prune::{PruneResult, evaluate_block};
    use crate::stripe::ColumnRole;
    use proximadb_filter_expression::{ComparisonOperator, FilterExpression};

    // A canonical i64 column meta with usable [min,max] bounds.
    fn i64_col(column_id: i32, min: i64, max: i64) -> ColumnMeta {
        let mut m = ColumnMeta {
            column_id,
            role: ColumnRole::UserDefined,
            data_type_id: 0x03, // DT_I64
            encoding_id: 0,
            nullable: false,
            has_bloom: false,
            is_sorted: false,
            is_lz4_compressed: false,
            stripe_offset: 0,
            stripe_len: 0,
            null_count: 0,
            distinct_hint: 8, // non-zero ⇒ bounds are usable
            min_val: [0; 16],
            max_val: [0; 16],
            bloom_offset: 0,
            bloom_len: 0,
        };
        m.min_val[0..8].copy_from_slice(&min.to_le_bytes());
        m.max_val[0..8].copy_from_slice(&max.to_le_bytes());
        m
    }

    fn eq_i64(field: &str, v: i64) -> FilterExpression {
        FilterExpression::Comparison {
            field: field.to_string(),
            operator: ComparisonOperator::Equals,
            value: serde_json::json!(v),
        }
    }

    #[test]
    fn from_metas_keeps_only_filterable_with_bounds() {
        let mut unbounded = i64_col(200, 0, 0);
        unbounded.distinct_hint = 0; // no usable bounds
        let metas = vec![i64_col(100, 5, 9), i64_col(101, 1, 3), unbounded];
        // keep 100 and 200; 200 is dropped (no bounds), 101 not requested.
        let stats = FooterBlockStats::from_column_metas(42, &metas, &[100, 200]);
        assert_eq!(stats.columns().len(), 1);
        assert_eq!(stats.columns()[0].column_id, 100);
        assert_eq!(stats.row_count_hint(), 42);
    }

    #[test]
    fn bytes_round_trip() {
        let metas = vec![i64_col(100, 5, 9), i64_col(101, -4, 4)];
        let stats = FooterBlockStats::from_column_metas(7, &metas, &[100, 101]);
        let bytes = stats.to_bytes();
        let back = FooterBlockStats::from_bytes(&bytes, 7);
        // Compare by the canonical column-meta bytes (ColumnMeta has no PartialEq).
        assert_eq!(back.columns().len(), 2);
        assert_eq!(back.to_bytes(), bytes);
        assert_eq!(back.row_count_hint(), 7);
    }

    #[test]
    fn empty_payload_is_all_may_match() {
        let stats = FooterBlockStats::from_bytes(&[], 10);
        assert!(stats.is_empty());
        // An empty source can never prune — every predicate "may match".
        assert!(stats.may_contain_i64(100, 12345));
        let filter = eq_i64("part", 5);
        let f2c: &dyn Fn(&str) -> Option<i32> = &|f| (f == "part").then_some(100);
        assert_eq!(evaluate_block(&stats, &filter, f2c), PruneResult::MayMatch);
    }

    #[test]
    fn evaluate_block_prunes_and_keeps_via_footer_stats() {
        // Column 100 holds values in [5, 9].
        let metas = vec![i64_col(100, 5, 9)];
        let stats = FooterBlockStats::from_column_metas(16, &metas, &[100]);
        let f2c: &dyn Fn(&str) -> Option<i32> = &|f| (f == "part").then_some(100);

        // == 7 is inside [5,9] ⇒ may match (must read the block).
        assert_eq!(
            evaluate_block(&stats, &eq_i64("part", 7), f2c),
            PruneResult::MayMatch
        );
        // == 100 is outside [5,9] ⇒ provably empty ⇒ Skip (prune, zero body GET).
        assert_eq!(
            evaluate_block(&stats, &eq_i64("part", 100), f2c),
            PruneResult::Skip
        );
    }

    #[test]
    fn footer_prune_matches_body_prune_for_same_metas() {
        // Soundness/parity: pruning from footer stats gives the same verdict as
        // pruning from the block body's ColumnMeta (both are the same bounds).
        let metas = vec![i64_col(100, 10, 20)];
        let stats = FooterBlockStats::from_column_metas(8, &metas, &[100]);
        let f2c: &dyn Fn(&str) -> Option<i32> = &|f| (f == "p").then_some(100);
        for (v, expected) in [
            (15, PruneResult::MayMatch),
            (10, PruneResult::MayMatch),
            (20, PruneResult::MayMatch),
            (9, PruneResult::Skip),
            (21, PruneResult::Skip),
        ] {
            assert_eq!(
                evaluate_block(&stats, &eq_i64("p", v), f2c),
                expected,
                "v={v}"
            );
        }
    }
}
