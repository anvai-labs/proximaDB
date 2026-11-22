// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Row allow-set for the filter-aware cascade (ADR-089 / TD-FPRUNE-1 P1).
//!
//! A dense bitset over a segment's **global row ordinals** (the shared currency
//! of the PAX regions — ADR-065 row-ordinal identity). The metadata pre-stage
//! marks the rows whose scalar predicate matched; the RaBitQ/SQ8 rank stages
//! then score **only** allowed rows, so the survivor pool is filled with
//! predicate-matching candidates instead of being diluted by rows the filter
//! would discard (post-filtering the pool would silently lose recall).

/// Dense bitset keyed by global row ordinal `0..n_slots`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowAllow {
    words: Vec<u64>,
    n_slots: usize,
    len: usize,
}

impl RowAllow {
    /// An empty allow-set over `n_slots` rows (nothing allowed).
    pub fn new(n_slots: usize) -> Self {
        Self {
            words: vec![0u64; n_slots.div_ceil(64)],
            n_slots,
            len: 0,
        }
    }

    /// Number of allowed rows.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when no row is allowed.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total row slots the set was sized for.
    pub fn n_slots(&self) -> usize {
        self.n_slots
    }

    /// Allow row `row`. Out-of-range rows are ignored (the set is sized from
    /// the footer row count; anything beyond it cannot be a real row).
    pub fn insert(&mut self, row: usize) {
        if row >= self.n_slots {
            return;
        }
        let (w, b) = (row / 64, row % 64);
        let mask = 1u64 << b;
        if self.words[w] & mask == 0 {
            self.words[w] |= mask;
            self.len += 1;
        }
    }

    /// Whether row `row` is allowed.
    pub fn contains(&self, row: usize) -> bool {
        if row >= self.n_slots {
            return false;
        }
        self.words[row / 64] & (1u64 << (row % 64)) != 0
    }

    /// Whether ANY row in `range` is allowed — used to skip whole A0 cells
    /// (contiguous global-row ranges) before their Region-A extents are
    /// fetched.
    pub fn any_in_range(&self, range: std::ops::Range<usize>) -> bool {
        let start = range.start.min(self.n_slots);
        let end = range.end.min(self.n_slots);
        if start >= end {
            return false;
        }
        let (first_w, last_w) = (start / 64, (end - 1) / 64);
        for (w, mut word) in (first_w..=last_w).zip(self.words[first_w..=last_w].iter().copied()) {
            if w == first_w {
                word &= u64::MAX << (start % 64);
            }
            if w == last_w {
                let tail = (end - 1) % 64;
                word &= u64::MAX >> (63 - tail);
            }
            if word != 0 {
                return true;
            }
        }
        false
    }

    /// Count of allowed rows in `range` — the number of predicate-matching rows
    /// in a contiguous global-row range (an A0 cell). TD-FPRUNE-1 M3 uses this to
    /// size the ADAPTIVE probe: walk cells nearest-first, accumulate matching
    /// rows, and stop once enough matching candidates are covered — all in RAM
    /// (cell row-ranges + this bitset), so no extra I/O.
    pub fn count_in_range(&self, range: std::ops::Range<usize>) -> usize {
        let start = range.start.min(self.n_slots);
        let end = range.end.min(self.n_slots);
        if start >= end {
            return 0;
        }
        let (first_w, last_w) = (start / 64, (end - 1) / 64);
        let mut count = 0usize;
        for (w, mut word) in (first_w..=last_w).zip(self.words[first_w..=last_w].iter().copied()) {
            if w == first_w {
                word &= u64::MAX << (start % 64);
            }
            if w == last_w {
                let tail = (end - 1) % 64;
                word &= u64::MAX >> (63 - tail);
            }
            count += word.count_ones() as usize;
        }
        count
    }
}

impl FromIterator<usize> for RowAllow {
    /// Build from row ordinals; the set is sized to the max ordinal + 1.
    /// (Production callers size via [`RowAllow::new`] from the footer row
    /// count; this is a test/utility convenience.)
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        let rows: Vec<usize> = iter.into_iter().collect();
        let n = rows.iter().max().map_or(0, |m| m + 1);
        let mut set = Self::new(n);
        for r in rows {
            set.insert(r);
        }
        set
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_contains_len_and_bounds() {
        let mut s = RowAllow::new(130);
        assert!(s.is_empty());
        s.insert(0);
        s.insert(63);
        s.insert(64);
        s.insert(129);
        s.insert(129); // idempotent
        s.insert(130); // out of range — ignored
        assert_eq!(s.len(), 4);
        assert!(s.contains(0) && s.contains(63) && s.contains(64) && s.contains(129));
        assert!(!s.contains(1) && !s.contains(128) && !s.contains(130) && !s.contains(9999));
    }

    #[test]
    fn any_in_range_hits_and_misses_across_word_boundaries() {
        let s: RowAllow = [70usize, 200].into_iter().collect();
        assert!(s.any_in_range(0..71));
        assert!(s.any_in_range(70..71));
        assert!(s.any_in_range(64..128));
        assert!(!s.any_in_range(0..70));
        assert!(!s.any_in_range(71..200));
        assert!(s.any_in_range(128..201));
        assert!(!s.any_in_range(201..500)); // beyond slots
        assert!(!s.any_in_range(10..10)); // empty range
    }

    #[test]
    fn count_in_range_across_word_boundaries() {
        // Rows spanning three 64-bit words: 5, 63, 64, 130, 200.
        let s: RowAllow = [5usize, 63, 64, 130, 200].into_iter().collect();
        assert_eq!(s.count_in_range(0..201), 5); // all
        assert_eq!(s.count_in_range(0..64), 2); // 5, 63
        assert_eq!(s.count_in_range(64..131), 2); // 64, 130
        assert_eq!(s.count_in_range(63..65), 2); // 63, 64 across the word edge
        assert_eq!(s.count_in_range(6..63), 0); // gap
        assert_eq!(s.count_in_range(200..500), 1); // 200 (clamped past slots)
        assert_eq!(s.count_in_range(10..10), 0); // empty range
    }
}
