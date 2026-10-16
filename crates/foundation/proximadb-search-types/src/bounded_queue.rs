//! Bounded priority queue for efficient top-k search
//!
//! This module provides a min-heap based priority queue that maintains
//! only the top-k results during vector search, enabling efficient
//! memory usage and early termination.

use crate::results::OptimizedSearchRecord;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A wrapper for search records that implements reverse ordering for min-heap
#[derive(Clone)]
struct MinHeapEntry {
    pub record: OptimizedSearchRecord,
}

impl PartialEq for MinHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.record.score == other.record.score
    }
}

impl Eq for MinHeapEntry {}

impl Ord for MinHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (lowest score at top)
        other
            .record
            .score
            .partial_cmp(&self.record.score)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for MinHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Bounded priority queue for maintaining top-k search results
pub struct BoundedPriorityQueue {
    heap: BinaryHeap<MinHeapEntry>,
    capacity: usize,
    min_score: f32,
}

impl BoundedPriorityQueue {
    /// Create a new bounded queue with specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            // Bounded up-front reservation; grows on demand (TD-SECSCAN-1 §5).
            heap: BinaryHeap::with_capacity(crate::top_k_limit::bounded_prealloc(capacity)),
            capacity,
            min_score: f32::NEG_INFINITY,
        }
    }

    /// Try to insert a new record
    /// Returns true if inserted, false if rejected due to low score
    pub fn try_insert(&mut self, record: OptimizedSearchRecord) -> bool {
        // If not at capacity, always insert
        if self.heap.len() < self.capacity {
            self.heap.push(MinHeapEntry { record });
            self.update_min_score();
            return true;
        }

        // At capacity - only insert if better than current minimum
        if record.score > self.min_score {
            // Remove worst (top of min-heap) and insert new
            self.heap.pop();
            self.heap.push(MinHeapEntry { record });
            self.update_min_score();
            true
        } else {
            false
        }
    }

    /// Get the current minimum score threshold
    /// Any new result must beat this score to be considered
    pub fn min_score_threshold(&self) -> f32 {
        if self.heap.len() < self.capacity {
            f32::NEG_INFINITY // Still accepting any score
        } else {
            self.min_score
        }
    }

    /// Check if we have enough results to potentially terminate early
    pub fn is_full(&self) -> bool {
        self.heap.len() >= self.capacity
    }

    /// Get the number of results currently in the queue
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Check if the queue is empty
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Convert to sorted vector (highest score first)
    pub fn into_sorted_vec(self) -> Vec<OptimizedSearchRecord> {
        let mut results: Vec<_> = self.heap.into_iter().map(|entry| entry.record).collect();

        // Sort by score descending (highest first)
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        results
    }

    /// Update the minimum score from the heap
    fn update_min_score(&mut self) {
        self.min_score = self
            .heap
            .peek()
            .map_or(f32::NEG_INFINITY, |e| e.record.score);
    }

    /// Check if a score has potential to enter the queue
    pub fn would_accept(&self, score: f32) -> bool {
        self.heap.len() < self.capacity || score > self.min_score
    }

    /// Merge another priority queue into this one.
    /// Takes the top-k across both queues. Used for combining thread-local
    /// results from parallel morsel-driven search.
    pub fn merge(&mut self, other: BoundedPriorityQueue) {
        for entry in other.heap {
            self.try_insert(entry.record);
        }
    }
}

/// Generic bounded top-k selection over `Ord` entries, keeping the k greatest.
///
/// Unlike [`BoundedPriorityQueue`] (typed to [`OptimizedSearchRecord`] and
/// score-ordered), callers define "better" through the entry's own [`Ord`]:
/// define it so that "better" ranks [`Ordering::Greater`]. For f32 keys use
/// `partial_cmp(..).unwrap_or(Ordering::Equal)` and carry an explicit
/// tie-break key (e.g. an insertion sequence number) in the entry — that
/// reproduces stable `sort_by` + `truncate(k)` semantics byte-identically
/// while costing O(n log k) instead of O(n log n) full sorts.
///
/// NaN keys fall through to `Equal` and therefore never rank better than a
/// real value already kept; entries equal to the current worst-kept entry are
/// rejected at capacity (first-inserted wins, matching stable sort).
pub struct TopKHeap<E: Ord> {
    capacity: usize,
    inner: BinaryHeap<std::cmp::Reverse<E>>,
}

impl<E: Ord> TopKHeap<E> {
    /// Create a heap keeping the `k` greatest entries. `k == 0` is clamped to
    /// `1` (a zero-capacity heap would reject everything, which callers almost
    /// never want; `k` comes from `top_k` query parameters).
    ///
    /// Because `k` is client-controlled, only the logical capacity follows it:
    /// the eager reservation is bounded by
    /// [`crate::top_k_limit::bounded_prealloc`] and the heap grows on demand,
    /// so `k = u32::MAX` no longer reserves hundreds of GB up front
    /// (TD-SECSCAN-1 §5).
    pub fn with_capacity(k: usize) -> Self {
        Self {
            capacity: k.max(1),
            inner: BinaryHeap::with_capacity(crate::top_k_limit::bounded_prealloc(k)),
        }
    }

    /// The maximum number of entries kept.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Try to push an entry. Returns `false` if it was rejected (not top-k).
    pub fn try_push(&mut self, entry: E) -> bool {
        if self.inner.len() < self.capacity {
            self.inner.push(std::cmp::Reverse(entry));
            return true;
        }
        // At capacity: the heap top (max of Reverse) is the worst kept entry.
        // Replace it only if the newcomer is strictly greater.
        if let Some(std::cmp::Reverse(worst)) = self.inner.peek()
            && entry.cmp(worst) == Ordering::Greater
        {
            self.inner.pop();
            self.inner.push(std::cmp::Reverse(entry));
            return true;
        }
        false
    }

    /// Whether an entry would be accepted right now.
    pub fn would_accept(&self, entry: &E) -> bool {
        if self.inner.len() < self.capacity {
            return true;
        }
        self.inner
            .peek()
            .is_none_or(|std::cmp::Reverse(worst)| entry.cmp(worst) == Ordering::Greater)
    }

    /// The current worst-kept entry (the admission threshold once full).
    pub fn threshold(&self) -> Option<&E> {
        self.inner.peek().map(|std::cmp::Reverse(e)| e)
    }

    /// Number of entries currently kept.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether no entries are kept.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Merge another heap into this one, keeping the top-k across both.
    /// Used for combining per-shard / per-thread-local top-k results.
    pub fn merge(&mut self, other: TopKHeap<E>) {
        for std::cmp::Reverse(entry) in other.inner {
            self.try_push(entry);
        }
    }

    /// Consume and return the kept entries sorted ascending (worst first).
    pub fn into_sorted_asc(self) -> Vec<E> {
        let mut entries: Vec<E> = self
            .inner
            .into_iter()
            .map(|std::cmp::Reverse(e)| e)
            .collect();
        entries.sort();
        entries
    }

    /// Consume and return the kept entries sorted descending (best first).
    pub fn into_sorted_desc(self) -> Vec<E> {
        let mut entries: Vec<E> = self
            .inner
            .into_iter()
            .map(|std::cmp::Reverse(e)| e)
            .collect();
        entries.sort_by(|a, b| b.cmp(a));
        entries
    }
}

#[cfg(test)]
mod topk_heap_tests {
    use super::*;

    /// Tiny deterministic xorshift64* PRNG — keeps the differential test free
    /// of a `rand` dependency while being reproducible everywhere.
    struct Lcg(u64);

    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }

        fn next_f32(&mut self) -> f32 {
            // Values in [0, 1000) with one decimal of quantization so ties
            // occur frequently — that is what stresses the tie-break path.
            (self.next_u64() % 10_000) as f32 / 10.0
        }
    }

    /// Entry shaped like a ranked candidate: total order over
    /// (rank_key, seq) with an explicit stable tie-break on insertion order.
    #[derive(Debug, PartialEq, Clone)]
    struct Cand {
        rank: f32,
        seq: u64,
        payload: u32,
    }

    impl Eq for Cand {}

    impl Ord for Cand {
        fn cmp(&self, other: &Self) -> Ordering {
            // "Better" = Greater = smaller distance first, then earlier insertion.
            other
                .rank
                .partial_cmp(&self.rank)
                .unwrap_or(Ordering::Equal)
                .then_with(|| other.seq.cmp(&self.seq))
        }
    }

    impl PartialOrd for Cand {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    #[test]
    fn topk_heap_keeps_k_greatest_and_evicts_worst() {
        let mut heap = TopKHeap::with_capacity(3);
        assert!(heap.try_push(Cand {
            rank: 5.0,
            seq: 0,
            payload: 0
        }));
        assert!(heap.try_push(Cand {
            rank: 1.0,
            seq: 1,
            payload: 1
        }));
        assert!(heap.try_push(Cand {
            rank: 3.0,
            seq: 2,
            payload: 2
        }));
        assert!(!heap.is_empty());
        assert_eq!(heap.len(), 3);

        // Worse than the worst kept (rank 5.0) — rejected.
        assert!(!heap.try_push(Cand {
            rank: 6.0,
            seq: 3,
            payload: 3
        }));
        // Better than the worst — replaces it.
        assert!(heap.try_push(Cand {
            rank: 2.0,
            seq: 4,
            payload: 4
        }));

        let kept = heap.into_sorted_desc();
        let ranks: Vec<f32> = kept.iter().map(|c| c.rank).collect();
        assert_eq!(ranks, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn topk_heap_zero_capacity_clamps_to_one() {
        let mut heap = TopKHeap::with_capacity(0);
        assert_eq!(heap.capacity(), 1);
        assert!(heap.try_push(Cand {
            rank: 9.0,
            seq: 0,
            payload: 0
        }));
        // Worse (larger distance) than the kept entry — rejected.
        assert!(!heap.try_push(Cand {
            rank: 99.0,
            seq: 1,
            payload: 1
        }));
        // Better — replaces it.
        assert!(heap.try_push(Cand {
            rank: 1.0,
            seq: 2,
            payload: 2
        }));
        assert_eq!(heap.len(), 1);
    }

    /// Differential test: the heap must reproduce stable `sort_by` +
    /// `truncate(k)` exactly — same kept set, same order — including on the
    /// heavy-tie workload (quantized ranks) where the seq tie-break decides.
    #[test]
    fn topk_heap_matches_stable_sort_truncate() {
        for k in [1usize, 7, 64, 500] {
            let mut rng = Lcg(0x9E3779B97F4A7C15 ^ k as u64);
            let mut heap = TopKHeap::with_capacity(k);
            let mut reference: Vec<Cand> = Vec::with_capacity(10_000);
            for seq in 0..10_000u64 {
                let cand = Cand {
                    rank: rng.next_f32(),
                    seq,
                    payload: seq as u32,
                };
                heap.try_push(cand.clone());
                reference.push(cand);
            }
            // Stable sort by the same "better ranks Greater" order, then take k.
            reference.sort_by(|a, b| b.cmp(a));
            reference.truncate(k);

            let got = heap.into_sorted_desc();
            assert_eq!(
                got, reference,
                "heap output diverged from stable sort+truncate at k={k}"
            );
        }
    }

    #[test]
    fn topk_heap_merge_keeps_topk_across_both() {
        let mut a = TopKHeap::with_capacity(3);
        let mut b = TopKHeap::with_capacity(3);
        for (rank, seq) in [(1.0, 0), (5.0, 1), (3.0, 2)] {
            a.try_push(Cand {
                rank,
                seq,
                payload: seq as u32,
            });
        }
        for (rank, seq) in [(2.0, 3), (4.0, 4), (6.0, 5)] {
            b.try_push(Cand {
                rank,
                seq,
                payload: seq as u32,
            });
        }
        a.merge(b);
        let ranks: Vec<f32> = a.into_sorted_desc().iter().map(|c| c.rank).collect();
        assert_eq!(ranks, vec![1.0, 2.0, 3.0]);
    }

    /// NaN rank keys compare `Equal` to everything, so they never displace a
    /// kept entry — deterministic, no panic, and documented as never-better.
    #[test]
    fn topk_heap_nan_keys_are_never_better() {
        let mut heap = TopKHeap::with_capacity(2);
        assert!(heap.try_push(Cand {
            rank: f32::NAN,
            seq: 0,
            payload: 0
        }));
        assert!(heap.try_push(Cand {
            rank: f32::NAN,
            seq: 1,
            payload: 1
        }));
        // A real value outranks Equal-to-everything NaN? No: NaN == NaN by our
        // Ord (Equal), so a real value must be strictly comparable — it is,
        // and real ranks compare Greater/Lesser against NaN via Equal only on
        // the NaN side. `partial_cmp(real, NaN)` is None → Equal → seq decides.
        assert!(!heap.try_push(Cand {
            rank: 1.0,
            seq: 2,
            payload: 2
        }));
        assert_eq!(heap.len(), 2);
        let _ = heap.into_sorted_asc(); // no panic
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn create_test_record(id: &str, score: f32) -> OptimizedSearchRecord {
        OptimizedSearchRecord {
            id: id.to_string(),
            vector_id: Some(id.to_string()),
            score,
            similarity: Some(score),
            vector: Some(Arc::new(vec![1.0, 2.0, 3.0])),
            metadata: Default::default(),
            debug_info: None,
            version: None,
            timestamp: None,
            updated_at: None,
            expires_at: None,
            source: None,
            expanded_context: vec![],
            semantic_similarity: None,
            quantization_info: None,
            engine_stats: None,
            index_path: None,
            ..Default::default()
        }
    }

    #[test]
    fn test_bounded_queue_basic() {
        let mut queue = BoundedPriorityQueue::new(3);

        // Insert first 3 records
        assert!(queue.try_insert(create_test_record("a", 0.5)));
        assert!(queue.try_insert(create_test_record("b", 0.6)));
        assert!(queue.try_insert(create_test_record("c", 0.8)));

        assert_eq!(queue.len(), 3);
        assert!(queue.is_full());
        assert_eq!(queue.min_score_threshold(), 0.5);

        // Try to insert a worse record - should be rejected
        assert!(!queue.try_insert(create_test_record("d", 0.4)));
        assert_eq!(queue.len(), 3);

        // Insert a better record - should replace worst
        assert!(queue.try_insert(create_test_record("e", 0.75)));
        assert_eq!(queue.len(), 3);

        // Final results should be [0.8, 0.75, 0.6]
        let results = queue.into_sorted_vec();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].score, 0.8);
        assert_eq!(results[1].score, 0.75);
        assert_eq!(results[2].score, 0.6);
    }

    #[test]
    fn test_merge_queues() {
        let mut queue_a = BoundedPriorityQueue::new(3);
        queue_a.try_insert(create_test_record("a1", 0.9));
        queue_a.try_insert(create_test_record("a2", 0.7));
        queue_a.try_insert(create_test_record("a3", 0.5));

        let mut queue_b = BoundedPriorityQueue::new(3);
        queue_b.try_insert(create_test_record("b1", 0.85));
        queue_b.try_insert(create_test_record("b2", 0.6));
        queue_b.try_insert(create_test_record("b3", 0.4));

        // Merge b into a (capacity 3), should keep top-3 across both
        queue_a.merge(queue_b);
        assert_eq!(queue_a.len(), 3);

        let results = queue_a.into_sorted_vec();
        assert_eq!(results[0].score, 0.9);
        assert_eq!(results[1].score, 0.85);
        assert_eq!(results[2].score, 0.7);
    }

    #[test]
    fn test_would_accept() {
        let mut queue = BoundedPriorityQueue::new(2);

        // Empty queue accepts anything
        assert!(queue.would_accept(0.0));

        queue.try_insert(create_test_record("a", 0.5));
        // Still not full, accepts anything
        assert!(queue.would_accept(0.0));

        queue.try_insert(create_test_record("b", 0.7));
        // Now full, only accepts > 0.5
        assert!(!queue.would_accept(0.4));
        assert!(queue.would_accept(0.6));
    }

    /// TD-SECSCAN-1 §5: `k` arrives from a client `top_k`. The logical
    /// capacity must honour it, but the eager pre-allocation must not — a
    /// `top_k = u32::MAX` request otherwise reserves hundreds of GB before a
    /// single candidate is scored.
    #[test]
    fn topk_heap_preallocation_is_bounded_but_capacity_is_not() {
        let huge = 1_000_000usize;
        let mut heap: TopKHeap<u64> = TopKHeap::with_capacity(huge);
        assert_eq!(heap.capacity(), huge, "logical top-k must be preserved");
        assert!(
            heap.inner.capacity() <= crate::top_k_limit::DEFAULT_MAX_TOP_K,
            "pre-allocation must be clamped, got {}",
            heap.inner.capacity()
        );
        // Grows on demand past the pre-allocation: nothing is truncated.
        for i in 0..20_000u64 {
            assert!(heap.try_push(i));
        }
        assert_eq!(heap.len(), 20_000);
    }

    #[test]
    fn bounded_priority_queue_preallocation_is_bounded() {
        let queue = BoundedPriorityQueue::new(1_000_000);
        assert!(
            queue.heap.capacity() <= crate::top_k_limit::DEFAULT_MAX_TOP_K,
            "pre-allocation must be clamped, got {}",
            queue.heap.capacity()
        );
    }
}
