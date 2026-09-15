//! Turning logical byte needs into physical requests.
//!
//! Object stores bill **per request** — Azure Hot, S3 Standard and GCS Standard
//! each charge one transaction per ranged GET regardless of its size, and
//! in-region bytes are free — so the number of physical reads is the billed read
//! cost. Merging nearby logical ranges trades a bounded over-read for a saved
//! round trip, which is a good trade precisely because bytes are not billed.
//!
//! This module is the *plan* half of a plan/execute split: it is pure, sync and
//! exhaustively testable, and it never performs I/O. [`FileSystem::read_ranges`]
//! executes the plan and slices the results back apart.
//!
//! The contract that makes it safe to slot underneath existing callers:
//!
//! * **Output is indexed by INPUT position.** Merging requires sorting
//!   internally, but callers index the returned buffers positionally — the PAX
//!   adapter builds `RecordBatch`es that way — so a sorted-order return would be
//!   a silent wrong-answer bug, not a crash.
//! * **Duplicates and overlaps are never deduplicated.** Two inputs may alias
//!   the same physical bytes; each still gets its own buffer.
//! * **Both bounds are required.** A gap bound alone (all upstream
//!   `object_store` enforces) leaves the over-read unbounded: a hundred
//!   scattered 4 KiB ranges under a 1 MiB gap would materialise ~100 MB.
//! * **An absent policy is the identity plan**, so an un-gated deployment issues
//!   exactly the requests it issues today.

use crate::{FilesystemError, FsResult, RangeCoalescePolicy};
use std::ops::Range;

/// Where one logical range's bytes live inside the physical read that covers it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanSlice {
    /// Index into [`RangePlan::physical`], or `None` for a zero-length range,
    /// which is satisfied without issuing any request at all.
    pub physical: Option<usize>,
    /// Byte offset of this range within that physical read.
    pub offset: usize,
    /// Length in bytes as requested. The physical buffer may be *shorter* when
    /// the backend clamped at EOF, which is why slicing must saturate.
    pub len: usize,
}

/// A set of physical reads plus, for every input range, where to find its bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangePlan {
    /// Physical reads to issue, sorted by start offset.
    ///
    /// Usually disjoint, but two entries **may overlap** when the caller passed
    /// overlapping logical ranges and [`RangeCoalescePolicy::max_merged_bytes`]
    /// refused to merge them. That costs a few bytes fetched twice, which is
    /// strictly better than the alternatives: dropping the size cap would let
    /// peak memory grow without bound, and clipping the second read would leave
    /// a logical range spanning two physical reads with no single buffer to
    /// slice from. Every mapping is always fully contained in exactly one
    /// physical read, so correctness does not depend on disjointness.
    pub physical: Vec<Range<u64>>,
    /// One entry per input range, in **input order**.
    pub mapping: Vec<PlanSlice>,
}

impl RangePlan {
    /// Total bytes the physical reads will transfer (including gap over-read).
    pub fn physical_bytes(&self) -> u64 {
        self.physical.iter().map(|r| r.end - r.start).sum()
    }

    /// Bytes the caller actually asked for.
    pub fn logical_bytes(&self) -> u64 {
        self.mapping.iter().map(|s| s.len as u64).sum()
    }

    /// Bytes fetched but not requested — the price paid for fewer requests.
    pub fn overread_bytes(&self) -> u64 {
        self.physical_bytes().saturating_sub(self.logical_bytes())
    }
}

/// Build the physical read plan for a set of logical ranges.
///
/// `policy == None` yields the identity plan: one physical read per non-empty
/// logical range, in input order, byte-identical to a per-range loop.
///
/// Fails closed on an inverted range. That is a deliberate behaviour change:
/// today `range.end - range.start` underflows, panicking in debug and producing
/// a `~u64::MAX` length in release, so a clean error strictly improves on
/// undefined behaviour.
pub fn coalesce_ranges_with_mapping(
    ranges: &[Range<u64>],
    policy: Option<RangeCoalescePolicy>,
) -> FsResult<RangePlan> {
    for range in ranges {
        if range.end < range.start {
            return Err(FilesystemError::InvalidOperation(format!(
                "inverted byte range {}..{}: end precedes start",
                range.start, range.end
            )));
        }
    }

    let Some(policy) = policy else {
        return Ok(identity_plan(ranges));
    };

    // Sort by start, carrying the input index so the mapping can be rebuilt in
    // input order. Zero-length ranges are excluded from the physical plan
    // entirely — they cost nothing and must not extend a merged span.
    let mut ordered: Vec<(usize, Range<u64>)> = ranges
        .iter()
        .enumerate()
        .filter(|(_, r)| r.end > r.start)
        .map(|(i, r)| (i, r.clone()))
        .collect();
    ordered.sort_by_key(|(_, r)| (r.start, r.end));

    let mut physical: Vec<Range<u64>> = Vec::new();
    // Parallel to `physical`: which input indices each physical read serves.
    let mut members: Vec<Vec<usize>> = Vec::new();

    for (input_idx, range) in ordered {
        let merged = match (physical.last_mut(), members.last_mut()) {
            (Some(last), Some(last_members)) => {
                // `saturating_sub`: overlapping ranges have a zero gap, not a
                // negative one.
                let gap = range.start.saturating_sub(last.end);
                let span = range.end.max(last.end).saturating_sub(last.start);
                if gap <= policy.max_gap_bytes && span <= policy.max_merged_bytes {
                    last.end = last.end.max(range.end);
                    last_members.push(input_idx);
                    true
                } else {
                    false
                }
            }
            _ => false,
        };
        if !merged {
            physical.push(range.clone());
            members.push(vec![input_idx]);
        }
    }

    // Rebuild the mapping in INPUT order.
    let mut mapping = vec![
        PlanSlice {
            physical: None,
            offset: 0,
            len: 0,
        };
        ranges.len()
    ];
    for (physical_idx, member_indices) in members.iter().enumerate() {
        let base = physical[physical_idx].start;
        for &input_idx in member_indices {
            let range = &ranges[input_idx];
            mapping[input_idx] = PlanSlice {
                physical: Some(physical_idx),
                offset: (range.start - base) as usize,
                len: (range.end - range.start) as usize,
            };
        }
    }

    Ok(RangePlan { physical, mapping })
}

/// One physical read per non-empty range, in input order.
fn identity_plan(ranges: &[Range<u64>]) -> RangePlan {
    let mut physical = Vec::with_capacity(ranges.len());
    let mapping = ranges
        .iter()
        .map(|range| {
            let len = (range.end - range.start) as usize;
            if len == 0 {
                // Preserved as a real physical read so the identity plan is
                // byte-identical to today's loop, which does issue it.
                physical.push(range.clone());
                PlanSlice {
                    physical: Some(physical.len() - 1),
                    offset: 0,
                    len: 0,
                }
            } else {
                physical.push(range.clone());
                PlanSlice {
                    physical: Some(physical.len() - 1),
                    offset: 0,
                    len,
                }
            }
        })
        .collect();
    RangePlan { physical, mapping }
}

/// Extract one logical range from the physical buffer that covers it.
///
/// Saturating on **both** ends. Every backend clamps at EOF — local returns a
/// short read, Azure/S3/GCS clamp the upper bound — so a merged buffer is
/// routinely shorter than the span requested. Clamping only the end (what
/// upstream `object_store` does) panics in `Bytes::slice` once `start > end`.
pub fn slice_from_physical(buffer: &[u8], slice: PlanSlice) -> Vec<u8> {
    let lo = slice.offset.min(buffer.len());
    let hi = slice.offset.saturating_add(slice.len).min(buffer.len());
    buffer[lo..hi.max(lo)].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(gap: u64, max: u64) -> Option<RangeCoalescePolicy> {
        Some(RangeCoalescePolicy {
            max_gap_bytes: gap,
            max_merged_bytes: max,
        })
    }

    #[test]
    fn identity_plan_when_policy_absent() {
        let ranges = vec![0..32, 200..232, 64..64];
        let plan = coalesce_ranges_with_mapping(&ranges, None).expect("identity");
        assert_eq!(
            plan.physical, ranges,
            "one physical read per input, in order"
        );
        assert_eq!(plan.overread_bytes(), 0);
    }

    // The lint is right in general and wrong here: an inverted range is exactly
    // the malformed input under test, and today it underflows rather than being
    // rejected.
    #[allow(clippy::reversed_empty_ranges)]
    #[test]
    fn inverted_range_fails_closed() {
        let inverted = 40..10;
        let ranges = std::slice::from_ref(&inverted);
        let err = coalesce_ranges_with_mapping(ranges, policy(64, 4096))
            .expect_err("inverted must be rejected");
        assert!(matches!(err, FilesystemError::InvalidOperation(_)));
        // ...and also when the policy is absent, so the identity path cannot
        // underflow either.
        assert!(coalesce_ranges_with_mapping(ranges, None).is_err());
    }

    #[test]
    fn zero_length_ranges_cost_no_physical_read() {
        let plan =
            coalesce_ranges_with_mapping(&[0..32, 64..64], policy(1024, 4096)).expect("plan");
        assert_eq!(
            plan.physical.len(),
            1,
            "only the non-empty range is fetched"
        );
        assert_eq!(plan.mapping[1].physical, None);
        assert_eq!(plan.mapping[1].len, 0);
    }

    #[test]
    fn mapping_is_input_indexed_under_sorting() {
        let ranges = vec![200..264, 0..64, 100..164];
        let plan = coalesce_ranges_with_mapping(&ranges, policy(4096, 8192)).expect("plan");
        assert_eq!(plan.physical, vec![0..264], "one merged span");
        assert_eq!(
            plan.mapping[0].offset, 200,
            "input 0 is the LAST by address"
        );
        assert_eq!(plan.mapping[1].offset, 0);
        assert_eq!(plan.mapping[2].offset, 100);
    }

    #[test]
    fn size_cap_bounds_the_overread() {
        let plan = coalesce_ranges_with_mapping(&[0..64, 64..128, 128..192], policy(64, 128))
            .expect("plan");
        assert_eq!(plan.physical.len(), 2, "span must never exceed the cap");
        assert!(plan.physical.iter().all(|r| r.end - r.start <= 128));
    }

    /// The three invariants any plan must satisfy, over a deterministic spread of
    /// shapes including overlaps, duplicates, gaps and zero-length ranges.
    #[test]
    fn plan_invariants_hold_across_shapes() {
        let shapes: Vec<Vec<Range<u64>>> = vec![
            vec![0..32, 0..32, 16..48],
            vec![0..16, 1_000..1_016, 2_000..2_016],
            vec![100..200, 0..50, 60..99, 199..201],
            vec![0..0, 5..5, 10..20],
            vec![0..4096],
        ];
        for (gap, max) in [(0_u64, 64_u64), (16, 4096), (1 << 20, 1 << 24)] {
            for ranges in &shapes {
                let plan = coalesce_ranges_with_mapping(ranges, policy(gap, max)).expect("plan");

                // Physical must cover the UNION of the logical ranges — not
                // their sum. Duplicate and overlapping inputs legitimately
                // share the same physical bytes, so `Σ logical` can exceed
                // `Σ physical` without any range going unserved.
                let mut union: Vec<Range<u64>> =
                    ranges.iter().filter(|r| r.end > r.start).cloned().collect();
                union.sort_by_key(|r| (r.start, r.end));
                let mut union_bytes = 0_u64;
                let mut cursor = 0_u64;
                for r in union {
                    let start = r.start.max(cursor);
                    if r.end > start {
                        union_bytes += r.end - start;
                        cursor = r.end;
                    }
                }
                assert!(
                    plan.physical_bytes() >= union_bytes,
                    "physical must cover the logical union: {ranges:?} @ gap={gap} max={max}"
                );
                for window in plan.physical.windows(2) {
                    assert!(
                        window[0].start <= window[1].start,
                        "physical reads must be sorted by start: {:?}",
                        plan.physical
                    );
                }
                // The cap bounds MERGING, not individual reads: a lone logical
                // range bigger than the cap must still be fetched whole.
                for (idx, phys) in plan.physical.iter().enumerate() {
                    if phys.end - phys.start > max {
                        let served = plan
                            .mapping
                            .iter()
                            .filter(|m| m.physical == Some(idx))
                            .count();
                        assert_eq!(
                            served, 1,
                            "a physical read may exceed max_merged_bytes only when it \
                             serves exactly one logical range: {phys:?} serves {served}"
                        );
                    }
                }
                for (i, slice) in plan.mapping.iter().enumerate() {
                    let Some(idx) = slice.physical else {
                        assert_eq!(ranges[i].end - ranges[i].start, 0);
                        continue;
                    };
                    let phys = &plan.physical[idx];
                    let start = phys.start + slice.offset as u64;
                    assert!(
                        start >= phys.start && start + slice.len as u64 <= phys.end,
                        "mapping {i} must land inside its physical read"
                    );
                    assert_eq!(start, ranges[i].start, "mapping must address its own range");
                }
            }
        }
    }

    #[test]
    fn slicing_saturates_when_the_backend_clamped_at_eof() {
        let short = vec![1_u8, 2, 3];
        // Asks for 10 bytes at offset 1 from a 3-byte buffer.
        let got = slice_from_physical(
            &short,
            PlanSlice {
                physical: Some(0),
                offset: 1,
                len: 10,
            },
        );
        assert_eq!(got, vec![2, 3]);
        // Entirely past the end of the returned buffer.
        let past = slice_from_physical(
            &short,
            PlanSlice {
                physical: Some(0),
                offset: 99,
                len: 4,
            },
        );
        assert!(past.is_empty(), "must not panic, must not wrap");
    }
}
