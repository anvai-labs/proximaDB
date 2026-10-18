// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! The ONE server-side `top_k` limit shared by every search surface
//! (TD-SECSCAN-1 §5).
//!
//! Before this module the cap lived inline in the REST v2 search handler only,
//! while Arrow Flight and gRPC passed a client `top_k` straight into an eager
//! `BinaryHeap::with_capacity(top_k)` — `top_k = u32::MAX` asked for a
//! multi-hundred-GB allocation and aborted the process. Every surface now
//! validates through [`check_top_k`], and the shared top-k sinks
//! ([`crate::bounded_queue::TopKHeap`], [`crate::bounded_queue::BoundedPriorityQueue`])
//! clamp their eager pre-allocation with [`bounded_prealloc`] as a second,
//! surface-independent layer.

use std::fmt;

/// Env override for the cap (documented in `docs/12-design/ENV_GATE_REGISTRY.adoc`).
pub const MAX_TOP_K_ENV: &str = "PROXIMADB_SEARCH_MAX_TOP_K";

/// Default cap: generous for production ANN workloads, small enough that one
/// request cannot exhaust memory.
pub const DEFAULT_MAX_TOP_K: usize = 10_000;

/// The effective server cap: `PROXIMADB_SEARCH_MAX_TOP_K` when set to a
/// positive integer, otherwise [`DEFAULT_MAX_TOP_K`]. A malformed or zero
/// override falls back to the default rather than disabling the cap.
pub fn max_top_k() -> usize {
    parse_max_top_k(std::env::var(MAX_TOP_K_ENV).ok().as_deref())
}

fn parse_max_top_k(raw: Option<&str>) -> usize {
    raw.and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&k| k > 0)
        .unwrap_or(DEFAULT_MAX_TOP_K)
}

/// A request asked for more results than the server cap allows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopKLimitExceeded {
    /// The requested `top_k`.
    pub requested: u64,
    /// The cap in force.
    pub max: usize,
}

impl fmt::Display for TopKLimitExceeded {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "top_k={} exceeds server cap {} (set {} to override)",
            self.requested, self.max, MAX_TOP_K_ENV
        )
    }
}

impl std::error::Error for TopKLimitExceeded {}

/// Reject a `top_k` above the server cap. Every search surface (REST, gRPC,
/// Arrow Flight) calls this at its boundary so clients get an explicit
/// invalid-argument error instead of a truncated result or an OOM.
///
/// `check_top_k` permits `0`: it bounds only the upper end, so each surface
/// decides for itself whether `top_k == 0` is an error or an empty result.
pub fn check_top_k(top_k: u64) -> Result<(), TopKLimitExceeded> {
    check_top_k_against(top_k, max_top_k())
}

fn check_top_k_against(requested: u64, max: usize) -> Result<(), TopKLimitExceeded> {
    // u64 comparison: a `usize` cap always fits in u64 on supported targets.
    if requested > max as u64 {
        Err(TopKLimitExceeded { requested, max })
    } else {
        Ok(())
    }
}

/// Eager pre-allocation for a top-k buffer of logical capacity `k`: at most
/// [`DEFAULT_MAX_TOP_K`]. The buffer still grows on demand up to `k`, so the
/// result is never truncated — only the up-front reservation is bounded, and
/// it no longer scales with an untrusted request field.
pub fn bounded_prealloc(k: usize) -> usize {
    k.min(DEFAULT_MAX_TOP_K)
}

/// Narrow a `top_k` to the `u16` carried by `SearchParams::top_k`,
/// SATURATING at `u16::MAX` (TD-SECSCAN-1 §5). A bare `k as u16` wrapped
/// (65,536 → 0, 70,000 → 4,464). Use this only where the field is a planning
/// hint or `k` is already bounded by the server cap; where the `u16` is a hard
/// result limit and `k` may exceed it, reject or split instead.
pub fn u16_top_k_hint(k: usize) -> u16 {
    u16::try_from(k).unwrap_or(u16::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_and_override_parsing() {
        assert_eq!(parse_max_top_k(None), DEFAULT_MAX_TOP_K);
        assert_eq!(parse_max_top_k(Some("50000")), 50_000);
        assert_eq!(parse_max_top_k(Some(" 7 ")), 7);
        // Malformed / zero never disables the cap.
        assert_eq!(parse_max_top_k(Some("lots")), DEFAULT_MAX_TOP_K);
        assert_eq!(parse_max_top_k(Some("0")), DEFAULT_MAX_TOP_K);
        assert_eq!(parse_max_top_k(Some("-1")), DEFAULT_MAX_TOP_K);
    }

    #[test]
    fn boundary_is_inclusive() {
        assert!(check_top_k_against(10_000, 10_000).is_ok());
        let err = check_top_k_against(10_001, 10_000).unwrap_err();
        assert_eq!(err.requested, 10_001);
        assert_eq!(
            err.to_string(),
            "top_k=10001 exceeds server cap 10000 (set PROXIMADB_SEARCH_MAX_TOP_K to override)"
        );
        assert!(check_top_k_against(u64::from(u32::MAX), 10_000).is_err());
    }

    #[test]
    fn prealloc_is_bounded() {
        assert_eq!(bounded_prealloc(5), 5);
        assert_eq!(bounded_prealloc(usize::MAX), DEFAULT_MAX_TOP_K);
    }

    /// TD-SECSCAN-1 §5 residual: the bare `k as u16` narrowing wrapped
    /// (65,536 → 0, 70,000 → 4,464). The hint saturates instead.
    #[test]
    fn u16_top_k_hint_saturates_at_the_boundary() {
        assert_eq!(u16_top_k_hint(0), 0);
        assert_eq!(u16_top_k_hint(10), 10);
        assert_eq!(u16_top_k_hint(65_535), u16::MAX);
        assert_eq!(u16_top_k_hint(65_536), u16::MAX);
        assert_eq!(u16_top_k_hint(70_000), u16::MAX);
        assert_eq!(u16_top_k_hint(usize::MAX), u16::MAX);
    }
}
