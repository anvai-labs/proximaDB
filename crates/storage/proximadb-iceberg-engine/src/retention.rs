//! # retention — the one generation window for Iceberg metadata and data files
//!
//! TD-GCRETAIN-1 (F8) decided the MATERIALIZE retention rule: keep the files
//! of the last N committed metadata versions — `PROXIMADB_MATERIALIZE_GC_RETAINED_GENERATIONS`
//! TOTAL generations including the committed one, default 3, floor 2 — and
//! do it **clock-free** (generation order is the create-only metadata CAS
//! order, never a wall clock; time-based grace was declined).
//!
//! ADR-098 D4.2 / TD-SPINE-3 make metadata follow the same window: the
//! retained `v*.metadata.json` versions, their manifest lists and manifests
//! are exactly those of the retained generations, and the `metadata-log` a
//! new version carries is bounded to the same window (an older entry would
//! name a snapshot whose data files the generation GC already deleted). The
//! window lives here, in the engine that owns metadata versions, so the root
//! crate's data-file GC and this crate's metadata GC + log bound cannot drift.

use proximadb_storage_common::gate_value::default_on_gate_value;

/// The env gate carrying the window size (registered in `ENV_GATE_REGISTRY.adoc`).
pub const RETAINED_GENERATIONS_ENV: &str = "PROXIMADB_MATERIALIZE_GC_RETAINED_GENERATIONS";

/// Default retained generations (TD-GCRETAIN-1).
pub const DEFAULT_RETAINED_GENERATIONS: u64 = 3;

/// Structural floor: the committed version and its parent.
pub const MIN_RETAINED_GENERATIONS: u64 = 2;

/// Iceberg table property that caps the number of previous metadata files the
/// `metadata-log` keeps. Here it may only LOWER the generation window, never
/// widen it (a wider log would name collected snapshots).
pub const PREVIOUS_VERSIONS_MAX_PROPERTY: &str = "write.metadata.previous-versions-max";

/// Kill-switch for the post-MATERIALIZE metadata GC (default: GC runs).
pub const METADATA_GC_DISABLE_ENV: &str = "PROXIMADB_ICEBERG_METADATA_GC_DISABLE";

/// Whether [`METADATA_GC_DISABLE_ENV`] disables the metadata GC
/// (`1|true|on|yes`, case-insensitive). The `metadata-log` bound is not
/// affected: it is a property of every new document.
pub fn metadata_gc_disabled() -> bool {
    disable_flag(std::env::var(METADATA_GC_DISABLE_ENV).ok().as_deref())
}

/// Pure core of the presence-style kill-switch spelling.
pub fn disable_flag(raw: Option<&str>) -> bool {
    raw.is_some_and(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        )
    })
}

/// How many committed Iceberg metadata generations MATERIALIZE keeps, TOTAL
/// including the committed one (read from [`RETAINED_GENERATIONS_ENV`]).
pub fn gc_retained_generations() -> u64 {
    retained_generations_from(std::env::var(RETAINED_GENERATIONS_ENV).ok().as_deref())
}

/// Pure core of the gate: unset/typo keeps the default 3; OFF/0/1 clamp to
/// the structural floor 2 (V and V−1) — the pre-F8 behavior, so OFF is a
/// faithful kill-switch to the narrow window, never "GC everything".
pub fn retained_generations_from(raw: Option<&str>) -> u64 {
    default_on_gate_value(raw, DEFAULT_RETAINED_GENERATIONS)
        .unwrap_or(MIN_RETAINED_GENERATIONS)
        .max(MIN_RETAINED_GENERATIONS)
}

/// The OLDER metadata versions whose referenced files GC must keep,
/// newest-first: `previous` down to and including `previous - (retained - 2)`,
/// saturating at 0. `retained` counts TOTAL generations including the
/// committed V (= `previous + 1`), so retained=2 ⇒ `[previous]`, retained=3 ⇒
/// `[previous, previous−1]`; version 0 is included (the first commit is v0).
pub fn retained_versions(previous: u64, retained: u64) -> Vec<u64> {
    let floor = previous.saturating_sub(retained.saturating_sub(MIN_RETAINED_GENERATIONS));
    (floor..=previous).rev().collect()
}

/// The oldest metadata version inside the window ending at `committed`:
/// versions `>= window_floor(committed, retained)` are retained. Consistent
/// with [`retained_versions`]: for `committed = previous + 1` the window is
/// `committed` plus `retained_versions(previous, retained)`.
pub fn window_floor(committed: u64, retained: u64) -> u64 {
    committed.saturating_sub(retained.max(1).saturating_sub(1))
}

/// How many previous-metadata entries a new version's `metadata-log` keeps:
/// the window minus the version itself (`retained - 1`), lowered — never
/// raised — by a `write.metadata.previous-versions-max` table property.
pub fn metadata_log_max_entries(retained: u64, property: Option<u64>) -> usize {
    let window = retained.max(MIN_RETAINED_GENERATIONS).saturating_sub(1);
    let bounded = property.map_or(window, |p| p.min(window));
    usize::try_from(bounded).unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_generations_spellings_clamp_to_the_floor() {
        assert_eq!(retained_generations_from(None), 3);
        assert_eq!(retained_generations_from(Some("garbage")), 3);
        assert_eq!(retained_generations_from(Some("5")), 5);
        assert_eq!(retained_generations_from(Some("1")), 2);
        assert_eq!(retained_generations_from(Some("0")), 2);
        assert_eq!(retained_generations_from(Some("off")), 2);
    }

    #[test]
    fn the_kill_switch_only_trips_on_truthy_spellings() {
        assert!(!disable_flag(None));
        assert!(!disable_flag(Some("0")));
        assert!(!disable_flag(Some("off")));
        assert!(disable_flag(Some("1")));
        assert!(disable_flag(Some(" TRUE ")));
    }

    #[test]
    fn window_floor_matches_retained_versions() {
        for retained in 2..6 {
            for previous in 0..8u64 {
                let committed = previous + 1;
                let floor = window_floor(committed, retained);
                let mut window: Vec<u64> = (floor..=committed).rev().collect();
                window.remove(0);
                assert_eq!(
                    window,
                    retained_versions(previous, retained),
                    "previous {previous} retained {retained}"
                );
            }
        }
        // The first commit's window is just itself.
        assert_eq!(window_floor(0, 3), 0);
    }

    #[test]
    fn metadata_log_bound_follows_the_window_and_the_property_only_lowers_it() {
        assert_eq!(metadata_log_max_entries(3, None), 2);
        assert_eq!(metadata_log_max_entries(2, None), 1);
        assert_eq!(metadata_log_max_entries(3, Some(100)), 2, "never widened");
        assert_eq!(metadata_log_max_entries(3, Some(1)), 1);
        assert_eq!(metadata_log_max_entries(3, Some(0)), 0);
    }
}
