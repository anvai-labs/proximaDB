//! TD-RDSTRAT-9: the policy-driven probe-band controller — the single place
//! the coarse-probe fan-out formula lives.
//!
//! The probe count is the recall↔GETs dial of the IVF-family read path: rank
//! cells in RAM, then read the chosen cells' extents as coalesced ranges.
//! Before this module the fan-out was one hardcoded power law
//! (`ceil(sqrt(k_c) × multiplier)`, process-global, no collection identity)
//! and measured GET fan-out grew ×3.6 per ×10 corpus at a fixed recall gate —
//! worse than √N. Now the formula is a POLICY resolved through the same
//! chain as the cache and compaction policies:
//!
//! ```text
//! collection property (probe.*) > operator tag (probe.algorithm:x)
//!   > server TOML [storage.sst_config.coarse_probe] > code default
//! ```
//!
//! The algorithms are a CLOSED ENUM of named pure functions — never a
//! formula interpreter: an expression string in config would be
//! unvalidatable at SET time and untestable (and a review/security surface).
//! Adding "some complex formula" means adding one variant + one pure fn +
//! one unit test in this file.
//!
//! The discrete band ladder is DERIVED from the rowgroup geometry (the
//! per-backend IOPS budget target — 4 MiB cloud, 8 MiB S3, 1 MiB local),
//! never configured and never duplicated at call sites: band edges live in
//! [`band_ladder`] and nowhere else.
//!
//! Default policy = `Sqrt` with the shipped multiplier/min/max: byte-for-byte
//! the previous behavior. A different default requires the bed experiment
//! (recall-gated) + a ledger claim — the storage-format migration doctrine.

use std::collections::HashMap;

/// The named probe-band algorithms. Each is one pure fn below; the policy
/// selects by name (`sqrt|cbrt|fourth_root|log|fixed|byte_budget`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ProbeAlgorithm {
    /// `ceil(cells^0.5 × multiplier)` — the historical default.
    #[default]
    Sqrt,
    /// `ceil(cells^0.333 × multiplier)` — the sub-linear candidate for
    /// large corpora.
    CubeRoot,
    /// `ceil(cells^0.25 × multiplier)` — the aggressive end of the root
    /// ladder.
    FourthRoot,
    /// `ceil(log2(cells) × multiplier + log2(base_bytes))` — logarithmic
    /// growth; the base budget scales the ladder's step.
    Log,
    /// `min_cells` passthrough (clamped) — exact control arm.
    Fixed,
    /// Probe a fixed BYTE budget per query, sub-linear in corpus:
    /// `budget = base × (rows / 1M)^0.25`, snapped to the band ladder and
    /// capped by `cap_bytes`; cells = budget / avg_cell_bytes. The
    /// cost-matched form: the plan maps directly to GET bytes.
    ByteBudget,
}

impl ProbeAlgorithm {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "sqrt" => Some(Self::Sqrt),
            "cbrt" | "cube_root" => Some(Self::CubeRoot),
            "fourth_root" | "4th_root" => Some(Self::FourthRoot),
            "log" | "logarithmic" => Some(Self::Log),
            "fixed" => Some(Self::Fixed),
            "byte_budget" | "budget" => Some(Self::ByteBudget),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sqrt => "sqrt",
            Self::CubeRoot => "cbrt",
            Self::FourthRoot => "fourth_root",
            Self::Log => "log",
            Self::Fixed => "fixed",
            Self::ByteBudget => "byte_budget",
        }
    }
}

/// A resolved probe policy (all layers merged). Numeric fields reuse the
/// `CoarseProbeConfig` names so the TOML surface stays familiar.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbePolicy {
    pub algorithm: ProbeAlgorithm,
    /// Scale factor for the root/log forms (TOML `nprobe_multiplier`,
    /// default 2.0 — recall-ratcheted).
    pub multiplier: f64,
    /// Floor on the probe count (TOML `nprobe_min`, default 3).
    pub min_cells: usize,
    /// Ceiling on the probe count (TOML `nprobe_max`; 0 = unlimited).
    pub max_cells: usize,
    /// Base per-query probe byte budget for `ByteBudget`/`Log` (bytes).
    pub base_bytes: u64,
    /// Hard per-query probe byte cap (bytes; 0 = none).
    pub cap_bytes: u64,
}

impl Default for ProbePolicy {
    fn default() -> Self {
        Self {
            algorithm: ProbeAlgorithm::Sqrt,
            multiplier: 2.0,
            min_cells: 3,
            max_cells: 0,
            base_bytes: 4 * 1024 * 1024,
            cap_bytes: 32 * 1024 * 1024,
        }
    }
}

/// The server-TOML layer of the chain (mirrors `ServerCachePolicy`'s role):
/// raw config fields, resolved against the code default per key.
#[derive(Debug, Clone, Default)]
pub struct ServerProbePolicy {
    pub algorithm: Option<String>,
    pub multiplier: Option<f64>,
    pub min_cells: Option<usize>,
    pub max_cells: Option<usize>,
    pub base_bytes: Option<u64>,
    pub cap_bytes: Option<u64>,
}

/// The result of one probe-plan computation: the cell count AND the byte
/// band it snapped to — the plan the caller reads, meters, and bills.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbePlan {
    pub cells: usize,
    /// Planned probe bytes (cells × avg_cell_bytes, band-snapped).
    pub probe_bytes: u64,
    /// The band rung the plan snapped to (0 when not band-constrained).
    pub band: u64,
}

/// The discrete band ladder, DERIVED from the rowgroup geometry:
/// rungs `[rg/4, rg/2, rg, 2·rg, 4·rg]` filtered to sane bounds (sub-512 KiB
/// probes are pointless; unbounded above is fine — `cap_bytes` bounds).
/// Band edges live HERE and nowhere else.
pub fn band_ladder(rowgroup_bytes: u64) -> Vec<u64> {
    let rg = rowgroup_bytes.max(1);
    [
        rg / 4,
        rg / 2,
        rg,
        rg.saturating_mul(2),
        rg.saturating_mul(4),
    ]
    .into_iter()
    .filter(|rung| *rung >= 512 * 1024)
    .collect()
}

/// Snap `bytes` to the nearest ladder rung (ties keep the LOWER rung —
/// fewer probes on ambiguity), bounded by `cap_bytes` (a cap above the top
/// rung still bites: the returned plan never exceeds it). When the ladder
/// is empty (a sub-512 KiB rowgroup target) the cap applies directly.
fn snap_to_band(bytes: u64, rowgroup_bytes: u64, cap_bytes: u64) -> u64 {
    let mut target = bytes;
    if cap_bytes > 0 {
        target = target.min(cap_bytes);
    }
    let ladder = band_ladder(rowgroup_bytes);
    if ladder.is_empty() {
        return target;
    }
    let mut best = ladder[0];
    let mut best_dist = u64::MAX;
    for &rung in &ladder {
        let dist = rung.abs_diff(target);
        if dist < best_dist {
            best_dist = dist;
            best = rung;
        }
    }
    best.min(target)
}

/// THE one pure function: the probe plan for a query under a policy.
///
/// * `cells` — the candidate pool: `k_c` per-segment coarse cells, or the
///   collection-total usable cells for directory planning.
/// * `corpus_rows` — rows the candidate pool covers (drives ByteBudget's
///   sub-linear scaling).
/// * `rowgroup_bytes` — the per-backend IOPS-budget target
///   (`IopsBudget::for_path(path).target_block_bytes()`): the band geometry.
/// * `avg_cell_bytes` — mean Region-A extent per cell (both call sites have
///   the parsed cell list; 0 falls back to rowgroup_bytes/8).
///
/// Clamped `[min_cells, max_cells]` then to `cells`, with the
/// max-then-min ordering so a pool smaller than `min_cells` cannot panic
/// (no-panic mandate #4).
///
/// SCALING CAVEAT (per-pool semantics): `corpus_rows` scales the budget of
/// THIS pool. The collection-directory site passes collection-total rows
/// (the authoritative path); a per-segment caller that lacks collection
/// context scales per-SEGMENT — more probes than the collection-level plan
/// would take, bounded by `cap_bytes`/`max_cells`. Callers should prefer
/// the collection path (TD-RDSTRAT-14 §Deferred).
pub fn resolve_probe_plan(
    policy: &ProbePolicy,
    cells: usize,
    corpus_rows: u64,
    rowgroup_bytes: u64,
    avg_cell_bytes: u64,
) -> ProbePlan {
    let pool = cells.max(1) as f64;
    let cell_bytes = if avg_cell_bytes > 0 {
        avg_cell_bytes
    } else {
        (rowgroup_bytes / 8).max(1)
    };
    match policy.algorithm {
        // ByteBudget plans in BYTES end-to-end: budget → cap → band → cells,
        // so the cell count, the planned bytes and the band can never
        // contradict each other and the cap genuinely bounds fan-out.
        ProbeAlgorithm::ByteBudget => {
            // Budget grows quarter-power with corpus; 1M rows is the base.
            let scale = (corpus_rows.max(1) as f64 / 1_000_000.0).powf(0.25);
            let mut budget = policy.base_bytes as f64 * scale;
            if policy.cap_bytes > 0 {
                budget = budget.min(policy.cap_bytes as f64);
            }
            let snapped = snap_to_band(budget as u64, rowgroup_bytes, policy.cap_bytes).max(1);
            let by_budget = ((snapped as f64) / (cell_bytes as f64)).ceil() as usize;
            // The recall floor (min_cells) can never exceed what the cap
            // affords: the CAP is the contract (it bounds fan-out), the
            // floor only raises the count when the budget has room.
            let cap_cells = if policy.cap_bytes > 0 {
                ((policy.cap_bytes / cell_bytes.max(1)) as usize).max(1)
            } else {
                usize::MAX
            };
            let floor = policy.min_cells.max(1).min(cap_cells);
            let floored = by_budget.max(floor);
            let bounded = if policy.max_cells > 0 {
                floored.min(policy.max_cells)
            } else {
                floored
            };
            let chosen = bounded.min(cells.max(1));
            ProbePlan {
                cells: chosen,
                probe_bytes: chosen as u64 * cell_bytes,
                band: snapped,
            }
        }
        _ => {
            let raw = match policy.algorithm {
                ProbeAlgorithm::Sqrt => pool.sqrt() * policy.multiplier,
                ProbeAlgorithm::CubeRoot => pool.cbrt() * policy.multiplier,
                ProbeAlgorithm::FourthRoot => pool.sqrt().sqrt() * policy.multiplier,
                ProbeAlgorithm::Log => pool.log2() * policy.multiplier + (snapped_log_base(policy)),
                ProbeAlgorithm::Fixed => policy.min_cells as f64,
                ProbeAlgorithm::ByteBudget => unreachable!("handled above"),
            };
            let raw = raw.ceil().max(1.0) as usize;
            let floored = raw.max(policy.min_cells.max(1));
            let bounded = if policy.max_cells > 0 {
                floored.min(policy.max_cells)
            } else {
                floored
            };
            let chosen = bounded.min(cells.max(1));
            ProbePlan {
                cells: chosen,
                probe_bytes: chosen as u64 * cell_bytes,
                band: 0,
            }
        }
    }
}

/// Log form's corpus-scale term: `log2(base_bytes)` so the budget knob
/// meaningfully scales the logarithmic ladder too.
fn snapped_log_base(policy: &ProbePolicy) -> f64 {
    (policy.base_bytes as f64).log2().max(1.0)
}

/// The property key for the algorithm (`probe.algorithm`).
pub const PROBE_ALGORITHM_PROPERTY_KEY: &str = "probe.algorithm";
/// The tag prefix for the algorithm channel (`probe.algorithm:x`).
pub const PROBE_ALGORITHM_TAG_PREFIX: &str = "probe.algorithm:";
const PROBE_MULTIPLIER_KEY: &str = "probe.multiplier";
const PROBE_MIN_CELLS_KEY: &str = "probe.min_cells";
const PROBE_MAX_CELLS_KEY: &str = "probe.max_cells";
const PROBE_BASE_MB_KEY: &str = "probe.base_mb";
const PROBE_CAP_MB_KEY: &str = "probe.cap_mb";

fn property_usize(properties: &HashMap<String, String>, key: &str) -> Option<usize> {
    properties
        .iter()
        .filter(|(k, _)| k.trim().to_ascii_lowercase() == key)
        .min_by_key(|(k, _)| k.as_str())
        .and_then(|(_, v)| v.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
}

fn property_u64_mb(properties: &HashMap<String, String>, key: &str) -> Option<u64> {
    property_usize(properties, key).map(|mb| mb as u64 * 1024 * 1024)
}

/// The full precedence chain — the ONLY exported resolution path (no
/// single-channel resolvers: a caller resolving only tags or only properties
/// silently misses the other layers, the "property set but inert" bug class
/// closed at the type level in the compaction policy first).
///
/// Per key: collection property > operator tag (algorithm only) > server
/// TOML > code default. An explicit value at any layer wins over defaults
/// below it; silence falls through.
pub fn resolve_probe_policy(
    properties: &HashMap<String, String>,
    tags: &[String],
    server: &ServerProbePolicy,
) -> ProbePolicy {
    let mut policy = ProbePolicy::default();

    // Server TOML layer (validated at load; junk is inert here, rejected there).
    if let Some(algorithm) = server.algorithm.as_deref().and_then(ProbeAlgorithm::parse) {
        policy.algorithm = algorithm;
    }
    if let Some(multiplier) = server.multiplier.filter(|m| *m > 0.0) {
        policy.multiplier = multiplier;
    }
    if let Some(min) = server.min_cells.filter(|m| *m > 0) {
        policy.min_cells = min;
    }
    if let Some(max) = server.max_cells {
        policy.max_cells = max;
    }
    if let Some(base) = server.base_bytes.filter(|b| *b > 0) {
        policy.base_bytes = base;
    }
    if let Some(cap) = server.cap_bytes.filter(|c| *c > 0) {
        policy.cap_bytes = cap;
    }

    // Operator tag layer (algorithm only — the legacy channel shape).
    let mut tag_algorithm = None;
    for tag in tags {
        if let Some(parsed) = tag
            .trim()
            .strip_prefix(PROBE_ALGORITHM_TAG_PREFIX)
            .and_then(ProbeAlgorithm::parse)
        {
            tag_algorithm = Some(parsed);
        }
    }
    if let Some(algorithm) = tag_algorithm {
        policy.algorithm = algorithm;
    }

    // Collection property layer (the top).
    if let Some(algorithm) = properties
        .iter()
        .filter(|(k, _)| k.trim().to_ascii_lowercase() == PROBE_ALGORITHM_PROPERTY_KEY)
        .min_by_key(|(k, _)| k.as_str())
        .and_then(|(_, v)| ProbeAlgorithm::parse(v))
    {
        policy.algorithm = algorithm;
    }
    if let Some(multiplier) = properties
        .iter()
        .filter(|(k, _)| k.trim().to_ascii_lowercase() == PROBE_MULTIPLIER_KEY)
        .min_by_key(|(k, _)| k.as_str())
        .and_then(|(_, v)| v.trim().parse::<f64>().ok())
        .filter(|m| *m > 0.0)
    {
        policy.multiplier = multiplier;
    }
    if let Some(min) = property_usize(properties, PROBE_MIN_CELLS_KEY) {
        policy.min_cells = min;
    }
    if let Some(max) = property_usize(properties, PROBE_MAX_CELLS_KEY) {
        policy.max_cells = max;
    }
    if let Some(base) = property_u64_mb(properties, PROBE_BASE_MB_KEY) {
        policy.base_bytes = base;
    }
    if let Some(cap) = property_u64_mb(properties, PROBE_CAP_MB_KEY) {
        policy.cap_bytes = cap;
    }
    policy
}

/// Validate `probe.*` properties at SET time — fail-closed with an
/// actionable message (mirror of `validate_compaction_properties`).
pub fn validate_probe_properties(properties: &HashMap<String, String>) -> Result<(), String> {
    for (raw_key, raw_value) in properties {
        let key = raw_key.trim().to_ascii_lowercase();
        if !key.starts_with("probe.") {
            continue;
        }
        match key.as_str() {
            PROBE_ALGORITHM_PROPERTY_KEY => {
                if ProbeAlgorithm::parse(raw_value).is_none() {
                    return Err(format!(
                        "probe.algorithm: expected one of sqrt|cbrt|fourth_root|log|fixed|byte_budget, got '{raw_value}'"
                    ));
                }
            }
            PROBE_MULTIPLIER_KEY => {
                let parsed = raw_value.trim().parse::<f64>().unwrap_or(0.0);
                if parsed <= 0.0 || !parsed.is_finite() {
                    return Err(format!(
                        "probe.multiplier: expected a positive finite number, got '{raw_value}'"
                    ));
                }
            }
            PROBE_MIN_CELLS_KEY | PROBE_MAX_CELLS_KEY | PROBE_BASE_MB_KEY | PROBE_CAP_MB_KEY => {
                let parsed = raw_value.trim().parse::<usize>().unwrap_or(0);
                if parsed == 0 {
                    return Err(format!(
                        "{key}: expected a positive integer, got '{raw_value}'"
                    ));
                }
            }
            _ => {
                return Err(format!(
                    "unknown probe property '{raw_key}': expected probe.algorithm, probe.multiplier, probe.min_cells, probe.max_cells, probe.base_mb or probe.cap_mb"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn props(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn sqrt_policy_reproduces_the_historical_formula_exactly() {
        // Regression: the default policy must equal the old geometric_nprobe
        // across the whole realistic pool range — zero behavior change.
        let policy = ProbePolicy::default();
        for k_c in 1..=4096usize {
            let old = {
                let raw = ((k_c as f32).sqrt() * 2.0f32).ceil() as usize;
                let floored = raw.max(3);
                floored.min(k_c.max(1))
            };
            let plan = resolve_probe_plan(&policy, k_c, 1_000_000, 4 * 1024 * 1024, 512 * 1024);
            assert_eq!(plan.cells, old, "k_c={k_c}");
        }
    }

    #[test]
    fn sublinear_ladders_flatten_fan_out() {
        let policy = |algorithm| ProbePolicy {
            algorithm,
            ..Default::default()
        };
        let plan = |p: &ProbePolicy, cells: usize| {
            resolve_probe_plan(p, cells, 10_000_000, 4 * 1024 * 1024, 256 * 1024).cells
        };
        // 10,000-cell pool (multiplier 2.0): sqrt 100×2=200, cbrt
        // ceil(21.54×2)=44, fourth-root 10×2=20 — strictly flattening.
        let sqrt = plan(&policy(ProbeAlgorithm::Sqrt), 10_000);
        let cbrt = plan(&policy(ProbeAlgorithm::CubeRoot), 10_000);
        let fourth = plan(&policy(ProbeAlgorithm::FourthRoot), 10_000);
        assert!(sqrt > cbrt && cbrt > fourth, "{sqrt} {cbrt} {fourth}");
        assert_eq!(sqrt, 200);
        assert_eq!(cbrt, 44);
        assert_eq!(fourth, 20);
    }

    #[test]
    fn byte_budget_scales_quarter_power_and_snaps_to_bands() {
        let policy = ProbePolicy {
            algorithm: ProbeAlgorithm::ByteBudget,
            ..Default::default() // base 4 MiB, cap 32 MiB
        };
        let cell = 512 * 1024; // 0.5 MiB cells
        let rg = 4 * 1024 * 1024;
        // 1M rows: budget 4 MiB -> snapped 4 MiB -> 8 cells of 0.5 MiB.
        let small = resolve_probe_plan(&policy, 10_000, 1_000_000, rg, cell);
        assert_eq!(small.cells, 8);
        assert_eq!(small.band, 4 * 1024 * 1024);
        // 10B rows: raw budget 40 MiB exceeds the 32 MiB cap -> snapped to
        // the 16 MiB rung -> 32 cells. cells/bytes/band AGREE (the cap
        // genuinely bounds fan-out — the plan never exceeds it).
        let big = resolve_probe_plan(&policy, 10_000, 10_000_000_000, rg, cell);
        assert_eq!(big.band, 16 * 1024 * 1024);
        assert_eq!(big.cells, 32);
        assert_eq!(big.probe_bytes, big.cells as u64 * cell);
        assert!(big.probe_bytes <= policy.cap_bytes);
        assert!(
            big.cells > small.cells,
            "sub-linear growth: {big:?} vs {small:?}"
        );
    }

    #[test]
    fn byte_budget_cap_bounds_cells_not_just_labels() {
        // The review regression: with a cap of 1 MiB and 0.5 MiB cells, no
        // corpus size may plan more than 2 cells — the cap limits the CELL
        // COUNT, not merely the reported bytes.
        let policy = ProbePolicy {
            algorithm: ProbeAlgorithm::ByteBudget,
            base_bytes: 4 * 1024 * 1024,
            cap_bytes: 1024 * 1024,
            ..Default::default()
        };
        for rows in [1_000_000u64, 10_000_000_000, 1_000_000_000_000] {
            let plan = resolve_probe_plan(&policy, 100_000, rows, 4 * 1024 * 1024, 512 * 1024);
            assert!(plan.cells <= 2, "rows={rows}: {plan:?}");
            assert!(
                plan.probe_bytes <= policy.cap_bytes,
                "rows={rows}: {plan:?}"
            );
        }
    }

    #[test]
    fn clamps_respect_min_max_and_pool() {
        let policy = ProbePolicy {
            min_cells: 5,
            max_cells: 10,
            ..Default::default()
        };
        assert_eq!(resolve_probe_plan(&policy, 2, 1000, 1 << 20, 4096).cells, 2);
        assert_eq!(
            resolve_probe_plan(&policy, 1_000_000, 1000, 1 << 20, 4096).cells,
            10
        );
    }

    #[test]
    fn band_ladder_derives_from_rowgroup_geometry() {
        let rungs = band_ladder(4 * 1024 * 1024);
        assert_eq!(
            rungs,
            vec![
                1024 * 1024,
                2 * 1024 * 1024,
                4 * 1024 * 1024,
                8 * 1024 * 1024,
                16 * 1024 * 1024
            ]
        );
        // Sub-megabyte rungs are pointless and filtered.
        assert!(band_ladder(1024).iter().all(|r| *r >= 512 * 1024));
    }

    #[test]
    fn policy_chain_property_over_tag_over_server_over_code() {
        let tag = vec!["probe.algorithm:cbrt".to_string()];
        let server = ServerProbePolicy {
            algorithm: Some("fourth_root".to_string()),
            ..Default::default()
        };
        // Property wins over tag and server.
        assert_eq!(
            resolve_probe_policy(&props(&[("probe.algorithm", "log")]), &tag, &server).algorithm,
            ProbeAlgorithm::Log
        );
        // Tag wins over server; explicit layers beat the code default.
        assert_eq!(
            resolve_probe_policy(&HashMap::new(), &tag, &server).algorithm,
            ProbeAlgorithm::CubeRoot
        );
        assert_eq!(
            resolve_probe_policy(&HashMap::new(), &[], &server).algorithm,
            ProbeAlgorithm::FourthRoot
        );
        assert_eq!(
            resolve_probe_policy(&HashMap::new(), &[], &ServerProbePolicy::default()).algorithm,
            ProbeAlgorithm::Sqrt
        );
        // Numeric layers resolve per key: property min over server base.
        let resolved = resolve_probe_policy(
            &props(&[("probe.min_cells", "7")]),
            &[],
            &ServerProbePolicy {
                base_bytes: Some(8 * 1024 * 1024),
                ..Default::default()
            },
        );
        assert_eq!(resolved.min_cells, 7);
        assert_eq!(resolved.base_bytes, 8 * 1024 * 1024);
    }

    #[test]
    fn case_duplicate_properties_resolve_deterministically() {
        for _ in 0..20 {
            let properties = props(&[
                ("PROBE.ALGORITHM", "sqrt"),
                ("Probe.Algorithm", "byte_budget"),
            ]);
            assert_eq!(
                resolve_probe_policy(&properties, &[], &ServerProbePolicy::default()).algorithm,
                // "PROBE.ALGORITHM" < "Probe.Algorithm" (ASCII 'R' < 'r')
                ProbeAlgorithm::Sqrt
            );
        }
    }

    #[test]
    fn validation_fails_closed() {
        assert!(validate_probe_properties(&props(&[("probe.algorithm", "cbrt")])).is_ok());
        assert!(validate_probe_properties(&props(&[("probe.multiplier", "1.5")])).is_ok());
        assert!(validate_probe_properties(&props(&[("probe.base_mb", "8")])).is_ok());
        // Non-compaction keys are ignored.
        assert!(validate_probe_properties(&props(&[("cache.index.mode", "pin")])).is_ok());
        assert!(validate_probe_properties(&props(&[("probe.algorithm", "sqrt2")])).is_err());
        assert!(validate_probe_properties(&props(&[("probe.multiplier", "0")])).is_err());
        assert!(validate_probe_properties(&props(&[("probe.min_cells", "0")])).is_err());
        assert!(validate_probe_properties(&props(&[("probe.unknown", "1")])).is_err());
    }
}
