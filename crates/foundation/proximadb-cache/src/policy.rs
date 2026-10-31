//! Per-collection cache policy over PAX region groups.
//!
//! A segment's regions are cached as six groups, one policy each:
//!
//! | group | regions | read by |
//! |---|---|---|
//! | [`RegionGroup::Control`] | header, footer, A0, RaBitQ params | every query |
//! | [`RegionGroup::Index`] | A (RaBitQ codes) | every vector query (probe) |
//! | [`RegionGroup::Rerank`] | B (SQ8 rerank) | every vector query (survivors) |
//! | [`RegionGroup::Filter`] | E (filter columns) | filtered queries |
//! | [`RegionGroup::Payload`] | D (row payload, OID chunks) | the top-k rows |
//! | [`RegionGroup::Exact`] | C (exact f32) | compaction, point reads |
//!
//! A group's effective policy is resolved field by field (mode, placement)
//! through four layers, highest first: the collection's `cache.*` properties
//! (Hive-TBLPROPERTIES-style and perf-only: they never change results), the
//! server's explicit defaults, an advisor recommendation (only when the
//! advisor is `on`, only for fields no layer above set), then the built-in
//! default. Each field records its [`Provenance`].
//!
//! The built-in defaults describe the caching the engine does today at the
//! shipped default config (`cache_on_write = invariant`, a configured
//! persistent disk store, non-zero cache budgets). Step 2b derives them from
//! the live engine config instead, so a server that changes those knobs
//! reports its real behaviour:
//! - the invariants cache seeds control and Region A on publish (RAM, then
//!   the persistent disk store), evicting Region A last;
//! - the survivor range cache serves Region B ranges and Region D OID chunks
//!   when read (RAM, then disk); Region D row blocks (rehydration, the
//!   filtered Stage-F pass) are not cached;
//! - Regions E and C are not cached;
//! - ECONOMY behaves as BALANCED (economy residency is not implemented);
//! - PERFORMANCE adds whole-segment disk residency while its reservation
//!   grant covers the collection. That residency is a property of the class,
//!   reported by [`EffectivePolicy::whole_segment_resident`], and no group
//!   setting can demote it.
//!
//! The advisor runs out of band: [`resolve_cache_policy`] takes its
//! precomputed recommendations, so resolving never runs a model.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// A cache unit: regions a query reads together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegionGroup {
    /// Header, footer, A0 and RaBitQ params.
    Control,
    /// Region A: RaBitQ codes.
    Index,
    /// Region B: SQ8 rerank vectors.
    Rerank,
    /// Region E: filter columns.
    Filter,
    /// Region D: row payload. Only its OID chunks are cached today.
    Payload,
    /// Region C: exact f32 vectors.
    Exact,
}

impl RegionGroup {
    pub const ALL: [RegionGroup; 6] = [
        RegionGroup::Control,
        RegionGroup::Index,
        RegionGroup::Rerank,
        RegionGroup::Filter,
        RegionGroup::Payload,
        RegionGroup::Exact,
    ];

    /// The group's name in `cache.<group>.*` property keys.
    pub fn as_str(self) -> &'static str {
        match self {
            RegionGroup::Control => "control",
            RegionGroup::Index => "index",
            RegionGroup::Rerank => "rerank",
            RegionGroup::Filter => "filter",
            RegionGroup::Payload => "payload",
            RegionGroup::Exact => "exact",
        }
    }

    fn slot(self) -> usize {
        self as usize
    }
}

impl FromStr for RegionGroup {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        RegionGroup::ALL
            .into_iter()
            .find(|g| g.as_str() == s)
            .ok_or(())
    }
}

/// How a group's bytes are cached.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroupMode {
    /// Resident: warmed on publish and kept (not evicted while pinned).
    Pin,
    /// Warmed on publish, then evictable.
    Warm,
    /// Cached when read, evictable.
    OnDemand,
    /// Never cached for serving.
    Off,
}

impl GroupMode {
    pub fn as_str(self) -> &'static str {
        match self {
            GroupMode::Pin => "pin",
            GroupMode::Warm => "warm",
            GroupMode::OnDemand => "on_demand",
            GroupMode::Off => "off",
        }
    }
}

impl FromStr for GroupMode {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "pin" => Ok(GroupMode::Pin),
            "warm" => Ok(GroupMode::Warm),
            "on_demand" | "on-demand" => Ok(GroupMode::OnDemand),
            "off" => Ok(GroupMode::Off),
            _ => Err(()),
        }
    }
}

/// Where a cached group's bytes live.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Placement {
    /// Process memory only.
    Ram,
    /// The persistent local-disk store only.
    Disk,
    /// Memory first, then the persistent local-disk store.
    RamThenDisk,
}

impl Placement {
    pub fn as_str(self) -> &'static str {
        match self {
            Placement::Ram => "ram",
            Placement::Disk => "disk",
            Placement::RamThenDisk => "ram_then_disk",
        }
    }
}

impl FromStr for Placement {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "ram" => Ok(Placement::Ram),
            "disk" => Ok(Placement::Disk),
            "ram_then_disk" | "ram-then-disk" => Ok(Placement::RamThenDisk),
            _ => Err(()),
        }
    }
}

/// One group's policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GroupPolicy {
    pub mode: GroupMode,
    pub placement: Placement,
}

impl GroupPolicy {
    pub const fn new(mode: GroupMode, placement: Placement) -> Self {
        Self { mode, placement }
    }
}

/// Whether, and how, advisor recommendations are used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdvisorMode {
    /// Recommendations are ignored.
    Off,
    /// Recommendations are reported (and metered) but never applied.
    #[default]
    Shadow,
    /// Recommendations are applied to fields no layer above decided.
    On,
}

impl AdvisorMode {
    pub fn as_str(self) -> &'static str {
        match self {
            AdvisorMode::Off => "off",
            AdvisorMode::Shadow => "shadow",
            AdvisorMode::On => "on",
        }
    }
}

impl FromStr for AdvisorMode {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "off" => Ok(AdvisorMode::Off),
            "shadow" => Ok(AdvisorMode::Shadow),
            "on" => Ok(AdvisorMode::On),
            _ => Err(()),
        }
    }
}

/// Which layer decided a field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provenance {
    /// A `cache.*` property on the collection.
    Collection,
    /// An explicit server-level default.
    Server,
    /// An applied advisor recommendation.
    Advisor,
    /// The built-in default.
    Builtin,
}

impl Provenance {
    pub fn as_str(self) -> &'static str {
        match self {
            Provenance::Collection => "collection",
            Provenance::Server => "server",
            Provenance::Advisor => "advisor",
            Provenance::Builtin => "builtin",
        }
    }
}

/// The storage class a collection is in (TD-TIER-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StorageClassHint {
    Economy,
    #[default]
    Balanced,
    Performance,
}

/// The property-key namespace this module owns.
pub const CACHE_PROPERTY_PREFIX: &str = "cache.";

/// A `cache.*` property that is not a known key, has an invalid value, or
/// is given twice under keys that differ only in case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachePropertyError {
    /// The key as the client wrote it.
    pub key: String,
    pub reason: String,
}

impl fmt::Display for CachePropertyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cache property {:?}: {}", self.key, self.reason)
    }
}

impl std::error::Error for CachePropertyError {}

/// Normalize the `cache.*` keys and values of `properties` (trimmed,
/// lowercase) and drop every other key. Keys that normalize to the same key
/// are an error, so a `SET` that differs from a stored key only in case must
/// overwrite it rather than add a second, conflicting key.
pub fn canonicalize_cache_properties(
    properties: &HashMap<String, String>,
) -> Result<BTreeMap<String, String>, CachePropertyError> {
    Ok(canonical_with_raw_keys(properties)?
        .into_iter()
        .map(|(key, (_, value))| (key, value))
        .collect())
}

/// [`canonicalize_cache_properties`], keeping each key as the client wrote
/// it (for error messages).
fn canonical_with_raw_keys(
    properties: &HashMap<String, String>,
) -> Result<BTreeMap<String, (String, String)>, CachePropertyError> {
    let mut raw: Vec<(&String, &String)> = properties.iter().collect();
    raw.sort();
    let mut out: BTreeMap<String, (String, String)> = BTreeMap::new();
    for (raw_key, raw_value) in raw {
        let key = raw_key.trim().to_ascii_lowercase();
        if !key.starts_with(CACHE_PROPERTY_PREFIX) {
            continue;
        }
        let value = raw_value.trim().to_ascii_lowercase();
        if let Some((first, _)) = out.get(&key) {
            return Err(CachePropertyError {
                key: raw_key.clone(),
                reason: format!("duplicates {first:?} (keys are case-insensitive)"),
            });
        }
        out.insert(key, (raw_key.clone(), value));
    }
    Ok(out)
}

/// The parsed `cache.*` properties of one collection.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CollectionCacheProperties {
    modes: BTreeMap<RegionGroup, GroupMode>,
    placements: BTreeMap<RegionGroup, Placement>,
    pub advisor: Option<AdvisorMode>,
}

impl CollectionCacheProperties {
    /// Parse every `cache.*` key of `properties`; other keys are ignored. An
    /// unknown `cache.*` key, an invalid value, or a case-only duplicate is
    /// an error, so a typo is refused at `SET` time rather than ignored.
    pub fn parse(properties: &HashMap<String, String>) -> Result<Self, CachePropertyError> {
        let mut parsed = CollectionCacheProperties::default();
        for (key, (raw_key, value)) in canonical_with_raw_keys(properties)? {
            let error = |reason: &str| CachePropertyError {
                key: raw_key.clone(),
                reason: reason.to_string(),
            };
            let rest = &key[CACHE_PROPERTY_PREFIX.len()..];
            if rest == "advisor" {
                parsed.advisor = Some(
                    value
                        .parse()
                        .map_err(|()| error("expected off, shadow or on"))?,
                );
                continue;
            }
            let (group, field) = rest
                .split_once('.')
                .ok_or_else(|| error("expected cache.<group>.<mode|placement> or cache.advisor"))?;
            let group: RegionGroup = group.parse().map_err(|()| {
                error("unknown group; expected control, index, rerank, filter, payload or exact")
            })?;
            match field {
                "mode" => {
                    let mode = value
                        .parse()
                        .map_err(|()| error("expected pin, warm, on_demand or off"))?;
                    parsed.modes.insert(group, mode);
                }
                "placement" => {
                    let placement = value
                        .parse()
                        .map_err(|()| error("expected ram, disk or ram_then_disk"))?;
                    parsed.placements.insert(group, placement);
                }
                _ => return Err(error("unknown field; expected mode or placement")),
            }
        }
        Ok(parsed)
    }

    pub fn mode(&self, group: RegionGroup) -> Option<GroupMode> {
        self.modes.get(&group).copied()
    }

    pub fn placement(&self, group: RegionGroup) -> Option<Placement> {
        self.placements.get(&group).copied()
    }
}

/// Validate client-supplied table properties: every `cache.*` key must parse.
pub fn validate_cache_properties(
    properties: &HashMap<String, String>,
) -> Result<(), CachePropertyError> {
    CollectionCacheProperties::parse(properties).map(|_| ())
}

/// The server's explicit per-group defaults and advisor mode
/// (`[storage.cache_policy]`). A field absent here falls through to the
/// advisor, then the built-in default.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerCachePolicy {
    pub modes: BTreeMap<RegionGroup, GroupMode>,
    pub placements: BTreeMap<RegionGroup, Placement>,
    pub advisor: AdvisorMode,
}

/// The built-in policy of `group`: today's caching (see the module docs).
/// Every class shares it; PERFORMANCE adds whole-segment residency on top.
pub fn builtin_group_policy(group: RegionGroup) -> GroupPolicy {
    use GroupMode::*;
    use Placement::*;
    match group {
        RegionGroup::Control | RegionGroup::Index => GroupPolicy::new(Warm, RamThenDisk),
        RegionGroup::Rerank | RegionGroup::Payload => GroupPolicy::new(OnDemand, RamThenDisk),
        RegionGroup::Filter | RegionGroup::Exact => GroupPolicy::new(Off, Disk),
    }
}

/// One advisor recommendation for one group.
#[derive(Debug, Clone, PartialEq)]
pub struct Recommendation {
    pub group: RegionGroup,
    pub policy: GroupPolicy,
    /// In `[0, 1]`.
    pub confidence: f64,
    /// Estimated monthly operator savings in USD (requests saved minus cache
    /// cost); negative means the change costs more than it saves.
    pub expected_monthly_savings_usd: f64,
}

/// Per-group measurements an advisor decides from.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GroupTelemetry {
    /// Bytes the group occupies across the collection's live segments.
    pub bytes: u64,
    /// Object-store GETs per query this group caused, over the window.
    pub object_gets_per_query: f64,
    /// Fraction of the group's reads served from cache.
    pub hit_ratio: f64,
}

/// What an advisor sees for one collection.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AdvisorInput {
    pub groups: BTreeMap<RegionGroup, GroupTelemetry>,
    pub queries_per_month: f64,
    pub usd_per_million_gets: f64,
    /// USD per GB-month for each placement.
    pub usd_per_gb_month: BTreeMap<Placement, f64>,
    /// Pin budget left in the collection's reservation pool.
    pub pool_headroom_bytes: u64,
}

/// The seam for policy tuning: from a collection's measurements, recommend
/// group policies. Implementations range from a deterministic break-even
/// rule to a learned (bandit/RL) policy. It runs out of band; the resolver
/// takes its output and enforces the guardrails (explicit settings win,
/// advisor mode, well-formed recommendations).
pub trait CachePolicyAdvisor: Send + Sync {
    fn recommend(&self, input: &AdvisorInput) -> Vec<Recommendation>;
}

/// Why a recommendation was not applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnappliedReason {
    /// The advisor mode is `shadow`.
    Shadow,
    /// Every field it would change was set by the collection or the server.
    Explicit,
    /// A higher-confidence recommendation for the same group won.
    Superseded,
    /// Some of its fields were set explicitly and kept; only the others
    /// were applied.
    Partial,
    /// Its confidence or savings is not a finite number in range.
    Invalid,
}

/// A recommendation that was not applied, and why.
#[derive(Debug, Clone, PartialEq)]
pub struct Unapplied {
    pub recommendation: Recommendation,
    pub reason: UnappliedReason,
}

/// One group's resolved policy and the layer that decided each field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedGroup {
    pub policy: GroupPolicy,
    pub mode_from: Provenance,
    pub placement_from: Provenance,
}

/// What [`resolve_cache_policy`] resolves from.
#[derive(Debug, Clone, Copy)]
pub struct ResolveInput<'a> {
    pub collection: &'a CollectionCacheProperties,
    pub server: &'a ServerCachePolicy,
    pub class: StorageClassHint,
    /// Whether a PERFORMANCE reservation grant currently covers the
    /// collection.
    pub performance_grant: bool,
    /// The advisor's latest recommendations, computed out of band.
    pub recommendations: &'a [Recommendation],
}

/// A collection's effective cache policy.
#[derive(Debug, Clone, PartialEq)]
pub struct EffectivePolicy {
    groups: [ResolvedGroup; 6],
    pub class: StorageClassHint,
    /// Whole-segment disk residency (PERFORMANCE with a grant). Independent
    /// of, and not demotable by, the group settings.
    pub whole_segment_resident: bool,
    pub advisor: AdvisorMode,
    pub unapplied: Vec<Unapplied>,
}

impl EffectivePolicy {
    pub fn group(&self, group: RegionGroup) -> ResolvedGroup {
        self.groups[group.slot()]
    }
}

fn well_formed(r: &Recommendation) -> bool {
    r.confidence.is_finite()
        && (0.0..=1.0).contains(&r.confidence)
        && r.expected_monthly_savings_usd.is_finite()
}

fn layered<T: Copy>(collection: Option<T>, server: Option<T>, builtin: T) -> (T, Provenance) {
    match (collection, server) {
        (Some(v), _) => (v, Provenance::Collection),
        (None, Some(v)) => (v, Provenance::Server),
        (None, None) => (builtin, Provenance::Builtin),
    }
}

/// Resolve a collection's effective policy, field by field: collection
/// property > server explicit default > advisor (mode `on`, only for fields
/// no layer above set; the highest-confidence well-formed recommendation per
/// group) > built-in default. A collection's `cache.advisor` overrides the
/// server's advisor mode.
pub fn resolve_cache_policy(input: ResolveInput<'_>) -> EffectivePolicy {
    let advisor = input.collection.advisor.unwrap_or(input.server.advisor);
    let mut groups = RegionGroup::ALL.map(|group| {
        let builtin = builtin_group_policy(group);
        let (mode, mode_from) = layered(
            input.collection.mode(group),
            input.server.modes.get(&group).copied(),
            builtin.mode,
        );
        let (placement, placement_from) = layered(
            input.collection.placement(group),
            input.server.placements.get(&group).copied(),
            builtin.placement,
        );
        ResolvedGroup {
            policy: GroupPolicy::new(mode, placement),
            mode_from,
            placement_from,
        }
    });
    let mut unapplied = Vec::new();
    if advisor != AdvisorMode::Off {
        let mut best: BTreeMap<RegionGroup, &Recommendation> = BTreeMap::new();
        for r in input.recommendations {
            if !well_formed(r) {
                unapplied.push(Unapplied {
                    recommendation: r.clone(),
                    reason: UnappliedReason::Invalid,
                });
                continue;
            }
            let loser = match best.get(&r.group) {
                Some(current) if current.confidence >= r.confidence => Some(r),
                Some(current) => {
                    let previous = *current;
                    best.insert(r.group, r);
                    Some(previous)
                }
                None => {
                    best.insert(r.group, r);
                    None
                }
            };
            if let Some(loser) = loser {
                unapplied.push(Unapplied {
                    recommendation: loser.clone(),
                    reason: UnappliedReason::Superseded,
                });
            }
        }
        for (group, r) in best {
            if advisor == AdvisorMode::Shadow {
                unapplied.push(Unapplied {
                    recommendation: r.clone(),
                    reason: UnappliedReason::Shadow,
                });
                continue;
            }
            let g = &mut groups[group.slot()];
            let mode_free = g.mode_from == Provenance::Builtin;
            let placement_free = g.placement_from == Provenance::Builtin;
            if mode_free {
                g.policy.mode = r.policy.mode;
                g.mode_from = Provenance::Advisor;
            }
            if placement_free {
                g.policy.placement = r.policy.placement;
                g.placement_from = Provenance::Advisor;
            }
            let reason = match (mode_free, placement_free) {
                (true, true) => None,
                (false, false) => Some(UnappliedReason::Explicit),
                _ => Some(UnappliedReason::Partial),
            };
            if let Some(reason) = reason {
                unapplied.push(Unapplied {
                    recommendation: r.clone(),
                    reason,
                });
            }
        }
    }
    EffectivePolicy {
        groups,
        class: input.class,
        whole_segment_resident: input.class == StorageClassHint::Performance
            && input.performance_grant,
        advisor,
        unapplied,
    }
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

    fn rec(group: RegionGroup, mode: GroupMode, confidence: f64) -> Recommendation {
        Recommendation {
            group,
            policy: GroupPolicy::new(mode, Placement::Disk),
            confidence,
            expected_monthly_savings_usd: 1.0,
        }
    }

    fn resolve(
        collection: &CollectionCacheProperties,
        server: &ServerCachePolicy,
        recommendations: &[Recommendation],
    ) -> EffectivePolicy {
        resolve_cache_policy(ResolveInput {
            collection,
            server,
            class: StorageClassHint::Balanced,
            performance_grant: false,
            recommendations,
        })
    }

    #[test]
    fn parse_reads_cache_keys_and_ignores_others() {
        let parsed = CollectionCacheProperties::parse(&props(&[
            ("cache.index.mode", "pin"),
            (" Cache.Payload.Placement ", " RAM "),
            ("cache.rerank.mode", "on-demand"),
            ("cache.advisor", "off"),
            ("workload_profile", "olap"),
        ]))
        .unwrap();
        assert_eq!(parsed.mode(RegionGroup::Index), Some(GroupMode::Pin));
        assert_eq!(parsed.mode(RegionGroup::Rerank), Some(GroupMode::OnDemand));
        assert_eq!(parsed.placement(RegionGroup::Payload), Some(Placement::Ram));
        assert_eq!(parsed.mode(RegionGroup::Payload), None);
        assert_eq!(parsed.advisor, Some(AdvisorMode::Off));
    }

    #[test]
    fn unknown_cache_keys_and_bad_values_are_refused() {
        for (key, value) in [
            ("cache.ann.mode", "pin"),
            ("cache.index.modee", "pin"),
            ("cache.index", "pin"),
            ("cache.index.mode", "always"),
            ("cache.index.placement", "nvme"),
            ("cache.advisor", "auto"),
        ] {
            let err = validate_cache_properties(&props(&[(key, value)]))
                .expect_err(&format!("{key}={value}"));
            assert_eq!(err.key, key);
        }
    }

    #[test]
    fn errors_name_the_key_as_written() {
        let err =
            validate_cache_properties(&props(&[(" Cache.Index.Mode ", "always")])).unwrap_err();
        assert_eq!(err.key, " Cache.Index.Mode ");
    }

    #[test]
    fn server_config_rejects_unknown_fields() {
        assert!(serde_json::from_str::<ServerCachePolicy>(r#"{"advisr":"on"}"#).is_err());
    }

    #[test]
    fn case_only_duplicate_keys_are_refused_deterministically() {
        let both = props(&[("cache.index.mode", "pin"), ("CACHE.INDEX.MODE", "off")]);
        for _ in 0..8 {
            let err = CollectionCacheProperties::parse(&both).unwrap_err();
            assert_eq!(
                err.key, "cache.index.mode",
                "the later raw key in sorted order is named"
            );
        }
        let canonical =
            canonicalize_cache_properties(&props(&[(" Cache.Index.Mode ", " PIN "), ("x", "y")]))
                .unwrap();
        assert_eq!(
            canonical.into_iter().collect::<Vec<_>>(),
            vec![("cache.index.mode".to_string(), "pin".to_string())]
        );
    }

    #[test]
    fn builtin_defaults_describe_todays_caching() {
        use GroupMode::*;
        let mode = |g| builtin_group_policy(g).mode;
        // Invariants cache seeds control + Region A on publish.
        assert_eq!(mode(RegionGroup::Control), Warm);
        assert_eq!(mode(RegionGroup::Index), Warm);
        // Survivor range cache serves B ranges and D OID chunks when read.
        assert_eq!(mode(RegionGroup::Rerank), OnDemand);
        assert_eq!(mode(RegionGroup::Payload), OnDemand);
        // Regions E and C have no cache.
        assert_eq!(mode(RegionGroup::Filter), Off);
        assert_eq!(mode(RegionGroup::Exact), Off);
        for g in [
            RegionGroup::Control,
            RegionGroup::Index,
            RegionGroup::Rerank,
            RegionGroup::Payload,
        ] {
            assert_eq!(
                builtin_group_policy(g).placement,
                Placement::RamThenDisk,
                "{g:?}"
            );
        }
    }

    #[test]
    fn classes_share_group_defaults_and_performance_adds_residency() {
        let none = CollectionCacheProperties::default();
        let server = ServerCachePolicy::default();
        let at = |class, performance_grant| {
            resolve_cache_policy(ResolveInput {
                collection: &none,
                server: &server,
                class,
                performance_grant,
                recommendations: &[],
            })
        };
        let balanced = at(StorageClassHint::Balanced, false);
        for class in [StorageClassHint::Economy, StorageClassHint::Performance] {
            let p = at(class, true);
            for g in RegionGroup::ALL {
                assert_eq!(p.group(g), balanced.group(g), "{class:?} {g:?}");
            }
        }
        assert!(at(StorageClassHint::Performance, true).whole_segment_resident);
        assert!(!at(StorageClassHint::Performance, false).whole_segment_resident);
        assert!(!at(StorageClassHint::Economy, true).whole_segment_resident);
    }

    #[test]
    fn residency_is_not_demotable_by_group_settings() {
        let collection =
            CollectionCacheProperties::parse(&props(&[("cache.exact.mode", "off")])).unwrap();
        let mut server = ServerCachePolicy {
            advisor: AdvisorMode::On,
            ..Default::default()
        };
        server.modes.insert(RegionGroup::Index, GroupMode::Off);
        let p = resolve_cache_policy(ResolveInput {
            collection: &collection,
            server: &server,
            class: StorageClassHint::Performance,
            performance_grant: true,
            recommendations: &[rec(RegionGroup::Rerank, GroupMode::Off, 0.9)],
        });
        assert!(p.whole_segment_resident);
    }

    #[test]
    fn fields_resolve_independently_with_provenance() {
        let collection =
            CollectionCacheProperties::parse(&props(&[("cache.payload.mode", "pin")])).unwrap();
        let mut server = ServerCachePolicy::default();
        server
            .placements
            .insert(RegionGroup::Payload, Placement::Ram);
        server
            .modes
            .insert(RegionGroup::Filter, GroupMode::OnDemand);
        let p = resolve(&collection, &server, &[]);
        let payload = p.group(RegionGroup::Payload);
        assert_eq!(
            payload.policy,
            GroupPolicy::new(GroupMode::Pin, Placement::Ram)
        );
        assert_eq!(
            (payload.mode_from, payload.placement_from),
            (Provenance::Collection, Provenance::Server)
        );
        let filter = p.group(RegionGroup::Filter);
        assert_eq!(
            (filter.mode_from, filter.placement_from),
            (Provenance::Server, Provenance::Builtin)
        );
    }

    #[test]
    fn advisor_applies_only_when_on_and_only_to_unset_fields() {
        let recs = [
            rec(RegionGroup::Index, GroupMode::Pin, 0.9),
            rec(RegionGroup::Payload, GroupMode::Pin, 0.8),
            rec(RegionGroup::Rerank, GroupMode::Pin, 0.7),
        ];
        // Payload: the collection set mode only. Rerank: placement only.
        let collection = CollectionCacheProperties::parse(&props(&[
            ("cache.payload.mode", "off"),
            ("cache.rerank.placement", "ram"),
        ]))
        .unwrap();

        let shadow = resolve(&collection, &ServerCachePolicy::default(), &recs);
        assert_eq!(shadow.advisor, AdvisorMode::Shadow);
        assert_eq!(
            shadow.group(RegionGroup::Index).mode_from,
            Provenance::Builtin
        );
        assert_eq!(shadow.unapplied.len(), 3);
        assert!(
            shadow
                .unapplied
                .iter()
                .all(|u| u.reason == UnappliedReason::Shadow)
        );

        let server = ServerCachePolicy {
            advisor: AdvisorMode::On,
            ..Default::default()
        };
        let on = resolve(&collection, &server, &recs);
        let index = on.group(RegionGroup::Index);
        assert_eq!(index.policy.mode, GroupMode::Pin);
        assert_eq!(
            (index.mode_from, index.placement_from),
            (Provenance::Advisor, Provenance::Advisor)
        );
        // Payload mode stays the collection's; its unset placement is tuned.
        let payload = on.group(RegionGroup::Payload);
        assert_eq!(
            (payload.policy.mode, payload.mode_from),
            (GroupMode::Off, Provenance::Collection)
        );
        assert_eq!(payload.placement_from, Provenance::Advisor);
        // Rerank placement stays the collection's; the advisor tunes mode.
        let rerank = on.group(RegionGroup::Rerank);
        assert_eq!(
            (rerank.policy.mode, rerank.mode_from),
            (GroupMode::Pin, Provenance::Advisor)
        );
        assert_eq!(
            (rerank.policy.placement, rerank.placement_from),
            (Placement::Ram, Provenance::Collection)
        );
        let partial: Vec<RegionGroup> = on
            .unapplied
            .iter()
            .filter(|u| u.reason == UnappliedReason::Partial)
            .map(|u| u.recommendation.group)
            .collect();
        assert_eq!(partial, vec![RegionGroup::Rerank, RegionGroup::Payload]);
        assert_eq!(on.unapplied.len(), 2);

        let opted_out =
            CollectionCacheProperties::parse(&props(&[("cache.advisor", "off")])).unwrap();
        let off = resolve(&opted_out, &server, &recs);
        assert_eq!(off.advisor, AdvisorMode::Off);
        assert_eq!(off.group(RegionGroup::Index).mode_from, Provenance::Builtin);
        assert!(off.unapplied.is_empty());
    }

    #[test]
    fn explicit_server_fields_block_the_advisor() {
        let mut server = ServerCachePolicy {
            advisor: AdvisorMode::On,
            ..Default::default()
        };
        server.modes.insert(RegionGroup::Index, GroupMode::OnDemand);
        server
            .placements
            .insert(RegionGroup::Index, Placement::Disk);
        let p = resolve(
            &CollectionCacheProperties::default(),
            &server,
            &[rec(RegionGroup::Index, GroupMode::Pin, 0.9)],
        );
        assert_eq!(p.group(RegionGroup::Index).policy.mode, GroupMode::OnDemand);
        assert_eq!(p.unapplied.len(), 1);
        assert_eq!(p.unapplied[0].reason, UnappliedReason::Explicit);
    }

    #[test]
    fn duplicate_and_invalid_recommendations_are_reported() {
        let server = ServerCachePolicy {
            advisor: AdvisorMode::On,
            ..Default::default()
        };
        let mut nan = rec(RegionGroup::Filter, GroupMode::Pin, 0.5);
        nan.expected_monthly_savings_usd = f64::NAN;
        let recs = [
            rec(RegionGroup::Index, GroupMode::OnDemand, 0.4),
            rec(RegionGroup::Index, GroupMode::Pin, 0.9),
            rec(RegionGroup::Rerank, GroupMode::Pin, 1.5),
            nan,
        ];
        let p = resolve(&CollectionCacheProperties::default(), &server, &recs);
        assert_eq!(
            p.group(RegionGroup::Index).policy.mode,
            GroupMode::Pin,
            "highest confidence wins"
        );
        let reasons: Vec<(RegionGroup, GroupMode, UnappliedReason)> = p
            .unapplied
            .iter()
            .map(|u| {
                (
                    u.recommendation.group,
                    u.recommendation.policy.mode,
                    u.reason,
                )
            })
            .collect();
        assert!(reasons.contains(&(
            RegionGroup::Index,
            GroupMode::OnDemand,
            UnappliedReason::Superseded
        )));
        assert!(reasons.contains(&(
            RegionGroup::Rerank,
            GroupMode::Pin,
            UnappliedReason::Invalid
        )));
        assert!(reasons.contains(&(
            RegionGroup::Filter,
            GroupMode::Pin,
            UnappliedReason::Invalid
        )));
        assert_eq!(p.group(RegionGroup::Rerank).mode_from, Provenance::Builtin);
    }

    #[test]
    fn server_policy_round_trips_as_config() {
        let mut server = ServerCachePolicy {
            advisor: AdvisorMode::On,
            ..Default::default()
        };
        server.modes.insert(RegionGroup::Index, GroupMode::Pin);
        server
            .placements
            .insert(RegionGroup::Payload, Placement::RamThenDisk);
        let json = serde_json::to_string(&server).unwrap();
        assert!(
            json.contains("\"index\":\"pin\"") && json.contains("ram_then_disk"),
            "{json}"
        );
        let back: ServerCachePolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(back, server);
        let empty: ServerCachePolicy = serde_json::from_str("{}").unwrap();
        assert_eq!(empty.advisor, AdvisorMode::Shadow);
    }
}
