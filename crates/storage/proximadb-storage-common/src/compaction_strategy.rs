//! TD-COMPACT-15: the per-collection compaction strategy.
//!
//! Two strategies exist:
//! - **Hybrid** (default): the current leveled policy — fill each level,
//!   promote upward. Best for write-heavy collections.
//! - **Consolidate**: merge smallest-first across ALL levels toward a
//!   `target_file_size_mb`, producing uniform segments. Best for read-heavy
//!   or batch-loaded collections.
//!
//! The strategy is resolved through [`resolve_effective_strategy`]:
//! collection property > operator tag (`compaction.strategy:<s>`) > code
//! default. There is no server layer while background consolidate is refused
//! (TD-COMPACT-16).
//! The `compaction:off` TAG always wins over any strategy (it disarms all
//! background compaction including consolidate).

use std::collections::HashMap;

/// The compaction strategy a collection uses.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum CompactionStrategy {
    /// The current leveled policy (count + training arms).
    #[default]
    Hybrid,
    /// Smallest-first merge toward `target_file_size_mb` (TD-COMPACT-15).
    Consolidate,
}

impl CompactionStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hybrid => "hybrid",
            Self::Consolidate => "consolidate",
        }
    }
}

/// The tag prefix for the strategy (the legacy tag channel, matching
/// `compaction:on/off` in [`crate::compaction_override`]).
pub const COMPACTION_STRATEGY_TAG_PREFIX: &str = "compaction.strategy:";

/// The property key (the catalog-properties channel, matching
/// `cache.index.mode` in the cache policy).
pub const COMPACTION_STRATEGY_PROPERTY_KEY: &str = "compaction.strategy";

/// The property key for the target file size (MB).
pub const COMPACTION_TARGET_FILE_SIZE_PROPERTY_KEY: &str = "compaction.target_file_size_mb";

/// Parse a strategy from its string form (case-insensitive).
pub fn parse_strategy(value: &str) -> Option<CompactionStrategy> {
    match value.trim().to_ascii_lowercase().as_str() {
        "hybrid" => Some(CompactionStrategy::Hybrid),
        "consolidate" => Some(CompactionStrategy::Consolidate),
        _ => None,
    }
}

/// The tag channel's explicit value, if any: `Some` means an operator
/// chose a strategy; `None` means the channel is silent and the next layer
/// down decides (see [`resolve_effective_strategy`]). Last valid wins.
/// Single-channel resolution is deliberately NOT exported: a caller
/// resolving only tags silently misses the property/server layers (the
/// "property set but inert" bug class).
pub(crate) fn resolve_strategy_explicit_from_tags(tags: &[String]) -> Option<CompactionStrategy> {
    let mut explicit = None;
    for tag in tags {
        if let Some(parsed) = tag
            .trim()
            .strip_prefix(COMPACTION_STRATEGY_TAG_PREFIX)
            .and_then(parse_strategy)
        {
            explicit = Some(parsed);
        }
    }
    explicit
}

/// The property channel's explicit value, if any. Deterministic under
/// duplicate spellings: the canonical key wins; among non-canonical
/// variants the lexicographically smallest key wins (never HashMap
/// iteration order). Single-channel resolution is deliberately NOT
/// exported — see [`resolve_strategy_explicit_from_tags`].
pub(crate) fn resolve_strategy_explicit_from_properties(
    properties: &HashMap<String, String>,
) -> Option<CompactionStrategy> {
    // A key that is PRESENT but unparseable is an explicit (if broken)
    // operator choice, so it must not fall through and inherit a lower
    // layer that might say `consolidate`. It fails safe to the code
    // default instead. (`ALTER TABLE ... SET` and `CREATE TABLE ... WITH`
    // validate; REST create/PATCH on this branch persist no properties at
    // all. This covers a catalog written before validation existed.)
    if let Some(value) = properties.get(COMPACTION_STRATEGY_PROPERTY_KEY) {
        return Some(parse_strategy(value).unwrap_or_default());
    }
    properties
        .iter()
        .filter(|(key, _)| key.trim().to_ascii_lowercase() == COMPACTION_STRATEGY_PROPERTY_KEY)
        .min_by_key(|(key, _)| key.as_str())
        .map(|(_, value)| parse_strategy(value).unwrap_or_default())
}

/// The full precedence chain, as one pure function (unit-testable without
/// an engine): **collection property > operator tag > code default**.
///
/// There is NO server layer while background consolidate is refused
/// (TD-COMPACT-16): the parameter that used to sit here was `None` at every
/// production call site, and keeping it invited wiring a server default back
/// in without the convergence gate that TD requires. The rework re-adds the
/// layer together with the merge that honours it.
pub fn resolve_effective_strategy(
    properties: &HashMap<String, String>,
    tags: &[String],
) -> CompactionStrategy {
    resolve_strategy_explicit_from_properties(properties)
        .or_else(|| resolve_strategy_explicit_from_tags(tags))
        .unwrap_or_default()
}

/// Validate a `compaction.*` property at SET time. Returns `Err` with a
/// human-readable message for unknown keys or invalid values.
pub fn validate_compaction_properties(properties: &HashMap<String, String>) -> Result<(), String> {
    for (raw_key, raw_value) in properties {
        let key = raw_key.trim().to_ascii_lowercase();
        if !key.starts_with("compaction.") {
            continue;
        }
        match key.as_str() {
            COMPACTION_STRATEGY_PROPERTY_KEY => match parse_strategy(raw_value) {
                Some(CompactionStrategy::Hybrid) => {}
                // Refused (TD-COMPACT-16): the background consolidate trigger
                // scheduled an ordinary L0 morsel, not the consolidate merge,
                // so it could re-fire every flush without converging.
                Some(CompactionStrategy::Consolidate) => {
                    return Err(
                        "compaction.strategy = 'consolidate' is not supported: background consolidate does not converge (TD-COMPACT-16). Collapse with COMPACT TABLE ... FULL instead".to_string(),
                    );
                }
                None => {
                    return Err(format!(
                        "compaction.strategy: expected 'hybrid', got '{raw_value}'"
                    ));
                }
            },
            COMPACTION_TARGET_FILE_SIZE_PROPERTY_KEY => {
                // REFUSED until the merge consumes it (#2418 review round
                // 7). Neither the consolidate trigger (which must agree with
                // the merge, see flush/mod.rs) nor the FULL merge geometry
                // reads it, so accepting it at SET would be exactly the
                // accepted-but-inert class this door exists to close. The
                // same carrier-pending refusal `probe.*` uses.
                return Err(
                    "compaction.target_file_size_mb is not consumed yet (the merge geometry does not read per-collection properties; TD-COMPACT-15 open slice). Set [storage.compaction_config] consolidate_target_file_size_mb on the server instead".to_string(),
                );
            }
            _ => {
                return Err(format!(
                    "unknown compaction property '{raw_key}': the only per-collection compaction property consumed today is compaction.strategy"
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
    fn default_is_hybrid() {
        assert_eq!(CompactionStrategy::default(), CompactionStrategy::Hybrid);
        assert_eq!(resolve_strategy_explicit_from_tags(&[]), None);
        assert_eq!(
            resolve_strategy_explicit_from_properties(&HashMap::new()),
            None
        );
    }

    #[test]
    fn tag_resolution_last_wins() {
        let tags = vec![
            "compaction.strategy:hybrid".to_string(),
            "compaction.strategy:consolidate".to_string(),
        ];
        assert_eq!(
            resolve_strategy_explicit_from_tags(&tags),
            Some(CompactionStrategy::Consolidate)
        );
        // Invalid value keeps the prior
        let tags = vec![
            "compaction.strategy:consolidate".to_string(),
            "compaction.strategy:garbage".to_string(),
        ];
        assert_eq!(
            resolve_strategy_explicit_from_tags(&tags),
            Some(CompactionStrategy::Consolidate)
        );
    }

    #[test]
    fn property_resolution_case_insensitive() {
        let props = props(&[("Compaction.Strategy", "CONSOLIDATE")]);
        assert_eq!(
            resolve_strategy_explicit_from_properties(&props),
            Some(CompactionStrategy::Consolidate)
        );
    }

    #[test]
    fn two_non_canonical_spellings_resolve_deterministically() {
        // Both non-canonical: the lexicographically smallest key wins,
        // never HashMap iteration order.
        for _ in 0..20 {
            let properties = props(&[
                ("COMPACTION.STRATEGY", "consolidate"),
                ("Compaction.Strategy", "hybrid"),
            ]);
            // "COMPACTION.STRATEGY" < "Compaction.Strategy" (ASCII 'O' < 'o')
            assert_eq!(
                resolve_strategy_explicit_from_properties(&properties),
                Some(CompactionStrategy::Consolidate)
            );
        }
    }

    #[test]
    fn validation_rejects_unknown_and_invalid() {
        assert!(
            validate_compaction_properties(&props(&[("compaction.strategy", "hybrid")])).is_ok()
        );
        // Background consolidate is refused, pointing at the path that works.
        let refusal =
            validate_compaction_properties(&props(&[("compaction.strategy", "consolidate")]))
                .expect_err("background consolidate must be refused (TD-COMPACT-16)");
        assert!(refusal.contains("COMPACT TABLE"), "{refusal}");
        // Refused until the merge consumes it: accepting it would be inert.
        let refusal =
            validate_compaction_properties(&props(&[("compaction.target_file_size_mb", "1024")]))
                .expect_err("an unconsumed property must be refused at the door");
        assert!(refusal.contains("not consumed yet"), "{refusal}");
        assert!(
            validate_compaction_properties(&props(&[("compaction.strategy", "always")])).is_err()
        );
        assert!(
            validate_compaction_properties(&props(&[("compaction.target_file_size_mb", "zero")]))
                .is_err()
        );
        assert!(
            validate_compaction_properties(&props(&[("compaction.unknown_key", "value")])).is_err()
        );
        // Non-compaction keys are ignored
        assert!(validate_compaction_properties(&props(&[("cache.index.mode", "pin")])).is_ok());
    }

    #[test]
    fn effective_strategy_chain_property_over_tag_over_code() {
        let consolidate_tag = vec!["compaction.strategy:consolidate".to_string()];
        let hybrid_tag = vec!["compaction.strategy:hybrid".to_string()];
        // Property > tag.
        assert_eq!(
            resolve_effective_strategy(
                &props(&[("compaction.strategy", "hybrid")]),
                &consolidate_tag
            ),
            CompactionStrategy::Hybrid
        );
        // Tag > code.
        assert_eq!(
            resolve_effective_strategy(&HashMap::new(), &consolidate_tag),
            CompactionStrategy::Consolidate
        );
        // An explicit hybrid tag wins over nothing below it (there is no
        // server layer while background consolidate is refused,
        // TD-COMPACT-16).
        assert_eq!(
            resolve_effective_strategy(&HashMap::new(), &hybrid_tag),
            CompactionStrategy::Hybrid
        );
        // Code default.
        assert_eq!(
            resolve_effective_strategy(&HashMap::new(), &[]),
            CompactionStrategy::Hybrid
        );
    }

    #[test]
    fn invalid_explicit_property_fails_safe_instead_of_inheriting() {
        // #2418 review F7: a present-but-unparseable property is an explicit
        // (broken) choice. It must NOT fall through to a tag layer that says
        // `consolidate` — it resolves to the safe code default. (There is no
        // server layer while background consolidate is refused,
        // TD-COMPACT-16.)
        let consolidate_tag = vec!["compaction.strategy:consolidate".to_string()];
        assert_eq!(
            resolve_effective_strategy(
                &props(&[("compaction.strategy", "consolidat")]),
                &consolidate_tag,
            ),
            CompactionStrategy::Hybrid
        );
        // Same for a non-canonical spelling carrying junk.
        assert_eq!(
            resolve_effective_strategy(
                &props(&[("Compaction.Strategy", "garbage")]),
                &consolidate_tag,
            ),
            CompactionStrategy::Hybrid
        );
        // An ABSENT property still falls through, as designed.
        assert_eq!(
            resolve_effective_strategy(&HashMap::new(), &consolidate_tag),
            CompactionStrategy::Consolidate
        );
    }

    #[test]
    fn explicit_resolvers_distinguish_absent_from_hybrid() {
        assert_eq!(resolve_strategy_explicit_from_tags(&[]), None);
        assert_eq!(
            resolve_strategy_explicit_from_tags(&["compaction.strategy:hybrid".into()]),
            Some(CompactionStrategy::Hybrid)
        );
        assert_eq!(
            resolve_strategy_explicit_from_properties(&HashMap::new()),
            None
        );
        assert_eq!(
            resolve_strategy_explicit_from_properties(&props(&[("compaction.strategy", "hybrid")])),
            Some(CompactionStrategy::Hybrid)
        );
    }

    #[test]
    fn case_duplicate_properties_resolve_deterministically() {
        // The canonical spelling wins over a case variant regardless of
        // HashMap iteration order.
        for _ in 0..20 {
            let properties = props(&[
                ("Compaction.Strategy", "hybrid"),
                ("compaction.strategy", "consolidate"),
            ]);
            assert_eq!(
                resolve_strategy_explicit_from_properties(&properties),
                Some(CompactionStrategy::Consolidate)
            );
        }
    }
}
