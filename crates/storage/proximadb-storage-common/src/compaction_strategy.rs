//! TD-COMPACT-15: the per-collection compaction strategy.
//!
//! Two strategies exist:
//! - **Hybrid** (default): the current leveled policy — fill each level,
//!   promote upward. Best for write-heavy collections.
//! - **Consolidate**: merge smallest-first across ALL levels toward a
//!   `target_file_size_mb`, producing uniform segments. Best for read-heavy
//!   or batch-loaded collections.
//!
//! The strategy is resolved through the same chain as the cache policy
//! (TD-COMPACT-2b): collection property > server TOML > code default.
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
    if let Some(value) = properties.get(COMPACTION_STRATEGY_PROPERTY_KEY) {
        return parse_strategy(value);
    }
    properties
        .iter()
        .filter(|(key, _)| key.trim().to_ascii_lowercase() == COMPACTION_STRATEGY_PROPERTY_KEY)
        .min_by_key(|(key, _)| key.as_str())
        .and_then(|(_, value)| parse_strategy(value))
}

/// The full precedence chain, as one pure function (unit-testable without
/// an engine): **collection property > operator tag > server TOML > code**.
/// An explicit `hybrid` at any layer WINS over a consolidate default at a
/// lower layer — silence falls through, an explicit choice does not.
pub fn resolve_effective_strategy(
    properties: &HashMap<String, String>,
    tags: &[String],
    server_consolidate_strategy: Option<&str>,
) -> CompactionStrategy {
    resolve_strategy_explicit_from_properties(properties)
        .or_else(|| resolve_strategy_explicit_from_tags(tags))
        .or_else(|| {
            server_consolidate_strategy
                .and_then(parse_strategy)
                .filter(|s| *s == CompactionStrategy::Consolidate)
        })
        .unwrap_or_default()
}

/// Resolve the target file size (MB) from properties, falling back to the
/// server default. Property: `compaction.target_file_size_mb = "1024"`.
/// Deterministic under case-duplicate keys (canonical spelling first), for
/// the same reason as [`resolve_strategy_explicit_from_properties`].
pub fn resolve_target_file_size_mb(
    properties: &HashMap<String, String>,
    server_default_mb: u64,
) -> u64 {
    let parse_valid =
        |v: &str| -> Option<u64> { v.trim().parse::<u64>().ok().filter(|mb| *mb > 0) };
    if let Some(value) = properties.get(COMPACTION_TARGET_FILE_SIZE_PROPERTY_KEY) {
        return parse_valid(value).unwrap_or(server_default_mb);
    }
    properties
        .iter()
        .filter(|(key, _)| {
            key.trim().to_ascii_lowercase() == COMPACTION_TARGET_FILE_SIZE_PROPERTY_KEY
        })
        .min_by_key(|(key, _)| key.as_str())
        .and_then(|(_, value)| parse_valid(value))
        .unwrap_or(server_default_mb)
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
            COMPACTION_STRATEGY_PROPERTY_KEY => {
                if parse_strategy(raw_value).is_none() {
                    return Err(format!(
                        "compaction.strategy: expected 'hybrid' or 'consolidate', got '{raw_value}'"
                    ));
                }
            }
            COMPACTION_TARGET_FILE_SIZE_PROPERTY_KEY => {
                let mb = raw_value.trim().parse::<u64>().unwrap_or(0);
                if mb == 0 {
                    return Err(format!(
                        "compaction.target_file_size_mb: expected a positive integer, got '{raw_value}'"
                    ));
                }
            }
            _ => {
                return Err(format!(
                    "unknown compaction property '{raw_key}': expected compaction.strategy or compaction.target_file_size_mb"
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
            let target = props(&[
                ("Compaction.Target_File_Size_MB", "256"),
                ("COMPACTION.TARGET_FILE_SIZE_MB", "1024"),
            ]);
            assert_eq!(resolve_target_file_size_mb(&target, 512), 1024);
        }
    }

    #[test]
    fn target_file_size_resolves_with_fallback() {
        assert_eq!(resolve_target_file_size_mb(&HashMap::new(), 512), 512);
        assert_eq!(
            resolve_target_file_size_mb(&props(&[("compaction.target_file_size_mb", "1024")]), 512),
            1024
        );
        // Invalid falls back
        assert_eq!(
            resolve_target_file_size_mb(&props(&[("compaction.target_file_size_mb", "0")]), 512),
            512
        );
    }

    #[test]
    fn validation_rejects_unknown_and_invalid() {
        assert!(
            validate_compaction_properties(&props(&[("compaction.strategy", "consolidate")]))
                .is_ok()
        );
        assert!(
            validate_compaction_properties(&props(&[("compaction.target_file_size_mb", "1024")]))
                .is_ok()
        );
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
    fn effective_strategy_chain_property_tag_server_code() {
        let hybrid_tag = vec!["compaction.strategy:hybrid".to_string()];
        let consolidate_tag = vec!["compaction.strategy:consolidate".to_string()];
        // Property wins over tag and server.
        assert_eq!(
            resolve_effective_strategy(
                &props(&[("compaction.strategy", "consolidate")]),
                &hybrid_tag,
                Some("hybrid"),
            ),
            CompactionStrategy::Consolidate
        );
        // Explicit tag wins over server (and absent property).
        assert_eq!(
            resolve_effective_strategy(&HashMap::new(), &consolidate_tag, Some("hybrid")),
            CompactionStrategy::Consolidate
        );
        // An EXPLICIT hybrid at a higher layer beats a consolidate default
        // below it — silence falls through, an explicit choice does not.
        assert_eq!(
            resolve_effective_strategy(&HashMap::new(), &hybrid_tag, Some("consolidate")),
            CompactionStrategy::Hybrid
        );
        assert_eq!(
            resolve_effective_strategy(
                &props(&[("compaction.strategy", "hybrid")]),
                &consolidate_tag,
                Some("consolidate"),
            ),
            CompactionStrategy::Hybrid
        );
        // Server default applies only when both collection layers are silent.
        assert_eq!(
            resolve_effective_strategy(&HashMap::new(), &[], Some("consolidate")),
            CompactionStrategy::Consolidate
        );
        // Code default last.
        assert_eq!(
            resolve_effective_strategy(&HashMap::new(), &[], None),
            CompactionStrategy::Hybrid
        );
        // A junk server value is inert (config load rejects it, but the
        // resolver must not invent a strategy).
        assert_eq!(
            resolve_effective_strategy(&HashMap::new(), &[], Some("aggressive")),
            CompactionStrategy::Hybrid
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

    #[test]
    fn target_deterministic_under_case_duplicates() {
        for _ in 0..20 {
            let properties = props(&[
                ("Compaction.Target_File_Size_MB", "256"),
                ("compaction.target_file_size_mb", "1024"),
            ]);
            assert_eq!(resolve_target_file_size_mb(&properties, 512), 1024);
        }
    }
}
