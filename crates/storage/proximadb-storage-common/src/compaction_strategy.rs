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

/// Resolve from collection TAGS (the legacy channel: last valid wins,
/// matching `resolve_compaction_armed` semantics). Tag:
/// `compaction.strategy:consolidate`
pub fn resolve_strategy_from_tags(tags: &[String]) -> CompactionStrategy {
    let mut strategy = CompactionStrategy::default();
    for tag in tags {
        if let Some(parsed) = tag
            .trim()
            .strip_prefix(COMPACTION_STRATEGY_TAG_PREFIX)
            .and_then(parse_strategy)
        {
            strategy = parsed;
        }
    }
    strategy
}

/// Resolve from catalog PROPERTIES (the property channel). Property:
/// `compaction.strategy = "consolidate"`
pub fn resolve_strategy_from_properties(
    properties: &HashMap<String, String>,
) -> CompactionStrategy {
    properties
        .iter()
        .find(|(k, _)| k.trim().to_ascii_lowercase() == COMPACTION_STRATEGY_PROPERTY_KEY)
        .and_then(|(_, v)| parse_strategy(v))
        .unwrap_or_default()
}

/// Resolve the target file size (MB) from properties, falling back to the
/// server default. Property: `compaction.target_file_size_mb = "1024"`
pub fn resolve_target_file_size_mb(
    properties: &HashMap<String, String>,
    server_default_mb: u64,
) -> u64 {
    properties
        .iter()
        .find(|(k, _)| k.trim().to_ascii_lowercase() == COMPACTION_TARGET_FILE_SIZE_PROPERTY_KEY)
        .and_then(|(_, v)| v.trim().parse::<u64>().ok())
        .filter(|mb| *mb > 0)
        .unwrap_or(server_default_mb)
}

/// The canonical tag form of the target-file-size property (last-wins
/// resolution, mirroring [`COMPACTION_STRATEGY_TAG_PREFIX`]).
pub const COMPACTION_TARGET_FILE_SIZE_TAG_PREFIX: &str = "compaction.target_file_size_mb:";

/// Resolve the target file size (MB) from the TAG channel, falling back to
/// the server default. Last valid occurrence wins, mirroring
/// [`resolve_strategy_from_tags`] — the property bridge appends these tags
/// AFTER operator tags so the property is authoritative.
pub fn resolve_target_file_size_mb_from_tags(tags: &[String], server_default_mb: u64) -> u64 {
    let mut resolved = server_default_mb;
    for tag in tags {
        if let Some(parsed) = tag
            .trim()
            .strip_prefix(COMPACTION_TARGET_FILE_SIZE_TAG_PREFIX)
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|mb| *mb > 0)
        {
            resolved = parsed;
        }
    }
    resolved
}

/// Project collection POLICY PROPERTIES into canonical tags, appended LAST
/// so last-wins resolution makes the property authoritative over operator
/// tags. This is the property→tag bridge at collection load
/// (`collection_from_catalog_schema`): the derived tags are never persisted
/// back — they are re-derived from the catalog properties on every load, so
/// there is no staleness channel and no invalidation hook to forget.
///
/// Only known `compaction.*` keys project (case-insensitive). Values that
/// fail validation are skipped defensively — they cannot exist past
/// SET-time validation, but a hand-edited catalog must not wedge the flush
/// path.
pub fn project_policy_properties_to_tags(
    properties: &HashMap<String, String>,
    tags: &mut Vec<String>,
) {
    for (raw_key, raw_value) in properties {
        let key = raw_key.trim().to_ascii_lowercase();
        match key.as_str() {
            COMPACTION_STRATEGY_PROPERTY_KEY => {
                if parse_strategy(raw_value).is_some() {
                    let canonical = raw_value.trim().to_ascii_lowercase();
                    tags.push(format!("{COMPACTION_STRATEGY_TAG_PREFIX}{canonical}"));
                }
            }
            COMPACTION_TARGET_FILE_SIZE_PROPERTY_KEY => {
                let trimmed = raw_value.trim();
                if trimmed.parse::<u64>().map(|mb| mb > 0).unwrap_or(false) {
                    tags.push(format!("{COMPACTION_TARGET_FILE_SIZE_TAG_PREFIX}{trimmed}"));
                }
            }
            _ => {}
        }
    }
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
        assert_eq!(resolve_strategy_from_tags(&[]), CompactionStrategy::Hybrid);
        assert_eq!(
            resolve_strategy_from_properties(&HashMap::new()),
            CompactionStrategy::Hybrid
        );
    }

    #[test]
    fn tag_resolution_last_wins() {
        let tags = vec![
            "compaction.strategy:hybrid".to_string(),
            "compaction.strategy:consolidate".to_string(),
        ];
        assert_eq!(
            resolve_strategy_from_tags(&tags),
            CompactionStrategy::Consolidate
        );
        // Invalid value keeps the prior
        let tags = vec![
            "compaction.strategy:consolidate".to_string(),
            "compaction.strategy:garbage".to_string(),
        ];
        assert_eq!(
            resolve_strategy_from_tags(&tags),
            CompactionStrategy::Consolidate
        );
    }

    #[test]
    fn property_resolution_case_insensitive() {
        let props = props(&[("Compaction.Strategy", "CONSOLIDATE")]);
        assert_eq!(
            resolve_strategy_from_properties(&props),
            CompactionStrategy::Consolidate
        );
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
    fn bridge_projects_known_keys_as_canonical_tags() {
        let properties = props(&[
            ("Compaction.Strategy", "Consolidate"),
            ("compaction.target_file_size_mb", "1024"),
            ("compaction.unknown", "ignored"),
            ("cache.index.mode", "pin"),
        ]);
        let mut tags = vec!["compaction.strategy:hybrid".to_string()];
        project_policy_properties_to_tags(&properties, &mut tags);
        // Appended last: property wins over the operator tag (last-wins).
        assert_eq!(
            resolve_strategy_from_tags(&tags),
            CompactionStrategy::Consolidate
        );
        assert_eq!(resolve_target_file_size_mb_from_tags(&tags, 512), 1024);
        // Exactly one projected tag per known key; unknown keys inert.
        assert_eq!(tags.iter().filter(|t| t.contains("compaction.")).count(), 3);
    }

    #[test]
    fn bridge_skips_invalid_values_defensively() {
        let properties = props(&[
            ("compaction.strategy", "garbage"),
            ("compaction.target_file_size_mb", "0"),
            ("compaction.target_file_size_mb", "not-a-number"),
        ]);
        let mut tags = vec![];
        project_policy_properties_to_tags(&properties, &mut tags);
        assert!(tags.is_empty());
    }

    #[test]
    fn target_from_tags_last_wins_and_falls_back() {
        assert_eq!(resolve_target_file_size_mb_from_tags(&[], 512), 512);
        let tags = vec![
            "compaction.target_file_size_mb:256".to_string(),
            "compaction.target_file_size_mb:2048".to_string(),
            "compaction.target_file_size_mb:0".to_string(), // invalid — prior kept
        ];
        assert_eq!(resolve_target_file_size_mb_from_tags(&tags, 512), 2048);
    }
}
