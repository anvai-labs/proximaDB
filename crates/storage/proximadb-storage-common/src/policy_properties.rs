//! The ONE validation boundary for per-collection policy properties.
//!
//! Three policy families now ride the Hive-TBLPROPERTIES channel —
//! `cache.*` (TD-COMPACT-2b), `compaction.*` (TD-COMPACT-15) and `probe.*`
//! (TD-RDSTRAT-14) — and each shipped with its own validator wired into
//! whichever door its PR touched. The result drifted:
//!
//! | door | cache.* | compaction.* | probe.* |
//! |---|---|---|---|
//! | `ALTER TABLE … SET` | yes | yes | yes |
//! | `PATCH /collections/{id}/properties` | yes | **no** | **no** |
//! | `POST /collections` (create) | **no** (its doc claimed yes) | **no** | **no** |
//!
//! So `PATCH {"compaction.strategy": "typo"}` was accepted and silently did
//! nothing, while the same key through SQL was rejected — the
//! accepted-but-inert class these validators exist to prevent, reintroduced
//! by having three of them and three doors.
//!
//! [`validate_policy_properties`] is the single function every door calls.
//! Adding a fourth family means adding one line HERE, and every door gets it
//! at once — the property cannot be half-wired again.

use std::collections::HashMap;

/// Validate every policy-property family at a write boundary (create, SET,
/// PATCH). Each family's validator ignores keys outside its own prefix, so
/// composition is order-independent and unknown non-policy keys pass through
/// untouched (they are stored verbatim by design).
///
/// Returns the first family error, already phrased for an operator.
pub fn validate_policy_properties(properties: &HashMap<String, String>) -> Result<(), String> {
    // The cache family carries a typed error; the other two are already
    // operator-phrased strings. Normalise to one string so every door gets
    // the same shape.
    proximadb_cache::policy::validate_cache_properties(properties)
        .map_err(|error| error.to_string())?;
    crate::compaction_strategy::validate_compaction_properties(properties)?;
    crate::probe_policy::validate_probe_properties(properties)?;
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
    fn every_family_is_reachable_through_the_one_door() {
        // A typo in ANY family must be rejected by the single call — this is
        // the regression that the per-door wiring kept missing.
        assert!(validate_policy_properties(&props(&[("cache.index.mode", "pinn")])).is_err());
        assert!(
            validate_policy_properties(&props(&[("compaction.strategy", "consolidat")])).is_err()
        );
        assert!(validate_policy_properties(&props(&[("probe.algorithm", "sqrt2")])).is_err());
        assert!(
            validate_policy_properties(&props(&[("compaction.target_file_size_mb", "0")])).is_err()
        );
        assert!(validate_policy_properties(&props(&[("probe.base_mb", "nope")])).is_err());
    }

    #[test]
    fn valid_properties_and_unknown_namespaces_pass() {
        assert!(
            validate_policy_properties(&props(&[
                ("cache.index.mode", "pin"),
                ("compaction.strategy", "consolidate"),
                ("compaction.target_file_size_mb", "1024"),
                ("probe.algorithm", "byte_budget"),
                ("probe.base_mb", "8"),
            ]))
            .is_ok()
        );
        // Non-policy keys are stored verbatim, never rejected.
        assert!(
            validate_policy_properties(&props(&[
                ("owner", "team-search"),
                ("my.custom.key", "whatever"),
            ]))
            .is_ok()
        );
    }

    #[test]
    fn unknown_key_inside_a_policy_namespace_is_still_rejected() {
        // The families fail closed on their OWN namespace — a new key must be
        // registered, not silently stored.
        assert!(validate_policy_properties(&props(&[("probe.nonsense", "1")])).is_err());
        assert!(validate_policy_properties(&props(&[("compaction.nonsense", "1")])).is_err());
    }
}
