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

/// The policy table-property namespaces. Keys under these are validated by
/// [`validate_policy_properties`] and resolved CASE- AND
/// WHITESPACE-insensitively (`trim().to_ascii_lowercase()`), so the catalog
/// imports this list for its SET apply instead of keeping its own copy. A
/// fourth family is added HERE and every consumer sees it.
pub const POLICY_PROPERTY_PREFIXES: [&str; 3] = ["cache.", "compaction.", "probe."];

/// The case-insensitive property lookup every knob resolver uses: the
/// FIRST key (by raw byte order) among case/whitespace variants wins, so
/// resolution is deterministic under HashMap iteration order. ONE
/// implementation — hand-copied filter/min_by_key chains could drift
/// (#2419 review round 19).
pub fn policy_value<'a>(properties: &'a HashMap<String, String>, key: &str) -> Option<&'a String> {
    properties
        .iter()
        .filter(|(k, _)| normalize_policy_key(k) == key)
        .min_by_key(|(k, _)| k.as_str())
        .map(|(_, v)| v)
}

/// Whether `key` names a ProximaDB POLICY property — the `cache.`,
/// `compaction.` or `probe.` namespaces this crate validates.
///
/// Used by the Iceberg commit door to refuse exactly these and nothing else.
/// It must NOT be confused with `is_reserved_table_property` (the
/// system-WRITTEN keys): the Iceberg engine's own namespaces (`write.*`,
/// `format-version`, `commit.*`) are neither, and must pass through
/// untouched or standard Spark CTAS breaks (#2419 review round 23).
pub fn is_policy_property(key: &str) -> bool {
    let normalized = normalize_policy_key(key);
    POLICY_PROPERTY_PREFIXES
        .iter()
        .any(|prefix| normalized.starts_with(prefix))
}

/// The normalized form every policy resolver keys on.
pub fn normalize_policy_key(key: &str) -> String {
    key.trim().to_ascii_lowercase()
}

/// Validate every policy-property family at a write boundary (create, SET,
/// PATCH). Each family's validator ignores keys outside its own prefix, so
/// composition is order-independent and unknown non-policy keys pass through
/// untouched (they are stored verbatim by design).
///
/// Returns the first family error, already phrased for an operator.
pub fn validate_policy_properties(properties: &HashMap<String, String>) -> Result<(), String> {
    // One map may not name the same policy knob twice. The resolvers key on
    // `normalize_policy_key`, so 'compaction.strategy' and
    // 'Compaction.Strategy ' are ONE knob; storing both left the effective
    // value to HashMap iteration order (per-process hash seed), which can
    // flip between restarts or replicas (#2419 review round 10). Only the
    // cache validator caught this before; it is enforced here for every
    // policy family at once.
    let mut seen: std::collections::HashMap<String, &str> = std::collections::HashMap::new();
    for key in properties.keys() {
        let normalized = normalize_policy_key(key);
        if !POLICY_PROPERTY_PREFIXES
            .iter()
            .any(|prefix| normalized.starts_with(prefix))
        {
            continue;
        }
        if let Some(previous) = seen.insert(normalized.clone(), key.as_str()) {
            return Err(format!(
                "duplicate policy property: '{previous}' and '{key}' name the same knob ({normalized}); keep one"
            ));
        }
    }
    // The cache family carries a typed error; the other two are already
    // operator-phrased strings. Normalise to one string so every door gets
    // the same shape.
    proximadb_cache::policy::validate_cache_properties(properties)
        .map_err(|error| error.to_string())?;
    crate::compaction_strategy::validate_compaction_properties(properties)?;
    // `probe.*` is validated for SHAPE (so the carrier work inherits a
    // correct validator) but REFUSED at the door, because nothing consumes
    // the property channel yet: `resolve_probe_policy` is only reached from
    // the process-global settings path, which passes no properties. Storing
    // a key that cannot take effect is the accepted-but-inert bug this
    // boundary exists to prevent — so until the per-collection carrier
    // lands (TD-RDSTRAT-14 "Deferred slices"), the honest answer to
    // `SET ('probe.algorithm', …)` is a refusal that names the knob that
    // does work, not a silent success.
    crate::probe_policy::validate_probe_properties(properties)?;
    if let Some((key, _)) = properties
        .iter()
        // `probe.*` is a RESERVED policy namespace, like `cache.*` (whose
        // validator likewise rejects unknown keys in it — see
        // proximadb-cache/src/policy.rs). Rejecting an unrecognised
        // `probe.algoritm` beats storing it inertly, which is the
        // accepted-but-inert class this door exists to close. The #2419
        // review raised that this narrows what the REST `properties` doc
        // comment promises; the doc is corrected rather than the guard
        // loosened.
        .find(|(key, _)| {
            // The probe family's prefix from the ONE list (index 2); a
            // second "probe." literal could drift from it.
            normalize_policy_key(key).starts_with(POLICY_PROPERTY_PREFIXES[2])
        })
    {
        return Err(format!(
            "{key}: per-collection probe properties are not consumed yet (TD-RDSTRAT-14 deferred the search-path carrier). Configure the probe policy with [storage.sst_config.coarse_probe] or PROXIMADB_PAX_READ_PROBE_* until it lands"
        ));
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
    fn every_family_is_reachable_through_the_one_door() {
        // A typo in ANY family must be rejected by the single call — this is
        // the regression that the per-door wiring kept missing.
        assert!(validate_policy_properties(&props(&[("cache.index.mode", "pinn")])).is_err());
        assert!(
            validate_policy_properties(&props(&[("compaction.strategy", "consolidat")])).is_err()
        );
        assert!(validate_policy_properties(&props(&[("probe.algorithm", "sqrt2")])).is_err());
        // NOTE: `compaction.strategy = "consolidate"` is deliberately NOT
        // asserted here — #2418 refuses it at this same door (TD-COMPACT-16),
        // and an assertion for either behavior breaks whichever PR merges
        // second. End-state (both merged): refused.
        // Even a SHAPE-VALID probe property is refused while the channel is
        // unconsumed — accepting it would store an inert key.
        // `probe.*` is reserved: an unrecognised key in it fails closed,
        // matching `cache.*`.
        assert!(validate_policy_properties(&props(&[("probe.owner", "team-search")])).is_err());
        let refusal = validate_policy_properties(&props(&[("probe.algorithm", "cbrt")]))
            .expect_err("probe.* must be refused until the carrier lands");
        assert!(refusal.contains("not consumed yet"), "{refusal}");
        assert!(
            validate_policy_properties(&props(&[("compaction.target_file_size_mb", "0")])).is_err()
        );
        assert!(validate_policy_properties(&props(&[("probe.base_mb", "nope")])).is_err());
        assert!(validate_policy_properties(&props(&[("probe.base_mb", "8")])).is_err());
    }

    #[test]
    fn variants_of_one_policy_knob_are_refused_in_a_single_map() {
        for (a, b) in [
            ("compaction.strategy", "Compaction.Strategy"),
            ("cache.index.mode", " cache.index.mode"),
            ("probe.algorithm", "PROBE.ALGORITHM"),
        ] {
            let error = validate_policy_properties(&props(&[(a, "hybrid"), (b, "hybrid")]))
                .expect_err("two spellings of one knob must be refused");
            assert!(error.contains("duplicate policy property"), "{error}");
        }
        // Non-policy keys keep exact-key semantics.
        assert!(validate_policy_properties(&props(&[("team", "a"), ("Team", "b")])).is_ok());
    }

    #[test]
    fn valid_properties_and_unknown_namespaces_pass() {
        assert!(
            validate_policy_properties(&props(&[
                ("cache.index.mode", "pin"), // Aligns with #2418 (TD-COMPACT-16): background consolidate is refused
                // at every door — this test must hold on develop with BOTH PRs
                // merged, whichever lands second.
                ("compaction.strategy", "hybrid"),
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
    #[test]
    fn policy_predicate_excludes_iceberg_engine_namespaces() {
        // The round-22 refusal assumed `write.*` was stripped as a reserved
        // key. It is not reserved, so it reached the bail and 400'd standard
        // Spark CTAS. These must NOT be policy properties.
        for key in [
            "write.parquet.compression-codec",
            "write.format.default",
            "format-version",
            "commit.retry.num-retries",
            "owner",
            "sort-order",
        ] {
            assert!(!is_policy_property(key), "{key} must pass the commit door");
        }
        // These must be refused there.
        for key in [
            "cache.index.mode",
            "compaction.strategy",
            "probe.algorithm",
            "  Probe.Algorithm  ",
        ] {
            assert!(is_policy_property(key), "{key} must be refused");
        }
    }
}
