//! TD-TIER-1 SC1/SC2: desired storage-class vocabulary, admission validation,
//! and the read-side status projection.
//!
//! One vocabulary shared by every facade: REST v2 parses user strings here,
//! gRPC/Flight/SQL reach the manager with raw proto enum numbers, and the
//! manager validates those numbers through [`validate_storage_class_raw`] so
//! an unknown value is rejected at admission on EVERY path (fail-closed), not
//! silently coerced at first use. Absent/`UNSPECIFIED` resolves to
//! [`StorageClass::Balanced`] — every pre-SC1 record decodes and serves
//! unchanged.
//!
//! v1 status truth (SC1): BALANCED is ACTIVE; ECONOMY and PERFORMANCE are
//! accepted-but-`NOT_ACTIVE` — their residency/reservation dependencies
//! (SC3/SC4) are not implemented, so the collection keeps serving with
//! balanced behavior and [`storage_class_status`] names the missing
//! dependency in the reason. The status is a pure projection of the desired
//! policy: never an input, never a claimed reservation.

use crate::proto::proximadb_v1::{
    CollectionConfig, StorageClass, StorageClassStatus, StorageEngine, storage_class_status::State,
};

/// The user-facing storage-class labels (REST v2 accepts exactly these,
/// case-insensitively, plus the proto enum names for parity with the
/// gRPC/Flight surfaces).
pub const STORAGE_CLASS_LABELS: [&str; 3] = ["economy", "balanced", "performance"];

/// Parse a user-supplied storage-class string (REST v2 `storage_class`).
///
/// Accepts the labels in [`STORAGE_CLASS_LABELS`] case-insensitively and the
/// proto enum names (`STORAGE_CLASS_ECONOMY`, …); an empty string is
/// `UNSPECIFIED` (⇒ balanced). Anything else is an error naming the valid
/// values — a misspelt class used to be indistinguishable from a silent
/// default (the TD-RESTGET-1 failure mode).
pub fn parse_storage_class(raw: &str) -> anyhow::Result<StorageClass> {
    let key = raw.trim().to_ascii_lowercase();
    if key.is_empty() {
        return Ok(StorageClass::Unspecified);
    }
    match key.as_str() {
        "unspecified" => Ok(StorageClass::Unspecified),
        "economy" => Ok(StorageClass::Economy),
        "balanced" => Ok(StorageClass::Balanced),
        "performance" => Ok(StorageClass::Performance),
        other => {
            // Parity with the proto surface: "storage_class_economy" and the
            // raw enum name both resolve, mirroring how the embedding
            // precision labels are accepted.
            if let Some(stripped) = other.strip_prefix("storage_class_")
                && let Some(parsed) = StorageClass::from_str_name(&format!(
                    "STORAGE_CLASS_{}",
                    stripped.to_ascii_uppercase()
                ))
            {
                return Ok(parsed);
            }
            Err(anyhow::anyhow!(
                "Invalid storage_class '{}'. Valid values: {:?}",
                raw.trim(),
                STORAGE_CLASS_LABELS
            ))
        }
    }
}

/// Validate a raw proto enum number from any facade and resolve it.
///
/// `None`/`Some(0)` (absent field / `UNSPECIFIED`) ⇒ [`StorageClass::Balanced`]
/// for compatibility; `Some(1..=3)` ⇒ that class; anything else is rejected —
/// prost preserves unknown wire numbers in the raw `i32`, so an unrecognised
/// enum value from a newer or hand-rolled client reaches this check intact
/// instead of being silently collapsed to a default.
pub fn validate_storage_class_raw(raw: Option<i32>) -> anyhow::Result<StorageClass> {
    match raw {
        None | Some(0) => Ok(StorageClass::Balanced),
        Some(value) => StorageClass::try_from(value).map_err(|_| {
            anyhow::anyhow!(
                "Invalid storage_class value {value}. Valid values: economy (1), \
                 balanced (2, default), performance (3)"
            )
        }),
    }
}

/// The collection's effective desired class: the stored value with
/// absent/`UNSPECIFIED` resolved to balanced.
pub fn storage_class_or_balanced(config: &CollectionConfig) -> StorageClass {
    validate_storage_class_raw(config.storage_class).unwrap_or(StorageClass::Balanced)
}

/// Stable lowercase label for the class (REST/gRPC-v2 read models).
pub fn storage_class_label(class: StorageClass) -> &'static str {
    match class {
        StorageClass::Economy => "economy",
        StorageClass::Balanced => "balanced",
        StorageClass::Performance => "performance",
        StorageClass::Unspecified => "balanced",
    }
}

/// The read-side observed status: a pure projection of the desired policy,
/// truthful about what SC1 actually delivers.
///
/// BALANCED (or an absent field) ⇒ ACTIVE. ECONOMY/PERFORMANCE ⇒ NOT_ACTIVE
/// with a reason naming the missing dependency, because SC3 (reservations)
/// and SC4 (residency coverage) are not implemented — the collection serves
/// balanced behavior and no guarantee is implied. `policy_generation` is 0
/// for pre-SC1 records (field absent) and 1 once any class is persisted at
/// create; a policy-change surface will bump it.
pub fn storage_class_status(config: &CollectionConfig) -> StorageClassStatus {
    let class = storage_class_or_balanced(config);
    let generation = if config.storage_class.is_some() { 1 } else { 0 };
    match class {
        StorageClass::Balanced | StorageClass::Unspecified => StorageClassStatus {
            state: State::Active as i32,
            reason: String::new(),
            policy_generation: generation,
            reserved_bytes: 0,
            resident_bytes: 0,
            covered_bytes: 0,
        },
        StorageClass::Economy => StorageClassStatus {
            state: State::NotActive as i32,
            reason: "economy residency is not implemented (TD-TIER-1 SC3/SC4): the \
                     collection serves balanced behavior"
                .to_string(),
            policy_generation: generation,
            reserved_bytes: 0,
            resident_bytes: 0,
            covered_bytes: 0,
        },
        StorageClass::Performance => StorageClassStatus {
            state: State::NotActive as i32,
            reason: "performance residency requires an admitted reservation and \
                     coverage (TD-TIER-1 SC3/SC4), not implemented: the collection \
                     serves balanced behavior"
                .to_string(),
            policy_generation: generation,
            reserved_bytes: 0,
            resident_bytes: 0,
            covered_bytes: 0,
        },
    }
}

/// SC2 admission check for the PERFORMANCE class, evaluated on the final
/// RESOLVED placement (not the raw client request): the engine must be the
/// PAX path (SST) and the resolved base location — the client's explicit
/// `storage_config.storage_path`, or the operator-CONFIGURED storage location
/// the manager falls back to — must be object-store-backed, because the
/// residency contract SC4 will build presumes a durable remote segment copy.
///
/// Authorization: an explicit client-supplied path is checked against the
/// canonical tenant-root convention (`{base}/tenants/{tenant_id}`, the same
/// shape `StoragePath::tenant_root_path` writes) — a path under ANOTHER
/// tenant's prefix is rejected. Anything failing admission is rejected here
/// with an actionable message — never accepted and silently degraded later.
pub fn check_performance_admission(
    config: &CollectionConfig,
    resolved_engine: StorageEngine,
    resolved_base_location: &str,
    tenant_id: Option<&str>,
) -> anyhow::Result<()> {
    if storage_class_or_balanced(config) != StorageClass::Performance {
        return Ok(());
    }
    if resolved_engine != StorageEngine::Sst {
        return Err(anyhow::anyhow!(
            "storage_class 'performance' requires the SST/PAX storage engine \
             (resolved engine: {:?}). Recreate the collection on SST, or use the \
             'balanced' class",
            resolved_engine
        ));
    }
    // Cross-tenant authorization on an explicit path (configured defaults are
    // operator-authorized by construction and carry no tenant segment).
    if let (Some(current_tenant), Some(explicit)) = (
        tenant_id.filter(|t| !t.is_empty()),
        config
            .storage_config
            .as_ref()
            .and_then(|storage| storage.storage_path.as_deref())
            .filter(|p| !p.is_empty()),
    ) {
        if let Some(owner) = tenant_segment_of(explicit)
            && owner != current_tenant
        {
            return Err(anyhow::anyhow!(
                "storage_class 'performance' placement '{explicit}' is under another \
                 tenant's prefix (tenants/{owner}); requesting tenant is \
                 '{current_tenant}'. Use an operator-configured location or this \
                 tenant's own prefix"
            ));
        }
    }
    if proximadb_storage_filesystem_types::object_store_scheme::ObjectStoreScheme::of(
        resolved_base_location,
    )
    .is_none()
    {
        return Err(anyhow::anyhow!(
            "storage_class 'performance' requires an object-store-backed placement \
             (s3://, gs:// or az-adls://); the resolved location '{}' is not one. \
             Configure a remote storage location, or use the 'balanced' class",
            if resolved_base_location.is_empty() {
                "<unset>"
            } else {
                resolved_base_location
            }
        ));
    }
    Ok(())
}

/// The tenant id named by a `{base}/tenants/{id}` path segment, if any
/// (read-only parse of the same convention `StoragePath::tenant_root_path`
/// writes; never constructs keys).
///
/// Parses the OUTERMOST `/tenants/` occurrence: a nested path such as
/// `tenants/victim/collections/tenants/attacker` must resolve to `victim`
/// (the prefix's actual owner). Parsing the innermost occurrence would let a
/// requesting tenant authorize itself into another tenant's subtree. The cost
/// is a false reject when an operator's configured base itself ends in
/// `tenants` (a doubled segment) — fail-closed is the correct side of that
/// trade.
fn tenant_segment_of(path: &str) -> Option<&str> {
    let idx = path.find("/tenants/")? + "/tenants/".len();
    path[idx..]
        .split('/')
        .next()
        .filter(|segment| !segment.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::proximadb_v1::StorageConfig;

    fn config_with_class(raw: Option<i32>) -> CollectionConfig {
        CollectionConfig {
            storage_class: raw,
            ..Default::default()
        }
    }

    #[test]
    fn absent_or_unspecified_resolves_balanced() {
        assert_eq!(
            validate_storage_class_raw(None).unwrap(),
            StorageClass::Balanced
        );
        assert_eq!(
            validate_storage_class_raw(Some(0)).unwrap(),
            StorageClass::Balanced
        );
        assert_eq!(
            storage_class_or_balanced(&config_with_class(None)),
            StorageClass::Balanced
        );
    }

    #[test]
    fn unknown_raw_number_is_rejected_not_coerced() {
        let error = validate_storage_class_raw(Some(99))
            .unwrap_err()
            .to_string();
        assert!(error.contains("99"), "{error}");
        assert!(error.contains("performance"), "{error}");
        assert!(validate_storage_class_raw(Some(-1)).is_err());
    }

    #[test]
    fn labels_parse_case_insensitively_and_proto_names_parse() {
        assert_eq!(
            parse_storage_class("Performance").unwrap(),
            StorageClass::Performance
        );
        assert_eq!(
            parse_storage_class(" ECONOMY ").unwrap(),
            StorageClass::Economy
        );
        assert_eq!(
            parse_storage_class("storage_class_balanced").unwrap(),
            StorageClass::Balanced
        );
        assert_eq!(
            parse_storage_class("STORAGE_CLASS_PERFORMANCE").unwrap(),
            StorageClass::Performance
        );
        assert_eq!(parse_storage_class("").unwrap(), StorageClass::Unspecified);
        let error = parse_storage_class("perfromance").unwrap_err().to_string();
        assert!(error.contains("perfromance"), "{error}");
        assert!(error.contains("balanced"), "{error}");
    }

    #[test]
    fn balanced_reports_active_and_pre_sc1_records_report_generation_zero() {
        let status = storage_class_status(&config_with_class(None));
        assert_eq!(status.state, State::Active as i32);
        assert_eq!(status.policy_generation, 0);
        assert_eq!(status.reserved_bytes, 0);

        let status = storage_class_status(&config_with_class(Some(StorageClass::Balanced as i32)));
        assert_eq!(status.state, State::Active as i32);
        assert_eq!(status.policy_generation, 1);
    }

    #[test]
    fn economy_and_performance_report_not_active_with_a_reason() {
        for (class, needle) in [
            (StorageClass::Economy, "economy"),
            (StorageClass::Performance, "performance"),
        ] {
            let status = storage_class_status(&config_with_class(Some(class as i32)));
            assert_eq!(status.state, State::NotActive as i32, "{needle}");
            assert!(
                status.reason.contains("SC3/SC4"),
                "reason must name the missing dependency: {}",
                status.reason
            );
            assert_eq!(status.policy_generation, 1);
            assert_eq!(status.reserved_bytes, 0);
            assert_eq!(status.resident_bytes, 0);
            assert_eq!(status.covered_bytes, 0);
        }
    }

    #[test]
    fn performance_admission_uses_the_resolved_placement_and_tenant_authorization() {
        let mut config = config_with_class(Some(StorageClass::Performance as i32));

        // Engine is not SST ⇒ rejected, naming the engine.
        let error = check_performance_admission(
            &config,
            StorageEngine::Helix,
            "s3://bucket/collections",
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("SST"), "{error}");

        // SST but the RESOLVED location (here the local default a bare
        // StorageConfig resolves to) is not object-store-backed ⇒ rejected.
        let error = check_performance_admission(&config, StorageEngine::Sst, "file://./data", None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("object-store-backed"), "{error}");
        assert!(error.contains("file://./data"), "{error}");

        // A configured remote DEFAULT (no client path at all) is admissible —
        // operator-configured targets are authorized by construction.
        check_performance_admission(
            &config,
            StorageEngine::Sst,
            "s3://bench/datasets",
            Some("acme"),
        )
        .unwrap();

        // An explicit client path under ANOTHER tenant's prefix is rejected.
        config.storage_config = Some(StorageConfig {
            storage_path: Some("s3://bench/tenants/other-tenant/coll".to_string()),
            ..Default::default()
        });
        let error = check_performance_admission(
            &config,
            StorageEngine::Sst,
            "s3://bench/tenants/other-tenant/coll",
            Some("acme"),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("other-tenant"), "{error}");
        assert!(error.contains("acme"), "{error}");

        // The SAME explicit path under the REQUESTING tenant's prefix is fine.
        config.storage_config = Some(StorageConfig {
            storage_path: Some("s3://bench/tenants/acme/coll".to_string()),
            ..Default::default()
        });
        check_performance_admission(
            &config,
            StorageEngine::Sst,
            "s3://bench/tenants/acme/coll",
            Some("acme"),
        )
        .unwrap();

        // A NESTED `tenants/` segment cannot smuggle authorization: the path
        // lives under the victim's prefix, so parsing the innermost segment
        // would name the requesting tenant and pass the check while writing
        // into the victim's subtree. The outermost owner is authoritative.
        config.storage_config = Some(StorageConfig {
            storage_path: Some(
                "s3://bench/tenants/other-tenant/collections/tenants/acme/coll".to_string(),
            ),
            ..Default::default()
        });
        let error = check_performance_admission(
            &config,
            StorageEngine::Sst,
            "s3://bench/tenants/other-tenant/collections/tenants/acme/coll",
            Some("acme"),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("other-tenant"), "{error}");

        // No tenant context (operator/internal path) skips the tenant check.
        config.storage_config = Some(StorageConfig {
            storage_path: Some("s3://bench/tenants/other-tenant/coll".to_string()),
            ..Default::default()
        });
        check_performance_admission(
            &config,
            StorageEngine::Sst,
            "s3://bench/tenants/other-tenant/coll",
            None,
        )
        .unwrap();

        // Balanced/economy never trigger the check.
        config.storage_class = Some(StorageClass::Balanced as i32);
        config.storage_config = None;
        check_performance_admission(&config, StorageEngine::Helix, "file://./data", None).unwrap();
    }
}
