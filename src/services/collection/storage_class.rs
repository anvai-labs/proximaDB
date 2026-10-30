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

/// The persisted monotonic generation, or the SC1-era heuristic for legacy
/// records without the field (0 = class never set, 1 = persisted at create).
pub fn policy_generation_or_legacy(config: &CollectionConfig) -> u64 {
    match config.policy_generation {
        Some(generation) => u64::from(generation),
        None => u64::from(config.storage_class.is_some()),
    }
}

/// The read-side observed status: a pure projection of the desired policy,
/// truthful about what is actually delivered.
///
/// BALANCED (or an absent field) ⇒ ACTIVE. ECONOMY/PERFORMANCE ⇒ NOT_ACTIVE
/// with a reason naming the missing dependency (SC4 residency coverage;
/// SC3 admission itself is wired — it reserves pool capacity at create and
/// at policy change, but no protected residency or coverage proof is
/// active), so the collection serves balanced behavior and no guarantee is
/// implied. `policy_generation` is the persisted monotonic counter (SC3-2b):
/// bumped on every actual class change through the mutation surface;
/// legacy records keep the SC1-era heuristic.
pub fn storage_class_status(config: &CollectionConfig) -> StorageClassStatus {
    let class = storage_class_or_balanced(config);
    let generation = policy_generation_or_legacy(config);
    match class {
        StorageClass::Balanced | StorageClass::Unspecified => StorageClassStatus {
            state: State::Active as i32,
            reason: String::new(),
            policy_generation: generation,
            reserved_bytes: 0,
            resident_bytes: 0,
            covered_bytes: 0,
            covered_generation: 0,
        },
        StorageClass::Economy => StorageClassStatus {
            state: State::NotActive as i32,
            reason: "economy residency is not implemented (TD-TIER-1 SC4): the \
                     collection serves balanced behavior"
                .to_string(),
            policy_generation: generation,
            reserved_bytes: 0,
            resident_bytes: 0,
            covered_bytes: 0,
            covered_generation: 0,
        },
        StorageClass::Performance => StorageClassStatus {
            state: State::NotActive as i32,
            reason: "performance residency coverage is not implemented \
                     (TD-TIER-1 SC4): admission reserves pool capacity at create, \
                     but no protected residency or coverage proof is active — the \
                     collection serves balanced behavior"
                .to_string(),
            policy_generation: generation,
            reserved_bytes: 0,
            resident_bytes: 0,
            covered_bytes: 0,
            covered_generation: 0,
        },
    }
}

/// The query profile READY is proven for (SC4-5): default PAX ANN search
/// (RaBitQ + SQ8 + f32 rerank, payload/metadata, filters) over the SST vector
/// readers. Other modalities and the DataFusion/legacy readers are not covered.
pub const READY_PROFILE: &str = "pax-ann-v1";

/// TD-TIER-1 SC4-5: the PERFORMANCE status from a grant and a residency proof
/// of its committed segment set. READY needs POSITIVE evidence (review B1):
/// - DEGRADED: the grant names no segment roots; the collection is not served
///   by [`READY_PROFILE`] (AXIS-routed `index_configs`); no persistent store;
///   or the set's STORED size exceeds the grant or the store's capacity
///   (coverage can never complete);
/// - WARMING: no published segment found under the roots yet (an empty
///   listing is never proof — the roots may not be where the segments are),
///   or coverage in progress;
/// - READY: every residency object is resident in full and protected, for
///   generation `g` and [`READY_PROFILE`].
///
/// Recomputed on every status read, so a lost entry (eviction, corruption,
/// retirement race) or a new publication drops READY immediately.
pub fn with_residency_proof(
    mut status: StorageClassStatus,
    reserved_bytes: u64,
    resident_bytes: u64,
    covered_bytes: u64,
    in_profile: bool,
    proof: &crate::storage::engines::sst::segment_io::ResidencyProof,
) -> StorageClassStatus {
    status.reserved_bytes = reserved_bytes;
    status.resident_bytes = resident_bytes;
    status.covered_bytes = covered_bytes;
    status.covered_generation = proof.generation;
    let capacity = reserved_bytes.min(proof.store_capacity);
    let (state, reason) = if proof.roots == 0 {
        (
            State::Degraded,
            "the grant names no segment roots (the collection has no storage assignment), \
             so no coverage can be proven"
                .to_string(),
        )
    } else if !in_profile {
        (
            State::Degraded,
            format!(
                "the collection is not served by profile {READY_PROFILE} (it must be on the \
                 SST engine with no AXIS index_configs), so residency cannot be proven for it"
            ),
        )
    } else if !proof.store_present {
        (
            State::Degraded,
            "no persistent local store is configured, so no segment can be resident \
             (set the cache local-disk path)"
                .to_string(),
        )
    } else if proof.segment_files == 0 {
        (
            State::Warming,
            "no published segment found under the grant's segment roots yet; READY needs a \
             proven, non-empty segment set"
                .to_string(),
        )
    } else if proof.covered_segments == proof.segments {
        (
            State::Ready,
            format!(
                "all {} objects of generation {:#x} are resident and protected \
                 (profile {READY_PROFILE})",
                proof.segments, proof.generation
            ),
        )
    } else if proof.inventory_stored_bytes > capacity {
        (
            State::Degraded,
            format!(
                "the committed segment set ({} bytes stored) exceeds what can be held \
                 ({capacity} bytes: the grant or the store's capacity, whichever is \
                 smaller); {} of {} objects resident — raise the reservation",
                proof.inventory_stored_bytes, proof.covered_segments, proof.segments
            ),
        )
    } else {
        (
            State::Warming,
            format!(
                "{} of {} objects of generation {:#x} resident; warming (profile \
                 {READY_PROFILE})",
                proof.covered_segments, proof.segments, proof.generation
            ),
        )
    };
    status.state = state as i32;
    status.reason = reason;
    status
}

/// TD-TIER-1 SC4-5: the PERFORMANCE status when the proof itself failed (e.g.
/// the listing errored): the progress figures stay, the state is DEGRADED
/// with the cause (review S5).
pub fn with_failed_proof(
    mut status: StorageClassStatus,
    reserved_bytes: u64,
    resident_bytes: u64,
    covered_bytes: u64,
    error: &str,
) -> StorageClassStatus {
    status.reserved_bytes = reserved_bytes;
    status.resident_bytes = resident_bytes;
    status.covered_bytes = covered_bytes;
    status.state = State::Degraded as i32;
    status.reason = format!("the residency proof could not run: {error}");
    status
}

/// The wire label of an observed status state, shared by REST v2 and gRPC v2
/// so the two surfaces can never disagree ("unspecified" for an unknown
/// discriminant).
pub fn state_label(state: i32) -> &'static str {
    State::try_from(state)
        .map(|state| match state {
            State::Active => "active",
            State::NotActive => "not_active",
            State::Warming => "warming",
            State::Ready => "ready",
            State::Degraded => "degraded",
            State::Unspecified => "unspecified",
        })
        .unwrap_or("unspecified")
}

/// The engine a record actually runs on, as the engine factory resolves it:
/// absent and `UNSPECIFIED` (legacy records persist `Some(0)`) run on SST; an
/// unknown number resolves to `None`, which no SST-only check accepts.
pub fn effective_storage_engine(raw: Option<i32>) -> Option<StorageEngine> {
    match raw.map(StorageEngine::try_from) {
        None | Some(Ok(StorageEngine::Unspecified)) => Some(StorageEngine::Sst),
        Some(Ok(engine)) => Some(engine),
        Some(Err(_)) => None,
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
    // An explicit path must be scheme-carrying or absolute, in every context:
    // a RELATIVE path ("tenants/x/...") resolves against process state, so it
    // is rejected outright rather than slipping the tenant-prefix parse.
    if let Some(explicit) = config
        .storage_config
        .as_ref()
        .and_then(|storage| storage.storage_path.as_deref())
        .filter(|p| !p.is_empty())
        && !explicit.contains("://")
        && !explicit.starts_with('/')
    {
        return Err(anyhow::anyhow!(
            "storage_class 'performance' placement '{explicit}' must be an \
             absolute path or carry a storage scheme (s3://, gs://, az:// ...); \
             relative paths are not admissible"
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
    ) && let Some(owner) = tenant_segment_of(explicit)
        && owner != current_tenant
    {
        return Err(anyhow::anyhow!(
            "storage_class 'performance' placement '{explicit}' is under another \
             tenant's prefix (tenants/{owner}); requesting tenant is \
             '{current_tenant}'. Use an operator-configured location or this \
             tenant's own prefix"
        ));
    }
    if proximadb_storage_filesystem_types::object_store_scheme::ObjectStoreScheme::of(
        resolved_base_location,
    )
    .is_none()
    {
        return Err(anyhow::anyhow!(
            "storage_class 'performance' requires an object-store-backed placement \
             (s3://, gs://, az:// — or the adls/abfs/azure aliases; https:// for \
             S3-compatible endpoints); the resolved location '{}' is not one. \
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

/// TD-TIER-1 SC4-2: the segment-root prefixes a collection's published
/// `.pax` objects live under — the persistent-L2 protection keys embed them.
///
/// Reconstructed with the SAME helper the SST flush and compaction paths use
/// (`collection_data_path_typed` over the storage assignment's tenant-scoped
/// `base_location`), for BOTH layouts: the legacy `{base}/{id}/data/` and,
/// when the typed triple is present, `{base}/accounts/{a}/{n}/{c}/data/` —
/// the layout follows the flusher's `PROXIMADB_TYPED_PATHS` at flush time,
/// which a restart can flip. Each prefix ends in `/`, so containment cannot
/// match a sibling collection whose id shares a decimal prefix (`.../12/`
/// vs `.../123/`). The base's trailing `/` is trimmed first: normal flushes
/// trim it, `Path::join` publishes do not.
pub fn residency_prefixes(
    base_location: &str,
    collection_id: &str,
    identity: Option<crate::core::stable_id::CollectionIdentity>,
) -> Vec<String> {
    use crate::storage::trait_components::path_resolver::collection_data_path_typed;
    let base = base_location.trim_end_matches('/');
    let mut prefixes = vec![format!(
        "{}/",
        collection_data_path_typed(base, collection_id, None)
    )];
    if let Some(identity) = identity {
        prefixes.push(format!(
            "{}/",
            collection_data_path_typed(base, collection_id, Some(identity))
        ));
    }
    prefixes
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

    /// Review NIT on SC4-5: a legacy record persisting UNSPECIFIED runs on SST
    /// (the engine factory's fallback), so it is in the SST-only profile and
    /// admissible; an unknown number resolves to nothing.
    #[test]
    fn effective_storage_engine_follows_the_factory() {
        assert_eq!(effective_storage_engine(None), Some(StorageEngine::Sst));
        assert_eq!(
            effective_storage_engine(Some(StorageEngine::Unspecified as i32)),
            Some(StorageEngine::Sst)
        );
        assert_eq!(
            effective_storage_engine(Some(StorageEngine::Helix as i32)),
            Some(StorageEngine::Helix)
        );
        assert_eq!(effective_storage_engine(Some(9_999)), None);
    }

    use super::*;
    use crate::proto::proximadb_v1::StorageConfig;

    fn config_with_class(raw: Option<i32>) -> CollectionConfig {
        CollectionConfig {
            storage_class: raw,
            ..Default::default()
        }
    }

    /// SC4-3: the status overlay reports the grant's bytes without claiming READY.
    fn proof(
        store_present: bool,
        segments: usize,
        covered_segments: usize,
        inventory_bytes: u64,
    ) -> crate::storage::engines::sst::segment_io::ResidencyProof {
        crate::storage::engines::sst::segment_io::ResidencyProof {
            store_present,
            roots: 1,
            segments,
            segment_files: segments,
            covered_segments,
            inventory_bytes,
            inventory_stored_bytes: inventory_bytes + 100 * segments as u64,
            store_capacity: 1 << 30,
            generation: 0xabc,
        }
    }

    /// SC4-5: READY only with positive evidence; WARMING while the set fits
    /// but is partial or nothing is published yet; DEGRADED when it can
    /// never fit (STORED size), has no roots or store, or the collection is
    /// outside the profile. Bytes and generation are reported in every state.
    #[test]
    fn residency_proof_drives_the_state() {
        let base =
            || storage_class_status(&config_with_class(Some(StorageClass::Performance as i32)));
        let ready = with_residency_proof(base(), 4096, 3000, 3000, true, &proof(true, 3, 3, 3000));
        assert_eq!(ready.state, State::Ready as i32, "{}", ready.reason);
        assert_eq!(ready.covered_generation, 0xabc);
        assert!(ready.reason.contains(READY_PROFILE));

        let warming =
            with_residency_proof(base(), 4096, 3000, 1000, true, &proof(true, 3, 1, 3000));
        assert_eq!(warming.state, State::Warming as i32, "{}", warming.reason);
        assert!(warming.reason.contains("1 of 3"));

        // Fits by value bytes (3000 <= 3100) but not by STORED bytes (3300).
        let over = with_residency_proof(base(), 3100, 3000, 1000, true, &proof(true, 3, 1, 3000));
        assert_eq!(over.state, State::Degraded as i32, "{}", over.reason);
        assert!(over.reason.contains("exceeds"));

        let no_store = with_residency_proof(base(), 4096, 0, 0, true, &proof(false, 3, 0, 3000));
        assert_eq!(no_store.state, State::Degraded as i32);
        assert_eq!(state_label(no_store.state), "degraded");
        assert_eq!(state_label(ready.state), "ready");
        assert_eq!(state_label(warming.state), "warming");

        // Review B1: an empty set is never proof.
        let empty = with_residency_proof(base(), 4096, 0, 0, true, &proof(true, 0, 0, 0));
        assert_eq!(empty.state, State::Warming as i32, "{}", empty.reason);
        // Review N-a: a set holding only a covered directory is not READY.
        let mut directory_only = proof(true, 1, 1, 100);
        directory_only.segment_files = 0;
        let lone = with_residency_proof(base(), 4096, 100, 100, true, &directory_only);
        assert_eq!(lone.state, State::Warming as i32, "{}", lone.reason);
        let mut no_roots = proof(true, 0, 0, 0);
        no_roots.roots = 0;
        let rootless = with_residency_proof(base(), 4096, 0, 0, true, &no_roots);
        assert_eq!(
            rootless.state,
            State::Degraded as i32,
            "{}",
            rootless.reason
        );
        // Review S4: outside the profile, never READY even when covered.
        let axis = with_residency_proof(base(), 4096, 3000, 3000, false, &proof(true, 3, 3, 3000));
        assert_eq!(axis.state, State::Degraded as i32, "{}", axis.reason);

        let failed = with_failed_proof(base(), 4096, 10, 5, "list failed");
        assert_eq!(failed.state, State::Degraded as i32);
        assert_eq!(
            (
                failed.reserved_bytes,
                failed.resident_bytes,
                failed.covered_bytes
            ),
            (4096, 10, 5)
        );
    }

    /// SC4-2: the prefixes are the flush/compaction segment roots, both
    /// layouts, trailing `/`, with the base's own trailing `/` trimmed.
    #[test]
    fn residency_prefixes_match_both_segment_layouts() {
        use crate::core::stable_id::CollectionIdentity;
        let legacy_only = residency_prefixes("s3://b/tenants/acme/", "12", None);
        assert_eq!(
            legacy_only,
            vec!["s3://b/tenants/acme/12/data/".to_string()]
        );

        let identity = CollectionIdentity {
            account_id: 1,
            namespace_id: 2,
            collection_id: 3,
        };
        let both = residency_prefixes("s3://b/tenants/acme", "12", Some(identity));
        assert_eq!(both.len(), 2);
        assert_eq!(both[0], "s3://b/tenants/acme/12/data/");
        assert!(both[1].starts_with("s3://b/tenants/acme/accounts/"));
        assert!(both[1].ends_with("/data/"));
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
                status.reason.contains("SC4"),
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

        // A RELATIVE explicit path is rejected in every context — it resolves
        // against process state and would slip the tenant-prefix parse
        // ("tenants/other/..." with no scheme or leading '/').
        config.storage_config = Some(StorageConfig {
            storage_path: Some("tenants/other-tenant/coll".to_string()),
            ..Default::default()
        });
        for tenant in [Some("acme"), None] {
            let error = check_performance_admission(
                &config,
                StorageEngine::Sst,
                "tenants/other-tenant/coll",
                tenant,
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains("relative"), "{error}");
        }

        // Balanced/economy never trigger the check.
        config.storage_class = Some(StorageClass::Balanced as i32);
        config.storage_config = None;
        check_performance_admission(&config, StorageEngine::Helix, "file://./data", None).unwrap();
    }
}
