//! Canonical tenant-attribution primitives for ProximaDB.
//!
//! # Why this is a foundation crate (cloud-native elasticity + economics)
//!
//! In a multi-tenant, horizontally-elastic cloud database, *who owns this data*
//! is not a storage detail — it is the key that every **metering** boundary
//! (KSU storage, KRU/KIU read/compute, KOU outgress, KEU embedding) and every
//! **isolation** boundary (`DrPathBuilder` prefixes, fail-closed `TenantContext`
//! gates) attributes work against. If two layers resolve the owning tenant
//! differently, billing drifts and isolation leaks. So tenant resolution must be
//! **one** primitive, shared by all tiers — storage, services, the network
//! gates, and the billing/governance plane — rather than a private helper on any
//! one of them.
//!
//! This crate is that single source of truth. It lives at the **foundation**
//! tier (depending only on `proximadb-proto`) precisely so every higher tier can
//! depend *down* on it; nothing has to reach *up* into the services layer to ask
//! "whose tenant is this?".
//!
//! * **Elasticity:** resolution is a pure function of the collection's own
//!   metadata — no shared state, no coordinator, no I/O. Any node in an
//!   autoscaled fleet resolves the same owner from the same `Collection`, so
//!   attribution is correct without cross-node consensus.
//! * **Economics:** because it is the *only* resolver, per-tenant meters across
//!   every dimension agree, and the bill reconciles.
//! * **Isolation (fail-closed):** an unresolved tenant returns [`None`]. Callers
//!   at an isolation boundary MUST treat `None` as *deny / do not attribute to a
//!   shared tenant* — never as a default or "public" tenant. Isolation is
//!   structural, not a best-effort guess.

use proximadb_proto::proximadb_v1::Collection;

/// Resolve the owning tenant id for a collection, or [`None`] if the collection
/// is not tenant-scoped (or carries no resolvable owner).
///
/// This is the **canonical** tenant resolver — the network gates, the storage
/// flush/compaction paths, and the collection service all call this one function
/// so tag/owner precedence is identical everywhere.
///
/// # Precedence (stable contract)
///
/// 1. An explicit `tenant:<id>` tag — the authoritative marker written at
///    provisioning time. The *first* `tenant:`-prefixed tag is taken; if its id
///    is empty, resolution stops at `None` (it does not skip to a later
///    `tenant:` tag). Wins over the isolated/owner path below.
/// 2. Otherwise, if the collection is flagged `tenant_isolated:true`, the
///    collection `owner` (when non-empty) is the tenant.
/// 3. Otherwise [`None`] — not tenant-scoped. **Fail closed** at isolation
///    boundaries: do not fall back to a shared/default tenant.
pub fn tenant_id_of(collection: &Collection) -> Option<String> {
    let config = collection.config.as_ref()?;

    if let Some(tag_tenant) = config
        .tags
        .iter()
        .find_map(|tag| tag.strip_prefix("tenant:"))
        .filter(|tenant_id| !tenant_id.is_empty())
    {
        return Some(tag_tenant.to_string());
    }

    let tenant_isolated = config.tags.iter().any(|tag| tag == "tenant_isolated:true");
    if tenant_isolated {
        return config
            .owner
            .as_ref()
            .filter(|owner| !owner.is_empty())
            .cloned();
    }

    None
}

/// The single canonical default **request** tenant.
///
/// Non-empty on purpose: an empty tenant id is rejected by
/// `DrPathBuilder::validate_id` and makes the catalog's `resolve_table_scoped`
/// skip the tenant namespace entirely — i.e. an *empty* default silently
/// **disables** structural isolation. A non-empty default keeps every request
/// attributable to a real tenant partition.
pub const DEFAULT_TENANT: &str = "default";

/// Resolve the **request** tenant (who is acting) from a raw per-surface signal —
/// the pgwire startup `database`, the REST `X-Tenant-ID` header, or the
/// gRPC/Arrow-Flight `x-tenant-id` metadata. Trims surrounding whitespace; an
/// absent or empty signal resolves to [`DEFAULT_TENANT`].
///
/// This is deliberately distinct from [`tenant_id_of`]: that resolves the
/// **owner** of a stored collection and fails closed to [`None`], because an
/// ownership gate must deny when it cannot attribute. A *request*, by contrast,
/// always acts as some tenant, so it defaults. Every network surface calls THIS
/// one function, so their defaults can never drift apart again (the pre-unify
/// state was pgwire/gRPC `""` vs REST `"default"` — a cross-surface data split).
pub fn resolve_request_tenant(raw: Option<&str>) -> String {
    match raw.map(str::trim).filter(|t| !t.is_empty()) {
        Some(tenant) => tenant.to_string(),
        None => DEFAULT_TENANT.to_string(),
    }
}

/// Explicit deployment-mode contract for request-tenant resolution.
///
/// `SingleTenant` preserves the existing compatibility default: an absent tenant signal resolves
/// to the configured default tenant. `MultiTenant` is fail-closed: every request must carry an
/// explicit tenant signal at the edge (REST header, gRPC/Flight metadata, pgwire database, or an
/// authenticated tenant claim). Both modes validate the resolved tenant before callers compose it
/// into catalog namespaces or object-store paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TenantDeploymentMode {
    /// Back-compatible single-tenant mode with a named default tenant.
    SingleTenant { default_tenant: String },
    /// SaaS/multi-tenant mode: a tenant signal is mandatory on every request.
    MultiTenant,
}

impl TenantDeploymentMode {
    /// Back-compatible default single-tenant mode.
    pub fn single_tenant_default() -> Self {
        Self::SingleTenant {
            default_tenant: DEFAULT_TENANT.to_string(),
        }
    }

    /// Construct a single-tenant mode with an explicit default tenant.
    pub fn single_tenant(default_tenant: impl Into<String>) -> Self {
        Self::SingleTenant {
            default_tenant: default_tenant.into(),
        }
    }
}

/// Request-tenant resolution failure under an explicit [`TenantDeploymentMode`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveRequestTenantError {
    /// Multi-tenant mode requires an explicit request tenant.
    MissingTenant,
    /// The resolved tenant is not safe as a structural catalog/path key.
    InvalidTenant(TenantError),
}

impl std::fmt::Display for ResolveRequestTenantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveRequestTenantError::MissingTenant => {
                write!(f, "tenant id is required in multi-tenant mode")
            }
            ResolveRequestTenantError::InvalidTenant(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for ResolveRequestTenantError {}

/// Resolve and validate the request tenant under an explicit deployment mode.
pub fn resolve_request_tenant_for_mode(
    raw: Option<&str>,
    mode: &TenantDeploymentMode,
) -> Result<String, ResolveRequestTenantError> {
    let raw = raw.map(str::trim).filter(|tenant| !tenant.is_empty());
    let tenant = match (mode, raw) {
        (_, Some(tenant)) => tenant.to_string(),
        (TenantDeploymentMode::SingleTenant { default_tenant }, None) => default_tenant.clone(),
        (TenantDeploymentMode::MultiTenant, None) => {
            return Err(ResolveRequestTenantError::MissingTenant);
        }
    };

    validate_request_tenant(&tenant).map_err(ResolveRequestTenantError::InvalidTenant)?;
    Ok(tenant)
}

/// Why a tenant id is not safe to compose into a storage key / path / catalog
/// `namespace[0]`. Returned by [`validate_request_tenant`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TenantError {
    /// Empty tenant id (would disable structural isolation).
    Empty,
    /// Begins with `_` — reserved for control-plane system subtrees
    /// (`_operator`, `_metering`, `_trace`, …), so a tenant can never shadow one.
    ReservedPrefix,
    /// Contains a path-traversal / separator / control / whitespace character.
    InvalidChar(char),
}

impl std::fmt::Display for TenantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TenantError::Empty => write!(f, "tenant id must not be empty"),
            TenantError::ReservedPrefix => {
                write!(
                    f,
                    "tenant id must not begin with '_' (reserved for system use)"
                )
            }
            TenantError::InvalidChar(c) => {
                write!(f, "tenant id contains an invalid character: {c:?}")
            }
        }
    }
}

impl std::error::Error for TenantError {}

/// Validate a resolved request tenant BEFORE it becomes a storage-key dimension,
/// a `DrPathBuilder` path segment, or a catalog `namespace[0]`. Mirrors the
/// structural guard in `DrPathBuilder::validate_id`: rejects empty, `_`-prefixed
/// (reserved system segments), path traversal (`..`), and separator / control /
/// whitespace characters (`/`, `\`, `\0`, …). Callers validate at the ingress
/// boundary and fail the request on error (fail-closed).
pub fn validate_request_tenant(tenant: &str) -> Result<(), TenantError> {
    if tenant.is_empty() {
        return Err(TenantError::Empty);
    }
    if tenant.starts_with('_') {
        return Err(TenantError::ReservedPrefix);
    }
    if tenant.contains("..") {
        return Err(TenantError::InvalidChar('.'));
    }
    if let Some(c) = tenant
        .chars()
        .find(|&c| c == '/' || c == '\\' || c == '\0' || c.is_whitespace())
    {
        return Err(TenantError::InvalidChar(c));
    }
    Ok(())
}

/// Why a tenant-scoped resource key could not be composed. Returned by
/// [`scoped_resource_key`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScopedKeyError {
    /// The tenant is not a safe key segment (see [`validate_request_tenant`]).
    Tenant(TenantError),
    /// The resource name is empty.
    EmptyName,
    /// The resource name contains a separator or control character. A `/` in a
    /// name would let it address another tenant's `{tenant}/{name}` key.
    InvalidNameChar(char),
}

impl std::fmt::Display for ScopedKeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScopedKeyError::Tenant(e) => write!(f, "invalid tenant: {e}"),
            ScopedKeyError::EmptyName => write!(f, "name must not be empty"),
            ScopedKeyError::InvalidNameChar(c) => {
                write!(f, "name contains an invalid character: {c:?}")
            }
        }
    }
}

impl std::error::Error for ScopedKeyError {}

/// Validate a tenant-clean resource name (document collection, graph id) before
/// it is folded into a tenant-scoped key: non-empty, and free of `/`, `\` and
/// control characters. Callers pass the name exactly as the client sent it
/// (after URL decoding), so an encoded separator is rejected too.
pub fn validate_scoped_resource_name(name: &str) -> Result<(), ScopedKeyError> {
    if name.is_empty() {
        return Err(ScopedKeyError::EmptyName);
    }
    if let Some(c) = name
        .chars()
        .find(|&c| c == '/' || c == '\\' || c.is_control())
    {
        return Err(ScopedKeyError::InvalidNameChar(c));
    }
    Ok(())
}

/// THE composition of a tenant-scoped resource key: `{tenant}/{name}`, with the
/// default tenant left bare (`name`). Both the tenant and the name are validated
/// first, so the first `/` of a key is always the tenant separator and a name can
/// never address another tenant's key.
pub fn scoped_resource_key(tenant: &str, name: &str) -> Result<String, ScopedKeyError> {
    validate_request_tenant(tenant).map_err(ScopedKeyError::Tenant)?;
    validate_scoped_resource_name(name)?;
    if tenant == DEFAULT_TENANT {
        return Ok(name.to_string());
    }
    Ok(format!("{tenant}/{name}"))
}

/// Inverse of [`scoped_resource_key`] for keys visible to `tenant`: the clean name
/// when `key` belongs to `tenant`, `None` otherwise.
pub fn unscoped_resource_name<'a>(tenant: &str, key: &'a str) -> Option<&'a str> {
    if tenant == DEFAULT_TENANT {
        return (!key.contains('/')).then_some(key);
    }
    key.strip_prefix(tenant)
        .and_then(|rest| rest.strip_prefix('/'))
        .filter(|name| !name.is_empty() && !name.contains('/'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_proto::proximadb_v1::CollectionConfig;

    #[test]
    fn scoped_resource_key_composes_and_rejects_separators() {
        assert_eq!(scoped_resource_key(DEFAULT_TENANT, "docs").unwrap(), "docs");
        assert_eq!(scoped_resource_key("acme", "docs").unwrap(), "acme/docs");
        for name in ["", "globex/docs", "a\\b", "a\0b"] {
            assert!(scoped_resource_key("acme", name).is_err(), "{name:?}");
            assert!(
                scoped_resource_key(DEFAULT_TENANT, name).is_err(),
                "{name:?}"
            );
        }
        assert!(matches!(
            scoped_resource_key("_sys", "docs"),
            Err(ScopedKeyError::Tenant(_))
        ));
    }

    #[test]
    fn unscoped_resource_name_keeps_only_the_tenants_own_keys() {
        assert_eq!(unscoped_resource_name("acme", "acme/docs"), Some("docs"));
        assert_eq!(unscoped_resource_name("acme", "globex/docs"), None);
        assert_eq!(unscoped_resource_name("acme", "acmex/docs"), None);
        assert_eq!(unscoped_resource_name("acme", "docs"), None);
        assert_eq!(unscoped_resource_name(DEFAULT_TENANT, "docs"), Some("docs"));
        assert_eq!(unscoped_resource_name(DEFAULT_TENANT, "acme/docs"), None);
    }

    fn collection(tags: &[&str], owner: Option<&str>) -> Collection {
        Collection {
            config: Some(CollectionConfig {
                tags: tags.iter().map(|t| t.to_string()).collect(),
                owner: owner.map(|o| o.to_string()),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn explicit_tenant_tag_wins() {
        let c = collection(&["tenant:acme", "tenant_isolated:true"], Some("someone"));
        // Tag precedence beats the isolated/owner path.
        assert_eq!(tenant_id_of(&c).as_deref(), Some("acme"));
    }

    #[test]
    fn deployment_mode_single_tenant_defaults_and_validates() {
        let mode = TenantDeploymentMode::single_tenant("tenant_a");

        assert_eq!(
            resolve_request_tenant_for_mode(None, &mode).as_deref(),
            Ok("tenant_a")
        );
        assert_eq!(
            resolve_request_tenant_for_mode(Some(" tenant_b "), &mode).as_deref(),
            Ok("tenant_b")
        );

        assert!(matches!(
            resolve_request_tenant_for_mode(Some("_operator"), &mode),
            Err(ResolveRequestTenantError::InvalidTenant(
                TenantError::ReservedPrefix
            ))
        ));
    }

    #[test]
    fn deployment_mode_multi_tenant_requires_explicit_tenant() {
        let mode = TenantDeploymentMode::MultiTenant;

        assert_eq!(
            resolve_request_tenant_for_mode(Some("tenant_a"), &mode).as_deref(),
            Ok("tenant_a")
        );
        assert_eq!(
            resolve_request_tenant_for_mode(None, &mode),
            Err(ResolveRequestTenantError::MissingTenant)
        );
        assert_eq!(
            resolve_request_tenant_for_mode(Some("   "), &mode),
            Err(ResolveRequestTenantError::MissingTenant)
        );
    }

    #[test]
    fn single_tenant_tag_resolves() {
        assert_eq!(
            tenant_id_of(&collection(&["env:prod", "tenant:beta"], None)).as_deref(),
            Some("beta")
        );
    }

    #[test]
    fn first_tenant_prefixed_tag_is_authoritative_even_if_empty() {
        // Documented contract: the first `tenant:`-prefixed tag is taken; if it
        // is empty the result is None (resolution does NOT skip ahead to a later
        // `tenant:` tag). Preserved verbatim from the original service helper.
        let c = collection(&["tenant:", "tenant:beta"], None);
        assert_eq!(tenant_id_of(&c), None);
    }

    #[test]
    fn isolated_falls_back_to_owner() {
        let c = collection(&["tenant_isolated:true"], Some("owner-co"));
        assert_eq!(tenant_id_of(&c).as_deref(), Some("owner-co"));
    }

    #[test]
    fn isolated_without_owner_is_unresolved() {
        // Fail closed: isolated but no owner -> None, never a default tenant.
        let c = collection(&["tenant_isolated:true"], None);
        assert_eq!(tenant_id_of(&c), None);
        let c_empty = collection(&["tenant_isolated:true"], Some(""));
        assert_eq!(tenant_id_of(&c_empty), None);
    }

    #[test]
    fn untagged_collection_is_not_tenant_scoped() {
        // No tenant markers -> None (fail closed at isolation boundaries).
        assert_eq!(
            tenant_id_of(&collection(&["env:prod"], Some("owner"))),
            None
        );
    }

    #[test]
    fn missing_config_is_unresolved() {
        let c = Collection {
            config: None,
            ..Default::default()
        };
        assert_eq!(tenant_id_of(&c), None);
    }

    // ── request-tenant resolution (the one default, shared by all surfaces) ──

    #[test]
    fn resolve_request_tenant_defaults_when_absent_or_empty() {
        assert_eq!(resolve_request_tenant(None), DEFAULT_TENANT);
        assert_eq!(resolve_request_tenant(Some("")), DEFAULT_TENANT);
        assert_eq!(resolve_request_tenant(Some("   ")), DEFAULT_TENANT);
        assert_eq!(DEFAULT_TENANT, "default");
    }

    #[test]
    fn resolve_request_tenant_passes_through_and_trims() {
        assert_eq!(resolve_request_tenant(Some("acme")), "acme");
        assert_eq!(resolve_request_tenant(Some("  acme  ")), "acme");
    }

    #[test]
    fn validate_request_tenant_accepts_ordinary_ids() {
        assert!(validate_request_tenant("acme").is_ok());
        assert!(validate_request_tenant("default").is_ok());
        assert!(validate_request_tenant("acct-lender-us").is_ok());
    }

    #[test]
    fn validate_request_tenant_rejects_unsafe_ids() {
        assert_eq!(validate_request_tenant(""), Err(TenantError::Empty));
        assert_eq!(
            validate_request_tenant("_operator"),
            Err(TenantError::ReservedPrefix)
        );
        assert_eq!(
            validate_request_tenant("../etc"),
            Err(TenantError::InvalidChar('.'))
        );
        assert_eq!(
            validate_request_tenant("a/b"),
            Err(TenantError::InvalidChar('/'))
        );
        assert_eq!(
            validate_request_tenant("a b"),
            Err(TenantError::InvalidChar(' '))
        );
    }
}

// ============================================================================
// Tenant consumption types (pre-extracted from root src/metrics + src/catalog
// for the observability crate extraction — Slice D root-crate shrinkage).
// Foundation-pure: String + u64 primitives + serde.
// ============================================================================

/// Per-tenant resident storage usage snapshot (for consumption metering).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TenantStorageUsage {
    pub tenant_id: String,
    pub resident_bytes: u64,
}

// ============================================================================

// ============================================================================
// Tenant tier system (moved from src/catalog/tenant_tier.rs — Slice D).
// ============================================================================
pub mod tenant_tier_types;
pub use tenant_tier_types::{
    FeatureFlags, ObjectEconomyQuantizationCeiling, TenantTierRecord, Tier,
    TierObjectEconomyConfig, tier_config,
};

// ============================================================================
// RBAC authorization-context data types (moved from
// `src/security/rbac_service.rs` — Slice D root-crate shrinkage).
// Leaf data types (permission enum, auth method, user/tenant context) with no
// root-internal dependencies — pure String / chrono / serde / std collections.
// The originating `rbac_service.rs` keeps a `pub use` re-export of every type
// below, so existing `crate::security::rbac_service::<Type>` paths are unchanged.
// ============================================================================
pub mod rbac_context;
pub use rbac_context::{
    RbacTenantContext, UnifiedAuthMethod, UnifiedPermission, UnifiedUserContext,
};

// ============================================================================
// Tenant-assertion trust (TD-TENANT-1): the ONE shared primitive every
// network surface (REST, gRPC, Arrow Flight, pgwire) uses to reconcile an
// asserted tenant against an authenticated binding under the deployment
// HeaderTrustPolicy, plus the ADR-031 stable-id resolver port.
// ============================================================================
pub mod identity_trust;
pub use identity_trust::{
    AuthClass, AuthenticatedTenantBinding, GATEWAY_ROLE, HeaderTrustPolicy, OPERATOR_ROLE,
    ReadEnforcementComposition, ResolvedRequestIdentity, ResolvedTenantAssertion,
    TenantAssertionError, TenantStableIdResolver, TierClaimRejection, read_enforcement_composition,
    resolve_subject_assertion, resolve_tenant_assertion, resolve_tier_claim,
};

// ============================================================================
// Claim vocabulary (TD-TENANT-3): the ONE set of wire names every ingress
// surface reads before calling the trust primitives above. The decision layer
// was already shared; the vocabulary was hand-rolled per surface and diverged
// (Flight accepted four tenant spellings, REST/gRPC one), so a claim honored
// on one port was silently ignored on another.
// ============================================================================
pub mod claim_vocabulary;
pub use claim_vocabulary::{
    ClaimHit, DEPRECATED_TENANT_CLAIM_ALIASES, TENANT_CLAIM_HEADER, TIER_CLAIM_HEADER,
    TIER_CLAIM_PG_PARAM, legacy_alias_present, tenant_claim, tenant_claim_with_legacy_aliases,
    tier_claim, tier_claim_pg,
};

// ============================================================================
// Collection ownership: the one rule for which request tenant may resolve a
// stored collection (search, get/scan, collection GET/list).
// ============================================================================
pub mod ownership;
pub use ownership::{
    CollectionOwner, clear_collection_owner, collection_owner, collection_visible_to,
    effective_owner, stamp_collection_owner,
};
