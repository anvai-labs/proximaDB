/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Tenant extraction middleware for multi-tenant isolation
//!
//! This middleware extracts tenant_id from incoming requests and injects it
//! into the request extensions for downstream handlers to access.
//!
//! # Tenant ID Sources (Priority Order)
//!
//! 1. **HTTP Header**: `X-Tenant-ID` header (explicit tenant selection)
//! 2. **JWT Claims**: `tenant_id` claim in authenticated JWT token
//! 3. **API Key Mapping**: Tenant associated with the API key
//! 4. **Default Tenant**: Configurable fallback for single-tenant deployments
//!
//! # Usage
//!
//! ```rust,ignore
//! use proximadb::network::middleware::tenant::{TenantExtractor, MiddlewareTenantContext};
//!
//! // Create tenant extractor with optional default tenant
//! let extractor = TenantExtractor::new()
//!     .with_default_tenant("default");
//!
//! // In handlers, access tenant context:
//! async fn handler(Extension(tenant): Extension<MiddlewareTenantContext>) -> impl IntoResponse {
//!     println!("Request from tenant: {}", tenant.tenant_id);
//! }
//! ```

use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::sync::Arc;
use tracing::{debug, warn};

/// HTTP header name for explicit tenant ID, in canonical display casing — the
/// spelling documented in the OpenAPI spec and used by REST clients.
///
/// TD-TENANT-3: this is a *display* constant. The wire lookup goes through the
/// shared claim vocabulary (`proximadb_tenant::TENANT_CLAIM_HEADER`), which is
/// the same header — `http::HeaderName` compares case-insensitively and HTTP/2
/// mandates lowercase on the wire. `x_tenant_id_matches_shared_vocabulary`
/// pins the two together so they cannot drift.
pub const X_TENANT_ID: &str = "X-Tenant-ID";

/// Tenant context extracted from request
///
/// This struct is injected into request extensions and can be accessed
/// by handlers using `Extension<TenantContext>`.
///
/// Backwards-compat alias for [`MiddlewareTenantContext`].
pub type TenantContext = MiddlewareTenantContext;

#[derive(Debug, Clone)]
pub struct MiddlewareTenantContext {
    /// The tenant identifier
    pub tenant_id: String,
    /// Owning customer **account** — the SaaS billing/isolation boundary above
    /// `tenant_id` (a tenant is a workspace/sub-org inside an account). `None`
    /// is single-account / default mode (resolve via
    /// [`account_or_default`](Self::account_or_default)); the account tier is
    /// inert until provisioned (Phase 5 two-tier operator/account model).
    pub account_id: Option<String>,
    /// Source of the tenant ID (for audit logging)
    pub source: TenantIdSource,
    /// Whether this is a system/admin tenant with elevated privileges. Doubles
    /// as the operator (control-plane) marker.
    pub is_system_tenant: bool,
    /// ADR-031 stable `u64` id of the tenant, resolved at the identity
    /// boundary when a [`proximadb_tenant::TenantStableIdResolver`] is wired
    /// on the extractor. `None` = not resolved / not yet minted — always an
    /// optimization for catalog/storage keying, never a second source of
    /// truth (the string `tenant_id` remains authoritative).
    pub tenant_stable_id: Option<u64>,
    /// ADR-074 S1: the resolved namespace id — the legacy string form of the
    /// `data/{tenant}/{namespace_id}/...` path tier. `None` = single-tenant /
    /// default (resolve via [`namespace_or_default`](Self::namespace_or_default)).
    /// Threaded to `StorageTenantContext` via the `From` bridge; populated from
    /// the alias→id resolution seam in S2/S3. NOTE: the stable numeric
    /// [`namespace_stable_id`](Self::namespace_stable_id) is the AUTHORITATIVE
    /// identity (rename-safe); this string is the derived legacy-compat form
    /// carried so the legacy string-resolved path can read existing data
    /// (mixed-read-safe, deprecated once typed paths go default-on).
    pub namespace_id: Option<String>,
    /// ADR-031 stable numeric namespace id (`NamespaceId = u16`). Threaded but
    /// unused until `PROXIMADB_TYPED_PATHS`; `None` = legacy string-resolved path.
    pub namespace_stable_id: Option<u16>,
    /// The authenticated principal's subject id (#1338 / TD-ABAC-6). Surfaced
    /// from `UnifiedUserContext.user_id` so REST read paths can thread it as
    /// the ABAC principal — the direct analog of gRPC's `user_id` accessor and
    /// Arrow's `AuthenticatedFlightContext.user_id`. `None` on the
    /// trust-asserted / unauthenticated path. Per-handler consumption (#1309)
    /// is wired separately; this field is the surfacing.
    pub subject: Option<String>,
}

impl MiddlewareTenantContext {
    /// Create a new tenant context
    pub fn new(tenant_id: impl Into<String>, source: TenantIdSource) -> Self {
        let tenant_id = tenant_id.into();
        let is_system_tenant = tenant_id == "system" || tenant_id == "admin";
        Self {
            tenant_id,
            account_id: None,
            source,
            is_system_tenant,
            tenant_stable_id: None,
            namespace_id: None,
            namespace_stable_id: None,
            subject: None,
        }
    }

    /// Create a default/anonymous tenant context. Uses the ONE canonical default
    /// (`proximadb_tenant::DEFAULT_TENANT`) shared by every surface.
    pub fn default_tenant() -> Self {
        Self::new(proximadb_tenant::DEFAULT_TENANT, TenantIdSource::Default)
    }

    /// Set the owning customer account (Phase 5). Activates the account-rooted
    /// isolation tree for this request; leaving it unset keeps single-account
    /// / default behaviour.
    pub fn with_account(mut self, account_id: impl Into<String>) -> Self {
        self.account_id = Some(account_id.into());
        self
    }

    /// The owning account ID, falling back to the reserved default-account ID
    /// when none was provisioned. Use at I/O boundaries that always need an
    /// account segment (e.g. `DrPathBuilder` account-rooted paths).
    pub fn account_or_default(&self) -> &str {
        self.account_id.as_deref().unwrap_or(
            crate::storage::trait_components::path_resolver::DrPathBuilder::DEFAULT_ACCOUNT_ID,
        )
    }

    /// Set the resolved namespace (ADR-074 S1). `namespace_stable_id` is the
    /// AUTHORITATIVE identity (rename-safe); `namespace_id` is the derived
    /// string form carried for the legacy string-resolved path (mixed-read-safe;
    /// deprecated when typed paths go default-on). Both are populated by the
    /// alias→id resolution seam in S2/S3.
    pub fn with_namespace(
        mut self,
        namespace_id: impl Into<String>,
        namespace_stable_id: u16,
    ) -> Self {
        self.namespace_id = Some(namespace_id.into());
        self.namespace_stable_id = Some(namespace_stable_id);
        self
    }

    /// The namespace id, falling back to the reserved default-namespace id when
    /// none was resolved (single-tenant / OSS). Use at I/O boundaries that render
    /// the legacy string path `data/{tenant}/{namespace_id}/…`.
    pub fn namespace_or_default(&self) -> &str {
        self.namespace_id.as_deref().unwrap_or(
            crate::storage::trait_components::path_resolver::DrPathBuilder::DEFAULT_NAMESPACE_ID,
        )
    }

    /// Check if this is a valid tenant (not empty)
    pub fn is_valid(&self) -> bool {
        !self.tenant_id.is_empty()
    }
}

/// ADR-074 S1: the **single identity-boundary seam**. Converts the network-layer
/// [`MiddlewareTenantContext`] into the storage-layer
/// [`StorageTenantContext`], carrying ALL identity tiers (tenant, account,
/// `tenant_stable_id`, `namespace_id`, `namespace_stable_id`) — replacing the
/// ad-hoc `StorageTenantContext::for_tenant_id(string)` re-derivation scattered
/// across ~18 protocol boundaries, which dropped account/stable/namespace on
/// the floor. Protocol boundaries should call `StorageTenantContext::from(ctx)`
/// (or `ctx.into()`) instead of `for_tenant_id`; S2/S3 populate the namespace
/// fields from the alias→stable-id resolution seam.
///
/// NOTE on authority (ADR-074): the stable numeric ids (`tenant_stable_id`,
/// `namespace_stable_id`) are the rename-safe source of truth; the string
/// `tenant_id`/`namespace_id` are derived legacy-compat forms carried so the
/// legacy string-resolved path can read existing data (mixed-read-safe). They
/// are deprecated for new code once typed paths go default-on.
///
/// [`StorageTenantContext`]: crate::storage::tenant::context::StorageTenantContext
impl From<&MiddlewareTenantContext> for crate::storage::tenant::context::StorageTenantContext {
    fn from(ctx: &MiddlewareTenantContext) -> Self {
        let mut s = Self::for_tenant_id(ctx.tenant_id.clone());
        s.account_id = ctx.account_id.clone();
        s.tenant_stable_id = ctx.tenant_stable_id;
        s.namespace_id = ctx.namespace_id.clone();
        s.namespace_stable_id = ctx.namespace_stable_id;
        s
    }
}

/// ADR-087: how this REST context's identity was established, mapped onto the
/// foundation [`proximadb_tenant::AuthClass`]. A present `subject` is always
/// credential-derived on REST (`authenticated_subject` reads the verified
/// `UnifiedUserContext`, #1338) ⇒ `Authenticated`; otherwise the tenant source
/// decides: bare header assertion ⇒ `TrustAsserted`; credential-bound,
/// default-tenant, or system sources with no subject ⇒ `Anonymous`.
fn auth_class_of(ctx: &MiddlewareTenantContext) -> proximadb_tenant::AuthClass {
    use proximadb_tenant::AuthClass;
    if ctx.subject.is_some() {
        AuthClass::Authenticated
    } else {
        match ctx.source {
            TenantIdSource::Header => AuthClass::TrustAsserted,
            _ => AuthClass::Anonymous,
        }
    }
}

/// ADR-087 (TD-ABAC-8): the middleware→foundation bridge. The REST tenant
/// middleware inserts this as a request Extension so ANY crate (notably
/// `proximadb-api` handlers, which cannot name root-crate types) can consume
/// the one caller identity and project it (`PortIdentity::from(&identity)`).
impl From<&MiddlewareTenantContext> for proximadb_tenant::ResolvedRequestIdentity {
    fn from(ctx: &MiddlewareTenantContext) -> Self {
        Self {
            tenant: ctx.tenant_id.clone(),
            subject: ctx.subject.clone(),
            auth_class: auth_class_of(ctx),
            tenant_stable_id: ctx.tenant_stable_id,
        }
    }
}

/// TD-ABAC-7: the one bridge from the middleware identity to the port-seam
/// caller identity ([`proximadb_runtime::PortIdentity`]). REST handlers build
/// it with `PortIdentity::from(&tenant)` instead of hand-threading the
/// `{tenant_id, subject, tenant_stable_id}` triple per call site.
impl<'a> From<&'a MiddlewareTenantContext> for proximadb_runtime::PortIdentity<'a> {
    fn from(ctx: &'a MiddlewareTenantContext) -> Self {
        Self {
            tenant_id: Some(&ctx.tenant_id),
            subject: ctx.subject.as_deref(),
            tenant_stable_id: ctx.tenant_stable_id,
            auth_class: auth_class_of(ctx),
        }
    }
}

/// Source of the tenant ID (for audit and debugging)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TenantIdSource {
    /// Extracted from X-Tenant-ID header
    Header,
    /// Extracted from JWT token claims
    JwtClaim,
    /// Derived from API key mapping
    ApiKey,
    /// Selected by an authenticated system/gateway principal.
    GatewayDelegation,
    /// Default tenant (single-tenant mode or fallback)
    Default,
    /// System/internal request
    System,
}

/// The bare-header trust policy — MOVED to the foundation crate
/// (`proximadb_tenant::identity_trust`, TD-TENANT-1 follow-up) so pgwire,
/// gRPC, and Arrow Flight share the exact same reconciliation primitive.
/// Re-exported here so `crate::network::middleware::tenant::HeaderTrustPolicy`
/// stays a stable path.
pub use proximadb_tenant::identity_trust::HeaderTrustPolicy;

use proximadb_tenant::identity_trust::{
    AuthenticatedTenantBinding, ResolvedTenantAssertion, TenantAssertionError,
    resolve_tenant_assertion,
};

impl std::fmt::Display for TenantIdSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Header => write!(f, "header"),
            Self::JwtClaim => write!(f, "jwt"),
            Self::ApiKey => write!(f, "api_key"),
            Self::GatewayDelegation => write!(f, "gateway_delegation"),
            Self::Default => write!(f, "default"),
            Self::System => write!(f, "system"),
        }
    }
}

/// Configuration for tenant extraction
#[derive(Debug, Clone)]
pub struct TenantExtractorConfig {
    /// Default tenant ID for single-tenant deployments
    pub default_tenant: Option<String>,
    /// Whether to require tenant ID (reject requests without tenant)
    pub require_tenant: bool,
    /// Whether to validate tenant exists in TenantManager
    pub validate_tenant: bool,
    /// System tenant IDs that bypass normal validation
    pub system_tenants: Vec<String>,
    /// Trust policy for the bare `X-Tenant-ID` header (see
    /// [`HeaderTrustPolicy`]). Default `Open` — default-safe for existing
    /// deployments; `multi_tenant()` strict mode uses `AuthenticatedOnly`.
    pub header_trust: HeaderTrustPolicy,
    /// Trust policy for the client-supplied `X-Tenant-Tier` claim (ADR-0053
    /// W8). The claim names an *entitlement* (cache share / cost
    /// multiplier), so a client with direct network access could otherwise
    /// self-stamp `enterprise`. Dropped claims never fail the request —
    /// the tenant simply resolves its default tier. Default `Open`;
    /// `multi_tenant()` strict mode uses `AuthenticatedOnly`.
    pub tier_header_trust: HeaderTrustPolicy,
}

impl Default for TenantExtractorConfig {
    fn default() -> Self {
        Self {
            // The ONE canonical default shared by every surface (foundation).
            default_tenant: Some(proximadb_tenant::DEFAULT_TENANT.to_string()),
            require_tenant: false,  // Allow single-tenant mode by default
            validate_tenant: false, // Disable validation by default (enable in production)
            system_tenants: vec!["system".to_string(), "admin".to_string()],
            header_trust: HeaderTrustPolicy::Open,
            tier_header_trust: HeaderTrustPolicy::Open,
        }
    }
}

impl TenantExtractorConfig {
    /// Create config from the explicit foundation deployment-mode contract (TD-CAT-3).
    pub fn from_deployment_mode(mode: proximadb_tenant::TenantDeploymentMode) -> Self {
        match mode {
            proximadb_tenant::TenantDeploymentMode::SingleTenant { default_tenant } => {
                Self::single_tenant(default_tenant)
            }
            proximadb_tenant::TenantDeploymentMode::MultiTenant => Self::multi_tenant(),
        }
    }

    /// Create config for single-tenant deployment
    pub fn single_tenant(default_tenant: impl Into<String>) -> Self {
        Self {
            default_tenant: Some(default_tenant.into()),
            require_tenant: false,
            validate_tenant: false,
            system_tenants: vec!["system".to_string()],
            header_trust: HeaderTrustPolicy::Open,
            tier_header_trust: HeaderTrustPolicy::Open,
        }
    }

    /// Create config for multi-tenant deployment (strict mode)
    pub fn multi_tenant() -> Self {
        Self {
            default_tenant: None,
            require_tenant: true,
            validate_tenant: true,
            system_tenants: vec!["system".to_string(), "admin".to_string()],
            // Strict mode: a tenant asserted via bare header without an
            // authenticated binding is a masquerade vector, not an identity.
            // Relax per-deployment with PROXIMADB_TENANT_HEADER_TRUST=open
            // (trusted-gateway topologies should prefer gateway-only).
            header_trust: HeaderTrustPolicy::AuthenticatedOnly,
            // Same rationale for the tier claim: on a multi-tenant
            // deployment the cache entitlement must not be self-stampable by
            // an unauthenticated caller. Gateway topologies stamping tier
            // claims should set PROXIMADB_TIER_HEADER_TRUST=gateway-only.
            tier_header_trust: HeaderTrustPolicy::AuthenticatedOnly,
        }
    }

    /// Builder: Set default tenant
    pub fn with_default_tenant(mut self, tenant: impl Into<String>) -> Self {
        self.default_tenant = Some(tenant.into());
        self
    }

    /// Builder: Require tenant ID on all requests
    pub fn require_tenant(mut self, require: bool) -> Self {
        self.require_tenant = require;
        self
    }

    /// Builder: Validate tenant exists
    pub fn validate_tenant(mut self, validate: bool) -> Self {
        self.validate_tenant = validate;
        self
    }

    /// Builder: Set the bare-header trust policy.
    pub fn with_header_trust(mut self, policy: HeaderTrustPolicy) -> Self {
        self.header_trust = policy;
        self
    }

    /// Builder: Set the tier-claim trust policy (ADR-0053 W8).
    pub fn with_tier_header_trust(mut self, policy: HeaderTrustPolicy) -> Self {
        self.tier_header_trust = policy;
        self
    }

    /// Apply deployment env overrides. Called at server construction (NOT in
    /// constructors, so tests and embedded uses stay hermetic). Delegates to
    /// the ONE shared resolution (`HeaderTrustPolicy::effective`): env
    /// `PROXIMADB_TENANT_HEADER_TRUST` wins; an unparseable value tightens
    /// to `authenticated-only` (fail-closed) rather than silently weakening.
    /// The tier-claim gate resolves identically from its own
    /// `PROXIMADB_TIER_HEADER_TRUST` key (ADR-0053 W8).
    pub fn apply_env_overrides(mut self) -> Self {
        let (policy, warning) = HeaderTrustPolicy::effective(self.header_trust, None);
        if let Some(warning) = warning {
            warn!("{warning}");
        } else if policy != self.header_trust {
            tracing::info!(
                %policy,
                "tenant header-trust policy set from PROXIMADB_TENANT_HEADER_TRUST"
            );
        }
        self.header_trust = policy;

        let (policy, warning) = HeaderTrustPolicy::effective_tier(self.tier_header_trust, None);
        if let Some(warning) = warning {
            warn!("{warning}");
        } else if policy != self.tier_header_trust {
            tracing::info!(
                %policy,
                "tier-claim trust policy set from PROXIMADB_TIER_HEADER_TRUST"
            );
        }
        self.tier_header_trust = policy;
        self
    }
}

/// Tenant extractor state (shared across requests)
#[derive(Clone)]
pub struct TenantExtractor {
    config: TenantExtractorConfig,
    /// Optional TenantManager for validation
    tenant_manager: Option<Arc<crate::storage::tenant::TenantManager>>,
    /// Optional ADR-031 stable-id resolver: when wired, the middleware stamps
    /// `MiddlewareTenantContext::tenant_stable_id` at the identity boundary.
    stable_id_resolver: Option<Arc<dyn proximadb_tenant::TenantStableIdResolver>>,
}

impl TenantExtractor {
    /// Create new tenant extractor with default config
    pub fn new() -> Self {
        Self {
            config: TenantExtractorConfig::default(),
            tenant_manager: None,
            stable_id_resolver: None,
        }
    }

    /// Create tenant extractor with custom config
    pub fn with_config(config: TenantExtractorConfig) -> Self {
        Self {
            config,
            tenant_manager: None,
            stable_id_resolver: None,
        }
    }

    /// Set TenantManager for validation
    pub fn with_tenant_manager(
        mut self,
        manager: Arc<crate::storage::tenant::TenantManager>,
    ) -> Self {
        self.tenant_manager = Some(manager);
        self
    }

    /// Wire the ADR-031 tenant stable-id resolver (catalog-backed once tenant
    /// stable-id minting lands; see TD-TENANT-1 follow-ups).
    pub fn with_stable_id_resolver(
        mut self,
        resolver: Arc<dyn proximadb_tenant::TenantStableIdResolver>,
    ) -> Self {
        self.stable_id_resolver = Some(resolver);
        self
    }

    /// Extract tenant ID from request. Delegates the assertion-vs-binding
    /// reconciliation to the ONE shared primitive
    /// (`proximadb_tenant::resolve_tenant_assertion`, TD-TENANT-1) — the same
    /// call pgwire, gRPC, and Arrow Flight make — and maps the result onto
    /// the REST source vocabulary + default-tenant fallback.
    fn extract_tenant_id(
        &self,
        req: &Request,
    ) -> Result<Option<(String, TenantIdSource)>, TenantAssertionError> {
        // TD-TENANT-3: the shared claim vocabulary, not a per-surface literal.
        // REST reads the canonical name only — Arrow Flight's legacy aliases
        // are deliberately NOT honored here (narrow, never widen).
        let headers = req.headers();
        let requested_tenant = proximadb_tenant::tenant_claim(|name| {
            headers.get(name).and_then(|value| value.to_str().ok())
        })
        .map(|hit| hit.value);

        let authenticated = self.authenticated_tenant_binding(req);
        let binding = authenticated
            .as_ref()
            .map(|(binding, _source)| binding.clone());

        match resolve_tenant_assertion(
            requested_tenant,
            binding.as_ref(),
            self.config.header_trust,
        )? {
            ResolvedTenantAssertion::Asserted(tenant_id) => {
                let source = if let Some((binding, _)) = &authenticated {
                    debug!(
                        gateway = %binding.tenant_id,
                        acting_tenant = %tenant_id,
                        "gateway principal delegated tenant via X-Tenant-ID"
                    );
                    TenantIdSource::GatewayDelegation
                } else {
                    debug!("Extracted tenant_id from header: {}", tenant_id);
                    TenantIdSource::Header
                };
                Ok(Some((tenant_id, source)))
            }
            ResolvedTenantAssertion::Credential(tenant_id) => {
                debug!("Extracted tenant_id from authenticated context: {tenant_id}");
                let source = authenticated
                    .map(|(_, source)| source)
                    .unwrap_or(TenantIdSource::JwtClaim);
                Ok(Some((tenant_id, source)))
            }
            ResolvedTenantAssertion::NoTenant => {
                if let Some(ref default_tenant) = self.config.default_tenant {
                    debug!("Using default tenant: {}", default_tenant);
                    return Ok(Some((default_tenant.clone(), TenantIdSource::Default)));
                }
                Ok(None)
            }
        }
    }

    /// The authenticated tenant binding for this request, if any, with the
    /// gateway-capability flag the `GatewayOnly` delegation consults: the
    /// first-class principal marker (`UnifiedUserContext::is_gateway_principal`,
    /// stamped from credential data) OR — compat — the bound tenant being in
    /// the deployment's `system_tenants` list.
    fn authenticated_tenant_binding(
        &self,
        req: &Request,
    ) -> Option<(AuthenticatedTenantBinding, TenantIdSource)> {
        if let Some(user_context) = req
            .extensions()
            .get::<crate::security::UnifiedUserContext>()
            && let Some(tenant_id) = user_context.tenant_id.as_ref()
        {
            let source = if matches!(
                &user_context.auth_method,
                crate::security::UnifiedAuthMethod::ApiKey
            ) {
                TenantIdSource::ApiKey
            } else {
                TenantIdSource::JwtClaim
            };
            // N3 (adversarial review pass 2): the system_tenants compat
            // fallback is for LOCALLY-authenticated principals whose bound
            // tenant is operator-designated. An OIDC token's tenant comes
            // from an IdP CLAIM — a token asserting tenant "system"/"admin"
            // must not thereby become a gateway-class principal (that grants
            // cross-tenant delegation under gateway-only + the tier-claim
            // gate). Exclude OIDC-provenance principals from the fallback.
            let oidc_provenance = user_context
                .metadata
                .get("oidc")
                .is_some_and(|v| v == "true");
            let is_gateway_principal = user_context.is_gateway_principal()
                || (!oidc_provenance && self.config.system_tenants.contains(tenant_id));
            return Some((
                AuthenticatedTenantBinding {
                    tenant_id: tenant_id.clone(),
                    is_gateway_principal,
                },
                source,
            ));
        }

        if let Some(user_info) = req.extensions().get::<super::auth::UserInfo>()
            && let Some(ref tenant_id) = user_info.tenant_id
        {
            let is_gateway_principal = self.config.system_tenants.contains(tenant_id);
            return Some((
                AuthenticatedTenantBinding {
                    tenant_id: tenant_id.clone(),
                    is_gateway_principal,
                },
                TenantIdSource::ApiKey,
            ));
        }

        None
    }

    /// Read and gate the `X-Tenant-Tier` claim (ADR-0053 W8). The claim is
    /// honored only from callers the tier policy trusts; a rejected claim is
    /// DROPPED (warned once per tenant) — never a 4xx, since the claim is an
    /// entitlement hint and the request proceeds at the tenant's default
    /// tier. The registry stamp stays insert-only: rejection depends on the
    /// requester's auth class, not the claim value, so clearing on rejection
    /// would let any anonymous request strip a legitimately-stamped tenant
    /// (downgrade-DoS).
    fn gated_tier_claim(&self, req: &Request) -> Option<String> {
        let headers = req.headers();
        let claim = proximadb_tenant::tier_claim(|name| {
            headers.get(name).and_then(|value| value.to_str().ok())
        });

        let binding = self
            .authenticated_tenant_binding(req)
            .map(|(binding, _source)| binding);

        match proximadb_tenant::resolve_tier_claim(
            claim,
            binding.as_ref(),
            self.config.tier_header_trust,
        ) {
            Ok(tier) => tier,
            Err(rejection) => {
                let tenant = binding
                    .as_ref()
                    .map(|b| b.tenant_id.as_str())
                    .unwrap_or_default()
                    .to_string();
                warn_once_per_tenant("rest", &tenant, &rejection);
                None
            }
        }
    }

    /// Validate tenant exists and is active
    fn validate_tenant(&self, tenant_id: &str, source: TenantIdSource) -> bool {
        // System tenants bypass validation
        if self.config.system_tenants.contains(&tenant_id.to_string()) {
            return true;
        }

        // If validation is disabled, allow all
        if !self.config.validate_tenant {
            return true;
        }

        // Validate with TenantManager if available
        if let Some(ref manager) = self.tenant_manager {
            match manager.get_tenant(tenant_id) {
                Ok(tenant) => {
                    // Check tenant status
                    matches!(
                        tenant.status,
                        crate::storage::tenant::context::TenantStatus::Active
                    )
                }
                Err(e) => {
                    // Tenant not found or lookup failed
                    warn!("Failed to validate tenant {}: {}", tenant_id, e);
                    false
                }
            }
        } else if matches!(
            source,
            TenantIdSource::JwtClaim | TenantIdSource::ApiKey | TenantIdSource::GatewayDelegation
        ) {
            // Without a local tenant catalog, only a credential-bound tenant
            // identity remains authoritative. A bare header/default must not
            // become trusted merely because validation infrastructure is absent.
            true
        } else {
            warn!(
                tenant_id,
                source = %source,
                "Tenant validation requested but no TenantManager is configured"
            );
            false
        }
    }
}

impl Default for TenantExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Tenant extraction middleware function
///
/// This middleware extracts tenant_id from the request and injects
/// MiddlewareTenantContext into request extensions.
/// #1338 (TD-ABAC-6): the authenticated principal's user id, read from the auth
/// layer's `UnifiedUserContext` extension. Surfaced onto
/// [`MiddlewareTenantContext::subject`] so REST read paths can thread it as the
/// ABAC principal — the analog of gRPC's `user_id` accessor / Arrow's `user_id`
/// field. `None` on the trust-asserted / unauthenticated path (no auth layer, or
/// auth disabled).
fn authenticated_subject(req: &Request) -> Option<String> {
    req.extensions()
        .get::<crate::security::UnifiedUserContext>()
        .map(|user| user.user_id.clone())
}

/// Warn once per (surface, tenant, rejection) that a tier claim was dropped,
/// so a strict policy facing a stamping control plane cannot flood the log.
/// Shared by every surface's tier-claim gate (REST here; gRPC via
/// [`warn_tier_claim_dropped`]). The `tenant` is the *acting* tenant when the
/// surface knows it, else the binding's (or empty for anonymous claims).
fn warn_once_per_tenant(
    surface: &str,
    tenant: &str,
    rejection: &proximadb_tenant::TierClaimRejection,
) {
    // ADR-0053 W8: count EVERY drop (the warn below is once per tenant —
    // aggregate drop volume must not inherit that suppression).
    crate::network::middleware::claim_metrics::record_tier_claim_dropped(
        bounded_surface(surface),
        rejection,
    );

    static WARNED: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> =
        std::sync::OnceLock::new();
    let warned = WARNED.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()));
    if let Ok(mut set) = warned.lock()
        && set.insert(format!("{surface}:{tenant}:{rejection}"))
    {
        tracing::warn!(
            target: "proximadb::tenant_audit",
            surface,
            tenant,
            "dropped x-tenant-tier claim: {rejection} \
             (governed by PROXIMADB_TIER_HEADER_TRUST)"
        );
    }
}

/// Map a surface name to a bounded metric label. Callers pass in-tree
/// literals; the fallback keeps label cardinality structurally bounded if a
/// new surface is added without extending this match (the metric reads
/// "other" and clippy/PR review catches the mismatch).
fn bounded_surface(surface: &str) -> &'static str {
    match surface {
        "rest" => "rest",
        "grpc" => "grpc",
        "flight" => "flight",
        "pgwire" => "pgwire",
        _ => "other",
    }
}

/// Wrapper the other network surfaces' tier-claim gates call (gRPC).
pub(crate) fn warn_tier_claim_dropped(
    surface: &str,
    tenant: &str,
    rejection: &proximadb_tenant::TierClaimRejection,
) {
    warn_once_per_tenant(surface, tenant, rejection);
}

pub async fn tenant_middleware(
    axum::extract::State(extractor): axum::extract::State<TenantExtractor>,
    mut req: Request,
    next: Next,
) -> Response {
    // Extract tenant ID
    match extractor.extract_tenant_id(&req) {
        Ok(Some((tenant_id, source))) => {
            if let Err(err) = proximadb_tenant::validate_request_tenant(&tenant_id) {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("Invalid tenant id '{}': {err}", tenant_id),
                )
                    .into_response();
            }

            // Validate tenant if configured
            if !extractor.validate_tenant(&tenant_id, source) {
                return (
                    StatusCode::FORBIDDEN,
                    format!("Tenant '{}' is not valid or not active", tenant_id),
                )
                    .into_response();
            }

            // Open-core cache tier hook: if the control plane stamped a tier
            // claim (`X-Tenant-Tier`), record it in the process-global registry
            // the cache LimitsResolver reads. The id is opaque (commercial tier
            // names + their cache shares are operator config, not OSS code).
            // ADR-0053 W8: the claim is gated by `tier_header_trust` — an
            // untrusted claim is dropped (warned), never a 4xx.
            let tier_claim = extractor.gated_tier_claim(&req);

            // Inject tenant context into request extensions
            let mut context = MiddlewareTenantContext::new(tenant_id, source);
            // ADR-031: stamp the tenant's stable u64 id at the boundary when a
            // resolver is wired, so downstream catalog/storage keying can use
            // it without re-resolving per operation.
            if let Some(resolver) = &extractor.stable_id_resolver {
                context.tenant_stable_id = resolver.stable_id_of(&context.tenant_id);
            }
            if let Some(tier) = tier_claim {
                crate::services::record_store::set_tenant_tier(&context.tenant_id, tier);
            }
            // #1338 (TD-ABAC-6): surface the authenticated subject so REST read
            // paths can thread it as the ABAC principal (mirrors gRPC's
            // `user_id` accessor / Arrow's `user_id` field).
            context.subject = authenticated_subject(&req);
            // Also inject api-crate MiddlewareTenantContext for port-backed handlers in proximadb-api
            req.extensions_mut()
                .insert(proximadb_api::rest::TenantContext {
                    tenant_id: context.tenant_id.clone(),
                    tenant_stable_id: context.tenant_stable_id,
                });
            // ADR-087 (TD-ABAC-8): the ONE foundation identity, visible to every
            // crate. api-crate handlers consume this; the tenant-only api
            // TenantContext above is its retirement candidate.
            req.extensions_mut()
                .insert(proximadb_tenant::ResolvedRequestIdentity::from(&context));
            req.extensions_mut().insert(context);

            next.run(req).await
        }
        Ok(None) => {
            if extractor.config.require_tenant {
                // Tenant required but not provided
                (
                    StatusCode::BAD_REQUEST,
                    "Tenant ID required. Provide X-Tenant-ID header or authenticate with tenant-bound credentials.",
                ).into_response()
            } else {
                // No tenant required, use anonymous context
                let mut default_ctx = MiddlewareTenantContext::default_tenant();
                default_ctx.subject = authenticated_subject(&req);
                req.extensions_mut()
                    .insert(proximadb_api::rest::TenantContext {
                        tenant_id: default_ctx.tenant_id.clone(),
                        tenant_stable_id: default_ctx.tenant_stable_id,
                    });
                req.extensions_mut()
                    .insert(proximadb_tenant::ResolvedRequestIdentity::from(
                        &default_ctx,
                    ));
                req.extensions_mut().insert(default_ctx);
                next.run(req).await
            }
        }
        Err(TenantAssertionError::Mismatch {
            asserted,
            authenticated,
        }) => {
            // Audit trail: a credentialed principal asserted a DIFFERENT
            // tenant via header — the masquerade signature.
            warn!(
                target: "proximadb::tenant_audit",
                requested = %asserted,
                authenticated = %authenticated,
                source = %TenantIdSource::Header,
                "rejected X-Tenant-ID: does not match authenticated tenant binding"
            );
            (
                StatusCode::FORBIDDEN,
                format!(
                    "Tenant '{}' does not match authenticated tenant '{}'",
                    asserted, authenticated
                ),
            )
                .into_response()
        }
        Err(TenantAssertionError::UnauthenticatedAssertionRejected { asserted }) => {
            warn!(
                target: "proximadb::tenant_audit",
                requested = %asserted,
                source = %TenantIdSource::Header,
                policy = %extractor.config.header_trust,
                "rejected bare X-Tenant-ID without authenticated tenant binding"
            );
            (
                StatusCode::FORBIDDEN,
                format!(
                    "Tenant '{}' asserted via X-Tenant-ID without authenticated credentials; \
                     this deployment requires a tenant-bound credential (JWT or API key)",
                    asserted
                ),
            )
                .into_response()
        }
    }
}

/// Create tenant extractor with config for use with middleware
///
/// # Example
/// ```rust,ignore
/// use axum::middleware;
///
/// let extractor = create_tenant_extractor(TenantExtractorConfig::default());
/// let router = Router::new()
///     .route("/api/v1/collections", get(list_collections))
///     .layer(middleware::from_fn_with_state(extractor, tenant_middleware));
/// ```
pub fn create_tenant_extractor(config: TenantExtractorConfig) -> TenantExtractor {
    TenantExtractor::with_config(config)
}

/// Extension trait to easily get tenant context from request
pub trait TenantContextExt {
    /// Get tenant context from request extensions
    fn tenant_context(&self) -> Option<&MiddlewareTenantContext>;

    /// Get tenant ID or default
    fn tenant_id_or_default(&self) -> String;
}

impl TenantContextExt for Request {
    fn tenant_context(&self) -> Option<&MiddlewareTenantContext> {
        self.extensions().get::<MiddlewareTenantContext>()
    }

    fn tenant_id_or_default(&self) -> String {
        self.tenant_context()
            .map_or_else(|| "default".to_string(), |ctx| ctx.tenant_id.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tenant_context_creation() {
        let ctx = MiddlewareTenantContext::new("tenant1", TenantIdSource::Header);
        assert_eq!(ctx.tenant_id, "tenant1");
        assert_eq!(ctx.source, TenantIdSource::Header);
        assert!(!ctx.is_system_tenant);
        assert!(ctx.is_valid());
    }

    #[test]
    fn test_system_tenant_detection() {
        let ctx = MiddlewareTenantContext::new("system", TenantIdSource::Header);
        assert!(ctx.is_system_tenant);

        let ctx = MiddlewareTenantContext::new("admin", TenantIdSource::Header);
        assert!(ctx.is_system_tenant);

        let ctx = MiddlewareTenantContext::new("customer1", TenantIdSource::Header);
        assert!(!ctx.is_system_tenant);
    }

    #[test]
    fn test_default_tenant_context() {
        let ctx = MiddlewareTenantContext::default_tenant();
        assert_eq!(ctx.tenant_id, "default");
        assert_eq!(ctx.source, TenantIdSource::Default);
    }

    #[test]
    fn rest_default_uses_the_one_canonical_constant() {
        // The middleware default context AND the extractor config default both
        // derive from the single `proximadb_tenant::DEFAULT_TENANT`, so REST can
        // never drift from pgwire/gRPC (which resolve the same constant).
        assert_eq!(
            MiddlewareTenantContext::default_tenant().tenant_id,
            proximadb_tenant::DEFAULT_TENANT
        );
        assert_eq!(
            TenantExtractorConfig::default().default_tenant.as_deref(),
            Some(proximadb_tenant::DEFAULT_TENANT)
        );
    }

    #[test]
    fn test_account_tier_inert_until_set() {
        // Unset account → resolves to the reserved default-account ID.
        let ctx = MiddlewareTenantContext::new("tnt_acme", TenantIdSource::Header);
        assert!(ctx.account_id.is_none());
        assert_eq!(ctx.account_or_default(), "default");

        // Provisioned account → used verbatim.
        let ctx = ctx.with_account("acct_acme");
        assert_eq!(ctx.account_id.as_deref(), Some("acct_acme"));
        assert_eq!(ctx.account_or_default(), "acct_acme");
    }

    #[test]
    fn test_config_single_tenant() {
        let config = TenantExtractorConfig::single_tenant("my-tenant");
        assert_eq!(config.default_tenant, Some("my-tenant".to_string()));
        assert!(!config.require_tenant);
        assert!(!config.validate_tenant);
    }

    #[test]
    fn test_config_multi_tenant() {
        let config = TenantExtractorConfig::multi_tenant();
        assert!(config.default_tenant.is_none());
        assert!(config.require_tenant);
        assert!(config.validate_tenant);
        assert_eq!(
            config.header_trust,
            HeaderTrustPolicy::AuthenticatedOnly,
            "multi-tenant deployments must reject unbound tenant headers"
        );
    }

    #[test]
    fn config_from_deployment_mode_matches_explicit_contract() {
        let single = TenantExtractorConfig::from_deployment_mode(
            proximadb_tenant::TenantDeploymentMode::single_tenant("tenant_a"),
        );
        assert_eq!(single.default_tenant.as_deref(), Some("tenant_a"));
        assert!(!single.require_tenant);
        assert!(!single.validate_tenant);

        let multi = TenantExtractorConfig::from_deployment_mode(
            proximadb_tenant::TenantDeploymentMode::MultiTenant,
        );
        assert!(multi.default_tenant.is_none());
        assert!(multi.require_tenant);
        assert!(multi.validate_tenant);
    }

    #[test]
    fn test_tenant_id_source_display() {
        assert_eq!(TenantIdSource::Header.to_string(), "header");
        assert_eq!(TenantIdSource::JwtClaim.to_string(), "jwt");
        assert_eq!(TenantIdSource::ApiKey.to_string(), "api_key");
        assert_eq!(
            TenantIdSource::GatewayDelegation.to_string(),
            "gateway_delegation"
        );
        assert_eq!(TenantIdSource::Default.to_string(), "default");
        assert_eq!(TenantIdSource::System.to_string(), "system");
    }

    // ── HeaderTrustPolicy (bare X-Tenant-ID hardening) ──────────────────────

    /// Build a request with an optional bare X-Tenant-ID header and an
    /// optional authenticated tenant binding (UnifiedUserContext extension,
    /// as the auth middleware would inject it).
    fn trust_request(header: Option<&str>, authenticated: Option<&str>) -> Request {
        let mut builder = axum::http::Request::builder().uri("/api/v2/anything");
        if let Some(tenant) = header {
            builder = builder.header(X_TENANT_ID, tenant);
        }
        let mut req = builder.body(axum::body::Body::empty()).expect("request");
        if let Some(tenant) = authenticated {
            let mut ctx = crate::security::UnifiedUserContext::anonymous();
            ctx.tenant_id = Some(tenant.to_string());
            ctx.auth_method = crate::security::UnifiedAuthMethod::JWT;
            req.extensions_mut().insert(ctx);
        }
        req
    }

    fn extractor(policy: HeaderTrustPolicy) -> TenantExtractor {
        TenantExtractor::with_config(TenantExtractorConfig {
            header_trust: policy,
            ..TenantExtractorConfig::default()
        })
    }

    /// TD-TENANT-3: the REST display constant and the shared wire vocabulary
    /// must name the same header. `HeaderMap` compares case-insensitively, so a
    /// drift here would be invisible until a surface that does NOT
    /// case-normalize (tonic `MetadataMap`) silently stopped matching.
    #[test]
    fn x_tenant_id_matches_shared_vocabulary() {
        assert_eq!(
            X_TENANT_ID.to_ascii_lowercase(),
            proximadb_tenant::TENANT_CLAIM_HEADER
        );
    }

    /// TD-TENANT-3 "narrow, never widen": Arrow Flight's legacy tenant aliases
    /// must NOT become effective on REST. Honoring them here would make a
    /// previously-ignored header newly authoritative on a second surface.
    #[test]
    fn rest_ignores_arrow_flight_legacy_tenant_aliases() {
        for alias in proximadb_tenant::DEPRECATED_TENANT_CLAIM_ALIASES {
            let mut req = trust_request(None, None);
            req.headers_mut().insert(
                axum::http::HeaderName::from_static(alias),
                axum::http::HeaderValue::from_static("smuggled"),
            );
            let result = extractor(HeaderTrustPolicy::Open)
                .extract_tenant_id(&req)
                .expect("alias must not error, just be ignored");
            assert!(
                result.is_none_or(|(tenant, _)| tenant != "smuggled"),
                "{alias} must not assert a tenant on REST"
            );
        }
    }

    /// N3 (adversarial review pass 2): an OIDC-provenance principal whose
    /// tenant CLAIM lands in system_tenants must NOT become a gateway-class
    /// principal via the compat fallback — that would transfer delegation
    /// trust by claim value, exactly what allow_delegation_roles=false
    /// withholds from ordinary IdP principals.
    #[test]
    fn oidc_provenance_is_excluded_from_the_system_tenants_fallback() {
        let extractor = TenantExtractor::with_config(TenantExtractorConfig {
            header_trust: HeaderTrustPolicy::GatewayOnly,
            ..TenantExtractorConfig::default()
        });
        let mut ctx = crate::security::UnifiedUserContext::anonymous();
        ctx.tenant_id = Some("system".to_string()); // IdP-asserted claim value
        ctx.metadata.insert("oidc".to_string(), "true".to_string());
        let req = axum::http::Request::builder()
            .uri("/")
            .body(axum::body::Body::empty())
            .expect("req");
        let binding = extractor
            .authenticated_tenant_binding(&req)
            .map(|(b, _)| b)
            .unwrap_or(AuthenticatedTenantBinding {
                tenant_id: "system".into(),
                is_gateway_principal: true, // visible if the test regresses
            });
        // We must reach the binding path: insert the context into extensions.
        let mut req2 = axum::http::Request::builder()
            .uri("/")
            .body(axum::body::Body::empty())
            .expect("req");
        req2.extensions_mut().insert(ctx);
        if let Some((binding, _)) = extractor.authenticated_tenant_binding(&req2) {
            assert!(
                !binding.is_gateway_principal,
                "OIDC tenant claim must not confer gateway-class via system_tenants"
            );
        }
        let _ = binding; // first probe just built the fallback shape
    }

    #[test]
    fn open_accepts_bare_header() {
        let result = extractor(HeaderTrustPolicy::Open)
            .extract_tenant_id(&trust_request(Some("demo1"), None))
            .expect("open mode accepts bare header");
        assert_eq!(result, Some(("demo1".to_string(), TenantIdSource::Header)));
    }

    #[test]
    fn strict_validation_without_manager_accepts_only_credential_bound_identity() {
        let extractor = TenantExtractor::with_config(TenantExtractorConfig::multi_tenant());

        assert!(extractor.validate_tenant("acme", TenantIdSource::JwtClaim));
        assert!(extractor.validate_tenant("acme", TenantIdSource::ApiKey));
        assert!(extractor.validate_tenant("acme", TenantIdSource::GatewayDelegation));
        assert!(!extractor.validate_tenant("acme", TenantIdSource::Header));
        assert!(!extractor.validate_tenant("acme", TenantIdSource::Default));
    }

    #[test]
    fn authenticated_only_rejects_bare_header() {
        let err = extractor(HeaderTrustPolicy::AuthenticatedOnly)
            .extract_tenant_id(&trust_request(Some("demo1"), None))
            .expect_err("bare header must be rejected without a credential binding");
        assert_eq!(
            err,
            TenantAssertionError::UnauthenticatedAssertionRejected {
                asserted: "demo1".to_string()
            }
        );
    }

    #[test]
    fn gateway_only_rejects_bare_header() {
        let err = extractor(HeaderTrustPolicy::GatewayOnly)
            .extract_tenant_id(&trust_request(Some("demo1"), None))
            .expect_err("bare header must be rejected without a credential binding");
        assert!(matches!(
            err,
            TenantAssertionError::UnauthenticatedAssertionRejected { .. }
        ));
    }

    #[test]
    fn credential_derived_tenant_flows_in_every_mode() {
        for policy in [
            HeaderTrustPolicy::Open,
            HeaderTrustPolicy::AuthenticatedOnly,
            HeaderTrustPolicy::GatewayOnly,
        ] {
            let result = extractor(policy)
                .extract_tenant_id(&trust_request(None, Some("acme")))
                .expect("credential-derived tenant is always accepted");
            assert_eq!(
                result,
                Some(("acme".to_string(), TenantIdSource::JwtClaim)),
                "policy {policy}"
            );
        }
    }

    #[test]
    fn matching_header_and_binding_resolve_to_binding() {
        let result = extractor(HeaderTrustPolicy::AuthenticatedOnly)
            .extract_tenant_id(&trust_request(Some("acme"), Some("acme")))
            .expect("matching header+binding is fine");
        assert_eq!(result, Some(("acme".to_string(), TenantIdSource::JwtClaim)));
    }

    #[test]
    fn spoofed_header_with_credential_rejected_in_every_mode_for_non_gateway() {
        for policy in [
            HeaderTrustPolicy::Open,
            HeaderTrustPolicy::AuthenticatedOnly,
            HeaderTrustPolicy::GatewayOnly,
        ] {
            let err = extractor(policy)
                .extract_tenant_id(&trust_request(Some("victim"), Some("acme")))
                .expect_err("header != binding is the masquerade signature");
            assert_eq!(
                err,
                TenantAssertionError::Mismatch {
                    asserted: "victim".to_string(),
                    authenticated: "acme".to_string(),
                },
                "policy {policy}"
            );
        }
    }

    #[test]
    fn gateway_only_lets_system_principal_delegate_tenant() {
        // The gateway authenticates with a service credential bound to the
        // "system" tenant (in system_tenants) and stamps the end user's
        // tenant via X-Tenant-ID.
        let result = extractor(HeaderTrustPolicy::GatewayOnly)
            .extract_tenant_id(&trust_request(Some("demo1"), Some("system")))
            .expect("gateway delegation must be allowed");
        assert_eq!(
            result,
            Some(("demo1".to_string(), TenantIdSource::GatewayDelegation))
        );
    }

    /// TD-TENANT-1 follow-up: the first-class principal marker. A credential
    /// carrying the `gateway` role may delegate under `GatewayOnly` even when
    /// its bound tenant is NOT in `system_tenants` — the marker, not the
    /// tenant name, is the capability.
    #[test]
    fn gateway_role_marker_delegates_without_system_tenant_membership() {
        let mut req = trust_request(Some("demo1"), None);
        let mut ctx = crate::security::UnifiedUserContext::anonymous();
        ctx.tenant_id = Some("svc-gw".to_string());
        ctx.auth_method = crate::security::UnifiedAuthMethod::JWT;
        ctx.roles.push(proximadb_tenant::GATEWAY_ROLE.to_string());
        req.extensions_mut().insert(ctx);

        let result = extractor(HeaderTrustPolicy::GatewayOnly)
            .extract_tenant_id(&req)
            .expect("gateway-role principal must be allowed to delegate");
        assert_eq!(
            result,
            Some(("demo1".to_string(), TenantIdSource::GatewayDelegation))
        );

        // Same principal WITHOUT the marker → mismatch.
        let mut req = trust_request(Some("demo1"), None);
        let mut ctx = crate::security::UnifiedUserContext::anonymous();
        ctx.tenant_id = Some("svc-gw".to_string());
        ctx.auth_method = crate::security::UnifiedAuthMethod::JWT;
        req.extensions_mut().insert(ctx);
        let err = extractor(HeaderTrustPolicy::GatewayOnly)
            .extract_tenant_id(&req)
            .expect_err("non-gateway principal must not delegate");
        assert!(matches!(err, TenantAssertionError::Mismatch { .. }));
    }

    // ── Tier-claim gate (ADR-0053 W8) ───────────────────────────────────────

    /// A request with an `X-Tenant-Tier` claim and an optional authenticated
    /// binding (mirrors `trust_request` for the tier header).
    fn tier_request(claim: Option<&str>, authenticated: Option<&str>) -> Request {
        let mut req = trust_request(None, authenticated);
        if let Some(claim) = claim {
            req.headers_mut().insert(
                "x-tenant-tier",
                axum::http::HeaderValue::from_str(claim).expect("header value"),
            );
        }
        req
    }

    fn tier_extractor(policy: HeaderTrustPolicy) -> TenantExtractor {
        TenantExtractor::with_config(TenantExtractorConfig {
            tier_header_trust: policy,
            ..TenantExtractorConfig::default()
        })
    }

    // Unique tenant ids per test: TENANT_TIERS is a process-global insert-only
    // registry with no removal API, so shared ids would couple tests.
    const T_OPEN: &str = "tier-test-open-tenant";
    const T_AUTHD: &str = "tier-test-authd-tenant";
    const T_BOUND: &str = "tier-test-bound-tenant";
    const T_GW: &str = "tier-test-gateway-tenant";
    const T_GW_OK: &str = "tier-test-gateway-ok-tenant";

    #[test]
    fn tier_claim_open_records_stamp_for_bare_header() {
        let got = tier_extractor(HeaderTrustPolicy::Open)
            .gated_tier_claim(&tier_request(Some("enterprise"), None));
        assert_eq!(got.as_deref(), Some("enterprise"));
    }

    #[test]
    fn tier_claim_authenticated_only_drops_anonymous_claim() {
        let got = tier_extractor(HeaderTrustPolicy::AuthenticatedOnly)
            .gated_tier_claim(&tier_request(Some("enterprise"), None));
        assert_eq!(got, None, "anonymous self-stamp must be dropped");
    }

    #[test]
    fn tier_claim_authenticated_only_accepts_credential_bound_claim() {
        let got = tier_extractor(HeaderTrustPolicy::AuthenticatedOnly)
            .gated_tier_claim(&tier_request(Some("pro"), Some(T_BOUND)));
        assert_eq!(got.as_deref(), Some("pro"));
    }

    #[test]
    fn tier_claim_gateway_only_drops_plain_credential_claim() {
        let got = tier_extractor(HeaderTrustPolicy::GatewayOnly)
            .gated_tier_claim(&tier_request(Some("enterprise"), Some(T_GW)));
        assert_eq!(got, None, "a plain tenant credential must not self-assign");
    }

    #[test]
    fn tier_claim_gateway_only_accepts_gateway_principal_claim() {
        // The gateway topology: a service credential with the gateway role
        // stamps the END USER's tier claim — accepted even though the claim's
        // tier has nothing to do with the credential's tenant.
        let mut req = tier_request(Some("enterprise"), None);
        let mut ctx = crate::security::UnifiedUserContext::anonymous();
        ctx.tenant_id = Some("svc-gw".to_string());
        ctx.auth_method = crate::security::UnifiedAuthMethod::JWT;
        ctx.roles.push(proximadb_tenant::GATEWAY_ROLE.to_string());
        req.extensions_mut().insert(ctx);

        let got = tier_extractor(HeaderTrustPolicy::GatewayOnly).gated_tier_claim(&req);
        assert_eq!(got.as_deref(), Some("enterprise"));
    }

    #[test]
    fn tier_claim_absent_header_is_ok_in_every_mode() {
        for policy in [
            HeaderTrustPolicy::Open,
            HeaderTrustPolicy::AuthenticatedOnly,
            HeaderTrustPolicy::GatewayOnly,
        ] {
            assert_eq!(
                tier_extractor(policy).gated_tier_claim(&tier_request(None, None)),
                None,
                "no header, no gate — policy {policy}"
            );
        }
    }

    #[test]
    fn multi_tenant_strict_mode_defaults_tier_gate_to_authenticated_only() {
        assert_eq!(
            TenantExtractorConfig::multi_tenant().tier_header_trust,
            HeaderTrustPolicy::AuthenticatedOnly,
            "multi-tenant deployments must not accept self-stamped tier claims"
        );
        assert_eq!(
            TenantExtractorConfig::default().tier_header_trust,
            HeaderTrustPolicy::Open
        );
        assert_eq!(
            TenantExtractorConfig::single_tenant("t").tier_header_trust,
            HeaderTrustPolicy::Open
        );
    }

    #[test]
    fn non_gateway_modes_do_not_allow_system_delegation() {
        // Delegation is a GatewayOnly capability — Open/AuthenticatedOnly keep
        // the strict header==binding contract even for system principals.
        for policy in [
            HeaderTrustPolicy::Open,
            HeaderTrustPolicy::AuthenticatedOnly,
        ] {
            let err = extractor(policy)
                .extract_tenant_id(&trust_request(Some("demo1"), Some("system")))
                .expect_err("delegation requires gateway-only mode");
            assert!(
                matches!(err, TenantAssertionError::Mismatch { .. }),
                "policy {policy}"
            );
        }
    }

    #[test]
    fn no_header_no_credential_falls_back_to_default_in_every_mode() {
        for policy in [
            HeaderTrustPolicy::Open,
            HeaderTrustPolicy::AuthenticatedOnly,
            HeaderTrustPolicy::GatewayOnly,
        ] {
            let result = extractor(policy)
                .extract_tenant_id(&trust_request(None, None))
                .expect("default fallback is not affected by header trust");
            assert_eq!(
                result,
                Some((
                    proximadb_tenant::DEFAULT_TENANT.to_string(),
                    TenantIdSource::Default
                )),
                "policy {policy}"
            );
        }
    }

    #[test]
    fn header_trust_policy_parses_and_displays() {
        use std::str::FromStr;
        assert_eq!(
            HeaderTrustPolicy::from_str("open").unwrap(),
            HeaderTrustPolicy::Open
        );
        assert_eq!(
            HeaderTrustPolicy::from_str("authenticated-only").unwrap(),
            HeaderTrustPolicy::AuthenticatedOnly
        );
        assert_eq!(
            HeaderTrustPolicy::from_str("AUTHENTICATED_MATCH").unwrap(),
            HeaderTrustPolicy::AuthenticatedOnly
        );
        assert_eq!(
            HeaderTrustPolicy::from_str("gateway-only").unwrap(),
            HeaderTrustPolicy::GatewayOnly
        );
        assert!(HeaderTrustPolicy::from_str("everything-goes").is_err());
        assert_eq!(HeaderTrustPolicy::default(), HeaderTrustPolicy::Open);
        assert_eq!(
            HeaderTrustPolicy::AuthenticatedOnly.to_string(),
            "authenticated-only"
        );
    }

    /// End-to-end through the axum layer: the policy maps to real HTTP
    /// statuses (403 for a rejected bare header, 200 for the default-tenant
    /// fallback and for gateway delegation).
    #[tokio::test]
    async fn middleware_maps_header_trust_to_http_statuses() {
        use axum::{Router, routing::get};
        use tower::ServiceExt;

        let app = |policy: HeaderTrustPolicy| {
            Router::new().route("/probe", get(|| async { "ok" })).layer(
                axum::middleware::from_fn_with_state(extractor(policy), tenant_middleware),
            )
        };

        // Bare header under authenticated-only → 403.
        let denied = app(HeaderTrustPolicy::AuthenticatedOnly)
            .oneshot(
                axum::http::Request::builder()
                    .uri("/probe")
                    .header(X_TENANT_ID, "demo1")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(denied.status(), StatusCode::FORBIDDEN);

        // No header → default-tenant fallback still flows (200).
        let allowed = app(HeaderTrustPolicy::AuthenticatedOnly)
            .oneshot(
                axum::http::Request::builder()
                    .uri("/probe")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(allowed.status(), StatusCode::OK);

        // Bare header under open → 200 (legacy behavior preserved).
        let open = app(HeaderTrustPolicy::Open)
            .oneshot(
                axum::http::Request::builder()
                    .uri("/probe")
                    .header(X_TENANT_ID, "demo1")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(open.status(), StatusCode::OK);
    }

    /// #1338 (TD-ABAC-6): the authenticated principal's user_id is surfaced on
    /// `MiddlewareTenantContext.subject` — the REST analog of gRPC's `user_id`
    /// accessor and Arrow's `user_id` field — so read paths can thread it as the
    /// ABAC principal. Proves the surfacing (per-handler consumption is #1309).
    #[tokio::test]
    async fn tenant_context_surfaces_the_authenticated_subject() {
        use axum::{Extension, Router, routing::get};
        use tower::ServiceExt;

        // Simulate the auth layer injecting an authenticated UnifiedUserContext.
        let inject_user = axum::middleware::from_fn(|mut req: Request, next: Next| async move {
            let mut user = crate::security::UnifiedUserContext::anonymous();
            user.user_id = "alice".to_string();
            user.tenant_id = Some("acme".to_string());
            user.auth_method = crate::security::UnifiedAuthMethod::JWT;
            req.extensions_mut().insert(user);
            next.run(req).await
        });

        let app = Router::new()
            .route(
                "/probe",
                get(
                    |Extension(ctx): Extension<MiddlewareTenantContext>| async move {
                        ctx.subject.unwrap_or_else(|| "none".to_string())
                    },
                ),
            )
            .layer(axum::middleware::from_fn_with_state(
                extractor(HeaderTrustPolicy::Open),
                tenant_middleware,
            ))
            .layer(inject_user);

        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/probe")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        assert_eq!(&body[..], b"alice");
    }

    #[test]
    fn multi_tenant_strict_mode_defaults_to_authenticated_only() {
        assert_eq!(
            TenantExtractorConfig::multi_tenant().header_trust,
            HeaderTrustPolicy::AuthenticatedOnly
        );
        // Default + single_tenant stay Open (default-safe for existing deployments).
        assert_eq!(
            TenantExtractorConfig::default().header_trust,
            HeaderTrustPolicy::Open
        );
        assert_eq!(
            TenantExtractorConfig::single_tenant("t").header_trust,
            HeaderTrustPolicy::Open
        );
    }
}
