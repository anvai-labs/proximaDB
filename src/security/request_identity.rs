// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! Shared request-identity helpers (TD-ABAC-6) — the root-crate half of the
//! unified identity seam.
//!
//! The foundation half (`AuthClass`, `ResolvedRequestIdentity`,
//! `resolve_subject_assertion`) lives in `proximadb_tenant::identity_trust`
//! (tenant/subject ids are bare strings there — no catalog dep). This module
//! holds the pieces that MUST live in the root crate because they name
//! `SecurityCoordinator` / `AuthenticationData` (security): the credential parser
//! (`parse_authorization`, PR-A) and the identity orchestrator
//! (`resolve_request_identity`, PR-B). The `&str → SubjectId` lift for the ABAC
//! boundary lands with the read-path wiring.

use crate::security::AuthenticationData;
use crate::security::SecurityCoordinator;
use crate::security::UnifiedUserContext;
use proximadb_tenant::identity_trust::{
    AuthenticatedTenantBinding, ResolvedTenantAssertion, TenantAssertionError,
    resolve_subject_assertion, resolve_tenant_assertion,
};
use proximadb_tenant::{
    AuthClass, HeaderTrustPolicy, ResolveRequestTenantError, ResolvedRequestIdentity,
    TenantDeploymentMode, resolve_request_tenant_for_mode,
};

/// Parse an `authorization` header/metadata **value** into
/// [`AuthenticationData`] — the ONE credential parser every network surface
/// reuses.
///
/// Collapses three near-identical copies: gRPC `auth_data_from_headers`
/// (`src/network/grpc/auth.rs`), Arrow `auth_data_from_metadata`
/// (`src/network/arrow_ipc/service.rs`), and REST `map_header_to_auth_data`
/// (`src/network/auth/middleware.rs`) — all three implemented the same
/// `Bearer` / `API-Key` / `Api-Key` / raw-as-ApiKey prefix logic.
///
/// Each surface's adapter extracts the raw value from its transport
/// (`http::HeaderMap` / `tonic::MetadataMap` / pgwire startup params) and keeps
/// only its surface-specific concerns: required-vs-optional semantics and any
/// extra fallbacks (e.g. Arrow's mTLS peer-cert and `x-api-key`/`api-key`
/// header fallbacks). The shared scheme-parsing lives here.
pub fn parse_authorization(value: &str) -> AuthenticationData {
    if let Some(token) = value.strip_prefix("Bearer ") {
        AuthenticationData::JWTToken(token.to_string())
    } else if let Some(key) = value
        .strip_prefix("API-Key ")
        .or_else(|| value.strip_prefix("Api-Key "))
    {
        AuthenticationData::ApiKey(key.to_string())
    } else {
        // No recognized scheme → treat the raw value as an API key. This is the
        // legacy behavior all three surfaces had; preserving it keeps the
        // consolidation behavior-neutral.
        AuthenticationData::ApiKey(value.to_string())
    }
}

/// The full result of [`resolve_request_identity`]: the uniform identity plus the
/// underlying `UnifiedUserContext` (when authenticated), so each surface can still
/// extract its surface-specific concerns (e.g. `DataPlaneCapability`, roles) that
/// are genuinely per-surface and not part of the shared identity contract.
pub struct ResolvedIdentity {
    /// The uniform tenant + subject + auth-class identity.
    pub identity: ResolvedRequestIdentity,
    /// The credential-resolved user context (None on the trust-asserted path).
    pub user_context: Option<UnifiedUserContext>,
}

/// Why request-identity resolution failed. Surfaces map this onto their own
/// protocol error vocabulary (HTTP 401/403, gRPC `UNAUTHENTICATED`/`PERMISSION_DENIED`,
/// SQLSTATE 28000).
#[derive(Debug, thiserror::Error)]
pub enum IdentityError {
    /// The credential was rejected by the `SecurityCoordinator`.
    #[error("authentication failed: {0}")]
    Authentication(String),
    /// A tenant or subject assertion was rejected by the deployment trust policy.
    /// Carries the structured [`TenantAssertionError`] so each surface can emit
    /// its own `tenant_audit` trail with full fidelity (Mismatch vs
    /// UnauthenticatedAssertionRejected) and map it onto its protocol's error
    /// vocabulary — the orchestrator itself never logs.
    #[error("identity assertion rejected: {0}")]
    Assertion(TenantAssertionError),
    /// The resolved tenant is invalid for the deployment mode (e.g. missing under
    /// `MultiTenant`, or an unsafe id).
    #[error(transparent)]
    TenantResolution(#[from] ResolveRequestTenantError),
}

/// The ONE identity resolver every network surface calls (TD-ABAC-6). Orchestrates
/// the credential → `authenticate_request` → tenant/subject trust-gate sequence that
/// gRPC, Arrow, REST, and pgwire each re-implemented. Each surface supplies only a
/// thin transport adapter (extract the credential + the asserted tenant/subject from
/// its transport) and maps [`IdentityError`] onto its protocol's error vocabulary.
///
/// Two paths, distinguished by whether a credential was verified:
/// * **Authenticated** (`coordinator` + `credential`): the credential is the proof.
///   Tenant is reconciled against the credential binding; subject = `user_id` (no
///   assertion gate — the credential IS the proof); `AuthClass::Authenticated`.
/// * **Trust-asserted** (no credential, e.g. pgwire trust auth or a no-coordinator
///   dev/embedded path): tenant and subject are client-asserted, both gated through
///   [`HeaderTrustPolicy`] (the same gate TD-TENANT-1 uses for tenant); `AuthClass::TrustAsserted`,
///   or `Anonymous` when nothing was asserted.
pub async fn resolve_request_identity(
    coordinator: Option<&SecurityCoordinator>,
    credential: Option<AuthenticationData>,
    asserted_tenant: Option<&str>,
    asserted_subject: Option<&str>,
    trust: HeaderTrustPolicy,
    mode: &TenantDeploymentMode,
    stable_id_resolver: Option<&dyn proximadb_tenant::TenantStableIdResolver>,
) -> Result<ResolvedIdentity, IdentityError> {
    let user_context: Option<UnifiedUserContext> = match (coordinator, credential) {
        (Some(coordinator), Some(credential)) => Some(
            coordinator
                .authenticate_request(credential)
                .await
                .map_err(|e| IdentityError::Authentication(e.to_string()))?,
        ),
        _ => None,
    };

    match user_context {
        // Authenticated: tenant reconciled against the credential binding; subject
        // is credential-derived (no assertion gate).
        Some(user_context) => {
            let binding =
                user_context
                    .tenant_id
                    .as_ref()
                    .map(|tenant_id| AuthenticatedTenantBinding {
                        tenant_id: tenant_id.clone(),
                        is_gateway_principal: user_context.is_gateway_principal(),
                    });
            let resolved = resolve_tenant_assertion(asserted_tenant, binding.as_ref(), trust)
                .map_err(IdentityError::Assertion)?;
            let tenant = resolve_tenant_for_mode(&resolved, mode)?;
            Ok(ResolvedIdentity {
                identity: ResolvedRequestIdentity {
                    tenant,
                    subject: Some(user_context.user_id.clone()),
                    auth_class: AuthClass::Authenticated,
                    tenant_stable_id: None,
                }
                .stamp_stable_id(stable_id_resolver),
                user_context: Some(user_context),
            })
        }
        // Trust-asserted (or anonymous): both tenant and subject are client-asserted
        // and gated through the deployment trust policy.
        None => {
            let resolved = resolve_tenant_assertion(asserted_tenant, None, trust)
                .map_err(IdentityError::Assertion)?;
            let tenant = resolve_tenant_for_mode(&resolved, mode)?;
            let subject = resolve_subject_assertion(asserted_subject, trust)
                .map_err(IdentityError::Assertion)?;
            let auth_class = match (&resolved, &subject) {
                (ResolvedTenantAssertion::NoTenant, None) => AuthClass::Anonymous,
                _ => AuthClass::TrustAsserted,
            };
            Ok(ResolvedIdentity {
                identity: ResolvedRequestIdentity {
                    tenant,
                    subject,
                    auth_class,
                    tenant_stable_id: None,
                }
                .stamp_stable_id(stable_id_resolver),
                user_context: None,
            })
        }
    }
}

/// Map a resolved tenant assertion to its effective string and apply the deployment
/// mode (single-tenant default / multi-tenant-required). Mirrors the sequence gRPC
/// and Arrow run inline today.
fn resolve_tenant_for_mode(
    resolved: &ResolvedTenantAssertion,
    mode: &TenantDeploymentMode,
) -> Result<String, IdentityError> {
    let tenant_str = match resolved {
        ResolvedTenantAssertion::Asserted(t) | ResolvedTenantAssertion::Credential(t) => {
            Some(t.as_str())
        }
        ResolvedTenantAssertion::NoTenant => None,
    };
    Ok(resolve_request_tenant_for_mode(tenant_str, mode)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bearer_prefix_is_jwt() {
        assert!(matches!(
            parse_authorization("Bearer eyJabc.def.ghi"),
            AuthenticationData::JWTToken(t) if t == "eyJabc.def.ghi"
        ));
    }

    #[test]
    fn api_key_prefixes_are_api_key() {
        assert!(matches!(
            parse_authorization("API-Key secret_123"),
            AuthenticationData::ApiKey(k) if k == "secret_123"
        ));
        assert!(matches!(
            parse_authorization("Api-Key secret_456"),
            AuthenticationData::ApiKey(k) if k == "secret_456"
        ));
    }

    #[test]
    fn raw_value_falls_back_to_api_key() {
        // Legacy behavior: an unrecognized scheme is treated as a bare API key.
        assert!(matches!(
            parse_authorization("bare-key-no-scheme"),
            AuthenticationData::ApiKey(k) if k == "bare-key-no-scheme"
        ));
    }

    // --- resolve_request_identity (TD-ABAC-6) ---
    //
    // The Authenticated path needs a real SecurityCoordinator (covered by the
    // surface integration tests when wired); these unit tests cover the
    // trust-asserted path (coordinator=None), which is the pgwire/dev parity case.

    fn single_tenant_mode() -> TenantDeploymentMode {
        TenantDeploymentMode::SingleTenant {
            default_tenant: "default-tenant".to_string(),
        }
    }

    #[tokio::test]
    async fn trust_asserted_open_accepts_tenant_and_subject() {
        let resolved = resolve_request_identity(
            None,
            None,
            Some("acme"),
            Some("alice"),
            HeaderTrustPolicy::Open,
            &single_tenant_mode(),
            None,
        )
        .await
        .expect("Open accepts bare assertions");

        assert_eq!(resolved.identity.tenant, "acme");
        assert_eq!(resolved.identity.subject.as_deref(), Some("alice"));
        assert_eq!(resolved.identity.auth_class, AuthClass::TrustAsserted);
        assert!(resolved.user_context.is_none());
    }

    #[tokio::test]
    async fn trust_asserted_strict_rejects_a_subject_assertion() {
        let result = resolve_request_identity(
            None,
            None,
            Some("acme"),
            Some("alice"),
            HeaderTrustPolicy::AuthenticatedOnly,
            &single_tenant_mode(),
            None,
        )
        .await;
        let Err(err) = result else {
            panic!("strict policy must reject a bare subject assertion");
        };
        assert!(matches!(err, IdentityError::Assertion(_)));
    }

    #[tokio::test]
    async fn nothing_asserted_is_anonymous_with_default_tenant() {
        // No coordinator, no assertions ⇒ Anonymous; SingleTenant supplies its
        // default tenant.
        let resolved = resolve_request_identity(
            None,
            None,
            None,
            None,
            HeaderTrustPolicy::Open,
            &single_tenant_mode(),
            None,
        )
        .await
        .expect("anonymous resolves");

        assert_eq!(resolved.identity.tenant, "default-tenant");
        assert!(resolved.identity.subject.is_none());
        assert_eq!(resolved.identity.auth_class, AuthClass::Anonymous);
    }

    #[tokio::test]
    async fn multitenant_without_a_tenant_is_rejected() {
        let result = resolve_request_identity(
            None,
            None,
            None,
            None,
            HeaderTrustPolicy::Open,
            &TenantDeploymentMode::MultiTenant,
            None,
        )
        .await;
        let Err(err) = result else {
            panic!("MultiTenant must reject a request with no tenant");
        };
        assert!(matches!(err, IdentityError::TenantResolution(_)));
    }
}
