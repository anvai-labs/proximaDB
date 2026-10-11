/*
 * Copyright 2026 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! REST operator endpoints for **provisioning ABAC policy at runtime** (TD-ABAC
//! control-plane / PR-B).
//!
//! ABAC enforcement is wired on every read surface, and PR-A made the three
//! durable stores shareable so a write is hot-visible to the live enforcer — but
//! there was no way for an operator to provision policy in a running server.
//! These endpoints are that control plane: they write through the SAME
//! `Arc<FileSystem*>` handles the live enforcer reads (`AppState.abac_*`), so a
//! provision takes effect on the next read with no restart.
//!
//! ## Security posture
//!
//! * **Auth gate** ([`authorize_operator`]): every handler requires
//!   `SystemAdmin` ∪ `ConfigureSystem` (cluster-scope operator authority). The
//!   binding's `tenant_stable_id` carries data-plane isolation — two layers.
//! * **Resolve-at-write, never a raw u64.** The path carries a tenant *string*;
//!   the handler resolves it server-side via
//!   [`CatalogTenantStableIdResolver`](crate::security::CatalogTenantStableIdResolver).
//!   Provisioning mints-or-resolves the tenant through the catalog before the
//!   policy write. This is the consistency guarantee: the request path and this
//!   admin both derive the u64 from the ONE durable catalog registry, so they
//!   cannot drift and a tenant does not need a dummy table before policy setup.
//! * **Fail-closed by construction.** ABAC is deny-by-default; provisioning only
//!   *adds* permits. An unprovisioned tenant stays denied; a dangling
//!   `predicate_ref` resolves to the unsatisfiable deny (safe), so it is accepted.
//! * **Audit.** Every mutation emits `tracing::info!` at
//!   `target = "proximadb.abac_policy.audit"` with the operator identity.
//!
//! ## Wire format
//!
//! Request/response bodies use the domain types directly (`PolicyBinding`,
//! `Scope`, `Effect`, `FieldMask`, `AttrValue`, `FilterExpression`). These
//! serialize as externally-tagged **PascalCase** — e.g. `{"Table": 200}`,
//! `"Permit"`, `{"Str":"eng"}`. There is deliberately **no DTO translation
//! layer**: a security API must not carry a mapping that could mis-translate a
//! field, and native serde means what you PUT is exactly what the enforcer stores.
//!
//! ## Routes (registered in `operator_and_control_v2_routes`)
//!
//! | Method + Path | Body → store write |
//! |---|---|
//! | `PUT /api/v2/abac/policy-bindings/{tenant}/{object_id}` | `{scope, effect, predicate_ref?, field_mask?}` → upsert |
//! | `DELETE /api/v2/abac/policy-bindings/{tenant}/{object_id}` | remove → 204 |
//! | `POST /api/v2/abac/attribute-bindings` | `{subject_id, tenant, attrs}` → upsert |
//! | `PUT /api/v2/abac/predicate-objects/{object_id}` | body is a `FilterExpression` → register |
//! | `DELETE /api/v2/abac/predicate-objects/{object_id}` | revoke → 204 |

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::{
    Extension, Json,
    extract::{Path, State},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};

use proximadb_abac::{
    AttributeBinding, FileSystemAttributeAuthority, FileSystemPolicyBindingStore,
    FileSystemPredicateObjectStore, PolicyBindingStore, PredicateObjectStore,
};
use proximadb_catalog::fc_metamodel::{
    AttrValue, Effect, FieldMask, ObjectId, PolicyBinding, Scope, SubjectId,
};
use proximadb_filter_expression::FilterExpression;
use proximadb_tenant::TenantStableIdResolver;

use crate::network::rest::canonical::handlers::AppState;
use crate::security::CatalogTenantStableIdResolver;
use crate::security::rbac_service::{UnifiedPermission, UnifiedUserContext};

// ===========================================================================
// Operator auth gate (mirrors primary_pod.rs; ABAC-specific message)
// ===========================================================================

/// The error body for a failed operator gate or provisioning precondition.
/// Shape matches `primary_pod::OperatorErrorResponse`.
#[derive(Debug, Serialize)]
pub struct OperatorErrorResponse {
    pub error: &'static str,
    pub message: String,
    pub code: u16,
}

/// Authorize a system operator. Returns the operator's `user_id` (for audit) on
/// success; `Err` is directly `?`-propagable from any handler whose error type is
/// `(StatusCode, Json<OperatorErrorResponse>)`.
///
/// * `None` (auth middleware bypassed/misconfigured) → 401.
/// * present but lacks `SystemAdmin` ∪ `ConfigureSystem` → 403.
pub fn authorize_operator(
    user_context: Option<&UnifiedUserContext>,
) -> Result<String, (StatusCode, Json<OperatorErrorResponse>)> {
    let Some(ctx) = user_context else {
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(OperatorErrorResponse {
                error: "missing_auth_context",
                message: "auth context not present — middleware misconfigured".to_string(),
                code: 401,
            }),
        ));
    };

    let allowed = ctx
        .effective_permissions
        .contains(&UnifiedPermission::SystemAdmin)
        || ctx
            .effective_permissions
            .contains(&UnifiedPermission::ConfigureSystem);

    if allowed {
        Ok(ctx.user_id.clone())
    } else {
        Err((
            StatusCode::FORBIDDEN,
            Json(OperatorErrorResponse {
                error: "operator_permission_required",
                message: "ABAC admin endpoints require SystemAdmin or ConfigureSystem permission"
                    .to_string(),
                code: 403,
            }),
        ))
    }
}

// ===========================================================================
// Provisioning preconditions + domain error
// ===========================================================================

/// Why a provisioning operation could not complete. (Store unavailability is
/// surfaced directly as HTTP 503 by the `require_*` helpers below; this enum
/// carries only the resolve-at-write failure, which is the security-critical
/// precondition that must be unit-testable.)
#[derive(Debug, PartialEq, Eq)]
pub enum ProvisionError {
    /// The tenant has no minted stable id — it cannot be the target of a policy
    /// until something (a table under it) has triggered the catalog to mint its
    /// account id. Maps to 422.
    TenantUnresolved {
        /// The tenant string the operator supplied.
        tenant: String,
    },
    /// The authoritative catalog could not durably mint the mapping. Maps to
    /// 503; no policy state is written with an unstable key.
    TenantMintFailed { tenant: String, message: String },
}

/// Ensure the tenant has a durable stable id before a policy mutation. The
/// subsequent sync resolve in the provisioning core intentionally re-reads the
/// same authority, catching any wiring mismatch before the store write.
async fn ensure_tenant(
    resolver: &CatalogTenantStableIdResolver,
    tenant: &str,
) -> Result<u64, ProvisionError> {
    resolver
        .ensure_stable_id(tenant)
        .await
        .map_err(|error| ProvisionError::TenantMintFailed {
            tenant: tenant.to_string(),
            message: error.to_string(),
        })?
        .ok_or_else(|| ProvisionError::TenantUnresolved {
            tenant: tenant.to_string(),
        })
}

/// Resolve a tenant string to its stable u64 via `resolver`, or fail closed.
fn resolve_tenant(
    resolver: &dyn TenantStableIdResolver,
    tenant: &str,
) -> Result<u64, ProvisionError> {
    resolver
        .stable_id_of(tenant)
        .ok_or_else(|| ProvisionError::TenantUnresolved {
            tenant: tenant.to_string(),
        })
}

fn map_provision_err(e: ProvisionError) -> (StatusCode, Json<OperatorErrorResponse>) {
    match e {
        ProvisionError::TenantUnresolved { tenant } => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(OperatorErrorResponse {
                error: "tenant_unresolved",
                message: format!(
                    "tenant '{tenant}' has no stable id and the configured catalog did not mint \
                     one; use the native or system catalog authority"
                ),
                code: 422,
            }),
        ),
        ProvisionError::TenantMintFailed { tenant, message } => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(OperatorErrorResponse {
                error: "tenant_mint_failed",
                message: format!(
                    "tenant '{tenant}' stable id could not be durably minted: {message}"
                ),
                code: 503,
            }),
        ),
    }
}

/// 503 when ABAC is off or no durable store opened (e.g. no `data_dir`).
fn unavailable(
    error: &'static str,
    message: &'static str,
) -> (StatusCode, Json<OperatorErrorResponse>) {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(OperatorErrorResponse {
            error,
            message: message.to_string(),
            code: 503,
        }),
    )
}

fn require_binding_store(
    state: &AppState,
) -> Result<&Arc<FileSystemPolicyBindingStore>, (StatusCode, Json<OperatorErrorResponse>)> {
    state.abac_binding_store.as_ref().ok_or_else(|| {
        unavailable(
            "abac_unavailable",
            "ABAC policy-binding store is not available (abac-policy off or no data_dir)",
        )
    })
}

fn require_authority(
    state: &AppState,
) -> Result<&Arc<FileSystemAttributeAuthority>, (StatusCode, Json<OperatorErrorResponse>)> {
    state.abac_authority.as_ref().ok_or_else(|| {
        unavailable(
            "abac_unavailable",
            "ABAC attribute authority is not available (abac-policy off or no data_dir)",
        )
    })
}

fn require_predicate_store(
    state: &AppState,
) -> Result<&Arc<FileSystemPredicateObjectStore>, (StatusCode, Json<OperatorErrorResponse>)> {
    state.abac_predicate_store.as_ref().ok_or_else(|| {
        unavailable(
            "abac_unavailable",
            "ABAC predicate-object store is not available (abac-policy off or no data_dir)",
        )
    })
}

// ===========================================================================
// Request / response DTOs
// ===========================================================================

/// Body for `PUT /api/v2/abac/policy-bindings/{tenant}/{object_id}`.
#[derive(Debug, Deserialize)]
pub struct PutPolicyBindingRequest {
    pub scope: Scope,
    pub effect: Effect,
    /// The predicate object this binding carries (a row-level rule). Omit for a
    /// predicate-free table-level grant. Dangling refs resolve fail-closed.
    #[serde(default)]
    pub predicate_ref: Option<ObjectId>,
    #[serde(default)]
    pub field_mask: Option<FieldMask>,
}

/// Body for `POST /api/v2/abac/attribute-bindings`.
#[derive(Debug, Deserialize)]
pub struct PostAttributeBindingRequest {
    pub subject_id: String,
    /// Tenant display name; resolved to the stable u64 server-side (never raw).
    pub tenant: String,
    pub attrs: BTreeMap<String, AttrValue>,
}

/// Response for `PUT /api/v2/abac/predicate-objects/{object_id}`.
#[derive(Debug, Serialize)]
pub struct PredicateObjectResponse {
    pub object_id: ObjectId,
    pub expression: FilterExpression,
}

// ── Read (GET) response DTOs ──

/// Response for `GET /api/v2/abac/policy-bindings/{tenant}` — the tenant's live
/// policy (the same set the enforcer composes for every read).
#[derive(Debug, Serialize)]
pub struct PolicyBindingsResponse {
    pub tenant: String,
    pub tenant_stable_id: u64,
    pub count: usize,
    pub bindings: Vec<PolicyBinding>,
}

/// Response for `GET /api/v2/abac/attribute-bindings` — every authority binding
/// (cluster-operator scope; cross-tenant by design).
#[derive(Debug, Serialize)]
pub struct AttributeBindingsResponse {
    pub count: usize,
    pub bindings: Vec<AttributeBinding>,
}

/// Response for `GET /api/v2/abac/predicate-objects` — every registered
/// predicate object.
#[derive(Debug, Serialize)]
pub struct PredicateObjectsResponse {
    pub count: usize,
    pub objects: Vec<PredicateObjectResponse>,
}

// ===========================================================================
// Testable provisioning cores (sync; take the shared handles + a resolver)
// ===========================================================================

/// Resolve the tenant, build the `PolicyBinding`, upsert it through the shared
/// store (hot-reload: the live enforcer reads this same instance). Returns the
/// stored binding.
fn provision_policy_binding(
    store: &Arc<FileSystemPolicyBindingStore>,
    resolver: &dyn TenantStableIdResolver,
    tenant: &str,
    object_id: ObjectId,
    req: PutPolicyBindingRequest,
) -> Result<PolicyBinding, ProvisionError> {
    let tenant_stable_id = resolve_tenant(resolver, tenant)?;
    let binding = PolicyBinding {
        object_id,
        tenant_stable_id,
        scope: req.scope,
        effect: req.effect,
        predicate_ref: req.predicate_ref,
        field_mask: req.field_mask,
    };
    store.upsert(binding.clone());
    Ok(binding)
}

/// Resolve the tenant, build the `AttributeBinding`, upsert it through the shared
/// authority.
fn provision_attribute_binding(
    authority: &Arc<FileSystemAttributeAuthority>,
    resolver: &dyn TenantStableIdResolver,
    subject_id: &str,
    tenant: &str,
    attrs: BTreeMap<String, AttrValue>,
) -> Result<AttributeBinding, ProvisionError> {
    let tenant_stable_id = resolve_tenant(resolver, tenant)?;
    let binding = AttributeBinding {
        subject_id: SubjectId(subject_id.to_string()),
        tenant_stable_id,
        attrs,
    };
    authority.upsert(binding.clone());
    Ok(binding)
}

/// Register (or replace) a predicate object. Predicate objects are global
/// `ObjectId`-keyed (not tenant-scoped), so no tenant resolution.
fn register_predicate_object(
    store: &Arc<FileSystemPredicateObjectStore>,
    object_id: ObjectId,
    expression: FilterExpression,
) -> PredicateObjectResponse {
    store.register(object_id, expression.clone());
    PredicateObjectResponse {
        object_id,
        expression,
    }
}

// ===========================================================================
// Handlers (thin: authorize → 503-check → resolve → core → audit → response)
// ===========================================================================

/// `PUT /api/v2/abac/policy-bindings/{tenant}/{object_id}` — upsert a policy
/// binding (the admission atom). Hot-reload: visible to the enforcer immediately.
pub async fn put_policy_binding(
    user_context: Option<Extension<UnifiedUserContext>>,
    State(state): State<AppState>,
    Path((tenant, object_id)): Path<(String, ObjectId)>,
    Json(body): Json<PutPolicyBindingRequest>,
) -> Result<Json<PolicyBinding>, (StatusCode, Json<OperatorErrorResponse>)> {
    let user_id = authorize_operator(user_context.as_ref().map(|e| &e.0))?;
    let store = require_binding_store(&state)?;
    let resolver = CatalogTenantStableIdResolver::new(state.catalog_manager.clone());
    ensure_tenant(&resolver, &tenant)
        .await
        .map_err(map_provision_err)?;
    let binding = provision_policy_binding(store, &resolver, &tenant, object_id, body)
        .map_err(map_provision_err)?;
    tracing::info!(
        target: "proximadb.abac_policy.audit",
        operator = %user_id,
        tenant = %tenant,
        object_id,
        tenant_stable_id = binding.tenant_stable_id,
        effect = ?binding.effect,
        action = "upsert_policy_binding",
        "ABAC policy binding provisioned"
    );
    Ok(Json(binding))
}

/// `DELETE /api/v2/abac/policy-bindings/{tenant}/{object_id}` — remove a binding.
/// Idempotent (204 whether or not it was present).
pub async fn delete_policy_binding(
    user_context: Option<Extension<UnifiedUserContext>>,
    State(state): State<AppState>,
    Path((tenant, object_id)): Path<(String, ObjectId)>,
) -> Result<StatusCode, (StatusCode, Json<OperatorErrorResponse>)> {
    let user_id = authorize_operator(user_context.as_ref().map(|e| &e.0))?;
    let store = require_binding_store(&state)?;
    let resolver = CatalogTenantStableIdResolver::new(state.catalog_manager.clone());
    let tenant_stable_id = resolve_tenant(&resolver, &tenant).map_err(map_provision_err)?;
    store.remove(tenant_stable_id, object_id);
    tracing::info!(
        target: "proximadb.abac_policy.audit",
        operator = %user_id,
        tenant = %tenant,
        object_id,
        action = "delete_policy_binding",
        "ABAC policy binding removed"
    );
    Ok(StatusCode::NO_CONTENT)
}

/// `POST /api/v2/abac/attribute-bindings` — upsert a subject's attribute binding
/// (the authority half). Hot-reload.
pub async fn post_attribute_binding(
    user_context: Option<Extension<UnifiedUserContext>>,
    State(state): State<AppState>,
    Json(body): Json<PostAttributeBindingRequest>,
) -> Result<Json<AttributeBinding>, (StatusCode, Json<OperatorErrorResponse>)> {
    let user_id = authorize_operator(user_context.as_ref().map(|e| &e.0))?;
    let authority = require_authority(&state)?;
    let resolver = CatalogTenantStableIdResolver::new(state.catalog_manager.clone());
    ensure_tenant(&resolver, &body.tenant)
        .await
        .map_err(map_provision_err)?;
    let binding = provision_attribute_binding(
        authority,
        &resolver,
        &body.subject_id,
        &body.tenant,
        body.attrs,
    )
    .map_err(map_provision_err)?;
    tracing::info!(
        target: "proximadb.abac_policy.audit",
        operator = %user_id,
        subject_id = %binding.subject_id.0,
        tenant = %body.tenant,
        tenant_stable_id = binding.tenant_stable_id,
        attrs = binding.attrs.len(),
        action = "upsert_attribute_binding",
        "ABAC attribute binding provisioned"
    );
    Ok(Json(binding))
}

/// `PUT /api/v2/abac/predicate-objects/{object_id}` — register/replace a
/// predicate object (a stored `FilterExpression` referenced by policy bindings).
pub async fn put_predicate_object(
    user_context: Option<Extension<UnifiedUserContext>>,
    State(state): State<AppState>,
    Path(object_id): Path<ObjectId>,
    Json(expression): Json<FilterExpression>,
) -> Result<Json<PredicateObjectResponse>, (StatusCode, Json<OperatorErrorResponse>)> {
    let user_id = authorize_operator(user_context.as_ref().map(|e| &e.0))?;
    let store = require_predicate_store(&state)?;
    let resp = register_predicate_object(store, object_id, expression);
    tracing::info!(
        target: "proximadb.abac_policy.audit",
        operator = %user_id,
        object_id,
        action = "register_predicate_object",
        "ABAC predicate object registered"
    );
    Ok(Json(resp))
}

/// `DELETE /api/v2/abac/predicate-objects/{object_id}` — revoke a predicate
/// object. Idempotent (204). Subsequent resolves of `object_id` fail-closed.
pub async fn delete_predicate_object(
    user_context: Option<Extension<UnifiedUserContext>>,
    State(state): State<AppState>,
    Path(object_id): Path<ObjectId>,
) -> Result<StatusCode, (StatusCode, Json<OperatorErrorResponse>)> {
    let user_id = authorize_operator(user_context.as_ref().map(|e| &e.0))?;
    let store = require_predicate_store(&state)?;
    store.revoke(object_id);
    tracing::info!(
        target: "proximadb.abac_policy.audit",
        operator = %user_id,
        object_id,
        action = "revoke_predicate_object",
        "ABAC predicate object revoked"
    );
    Ok(StatusCode::NO_CONTENT)
}

// ===========================================================================
// Read (GET) handlers — operator inspection of the live policy
// ===========================================================================

/// `GET /api/v2/abac/policy-bindings/{tenant}` — the tenant's live policy
/// bindings (the set the enforcer composes per read). Resolve-at-read, same as
/// write: 422 if the tenant has no stable id.
pub async fn get_policy_bindings(
    user_context: Option<Extension<UnifiedUserContext>>,
    State(state): State<AppState>,
    Path(tenant): Path<String>,
) -> Result<Json<PolicyBindingsResponse>, (StatusCode, Json<OperatorErrorResponse>)> {
    authorize_operator(user_context.as_ref().map(|e| &e.0))?;
    let store = require_binding_store(&state)?;
    let resolver = CatalogTenantStableIdResolver::new(state.catalog_manager.clone());
    let tenant_stable_id = resolve_tenant(&resolver, &tenant).map_err(map_provision_err)?;
    let bindings = store.bindings_for(tenant_stable_id);
    let count = bindings.len();
    Ok(Json(PolicyBindingsResponse {
        tenant,
        tenant_stable_id,
        count,
        bindings,
    }))
}

/// `GET /api/v2/abac/attribute-bindings` — every authority binding
/// (cluster-operator scope; cross-tenant by design).
pub async fn list_attribute_bindings(
    user_context: Option<Extension<UnifiedUserContext>>,
    State(state): State<AppState>,
) -> Result<Json<AttributeBindingsResponse>, (StatusCode, Json<OperatorErrorResponse>)> {
    authorize_operator(user_context.as_ref().map(|e| &e.0))?;
    let authority = require_authority(&state)?;
    let bindings = authority.bindings();
    let count = bindings.len();
    Ok(Json(AttributeBindingsResponse { count, bindings }))
}

/// `GET /api/v2/abac/predicate-objects` — every registered predicate object.
pub async fn list_predicate_objects(
    user_context: Option<Extension<UnifiedUserContext>>,
    State(state): State<AppState>,
) -> Result<Json<PredicateObjectsResponse>, (StatusCode, Json<OperatorErrorResponse>)> {
    authorize_operator(user_context.as_ref().map(|e| &e.0))?;
    let store = require_predicate_store(&state)?;
    let objects = store
        .objects()
        .into_iter()
        .map(|(object_id, expression)| PredicateObjectResponse {
            object_id,
            expression,
        })
        .collect::<Vec<_>>();
    let count = objects.len();
    Ok(Json(PredicateObjectsResponse { count, objects }))
}

/// `GET /api/v2/abac/predicate-objects/{object_id}` — one predicate object.
/// 404 if unknown (a dangling ref resolves fail-closed regardless).
pub async fn get_predicate_object(
    user_context: Option<Extension<UnifiedUserContext>>,
    State(state): State<AppState>,
    Path(object_id): Path<ObjectId>,
) -> Result<Json<PredicateObjectResponse>, (StatusCode, Json<OperatorErrorResponse>)> {
    authorize_operator(user_context.as_ref().map(|e| &e.0))?;
    let store = require_predicate_store(&state)?;
    match store.get(object_id) {
        Some(expression) => Ok(Json(PredicateObjectResponse {
            object_id,
            expression,
        })),
        None => Err((
            StatusCode::NOT_FOUND,
            Json(OperatorErrorResponse {
                error: "predicate_object_not_found",
                message: format!("no predicate object registered under object_id {object_id}"),
                code: 404,
            }),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::rbac_service::UnifiedAuthMethod;
    use crate::security::rls::AbacEnforcer;
    use chrono::Utc;
    use proximadb_abac::InMemoryPolicyEpochs;
    use proximadb_catalog::fc_metamodel::Target;
    use proximadb_filter_expression::ComparisonOperator;
    use serde_json::json;
    use std::collections::{BTreeMap, HashMap, HashSet};

    // ── gate tests (authorize_operator as a pure fn, mirroring primary_pod) ──

    fn ctx_with_permissions(perms: Vec<UnifiedPermission>) -> UnifiedUserContext {
        UnifiedUserContext {
            user_id: "test-operator".to_string(),
            tenant_id: Some("tenant-x".to_string()),
            roles: vec!["test".to_string()],
            effective_permissions: perms.into_iter().collect::<HashSet<_>>(),
            auth_method: UnifiedAuthMethod::ApiKey,
            session_id: "test-session".to_string(),
            expires_at: None,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn missing_auth_context_is_401() {
        let (status, body) = authorize_operator(None).expect_err("None ⇒ 401");
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(body.0.code, 401);
        assert_eq!(body.0.error, "missing_auth_context");
    }

    #[test]
    fn tenant_admin_alone_is_403() {
        let ctx = ctx_with_permissions(vec![UnifiedPermission::TenantAdmin]);
        let (status, body) = authorize_operator(Some(&ctx)).expect_err("TenantAdmin ⇒ 403");
        assert_eq!(status, StatusCode::FORBIDDEN);
        assert_eq!(body.0.code, 403);
    }

    #[test]
    fn data_plane_permissions_are_403() {
        // CollectionRead / VectorSearch are scoped (collection-string-carrying)
        // permissions — neither is an operator permission, so both must reject.
        for perm in [
            UnifiedPermission::CollectionRead("c".to_string()),
            UnifiedPermission::VectorSearch("c".to_string()),
        ] {
            let ctx = ctx_with_permissions(vec![perm]);
            assert!(
                authorize_operator(Some(&ctx)).is_err(),
                "a data-plane permission must not satisfy the operator gate"
            );
        }
    }

    #[test]
    fn system_admin_is_authorized() {
        let ctx = ctx_with_permissions(vec![UnifiedPermission::SystemAdmin]);
        let user_id = authorize_operator(Some(&ctx)).expect("SystemAdmin ⇒ Ok");
        assert_eq!(user_id, "test-operator");
    }

    #[test]
    fn configure_system_is_authorized() {
        let ctx = ctx_with_permissions(vec![UnifiedPermission::ConfigureSystem]);
        authorize_operator(Some(&ctx)).expect("ConfigureSystem ⇒ Ok");
    }

    #[test]
    fn either_operator_permission_satisfies_the_gate() {
        let ctx = ctx_with_permissions(vec![
            UnifiedPermission::SystemAdmin,
            UnifiedPermission::ConfigureSystem,
        ]);
        authorize_operator(Some(&ctx)).expect("both ⇒ Ok");
    }

    // ── resolve-at-write (the security-critical consistency guarantee) ──

    /// A test resolver: "acme" → 7, anything else → None (unminted).
    struct TestResolver;
    impl TenantStableIdResolver for TestResolver {
        fn stable_id_of(&self, tenant_id: &str) -> Option<u64> {
            (tenant_id == "acme").then_some(7)
        }
    }

    fn unique_dir(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "proximadb-abac-admin-{label}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    #[test]
    fn unminted_tenant_is_rejected_with_tenant_unresolved() {
        let dir = unique_dir("resolve");
        let store = Arc::new(FileSystemPolicyBindingStore::open(dir.join("policy.json")).unwrap());
        let resolver = TestResolver;

        // "ghost" has no stable id ⇒ TenantUnresolved (the handler maps this to 422).
        let err = provision_policy_binding(
            &store,
            &resolver,
            "ghost",
            1,
            PutPolicyBindingRequest {
                scope: Scope::Table(200),
                effect: Effect::Permit,
                predicate_ref: None,
                field_mask: None,
            },
        )
        .expect_err("unminted tenant must not provision");
        assert_eq!(
            err,
            ProvisionError::TenantUnresolved {
                tenant: "ghost".into()
            }
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn provisioning_mints_tenant_without_a_dummy_table() {
        let dir = unique_dir("catalog-mint");
        let manager = Arc::new(crate::catalog::CatalogManager::new());
        manager
            .create_native_catalog("default", &format!("file://{}", dir.display()))
            .await
            .expect("native catalog");
        let resolver = CatalogTenantStableIdResolver::new(manager.clone());

        let minted = ensure_tenant(&resolver, "new-tenant")
            .await
            .expect("policy provisioning must mint the tenant");
        assert_eq!(resolver.stable_id_of("new-tenant"), Some(minted));

        let store = Arc::new(
            FileSystemPolicyBindingStore::open(dir.join("policy.json")).expect("policy store"),
        );
        let binding = provision_policy_binding(
            &store,
            &resolver,
            "new-tenant",
            9,
            PutPolicyBindingRequest {
                scope: Scope::Table(200),
                effect: Effect::Permit,
                predicate_ref: None,
                field_mask: None,
            },
        )
        .expect("resolved-at-write binding");
        assert_eq!(binding.tenant_stable_id, minted);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn provisioned_binding_round_trips_through_the_store() {
        let dir = unique_dir("roundtrip");
        let path = dir.join("policy.json");
        let store = Arc::new(FileSystemPolicyBindingStore::open(&path).unwrap());
        let resolver = TestResolver;

        let binding = provision_policy_binding(
            &store,
            &resolver,
            "acme",
            1,
            PutPolicyBindingRequest {
                scope: Scope::Table(200),
                effect: Effect::Permit,
                predicate_ref: Some(42),
                field_mask: None,
            },
        )
        .expect("acme is minted");
        assert_eq!(binding.tenant_stable_id, 7); // resolve-at-write stamped the u64

        // Reopen ⇒ the provision survived (durable).
        let reopened = FileSystemPolicyBindingStore::open(&path).unwrap();
        let for_acme = reopened.bindings_for(7);
        assert_eq!(for_acme.len(), 1);
        assert_eq!(for_acme[0].object_id, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── the heart of PR-B: provision → enforce at the service seam ──
    //
    // (REST→read→admit over HTTP is not yet possible — TD-ABAC-5 wires the
    // request subject into the enforcer on a separate follow-on — so the honest
    // assertion is at the service seam: `resolve_read_context`.)

    #[test]
    fn provisioned_policy_admits_unprovisioned_denies_at_service_seam() {
        let dir = unique_dir("seam");
        let binding_store =
            Arc::new(FileSystemPolicyBindingStore::open(dir.join("policy.json")).unwrap());
        let authority =
            Arc::new(FileSystemAttributeAuthority::open(dir.join("attrs.json")).unwrap());
        let predicate_store =
            Arc::new(FileSystemPredicateObjectStore::open(dir.join("preds.json")).unwrap());

        // The enforcer holds Arc clones of the SAME three stores the cores write.
        let enforcer = AbacEnforcer::new(
            authority.clone(),
            predicate_store.clone(),
            Arc::new(InMemoryPolicyEpochs::new()),
        )
        .with_binding_store(binding_store.clone());

        let target = Target {
            namespace: 3,
            table: 200,
            column: None,
        };
        let resolver = TestResolver;

        // 1. Before provisioning: alice@7 is denied (no attrs, no binding).
        assert!(
            enforcer
                .resolve_read_context(&SubjectId("alice".into()), 7, target)
                .is_err(),
            "an unprovisioned subject must be denied (fail-closed)"
        );

        // 2. Provision alice's authority + the predicate object + the policy binding
        //    through the SAME shared handles (resolve-at-write + hot-reload).
        let mut attrs = BTreeMap::new();
        attrs.insert("dept".to_string(), AttrValue::Str("eng".into()));
        provision_attribute_binding(&authority, &resolver, "alice", "acme", attrs)
            .expect("acme is minted");
        register_predicate_object(
            &predicate_store,
            42,
            FilterExpression::Comparison {
                field: "dept".to_string(),
                operator: ComparisonOperator::Equals,
                value: json!("eng"),
            },
        );
        provision_policy_binding(
            &binding_store,
            &resolver,
            "acme",
            1,
            PutPolicyBindingRequest {
                scope: Scope::Table(200),
                effect: Effect::Permit,
                predicate_ref: Some(42),
                field_mask: None,
            },
        )
        .expect("acme is minted");

        // 3. After provisioning: alice@7 is admitted with the dept=eng predicate ref.
        let ctx = enforcer
            .resolve_read_context(&SubjectId("alice".into()), 7, target)
            .expect("alice must be admitted after provisioning");
        assert!(
            ctx.row_predicate_refs().contains(&42),
            "the admitted context carries the provisioned predicate ref"
        );

        // 4. A subject in an unminted tenant can't even be provisioned (422 path).
        let err = provision_policy_binding(
            &binding_store,
            &resolver,
            "ghost",
            2,
            PutPolicyBindingRequest {
                scope: Scope::Table(200),
                effect: Effect::Permit,
                predicate_ref: None,
                field_mask: None,
            },
        )
        .expect_err("ghost has no stable id");
        assert_eq!(
            err,
            ProvisionError::TenantUnresolved {
                tenant: "ghost".into()
            }
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_endpoints_reflect_provisioned_policy() {
        // Round-trip: provision via the write cores, then read back through the
        // SAME accessors the GET handlers use — `authority.bindings()` (the new
        // accessor), `bindings_for`, `objects`, `get`. Proves the read surface
        // observes the live, hot-reloaded policy (and validates PR-B's writes).
        let dir = unique_dir("reads");
        let binding_store =
            Arc::new(FileSystemPolicyBindingStore::open(dir.join("policy.json")).unwrap());
        let authority =
            Arc::new(FileSystemAttributeAuthority::open(dir.join("attrs.json")).unwrap());
        let predicate_store =
            Arc::new(FileSystemPredicateObjectStore::open(dir.join("preds.json")).unwrap());
        let resolver = TestResolver;

        // Provision one of each.
        let mut attrs = BTreeMap::new();
        attrs.insert("dept".to_string(), AttrValue::Str("eng".into()));
        provision_attribute_binding(&authority, &resolver, "alice", "acme", attrs).unwrap();
        provision_policy_binding(
            &binding_store,
            &resolver,
            "acme",
            1,
            PutPolicyBindingRequest {
                scope: Scope::Table(200),
                effect: Effect::Permit,
                predicate_ref: Some(42),
                field_mask: None,
            },
        )
        .unwrap();
        register_predicate_object(
            &predicate_store,
            42,
            FilterExpression::Comparison {
                field: "dept".to_string(),
                operator: ComparisonOperator::Equals,
                value: json!("eng"),
            },
        );

        // GET /policy-bindings/{tenant} path → bindings_for.
        let acme_bindings = binding_store.bindings_for(7);
        assert_eq!(acme_bindings.len(), 1);
        assert_eq!(acme_bindings[0].object_id, 1);

        // GET /attribute-bindings path → authority.bindings() (the new accessor).
        let attr_bindings = authority.bindings();
        assert_eq!(attr_bindings.len(), 1);
        assert_eq!(attr_bindings[0].subject_id.0, "alice");
        assert_eq!(attr_bindings[0].tenant_stable_id, 7);

        // GET /predicate-objects path → objects().
        let pred_objects = predicate_store.objects();
        assert_eq!(pred_objects.len(), 1);
        assert_eq!(pred_objects[0].0, 42);

        // GET /predicate-objects/{object_id} path → get() (Some) or 404 (None).
        assert!(predicate_store.get(42).is_some());
        assert!(predicate_store.get(999).is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }
}

// ===========================================================================
// ADR-090 grants admin (TD-AUTHZ-1) — provision/list/revoke the entitlement
// layer. Same doctrine as the policy endpoints above: operator-gated, 503 when
// the durable store is absent, tenants arrive as STRINGS and resolve through
// the catalog resolver (fail-closed on unminted tenants), writes are
// hot-visible to the live enforcer through the shared Arc.
// ===========================================================================

/// Grantee as provisioned over the wire: a tenant-wide share, or one user of
/// a (possibly foreign) tenant. Tenants are strings here; stable ids are
/// resolved server-side — clients never supply raw stable ids.
#[derive(Debug, serde::Deserialize)]
pub struct GrantGranteeRequest {
    pub tenant: String,
    #[serde(default)]
    pub subject: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
pub struct PostGrantRequest {
    /// The RESOURCE OWNER tenant (string; resolved + must be minted).
    pub owner_tenant: String,
    pub resource: proximadb_catalog::fc_metamodel::Scope,
    pub grantee: GrantGranteeRequest,
    pub actions: std::collections::BTreeSet<proximadb_catalog::grants::GrantAction>,
    #[serde(default)]
    pub predicate_ref: Option<ObjectId>,
    #[serde(default)]
    pub field_mask: Option<proximadb_catalog::fc_metamodel::FieldMask>,
    #[serde(default)]
    pub expires_at_ms: Option<i64>,
}

#[derive(Debug, serde::Serialize)]
pub struct PostGrantResponse {
    pub grant_id: String,
    pub owner_tenant_stable_id: u64,
}

/// Like [`unavailable`] but for a runtime message (store errors carry dynamic
/// detail); `unavailable` deliberately takes `&'static str` so fixed messages
/// cannot accidentally interpolate request data.
fn store_unavailable(
    error: &'static str,
    message: String,
) -> (StatusCode, Json<OperatorErrorResponse>) {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(OperatorErrorResponse {
            error,
            message,
            code: 503,
        }),
    )
}

fn require_grant_store(
    state: &AppState,
) -> Result<
    &Arc<proximadb_catalog::grants::FileSystemGrantStore>,
    (StatusCode, Json<OperatorErrorResponse>),
> {
    state.abac_grant_store.as_ref().ok_or_else(|| {
        unavailable(
            "abac_unavailable",
            "ADR-090 grant store is not available (abac-policy off or no data_dir)",
        )
    })
}

/// `POST /api/v2/abac/grants` — provision a grant. Both the owner tenant and
/// the grantee tenant must resolve (fail-closed on unminted tenants); the
/// grantee MAY be a foreign tenant — that is the point of grants (ADR-090 L1).
/// The grantee SUBJECT is deliberately NOT validated against the principal
/// registry: a share may be provisioned before its recipient's first login.
pub async fn post_grant(
    user_context: Option<Extension<UnifiedUserContext>>,
    State(state): State<AppState>,
    Json(body): Json<PostGrantRequest>,
) -> Result<Json<PostGrantResponse>, (StatusCode, Json<OperatorErrorResponse>)> {
    let user_id = authorize_operator(user_context.as_ref().map(|e| &e.0))?;
    let store = require_grant_store(&state)?;
    let resolver = CatalogTenantStableIdResolver::new(state.catalog_manager.clone());
    let owner = ensure_tenant(&resolver, &body.owner_tenant)
        .await
        .map_err(map_provision_err)?;
    let grantee_tenant = ensure_tenant(&resolver, &body.grantee.tenant)
        .await
        .map_err(map_provision_err)?;
    let grantee = match &body.grantee.subject {
        Some(subject) if !subject.trim().is_empty() => proximadb_catalog::grants::Grantee::User {
            tenant_stable_id: grantee_tenant,
            subject: proximadb_catalog::fc_metamodel::SubjectId(subject.clone()),
        },
        _ => proximadb_catalog::grants::Grantee::Tenant(grantee_tenant),
    };
    let grant_id = store
        .grant(
            owner,
            body.resource,
            grantee,
            body.actions,
            body.predicate_ref,
            body.field_mask,
            body.expires_at_ms,
        )
        .map_err(|e| store_unavailable("grant_store_error", e.to_string()))?;
    tracing::info!(
        target: "proximadb.abac_policy.audit",
        operator = %user_id,
        owner_tenant = %body.owner_tenant,
        grantee_tenant = %body.grantee.tenant,
        grant_id = %grant_id,
        action = "provision_grant",
        "ADR-090 grant provisioned"
    );
    Ok(Json(PostGrantResponse {
        grant_id,
        owner_tenant_stable_id: owner,
    }))
}

/// `GET /api/v2/abac/grants/{owner_tenant}` — list an owner's grants (revoked
/// grants stay listed with `revoked_at_ms` set — the audit trail is the point).
pub async fn list_grants(
    user_context: Option<Extension<UnifiedUserContext>>,
    State(state): State<AppState>,
    Path(owner_tenant): Path<String>,
) -> Result<
    Json<Vec<proximadb_catalog::grants::GrantRecord>>,
    (StatusCode, Json<OperatorErrorResponse>),
> {
    authorize_operator(user_context.as_ref().map(|e| &e.0))?;
    let store = require_grant_store(&state)?;
    let resolver = CatalogTenantStableIdResolver::new(state.catalog_manager.clone());
    let owner = ensure_tenant(&resolver, &owner_tenant)
        .await
        .map_err(map_provision_err)?;
    Ok(Json(store.grants_for_owner(owner)))
}

/// `DELETE /api/v2/abac/grants/{owner_tenant}/{grant_id}` — revoke. Idempotent
/// like the sibling deletes: 204 whether the grant existed (and was revoked) or
/// was already unknown/revoked. Revocation under the WRONG owner cannot even
/// name the grant (owner-partitioned store).
pub async fn delete_grant(
    user_context: Option<Extension<UnifiedUserContext>>,
    State(state): State<AppState>,
    Path((owner_tenant, grant_id)): Path<(String, String)>,
) -> Result<StatusCode, (StatusCode, Json<OperatorErrorResponse>)> {
    let user_id = authorize_operator(user_context.as_ref().map(|e| &e.0))?;
    let store = require_grant_store(&state)?;
    let resolver = CatalogTenantStableIdResolver::new(state.catalog_manager.clone());
    let owner = ensure_tenant(&resolver, &owner_tenant)
        .await
        .map_err(map_provision_err)?;
    match store.revoke(owner, &grant_id) {
        Ok(()) => {
            tracing::info!(
                target: "proximadb.abac_policy.audit",
                operator = %user_id,
                owner_tenant = %owner_tenant,
                grant_id = %grant_id,
                action = "revoke_grant",
                "ADR-090 grant revoked"
            );
        }
        Err(proximadb_catalog::grants::GrantStoreError::UnknownGrant { .. }) => {}
        Err(e) => return Err(store_unavailable("grant_store_error", e.to_string())),
    }
    Ok(StatusCode::NO_CONTENT)
}

// ===========================================================================
// TD-SEC-2 Slice C — per-tenant security posture admin.
// Same doctrine as the grant/policy endpoints: operator-gated, 503 when the
// durable store is absent, tenants arrive as STRINGS and resolve through the
// catalog resolver, writes are hot-visible to the live enforcer.
// ===========================================================================

#[derive(Debug, Deserialize)]
pub struct PutPostureRequest {
    /// `Off` | `Audit` | `Enforce` (externally tagged, PascalCase on the wire —
    /// lowercase values are a serde parse error, not an alias).
    pub grant_enforcement: proximadb_catalog::tenant_posture::GrantEnforcement,
}

fn require_posture_store(
    state: &AppState,
) -> Result<
    &Arc<proximadb_catalog::tenant_posture::FileSystemTenantPostureStore>,
    (StatusCode, Json<OperatorErrorResponse>),
> {
    state.abac_posture_store.as_ref().ok_or_else(|| {
        unavailable(
            "abac_unavailable",
            "tenant posture store is not available (abac-policy off or no data_dir)",
        )
    })
}

/// `PUT /api/v2/abac/tenant-posture/{tenant}` — set a tenant's enforcement mode.
///
/// The intended rollout is `Off` → `Audit` (watch the would-be-denial warnings
/// fall to zero for this tenant) → `Enforce`. Skipping the `Audit` rehearsal is
/// how an operator breaks a customer's traffic.
pub async fn put_tenant_posture(
    user_context: Option<Extension<UnifiedUserContext>>,
    State(state): State<AppState>,
    Path(tenant): Path<String>,
    Json(body): Json<PutPostureRequest>,
) -> Result<
    Json<proximadb_catalog::tenant_posture::TenantSecurityPosture>,
    (StatusCode, Json<OperatorErrorResponse>),
> {
    let user_id = authorize_operator(user_context.as_ref().map(|e| &e.0))?;
    let store = require_posture_store(&state)?;
    let resolver = CatalogTenantStableIdResolver::new(state.catalog_manager.clone());
    let tenant_stable_id = ensure_tenant(&resolver, &tenant)
        .await
        .map_err(map_provision_err)?;
    let now_ms = chrono::Utc::now().timestamp_millis();
    store
        .set(tenant_stable_id, body.grant_enforcement, now_ms)
        .map_err(|e| store_unavailable("posture_store_error", e.to_string()))?;
    tracing::info!(
        target: "proximadb.abac_policy.audit",
        operator = %user_id,
        tenant = %tenant,
        mode = ?body.grant_enforcement,
        action = "set_tenant_posture",
        "tenant security posture updated"
    );
    Ok(Json(
        proximadb_catalog::tenant_posture::TenantSecurityPosture {
            tenant_stable_id,
            grant_enforcement: body.grant_enforcement,
            updated_at_ms: now_ms,
        },
    ))
}

/// `GET /api/v2/abac/tenant-posture/{tenant}` — the tenant's EXPLICIT posture.
///
/// 404 when the tenant has no record: that is meaningfully different from
/// `Off`, because an absent record means "inherit the process default", which
/// may itself be `Enforce` when the env gate is armed.
pub async fn get_tenant_posture(
    user_context: Option<Extension<UnifiedUserContext>>,
    State(state): State<AppState>,
    Path(tenant): Path<String>,
) -> Result<
    Json<proximadb_catalog::tenant_posture::TenantSecurityPosture>,
    (StatusCode, Json<OperatorErrorResponse>),
> {
    authorize_operator(user_context.as_ref().map(|e| &e.0))?;
    let store = require_posture_store(&state)?;
    let resolver = CatalogTenantStableIdResolver::new(state.catalog_manager.clone());
    let tenant_stable_id = ensure_tenant(&resolver, &tenant)
        .await
        .map_err(map_provision_err)?;
    match store.get(tenant_stable_id) {
        Some(posture) => Ok(Json(posture)),
        None => Err((
            StatusCode::NOT_FOUND,
            Json(OperatorErrorResponse {
                error: "no_explicit_posture",
                message: format!(
                    "tenant '{tenant}' has no explicit posture; it inherits the process default"
                ),
                code: 404,
            }),
        )),
    }
}
