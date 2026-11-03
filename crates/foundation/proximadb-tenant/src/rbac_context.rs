//! RBAC authorization-context data types (moved from `src/security/rbac_service.rs` — Slice D).
//!
//! These are the **leaf data** types that flow through the RBAC / authorization
//! surface: the permission enum, the auth-method discriminator, the
//! authenticated user context, and the resolved tenant context. They are pure
//! data (no behavior, no root-internal dependencies — only `String` / `chrono`
//! / `serde` / std collections), so they belong at the foundation tier where
//! every higher layer can depend *down* on them without reaching into the
//! services/security layer.
//!
//! The originating `src/security/rbac_service.rs` keeps a `pub use` re-export of
//! every type below, so all existing `crate::security::rbac_service::<Type>`
//! call sites continue to resolve unchanged (the move is source-compatible).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Unified permission model consolidating all permission types (tenant / domain /
/// collection / vector / entity / graph / query / system / business-context /
/// field-level).
///
/// Self-contained leaf enum: every payload is a `String` (or pair of `String`s),
/// so this type has no dependencies beyond std + serde. It is the field type of
/// [`UnifiedUserContext::effective_permissions`] and is consulted throughout the
/// RBAC manager for wildcard / hierarchy resolution.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnifiedPermission {
    // === TENANT LEVEL PERMISSIONS ===
    TenantAdmin,
    TenantRead,
    TenantWrite,

    // === DOMAIN LEVEL PERMISSIONS ===
    DomainCreate,
    DomainRead(String),
    DomainWrite(String),
    DomainAdmin(String),

    // === COLLECTION LEVEL PERMISSIONS ===
    // Collection management
    CollectionCreate,
    CollectionRead(String),
    CollectionWrite(String),
    CollectionDelete(String),
    CollectionAdmin(String),

    // Collection metadata
    ReadCollectionMetadata(String),
    UpdateCollectionMetadata(String),
    ListCollections,

    // === VECTOR LEVEL PERMISSIONS ===
    VectorInsert(String), // Collection-specific vector operations
    VectorDelete(String),
    VectorSearch(String),
    VectorUpdate(String),
    VectorRead(String),

    // === ENTITY LEVEL PERMISSIONS ===
    EntityRead(String),
    EntityWrite(String),
    EntityDelete(String),

    // === GRAPH LEVEL PERMISSIONS ===
    GraphCreateRelations(String), // Collection-specific graph operations
    GraphDeleteRelations(String),
    GraphTraverse(String),
    GraphReadRelations(String),

    // === QUERY LEVEL PERMISSIONS ===
    ExecuteSqlQueries(String),   // Collection-specific SQL queries
    ExecuteSksFunctions(String), // SKS function execution

    // === SYSTEM LEVEL PERMISSIONS ===
    ViewSystemMetrics,
    ViewSystemHealth,
    ConfigureSystem,
    AuditRead,
    SystemAdmin,
    /// Operator maintenance on collections (compaction and similar
    /// non-destructive upkeep) — the PG17 `MAINTAIN` analog. Granted by
    /// `TenantAdmin` and `SystemAdmin` (see `check_wildcard_permissions`).
    MaintainCollections,

    // === BUSINESS CONTEXT PERMISSIONS ===
    // (From Enhanced RBAC for business intelligence)
    RiskDataAccess,
    FinancialDataAccess,
    ComplianceDataAccess,
    CustomerDataAccess,

    // === SPECIAL PERMISSIONS ===
    FieldLevelRead(String, String), // (collection, field)
    FieldLevelWrite(String, String),
}

/// Authentication method enum for the unified security surface.
///
/// This lives in the security/rbac boundary and intentionally differs from the
/// inbound transport/auth-provider `AuthMethod` modeled at the network edge: it
/// discriminates *how an authenticated principal authenticated*, not which wire
/// provider carried the credential.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum UnifiedAuthMethod {
    SSO { provider: String },
    JWT,
    ApiKey,
    ClientCertificate,
    Internal,
}

/// Unified user context for all authentication methods.
///
/// Carries the resolved identity of an authenticated principal: who they are
/// (`user_id`), which tenant they act under (`tenant_id`), their granted roles
/// and effective permission set, the method by which they authenticated, and
/// session bookkeeping. Leaf data type — fields are primitives, `chrono`
/// timestamps, std collections, and the two leaf enums above.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedUserContext {
    pub user_id: String,
    pub tenant_id: Option<String>,
    pub roles: Vec<String>,
    pub effective_permissions: HashSet<UnifiedPermission>,
    pub auth_method: UnifiedAuthMethod,
    pub session_id: String,
    pub expires_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

impl UnifiedUserContext {
    /// Create an anonymous user context.
    ///
    /// Used by auth surfaces when no principal could be resolved (e.g. auth
    /// disabled). Moved alongside the type from `src/security/auth_service.rs`
    /// (Slice D) — inherent methods must live in the defining crate.
    pub fn anonymous() -> Self {
        Self {
            user_id: "anonymous".to_string(),
            tenant_id: None,
            roles: vec!["anonymous".to_string()],
            effective_permissions: HashSet::new(),
            auth_method: UnifiedAuthMethod::Internal,
            session_id: format!("anon_{}", uuid::Uuid::new_v4()),
            expires_at: None,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Check if user is authenticated (i.e. not the anonymous principal).
    pub fn is_authenticated(&self) -> bool {
        self.user_id != "anonymous"
    }

    /// First-class gateway/operator principal marker (TD-TENANT-1). A gateway
    /// principal may delegate — assert a different acting tenant — under
    /// [`crate::identity_trust::HeaderTrustPolicy::GatewayOnly`]. Stamped
    /// from credential data at construction: a `gateway`/`operator` role
    /// (e.g. from a `gateway: true` JWT claim) or a `gateway: "true"`
    /// metadata entry. Prefer this over tenant-name membership in a
    /// system-tenant list (the compat fallback surfaces still honor).
    pub fn is_gateway_principal(&self) -> bool {
        self.roles.iter().any(|role| {
            role == crate::identity_trust::GATEWAY_ROLE
                || role == crate::identity_trust::OPERATOR_ROLE
        }) || self
            .metadata
            .get("gateway")
            .is_some_and(|value| value == "true")
    }

    /// Check if the user session has expired. A session with no expiry is never
    /// considered expired.
    pub fn is_session_expired(&self) -> bool {
        match self.expires_at {
            Some(expires_at) => Utc::now() > expires_at,
            None => false,
        }
    }

    /// Get the user display name — falls back to the `user_id` when no
    /// `display_name` metadata entry is present.
    pub fn display_name(&self) -> String {
        self.metadata
            .get("display_name")
            .unwrap_or(&self.user_id)
            .clone()
    }
}

/// Tenant context for authorized operations.
///
/// The resolved tenant against which an authorized request executes — used by
/// the RBAC manager to scope `AuthorizationResult::tenant_context`. The
/// back-compat alias `TenantContext` (re-exported by the originating
/// `rbac_service.rs`) points here.
#[derive(Debug, Clone)]
pub struct RbacTenantContext {
    pub tenant_id: String,
    pub tenant_name: String,
    pub security_policy: String,
    pub compliance_frameworks: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that every `UnifiedPermission` variant can be constructed and that
    /// each produces a distinct `Debug` representation. Moved alongside the type
    /// from `src/security/rbac_service.rs` — this is a data-type test (no manager
    /// behavior), so it travels with the type.
    #[test]
    fn unified_permission_variants_are_distinct() {
        let permissions: Vec<UnifiedPermission> = vec![
            UnifiedPermission::TenantAdmin,
            UnifiedPermission::TenantRead,
            UnifiedPermission::TenantWrite,
            UnifiedPermission::DomainCreate,
            UnifiedPermission::DomainRead("d1".into()),
            UnifiedPermission::DomainWrite("d1".into()),
            UnifiedPermission::DomainAdmin("d1".into()),
            UnifiedPermission::CollectionCreate,
            UnifiedPermission::CollectionRead("c1".into()),
            UnifiedPermission::CollectionWrite("c1".into()),
            UnifiedPermission::CollectionDelete("c1".into()),
            UnifiedPermission::CollectionAdmin("c1".into()),
            UnifiedPermission::ReadCollectionMetadata("c1".into()),
            UnifiedPermission::UpdateCollectionMetadata("c1".into()),
            UnifiedPermission::ListCollections,
            UnifiedPermission::VectorInsert("c1".into()),
            UnifiedPermission::VectorDelete("c1".into()),
            UnifiedPermission::VectorSearch("c1".into()),
            UnifiedPermission::VectorUpdate("c1".into()),
            UnifiedPermission::VectorRead("c1".into()),
            UnifiedPermission::EntityRead("e1".into()),
            UnifiedPermission::EntityWrite("e1".into()),
            UnifiedPermission::EntityDelete("e1".into()),
            UnifiedPermission::GraphCreateRelations("g1".into()),
            UnifiedPermission::GraphDeleteRelations("g1".into()),
            UnifiedPermission::GraphTraverse("g1".into()),
            UnifiedPermission::GraphReadRelations("g1".into()),
            UnifiedPermission::ExecuteSqlQueries("c1".into()),
            UnifiedPermission::ExecuteSksFunctions("c1".into()),
            UnifiedPermission::ViewSystemMetrics,
            UnifiedPermission::ViewSystemHealth,
            UnifiedPermission::ConfigureSystem,
            UnifiedPermission::AuditRead,
            UnifiedPermission::SystemAdmin,
            UnifiedPermission::RiskDataAccess,
            UnifiedPermission::FinancialDataAccess,
            UnifiedPermission::ComplianceDataAccess,
            UnifiedPermission::CustomerDataAccess,
            UnifiedPermission::FieldLevelRead("c1".into(), "field_a".into()),
            UnifiedPermission::FieldLevelWrite("c1".into(), "field_b".into()),
        ];

        // Each variant should produce a distinct Debug string.
        let debug_strings: HashSet<String> = permissions.iter().map(|p| format!("{p:?}")).collect();
        assert_eq!(
            debug_strings.len(),
            permissions.len(),
            "all permission variants should produce unique Debug representations"
        );
    }

    #[test]
    fn unified_auth_method_round_trips_serde() {
        let methods = vec![
            UnifiedAuthMethod::SSO {
                provider: "okta".into(),
            },
            UnifiedAuthMethod::JWT,
            UnifiedAuthMethod::ApiKey,
            UnifiedAuthMethod::ClientCertificate,
            UnifiedAuthMethod::Internal,
        ];
        for method in methods {
            let json = serde_json::to_string(&method).expect("serialize");
            let back: UnifiedAuthMethod = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(method, back);
        }
    }

    #[test]
    fn unified_user_context_round_trips_serde() {
        let ctx = UnifiedUserContext {
            user_id: "u1".into(),
            tenant_id: Some("t1".into()),
            roles: vec!["admin".into()],
            effective_permissions: [
                UnifiedPermission::TenantRead,
                UnifiedPermission::VectorSearch("c1".into()),
            ]
            .into_iter()
            .collect(),
            auth_method: UnifiedAuthMethod::JWT,
            session_id: "s1".into(),
            expires_at: None,
            created_at: Utc::now(),
            metadata: HashMap::from([("k".into(), "v".into())]),
        };
        let json = serde_json::to_string(&ctx).expect("serialize");
        let back: UnifiedUserContext = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.user_id, "u1");
        assert_eq!(back.tenant_id.as_deref(), Some("t1"));
        assert_eq!(back.auth_method, UnifiedAuthMethod::JWT);
        assert_eq!(back.effective_permissions.len(), 2);
    }

    #[test]
    fn rbac_tenant_context_constructs() {
        let ctx = RbacTenantContext {
            tenant_id: "tenant_1".into(),
            tenant_name: "Acme Corp".into(),
            security_policy: "strict".into(),
            compliance_frameworks: vec!["SOC2".into(), "GDPR".into()],
        };
        assert_eq!(ctx.tenant_id, "tenant_1");
        assert_eq!(ctx.compliance_frameworks.len(), 2);
    }
}
