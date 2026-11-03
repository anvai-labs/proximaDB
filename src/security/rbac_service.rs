//! Unified RBAC System for ProximaDB
//!
//! Consolidates the Enhanced RBAC Manager (storage/tenant/rbac.rs) and
//! Network RBAC (network/auth/rbac.rs) into a single, coherent permission system.

use anyhow::{Result, anyhow};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, info};

// Re-export types from Enhanced RBAC for compatibility
pub use crate::storage::tenant::rbac::{
    CollectionOperation as EnhancedCollectionOperation, Permission as EnhancedPermission,
};

/// Data model enum for cross-model permission validation — re-exported from canonical definition.
pub use proximadb_data_model::DataModel;

/// Permission cache entry with TTL
#[derive(Debug, Clone)]
struct PermissionCacheEntry {
    /// Whether the permission check was allowed
    allowed: bool,
    /// Timestamp when this cache entry was created, for TTL expiry
    cached_at: DateTime<Utc>,
}

/// Unified permission model consolidating all permission types.
///
/// Hoisted to the `proximadb-tenant` foundation crate (Slice D root-crate
/// shrinkage) — re-exported here so all existing `crate::security::rbac_service`
/// call sites resolve unchanged. See [`proximadb_tenant::UnifiedPermission`].
pub use proximadb_tenant::UnifiedPermission;

/// Unified role definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedRole {
    pub role_id: String,
    pub role_name: String,
    pub tenant_id: Option<String>, // None for system-wide roles
    pub permissions: HashSet<UnifiedPermission>,
    pub description: String,
    pub created_at: DateTime<Utc>,
    pub created_by: String,
    pub is_system_role: bool,
}

/// Unified user context for all authentication methods.
///
/// Hoisted to the `proximadb-tenant` foundation crate (Slice D root-crate
/// shrinkage) — re-exported here so all existing `crate::security::rbac_service`
/// call sites resolve unchanged. See [`proximadb_tenant::UnifiedUserContext`].
pub use proximadb_tenant::UnifiedUserContext;

/// Authentication method enum for the unified security surface.
/// This lives in the security/rbac boundary and intentionally differs from
/// `crate::network::auth::AuthMethod`, which models inbound transport/auth-provider
/// style handling at the network edge.
///
/// Hoisted to the `proximadb-tenant` foundation crate (Slice D) — re-exported
/// here. See [`proximadb_tenant::UnifiedAuthMethod`].
pub use proximadb_tenant::UnifiedAuthMethod;

/// Compatibility alias preserved for existing call sites in the security crate.
pub type AuthMethod = UnifiedAuthMethod;

/// Consolidated RBAC Manager
pub struct ConsolidatedRBACManager {
    /// Tenant-specific role definitions
    tenant_roles: Arc<DashMap<String, Arc<DashMap<String, UnifiedRole>>>>,

    /// System-wide roles (cross-tenant)
    system_roles: Arc<DashMap<String, UnifiedRole>>,

    /// Collection-specific permissions
    #[allow(dead_code)]
    collection_permissions: Arc<DashMap<String, CollectionPermissions>>,

    /// User role assignments
    user_role_assignments: Arc<DashMap<String, UserRoleAssignment>>,

    /// Audit logger for RBAC events
    audit_logger: Option<Arc<dyn RBACEventLogger + Send + Sync>>,

    /// Configuration
    config: RBACConfig,

    /// Permission cache for performance (user_id -> permission -> entry)
    permission_cache:
        Arc<RwLock<HashMap<String, HashMap<UnifiedPermission, PermissionCacheEntry>>>>,

    /// Reference to Enhanced RBAC Manager for multi-tenant operations
    enhanced_rbac: Option<Arc<crate::storage::tenant::rbac::EnhancedRBACManager>>,
}

/// Collection permissions with unified model
#[derive(Debug, Clone)]
pub struct CollectionPermissions {
    pub tenant_id: String,
    pub collection_id: String,
    pub read_roles: HashSet<String>,
    pub write_roles: HashSet<String>,
    pub admin_roles: HashSet<String>,
    pub field_level_permissions: Option<FieldLevelPermissions>,
    pub created_at: DateTime<Utc>,
}

/// Field-level permissions for granular access control
#[derive(Debug, Clone)]
pub struct FieldLevelPermissions {
    pub read_fields: HashMap<String, HashSet<String>>, // role -> fields
    pub write_fields: HashMap<String, HashSet<String>>, // role -> fields
    pub restricted_fields: HashSet<String>,            // Always restricted
}

/// User role assignment
#[derive(Debug, Clone)]
pub struct UserRoleAssignment {
    pub user_id: String,
    pub tenant_id: Option<String>,
    pub roles: HashSet<String>,
    pub direct_permissions: HashSet<UnifiedPermission>,
    pub assigned_at: DateTime<Utc>,
    pub assigned_by: String,
    pub expires_at: Option<DateTime<Utc>>,
}

/// RBAC configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RBACConfig {
    pub enabled: bool,
    pub enable_field_level_permissions: bool,
    pub enable_audit_logging: bool,
    pub default_deny: bool,
    pub cache_permissions: bool,
    pub permission_cache_ttl_minutes: u64,
}

impl Default for RBACConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            enable_field_level_permissions: true,
            enable_audit_logging: true,
            default_deny: false,
            cache_permissions: true,
            permission_cache_ttl_minutes: 15,
        }
    }
}

/// RBAC event logger trait
pub trait RBACEventLogger: Send + Sync {
    fn log_permission_check(
        &self,
        user_context: &UnifiedUserContext,
        permission: &UnifiedPermission,
        result: bool,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;

    fn log_role_assignment(
        &self,
        user_id: &str,
        tenant_id: Option<&str>,
        roles: &[String],
        assigned_by: &str,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;

    fn log_permission_denial(
        &self,
        user_context: &UnifiedUserContext,
        attempted_permission: &UnifiedPermission,
        reason: &str,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;
}

impl ConsolidatedRBACManager {
    /// Create new consolidated RBAC manager
    pub fn new(config: RBACConfig) -> Self {
        let manager = Self {
            tenant_roles: Arc::new(DashMap::new()),
            system_roles: Arc::new(DashMap::new()),
            collection_permissions: Arc::new(DashMap::new()),
            user_role_assignments: Arc::new(DashMap::new()),
            audit_logger: None,
            config,
            permission_cache: Arc::new(RwLock::new(HashMap::new())),
            enhanced_rbac: None,
        };

        // Initialize default system roles
        manager.initialize_default_system_roles();
        manager
    }

    /// Create new consolidated RBAC manager with Enhanced RBAC integration
    pub fn with_enhanced_rbac(
        config: RBACConfig,
        enhanced_rbac: Arc<crate::storage::tenant::rbac::EnhancedRBACManager>,
    ) -> Self {
        let manager = Self {
            tenant_roles: Arc::new(DashMap::new()),
            system_roles: Arc::new(DashMap::new()),
            collection_permissions: Arc::new(DashMap::new()),
            user_role_assignments: Arc::new(DashMap::new()),
            audit_logger: None,
            config,
            permission_cache: Arc::new(RwLock::new(HashMap::new())),
            enhanced_rbac: Some(enhanced_rbac),
        };

        manager.initialize_default_system_roles();
        manager
    }

    /// Set audit logger for RBAC events
    pub fn set_audit_logger(&mut self, logger: Arc<dyn RBACEventLogger + Send + Sync>) {
        self.audit_logger = Some(logger);
    }

    /// Check permission with caching support
    pub async fn check_permission_cached(
        &self,
        user_id: &str,
        permission: &UnifiedPermission,
    ) -> Result<bool> {
        if !self.config.cache_permissions {
            return self
                .check_permission_with_context(user_id, permission)
                .await;
        }

        // Check cache first
        {
            let cache = self.permission_cache.read().await;
            if let Some(user_cache) = cache.get(user_id)
                && let Some(entry) = user_cache.get(permission)
            {
                let now = Utc::now();
                let ttl = Duration::from_secs(self.config.permission_cache_ttl_minutes * 60);

                if now
                    .signed_duration_since(entry.cached_at)
                    .to_std()
                    .unwrap_or(Duration::ZERO)
                    < ttl
                {
                    debug!("Cache hit for user '{}': {:?}", user_id, permission);
                    return Ok(entry.allowed);
                }
            }
        }

        // Cache miss - perform actual check
        let allowed = self
            .check_permission_with_context(user_id, permission)
            .await?;

        // Update cache
        {
            let mut cache = self.permission_cache.write().await;
            let user_cache = cache.entry(user_id.to_string()).or_default();
            user_cache.insert(
                permission.clone(),
                PermissionCacheEntry {
                    allowed,
                    cached_at: Utc::now(),
                },
            );
        }

        Ok(allowed)
    }

    /// Validate collection access across data models
    pub async fn validate_collection_access_cross_model(
        &self,
        user_ctx: &UnifiedUserContext,
        collection_id: &str,
        operation: EnhancedCollectionOperation,
        data_model: DataModel,
    ) -> Result<AuthorizationResult> {
        let permission = match (data_model, &operation) {
            (DataModel::Vector, EnhancedCollectionOperation::Read) => {
                UnifiedPermission::CollectionRead(collection_id.to_string())
            }
            (DataModel::Vector, EnhancedCollectionOperation::Write) => {
                UnifiedPermission::CollectionWrite(collection_id.to_string())
            }
            (DataModel::Graph, EnhancedCollectionOperation::Read) => {
                UnifiedPermission::CollectionRead(collection_id.to_string())
            }
            (DataModel::Graph, EnhancedCollectionOperation::Delete) => {
                UnifiedPermission::CollectionDelete(collection_id.to_string())
            }
            (DataModel::Graph, EnhancedCollectionOperation::Admin) => {
                UnifiedPermission::CollectionAdmin(collection_id.to_string())
            }
            (DataModel::Document, EnhancedCollectionOperation::Read) => {
                UnifiedPermission::CollectionRead(collection_id.to_string())
            }
            (DataModel::Document, EnhancedCollectionOperation::Write) => {
                UnifiedPermission::CollectionWrite(collection_id.to_string())
            }
            (DataModel::Document, EnhancedCollectionOperation::Delete) => {
                UnifiedPermission::CollectionDelete(collection_id.to_string())
            }
            (DataModel::Document, EnhancedCollectionOperation::Admin) => {
                UnifiedPermission::CollectionAdmin(collection_id.to_string())
            }
            _ => {
                return Ok(AuthorizationResult {
                    allowed: false,
                    permissions: HashSet::new(),
                    tenant_context: None,
                    reason: Some(format!(
                        "Unsupported operation {:?} for data model {:?}",
                        operation, data_model
                    )),
                });
            }
        };

        let allowed = self
            .check_permission_cached(&user_ctx.user_id, &permission)
            .await?;

        let mut permissions = HashSet::new();
        permissions.insert(permission.clone());

        Ok(AuthorizationResult {
            allowed,
            permissions,
            tenant_context: user_ctx.tenant_id.as_ref().map(|tid| RbacTenantContext {
                tenant_id: tid.clone(),
                tenant_name: tid.clone(),
                security_policy: "default".to_string(),
                compliance_frameworks: Vec::new(),
            }),
            reason: if allowed {
                None
            } else {
                Some("Insufficient permissions".to_string())
            },
        })
    }

    /// Bridge to Enhanced RBAC for tenant-level operations
    pub async fn validate_bridge_enhanced_rbac(
        &self,
        user_context: &crate::storage::tenant::UserContext,
        collection_id: &str,
        operation: EnhancedCollectionOperation,
    ) -> Result<crate::storage::tenant::rbac::AccessValidationResult> {
        if let Some(enhanced) = &self.enhanced_rbac {
            return enhanced
                .validate_collection_access(
                    &user_context.tenant_id,
                    collection_id,
                    operation,
                    user_context,
                )
                .await;
        }

        // Fallback to internal validation if no Enhanced RBAC
        let user_ctx = UnifiedUserContext {
            user_id: user_context.user_id.clone(),
            tenant_id: Some(user_context.tenant_id.clone()),
            roles: user_context.roles.clone(),
            effective_permissions: HashSet::new(),
            auth_method: UnifiedAuthMethod::Internal,
            session_id: uuid::Uuid::new_v4().to_string(),
            expires_at: None,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        };

        let auth_result = self
            .validate_collection_access_cross_model(
                &user_ctx,
                collection_id,
                operation,
                DataModel::Vector,
            )
            .await?;

        Ok(crate::storage::tenant::rbac::AccessValidationResult {
            granted: auth_result.allowed,
            user_context: user_context.clone(),
            collection_context: crate::storage::tenant::rbac::CollectionContext {
                tenant_id: user_context.tenant_id.clone(),
                collection_id: collection_id.to_string(),
                operation: crate::storage::tenant::rbac::CollectionOperation::Read,
            },
            validation_metadata: crate::storage::tenant::rbac::ValidationMetadata {
                validated_at: Utc::now(),
                permissions_checked: HashSet::new(),
                validation_reason: auth_result.reason.unwrap_or_default(),
            },
        })
    }

    /// Clear permission cache for a specific user
    pub async fn clear_user_cache(&self, user_id: &str) {
        let mut cache = self.permission_cache.write().await;
        cache.remove(user_id);
        debug!("Cleared permission cache for user '{}'", user_id);
    }

    /// Clear entire permission cache
    pub async fn clear_all_cache(&self) {
        let mut cache = self.permission_cache.write().await;
        cache.clear();
        debug!("Cleared entire permission cache");
    }

    /// Check permission with full context (internal method)
    async fn check_permission_with_context(
        &self,
        user_id: &str,
        permission: &UnifiedPermission,
    ) -> Result<bool> {
        // Get user role assignment
        let assignment = self.user_role_assignments.get(user_id);

        if let Some(assignment) = assignment {
            let effective_permissions = self
                .get_effective_permissions_for_assignment(&assignment)
                .await?;

            let has_permission = effective_permissions.contains(permission)
                || self.check_wildcard_permissions(&effective_permissions, permission);

            Ok(has_permission)
        } else {
            // No assignment found - deny unless default_deny is false
            Ok(!self.config.default_deny)
        }
    }

    /// Get effective permissions for a user assignment
    async fn get_effective_permissions_for_assignment(
        &self,
        assignment: &UserRoleAssignment,
    ) -> Result<HashSet<UnifiedPermission>> {
        let mut effective_permissions = HashSet::new();

        // Add direct permissions
        effective_permissions.extend(assignment.direct_permissions.clone());

        // Add role-based permissions
        for role_name in &assignment.roles {
            if let Some(permissions) = self
                .get_role_permissions(role_name, assignment.tenant_id.as_deref())
                .await?
            {
                effective_permissions.extend(permissions);
            }
        }

        Ok(effective_permissions)
    }

    /// Check if user has specific permission
    pub async fn check_permission(
        &self,
        user_context: &UnifiedUserContext,
        permission: &UnifiedPermission,
    ) -> Result<bool> {
        // Get effective permissions for user
        let effective_permissions = self.get_effective_permissions(user_context).await?;

        let has_permission = effective_permissions.contains(permission)
            || self.check_wildcard_permissions(&effective_permissions, permission);

        // Log permission check if audit enabled
        if let Some(audit_logger) = &self.audit_logger {
            audit_logger
                .log_permission_check(user_context, permission, has_permission)
                .await?;
        }

        Ok(has_permission)
    }

    /// Get effective permissions for user (combining role permissions and direct permissions)
    pub async fn get_effective_permissions(
        &self,
        user_context: &UnifiedUserContext,
    ) -> Result<HashSet<UnifiedPermission>> {
        let mut effective_permissions = HashSet::new();

        // Start with permissions already in the user context
        effective_permissions.extend(user_context.effective_permissions.clone());

        // Get user role assignment
        if let Some(assignment) = self.user_role_assignments.get(&user_context.user_id) {
            // Add direct permissions
            effective_permissions.extend(assignment.direct_permissions.clone());

            // Add role-based permissions
            for role_name in &assignment.roles {
                if let Some(permissions) = self
                    .get_role_permissions(role_name, user_context.tenant_id.as_deref())
                    .await?
                {
                    effective_permissions.extend(permissions);
                }
            }
        }

        // Apply default deny policy if configured
        if self.config.default_deny && effective_permissions.is_empty() {
            return Ok(HashSet::new());
        }

        Ok(effective_permissions)
    }

    /// Get permissions for a specific role
    async fn get_role_permissions(
        &self,
        role_name: &str,
        tenant_id: Option<&str>,
    ) -> Result<Option<HashSet<UnifiedPermission>>> {
        // Check tenant-specific roles first
        if let Some(tenant_id) = tenant_id
            && let Some(tenant_roles) = self.tenant_roles.get(tenant_id)
            && let Some(role) = tenant_roles.get(role_name)
        {
            return Ok(Some(role.permissions.clone()));
        }

        // Check system-wide roles
        if let Some(role) = self.system_roles.get(role_name) {
            return Ok(Some(role.permissions.clone()));
        }

        Ok(None)
    }

    /// Check wildcard permissions (e.g., CollectionAdmin grants all collection permissions)
    fn check_wildcard_permissions(
        &self,
        user_permissions: &HashSet<UnifiedPermission>,
        requested_permission: &UnifiedPermission,
    ) -> bool {
        match requested_permission {
            // Collection permissions
            UnifiedPermission::CollectionRead(collection_id)
            | UnifiedPermission::CollectionWrite(collection_id)
            | UnifiedPermission::VectorRead(collection_id)
            | UnifiedPermission::VectorInsert(collection_id)
            | UnifiedPermission::VectorSearch(collection_id) => {
                user_permissions
                    .contains(&UnifiedPermission::CollectionAdmin(collection_id.clone()))
                    || user_permissions.contains(&UnifiedPermission::SystemAdmin)
            }

            // Domain permissions
            UnifiedPermission::DomainRead(domain_id)
            | UnifiedPermission::DomainWrite(domain_id) => {
                user_permissions.contains(&UnifiedPermission::DomainAdmin(domain_id.clone()))
                    || user_permissions.contains(&UnifiedPermission::TenantAdmin)
                    || user_permissions.contains(&UnifiedPermission::SystemAdmin)
            }

            // Tenant permissions
            UnifiedPermission::TenantRead | UnifiedPermission::TenantWrite => {
                user_permissions.contains(&UnifiedPermission::TenantAdmin)
                    || user_permissions.contains(&UnifiedPermission::SystemAdmin)
            }

            // Operator maintenance (PG17 MAINTAIN analog): a tenant admin or
            // system admin can always compact their collections.
            UnifiedPermission::MaintainCollections => {
                user_permissions.contains(&UnifiedPermission::TenantAdmin)
                    || user_permissions.contains(&UnifiedPermission::SystemAdmin)
            }

            _ => false,
        }
    }

    /// Initialize default system roles
    fn initialize_default_system_roles(&self) {
        let system_admin_role = UnifiedRole {
            role_id: "system_admin".to_string(),
            role_name: "System Administrator".to_string(),
            tenant_id: None,
            permissions: vec![UnifiedPermission::SystemAdmin].into_iter().collect(),
            description: "Full system administration access".to_string(),
            created_at: Utc::now(),
            created_by: "system".to_string(),
            is_system_role: true,
        };

        let tenant_admin_role = UnifiedRole {
            role_id: "tenant_admin".to_string(),
            role_name: "Tenant Administrator".to_string(),
            tenant_id: None,
            permissions: vec![
                UnifiedPermission::TenantAdmin,
                UnifiedPermission::CollectionCreate,
                UnifiedPermission::AuditRead,
                // TD-COMPACT-14: the PG17 MAINTAIN analog — a tenant admin can
                // always run operator compaction on the tenant's collections.
                UnifiedPermission::MaintainCollections,
            ]
            .into_iter()
            .collect(),
            description: "Tenant administration access".to_string(),
            created_at: Utc::now(),
            created_by: "system".to_string(),
            is_system_role: true,
        };

        let collection_user_role = UnifiedRole {
            role_id: "collection_user".to_string(),
            role_name: "Collection User".to_string(),
            tenant_id: None,
            permissions: vec![
                UnifiedPermission::ListCollections,
                UnifiedPermission::ReadCollectionMetadata("*".to_string()),
                UnifiedPermission::ViewSystemHealth,
            ]
            .into_iter()
            .collect(),
            description: "Basic collection access".to_string(),
            created_at: Utc::now(),
            created_by: "system".to_string(),
            is_system_role: true,
        };

        // Insert default roles
        self.system_roles
            .insert("system_admin".to_string(), system_admin_role);
        self.system_roles
            .insert("tenant_admin".to_string(), tenant_admin_role);
        self.system_roles
            .insert("collection_user".to_string(), collection_user_role);

        info!(
            "Initialized {} default system roles",
            self.system_roles.len()
        );
    }

    /// Assign role to user
    pub async fn assign_role_to_user(
        &self,
        user_id: &str,
        tenant_id: Option<&str>,
        role_name: &str,
        assigned_by: &str,
    ) -> Result<()> {
        // Validate role exists
        let role_exists = if let Some(tenant_id) = tenant_id {
            self.tenant_roles
                .get(tenant_id)
                .is_some_and(|roles| roles.contains_key(role_name))
        } else {
            false
        };

        if !role_exists && !self.system_roles.contains_key(role_name) {
            return Err(anyhow!("Role '{}' does not exist", role_name));
        }

        // Create or update user role assignment
        let assignment = UserRoleAssignment {
            user_id: user_id.to_string(),
            tenant_id: tenant_id.map(|t| t.to_string()),
            roles: vec![role_name.to_string()].into_iter().collect(),
            direct_permissions: HashSet::new(),
            assigned_at: Utc::now(),
            assigned_by: assigned_by.to_string(),
            expires_at: None,
        };

        self.user_role_assignments
            .insert(user_id.to_string(), assignment);

        // Log role assignment
        if let Some(audit_logger) = &self.audit_logger {
            audit_logger
                .log_role_assignment(user_id, tenant_id, &[role_name.to_string()], assigned_by)
                .await?;
        }

        info!(
            "Assigned role '{}' to user '{}' in tenant '{:?}'",
            role_name, user_id, tenant_id
        );

        Ok(())
    }

    /// Grant a direct permission to a user (bypassing roles)
    pub async fn grant_permission(
        &self,
        user_id: &str,
        permission: &UnifiedPermission,
    ) -> Result<()> {
        // Get or create user role assignment
        let mut assignment = self.user_role_assignments.get(user_id).map_or_else(
            || UserRoleAssignment {
                user_id: user_id.to_string(),
                tenant_id: None,
                roles: HashSet::new(),
                direct_permissions: HashSet::new(),
                assigned_at: Utc::now(),
                assigned_by: "system".to_string(),
                expires_at: None,
            },
            |a| a.clone(),
        );

        // Add the direct permission
        assignment.direct_permissions.insert(permission.clone());

        // Update the assignment
        self.user_role_assignments
            .insert(user_id.to_string(), assignment);

        info!("Granted permission {:?} to user '{}'", permission, user_id);

        Ok(())
    }

    /// Create custom tenant role
    pub async fn create_tenant_role(
        &self,
        tenant_id: &str,
        role_name: &str,
        permissions: HashSet<UnifiedPermission>,
        created_by: &str,
    ) -> Result<()> {
        let role = UnifiedRole {
            role_id: format!("{}_{}", tenant_id, role_name),
            role_name: role_name.to_string(),
            tenant_id: Some(tenant_id.to_string()),
            permissions,
            description: format!("Custom role for tenant {}", tenant_id),
            created_at: Utc::now(),
            created_by: created_by.to_string(),
            is_system_role: false,
        };

        // Ensure tenant roles map exists
        if !self.tenant_roles.contains_key(tenant_id) {
            self.tenant_roles
                .insert(tenant_id.to_string(), Arc::new(DashMap::new()));
        }

        // Insert role
        if let Some(tenant_roles) = self.tenant_roles.get(tenant_id) {
            tenant_roles.insert(role_name.to_string(), role);
        }

        info!(
            "Created custom role '{}' for tenant '{}'",
            role_name, tenant_id
        );
        Ok(())
    }

    /// Check collection-specific permission
    pub async fn check_collection_permission(
        &self,
        user_context: &UnifiedUserContext,
        collection_id: &str,
        permission_type: CollectionPermissionType,
    ) -> Result<bool> {
        let permission = match permission_type {
            CollectionPermissionType::Read => {
                UnifiedPermission::CollectionRead(collection_id.to_string())
            }
            CollectionPermissionType::Write => {
                UnifiedPermission::CollectionWrite(collection_id.to_string())
            }
            CollectionPermissionType::Admin => {
                UnifiedPermission::CollectionAdmin(collection_id.to_string())
            }
            CollectionPermissionType::VectorInsert => {
                UnifiedPermission::VectorInsert(collection_id.to_string())
            }
            CollectionPermissionType::VectorSearch => {
                UnifiedPermission::VectorSearch(collection_id.to_string())
            }
        };

        self.check_permission(user_context, &permission).await
    }

    /// Get all collections user has access to
    pub async fn get_accessible_collections(
        &self,
        user_context: &UnifiedUserContext,
        _permission_type: CollectionPermissionType,
    ) -> Result<Vec<String>> {
        let mut accessible_collections = Vec::new();

        // Check if user has system admin (access to all)
        if user_context
            .effective_permissions
            .contains(&UnifiedPermission::SystemAdmin)
        {
            // Return all collections for system admin
            // This would need integration with collection service
            return Ok(vec!["*".to_string()]); // Placeholder
        }

        // Check specific collection permissions
        for permission in &user_context.effective_permissions {
            match permission {
                UnifiedPermission::CollectionRead(coll_id)
                | UnifiedPermission::CollectionWrite(coll_id)
                | UnifiedPermission::CollectionAdmin(coll_id) => {
                    accessible_collections.push(coll_id.clone());
                }
                _ => {}
            }
        }

        Ok(accessible_collections)
    }
}

/// Collection permission types for easier API usage
#[derive(Debug, Clone)]
pub enum CollectionPermissionType {
    Read,
    Write,
    Admin,
    VectorInsert,
    VectorSearch,
}

/// Authorization result
#[derive(Debug, Clone)]
pub struct AuthorizationResult {
    pub allowed: bool,
    pub permissions: HashSet<UnifiedPermission>,
    pub tenant_context: Option<RbacTenantContext>,
    pub reason: Option<String>,
}

/// Tenant context for authorized operations.
///
/// Hoisted to the `proximadb-tenant` foundation crate (Slice D root-crate
/// shrinkage) — re-exported here so all existing `crate::security::rbac_service`
/// call sites resolve unchanged. See [`proximadb_tenant::RbacTenantContext`].
pub use proximadb_tenant::RbacTenantContext;

/// Backwards-compat alias for [`RbacTenantContext`].
pub type TenantContext = RbacTenantContext;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_unified_permission_checking() {
        let config = RBACConfig::default();
        let rbac_manager = ConsolidatedRBACManager::new(config);

        // Create test user context
        let user_context = UnifiedUserContext {
            user_id: "test_user".to_string(),
            tenant_id: Some("test_tenant".to_string()),
            roles: vec!["collection_user".to_string()],
            effective_permissions: vec![
                UnifiedPermission::CollectionRead("test_collection".to_string()),
                UnifiedPermission::VectorSearch("test_collection".to_string()),
            ]
            .into_iter()
            .collect(),
            auth_method: UnifiedAuthMethod::JWT,
            session_id: "session_123".to_string(),
            expires_at: None,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        };

        // Test permission checking
        let can_read = rbac_manager
            .check_permission(
                &user_context,
                &UnifiedPermission::CollectionRead("test_collection".to_string()),
            )
            .await
            .unwrap();

        assert!(can_read);

        let cannot_admin = rbac_manager
            .check_permission(&user_context, &UnifiedPermission::SystemAdmin)
            .await
            .unwrap();

        assert!(!cannot_admin);
    }

    #[tokio::test]
    async fn test_wildcard_permissions() {
        let config = RBACConfig::default();
        let rbac_manager = ConsolidatedRBACManager::new(config);

        // User with admin permission should have access to specific operations
        let admin_permissions = vec![UnifiedPermission::CollectionAdmin(
            "test_collection".to_string(),
        )]
        .into_iter()
        .collect();

        let has_read = rbac_manager.check_wildcard_permissions(
            &admin_permissions,
            &UnifiedPermission::CollectionRead("test_collection".to_string()),
        );

        assert!(has_read);
    }

    #[tokio::test]
    async fn test_role_assignment() {
        let config = RBACConfig::default();
        let rbac_manager = ConsolidatedRBACManager::new(config);

        // Assign role to user
        let result = rbac_manager
            .assign_role_to_user("test_user", Some("test_tenant"), "collection_user", "admin")
            .await;

        assert!(result.is_ok());

        // Verify assignment exists
        let assignment = rbac_manager.user_role_assignments.get("test_user");
        assert!(assignment.is_some());
        assert!(assignment.unwrap().roles.contains("collection_user"));
    }

    // =========================================================================
    // New infrastructure tests
    // =========================================================================

    // NOTE: `test_unified_permission_variants` moved to the `proximadb-tenant`
    // crate alongside the `UnifiedPermission` type (Slice D hoist — it is a pure
    // data-type test with no manager behavior).

    #[test]
    fn test_data_model_alias() {
        // DataModel is the canonical cross-model discriminator. Verify key
        // variants are accessible through the RBAC re-export.
        let vector = DataModel::Vector;
        let document = DataModel::Document;
        let graph = DataModel::Graph;

        assert_ne!(vector, document);
        assert_ne!(document, graph);
        assert_eq!(vector, DataModel::Vector);

        // Verify all variants listed in the canonical DataModel are reachable.
        let _observability = DataModel::Observability;
        let _relational = DataModel::Relational;
        let _time_series = DataModel::TimeSeries;
        let _event = DataModel::Event;
    }

    #[test]
    fn test_permission_cache_entry() {
        // PermissionCacheEntry is private, but we can exercise it indirectly
        // through the public cache API. Here we just verify the struct can be
        // constructed (it is in scope for this test module via `super::*`).
        let entry = PermissionCacheEntry {
            allowed: true,
            cached_at: Utc::now(),
        };
        assert!(entry.allowed);
        assert!(entry.cached_at <= Utc::now());

        let denied_entry = PermissionCacheEntry {
            allowed: false,
            cached_at: Utc::now() - chrono::Duration::minutes(5),
        };
        assert!(!denied_entry.allowed);
        // TTL check: the entry was created 5 minutes ago
        let age = Utc::now()
            .signed_duration_since(denied_entry.cached_at)
            .num_seconds();
        assert!(
            age >= 299,
            "cache entry age should be approximately 300 seconds"
        );
    }

    #[test]
    fn test_authorization_result_allowed() {
        let mut permissions = HashSet::new();
        permissions.insert(UnifiedPermission::CollectionRead("my_coll".into()));

        let result = AuthorizationResult {
            allowed: true,
            permissions,
            tenant_context: Some(RbacTenantContext {
                tenant_id: "tenant_1".into(),
                tenant_name: "Acme Corp".into(),
                security_policy: "strict".into(),
                compliance_frameworks: vec!["SOC2".into(), "GDPR".into()],
            }),
            reason: None,
        };

        assert!(result.allowed);
        assert!(
            result.reason.is_none(),
            "allowed result should have no denial reason"
        );
        assert_eq!(result.permissions.len(), 1);
        let ctx = result
            .tenant_context
            .as_ref()
            .expect("should have tenant context");
        assert_eq!(ctx.tenant_id, "tenant_1");
        assert_eq!(ctx.compliance_frameworks.len(), 2);
    }

    #[test]
    fn test_authorization_result_denied() {
        let result = AuthorizationResult {
            allowed: false,
            permissions: HashSet::new(),
            tenant_context: None,
            reason: Some("Insufficient permissions for VectorSearch on collection 'secret'".into()),
        };

        assert!(!result.allowed);
        assert!(result.permissions.is_empty());
        assert!(result.tenant_context.is_none());
        let reason = result
            .reason
            .as_ref()
            .expect("denied result should carry a reason");
        assert!(
            reason.contains("Insufficient permissions"),
            "reason should describe the denial"
        );
    }

    #[tokio::test]
    async fn test_cross_model_permission_validation() {
        // Validate that validate_collection_access_cross_model produces
        // correct AuthorizationResult for various DataModel + operation combos.
        let config = RBACConfig {
            default_deny: false, // allow by default (no assignment => allowed)
            ..RBACConfig::default()
        };
        let rbac = ConsolidatedRBACManager::new(config);

        let user_ctx = UnifiedUserContext {
            user_id: "cross_model_user".into(),
            tenant_id: Some("tenant_x".into()),
            roles: vec![],
            effective_permissions: HashSet::new(),
            auth_method: UnifiedAuthMethod::ApiKey,
            session_id: "sess_cross".into(),
            expires_at: None,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        };

        // Vector + Read should map to CollectionRead
        let result = rbac
            .validate_collection_access_cross_model(
                &user_ctx,
                "embeddings",
                EnhancedCollectionOperation::Read,
                DataModel::Vector,
            )
            .await
            .expect("should not fail");
        // default_deny=false and no assignment => allowed
        assert!(
            result.allowed,
            "Vector Read should be allowed with default_deny=false"
        );

        // Graph + Delete should map to CollectionDelete
        let result = rbac
            .validate_collection_access_cross_model(
                &user_ctx,
                "social_graph",
                EnhancedCollectionOperation::Delete,
                DataModel::Graph,
            )
            .await
            .expect("should not fail");
        assert!(result.allowed);

        // Document + Write should map to CollectionWrite
        let result = rbac
            .validate_collection_access_cross_model(
                &user_ctx,
                "articles",
                EnhancedCollectionOperation::Write,
                DataModel::Document,
            )
            .await
            .expect("should not fail");
        assert!(result.allowed);

        // Unsupported combo: Vector + Admin has no explicit mapping => denied
        let result = rbac
            .validate_collection_access_cross_model(
                &user_ctx,
                "embeddings",
                EnhancedCollectionOperation::Admin,
                DataModel::Vector,
            )
            .await
            .expect("should not fail");
        assert!(
            !result.allowed,
            "Vector + Admin is not an explicitly mapped combination"
        );
        assert!(
            result.reason.is_some(),
            "unsupported combo should carry a reason"
        );
    }
}
