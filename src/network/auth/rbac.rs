/*
 * Copyright 2025 ProximaDB
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

//! Role-Based Access Control (RBAC) for ProximaDB

use crate::network::auth::{AuthError, Permission, RbacConfig};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tokio::sync::RwLock;

/// RBAC service for managing users, roles, and permissions
pub struct RbacService {
    /// User to roles mapping
    user_roles: RwLock<HashMap<String, HashSet<String>>>,

    /// Role to permissions mapping
    role_permissions: RwLock<HashMap<String, HashSet<Permission>>>,

    /// Configuration
    config: RbacConfig,
}

impl RbacService {
    /// Create a new RBAC service with default configuration
    pub fn new() -> Self {
        let config = RbacConfig::default();
        let service = Self {
            user_roles: RwLock::new(HashMap::new()),
            role_permissions: RwLock::new(HashMap::new()),
            config,
        };

        // Initialize default roles
        service.initialize_default_roles();
        service
    }

    /// Create RBAC service with custom configuration
    pub fn with_config(config: RbacConfig) -> Self {
        let service = Self {
            user_roles: RwLock::new(HashMap::new()),
            role_permissions: RwLock::new(HashMap::new()),
            config,
        };

        service.initialize_default_roles();
        service
    }

    /// Initialize default roles from configuration
    fn initialize_default_roles(&self) {
        let runtime = tokio::runtime::Handle::current();
        runtime.spawn(async move {
            // This would be implemented with proper async initialization
            // For now, we'll defer role setup to first access
        });
    }

    /// Get roles for a user
    pub fn get_user_roles(&self, user_id: &str) -> Result<Vec<String>, AuthError> {
        let user_roles = self
            .user_roles
            .try_read()
            .map_err(|_| AuthError::InvalidCredentials)?;
        let roles = user_roles.get(user_id).map_or_else(
            || vec![self.config.default_role.clone()],
            |roles| roles.iter().cloned().collect(),
        );
        Ok(roles)
    }

    /// Assign role to user
    pub async fn assign_role(&self, user_id: &str, role: &str) -> Result<(), AuthError> {
        // Verify role exists
        let role_permissions = self.role_permissions.read().await;
        if !role_permissions.contains_key(role) && !self.config.roles.contains_key(role) {
            return Err(AuthError::RoleNotFound(role.to_string()));
        }
        drop(role_permissions);

        // Assign role to user
        let mut user_roles = self.user_roles.write().await;
        user_roles
            .entry(user_id.to_string())
            .or_default()
            .insert(role.to_string());

        Ok(())
    }

    /// Remove role from user
    pub async fn remove_role(&self, user_id: &str, role: &str) -> Result<(), AuthError> {
        let mut user_roles = self.user_roles.write().await;
        if let Some(roles) = user_roles.get_mut(user_id) {
            roles.remove(role);
        }
        Ok(())
    }

    /// Get permissions for a list of roles
    pub fn get_permissions_for_roles(
        &self,
        roles: &[String],
    ) -> Result<Vec<Permission>, AuthError> {
        let mut permissions = HashSet::new();

        // Get cached permissions
        let role_permissions = self
            .role_permissions
            .try_read()
            .map_err(|_| AuthError::InvalidCredentials)?;

        for role in roles {
            // Check cached permissions first
            if let Some(role_perms) = role_permissions.get(role) {
                permissions.extend(role_perms.iter().cloned());
            }
            // Fallback to config permissions
            else if let Some(config_perms) = self.config.roles.get(role) {
                for perm_str in config_perms {
                    if let Ok(permission) = self.parse_permission(perm_str) {
                        permissions.insert(permission);
                    }
                }
            }
        }

        Ok(permissions.into_iter().collect())
    }

    /// Create a new role with permissions
    pub async fn create_role(
        &self,
        role_name: &str,
        permissions: Vec<Permission>,
    ) -> Result<(), AuthError> {
        let mut role_permissions = self.role_permissions.write().await;
        role_permissions.insert(role_name.to_string(), permissions.into_iter().collect());
        Ok(())
    }

    /// Delete a role
    pub async fn delete_role(&self, role_name: &str) -> Result<(), AuthError> {
        // Don't allow deletion of built-in roles
        if ["admin", "user", "readonly"].contains(&role_name) {
            return Err(AuthError::AuthorizationDenied(Permission::ManageRoles));
        }

        let mut role_permissions = self.role_permissions.write().await;
        role_permissions.remove(role_name);

        // Remove role from all users
        let mut user_roles = self.user_roles.write().await;
        for roles in user_roles.values_mut() {
            roles.remove(role_name);
        }

        Ok(())
    }

    /// Check if user has specific permission
    pub async fn user_has_permission(&self, user_id: &str, permission: Permission) -> bool {
        let roles = match self.get_user_roles(user_id) {
            Ok(roles) => roles,
            Err(_) => return false,
        };

        let permissions = match self.get_permissions_for_roles(&roles) {
            Ok(permissions) => permissions,
            Err(_) => return false,
        };

        permissions.contains(&permission)
    }

    /// List all roles
    pub async fn list_roles(&self) -> Vec<String> {
        let role_permissions = self.role_permissions.read().await;
        let mut roles: Vec<String> = role_permissions.keys().cloned().collect();

        // Add config roles that aren't cached yet
        for config_role in self.config.roles.keys() {
            if !roles.contains(config_role) {
                roles.push(config_role.clone());
            }
        }

        roles.sort();
        roles
    }

    /// Get role permissions
    pub async fn get_role_permissions(&self, role: &str) -> Result<Vec<Permission>, AuthError> {
        let role_permissions = self.role_permissions.read().await;

        if let Some(permissions) = role_permissions.get(role) {
            Ok(permissions.iter().cloned().collect())
        } else if let Some(config_perms) = self.config.roles.get(role) {
            let mut permissions = Vec::new();
            for perm_str in config_perms {
                if let Ok(permission) = self.parse_permission(perm_str) {
                    permissions.push(permission);
                }
            }
            Ok(permissions)
        } else {
            Err(AuthError::RoleNotFound(role.to_string()))
        }
    }

    /// Parse permission string to Permission enum
    fn parse_permission(&self, perm_str: &str) -> Result<Permission, AuthError> {
        match perm_str {
            "CreateCollection" => Ok(Permission::CreateCollection),
            "DeleteCollection" => Ok(Permission::DeleteCollection),
            "ListCollections" => Ok(Permission::ListCollections),
            "ReadCollectionMetadata" => Ok(Permission::ReadCollectionMetadata),
            "UpdateCollectionMetadata" => Ok(Permission::UpdateCollectionMetadata),
            "InsertVectors" => Ok(Permission::InsertVectors),
            "DeleteVectors" => Ok(Permission::DeleteVectors),
            "SearchVectors" => Ok(Permission::SearchVectors),
            "UpdateVectors" => Ok(Permission::UpdateVectors),
            "ReadVectors" => Ok(Permission::ReadVectors),
            "CreateGraphRelations" => Ok(Permission::CreateGraphRelations),
            "DeleteGraphRelations" => Ok(Permission::DeleteGraphRelations),
            "TraverseGraph" => Ok(Permission::TraverseGraph),
            "ReadGraphRelations" => Ok(Permission::ReadGraphRelations),
            "ExecuteSqlQueries" => Ok(Permission::ExecuteSqlQueries),
            "ExecuteSksFunctions" => Ok(Permission::ExecuteSksFunctions),
            "ViewSystemMetrics" => Ok(Permission::ViewSystemMetrics),
            "ViewSystemHealth" => Ok(Permission::ViewSystemHealth),
            "ConfigureSystem" => Ok(Permission::ConfigureSystem),
            "ManageUsers" => Ok(Permission::ManageUsers),
            "ManageRoles" => Ok(Permission::ManageRoles),
            "ManageApiKeys" => Ok(Permission::ManageApiKeys),
            "ViewAuditLogs" => Ok(Permission::ViewAuditLogs),
            _ => Err(AuthError::AuthenticationFailed(format!(
                "Unknown permission: {}",
                perm_str
            ))),
        }
    }

    /// Add permission to role
    pub async fn add_permission_to_role(
        &self,
        role: &str,
        permission: Permission,
    ) -> Result<(), AuthError> {
        let mut role_permissions = self.role_permissions.write().await;
        role_permissions
            .entry(role.to_string())
            .or_default()
            .insert(permission);
        Ok(())
    }

    /// Remove permission from role
    pub async fn remove_permission_from_role(
        &self,
        role: &str,
        permission: Permission,
    ) -> Result<(), AuthError> {
        let mut role_permissions = self.role_permissions.write().await;
        if let Some(permissions) = role_permissions.get_mut(role) {
            permissions.remove(&permission);
        }
        Ok(())
    }

    /// Get all users with a specific role
    pub async fn get_users_with_role(&self, role: &str) -> Vec<String> {
        let user_roles = self.user_roles.read().await;
        user_roles
            .iter()
            .filter(|(_, roles)| roles.contains(role))
            .map(|(user_id, _)| user_id.clone())
            .collect()
    }
}

impl Default for RbacService {
    fn default() -> Self {
        Self::new()
    }
}

/// Resource-based permission context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionContext {
    /// Resource type (collection, system, etc.)
    pub resource_type: ResourceType,

    /// Specific resource ID (optional)
    pub resource_id: Option<String>,

    /// Tenant ID for multi-tenancy
    pub tenant_id: Option<String>,

    /// Additional context metadata
    pub metadata: HashMap<String, String>,
}

/// Types of resources that can be protected
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ResourceType {
    /// Vector collection resource
    Collection,
    /// Individual vector resource
    Vector,
    /// Graph database resource
    Graph,
    /// System-level resource (configuration, metrics)
    System,
    /// User account resource
    User,
    /// RBAC role resource
    Role,
}

/// Extended RBAC service with resource-based permissions
impl RbacService {
    /// Check permission with resource context
    pub async fn check_permission_with_context(
        &self,
        user_id: &str,
        permission: Permission,
        context: &PermissionContext,
    ) -> Result<(), AuthError> {
        // Basic permission check
        if !self.user_has_permission(user_id, permission.clone()).await {
            return Err(AuthError::AuthorizationDenied(permission));
        }

        // Additional context-based checks
        match context.resource_type {
            ResourceType::Collection => {
                // Check if user has access to specific collection
                if let Some(collection_id) = &context.resource_id
                    && !self
                        .user_has_collection_access(user_id, collection_id)
                        .await
                {
                    return Err(AuthError::AuthorizationDenied(permission));
                }
            }
            ResourceType::System => {
                // System operations require admin role
                let roles = self.get_user_roles(user_id)?;
                if !roles.contains(&"admin".to_string()) {
                    return Err(AuthError::AuthorizationDenied(permission));
                }
            }
            _ => {} // Other resource types handled by basic permission check
        }

        // Tenant isolation check: fail-closed. This layer holds no
        // tenant-membership data, so it cannot prove the user belongs to the
        // tenant a context asserts — and an unprovable tenant claim is DENIED,
        // never silently allowed. Real membership validation would require a
        // membership store this service does not have (TD-FACADE-1). The
        // middleware never asserts a resource tenant (it cannot know one from
        // a request path), so mounted use is unaffected; a caller that asserts
        // one must bring its own tenancy decision. The system's structural
        // isolation (TenantContext + DrPathBuilder paths) is unaffected by
        // this layer either way.
        if let Some(resource_tenant) = &context.tenant_id
            && !self.user_belongs_to_tenant(user_id, resource_tenant).await
        {
            return Err(AuthError::AuthorizationDenied(permission));
        }

        Ok(())
    }

    /// Collection-access predicate.
    ///
    /// No per-collection ACLs exist in this service: access to ANY collection
    /// is granted whenever the caller's role permissions already allow the
    /// operation (checked separately by `check_permission_with_context`) and
    /// the roles read succeeds (`try_read` contention is the only failure
    /// mode; unknown users resolve to the default role). Enforcing per-collection ACLs would require
    /// an ACL store this layer does not have (TD-FACADE-1); until one exists
    /// this intentionally reduces to role permissions, and the `admin` short
    /// circuit below is the only collection-independent override.
    async fn user_has_collection_access(&self, user_id: &str, _collection_id: &str) -> bool {
        let roles = match self.get_user_roles(user_id) {
            Ok(roles) => roles,
            Err(_) => return false,
        };

        // Admins have access to all collections
        if roles.contains(&"admin".to_string()) {
            return true;
        }

        true
    }

    /// Fail-closed tenant-membership predicate (TD-FACADE-1).
    ///
    /// This service stores roles and permissions only — it has no
    /// tenant-membership data, so it cannot prove ANY user belongs to ANY
    /// tenant, and it answers `false` (deny) rather than guessing `true`.
    /// The previous body returned `true` unconditionally ("for now, assume
    /// all users can access all tenants"), which made
    /// [`Self::check_permission_with_context`] approve a context asserting a
    /// foreign tenant — a cross-tenant bypass for direct callers.
    ///
    /// Implementing real validation would require a membership store this
    /// layer does not have; until one exists, a context that asserts a
    /// resource tenant is denied. Callers that do not want a tenant decision
    /// here pass `tenant_id: None` (as the middleware does) and enforce
    /// tenancy structurally via `TenantContext` / `DrPathBuilder`.
    async fn user_belongs_to_tenant(&self, _user_id: &str, _tenant_id: &str) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rbac_service_creation() {
        let rbac = RbacService::new();
        let roles = rbac.list_roles().await;
        assert!(roles.len() >= 3); // admin, user, readonly
    }

    /// TD-FACADE-1 pin: the tenant predicate must be FAIL-CLOSED. It returned
    /// `true` unconditionally, which made `check_permission_with_context`
    /// approve a context asserting a foreign tenant — a cross-tenant bypass
    /// for direct callers. This layer holds no membership data, so any
    /// asserted tenant is unprovable and must be denied.
    #[tokio::test]
    async fn tenant_asserted_context_is_denied_fail_closed() {
        let rbac = RbacService::new();
        let user_id = "test_user"; // default "user" role: has ReadVectors

        let context = PermissionContext {
            resource_type: ResourceType::Collection,
            resource_id: Some("some_collection".to_string()),
            tenant_id: Some("any-tenant-at-all".to_string()), // even one that
            // might be the caller's own: indistinguishable without data
            metadata: HashMap::new(),
        };

        let result = rbac
            .check_permission_with_context(user_id, Permission::ReadVectors, &context)
            .await;
        assert!(
            result.is_err(),
            "a context asserting a tenant must be denied while membership cannot be proven"
        );
    }

    /// Companion to the pin above: no tenant asserted => the permission check
    /// itself decides. This is the middleware shape after it stopped echoing
    /// the caller's own tenant into the context, and it must still ALLOW a
    /// permitted operation — otherwise the pin would pass merely because
    /// everything is denied.
    #[tokio::test]
    async fn no_tenant_context_allows_permitted_operation() {
        let rbac = RbacService::new();
        let user_id = "test_user"; // default "user" role: has ReadVectors

        let context = PermissionContext {
            resource_type: ResourceType::Collection,
            resource_id: Some("some_collection".to_string()),
            tenant_id: None,
            metadata: HashMap::new(),
        };

        let result = rbac
            .check_permission_with_context(user_id, Permission::ReadVectors, &context)
            .await;
        assert!(
            result.is_ok(),
            "without a tenant assertion the role permission alone decides, and this role has ReadVectors"
        );
    }

    #[tokio::test]
    async fn test_role_assignment() {
        let rbac = RbacService::new();
        let user_id = "test_user";

        // Assign admin role
        rbac.assign_role(user_id, "admin").await.unwrap();

        // Check user roles
        let roles = rbac.get_user_roles(user_id).unwrap();
        assert!(roles.contains(&"admin".to_string()));

        // Check user permissions
        let permissions = rbac.get_permissions_for_roles(&roles).unwrap();
        assert!(permissions.contains(&Permission::ManageUsers));
    }

    #[tokio::test]
    async fn test_permission_check() {
        let rbac = RbacService::new();
        let user_id = "test_user";

        // Initially user should have default role permissions
        let has_read = rbac
            .user_has_permission(user_id, Permission::ReadVectors)
            .await;
        assert!(has_read); // default "user" role should have read permissions

        // User should not have admin permissions initially
        let has_manage = rbac
            .user_has_permission(user_id, Permission::ManageUsers)
            .await;
        assert!(!has_manage);

        // Assign admin role
        rbac.assign_role(user_id, "admin").await.unwrap();

        // Now should have admin permissions
        let has_manage = rbac
            .user_has_permission(user_id, Permission::ManageUsers)
            .await;
        assert!(has_manage);
    }

    #[tokio::test]
    async fn test_custom_role_creation() {
        let rbac = RbacService::new();
        let role_name = "custom_role";
        let permissions = vec![Permission::ReadVectors, Permission::SearchVectors];

        // Create custom role
        rbac.create_role(role_name, permissions.clone())
            .await
            .unwrap();

        // Verify role exists
        let roles = rbac.list_roles().await;
        assert!(roles.contains(&role_name.to_string()));

        // Verify role permissions
        let role_permissions = rbac.get_role_permissions(role_name).await.unwrap();
        assert_eq!(role_permissions.len(), 2);
        assert!(role_permissions.contains(&Permission::ReadVectors));
        assert!(role_permissions.contains(&Permission::SearchVectors));
    }

    #[tokio::test]
    async fn test_permission_context() {
        let rbac = RbacService::new();
        let user_id = "test_user";

        // Assign user role
        rbac.assign_role(user_id, "user").await.unwrap();

        let context = PermissionContext {
            resource_type: ResourceType::Collection,
            resource_id: Some("test_collection".to_string()),
            tenant_id: None,
            metadata: HashMap::new(),
        };

        // Should succeed for read permission
        let result = rbac
            .check_permission_with_context(user_id, Permission::ReadVectors, &context)
            .await;
        assert!(result.is_ok());

        // Should fail for admin permission
        let result = rbac
            .check_permission_with_context(user_id, Permission::ManageUsers, &context)
            .await;
        assert!(result.is_err());
    }
}
