//! Security Coordinator for ProximaDB
//!
//! Central coordination point for all security operations including
//! authentication, authorization, audit, Row-Level Security (RLS),
//! and security policy enforcement.

use super::auth_service::{AuthenticationConfig, AuthenticationData, UnifiedAuthService};
use super::encryption::{
    EncryptedField, EncryptionConfig, FieldEncryption, FieldEncryptionError, KeyStore,
    KeyStoreConfig,
};
use super::rbac_service::{
    ConsolidatedRBACManager, RBACConfig, UnifiedAuthMethod, UnifiedPermission, UnifiedUserContext,
};
use super::rls::{CollectionRLS, Operation as RLSOperation, RLSConfig, RLSFilterResult, RLSPolicy};
use crate::audit::logger::AuditLogger;
use proximadb_security::AuditConfig;
use std::collections::HashMap;

use anyhow::{Result, anyhow};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Security coordinator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub enabled: bool,
    pub mode: SecurityMode,
    pub authentication: AuthenticationConfig,
    pub rbac: RBACConfig,
    pub audit: AuditConfig,
    pub tls: TlsConfig,
    pub compliance: ComplianceConfig,
    /// Field-level encryption configuration
    #[serde(default)]
    pub encryption: EncryptionConfig,
    /// Key store configuration
    #[serde(default)]
    pub key_store: KeyStoreConfig,
    /// Tenant-identity trust configuration (`[security.tenant]`, TD-TENANT-1).
    #[serde(default)]
    pub tenant: TenantTrustConfig,
    /// pgwire authentication posture (`[security.pgwire]`, TD-PGWIRE-AUTH-1).
    #[serde(default)]
    pub pgwire: PgwireSecurityConfig,
}

/// `[security.pgwire]` — the pgwire authentication knob (TD-PGWIRE-AUTH-1).
/// Sibling of `[security.tenant]`, NOT a reuse of `authentication.enabled`:
/// that flag already means "REST/gRPC/Flight credentials required" and is the
/// *forcing* condition; this knob can RAISE pgwire above the global default
/// (require SCRAM while REST auth is off) but can never LOWER it (an env value
/// of `trust` while `authentication.enabled=true` is warn-ignored).
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct PgwireSecurityConfig {
    /// `"trust"` (default when absent) | `"password"` (SCRAM-SHA-256
    /// required). The `PROXIMADB_PGWIRE_AUTH` env override wins over the
    /// config value; the effective mode is resolved once at startup in
    /// `database.rs`.
    #[serde(default)]
    pub auth: Option<String>,
}

/// `[security.tenant]` — tenant-identity trust knobs (TD-TENANT-1).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TenantTrustConfig {
    /// Bare tenant-assertion trust policy for every network surface (REST
    /// `X-Tenant-ID`, gRPC / Arrow Flight `x-tenant-id` metadata, pgwire
    /// startup `database` / tenant session vars): `"open"` |
    /// `"authenticated-only"` | `"gateway-only"`. Unset ⇒ the deployment-mode
    /// preset (`open`; `authenticated-only` under multi-tenant strict mode).
    /// The `PROXIMADB_TENANT_HEADER_TRUST` env override wins over both.
    #[serde(default)]
    pub header_trust: Option<proximadb_tenant::HeaderTrustPolicy>,
    /// Tier-claim trust policy (ADR-0053 W8) for the entitlement claim every
    /// network surface reads (REST `X-Tenant-Tier`, gRPC `x-tenant-tier`
    /// metadata, pgwire `proximadb_tier` startup parameter). Unset ⇒ the
    /// deployment-mode preset (same ladder as `header_trust`). The
    /// `PROXIMADB_TIER_HEADER_TRUST` env override wins over both. A rejected
    /// claim is dropped (never a 4xx) — the tenant resolves its default tier.
    #[serde(default)]
    pub tier_header_trust: Option<proximadb_tenant::HeaderTrustPolicy>,
}

/// Security mode for different deployment scenarios
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SecurityMode {
    #[serde(rename = "development")]
    Development,
    #[serde(rename = "production")]
    Production,
    #[serde(rename = "enterprise")]
    Enterprise,
}

/// TLS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    pub enabled: bool,
    pub require_client_certificates: bool,
    pub cert_file: Option<String>,
    pub key_file: Option<String>,
    pub ca_file: Option<String>,
}

/// Compliance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceConfig {
    pub frameworks: Vec<String>,
    pub data_residency: Option<String>,
    pub encryption_at_rest: bool,
    pub encryption_in_transit: bool,
}

/// Central security coordinator
pub struct SecurityCoordinator {
    /// Unified authentication service
    auth_service: Arc<UnifiedAuthService>,

    /// Consolidated RBAC manager
    rbac_manager: Arc<ConsolidatedRBACManager>,

    /// Row-Level Security service
    rls_service: Arc<CollectionRLS>,

    /// Field-level encryption service
    encryption_service: Option<Arc<FieldEncryption>>,

    /// Key store for encryption keys
    key_store: Option<Arc<KeyStore>>,

    /// Audit logger
    audit_logger: Arc<AuditLogger>,

    /// Security configuration
    config: SecurityConfig,
}

impl SecurityCoordinator {
    /// Create new security coordinator
    pub fn new(
        auth_service: UnifiedAuthService,
        rbac_manager: ConsolidatedRBACManager,
        audit_logger: Arc<AuditLogger>,
        config: SecurityConfig,
    ) -> Self {
        // Initialize RLS with default config
        let rls_service = CollectionRLS::new(RLSConfig::default());

        // Initialize encryption if enabled
        let (key_store, encryption_service) = if config.encryption.enabled {
            match KeyStore::new(config.key_store.clone()) {
                Ok(ks) => {
                    let key_store = Arc::new(ks);
                    // Create default encryption key if needed
                    if !key_store.key_exists("default") {
                        let _ = key_store.create_key("default", "Default encryption key");
                    }
                    match FieldEncryption::new(Arc::clone(&key_store), config.encryption.clone()) {
                        Ok(fe) => (Some(key_store), Some(Arc::new(fe))),
                        Err(_) => (Some(key_store), None),
                    }
                }
                Err(_) => (None, None),
            }
        } else {
            (None, None)
        };

        Self {
            auth_service: Arc::new(auth_service),
            rbac_manager: Arc::new(rbac_manager),
            rls_service: Arc::new(rls_service),
            encryption_service,
            key_store,
            audit_logger,
            config,
        }
    }

    /// Get the RLS service for direct access
    pub fn rls_service(&self) -> Arc<CollectionRLS> {
        Arc::clone(&self.rls_service)
    }

    /// Register an RLS policy for a collection
    pub async fn register_rls_policy(&self, policy: RLSPolicy) -> Result<()> {
        self.rls_service.register_policy(policy).await
    }

    /// Remove an RLS policy from a collection
    pub async fn remove_rls_policy(&self, collection: &str, policy_name: &str) -> Result<()> {
        self.rls_service
            .remove_policy(collection, policy_name)
            .await
    }

    /// Apply RLS filters for a search operation
    ///
    /// Returns the security filters that should be applied to the search based on
    /// the user's context and the collection's RLS policies.
    pub async fn apply_rls_filter(
        &self,
        collection: &str,
        operation: RLSOperation,
        user_context: &UnifiedUserContext,
    ) -> Result<Arc<RLSFilterResult>> {
        self.rls_service
            .apply_security_filter(collection, &operation, user_context)
            .await
    }

    // ============== Field-Level Encryption Methods ==============

    /// Check if field-level encryption is enabled
    pub fn encryption_enabled(&self) -> bool {
        self.encryption_service.is_some()
    }

    /// Get the encryption service for direct access
    pub fn encryption_service(&self) -> Option<Arc<FieldEncryption>> {
        self.encryption_service.clone()
    }

    /// Get the key store for key management
    pub fn key_store(&self) -> Option<Arc<KeyStore>> {
        self.key_store.clone()
    }

    /// Encrypt a metadata field value
    pub fn encrypt_field(
        &self,
        field_name: &str,
        value: &serde_json::Value,
    ) -> Result<EncryptedField, FieldEncryptionError> {
        let service = self.encryption_service.as_ref().ok_or_else(|| {
            FieldEncryptionError::EncryptionFailed("Encryption not enabled".into())
        })?;
        service.encrypt_field(field_name, value)
    }

    /// Decrypt an encrypted field
    pub fn decrypt_field(
        &self,
        encrypted: &EncryptedField,
    ) -> Result<serde_json::Value, FieldEncryptionError> {
        let service = self.encryption_service.as_ref().ok_or_else(|| {
            FieldEncryptionError::DecryptionFailed("Encryption not enabled".into())
        })?;
        service.decrypt_field(encrypted)
    }

    /// Encrypt multiple metadata fields in a record
    pub fn encrypt_record_metadata(
        &self,
        metadata: &mut HashMap<String, serde_json::Value>,
    ) -> Result<HashMap<String, EncryptedField>, FieldEncryptionError> {
        let service = self.encryption_service.as_ref().ok_or_else(|| {
            FieldEncryptionError::EncryptionFailed("Encryption not enabled".into())
        })?;
        service.encrypt_record_metadata(metadata)
    }

    /// Decrypt all encrypted fields in a record
    pub fn decrypt_record_metadata(
        &self,
        encrypted_fields: &HashMap<String, EncryptedField>,
    ) -> Result<HashMap<String, serde_json::Value>, FieldEncryptionError> {
        let service = self.encryption_service.as_ref().ok_or_else(|| {
            FieldEncryptionError::DecryptionFailed("Encryption not enabled".into())
        })?;
        service.decrypt_record_metadata(encrypted_fields)
    }

    /// Generate a search index for an encrypted field query
    pub fn generate_search_index(
        &self,
        value: &serde_json::Value,
        truncate_bytes: Option<usize>,
    ) -> Result<String, FieldEncryptionError> {
        let service = self.encryption_service.as_ref().ok_or_else(|| {
            FieldEncryptionError::EncryptionFailed("Encryption not enabled".into())
        })?;
        service.generate_search_index(value, truncate_bytes)
    }

    /// Create an encryption key for a specific purpose
    pub fn create_encryption_key(
        &self,
        key_id: &str,
        purpose: &str,
    ) -> Result<(), FieldEncryptionError> {
        let store = self.key_store.as_ref().ok_or_else(|| {
            FieldEncryptionError::EncryptionFailed("Key store not enabled".into())
        })?;
        store.create_key(key_id, purpose)?;
        Ok(())
    }

    /// Rotate an encryption key
    pub fn rotate_encryption_key(&self, key_id: &str) -> Result<(), FieldEncryptionError> {
        let store = self.key_store.as_ref().ok_or_else(|| {
            FieldEncryptionError::EncryptionFailed("Key store not enabled".into())
        })?;
        store.rotate_key(key_id)?;
        Ok(())
    }

    // ============== Authentication & Authorization Methods ==============

    /// Full authentication and authorization flow
    pub async fn authenticate_and_authorize(
        &self,
        auth_data: AuthenticationData,
        requested_permission: UnifiedPermission,
    ) -> Result<AuthorizedContext> {
        let start_time = Utc::now();

        // Step 1: Authenticate user
        let auth_result = self.auth_service.authenticate(auth_data).await?;

        if !auth_result.success {
            return Err(anyhow!(
                "Authentication failed: {}",
                auth_result
                    .error_message
                    .unwrap_or("Unknown error".to_string())
            ));
        }

        // Step 2: Populate effective permissions via RBAC
        let mut user_context = auth_result.user_context;
        let effective_permissions = self
            .rbac_manager
            .get_effective_permissions(&user_context)
            .await?;
        user_context.effective_permissions = effective_permissions;

        // Step 3: Check authorization for requested permission
        let authorized = self
            .rbac_manager
            .check_permission(&user_context, &requested_permission)
            .await?;

        if !authorized {
            // Log the authorization failure. The `?` here used to REPLACE the
            // permission error with the audit error: the caller was told
            // "audit write failed" instead of "insufficient permissions",
            // which both misclassifies the outcome and leaks internal detail
            // to a caller that was being denied anyway. Record the audit
            // failure loudly, then always return the permission error.
            if let Err(error) = self
                .audit_logger
                .log_event(create_authorization_failure_event(
                    &user_context,
                    &requested_permission,
                    start_time,
                ))
                .await
            {
                crate::security::auth_service::AUDIT_WRITE_FAILURES
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                tracing::error!(
                    target: "proximadb.security.audit",
                    %error,
                    "AUDIT WRITE FAILED for an authorization denial - the denial still stands"
                );
            }

            return Err(anyhow!("Insufficient permissions for requested operation"));
        }

        // Step 4: Create authorized context
        Ok(AuthorizedContext {
            user_context,
            granted_permission: requested_permission,
            session_metadata: SessionMetadata {
                authenticated_at: start_time,
                auth_method: auth_result.auth_method,
                requires_mfa: auth_result.requires_mfa,
            },
        })
    }

    /// Simplified authentication for API endpoints
    pub async fn authenticate_request(
        &self,
        auth_data: AuthenticationData,
    ) -> Result<UnifiedUserContext> {
        let auth_result = self.auth_service.authenticate(auth_data).await?;

        if !auth_result.success {
            return Err(anyhow!("Authentication failed"));
        }

        // Populate effective permissions
        let mut user_context = auth_result.user_context;
        let effective_permissions = self
            .rbac_manager
            .get_effective_permissions(&user_context)
            .await?;
        user_context.effective_permissions = effective_permissions;

        Ok(user_context)
    }

    /// The SCRAM verifier for `username`, if configured (TD-PGWIRE-AUTH-1).
    /// The pgwire protocol layer runs the SASL exchange against it BEFORE any
    /// identity exists; resolution then flows through
    /// [`Self::authenticate_request`] with
    /// `AuthenticationData::ScramAuthenticated`.
    pub fn scram_verifier(
        &self,
        username: &str,
    ) -> Option<crate::network::postgres::scram::ScramVerifier> {
        self.auth_service.scram_verifier(username)
    }

    /// True when any pgwire SCRAM identity is configured.
    pub fn has_scram_users(&self) -> bool {
        self.auth_service.has_scram_users()
    }

    /// Check if user has permission for operation
    pub async fn check_authorization(
        &self,
        user_context: &UnifiedUserContext,
        permission: &UnifiedPermission,
    ) -> Result<bool> {
        self.rbac_manager
            .check_permission(user_context, permission)
            .await
    }

    /// Get security policy for tenant
    pub async fn get_tenant_security_policy(
        &self,
        _tenant_id: &str,
    ) -> Result<TenantSecurityPolicy> {
        // This would integrate with tenant manager
        // For now, return default policy based on security mode
        Ok(match self.config.mode {
            SecurityMode::Development => TenantSecurityPolicy::development(),
            SecurityMode::Production => TenantSecurityPolicy::production(),
            SecurityMode::Enterprise => TenantSecurityPolicy::enterprise(),
        })
    }

    /// Initialize security coordinator from configuration
    pub async fn from_config(config: SecurityConfig) -> Result<Self> {
        // Create RBAC manager
        let rbac_manager = ConsolidatedRBACManager::new(config.rbac.clone());

        // Create auth service
        let auth_service = UnifiedAuthService::new(config.authentication.clone())?;

        // Create audit logger - config.audit is already AuditConfig
        let audit_logger = AuditLogger::new(config.audit.clone()).await?;

        Ok(Self::new(
            auth_service,
            rbac_manager,
            Arc::new(audit_logger),
            config,
        ))
    }

    /// Health check for security subsystem
    pub async fn health_check(&self) -> SecurityHealthStatus {
        let mut status = SecurityHealthStatus {
            overall_healthy: true,
            authentication_healthy: true,
            rbac_healthy: true,
            audit_healthy: true,
            tls_healthy: true,
            issues: vec![],
        };

        // Check authentication health
        if !self.config.authentication.enabled && self.config.mode != SecurityMode::Development {
            status.authentication_healthy = false;
            status
                .issues
                .push("Authentication disabled in non-development mode".to_string());
        }

        // Check TLS health
        if !self.config.tls.enabled && self.config.mode == SecurityMode::Production {
            status.tls_healthy = false;
            status
                .issues
                .push("TLS disabled in production mode".to_string());
        }

        // Check audit health
        if !self.config.audit.enable_audit_logging {
            status.audit_healthy = false;
            status.issues.push("Audit logging disabled".to_string());
        }

        status.overall_healthy = status.authentication_healthy
            && status.rbac_healthy
            && status.audit_healthy
            && status.tls_healthy;

        status
    }
}

/// Authorized context after successful authentication and authorization
#[derive(Debug, Clone)]
pub struct AuthorizedContext {
    pub user_context: UnifiedUserContext,
    pub granted_permission: UnifiedPermission,
    pub session_metadata: SessionMetadata,
}

/// Session metadata
#[derive(Debug, Clone)]
pub struct SessionMetadata {
    pub authenticated_at: chrono::DateTime<Utc>,
    pub auth_method: UnifiedAuthMethod,
    pub requires_mfa: bool,
}

/// Tenant security policy
#[derive(Debug, Clone)]
pub struct TenantSecurityPolicy {
    pub require_authentication: bool,
    pub require_mfa: bool,
    pub session_timeout_minutes: u64,
    pub allowed_auth_methods: Vec<super::auth_service::AuthenticationMethod>,
    pub audit_level: AuditLevel,
    pub compliance_frameworks: Vec<String>,
}

impl TenantSecurityPolicy {
    pub fn development() -> Self {
        Self {
            require_authentication: false,
            require_mfa: false,
            session_timeout_minutes: 480,
            allowed_auth_methods: vec![super::auth_service::AuthenticationMethod::ApiKey],
            audit_level: AuditLevel::Basic,
            compliance_frameworks: vec![],
        }
    }

    pub fn production() -> Self {
        Self {
            require_authentication: true,
            require_mfa: false,
            session_timeout_minutes: 240,
            allowed_auth_methods: vec![
                super::auth_service::AuthenticationMethod::JWT,
                super::auth_service::AuthenticationMethod::ApiKey,
            ],
            audit_level: AuditLevel::Comprehensive,
            compliance_frameworks: vec!["SOC2".to_string()],
        }
    }

    pub fn enterprise() -> Self {
        Self {
            require_authentication: true,
            require_mfa: true,
            session_timeout_minutes: 120,
            // SSO removed (TD-SSO-1): the legacy SSO path is fail-closed;
            // enterprise OIDC federation rides the JWT method.
            allowed_auth_methods: vec![
                super::auth_service::AuthenticationMethod::JWT,
                super::auth_service::AuthenticationMethod::ClientCertificate,
            ],
            audit_level: AuditLevel::Full,
            compliance_frameworks: vec![
                "SOC2".to_string(),
                "GDPR".to_string(),
                "HIPAA".to_string(),
            ],
        }
    }
}

/// Audit level configuration
#[derive(Debug, Clone)]
pub enum AuditLevel {
    None,
    Basic,
    Comprehensive,
    Full,
}

/// Security health status
#[derive(Debug, Clone)]
pub struct SecurityHealthStatus {
    pub overall_healthy: bool,
    pub authentication_healthy: bool,
    pub rbac_healthy: bool,
    pub audit_healthy: bool,
    pub tls_healthy: bool,
    pub issues: Vec<String>,
}

/// Create authorization failure audit event
fn create_authorization_failure_event(
    user_context: &UnifiedUserContext,
    requested_permission: &UnifiedPermission,
    _start_time: chrono::DateTime<Utc>,
) -> proximadb_security::AuditEvent {
    use proximadb_security::{AuditEvent, AuditEventType, AuditResource, AuditResult};
    use std::collections::HashMap;

    AuditEvent {
        event_id: uuid::Uuid::new_v4().to_string(),
        event_type: AuditEventType::Authorization,
        timestamp: Utc::now(),
        user_id: Some(user_context.user_id.clone()),
        tenant_id: user_context.tenant_id.clone(),
        resource: AuditResource {
            resource_type: "permission".to_string(),
            resource_id: format!("{:?}", requested_permission),
            parent_resource: None,
            resource_tenant_id: None,
        },
        action: "check_permission".to_string(),
        result: AuditResult::Failure {
            error_code: "PERMISSION_DENIED".to_string(),
            error_message: "Permission denied".to_string(),
        },
        ip_address: None,
        user_agent: None,
        session_id: Some(user_context.session_id.clone()),
        request_id: None,
        details: {
            let mut details = HashMap::new();
            details.insert(
                "requested_permission".to_string(),
                serde_json::json!(format!("{:?}", requested_permission)),
            );
            details.insert(
                "user_roles".to_string(),
                serde_json::json!(user_context.roles),
            );
            details.insert(
                "auth_method".to_string(),
                serde_json::json!(format!("{:?}", user_context.auth_method)),
            );
            if let Some(tenant) = &user_context.tenant_id {
                details.insert("tenant_id".to_string(), serde_json::json!(tenant));
            }
            details
        },
        risk_score: Some(0.75), // High risk for authorization failures
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::AuthenticationMethod;
    use crate::security::auth_service::{JwtConfig, MtlsConfig, SSOConfig};
    use std::collections::HashMap;
    use tracing::{debug, info, warn};

    fn create_test_config() -> SecurityConfig {
        SecurityConfig {
            enabled: true,
            mode: SecurityMode::Development,
            authentication: AuthenticationConfig {
                enabled: true,
                methods: vec![AuthenticationMethod::ApiKey],
                require_authentication: false,
                default_session_timeout_minutes: 480,
                api_keys: HashMap::new(),
                scram_users: HashMap::new(),
                jwt: JwtConfig {
                    enabled: false,
                    secret: "test-secret".to_string(),
                    access_token_expiration_minutes: 15,
                    refresh_token_expiration_days: 7,
                    issuer: "test".to_string(),
                    audience: "test".to_string(),
                    algorithm: "HS256".to_string(),
                },
                sso: SSOConfig {
                    enabled: false,
                    providers: vec![],
                    token_cache_ttl_minutes: 5,
                },
                mtls: MtlsConfig::default(),
                oidc: None,
                audit_fail_closed: false,
            },
            rbac: RBACConfig::default(),
            audit: AuditConfig::default(),
            tls: TlsConfig {
                enabled: false,
                require_client_certificates: false,
                cert_file: None,
                key_file: None,
                ca_file: None,
            },
            compliance: ComplianceConfig {
                frameworks: vec![],
                data_residency: None,
                encryption_at_rest: false,
                encryption_in_transit: false,
            },
            encryption: EncryptionConfig::default(),
            key_store: KeyStoreConfig::default(),
            tenant: Default::default(),
            pgwire: PgwireSecurityConfig::default(),
        }
    }

    #[tokio::test]
    async fn test_security_coordinator_creation() {
        let config = create_test_config();
        let coordinator_result = SecurityCoordinator::from_config(config).await;

        // This test will fail until we implement the dependencies
        // but it documents the expected interface
        match coordinator_result {
            Ok(_) => info!("Security coordinator created successfully"),
            Err(e) => warn!("Security coordinator creation failed: {}", e),
        }
    }

    #[tokio::test]
    async fn test_security_health_check() {
        let config = create_test_config();

        // Create minimal coordinator for health check testing
        // This is a placeholder until full implementation is complete
        match SecurityCoordinator::from_config(config).await {
            Ok(coordinator) => {
                let health = coordinator.health_check().await;
                debug!("Security health check: {:?}", health);
            }
            Err(e) => {
                warn!("Could not create coordinator for health check: {}", e);
            }
        }
    }

    #[test]
    fn test_tenant_security_policies() {
        let dev_policy = TenantSecurityPolicy::development();
        assert!(!dev_policy.require_authentication);
        assert!(!dev_policy.require_mfa);

        let prod_policy = TenantSecurityPolicy::production();
        assert!(prod_policy.require_authentication);
        assert!(!prod_policy.require_mfa);

        let enterprise_policy = TenantSecurityPolicy::enterprise();
        assert!(enterprise_policy.require_authentication);
        assert!(enterprise_policy.require_mfa);
        assert!(
            enterprise_policy
                .compliance_frameworks
                .contains(&"SOC2".to_string())
        );
    }

    #[tokio::test]
    async fn test_encryption_integration() {
        use super::super::encryption::{EncryptionType, FieldEncryptionSettings};

        // Create config with encryption enabled
        let mut config = create_test_config();
        config.encryption.enabled = true;
        config.encryption.field_settings.insert(
            "ssn".to_string(),
            FieldEncryptionSettings {
                encryption_type: EncryptionType::Deterministic,
                key_id: "default".to_string(),
                blind_index: true,
                blind_index_bytes: Some(8),
            },
        );

        match SecurityCoordinator::from_config(config).await {
            Ok(coordinator) => {
                // Test encryption is enabled
                assert!(coordinator.encryption_enabled());

                // Test field encryption
                let value = serde_json::json!("123-45-6789");
                let encrypted = coordinator.encrypt_field("ssn", &value);
                assert!(encrypted.is_ok());

                let encrypted = encrypted.unwrap();
                assert_eq!(encrypted.encryption_type, EncryptionType::Deterministic);
                assert!(encrypted.blind_index.is_some());

                // Test decryption
                let decrypted = coordinator.decrypt_field(&encrypted);
                assert!(decrypted.is_ok());
                assert_eq!(decrypted.unwrap(), value);

                info!("Encryption integration test passed");
            }
            Err(e) => {
                warn!("Could not create coordinator for encryption test: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_encryption_disabled() {
        let config = create_test_config();
        // encryption.enabled defaults to false

        match SecurityCoordinator::from_config(config).await {
            Ok(coordinator) => {
                assert!(!coordinator.encryption_enabled());

                // Encryption should fail when disabled
                let value = serde_json::json!("test");
                let result = coordinator.encrypt_field("field", &value);
                assert!(result.is_err());
            }
            Err(e) => {
                warn!("Could not create coordinator: {}", e);
            }
        }
    }
}
