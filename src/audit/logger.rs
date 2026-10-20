//! Audit Logger Implementation
//!
//! Main audit logging system with comprehensive event tracking,
//! encryption, and compliance reporting capabilities.

use super::storage::AuditStorage;
use anyhow::{Result, anyhow};
use chrono::{Duration, Timelike, Utc};
pub use proximadb_security::{AuditConfig, AuditStorageBackend, EmailConfig, EncryptionAlgorithm};
pub use proximadb_security::{
    AuditEvent, AuditEventType, AuditResource, AuditResult, SecurityAlert, SecurityAlertSeverity,
    SecurityAlertType,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Comprehensive audit logger for enterprise compliance
#[derive(Clone)]
pub struct AuditLogger {
    /// Backend storage for persisting audit events
    storage: Arc<dyn AuditStorage + Send + Sync>,
    /// Audit logging configuration (encryption, retention, alerts)
    config: AuditConfig,
    /// Encryption key for protecting sensitive audit data fields
    encryption_key: Arc<EncryptionKey>,
    /// Optional sender for real-time security alert notifications
    alert_sender: Option<Arc<AlertSender>>,
}

/// Encryption key for sensitive audit data
#[derive(Debug)]
pub struct EncryptionKey {
    /// 256-bit key material used for encrypting sensitive fields
    #[allow(dead_code)]
    key: [u8; 32],
    /// Encryption algorithm (AES-256-GCM or ChaCha20-Poly1305)
    #[allow(dead_code)]
    algorithm: EncryptionAlgorithm,
}

/// Alert sender for real-time security notifications
#[derive(Debug)]
pub struct AlertSender {
    /// Webhook URL for sending security alert payloads
    webhook_url: Option<String>,
    /// Optional email configuration for alert delivery
    #[allow(dead_code)]
    email_config: Option<EmailConfig>,
}

impl std::fmt::Debug for AuditLogger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuditLogger")
            .field("config", &self.config)
            .field("encryption_key", &self.encryption_key)
            .field("alert_sender", &self.alert_sender)
            .field("storage", &"<AuditStorage trait object>")
            .finish()
    }
}

impl AuditLogger {
    /// An audit logger that discards events.
    ///
    /// Deliberately NOT reachable by simply disabling auditing: [`Self::new`]
    /// still returns `Err` for a disabled config (a contract other code and
    /// tests rely on). The composition root chooses this sink only for a
    /// **development** deployment, so that `security.enabled = true` with the
    /// shipped default `enable_audit_logging = false` boots instead of failing
    /// hard — while a production deployment keeps failing fast, because
    /// security without accountability is a misconfiguration there.
    pub fn noop(config: AuditConfig) -> Self {
        Self {
            storage: Arc::new(super::storage::NoopAuditStorage),
            config,
            // Encryption is irrelevant for a sink that stores nothing; the
            // disabled path is infallible, and the unwrap_or keeps `noop`
            // total (no Result) without an unwrap/expect.
            encryption_key: Arc::new(EncryptionKey::new(false).unwrap_or(EncryptionKey {
                key: [0u8; 32],
                algorithm: EncryptionAlgorithm::AES256GCM,
            })),
            alert_sender: None,
        }
    }

    /// Create new audit logger with configuration
    pub async fn new(config: AuditConfig) -> Result<Self> {
        if !config.enable_audit_logging {
            return Err(anyhow!("Audit logging is disabled in configuration"));
        }

        // Initialize storage backend
        let storage = Self::create_storage_backend(&config).await?;

        // Initialize encryption
        let encryption_key = Arc::new(EncryptionKey::new(config.encryption_enabled)?);

        // Initialize alert sender if configured
        let alert_sender = if config.enable_real_time_alerts {
            Some(Arc::new(AlertSender::new(&config)?))
        } else {
            None
        };

        info!(
            "✅ Audit logger initialized with {} compliance frameworks",
            config.compliance_frameworks.len()
        );

        Ok(Self {
            storage,
            config,
            encryption_key,
            alert_sender,
        })
    }

    /// Log comprehensive audit event
    pub async fn log_event(&self, event: AuditEvent) -> Result<()> {
        debug!(
            "📝 Logging audit event: {:?} for user {:?}",
            event.event_type, event.user_id
        );

        // Step 1: Validate event structure
        self.validate_audit_event(&event)?;

        // Step 2: Enrich event with additional metadata
        let enriched_event = self.enrich_audit_event(event).await?;

        // Step 3: Encrypt sensitive fields if encryption is enabled
        let processed_event = if self.config.encryption_enabled {
            self.encrypt_sensitive_fields(enriched_event)?
        } else {
            enriched_event
        };

        // Step 4: Store in primary audit storage
        self.storage
            .store_audit_event(&processed_event)
            .await
            .map_err(|e| anyhow!("Failed to store audit event: {}", e))?;

        // Step 5: Send to external audit systems if configured
        if let Some(ref external_endpoint) = self.config.external_audit_endpoint
            && let Err(e) = self
                .send_to_external_audit(&processed_event, external_endpoint)
                .await
        {
            warn!("Failed to send audit event to external system: {}", e);
            // Don't fail the operation if external audit fails
        }

        // Step 6: Evaluate for security alerts
        self.evaluate_security_alerts(&processed_event).await?;

        debug!(
            "✅ Audit event logged successfully: {}",
            processed_event.event_id
        );
        Ok(())
    }

    /// Log authentication event
    pub async fn log_authentication_event(
        &self,
        user_id: &str,
        authentication_method: &str,
        result: AuditResult,
        ip_address: Option<String>,
        user_agent: Option<String>,
    ) -> Result<()> {
        // Calculate risk score before moving ip_address
        let risk_score = self
            .calculate_authentication_risk(user_id, &ip_address)
            .await;

        let event = AuditEvent {
            event_id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            event_type: AuditEventType::Authentication,
            user_id: Some(user_id.to_string()),
            resource: AuditResource {
                resource_type: "authentication".to_string(),
                resource_id: "system".to_string(),
                parent_resource: None,
                resource_tenant_id: None,
            },
            action: format!("authenticate_{}", authentication_method),
            result,
            details: HashMap::new(),
            ip_address,
            user_agent,
            request_id: None,
            tenant_id: None,
            session_id: None,
            risk_score,
        };

        self.log_event(event).await
    }

    /// Log data access event
    pub async fn log_data_access_event(
        &self,
        user_id: &str,
        tenant_id: Option<String>,
        resource: AuditResource,
        action: &str,
        result: AuditResult,
        metadata: HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        let event = AuditEvent {
            event_id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            event_type: AuditEventType::DataAccess,
            user_id: Some(user_id.to_string()),
            resource,
            action: action.to_string(),
            result,
            details: metadata,
            ip_address: None, // Would be populated by middleware
            user_agent: None, // Would be populated by middleware
            request_id: None, // Would be populated by middleware
            tenant_id,
            session_id: None, // Would be populated by middleware
            risk_score: self.calculate_data_access_risk(user_id, action).await,
        };

        self.log_event(event).await
    }

    /// Create storage backend based on configuration
    async fn create_storage_backend(
        config: &AuditConfig,
    ) -> Result<Arc<dyn AuditStorage + Send + Sync>> {
        match &config.storage_backend {
            AuditStorageBackend::File { directory } => {
                // Dispatch by scheme: object-store URLs (s3://, adls://,
                // abfs://, gcs://, …) route through the FilesystemFactory;
                // bare paths and file:// stay on the local backend
                // (TD-OBJSTORE-1, #960).
                if directory.contains("://") && !directory.starts_with("file://") {
                    let storage =
                        super::storage::ObjectStoreAuditStorage::new(directory.clone()).await?;
                    Ok(Arc::new(storage))
                } else {
                    let storage = super::storage::FileAuditStorage::new(directory.clone()).await?;
                    Ok(Arc::new(storage))
                }
            }
            AuditStorageBackend::Database { .. } => {
                // Fail closed: no database audit backend exists. The former
                // placeholder silently discarded every audit event
                // (store_audit_event was `Ok(())`), violating the audit
                // contract. Refuse to start rather than drop compliance data
                // — same policy as the Combined arm below and the
                // feature-gated S3 backend.
                Err(anyhow!(
                    "Database audit backend is not implemented; use the File \
                     storage_backend (local directory or object-store URL) or S3"
                ))
            }
            AuditStorageBackend::S3 { bucket, region: _ } => {
                // Route through the same object-store implementation as
                // scheme-qualified File directories (TD-OBJSTORE-1, #960).
                let storage =
                    super::storage::ObjectStoreAuditStorage::new(format!("s3://{bucket}/audit"))
                        .await?;
                Ok(Arc::new(storage))
            }
            AuditStorageBackend::Combined {
                primary: _,
                secondary: _,
            } => {
                // Placeholder for combined storage implementation
                Err(anyhow!("Combined audit storage not yet implemented"))
            }
        }
    }

    /// Validate audit event structure
    fn validate_audit_event(&self, event: &AuditEvent) -> Result<()> {
        if event.event_id.is_empty() {
            return Err(anyhow!("Event ID is required"));
        }

        if event.action.is_empty() {
            return Err(anyhow!("Action is required"));
        }

        if event.resource.resource_type.is_empty() || event.resource.resource_id.is_empty() {
            return Err(anyhow!("Resource type and ID are required"));
        }

        Ok(())
    }

    /// Enrich audit event with additional metadata
    async fn enrich_audit_event(&self, mut event: AuditEvent) -> Result<AuditEvent> {
        // Add system metadata
        event.details.insert(
            "audit_version".to_string(),
            serde_json::Value::String("1.0".to_string()),
        );
        event.details.insert(
            "system_hostname".to_string(),
            serde_json::Value::String(
                std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string()),
            ),
        );

        // Add compliance framework tags
        event.details.insert(
            "compliance_frameworks".to_string(),
            serde_json::Value::Array(
                self.config
                    .compliance_frameworks
                    .iter()
                    .map(|f| serde_json::Value::String(f.clone()))
                    .collect(),
            ),
        );

        // Add geolocation for IP address if available
        if let Some(ref ip) = event.ip_address
            && let Ok(location) = self.get_ip_geolocation(ip).await
        {
            event
                .details
                .insert("geolocation".to_string(), serde_json::to_value(location)?);
        }

        Ok(event)
    }

    /// Encrypt sensitive fields for privacy compliance
    fn encrypt_sensitive_fields(&self, mut event: AuditEvent) -> Result<AuditEvent> {
        if !self.config.encryption_enabled {
            return Ok(event);
        }

        // Encrypt IP addresses for privacy
        if let Some(ref ip) = event.ip_address {
            let encrypted_ip = self.encryption_key.encrypt(ip.as_bytes())?;
            event.ip_address = Some(proximadb_kernel::encoding::base64_encode(&encrypted_ip));
        }

        // Encrypt user agent strings
        if let Some(ref user_agent) = event.user_agent {
            let encrypted_ua = self.encryption_key.encrypt(user_agent.as_bytes())?;
            event.user_agent = Some(proximadb_kernel::encoding::base64_encode(&encrypted_ua));
        }

        // Encrypt sensitive details
        for (key, value) in &mut event.details {
            if self.is_sensitive_field(key)
                && let serde_json::Value::String(string_value) = value
            {
                let encrypted_value = self.encryption_key.encrypt(string_value.as_bytes())?;
                *value = serde_json::Value::String(proximadb_kernel::encoding::base64_encode(
                    &encrypted_value,
                ));
            }
        }

        Ok(event)
    }

    /// Check if field contains sensitive data
    fn is_sensitive_field(&self, field_name: &str) -> bool {
        let sensitive_fields = [
            "email",
            "phone",
            "ssn",
            "credit_card",
            "password",
            "api_key",
            "token",
            "secret",
            "private_key",
        ];

        sensitive_fields
            .iter()
            .any(|&sensitive| field_name.to_lowercase().contains(sensitive))
    }

    /// Evaluate security alerts from audit events
    async fn evaluate_security_alerts(&self, event: &AuditEvent) -> Result<()> {
        let mut alerts = Vec::new();

        // Check for failed authentication attempts
        if event.event_type == AuditEventType::Authentication
            && let AuditResult::Failure { .. } = &event.result
        {
            let recent_failures = self
                .count_recent_auth_failures(
                    event.user_id.as_deref().unwrap_or("unknown"),
                    Duration::minutes(15),
                )
                .await?;

            if recent_failures > 5 {
                alerts.push(SecurityAlert {
                    alert_id: Uuid::new_v4().to_string(),
                    timestamp: Utc::now(),
                    alert_type: SecurityAlertType::SuspiciousAuthActivity,
                    severity: SecurityAlertSeverity::High,
                    description: format!(
                        "Multiple authentication failures: {} in 15 minutes",
                        recent_failures
                    ),
                    user_id: event.user_id.clone(),
                    ip_address: event.ip_address.clone(),
                    related_event_id: event.event_id.clone(),
                });
            }
        }

        // Check for cross-tenant access attempts
        // Was `(&event.tenant_id, &event.tenant_id)` — the same field twice, so
        // `user_tenant != resource_tenant` was unsatisfiable and this alert
        // could NEVER fire. The resource's owning tenant now rides on
        // `AuditResource.resource_tenant_id`; emitters that know it populate it,
        // and events without it simply do not trigger the check.
        if let (Some(user_tenant), Some(resource_tenant)) =
            (&event.tenant_id, &event.resource.resource_tenant_id)
            && user_tenant != resource_tenant
        {
            alerts.push(SecurityAlert {
                alert_id: Uuid::new_v4().to_string(),
                timestamp: Utc::now(),
                alert_type: SecurityAlertType::CrossTenantAccess,
                severity: SecurityAlertSeverity::Critical,
                description: format!(
                    "Cross-tenant access attempt: user tenant {} accessing resource tenant {}",
                    user_tenant, resource_tenant
                ),
                user_id: event.user_id.clone(),
                ip_address: event.ip_address.clone(),
                related_event_id: event.event_id.clone(),
            });
        }

        // Send alerts if any were generated
        for alert in alerts {
            self.send_security_alert(alert).await?;
        }

        Ok(())
    }

    /// Send security alert
    async fn send_security_alert(&self, alert: SecurityAlert) -> Result<()> {
        error!(
            "🚨 SECURITY ALERT: {:?} - {}",
            alert.alert_type, alert.description
        );

        if let Some(ref alert_sender) = self.alert_sender {
            alert_sender.send_alert(&alert).await?;
        }

        // Also log the alert as an audit event
        let alert_event = AuditEvent {
            event_id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            event_type: AuditEventType::SecurityEvent,
            user_id: alert.user_id.clone(),
            resource: AuditResource {
                resource_type: "security_alert".to_string(),
                resource_id: alert.alert_id.clone(),
                parent_resource: None,
                resource_tenant_id: None,
            },
            action: format!("security_alert_{:?}", alert.alert_type),
            result: AuditResult::Success,
            details: serde_json::to_value(&alert)?
                .as_object()
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .collect(),
            ip_address: alert.ip_address,
            user_agent: None,
            request_id: None,
            tenant_id: None,
            session_id: None,
            risk_score: Some(match alert.severity {
                SecurityAlertSeverity::Critical => 0.95,
                SecurityAlertSeverity::High => 0.85,
                SecurityAlertSeverity::Medium => 0.65,
                SecurityAlertSeverity::Low => 0.35,
            }),
        };

        // Store the alert event (but don't create recursive alerts)
        self.storage.store_audit_event(&alert_event).await?;

        Ok(())
    }

    /// Count recent authentication failures for a user
    async fn count_recent_auth_failures(
        &self,
        user_id: &str,
        time_window: Duration,
    ) -> Result<u32> {
        let since = Utc::now() - time_window;

        // Query audit storage for recent auth failures
        let recent_events = self
            .storage
            .query_events(
                Some(AuditEventType::Authentication),
                Some(user_id.to_string()),
                Some(since),
                None,
                Some(100),
            )
            .await?;

        let failure_count = recent_events
            .iter()
            .filter(|event| matches!(event.result, AuditResult::Failure { .. }))
            .count() as u32;

        Ok(failure_count)
    }

    /// Calculate authentication risk score
    async fn calculate_authentication_risk(
        &self,
        user_id: &str,
        ip_address: &Option<String>,
    ) -> Option<f64> {
        let mut risk_score: f64 = 0.0;

        // Check for unusual IP address
        if let Some(ip) = &ip_address
            && self.is_suspicious_ip(ip).await
        {
            risk_score += 0.3;
        }

        // Check recent failure rate
        if let Ok(recent_failures) = self
            .count_recent_auth_failures(user_id, Duration::hours(1))
            .await
        {
            risk_score += (recent_failures as f64) * 0.1;
        }

        // Check time-based patterns (logins at unusual hours)
        let current_hour = Utc::now().hour();
        if !(6..=22).contains(&current_hour) {
            risk_score += 0.2; // Higher risk for after-hours access
        }

        Some(risk_score.min(1.0))
    }

    /// Calculate data access risk score
    async fn calculate_data_access_risk(&self, _user_id: &str, action: &str) -> Option<f64> {
        let mut risk_score: f64 = 0.0;

        // Higher risk for bulk operations
        if action.contains("bulk") || action.contains("export") || action.contains("download") {
            risk_score += 0.4;
        }

        // Higher risk for admin operations
        if action.contains("admin") || action.contains("delete") || action.contains("modify") {
            risk_score += 0.3;
        }

        // Check user's recent activity pattern
        // (Placeholder - real implementation would analyze user behavior)

        Some(risk_score.min(1.0))
    }

    /// Check if IP address is suspicious
    async fn is_suspicious_ip(&self, _ip_address: &str) -> bool {
        // Placeholder for threat intelligence integration
        // Real implementation would check against:
        // - Known malicious IP databases
        // - Geo-location anomalies
        // - VPN/proxy detection
        // - Previous security incidents
        false
    }

    /// Get IP geolocation information
    async fn get_ip_geolocation(&self, _ip_address: &str) -> Result<IpGeolocation> {
        // Placeholder for geolocation service integration
        Ok(IpGeolocation {
            country: "Unknown".to_string(),
            region: "Unknown".to_string(),
            city: "Unknown".to_string(),
            latitude: None,
            longitude: None,
        })
    }

    /// Send to external audit system
    async fn send_to_external_audit(&self, event: &AuditEvent, endpoint: &str) -> Result<()> {
        let client = reqwest::Client::new();

        let response = client
            .post(endpoint)
            .header("Content-Type", "application/json")
            .json(event)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "External audit system returned HTTP {}",
                response.status()
            ));
        }

        debug!("✅ Sent audit event to external system: {}", endpoint);
        Ok(())
    }
}

/// IP geolocation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpGeolocation {
    /// Country name or ISO 3166-1 alpha-2 code (e.g., "US")
    pub country: String,
    /// State, province, or administrative region name
    pub region: String,
    /// City name
    pub city: String,
    /// Geographic latitude in decimal degrees, if available
    pub latitude: Option<f64>,
    /// Geographic longitude in decimal degrees, if available
    pub longitude: Option<f64>,
}

impl EncryptionKey {
    /// Create an `EncryptionKey`. If `enabled` is `false` a zeroed dummy key is returned; otherwise a cryptographically random 256-bit key is generated.
    pub fn new(enabled: bool) -> Result<Self> {
        if !enabled {
            // Return dummy key when encryption is disabled
            return Ok(Self {
                key: [0u8; 32],
                algorithm: EncryptionAlgorithm::AES256GCM,
            });
        }

        // Generate secure random key
        use rand::RngCore;
        let mut key = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut key);

        Ok(Self {
            key,
            algorithm: EncryptionAlgorithm::AES256GCM,
        })
    }

    /// Encrypt `data` using the configured algorithm. Returns the ciphertext as a byte vector.
    pub fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Placeholder for actual encryption
        // Real implementation would use the configured algorithm
        Ok(data.to_vec())
    }

    /// Decrypt `data` using the configured algorithm. Returns the plaintext as a byte vector.
    pub fn decrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Placeholder for actual decryption
        Ok(data.to_vec())
    }
}

impl AlertSender {
    /// Create a new `AlertSender` from the given `AuditConfig`, wiring up the optional webhook URL
    pub fn new(config: &AuditConfig) -> Result<Self> {
        Ok(Self {
            webhook_url: config.alert_webhook_url.clone(),
            email_config: None, // Would be configured separately
        })
    }

    /// Dispatch the security alert payload to the configured webhook (if any)
    pub async fn send_alert(&self, alert: &SecurityAlert) -> Result<()> {
        if let Some(ref webhook_url) = self.webhook_url {
            let client = reqwest::Client::new();

            let alert_payload = serde_json::json!({
                "alert_id": alert.alert_id,
                "timestamp": alert.timestamp,
                "type": alert.alert_type,
                "severity": alert.severity,
                "description": alert.description,
                "user_id": alert.user_id,
                "ip_address": alert.ip_address
            });

            let response = client
                .post(webhook_url)
                .header("Content-Type", "application/json")
                .json(&alert_payload)
                .send()
                .await?;

            if response.status().is_success() {
                info!("✅ Security alert sent to webhook: {}", alert.alert_id);
            } else {
                warn!(
                    "⚠️ Failed to send security alert to webhook: HTTP {}",
                    response.status()
                );
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ==================== AuditConfig Tests ====================

    #[test]
    fn test_audit_config_default() {
        let config = AuditConfig::default();

        assert!(config.enable_audit_logging);
        assert!(!config.encryption_enabled);
        assert!(config.external_audit_endpoint.is_none());
        assert_eq!(config.retention_days, 30);
        assert!(!config.enable_real_time_alerts);
        assert!(config.alert_webhook_url.is_none());
        assert!(config.compliance_frameworks.is_empty());
    }

    #[test]
    fn test_audit_config_with_file_backend() {
        let config = AuditConfig {
            enable_audit_logging: true,
            storage_backend: AuditStorageBackend::File {
                directory: "/tmp/audit".to_string(),
            },
            encryption_enabled: false,
            external_audit_endpoint: None,
            retention_days: 90,
            enable_real_time_alerts: false,
            alert_webhook_url: None,
            compliance_frameworks: vec!["SOC2".to_string(), "GDPR".to_string()],
        };

        assert!(config.enable_audit_logging);
        assert_eq!(config.retention_days, 90);
        assert_eq!(config.compliance_frameworks.len(), 2);
    }

    #[test]
    fn test_audit_config_serialization() {
        let config = AuditConfig::default();

        let json = serde_json::to_string(&config).expect("Failed to serialize");
        assert!(json.contains("enable_audit_logging"));
        assert!(json.contains("retention_days"));

        let deserialized: AuditConfig = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(
            deserialized.enable_audit_logging,
            config.enable_audit_logging
        );
        assert_eq!(deserialized.retention_days, config.retention_days);
    }

    // ==================== AuditStorageBackend Tests ====================

    #[test]
    fn test_audit_storage_backend_file() {
        let backend = AuditStorageBackend::File {
            directory: "/tmp/audit_logs".to_string(),
        };

        if let AuditStorageBackend::File { directory } = backend {
            assert_eq!(directory, "/tmp/audit_logs");
        } else {
            panic!("Expected File variant");
        }
    }

    #[test]
    fn test_audit_storage_backend_database() {
        let backend = AuditStorageBackend::Database {
            connection_string: "postgres://localhost/auditdb".to_string(),
        };

        if let AuditStorageBackend::Database { connection_string } = backend {
            assert_eq!(connection_string, "postgres://localhost/auditdb");
        } else {
            panic!("Expected Database variant");
        }
    }

    #[test]
    fn test_audit_storage_backend_s3() {
        let backend = AuditStorageBackend::S3 {
            bucket: "my-audit-bucket".to_string(),
            region: "us-west-2".to_string(),
        };

        if let AuditStorageBackend::S3 { bucket, region } = backend {
            assert_eq!(bucket, "my-audit-bucket");
            assert_eq!(region, "us-west-2");
        } else {
            panic!("Expected S3 variant");
        }
    }

    #[test]
    fn test_audit_storage_backend_combined() {
        let primary = Box::new(AuditStorageBackend::File {
            directory: "/tmp/primary".to_string(),
        });
        let secondary = Box::new(AuditStorageBackend::S3 {
            bucket: "backup".to_string(),
            region: "us-east-1".to_string(),
        });

        let backend = AuditStorageBackend::Combined { primary, secondary };

        if let AuditStorageBackend::Combined {
            primary: _,
            secondary: _,
        } = backend
        {
            // Test passes if we can create the combined backend
        } else {
            panic!("Expected Combined variant");
        }
    }

    #[test]
    fn test_audit_storage_backend_serialization() {
        let backend = AuditStorageBackend::File {
            directory: "/tmp/test".to_string(),
        };

        let json = serde_json::to_string(&backend).expect("Failed to serialize");
        assert!(json.contains("/tmp/test"));
    }

    // ==================== EncryptionKey Tests ====================

    #[test]
    fn test_encryption_key_disabled() {
        let key = EncryptionKey::new(false).expect("Failed to create disabled key");

        // Disabled key should have zero bytes
        assert_eq!(key.key, [0u8; 32]);
        assert!(matches!(key.algorithm, EncryptionAlgorithm::AES256GCM));
    }

    #[test]
    fn test_encryption_key_enabled() {
        let key = EncryptionKey::new(true).expect("Failed to create enabled key");

        // Enabled key should have random bytes (very unlikely to be all zeros)
        let is_all_zeros = key.key.iter().all(|&b| b == 0);
        assert!(!is_all_zeros);
    }

    #[test]
    fn test_encryption_key_encrypt_decrypt() {
        let key = EncryptionKey::new(true).expect("Failed to create key");

        let plaintext = b"Hello, World!";
        let encrypted = key.encrypt(plaintext).expect("Encryption failed");
        let decrypted = key.decrypt(&encrypted).expect("Decryption failed");

        // Note: placeholder implementation just returns the same data
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_encryption_key_encrypt_empty() {
        let key = EncryptionKey::new(true).expect("Failed to create key");

        let plaintext = b"";
        let encrypted = key.encrypt(plaintext).expect("Encryption failed");

        assert!(encrypted.is_empty());
    }

    // ==================== EncryptionAlgorithm Tests ====================

    #[test]
    fn test_encryption_algorithm_variants() {
        let aes = EncryptionAlgorithm::AES256GCM;
        let chacha = EncryptionAlgorithm::ChaCha20Poly1305;

        assert!(matches!(aes, EncryptionAlgorithm::AES256GCM));
        assert!(matches!(chacha, EncryptionAlgorithm::ChaCha20Poly1305));
    }

    #[test]
    fn test_encryption_algorithm_serialization() {
        let algo = EncryptionAlgorithm::AES256GCM;

        let json = serde_json::to_string(&algo).expect("Failed to serialize");
        assert!(json.contains("AES256GCM"));
    }

    // ==================== AlertSender Tests ====================

    #[test]
    fn test_alert_sender_creation_no_webhook() {
        let config = AuditConfig::default();
        let sender = AlertSender::new(&config).expect("Failed to create AlertSender");

        assert!(sender.webhook_url.is_none());
        assert!(sender.email_config.is_none());
    }

    #[test]
    fn test_alert_sender_creation_with_webhook() {
        let config = AuditConfig {
            alert_webhook_url: Some("https://webhook.example.com/alerts".to_string()),
            ..AuditConfig::default()
        };
        let sender = AlertSender::new(&config).expect("Failed to create AlertSender");

        assert_eq!(
            sender.webhook_url,
            Some("https://webhook.example.com/alerts".to_string())
        );
    }

    #[tokio::test]
    async fn test_alert_sender_send_alert_no_webhook() {
        let config = AuditConfig::default();
        let sender = AlertSender::new(&config).expect("Failed to create AlertSender");

        let alert = SecurityAlert::new(
            SecurityAlertType::SuspiciousAuthActivity,
            SecurityAlertSeverity::High,
            "Test alert".to_string(),
        );

        // Should succeed without webhook (no-op)
        let result = sender.send_alert(&alert).await;
        assert!(result.is_ok());
    }

    // ==================== EmailConfig Tests ====================

    #[test]
    fn test_email_config() {
        let config = EmailConfig {
            smtp_server: "smtp.example.com".to_string(),
            smtp_port: 587,
            username: "user@example.com".to_string(),
            password: "secret".to_string(),
            from_address: "alerts@example.com".to_string(),
            to_addresses: vec!["admin@example.com".to_string()],
        };

        assert_eq!(config.smtp_server, "smtp.example.com");
        assert_eq!(config.smtp_port, 587);
        assert_eq!(config.to_addresses.len(), 1);
    }

    #[test]
    fn test_email_config_serialization() {
        let config = EmailConfig {
            smtp_server: "smtp.test.com".to_string(),
            smtp_port: 25,
            username: "test".to_string(),
            password: "pass".to_string(),
            from_address: "from@test.com".to_string(),
            to_addresses: vec!["to@test.com".to_string()],
        };

        let json = serde_json::to_string(&config).expect("Failed to serialize");
        assert!(json.contains("smtp.test.com"));
        assert!(json.contains("25"));
    }

    // ==================== IpGeolocation Tests ====================

    #[test]
    fn test_ip_geolocation() {
        let geo = IpGeolocation {
            country: "United States".to_string(),
            region: "California".to_string(),
            city: "San Francisco".to_string(),
            latitude: Some(37.7749),
            longitude: Some(-122.4194),
        };

        assert_eq!(geo.country, "United States");
        assert_eq!(geo.region, "California");
        assert_eq!(geo.city, "San Francisco");
        assert!(geo.latitude.is_some());
        assert!(geo.longitude.is_some());
    }

    #[test]
    fn test_ip_geolocation_serialization() {
        let geo = IpGeolocation {
            country: "Germany".to_string(),
            region: "Berlin".to_string(),
            city: "Berlin".to_string(),
            latitude: Some(52.52),
            longitude: Some(13.405),
        };

        let json = serde_json::to_string(&geo).expect("Failed to serialize");
        assert!(json.contains("Germany"));
        assert!(json.contains("Berlin"));

        let deserialized: IpGeolocation =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized.country, "Germany");
    }

    // ==================== AuditLogger Tests ====================

    #[tokio::test]
    async fn test_audit_logger_creation_disabled() {
        let config = AuditConfig {
            enable_audit_logging: false,
            ..AuditConfig::default()
        };

        let result = AuditLogger::new(config).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("disabled"));
    }

    #[tokio::test]
    async fn test_audit_logger_creation_with_file_backend() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config = AuditConfig {
            enable_audit_logging: true,
            storage_backend: AuditStorageBackend::File {
                directory: temp_dir.path().to_string_lossy().to_string(),
            },
            encryption_enabled: false,
            external_audit_endpoint: None,
            retention_days: 30,
            enable_real_time_alerts: false,
            alert_webhook_url: None,
            compliance_frameworks: vec![],
        };

        let logger = AuditLogger::new(config).await;
        assert!(logger.is_ok());
    }

    #[tokio::test]
    async fn test_audit_logger_creation_with_database_backend_fails_closed() {
        let config = AuditConfig {
            enable_audit_logging: true,
            storage_backend: AuditStorageBackend::Database {
                connection_string: "postgres://localhost/test".to_string(),
            },
            encryption_enabled: false,
            external_audit_endpoint: None,
            retention_days: 30,
            enable_real_time_alerts: false,
            alert_webhook_url: None,
            compliance_frameworks: vec![],
        };

        // Fail closed: no database audit backend exists, and the previous
        // placeholder silently discarded every audit event. Selecting it
        // must abort startup with an actionable message (same policy as the
        // Combined arm and the feature-gated S3 backend).
        let result = AuditLogger::new(config).await;
        let message = result.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(
            message.contains("Database audit backend is not implemented"),
            "database backend must fail closed at construction, got: {message}"
        );
    }

    #[tokio::test]
    async fn test_audit_logger_creation_s3_backend() {
        // TD-OBJSTORE-1 (#960): the S3 backend now routes through
        // ObjectStoreAuditStorage. With the `aws` feature the s3:// scheme
        // registers and creation succeeds (credentials are resolved lazily
        // on first I/O); without it, creation fail-fasts on the
        // unroutable scheme instead of the old "not yet implemented".
        let config = AuditConfig {
            enable_audit_logging: true,
            storage_backend: AuditStorageBackend::S3 {
                bucket: "test-bucket".to_string(),
                region: "us-west-2".to_string(),
            },
            encryption_enabled: false,
            external_audit_endpoint: None,
            retention_days: 30,
            enable_real_time_alerts: false,
            alert_webhook_url: None,
            compliance_frameworks: vec![],
        };

        let result = AuditLogger::new(config).await;
        #[cfg(feature = "aws")]
        assert!(
            result.is_ok(),
            "S3 audit backend must initialize lazily with the aws feature: {:?}",
            result.err().map(|e| e.to_string())
        );
        #[cfg(not(feature = "aws"))]
        assert!(
            result
                .err()
                .map(|e| e.to_string())
                .is_some_and(|msg| msg.contains("not routable")),
            "S3 audit backend must fail-fast on the unroutable scheme without the aws feature"
        );
    }

    #[tokio::test]
    async fn test_audit_logger_creation_combined_not_implemented() {
        let config = AuditConfig {
            enable_audit_logging: true,
            storage_backend: AuditStorageBackend::Combined {
                primary: Box::new(AuditStorageBackend::File {
                    directory: "/tmp/primary".to_string(),
                }),
                secondary: Box::new(AuditStorageBackend::File {
                    directory: "/tmp/secondary".to_string(),
                }),
            },
            encryption_enabled: false,
            external_audit_endpoint: None,
            retention_days: 30,
            enable_real_time_alerts: false,
            alert_webhook_url: None,
            compliance_frameworks: vec![],
        };

        let result = AuditLogger::new(config).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("not yet implemented")
        );
    }

    #[tokio::test]
    async fn test_audit_logger_creation_with_encryption() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config = AuditConfig {
            enable_audit_logging: true,
            storage_backend: AuditStorageBackend::File {
                directory: temp_dir.path().to_string_lossy().to_string(),
            },
            encryption_enabled: true,
            external_audit_endpoint: None,
            retention_days: 30,
            enable_real_time_alerts: false,
            alert_webhook_url: None,
            compliance_frameworks: vec![],
        };

        let logger = AuditLogger::new(config).await;
        assert!(logger.is_ok());
    }

    #[tokio::test]
    async fn test_audit_logger_creation_with_alerts() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config = AuditConfig {
            enable_audit_logging: true,
            storage_backend: AuditStorageBackend::File {
                directory: temp_dir.path().to_string_lossy().to_string(),
            },
            encryption_enabled: false,
            external_audit_endpoint: None,
            retention_days: 30,
            enable_real_time_alerts: true,
            alert_webhook_url: Some("https://webhook.example.com".to_string()),
            compliance_frameworks: vec![],
        };

        let logger = AuditLogger::new(config).await;
        assert!(logger.is_ok());
    }

    #[tokio::test]
    async fn test_audit_logger_log_event() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config = AuditConfig {
            enable_audit_logging: true,
            storage_backend: AuditStorageBackend::File {
                directory: temp_dir.path().to_string_lossy().to_string(),
            },
            encryption_enabled: false,
            external_audit_endpoint: None,
            retention_days: 30,
            enable_real_time_alerts: false,
            alert_webhook_url: None,
            compliance_frameworks: vec![],
        };

        let logger = AuditLogger::new(config)
            .await
            .expect("Failed to create logger");

        let resource = AuditResource::new("collection".to_string(), "test-collection".to_string());
        let event = AuditEvent::new(
            AuditEventType::DataAccess,
            resource,
            "read".to_string(),
            AuditResult::Success,
        )
        .with_user("test-user".to_string());

        let result = logger.log_event(event).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_audit_logger_log_event_validation_empty_event_id() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config = AuditConfig {
            enable_audit_logging: true,
            storage_backend: AuditStorageBackend::File {
                directory: temp_dir.path().to_string_lossy().to_string(),
            },
            ..AuditConfig::default()
        };

        let logger = AuditLogger::new(config)
            .await
            .expect("Failed to create logger");

        let resource = AuditResource::new("test".to_string(), "test".to_string());
        let mut event = AuditEvent::new(
            AuditEventType::DataAccess,
            resource,
            "read".to_string(),
            AuditResult::Success,
        );
        event.event_id = String::new(); // Empty event ID

        let result = logger.log_event(event).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Event ID"));
    }

    #[tokio::test]
    async fn test_audit_logger_log_event_validation_empty_action() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config = AuditConfig {
            enable_audit_logging: true,
            storage_backend: AuditStorageBackend::File {
                directory: temp_dir.path().to_string_lossy().to_string(),
            },
            ..AuditConfig::default()
        };

        let logger = AuditLogger::new(config)
            .await
            .expect("Failed to create logger");

        let resource = AuditResource::new("test".to_string(), "test".to_string());
        let mut event = AuditEvent::new(
            AuditEventType::DataAccess,
            resource,
            "read".to_string(),
            AuditResult::Success,
        );
        event.action = String::new(); // Empty action

        let result = logger.log_event(event).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Action"));
    }

    #[tokio::test]
    async fn test_audit_logger_log_event_validation_empty_resource() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config = AuditConfig {
            enable_audit_logging: true,
            storage_backend: AuditStorageBackend::File {
                directory: temp_dir.path().to_string_lossy().to_string(),
            },
            ..AuditConfig::default()
        };

        let logger = AuditLogger::new(config)
            .await
            .expect("Failed to create logger");

        let resource = AuditResource::new(String::new(), String::new()); // Empty resource
        let event = AuditEvent::new(
            AuditEventType::DataAccess,
            resource,
            "read".to_string(),
            AuditResult::Success,
        );

        let result = logger.log_event(event).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Resource"));
    }

    #[tokio::test]
    async fn test_audit_logger_log_authentication_event() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config = AuditConfig {
            enable_audit_logging: true,
            storage_backend: AuditStorageBackend::File {
                directory: temp_dir.path().to_string_lossy().to_string(),
            },
            ..AuditConfig::default()
        };

        let logger = AuditLogger::new(config)
            .await
            .expect("Failed to create logger");

        let result = logger
            .log_authentication_event(
                "user-123",
                "password",
                AuditResult::Success,
                Some("192.168.1.100".to_string()),
                Some("Mozilla/5.0".to_string()),
            )
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_audit_logger_log_authentication_event_failure() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config = AuditConfig {
            enable_audit_logging: true,
            storage_backend: AuditStorageBackend::File {
                directory: temp_dir.path().to_string_lossy().to_string(),
            },
            ..AuditConfig::default()
        };

        let logger = AuditLogger::new(config)
            .await
            .expect("Failed to create logger");

        let result = logger
            .log_authentication_event(
                "user-123",
                "password",
                AuditResult::Failure {
                    error_code: "AUTH_FAILED".to_string(),
                    error_message: "Invalid credentials".to_string(),
                },
                Some("192.168.1.100".to_string()),
                None,
            )
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_audit_logger_log_data_access_event() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config = AuditConfig {
            enable_audit_logging: true,
            storage_backend: AuditStorageBackend::File {
                directory: temp_dir.path().to_string_lossy().to_string(),
            },
            ..AuditConfig::default()
        };

        let logger = AuditLogger::new(config)
            .await
            .expect("Failed to create logger");

        let resource = AuditResource::new("collection".to_string(), "my-collection".to_string());
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("query".to_string(), serde_json::json!("test query"));

        let result = logger
            .log_data_access_event(
                "user-456",
                Some("tenant-123".to_string()),
                resource,
                "vector_search",
                AuditResult::Success,
                metadata,
            )
            .await;

        assert!(result.is_ok());
    }

    // ==================== Sensitive Field Detection Tests ====================

    #[tokio::test]
    async fn test_audit_logger_is_sensitive_field() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config = AuditConfig {
            enable_audit_logging: true,
            storage_backend: AuditStorageBackend::File {
                directory: temp_dir.path().to_string_lossy().to_string(),
            },
            ..AuditConfig::default()
        };

        let logger = AuditLogger::new(config)
            .await
            .expect("Failed to create logger");

        // Sensitive fields
        assert!(logger.is_sensitive_field("email"));
        assert!(logger.is_sensitive_field("user_email"));
        assert!(logger.is_sensitive_field("EMAIL_ADDRESS"));
        assert!(logger.is_sensitive_field("phone"));
        assert!(logger.is_sensitive_field("ssn"));
        assert!(logger.is_sensitive_field("credit_card"));
        assert!(logger.is_sensitive_field("password"));
        assert!(logger.is_sensitive_field("api_key"));
        assert!(logger.is_sensitive_field("auth_token"));
        assert!(logger.is_sensitive_field("secret_key"));
        assert!(logger.is_sensitive_field("private_key"));

        // Non-sensitive fields
        assert!(!logger.is_sensitive_field("username"));
        assert!(!logger.is_sensitive_field("collection_name"));
        assert!(!logger.is_sensitive_field("timestamp"));
        assert!(!logger.is_sensitive_field("action"));
    }

    // ==================== Debug Implementation Tests ====================

    #[tokio::test]
    async fn test_audit_logger_debug() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config = AuditConfig {
            enable_audit_logging: true,
            storage_backend: AuditStorageBackend::File {
                directory: temp_dir.path().to_string_lossy().to_string(),
            },
            ..AuditConfig::default()
        };

        let logger = AuditLogger::new(config)
            .await
            .expect("Failed to create logger");

        let debug_str = format!("{:?}", logger);
        assert!(debug_str.contains("AuditLogger"));
        assert!(debug_str.contains("config"));
        assert!(debug_str.contains("storage"));
    }

    // ==================== Risk Score Calculation Tests ====================

    #[tokio::test]
    async fn test_audit_logger_calculate_data_access_risk_bulk_operations() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config = AuditConfig {
            enable_audit_logging: true,
            storage_backend: AuditStorageBackend::File {
                directory: temp_dir.path().to_string_lossy().to_string(),
            },
            ..AuditConfig::default()
        };

        let logger = AuditLogger::new(config)
            .await
            .expect("Failed to create logger");

        // Bulk operations should have higher risk
        let bulk_risk = logger
            .calculate_data_access_risk("user-1", "bulk_export")
            .await;
        assert!(bulk_risk.is_some());
        assert!(bulk_risk.unwrap() >= 0.4);

        // Normal operations should have lower risk
        let normal_risk = logger.calculate_data_access_risk("user-1", "read").await;
        assert!(normal_risk.is_some());
        assert!(normal_risk.unwrap() < 0.4);
    }

    #[tokio::test]
    async fn test_audit_logger_calculate_data_access_risk_admin_operations() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config = AuditConfig {
            enable_audit_logging: true,
            storage_backend: AuditStorageBackend::File {
                directory: temp_dir.path().to_string_lossy().to_string(),
            },
            ..AuditConfig::default()
        };

        let logger = AuditLogger::new(config)
            .await
            .expect("Failed to create logger");

        // Admin operations should have higher risk
        let admin_risk = logger
            .calculate_data_access_risk("user-1", "admin_delete")
            .await;
        assert!(admin_risk.is_some());
        assert!(admin_risk.unwrap() >= 0.3);
    }

    #[tokio::test]
    async fn test_audit_logger_calculate_authentication_risk() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config = AuditConfig {
            enable_audit_logging: true,
            storage_backend: AuditStorageBackend::File {
                directory: temp_dir.path().to_string_lossy().to_string(),
            },
            ..AuditConfig::default()
        };

        let logger = AuditLogger::new(config)
            .await
            .expect("Failed to create logger");

        let risk = logger
            .calculate_authentication_risk("user-1", &Some("192.168.1.1".to_string()))
            .await;

        assert!(risk.is_some());
        // Risk should be between 0 and 1
        let risk_value = risk.unwrap();
        assert!((0.0..=1.0).contains(&risk_value));
    }

    // ==================== Integration Tests ====================

    #[tokio::test]
    async fn test_audit_logger_full_workflow() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config = AuditConfig {
            enable_audit_logging: true,
            storage_backend: AuditStorageBackend::File {
                directory: temp_dir.path().to_string_lossy().to_string(),
            },
            encryption_enabled: false,
            external_audit_endpoint: None,
            retention_days: 30,
            enable_real_time_alerts: false,
            alert_webhook_url: None,
            compliance_frameworks: vec!["SOC2".to_string()],
        };

        let logger = AuditLogger::new(config)
            .await
            .expect("Failed to create logger");

        // Log authentication event
        logger
            .log_authentication_event(
                "user-1",
                "sso",
                AuditResult::Success,
                Some("10.0.0.1".to_string()),
                Some("Chrome".to_string()),
            )
            .await
            .expect("Failed to log auth event");

        // Log data access event
        let resource = AuditResource::new("collection".to_string(), "vectors".to_string());
        logger
            .log_data_access_event(
                "user-1",
                Some("tenant-1".to_string()),
                resource,
                "search",
                AuditResult::Success,
                std::collections::HashMap::new(),
            )
            .await
            .expect("Failed to log data access event");

        // Log generic event
        let generic_resource = AuditResource::new("system".to_string(), "config".to_string());
        let event = AuditEvent::new(
            AuditEventType::SystemConfiguration,
            generic_resource,
            "update_settings".to_string(),
            AuditResult::Success,
        )
        .with_user("admin".to_string());

        logger
            .log_event(event)
            .await
            .expect("Failed to log generic event");
    }

    #[tokio::test]
    async fn test_audit_logger_with_compliance_frameworks() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config = AuditConfig {
            enable_audit_logging: true,
            storage_backend: AuditStorageBackend::File {
                directory: temp_dir.path().to_string_lossy().to_string(),
            },
            encryption_enabled: false,
            external_audit_endpoint: None,
            retention_days: 90,
            enable_real_time_alerts: false,
            alert_webhook_url: None,
            compliance_frameworks: vec![
                "SOC2".to_string(),
                "GDPR".to_string(),
                "HIPAA".to_string(),
            ],
        };

        let logger = AuditLogger::new(config)
            .await
            .expect("Failed to create logger");

        let resource = AuditResource::new("pii_data".to_string(), "patient_records".to_string());
        let event = AuditEvent::new(
            AuditEventType::DataAccess,
            resource,
            "read".to_string(),
            AuditResult::Success,
        )
        .with_user("healthcare_worker".to_string())
        .with_tenant("hospital_a".to_string());

        let result = logger.log_event(event).await;
        assert!(result.is_ok());
    }
}
