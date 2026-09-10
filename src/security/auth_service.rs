//! Unified Authentication Service for ProximaDB
//!
//! Consolidates authentication logic from multiple sources:
//! - Network Auth Service (src/network/auth/mod.rs)
//! - Auth Middleware (src/network/middleware/auth.rs)

use super::rbac_service::{UnifiedAuthMethod, UnifiedPermission, UnifiedUserContext};
use crate::audit::logger::AuditLogger;
use crate::network::auth::{JwtService, TokenPair};
use crate::network::postgres::scram;
use proximadb_catalog::principal_registry::{
    FileSystemPrincipalRegistry, KEY_PREFIX as REGISTRY_KEY_PREFIX,
};

use anyhow::{Result, anyhow};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use proximadb_security::{AuditEvent, AuditEventType, AuditResource, AuditResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::{info, warn};
use x509_parser::prelude::*;

/// Unified authentication service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationConfig {
    pub enabled: bool,
    pub methods: Vec<AuthenticationMethod>,
    pub require_authentication: bool,
    pub default_session_timeout_minutes: u64,
    pub api_keys: HashMap<String, ApiKeyInfo>,
    pub jwt: JwtConfig,
    pub sso: SSOConfig,
    /// mTLS configuration for client certificate authentication
    #[serde(default)]
    pub mtls: MtlsConfig,
    /// Deny authentication when the audit write fails, instead of proceeding
    /// unaudited. Default OFF — see the policy note at the emit site: the
    /// credible audit-write failure is operational (full disk), and denying on
    /// it converts that into a total authentication outage. Regulated
    /// deployments that must never serve unaudited traffic opt in here.
    #[serde(default)]
    pub audit_fail_closed: bool,
    /// Generic OIDC provider (TD-SSO-1): bearer-token validation against an
    /// external IdP's JWKS, with role-claim sanitization at the seam. Absent
    /// ⇒ inert; RS*/ES* bearer tokens then fail closed (no local asymmetric
    /// verifier exists).
    #[serde(default)]
    pub oidc: Option<crate::network::auth::oidc::OidcProviderConfig>,
    /// pgwire SCRAM-SHA-256 identities (`[security.authentication.scram_users]`,
    /// TD-PGWIRE-AUTH-1). Deliberately a SEPARATE namespace from `api_keys`: the
    /// identity payload maps to a key-equivalent `UnifiedUserContext`, but the
    /// secret material lives apart so a leaked StoredKey/ServerKey can never be
    /// replayed as an API key and vice versa.
    #[serde(default)]
    pub scram_users: HashMap<String, ScramUserConfig>,
}

/// One pgwire SCRAM identity (TD-PGWIRE-AUTH-1). Exactly one of `password` /
/// `verifier` is required; a configured plaintext is derived into a verifier at
/// coordinator build time and then dropped.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScramUserConfig {
    /// Plaintext password — converted to a [`ScramVerifierConfig`] at build,
    /// never retained in memory or re-serialized.
    #[serde(default)]
    pub password: Option<String>,
    /// Precomputed verifier (salt/StoredKey/ServerKey, base64) for deployments
    /// that refuse plaintext passwords in config files.
    #[serde(default)]
    pub verifier: Option<ScramVerifierConfig>,
    /// Tenant binding: on successful authentication the session identity binds
    /// to this tenant (the credential's `AuthenticatedTenantBinding`).
    #[serde(default)]
    pub tenant_id: Option<String>,
    /// Delegation-ONLY role claims, same policy as `ApiKeyInfo::roles`: only
    /// the exact strings `gateway`/`operator` are recognized, and a role grants
    /// NO `UnifiedPermission`.
    #[serde(default)]
    pub roles: Vec<String>,
}

/// Precomputed SCRAM verifier material (base64-encoded parts, TD-PGWIRE-AUTH-1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScramVerifierConfig {
    pub salt: String,
    pub stored_key: String,
    pub server_key: String,
    #[serde(default = "default_scram_iterations")]
    pub iterations: u32,
}

fn default_scram_iterations() -> u32 {
    crate::network::postgres::scram::DEFAULT_ITERATIONS
}

/// Count of audit writes that failed on the authentication path. Exposed so an
/// operator can alarm on "we are serving unaudited traffic" — the failure is
/// fail-open by default, so this counter is the ONLY signal that accountability
/// has been lost.
pub static AUDIT_WRITE_FAILURES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Authentication methods supported
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AuthenticationMethod {
    #[serde(rename = "sso")]
    SSO,
    #[serde(rename = "jwt")]
    JWT,
    #[serde(rename = "api_key")]
    ApiKey,
    #[serde(rename = "mtls")]
    ClientCertificate,
}

/// API key information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyInfo {
    pub user_id: String,
    pub tenant_id: Option<String>,
    pub permissions: Vec<String>,
    /// TD-TENANT-1 follow-up (a): optional role claims for this key. The
    /// `gateway`/`operator` roles make the credential a gateway principal
    /// (`is_gateway_principal`), enabling `GatewayOnly` tenant delegation for
    /// config-backed API-key gateways — previously structurally impossible
    /// (roles were hardcoded to `api_user`). ADR-090 `pxk_` registry keys have
    /// a separate, registry-authoritative identity path and do not inherit
    /// these config claims. Delegation-ONLY: a role here grants NO
    /// `UnifiedPermission` (authz stays permission-driven), and only the
    /// exact strings `gateway`/`operator` match — a typo grants nothing.
    #[serde(default)]
    pub roles: Vec<String>,
    pub created_at: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
    pub rate_limit_per_minute: Option<u32>,
    pub ip_restrictions: Vec<String>,
}

/// JWT configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtConfig {
    pub enabled: bool,
    pub secret: String,
    pub access_token_expiration_minutes: u64,
    pub refresh_token_expiration_days: u64,
    pub issuer: String,
    pub audience: String,
    pub algorithm: String,
}

/// SSO configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSOConfig {
    pub enabled: bool,
    pub providers: Vec<String>,
    pub token_cache_ttl_minutes: u64,
    // NOTE (provider portability): the former `aws_iam`/`azure_ad` fields and
    // their types were REMOVED — they had zero production readers (every
    // construction set them `None` inside `#[cfg(test)]`) and the backing
    // `azure_ad.rs` stub returned `system_admin()` for any unexpired token
    // with no signature verification (a latent privilege escalation). Generic
    // OIDC covers AWS/Azure via `[security.authentication.oidc]`.
}

/// mTLS configuration for client certificate authentication
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MtlsConfig {
    /// Whether mTLS authentication is enabled
    pub enabled: bool,
    /// Path to the CA certificate used to verify client certificates
    pub ca_cert_path: Option<String>,
    /// Whether client certificates are required (vs optional)
    pub require_client_cert: bool,
    /// Maps Common Name patterns to RBAC role names.
    /// Patterns support prefix matching with a trailing '*' wildcard.
    /// Example: {"*.admin" => "admin", "service-*" => "service_role"}
    #[serde(default)]
    pub cn_role_mapping: HashMap<String, String>,
}

/// Validated client identity extracted from an X.509 certificate
#[derive(Debug, Clone)]
pub struct ClientIdentity {
    /// The Common Name (CN) from the certificate subject
    pub common_name: String,
    /// Subject Alternative Name entries (DNS names, emails, URIs, IP addresses)
    pub san_entries: Vec<String>,
    /// RBAC roles resolved from the CN via cn_role_mapping
    pub roles: Vec<String>,
}

/// Authentication result from unified service
#[derive(Debug, Clone)]
pub struct AuthenticationResult {
    pub user_context: UnifiedUserContext,
    pub auth_method: UnifiedAuthMethod,
    pub success: bool,
    pub error_message: Option<String>,
    pub requires_mfa: bool,
}

/// Canonical generic authentication result for unified security auth flows.
pub type SecurityAuthenticationResult = AuthenticationResult;

/// Unified authentication service
pub struct UnifiedAuthService {
    /// JWT service for token authentication
    jwt_service: Option<Arc<JwtService>>,
    /// Generic OIDC bearer verifier (TD-SSO-1). Present only when
    /// `[security.authentication.oidc]` is configured AND enabled.
    oidc_verifier: Option<Arc<crate::network::auth::oidc::OidcTokenVerifier>>,

    /// API key store
    api_keys: Arc<DashMap<String, ApiKeyInfo>>,

    /// pgwire SCRAM verifiers (TD-PGWIRE-AUTH-1): derived once at build from
    /// `config.scram_users` (plaintext passwords are converted here and the
    /// verifier is what authentication consumes).
    scram_verifiers: Arc<DashMap<String, scram::ScramVerifier>>,
    /// The identity side of [`Self::scram_verifiers`] (tenant binding + roles).
    scram_identities: Arc<DashMap<String, ScramUserConfig>>,

    /// ADR-090 L0: catalog-resident principal/key registry. When present it is
    /// AUTHORITATIVE for the `pxk_` key namespace; the config-table keys above
    /// remain only as the legacy bootstrap path.
    principal_registry: Option<Arc<FileSystemPrincipalRegistry>>,
    /// Failed-authentication throttle (TD-SEC-2). `None` = no throttling, the
    /// pre-existing behavior.
    rate_limiter: Option<Arc<crate::security::advanced_features::RateLimitingService>>,

    /// Configuration
    config: AuthenticationConfig,

    /// Audit logger for authentication events
    audit_logger: Option<Arc<AuditLogger>>,

    /// Cached CA certificate DER bytes for mTLS validation
    ca_cert_der: Option<Vec<u8>>,
}

impl std::fmt::Debug for UnifiedAuthService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnifiedAuthService")
            .field("has_jwt_service", &self.jwt_service.is_some())
            .field("api_key_count", &self.api_keys.len())
            .field("has_principal_registry", &self.principal_registry.is_some())
            .field("has_ca_cert", &self.ca_cert_der.is_some())
            .finish()
    }
}

impl UnifiedAuthService {
    /// Create new unified authentication service
    pub fn new(config: AuthenticationConfig) -> Result<Self> {
        // Load CA certificate if mTLS is configured
        let ca_cert_der = if config.mtls.enabled {
            if let Some(ref ca_path) = config.mtls.ca_cert_path {
                Some(Self::load_ca_certificate(ca_path)?)
            } else {
                return Err(anyhow!("mTLS is enabled but no ca_cert_path is configured"));
            }
        } else {
            None
        };

        let mut service = Self {
            jwt_service: None,
            oidc_verifier: None,
            api_keys: Arc::new(DashMap::new()),
            scram_verifiers: Arc::new(DashMap::new()),
            scram_identities: Arc::new(DashMap::new()),
            principal_registry: None,
            rate_limiter: None,
            config: config.clone(),
            audit_logger: None,
            ca_cert_der,
        };

        // TD-PGWIRE-AUTH-1: derive SCRAM verifiers once at build. A configured
        // plaintext is converted to a verifier here; the runtime auth path
        // never touches it. Malformed verifier material fails the boot.
        for (username, user) in &config.scram_users {
            let verifier = match (&user.password, &user.verifier) {
                (Some(password), None) => scram::ScramVerifier::generate(
                    password,
                    user.verifier
                        .as_ref()
                        .map(|v| v.iterations)
                        .unwrap_or(crate::network::postgres::scram::DEFAULT_ITERATIONS),
                )
                .map_err(|error| {
                    anyhow!("scram_users[{username}]: verifier derivation failed: {error:?}")
                })?,
                (None, Some(material)) => {
                    use base64::Engine as _;
                    let decode32 = |s: &str| -> Result<[u8; 32]> {
                        let raw = base64::engine::general_purpose::STANDARD
                            .decode(s)
                            .map_err(|e| anyhow!("scram_users[{username}]: bad base64: {e}"))?;
                        raw.try_into()
                            .map_err(|_| anyhow!("scram_users[{username}]: key must be 32 bytes"))
                    };
                    let salt = base64::engine::general_purpose::STANDARD
                        .decode(&material.salt)
                        .map_err(|e| anyhow!("scram_users[{username}]: bad salt base64: {e}"))?;
                    scram::ScramVerifier {
                        salt,
                        stored_key: decode32(&material.stored_key)?,
                        server_key: decode32(&material.server_key)?,
                        iterations: material.iterations,
                    }
                }
                (Some(_), Some(_)) => {
                    return Err(anyhow!(
                        "scram_users[{username}]: configure exactly one of `password` or `verifier`"
                    ));
                }
                (None, None) => {
                    return Err(anyhow!(
                        "scram_users[{username}]: one of `password` or `verifier` is required"
                    ));
                }
            };
            service.scram_verifiers.insert(username.clone(), verifier);
            service
                .scram_identities
                .insert(username.clone(), user.clone());
        }

        // Initialize JWT service if enabled
        if config.jwt.enabled {
            // Convert unified JwtConfig to network JwtConfig
            let network_jwt_config = crate::network::auth::config::JwtConfig {
                secret: Some(config.jwt.secret.clone()),
                expiration_secs: config.jwt.access_token_expiration_minutes * 60,
                refresh_expiration_secs: config.jwt.refresh_token_expiration_days * 24 * 3600,
                issuer: config.jwt.issuer.clone(),
                audience: config.jwt.audience.clone(),
                algorithm: match config.jwt.algorithm.as_str() {
                    "HS256" => crate::network::auth::config::JwtAlgorithm::HS256,
                    "HS384" => crate::network::auth::config::JwtAlgorithm::HS384,
                    "HS512" => crate::network::auth::config::JwtAlgorithm::HS512,
                    other => {
                        return Err(anyhow!(
                            "invalid [security.authentication.jwt] algorithm {other:?}: \
                             HS256/HS384/HS512 supported (use [security.authentication.oidc] \
                             for RS256/ES256)"
                        ));
                    }
                },
            };
            let jwt_service = JwtService::new(network_jwt_config)?;
            service.jwt_service = Some(Arc::new(jwt_service));
        }

        // TD-SSO-1: build the OIDC verifier when configured. Construction
        // never contacts the IdP (lazy JWKS) — boot-order robustness.
        if let Some(oidc_cfg) = config.oidc.as_ref()
            && oidc_cfg.enabled
        {
            let verifier = crate::network::auth::oidc::OidcTokenVerifier::new(oidc_cfg.clone())
                .map_err(|e| anyhow!("invalid [security.authentication.oidc]: {e}"))?;
            // F8: the OIDC branch is reachable only via the `jwt` method —
            // warn loudly when the operator enabled the provider but not the
            // method, or every bearer token will read "JWT authentication
            // disabled".
            if !config.methods.contains(&AuthenticationMethod::JWT) {
                warn!(
                    "[security.authentication.oidc] is enabled but methods lacks \"jwt\" — OIDC bearer tokens will be rejected until it is added"
                );
            }
            info!(
                issuer = %oidc_cfg.issuer_url,
                "OIDC bearer validation enabled (TD-SSO-1; roles sanitize at the seam; \
                 delegation roles default OFF pending allow_delegation_roles)"
            );
            service.oidc_verifier = Some(Arc::new(verifier));
        }

        // Load API keys
        for (key, info) in config.api_keys {
            service.api_keys.insert(key, info);
        }

        Ok(service)
    }

    /// Set audit logger
    pub fn set_audit_logger(&mut self, audit_logger: Arc<AuditLogger>) {
        self.audit_logger = Some(audit_logger);
    }

    /// Authenticate request using multiple methods
    pub async fn authenticate(
        &self,
        auth_data: AuthenticationData,
    ) -> Result<AuthenticationResult> {
        let start_time = Utc::now();

        let result = match &auth_data {
            AuthenticationData::SSOToken(_) => Ok(AuthenticationResult {
                user_context: UnifiedUserContext::anonymous(),
                auth_method: UnifiedAuthMethod::SSO {
                    provider: "removed".to_string(),
                },
                success: false,
                error_message: Some(
                    "Legacy SSO removed: use [security.authentication.oidc] for any OIDC IdP"
                        .to_string(),
                ),
                requires_mfa: false,
            }),
            AuthenticationData::JWTToken(token) => self.authenticate_jwt_token(token).await,
            AuthenticationData::ApiKey(key) => self.authenticate_api_key(key).await,
            AuthenticationData::ClientCertificate(cert_data) => {
                self.authenticate_client_certificate(cert_data).await
            }
            AuthenticationData::ScramAuthenticated { username } => {
                self.authenticate_scram_user(username).await
            }
        };

        // Log authentication attempt
        if let Some(audit_logger) = &self.audit_logger {
            let auth_event = match &result {
                Ok(auth_result) => create_auth_audit_event(&auth_data, auth_result, start_time),
                Err(_) => {
                    // Attribute to the credential PRESENTED, never to
                    // `anonymous()` — see `attempted_principal`.
                    let mut attempted = UnifiedUserContext::anonymous();
                    attempted.user_id = attempted_principal(&auth_data);
                    // Count this failure against the principal's budget. The
                    // limiter is advisory here (the request already failed);
                    // exceeding the budget is itself an auditable signal.
                    if let Some(limiter) = &self.rate_limiter {
                        let verdict = limiter.record_failed_auth(&attempted.user_id).await;
                        if !verdict.allowed {
                            tracing::warn!(
                                target: "proximadb.security.audit",
                                principal = %attempted.user_id,
                                "failed-authentication budget exceeded - possible brute force"
                            );
                        }
                    }
                    let failed_result = AuthenticationResult {
                        user_context: attempted,
                        auth_method: UnifiedAuthMethod::Internal,
                        success: false,
                        error_message: Some("Authentication failed".to_string()),
                        requires_mfa: false,
                    };
                    create_auth_audit_event(&auth_data, &failed_result, start_time)
                }
            };
            // An audit-write failure must never be silent (this was `let _ =`).
            // Policy: LOUD, fail-open by default. The credible failure mode is
            // operational (a full disk — note the retention job has no
            // production caller), and failing closed converts that into a total
            // authentication outage for every tenant: a self-inflicted DoS. An
            // adversary who can break the sink can already rewrite the
            // unchained history, so failing closed buys no accountability
            // against them either. What matters is that the loss of
            // accountability is VISIBLE.
            if let Err(error) = audit_logger.log_event(auth_event).await {
                AUDIT_WRITE_FAILURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                tracing::error!(
                    target: "proximadb.security.audit",
                    %error,
                    "AUDIT WRITE FAILED - proceeding unaudited (fail-open); set \
                     authentication.audit_fail_closed = true to deny instead"
                );
                if self.config.audit_fail_closed {
                    return Err(anyhow!("authentication unavailable: audit write failed"));
                }
            }
        }

        result
    }

    /// Authenticate JWT token
    async fn authenticate_jwt_token(&self, token: &str) -> Result<AuthenticationResult> {
        if !self.config.methods.contains(&AuthenticationMethod::JWT) {
            return Ok(AuthenticationResult {
                user_context: UnifiedUserContext::anonymous(),
                auth_method: UnifiedAuthMethod::JWT,
                success: false,
                error_message: Some("JWT authentication disabled".to_string()),
                requires_mfa: false,
            });
        }

        // TD-SSO-1 dispatch: the UNAUTHENTICATED header `alg` only ROUTES —
        // it selects a verifier that either cryptographically validates or
        // rejects; it never authorizes. RS*/ES* goes to the OIDC verifier
        // when configured (the local service is HMAC-only); HS* stays local.
        let is_asymmetric = jsonwebtoken::decode_header(token)
            .map(|header| {
                matches!(
                    header.alg,
                    jsonwebtoken::Algorithm::RS256
                        | jsonwebtoken::Algorithm::RS384
                        | jsonwebtoken::Algorithm::RS512
                        | jsonwebtoken::Algorithm::ES256
                        | jsonwebtoken::Algorithm::ES384
                )
            })
            .unwrap_or(false);

        if is_asymmetric && let Some(verifier) = &self.oidc_verifier {
            return match verifier.verify(token).await {
                Ok(claims) => {
                    let user_context =
                        self.convert_oidc_claims_to_unified(claims, verifier.config());
                    Ok(AuthenticationResult {
                        user_context,
                        auth_method: UnifiedAuthMethod::JWT,
                        success: true,
                        error_message: None,
                        requires_mfa: false,
                    })
                }
                Err(e) => {
                    warn!("OIDC bearer validation failed: {}", e);
                    Ok(AuthenticationResult {
                        user_context: UnifiedUserContext::anonymous(),
                        auth_method: UnifiedAuthMethod::JWT,
                        success: false,
                        error_message: Some(e.to_string()),
                        requires_mfa: false,
                    })
                }
            };
        }

        match &self.jwt_service {
            Some(jwt_service) => match jwt_service.verify_token(token).await {
                Ok(claims) => {
                    let user_context = self.convert_jwt_claims_to_unified(claims);
                    Ok(AuthenticationResult {
                        user_context,
                        auth_method: UnifiedAuthMethod::JWT,
                        success: true,
                        error_message: None,
                        requires_mfa: false,
                    })
                }
                Err(e) => {
                    warn!("JWT authentication failed: {}", e);
                    Ok(AuthenticationResult {
                        user_context: UnifiedUserContext::anonymous(),
                        auth_method: UnifiedAuthMethod::JWT,
                        success: false,
                        error_message: Some(e.to_string()),
                        requires_mfa: false,
                    })
                }
            },
            None => Err(anyhow!(
                "JWT authentication enabled but JWT service not configured"
            )),
        }
    }

    /// TD-SSO-1: verified OIDC claims → UnifiedUserContext. Roles cross ONLY
    /// through `sanitize_idp_roles` (the #1791 invariant, uniform); the
    /// optional tenant claim maps to tenant_id; ZERO permissions are derived
    /// from IdP claims (RBAC stays permission-driven).
    fn convert_oidc_claims_to_unified(
        &self,
        claims: crate::network::auth::oidc::OidcClaims,
        cfg: &crate::network::auth::oidc::OidcProviderConfig,
    ) -> UnifiedUserContext {
        let raw_roles = crate::network::auth::oidc::roles_raw_to_strings(&claims.roles_raw);
        let roles = crate::network::auth::oidc::sanitize_idp_roles(&raw_roles, cfg);
        let tenant_id = crate::network::auth::oidc::tenant_raw_to_option(&claims.tenant_raw);
        // F3 (adversarial review): the IdP's issuer-scoped `sub` must NOT be
        // the raw local join key — get_effective_permissions looks up
        // user_role_assignments[user_id], so a colliding sub (email-as-sub
        // providers; a local principal named "admin") would INHERIT that
        // principal's locally granted permissions. Namespace it: local role
        // assignments targeting OIDC users must use the `oidc:{sub}` form.
        let oidc_user_id = format!("oidc:{}", claims.sub);
        let oidc_session = format!("oidc_{}", claims.sub);
        let mut metadata = HashMap::new();
        // N3 (adversarial review pass 2): mark the principal's provenance so
        // the tenant middleware can exclude IdP-asserted tenant claims from
        // the system_tenants compat fallback (a token whose `tenant` claim
        // lands in system_tenants must not become a gateway-class principal).
        metadata.insert("oidc".to_string(), "true".to_string());
        UnifiedUserContext {
            user_id: oidc_user_id,
            tenant_id,
            roles,
            effective_permissions: HashSet::new(),
            auth_method: UnifiedAuthMethod::JWT,
            session_id: oidc_session,
            expires_at: DateTime::from_timestamp(claims.exp, 0),
            created_at: DateTime::from_timestamp(claims.iat, 0).unwrap_or_else(Utc::now),
            metadata,
        }
    }

    /// Authenticate API key
    async fn authenticate_api_key(&self, api_key: &str) -> Result<AuthenticationResult> {
        if !self.config.methods.contains(&AuthenticationMethod::ApiKey) {
            return Ok(AuthenticationResult {
                user_context: UnifiedUserContext::anonymous(),
                auth_method: UnifiedAuthMethod::ApiKey,
                success: false,
                error_message: Some("API key authentication disabled".to_string()),
                requires_mfa: false,
            });
        }

        // ADR-090 L0: the `pxk_` namespace is REGISTRY-AUTHORITATIVE. A key that
        // *looks like* a registry key never falls through to the config table —
        // with no registry attached, or on any resolve failure (unknown, wrong
        // secret, revoked, expired, principal disabled), it fails closed with
        // the same uniform error. This makes a config-table entry that mimics
        // the registry format a spoof attempt, not an alternate authority.
        if api_key.starts_with(REGISTRY_KEY_PREFIX) {
            let resolved = self
                .principal_registry
                .as_ref()
                .and_then(|registry| registry.resolve_key(api_key));
            return Ok(match resolved {
                Some(ref key) => AuthenticationResult {
                    user_context: self.registry_key_to_unified(key),
                    auth_method: UnifiedAuthMethod::ApiKey,
                    success: true,
                    error_message: None,
                    requires_mfa: false,
                },
                None => AuthenticationResult {
                    user_context: UnifiedUserContext::anonymous(),
                    auth_method: UnifiedAuthMethod::ApiKey,
                    success: false,
                    error_message: Some("Invalid API key".to_string()),
                    requires_mfa: false,
                },
            });
        }

        match self.api_keys.get(api_key) {
            Some(api_key_info) => {
                // Check if API key has expired
                if let Some(expires_at) = api_key_info.expires_at
                    && Utc::now() > expires_at
                {
                    return Ok(AuthenticationResult {
                        user_context: UnifiedUserContext::anonymous(),
                        auth_method: UnifiedAuthMethod::ApiKey,
                        success: false,
                        error_message: Some("API key expired".to_string()),
                        requires_mfa: false,
                    });
                }

                let user_context = self.convert_api_key_to_unified(api_key_info.clone());
                Ok(AuthenticationResult {
                    user_context,
                    auth_method: UnifiedAuthMethod::ApiKey,
                    success: true,
                    error_message: None,
                    requires_mfa: false,
                })
            }
            None => {
                warn!(
                    "Invalid API key attempted: {}",
                    &api_key[..std::cmp::min(8, api_key.len())]
                );
                Ok(AuthenticationResult {
                    user_context: UnifiedUserContext::anonymous(),
                    auth_method: UnifiedAuthMethod::ApiKey,
                    success: false,
                    error_message: Some("Invalid API key".to_string()),
                    requires_mfa: false,
                })
            }
        }
    }

    /// Authenticate client certificate via mTLS validation.
    ///
    /// When raw DER bytes are present in `cert_data`, the certificate is fully
    /// parsed and validated (expiry, CA signature, CN/SAN extraction, role mapping).
    /// Without raw bytes the method falls back to the pre-populated metadata fields
    /// on `ClientCertificateData`.
    async fn authenticate_client_certificate(
        &self,
        cert_data: &ClientCertificateData,
    ) -> Result<AuthenticationResult> {
        if !self
            .config
            .methods
            .contains(&AuthenticationMethod::ClientCertificate)
        {
            return Ok(AuthenticationResult {
                user_context: UnifiedUserContext::anonymous(),
                auth_method: UnifiedAuthMethod::ClientCertificate,
                success: false,
                error_message: Some("Client certificate authentication disabled".to_string()),
                requires_mfa: false,
            });
        }

        if !self.config.mtls.enabled {
            return Ok(AuthenticationResult {
                user_context: UnifiedUserContext::anonymous(),
                auth_method: UnifiedAuthMethod::ClientCertificate,
                success: false,
                error_message: Some("mTLS is not enabled in configuration".to_string()),
                requires_mfa: false,
            });
        }

        // Validate the certificate and extract identity
        let identity = match &cert_data.raw_cert_der {
            Some(der_bytes) => match self.validate_client_certificate(der_bytes) {
                Ok(id) => id,
                Err(e) => {
                    warn!("mTLS certificate validation failed: {}", e);
                    return Ok(AuthenticationResult {
                        user_context: UnifiedUserContext::anonymous(),
                        auth_method: UnifiedAuthMethod::ClientCertificate,
                        success: false,
                        error_message: Some(format!("Certificate validation failed: {}", e)),
                        requires_mfa: false,
                    });
                }
            },
            None => {
                // Fallback: use pre-populated metadata when raw bytes are unavailable.
                // Only basic expiry checking is possible in this path.
                let now = Utc::now();
                if now < cert_data.not_before || now > cert_data.not_after {
                    return Ok(AuthenticationResult {
                        user_context: UnifiedUserContext::anonymous(),
                        auth_method: UnifiedAuthMethod::ClientCertificate,
                        success: false,
                        error_message: Some(
                            "Client certificate has expired or is not yet valid".to_string(),
                        ),
                        requires_mfa: false,
                    });
                }

                let roles = self.resolve_roles_for_cn(&cert_data.subject);
                ClientIdentity {
                    common_name: cert_data.subject.clone(),
                    san_entries: Vec::new(),
                    roles,
                }
            }
        };

        if identity.roles.is_empty() {
            warn!(
                "No roles mapped for client certificate CN={}",
                identity.common_name
            );
            return Ok(AuthenticationResult {
                user_context: UnifiedUserContext::anonymous(),
                auth_method: UnifiedAuthMethod::ClientCertificate,
                success: false,
                error_message: Some(format!(
                    "No roles mapped for certificate CN={}",
                    identity.common_name
                )),
                requires_mfa: false,
            });
        }

        info!(
            "mTLS authentication succeeded for CN={}, roles={:?}",
            identity.common_name, identity.roles
        );

        let user_context = UnifiedUserContext {
            user_id: identity.common_name.clone(),
            tenant_id: None,
            roles: identity.roles,
            effective_permissions: HashSet::new(), // Populated by RBAC manager
            auth_method: UnifiedAuthMethod::ClientCertificate,
            session_id: format!("mtls_{}", uuid::Uuid::new_v4()),
            expires_at: None, // Certificate expiry is validated at authentication time
            created_at: Utc::now(),
            metadata: {
                let mut m = HashMap::new();
                m.insert("auth_cn".to_string(), identity.common_name);
                if !identity.san_entries.is_empty() {
                    m.insert("auth_sans".to_string(), identity.san_entries.join(","));
                }
                m
            },
        };

        Ok(AuthenticationResult {
            user_context,
            auth_method: UnifiedAuthMethod::ClientCertificate,
            success: true,
            error_message: None,
            requires_mfa: false,
        })
    }

    /// Load a CA certificate from a PEM or DER file and return its DER bytes.
    fn load_ca_certificate(path: &str) -> Result<Vec<u8>> {
        let file_bytes = std::fs::read(path)
            .map_err(|e| anyhow!("Failed to read CA certificate at '{}': {}", path, e))?;

        // Try PEM first
        if let Ok(pem_contents) = std::str::from_utf8(&file_bytes) {
            let mut reader = std::io::BufReader::new(pem_contents.as_bytes());
            // rustls-pemfile 2.0: certs() yields Result<CertificateDer>; take the first.
            if let Some(first_cert) = rustls_pemfile::certs(&mut reader).next() {
                let first_cert =
                    first_cert.map_err(|e| anyhow!("Failed to parse PEM certificates: {}", e))?;
                return Ok(first_cert.as_ref().to_vec());
            }
        }

        // Fallback: treat as raw DER
        // Validate it parses as an X.509 certificate
        X509Certificate::from_der(&file_bytes)
            .map_err(|e| anyhow!("CA file is neither valid PEM nor DER: {:?}", e))?;
        Ok(file_bytes)
    }

    /// Validate a DER-encoded client certificate against the configured CA,
    /// check expiry, extract CN and SANs, and resolve RBAC roles.
    pub fn validate_client_certificate(&self, cert_der: &[u8]) -> Result<ClientIdentity> {
        let ca_der = self
            .ca_cert_der
            .as_ref()
            .ok_or_else(|| anyhow!("No CA certificate loaded for mTLS validation"))?;

        // Parse the CA certificate
        let (_, ca_cert) = X509Certificate::from_der(ca_der)
            .map_err(|e| anyhow!("Failed to parse CA certificate: {:?}", e))?;

        // Parse the client certificate
        let (_, client_cert) = X509Certificate::from_der(cert_der)
            .map_err(|e| anyhow!("Failed to parse client certificate: {:?}", e))?;

        // 1. Verify the client certificate is within its validity period
        let now = ASN1Time::now();
        if !client_cert.validity().is_valid_at(now) {
            let not_before = client_cert.validity().not_before;
            let not_after = client_cert.validity().not_after;
            return Err(anyhow!(
                "Client certificate is not valid at current time (valid {} to {})",
                not_before,
                not_after
            ));
        }

        // 2. Verify the client certificate was issued by the CA.
        //    Compare the client cert's issuer DN with the CA cert's subject DN.
        //    Additionally verify that the Authority Key Identifier (if present)
        //    matches the CA's Subject Key Identifier.
        if client_cert.issuer() != ca_cert.subject() {
            return Err(anyhow!(
                "Certificate issuer does not match the configured CA subject (issuer={}, ca_subject={})",
                client_cert.issuer(),
                ca_cert.subject()
            ));
        }

        // Cross-check key identifiers when both extensions are present
        // x509_parser exposes extensions via the parsed extensions map
        let client_aki = client_cert
            .tbs_certificate
            .extensions_map()
            .ok()
            .and_then(|map| {
                map.get(&x509_parser::oid_registry::OID_X509_EXT_AUTHORITY_KEY_IDENTIFIER)
                    .cloned()
            })
            .and_then(|ext| {
                x509_parser::extensions::AuthorityKeyIdentifier::from_der(ext.value)
                    .ok()
                    .and_then(|(_, aki)| aki.key_identifier.map(|ki| ki.0.to_vec()))
            });
        let ca_ski = ca_cert
            .tbs_certificate
            .extensions_map()
            .ok()
            .and_then(|map| {
                map.get(&x509_parser::oid_registry::OID_X509_EXT_SUBJECT_KEY_IDENTIFIER)
                    .cloned()
            })
            .map(|ext| ext.value.to_vec());

        if let (Some(aki), Some(ski)) = (&client_aki, &ca_ski)
            && aki != ski
        {
            return Err(anyhow!(
                "Certificate Authority Key Identifier does not match CA Subject Key Identifier"
            ));
        }

        // 3. Extract Common Name from the subject
        let common_name = client_cert
            .subject()
            .iter_common_name()
            .next()
            .and_then(|cn| cn.as_str().ok())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow!("Client certificate has no Common Name (CN) in subject"))?;

        // 4. Extract Subject Alternative Names
        let san_entries = Self::extract_san_entries(&client_cert);

        // 5. Resolve RBAC roles from CN
        let roles = self.resolve_roles_for_cn(&common_name);

        Ok(ClientIdentity {
            common_name,
            san_entries,
            roles,
        })
    }

    /// Extract Subject Alternative Name entries from an X.509 certificate.
    fn extract_san_entries(cert: &X509Certificate<'_>) -> Vec<String> {
        let mut entries = Vec::new();
        if let Ok(Some(san_ext)) = cert.subject_alternative_name() {
            for name in &san_ext.value.general_names {
                match name {
                    GeneralName::DNSName(dns) => {
                        entries.push(format!("DNS:{}", dns));
                    }
                    GeneralName::RFC822Name(email) => {
                        entries.push(format!("email:{}", email));
                    }
                    GeneralName::URI(uri) => {
                        entries.push(format!("URI:{}", uri));
                    }
                    GeneralName::IPAddress(ip_bytes) => {
                        let ip_str = if ip_bytes.len() == 4 {
                            format!(
                                "IP:{}.{}.{}.{}",
                                ip_bytes[0], ip_bytes[1], ip_bytes[2], ip_bytes[3]
                            )
                        } else {
                            format!("IP:{}", hex::encode(ip_bytes))
                        };
                        entries.push(ip_str);
                    }
                    _ => {
                        entries.push(format!("other:{:?}", name));
                    }
                }
            }
        }
        entries
    }

    /// Resolve RBAC roles for a given Common Name using the configured cn_role_mapping.
    ///
    /// Supports exact matches and simple wildcard patterns:
    /// - Trailing wildcard: "service-*" matches "service-auth", "service-gateway"
    /// - Leading wildcard: "*.admin" matches "db.admin", "cluster.admin"
    /// - Exact match: "root" matches only "root"
    fn resolve_roles_for_cn(&self, cn: &str) -> Vec<String> {
        let mut roles = Vec::new();
        for (pattern, role) in &self.config.mtls.cn_role_mapping {
            let matched = if let Some(suffix) = pattern.strip_prefix('*') {
                // Leading wildcard: "*.admin" matches anything ending with ".admin"
                cn.ends_with(suffix)
            } else if pattern.ends_with('*') {
                // Trailing wildcard: "service-*" matches anything starting with "service-"
                let prefix = &pattern[..pattern.len() - 1];
                cn.starts_with(prefix)
            } else {
                // Exact match
                cn == pattern
            };

            if matched && !roles.contains(role) {
                roles.push(role.clone());
            }
        }
        roles
    }

    /// Convert enterprise user context to unified context
    fn convert_jwt_claims_to_unified(
        &self,
        claims: crate::network::auth::Claims,
    ) -> UnifiedUserContext {
        let mut metadata = HashMap::new();
        if let Some(value) = claims.capability_type {
            metadata.insert("capability_type".to_string(), value);
        }
        if let Some(value) = claims.collection {
            metadata.insert("collection".to_string(), value);
        }
        if let Some(value) = claims.operation {
            metadata.insert("operation".to_string(), value);
        }
        if let Some(value) = claims.protocol {
            metadata.insert("protocol".to_string(), value);
        }
        if let Some(value) = claims.mode {
            metadata.insert("mode".to_string(), value);
        }
        if !claims.scopes.is_empty() {
            metadata.insert("scopes".to_string(), claims.scopes.join(" "));
        }
        if let Some(value) = claims.max_records {
            metadata.insert("max_records".to_string(), value.to_string());
        }
        if let Some(value) = claims.max_bytes {
            metadata.insert("max_bytes".to_string(), value.to_string());
        }
        if let Some(value) = claims.tier {
            metadata.insert("tier".to_string(), value);
        }
        if let Some(value) = claims.route_visibility {
            metadata.insert("route_visibility".to_string(), value);
        }
        if let Some(value) = claims.metering_required {
            metadata.insert("metering_required".to_string(), value.to_string());
        }

        // TD-SSO-1 uniform seam: when an OIDC provider is configured, LOCAL
        // tokens sanitize through the SAME allowlist as IdP tokens (the
        // #1791 invariant holds on every bearer path). With no OIDC config,
        // local behavior is unchanged — the local mint requires the engine's
        // own secret, and existing capability tokens carry role-like strings
        // (e.g. `data_plane`) that an empty default allowlist would strip.
        let roles = match self.oidc_verifier.as_ref() {
            Some(verifier) => {
                crate::network::auth::oidc::sanitize_idp_roles(&claims.roles, verifier.config())
            }
            None => claims.roles,
        };

        UnifiedUserContext {
            user_id: claims.sub,
            tenant_id: claims.tenant_id,
            roles,
            effective_permissions: HashSet::new(), // Will be populated by RBAC manager
            auth_method: UnifiedAuthMethod::JWT,
            session_id: claims.jti,
            expires_at: Some(DateTime::from_timestamp(claims.exp, 0).unwrap_or_else(Utc::now)),
            created_at: DateTime::from_timestamp(claims.iat, 0).unwrap_or_else(Utc::now),
            metadata,
        }
    }

    /// Convert API key info to unified context
    /// Attach the ADR-090 catalog principal registry. Once attached, keys in
    /// the `pxk_` namespace resolve exclusively through it (fail-closed).
    /// Attach the failed-authentication throttle. Without this the brute-force
    /// budget is unenforced (today's behavior).
    pub fn set_rate_limiter(
        &mut self,
        limiter: Arc<crate::security::advanced_features::RateLimitingService>,
    ) {
        self.rate_limiter = Some(limiter);
    }

    pub fn set_principal_registry(&mut self, registry: Arc<FileSystemPrincipalRegistry>) {
        self.principal_registry = Some(registry);
    }

    /// Build the identity for a registry-resolved key. Permissions are
    /// deliberately EMPTY: under ADR-090 the credential proves who you are
    /// (subject + tenant); what you may do comes from grants/ABAC (L1/L2),
    /// not from strings stored next to the key.
    fn registry_key_to_unified(
        &self,
        resolved: &proximadb_catalog::principal_registry::ResolvedApiKey,
    ) -> UnifiedUserContext {
        UnifiedUserContext {
            user_id: resolved.subject.0.clone(),
            tenant_id: Some(resolved.tenant_id.clone()),
            roles: vec!["api_user".to_string()],
            effective_permissions: std::collections::HashSet::new(),
            auth_method: UnifiedAuthMethod::ApiKey,
            session_id: format!("apikey_{}", uuid::Uuid::new_v4()),
            expires_at: None,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    fn convert_api_key_to_unified(&self, api_key_info: ApiKeyInfo) -> UnifiedUserContext {
        // Convert string permissions to UnifiedPermission enum.
        // A bare `"*"` in the config is the "all permissions" shorthand
        // operators reach for in dev / single-node setups — fan it out
        // to the SystemAdmin + ConfigureSystem + TenantAdmin + write
        // permissions the operator-gated endpoints actually check for
        // (recall-tune, recluster, suspend, resume, primary-pod
        // mutations). Specific tokens still parse via
        // `parse_permission_string`.
        let mut permissions: std::collections::HashSet<UnifiedPermission> =
            std::collections::HashSet::new();
        for p in &api_key_info.permissions {
            if p == "*" {
                permissions.extend([
                    UnifiedPermission::SystemAdmin,
                    UnifiedPermission::ConfigureSystem,
                    UnifiedPermission::TenantAdmin,
                    UnifiedPermission::TenantRead,
                    UnifiedPermission::TenantWrite,
                    UnifiedPermission::CollectionCreate,
                ]);
            } else if let Some(parsed) = self.parse_permission_string(p) {
                permissions.insert(parsed);
            }
        }

        let key_user_id = api_key_info.user_id.clone();
        UnifiedUserContext {
            user_id: api_key_info.user_id,
            tenant_id: api_key_info.tenant_id,
            // TD-TENANT-1 follow-up (a): configured roles ride alongside the
            // default (nothing checking `api_user` regresses); `gateway`/
            // `operator` here are what `is_gateway_principal` reads.
            //
            // ADVERSARIAL REVIEW (2026-08-31): pass-through of ARBITRARY role
            // strings was an escalation — `SecurityPredicate::RoleBased`
            // (security/rls/service.rs) grants Unrestricted row access when
            // roles contain an allowed value, so a config-supplied business
            // role would silently satisfy RLS policies (impossible before:
            // API-key roles were hardcoded to api_user). Sanitize at this
            // seam: ONLY the exact gateway/operator delegation markers pass;
            // every other configured role is DROPPED with a warn. The
            // delegation-only invariant now holds by construction.
            roles: {
                const DELEGATION_ROLES: [&str; 2] = [
                    proximadb_tenant::GATEWAY_ROLE,
                    proximadb_tenant::OPERATOR_ROLE,
                ];
                let mut roles = vec!["api_user".to_string()];
                for r in &api_key_info.roles {
                    if DELEGATION_ROLES.contains(&r.as_str()) {
                        if !roles.iter().any(|existing| existing == r) {
                            roles.push(r.clone());
                        }
                    } else {
                        tracing::warn!(
                            key_user = %key_user_id,
                            dropped_role = %r,
                            "API-key role ignored: only the gateway/operator                              delegation markers are honored (RLS RoleBased                              predicates match role strings — arbitrary                              pass-through would be an escalation)"
                        );
                    }
                }
                roles
            },
            effective_permissions: permissions,
            auth_method: UnifiedAuthMethod::ApiKey,
            session_id: format!("apikey_{}", uuid::Uuid::new_v4()),
            expires_at: api_key_info.expires_at,
            created_at: api_key_info.created_at.unwrap_or_else(Utc::now),
            metadata: HashMap::new(),
        }
    }

    /// The SCRAM verifier for `username`, if configured (TD-PGWIRE-AUTH-1).
    /// Consumed by the pgwire protocol layer to run the SASL exchange.
    pub fn scram_verifier(&self, username: &str) -> Option<scram::ScramVerifier> {
        self.scram_verifiers.get(username).map(|e| e.clone())
    }

    /// True when any pgwire SCRAM identity is configured.
    pub fn has_scram_users(&self) -> bool {
        !self.scram_verifiers.is_empty()
    }

    /// Authenticate a pgwire SCRAM identity (TD-PGWIRE-AUTH-1). The client
    /// proof has ALREADY been verified by the protocol layer's
    /// `ScramExchange` — this resolves the verified username into a
    /// `UnifiedUserContext` from the config tables. Failure here is a
    /// configuration mismatch (identity removed between exchange and resolve),
    /// never a secret comparison.
    async fn authenticate_scram_user(&self, username: &str) -> Result<AuthenticationResult> {
        if self.scram_verifiers.is_empty() {
            return Ok(AuthenticationResult {
                user_context: UnifiedUserContext::anonymous(),
                auth_method: UnifiedAuthMethod::Internal,
                success: false,
                error_message: Some("SCRAM authentication not configured".to_string()),
                requires_mfa: false,
            });
        }
        let Some(identity) = self
            .scram_identities
            .get(username)
            .map(|entry| entry.clone())
        else {
            return Ok(AuthenticationResult {
                user_context: UnifiedUserContext::anonymous(),
                auth_method: UnifiedAuthMethod::Internal,
                success: false,
                error_message: Some("Invalid SCRAM credentials".to_string()),
                requires_mfa: false,
            });
        };
        Ok(AuthenticationResult {
            user_context: self.convert_scram_user_to_unified(username, &identity),
            auth_method: UnifiedAuthMethod::Internal,
            success: true,
            error_message: None,
            requires_mfa: false,
        })
    }

    /// Build the [`UnifiedUserContext`] for a verified SCRAM identity. Role
    /// claims follow the SAME delegation-only sanitize policy as API keys:
    /// only the exact `gateway`/`operator` markers pass (an arbitrary role
    /// string would satisfy RLS RoleBased predicates — an escalation).
    fn convert_scram_user_to_unified(
        &self,
        username: &str,
        user: &ScramUserConfig,
    ) -> UnifiedUserContext {
        const DELEGATION_ROLES: [&str; 2] = [
            proximadb_tenant::GATEWAY_ROLE,
            proximadb_tenant::OPERATOR_ROLE,
        ];
        let mut roles = vec!["api_user".to_string()];
        for r in &user.roles {
            if DELEGATION_ROLES.contains(&r.as_str()) {
                if !roles.iter().any(|existing| existing == r) {
                    roles.push(r.clone());
                }
            } else {
                warn!(
                    scram_user = %username,
                    dropped_role = %r,
                    "SCRAM role ignored: only the gateway/operator delegation \
                     markers are honored (same escalation rule as API keys)"
                );
            }
        }
        UnifiedUserContext {
            user_id: username.to_string(),
            tenant_id: user.tenant_id.clone(),
            roles,
            // No configured permissions: authorization stays permission/RBAC-
            // driven exactly as for a zero-permission API key.
            effective_permissions: HashSet::new(),
            auth_method: UnifiedAuthMethod::Internal,
            session_id: format!("scram_{}", uuid::Uuid::new_v4()),
            expires_at: None,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Parse permission string to UnifiedPermission enum
    fn parse_permission_string(&self, permission_str: &str) -> Option<UnifiedPermission> {
        match permission_str {
            "read" => Some(UnifiedPermission::TenantRead),
            "write" => Some(UnifiedPermission::TenantWrite),
            "admin" => Some(UnifiedPermission::TenantAdmin),
            "collection_create" => Some(UnifiedPermission::CollectionCreate),
            "system_admin" => Some(UnifiedPermission::SystemAdmin),
            _ => {
                warn!("Unknown permission string: {}", permission_str);
                None
            }
        }
    }

    /// Generate JWT token pair
    pub async fn generate_token_pair(
        &self,
        user_context: &UnifiedUserContext,
    ) -> Result<TokenPair> {
        match &self.jwt_service {
            Some(jwt_service) => jwt_service
                .generate_token_pair(
                    &user_context.user_id,
                    user_context.tenant_id.clone(),
                    user_context.roles.clone(),
                )
                .await
                .map_err(|e| anyhow!(e)),
            None => Err(anyhow!("JWT service not configured")),
        }
    }

    /// Refresh JWT token
    pub async fn refresh_token(&self, refresh_token: &str) -> Result<TokenPair> {
        match &self.jwt_service {
            Some(jwt_service) => jwt_service
                .refresh_token(refresh_token, None)
                .await
                .map_err(|e| anyhow!(e)),
            None => Err(anyhow!("JWT service not configured")),
        }
    }

    /// Validate if authentication method is enabled
    pub fn is_method_enabled(&self, method: &AuthenticationMethod) -> bool {
        self.config.methods.contains(method)
    }
}

/// Authentication data from request
#[derive(Debug)]
pub enum AuthenticationData {
    /// Legacy SSO — the backing manager was removed (provider portability);
    /// carries the opaque token for the fail-closed stub.
    SSOToken(String),
    JWTToken(String),
    ApiKey(String),
    ClientCertificate(ClientCertificateData),
    /// pgwire SCRAM-SHA-256, server-side-only (TD-PGWIRE-AUTH-1): the
    /// protocol layer constructs this ONLY after the SASL exchange verified
    /// the client proof. `parse_authorization` can never produce it, so no
    /// HTTP surface can forge this shape.
    ScramAuthenticated {
        username: String,
    },
}

/// Client certificate data for mTLS
#[derive(Debug, Clone)]
pub struct ClientCertificateData {
    pub subject: String,
    pub issuer: String,
    pub serial_number: String,
    pub not_before: DateTime<Utc>,
    pub not_after: DateTime<Utc>,
    /// Raw DER-encoded client certificate bytes for full validation.
    /// When provided, the mTLS validator will parse and verify the certificate
    /// against the configured CA, extract CN/SANs, and map to roles.
    pub raw_cert_der: Option<Vec<u8>>,
}

// NOTE: the inherent `impl UnifiedUserContext { anonymous / is_authenticated /
// is_session_expired / display_name }` block previously defined here was moved
// to `proximadb_tenant::rbac_context` (Slice D hoist) — inherent impls must live
// in the defining crate, and the type now lives in the foundation `proximadb-
// tenant` crate. All methods remain accessible in scope via the
// `crate::security::rbac_service` re-export.

/// Create audit event for authentication attempt
/// A stable, **non-secret** identifier for the credential that was PRESENTED,
/// used to attribute failed authentications.
///
/// This is what makes per-principal brute-force detection possible at all: the
/// detector counts recent failures keyed on the audit event's `user_id`, and a
/// failed authentication has no authenticated user yet. Attributing every
/// failure to `anonymous()` (the previous behavior) collapses every failure in
/// the system into ONE bucket, so one account under attack is indistinguishable
/// from unrelated background noise.
///
/// SECRECY RULE: the result is persisted in an audit record, so it must never
/// contain secret material. A `pxk_` key carries a public key id before the `.`
/// separator — already public, and the natural attribution. Every other
/// credential shape offers only the secret itself as an identifier, so we emit
/// a truncated SHA-256 **fingerprint**: stable enough to count against, useless
/// to replay.
fn attempted_principal(auth_data: &AuthenticationData) -> String {
    fn fingerprint(secret: &str) -> String {
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(secret.as_bytes());
        let mut out = String::with_capacity(16);
        for byte in digest.iter().take(8) {
            use std::fmt::Write as _;
            let _ = write!(out, "{byte:02x}");
        }
        format!("fp:{out}")
    }

    match auth_data {
        AuthenticationData::ApiKey(key) => match key
            .strip_prefix(REGISTRY_KEY_PREFIX)
            .and_then(|rest| rest.split_once('.'))
        {
            // Registry key: the id before '.' is public by construction.
            Some((key_id, _secret)) => format!("{REGISTRY_KEY_PREFIX}{key_id}"),
            // Legacy/config key: no public component exists — fingerprint it.
            None => fingerprint(key),
        },
        AuthenticationData::JWTToken(token) => fingerprint(token),
        AuthenticationData::SSOToken(_) => "sso:unverified".to_string(),
        AuthenticationData::ClientCertificate(cert) => format!("cert:{}", cert.subject),
        // The username is not secret (it is presented in the clear on the wire).
        AuthenticationData::ScramAuthenticated { username } => format!("scram:{username}"),
    }
}

fn create_auth_audit_event(
    auth_data: &AuthenticationData,
    result: &AuthenticationResult,
    _start_time: DateTime<Utc>,
) -> AuditEvent {
    let auth_method_name = match auth_data {
        AuthenticationData::SSOToken(_) => "sso",
        AuthenticationData::JWTToken(_) => "jwt",
        AuthenticationData::ApiKey(_) => "api_key",
        AuthenticationData::ClientCertificate(_) => "client_certificate",
        AuthenticationData::ScramAuthenticated { .. } => "scram_sha_256",
    };

    AuditEvent {
        event_id: uuid::Uuid::new_v4().to_string(),
        event_type: AuditEventType::Authentication,
        timestamp: Utc::now(),
        user_id: Some(result.user_context.user_id.clone()),
        tenant_id: result.user_context.tenant_id.clone(),
        resource: AuditResource {
            resource_type: "authentication".to_string(),
            resource_id: result.user_context.session_id.clone(),
            parent_resource: None,
            resource_tenant_id: None,
        },
        action: "authenticate".to_string(),
        result: if result.success {
            AuditResult::Success
        } else {
            AuditResult::Failure {
                error_code: "AUTH_FAILED".to_string(),
                error_message: result
                    .error_message
                    .clone()
                    .unwrap_or_else(|| "Authentication failed".to_string()),
            }
        },
        ip_address: None, // Would be populated by middleware
        user_agent: None,
        session_id: Some(result.user_context.session_id.clone()),
        request_id: None,
        details: {
            let mut details = HashMap::new();
            details.insert(
                "auth_method".to_string(),
                serde_json::json!(auth_method_name),
            );
            details.insert("success".to_string(), serde_json::json!(result.success));
            if let Some(err_msg) = &result.error_message {
                details.insert("error_message".to_string(), serde_json::json!(err_msg));
            }
            details.insert(
                "requires_mfa".to_string(),
                serde_json::json!(result.requires_mfa),
            );
            details.insert(
                "roles".to_string(),
                serde_json::json!(result.user_context.roles),
            );
            details
        },
        risk_score: if result.success { Some(0.0) } else { Some(0.5) },
    }
}

#[cfg(test)]
mod tests {
    use super::ApiKeyInfo;

    /// N5 (adversarial review pass 2): the LOCAL-JWT retrofit arm is pinned
    /// too — the same blind-test class as F5, one function below it. With an
    /// OIDC verifier present, a locally-minted token's roles cross the SAME
    /// seam; deleting the `Some(verifier) => sanitize…` arm reverts to
    /// pass-through and THIS test fails.
    #[test]
    fn local_jwt_roles_sanitize_when_oidc_is_configured() {
        use crate::network::auth::oidc::test_fixtures as fx;
        let cfg = super::AuthenticationConfig {
            oidc: Some(fx::verifier(&["analyst"], false)),
            ..serde_json::from_str(
                r#"{"enabled":false,"methods":["jwt"],"require_authentication":false,
                    "default_session_timeout_minutes":30,"api_keys":{},
                    "jwt":{"enabled":false,"secret":"x","issuer":"t","audience":"t",
                            "access_token_expiration_minutes":1,
                            "refresh_token_expiration_days":1,"algorithm":"HS256"},
                    "sso":{"enabled":false,"providers":[],"token_cache_ttl_minutes":1,
                            "aws_iam":null,"azure_ad":null}}"#,
            )
            .expect("minimal auth config")
        };
        let service = super::UnifiedAuthService::new(cfg).expect("service");
        let claims = crate::network::auth::Claims {
            sub: "local-user".into(),
            iat: 1,
            exp: 2,
            nbf: 1,
            iss: "local".into(),
            aud: "local".into(),
            jti: "j".into(),
            typ: crate::network::auth::jwt::TokenType::Access,
            tenant_id: None,
            roles: vec![
                "gateway".to_string(),
                "analyst".to_string(),
                "admin".to_string(),
            ],
            scopes: vec![],
            capability_type: None,
            collection: None,
            operation: None,
            protocol: None,
            mode: None,
            max_records: None,
            max_bytes: None,
            tier: None,
            route_visibility: None,
            metering_required: None,
        };
        let ctx = service.convert_jwt_claims_to_unified(claims);
        // Delegation is OFF in this fixture's oidc config (F2 default), and
        // only the allowlisted business role crosses.
        assert_eq!(
            ctx.roles,
            vec!["analyst".to_string()],
            "local tokens must hit the same seam when OIDC is configured (N5)"
        );
    }

    /// F5 (adversarial review): the seam invariant pinned WHERE IT LIVES.
    /// The reviewer's sabotage — deleting the `sanitize_idp_roles` call in
    /// `convert_oidc_claims_to_unified` — passed every oidc.rs test, because
    /// they pin the helper, not the production conversion. This test drives
    /// the real `authenticate` dispatch with a real RS256 token against a
    /// mock JWKS and asserts the resulting UnifiedUserContext.
    #[tokio::test]
    async fn oidc_seam_is_enforced_at_the_production_conversion() {
        use crate::network::auth::oidc::test_fixtures as fx;

        let server = httpmock::MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(httpmock::Method::GET).path("/jwks");
                then.status(200)
                    .header("content-type", "application/json")
                    .body(fx::TEST_JWKS_JSON);
            })
            .await;
        server
            .mock_async(|when, then| {
                when.method(httpmock::Method::GET)
                    .path("/.well-known/openid-configuration");
                then.status(200)
                    .header("content-type", "application/json")
                    .body(
                        serde_json::json!({
                            "issuer": "https://idp.example.test",
                            "jwks_uri": format!("{}/jwks", server.base_url())
                        })
                        .to_string(),
                    );
            })
            .await;

        // OIDC issuer must be https — point the config at the loopback http
        // mock by supplying jwks_url explicitly (issuer stays the https
        // claim value; only the FETCH goes to the loopback).
        let cfg = super::AuthenticationConfig {
            enabled: true,
            methods: vec![super::AuthenticationMethod::JWT],
            require_authentication: false,
            default_session_timeout_minutes: 30,
            api_keys: HashMap::new(),
            scram_users: HashMap::new(),
            jwt: super::JwtConfig {
                enabled: false,
                secret: "x".to_string(),
                access_token_expiration_minutes: 1,
                refresh_token_expiration_days: 1,
                issuer: "local".to_string(),
                audience: "local".to_string(),
                algorithm: "HS256".to_string(),
            },
            sso: super::SSOConfig {
                enabled: false,
                providers: vec![],
                token_cache_ttl_minutes: 1,
            },
            mtls: super::MtlsConfig::default(),
            audit_fail_closed: false,
            oidc: Some(crate::network::auth::oidc::OidcProviderConfig {
                jwks_url: Some(format!("{}/jwks", server.base_url())),
                role_allowlist: vec!["analyst".to_string()],
                ..fx::verifier(&[], false)
            }),
        };
        let service = super::UnifiedAuthService::new(cfg).expect("service");

        let now = chrono::Utc::now().timestamp();
        let token = fx::sign_rs256(&fx::std_claims(now));
        let result = service
            .authenticate(crate::security::AuthenticationData::JWTToken(token))
            .await
            .expect("authenticate dispatch");

        assert!(result.success, "err: {:?}", result.error_message);
        let ctx = &result.user_context;
        // THE invariant, at the production seam: IdP groups ["gateway",
        // "analyst"] — analyst allowlisted, gateway delegation OFF by
        // default (F2) — only ["analyst"] crosses.
        assert_eq!(
            ctx.roles,
            vec!["analyst".to_string()],
            "seam must sanitize at the production conversion (F5)"
        );
        // F3: identity is namespaced, not the raw sub.
        assert!(ctx.user_id.starts_with("oidc:"), "got {}", ctx.user_id);
        // Tenant claim mapped at the production seam (reviewer E-gap).
        assert_eq!(ctx.tenant_id.as_deref(), Some("tenant-9"));
        // N3: OIDC provenance is marked for the middleware's fallback gate.
        assert_eq!(ctx.metadata.get("oidc").map(String::as_str), Some("true"));
        // Claim 4: no permissions derived.
        assert!(ctx.effective_permissions.is_empty());
    }

    fn key(user: &str, roles: Vec<&str>) -> ApiKeyInfo {
        ApiKeyInfo {
            user_id: user.to_string(),
            tenant_id: Some("tenant-a".to_string()),
            permissions: vec![],
            roles: roles.into_iter().map(String::from).collect(),
            created_at: None,
            expires_at: None,
            rate_limit_per_minute: None,
            ip_restrictions: vec![],
        }
    }

    fn convert(info: ApiKeyInfo) -> crate::security::rbac_service::UnifiedUserContext {
        // The conversion is pure given a config; reach it through a minimal
        // service rather than replicating its logic here. mTLS disabled so
        // ::new never touches the filesystem.
        let cfg = super::AuthenticationConfig {
            mtls: super::MtlsConfig::default(),
            oidc: None,
            ..serde_json::from_str(
                r#"{"enabled":false,"methods":["api_key"],"require_authentication":false,
                    "default_session_timeout_minutes":30,"api_keys":{},
                    "jwt":{"enabled":false,"secret":"x","issuer":"t","audience":"t",
                            "access_token_expiration_minutes":1,
                            "refresh_token_expiration_days":1,"algorithm":"HS256"},
                    "sso":{"enabled":false,"providers":[],"token_cache_ttl_minutes":1,
                            "aws_iam":null,"azure_ad":null}}"#,
            )
            .expect("minimal auth config")
        };
        let svc = super::UnifiedAuthService::new(cfg).expect("service");
        svc.convert_api_key_to_unified(info)
    }

    /// TD-TENANT-1 follow-up (a): a configured `gateway` role makes the API
    /// key a gateway principal — previously structurally impossible (roles
    /// were hardcoded to `api_user`).
    #[test]
    fn apikey_gateway_role_makes_a_gateway_principal() {
        let ctx = convert(key("gw", vec!["gateway"]));
        assert!(
            ctx.is_gateway_principal(),
            "gateway role must stamp the principal"
        );
    }

    /// Fail-closed: a typo grants nothing — only the exact `gateway`/
    /// `operator` strings match.
    #[test]
    fn apikey_typo_role_grants_nothing() {
        let ctx = convert(key("typo", vec!["gateways", "Gateway", ""]));
        assert!(!ctx.is_gateway_principal());
    }

    /// ADVERSARIAL REVIEW (2026-08-31) regression: a config-supplied BUSINESS
    /// role must NOT reach the user context — `SecurityPredicate::RoleBased`
    /// grants Unrestricted row access on role-string match, so arbitrary
    /// pass-through would silently satisfy RLS policies. Only the delegation
    /// markers survive the conversion.
    #[test]
    fn apikey_business_roles_are_dropped_not_passed_through() {
        let ctx = convert(key("biz", vec!["analyst", "admin", "collection_user"]));
        assert!(!ctx.is_gateway_principal());
        assert_eq!(
            ctx.roles,
            vec!["api_user".to_string()],
            "only delegation markers may pass; business roles are an RLS escalation"
        );
        // And the delegation marker still passes alongside a business role.
        let ctx = convert(key("mix", vec!["analyst", "gateway"]));
        assert!(ctx.is_gateway_principal());
        assert!(!ctx.roles.contains(&"analyst".to_string()));
    }

    /// The default (no roles) keeps today's behavior exactly.
    #[test]
    fn apikey_default_roles_unchanged() {
        let ctx = convert(key("plain", vec![]));
        assert!(!ctx.is_gateway_principal());
        assert_eq!(ctx.roles, vec!["api_user".to_string()]);
    }

    /// Defense in depth: `gateway` unlocks ONLY delegation under GatewayOnly
    /// — it implies NO data-plane permission. Authz stays permission-driven.
    #[test]
    fn apikey_gateway_role_grants_no_data_plane_permission() {
        let ctx = convert(key("gw", vec!["gateway"]));
        assert!(
            ctx.effective_permissions.is_empty(),
            "gateway role must not imply any UnifiedPermission"
        );
    }

    /// Config compatibility: TOML without `roles` parses (serde default).
    #[test]
    fn apikey_toml_without_roles_parses() {
        let toml = r#"
user_id = "u1"
tenant_id = "t1"
permissions = ["read"]
ip_restrictions = []
"#;
        let info: ApiKeyInfo = toml::from_str(toml).expect("legacy TOML must parse");
        assert!(info.roles.is_empty());
    }

    /// The end-to-end contract: an API-key principal stamped `gateway` by
    /// CONFIG is accepted by the ONE shared trust primitive for GatewayOnly
    /// delegation — and one without roles is refused. This is the property
    /// TD-TENANT-1 follow-up (a) exists for; tested at the primitive seam the
    /// surfaces actually call.
    #[test]
    fn apikey_gateway_principal_is_accepted_for_gatewayonly_delegation() {
        use proximadb_tenant::{
            AuthenticatedTenantBinding, HeaderTrustPolicy, ResolvedTenantAssertion,
            resolve_tenant_assertion,
        };
        let bind =
            |ctx: &crate::security::rbac_service::UnifiedUserContext| AuthenticatedTenantBinding {
                tenant_id: ctx.tenant_id.clone().expect("key bound to tenant-a"),
                is_gateway_principal: ctx.is_gateway_principal(),
            };

        let gw = convert(key("gw", vec!["gateway"]));
        let binding = bind(&gw);
        match resolve_tenant_assertion(
            Some("tenant-b"),
            Some(&binding),
            HeaderTrustPolicy::GatewayOnly,
        ) {
            Ok(ResolvedTenantAssertion::Asserted(t)) => assert_eq!(t, "tenant-b"),
            other => panic!("gateway-role API key must delegate, got {other:?}"),
        }

        let plain = convert(key("plain", vec![]));
        let binding = bind(&plain);
        assert!(
            resolve_tenant_assertion(
                Some("tenant-b"),
                Some(&binding),
                HeaderTrustPolicy::GatewayOnly
            )
            .is_err(),
            "role-less API key must NOT delegate under GatewayOnly"
        );
    }

    #[test]
    fn apikey_toml_with_roles_round_trips() {
        let toml = r#"
user_id = "u1"
permissions = []
roles = ["gateway"]
ip_restrictions = []
"#;
        let info: ApiKeyInfo = toml::from_str(toml).expect("roles TOML must parse");
        assert_eq!(info.roles, vec!["gateway".to_string()]);
    }

    /// Exercise the public authentication path, including the configured-key
    /// lookup, instead of proving only the private conversion helper. A role
    /// stamped by config must survive authentication and reach the shared
    /// tenant-delegation primitive.
    #[tokio::test]
    async fn configured_apikey_gateway_role_survives_authentication() {
        use proximadb_tenant::{
            AuthenticatedTenantBinding, HeaderTrustPolicy, ResolvedTenantAssertion,
            resolve_tenant_assertion,
        };

        let mut api_keys = HashMap::new();
        api_keys.insert("gateway-key".to_string(), key("gw", vec!["gateway"]));
        let svc = UnifiedAuthService::new(api_key_test_config(api_keys)).expect("service");
        let authenticated = svc
            .authenticate_api_key("gateway-key")
            .await
            .expect("authentication result");
        assert!(authenticated.success);
        assert!(authenticated.user_context.is_gateway_principal());

        let binding = AuthenticatedTenantBinding {
            tenant_id: authenticated
                .user_context
                .tenant_id
                .clone()
                .expect("configured key tenant"),
            is_gateway_principal: authenticated.user_context.is_gateway_principal(),
        };
        assert!(matches!(
            resolve_tenant_assertion(
                Some("tenant-b"),
                Some(&binding),
                HeaderTrustPolicy::GatewayOnly,
            ),
            Ok(ResolvedTenantAssertion::Asserted(ref tenant)) if tenant == "tenant-b"
        ));
    }

    use super::*;

    fn api_key_test_config(api_keys: HashMap<String, ApiKeyInfo>) -> AuthenticationConfig {
        AuthenticationConfig {
            enabled: true,
            methods: vec![AuthenticationMethod::ApiKey],
            require_authentication: false,
            default_session_timeout_minutes: 480,
            api_keys,
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
        }
    }

    /// ADR-090 L0.1 wiring spec: a registry-minted key authenticates through
    /// the service and yields the tenant-bound user identity — with EMPTY
    /// permissions (authorization is grants/ABAC, not credential strings).
    #[tokio::test]
    async fn registry_key_authenticates_with_tenant_bound_identity() {
        use proximadb_catalog::fc_metamodel::SubjectId;
        let dir = tempfile::tempdir().expect("tempdir");
        let registry = Arc::new(FileSystemPrincipalRegistry::open(dir.path()).expect("open"));
        registry
            .create_principal(SubjectId("alice".into()), "tenant-a", None)
            .expect("create");
        let minted = registry.mint_key("tenant-a", "alice", None).expect("mint");

        let mut svc = UnifiedAuthService::new(api_key_test_config(HashMap::new())).expect("svc");
        svc.set_principal_registry(registry.clone());

        let out = svc.authenticate_api_key(&minted.key).await.expect("auth");
        assert!(out.success);
        assert_eq!(out.user_context.user_id, "alice");
        assert_eq!(out.user_context.tenant_id.as_deref(), Some("tenant-a"));
        assert!(
            out.user_context.effective_permissions.is_empty(),
            "registry keys must not carry config-style permission strings"
        );
        assert_eq!(out.user_context.roles, vec!["api_user".to_string()]);
        assert!(
            !out.user_context.is_gateway_principal(),
            "registry keys must stay non-gateway until the registry has an explicit role authority"
        );

        // Revocation is honored through the service, not just the registry.
        registry.revoke_key(&minted.key_id).expect("revoke");
        let out = svc.authenticate_api_key(&minted.key).await.expect("auth2");
        assert!(!out.success, "revoked key must fail through the service");
    }

    /// The `pxk_` namespace is registry-authoritative: a config-table entry
    /// that mimics the registry format must NOT authenticate — whether a
    /// registry is attached (empty registry ⇒ fail-closed) or not (no
    /// registry ⇒ still fail-closed, never the config fallback).
    #[tokio::test]
    async fn pxk_namespace_never_falls_through_to_the_config_table() {
        let spoof_key = format!("{REGISTRY_KEY_PREFIX}deadbeefdeadbeef.{}", "a".repeat(64));
        let mut keys = HashMap::new();
        keys.insert(
            spoof_key.clone(),
            ApiKeyInfo {
                user_id: "mallory".to_string(),
                tenant_id: Some("tenant-x".to_string()),
                permissions: vec!["*".to_string()],
                roles: vec![],
                created_at: None,
                expires_at: None,
                rate_limit_per_minute: None,
                ip_restrictions: vec![],
            },
        );

        // No registry attached: the pxk_ key still must not hit the config table.
        let svc = UnifiedAuthService::new(api_key_test_config(keys.clone())).expect("svc");
        let out = svc.authenticate_api_key(&spoof_key).await.expect("auth");
        assert!(
            !out.success,
            "config-table spoof of a registry key authenticated (no registry)"
        );

        // Empty registry attached: same fail-closed outcome.
        let dir = tempfile::tempdir().expect("tempdir");
        let registry = Arc::new(FileSystemPrincipalRegistry::open(dir.path()).expect("open"));
        let mut svc = UnifiedAuthService::new(api_key_test_config(keys)).expect("svc2");
        svc.set_principal_registry(registry);
        let out = svc.authenticate_api_key(&spoof_key).await.expect("auth2");
        assert!(
            !out.success,
            "config-table spoof authenticated (empty registry)"
        );
    }

    #[tokio::test]
    async fn test_auth_service_service_creation() {
        let config = AuthenticationConfig {
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
        };

        let auth_service = UnifiedAuthService::new(config);
        assert!(auth_service.is_ok());
    }

    #[tokio::test]
    async fn test_anonymous_user_context() {
        let anonymous = UnifiedUserContext::anonymous();
        assert_eq!(anonymous.user_id, "anonymous");
        assert!(!anonymous.is_authenticated());
        assert!(!anonymous.is_session_expired());
    }

    #[tokio::test]
    async fn test_permission_string_parsing() {
        let config = AuthenticationConfig {
            enabled: true,
            methods: vec![AuthenticationMethod::ApiKey],
            require_authentication: false,
            default_session_timeout_minutes: 480,
            api_keys: HashMap::new(),
            scram_users: HashMap::new(),
            jwt: JwtConfig {
                enabled: false,
                secret: "test".to_string(),
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
        };

        let auth_service = UnifiedAuthService::new(config).unwrap();

        let read_perm = auth_service.parse_permission_string("read");
        assert_eq!(read_perm, Some(UnifiedPermission::TenantRead));

        let admin_perm = auth_service.parse_permission_string("admin");
        assert_eq!(admin_perm, Some(UnifiedPermission::TenantAdmin));

        let unknown_perm = auth_service.parse_permission_string("unknown");
        assert_eq!(unknown_perm, None);
    }

    // ──────────────────────────────────────────────────────────
    //  Wildcard "*" permission expansion (auth convergence pass)
    //
    //  The default dev config + most operator-provided API keys
    //  use `permissions = ["*"]` as shorthand for "all admin
    //  permissions". `convert_api_key_to_unified` must expand
    //  that to the six permissions the operator-gated endpoints
    //  actually check for; otherwise the recall-tune / recluster
    //  / suspend / resume handlers 403 every request.
    // ──────────────────────────────────────────────────────────

    fn minimal_auth_config(api_keys: HashMap<String, ApiKeyInfo>) -> AuthenticationConfig {
        AuthenticationConfig {
            enabled: true,
            methods: vec![AuthenticationMethod::ApiKey],
            require_authentication: true,
            default_session_timeout_minutes: 60,
            api_keys,
            scram_users: HashMap::new(),
            jwt: JwtConfig {
                enabled: false,
                secret: "t".to_string(),
                access_token_expiration_minutes: 15,
                refresh_token_expiration_days: 7,
                issuer: "t".to_string(),
                audience: "t".to_string(),
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
        }
    }

    fn api_key_info(user_id: &str, permissions: Vec<&str>) -> ApiKeyInfo {
        ApiKeyInfo {
            user_id: user_id.to_string(),
            tenant_id: None,
            permissions: permissions.into_iter().map(String::from).collect(),
            roles: vec![],
            created_at: None,
            expires_at: None,
            rate_limit_per_minute: None,
            ip_restrictions: Vec::new(),
        }
    }

    #[test]
    fn wildcard_expands_to_full_admin_set() {
        let service = UnifiedAuthService::new(minimal_auth_config(HashMap::new())).unwrap();
        let ctx = service.convert_api_key_to_unified(api_key_info("dev", vec!["*"]));
        // All six fan-out permissions must be present.
        assert!(
            ctx.effective_permissions
                .contains(&UnifiedPermission::SystemAdmin),
            "wildcard must grant SystemAdmin so recall-tune / recluster / suspend pass require_recall_admin"
        );
        assert!(
            ctx.effective_permissions
                .contains(&UnifiedPermission::ConfigureSystem),
            "wildcard must grant ConfigureSystem (the alternative gate on require_recall_admin)"
        );
        assert!(
            ctx.effective_permissions
                .contains(&UnifiedPermission::TenantAdmin)
        );
        assert!(
            ctx.effective_permissions
                .contains(&UnifiedPermission::TenantRead)
        );
        assert!(
            ctx.effective_permissions
                .contains(&UnifiedPermission::TenantWrite)
        );
        assert!(
            ctx.effective_permissions
                .contains(&UnifiedPermission::CollectionCreate)
        );
        assert_eq!(
            ctx.effective_permissions.len(),
            6,
            "wildcard expands to exactly the 6 documented admin-tier permissions"
        );
        assert_eq!(ctx.user_id, "dev");
        assert!(matches!(ctx.auth_method, UnifiedAuthMethod::ApiKey));
    }

    #[test]
    fn wildcard_unions_with_specific_tokens_without_duplicates() {
        // "*" + "admin" → the union is still 6 permissions (TenantAdmin
        // is already in the wildcard set). The HashSet dedup keeps the
        // count stable so audit logs don't double-count.
        let service = UnifiedAuthService::new(minimal_auth_config(HashMap::new())).unwrap();
        let ctx = service.convert_api_key_to_unified(api_key_info(
            "u",
            vec!["*", "admin", "read", "system_admin"],
        ));
        assert_eq!(
            ctx.effective_permissions.len(),
            6,
            "duplicates from specific tokens overlapping with wildcard must be deduped"
        );
        assert!(
            ctx.effective_permissions
                .contains(&UnifiedPermission::SystemAdmin)
        );
    }

    #[test]
    fn specific_permissions_without_wildcard_parse_unchanged() {
        // Regression guard: callers that don't use "*" still get the
        // exact per-token mapping from parse_permission_string.
        let service = UnifiedAuthService::new(minimal_auth_config(HashMap::new())).unwrap();
        let ctx = service.convert_api_key_to_unified(api_key_info(
            "u",
            vec!["read", "write", "collection_create"],
        ));
        assert_eq!(ctx.effective_permissions.len(), 3);
        assert!(
            ctx.effective_permissions
                .contains(&UnifiedPermission::TenantRead)
        );
        assert!(
            ctx.effective_permissions
                .contains(&UnifiedPermission::TenantWrite)
        );
        assert!(
            ctx.effective_permissions
                .contains(&UnifiedPermission::CollectionCreate)
        );
        assert!(
            !ctx.effective_permissions
                .contains(&UnifiedPermission::SystemAdmin),
            "specific tokens must NOT silently escalate to admin"
        );
    }

    #[test]
    fn unknown_tokens_are_skipped_not_errors() {
        // The wildcard path coexists with the "unknown token → warn and
        // skip" path. An unrecognised string must NOT poison the rest
        // of the permission set.
        let service = UnifiedAuthService::new(minimal_auth_config(HashMap::new())).unwrap();
        let ctx =
            service.convert_api_key_to_unified(api_key_info("u", vec!["bogus_perm", "read", "*"]));
        // 6 (from wildcard) + 0 (bogus_perm dropped) — read is already
        // inside the wildcard fan-out so no extra entry is added.
        assert_eq!(ctx.effective_permissions.len(), 6);
    }

    #[test]
    fn empty_permission_list_yields_no_authority() {
        // Operator explicitly opts out of granting anything. The
        // resulting context must be authenticated but powerless
        // (every operator-gated endpoint should 403 it).
        let service = UnifiedAuthService::new(minimal_auth_config(HashMap::new())).unwrap();
        let ctx = service.convert_api_key_to_unified(api_key_info("u", Vec::new()));
        assert!(
            ctx.effective_permissions.is_empty(),
            "empty perms list must remain empty — not silently escalated"
        );
        assert_eq!(ctx.user_id, "u");
    }

    #[test]
    fn test_mtls_config_default() {
        let config = MtlsConfig::default();
        assert!(!config.enabled);
        assert!(config.ca_cert_path.is_none());
        assert!(!config.require_client_cert);
        assert!(config.cn_role_mapping.is_empty());
    }

    #[test]
    fn test_resolve_roles_exact_match() {
        let mut cn_role_mapping = HashMap::new();
        cn_role_mapping.insert("db-admin".to_string(), "admin".to_string());
        cn_role_mapping.insert("service-gateway".to_string(), "gateway".to_string());

        let config = AuthenticationConfig {
            enabled: true,
            methods: vec![AuthenticationMethod::ClientCertificate],
            require_authentication: true,
            default_session_timeout_minutes: 60,
            api_keys: HashMap::new(),
            scram_users: HashMap::new(),
            jwt: JwtConfig {
                enabled: false,
                secret: "test".to_string(),
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
            mtls: MtlsConfig {
                enabled: false,
                ca_cert_path: None,
                require_client_cert: false,
                cn_role_mapping,
            },
            audit_fail_closed: false,
            oidc: None,
        };

        let service = UnifiedAuthService::new(config).unwrap_or_else(|_| {
            // Service creation does not fail when mtls.enabled is false
            unreachable!()
        });

        let roles = service.resolve_roles_for_cn("db-admin");
        assert_eq!(roles, vec!["admin".to_string()]);

        let roles = service.resolve_roles_for_cn("unknown-service");
        assert!(roles.is_empty());
    }

    #[test]
    fn test_resolve_roles_wildcard_prefix() {
        let mut cn_role_mapping = HashMap::new();
        cn_role_mapping.insert("service-*".to_string(), "service_role".to_string());

        let config = AuthenticationConfig {
            enabled: true,
            methods: vec![AuthenticationMethod::ClientCertificate],
            require_authentication: true,
            default_session_timeout_minutes: 60,
            api_keys: HashMap::new(),
            scram_users: HashMap::new(),
            jwt: JwtConfig {
                enabled: false,
                secret: "test".to_string(),
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
            mtls: MtlsConfig {
                enabled: false,
                ca_cert_path: None,
                require_client_cert: false,
                cn_role_mapping,
            },
            audit_fail_closed: false,
            oidc: None,
        };

        let service = UnifiedAuthService::new(config).unwrap_or_else(|_| unreachable!());

        let roles = service.resolve_roles_for_cn("service-auth");
        assert_eq!(roles, vec!["service_role".to_string()]);

        let roles = service.resolve_roles_for_cn("service-gateway");
        assert_eq!(roles, vec!["service_role".to_string()]);

        // Does not match without the prefix
        let roles = service.resolve_roles_for_cn("other-auth");
        assert!(roles.is_empty());
    }

    #[test]
    fn test_resolve_roles_wildcard_suffix() {
        let mut cn_role_mapping = HashMap::new();
        cn_role_mapping.insert("*.admin".to_string(), "admin".to_string());

        let config = AuthenticationConfig {
            enabled: true,
            methods: vec![AuthenticationMethod::ClientCertificate],
            require_authentication: true,
            default_session_timeout_minutes: 60,
            api_keys: HashMap::new(),
            scram_users: HashMap::new(),
            jwt: JwtConfig {
                enabled: false,
                secret: "test".to_string(),
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
            mtls: MtlsConfig {
                enabled: false,
                ca_cert_path: None,
                require_client_cert: false,
                cn_role_mapping,
            },
            audit_fail_closed: false,
            oidc: None,
        };

        let service = UnifiedAuthService::new(config).unwrap_or_else(|_| unreachable!());

        let roles = service.resolve_roles_for_cn("cluster.admin");
        assert_eq!(roles, vec!["admin".to_string()]);

        let roles = service.resolve_roles_for_cn("db.admin");
        assert_eq!(roles, vec!["admin".to_string()]);

        // Does not match without the suffix
        let roles = service.resolve_roles_for_cn("cluster.reader");
        assert!(roles.is_empty());
    }

    #[test]
    fn test_mtls_enabled_without_ca_path_fails() {
        let config = AuthenticationConfig {
            enabled: true,
            methods: vec![AuthenticationMethod::ClientCertificate],
            require_authentication: true,
            default_session_timeout_minutes: 60,
            api_keys: HashMap::new(),
            scram_users: HashMap::new(),
            jwt: JwtConfig {
                enabled: false,
                secret: "test".to_string(),
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
            mtls: MtlsConfig {
                enabled: true,
                ca_cert_path: None, // Missing CA path
                require_client_cert: true,
                cn_role_mapping: HashMap::new(),
            },
            audit_fail_closed: false,
            oidc: None,
        };

        let result = UnifiedAuthService::new(config);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("ca_cert_path"));
    }

    #[tokio::test]
    async fn test_mtls_disabled_returns_failure() {
        let config = AuthenticationConfig {
            enabled: true,
            methods: vec![AuthenticationMethod::ClientCertificate],
            require_authentication: true,
            default_session_timeout_minutes: 60,
            api_keys: HashMap::new(),
            scram_users: HashMap::new(),
            jwt: JwtConfig {
                enabled: false,
                secret: "test".to_string(),
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
            mtls: MtlsConfig {
                enabled: false,
                ca_cert_path: None,
                require_client_cert: false,
                cn_role_mapping: HashMap::new(),
            },
            audit_fail_closed: false,
            oidc: None,
        };

        let service = UnifiedAuthService::new(config).unwrap_or_else(|_| unreachable!());
        let cert_data = ClientCertificateData {
            subject: "test-service".to_string(),
            issuer: "test-ca".to_string(),
            serial_number: "1234".to_string(),
            not_before: Utc::now() - chrono::Duration::hours(1),
            not_after: Utc::now() + chrono::Duration::hours(1),
            raw_cert_der: None,
        };

        let result = service
            .authenticate_client_certificate(&cert_data)
            .await
            .unwrap_or_else(|_| unreachable!());
        assert!(!result.success);
        assert!(
            result
                .error_message
                .as_deref()
                .unwrap_or("")
                .contains("not enabled")
        );
    }
}

// ===========================================================================
// TD-SEC-2 Slice A spec: the controls that existed but never fired.
// Each test names the contract clause it pins.
// ===========================================================================
#[cfg(test)]
mod arm_dead_controls_tests {
    use super::*;

    /// A registry key attributes to its PUBLIC key id — and the secret never
    /// appears in the attribution (it would otherwise be written to the audit
    /// record).
    #[test]
    fn registry_key_attributes_to_public_id_and_never_the_secret() {
        let data = AuthenticationData::ApiKey("pxk_abcd1234.SUPERSECRETVALUE".to_string());
        let principal = attempted_principal(&data);
        assert_eq!(principal, "pxk_abcd1234");
        assert!(
            !principal.contains("SUPERSECRETVALUE"),
            "secret material must never reach an audit record"
        );
    }

    /// A legacy/config key has no public component, so it fingerprints —
    /// stable (same key ⇒ same attribution, which is what makes counting
    /// possible) but not the secret itself.
    #[test]
    fn legacy_key_fingerprints_stably_without_revealing_it() {
        let data = AuthenticationData::ApiKey("legacy-secret-key".to_string());
        let a = attempted_principal(&data);
        let b = attempted_principal(&AuthenticationData::ApiKey("legacy-secret-key".to_string()));
        assert_eq!(a, b, "attribution must be stable to be countable");
        assert!(a.starts_with("fp:"));
        assert!(!a.contains("legacy-secret-key"));
    }

    /// THE defect this slice exists to fix: distinct credentials must produce
    /// distinct attributions. Previously every failure was `anonymous()`, so
    /// all failures shared one counter bucket and per-principal brute-force
    /// detection was impossible.
    #[test]
    fn distinct_credentials_do_not_share_one_bucket() {
        let a = attempted_principal(&AuthenticationData::ApiKey("pxk_aaa.s1".to_string()));
        let b = attempted_principal(&AuthenticationData::ApiKey("pxk_bbb.s2".to_string()));
        let c = attempted_principal(&AuthenticationData::ApiKey("legacy-one".to_string()));
        let d = attempted_principal(&AuthenticationData::ApiKey("legacy-two".to_string()));
        assert_ne!(a, b);
        assert_ne!(c, d);
        assert_ne!(a, c);
    }

    /// The failure budget is enforced per principal, and one principal
    /// exhausting it must not deny another (isolation).
    #[tokio::test]
    async fn failed_auth_budget_is_per_principal() {
        use crate::security::advanced_features::{RateLimitConfig, RateLimitingService};
        let limiter = RateLimitingService::new(RateLimitConfig {
            enabled: true,
            requests_per_minute_per_user: 1_000,
            requests_per_minute_per_tenant: 1_000,
            requests_per_minute_per_ip: 1_000,
            burst_allowance: 0,
            cleanup_interval_minutes: 60,
            failed_auth_per_minute_per_principal: 3,
        });

        for attempt in 1..=3 {
            assert!(
                limiter.record_failed_auth("pxk_victim").await.allowed,
                "attempt {attempt} is within budget"
            );
        }
        assert!(
            !limiter.record_failed_auth("pxk_victim").await.allowed,
            "the 4th failure exceeds a budget of 3"
        );
        assert!(
            limiter.record_failed_auth("pxk_bystander").await.allowed,
            "another principal must be unaffected — otherwise one attacker \
             locks out every user"
        );
    }
}
