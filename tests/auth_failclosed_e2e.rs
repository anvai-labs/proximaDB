// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! Auth fail-closed e2e (TD-AUTH-FC).
//!
//! Posture B — `[security] enabled = true` with
//! `[security.authentication] enabled = false` — previously failed OPEN on
//! every network surface: the coordinator existed (RBAC/audit active) but no
//! auth layer was attached, so unauthenticated data-plane calls were SERVED.
//! After TD-AUTH-FC the coordinator presence (not the authentication flag)
//! decides, so posture B enforces credentials exactly like posture C.
//!
//! Matrix:
//! 1. Posture B + no credential  → REST data-plane REJECTED (401).
//! 2. Posture B + valid API key  → the same call SUCCEEDS.
//! 3. `/health` stays exempt     — liveness probes never authenticate.
//! 4. Security disabled entirely → unauthenticated calls still work
//!    (the dev default; unchanged by TD-AUTH-FC).

use std::net::TcpListener;
use std::time::Duration;

use proximadb::core::Config;
use proximadb::database::ProximaDB;
use proximadb::security::auth_service::{
    ApiKeyInfo, AuthenticationConfig, AuthenticationMethod, JwtConfig, MtlsConfig, SSOConfig,
};
use proximadb::security::rbac_service::RBACConfig;
use proximadb::security::security_coordinator::{
    ComplianceConfig, SecurityConfig, SecurityMode, TlsConfig,
};
use proximadb_security::AuditConfig;
use tempfile::TempDir;
use tokio::time::sleep;

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind port 0");
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);
    port
}

struct AuthTestServer {
    rest_port: u16,
    db: Option<ProximaDB>,
    _tmp: TempDir,
}

impl Drop for AuthTestServer {
    fn drop(&mut self) {
        if let Some(mut db) = self.db.take() {
            tokio::spawn(async move {
                let _ = db.shutdown().await;
            });
        }
    }
}

impl AuthTestServer {
    /// `security_enabled = false` boots with NO `[security]` section (the dev
    /// default). `true` boots posture B: coordinator present, authentication
    /// flag off, one API key configured.
    async fn start(security_enabled: bool) -> anyhow::Result<Self> {
        let rest_port = free_port();
        let grpc_port = free_port();
        let pg_port = free_port();
        let tmp = TempDir::new()?;

        let mut config = Config::default();
        config.server.bind_address = "127.0.0.1".to_string();
        config.server.port = rest_port;
        config.server.data_dir = tmp.path().to_path_buf();
        config.api.rest_port = rest_port;
        config.api.grpc_port = grpc_port;
        config.api.unified_mode = false;
        config.api.pg_port = Some(pg_port);

        if security_enabled {
            let mut api_keys = std::collections::HashMap::new();
            api_keys.insert(
                "td-auth-fc-test-key".to_string(),
                ApiKeyInfo {
                    user_id: "auth-fc-tester".to_string(),
                    tenant_id: Some("default-tenant".to_string()),
                    permissions: vec!["read".to_string(), "write".to_string()],
                    roles: Vec::new(),
                    created_at: None,
                    expires_at: None,
                    rate_limit_per_minute: None,
                    ip_restrictions: Vec::new(),
                },
            );
            let authentication = AuthenticationConfig {
                // THE posture-B crux: the authentication FLAG is off — before
                // TD-AUTH-FC this made every surface fail open.
                enabled: false,
                methods: vec![AuthenticationMethod::ApiKey],
                require_authentication: false,
                default_session_timeout_minutes: 30,
                api_keys,
                jwt: JwtConfig {
                    enabled: false,
                    secret: String::new(),
                    access_token_expiration_minutes: 15,
                    refresh_token_expiration_days: 7,
                    issuer: String::new(),
                    audience: String::new(),
                    algorithm: "HS256".to_string(),
                },
                sso: SSOConfig {
                    enabled: false,
                    providers: Vec::new(),
                    token_cache_ttl_minutes: 5,
                },
                mtls: MtlsConfig {
                    enabled: false,
                    ca_cert_path: None,
                    require_client_cert: false,
                    cn_role_mapping: std::collections::HashMap::new(),
                },
                audit_fail_closed: false,
                oidc: None,
            };
            config.security = Some(SecurityConfig {
                enabled: true,
                mode: SecurityMode::Development,
                authentication,
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
                    frameworks: Vec::new(),
                    data_residency: None,
                    encryption_at_rest: false,
                    encryption_in_transit: false,
                },
                encryption: Default::default(),
                key_store: Default::default(),
                tenant: Default::default(),
            });
        }

        let mut db = ProximaDB::new(config).await?;
        db.start().await?;

        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .no_proxy()
            .build()?;
        let health = format!("http://127.0.0.1:{rest_port}/health");
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        loop {
            match http.get(&health).send().await {
                Ok(r) if r.status().is_success() => break,
                _ if std::time::Instant::now() > deadline => anyhow::bail!("REST not ready"),
                _ => sleep(Duration::from_millis(100)).await,
            }
        }
        sleep(Duration::from_millis(200)).await;

        Ok(Self {
            rest_port,
            db: Some(db),
            _tmp: tmp,
        })
    }
}

#[test]
fn auth_failclosed_posture_b_matrix() {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("test runtime")
        .block_on(auth_failclosed_posture_b_matrix_impl());
}

async fn auth_failclosed_posture_b_matrix_impl() {
    // ── 1. Posture B: unauthenticated data-plane call REJECTED. ──
    let server = AuthTestServer::start(true)
        .await
        .expect("posture-B server start");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .no_proxy()
        .build()
        .unwrap();
    let base = format!("http://127.0.0.1:{}", server.rest_port);

    let status = http
        .get(format!("{base}/api/v2/collections"))
        .send()
        .await
        .expect("unauthenticated call")
        .status();
    assert!(
        status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN,
        "posture B must reject unauthenticated data-plane calls, got {status}"
    );

    // ── 2. The SAME call with the configured API key SUCCEEDS. ──
    let status = http
        .get(format!("{base}/api/v2/collections"))
        .header("Authorization", "Api-Key td-auth-fc-test-key")
        .send()
        .await
        .expect("authenticated call")
        .status();
    assert!(
        status.is_success(),
        "a valid API key must pass posture-B enforcement, got {status}"
    );

    // ── 3. /health stays exempt — liveness probes never authenticate. ──
    let status = http
        .get(format!("{base}/health"))
        .send()
        .await
        .expect("health probe")
        .status();
    assert!(
        status.is_success(),
        "health must stay unauthenticated-exempt, got {status}"
    );

    // ── 4. Security disabled entirely: unauthenticated calls still work. ──
    let dev_server = AuthTestServer::start(false)
        .await
        .expect("dev server start");
    let status = http
        .get(format!(
            "http://127.0.0.1:{}/api/v2/collections",
            dev_server.rest_port
        ))
        .send()
        .await
        .expect("dev data-plane call")
        .status();
    assert!(
        status.is_success(),
        "the dev default (no security) must stay open, got {status}"
    );
}
