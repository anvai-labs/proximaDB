// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! pgwire SCRAM-SHA-256 authentication e2e (TD-PGWIRE-AUTH-1).
//!
//! Drives the REAL client path — `tokio_postgres` natively implements
//! SCRAM-SHA-256 over a NoTls connection (it advertises the mechanism, and
//! selects plain SCRAM-SHA-256 when the server offers no `-PLUS`) — so these
//! tests exercise exactly what psql/JDBC do.
//!
//! Matrix:
//! 1. SCRAM required + correct password  → connect succeeds, query works.
//! 2. SCRAM required + wrong password    → rejected (28P01 class).
//! 3. SCRAM required + no password       → rejected.
//! 4. SCRAM required + unknown user      → rejected (same failure class — the
//!    mock-verifier path keeps shape/timing uniform).
//! 5. Default trust posture (no security) → credential-less connect works,
//!    byte-identical to pre-TD-PGWIRE-AUTH-1 behavior (guards the existing
//!    pgwire e2e suites).

use std::collections::HashMap;
use std::net::TcpListener;
use std::time::Duration;

use proximadb::core::Config;
use proximadb::database::ProximaDB;
use proximadb::security::auth_service::{
    AuthenticationConfig, AuthenticationMethod, JwtConfig, MtlsConfig, SSOConfig, ScramUserConfig,
};
use proximadb::security::rbac_service::RBACConfig;
use proximadb::security::security_coordinator::{
    ComplianceConfig, PgwireSecurityConfig, SecurityConfig, SecurityMode, TlsConfig,
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

struct ScramTestServer {
    pg_port: u16,
    rest_port: u16,
    db: Option<ProximaDB>,
    _tmp: TempDir,
}

impl Drop for ScramTestServer {
    fn drop(&mut self) {
        if let Some(mut db) = self.db.take() {
            tokio::spawn(async move {
                let _ = db.shutdown().await;
            });
        }
    }
}

impl ScramTestServer {
    /// Boot a server with the given pgwire posture and scram-users map
    /// (`None` auth ⇒ trust; `None` users ⇒ no security section at all —
    /// the trust default every existing pgwire e2e relies on).
    async fn start(
        pgwire_auth: Option<&str>,
        scram_users: Option<Vec<(&'static str, &'static str)>>,
    ) -> anyhow::Result<Self> {
        let pg_port = free_port();
        let rest_port = free_port();
        let grpc_port = free_port();
        let tmp = TempDir::new()?;

        let mut config = Config::default();
        config.server.bind_address = "127.0.0.1".to_string();
        config.server.port = rest_port;
        config.server.data_dir = tmp.path().to_path_buf();
        config.api.rest_port = rest_port;
        config.api.grpc_port = grpc_port;
        config.api.unified_mode = false;
        config.api.pg_port = Some(pg_port);

        if let Some(scram_users) = scram_users {
            // Typed config so the test owns the plaintext→verifier conversion
            // path (the same code the config-file boot runs).
            let mut users: HashMap<String, ScramUserConfig> = HashMap::new();
            for (username, password) in scram_users {
                users.insert(
                    username.to_string(),
                    ScramUserConfig {
                        password: Some(password.to_string()),
                        verifier: None,
                        // Bind the credential to the tenant the test client
                        // asserts via `dbname=proximadb` — a mismatch is (correctly)
                        // rejected by the tenant reconciliation.
                        tenant_id: Some("proximadb".to_string()),
                        roles: Vec::new(),
                    },
                );
            }
            let authentication = AuthenticationConfig {
                enabled: false,
                methods: vec![AuthenticationMethod::ApiKey],
                require_authentication: false,
                default_session_timeout_minutes: 30,
                api_keys: HashMap::new(),
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
                    cn_role_mapping: HashMap::new(),
                },
                audit_fail_closed: false,
                oidc: None,
                scram_users: users,
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
                pgwire: PgwireSecurityConfig {
                    auth: pgwire_auth.map(str::to_string),
                },
            });
        }

        let mut db = ProximaDB::new(config).await?;
        db.start().await?;

        // Health-wait (same shape as the other pgwire harnesses).
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .no_proxy()
            .build()?;
        let health = format!("http://127.0.0.1:{rest_port}/health");
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        loop {
            match http.get(&health).send().await {
                Ok(r) if r.status().is_success() => break,
                _ if std::time::Instant::now() > deadline => {
                    anyhow::bail!("REST not ready");
                }
                _ => sleep(Duration::from_millis(100)).await,
            }
        }
        sleep(Duration::from_millis(200)).await;

        Ok(Self {
            pg_port,
            rest_port,
            db: Some(db),
            _tmp: tmp,
        })
    }

    fn conn_str(&self, user: &str, password: Option<&str>) -> String {
        let mut s = format!(
            "host=127.0.0.1 port={} user={user} dbname=proximadb sslmode=disable",
            self.pg_port
        );
        if let Some(password) = password {
            s.push_str(&format!(" password={password}"));
        }
        s
    }
}

async fn connect(conn_str: &str) -> Result<(), tokio_postgres::Error> {
    let (client, conn) = tokio_postgres::connect(conn_str, tokio_postgres::NoTls).await?;
    let conn_task = tokio::spawn(async move {
        let _ = conn.await;
    });
    // A trivial statement proves the session is fully usable post-auth.
    let _ = client.simple_query("SELECT 1").await?;
    // Dropping the client closes the connection.
    drop(client);
    conn_task.abort();
    Ok(())
}

fn scram_users() -> Vec<(&'static str, &'static str)> {
    vec![("postgres", "s3cret")]
}

#[test]
fn pgwire_scram_auth_matrix() {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_stack_size(8 * 1024 * 1024)
        .enable_all()
        .build()
        .expect("multi-thread test runtime")
        .block_on(pgwire_scram_auth_matrix_impl());
}

async fn pgwire_scram_auth_matrix_impl() {
    // ── 1. Trust default (no security): credential-less connect works. ──
    let trust_server = ScramTestServer::start(None, None)
        .await
        .expect("trust server start");
    connect(&trust_server.conn_str("postgres", None))
        .await
        .expect("trust default must accept a credential-less connection");

    // ── 2. SCRAM required + correct password → works. ──
    let scram_server = ScramTestServer::start(Some("password"), Some(scram_users()))
        .await
        .expect("scram server start");
    connect(&scram_server.conn_str("postgres", Some("s3cret")))
        .await
        .expect("SCRAM with the correct password must authenticate");

    // ── 3. SCRAM required + wrong password → rejected. ──
    let err = connect(&scram_server.conn_str("postgres", Some("wrong")))
        .await
        .expect_err("wrong password must be rejected");
    let db = err
        .as_db_error()
        .expect("wrong password must surface a db error");
    assert_eq!(
        db.code().code(),
        "28P01",
        "wrong password must be SQLSTATE 28P01, got: {db}"
    );

    // ── 4. SCRAM required + no credential → rejected. ──
    // Rejected client-side (no password to answer the SASL challenge with) or
    // server-side (28P01) — either way the connection MUST NOT open.
    connect(&scram_server.conn_str("postgres", None))
        .await
        .expect_err("credential-less connect must be rejected when SCRAM is required");

    // ── 5. SCRAM required + unknown user → same rejection class. ──
    let err = connect(&scram_server.conn_str("nobody", Some("s3cret")))
        .await
        .expect_err("unknown user must be rejected");
    let db = err
        .as_db_error()
        .expect("unknown user must surface a db error");
    assert_eq!(
        db.code().code(),
        "28P01",
        "unknown-user failure must be SQLSTATE 28P01 (indistinguishable from a wrong password), got: {db}"
    );
    assert_eq!(
        db.message(),
        "password authentication failed",
        "unknown-user message must match the wrong-password message (anti-enumeration)"
    );

    // The env/config ladder (env > config > forced; downgrade warn-ignored) is
    // covered by the pure `PgwireAuthMode::resolve` unit tests in
    // protocol_tests.rs; mutating process env here would race other tests.
}
