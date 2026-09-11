//! Unified-port auth convergence + recall-tune dispatch — end-to-end.
//!
//! Pins three production invariants the convergence pass in
//! `feat(rest-auth): converge unified-port + multi-port auth onto a
//! single canonical surface` is supposed to deliver:
//!
//! 1. `build_router_for_unified` attaches `auth_middleware_unified`
//!    when `multi_server` passes a `SecurityCoordinator` (the
//!    `rest_auth_enabled = true` branch). Unauthenticated requests
//!    must 401 from the middleware — not from `require_recall_admin`'s
//!    surrogate-bypass branch.
//!
//! 2. A valid `Api-Key dev-key` header passes the middleware AND the
//!    operator-permission check (because `convert_api_key_to_unified`
//!    expands the wildcard `"*"` permission to the SystemAdmin-tier
//!    set). The recall-tune handler runs and reaches the IVF
//!    dispatch arm — proved by `algorithm:"ivf"` in the response.
//!
//! 3. The "wrong auth scheme" path stays rejected. `Authorization:
//!    Bearer <opaque>` is parsed as a JWT token by the auth
//!    middleware (the JWT extractor strips the `Bearer ` prefix);
//!    an opaque key under that scheme must 401.
//!
//! One ProximaDB boot per process (global WAL manifest is a set-once
//! singleton — see `feedback_one_proximadb_boot_per_test_process`).
//! All three observable behaviours are covered in a single test that
//! shares the server.

use std::net::TcpListener;
use std::time::Duration;

use proximadb::core::Config;
use proximadb::database::ProximaDB;
use proximadb::security::SecurityMode;
use proximadb::security::auth_service::{
    ApiKeyInfo, AuthenticationConfig, AuthenticationMethod, JwtConfig, MtlsConfig, SSOConfig,
};
use proximadb::security::rbac_service::RBACConfig;
use proximadb::security::security_coordinator::{ComplianceConfig, TlsConfig};
use proximadb::security::{AuditConfig, EncryptionConfig, KeyStoreConfig, SecurityConfig};
use tempfile::TempDir;
use tokio::time::sleep;

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind port 0");
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);
    port
}

fn auth_config_with_dev_key() -> SecurityConfig {
    let mut api_keys = std::collections::HashMap::new();
    api_keys.insert(
        "dev-key".to_string(),
        ApiKeyInfo {
            user_id: "devuser".to_string(),
            tenant_id: None,
            // The wildcard expansion is the load-bearing fixture
            // detail. Without it, the recall-tune endpoint would 403
            // even with a valid auth header, hiding the convergence
            // bug. With it, the integration test validates the full
            // chain: middleware injection → wildcard fan-out →
            // require_recall_admin gate.
            permissions: vec!["*".to_string()],
            roles: vec![],
            created_at: None,
            expires_at: None,
            rate_limit_per_minute: None,
            ip_restrictions: vec![],
        },
    );
    api_keys.insert(
        "read-only-key".to_string(),
        ApiKeyInfo {
            user_id: "reader".to_string(),
            tenant_id: None,
            permissions: vec!["read".to_string()],
            roles: vec![],
            created_at: None,
            expires_at: None,
            rate_limit_per_minute: None,
            ip_restrictions: vec![],
        },
    );

    SecurityConfig {
        enabled: true,
        mode: SecurityMode::Development,
        authentication: AuthenticationConfig {
            enabled: true,
            methods: vec![AuthenticationMethod::ApiKey],
            require_authentication: true,
            default_session_timeout_minutes: 60,
            api_keys,
            scram_users: std::collections::HashMap::new(),
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
        pgwire: proximadb::security::security_coordinator::PgwireSecurityConfig::default(),
    }
}

struct AuthTestServer {
    rest_port: u16,
    db: Option<ProximaDB>,
    _tmp_data: TempDir,
}

impl AuthTestServer {
    async fn start() -> anyhow::Result<Self> {
        let rest_port = free_port();
        let grpc_port = free_port();
        let pg_port = free_port();
        let tmp_data = TempDir::new()?;

        let mut config = Config::default();
        config.server.bind_address = "127.0.0.1".to_string();
        config.server.port = rest_port;
        config.server.data_dir = tmp_data.path().to_path_buf();
        // Unified-port mode is the path the convergence change
        // actually touched. Multi-port already worked.
        config.api.unified_mode = true;
        config.api.unified_port = rest_port;
        config.api.rest_port = rest_port;
        config.api.grpc_port = grpc_port;
        config.api.pg_port = Some(pg_port);
        config.storage.storage_locations = vec![proximadb::core::config::StorageLocation {
            url: format!("file://{}", tmp_data.path().display()),
            ..Default::default()
        }];
        // metadata_url defaults to `file://./metadata` — a path
        // relative to the cargo working directory, NOT the test's
        // tempdir. Without this override, every run shares the same
        // metadata catalog and the collection name collides between
        // runs ("create_collection returned no collection:
        // error_code=Some(COLLECTION_EXISTS)").
        config.storage.metadata_url = format!("file://{}/metadata", tmp_data.path().display());
        config.storage.wal_config.write_buffer_directory =
            format!("file://{}/wal", tmp_data.path().display());
        config.security = Some(auth_config_with_dev_key());

        let mut db = ProximaDB::new(config).await?;
        db.start().await?;

        // Health probe — auth is skip-listed by `should_skip_auth`
        // for `/health`, so this works even with auth enabled.
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(3))
            .no_proxy()
            .build()?;
        let health = format!("http://127.0.0.1:{rest_port}/health");
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        loop {
            match http.get(&health).send().await {
                Ok(r) if r.status().is_success() => break,
                _ => {
                    if std::time::Instant::now() > deadline {
                        anyhow::bail!("REST not ready on {rest_port} within 15s");
                    }
                    sleep(Duration::from_millis(100)).await;
                }
            }
        }
        // Extra settle so the auth middleware is fully wired by
        // the time we probe it.
        sleep(Duration::from_millis(200)).await;

        Ok(Self {
            rest_port,
            db: Some(db),
            _tmp_data: tmp_data,
        })
    }

    fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{}", self.rest_port, path)
    }
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

// Creates its IVF collection via the canonical `POST /api/v2/collections`
// (WS1 restored `index_configs` + `tags` parity to CreateCollectionV2Request),
// then proves auth-converged recall-tune dispatches to the IVF arm.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unified_port_auth_converged_recall_tune_e2e() {
    let server = AuthTestServer::start().await.expect("server start");

    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .no_proxy()
        .build()
        .expect("http client");

    // ── Phase 1: invariant (1) — no header → 401 from middleware ──
    //
    // Hits the v2 collections list endpoint which is auth-gated
    // by `auth_middleware_unified` at the router layer.
    let resp = http
        .get(server.url("/api/v2/collections"))
        .send()
        .await
        .expect("send no-auth");
    assert_eq!(
        resp.status().as_u16(),
        401,
        "auth middleware must reject unauthenticated requests on the converged unified port"
    );

    // ── Phase 2: invariant (3) — Bearer <opaque> → 401 ─────────
    //
    // Auth middleware strips `Bearer ` → JWT extractor. An opaque
    // key under the wrong scheme must NOT succeed.
    let resp = http
        .get(server.url("/api/v2/collections"))
        .header("Authorization", "Bearer dev-key")
        .send()
        .await
        .expect("send bearer");
    assert_eq!(
        resp.status().as_u16(),
        401,
        "Bearer-prefixed dev-key routes to the JWT path and must fail (it is not a JWT)"
    );

    // ── Phase 3: invariant (2) — Api-Key dev-key → wildcard ──
    //
    // List collections — wildcard "*" includes TenantRead so this
    // must succeed.
    let resp = http
        .get(server.url("/api/v2/collections"))
        .header("Authorization", "Api-Key dev-key")
        .send()
        .await
        .expect("send api-key");
    assert_eq!(
        resp.status().as_u16(),
        200,
        "Api-Key dev-key must pass the converged auth chain (middleware → wildcard expansion)"
    );

    // ── Phase 4: IVF collection create + recall-tune dispatch ──
    //
    // Drive an explicit IVF index_config so `active_algorithm_for`
    // resolves to "ivf" and recall-tune dispatches to the IVF arm.
    // Canonical v2 create with an explicit IVF index + recall tags (WS1 parity).
    let create_body = serde_json::json!({
        "name": "ivf_auth_e2e",
        "dimension": 32,
        "distance_metric": "cosine",
        "tags": ["recall_target:0.70", "target_vector_count:1000"],
        "index_configs": [{
            "index_name": "ivf_primary",
            "algorithm": "ivf",
            "ivf_config": {"n_lists": 100, "n_probe": 50}
        }]
    });
    let resp = http
        .post(server.url("/api/v2/collections"))
        .header("Authorization", "Api-Key dev-key")
        .json(&create_body)
        .send()
        .await
        .expect("send create");
    let status = resp.status();
    let body_text = resp.text().await.unwrap_or_default();
    assert_eq!(
        status.as_u16(),
        200,
        "v2 collection create with admin Api-Key must succeed; got {} body={}",
        status,
        body_text
    );
    let create_json: serde_json::Value =
        serde_json::from_str(&body_text).expect("create json parse");
    // v2 CreateCollectionV2Response echoes the name + a canonical collection_id
    // on success. ADR-031: collection_id is the numeric object_id (decimal u64),
    // not the collection name ("ivf_auth_e2e"). Assert the echoed name + numeric id.
    assert_eq!(
        create_json.get("name").and_then(|v| v.as_str()),
        Some("ivf_auth_e2e"),
        "v2 create response must echo name: {:?}",
        create_json
    );
    assert!(
        create_json
            .get("collection_id")
            .and_then(|v| v.as_str())
            .map(|s| s.parse::<u64>().is_ok())
            .unwrap_or(false),
        "v2 create response must include a numeric collection_id: {:?}",
        create_json
    );

    // recall-tune is the load-bearing operator endpoint the
    // convergence work targeted. It require_recall_admin → wildcard
    // fan-out grants this. The 200 status is the convergence
    // proof; the `algorithm:"ivf"` field is the P2.4 dispatch
    // proof.
    let resp = http
        .post(server.url("/api/v2/_diagnostics/collections/ivf_auth_e2e/recall-tune"))
        .header("Authorization", "Api-Key dev-key")
        .send()
        .await
        .expect("send recall-tune");
    assert_eq!(
        resp.status().as_u16(),
        200,
        "recall-tune must pass require_recall_admin under wildcard Api-Key; got {}",
        resp.status()
    );
    let tune_json: serde_json::Value = resp.json().await.expect("tune json");
    let report = tune_json.get("report").expect("response carries `report`");
    assert_eq!(
        report.get("algorithm").and_then(|v| v.as_str()),
        Some("ivf"),
        "recall-tune must dispatch to the IVF arm via active_algorithm_for; report={}",
        report
    );
    assert_eq!(
        report.get("wired").and_then(|v| v.as_bool()),
        Some(true),
        "recall-tune report must report wired=true for the IVF tag-driven collection"
    );

    // ── Phase 5: insufficient-permission Api-Key → 403 ─────────
    //
    // The read-only key authenticates (middleware passes) but
    // lacks SystemAdmin / ConfigureSystem → require_recall_admin
    // forbids. This pins the "wildcard is the only admin path"
    // invariant: removing the "*" fan-out would silently demote
    // the dev-key to this state.
    let resp = http
        .post(server.url("/api/v2/_diagnostics/collections/ivf_auth_e2e/recall-tune"))
        .header("Authorization", "Api-Key read-only-key")
        .send()
        .await
        .expect("send recall-tune read-only");
    assert_eq!(
        resp.status().as_u16(),
        403,
        "read-only Api-Key must be authenticated but forbidden on recall-tune"
    );
}
