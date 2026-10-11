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

struct AuthTestServer {
    rest_port: u16,
    grpc_port: u16,
    flight_port: u16,
    api_key: String,
    db: Option<ProximaDB>,
    _tmp: TempDir,
}

impl AuthTestServer {
    /// `security_enabled = false` boots with NO `[security]` section (the dev
    /// default). `true` boots posture B: coordinator present, authentication
    /// flag off, one API key configured.
    async fn start(security_enabled: bool) -> anyhow::Result<Self> {
        Self::start_with(security_enabled, |_| {}).await
    }

    async fn shutdown(mut self) -> anyhow::Result<()> {
        if let Some(mut db) = self.db.take() {
            db.shutdown().await?;
        }
        Ok(())
    }

    fn config(security_enabled: bool) -> anyhow::Result<(Config, TempDir, String)> {
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
        config.api.enable_pgwire = false;
        config.api.arrow_flight_port = free_port();
        config.storage.storage_locations = vec![proximadb::core::config::StorageLocation {
            url: format!("file://{}", tmp.path().display()),
            ..Default::default()
        }];
        config.storage.metadata_url = format!("file://{}/metadata", tmp.path().display());
        config.storage.wal_config.write_buffer_directory =
            format!("file://{}/wal", tmp.path().display());
        let api_key = uuid::Uuid::new_v4().to_string();

        if security_enabled {
            let mut api_keys = std::collections::HashMap::new();
            api_keys.insert(
                api_key.clone(),
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
                scram_users: std::collections::HashMap::new(),
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
                pgwire: PgwireSecurityConfig { auth: None },
            });
        }

        Ok((config, tmp, api_key))
    }

    async fn start_with(
        security_enabled: bool,
        configure: impl FnOnce(&mut Config),
    ) -> anyhow::Result<Self> {
        let (mut config, tmp, api_key) = Self::config(security_enabled)?;
        configure(&mut config);
        let rest_port = config.api.rest_port;
        let grpc_port = if config.api.unified_mode {
            config.api.unified_port
        } else {
            config.api.grpc_port
        };
        let flight_port = if config.api.unified_mode {
            config.api.unified_port
        } else {
            config.api.arrow_flight_port
        };
        let mut db = ProximaDB::new(config).await?;
        if let Err(error) = db.start().await {
            db.shutdown().await?;
            return Err(error);
        }

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
            grpc_port,
            flight_port,
            api_key,
            db: Some(db),
            _tmp: tmp,
        })
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn auth_failclosed_flight_wrapper_preserves_injected_coordinator() {
    use arrow_flight::{Criteria, flight_service_client::FlightServiceClient};
    use futures::TryStreamExt;
    use proximadb::network::arrow_ipc::{ArrowFlightServer, service::ProximaFlightService};
    use proximadb::network::multi_server::{BindTarget, ServiceProfile, SharedServices};
    use std::sync::Arc;

    let (config, _tmp, api_key) = AuthTestServer::config(true).expect("isolated config");
    let coordinator = Arc::new(
        proximadb::security::initialize_security(config.security.clone().expect("security"))
            .await
            .expect("coordinator"),
    );
    let (services, _) = SharedServices::new(
        None,
        &config.storage,
        None,
        Some(&config),
        ServiceProfile::Embedded,
    )
    .await
    .expect("shared services");
    let service = ProximaFlightService::from_services(
        services.record_ops.clone(),
        services.record_ops.clone(),
        services.vector_operations_service.clone(),
        services.collection_service.clone(),
        services.graph_service.clone(),
    )
    .with_security_coordinator(Some(coordinator));
    let address = ([127, 0, 0, 1], config.api.arrow_flight_port).into();
    // Deliberately do NOT repeat the service coordinator on the wrapper.
    let server = ArrowFlightServer::new(BindTarget::Tcp(address), service);
    let task = tokio::spawn(server.start());
    let endpoint =
        tonic::transport::Endpoint::from_shared(format!("http://{address}")).expect("endpoint");
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let channel = loop {
        match endpoint.connect().await {
            Ok(channel) => break channel,
            Err(error) if std::time::Instant::now() >= deadline => {
                task.abort();
                panic!("Flight not ready: {error}");
            }
            Err(_) => sleep(Duration::from_millis(25)).await,
        }
    };
    let mut client = FlightServiceClient::new(channel);
    for key in [None, Some("invalid"), Some(api_key.as_str())] {
        let mut request = tonic::Request::new(Criteria {
            expression: Default::default(),
        });
        if let Some(key) = key {
            request
                .metadata_mut()
                .insert("x-api-key", key.parse().expect("key"));
        }
        let result = client.list_flights(request).await;
        if key == Some(api_key.as_str()) {
            let mut stream = result.expect("valid credential").into_inner();
            while stream.try_next().await.expect("Flight stream").is_some() {}
        } else {
            assert!(matches!(
                result
                    .expect_err("missing/invalid credential must be rejected")
                    .code(),
                tonic::Code::Unauthenticated | tonic::Code::PermissionDenied
            ));
        }
    }
    task.abort();
    assert!(task.await.expect_err("cancelled listener").is_cancelled());
}

#[test]
fn auth_failclosed_posture_b_matrix() {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_stack_size(8 * 1024 * 1024)
        .enable_all()
        .build()
        .expect("test runtime")
        .block_on(auth_failclosed_posture_b_matrix_impl());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn auth_failclosed_grpc_and_flight_credentials_in_both_port_modes() {
    use arrow_flight::{Criteria, flight_service_client::FlightServiceClient};
    use futures::TryStreamExt;
    use proximadb_proto::proximadb_v2::{
        V2ListCollectionsRequest, proxima_record_service_client::ProximaRecordServiceClient,
    };

    for unified in [false, true] {
        let server = AuthTestServer::start_with(true, |config| {
            config.api.unified_mode = unified;
            config.api.unified_port = config.api.rest_port;
            config.api.internal_mux_port = Some(free_port());
        })
        .await
        .expect("transport fixture");
        let http = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("HTTP client");
        let collections_url = format!("http://127.0.0.1:{}/api/v2/collections", server.rest_port);
        let collection_name = format!("auth_control_{}", uuid::Uuid::new_v4().simple());
        let created = http
            .post(&collections_url)
            .header("Authorization", format!("Api-Key {}", server.api_key))
            .json(&serde_json::json!({"name": collection_name, "dimension": 4}))
            .send()
            .await
            .expect("create positive-control collection");
        assert!(
            created.status().is_success(),
            "collection setup: {}",
            created.text().await.expect("body")
        );
        let mut grpc =
            ProximaRecordServiceClient::connect(format!("http://127.0.0.1:{}", server.grpc_port))
                .await
                .expect("gRPC connect");
        let flight_channel = tonic::transport::Endpoint::from_shared(format!(
            "http://127.0.0.1:{}",
            server.flight_port
        ))
        .expect("Flight endpoint")
        .connect()
        .await
        .expect("Flight connect");
        let mut flight = FlightServiceClient::new(flight_channel);
        let invalid = uuid::Uuid::new_v4().to_string();
        for credential in [None, Some(invalid.as_str()), Some(server.api_key.as_str())] {
            let mut query = tonic::Request::new(V2ListCollectionsRequest {
                limit: None,
                offset: None,
            });
            let mut list = tonic::Request::new(Criteria {
                expression: Default::default(),
            });
            if let Some(key) = credential {
                let value = format!("Api-Key {key}")
                    .parse::<tonic::metadata::MetadataValue<tonic::metadata::Ascii>>()
                    .unwrap();
                query.metadata_mut().insert("authorization", value.clone());
                list.metadata_mut().insert("authorization", value);
            }
            let rest_result = match credential {
                Some(key) => http
                    .get(&collections_url)
                    .header("Authorization", format!("Api-Key {key}")),
                None => http.get(&collections_url),
            }
            .send()
            .await
            .expect("REST credentials control");
            let grpc_result = grpc.list_collections(query).await;
            let flight_result = flight.list_flights(list).await;
            if credential == Some(server.api_key.as_str()) {
                assert!(rest_result.status().is_success());
                assert!(
                    grpc_result
                        .expect("valid gRPC credential")
                        .into_inner()
                        .collections
                        .iter()
                        .any(|collection| collection
                            .config
                            .as_ref()
                            .is_some_and(|config| config.name == collection_name))
                );
                flight_result
                    .expect("valid Flight credential")
                    .into_inner()
                    .try_collect::<Vec<_>>()
                    .await
                    .expect("Flight stream completion");
            } else {
                assert!(matches!(
                    rest_result.status(),
                    reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN
                ));
                assert!(matches!(
                    grpc_result.expect_err("gRPC must deny").code(),
                    tonic::Code::Unauthenticated | tonic::Code::PermissionDenied
                ));
                assert!(matches!(
                    flight_result.expect_err("Flight must deny").code(),
                    tonic::Code::Unauthenticated | tonic::Code::PermissionDenied
                ));
            }
        }
        server.shutdown().await.expect("transport shutdown");
    }
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
        .header("Authorization", format!("Api-Key {}", server.api_key))
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
    dev_server.shutdown().await.expect("dev shutdown");
    server.shutdown().await.expect("authenticated shutdown");
}

#[tokio::test]
async fn auth_failclosed_refuses_unbound_optional_listeners_before_binding() {
    for mcp in [false, true] {
        let port = free_port();
        let (mut config, _tmp, _) = AuthTestServer::config(true).expect("isolated config");
        if mcp {
            config.api.mcp_port = Some(port);
        } else {
            config.api.enable_pgwire = true;
            config.api.pg_port = Some(port);
        }
        let ports = [
            config.api.rest_port,
            config.api.grpc_port,
            config.api.arrow_flight_port,
            port,
        ];
        let mut db = ProximaDB::new(config).await.expect("database construction");
        let error = db
            .start()
            .await
            .expect_err("unbound listener must not be admitted");
        assert!(
            error
                .to_string()
                .contains(if mcp { "MCP" } else { "pgwire" }),
            "{error:#}"
        );
        // Check BEFORE shutdown: cleanup must not hide a partial startup.
        for port in ports {
            assert!(
                tokio::net::TcpStream::connect(("127.0.0.1", port))
                    .await
                    .is_err(),
                "rejected configuration bound port {port} before validation"
            );
        }
        db.shutdown().await.expect("rejected database cleanup");
    }
}
