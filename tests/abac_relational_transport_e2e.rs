// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Live cross-surface ABAC provisioning ratchet (TD-ABAC-10b / TD-ABAC-11).
//!
//! This is deliberately a transport test, not another component test. It boots
//! one real server and crosses the production surfaces involved in the operator
//! journey:
//!
//! ```text
//! pgwire setup + trust-auth SELECT ----+
//!                                      v
//! authenticated gRPC ExecuteQuery -> relational DML scan
//! authenticated REST /api/v2/sql ----> shared typed SQL port
//! authenticated REST record reads ---> shared record ABAC seam
//! authenticated gRPC record reads ---> shared record ABAC seam
//! authenticated Flight `DoGet` ------> shared record ABAC seam
//!                                      ^
//! authenticated REST policy control -> shared live ABAC stores
//! ```
//!
//! The test proves deny-before-provision, hot permit without reconnect/restart,
//! isolation of an unbound subject, and hot revoke over trust-auth pgwire and
//! the credential-authenticated REST, gRPC, and Flight transports. It also
//! discovers the table's stable object id through `xcatalog.tables`; policy
//! tooling must never guess an allocator result or scrape catalog persistence.

#![cfg(feature = "abac-policy")]

use std::collections::HashMap;
use std::net::TcpListener;
use std::time::Duration;

use arrow_array::{Array, RecordBatch, StringArray};
use arrow_flight::Ticket;
use arrow_flight::decode::FlightRecordBatchStream;
use arrow_flight::error::FlightError;
use arrow_flight::flight_service_client::FlightServiceClient;
use futures::TryStreamExt;
use proximadb::core::Config;
use proximadb::database::ProximaDB;
use proximadb::proto::proximadb_v2::proxima_record_service_client::ProximaRecordServiceClient;
use proximadb::proto::proximadb_v2::{
    GetRecordRequest, TypedSearchRequest, V2QueryRequest, V2QueryResponse,
};
use proximadb::security::SecurityMode;
use proximadb::security::auth_service::{
    ApiKeyInfo, AuthenticationConfig, AuthenticationMethod, JwtConfig, MtlsConfig, SSOConfig,
};
use proximadb::security::rbac_service::RBACConfig;
use proximadb::security::security_coordinator::{ComplianceConfig, TlsConfig};
use proximadb::security::{AuditConfig, EncryptionConfig, KeyStoreConfig, SecurityConfig};
use reqwest::{Client as HttpClient, StatusCode};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::time::sleep;
use tokio_postgres::{Client as PgClient, SimpleQueryMessage};

// The composition root mints this catalog tenant before listeners start, so the
// pgwire handshake can stamp its stable id once and retain it for the session.
const TENANT: &str = proximadb_tenant::DEFAULT_TENANT;
const OPERATOR_KEY: &str = "abac-operator-key";
const ALICE_KEY: &str = "abac-alice-key";
const BOB_KEY: &str = "abac-bob-key";

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind port 0");
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);
    port
}

fn transport_security_config() -> SecurityConfig {
    let mut api_keys = HashMap::new();
    for (key, user_id) in [
        (OPERATOR_KEY, "abac-operator"),
        (ALICE_KEY, "alice"),
        (BOB_KEY, "bob"),
    ] {
        api_keys.insert(
            key.to_string(),
            ApiKeyInfo {
                user_id: user_id.to_string(),
                tenant_id: Some(TENANT.to_string()),
                permissions: vec!["*".to_string()],
                roles: vec![],
                created_at: None,
                expires_at: None,
                rate_limit_per_minute: None,
                ip_restrictions: vec![],
            },
        );
    }
    SecurityConfig {
        enabled: true,
        mode: SecurityMode::Development,
        authentication: AuthenticationConfig {
            enabled: true,
            methods: vec![AuthenticationMethod::ApiKey],
            require_authentication: true,
            default_session_timeout_minutes: 60,
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
        pgwire: crate::security::security_coordinator::PgwireSecurityConfig::default(),
    }
}

struct LiveServer {
    rest_port: u16,
    grpc_port: u16,
    flight_port: u16,
    pg_port: u16,
    db: Option<ProximaDB>,
    _tmp: TempDir,
}

impl LiveServer {
    async fn start() -> anyhow::Result<Self> {
        let rest_port = free_port();
        let grpc_port = free_port();
        let flight_port = free_port();
        let pg_port = free_port();
        let tmp = TempDir::new()?;
        let mut config = Config::default();
        config.server.bind_address = "127.0.0.1".to_string();
        config.server.port = rest_port;
        config.server.data_dir = tmp.path().to_path_buf();
        config.api.unified_mode = false;
        config.api.rest_port = rest_port;
        config.api.grpc_port = grpc_port;
        config.api.arrow_flight_port = flight_port;
        config.api.pg_port = Some(pg_port);
        config.storage.storage_locations = vec![proximadb::core::config::StorageLocation {
            url: format!("file://{}", tmp.path().display()),
            ..Default::default()
        }];
        config.storage.metadata_url = format!("file://{}/metadata", tmp.path().display());
        config.storage.wal_config.write_buffer_directory =
            format!("file://{}/wal", tmp.path().display());
        config.security = Some(transport_security_config());

        let mut db = ProximaDB::new(config).await?;
        db.start().await?;
        let http = HttpClient::builder()
            .timeout(Duration::from_secs(3))
            .no_proxy()
            .build()?;
        let health = format!("http://127.0.0.1:{rest_port}/health");
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        loop {
            match http.get(&health).send().await {
                Ok(response) if response.status().is_success() => break,
                _ if std::time::Instant::now() > deadline => {
                    anyhow::bail!("REST server did not become ready")
                }
                _ => sleep(Duration::from_millis(100)).await,
            }
        }
        sleep(Duration::from_millis(200)).await;
        Ok(Self {
            rest_port,
            grpc_port,
            flight_port,
            pg_port,
            db: Some(db),
            _tmp: tmp,
        })
    }

    fn pg_conn_str(&self, subject: &str) -> String {
        format!(
            "host=127.0.0.1 port={} user={subject} dbname={TENANT} sslmode=disable",
            self.pg_port
        )
    }

    fn admin_url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{path}", self.rest_port)
    }

    fn grpc_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.grpc_port)
    }

    fn flight_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.flight_port)
    }
}

impl Drop for LiveServer {
    fn drop(&mut self) {
        if let Some(mut db) = self.db.take() {
            tokio::spawn(async move {
                let _ = db.shutdown().await;
            });
        }
    }
}

async fn connect(server: &LiveServer, subject: &str) -> PgClient {
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    let (client, connection) = loop {
        match tokio_postgres::connect(&server.pg_conn_str(subject), tokio_postgres::NoTls).await {
            Ok(connection) => break connection,
            Err(error) if std::time::Instant::now() < deadline => {
                sleep(Duration::from_millis(100)).await;
                tracing::debug!(%error, %subject, "waiting for pgwire listener");
            }
            Err(error) => panic!("connect {subject}: {error}"),
        }
    };
    tokio::spawn(async move {
        if let Err(error) = connection.await {
            eprintln!("pgwire connection error: {error}");
        }
    });
    client
}

async fn exec(client: &PgClient, sql: &str) {
    client
        .simple_query(sql)
        .await
        .unwrap_or_else(|error| panic!("execute `{sql}`: {error}"));
}

async fn scalar(client: &PgClient, sql: &str) -> String {
    for message in client
        .simple_query(sql)
        .await
        .unwrap_or_else(|error| panic!("query `{sql}`: {error}"))
    {
        if let SimpleQueryMessage::Row(row) = message {
            return row.get(0).unwrap_or("NULL").to_string();
        }
    }
    panic!("query `{sql}` returned no row")
}

async fn table_object_id(client: &PgClient, table: &str) -> u64 {
    let sql = format!("SELECT * FROM xcatalog.tables WHERE table_name = '{table}'");
    // `resolve_table_scoped` avoids duplicating the tenant when the default
    // namespace already equals it; introspection renders that namespace as the
    // PostgreSQL-compatible `public` alias.
    let expected_namespace = "public";
    for message in client
        .simple_query(&sql)
        .await
        .unwrap_or_else(|error| panic!("catalog query `{sql}`: {error}"))
    {
        if let SimpleQueryMessage::Row(row) = message
            && row.get(1) == Some(expected_namespace)
        {
            return row
                .get(9)
                .expect("xcatalog.tables object_id column")
                .parse()
                .expect("numeric stable table object id");
        }
    }
    panic!("catalog query did not return tenant-scoped table {expected_namespace}.{table}")
}

async fn assert_admin_success(response: reqwest::Response, operation: &str) -> Value {
    let status = response.status();
    let body = response.text().await.expect("admin response body");
    assert!(
        status.is_success(),
        "{operation} failed with {status}: {body}"
    );
    if body.is_empty() {
        Value::Null
    } else {
        serde_json::from_str(&body)
            .unwrap_or_else(|error| panic!("{operation} returned invalid JSON `{body}`: {error}"))
    }
}

type GrpcClient = ProximaRecordServiceClient<tonic::transport::Channel>;

async fn connect_grpc(server: &LiveServer) -> GrpcClient {
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    loop {
        match ProximaRecordServiceClient::connect(server.grpc_url()).await {
            Ok(client) => return client,
            Err(error) if std::time::Instant::now() > deadline => {
                panic!("gRPC server did not become ready: {error}")
            }
            Err(_) => sleep(Duration::from_millis(100)).await,
        }
    }
}

fn authenticated_grpc_request<T>(body: T, api_key: &str) -> tonic::Request<T> {
    let mut request = tonic::Request::new(body);
    request.metadata_mut().insert(
        "authorization",
        format!("Api-Key {api_key}")
            .parse()
            .expect("authorization metadata"),
    );
    request
        .metadata_mut()
        .insert("x-tenant-id", TENANT.parse().expect("tenant metadata"));
    request
}

async fn grpc_query(client: &mut GrpcClient, api_key: &str, sql: &str) -> V2QueryResponse {
    let request = authenticated_grpc_request(
        V2QueryRequest {
            query: sql.to_string(),
            collection_id: String::new(),
            limit: None,
            offset: None,
        },
        api_key,
    );
    client
        .execute_query(request)
        .await
        .unwrap_or_else(|error| panic!("gRPC ExecuteQuery `{sql}`: {error}"))
        .into_inner()
}

async fn grpc_row_count(client: &mut GrpcClient, api_key: &str, sql: &str) -> usize {
    grpc_query(client, api_key, sql).await.rows.len()
}

async fn rest_sql_row_count(
    client: &HttpClient,
    server: &LiveServer,
    api_key: &str,
    sql: &str,
) -> usize {
    let response = client
        .post(server.admin_url("/api/v2/sql"))
        .header(reqwest::header::AUTHORIZATION, format!("Api-Key {api_key}"))
        .json(&json!({"query": sql}))
        .send()
        .await
        .unwrap_or_else(|error| panic!("REST SQL `{sql}`: {error}"));
    let status = response.status();
    let body: Value = response
        .json()
        .await
        .unwrap_or_else(|error| panic!("REST SQL `{sql}` returned invalid JSON: {error}"));
    assert!(
        status.is_success(),
        "REST SQL `{sql}` failed with {status}: {body}"
    );
    body["rows"]
        .as_array()
        .unwrap_or_else(|| panic!("REST SQL response has no rows array: {body}"))
        .len()
}

async fn rest_record_search_ids(
    client: &HttpClient,
    server: &LiveServer,
    api_key: &str,
    collection: &str,
) -> Vec<String> {
    let response = client
        .post(server.admin_url(&format!("/api/v2/collections/{collection}/search")))
        .header(reqwest::header::AUTHORIZATION, format!("Api-Key {api_key}"))
        .json(&json!({
            "vector": [1.0, 0.0, 0.0, 0.0],
            "top_k": 10
        }))
        .send()
        .await
        .unwrap_or_else(|error| panic!("REST record search: {error}"));
    let status = response.status();
    let body: Value = response.json().await.expect("REST search JSON body");
    assert!(
        status.is_success(),
        "REST record search failed with {status}: {body}"
    );
    body["results"]
        .as_array()
        .unwrap_or_else(|| panic!("REST search response has no results: {body}"))
        .iter()
        .map(|result| {
            result["id"]
                .as_str()
                .unwrap_or_else(|| panic!("REST search result has no id: {result}"))
                .to_string()
        })
        .collect()
}

async fn rest_record_get_status(
    client: &HttpClient,
    server: &LiveServer,
    api_key: &str,
    collection: &str,
    record_id: &str,
) -> StatusCode {
    client
        .get(server.admin_url(&format!(
            "/api/v2/collections/{collection}/records/{record_id}?include_vector=false"
        )))
        .header(reqwest::header::AUTHORIZATION, format!("Api-Key {api_key}"))
        .send()
        .await
        .unwrap_or_else(|error| panic!("REST record get: {error}"))
        .status()
}

async fn grpc_record_search_ids(
    client: &mut GrpcClient,
    api_key: &str,
    collection: &str,
) -> Vec<String> {
    let request = authenticated_grpc_request(
        TypedSearchRequest {
            collection_id: collection.to_string(),
            query_vector: vec![1.0, 0.0, 0.0, 0.0],
            top_k: 10,
            ..Default::default()
        },
        api_key,
    );
    client
        .search(request)
        .await
        .unwrap_or_else(|error| panic!("gRPC record search: {error}"))
        .into_inner()
        .results
        .into_iter()
        .map(|result| result.id)
        .collect()
}

async fn grpc_record_stream_ids(
    client: &mut GrpcClient,
    api_key: &str,
    collection: &str,
) -> Vec<String> {
    let request = authenticated_grpc_request(
        TypedSearchRequest {
            collection_id: collection.to_string(),
            query_vector: vec![1.0, 0.0, 0.0, 0.0],
            top_k: 10,
            ..Default::default()
        },
        api_key,
    );
    let mut stream = client
        .search_stream(request)
        .await
        .unwrap_or_else(|error| panic!("gRPC record search stream: {error}"))
        .into_inner();
    let mut ids = Vec::new();
    while let Some(result) = stream
        .message()
        .await
        .unwrap_or_else(|error| panic!("gRPC record search stream item: {error}"))
    {
        ids.push(result.id);
    }
    ids
}

async fn grpc_record_found(
    client: &mut GrpcClient,
    api_key: &str,
    collection: &str,
    record_id: &str,
) -> bool {
    let request = authenticated_grpc_request(
        GetRecordRequest {
            collection_id: collection.to_string(),
            id: record_id.to_string(),
            include_vector: false,
        },
        api_key,
    );
    client
        .get_record(request)
        .await
        .unwrap_or_else(|error| panic!("gRPC record get: {error}"))
        .into_inner()
        .found
}

async fn connect_flight(server: &LiveServer) -> FlightServiceClient<tonic::transport::Channel> {
    let channel = tonic::transport::Endpoint::from_shared(server.flight_url())
        .expect("Flight endpoint")
        .connect()
        .await
        .expect("Flight channel");
    FlightServiceClient::new(channel)
}

async fn flight_record_search_ids(
    client: &mut FlightServiceClient<tonic::transport::Channel>,
    api_key: &str,
    collection: &str,
) -> Vec<String> {
    let ticket = Ticket {
        ticket: serde_json::to_vec(&json!({
            "type": "vector_search",
            "collection_id": collection,
            "query_vector": [1.0, 0.0, 0.0, 0.0],
            "top_k": 10,
            "include_vector": false
        }))
        .expect("serialize Flight search ticket")
        .into(),
    };
    let response = client
        .do_get(authenticated_grpc_request(ticket, api_key))
        .await
        .unwrap_or_else(|error| panic!("Flight record search: {error}"))
        .into_inner();
    let stream = FlightRecordBatchStream::new_from_flight_data(
        response.map_err(|status| FlightError::Tonic(Box::new(status))),
    );
    let batches: Vec<RecordBatch> = stream
        .try_collect()
        .await
        .unwrap_or_else(|error| panic!("decode Flight record search: {error}"));

    let mut ids = Vec::new();
    for batch in batches {
        let id_column = batch
            .column_by_name("id")
            .expect("Flight search id column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("Flight search id column must be Utf8");
        ids.extend((0..id_column.len()).map(|index| id_column.value(index).to_string()));
    }
    ids
}

struct LiveRecordFixture {
    collection: String,
    eng_id: &'static str,
    hr_id: &'static str,
}

async fn create_live_record_fixture(
    client: &HttpClient,
    server: &LiveServer,
    name_prefix: &str,
) -> LiveRecordFixture {
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let fixture = LiveRecordFixture {
        collection: format!("{name_prefix}_{suffix}"),
        eng_id: "record-eng",
        hr_id: "record-hr",
    };
    let operator_auth = format!("Api-Key {OPERATOR_KEY}");
    let response = client
        .post(server.admin_url("/api/v2/collections"))
        .header(reqwest::header::AUTHORIZATION, &operator_auth)
        .json(&json!({
            "name": fixture.collection,
            "dimension": 4,
            "engine": "sst",
            "enable_proxima_record": true
        }))
        .send()
        .await
        .expect("create record collection");
    assert_admin_success(response, "create record collection").await;

    let response = client
        .post(server.admin_url(&format!(
            "/api/v2/collections/{}/records/batch",
            fixture.collection
        )))
        .header(reqwest::header::AUTHORIZATION, &operator_auth)
        .json(&json!({
            "records": [
                {
                    "id": fixture.eng_id,
                    "vector": [1.0, 0.0, 0.0, 0.0],
                    "props": {"dept": "eng"}
                },
                {
                    "id": fixture.hr_id,
                    "vector": [0.99, 0.01, 0.0, 0.0],
                    "props": {"dept": "hr"}
                }
            ],
            "validate_schema": false
        }))
        .send()
        .await
        .expect("insert record fixtures");
    let inserted = assert_admin_success(response, "insert record fixtures").await;
    assert_eq!(inserted["inserted_count"], 2);
    fixture
}

async fn provision_live_record_policy(
    client: &HttpClient,
    server: &LiveServer,
    predicate_object_id: u64,
) -> String {
    let operator_auth = format!("Api-Key {OPERATOR_KEY}");
    let response = client
        .post(server.admin_url("/api/v2/abac/attribute-bindings"))
        .header(reqwest::header::AUTHORIZATION, &operator_auth)
        .json(&json!({
            "subject_id": "alice",
            "tenant": TENANT,
            "attrs": {"dept": {"Str": "eng"}}
        }))
        .send()
        .await
        .expect("provision alice attributes");
    let binding = assert_admin_success(response, "provision alice attributes").await;

    let response = client
        .put(server.admin_url(&format!(
            "/api/v2/abac/predicate-objects/{predicate_object_id}"
        )))
        .header(reqwest::header::AUTHORIZATION, &operator_auth)
        .json(&json!({
            "Comparison": {
                "field": "dept",
                "operator": "Equals",
                "value": "eng"
            }
        }))
        .send()
        .await
        .expect("provision row predicate");
    assert_admin_success(response, "provision row predicate").await;

    let policy_path = format!(
        "/api/v2/abac/policy-bindings/{TENANT}/{}",
        predicate_object_id + 1
    );
    let response = client
        .put(server.admin_url(&policy_path))
        .header(reqwest::header::AUTHORIZATION, &operator_auth)
        .json(&json!({
            "scope": {"Namespace": 0},
            "effect": "Permit",
            "predicate_ref": predicate_object_id
        }))
        .send()
        .await
        .expect("provision record permit");
    let policy = assert_admin_success(response, "provision record permit").await;
    assert_eq!(
        policy["tenant_stable_id"].as_u64(),
        binding["tenant_stable_id"].as_u64()
    );
    policy_path
}

/// Same shape as [`provision_live_record_policy`], except the stored predicate
/// compares against `$subject.dept` instead of the literal `"eng"`.
///
/// ADR-090 L2.1: the placeholder must resolve to alice's SERVER-RESOLVED `dept`
/// attribute at compile time, so the visible rows are identical to the literal
/// case. The unit tests in `proximadb-abac` prove the substitution; this proves
/// it survives the real provisioning API and the real read path — an attribute
/// stored as a claim rather than server-resolved would deny here, and no unit
/// test could tell us that.
async fn provision_live_subject_parameterized_policy(
    client: &HttpClient,
    server: &LiveServer,
    predicate_object_id: u64,
) -> String {
    let operator_auth = format!("Api-Key {OPERATOR_KEY}");
    let response = client
        .post(server.admin_url("/api/v2/abac/attribute-bindings"))
        .header(reqwest::header::AUTHORIZATION, &operator_auth)
        .json(&json!({
            "subject_id": "alice",
            "tenant": TENANT,
            "attrs": {"dept": {"Str": "eng"}}
        }))
        .send()
        .await
        .expect("provision alice attributes");
    assert_admin_success(response, "provision alice attributes").await;

    let response = client
        .put(server.admin_url(&format!(
            "/api/v2/abac/predicate-objects/{predicate_object_id}"
        )))
        .header(reqwest::header::AUTHORIZATION, &operator_auth)
        .json(&json!({
            "Comparison": {
                "field": "dept",
                "operator": "Equals",
                "value": "$subject.dept"
            }
        }))
        .send()
        .await
        .expect("provision subject-parameterized predicate");
    assert_admin_success(response, "provision subject-parameterized predicate").await;

    let policy_path = format!(
        "/api/v2/abac/policy-bindings/{TENANT}/{}",
        predicate_object_id + 1
    );
    let response = client
        .put(server.admin_url(&policy_path))
        .header(reqwest::header::AUTHORIZATION, &operator_auth)
        .json(&json!({
            "scope": {"Namespace": 0},
            "effect": "Permit",
            "predicate_ref": predicate_object_id
        }))
        .send()
        .await
        .expect("provision record permit");
    assert_admin_success(response, "provision record permit").await;
    policy_path
}

/// ADR-090 L2.1 end to end: a `$subject.<attr>` predicate selects exactly the
/// rows the equivalent literal predicate selects, through the real REST
/// provisioning API and the real read path.
///
/// The negative half matters as much as the positive: `hr_id` must stay hidden.
/// A substitution that silently failed would compare `dept` against the literal
/// string `"$subject.dept"`, match nothing, and hide BOTH rows — which reads as
/// "the policy is just restrictive" unless the admitted row is asserted too.
#[test]
fn rest_records_follow_a_subject_parameterized_predicate() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_stack_size(8 * 1024 * 1024)
        .enable_all()
        .build()
        .expect("tokio runtime");
    runtime.block_on(rest_records_follow_a_subject_parameterized_predicate_inner());
}

async fn rest_records_follow_a_subject_parameterized_predicate_inner() {
    let server = LiveServer::start().await.expect("start live server");
    let http = HttpClient::builder()
        .timeout(Duration::from_secs(5))
        .no_proxy()
        .build()
        .expect("HTTP client");
    let fixture = create_live_record_fixture(&http, &server, "abac_subject_param").await;

    let policy_path =
        provision_live_subject_parameterized_policy(&http, &server, 7_000_000_000).await;

    assert_eq!(
        rest_record_search_ids(&http, &server, ALICE_KEY, &fixture.collection).await,
        vec![fixture.eng_id.to_string()],
        "`$subject.dept` must resolve to alice's server-resolved dept and admit her row"
    );
    assert_eq!(
        rest_record_get_status(
            &http,
            &server,
            ALICE_KEY,
            &fixture.collection,
            fixture.hr_id
        )
        .await,
        StatusCode::NOT_FOUND,
        "the substituted predicate must still exclude a row from another dept"
    );

    revoke_live_policy(&http, &server, &policy_path).await;
    assert!(
        rest_record_search_ids(&http, &server, ALICE_KEY, &fixture.collection)
            .await
            .is_empty(),
        "a subject-parameterized policy must observe the hot revoke like any other"
    );
}

async fn revoke_live_policy(client: &HttpClient, server: &LiveServer, policy_path: &str) {
    let response = client
        .delete(server.admin_url(policy_path))
        .header(
            reqwest::header::AUTHORIZATION,
            format!("Api-Key {OPERATOR_KEY}"),
        )
        .send()
        .await
        .expect("revoke record permit");
    assert_eq!(response.status(), StatusCode::NO_CONTENT);
}

/// The root crate's debug-build server future exceeds Tokio's 2 MiB default
/// worker stack in several existing pgwire E2Es. Use the repository's standard
/// 8 MiB integration-test runtime so the test measures the transport contract,
/// not debug-frame size (see `tpch_pgwire_e2e`).
#[test]
fn pgwire_reads_follow_live_rest_policy_provision_and_revoke() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_stack_size(8 * 1024 * 1024)
        .enable_all()
        .build()
        .expect("tokio runtime");
    runtime.block_on(pgwire_reads_follow_live_rest_policy_provision_and_revoke_inner());
}

async fn pgwire_reads_follow_live_rest_policy_provision_and_revoke_inner() {
    let server = LiveServer::start().await.expect("start live server");
    let alice = connect(&server, "alice").await;
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let table = format!("abac_live_{suffix}");
    exec(
        &alice,
        &format!("CREATE TABLE {table} (id INT PRIMARY KEY, value INT)"),
    )
    .await;
    exec(&alice, &format!("INSERT INTO {table} VALUES (1, 10)")).await;
    exec(&alice, &format!("INSERT INTO {table} VALUES (2, 20)")).await;

    let object_id = table_object_id(&alice, &table).await;
    let table_scope = u32::try_from(object_id).expect("ABAC table scope is u32");

    // Empty durable stores are deny-by-default. The request carries all four
    // identity fields through real pgwire, but neither subject membership nor a
    // table permit exists yet.
    assert_eq!(
        scalar(&alice, &format!("SELECT COUNT(*) FROM {table}")).await,
        "0",
        "unprovisioned governed reads must fail closed"
    );

    let http = HttpClient::builder()
        .timeout(Duration::from_secs(5))
        .no_proxy()
        .build()
        .expect("HTTP client");
    let auth = format!("Api-Key {OPERATOR_KEY}");

    let response = http
        .post(server.admin_url("/api/v2/abac/attribute-bindings"))
        .header(reqwest::header::AUTHORIZATION, &auth)
        .json(&json!({
            "subject_id": "alice",
            "tenant": TENANT,
            "attrs": {}
        }))
        .send()
        .await
        .expect("provision alice binding");
    let binding = assert_admin_success(response, "provision attribute binding").await;
    assert_eq!(binding["subject_id"], "alice");

    // The URL object id identifies this policy-binding object; the scope carries
    // the independently discovered table object id.
    let policy_object_id = object_id + 1_000_000_000;
    let policy_path = format!("/api/v2/abac/policy-bindings/{TENANT}/{policy_object_id}");
    let response = http
        .put(server.admin_url(&policy_path))
        .header(reqwest::header::AUTHORIZATION, &auth)
        .json(&json!({
            "scope": {"Table": table_scope},
            "effect": "Permit"
        }))
        .send()
        .await
        .expect("provision table permit");
    let policy = assert_admin_success(response, "provision table permit").await;
    assert_eq!(
        policy["tenant_stable_id"].as_u64(),
        binding["tenant_stable_id"].as_u64()
    );

    assert_eq!(
        scalar(&alice, &format!("SELECT COUNT(*) FROM {table}")).await,
        "2",
        "policy mutation must be hot-visible to the existing pgwire session"
    );

    let bob = connect(&server, "bob").await;
    assert_eq!(
        scalar(&bob, &format!("SELECT COUNT(*) FROM {table}")).await,
        "0",
        "a table permit must not admit a subject without an authority binding"
    );

    let response = http
        .delete(server.admin_url(&policy_path))
        .header(reqwest::header::AUTHORIZATION, &auth)
        .send()
        .await
        .expect("revoke table permit");
    assert_eq!(
        response.status(),
        StatusCode::NO_CONTENT,
        "policy revoke must succeed"
    );
    assert_eq!(
        scalar(&alice, &format!("SELECT COUNT(*) FROM {table}")).await,
        "0",
        "policy revoke must be hot-visible to the existing pgwire session"
    );
}

/// The protobuf programmatic SQL surface is authenticated gRPC
/// `ProximaRecordService.ExecuteQuery`. Keep this as a separate live ratchet
/// from pgwire and REST: gRPC obtains the subject
/// from a verified API key and therefore proves the load-bearing authenticated
/// carrier, not pgwire's deliberately trust-asserted user name.
#[test]
fn grpc_reads_follow_live_rest_policy_provision_and_revoke() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_stack_size(8 * 1024 * 1024)
        .enable_all()
        .build()
        .expect("tokio runtime");
    runtime.block_on(grpc_reads_follow_live_rest_policy_provision_and_revoke_inner());
}

async fn grpc_reads_follow_live_rest_policy_provision_and_revoke_inner() {
    let server = LiveServer::start().await.expect("start live server");

    // Use pgwire only as the setup/operator SQL surface. The assertions below
    // all cross the independently authenticated gRPC transport.
    let setup = connect(&server, "setup-operator").await;
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let table = format!("abac_grpc_live_{suffix}");
    exec(
        &setup,
        &format!("CREATE TABLE {table} (id INT PRIMARY KEY, value INT)"),
    )
    .await;
    exec(&setup, &format!("INSERT INTO {table} VALUES (1, 10)")).await;
    exec(&setup, &format!("INSERT INTO {table} VALUES (2, 20)")).await;

    let object_id = table_object_id(&setup, &table).await;
    let table_scope = u32::try_from(object_id).expect("ABAC table scope is u32");
    let sql = format!("SELECT id FROM {table}");
    let mut alice = connect_grpc(&server).await;
    let mut bob = connect_grpc(&server).await;

    assert_eq!(
        grpc_row_count(&mut alice, ALICE_KEY, &sql).await,
        0,
        "an authenticated but unprovisioned gRPC subject must fail closed"
    );

    let http = HttpClient::builder()
        .timeout(Duration::from_secs(5))
        .no_proxy()
        .build()
        .expect("HTTP client");
    let auth = format!("Api-Key {OPERATOR_KEY}");
    let response = http
        .post(server.admin_url("/api/v2/abac/attribute-bindings"))
        .header(reqwest::header::AUTHORIZATION, &auth)
        .json(&json!({
            "subject_id": "alice",
            "tenant": TENANT,
            "attrs": {}
        }))
        .send()
        .await
        .expect("provision alice binding");
    let binding = assert_admin_success(response, "provision attribute binding").await;
    assert_eq!(binding["subject_id"], "alice");

    let policy_object_id = object_id + 2_000_000_000;
    let policy_path = format!("/api/v2/abac/policy-bindings/{TENANT}/{policy_object_id}");
    let response = http
        .put(server.admin_url(&policy_path))
        .header(reqwest::header::AUTHORIZATION, &auth)
        .json(&json!({
            "scope": {"Table": table_scope},
            "effect": "Permit"
        }))
        .send()
        .await
        .expect("provision table permit");
    let policy = assert_admin_success(response, "provision table permit").await;
    assert_eq!(
        policy["tenant_stable_id"].as_u64(),
        binding["tenant_stable_id"].as_u64()
    );

    let permitted = grpc_query(&mut alice, ALICE_KEY, &sql).await;
    assert_eq!(
        permitted.rows.len(),
        2,
        "the existing authenticated gRPC client must observe the hot permit"
    );
    assert!(
        permitted.rows.iter().all(|row| {
            row.values.get("id").is_some_and(|value| {
                value.value.as_ref().is_some_and(|inner| {
                    !matches!(
                        inner,
                        proximadb::proto::proximadb_v2::typed_value::Value::TextValue(_)
                    )
                })
            })
        }),
        "v2 gRPC SQL must carry native TypedValue cells, not JSON-wrapped v1 SqlValue"
    );
    assert_eq!(
        grpc_row_count(&mut bob, BOB_KEY, &sql).await,
        0,
        "the table permit must not admit another authenticated principal"
    );

    let response = http
        .delete(server.admin_url(&policy_path))
        .header(reqwest::header::AUTHORIZATION, &auth)
        .send()
        .await
        .expect("revoke table permit");
    assert_eq!(
        response.status(),
        StatusCode::NO_CONTENT,
        "policy revoke must succeed"
    );
    assert_eq!(
        grpc_row_count(&mut alice, ALICE_KEY, &sql).await,
        0,
        "the existing authenticated gRPC client must observe the hot revoke"
    );
}

/// The canonical SQL-over-REST surface must carry the same verified identity
/// and reach the same ABAC-protected relational scan as pgwire and gRPC.
#[test]
fn rest_sql_reads_follow_live_policy_provision_and_revoke() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_stack_size(8 * 1024 * 1024)
        .enable_all()
        .build()
        .expect("tokio runtime");
    runtime.block_on(rest_sql_reads_follow_live_policy_provision_and_revoke_inner());
}

async fn rest_sql_reads_follow_live_policy_provision_and_revoke_inner() {
    let server = LiveServer::start().await.expect("start live server");
    let setup = connect(&server, "setup-operator").await;
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let table = format!("abac_rest_sql_live_{suffix}");
    exec(
        &setup,
        &format!("CREATE TABLE {table} (id INT PRIMARY KEY, value INT)"),
    )
    .await;
    exec(&setup, &format!("INSERT INTO {table} VALUES (1, 10)")).await;
    exec(&setup, &format!("INSERT INTO {table} VALUES (2, 20)")).await;

    let object_id = table_object_id(&setup, &table).await;
    let table_scope = u32::try_from(object_id).expect("ABAC table scope is u32");
    let sql = format!("SELECT id FROM {table}");
    let http = HttpClient::builder()
        .timeout(Duration::from_secs(5))
        .no_proxy()
        .build()
        .expect("HTTP client");

    assert_eq!(
        rest_sql_row_count(&http, &server, ALICE_KEY, &sql).await,
        0,
        "an authenticated but unprovisioned REST SQL subject must fail closed"
    );

    let operator_auth = format!("Api-Key {OPERATOR_KEY}");
    let response = http
        .post(server.admin_url("/api/v2/abac/attribute-bindings"))
        .header(reqwest::header::AUTHORIZATION, &operator_auth)
        .json(&json!({
            "subject_id": "alice",
            "tenant": TENANT,
            "attrs": {}
        }))
        .send()
        .await
        .expect("provision alice binding");
    let binding = assert_admin_success(response, "provision attribute binding").await;

    let policy_object_id = object_id + 3_000_000_000;
    let policy_path = format!("/api/v2/abac/policy-bindings/{TENANT}/{policy_object_id}");
    let response = http
        .put(server.admin_url(&policy_path))
        .header(reqwest::header::AUTHORIZATION, &operator_auth)
        .json(&json!({
            "scope": {"Table": table_scope},
            "effect": "Permit"
        }))
        .send()
        .await
        .expect("provision table permit");
    let policy = assert_admin_success(response, "provision table permit").await;
    assert_eq!(
        policy["tenant_stable_id"].as_u64(),
        binding["tenant_stable_id"].as_u64()
    );

    assert_eq!(
        rest_sql_row_count(&http, &server, ALICE_KEY, &sql).await,
        2,
        "REST SQL must observe the hot permit"
    );
    assert_eq!(
        rest_sql_row_count(&http, &server, BOB_KEY, &sql).await,
        0,
        "the table permit must not admit another REST principal"
    );

    let response = http
        .delete(server.admin_url(&policy_path))
        .header(reqwest::header::AUTHORIZATION, &operator_auth)
        .send()
        .await
        .expect("revoke table permit");
    assert_eq!(response.status(), StatusCode::NO_CONTENT);
    assert_eq!(
        rest_sql_row_count(&http, &server, ALICE_KEY, &sql).await,
        0,
        "REST SQL must observe the hot revoke"
    );
}

/// Canonical REST record search and point-get must consume the same hot policy
/// stores as the operator control plane. The stored predicate is deliberately
/// row-selective so this proves enforcement, not merely route authentication.
#[test]
fn rest_records_follow_live_row_policy_provision_and_revoke() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_stack_size(8 * 1024 * 1024)
        .enable_all()
        .build()
        .expect("tokio runtime");
    runtime.block_on(rest_records_follow_live_row_policy_provision_and_revoke_inner());
}

async fn rest_records_follow_live_row_policy_provision_and_revoke_inner() {
    let server = LiveServer::start().await.expect("start live server");
    let http = HttpClient::builder()
        .timeout(Duration::from_secs(5))
        .no_proxy()
        .build()
        .expect("HTTP client");
    let fixture = create_live_record_fixture(&http, &server, "abac_rest_records").await;

    assert!(
        rest_record_search_ids(&http, &server, ALICE_KEY, &fixture.collection)
            .await
            .is_empty(),
        "an unprovisioned REST record principal must fail closed"
    );
    assert_eq!(
        rest_record_get_status(
            &http,
            &server,
            ALICE_KEY,
            &fixture.collection,
            fixture.eng_id,
        )
        .await,
        StatusCode::NOT_FOUND,
        "unprovisioned point-get must hide the record"
    );

    let policy_path = provision_live_record_policy(&http, &server, 4_000_000_000).await;

    let alice_ids = rest_record_search_ids(&http, &server, ALICE_KEY, &fixture.collection).await;
    assert_eq!(alice_ids, vec![fixture.eng_id.to_string()]);
    assert_eq!(
        rest_record_get_status(
            &http,
            &server,
            ALICE_KEY,
            &fixture.collection,
            fixture.eng_id,
        )
        .await,
        StatusCode::OK
    );
    assert_eq!(
        rest_record_get_status(
            &http,
            &server,
            ALICE_KEY,
            &fixture.collection,
            fixture.hr_id,
        )
        .await,
        StatusCode::NOT_FOUND,
        "point-get must post-filter a row that violates the stored predicate"
    );
    assert!(
        rest_record_search_ids(&http, &server, BOB_KEY, &fixture.collection)
            .await
            .is_empty(),
        "an unbound authenticated principal must remain denied"
    );

    revoke_live_policy(&http, &server, &policy_path).await;
    assert!(
        rest_record_search_ids(&http, &server, ALICE_KEY, &fixture.collection)
            .await
            .is_empty(),
        "record search must observe the hot revoke"
    );
    assert_eq!(
        rest_record_get_status(
            &http,
            &server,
            ALICE_KEY,
            &fixture.collection,
            fixture.eng_id,
        )
        .await,
        StatusCode::NOT_FOUND,
        "point-get must observe the hot revoke"
    );
}

/// Authenticated gRPC v2 search and point-get must project the credential-bound
/// subject into the same record read-context composition used by REST.
#[test]
fn grpc_records_follow_live_row_policy_provision_and_revoke() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_stack_size(8 * 1024 * 1024)
        .enable_all()
        .build()
        .expect("tokio runtime");
    runtime.block_on(grpc_records_follow_live_row_policy_provision_and_revoke_inner());
}

async fn grpc_records_follow_live_row_policy_provision_and_revoke_inner() {
    let server = LiveServer::start().await.expect("start live server");
    let http = HttpClient::builder()
        .timeout(Duration::from_secs(5))
        .no_proxy()
        .build()
        .expect("HTTP client");
    let fixture = create_live_record_fixture(&http, &server, "abac_grpc_records").await;
    let mut alice = connect_grpc(&server).await;
    let mut bob = connect_grpc(&server).await;

    assert!(
        grpc_record_search_ids(&mut alice, ALICE_KEY, &fixture.collection)
            .await
            .is_empty(),
        "an unprovisioned gRPC record principal must fail closed"
    );
    assert!(
        grpc_record_stream_ids(&mut alice, ALICE_KEY, &fixture.collection)
            .await
            .is_empty(),
        "the unprovisioned streaming search path must also fail closed"
    );
    assert!(
        !grpc_record_found(&mut alice, ALICE_KEY, &fixture.collection, fixture.eng_id,).await,
        "unprovisioned gRPC point-get must hide the record"
    );

    let policy_path = provision_live_record_policy(&http, &server, 5_000_000_000).await;

    let alice_ids = grpc_record_search_ids(&mut alice, ALICE_KEY, &fixture.collection).await;
    assert_eq!(alice_ids, vec![fixture.eng_id.to_string()]);
    let alice_stream_ids = grpc_record_stream_ids(&mut alice, ALICE_KEY, &fixture.collection).await;
    assert_eq!(alice_stream_ids, vec![fixture.eng_id.to_string()]);
    assert!(grpc_record_found(&mut alice, ALICE_KEY, &fixture.collection, fixture.eng_id,).await);
    assert!(
        !grpc_record_found(&mut alice, ALICE_KEY, &fixture.collection, fixture.hr_id,).await,
        "gRPC point-get must hide a row that violates the stored predicate"
    );
    assert!(
        grpc_record_search_ids(&mut bob, BOB_KEY, &fixture.collection)
            .await
            .is_empty(),
        "an unbound authenticated gRPC principal must remain denied"
    );

    revoke_live_policy(&http, &server, &policy_path).await;
    assert!(
        grpc_record_search_ids(&mut alice, ALICE_KEY, &fixture.collection)
            .await
            .is_empty(),
        "gRPC record search must observe the hot revoke"
    );
    assert!(
        grpc_record_stream_ids(&mut alice, ALICE_KEY, &fixture.collection)
            .await
            .is_empty(),
        "gRPC streaming search must observe the hot revoke"
    );
    assert!(
        !grpc_record_found(&mut alice, ALICE_KEY, &fixture.collection, fixture.eng_id,).await,
        "gRPC point-get must observe the hot revoke"
    );
}

/// Arrow Flight's canonical `type: vector_search` ticket must authenticate the
/// API key and project that principal into the same record-read authority as
/// REST and gRPC. Reusing each connected Flight client across policy mutations
/// proves authorization is evaluated per operation rather than cached at
/// connection establishment.
#[test]
fn flight_records_follow_live_row_policy_provision_and_revoke() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_stack_size(8 * 1024 * 1024)
        .enable_all()
        .build()
        .expect("tokio runtime");
    runtime.block_on(flight_records_follow_live_row_policy_provision_and_revoke_inner());
}

async fn flight_records_follow_live_row_policy_provision_and_revoke_inner() {
    let server = LiveServer::start().await.expect("start live server");
    let http = HttpClient::builder()
        .timeout(Duration::from_secs(5))
        .no_proxy()
        .build()
        .expect("HTTP client");
    let fixture = create_live_record_fixture(&http, &server, "abac_flight_records").await;
    let mut alice = connect_flight(&server).await;
    let mut bob = connect_flight(&server).await;

    assert!(
        flight_record_search_ids(&mut alice, ALICE_KEY, &fixture.collection)
            .await
            .is_empty(),
        "an unprovisioned Flight record principal must fail closed"
    );

    let policy_path = provision_live_record_policy(&http, &server, 6_000_000_000).await;

    assert_eq!(
        flight_record_search_ids(&mut alice, ALICE_KEY, &fixture.collection).await,
        vec![fixture.eng_id.to_string()],
        "Flight search must apply the stored row predicate"
    );
    assert!(
        flight_record_search_ids(&mut bob, BOB_KEY, &fixture.collection)
            .await
            .is_empty(),
        "an unbound authenticated Flight principal must remain denied"
    );

    revoke_live_policy(&http, &server, &policy_path).await;
    assert!(
        flight_record_search_ids(&mut alice, ALICE_KEY, &fixture.collection)
            .await
            .is_empty(),
        "the existing Flight client must observe the hot revoke"
    );
}

// ===========================================================================
// TD-AUTHZ-1 L1.2 — the GRANT ratchet: provision -> permit -> deny -> revoke
// over a live server. This is the documented flip-precondition for arming
// grant enforcement (PROXIMADB_AUTHZ_REQUIRE_GRANTS, or a tenant's `Enforce`
// posture): enforcement must not be turned on for anyone until this is green.
//
// SCOPE — same-tenant, deliberately. The cross-tenant variant is NOT written
// here because it cannot pass yet: three live read paths still retain rows by
// `record.tenant_id == tenant_context.tenant_id`
// (src/services/operations/vectors/legacy.rs:1123/:1175/:1346), so a foreign
// grantee is admitted by the catalog and then filtered to zero rows. Writing
// it now would produce a test that fails for reasons unrelated to grants. See
// the 2026-08-07 addendum in TD-TENANT-2; that predicate is itself load-bearing
// until the WAL read is tenant-keyed (`get_collection_vectors_raw` takes a bare
// collection id), so it cannot simply be deleted either.
// ===========================================================================

/// Set a tenant's grant-enforcement posture through the live admin API
/// (TD-SEC-2 Slice C). Returns once the write is durable and hot-visible.
async fn set_tenant_posture(client: &HttpClient, server: &LiveServer, tenant: &str, mode: &str) {
    let response = client
        .put(server.admin_url(&format!("/api/v2/abac/tenant-posture/{tenant}")))
        .header(
            reqwest::header::AUTHORIZATION,
            format!("Api-Key {OPERATOR_KEY}"),
        )
        .json(&json!({ "grant_enforcement": mode }))
        .send()
        .await
        .expect("put tenant posture");
    assert_admin_success(response, &format!("set posture {mode}")).await;
}

/// Provision a Read grant over one collection for one subject of `tenant`.
/// Returns the grant id, so the revoke half of the ratchet can name it.
async fn provision_read_grant(
    client: &HttpClient,
    server: &LiveServer,
    tenant: &str,
    subject: &str,
    table_object_id: u64,
) -> String {
    let response = client
        .post(server.admin_url("/api/v2/abac/grants"))
        .header(
            reqwest::header::AUTHORIZATION,
            format!("Api-Key {OPERATOR_KEY}"),
        )
        .json(&json!({
            "owner_tenant": tenant,
            "resource": { "Table": table_object_id },
            "grantee": { "tenant": tenant, "subject": subject },
            "actions": ["Read"],
        }))
        .send()
        .await
        .expect("post grant");
    let body = assert_admin_success(response, "provision grant").await;
    body.get("grant_id")
        .and_then(|v| v.as_str())
        .expect("grant_id in response")
        .to_string()
}

async fn revoke_grant(client: &HttpClient, server: &LiveServer, tenant: &str, grant_id: &str) {
    let response = client
        .delete(server.admin_url(&format!("/api/v2/abac/grants/{tenant}/{grant_id}")))
        .header(
            reqwest::header::AUTHORIZATION,
            format!("Api-Key {OPERATOR_KEY}"),
        )
        .send()
        .await
        .expect("delete grant");
    assert_eq!(
        response.status(),
        StatusCode::NO_CONTENT,
        "revoke must be idempotent 204"
    );
}

#[test]
fn grant_enforcement_follows_live_provision_permit_deny_revoke() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_stack_size(8 * 1024 * 1024)
        .enable_all()
        .build()
        .expect("tokio runtime");
    runtime.block_on(grant_ratchet_inner());
}

async fn grant_ratchet_inner() {
    let server = LiveServer::start().await.expect("start live server");
    let http = HttpClient::builder()
        .timeout(Duration::from_secs(5))
        .no_proxy()
        .build()
        .expect("HTTP client");
    let fixture = create_live_record_fixture(&http, &server, "authz_grant_ratchet").await;

    // ABAC is fail-closed, so an unprovisioned principal reads nothing even
    // before grants exist. Provision the POLICY layer first: grants compose as
    // deny > absence-of-grant > grant, and the enforcement seam consults grants
    // only on the policy `Admit` arm — so a policy denial would mask everything
    // this test is trying to prove.
    let _policy_path = provision_live_record_policy(&http, &server, 4_100_000_000).await;

    // Baseline: posture is Off (the default), so admission is policy-only and
    // alice's row is visible. If this fails, every "denied ⇒ empty" assertion
    // below would pass vacuously against a broken fixture.
    let baseline = rest_record_search_ids(&http, &server, ALICE_KEY, &fixture.collection).await;
    assert!(
        !baseline.is_empty(),
        "the POLICY layer must admit before enforcement is armed, otherwise the \
         deny assertions below prove nothing"
    );

    let pg = connect(&server, "alice").await;
    let table_object_id = table_object_id(&pg, &fixture.collection).await;

    // 1. ARM: this tenant now requires a grant for every read.
    set_tenant_posture(&http, &server, TENANT, "Enforce").await;

    // 2. DENY: no grant exists yet, so the read is refused. Enforcement
    //    manifests as an empty result set — the read path fails CLOSED on
    //    DenyReason::NoApplicableGrant rather than returning rows.
    let denied = rest_record_search_ids(&http, &server, ALICE_KEY, &fixture.collection).await;
    assert!(
        denied.is_empty(),
        "with Enforce posture and no grant the read must be denied, got {denied:?}"
    );

    // 3. PERMIT: provision a Read grant for this subject over this collection.
    let grant_id = provision_read_grant(&http, &server, TENANT, "alice", table_object_id).await;
    let permitted = rest_record_search_ids(&http, &server, ALICE_KEY, &fixture.collection).await;
    assert!(
        !permitted.is_empty(),
        "an applicable grant must restore the read (hot-visible, no restart)"
    );
    assert_eq!(
        permitted.len(),
        baseline.len(),
        "a Read grant admits the same rows policy-only admission did — the \
         grant gates ADMISSION, it does not additionally filter rows (row-level \
         narrowing rides predicate_ref, which is L2)"
    );

    // 4. REVOKE: the entitlement is withdrawn and the read closes again.
    revoke_grant(&http, &server, TENANT, &grant_id).await;
    let revoked = rest_record_search_ids(&http, &server, ALICE_KEY, &fixture.collection).await;
    assert!(
        revoked.is_empty(),
        "revoking the grant must close the read again, got {revoked:?}"
    );

    // 5. DISARM: returning the tenant to Off restores policy-only admission.
    //    This is the Slice C guarantee that posture is reversible per tenant.
    set_tenant_posture(&http, &server, TENANT, "Off").await;
    let disarmed = rest_record_search_ids(&http, &server, ALICE_KEY, &fixture.collection).await;
    assert_eq!(
        disarmed.len(),
        baseline.len(),
        "posture Off must restore the pre-enforcement baseline"
    );
}
