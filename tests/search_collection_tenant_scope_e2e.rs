//! Search collection resolution is scoped to the request tenant.
//!
//! One in-process server, two tenants (`acme`, `globex`) plus requests with
//! no tenant signal (the default tenant). Each tenant owns a collection with a
//! distinct name and disjoint records (collection names are unique across the
//! deployment). On every search / read surface:
//!
//! * a tenant searching its OWN collection is served;
//! * a tenant addressing ANOTHER tenant's collection — by name, by object id,
//!   or with a wrong-dimension vector — gets exactly the response it would get
//!   for a collection that does not exist (same status, same body once the
//!   identifier and per-request ids are normalised);
//! * collection listing omits other tenants' collections;
//! * requests without a tenant signal keep reading the default tenant's
//!   collection (single-tenant deployments are unchanged);
//! * a collection created on any surface records its creator as owner.

use std::net::TcpListener;
use std::time::Duration;

use arrow_flight::Ticket;
use arrow_flight::flight_service_client::FlightServiceClient;
use futures::TryStreamExt;
use proximadb::core::Config;
use proximadb::database::ProximaDB;
use proximadb::proto::proximadb_v2 as pv2;
use pv2::proxima_entity_service_client::ProximaEntityServiceClient;
use pv2::proxima_fusion_service_client::ProximaFusionServiceClient;
use pv2::proxima_record_service_client::ProximaRecordServiceClient;
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::time::sleep;

const DIM: usize = 4;
const ACME: &str = "acme";
const GLOBEX: &str = "globex";
const ACME_DOCS: &str = "acme_docs";
const GLOBEX_DOCS: &str = "globex_docs";
const DEFAULT_DOCS: &str = "default_docs";
/// Never created: the "does not exist" reference.
const MISSING_DOCS: &str = "missing_docs";
/// An object id no collection has.
const MISSING_OBJECT_ID: &str = "987654321";

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind port 0");
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);
    port
}

struct Server {
    rest_port: u16,
    grpc_port: u16,
    flight_port: u16,
    pg_port: u16,
    db: Option<ProximaDB>,
    _tmp: TempDir,
}

impl Server {
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
        config.api.enable_pgwire = true;
        config.storage.storage_locations = vec![proximadb::core::config::StorageLocation {
            url: format!("file://{}", tmp.path().display()),
            ..Default::default()
        }];
        config.storage.metadata_url = format!("file://{}/metadata", tmp.path().display());
        config.storage.wal_config.write_buffer_directory =
            format!("file://{}/wal", tmp.path().display());

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
                Ok(resp) if resp.status().is_success() => break,
                _ if std::time::Instant::now() > deadline => {
                    anyhow::bail!("server did not become ready in 20s")
                }
                _ => sleep(Duration::from_millis(100)).await,
            }
        }
        sleep(Duration::from_millis(300)).await;

        Ok(Self {
            rest_port,
            grpc_port,
            flight_port,
            pg_port,
            db: Some(db),
            _tmp: tmp,
        })
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Some(mut db) = self.db.take() {
            tokio::spawn(async move {
                let _ = db.shutdown().await;
            });
        }
    }
}

/// A response reduced to what a client can observe, comparable across
/// requests: the status plus the body with the addressed identifier replaced
/// by `<C>` and per-request ids and timings removed.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Observed {
    status: String,
    body: String,
}

fn strip_request_ids(value: &mut Value) {
    match value {
        Value::Object(map) => {
            // Per-request ids and timings are not observable differences.
            for volatile in [
                "request_id",
                "query_id",
                "started_at_ms",
                "finished_at_ms",
                "wall_time_us",
                "latency_ms",
                "processing_time_us",
                "execution_time_ms",
            ] {
                map.remove(volatile);
            }
            for child in map.values_mut() {
                strip_request_ids(child);
            }
        }
        Value::Array(items) => items.iter_mut().for_each(strip_request_ids),
        _ => {}
    }
}

/// Replace whole-word occurrences of `identifier` (word = `[A-Za-z0-9_]` run)
/// with `<C>`, so a numeric object id never rewrites digits elsewhere.
fn mask(text: &str, identifier: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut word = String::new();
    let flush = |word: &mut String, out: &mut String| {
        if word.as_str() == identifier {
            out.push_str("<C>");
        } else {
            out.push_str(word);
        }
        word.clear();
    };
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            word.push(ch);
        } else {
            flush(&mut word, &mut out);
            out.push(ch);
        }
    }
    flush(&mut word, &mut out);
    out
}

fn with_tenant<T>(message: T, tenant: &str) -> tonic::Request<T> {
    let mut request = tonic::Request::new(message);
    if let Ok(value) = tenant.parse() {
        request.metadata_mut().insert("x-tenant-id", value);
    }
    request
}

fn normalise(status: String, body: &str, identifier: &str) -> Observed {
    let body = match serde_json::from_str::<Value>(body) {
        Ok(mut json) => {
            strip_request_ids(&mut json);
            json.to_string()
        }
        Err(_) => body.to_string(),
    };
    // `(query_id: …)` suffixes in AQL error text.
    let body = match body.find("(query_id:") {
        Some(at) => body[..at].to_string(),
        None => body,
    };
    Observed {
        status,
        body: mask(&body, identifier),
    }
}

fn vector() -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    v[0] = 1.0;
    v
}

fn short_vector() -> Vec<f32> {
    vec![1.0f32; DIM - 1]
}

struct Rest {
    http: reqwest::Client,
    base: String,
}

impl Rest {
    async fn call(
        &self,
        tenant: Option<&str>,
        method: reqwest::Method,
        path: &str,
        body: Option<Value>,
    ) -> (reqwest::StatusCode, String) {
        let mut request = self.http.request(method, format!("{}{path}", self.base));
        if let Some(tenant) = tenant {
            request = request.header("X-Tenant-ID", tenant);
        }
        if let Some(body) = body {
            request = request.json(&body);
        }
        let response = request.send().await.expect("request sent");
        let status = response.status();
        (status, response.text().await.unwrap_or_default())
    }

    /// Create a collection as `tenant`; returns its object id.
    async fn create(&self, tenant: Option<&str>, name: &str) -> String {
        let (status, body) = self
            .call(
                tenant,
                reqwest::Method::POST,
                "/api/v2/collections",
                Some(json!({
                    "name": name, "dimension": DIM, "engine": "sst",
                    "distance_metric": "cosine", "enable_proxima_record": false,
                })),
            )
            .await;
        assert!(status.is_success(), "create {name}: {status} {body}");
        let created: Value = serde_json::from_str(&body).expect("create json");
        created["collection_id"]
            .as_str()
            .filter(|id| !id.is_empty())
            .unwrap_or_else(|| panic!("create {name} returned no object id: {body}"))
            .to_string()
    }

    async fn insert(&self, tenant: Option<&str>, name: &str, prefix: &str) {
        let records: Vec<Value> = (0..4)
            .map(|i| {
                let mut v = vec![0.0f32; DIM];
                v[i % DIM] = 1.0;
                let topic = if i % 2 == 0 { "alpha" } else { "beta" };
                json!({ "id": format!("{prefix}-r{i}"), "vector": v, "props": { "topic": topic } })
            })
            .collect();
        let (status, body) = self
            .call(
                tenant,
                reqwest::Method::POST,
                &format!("/api/v2/collections/{name}/records/batch"),
                Some(json!({ "records": records })),
            )
            .await;
        assert!(status.is_success(), "insert into {name}: {status} {body}");
    }
}

/// One REST surface: how to address collection `c` with query vector `v`.
struct RestSurface {
    label: &'static str,
    method: reqwest::Method,
    path: fn(&str) -> String,
    body: fn(&str, &[f32]) -> Option<Value>,
}

fn rest_surfaces() -> Vec<RestSurface> {
    vec![
        RestSurface {
            label: "REST v2 collection search",
            method: reqwest::Method::POST,
            path: |c| format!("/api/v2/collections/{c}/search"),
            body: |_, v| Some(json!({ "vector": v, "top_k": 5 })),
        },
        RestSurface {
            label: "REST v1 progressive search",
            method: reqwest::Method::POST,
            path: |c| format!("/api/v2/progressive/search/{c}"),
            body: |c, v| {
                Some(json!({ "collection_id": c, "queries": [{ "vector": v }], "top_k": 5 }))
            },
        },
        RestSurface {
            label: "REST hybrid search",
            method: reqwest::Method::POST,
            path: |_| "/api/v2/hybrid/search".to_string(),
            body: |c, v| Some(json!({ "collection": c, "vector": v, "top_k": 5 })),
        },
        RestSurface {
            label: "REST graph fusion-search",
            method: reqwest::Method::POST,
            path: |c| format!("/api/v2/graphs/{c}/fusion-search"),
            body: |c, v| Some(json!({ "vector_collection": c, "query_vector": v, "limit": 5 })),
        },
        RestSurface {
            label: "REST v2 entity search",
            method: reqwest::Method::POST,
            path: |c| format!("/api/v2/collections/{c}/entities/search"),
            body: |_, v| Some(json!({ "query_vector": v, "top_k": 5 })),
        },
        RestSurface {
            label: "REST v1 entity search",
            method: reqwest::Method::POST,
            path: |c| format!("/api/v1/collections/{c}/entities/search"),
            body: |_, v| Some(json!({ "query_vector": v, "top_k": 5 })),
        },
        RestSurface {
            label: "REST analytics entanglement",
            method: reqwest::Method::GET,
            path: |c| {
                format!("/api/v2/analytics/collections/{c}/entanglement?topic_field=topic&limit=10")
            },
            body: |_, _| None,
        },
        RestSurface {
            label: "REST collection GET",
            method: reqwest::Method::GET,
            path: |c| format!("/api/v2/collections/{c}"),
            body: |_, _| None,
        },
        RestSurface {
            label: "REST record scan",
            method: reqwest::Method::POST,
            path: |c| format!("/api/v2/collections/{c}/records/scan"),
            body: |_, _| Some(json!({ "limit": 10 })),
        },
    ]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn search_resolution_is_scoped_to_the_request_tenant() {
    let server = Server::start().await.expect("server start");
    let rest = Rest {
        http: reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .no_proxy()
            .build()
            .expect("http client"),
        base: format!("http://127.0.0.1:{}", server.rest_port),
    };

    let acme_id = rest.create(Some(ACME), ACME_DOCS).await;
    let globex_id = rest.create(Some(GLOBEX), GLOBEX_DOCS).await;
    let _default_id = rest.create(None, DEFAULT_DOCS).await;
    rest.insert(Some(ACME), ACME_DOCS, ACME).await;
    rest.insert(Some(GLOBEX), GLOBEX_DOCS, GLOBEX).await;
    rest.insert(None, DEFAULT_DOCS, "default").await;
    sleep(Duration::from_millis(300)).await;

    let mut failures: Vec<String> = Vec::new();

    // --- REST surfaces.
    for surface in rest_surfaces() {
        // Own collection is served.
        for (tenant, own) in [(Some(ACME), ACME_DOCS), (None, DEFAULT_DOCS)] {
            let (status, body) = rest
                .call(
                    tenant,
                    surface.method.clone(),
                    &(surface.path)(own),
                    (surface.body)(own, &vector()),
                )
                .await;
            if !status.is_success() {
                failures.push(format!(
                    "{}: tenant {tenant:?} reading its own {own}: {status} {body}",
                    surface.label
                ));
            }
        }
        // Another tenant's collection looks exactly like a missing one.
        let probes = [
            ("by name", GLOBEX_DOCS, MISSING_DOCS, vector()),
            (
                "by object id",
                globex_id.as_str(),
                MISSING_OBJECT_ID,
                vector(),
            ),
            (
                "wrong-dimension vector",
                GLOBEX_DOCS,
                MISSING_DOCS,
                short_vector(),
            ),
            (
                "default tenant's collection",
                DEFAULT_DOCS,
                MISSING_DOCS,
                vector(),
            ),
        ];
        for (how, foreign, missing, v) in probes {
            let (s1, b1) = rest
                .call(
                    Some(ACME),
                    surface.method.clone(),
                    &(surface.path)(foreign),
                    (surface.body)(foreign, &v),
                )
                .await;
            let (s2, b2) = rest
                .call(
                    Some(ACME),
                    surface.method.clone(),
                    &(surface.path)(missing),
                    (surface.body)(missing, &v),
                )
                .await;
            let foreign_seen = normalise(s1.to_string(), &b1, foreign);
            let missing_seen = normalise(s2.to_string(), &b2, missing);
            if foreign_seen != missing_seen {
                failures.push(format!(
                    "{} {how}: another tenant's collection is distinguishable from a missing one:\n      foreign: {foreign_seen:?}\n      missing: {missing_seen:?}",
                    surface.label
                ));
            }
            if s1.is_success() {
                failures.push(format!(
                    "{} {how}: acme was served another tenant's collection: {s1} {b1}",
                    surface.label
                ));
            }
        }
        // The other tenant still reads its own collection by object id.
        let (status, body) = rest
            .call(
                Some(GLOBEX),
                surface.method.clone(),
                &(surface.path)(&globex_id),
                (surface.body)(&globex_id, &vector()),
            )
            .await;
        if !status.is_success() {
            failures.push(format!(
                "{}: globex reading its own collection by object id: {status} {body}",
                surface.label
            ));
        }
    }

    // --- REST record get by id.
    let (s1, b1) = rest
        .call(
            Some(ACME),
            reqwest::Method::GET,
            &format!("/api/v2/collections/{GLOBEX_DOCS}/records/{GLOBEX}-r0"),
            None,
        )
        .await;
    let (s2, b2) = rest
        .call(
            Some(ACME),
            reqwest::Method::GET,
            &format!("/api/v2/collections/{MISSING_DOCS}/records/{GLOBEX}-r0"),
            None,
        )
        .await;
    if normalise(s1.to_string(), &b1, GLOBEX_DOCS) != normalise(s2.to_string(), &b2, MISSING_DOCS)
        || s1.is_success()
    {
        failures.push(format!(
            "REST record get: foreign {s1} {b1} vs missing {s2} {b2}"
        ));
    }

    // --- Collection listing omits other tenants' collections.
    for (tenant, own, others) in [
        (Some(ACME), ACME_DOCS, [GLOBEX_DOCS, DEFAULT_DOCS]),
        (None, DEFAULT_DOCS, [ACME_DOCS, GLOBEX_DOCS]),
    ] {
        let (status, body) = rest
            .call(tenant, reqwest::Method::GET, "/api/v2/collections", None)
            .await;
        if !status.is_success() || !body.contains(own) {
            failures.push(format!(
                "REST list as {tenant:?}: own collection missing: {status} {body}"
            ));
        }
        for other in others {
            if body.contains(other) {
                failures.push(format!("REST list as {tenant:?} includes {other}: {body}"));
            }
        }
    }

    // --- AQL `vector` source (its collection is the source name `vector`).
    let aql = |tenant: Option<&'static str>| {
        let rest = &rest;
        async move {
            rest.call(
                tenant,
                reqwest::Method::POST,
                "/api/v2/aql/execute",
                Some(json!({
                    "query": {
                        "find": { "projections": [] },
                        "from": { "Source": { "name": "vector", "alias": null } },
                        "where_clause": { "predicate": { "And": {
                            "lhs": { "Equals": { "field": "vector", "value": { "Vector": vector() } } },
                            "rhs": { "SemanticMatch": {
                                "field": "embedding", "query": "", "threshold": 0.0, "top_k": 5
                            }}
                        }}}
                    }
                })),
            )
            .await
        }
    };
    let (s_missing, b_missing) = aql(Some(ACME)).await;
    rest.create(Some(GLOBEX), "vector").await;
    rest.insert(Some(GLOBEX), "vector", "gvec").await;
    sleep(Duration::from_millis(200)).await;
    let (s_foreign, b_foreign) = aql(Some(ACME)).await;
    if normalise(s_foreign.to_string(), &b_foreign, "vector")
        != normalise(s_missing.to_string(), &b_missing, "vector")
    {
        failures.push(format!(
            "AQL vector source: another tenant's collection is distinguishable: foreign {s_foreign} {b_foreign} vs missing {s_missing} {b_missing}"
        ));
    }
    let (s_own, b_own) = aql(Some(GLOBEX)).await;
    if !s_own.is_success() || b_own.contains("Failure") {
        failures.push(format!("AQL vector source as owner: {s_own} {b_own}"));
    }

    // --- gRPC.
    let endpoint = format!("http://127.0.0.1:{}", server.grpc_port);
    let observe = |outcome: Result<String, tonic::Status>, identifier: &str| match outcome {
        Ok(served) => Observed {
            status: "OK".to_string(),
            body: served,
        },
        Err(status) => Observed {
            status: format!("{:?}", status.code()),
            body: mask(status.message(), identifier),
        },
    };
    let mut records = ProximaRecordServiceClient::connect(endpoint.clone())
        .await
        .expect("record gRPC connect");
    let mut fusion = ProximaFusionServiceClient::connect(endpoint.clone())
        .await
        .expect("fusion gRPC connect");
    let mut entities = ProximaEntityServiceClient::connect(endpoint)
        .await
        .expect("entity gRPC connect");
    for (how, foreign, missing) in [
        ("by name", GLOBEX_DOCS, MISSING_DOCS),
        ("by object id", globex_id.as_str(), MISSING_OBJECT_ID),
    ] {
        let mut grpc_cases: Vec<(&str, Observed, Observed)> = Vec::new();
        for (target, slot) in [(foreign, 0usize), (missing, 1usize)] {
            let search = records
                .search(with_tenant(
                    pv2::TypedSearchRequest {
                        collection_id: target.to_string(),
                        query_vector: vector(),
                        top_k: 5,
                        ..Default::default()
                    },
                    ACME,
                ))
                .await
                .map(|r| format!("{} results", r.into_inner().results.len()));
            let get = records
                .get_collection(with_tenant(
                    pv2::V2GetCollectionRequest {
                        collection_id: target.to_string(),
                    },
                    ACME,
                ))
                .await
                .map(|r| r.into_inner().config.map(|c| c.name).unwrap_or_default());
            let fused = fusion
                .fusion_search(with_tenant(
                    pv2::FusionSearchRequest {
                        graph_id: target.to_string(),
                        vector_collection: target.to_string(),
                        query_vector: vector(),
                        limit: 5,
                        ..Default::default()
                    },
                    ACME,
                ))
                .await
                .map(|r| format!("{} hits", r.into_inner().results.len()));
            let entity = entities
                .search_entities(with_tenant(
                    pv2::SearchEntitiesRequest {
                        collection_id: target.to_string(),
                        similar: Some(pv2::SimilarQuery {
                            query: Some(pv2::similar_query::Query::Vector(pv2::VectorData {
                                values: vector(),
                            })),
                            ..Default::default()
                        }),
                        top_k: 5,
                        ..Default::default()
                    },
                    ACME,
                ))
                .await
                .map(|r| format!("{} entities", r.into_inner().results.len()));
            let outcomes = [
                ("gRPC Search", search),
                ("gRPC GetCollection", get),
                ("gRPC FusionSearch", fused),
                ("gRPC SearchEntities", entity),
            ];
            for (i, (label, outcome)) in outcomes.into_iter().enumerate() {
                let seen = observe(outcome, target);
                if slot == 0 {
                    grpc_cases.push((label, seen.clone(), seen));
                } else if let Some(case) = grpc_cases.get_mut(i) {
                    case.2 = seen;
                }
            }
        }
        for (label, foreign_seen, missing_seen) in grpc_cases {
            if foreign_seen != missing_seen || foreign_seen.status == "OK" {
                failures.push(format!(
                    "{label} {how}: foreign {foreign_seen:?} vs missing {missing_seen:?}"
                ));
            }
        }
    }
    let own = records
        .search(with_tenant(
            pv2::TypedSearchRequest {
                collection_id: ACME_DOCS.to_string(),
                query_vector: vector(),
                top_k: 5,
                ..Default::default()
            },
            ACME,
        ))
        .await;
    match own {
        Ok(r)
            if r.get_ref()
                .results
                .iter()
                .all(|h| h.id.starts_with("acme-")) => {}
        other => failures.push(format!("gRPC Search own collection: {other:?}")),
    }
    let listed = records
        .list_collections(with_tenant(pv2::V2ListCollectionsRequest::default(), ACME))
        .await
        .map(|r| {
            r.into_inner()
                .collections
                .into_iter()
                .filter_map(|c| c.config.map(|cfg| cfg.name))
                .collect::<Vec<_>>()
        });
    match listed {
        Ok(names)
            if names.iter().any(|n| n == ACME_DOCS)
                && !names.iter().any(|n| n == GLOBEX_DOCS || n == DEFAULT_DOCS) => {}
        other => failures.push(format!("gRPC ListCollections as acme: {other:?}")),
    }

    // --- Arrow Flight DoGet `vector_search`.
    let channel =
        tonic::transport::Endpoint::from_shared(format!("http://127.0.0.1:{}", server.flight_port))
            .expect("flight endpoint")
            .connect()
            .await
            .expect("flight channel");
    let mut flight = FlightServiceClient::new(channel);
    let mut flight_get = async |target: &str, tenant: &str| -> Observed {
        let ticket = Ticket {
            ticket: serde_json::to_vec(&json!({
                "type": "vector_search", "collection_id": target,
                "query_vector": vector(), "top_k": 5, "include_vector": false,
            }))
            .expect("ticket json")
            .into(),
        };
        let outcome = match flight.do_get(with_tenant(ticket, tenant)).await {
            Ok(stream) => stream
                .into_inner()
                .try_collect::<Vec<_>>()
                .await
                .map(|frames| format!("{} frames", frames.len())),
            Err(status) => Err(status),
        };
        observe(outcome, target)
    };
    for (how, foreign, missing) in [
        ("by name", GLOBEX_DOCS, MISSING_DOCS),
        ("by object id", globex_id.as_str(), MISSING_OBJECT_ID),
    ] {
        let foreign_seen = flight_get(foreign, ACME).await;
        let missing_seen = flight_get(missing, ACME).await;
        if foreign_seen != missing_seen || foreign_seen.status == "OK" {
            failures.push(format!(
                "Flight DoGet {how}: foreign {foreign_seen:?} vs missing {missing_seen:?}"
            ));
        }
    }
    let own_flight = flight_get(ACME_DOCS, ACME).await;
    if own_flight.status != "OK" {
        failures.push(format!("Flight DoGet own collection: {own_flight:?}"));
    }

    // --- pgwire pgvector search (startup `database` = tenant).
    let pg = |database: &str| {
        format!(
            "host=127.0.0.1 port={} user=postgres dbname={database} sslmode=disable",
            server.pg_port
        )
    };
    let (client, connection) = tokio_postgres::connect(&pg(ACME), tokio_postgres::NoTls)
        .await
        .expect("pgwire connect");
    tokio::spawn(connection);
    let pg_search = async |target: &str| -> Observed {
        // pgvector nearest-neighbour over a vector collection (the native
        // vector-search route; standalone `vector_search(...)` needs a join).
        let sql = format!("SELECT id FROM {target} ORDER BY embedding <-> '[1,0,0,0]' LIMIT 5");
        match client.query(sql.as_str(), &[]).await {
            Ok(rows) => Observed {
                status: "OK".to_string(),
                body: format!("{} rows", rows.len()),
            },
            Err(error) => {
                let (code, message) = error
                    .as_db_error()
                    .map(|db| (db.code().code().to_string(), db.message().to_string()))
                    .unwrap_or_else(|| ("client".to_string(), error.to_string()));
                Observed {
                    status: code,
                    body: mask(&message, target),
                }
            }
        }
    };
    let foreign_seen = pg_search(GLOBEX_DOCS).await;
    let missing_seen = pg_search(MISSING_DOCS).await;
    if foreign_seen != missing_seen || foreign_seen.status == "OK" {
        failures.push(format!(
            "pgwire pgvector search: foreign {foreign_seen:?} vs missing {missing_seen:?}"
        ));
    }
    let own_pg = pg_search(ACME_DOCS).await;
    if own_pg.status != "OK" {
        failures.push(format!("pgwire pgvector search own collection: {own_pg:?}"));
    }

    // --- Owner stamping at create: a collection created as acme on each
    // surface is not visible to the default tenant or globex.
    let grpc_created = records
        .create_collection(with_tenant(
            pv2::V2CollectionConfig {
                name: "acme_grpc_docs".to_string(),
                dimension: DIM as u32,
                distance_metric: "cosine".to_string(),
                storage_engine: "sst".to_string(),
                ..Default::default()
            },
            ACME,
        ))
        .await;
    if let Err(status) = grpc_created {
        failures.push(format!("gRPC CreateCollection as acme: {status:?}"));
    }
    rest.create(Some(ACME), "acme_v1_docs").await;
    for created in ["acme_grpc_docs", "acme_v1_docs"] {
        for viewer in [None, Some(GLOBEX)] {
            let (s1, b1) = rest
                .call(
                    viewer,
                    reqwest::Method::GET,
                    &format!("/api/v2/collections/{created}"),
                    None,
                )
                .await;
            let (s2, b2) = rest
                .call(
                    viewer,
                    reqwest::Method::GET,
                    &format!("/api/v2/collections/{MISSING_DOCS}"),
                    None,
                )
                .await;
            if normalise(s1.to_string(), &b1, created)
                != normalise(s2.to_string(), &b2, MISSING_DOCS)
            {
                failures.push(format!(
                    "{created} (created as acme) visible to {viewer:?}: {s1} {b1}"
                ));
            }
        }
        let (status, body) = rest
            .call(
                Some(ACME),
                reqwest::Method::GET,
                &format!("/api/v2/collections/{created}"),
                None,
            )
            .await;
        if !status.is_success() {
            failures.push(format!(
                "{created}: creator acme cannot read it: {status} {body}"
            ));
        }
    }

    // acme's own collection is not readable by object id from the default tenant.
    let (s1, b1) = rest
        .call(
            None,
            reqwest::Method::POST,
            &format!("/api/v2/collections/{acme_id}/search"),
            Some(json!({ "vector": vector(), "top_k": 5 })),
        )
        .await;
    if s1.is_success() {
        failures.push(format!(
            "default tenant read acme's collection by object id: {b1}"
        ));
    }

    assert!(
        failures.is_empty(),
        "search collection resolution must be scoped to the request tenant:\n  {}",
        failures.join("\n  ")
    );
}

/// One REST probe: `(label, method, path for collection c, body for c)`.
type RestProbe = (
    &'static str,
    reqwest::Method,
    fn(&str) -> String,
    fn(&str) -> Option<Value>,
);

fn sql_vector_search(c: &str) -> String {
    format!("SELECT * FROM VECTOR_SEARCH('{c}', '[1.0,0.0,0.0,0.0]', 5)")
}

fn beyond_search_probes() -> Vec<RestProbe> {
    vec![
        (
            "REST /api/v2/sql VECTOR_SEARCH",
            reqwest::Method::POST,
            |_| "/api/v2/sql".to_string(),
            |c| Some(json!({ "query": sql_vector_search(c) })),
        ),
        (
            "REST /api/v2/query federated",
            reqwest::Method::POST,
            |_| "/api/v2/query".to_string(),
            |c| Some(json!({ "language": "federated", "query": sql_vector_search(c) })),
        ),
        (
            "REST /api/v2/query uql",
            reqwest::Method::POST,
            |_| "/api/v2/query".to_string(),
            |c| {
                Some(json!({
                    "language": "uql",
                    "query": format!("SELECT * FROM vectors.{c} WHERE VECTOR_SIMILAR(embedding, ?, 0.0) LIMIT 5"),
                    "parameters": [vector()],
                }))
            },
        ),
        (
            "REST document ingest",
            reqwest::Method::POST,
            |c| format!("/api/v2/collections/{c}/documents"),
            |_| {
                Some(json!({
                    "documents": [{ "id": "planted-doc", "text": "planted text", "vector": vector() }]
                }))
            },
        ),
        (
            "REST discovery dedup create",
            reqwest::Method::POST,
            |c| format!("/api/v2/collections/{c}/discovery-jobs"),
            |_| Some(json!({ "kind": "dedup" })),
        ),
        (
            "REST discovery list",
            reqwest::Method::GET,
            |c| format!("/api/v2/collections/{c}/discovery-jobs"),
            |_| None,
        ),
        (
            "REST v1 entity upsert",
            reqwest::Method::POST,
            |c| format!("/api/v1/collections/{c}/entities"),
            |_| {
                Some(json!({ "entity": {
                    "id": "planted-entity",
                    "embeddings": [{ "model_id": "m", "modality": "Text", "dimension": 4,
                                     "vector": [0.0, 0.0, 0.0, 1.0] }]
                }}))
            },
        ),
        (
            "REST v1 entity list",
            reqwest::Method::GET,
            |c| format!("/api/v1/collections/{c}/entities"),
            |_| None,
        ),
        (
            "REST v1 entity get",
            reqwest::Method::GET,
            |c| format!("/api/v1/collections/{c}/entities/{GLOBEX}-r0"),
            |_| None,
        ),
        (
            "REST v1 entity delete",
            reqwest::Method::DELETE,
            |c| format!("/api/v1/collections/{c}/entities/{GLOBEX}-r1"),
            |_| None,
        ),
        (
            "REST hybrid index",
            reqwest::Method::POST,
            |_| "/api/v2/hybrid/index".to_string(),
            |c| {
                Some(json!({
                    "collection": c,
                    "documents": [{ "id": "planted-bm25", "text": "quarterly planted" }]
                }))
            },
        ),
    ]
}

/// Surfaces beyond vector search — SQL/federated/UQL queries, document
/// ingest, discovery jobs, legacy entities, BM25 indexing, and the Arrow
/// Flight actions — also resolve collections within the request tenant: a
/// foreign collection is answered exactly like a missing one, nothing in it
/// changes, and listings omit it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_collection_surface_is_scoped_to_the_request_tenant() {
    let server = Server::start().await.expect("server start");
    let rest = Rest {
        http: reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .no_proxy()
            .build()
            .expect("http client"),
        base: format!("http://127.0.0.1:{}", server.rest_port),
    };
    rest.create(Some(ACME), ACME_DOCS).await;
    let globex_id = rest.create(Some(GLOBEX), GLOBEX_DOCS).await;
    rest.insert(Some(ACME), ACME_DOCS, ACME).await;
    rest.insert(Some(GLOBEX), GLOBEX_DOCS, GLOBEX).await;
    sleep(Duration::from_millis(300)).await;

    let globex_snapshot = |rest: &Rest| {
        let http = rest.http.clone();
        let base = rest.base.clone();
        async move {
            let response = http
                .post(format!(
                    "{base}/api/v2/collections/{GLOBEX_DOCS}/records/scan"
                ))
                .header("X-Tenant-ID", GLOBEX)
                .json(&json!({ "limit": 100 }))
                .send()
                .await
                .expect("globex scan");
            let body: Value = response.json().await.unwrap_or(Value::Null);
            let mut ids: Vec<String> = body["records"]
                .as_array()
                .map(|records| {
                    records
                        .iter()
                        .filter_map(|r| r["id"].as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            ids.sort();
            ids
        }
    };
    let before = globex_snapshot(&rest).await;
    assert!(!before.is_empty(), "globex must see its own records");

    let mut failures: Vec<String> = Vec::new();

    for (label, method, path, body) in beyond_search_probes() {
        for (how, foreign, missing) in [
            ("by name", GLOBEX_DOCS, MISSING_DOCS),
            ("by object id", globex_id.as_str(), MISSING_OBJECT_ID),
        ] {
            let (s1, b1) = rest
                .call(Some(ACME), method.clone(), &path(foreign), body(foreign))
                .await;
            let (s2, b2) = rest
                .call(Some(ACME), method.clone(), &path(missing), body(missing))
                .await;
            let foreign_seen = normalise(s1.to_string(), &b1, foreign);
            let missing_seen = normalise(s2.to_string(), &b2, missing);
            // Document ingest auto-creates a missing collection, so for it the
            // check is that the foreign collection is neither served nor
            // written (the record snapshot below).
            if foreign_seen != missing_seen && label != "REST document ingest" {
                failures.push(format!(
                    "{label} {how}: foreign {foreign_seen:?}\n      vs missing {missing_seen:?}"
                ));
            }
            if s1.is_success() {
                failures.push(format!("{label} {how}: acme was served: {s1} {b1}"));
            }
            // A missing collection named by NAME is a client error. (A bare
            // numeric object id is not a valid UQL source name, so that
            // probe fails as a malformed query — identically for both.)
            if s1.is_server_error() && how == "by name" {
                failures.push(format!(
                    "{label} {how}: a missing collection must be a client error, got {s1} {b1}"
                ));
            }
        }
    }
    // The owner still gets served on the query surfaces.
    let (status, body) = rest
        .call(
            Some(GLOBEX),
            reqwest::Method::POST,
            "/api/v2/sql",
            Some(json!({ "query": sql_vector_search(GLOBEX_DOCS) })),
        )
        .await;
    if !status.is_success() {
        failures.push(format!("REST /api/v2/sql as owner: {status} {body}"));
    }

    // --- Arrow Flight actions.
    let channel =
        tonic::transport::Endpoint::from_shared(format!("http://127.0.0.1:{}", server.flight_port))
            .expect("flight endpoint")
            .connect()
            .await
            .expect("flight channel");
    let mut flight = FlightServiceClient::new(channel);
    let mut act = async |tenant: &str, kind: &str, body: Value, identifier: &str| -> Observed {
        let action = arrow_flight::Action {
            r#type: kind.to_string(),
            body: serde_json::to_vec(&body).unwrap_or_default().into(),
        };
        let outcome = match flight.do_action(with_tenant(action, tenant)).await {
            Ok(stream) => stream
                .into_inner()
                .try_collect::<Vec<_>>()
                .await
                .map(|results| {
                    results
                        .iter()
                        .map(|r| String::from_utf8_lossy(&r.body).into_owned())
                        .collect::<Vec<_>>()
                        .join("|")
                }),
            Err(status) => Err(status),
        };
        match outcome {
            Ok(served) => {
                let mut normalised = normalise("OK".to_string(), &served, identifier);
                normalised.status = "OK".to_string();
                normalised
            }
            Err(status) => Observed {
                status: format!("{:?}", status.code()),
                body: mask(status.message(), identifier),
            },
        }
    };
    let flight_actions: Vec<(&str, fn(&str) -> Value)> = vec![
        ("get_collection", |c| json!({ "collection_id": c })),
        (
            "get_vectors",
            |c| json!({ "collection_id": c, "vector_ids": [format!("{GLOBEX}-r0")] }),
        ),
        ("list_arrow_files", |c| json!({ "collection_id": c })),
        ("flush_collection", |c| json!({ "collection_id": c })),
        ("compact_collection", |c| json!({ "collection_id": c })),
        ("flush_and_compact", |c| json!({ "collection_id": c })),
        (
            "delete_vectors",
            |c| json!({ "collection_id": c, "vector_ids": [format!("{GLOBEX}-r2")] }),
        ),
        (
            "rank_features_export",
            |c| json!({ "collection": c, "query_vector": vector(), "k": 5 }),
        ),
        ("delete_collection", |c| json!({ "collection_id": c })),
    ];
    for (kind, body) in flight_actions {
        for (how, foreign, missing) in [
            ("by name", GLOBEX_DOCS, MISSING_DOCS),
            ("by object id", globex_id.as_str(), MISSING_OBJECT_ID),
        ] {
            let foreign_seen = act(ACME, kind, body(foreign), foreign).await;
            let missing_seen = act(ACME, kind, body(missing), missing).await;
            if foreign_seen != missing_seen {
                failures.push(format!(
                    "Flight {kind} {how}: foreign {foreign_seen:?}\n      vs missing {missing_seen:?}"
                ));
            }
        }
    }
    let listed = act(ACME, "list_collections", json!({}), "").await;
    if listed.body.contains(GLOBEX_DOCS) || !listed.body.contains(ACME_DOCS) {
        failures.push(format!("Flight list_collections as acme: {listed:?}"));
    }
    match flight
        .list_flights(with_tenant(arrow_flight::Criteria::default(), ACME))
        .await
    {
        Ok(stream) => {
            let infos = stream.into_inner().try_collect::<Vec<_>>().await;
            if format!("{infos:?}").contains(GLOBEX_DOCS) {
                failures.push("Flight list_flights as acme lists globex_docs".to_string());
            }
        }
        Err(status) => failures.push(format!("Flight list_flights as acme: {status:?}")),
    }

    // Nothing in globex's collection changed, and it still exists.
    sleep(Duration::from_secs(3)).await;
    let after = globex_snapshot(&rest).await;
    if after != before {
        failures.push(format!(
            "globex_docs records changed: before {before:?}, after {after:?}"
        ));
    }
    let (status, body) = rest
        .call(
            Some(GLOBEX),
            reqwest::Method::GET,
            &format!("/api/v2/collections/{GLOBEX_DOCS}"),
            None,
        )
        .await;
    if !status.is_success() {
        failures.push(format!(
            "globex_docs gone after acme's delete: {status} {body}"
        ));
    }
    let (status, body) = rest
        .call(
            Some(GLOBEX),
            reqwest::Method::POST,
            "/api/v2/hybrid/search",
            Some(json!({ "collection": GLOBEX_DOCS, "text_query": "quarterly", "vector": vector(), "top_k": 5 })),
        )
        .await;
    if body.contains("planted-bm25") {
        failures.push(format!(
            "BM25 index of globex_docs was written by acme: {status} {body}"
        ));
    }

    assert!(
        failures.is_empty(),
        "every collection surface must be scoped to the request tenant:\n  {}",
        failures.join("\n  ")
    );
}
