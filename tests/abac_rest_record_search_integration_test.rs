// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! TD-ABAC-5/6 (REST records slice): proves ABAC enforcement FIRES on the REST
//! record-search path (`search_records_with_tenant_context`), not just the
//! fusion/service seam.
//!
//! A `VectorOperationsService` is built with a durable ABAC enforcer; the REST
//! records-search entry point threads the request subject + tenant stable id into
//! `_inner`, which resolves the subject's read context and ANDs the security
//! predicate (`dept=eng`) into the search. So:
//! * an admitted subject (`alice`, dept=eng) sees only `eng` records — the `hr`
//!   record is filtered out;
//! * a denied subject (`mallory`, no attribute binding) gets an **empty** result
//!   (fail-closed);
//! * a `None` subject (the gRPC/Flight callers that still delegate with `None`)
//!   gets the unfiltered results (passthrough — today's behavior preserved).
//!
//! The binding is `Scope::Namespace(0)` because `resolve_vector_read_context`
//! builds `Target { namespace: 0, .. }` — a namespace-scoped permit matches
//! regardless of the collection's catalog object id (the test harness mints
//! name-shaped ids that don't parse as the numeric object id).
//!
//! The `v1_*` tests prove the same enforcement on the **v1 search path**
//! (`search_v1` — the entry the REST v1 + `/progressive/search` routes reach via
//! `ApiHandlersPort::handle_vector_search_v1_for_tenant` → `VectorOpsPort::search`;
//! progressive is a thin delegate to the same v1 handler, so one v1 test covers
//! both routes).
//!
//! Default-OFF: this file compiles only under `--features abac-policy`.

#![cfg(all(test, feature = "abac-policy"))]

#[path = "common/integration_test_helpers.rs"]
mod harness;

use harness::UnifiedTestEnvironment;
use proximadb::core::search::{ComparisonOperator, FilterExpression};
use proximadb::security::rls::AbacEnforcer;
use proximadb_abac::{
    InMemoryAttributeAuthority, InMemoryPolicyEpochs, InMemoryPredicateObjectStore,
};
use proximadb_catalog::fc_metamodel::{AttrValue, Effect, PolicyBinding, Scope};
use proximadb_data_model::ProximaValue;
use proximadb_records::{EmbeddingCell, ProximaRecord, ProximaTreeNode};
use proximadb_runtime::{PortIdentity, RichRecordGetRequest, RichSearchRequest};
use serde_json::json;
use std::sync::Arc;

const COLLECTION: &str = "abac_rest_record_search";
const DIM: u32 = 4;
const TENANT: u64 = 7;
const OID_ENG: &str = "abac/record/eng";
const OID_HR: &str = "abac/record/hr";

/// The port-seam identity for a named ABAC subject under [`TENANT`].
fn subject_identity(subject: &str) -> PortIdentity<'_> {
    PortIdentity {
        tenant_id: None,
        subject: Some(subject),
        tenant_stable_id: Some(TENANT),
        auth_class: proximadb_tenant::AuthClass::Authenticated,
    }
}

/// Admit `alice` (dept=eng) under a `dept=eng` row predicate, scoped to
/// `Namespace(0)` so it matches `resolve_vector_read_context`'s
/// `Target { namespace: 0, .. }` regardless of the collection's object id.
fn enforcer_for_alice() -> Arc<AbacEnforcer> {
    let mut authority = InMemoryAttributeAuthority::new();
    authority.upsert(
        proximadb_abac::AttributeBinding::new("alice", TENANT)
            .with_attr("dept", AttrValue::Str("eng".into())),
    );
    let mut store = InMemoryPredicateObjectStore::new();
    store.register(
        42,
        FilterExpression::Comparison {
            field: "dept".to_string(),
            operator: ComparisonOperator::Equals,
            value: json!("eng"),
        },
    );
    let bindings = vec![PolicyBinding {
        object_id: 1,
        tenant_stable_id: TENANT,
        scope: Scope::Namespace(0),
        effect: Effect::Permit,
        predicate_ref: Some(42),
        field_mask: None,
    }];
    Arc::new(
        AbacEnforcer::new(
            Arc::new(authority),
            Arc::new(store),
            Arc::new(InMemoryPolicyEpochs::new()),
        )
        .with_bindings(bindings),
    )
}

fn record(oid: &str, dept: &str, vector: Vec<f32>) -> ProximaRecord {
    let mut rec = ProximaRecord {
        oid: oid.to_string(),
        record_version: 1,
        embeddings: vec![EmbeddingCell::new_fp32("test", "dense_vector", DIM, vector)],
        ..Default::default()
    };
    rec.props.insert(
        "dept".to_string(),
        ProximaTreeNode::Value(ProximaValue::String(dept.to_string())),
    );
    rec
}

fn request() -> RichSearchRequest {
    RichSearchRequest {
        collection_id: COLLECTION.to_string(),
        query_vector: vec![1.0, 0.0, 0.0, 0.0],
        top_k: 10,
        filters: vec![],
    }
}

/// The proto-typed request the v1 REST + progressive routes carry.
fn v1_request() -> proximadb::proto::proximadb_v1::VectorSearchRequest {
    proximadb::proto::proximadb_v1::VectorSearchRequest {
        collection_id: COLLECTION.to_string(),
        queries: vec![proximadb::proto::proximadb_v1::SearchQuery {
            vector: vec![1.0, 0.0, 0.0, 0.0],
            filters: Default::default(),
            advanced_filter: None,
        }],
        top_k: 10,
        include_fields: None,
        search_params: None,
        distance_metric_override: None,
        search_optimization: None,
    }
}

fn v1_result_ids(resp: &proximadb::proto::proximadb_v1::VectorOperationResponse) -> Vec<String> {
    resp.results
        .as_ref()
        .map(|r| r.results.iter().map(|rec| rec.id.clone()).collect())
        .unwrap_or_default()
}

/// Build the service + collection + records once; the per-subject assertions
/// share it.
async fn fixture() -> (
    Arc<proximadb::services::VectorOperationsService>,
    Arc<proximadb::services::CollectionService>,
) {
    let env = UnifiedTestEnvironment::new().await.expect("test env");
    let (svc, collections) = env
        .vector_operations_service_with_abac(enforcer_for_alice())
        .await
        .expect("vector service with abac");
    UnifiedTestEnvironment::create_vector_collection(&collections, COLLECTION, DIM)
        .await
        .expect("create collection");
    svc.insert_vectors_direct(
        COLLECTION,
        Arc::new(vec![
            record(OID_ENG, "eng", vec![1.0, 0.0, 0.0, 0.0]),
            record(OID_HR, "hr", vec![0.99, 0.01, 0.0, 0.0]),
        ]),
    )
    .await
    .expect("insert records");
    (svc, collections)
}

/// An admitted subject sees only `dept=eng` records — the `hr` record is
/// filtered out by the pushed-down / post-filtered security predicate.
#[tokio::test]
async fn admitted_subject_sees_only_accessible_records_on_rest_records_path() {
    let (svc, _collections) = fixture().await;

    let resp = svc
        .search_records_with_tenant_context(request(), None, subject_identity("alice"))
        .await
        .expect("abac search");

    let ids: Vec<&str> = resp.results.iter().map(|r| r.id.as_str()).collect();
    assert!(
        ids.contains(&OID_ENG),
        "alice (dept=eng) must see the eng record; got {ids:?}"
    );
    assert!(
        !ids.contains(&OID_HR),
        "the hr record must be filtered out; got {ids:?}"
    );
}

/// A denied subject (no attribute binding) gets an EMPTY result — fail-closed.
#[tokio::test]
async fn denied_subject_gets_empty_results_on_rest_records_path() {
    let (svc, _collections) = fixture().await;

    let resp = svc
        .search_records_with_tenant_context(request(), None, subject_identity("mallory"))
        .await
        .expect("abac search");

    assert!(
        resp.results.is_empty(),
        "mallory (unbound) must be denied — empty results (fail-closed); got {}",
        resp.results.len()
    );
}

/// A `None` subject (the gRPC/Flight callers that still delegate with `None`)
/// gets the unfiltered results — today's behavior is preserved until those
/// surfaces adopt the `_abac` variant.
#[tokio::test]
async fn none_subject_is_passthrough_on_rest_records_path() {
    let (svc, _collections) = fixture().await;

    let resp = svc
        .search_records_with_tenant_context(request(), None, PortIdentity::anonymous())
        .await
        .expect("abac search");

    let ids: Vec<&str> = resp.results.iter().map(|r| r.id.as_str()).collect();
    assert!(
        ids.contains(&OID_ENG) && ids.contains(&OID_HR),
        "a None subject (gRPC/Flight path) must see both records (passthrough); got {ids:?}"
    );
}

// ── get-by-id path (`get_record_with_tenant_context`) ──
//
// A point lookup has no filter slot to push into, so enforcement is a post-check
// on the single fetched record: denied subject ⇒ None (fail-closed); admitted
// subject ⇒ the record only if it satisfies the security predicate; None subject
// ⇒ passthrough.

fn get_request(record_id: &str) -> RichRecordGetRequest {
    RichRecordGetRequest {
        collection_id: COLLECTION.to_string(),
        record_id: record_id.to_string(),
        include_vector: false,
        include_props: true,
    }
}

/// An admitted subject (dept=eng) may GET the eng record, NOT the hr record.
#[tokio::test]
async fn admitted_subject_gets_only_accessible_record_on_getbyid_path() {
    let (svc, _collections) = fixture().await;

    let eng = svc
        .get_record_with_tenant_context(get_request(OID_ENG), None, subject_identity("alice"))
        .await
        .expect("abac get");
    assert!(eng.is_some(), "alice (dept=eng) may GET the eng record");

    let hr = svc
        .get_record_with_tenant_context(get_request(OID_HR), None, subject_identity("alice"))
        .await
        .expect("abac get");
    assert!(
        hr.is_none(),
        "alice must NOT GET the hr record — it fails the dept=eng security predicate"
    );
}

/// A denied subject (no attribute binding) gets NONE — fail-closed.
#[tokio::test]
async fn denied_subject_gets_no_record_on_getbyid_path() {
    let (svc, _collections) = fixture().await;

    let resp = svc
        .get_record_with_tenant_context(get_request(OID_ENG), None, subject_identity("mallory"))
        .await
        .expect("abac get");
    assert!(
        resp.is_none(),
        "mallory (unbound) must be denied — no record (fail-closed)"
    );
}

/// A `None` subject (the gRPC/internal callers) gets the record — passthrough.
#[tokio::test]
async fn none_subject_is_passthrough_on_getbyid_path() {
    let (svc, _collections) = fixture().await;

    let resp = svc
        .get_record_with_tenant_context(get_request(OID_ENG), None, PortIdentity::anonymous())
        .await
        .expect("abac get");
    assert!(
        resp.is_some(),
        "a None subject (gRPC/internal path) may GET the record (passthrough)"
    );
}

/// v1/progressive path: an admitted subject sees only `dept=eng` records.
#[tokio::test]
async fn v1_admitted_subject_sees_only_accessible_records() {
    let (svc, _collections) = fixture().await;

    let resp = svc
        .search_v1(v1_request(), subject_identity("alice"))
        .await
        .expect("v1 abac search");

    let ids = v1_result_ids(&resp);
    assert!(
        ids.iter().any(|id| id == OID_ENG),
        "alice (dept=eng) must see the eng record on the v1 path; got {ids:?}"
    );
    assert!(
        !ids.iter().any(|id| id == OID_HR),
        "the hr record must be filtered out on the v1 path; got {ids:?}"
    );
}

/// v1/progressive path: a denied subject gets an EMPTY result — fail-closed.
#[tokio::test]
async fn v1_denied_subject_gets_empty_results() {
    let (svc, _collections) = fixture().await;

    let resp = svc
        .search_v1(v1_request(), subject_identity("mallory"))
        .await
        .expect("v1 abac search");

    let ids = v1_result_ids(&resp);
    assert!(
        ids.is_empty(),
        "mallory (unbound) must be denied on the v1 path — empty results (fail-closed); got {ids:?}"
    );
}

/// v1/progressive path: a `None` subject (internal callers) is passthrough.
#[tokio::test]
async fn v1_none_subject_is_passthrough() {
    let (svc, _collections) = fixture().await;

    let resp = svc
        .search_v1(v1_request(), PortIdentity::anonymous())
        .await
        .expect("v1 search");

    let ids = v1_result_ids(&resp);
    assert!(
        ids.iter().any(|id| id == OID_ENG) && ids.iter().any(|id| id == OID_HR),
        "a None subject must see both records on the v1 path (passthrough); got {ids:?}"
    );
}

/// Exercise lowered REST and raw federated SQL through the real query strategy,
/// scoped collection lookup, vector policy evaluator, and an enabled cache.
#[tokio::test]
async fn federated_reads_keep_the_policy_principal_and_skip_shared_results() {
    use axum::{Extension, body::Body, http::Request, http::StatusCode};
    use proximadb::network::rest::canonical::rank::{MockRangeCandidateProvider, RankServices};
    use proximadb::query::{
        FederatedQueryContext, QueryFacadeAdapter, SqlStrategy, UnifiedQueryFacade,
        UnifiedQueryPortImpl,
        cache::{FederatedQueryResultCache, query_result_cache::QueryResultCacheConfig},
    };
    use proximadb::storage::MultiModelStorageFacade;
    use proximadb_api::rest::{UnifiedQueryRestState, create_multimodal_router};
    use proximadb_tenant::{AuthClass, DEFAULT_TENANT, ResolvedRequestIdentity};
    use tower::ServiceExt;

    let (vectors, collections) = fixture().await;
    let cache = Arc::new(FederatedQueryResultCache::new(QueryResultCacheConfig {
        enable_cleanup: false,
        ..Default::default()
    }));
    let federated =
        FederatedQueryContext::with_cache(Arc::new(MultiModelStorageFacade::new()), cache.clone())
            .with_collection_port(collections)
            .with_vector_operations(vectors)
            .with_rank_services(Arc::new(RankServices::new(Arc::new(
                MockRangeCandidateProvider::default(),
            ))));
    let facade = Arc::new(UnifiedQueryFacade::with_strategies(vec![Arc::new(
        SqlStrategy::new(Arc::new(federated)),
    )]));
    let router = create_multimodal_router().with_state(UnifiedQueryRestState {
        unified_query_port: Arc::new(UnifiedQueryPortImpl::new(Arc::new(
            QueryFacadeAdapter::without_validation(facade),
        ))),
    });
    let request = |path: &'static str,
                   body: serde_json::Value,
                   subject: Option<&'static str>,
                   stable: Option<u64>| {
        let router = router.clone();
        async move {
            let identity = ResolvedRequestIdentity {
                tenant: DEFAULT_TENANT.to_owned(),
                subject: subject.map(str::to_owned),
                auth_class: if subject.is_some() {
                    AuthClass::Authenticated
                } else {
                    AuthClass::Anonymous
                },
                tenant_stable_id: stable,
            };
            let response = router
                .layer(Extension(identity))
                .oneshot(
                    Request::post(path)
                        .header("content-type", "application/json")
                        .body(Body::from(body.to_string()))
                        .expect("request"),
                )
                .await
                .expect("response");
            let status = response.status();
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("body");
            let body = serde_json::from_slice::<serde_json::Value>(&bytes).expect("JSON");
            (status, body)
        }
    };

    let lowered = json!({ "components": [{ "component_type": "vector", "config": {
        "collection": COLLECTION, "query_vector": [1.0, 0.0, 0.0, 0.0], "top_k": 10,
    }}] });
    let raw = json!({ "query": format!(
        "SELECT * FROM VECTOR_SEARCH('{}', '[1.0, 0.0, 0.0, 0.0]', 10)", COLLECTION,
    ) });
    let ids = |body: &serde_json::Value| -> Vec<String> {
        body["records"]
            .as_array()
            .expect("records")
            .iter()
            .map(|row| row["id"].as_str().expect("record ID").to_owned())
            .collect()
    };

    let (status, broad) = request("/multi-model", lowered.clone(), None, Some(TENANT)).await;
    assert_eq!(status, StatusCode::OK, "{broad}");
    assert_eq!(ids(&broad).len(), 2);
    let (status, hit) = request("/multi-model", lowered.clone(), None, Some(TENANT)).await;
    assert_eq!(status, StatusCode::OK, "{hit}");
    assert_eq!(ids(&hit), ids(&broad));
    let warm_cache = cache.stats();
    assert!(
        warm_cache.hits > 0,
        "the subjectless control must warm the shared cache"
    );

    let rerank = json!({ "query": format!(
        "SELECT * FROM RERANK('{}', 'q', '[1.0, 0.0, 0.0, 0.0]', 10)", COLLECTION,
    ) });
    for (path, query) in [
        ("/multi-model", lowered),
        ("/federated", raw),
        ("/federated", rerank),
    ] {
        for (subject, stable, expected) in [
            ("alice", Some(TENANT), vec![OID_ENG.to_owned()]),
            ("mallory", Some(TENANT), Vec::new()),
            ("alice", None, Vec::new()),
        ] {
            let (status, body) = request(path, query.clone(), Some(subject), stable).await;
            assert_eq!(status, StatusCode::OK, "{path} {subject}: {body}");
            assert_eq!(
                ids(&body),
                expected,
                "{path} {subject} (stable ID {stable:?})"
            );
        }
    }
    let after_subject_reads = cache.stats();
    assert_eq!(after_subject_reads.hits, warm_cache.hits);
    assert_eq!(after_subject_reads.misses, warm_cache.misses);
    assert_eq!(after_subject_reads.inserts, warm_cache.inserts);

    let unsupported = [
        (
            "/federated",
            json!({ "query": "SELECT * FROM GRAPH_QUERY('MATCH (n) RETURN n')" }),
        ),
        (
            "/multi-model",
            json!({ "components": [{ "component_type": "graph", "config": {
            "cypher": "MATCH (n) RETURN n",
        }}] }),
        ),
        (
            "/federated",
            json!({ "query": format!(
            "SELECT * FROM RERANK('{}', 'q', '[1.0, 0.0, 0.0, 0.0]', 10, 'profile')", COLLECTION,
        ) }),
        ),
    ];
    for (path, query) in unsupported {
        let (status, body) = request(path, query, Some("alice"), Some(TENANT)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {body}");
        assert!(
            body["error"]
                .as_str()
                .expect("policy error")
                .contains("subject-level read policies"),
            "{path}: {body}",
        );
    }
}

// ── The ONE composition rule (ADR-087 / TD-ABAC-10c) ────────────────────────
//
// `records_read_context` is the single definition of "what does this
// (subject, tenant_stable_id) get to read" — the REST/v1 seam AND the pgwire
// pgvector path both resolve through it, so its truth table is pinned here
// rather than re-asserted per surface. A surface that stops calling it is a
// bypass by construction.

/// Admitted subject ⇒ a CLIENT context (the security predicate is applied).
#[tokio::test]
async fn composition_rule_admits_a_bound_subject_as_client() {
    let (svc, _collections) = fixture().await;

    let context = svc
        .records_read_context(
            Some("alice"),
            Some(TENANT),
            proximadb_tenant::AuthClass::Authenticated,
            COLLECTION,
        )
        .await;

    assert!(
        matches!(context, Some(proximadb_abac::ReadContext::Client(_))),
        "alice (dept=eng) must resolve to a Client context; got {context:?}"
    );
}

/// Denied subject ⇒ `None`. THIS is the cell every caller must fail closed on:
/// the REST seam returns empty results, and pgwire's pgvector path now emits an
/// empty result set instead of the rows it used to serve unfiltered.
#[tokio::test]
async fn composition_rule_denies_an_unbound_subject_so_callers_fail_closed() {
    let (svc, _collections) = fixture().await;

    let context = svc
        .records_read_context(
            Some("mallory"),
            Some(TENANT),
            proximadb_tenant::AuthClass::Authenticated,
            COLLECTION,
        )
        .await;

    assert!(
        context.is_none(),
        "an unbound subject must DENY (None) so every caller fails closed; got {context:?}"
    );
}

/// No subject ⇒ System passthrough (internal/unauthenticated callers).
#[tokio::test]
async fn composition_rule_passes_through_when_there_is_no_client_subject() {
    let (svc, _collections) = fixture().await;

    let context = svc
        .records_read_context(
            None,
            None,
            proximadb_tenant::AuthClass::Anonymous,
            COLLECTION,
        )
        .await;

    assert!(
        matches!(context, Some(proximadb_abac::ReadContext::System(_))),
        "no subject ⇒ System passthrough; got {context:?}"
    );
}

/// TD-ABAC-11 strict cell: an armed enforcer cannot consult policy without the
/// tenant stable id, so a present subject MUST be denied.
#[tokio::test]
async fn composition_rule_denies_a_subject_without_a_stable_id() {
    let (svc, _collections) = fixture().await;

    let context = svc
        .records_read_context(
            Some("alice"),
            None,
            proximadb_tenant::AuthClass::Authenticated,
            COLLECTION,
        )
        .await;

    assert!(
        context.is_none(),
        "TD-ABAC-11: subject + absent stable id must deny; got {context:?}"
    );
}
