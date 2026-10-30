// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! FA-2 (abac-policy): proves vector-path ABAC enforcement FIRES end-to-end on
//! the real search kernel.
//!
//! A `VectorOperationsService` is built with a durable ABAC enforcer (the
//! composition wiring that `SharedServices` applies in production); an admitted
//! `Client` `ReadContext` — resolved from the subject via `resolve_read_context`
//! — is threaded into `unified_search_native_with_tenant_context`, the SAME seam
//! the fusion vector leg uses. The subject's security predicate (`dept=eng`) is
//! pushed DOWN into the ANN search, so an `eng` record is returned while an `hr`
//! record (though nearest to the query) is excluded. A `System` context leaves
//! the search untouched (normal search still works; enforcement is conditional
//! on a `Client` context).
//!
//! Default-OFF: this file compiles only under `--features abac-policy`.

#![cfg(all(test, feature = "abac-policy"))]

#[path = "common/integration_test_helpers.rs"]
mod harness;

use harness::UnifiedTestEnvironment;
use proximadb::core::search::{ComparisonOperator, FilterExpression};
use proximadb::security::rls::AbacEnforcer;
use proximadb_abac::{
    InMemoryAttributeAuthority, InMemoryPolicyEpochs, InMemoryPredicateObjectStore, ReadContext,
    SystemReadReason,
};
use proximadb_catalog::fc_metamodel::{AttrValue, Effect, PolicyBinding, Scope, SubjectId, Target};
use proximadb_data_model::ProximaValue;
use proximadb_records::{EmbeddingCell, EmbeddingValues, ProximaRecord, ProximaTreeNode};
use serde_json::json;
use std::sync::Arc;

const COLLECTION: &str = "abac_vector_pushdown";
const DIM: u32 = 4;
const TENANT: u64 = 7;
const NAMESPACE: u16 = 3;
const TABLE: u32 = 200;
const OID_ENG: &str = "abac/record/eng";
const OID_HR: &str = "abac/record/hr";

/// Build an enforcer that admits `alice` (dept=eng) with a `dept=eng` row
/// predicate (mirrors the `abac_adapter` unit-test fixture). `with_bindings`
/// installs the held binding set that `resolve_read_context` consults.
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
        scope: Scope::Table(TABLE),
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

/// An admitted subject sees only `dept=eng` records — the `hr` record, though
/// nearest to the query, is filtered out by the pushed-down security predicate.
#[tokio::test]
async fn admitted_subject_sees_only_accessible_records() {
    let env = UnifiedTestEnvironment::new().await.expect("test env");
    let enforcer = enforcer_for_alice();
    let (svc, collections) = env
        .vector_operations_service_with_abac(enforcer.clone())
        .await
        .expect("vector service with abac");
    UnifiedTestEnvironment::create_vector_collection(&collections, COLLECTION, DIM)
        .await
        .expect("create collection");

    // Both records sit nearest the query; without ABAC both would be returned.
    svc.insert_vectors_direct(
        COLLECTION,
        Arc::new(vec![
            record(OID_ENG, "eng", vec![1.0, 0.0, 0.0, 0.0]),
            record(OID_HR, "hr", vec![0.99, 0.01, 0.0, 0.0]),
        ]),
    )
    .await
    .expect("insert records");

    // Resolve alice → admitted Client ReadContext (the fusion vector-leg seam).
    let ctx = enforcer
        .resolve_read_context(
            &SubjectId("alice".into()),
            TENANT,
            Target {
                namespace: NAMESPACE,
                table: TABLE,
                column: None,
            },
        )
        .expect("alice is admitted");
    let read_ctx = ReadContext::Client(ctx);

    let results = svc
        .unified_search_native_with_tenant_context(
            COLLECTION,
            vec![1.0, 0.0, 0.0, 0.0],
            10,
            None,
            None,
            None,
            &read_ctx,
        )
        .await
        .expect("search");

    let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
    assert!(
        ids.contains(&OID_ENG),
        "alice (dept=eng) must see the eng record; got {ids:?}"
    );
    assert!(
        !ids.contains(&OID_HR),
        "the hr record must be filtered out by the security predicate; got {ids:?}"
    );
}

/// A `System` context (trusted-internal / unwired) leaves the search untouched —
/// both records are returned. Proves enforcement is conditional on a `Client`
/// context and that normal search is unaffected by the wired enforcer.
#[tokio::test]
async fn system_context_does_not_filter() {
    let env = UnifiedTestEnvironment::new().await.expect("test env");
    let enforcer = enforcer_for_alice();
    let (svc, collections) = env
        .vector_operations_service_with_abac(enforcer)
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

    let read_ctx = ReadContext::system(SystemReadReason::Statistics, "abac_system_test");
    let results = svc
        .unified_search_native_with_tenant_context(
            COLLECTION,
            vec![1.0, 0.0, 0.0, 0.0],
            10,
            None,
            None,
            None,
            &read_ctx,
        )
        .await
        .expect("search");

    let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
    assert!(
        ids.contains(&OID_ENG) && ids.contains(&OID_HR),
        "a System context must not filter — both records expected; got {ids:?}"
    );
}

/// FA-2 PR-D3: the centralized vector resolver (`resolve_vector_read_context`)
/// resolves the collection and builds the `Target` the enforcer matches. An
/// admitted subject ⇒ `Some(Ok(ctx))`; an unbound subject ⇒ `Some(Err(_))`
/// (deny — the caller fail-closes). Uses a namespace-scoped binding (matches the
/// resolver's `Target.namespace = 0`) so the assertion is independent of the
/// collection's generated object id.
#[tokio::test]
async fn resolver_denies_rather_than_degrading_to_unfiltered_when_it_cannot_resolve() {
    // REGRESSION: `resolve_vector_read_context` used to `?` out to `None` when
    // the collection would not load. `records_read_context` maps `None` to
    // `ReadContext::system` — an UNFILTERED read — so a failure to authorize
    // silently became unrestricted access to every row.
    //
    // A failure to resolve is a DENIAL, not an absence of policy. This asserts
    // the distinction that the `Option<Result<..>>` shape makes so easy to get
    // backwards: `Some(Err(..))` (deny), never `None` (ABAC off ⇒ unfiltered).
    let env = UnifiedTestEnvironment::new().await.expect("test env");
    let enforcer = Arc::new(
        AbacEnforcer::new(
            Arc::new(InMemoryAttributeAuthority::default()),
            Arc::new(InMemoryPredicateObjectStore::default()),
            Arc::new(InMemoryPolicyEpochs::new()),
        )
        .with_bindings(vec![]),
    );
    let (svc, _collections) = env
        .vector_operations_service_with_abac(enforcer)
        .await
        .expect("vector service with abac");

    // A collection that does not exist cannot be resolved.
    let outcome = svc
        .resolve_vector_read_context(&SubjectId("alice".into()), TENANT, "no-such-collection")
        .await;

    match outcome {
        Some(Err(_)) => {} // correct: an explicit denial
        None => panic!(
            "resolver returned None for an unresolvable collection — that maps to \
             ReadContext::system (UNFILTERED). A failure to authorize must DENY."
        ),
        Some(Ok(_)) => panic!("an unresolvable collection must not be admitted"),
    }
}

#[tokio::test]
async fn resolve_vector_read_context_admits_and_denies() {
    let env = UnifiedTestEnvironment::new().await.expect("test env");
    let mut authority = InMemoryAttributeAuthority::new();
    authority.upsert(
        proximadb_abac::AttributeBinding::new("alice", TENANT)
            .with_attr("dept", AttrValue::Str("eng".into())),
    );
    let bindings = vec![PolicyBinding {
        object_id: 1,
        tenant_stable_id: TENANT,
        scope: Scope::Namespace(0),
        effect: Effect::Permit,
        predicate_ref: None,
        field_mask: None,
    }];
    let enforcer = Arc::new(
        AbacEnforcer::new(
            Arc::new(authority),
            Arc::new(InMemoryPredicateObjectStore::new()),
            Arc::new(InMemoryPolicyEpochs::new()),
        )
        .with_bindings(bindings),
    );
    let (svc, collections) = env
        .vector_operations_service_with_abac(enforcer)
        .await
        .expect("vector service with abac");
    UnifiedTestEnvironment::create_vector_collection(&collections, COLLECTION, DIM)
        .await
        .expect("create collection");

    // alice (binding present) ⇒ admitted: the resolver found the collection and
    // built a Target the enforcer matched.
    let admitted = svc
        .resolve_vector_read_context(&SubjectId("alice".into()), TENANT, COLLECTION)
        .await;
    assert!(
        matches!(admitted, Some(Ok(_))),
        "alice must be admitted (Some(Ok))"
    );

    // mallory (no attribute binding) ⇒ denied — caller must fail-closed.
    let denied = svc
        .resolve_vector_read_context(&SubjectId("mallory".into()), TENANT, COLLECTION)
        .await;
    assert!(
        matches!(denied, Some(Err(_))),
        "mallory (unbound) must be denied (Some(Err))"
    );
}
