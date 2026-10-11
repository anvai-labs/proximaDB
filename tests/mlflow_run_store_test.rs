//! TD-MLOPS-1 slice 1: the substrate-backed RunStore passes the same port
//! conformance battery as the in-memory reference (experiments, runs, params,
//! metrics history, tags, datasets, soft-delete/restore, finish-freeze) and
//! enforces structural tenant isolation.

use std::sync::Arc;

use proximadb::services::mlflow_run_store::SubstrateRunStore;
use proximadb::storage::document::{DocumentRecord, DocumentService};
use proximadb::storage::engines::sst::SstEngine;
use proximadb_catalog::run_store::{RunStore, RunStoreError};

#[tokio::test]
async fn substrate_run_store_passes_port_conformance() {
    let engine = Arc::new(SstEngine::new().await.unwrap());
    let document = Arc::new(DocumentService::new(engine));
    let store = SubstrateRunStore::for_tenant(document, "default").unwrap();

    proximadb_catalog::run_store::conformance_tests::port_conformance(&store).await;
}

#[tokio::test]
async fn concurrent_metric_appends_to_one_run_are_all_retained() {
    // TD-MLOPS-1 review round 1: interleaved writers at the same step must
    // not lose points — history length is exactly the number of appends and
    // ordering is stable (the store serializes mutations).
    use proximadb_catalog::run_store::{MetricPoint, RunStore};

    let engine = Arc::new(SstEngine::new().await.unwrap());
    let document = Arc::new(DocumentService::new(engine));
    let store = SubstrateRunStore::for_tenant(document, "default").unwrap();
    let exp = store
        .create_experiment("concurrent", None, Default::default())
        .await
        .unwrap();
    store
        .create_run(
            exp.experiment_id,
            "run-c",
            None,
            None,
            Default::default(),
            0,
        )
        .await
        .unwrap();

    let mut tasks = Vec::new();
    for i in 0..16 {
        let point = MetricPoint {
            key: "loss".to_string(),
            value: i as f64,
            timestamp_ms: 1_000 + i as i64,
            step: i / 4, // deliberately colliding steps
        };
        tasks.push(store.log_metric("run-c", point));
    }
    let results = futures::future::join_all(tasks).await;
    for r in &results {
        r.as_ref().unwrap();
    }
    let history = store.metric_history("run-c", "loss").await.unwrap();
    assert_eq!(history.len(), 16, "no append may be lost under concurrency");
    let mut sorted: Vec<f64> = history.iter().map(|p| p.value).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let expected: Vec<f64> = (0..16).map(|i| i as f64).collect();
    assert_eq!(sorted, expected, "all 16 distinct values retained");
}

#[tokio::test]
async fn substrate_run_store_isolates_tenants_structurally() {
    let engine = Arc::new(SstEngine::new().await.unwrap());
    let document = Arc::new(DocumentService::new(engine));
    let alpha = SubstrateRunStore::for_tenant(document.clone(), "alpha").unwrap();
    let beta = SubstrateRunStore::for_tenant(document, "beta").unwrap();

    let exp_a = alpha
        .create_experiment("shared-name", None, Default::default())
        .await
        .unwrap();

    // Same name is legal in the other tenant; ids are per-tenant sequences.
    let exp_b = beta
        .create_experiment("shared-name", None, Default::default())
        .await
        .unwrap();
    assert_eq!(exp_a.experiment_id, exp_b.experiment_id);

    // Cross-tenant run visibility: none.
    alpha
        .create_run(
            exp_a.experiment_id,
            "run-a",
            None,
            None,
            Default::default(),
            1,
        )
        .await
        .unwrap();
    beta.create_run(
        exp_b.experiment_id,
        "run-b",
        None,
        None,
        Default::default(),
        1,
    )
    .await
    .unwrap();
    let beta_runs = beta.list_runs(exp_b.experiment_id, true).await.unwrap();
    assert_eq!(beta_runs.len(), 1);
    assert_eq!(beta_runs[0].run_id, "run-b");
    assert!(beta.get_run("run-a").await.is_err());
    assert!(alpha.get_run("run-b").await.is_err());
}

#[tokio::test]
async fn existing_run_without_payload_is_reported_as_corrupt_not_unknown() {
    let engine = Arc::new(SstEngine::new().await.unwrap());
    let document = Arc::new(DocumentService::new(engine));
    let store = SubstrateRunStore::for_tenant(document.clone(), "default").unwrap();
    store
        .create_experiment("ensure-tracking-collection", None, Default::default())
        .await
        .unwrap();

    let corrupt = DocumentRecord::from_tree(
        "run-corrupt".to_string(),
        Default::default(),
        "mlflow_tracking_system".to_string(),
        None,
        Some("mlflow_tracking_system".to_string()),
    );
    document
        .insert_document_record("mlflow_tracking_system", corrupt)
        .await
        .unwrap();

    let error = store.get_run("corrupt").await.unwrap_err();
    assert!(
        matches!(error, RunStoreError::Internal { .. }),
        "existing corrupt records must not masquerade as unknown: {error:?}"
    );
}
