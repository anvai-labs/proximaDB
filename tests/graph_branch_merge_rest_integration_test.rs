//! T3.1 Slice 9 — REST router integration tests for the branch-merge endpoint.
//!
//! Drives `merge_graph_branch_inner` through a minimal axum Router via
//! `tower::ServiceExt::oneshot`. This is the first integration coverage of
//! `POST /api/v1/collections/:collection/branches/:branch/merge` that
//! exercises the full extractor / JSON / response-shape path — earlier
//! coverage either bypassed the router (Slice 8 WAL-roundtrip tests) or
//! tested data-layer helpers in isolation (handlers.rs#mod tests).
//!
//! ## Design
//!
//! The production handler (`merge_graph_branch`) takes `State<AppState>`,
//! which would require stubbing ~10 services to construct in a test.
//! Slice 9 refactored the handler into a thin `State<AppState>` shim over
//! a pure-logic `merge_graph_branch_inner(data_dir, collection, branch,
//! request)` function. This test mounts a minimal Router with
//! `State<PathBuf>` (data_dir only) on a test-local handler that forwards
//! into `merge_graph_branch_inner`. The pattern is reusable for future
//! REST endpoint integration tests.

use anyhow::Result;
use axum::body::Body;
use axum::extract::{Path as AxumPath, State};
use axum::http::{Request, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use proximadb::errors::ApiResult;
use proximadb::network::rest::handlers::{GraphBranchMergeRequest, merge_graph_branch_inner};
use proximadb::services::FramedTableWalAppender;
use proximadb::services::record_store::TableWalAppender;
use proximadb_records::ProximaRecord;
use proximadb_storage_common::CanonicalOperation;
use std::path::PathBuf;
use tempfile::TempDir;
use tokio::time::{Duration, sleep};
use tower::ServiceExt; // for `oneshot`

type JsonResponse<T> = axum::Json<T>;

const COLLECTION: &str = "col";
const BRANCH_A: &str = "a";
const BRANCH_B: &str = "b";
const EXPECTED_ORIGIN: &str = "branch_merge:a:b";

/// Build a minimal axum Router scoped to the merge endpoint. State is just
/// the data_dir PathBuf — no AppState needed.
fn router_for_test(data_dir: PathBuf) -> Router {
    Router::new()
        .route(
            "/api/v1/collections/{collection}/branches/{branch}/merge",
            post(test_merge_handler),
        )
        .with_state(data_dir)
}

/// Test-local extractor shim. Mirrors the production
/// `merge_graph_branch` shim but pulls data_dir from `State<PathBuf>`
/// instead of `State<AppState>`.
async fn test_merge_handler(
    State(data_dir): State<PathBuf>,
    AxumPath((collection, branch)): AxumPath<(String, String)>,
    Json(request): Json<GraphBranchMergeRequest>,
) -> ApiResult<JsonResponse<serde_json::Value>> {
    merge_graph_branch_inner(&data_dir, "default", &collection, &branch, request).await
}

/// Returns (TempDir guard, data_dir path). The TempDir must be held for the
/// lifetime of the test; dropping it cleans up the on-disk WAL.
fn fresh_data_dir() -> Result<(TempDir, PathBuf)> {
    let tmp = tempfile::tempdir()?;
    let data_dir = tmp.path().to_path_buf();
    Ok((tmp, data_dir))
}

/// Wal path the handler reads — mirrors `graph_branch_merge_wal_path`.
fn wal_path_for(data_dir: &PathBuf) -> PathBuf {
    data_dir.join("pgwire").join("canonical-records.wal")
}

fn upsert_op(oid: &str, branch: Option<&str>) -> CanonicalOperation {
    let mut record = ProximaRecord::default();
    record.oid = oid.to_string();
    record.branch_id = branch.map(String::from);
    CanonicalOperation::RecordUpsert {
        collection_id: COLLECTION.to_string(),
        record: Box::new(record),
        projections: vec![],
    }
}

/// Append `ops` to the canonical WAL and drop the appender so the next
/// reader sees a fully flushed file.
async fn append_and_drop(wal_path: &PathBuf, ops: Vec<CanonicalOperation>) -> Result<()> {
    let appender = FramedTableWalAppender::open(wal_path).await?;
    appender.append_operations(ops, None).await?;
    drop(appender);
    Ok(())
}

/// Seed a base + two divergent branch entries for the same OID. The right
/// branch is appended later → has a later timestamp_ms → wins LWW.
async fn seed_two_branch_conflict(wal_path: &PathBuf) -> Result<()> {
    append_and_drop(wal_path, vec![upsert_op("shared", None)]).await?;
    sleep(Duration::from_millis(8)).await;
    append_and_drop(wal_path, vec![upsert_op("shared", Some(BRANCH_A))]).await?;
    sleep(Duration::from_millis(8)).await;
    append_and_drop(wal_path, vec![upsert_op("shared", Some(BRANCH_B))]).await?;
    Ok(())
}

/// Send a JSON POST through the router via oneshot and return (status, body).
async fn post_json(
    app: Router,
    uri: &str,
    body_json: serde_json::Value,
) -> Result<(StatusCode, serde_json::Value)> {
    let request = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body_json)?))?;
    let response = app.oneshot(request).await?;
    let status = response.status();
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await?;
    let body: serde_json::Value =
        serde_json::from_slice(&body_bytes).unwrap_or(serde_json::Value::Null);
    Ok((status, body))
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Dry-run returns merge report without writing.
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn merge_endpoint_dry_run_returns_report_without_writing() -> Result<()> {
    let (_tmp, data_dir) = fresh_data_dir()?;
    let wal_path = wal_path_for(&data_dir);
    seed_two_branch_conflict(&wal_path).await?;

    let entries_before = FramedTableWalAppender::read_entries_from_path(&wal_path).await?;

    let app = router_for_test(data_dir.clone());
    let (status, body) = post_json(
        app,
        "/api/v1/collections/col/branches/a/merge",
        serde_json::json!({"target_branch": "b", "dry_run": true}),
    )
    .await?;

    assert_eq!(
        status,
        StatusCode::OK,
        "dry-run should return 200; body: {}",
        body
    );
    assert_eq!(body["dry_run"], serde_json::Value::Bool(true));
    assert_eq!(body["source_branch"], "a");
    assert_eq!(body["target_branch"], "b");
    assert!(body.get("merge_base_lsn").is_some());
    assert!(body.get("conflicts").is_some());
    assert!(body.get("resolutions").is_some());
    assert!(
        body["write_back"].is_null(),
        "dry-run write_back must be null; got {}",
        body["write_back"]
    );

    let entries_after = FramedTableWalAppender::read_entries_from_path(&wal_path).await?;
    assert_eq!(
        entries_after.len(),
        entries_before.len(),
        "dry-run must not append to the WAL"
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Commit (dry_run=false) writes through and response carries write_back.
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn merge_endpoint_commit_writes_through_and_returns_write_back_block() -> Result<()> {
    let (_tmp, data_dir) = fresh_data_dir()?;
    let wal_path = wal_path_for(&data_dir);
    seed_two_branch_conflict(&wal_path).await?;

    let entries_before = FramedTableWalAppender::read_entries_from_path(&wal_path).await?;

    let app = router_for_test(data_dir.clone());
    let (status, body) = post_json(
        app,
        "/api/v1/collections/col/branches/a/merge",
        serde_json::json!({"target_branch": "b", "dry_run": false}),
    )
    .await?;

    assert_eq!(
        status,
        StatusCode::OK,
        "commit should return 200; body: {}",
        body
    );
    assert_eq!(body["dry_run"], serde_json::Value::Bool(false));

    let write_back = &body["write_back"];
    assert!(
        write_back.is_object(),
        "commit write_back must be an object; got {}",
        write_back
    );
    let entries_written = write_back["entries_written"]
        .as_u64()
        .expect("entries_written must be a number");
    assert!(entries_written >= 1, "commit must write at least one entry");
    assert!(write_back["first_lsn"].is_number());
    assert!(write_back["last_lsn"].is_number());

    let entries_after = FramedTableWalAppender::read_entries_from_path(&wal_path).await?;
    assert_eq!(
        entries_after.len(),
        entries_before.len() + (entries_written as usize),
        "WAL must grow by entries_written"
    );

    // Every newly appended RecordUpsert must carry the merge origin.
    for entry in &entries_after[entries_before.len()..] {
        match &entry.operation {
            CanonicalOperation::RecordUpsert { record, .. } => {
                assert_eq!(
                    record.origin.as_deref(),
                    Some(EXPECTED_ORIGIN),
                    "merge upsert must carry origin = {EXPECTED_ORIGIN}"
                );
            }
            CanonicalOperation::RecordDelete { .. } => {
                // tombstones don't carry origin — accepted
            }
            other => panic!("unexpected operation in merge output: {:?}", other),
        }
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. 404 when the canonical WAL file is missing entirely.
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn merge_endpoint_returns_404_for_missing_wal_file() -> Result<()> {
    // Don't seed anything — data_dir/pgwire/canonical-records.wal won't exist.
    let (_tmp, data_dir) = fresh_data_dir()?;

    let app = router_for_test(data_dir.clone());
    let (status, body) = post_json(
        app,
        "/api/v1/collections/col/branches/a/merge",
        serde_json::json!({"target_branch": "b", "dry_run": true}),
    )
    .await?;

    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "missing WAL should return 404; body: {}",
        body
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. 400 when target_branch is empty.
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn merge_endpoint_returns_400_for_empty_target_branch() -> Result<()> {
    let (_tmp, data_dir) = fresh_data_dir()?;

    let app = router_for_test(data_dir.clone());
    let (status, body) = post_json(
        app,
        "/api/v1/collections/col/branches/a/merge",
        serde_json::json!({"target_branch": "", "dry_run": true}),
    )
    .await?;

    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "empty target_branch should return 400; body: {}",
        body
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. 404 when WAL exists but holds no branch entries for the requested pair.
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn merge_endpoint_returns_404_when_no_branch_entries_for_collection() -> Result<()> {
    let (_tmp, data_dir) = fresh_data_dir()?;
    let wal_path = wal_path_for(&data_dir);
    // Seed a record on the WRONG collection so the filter strips it.
    let mut record = ProximaRecord::default();
    record.oid = "shared".into();
    record.branch_id = Some("a".into());
    append_and_drop(
        &wal_path,
        vec![CanonicalOperation::RecordUpsert {
            collection_id: "other-collection".to_string(),
            record: Box::new(record),
            projections: vec![],
        }],
    )
    .await?;

    let app = router_for_test(data_dir.clone());
    let (status, body) = post_json(
        app,
        "/api/v1/collections/col/branches/a/merge",
        serde_json::json!({"target_branch": "b", "dry_run": true}),
    )
    .await?;

    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "collection with no matching branch entries should return 404; body: {}",
        body
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// TD-SEAM-12 hazard 2: truncated history refuses the merge (410), never a
// silently wrong merge base. A floor marker with no merge-base generation in
// the WAL is history truncated before merge bases were tracked: whether the
// collection's base survived cannot be told, so the merge refuses.
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn merge_endpoint_refuses_when_wal_history_was_truncated() -> Result<()> {
    let (_tmp, data_dir) = fresh_data_dir()?;
    let wal_path = wal_path_for(&data_dir);
    seed_two_branch_conflict(&wal_path).await?;
    let uri = format!("/api/v1/collections/{COLLECTION}/branches/{BRANCH_A}/merge");
    let body = serde_json::json!({ "target_branch": BRANCH_B, "dry_run": true });

    // Complete history: the merge runs.
    let (status, _) = post_json(router_for_test(data_dir.clone()), &uri, body.clone()).await?;
    assert_eq!(status, StatusCode::OK);

    // The durable floor marker says the main-line base write (seq 1) is gone,
    // and no merge-base generation recorded what it held.
    let mut floor = wal_path.as_os_str().to_owned();
    floor.push(".floor");
    std::fs::write(&floor, "1")?;

    let (status, response) = post_json(router_for_test(data_dir), &uri, body).await?;
    assert_eq!(
        status,
        StatusCode::GONE,
        "a merge base computed from truncated history must be refused: {response}"
    );
    assert!(
        response
            .to_string()
            .contains("before merge bases were tracked"),
        "refused because nothing tracked the truncated history: {response}"
    );
    Ok(())
}
