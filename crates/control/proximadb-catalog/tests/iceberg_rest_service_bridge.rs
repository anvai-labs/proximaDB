// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Integration test for `IcebergRestService` against the concrete
//! `IcebergObjectStoreBridge`.
//!
//! Lives in `tests/` (not the lib's `#[cfg(test)]` module) because
//! `proximadb-iceberg-engine` is a dev-dependency that itself depends on this
//! crate: inside a unit test, `crate::object_store_bridge::ObjectStoreBridge`
//! is a different compilation unit than the one the engine implements, so the
//! trait bound can only be satisfied from an external test target.

use std::sync::Arc;

use proximadb_catalog::object_store_bridge::{
    BridgeObjectPath, CommitOutcome, ObjectStoreBridge as _,
};
use proximadb_catalog::{
    CatalogColumn, CatalogManager, CatalogTableSchema, TableIdentifier,
    iceberg_rest_service::IcebergRestService,
};
use proximadb_data_model::ProximaType;
use proximadb_iceberg_engine::IcebergObjectStoreBridge;

#[tokio::test]
async fn ensure_table_metadata_materializes_history_from_manifest_log() {
    // In-memory warehouse with three committed (empty) data-manifest versions: 0,1,2.
    let bridge = IcebergObjectStoreBridge::from_url("memory://").expect("memory bridge");
    let base = "warehouse_tables/events";
    let manifest_prefix = format!("{base}/_manifests");
    let data_prefix = BridgeObjectPath::from(format!("{base}/data"));
    let mut parent = None;
    for _ in 0..3 {
        match bridge
            .publish_snapshot(&data_prefix, &manifest_prefix, parent)
            .await
            .expect("seed manifest")
        {
            CommitOutcome::Committed(v) => parent = Some(v),
            other => panic!("unexpected seed outcome: {other:?}"),
        }
    }

    let svc = IcebergRestService::new(
        Arc::new(CatalogManager::new()),
        "wh",
        "grpc://localhost:5680",
        "http://localhost:5678/iceberg/v1",
    )
    .with_object_store_bridge(Arc::new(bridge));

    let mut schema = CatalogTableSchema::new("events")
        .with_column(CatalogColumn::new(1, "id", ProximaType::String))
        .with_primary_key(vec!["id".to_string()]);
    schema.location = Some(base.to_string());
    let id = TableIdentifier::new(vec!["default".to_string()], "events".to_string());

    let (md, location) = svc
        .ensure_table_metadata(&id, &schema)
        .await
        .expect("ensure");

    // History reflects the manifest log: 3 parent-chained snapshots.
    assert_eq!(md.snapshots.len(), 3);
    assert_eq!(md.snapshots[0].parent_snapshot_id, None);
    assert_eq!(
        md.snapshots[2].parent_snapshot_id,
        Some(md.snapshots[1].snapshot_id)
    );
    assert_eq!(md.snapshots[2].sequence_number, 3);
    assert_eq!(md.current_snapshot_id, Some(md.snapshots[2].snapshot_id));
    assert_eq!(
        md.refs.get("main").expect("main ref").snapshot_id,
        md.snapshots[2].snapshot_id
    );
    assert!(
        location.ends_with(".metadata.json"),
        "location = {location}"
    );

    // Idempotent: a second materialization sees the persisted metadata is current and
    // returns the same history (no runaway metadata versions on repeated reads).
    let (md2, _) = svc
        .ensure_table_metadata(&id, &schema)
        .await
        .expect("ensure 2");
    assert_eq!(md2.snapshots.len(), 3);
    assert_eq!(md2.current_snapshot_id, md.current_snapshot_id);

    // Another table recorded at the same location is served its own metadata,
    // never the first table's.
    let mut other_schema = CatalogTableSchema::new("other")
        .with_column(CatalogColumn::new(1, "id", ProximaType::String))
        .with_primary_key(vec!["id".to_string()]);
    other_schema.location = Some(base.to_string());
    let other_id = TableIdentifier::new(vec!["default".to_string()], "other".to_string());
    let (other_md, other_location) = svc
        .ensure_table_metadata(&other_id, &other_schema)
        .await
        .expect("ensure other");
    assert_ne!(other_md.table_uuid, md.table_uuid);
    assert!(
        other_location.contains("/tables/other/"),
        "{other_location}"
    );
    let (again, _) = svc
        .ensure_table_metadata(&id, &schema)
        .await
        .expect("ensure 3");
    assert_eq!(again.table_uuid, md.table_uuid);
}
