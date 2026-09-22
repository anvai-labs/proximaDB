use super::*;
#[cfg(feature = "abac-policy")]
use proximadb_abac::{ReadContext, SystemReadReason};

/// TD-OLAP-6: explicit `cluster_key` property wins; else the first
/// DATE/TIMESTAMP column; NULL-keyed rows sort last.
#[test]
fn cluster_key_resolution_and_sort_order() {
    fn col(name: &str, ty: ProximaType) -> CatalogColumn {
        CatalogColumn {
            id: 0,
            object_id: None,
            name: name.to_string(),
            data_type: ty,
            nullable: true,
            default_value: None,
            comment: None,
            properties: Default::default(),
            is_deleted: false,
            original_id: None,
        }
    }
    let mut schema = CatalogTableSchema {
        name: "t".to_string(),
        columns: vec![
            col("id", ProximaType::Int64),
            col("created", ProximaType::Date),
            col(
                "updated",
                ProximaType::Timestamp(proximadb_data_model::TimeUnit::Microsecond),
            ),
        ],
        ..Default::default()
    };
    // Heuristic: first temporal column.
    assert_eq!(resolve_cluster_key(&schema).as_deref(), Some("created"));
    // Explicit property wins.
    schema
        .properties
        .insert("cluster_key".to_string(), "id".to_string());
    assert_eq!(resolve_cluster_key(&schema).as_deref(), Some("id"));
    // Bogus property falls back to the heuristic.
    schema
        .properties
        .insert("cluster_key".to_string(), "nope".to_string());
    assert_eq!(resolve_cluster_key(&schema).as_deref(), Some("created"));

    let rec = |d: Option<i32>| {
        let mut r = ProximaRecord::default();
        if let Some(days) = d {
            r.props.insert(
                "created".to_string(),
                ProximaTreeNode::Value(ProximaValue::Date(days)),
            );
        }
        r
    };
    let mut records = [rec(Some(30)), rec(None), rec(Some(10)), rec(Some(20))];
    records.sort_by_cached_key(|r| cluster_sort_key(r, "created"));
    let keys: Vec<Option<i32>> = records
        .iter()
        .map(|r| match r.props.get("created") {
            Some(ProximaTreeNode::Value(ProximaValue::Date(d))) => Some(*d),
            _ => None,
        })
        .collect();
    assert_eq!(keys, vec![Some(10), Some(20), Some(30), None], "NULLs last");
}

/// TD-OLAP-2 (A2): `publish_ndv_statistics` (the materialize write-boundary
/// hook) folds `ProximaRecord`s into per-column HLL sketches and publishes them
/// to the ADR-037 registry keyed by table name; `statistics_from_splits` (the
/// DF reader's `statistics()`) then overlays `distinct_count = Inexact(ndv)`
/// from that registry. This proves the FULL A2 chain — the mechanism that lets
/// DataFusion's `EnforceDistribution` pick a partitioned (multi-core) join
/// build instead of the single-partition fallback when `distinct_count` is
/// `Absent` (the measured 100-700× gap vs DuckDB). Unlike
/// `schema_inference::statistics_from_splits_overlays_registry_ndv` (which
/// populates the registry by hand with json literals), this exercises the real
/// `ProximaValue`→json conversion + schema-driven column iteration that the
/// materialize path uses. Empty splits isolate the overlay (the footer-stats
/// block is skipped, but the registry overlay at schema_inference.rs:218 fires
/// regardless of split coverage).
#[test]
fn publish_ndv_statistics_feeds_distinct_count_overlay() {
    use crate::core::statistics::statistics_registry;
    use crate::datafusion::schema_inference::statistics_from_splits;
    use datafusion_common::stats::Precision;

    fn col(name: &str, ty: ProximaType) -> CatalogColumn {
        CatalogColumn {
            id: 0,
            object_id: None,
            name: name.to_string(),
            data_type: ty,
            nullable: true,
            default_value: None,
            comment: None,
            properties: Default::default(),
            is_deleted: false,
            original_id: None,
        }
    }
    let table = "ndv_publish_test_tbl";
    let schema = CatalogTableSchema {
        name: table.to_string(),
        columns: vec![
            col("id", ProximaType::Int64),
            col("flag", ProximaType::String),
            col("created", ProximaType::Date),
        ],
        ..Default::default()
    };
    // 100 rows: id NDV=100, flag NDV=2 (A/B alternating), created NDV=10 (cyclic).
    let mut records = Vec::new();
    for i in 0..100i64 {
        let mut r = ProximaRecord::default();
        r.props.insert(
            "id".to_string(),
            ProximaTreeNode::Value(ProximaValue::Int64(i)),
        );
        r.props.insert(
            "flag".to_string(),
            ProximaTreeNode::Value(ProximaValue::String(
                if i % 2 == 0 { "A" } else { "B" }.to_string(),
            )),
        );
        r.props.insert(
            "created".to_string(),
            ProximaTreeNode::Value(ProximaValue::Date((i % 10) as i32)),
        );
        records.push(r);
    }

    publish_ndv_statistics(table, &schema, &records);

    // Write side: the registry envelope carries per-column distinct_estimate.
    let env = statistics_registry()
        .envelope(table)
        .expect("summary published by materialize");
    assert_eq!(env.record_count, 100);
    let ndv = |name: &str| {
        env.fields
            .iter()
            .find(|f| f.name == name)
            .and_then(|f| f.distinct_estimate)
            .unwrap_or_else(|| panic!("field {name} has distinct_estimate"))
    };
    // HLL standard error ~2%; allow a band.
    assert!((ndv("id") as i64 - 100).abs() <= 5, "id ndv {}", ndv("id"));
    assert!(ndv("flag") <= 3, "flag ndv {}", ndv("flag"));
    assert!(
        (ndv("created") as i64 - 10).abs() <= 2,
        "created ndv {}",
        ndv("created")
    );

    // Read side: `statistics_from_splits` overlays distinct_count from the
    // registry — the exact seam the DF reader's `statistics()` calls.
    let arrow = schema.to_arrow_schema();
    let stats = statistics_from_splits(
        &[],
        arrow.as_ref(),
        Some(table),
        false,
        proximadb_data_model::StatsTrust::Trusted,
    );
    let col_ndv = |name: &str| {
        let idx = arrow.index_of(name).unwrap();
        match stats.column_statistics[idx].distinct_count {
            Precision::Inexact(n) => n,
            ref other => panic!("distinct_count for {name} not Inexact: {other:?}"),
        }
    };
    assert!((col_ndv("id") as i64 - 100).abs() <= 5);
    assert!(col_ndv("flag") <= 3);
    assert!((col_ndv("created") as i64 - 10).abs() <= 2);

    // Cleanup the process-global registry so this test is order-independent.
    statistics_registry().remove(table);
}

#[test]
fn resolve_materialize_prefix_drpath_and_cross_tenant() {
    use proximadb_catalog::StoragePoolClass;
    let pc = StoragePoolClass::default();
    let resolve =
        |nsid, owner| resolve_materialize_prefix("tnt_acme", nsid, owner, pc, "orders", None);

    // The single canonical DrPath layout (a real namespace_id).
    assert_eq!(resolve("ns_1", None).unwrap(), "data/tnt_acme/ns_1/orders");
    // Embedded / single-tenant uses the well-known ns_default — same layout, no
    // legacy fork (single-tenant is a degenerate multi-tenant).
    assert_eq!(
        resolve(DrPathBuilder::DEFAULT_NAMESPACE_ID, None).unwrap(),
        "data/tnt_acme/ns_default/orders"
    );
    // Cross-tenant (namespace owned by another tenant) → refused.
    assert!(resolve("ns_1", Some("tnt_globex")).is_err());
    // Injection in ANY segment is refused — `build_from_parts` validates the
    // tenant, namespace_id, AND table (the guard the removed manual validator gave).
    assert!(resolve_materialize_prefix("tnt_acme", "..", None, pc, "orders", None).is_err());
    assert!(resolve_materialize_prefix("tnt_acme", "ns_1", None, pc, "bad/name", None).is_err());
    assert!(resolve_materialize_prefix("..", "ns_1", None, pc, "orders", None).is_err());
}

/// ADR-031: the materialized object-path segment is the stable object_id when oid
/// paths are on AND the table carries one, else the bare name — the unit that makes
/// a rename a metadata-only op and keeps the path rename-safe. Pure (gate passed
/// in), so no process-env coupling.
#[test]
fn materialize_path_segment_prefers_object_id_when_enabled() {
    // No object_id → always the name, regardless of the gate.
    assert_eq!(materialize_path_segment("orders", None, false), "orders");
    assert_eq!(materialize_path_segment("orders", None, true), "orders");
    // With an object_id: gate off → name; gate on → the decimal id, never the
    // mutable name (so RENAME TABLE never orphans the materialized snapshot).
    assert_eq!(
        materialize_path_segment("orders", Some(42), false),
        "orders"
    );
    assert_eq!(materialize_path_segment("orders", Some(42), true), "42");
}

use crate::cluster::partition_lease::{
    DmlLockScope, DmlLockService, LockOutcome, PartitionLeaseManager, PartitionLeaseStore,
};
use crate::cluster::primary_pod_registry::PrimaryPodRegistry;
use crate::query::table_write_executor::PlannedOnlyTableWriteExecutor;
use crate::services::operations::batch_result::OperationMetrics;
use crate::services::record_store::{
    TableRecordGetResponse, TableRecordScanRequest, TableRecordScanResponse, TableRecordWriteResult,
};
use crate::services::{DdlService, DdlStatement};
use proximadb_catalog::{
    CatalogColumn, CatalogProjection, CatalogProjectionKind, CatalogStorageSpecialization,
    CatalogWorkloadProfile,
};
use proximadb_iceberg_engine::IcebergObjectStoreBridge;
use proximadb_object_store::ProximaObjectStore;

struct ExplainOnlyRecordStore;

#[async_trait::async_trait]
impl TableRecordStore for ExplainOnlyRecordStore {
    async fn write_mutations(
        &self,
        _table_schema: &CatalogTableSchema,
        mutations: Vec<TableRecordMutation>,
        _tenant_context: Option<&crate::storage::tenant::context::TenantContext>,
    ) -> Result<TableRecordWriteResult> {
        Ok(TableRecordWriteResult {
            success: true,
            record_ids: mutations
                .into_iter()
                .map(|mutation| mutation.record.oid)
                .collect(),
            metrics: OperationMetrics::default(),
            errors: Vec::new(),
            error_code: None,
        })
    }

    async fn get_by_key(
        &self,
        _table_schema: &CatalogTableSchema,
        _request: TableRecordGetRequest,
        _tenant_context: Option<&crate::storage::tenant::context::TenantContext>,
        #[cfg(feature = "abac-policy")] _read_context: &proximadb_abac::ReadContext,
    ) -> Result<TableRecordGetResponse> {
        Ok(None)
    }

    async fn scan_records(
        &self,
        _table_schema: &CatalogTableSchema,
        _request: TableRecordScanRequest,
        _tenant_context: Option<&crate::storage::tenant::context::TenantContext>,
        #[cfg(feature = "abac-policy")] _read_context: &proximadb_abac::ReadContext,
    ) -> Result<TableRecordScanResponse> {
        Ok(Vec::new())
    }
}

fn update_test_schema() -> CatalogTableSchema {
    CatalogTableSchema::new("agent_store")
        .with_column(CatalogColumn::new(1, "record_id", ProximaType::String).nullable(false))
        .with_column(CatalogColumn::new(2, "name", ProximaType::String).nullable(false))
        .with_column(
            CatalogColumn::new(3, "payload", ProximaType::Json).with_default("'{}'::jsonb"),
        )
        .with_column(CatalogColumn::new(4, "notes", ProximaType::String))
        .with_primary_key(vec!["record_id".to_string()])
}

/// A7: the production wiring (SharedServices) builds the lease stack from an
/// object-store URL via `PartitionLeaseStore::from_url` — the same path used
/// with `storage_config.metadata_url`. Verify that construction path yields a
/// working DmlLockService wired into DmlService (cross-pod coordination on).
#[tokio::test]
async fn dml_lock_service_buildable_from_url_like_production() -> Result<()> {
    let dir = tempfile::tempdir().expect("tempdir");
    let url = format!("file://{}", dir.path().display());
    let store = PartitionLeaseStore::from_url(&url, "_operator/leases")?;
    let lease_manager = Arc::new(PartitionLeaseManager::new(
        Arc::new(store),
        Arc::new(PrimaryPodRegistry::new()),
        "pod-prod",
        10_000,
    ));
    let lock_service = Arc::new(DmlLockService::new(lease_manager, "pod-prod"));
    let dml = DmlService::with_record_store_and_table_write_executor(
        Arc::new(CatalogManager::new()),
        Arc::new(ExplainOnlyRecordStore),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    )
    .with_dml_lock_service(lock_service);
    let tenant = TenantContext::for_tenant_id("tenant_a");
    let table_id =
        TableIdentifier::new(vec!["tenant_a".to_string(), "default".to_string()], "users");
    let guard = dml
        .acquire_table_dml_lock(&table_id, Some(&tenant), LockIntent::Write)
        .await?
        .expect("wired DmlService should acquire the table lock");
    assert_eq!(guard.lease_generation(), 1);
    guard.release().await;
    Ok(())
}

#[tokio::test]
async fn dml_service_table_lock_uses_resolved_tenant_namespace() -> Result<()> {
    let backing: Arc<dyn object_store::ObjectStore> =
        Arc::new(object_store::memory::InMemory::new());
    let lease_store = Arc::new(PartitionLeaseStore::new(
        ProximaObjectStore::new(backing),
        "_operator/leases",
    ));
    let lease_manager = Arc::new(PartitionLeaseManager::new(
        lease_store,
        Arc::new(PrimaryPodRegistry::new()),
        "pod-1",
        10_000,
    ));
    let lock_service = Arc::new(DmlLockService::new(lease_manager, "pod-1"));

    let dml = DmlService::with_record_store_and_table_write_executor(
        Arc::new(CatalogManager::new()),
        Arc::new(ExplainOnlyRecordStore),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    )
    .with_dml_lock_service(lock_service.clone());

    let tenant = TenantContext::for_tenant_id("tenant_a");
    let table_id =
        TableIdentifier::new(vec!["tenant_a".to_string(), "default".to_string()], "users");
    let guard = dml
        .acquire_table_dml_lock(&table_id, Some(&tenant), LockIntent::Write)
        .await?
        .expect("lock guard");
    assert_eq!(guard.lease_generation(), 1);

    let schema_scope = DmlLockScope::Schema {
        schema_name: "default".to_string(),
    };
    let conflict = lock_service
        .acquire_dml_lock(
            "tenant_a",
            Some("default"),
            &schema_scope,
            LockIntent::Write,
            1,
        )
        .await?;
    assert!(matches!(conflict, LockOutcome::Conflict));

    guard.release().await;
    let after_release = lock_service
        .acquire_dml_lock(
            "tenant_a",
            Some("default"),
            &schema_scope,
            LockIntent::Write,
            2,
        )
        .await?;
    assert!(matches!(after_release, LockOutcome::Acquired { .. }));

    Ok(())
}

#[test]
fn test_resolve_select_projection_uses_catalog_columns() {
    let schema = update_test_schema();

    let projected = DmlService::resolve_select_projection(
        &schema,
        &["record_id".to_string(), "payload".to_string()],
    )
    .expect("projection should resolve");

    assert_eq!(
        projected
            .iter()
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>(),
        vec!["record_id", "payload"]
    );
    assert!(DmlService::resolve_select_projection(&schema, &["missing".to_string()]).is_err());
}

#[test]
fn test_project_select_rows_uses_catalog_primary_key_props_and_vectors() {
    let schema = CatalogTableSchema::new("items")
        .with_column(CatalogColumn::new(1, "id", ProximaType::Int32))
        .with_column(CatalogColumn::new(2, "name", ProximaType::String))
        .with_column(CatalogColumn::new(3, "active", ProximaType::Boolean))
        .with_column(CatalogColumn::new(
            4,
            "embedding",
            ProximaType::DenseVector {
                element: proximadb_data_model::VectorElement::Float32,
                dim: 0,
            },
        ))
        .with_primary_key(vec!["id".to_string()]);
    let selected_columns = schema.columns.clone();
    let record = ProximaRecord {
        oid: "7".to_string(),
        props: proximadb_records::ProximaTree::from([
            (
                "name".to_string(),
                ProximaTreeNode::Value(ProximaValue::String("alice".to_string())),
            ),
            (
                "active".to_string(),
                ProximaTreeNode::Value(ProximaValue::Boolean(true)),
            ),
        ]),
        embeddings: vec![EmbeddingCell {
            model_id: "default".to_string(),
            modality: "vector".to_string(),
            dim: 2,
            values: proximadb_records::EmbeddingValues::Fp32(vec![0.1, 0.2]),
            ..Default::default()
        }],
        ..Default::default()
    };

    let rows = DmlService::project_select_rows(&[record], &schema, &selected_columns)
        .expect("row projection should succeed");

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], ProximaValue::Int32(7));
    assert_eq!(rows[0][1], ProximaValue::String("alice".to_string()));
    assert_eq!(rows[0][2], ProximaValue::Boolean(true));
    assert_eq!(rows[0][3], ProximaValue::DenseVector(vec![0.1, 0.2]));
}

#[test]
fn test_select_route_metadata_carries_catalog_read_route() {
    let mut schema = CatalogTableSchema::new("items")
        .with_column(CatalogColumn::new(1, "id", ProximaType::Int32))
        .with_column(CatalogColumn::new(2, "name", ProximaType::String))
        .with_primary_key(vec!["id".to_string()])
        .with_workload_profile(CatalogWorkloadProfile::Oltp)
        .with_storage_specialization(CatalogStorageSpecialization::PaxOltp);
    schema
        .properties
        .insert("policy_boundary".to_string(), "engine-enforced".to_string());
    let predicates = [RelationalSelectPredicate {
        column: schema.columns[0].clone(),
        condition: RelationalSelectPredicateCondition::Comparison {
            operator: RelationalSelectPredicateOperator::Equal,
            literal: "7".to_string(),
        },
    }];

    let route = DmlService::select_route_metadata(
        &schema,
        &schema.columns,
        predicates.len(),
        Some(1),
        RelationalSelectAccessPath::PrimaryKeyLookup,
    );

    assert_eq!(
        route.access_path,
        RelationalSelectAccessPath::PrimaryKeyLookup
    );
    assert_eq!(route.authority_mode, "ProximaAuthoritative");
    assert_eq!(route.workload_profile, "oltp");
    assert_eq!(route.storage_specialization, "pax_oltp");
    assert_eq!(route.policy_boundary, "engine-enforced");
    assert_eq!(route.predicate_count, 1);
    assert_eq!(route.projected_column_count, 2);
    assert_eq!(route.limit, Some(1));
}

#[test]
fn test_dml_result_success() {
    let result = DmlResult::success(5, "Operation completed");
    assert!(result.success);
    assert_eq!(result.rows_affected, 5);
}

#[test]
fn test_sql_value_literal_types() {
    let null = SqlValueLiteral::Null;
    let bool_val = SqlValueLiteral::Boolean(true);
    let int_val = SqlValueLiteral::Integer(42);
    let _float_val = SqlValueLiteral::Float(3.5);
    let _string_val = SqlValueLiteral::String("hello".to_string());
    let _array_val = SqlValueLiteral::Array(vec![
        SqlValueLiteral::Float(1.0),
        SqlValueLiteral::Float(2.0),
    ]);

    match null {
        SqlValueLiteral::Null => (),
        _ => panic!("Expected Null"),
    }
    match bool_val {
        SqlValueLiteral::Boolean(true) => (),
        _ => panic!("Expected Boolean(true)"),
    }
    match int_val {
        SqlValueLiteral::Integer(42) => (),
        _ => panic!("Expected Integer(42)"),
    }
}

#[test]
fn test_comparison_operators() {
    let _eq = ComparisonOperator::Equal;
    let _ne = ComparisonOperator::NotEqual;
    let _lt = ComparisonOperator::LessThan;
    let _gt = ComparisonOperator::GreaterThan;
}

#[test]
fn test_where_clause() {
    let wc = WhereClause {
        conditions: vec![Condition::Comparison {
            column: "id".to_string(),
            operator: ComparisonOperator::Equal,
            value: SqlValueLiteral::String("test123".to_string()),
        }],
        operator: LogicalOperator::And,
    };

    assert_eq!(wc.conditions.len(), 1);
}

#[test]
fn test_parse_jsonb_default_literal() {
    let literal = DmlService::parse_default_literal("'{}'::jsonb").unwrap();
    match literal {
        SqlValueLiteral::Json(value) => {
            assert_eq!(value, serde_json::json!({}));
        }
        other => panic!("expected JSON default literal, got {other:?}"),
    }
}

#[test]
fn test_parse_default_literal_unescapes_sql_string() {
    let literal = DmlService::parse_default_literal("'agent''s note'").unwrap();
    match literal {
        SqlValueLiteral::String(value) => {
            assert_eq!(value, "agent's note");
        }
        other => panic!("expected string default literal, got {other:?}"),
    }
}

#[test]
fn test_update_assignment_validation_rejects_primary_key_change() {
    let err = DmlService::validate_update_assignments(
        &[(
            "record_id".to_string(),
            SqlValueLiteral::String("r2".to_string()),
        )],
        &update_test_schema(),
    )
    .unwrap_err();

    assert!(err.to_string().contains("cannot modify primary key"));
}

#[test]
fn test_update_assignment_validation_rejects_null_for_not_null_column() {
    let err = DmlService::validate_update_assignments(
        &[("name".to_string(), SqlValueLiteral::Null)],
        &update_test_schema(),
    )
    .unwrap_err();

    assert!(err.to_string().contains("cannot be NULL"));
}

#[test]
fn test_update_assignment_validation_accepts_default_with_catalog_default() {
    DmlService::validate_update_assignments(
        &[("payload".to_string(), SqlValueLiteral::Default)],
        &update_test_schema(),
    )
    .unwrap();
}

#[test]
fn test_update_assignment_validation_rejects_default_without_catalog_default() {
    let err = DmlService::validate_update_assignments(
        &[("notes".to_string(), SqlValueLiteral::Default)],
        &update_test_schema(),
    )
    .unwrap_err();

    assert!(err.to_string().contains("has no DEFAULT"));
}

#[test]
fn test_delete_tombstone_record_uses_catalog_primary_key_shape() {
    let record = DmlService::build_delete_tombstone_record("r1", &update_test_schema(), 123)
        .expect("delete tombstone should build");

    assert_eq!(record.oid, "r1");
    assert_eq!(record.local_id.as_deref(), Some("r1"));
    assert_eq!(record.variation_id.as_deref(), Some("agent_store"));
    assert_eq!(record.created_at_ns, 123);
    assert_eq!(record.updated_at_ns, 123);
    assert_eq!(record.valid_to_ns, Some(0));
    assert_eq!(record.origin.as_deref(), Some("delete"));
    assert!(record.embeddings.is_empty());
}

#[test]
fn test_mutation_methods_distinguish_insert_and_upsert() {
    assert_eq!(
        DmlService::mutation_method(RelationalMutationKind::Insert),
        "sql_insert"
    );
    assert_eq!(
        DmlService::mutation_method(RelationalMutationKind::Upsert),
        "sql_upsert"
    );
}

#[test]
fn test_row_dml_write_intent_routes_mutations_to_wal_lane() {
    let schema = update_test_schema();
    for operation in [
        WriteOperationKind::Insert,
        WriteOperationKind::Upsert,
        WriteOperationKind::Update,
        WriteOperationKind::Delete,
    ] {
        let (intent, decision) = DmlService::route_row_dml_write_intent(&schema, operation, 3);

        assert_eq!(intent.durability, WriteDurabilityRequirement::WalRequired);
        assert_eq!(format!("{:?}", decision.lane), "WalCurrentState");
    }
}

#[test]
fn test_update_reconstructs_catalog_validated_row_shape() {
    let existing = RichSearchResult {
        id: "r1".to_string(),
        score: 1.0,
        similarity: None,
        vector: Vec::new(),
        props: HashMap::from([
            (
                "name".to_string(),
                ProximaValue::String("before".to_string()),
            ),
            ("notes".to_string(), ProximaValue::String("old".to_string())),
        ]),
        version: Some(7),
        timestamp: None,
        source: None,
    };
    let schema = update_test_schema();
    let mut row_values = DmlService::row_values_from_existing(&existing, &schema)
        .expect("existing row should map to catalog values");
    row_values.insert(
        "notes".to_string(),
        ProximaValue::String("updated".to_string()),
    );

    let row = CatalogRow::validate(&schema, row_values, &RelationalWriteProfile::oltp())
        .expect("updated row should validate");
    let record = row
        .to_mutation_record(
            &schema,
            RelationalMutationKind::Update,
            RelationalRecordOptions {
                method: Some("sql_dml_update".to_string()),
                record_version: Some(8),
                ..RelationalRecordOptions::default()
            },
        )
        .expect("updated row should project");

    assert_eq!(record.oid, "r1");
    assert_eq!(record.record_version, 8);
    assert_eq!(record.method.as_deref(), Some("sql_dml_update"));
    assert_eq!(
        proximadb_records::tree_get(&record.props, "notes"),
        Some(&ProximaValue::String("updated".to_string()))
    );
}

#[tokio::test]
async fn test_explain_table_write_returns_route_explanation() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");

    let ddl = DdlService::new(manager.clone());
    ddl.execute(DdlStatement::CreateNamespace {
        namespace: vec!["default".to_string()],
        if_not_exists: true,
        properties: HashMap::new(),
    })
    .await
    .expect("create namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let statement = parser
        .parse_ddl("CREATE TABLE staging (id TEXT NOT NULL, payload JSONB);")
        .expect("parse ddl")
        .expect("ddl statement");
    ddl.execute(statement).await.expect("execute ddl");

    let mut facts_schema = CatalogTableSchema::new("facts")
        .with_column(CatalogColumn::new(1, "id", ProximaType::String).nullable(false))
        .with_column(CatalogColumn::new(2, "payload", ProximaType::Json))
        .with_workload_profile(CatalogWorkloadProfile::Olap)
        .with_storage_specialization(CatalogStorageSpecialization::ColumnarAnalytics)
        .with_projection(
            CatalogProjection::rebuildable(
                "facts_iceberg_publication",
                CatalogProjectionKind::Columnar,
                "primary",
            )
            .with_bounded_lag(5_000)
            .with_lineage("wal:1..42", "wal:42")
            .with_policy_and_gate("engine-enforced", "projection-publication-smoke"),
        );
    facts_schema
        .properties
        .insert("compute_route".to_string(), "datafusion-local".to_string());
    facts_schema
        .properties
        .insert("freshness_sla".to_string(), "5s".to_string());

    let (catalog, table_id) = manager.resolve_table("facts").await.expect("resolve facts");
    catalog
        .create_table(&table_id, facts_schema)
        .await
        .expect("create facts schema with projection metadata");

    for (table_name, stats) in [
        (
            "staging",
            CatalogTableStatistics {
                row_count: 1_000,
                size_bytes: 512_000,
                file_count: 1,
                ..Default::default()
            },
        ),
        (
            "facts",
            CatalogTableStatistics {
                row_count: 10_000,
                size_bytes: 4_000_000,
                file_count: 4,
                ..Default::default()
            },
        ),
    ] {
        let (catalog, table_id) = manager
            .resolve_table(table_name)
            .await
            .expect("resolve table for stats");
        catalog
            .update_statistics(&table_id, stats)
            .await
            .expect("update stats");
    }

    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        Arc::new(ExplainOnlyRecordStore),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );
    let statement = parser
        .parse_dml("INSERT INTO facts SELECT * FROM staging;")
        .expect("parse dml")
        .expect("dml statement");

    let explanation = dml
        .explain_table_write(statement)
        .await
        .expect("explain route");

    assert_eq!(explanation.target_table, "facts");
    assert_eq!(explanation.selected_backend, "DataFusionLocal");
    assert_eq!(explanation.route_metadata.workload_profile, "olap");
    assert_eq!(
        explanation.route_metadata.storage_specialization,
        "columnar_analytics"
    );
    assert_eq!(
        explanation
            .route_metadata
            .preferred_compute_route
            .as_deref(),
        Some("datafusion-local")
    );
    assert_eq!(
        explanation.route_metadata.freshness_sla.as_deref(),
        Some("5s")
    );
    assert_eq!(
        explanation
            .route_metadata
            .projection_freshness_state
            .as_deref(),
        Some("Fresh")
    );
    assert_eq!(explanation.route_metadata.projection_metadata.len(), 1);
    let projection = &explanation.route_metadata.projection_metadata[0];
    assert_eq!(projection.name, "facts_iceberg_publication");
    assert_eq!(projection.kind, "Columnar");
    assert_eq!(projection.rebuild_source, "primary");
    assert_eq!(projection.freshness, "BoundedLag");
    assert_eq!(projection.freshness_state, "Fresh");
    assert_eq!(projection.max_lag_ms, Some(5_000));
    assert_eq!(projection.source_range.as_deref(), Some("wal:1..42"));
    assert_eq!(projection.last_included_position.as_deref(), Some("wal:42"));
    assert_eq!(
        projection.policy_boundary.as_deref(),
        Some("engine-enforced")
    );
    assert_eq!(
        projection.benchmark_gate.as_deref(),
        Some("projection-publication-smoke")
    );
    assert_eq!(explanation.data_movement.source_rows, Some(1_000));
    assert_eq!(explanation.data_movement.source_bytes, Some(512_000));
    assert_eq!(
        explanation.data_movement.target_bytes_before_write,
        Some(4_000_000)
    );
    assert!(
        explanation
            .candidate_paths
            .iter()
            .any(|path| path.backend == "DataFusionLocal")
    );
}

#[tokio::test]
async fn object_store_bridge_insert_select_executes_through_datafusion_route() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");

    let ddl = DdlService::new(manager.clone());
    ddl.execute(DdlStatement::CreateNamespace {
        namespace: vec!["default".to_string()],
        if_not_exists: true,
        properties: HashMap::new(),
    })
    .await
    .expect("create namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    for ddl_sql in [
        "CREATE TABLE staging (id TEXT NOT NULL, amount INTEGER NOT NULL, PRIMARY KEY (id));",
        "CREATE TABLE facts (id TEXT NOT NULL, amount INTEGER NOT NULL, PRIMARY KEY (id))
             WITH (
                workload = 'olap',
                layout = 'columnar',
                compute_route = 'datafusion-local',
                freshness_sla = '5s'
             );",
    ] {
        let statement = parser
            .parse_ddl(ddl_sql)
            .expect("parse ddl")
            .expect("ddl statement");
        ddl.execute(statement).await.expect("execute ddl");
    }

    let bridge: Arc<dyn ObjectStoreBridge> =
        Arc::new(IcebergObjectStoreBridge::from_url("memory://").expect("object bridge"));
    let iceberg_store: Arc<dyn TableRecordStore> =
        Arc::new(ObjectStoreIcebergRecordStore::new(bridge.clone()));
    let vector_store: Arc<dyn TableRecordStore> =
        Arc::new(ObjectStoreVectorRecordStore::new(bridge.clone()));
    let routed_store: Arc<dyn TableRecordStore> = Arc::new(CatalogRoutingTableRecordStore::new(
        iceberg_store,
        vector_store.clone(),
        vector_store,
    ));
    let source_reader = Arc::new(TableRecordStoreSourceReader::new(routed_store.clone()));
    let table_write_executor = Arc::new(
        DataFusionTableWriteExecutor::new(source_reader, routed_store.clone())
            .with_object_store_bridge(bridge),
    );
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager,
        routed_store,
        table_write_executor,
    );

    let insert = dml
        .execute(DmlStatement::Insert {
            table_name: "staging".to_string(),
            columns: vec!["id".to_string(), "amount".to_string()],
            values: vec![
                vec![
                    SqlValueLiteral::String("s1".to_string()),
                    SqlValueLiteral::Integer(42),
                ],
                vec![
                    SqlValueLiteral::String("s2".to_string()),
                    SqlValueLiteral::Integer(77),
                ],
            ],
        })
        .await
        .expect("insert source rows");
    assert_eq!(insert.rows_affected, 2);

    let statement = parser
        .parse_dml("INSERT INTO facts SELECT * FROM staging;")
        .expect("parse dml")
        .expect("dml statement");
    let copy = dml.execute(statement).await.expect("execute insert select");

    assert_eq!(copy.rows_affected, 2);
    assert!(copy.message.contains("DataFusionLocal"));

    let (_schema, mut records) = dml
        .scan_table_records("facts", None)
        .await
        .expect("scan target rows");
    records.sort_by(|left, right| left.oid.cmp(&right.oid));
    assert_eq!(
        records
            .iter()
            .map(|record| record.oid.as_str())
            .collect::<Vec<_>>(),
        vec!["s1", "s2"]
    );
    let amounts = records
        .iter()
        .map(|record| proximadb_records::tree_get(&record.props, "amount"))
        .collect::<Vec<_>>();
    assert_eq!(
        amounts,
        vec![
            Some(&ProximaValue::Int32(42)),
            Some(&ProximaValue::Int32(77))
        ]
    );
}

#[tokio::test]
async fn explain_insert_values_returns_native_oltp_route() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    DdlService::new(manager.clone())
        .execute(DdlStatement::CreateNamespace {
            namespace: vec!["default".to_string()],
            if_not_exists: true,
            properties: HashMap::new(),
        })
        .await
        .expect("namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
        .parse_ddl("CREATE TABLE orders (id TEXT NOT NULL, amount FLOAT);")
        .expect("parse ddl")
        .expect("ddl");
    DdlService::new(manager.clone())
        .execute(ddl_stmt)
        .await
        .expect("create table");

    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        Arc::new(ExplainOnlyRecordStore),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );
    let stmt = parser
        .parse_dml("INSERT INTO orders (id, amount) VALUES ('r1', 9.99);")
        .expect("parse dml")
        .expect("dml stmt");

    let explanation = dml
        .explain_table_write(stmt)
        .await
        .expect("explain values insert");

    assert_eq!(explanation.target_table, "orders");
    assert_eq!(explanation.selected_backend, "Native");
    // Default table (no WITH options) gets the htap workload profile.
    assert!(
        explanation.route_metadata.workload_profile == "htap"
            || explanation.route_metadata.workload_profile == "oltp",
        "unexpected workload_profile: {}",
        explanation.route_metadata.workload_profile
    );
    assert!(
        explanation.write_lane.contains("Wal"),
        "expected WAL lane, got {:?}",
        explanation.write_lane
    );
}

#[tokio::test]
async fn explain_update_and_delete_return_routes() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    DdlService::new(manager.clone())
        .execute(DdlStatement::CreateNamespace {
            namespace: vec!["default".to_string()],
            if_not_exists: true,
            properties: HashMap::new(),
        })
        .await
        .expect("namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
        .parse_ddl("CREATE TABLE accounts (id TEXT NOT NULL, balance FLOAT);")
        .expect("parse ddl")
        .expect("ddl");
    DdlService::new(manager.clone())
        .execute(ddl_stmt)
        .await
        .expect("create table");

    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        Arc::new(ExplainOnlyRecordStore),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );

    let update_stmt = parser
        .parse_dml("UPDATE accounts SET balance = 100.0 WHERE id = 'a1';")
        .expect("parse dml")
        .expect("update stmt");
    let update_explanation = dml
        .explain_table_write(update_stmt)
        .await
        .expect("explain update");
    assert_eq!(update_explanation.target_table, "accounts");
    assert_eq!(update_explanation.selected_backend, "Native");

    let delete_stmt = parser
        .parse_dml("DELETE FROM accounts WHERE id = 'a1';")
        .expect("parse dml")
        .expect("delete stmt");
    let delete_explanation = dml
        .explain_table_write(delete_stmt)
        .await
        .expect("explain delete");
    assert_eq!(delete_explanation.target_table, "accounts");
    assert_eq!(delete_explanation.selected_backend, "Native");
}

/// TD-110: VALUES DML remains on the native WAL/row-delta route even when
/// the target is an OLAP-profile table with a preferred DataFusion route.
/// DataFusion is for analytical reads/transforms and OLAP publication, not
/// the direct authority for row-level PostgreSQL-style writes.
#[tokio::test]
async fn explain_values_insert_to_olap_table_stays_native() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    DdlService::new(manager.clone())
        .execute(DdlStatement::CreateNamespace {
            namespace: vec!["default".to_string()],
            if_not_exists: true,
            properties: HashMap::new(),
        })
        .await
        .expect("namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
        .parse_ddl(
            "CREATE TABLE metrics (id TEXT NOT NULL, value FLOAT)
                 WITH (
                     workload = 'olap',
                     layout = 'columnar',
                     compute_route = 'datafusion-local',
                     freshness_sla = '10s'
                 );",
        )
        .expect("parse ddl")
        .expect("ddl");
    DdlService::new(manager.clone())
        .execute(ddl_stmt)
        .await
        .expect("create olap table");

    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        Arc::new(ExplainOnlyRecordStore),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );
    let stmt = parser
        .parse_dml("INSERT INTO metrics (id, value) VALUES ('m1', 42.0);")
        .expect("parse dml")
        .expect("dml stmt");

    let explanation = dml
        .explain_table_write(stmt)
        .await
        .expect("explain olap insert values");

    assert_eq!(explanation.target_table, "metrics");
    assert_eq!(
        explanation.selected_backend, "Native",
        "VALUES DML should commit through WAL/row-delta before OLAP publication"
    );
    assert_eq!(explanation.route_metadata.workload_profile, "olap");
    assert_eq!(
        explanation.route_metadata.storage_specialization,
        "columnar_analytics"
    );
    assert_eq!(
        explanation
            .route_metadata
            .preferred_compute_route
            .as_deref(),
        Some("datafusion-local")
    );
    assert!(
        explanation
            .rejected_paths
            .iter()
            .any(|path| path.backend == "DataFusionLocal"
                && path.reason.contains("row/delta commit path")),
        "DataFusion rejection should explain the TD-110 row/delta gate: {:?}",
        explanation.rejected_paths
    );
    assert!(
        explanation.write_lane.contains("Wal"),
        "VALUES DML should remain WAL-backed, got {:?}",
        explanation.write_lane
    );
}

/// End-to-end smoke test: `DmlService` with a `DirectWalTableRecordStore` performs a
/// VALUES INSERT and a primary-key SELECT through the canonical WAL + memtable path,
/// then replays the WAL into a fresh memtable to verify durability.
#[tokio::test]
async fn direct_record_storage_insert_select_and_wal_replay() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("dml-smoke.wal");

    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");

    let ddl = DdlService::new(manager.clone());
    ddl.execute(DdlStatement::CreateNamespace {
        namespace: vec!["default".to_string()],
        if_not_exists: true,
        properties: HashMap::new(),
    })
    .await
    .expect("create namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let create_sql =
        "CREATE TABLE users (id TEXT NOT NULL, email TEXT NOT NULL, age INT, PRIMARY KEY (id));";
    let ddl_stmt = parser
        .parse_ddl(create_sql)
        .expect("parse create table")
        .expect("ddl statement");
    ddl.execute(ddl_stmt).await.expect("create table");

    // Wire DmlService over the canonical direct WAL writer.
    let wal_appender = Arc::new(
        FramedTableWalAppender::open(&wal_path)
            .await
            .expect("open WAL"),
    );
    let record_storage = Arc::new(MemtableRecordStorage::new());
    let record_store = Arc::new(DirectWalTableRecordStore::new(
        record_storage.clone(),
        wal_appender.clone(),
    ));
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store,
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );

    // INSERT via SQL VALUES DML.
    let insert_sql = "INSERT INTO users (id, email, age) VALUES ('u1', 'alice@example.com', 30);";
    let stmt = parser
        .parse_dml(insert_sql)
        .expect("parse insert")
        .expect("dml statement");
    let result = dml.execute(stmt).await.expect("execute insert");
    assert!(result.success, "INSERT must succeed");
    assert_eq!(result.rows_affected, 1);

    // SELECT via DmlService's relational scan/projection path — verifies current-state is
    // visible through the canonical record store.
    let sel = dml
        .select_table_records_with_projection(
            "users",
            &["id".to_string(), "email".to_string(), "age".to_string()],
            None,
            &[],
            None,
        )
        .await
        .expect("select users");
    assert_eq!(sel.rows.len(), 1, "SELECT must return the inserted row");
    assert_eq!(
        sel.rows[0][0],
        ProximaValue::String("u1".to_string()),
        "id column must match"
    );
    assert_eq!(
        sel.rows[0][1],
        ProximaValue::String("alice@example.com".to_string()),
        "email column must match"
    );

    // Replay WAL into a fresh memtable — verifies Layer 0 durability.
    let replay_storage = Arc::new(MemtableRecordStorage::new());
    let replay_wal = FramedTableWalAppender::open(&wal_path)
        .await
        .expect("reopen WAL for replay");
    let entries = replay_wal.read_entries().await.expect("read WAL entries");
    assert!(!entries.is_empty(), "WAL must contain at least one entry");
    let summary = replay_storage
        .replay_wal_entries(entries)
        .await
        .expect("replay WAL");
    assert_eq!(
        summary.upserts_replayed, 1,
        "WAL replay must recover the INSERT as an upsert"
    );
    assert_eq!(
        replay_storage.len(),
        1,
        "replayed memtable must hold the record"
    );
}

/// CDC change-feed (P2): INSERT → UPDATE → DELETE produce ordered change rows over the
/// canonical WAL; `changes_since` filters by table + lsn and reports correct op tags.
#[tokio::test]
async fn cdc_change_feed_reports_ops_since_lsn() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("cdc.wal");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    let ddl = DdlService::new(manager.clone());
    ddl.execute(DdlStatement::CreateNamespace {
        namespace: vec!["default".to_string()],
        if_not_exists: true,
        properties: HashMap::new(),
    })
    .await
    .expect("create namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
        .parse_ddl("CREATE TABLE acct (id TEXT NOT NULL, bal INT, PRIMARY KEY (id));")
        .expect("parse create")
        .expect("ddl");
    ddl.execute(ddl_stmt).await.expect("create table");

    let wal_appender = Arc::new(
        FramedTableWalAppender::open(&wal_path)
            .await
            .expect("open WAL"),
    );
    let record_store = Arc::new(DirectWalTableRecordStore::new(
        Arc::new(MemtableRecordStorage::new()),
        wal_appender,
    ));
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store,
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );

    let run = |sql: &str| {
        let parser = &parser;
        let dml = &dml;
        let sql = sql.to_string();
        async move {
            let stmt = parser.parse_dml(&sql).expect("parse dml").expect("dml");
            dml.execute(stmt).await.expect("execute dml")
        }
    };
    run("INSERT INTO acct (id, bal) VALUES ('a', 100);").await;
    run("UPDATE acct SET bal = 150 WHERE id = 'a';").await;
    run("DELETE FROM acct WHERE id = 'a';").await;

    // Full feed from the beginning: upsert, upsert (update), delete — lsn-ordered.
    let changes = dml.changes_since("acct", 0).await.expect("changes");
    assert_eq!(changes.len(), 3, "three changes: insert, update, delete");
    assert_eq!(changes[0].op, "upsert");
    assert_eq!(changes[1].op, "upsert");
    assert_eq!(changes[2].op, "delete");
    assert!(
        changes[0].lsn < changes[1].lsn && changes[1].lsn < changes[2].lsn,
        "changes are lsn-ordered"
    );
    assert_eq!(changes[2].key, "a", "delete carries the key");

    // Incremental: only changes after the first lsn (the two later ops).
    let tail = dml
        .changes_since("acct", changes[0].lsn)
        .await
        .expect("tail");
    assert_eq!(tail.len(), 2, "since first lsn → update + delete only");

    // A different table sees nothing.
    assert!(
        dml.changes_since("other", 0)
            .await
            .expect("other")
            .is_empty()
    );
}

#[tokio::test]
async fn insert_rejects_duplicate_primary_key() {
    // TD-110 Slice B: a second INSERT with the same primary key (and a duplicate within one
    // INSERT) is rejected; a distinct key still succeeds.
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    let ddl = DdlService::new(manager.clone());
    ddl.execute(DdlStatement::CreateNamespace {
        namespace: vec!["default".to_string()],
        if_not_exists: true,
        properties: HashMap::new(),
    })
    .await
    .expect("create namespace");
    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
        .parse_ddl("CREATE TABLE users (id TEXT NOT NULL, email TEXT NOT NULL, PRIMARY KEY (id));")
        .expect("parse create")
        .expect("ddl");
    ddl.execute(ddl_stmt).await.expect("create table");

    let wal_appender = Arc::new(
        FramedTableWalAppender::open(temp_dir.path().join("pk.wal"))
            .await
            .expect("open WAL"),
    );
    let record_store = Arc::new(DirectWalTableRecordStore::new(
        Arc::new(MemtableRecordStorage::new()),
        wal_appender,
    ));
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store,
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );

    let insert = |sql: &'static str| {
        let p = &parser;

        p.parse_dml(sql).expect("parse dml").expect("dml")
    };

    // First insert succeeds.
    dml.execute(insert(
        "INSERT INTO users (id, email) VALUES ('u1', 'a@x.com');",
    ))
    .await
    .expect("first insert");

    // Re-inserting the same PK is rejected against the committed row.
    let err = dml
        .execute(insert(
            "INSERT INTO users (id, email) VALUES ('u1', 'b@x.com');",
        ))
        .await
        .expect_err("duplicate PK must be rejected");
    assert!(
        err.to_string()
            .contains("duplicate key value violates primary key"),
        "unexpected error: {err}"
    );

    // A duplicate PK within a single INSERT is also rejected.
    let err = dml
        .execute(insert(
            "INSERT INTO users (id, email) VALUES ('u2', 'c@x.com'), ('u2', 'd@x.com');",
        ))
        .await
        .expect_err("within-batch duplicate PK must be rejected");
    assert!(
        err.to_string().contains("appears more than once"),
        "unexpected error: {err}"
    );

    // A distinct key still inserts.
    dml.execute(insert(
        "INSERT INTO users (id, email) VALUES ('u3', 'e@x.com');",
    ))
    .await
    .expect("distinct key insert");
}

/// F2 (ADR-072 D3/D5): with a fenced `ConditionalKeyStore` wired via
/// `with_conditional_key_store`, a duplicate PK is rejected by the store's
/// `put_if_absent` (a live holder) rather than the `get_by_key` probe.
#[cfg(feature = "oltp-integrity")]
#[tokio::test]
async fn insert_rejects_duplicate_primary_key_via_fenced_store() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};
    use proximadb_cks_local::{LocalWalKeyStore, SyncPolicy};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    let ddl = DdlService::new(manager.clone());
    ddl.execute(DdlStatement::CreateNamespace {
        namespace: vec!["default".to_string()],
        if_not_exists: true,
        properties: HashMap::new(),
    })
    .await
    .expect("create namespace");
    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
        .parse_ddl("CREATE TABLE users (id TEXT NOT NULL, email TEXT NOT NULL, PRIMARY KEY (id));")
        .expect("parse create")
        .expect("ddl");
    ddl.execute(ddl_stmt).await.expect("create table");

    let wal_appender = Arc::new(
        FramedTableWalAppender::open(temp_dir.path().join("pk.wal"))
            .await
            .expect("open WAL"),
    );
    let record_store = Arc::new(DirectWalTableRecordStore::new(
        Arc::new(MemtableRecordStorage::new()),
        wal_appender,
    ));
    let cks = Arc::new(
        LocalWalKeyStore::open(temp_dir.path().join("cks.wal"), SyncPolicy::PerOp)
            .expect("open cks"),
    );
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store,
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    )
    .with_conditional_key_store(cks);

    let insert = |sql: &'static str| {
        let p = &parser;
        p.parse_dml(sql).expect("parse dml").expect("dml")
    };

    dml.execute(insert(
        "INSERT INTO users (id, email) VALUES ('u1', 'a@x.com');",
    ))
    .await
    .expect("first insert");

    let err = dml
        .execute(insert(
            "INSERT INTO users (id, email) VALUES ('u1', 'b@x.com');",
        ))
        .await
        .expect_err("duplicate PK must be rejected by the fenced store");
    assert!(
        err.to_string()
            .contains("duplicate key value violates primary key"),
        "unexpected error: {err}"
    );

    // A distinct key still inserts through the fenced path.
    dml.execute(insert(
        "INSERT INTO users (id, email) VALUES ('u3', 'e@x.com');",
    ))
    .await
    .expect("distinct key insert");
}

#[tokio::test]
async fn insert_rejects_duplicate_unique_constraint() {
    // TD-110: a non-PK UNIQUE column rejects a duplicate value against a
    // committed row AND within one INSERT; NULL tuples are exempt (multiple
    // NULLs allowed); distinct values still insert.
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    let ddl = DdlService::new(manager.clone());
    ddl.execute(DdlStatement::CreateNamespace {
        namespace: vec!["default".to_string()],
        if_not_exists: true,
        properties: HashMap::new(),
    })
    .await
    .expect("create namespace");
    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
            .parse_ddl(
                "CREATE TABLE members (id TEXT NOT NULL, email TEXT, PRIMARY KEY (id), UNIQUE (email));",
            )
            .expect("parse create")
            .expect("ddl");
    ddl.execute(ddl_stmt).await.expect("create table");

    let wal_appender = Arc::new(
        FramedTableWalAppender::open(temp_dir.path().join("unique.wal"))
            .await
            .expect("open WAL"),
    );
    let record_store = Arc::new(DirectWalTableRecordStore::new(
        Arc::new(MemtableRecordStorage::new()),
        wal_appender,
    ));
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store,
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );

    let insert = |sql: &'static str| parser.parse_dml(sql).expect("parse dml").expect("dml");

    // First insert succeeds.
    dml.execute(insert(
        "INSERT INTO members (id, email) VALUES ('m1', 'a@x.com');",
    ))
    .await
    .expect("first insert");

    // A different PK but duplicate UNIQUE email is rejected against the committed row.
    let err = dml
        .execute(insert(
            "INSERT INTO members (id, email) VALUES ('m2', 'a@x.com');",
        ))
        .await
        .expect_err("duplicate UNIQUE email must be rejected");
    assert!(
        err.to_string()
            .contains("duplicate key value violates unique constraint"),
        "unexpected error: {err}"
    );

    // A duplicate UNIQUE value within a single INSERT is also rejected.
    let err = dml
        .execute(insert(
            "INSERT INTO members (id, email) VALUES ('m3', 'b@x.com'), ('m4', 'b@x.com');",
        ))
        .await
        .expect_err("within-batch duplicate UNIQUE value must be rejected");
    assert!(
        err.to_string().contains("appears more than once"),
        "unexpected error: {err}"
    );

    // NULL UNIQUE tuples are exempt — multiple NULL emails are allowed.
    dml.execute(insert("INSERT INTO members (id) VALUES ('m5');"))
        .await
        .expect("first NULL email insert");
    dml.execute(insert("INSERT INTO members (id) VALUES ('m6');"))
        .await
        .expect("second NULL email allowed (NULLs exempt from UNIQUE)");

    // A distinct UNIQUE value still inserts.
    dml.execute(insert(
        "INSERT INTO members (id, email) VALUES ('m7', 'c@x.com');",
    ))
    .await
    .expect("distinct UNIQUE value insert");

    // Slice-C increment: a multi-row INSERT where ONE row (amid non-colliding
    // rows) duplicates a committed value is rejected by the single batch scan.
    let err = dml
            .execute(insert(
                "INSERT INTO members (id, email) VALUES ('m8', 'fresh1@x.com'), ('m9', 'c@x.com'), ('m10', 'fresh2@x.com');",
            ))
            .await
            .expect_err("a colliding row anywhere in the batch must be rejected");
    assert!(
        err.to_string()
            .contains("duplicate key value violates unique constraint"),
        "unexpected error: {err}"
    );

    // And a fully-distinct multi-row batch still inserts.
    dml.execute(insert(
        "INSERT INTO members (id, email) VALUES ('m11', 'd@x.com'), ('m12', 'e@x.com');",
    ))
    .await
    .expect("fully-distinct batch insert");
}

#[tokio::test]
async fn unique_index_frees_value_on_delete_and_update() {
    // TD-110 Slice C: the store-layer UNIQUE index must release a value when
    // its owning row is DELETEd or UPDATEd off it, and re-claim it on the new
    // value — otherwise a duplicate would be wrongly rejected (stale index) or
    // wrongly accepted (missed update). Exercises the index maintenance path
    // (DirectWalTableRecordStore::check_unique_conflict override).
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    let ddl = DdlService::new(manager.clone());
    ddl.execute(DdlStatement::CreateNamespace {
        namespace: vec!["default".to_string()],
        if_not_exists: true,
        properties: HashMap::new(),
    })
    .await
    .expect("create namespace");
    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
            .parse_ddl(
                "CREATE TABLE members (id TEXT NOT NULL, email TEXT, PRIMARY KEY (id), UNIQUE (email));",
            )
            .expect("parse create")
            .expect("ddl");
    ddl.execute(ddl_stmt).await.expect("create table");

    let wal_appender = Arc::new(
        FramedTableWalAppender::open(temp_dir.path().join("unique_idx.wal"))
            .await
            .expect("open WAL"),
    );
    let record_store = Arc::new(DirectWalTableRecordStore::new(
        Arc::new(MemtableRecordStorage::new()),
        wal_appender,
    ));
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store,
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );
    let run = |sql: &'static str| parser.parse_dml(sql).expect("parse dml").expect("dml");

    // DELETE frees the value: insert d1=x@x.com, delete it, then d2=x@x.com inserts.
    dml.execute(run(
        "INSERT INTO members (id, email) VALUES ('d1', 'x@x.com');",
    ))
    .await
    .expect("insert d1");
    dml.execute(run("DELETE FROM members WHERE id = 'd1';"))
        .await
        .expect("delete d1");
    dml.execute(run(
        "INSERT INTO members (id, email) VALUES ('d2', 'x@x.com');",
    ))
    .await
    .expect("x@x.com is free after delete — d2 must insert");

    // UPDATE moves a value: u3 holds y@x.com, update it to z@x.com.
    dml.execute(run(
        "INSERT INTO members (id, email) VALUES ('u3', 'y@x.com');",
    ))
    .await
    .expect("insert u3");
    dml.execute(run("UPDATE members SET email = 'z@x.com' WHERE id = 'u3';"))
        .await
        .expect("update u3 email");

    // The vacated value (y@x.com) is now insertable…
    dml.execute(run(
        "INSERT INTO members (id, email) VALUES ('u4', 'y@x.com');",
    ))
    .await
    .expect("y@x.com freed by update — u4 must insert");

    // …and the new value (z@x.com) is now claimed by u3 → rejected.
    let err = dml
        .execute(run(
            "INSERT INTO members (id, email) VALUES ('u5', 'z@x.com');",
        ))
        .await
        .expect_err("z@x.com taken by u3 after update — must be rejected");
    assert!(
        err.to_string()
            .contains("duplicate key value violates unique constraint"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn update_rejects_duplicate_unique_value() {
    // TD-110: UPDATE that sets a UNIQUE column to a value owned by ANOTHER row
    // is rejected; setting it to the row's OWN current value (or a free value)
    // is allowed (the updated row is excluded from its own conflict check).
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    let ddl = DdlService::new(manager.clone());
    ddl.execute(DdlStatement::CreateNamespace {
        namespace: vec!["default".to_string()],
        if_not_exists: true,
        properties: HashMap::new(),
    })
    .await
    .expect("create namespace");
    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
            .parse_ddl(
                "CREATE TABLE members (id TEXT NOT NULL, email TEXT, PRIMARY KEY (id), UNIQUE (email));",
            )
            .expect("parse create")
            .expect("ddl");
    ddl.execute(ddl_stmt).await.expect("create table");

    let wal_appender = Arc::new(
        FramedTableWalAppender::open(temp_dir.path().join("update_unique.wal"))
            .await
            .expect("open WAL"),
    );
    let record_store = Arc::new(DirectWalTableRecordStore::new(
        Arc::new(MemtableRecordStorage::new()),
        wal_appender,
    ));
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store,
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );
    let run = |sql: &'static str| parser.parse_dml(sql).expect("parse dml").expect("dml");

    dml.execute(run(
        "INSERT INTO members (id, email) VALUES ('a', 'a@x.com');",
    ))
    .await
    .expect("insert a");
    dml.execute(run(
        "INSERT INTO members (id, email) VALUES ('b', 'b@x.com');",
    ))
    .await
    .expect("insert b");

    // UPDATE a -> b@x.com (owned by b) must be rejected.
    let err = dml
        .execute(run("UPDATE members SET email = 'b@x.com' WHERE id = 'a';"))
        .await
        .expect_err("UPDATE to another row's unique value must be rejected");
    assert!(
        err.to_string()
            .contains("duplicate key value violates unique constraint"),
        "unexpected error: {err}"
    );

    // UPDATE a -> its OWN current value (a@x.com) is a no-op conflict-wise → allowed.
    dml.execute(run("UPDATE members SET email = 'a@x.com' WHERE id = 'a';"))
        .await
        .expect("UPDATE to the row's own current value must be allowed");

    // UPDATE a -> a free value is allowed.
    dml.execute(run("UPDATE members SET email = 'c@x.com' WHERE id = 'a';"))
        .await
        .expect("UPDATE to a free unique value must be allowed");
}

/// F5 (ADR-072): with the fenced `ConditionalKeyStore` wired, a FOREIGN KEY
/// existence check resolves against the SAME store that fences the parent's PK
/// uniqueness (its `get` — "foreign-key existence checks (phase 2) ride this
/// method"), not the record-store probe. A child whose FK matches a live parent
/// PK inserts; a child referencing an absent parent is rejected by the store.
#[cfg(feature = "oltp-integrity")]
#[tokio::test]
async fn insert_enforces_foreign_key_via_fenced_store() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};
    use proximadb_cks_local::{LocalWalKeyStore, SyncPolicy};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    let ddl = DdlService::new(manager.clone());
    ddl.execute(DdlStatement::CreateNamespace {
        namespace: vec!["default".to_string()],
        if_not_exists: true,
        properties: HashMap::new(),
    })
    .await
    .expect("create namespace");
    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    for create in [
        "CREATE TABLE customers (id TEXT NOT NULL, name TEXT, PRIMARY KEY (id));",
        "CREATE TABLE orders (id TEXT NOT NULL, customer_id TEXT, PRIMARY KEY (id), FOREIGN KEY (customer_id) REFERENCES customers (id));",
    ] {
        let stmt = parser
            .parse_ddl(create)
            .expect("parse create")
            .expect("ddl");
        ddl.execute(stmt).await.expect("create table");
    }

    let wal_appender = Arc::new(
        FramedTableWalAppender::open(temp_dir.path().join("fk.wal"))
            .await
            .expect("open WAL"),
    );
    let record_store = Arc::new(DirectWalTableRecordStore::new(
        Arc::new(MemtableRecordStorage::new()),
        wal_appender,
    ));
    let cks = Arc::new(
        LocalWalKeyStore::open(temp_dir.path().join("cks.wal"), SyncPolicy::PerOp)
            .expect("open cks"),
    );
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store,
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    )
    .with_conditional_key_store(cks);
    let run = |sql: &'static str| parser.parse_dml(sql).expect("parse dml").expect("dml");

    // Parent PK 'c1' is registered in the fenced store by execute_insert.
    dml.execute(run(
        "INSERT INTO customers (id, name) VALUES ('c1', 'Alice');",
    ))
    .await
    .expect("insert parent customer");
    // Child FK resolves via the CKS `get` → parent present → allowed.
    dml.execute(run(
        "INSERT INTO orders (id, customer_id) VALUES ('o1', 'c1');",
    ))
    .await
    .expect("child referencing a live parent must insert (via the fenced store)");
    // Child FK to an absent parent → CKS `get` returns None → rejected.
    let err = dml
        .execute(run(
            "INSERT INTO orders (id, customer_id) VALUES ('o2', 'c99');",
        ))
        .await
        .expect_err("FK to a missing parent must be rejected via the fenced store");
    assert!(
        err.to_string().contains("FOREIGN KEY"),
        "unexpected error: {err}"
    );
}

/// F5 (ADR-072): a DELETE tombstones the row's PK in the fenced
/// `ConditionalKeyStore`, so the same PK re-inserts afterward. Without the
/// tombstone, `put_if_absent` would see the lingering holder and wrongly reject —
/// so this also proves the delete-side `Identity` matches the insert-side
/// registration exactly (same tenant placeholder + keyspace + typed value).
#[cfg(feature = "oltp-integrity")]
#[tokio::test]
async fn delete_tombstones_pk_in_fenced_store_allowing_reinsert() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};
    use proximadb_cks_local::{LocalWalKeyStore, SyncPolicy};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    let ddl = DdlService::new(manager.clone());
    ddl.execute(DdlStatement::CreateNamespace {
        namespace: vec!["default".to_string()],
        if_not_exists: true,
        properties: HashMap::new(),
    })
    .await
    .expect("create namespace");
    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
        .parse_ddl("CREATE TABLE users (id TEXT NOT NULL, email TEXT NOT NULL, PRIMARY KEY (id));")
        .expect("parse create")
        .expect("ddl");
    ddl.execute(ddl_stmt).await.expect("create table");

    let wal_appender = Arc::new(
        FramedTableWalAppender::open(temp_dir.path().join("del.wal"))
            .await
            .expect("open WAL"),
    );
    let record_store = Arc::new(DirectWalTableRecordStore::new(
        Arc::new(MemtableRecordStorage::new()),
        wal_appender,
    ));
    let cks = Arc::new(
        LocalWalKeyStore::open(temp_dir.path().join("cks.wal"), SyncPolicy::PerOp)
            .expect("open cks"),
    );
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store,
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    )
    .with_conditional_key_store(cks);
    let run = |sql: &'static str| parser.parse_dml(sql).expect("parse dml").expect("dml");

    // Insert u1 (registers the PK in the fenced store).
    dml.execute(run(
        "INSERT INTO users (id, email) VALUES ('u1', 'a@x.com');",
    ))
    .await
    .expect("insert u1");
    // A duplicate before delete is rejected (baseline: the fence is live).
    dml.execute(run(
        "INSERT INTO users (id, email) VALUES ('u1', 'dup@x.com');",
    ))
    .await
    .expect_err("duplicate PK must be rejected while the row is live");
    // Delete u1 — must tombstone the fenced entry.
    dml.execute(run("DELETE FROM users WHERE id = 'u1';"))
        .await
        .expect("delete u1");
    // Re-insert the same PK — must SUCCEED (the tombstone freed the fenced entry;
    // proves the delete-side Identity matched the insert-side registration).
    dml.execute(run(
        "INSERT INTO users (id, email) VALUES ('u1', 'b@x.com');",
    ))
    .await
    .expect("re-inserting a deleted PK must succeed after the tombstone");
}

#[tokio::test]
async fn insert_enforces_foreign_key_reference() {
    // TD-110: a FOREIGN KEY referencing the parent PK is enforced on INSERT —
    // present parent ok, missing parent rejected, NULL FK exempt; an UPDATE
    // re-checks. Parent + child live in the same store (same-partition).
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    let ddl = DdlService::new(manager.clone());
    ddl.execute(DdlStatement::CreateNamespace {
        namespace: vec!["default".to_string()],
        if_not_exists: true,
        properties: HashMap::new(),
    })
    .await
    .expect("create namespace");
    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    for create in [
        "CREATE TABLE customers (id TEXT NOT NULL, name TEXT, PRIMARY KEY (id));",
        "CREATE TABLE orders (id TEXT NOT NULL, customer_id TEXT, PRIMARY KEY (id), FOREIGN KEY (customer_id) REFERENCES customers (id));",
    ] {
        let stmt = parser
            .parse_ddl(create)
            .expect("parse create")
            .expect("ddl");
        ddl.execute(stmt).await.expect("create table");
    }

    let wal_appender = Arc::new(
        FramedTableWalAppender::open(temp_dir.path().join("fk.wal"))
            .await
            .expect("open WAL"),
    );
    let record_store = Arc::new(DirectWalTableRecordStore::new(
        Arc::new(MemtableRecordStorage::new()),
        wal_appender,
    ));
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store,
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );
    let run = |sql: &'static str| parser.parse_dml(sql).expect("parse dml").expect("dml");

    // Parent row, then a child referencing it — allowed.
    dml.execute(run(
        "INSERT INTO customers (id, name) VALUES ('c1', 'Alice');",
    ))
    .await
    .expect("insert parent customer");
    dml.execute(run(
        "INSERT INTO orders (id, customer_id) VALUES ('o1', 'c1');",
    ))
    .await
    .expect("child referencing existing parent must insert");

    // Child referencing a missing parent — rejected.
    let err = dml
        .execute(run(
            "INSERT INTO orders (id, customer_id) VALUES ('o2', 'c99');",
        ))
        .await
        .expect_err("FK to a missing parent must be rejected");
    assert!(
        err.to_string().contains("violates reference"),
        "unexpected error: {err}"
    );

    // NULL FK (customer_id omitted) is exempt.
    dml.execute(run("INSERT INTO orders (id) VALUES ('o3');"))
        .await
        .expect("NULL foreign key is exempt from the reference check");

    // UPDATE re-checks: pointing an order at a missing parent is rejected.
    let err = dml
        .execute(run(
            "UPDATE orders SET customer_id = 'c99' WHERE id = 'o1';",
        ))
        .await
        .expect_err("UPDATE to a missing FK parent must be rejected");
    assert!(
        err.to_string().contains("violates reference"),
        "unexpected error: {err}"
    );
}

/// SQL UPDATE and DELETE through `DirectWalTableRecordStore` — T9 conformance.
///
/// Verifies that UPDATE rewrites the current visible record and DELETE leaves
/// the row invisible to subsequent scans, both through the canonical WAL path.
#[tokio::test]
async fn direct_record_storage_update_and_delete_conformance() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("dml-ud.wal");

    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");

    let ddl = DdlService::new(manager.clone());
    ddl.execute(DdlStatement::CreateNamespace {
        namespace: vec!["default".to_string()],
        if_not_exists: true,
        properties: HashMap::new(),
    })
    .await
    .expect("create namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let create_sql = "CREATE TABLE items (id TEXT NOT NULL, label TEXT, PRIMARY KEY (id));";
    let ddl_stmt = parser
        .parse_ddl(create_sql)
        .expect("parse ddl")
        .expect("ddl stmt");
    ddl.execute(ddl_stmt).await.expect("create table");

    let wal_appender = Arc::new(
        FramedTableWalAppender::open(&wal_path)
            .await
            .expect("open WAL"),
    );
    let record_storage = Arc::new(MemtableRecordStorage::new());
    let record_store = Arc::new(DirectWalTableRecordStore::new(
        record_storage.clone(),
        wal_appender,
    ));
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store,
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );

    // INSERT two rows.
    for (id, label) in [("i1", "alpha"), ("i2", "beta")] {
        let sql = format!(
            "INSERT INTO items (id, label) VALUES ('{}', '{}');",
            id, label
        );
        let stmt = parser
            .parse_dml(&sql)
            .expect("parse insert")
            .expect("dml stmt");
        let r = dml.execute(stmt).await.expect("insert");
        assert!(r.success, "INSERT {id} must succeed");
    }

    // UPDATE i1's label.
    let update_sql = "UPDATE items SET label = 'alpha-updated' WHERE id = 'i1';";
    let update_stmt = parser
        .parse_dml(update_sql)
        .expect("parse update")
        .expect("dml stmt");
    let update_r = dml.execute(update_stmt).await.expect("update");
    assert!(update_r.success, "UPDATE must succeed");

    // Verify the updated label is visible via SELECT projection.
    let after_update = dml
        .select_table_records_with_projection(
            "items",
            &["id".to_string(), "label".to_string()],
            None,
            &[RelationalSelectPredicateInput {
                column_name: "id".to_string(),
                condition: RelationalSelectPredicateCondition::Comparison {
                    operator: RelationalSelectPredicateOperator::Equal,
                    literal: "i1".to_string(),
                },
            }],
            None,
        )
        .await
        .expect("select after update");
    assert_eq!(
        after_update.rows.len(),
        1,
        "SELECT must find i1 after update"
    );
    assert_eq!(
        after_update.rows[0][1],
        ProximaValue::String("alpha-updated".to_string()),
        "updated label must be visible"
    );

    // DELETE i2.
    let delete_sql = "DELETE FROM items WHERE id = 'i2';";
    let delete_stmt = parser
        .parse_dml(delete_sql)
        .expect("parse delete")
        .expect("dml stmt");
    let delete_r = dml.execute(delete_stmt).await.expect("delete");
    assert!(delete_r.success, "DELETE must succeed");

    // Verify i2 is no longer returned by a full scan.
    let after_delete = dml
        .select_table_records_with_projection("items", &["id".to_string()], None, &[], None)
        .await
        .expect("select after delete");
    let ids: Vec<&ProximaValue> = after_delete.rows.iter().map(|r| &r[0]).collect();
    assert!(
        !ids.contains(&&ProximaValue::String("i2".to_string())),
        "deleted row must not appear in scan"
    );
    assert_eq!(after_delete.rows.len(), 1, "only i1 must remain");
}

/// UPDATE/DELETE WHERE supports OR / nested groups / BETWEEN / NOT BETWEEN
/// via the resolved predicate tree (reusing the catalog-aware leaf eval), and
/// the PK fast-path stays OR-safe (a PK leaf under OR forces a full scan).
#[tokio::test]
async fn update_delete_support_or_nested_between_where() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("dml-tree.wal");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    let ddl = DdlService::new(manager.clone());
    ddl.execute(DdlStatement::CreateNamespace {
        namespace: vec!["default".to_string()],
        if_not_exists: true,
        properties: HashMap::new(),
    })
    .await
    .expect("create namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
        .parse_ddl("CREATE TABLE inv (id TEXT NOT NULL, status TEXT, qty INT, PRIMARY KEY (id));")
        .expect("parse ddl")
        .expect("ddl stmt");
    ddl.execute(ddl_stmt).await.expect("create table");

    let record_store = Arc::new(DirectWalTableRecordStore::new(
        Arc::new(MemtableRecordStorage::new()),
        Arc::new(
            FramedTableWalAppender::open(&wal_path)
                .await
                .expect("open WAL"),
        ),
    ));
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store,
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );

    async fn exec(
        dml: &DmlService,
        parser: &crate::query::sql_frontend::SqlFrontendParser,
        sql: &str,
    ) -> DmlResult {
        let stmt = parser.parse_dml(sql).expect("parse").expect("dml stmt");
        dml.execute(stmt).await.expect("execute")
    }
    async fn status_of(dml: &DmlService, id: &str) -> Option<String> {
        let sel = dml
            .select_table_records_with_projection(
                "inv",
                &["id".to_string(), "status".to_string()],
                None,
                &[RelationalSelectPredicateInput {
                    column_name: "id".to_string(),
                    condition: RelationalSelectPredicateCondition::Comparison {
                        operator: RelationalSelectPredicateOperator::Equal,
                        literal: id.to_string(),
                    },
                }],
                None,
            )
            .await
            .expect("select");
        sel.rows.first().map(|row| match &row[1] {
            ProximaValue::String(s) => s.clone(),
            other => format!("{other:?}"),
        })
    }

    for (id, status, qty) in [
        ("i1", "active", 5),
        ("i2", "active", 15),
        ("i3", "idle", 25),
        ("i4", "idle", 35),
    ] {
        exec(
            &dml,
            &parser,
            &format!("INSERT INTO inv (id, status, qty) VALUES ('{id}', '{status}', {qty});"),
        )
        .await;
    }

    // (1) OR: status='active' OR qty >= 30 → i1,i2 (active) + i4 (qty 35).
    let r = exec(
        &dml,
        &parser,
        "UPDATE inv SET status = 'archived' WHERE status = 'active' OR qty >= 30;",
    )
    .await;
    assert_eq!(r.rows_affected, 3, "OR union");
    assert_eq!(status_of(&dml, "i1").await.as_deref(), Some("archived"));
    assert_eq!(status_of(&dml, "i2").await.as_deref(), Some("archived"));
    assert_eq!(
        status_of(&dml, "i3").await.as_deref(),
        Some("idle"),
        "untouched"
    );
    assert_eq!(status_of(&dml, "i4").await.as_deref(), Some("archived"));

    // (2) BETWEEN on an INT column (catalog-aware): qty 20..30 → i3 only.
    let r = exec(
        &dml,
        &parser,
        "UPDATE inv SET status = 'mid' WHERE qty BETWEEN 20 AND 30;",
    )
    .await;
    assert_eq!(r.rows_affected, 1, "BETWEEN matches i3 (qty 25)");
    assert_eq!(status_of(&dml, "i3").await.as_deref(), Some("mid"));

    // (3) NOT BETWEEN: qty outside 10..30 → i1 (5) and i4 (35).
    let r = exec(
        &dml,
        &parser,
        "UPDATE inv SET status = 'extreme' WHERE qty NOT BETWEEN 10 AND 30;",
    )
    .await;
    assert_eq!(r.rows_affected, 2, "NOT BETWEEN matches i1 + i4");
    assert_eq!(status_of(&dml, "i1").await.as_deref(), Some("extreme"));
    assert_eq!(status_of(&dml, "i4").await.as_deref(), Some("extreme"));

    // (4) Nested + PK-under-OR safety: id='i2' OR (status='extreme' AND qty < 10)
    // → i2 (PK) + i1 (extreme AND qty 5<10). The PK leaf under OR must NOT
    // shortcut to fetching only i2 and miss i1.
    let r = exec(
        &dml,
        &parser,
        "DELETE FROM inv WHERE id = 'i2' OR (status = 'extreme' AND qty < 10);",
    )
    .await;
    assert_eq!(
        r.rows_affected, 2,
        "i2 (pk) + i1 (nested) — PK fast-path stayed OR-safe"
    );
    assert_eq!(
        status_of(&dml, "i1").await,
        None,
        "i1 deleted via nested branch"
    );
    assert_eq!(
        status_of(&dml, "i2").await,
        None,
        "i2 deleted via PK branch"
    );
    assert_eq!(
        status_of(&dml, "i3").await.as_deref(),
        Some("mid"),
        "i3 survives"
    );
    assert_eq!(
        status_of(&dml, "i4").await.as_deref(),
        Some("extreme"),
        "i4 survives"
    );
}

/// SELECT WHERE supports OR / mixed-AND-OR / nested groups / NOT IN through
/// the same resolved predicate tree as UPDATE/DELETE, pushed into the record
/// scan via `select_table_records_with_projection_where`. The PK fast-path
/// stays OR-safe (a PK leaf under OR forces a full scan), and nested groups
/// are NOT flattened.
#[tokio::test]
async fn select_where_supports_or_nested_and_pk_or_safety() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("dml-select-tree.wal");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    let ddl = DdlService::new(manager.clone());
    ddl.execute(DdlStatement::CreateNamespace {
        namespace: vec!["default".to_string()],
        if_not_exists: true,
        properties: HashMap::new(),
    })
    .await
    .expect("create namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
        .parse_ddl("CREATE TABLE inv (id TEXT NOT NULL, status TEXT, qty INT, PRIMARY KEY (id));")
        .expect("parse ddl")
        .expect("ddl stmt");
    ddl.execute(ddl_stmt).await.expect("create table");

    let record_store = Arc::new(DirectWalTableRecordStore::new(
        Arc::new(MemtableRecordStorage::new()),
        Arc::new(
            FramedTableWalAppender::open(&wal_path)
                .await
                .expect("open WAL"),
        ),
    ));
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store,
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );

    for (id, status, qty) in [
        ("i1", "active", 5),
        ("i2", "active", 15),
        ("i3", "idle", 25),
        ("i4", "idle", 35),
    ] {
        let stmt = parser
            .parse_dml(&format!(
                "INSERT INTO inv (id, status, qty) VALUES ('{id}', '{status}', {qty});"
            ))
            .expect("parse insert")
            .expect("insert stmt");
        dml.execute(stmt).await.expect("insert");
    }

    // Run a full SELECT string through the WhereClause-tree path; return the
    // chosen access path, the route-metadata predicate_count (tree leaf count),
    // and the sorted matching ids.
    async fn run(
        dml: &DmlService,
        parser: &crate::query::sql_frontend::SqlFrontendParser,
        sql: &str,
        limit: Option<usize>,
    ) -> (RelationalSelectAccessPath, usize, Vec<String>) {
        let where_clause = parser.parse_select_where_clause(sql).expect("parse where");
        let res = dml
            .select_table_records_with_projection_where(
                "inv",
                &["id".to_string()],
                limit,
                where_clause.as_ref(),
                None,
            )
            .await
            .expect("select");
        let mut ids: Vec<String> = res
            .rows
            .iter()
            .map(|row| match &row[0] {
                ProximaValue::String(s) => s.clone(),
                other => format!("{other:?}"),
            })
            .collect();
        ids.sort();
        (
            res.route_metadata.access_path,
            res.route_metadata.predicate_count,
            ids,
        )
    }

    // (1) OR union: status='active' OR qty >= 30 → i1,i2 (active) + i4 (35).
    // predicate_count = 2 leaves.
    let (path, pc, ids) = run(
        &dml,
        &parser,
        "SELECT id FROM inv WHERE status = 'active' OR qty >= 30",
        None,
    )
    .await;
    assert_eq!(path, RelationalSelectAccessPath::TableScan);
    assert_eq!(pc, 2, "route-metadata predicate_count == tree leaf count");
    assert_eq!(ids, vec!["i1", "i2", "i4"], "OR union");

    // (2) PK-under-OR safety: id='i2' OR status='idle'. The PK leaf must NOT
    // shortcut to a point lookup that misses the idle rows.
    let (path, _pc, ids) = run(
        &dml,
        &parser,
        "SELECT id FROM inv WHERE id = 'i2' OR status = 'idle'",
        None,
    )
    .await;
    assert_eq!(
        path,
        RelationalSelectAccessPath::TableScan,
        "PK leaf under OR must force a full scan"
    );
    assert_eq!(ids, vec!["i2", "i3", "i4"]);

    // (3) PK fast-path + full-predicate re-check: id IN (i1,i2,i3) AND
    // status='active' → only i1,i2 (i3 is idle and is dropped by the re-check).
    let (path, _pc, ids) = run(
        &dml,
        &parser,
        "SELECT id FROM inv WHERE id IN ('i1','i2','i3') AND status = 'active'",
        None,
    )
    .await;
    assert_eq!(path, RelationalSelectAccessPath::PrimaryKeyLookup);
    assert_eq!(ids, vec!["i1", "i2"]);

    // (4) Nested grouping must NOT flatten: status='idle' AND (qty < 30 OR
    // id='i1') → i3 only. Flattening to `idle AND qty<30 AND id='i1'` would
    // wrongly return zero rows. predicate_count = 3 leaves.
    let (path, pc, ids) = run(
        &dml,
        &parser,
        "SELECT id FROM inv WHERE status = 'idle' AND (qty < 30 OR id = 'i1')",
        None,
    )
    .await;
    assert_eq!(path, RelationalSelectAccessPath::TableScan);
    assert_eq!(pc, 3, "AND of [idle, OR(qty<30, id=i1)] has 3 leaves");
    assert_eq!(ids, vec!["i3"]);

    // (4b) OR-under-AND, no PK predicate → full scan, not flattened:
    // (status='active' OR qty >= 30) AND qty < 20 → {i1,i2,i4} ∩ {i1,i2} = i1,i2.
    let (path, _pc, ids) = run(
        &dml,
        &parser,
        "SELECT id FROM inv WHERE (status = 'active' OR qty >= 30) AND qty < 20",
        None,
    )
    .await;
    assert_eq!(path, RelationalSelectAccessPath::TableScan);
    assert_eq!(ids, vec!["i1", "i2"], "(a OR b) AND c grouping preserved");

    // (5) NOT IN mixed with OR over a never-true branch: qty NOT IN (5,15) OR
    // status IS NULL → i3,i4 (no row has a NULL status).
    let (_path, _pc, ids) = run(
        &dml,
        &parser,
        "SELECT id FROM inv WHERE qty NOT IN (5, 15) OR status IS NULL",
        None,
    )
    .await;
    assert_eq!(ids, vec!["i3", "i4"]);

    // (5b) NOT BETWEEN: qty outside 10..30 → i1 (5) and i4 (35).
    let (_path, _pc, ids) = run(
        &dml,
        &parser,
        "SELECT id FROM inv WHERE qty NOT BETWEEN 10 AND 30",
        None,
    )
    .await;
    assert_eq!(ids, vec!["i1", "i4"], "NOT BETWEEN matches the extremes");

    // (6) LIMIT honored on the OR scan path.
    let (_path, _pc, ids) = run(
        &dml,
        &parser,
        "SELECT id FROM inv WHERE status = 'active' OR status = 'idle'",
        Some(2),
    )
    .await;
    assert_eq!(ids.len(), 2, "limit pushed into the predicate scan");

    // (7) No WHERE → scan all rows.
    let (path, pc, ids) = run(&dml, &parser, "SELECT id FROM inv", None).await;
    assert_eq!(path, RelationalSelectAccessPath::TableScan);
    assert_eq!(pc, 0, "no WHERE → zero predicate leaves");
    assert_eq!(ids, vec!["i1", "i2", "i3", "i4"]);
}

/// TD-127: a single-column equality / IN-list on a non-PK secondary-indexed
/// column is answered by the OLTP secondary index (`SecondaryIndexLookup`),
/// not a full `TableScan`, and returns exactly the rows the scan would — while
/// the kill-switch falls back to a scan with identical rows. The schema is
/// registered programmatically because no DDL surface declares secondary
/// indexes yet (a noted follow-on — until a `CREATE INDEX` DDL lands, the
/// Volcano relational stack's `SecondaryLookup` path is correct-but-dormant);
/// the cataloged `secondary_indexes` survive the `ObjectSchema` round-trip, so
/// the read path sees them. This test also covers the Volcano-pipeline
/// forwarders `secondary_lookup_relational` (TD-127) and
/// `point_lookup_batch_relational` (TD-128), which reuse this same index +
/// `get_by_key` path.
#[tokio::test]
async fn select_where_uses_secondary_index_for_nonpk_name_and_file() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};
    use proximadb_catalog::{
        CatalogColumn, CatalogIndex, CatalogIndexType, CatalogStorageSpecialization,
        CatalogTableSchema, RelationalCapabilities,
    };
    use proximadb_data_model::ProximaType;

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("dml-secondary-index.wal");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");

    // Register a `code_symbol` table with non-unique secondary indexes on the
    // code-graph lookup columns (`name`, `file`), PK `oid`.
    let (catalog, table_id) = manager
        .resolve_table_scoped("code_symbol", None)
        .await
        .expect("resolve table");
    if !catalog
        .namespace_exists(&table_id.namespace)
        .await
        .expect("namespace_exists")
    {
        catalog
            .create_namespace_for_tenant(&table_id.namespace, HashMap::new(), None)
            .await
            .expect("create namespace");
    }
    let schema = CatalogTableSchema::new(&table_id.name)
        .with_column(CatalogColumn::new(1, "oid", ProximaType::String).nullable(false))
        .with_column(CatalogColumn::new(2, "name", ProximaType::String))
        .with_column(CatalogColumn::new(3, "file", ProximaType::String))
        .with_primary_key(vec!["oid".to_string()])
        .with_storage_specialization(CatalogStorageSpecialization::PaxOltp)
        .with_relational_capabilities(RelationalCapabilities {
            primary_key: vec!["oid".to_string()],
            secondary_indexes: vec![
                CatalogIndex::new(
                    "sym_name_idx",
                    vec!["name".to_string()],
                    CatalogIndexType::Hash,
                ),
                CatalogIndex::new(
                    "sym_file_idx",
                    vec!["file".to_string()],
                    CatalogIndexType::Hash,
                ),
            ],
            ..Default::default()
        });
    catalog
        .create_table(&table_id, schema)
        .await
        .expect("register table");

    let record_store = Arc::new(DirectWalTableRecordStore::new(
        Arc::new(MemtableRecordStorage::new()),
        Arc::new(
            FramedTableWalAppender::open(&wal_path)
                .await
                .expect("open WAL"),
        ),
    ));
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store,
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    for (oid, name, file) in [
        ("s1", "parse", "a.rs"),
        ("s2", "parse", "b.rs"),
        ("s3", "emit", "b.rs"),
        ("s4", "emit", "c.rs"),
    ] {
        let stmt = parser
            .parse_dml(&format!(
                "INSERT INTO code_symbol (oid, name, file) VALUES ('{oid}', '{name}', '{file}');"
            ))
            .expect("parse insert")
            .expect("insert stmt");
        dml.execute(stmt).await.expect("insert");
    }

    async fn run(
        dml: &DmlService,
        parser: &crate::query::sql_frontend::SqlFrontendParser,
        sql: &str,
    ) -> (RelationalSelectAccessPath, Vec<String>) {
        let where_clause = parser.parse_select_where_clause(sql).expect("parse where");
        let res = dml
            .select_table_records_with_projection_where(
                "code_symbol",
                &["oid".to_string()],
                None,
                where_clause.as_ref(),
                None,
            )
            .await
            .expect("select");
        let mut ids: Vec<String> = res
            .rows
            .iter()
            .map(|row| match &row[0] {
                ProximaValue::String(s) => s.clone(),
                other => format!("{other:?}"),
            })
            .collect();
        ids.sort();
        (res.route_metadata.access_path, ids)
    }

    // Equality on the indexed non-PK `name` → secondary-index lookup, both rows.
    let (path, ids) = run(
        &dml,
        &parser,
        "SELECT oid FROM code_symbol WHERE name = 'parse'",
    )
    .await;
    assert_eq!(path, RelationalSelectAccessPath::SecondaryIndexLookup);
    assert_eq!(ids, vec!["s1", "s2"]);

    // IN-list on the indexed `file` → secondary-index lookup, union a.rs + c.rs.
    let (path, ids) = run(
        &dml,
        &parser,
        "SELECT oid FROM code_symbol WHERE file IN ('a.rs', 'c.rs')",
    )
    .await;
    assert_eq!(path, RelationalSelectAccessPath::SecondaryIndexLookup);
    assert_eq!(ids, vec!["s1", "s4"]);

    // The index narrows; the FULL predicate still decides: name='parse' AND
    // file='b.rs' → only s2 (s1 is a.rs, dropped by the re-check).
    let (path, ids) = run(
        &dml,
        &parser,
        "SELECT oid FROM code_symbol WHERE name = 'parse' AND file = 'b.rs'",
    )
    .await;
    assert_eq!(path, RelationalSelectAccessPath::SecondaryIndexLookup);
    assert_eq!(ids, vec!["s2"]);

    // Kill-switch → scan fallback with identical rows.
    // SAFETY: single-threaded test; restored immediately after the probe.
    unsafe { std::env::set_var("PROXIMADB_SECONDARY_INDEX_DISABLE", "1") };
    let (path, ids) = run(
        &dml,
        &parser,
        "SELECT oid FROM code_symbol WHERE name = 'parse'",
    )
    .await;
    unsafe { std::env::remove_var("PROXIMADB_SECONDARY_INDEX_DISABLE") };
    assert_eq!(path, RelationalSelectAccessPath::TableScan);
    assert_eq!(ids, vec!["s1", "s2"], "scan fallback returns the same rows");

    // TD-127: the Volcano-pipeline forwarder `secondary_lookup_relational`
    // reuses the same index + `get_by_key` path and returns FULL projected
    // rows for each live candidate. `name='emit'` → s3,s4.
    let mut got: Vec<String> = dml
        .secondary_lookup_relational("code_symbol", "name", &["emit".to_string()], None)
        .await
        .expect("secondary lookup")
        .expect("name is indexed → Some(rows)")
        .into_iter()
        .map(|row| match &row[0] {
            ProximaValue::String(s) => s.clone(),
            other => format!("{other:?}"),
        })
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec!["s3", "s4"],
        "secondary_lookup_relational candidates"
    );

    // A non-indexed column → Ok(None) (the Volcano executor falls back to a
    // full scan + residual filter).
    let none = dml
        .secondary_lookup_relational("code_symbol", "oid", &["s1".to_string()], None)
        .await
        .expect("secondary lookup on non-indexed column");
    assert!(none.is_none(), "non-indexed column → None (scan fallback)");

    // TD-128: the discrete-batch forwarder `point_lookup_batch_relational`
    // reuses `get_by_key` per key and returns FULL rows for the hits only
    // (the missing key is absent).
    let mut batch: Vec<String> = dml
        .point_lookup_batch_relational(
            "code_symbol",
            &["s1".to_string(), "s3".to_string(), "missing".to_string()],
            None,
        )
        .await
        .expect("point batch lookup")
        .into_iter()
        .map(|row| match &row[0] {
            ProximaValue::String(s) => s.clone(),
            other => format!("{other:?}"),
        })
        .collect();
    batch.sort();
    assert_eq!(
        batch,
        vec!["s1", "s3"],
        "batch lookup returns hits, omits miss"
    );
}

/// `scan_table_relational` (PATH B reader backend) pushes the output
/// projection + a full-row predicate + limit into the record-store scan.
#[tokio::test]
async fn scan_table_relational_pushes_projection_predicate_limit() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("dml-scan-rel.wal");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    let ddl = DdlService::new(manager.clone());
    ddl.execute(DdlStatement::CreateNamespace {
        namespace: vec!["default".to_string()],
        if_not_exists: true,
        properties: HashMap::new(),
    })
    .await
    .expect("create namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
        .parse_ddl("CREATE TABLE inv (id TEXT NOT NULL, status TEXT, qty INT, PRIMARY KEY (id));")
        .expect("parse ddl")
        .expect("ddl stmt");
    ddl.execute(ddl_stmt).await.expect("create table");

    let record_store = Arc::new(DirectWalTableRecordStore::new(
        Arc::new(MemtableRecordStorage::new()),
        Arc::new(
            FramedTableWalAppender::open(&wal_path)
                .await
                .expect("open WAL"),
        ),
    ));
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store,
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );
    for (id, status, qty) in [
        ("i1", "active", 5),
        ("i2", "active", 15),
        ("i3", "idle", 25),
        ("i4", "idle", 35),
    ] {
        let stmt = parser
            .parse_dml(&format!(
                "INSERT INTO inv (id, status, qty) VALUES ('{id}', '{status}', {qty});"
            ))
            .expect("parse insert")
            .expect("insert stmt");
        dml.execute(stmt).await.expect("insert");
    }

    // (a) No predicate / no projection → all rows, full column order [id,status,qty].
    let (schema, rows) = dml
        .scan_table_relational(
            "inv",
            None,
            None,
            None,
            None,
            proximadb_runtime::PortIdentity::anonymous(),
        )
        .await
        .expect("scan all");
    assert_eq!(schema.columns.len(), 3);
    assert_eq!(rows.len(), 4);
    assert!(rows.iter().all(|r| r.len() == 3));

    // (b) Predicate over the FULL row: status (ordinal 1) == 'active' → i1,i2.
    let pred = |row: &[ProximaValue]| -> Result<bool, ExprError> {
        Ok(matches!(&row[1], ProximaValue::String(s) if s == "active"))
    };
    let (_s, rows) = dml
        .scan_table_relational(
            "inv",
            None,
            Some(&pred),
            None,
            None,
            proximadb_runtime::PortIdentity::anonymous(),
        )
        .await
        .expect("scan predicate");
    let mut ids: Vec<String> = rows
        .iter()
        .map(|r| match &r[0] {
            ProximaValue::String(s) => s.clone(),
            other => format!("{other:?}"),
        })
        .collect();
    ids.sort();
    assert_eq!(ids, vec!["i1", "i2"], "predicate filters to active rows");

    // (b2) ADR-043 Invariant 1: a predicate that errors is surfaced as a hard
    // error, NEVER silently dropped to an empty result.
    let failing_pred = |_row: &[ProximaValue]| -> Result<bool, ExprError> {
        Err(ExprError::UnknownFunction {
            name: "does_not_exist".to_string(),
        })
    };
    let err = dml
        .scan_table_relational(
            "inv",
            None,
            Some(&failing_pred),
            None,
            None,
            proximadb_runtime::PortIdentity::anonymous(),
        )
        .await
        .expect_err("a predicate eval error must surface, not silently drop rows");
    assert!(
        err.to_string().contains("predicate evaluation failed"),
        "expected a loud predicate error, got: {err}"
    );

    // (c) Output projection narrows + orders columns → just [status].
    let cols = vec!["status".to_string()];
    let (_s, rows) = dml
        .scan_table_relational(
            "inv",
            Some(&cols),
            None,
            None,
            None,
            proximadb_runtime::PortIdentity::anonymous(),
        )
        .await
        .expect("scan projection");
    assert_eq!(rows.len(), 4);
    assert!(rows.iter().all(|r| r.len() == 1));

    // (d) Limit caps the result.
    let (_s, rows) = dml
        .scan_table_relational(
            "inv",
            None,
            None,
            Some(2),
            None,
            proximadb_runtime::PortIdentity::anonymous(),
        )
        .await
        .expect("scan limit");
    assert_eq!(rows.len(), 2, "limit caps the scan");
}

/// P3.2: `materialize_table_to_parquet` snapshots the table's rows to a Parquet
/// object on the bridge AND flips the catalog layout to Parquet/ProjectionPublication
/// at the published location, so the OLAP router will treat it as Parquet-backed.
#[tokio::test]
async fn materialize_table_writes_parquet_and_flips_catalog_layout() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};
    use futures::StreamExt;
    use proximadb_iceberg_engine::IcebergObjectStoreBridge;

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("dml-materialize.wal");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    let ddl = DdlService::new(manager.clone());
    ddl.execute(DdlStatement::CreateNamespace {
        namespace: vec!["default".to_string()],
        if_not_exists: true,
        properties: HashMap::new(),
    })
    .await
    .expect("create namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
        .parse_ddl("CREATE TABLE inv (id TEXT NOT NULL, status TEXT, qty INT, PRIMARY KEY (id));")
        .expect("parse ddl")
        .expect("ddl stmt");
    ddl.execute(ddl_stmt).await.expect("create table");

    let record_store = Arc::new(DirectWalTableRecordStore::new(
        Arc::new(MemtableRecordStorage::new()),
        Arc::new(
            FramedTableWalAppender::open(&wal_path)
                .await
                .expect("open WAL"),
        ),
    ));
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store,
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );
    for (id, status, qty) in [
        ("i1", "active", 5),
        ("i2", "active", 15),
        ("i3", "idle", 25),
    ] {
        let stmt = parser
            .parse_dml(&format!(
                "INSERT INTO inv (id, status, qty) VALUES ('{id}', '{status}', {qty});"
            ))
            .expect("parse insert")
            .expect("insert stmt");
        dml.execute(stmt).await.expect("insert");
    }

    // A shared in-memory bridge: we reuse the SAME handle to read the snapshot
    // back (from_url("memory://") would open a fresh, empty store).
    let bridge = Arc::new(IcebergObjectStoreBridge::from_url("memory:///warehouse").unwrap());

    let location = dml
        .materialize_table_to_parquet(&*bridge, "memory:///warehouse", "inv", None)
        .await
        .expect("materialize");

    // The published location is the tenant-isolated base URL in the single
    // canonical DrPath layout (`data/{tenant}/{namespace_id}/{table}`) — the
    // no-tenant path is just a degenerate multi-tenant under `default_tenant` and
    // the namespace's own (rename-stable) `namespace_id`, no legacy fork.
    assert!(
        location.starts_with("memory:///warehouse/data/default_tenant/ns_")
            && location.ends_with("/inv/data"),
        "embedded materialize must use the canonical DrPath layout: {location}"
    );

    // The Parquet snapshot landed where the OLAP reader lists `{location}/*.parquet`
    // (ADR-059: `location` is the dir where the parquet files are IMMEDIATE
    // children), and reads back all three rows. Derive the prefix from the
    // resolved location so the read tracks the real (opaque) namespace_id.
    let prefix = location
        .strip_prefix("memory:///warehouse/")
        .expect("location under warehouse root");
    let data_object = object_store::path::Path::from(format!("{prefix}/part-0.parquet"));
    let mut stream = bridge
        .read_parquet_batches(
            &data_object,
            Arc::new(arrow_schema::Schema::empty()),
            1024,
            None,
        )
        .await
        .expect("read materialized parquet");
    let mut total = 0usize;
    while let Some(batch) = stream.next().await {
        total += batch.expect("batch").num_rows();
    }
    assert_eq!(total, 3, "all rows materialized into the snapshot");

    // The catalog layout is now a published Parquet projection at the location.
    let (catalog, id) = manager.resolve_table("inv").await.expect("resolve");
    let schema = catalog.get_table(&id).await.expect("get table");
    assert_eq!(schema.storage_layouts.len(), 1);
    let layout = &schema.storage_layouts[0];
    assert!(matches!(
        layout.physical_format,
        proximadb_catalog::CatalogPhysicalFormat::Parquet
    ));
    assert!(matches!(
        layout.authority,
        proximadb_catalog::CatalogAuthorityMode::ProjectionPublication
    ));
    assert_eq!(layout.location.as_deref(), Some(location.as_str()));
}

/// P3.3: `ALTER TABLE … MATERIALIZE` routed through DdlService + a wired
/// DmlTableMaterializer flips the catalog layout; an unwired DdlService errors cleanly.
#[tokio::test]
async fn alter_table_materialize_via_ddl_flips_catalog_layout() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};
    use proximadb_iceberg_engine::IcebergObjectStoreBridge;

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("dml-mat-ddl.wal");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    let ddl = DdlService::new(manager.clone());
    ddl.execute(DdlStatement::CreateNamespace {
        namespace: vec!["default".to_string()],
        if_not_exists: true,
        properties: HashMap::new(),
    })
    .await
    .expect("create namespace");
    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
        .parse_ddl("CREATE TABLE inv (id TEXT NOT NULL, status TEXT, qty INT, PRIMARY KEY (id));")
        .expect("parse ddl")
        .expect("ddl stmt");
    ddl.execute(ddl_stmt).await.expect("create table");
    let record_store = Arc::new(DirectWalTableRecordStore::new(
        Arc::new(MemtableRecordStorage::new()),
        Arc::new(
            FramedTableWalAppender::open(&wal_path)
                .await
                .expect("open WAL"),
        ),
    ));
    let dml = Arc::new(DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store,
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    ));
    for (id, status, qty) in [("i1", "active", 5), ("i2", "idle", 25)] {
        let stmt = parser
            .parse_dml(&format!(
                "INSERT INTO inv (id, status, qty) VALUES ('{id}', '{status}', {qty});"
            ))
            .expect("parse insert")
            .expect("insert stmt");
        dml.execute(stmt).await.expect("insert");
    }

    // Wire a materializer (DmlService + bridge) into a DdlService and run the trigger.
    let bridge = Arc::new(IcebergObjectStoreBridge::from_url("memory:///wh").unwrap());
    let materializer = Arc::new(DmlTableMaterializer::new(
        dml.clone(),
        bridge.clone(),
        "memory:///wh",
    ));
    let ddl_mat = DdlService::new(manager.clone()).with_materializer(materializer);
    ddl_mat
        .execute(DdlStatement::MaterializeTable {
            name: "inv".to_string(),
        })
        .await
        .expect("materialize via DDL");

    // The catalog layout is now a published Parquet projection.
    let (catalog, id) = manager.resolve_table("inv").await.expect("resolve");
    let schema = catalog.get_table(&id).await.expect("get table");
    assert!(matches!(
        schema.storage_layouts[0].physical_format,
        proximadb_catalog::CatalogPhysicalFormat::Parquet
    ));
    assert!(matches!(
        schema.storage_layouts[0].authority,
        proximadb_catalog::CatalogAuthorityMode::ProjectionPublication
    ));

    // A DdlService WITHOUT a materializer rejects the statement cleanly.
    let ddl_bare = DdlService::new(manager.clone());
    assert!(
        ddl_bare
            .execute(DdlStatement::MaterializeTable {
                name: "inv".to_string()
            })
            .await
            .is_err(),
        "MATERIALIZE without a configured materializer must error"
    );
}

/// TD-113: the same table name materialized under two different request
/// tenants lands under disjoint, tenant-isolated object prefixes — never
/// co-mingled under the `default_tenant` placeholder. CREATE + INSERT +
/// MATERIALIZE are all tenant-scoped (TD-064), so each tenant owns its row
/// and its snapshot prefix.
#[tokio::test]
async fn materialize_is_tenant_isolated_by_request_tenant() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};
    use proximadb_iceberg_engine::IcebergObjectStoreBridge;

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("dml-mat-tenant.wal");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    let ddl = DdlService::new(manager.clone());
    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let record_store = Arc::new(DirectWalTableRecordStore::new(
        Arc::new(MemtableRecordStorage::new()),
        Arc::new(
            FramedTableWalAppender::open(&wal_path)
                .await
                .expect("open WAL"),
        ),
    ));
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store,
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );
    let bridge = Arc::new(IcebergObjectStoreBridge::from_url("memory:///wh").unwrap());

    let mut locations = Vec::new();
    for tenant in ["acmecorp", "globexco"] {
        let tctx = TenantContext::for_tenant_id(tenant);
        let create = parser
            .parse_ddl("CREATE TABLE inv (id TEXT NOT NULL, qty INT, PRIMARY KEY (id));")
            .expect("parse ddl")
            .expect("ddl stmt");
        ddl.execute_scoped(create, Some(tenant))
            .await
            .expect("create table");
        // Distinct PK per tenant: this low-level DirectWalTableRecordStore is
        // not row-partitioned by tenant (that is TD-064's routing store), and
        // TD-113 only governs the object prefix, not row isolation here.
        let insert = parser
            .parse_dml(&format!(
                "INSERT INTO inv (id, qty) VALUES ('{tenant}', 5);"
            ))
            .expect("parse insert")
            .expect("insert stmt");
        dml.execute_scoped(insert, Some(&tctx))
            .await
            .expect("insert");
        let loc = dml
            .materialize_table_to_parquet(&*bridge, "memory:///wh", "inv", Some(&tctx))
            .await
            .expect("materialize");
        assert!(
            loc.contains(&format!("/{tenant}/")),
            "location must carry the tenant segment: {loc}"
        );
        assert!(
            !loc.contains("default_tenant"),
            "location must not use the default_tenant placeholder: {loc}"
        );
        locations.push(loc);
    }
    assert_ne!(
        locations[0], locations[1],
        "two tenants must materialize to disjoint prefixes"
    );
}

/// TD-113 Phase 2: DrPath is the canonical layout, so the snapshot prefix uses the
/// rename-stable opaque `namespace_id` (`data/{tenant}/{ns_<uuid>}/{table}`) instead
/// of the human namespace path whenever the namespace has a `namespace_id` — driven
/// by the catalog (no env flag).
#[tokio::test]
async fn materialize_drpath_layout_uses_opaque_namespace_id() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};
    use proximadb_iceberg_engine::IcebergObjectStoreBridge;

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("dml-mat-drpath.wal");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    let ddl = DdlService::new(manager.clone());
    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let record_store = Arc::new(DirectWalTableRecordStore::new(
        Arc::new(MemtableRecordStorage::new()),
        Arc::new(
            FramedTableWalAppender::open(&wal_path)
                .await
                .expect("open WAL"),
        ),
    ));
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store,
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );
    let create = parser
        .parse_ddl("CREATE TABLE inv (id TEXT NOT NULL, qty INT, PRIMARY KEY (id));")
        .expect("parse ddl")
        .expect("ddl stmt");
    ddl.execute_scoped(create, Some("acmecorp"))
        .await
        .expect("create table");
    let tctx = TenantContext::for_tenant_id("acmecorp");
    let insert = parser
        .parse_dml("INSERT INTO inv (id, qty) VALUES ('i1', 5);")
        .expect("parse insert")
        .expect("insert stmt");
    dml.execute_scoped(insert, Some(&tctx))
        .await
        .expect("insert");

    let bridge = Arc::new(IcebergObjectStoreBridge::from_url("memory:///wh").unwrap());
    let loc = dml
        .materialize_table_to_parquet(&*bridge, "memory:///wh", "inv", Some(&tctx))
        .await
        .expect("materialize (drpath)");

    assert!(
        loc.starts_with("memory:///wh/data/acmecorp/ns_"),
        "drpath layout must use the opaque namespace_id: {loc}"
    );
    assert!(loc.ends_with("/inv/data"), "loc={loc}");
    assert!(!loc.contains("default_tenant"), "loc={loc}");
}

/// The bulk-append `INSERT ... SELECT` prefix resolver produces the SAME
/// namespace-aware, tenant-scoped layout as the materialize path
/// (`data/{tenant}/{namespace}/{table}`) — not the flat `tables/{name}`
/// fallback — and yields `None` (legacy fallback) when there is no tenant scope.
#[tokio::test]
async fn resolve_warehouse_object_prefix_is_namespace_aware_and_tenant_scoped() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("dml-bulk-prefix.wal");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    let ddl = DdlService::new(manager.clone());
    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let record_store = Arc::new(DirectWalTableRecordStore::new(
        Arc::new(MemtableRecordStorage::new()),
        Arc::new(
            FramedTableWalAppender::open(&wal_path)
                .await
                .expect("open WAL"),
        ),
    ));
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store,
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );
    let create = parser
        .parse_ddl("CREATE TABLE facts (id TEXT NOT NULL, qty INT, PRIMARY KEY (id));")
        .expect("parse ddl")
        .expect("ddl stmt");
    ddl.execute_scoped(create, Some("acmecorp"))
        .await
        .expect("create table");

    // Tenant scope → namespace-aware, tenant-isolated prefix (NOT the flat fallback).
    let prefix = dml
        .resolve_warehouse_object_prefix("facts", Some("acmecorp"))
        .await
        .expect("resolve prefix")
        .expect("a tenant scope yields a prefix");
    assert!(
        prefix.starts_with("data/acmecorp/"),
        "prefix must be tenant-isolated: {prefix}"
    );
    assert!(
        prefix.ends_with("/facts"),
        "prefix must end with table: {prefix}"
    );
    assert!(
        !prefix.contains("/tables/"),
        "prefix must be namespace-aware, not the flat `tables/` fallback: {prefix}"
    );
    assert!(!prefix.contains("default_tenant"), "prefix={prefix}");

    // No tenant scope → None (single-tenant/embedded keeps the legacy fallback).
    assert!(
        dml.resolve_warehouse_object_prefix("facts", None)
            .await
            .expect("resolve prefix (no tenant)")
            .is_none()
    );
}

/// TD-113 Phase 2: a tenant-scoped CREATE records the owning tenant on the
/// namespace, so it becomes DR-addressable (both `namespace_id` and
/// `tenant_id` populated).
#[tokio::test]
async fn scoped_create_makes_namespace_dr_addressable() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    let ddl = DdlService::new(manager.clone());
    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let create = parser
        .parse_ddl("CREATE TABLE inv (id TEXT NOT NULL, PRIMARY KEY (id));")
        .expect("parse ddl")
        .expect("ddl stmt");
    ddl.execute_scoped(create, Some("acmecorp"))
        .await
        .expect("create table");

    let (catalog, table_id) = manager
        .resolve_table_scoped("inv", Some("acmecorp"))
        .await
        .expect("resolve scoped");
    let ns = catalog
        .get_namespace(&table_id.namespace)
        .await
        .expect("get namespace");
    assert_eq!(ns.tenant_id.as_deref(), Some("acmecorp"));
    assert!(ns.namespace_id.is_some());
    assert!(ns.is_dr_addressable(), "namespace must be DR-addressable");
}

/// P3 end-to-end: materialize a table to a Parquet snapshot on a REOPENABLE
/// (file://) object store, then read it back through the DataFusion OLAP reader
/// (`ObjectStoreParquetTable::open(location)` + `ctx.sql`) — proving the published
/// `location` is exactly what the router registers and queries. Feature-gated
/// because the DataFusion reader lives behind `datafusion-integration`.
#[cfg(feature = "datafusion-integration")]
// Watchdog-bounded `current_thread` runtime instead of `#[tokio::test]`'s
// multi-threaded runtime, which intermittently hangs late in the suite and —
// with no 30s bound — rides to nextest's 120s slow-timeout (CLAUDE.md #11).
#[test]
fn materialized_table_is_readable_through_datafusion_reader() {
    crate::query::execution::test_runtime::run_with_timeout(30, async {
        use crate::datafusion::create_session_context;
        use crate::datafusion::engine_adapters::register_object_store_parquet_location;
        use crate::services::record_store::DirectWalTableRecordStore;
        use crate::services::{FramedTableWalAppender, MemtableRecordStorage};
        use proximadb_iceberg_engine::IcebergObjectStoreBridge;

        let temp_dir = tempfile::tempdir().expect("tempdir");
        let wal_path = temp_dir.path().join("dml-mat-e2e.wal");
        let manager = Arc::new(CatalogManager::new());
        manager
            .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
            .await
            .expect("native catalog");
        let ddl = DdlService::new(manager.clone());
        ddl.execute(DdlStatement::CreateNamespace {
            namespace: vec!["default".to_string()],
            if_not_exists: true,
            properties: HashMap::new(),
        })
        .await
        .expect("create namespace");
        let parser = crate::query::sql_frontend::SqlFrontendParser::new();
        let ddl_stmt = parser
            .parse_ddl(
                "CREATE TABLE inv (id TEXT NOT NULL, status TEXT, qty INT, PRIMARY KEY (id));",
            )
            .expect("parse ddl")
            .expect("ddl stmt");
        ddl.execute(ddl_stmt).await.expect("create table");
        let record_store = Arc::new(DirectWalTableRecordStore::new(
            Arc::new(MemtableRecordStorage::new()),
            Arc::new(
                FramedTableWalAppender::open(&wal_path)
                    .await
                    .expect("open WAL"),
            ),
        ));
        let dml = DmlService::with_record_store_and_table_write_executor(
            manager.clone(),
            record_store,
            Arc::new(PlannedOnlyTableWriteExecutor::new()),
        );
        for (id, status, qty) in [
            ("i1", "active", 5),
            ("i2", "active", 15),
            ("i3", "idle", 25),
        ] {
            let stmt = parser
                .parse_dml(&format!(
                    "INSERT INTO inv (id, status, qty) VALUES ('{id}', '{status}', {qty});"
                ))
                .expect("parse insert")
                .expect("insert stmt");
            dml.execute(stmt).await.expect("insert");
        }

        // A file:// store the OLAP reader can REOPEN from the published URL.
        let store_dir = tempfile::tempdir().expect("store tempdir");
        let root_url = format!("file://{}", store_dir.path().display());
        let bridge = Arc::new(IcebergObjectStoreBridge::from_url(&root_url).expect("bridge"));

        let location = dml
            .materialize_table_to_parquet(&*bridge, &root_url, "inv", None)
            .await
            .expect("materialize");

        // Read the published location back through the DataFusion OLAP reader.
        let ctx = create_session_context().expect("session ctx");
        register_object_store_parquet_location(
            &ctx,
            "inv_parquet",
            &location,
            None,
            proximadb_data_model::StatsTrust::Trusted,
        )
        .await
        .expect("register parquet location");
        let batches = ctx
            .sql("SELECT * FROM inv_parquet")
            .await
            .expect("plan select")
            .collect()
            .await
            .expect("collect");
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(
            total, 3,
            "DataFusion reads all materialized rows from the published location"
        );
    });
}

/// `point_lookup_relational` (PATH B PkLookup backend) returns the full row by
/// primary key in `schema.columns` order, and `None` for a missing key.
#[tokio::test]
async fn point_lookup_relational_returns_full_row_by_pk() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("dml-pklookup.wal");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    let ddl = DdlService::new(manager.clone());
    ddl.execute(DdlStatement::CreateNamespace {
        namespace: vec!["default".to_string()],
        if_not_exists: true,
        properties: HashMap::new(),
    })
    .await
    .expect("create namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
        .parse_ddl("CREATE TABLE inv (id TEXT NOT NULL, status TEXT, qty INT, PRIMARY KEY (id));")
        .expect("parse ddl")
        .expect("ddl stmt");
    ddl.execute(ddl_stmt).await.expect("create table");

    let record_store = Arc::new(DirectWalTableRecordStore::new(
        Arc::new(MemtableRecordStorage::new()),
        Arc::new(
            FramedTableWalAppender::open(&wal_path)
                .await
                .expect("open WAL"),
        ),
    ));
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store,
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );
    for (id, status, qty) in [("i1", "active", 5), ("i2", "active", 15)] {
        let stmt = parser
            .parse_dml(&format!(
                "INSERT INTO inv (id, status, qty) VALUES ('{id}', '{status}', {qty});"
            ))
            .expect("parse insert")
            .expect("insert stmt");
        dml.execute(stmt).await.expect("insert");
    }

    // Existing key → full row [id, status, qty] in schema order.
    let row = dml
        .point_lookup_relational("inv", "i2", None)
        .await
        .expect("lookup")
        .expect("row present");
    assert_eq!(row.len(), 3);
    assert_eq!(row[0], ProximaValue::String("i2".to_string()));
    assert_eq!(row[1], ProximaValue::String("active".to_string()));
    assert_eq!(row[2], ProximaValue::Int32(15));

    // Missing key → None.
    let missing = dml
        .point_lookup_relational("inv", "nope", None)
        .await
        .expect("lookup");
    assert!(missing.is_none(), "absent key returns None");
}

/// UPDATE/DELETE WHERE must honor NON-primary-key predicates (and the full
/// predicate of a mixed `pk = x AND col = y`), via the shared scan-filter
/// push-down — not the prior PK-only `extract_ids_from_where` which silently
/// ignored non-PK conditions.
#[tokio::test]
async fn update_delete_honor_non_primary_key_where_predicates() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("dml-nonpk.wal");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    let ddl = DdlService::new(manager.clone());
    ddl.execute(DdlStatement::CreateNamespace {
        namespace: vec!["default".to_string()],
        if_not_exists: true,
        properties: HashMap::new(),
    })
    .await
    .expect("create namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
        .parse_ddl("CREATE TABLE orders (id TEXT NOT NULL, status TEXT, PRIMARY KEY (id));")
        .expect("parse ddl")
        .expect("ddl stmt");
    ddl.execute(ddl_stmt).await.expect("create table");

    let wal_appender = Arc::new(
        FramedTableWalAppender::open(&wal_path)
            .await
            .expect("open WAL"),
    );
    let record_store = Arc::new(DirectWalTableRecordStore::new(
        Arc::new(MemtableRecordStorage::new()),
        wal_appender,
    ));
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store,
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );

    async fn exec(
        dml: &DmlService,
        parser: &crate::query::sql_frontend::SqlFrontendParser,
        sql: &str,
    ) -> DmlResult {
        let stmt = parser.parse_dml(sql).expect("parse").expect("dml stmt");
        dml.execute(stmt).await.expect("execute")
    }
    async fn status_of(dml: &DmlService, id: &str) -> Option<String> {
        let sel = dml
            .select_table_records_with_projection(
                "orders",
                &["id".to_string(), "status".to_string()],
                None,
                &[RelationalSelectPredicateInput {
                    column_name: "id".to_string(),
                    condition: RelationalSelectPredicateCondition::Comparison {
                        operator: RelationalSelectPredicateOperator::Equal,
                        literal: id.to_string(),
                    },
                }],
                None,
            )
            .await
            .expect("select");
        sel.rows.first().map(|row| match &row[1] {
            ProximaValue::String(s) => s.clone(),
            other => format!("{other:?}"),
        })
    }

    for (id, status) in [("o1", "active"), ("o2", "active"), ("o3", "inactive")] {
        exec(
            &dml,
            &parser,
            &format!("INSERT INTO orders (id, status) VALUES ('{id}', '{status}');"),
        )
        .await;
    }

    // (1) Non-PK WHERE UPDATE: only the two 'active' rows change.
    let r = exec(
        &dml,
        &parser,
        "UPDATE orders SET status = 'archived' WHERE status = 'active';",
    )
    .await;
    assert!(r.success);
    assert_eq!(r.rows_affected, 2, "only the two active rows update");
    assert_eq!(status_of(&dml, "o1").await.as_deref(), Some("archived"));
    assert_eq!(status_of(&dml, "o2").await.as_deref(), Some("archived"));
    assert_eq!(
        status_of(&dml, "o3").await.as_deref(),
        Some("inactive"),
        "non-matching row must be untouched"
    );

    // (2) Mixed pk + non-pk, the silent-bug fix: o3 is 'inactive', so the
    // `AND status = 'active'` must prevent the update even though id matches.
    let r = exec(
        &dml,
        &parser,
        "UPDATE orders SET status = 'hacked' WHERE id = 'o3' AND status = 'active';",
    )
    .await;
    assert!(r.success);
    assert_eq!(
        r.rows_affected, 0,
        "id matches but the non-PK condition fails"
    );
    assert_eq!(
        status_of(&dml, "o3").await.as_deref(),
        Some("inactive"),
        "row must NOT be mutated when the full predicate is not satisfied"
    );

    // (3) Non-PK WHERE DELETE: removes only the archived rows.
    let r = exec(
        &dml,
        &parser,
        "DELETE FROM orders WHERE status = 'archived';",
    )
    .await;
    assert!(r.success);
    assert_eq!(r.rows_affected, 2);
    assert_eq!(status_of(&dml, "o1").await, None, "o1 deleted");
    assert_eq!(status_of(&dml, "o2").await, None, "o2 deleted");
    assert_eq!(
        status_of(&dml, "o3").await.as_deref(),
        Some("inactive"),
        "o3 survives"
    );
}

/// INSERT and DELETE through `DmlService` bump catalog row-count statistics
/// so that subsequent route decisions reflect the current approximate cardinality.
#[tokio::test]
async fn insert_and_delete_update_catalog_row_count_statistics() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("stats-feedback.wal");

    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");

    let ddl = DdlService::new(manager.clone());
    ddl.execute(DdlStatement::CreateNamespace {
        namespace: vec!["default".to_string()],
        if_not_exists: true,
        properties: HashMap::new(),
    })
    .await
    .expect("create namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let create_sql = "CREATE TABLE stat_rows (id TEXT NOT NULL, val TEXT, PRIMARY KEY (id));";
    let ddl_stmt = parser
        .parse_ddl(create_sql)
        .expect("parse ddl")
        .expect("ddl stmt");
    ddl.execute(ddl_stmt).await.expect("create table");

    let wal_appender = Arc::new(
        FramedTableWalAppender::open(&wal_path)
            .await
            .expect("open WAL"),
    );
    let record_store = Arc::new(DirectWalTableRecordStore::new(
        Arc::new(MemtableRecordStorage::new()),
        wal_appender,
    ));
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store,
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );

    // Pre-condition: row_count starts at 0 (no stats written yet).
    let (catalog_pre, table_id_pre) = manager
        .resolve_table("stat_rows")
        .await
        .expect("resolve table");
    let stats_pre = catalog_pre
        .get_statistics(&table_id_pre)
        .await
        .unwrap_or_default();
    assert_eq!(stats_pre.row_count, 0, "row_count must start at 0");

    // INSERT 3 rows → bump_row_count_stats adds +3.
    for i in 1..=3u32 {
        let sql = format!(
            "INSERT INTO stat_rows (id, val) VALUES ('r{}', 'v{}');",
            i, i
        );
        let stmt = parser
            .parse_dml(&sql)
            .expect("parse insert")
            .expect("dml stmt");
        dml.execute(stmt).await.expect("insert");
    }

    let (catalog_after_insert, table_id_after_insert) = manager
        .resolve_table("stat_rows")
        .await
        .expect("resolve table after insert");
    let stats_after_insert = catalog_after_insert
        .get_statistics(&table_id_after_insert)
        .await
        .unwrap_or_default();
    assert_eq!(
        stats_after_insert.row_count, 3,
        "row_count must be 3 after three inserts"
    );

    // DELETE 1 row → bump_row_count_stats subtracts 1.
    let del_stmt = parser
        .parse_dml("DELETE FROM stat_rows WHERE id = 'r1';")
        .expect("parse delete")
        .expect("dml stmt");
    dml.execute(del_stmt).await.expect("delete");

    let (catalog_after_delete, table_id_after_delete) = manager
        .resolve_table("stat_rows")
        .await
        .expect("resolve table after delete");
    let stats_after_delete = catalog_after_delete
        .get_statistics(&table_id_after_delete)
        .await
        .unwrap_or_default();
    assert_eq!(
        stats_after_delete.row_count, 2,
        "row_count must be 2 after one delete"
    );
    assert!(
        stats_after_delete.last_analyzed_ms.is_some(),
        "last_analyzed_ms must be set"
    );
}

/// T8: After INSERT with some NULL column values, `column_stats[col].null_count`
/// reflects the number of NULLs written. Null-free inserts leave null_count at 0/absent.
#[tokio::test]
async fn insert_null_values_update_column_null_count_statistics() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("col-stats.wal");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    DdlService::new(manager.clone())
        .execute(DdlStatement::CreateNamespace {
            namespace: vec!["default".to_string()],
            if_not_exists: true,
            properties: HashMap::new(),
        })
        .await
        .expect("namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
        .parse_ddl("CREATE TABLE nullable_tbl (id TEXT NOT NULL, note TEXT, score FLOAT);")
        .expect("parse ddl")
        .expect("ddl");
    DdlService::new(manager.clone())
        .execute(ddl_stmt)
        .await
        .expect("create table");

    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        Arc::new(DirectWalTableRecordStore::new(
            Arc::new(MemtableRecordStorage::new()),
            Arc::new(
                FramedTableWalAppender::open(&wal_path)
                    .await
                    .expect("open WAL"),
            ),
        )),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );

    // Row 1: note = NULL, score present.
    // Row 2: note present, score = NULL.
    // Row 3: both present.
    for stmt_sql in [
        "INSERT INTO nullable_tbl (id, note, score) VALUES ('r1', NULL, 1.0);",
        "INSERT INTO nullable_tbl (id, note, score) VALUES ('r2', 'hello', NULL);",
        "INSERT INTO nullable_tbl (id, note, score) VALUES ('r3', 'world', 2.0);",
    ] {
        let stmt = parser
            .parse_dml(stmt_sql)
            .expect("parse insert")
            .expect("dml stmt");
        dml.execute(stmt).await.expect("insert");
    }

    let (catalog, table_id) = manager
        .resolve_table("nullable_tbl")
        .await
        .expect("resolve");
    let stats = catalog.get_statistics(&table_id).await.unwrap_or_default();

    assert_eq!(stats.row_count, 3, "three rows inserted");
    assert_eq!(
        stats.column_stats.get("note").and_then(|cs| cs.null_count),
        Some(1),
        "note has 1 NULL across 3 inserts"
    );
    assert_eq!(
        stats.column_stats.get("score").and_then(|cs| cs.null_count),
        Some(1),
        "score has 1 NULL across 3 inserts"
    );
    // id is NOT NULL — null_count entry should be absent or 0.
    let id_null_count = stats
        .column_stats
        .get("id")
        .and_then(|cs| cs.null_count)
        .unwrap_or(0);
    assert_eq!(
        id_null_count, 0,
        "id is NOT NULL, no nulls should be counted"
    );
}

/// T9: After INSERT with NULL in nullable columns, `scan_table_records` returns rows with
/// `ProximaValue::Null` for those fields, and projection produces empty string for NULL values.
#[tokio::test]
async fn insert_nullable_values_are_scannable_and_project_null_correctly() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("scan-null.wal");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    DdlService::new(manager.clone())
        .execute(DdlStatement::CreateNamespace {
            namespace: vec!["default".to_string()],
            if_not_exists: true,
            properties: HashMap::new(),
        })
        .await
        .expect("namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
            .parse_ddl("CREATE TABLE scan_null_tbl (id TEXT NOT NULL, tag TEXT, rating FLOAT, PRIMARY KEY (id));")
            .expect("parse ddl")
            .expect("ddl");
    DdlService::new(manager.clone())
        .execute(ddl_stmt)
        .await
        .expect("create table");

    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        Arc::new(DirectWalTableRecordStore::new(
            Arc::new(MemtableRecordStorage::new()),
            Arc::new(
                FramedTableWalAppender::open(&wal_path)
                    .await
                    .expect("open WAL"),
            ),
        )),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );

    for sql in [
        "INSERT INTO scan_null_tbl (id, tag, rating) VALUES ('x1', NULL, 9.5);",
        "INSERT INTO scan_null_tbl (id, tag, rating) VALUES ('x2', 'beta', NULL);",
    ] {
        let stmt = parser.parse_dml(sql).expect("parse").expect("dml");
        dml.execute(stmt).await.expect("insert");
    }

    let (_schema, records) = dml
        .scan_table_records("scan_null_tbl", None)
        .await
        .expect("scan");
    assert_eq!(records.len(), 2, "two rows scanned");

    let find = |oid: &str| {
        records
            .iter()
            .find(|r| r.oid == oid)
            .unwrap_or_else(|| panic!("row {oid} not found"))
    };
    let prop_value = |record: &ProximaRecord, col: &str| -> Option<ProximaValue> {
        match record.props.get(col) {
            Some(proximadb_records::ProximaTreeNode::Value(v)) => Some(v.clone()),
            _ => None,
        }
    };
    let r_x1 = find("x1");
    assert_eq!(
        prop_value(r_x1, "tag"),
        Some(ProximaValue::Null),
        "x1.tag should be Null"
    );
    let r_x2 = find("x2");
    assert_eq!(
        prop_value(r_x2, "rating"),
        Some(ProximaValue::Null),
        "x2.rating should be Null"
    );

    // Projection: NULL columns surface as ProximaValue::Null in SELECT output.
    let result = dml
        .select_table_records_with_projection(
            "scan_null_tbl",
            &["id".to_string(), "tag".to_string(), "rating".to_string()],
            None,
            &[],
            None,
        )
        .await
        .expect("select");
    let x1_id = ProximaValue::String("x1".to_string());
    let x2_id = ProximaValue::String("x2".to_string());
    let row_x1 = result
        .rows
        .iter()
        .find(|r| r.first() == Some(&x1_id))
        .expect("x1 row in projection");
    // columns order: id, tag, rating → indices 0, 1, 2
    assert_eq!(
        row_x1.get(1),
        Some(&ProximaValue::Null),
        "x1.tag projects as Null"
    );
    let row_x2 = result
        .rows
        .iter()
        .find(|r| r.first() == Some(&x2_id))
        .expect("x2 row in projection");
    assert_eq!(
        row_x2.get(2),
        Some(&ProximaValue::Null),
        "x2.rating projects as Null"
    );
}

/// T9: `IS NULL` and `IS NOT NULL` predicates correctly filter rows with `ProximaValue::Null`
/// versus non-null values in `scan_table_records_with_predicates`.
#[tokio::test]
async fn is_null_predicate_filters_nullable_column_rows() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("predicate-null.wal");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    DdlService::new(manager.clone())
        .execute(DdlStatement::CreateNamespace {
            namespace: vec!["default".to_string()],
            if_not_exists: true,
            properties: HashMap::new(),
        })
        .await
        .expect("namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
        .parse_ddl("CREATE TABLE null_pred_tbl (id TEXT NOT NULL, label TEXT, PRIMARY KEY (id));")
        .expect("parse ddl")
        .expect("ddl");
    DdlService::new(manager.clone())
        .execute(ddl_stmt)
        .await
        .expect("create table");

    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        Arc::new(DirectWalTableRecordStore::new(
            Arc::new(MemtableRecordStorage::new()),
            Arc::new(
                FramedTableWalAppender::open(&wal_path)
                    .await
                    .expect("open WAL"),
            ),
        )),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );

    for sql in [
        "INSERT INTO null_pred_tbl (id, label) VALUES ('p1', NULL);",
        "INSERT INTO null_pred_tbl (id, label) VALUES ('p2', 'hello');",
        "INSERT INTO null_pred_tbl (id, label) VALUES ('p3', NULL);",
    ] {
        let stmt = parser.parse_dml(sql).expect("parse").expect("dml");
        dml.execute(stmt).await.expect("insert");
    }

    // IS NULL: should return p1 and p3 only.
    let is_null_predicate = RelationalSelectPredicateInput {
        column_name: "label".to_string(),
        condition: RelationalSelectPredicateCondition::IsNull { negated: false },
    };
    let (_schema, null_rows) = dml
        .scan_table_records_with_predicates("null_pred_tbl", None, &[is_null_predicate])
        .await
        .expect("scan IS NULL");
    let null_oids: Vec<&str> = null_rows.iter().map(|r| r.oid.as_str()).collect();
    assert!(null_oids.contains(&"p1"), "p1 must match IS NULL");
    assert!(null_oids.contains(&"p3"), "p3 must match IS NULL");
    assert!(!null_oids.contains(&"p2"), "p2 must not match IS NULL");

    // IS NOT NULL: should return only p2.
    let is_not_null_predicate = RelationalSelectPredicateInput {
        column_name: "label".to_string(),
        condition: RelationalSelectPredicateCondition::IsNull { negated: true },
    };
    let (_schema, not_null_rows) = dml
        .scan_table_records_with_predicates("null_pred_tbl", None, &[is_not_null_predicate])
        .await
        .expect("scan IS NOT NULL");
    let not_null_oids: Vec<&str> = not_null_rows.iter().map(|r| r.oid.as_str()).collect();
    assert!(not_null_oids.contains(&"p2"), "p2 must match IS NOT NULL");
    assert!(
        !not_null_oids.contains(&"p1"),
        "p1 must not match IS NOT NULL"
    );
    assert!(
        !not_null_oids.contains(&"p3"),
        "p3 must not match IS NOT NULL"
    );
}

/// T4: NOT IN predicate via `scan_table_records_with_predicates` excludes rows whose
/// column value appears in the exclusion set; IN includes only rows whose value is in the set.
#[tokio::test]
async fn in_and_not_in_predicates_filter_correctly() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("in-pred.wal");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    DdlService::new(manager.clone())
        .execute(DdlStatement::CreateNamespace {
            namespace: vec!["default".to_string()],
            if_not_exists: true,
            properties: HashMap::new(),
        })
        .await
        .expect("namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
        .parse_ddl(
            "CREATE TABLE in_pred_tbl (id TEXT NOT NULL, status TEXT NOT NULL, PRIMARY KEY (id));",
        )
        .expect("parse ddl")
        .expect("ddl");
    DdlService::new(manager.clone())
        .execute(ddl_stmt)
        .await
        .expect("create table");

    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        Arc::new(DirectWalTableRecordStore::new(
            Arc::new(MemtableRecordStorage::new()),
            Arc::new(
                FramedTableWalAppender::open(&wal_path)
                    .await
                    .expect("open WAL"),
            ),
        )),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );

    for sql in [
        "INSERT INTO in_pred_tbl (id, status) VALUES ('i1', 'active');",
        "INSERT INTO in_pred_tbl (id, status) VALUES ('i2', 'inactive');",
        "INSERT INTO in_pred_tbl (id, status) VALUES ('i3', 'pending');",
        "INSERT INTO in_pred_tbl (id, status) VALUES ('i4', 'active');",
    ] {
        let stmt = parser.parse_dml(sql).expect("parse").expect("dml");
        dml.execute(stmt).await.expect("insert");
    }

    // IN ('active', 'pending'): should return i1, i3, i4.
    let in_predicate = RelationalSelectPredicateInput {
        column_name: "status".to_string(),
        condition: RelationalSelectPredicateCondition::In {
            literals: vec!["active".to_string(), "pending".to_string()],
            negated: false,
        },
    };
    let (_schema, in_rows) = dml
        .scan_table_records_with_predicates("in_pred_tbl", None, &[in_predicate])
        .await
        .expect("scan IN");
    let in_oids: Vec<&str> = in_rows.iter().map(|r| r.oid.as_str()).collect();
    assert!(in_oids.contains(&"i1"), "i1 (active) matches IN");
    assert!(in_oids.contains(&"i3"), "i3 (pending) matches IN");
    assert!(in_oids.contains(&"i4"), "i4 (active) matches IN");
    assert!(!in_oids.contains(&"i2"), "i2 (inactive) excluded from IN");

    // NOT IN ('active'): should return i2 and i3.
    let not_in_predicate = RelationalSelectPredicateInput {
        column_name: "status".to_string(),
        condition: RelationalSelectPredicateCondition::In {
            literals: vec!["active".to_string()],
            negated: true,
        },
    };
    let (_schema, not_in_rows) = dml
        .scan_table_records_with_predicates("in_pred_tbl", None, &[not_in_predicate])
        .await
        .expect("scan NOT IN");
    let not_in_oids: Vec<&str> = not_in_rows.iter().map(|r| r.oid.as_str()).collect();
    assert!(
        not_in_oids.contains(&"i2"),
        "i2 (inactive) matches NOT IN ('active')"
    );
    assert!(
        not_in_oids.contains(&"i3"),
        "i3 (pending) matches NOT IN ('active')"
    );
    assert!(
        !not_in_oids.contains(&"i1"),
        "i1 (active) excluded by NOT IN"
    );
    assert!(
        !not_in_oids.contains(&"i4"),
        "i4 (active) excluded by NOT IN"
    );
}

/// T4: LIKE and NOT LIKE predicates filter rows correctly via `scan_table_records_with_predicates`.
#[tokio::test]
async fn like_predicate_filters_correctly() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("like-pred.wal");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    DdlService::new(manager.clone())
        .execute(DdlStatement::CreateNamespace {
            namespace: vec!["default".to_string()],
            if_not_exists: true,
            properties: HashMap::new(),
        })
        .await
        .expect("namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
        .parse_ddl(
            "CREATE TABLE like_tbl (id TEXT NOT NULL, name TEXT NOT NULL, PRIMARY KEY (id));",
        )
        .expect("parse ddl")
        .expect("ddl");
    DdlService::new(manager.clone())
        .execute(ddl_stmt)
        .await
        .expect("create table");

    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        Arc::new(DirectWalTableRecordStore::new(
            Arc::new(MemtableRecordStorage::new()),
            Arc::new(
                FramedTableWalAppender::open(&wal_path)
                    .await
                    .expect("open WAL"),
            ),
        )),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );

    for sql in [
        "INSERT INTO like_tbl (id, name) VALUES ('l1', 'alice_admin');",
        "INSERT INTO like_tbl (id, name) VALUES ('l2', 'bob_user');",
        "INSERT INTO like_tbl (id, name) VALUES ('l3', 'alice_user');",
        "INSERT INTO like_tbl (id, name) VALUES ('l4', 'charlie');",
    ] {
        let stmt = parser.parse_dml(sql).expect("parse").expect("dml");
        dml.execute(stmt).await.expect("insert");
    }

    // LIKE 'alice%': should return l1 and l3.
    let like_predicate = RelationalSelectPredicateInput {
        column_name: "name".to_string(),
        condition: RelationalSelectPredicateCondition::Like {
            pattern: "alice%".to_string(),
            negated: false,
        },
    };
    let (_schema, like_rows) = dml
        .scan_table_records_with_predicates("like_tbl", None, &[like_predicate])
        .await
        .expect("scan LIKE");
    let like_oids: Vec<&str> = like_rows.iter().map(|r| r.oid.as_str()).collect();
    assert!(
        like_oids.contains(&"l1"),
        "l1 (alice_admin) matches LIKE 'alice%'"
    );
    assert!(
        like_oids.contains(&"l3"),
        "l3 (alice_user) matches LIKE 'alice%'"
    );
    assert!(!like_oids.contains(&"l2"), "l2 (bob_user) excluded");
    assert!(!like_oids.contains(&"l4"), "l4 (charlie) excluded");

    // NOT LIKE 'alice%': should return l2 and l4.
    let not_like_predicate = RelationalSelectPredicateInput {
        column_name: "name".to_string(),
        condition: RelationalSelectPredicateCondition::Like {
            pattern: "alice%".to_string(),
            negated: true,
        },
    };
    let (_schema, not_like_rows) = dml
        .scan_table_records_with_predicates("like_tbl", None, &[not_like_predicate])
        .await
        .expect("scan NOT LIKE");
    let not_like_oids: Vec<&str> = not_like_rows.iter().map(|r| r.oid.as_str()).collect();
    assert!(
        not_like_oids.contains(&"l2"),
        "l2 (bob_user) matches NOT LIKE 'alice%'"
    );
    assert!(
        not_like_oids.contains(&"l4"),
        "l4 (charlie) matches NOT LIKE 'alice%'"
    );
    assert!(!not_like_oids.contains(&"l1"), "l1 excluded by NOT LIKE");
    assert!(!not_like_oids.contains(&"l3"), "l3 excluded by NOT LIKE");
}

/// T9: UPDATE SET col = NULL on a nullable column succeeds and the column reads back as
/// `ProximaValue::Null`; UPDATE SET col = NULL on a NOT NULL column is rejected.
#[tokio::test]
async fn update_nullable_column_to_null_succeeds() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("update-null.wal");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    DdlService::new(manager.clone())
        .execute(DdlStatement::CreateNamespace {
            namespace: vec!["default".to_string()],
            if_not_exists: true,
            properties: HashMap::new(),
        })
        .await
        .expect("namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
            .parse_ddl("CREATE TABLE upd_null_tbl (id TEXT NOT NULL, note TEXT, score FLOAT NOT NULL, PRIMARY KEY (id));")
            .expect("parse ddl")
            .expect("ddl");
    DdlService::new(manager.clone())
        .execute(ddl_stmt)
        .await
        .expect("create table");

    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        Arc::new(DirectWalTableRecordStore::new(
            Arc::new(MemtableRecordStorage::new()),
            Arc::new(
                FramedTableWalAppender::open(&wal_path)
                    .await
                    .expect("open WAL"),
            ),
        )),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );

    let insert_stmt = parser
        .parse_dml("INSERT INTO upd_null_tbl (id, note, score) VALUES ('u1', 'initial', 1.5);")
        .expect("parse")
        .expect("dml");
    dml.execute(insert_stmt).await.expect("insert");

    // UPDATE nullable column to NULL — should succeed.
    let upd_stmt = parser
        .parse_dml("UPDATE upd_null_tbl SET note = NULL WHERE id = 'u1';")
        .expect("parse update")
        .expect("dml");
    dml.execute(upd_stmt)
        .await
        .expect("UPDATE note=NULL should succeed for nullable column");

    let prop_val = |record: &ProximaRecord, col: &str| -> Option<ProximaValue> {
        match record.props.get(col) {
            Some(proximadb_records::ProximaTreeNode::Value(v)) => Some(v.clone()),
            _ => None,
        }
    };

    let (_schema, rows) = dml
        .scan_table_records("upd_null_tbl", None)
        .await
        .expect("scan");
    let u1 = rows.iter().find(|r| r.oid == "u1").expect("u1 row");
    assert_eq!(
        prop_val(u1, "note"),
        Some(ProximaValue::Null),
        "note should be Null after UPDATE SET note=NULL"
    );

    // UPDATE NOT NULL column to NULL — should be rejected.
    let bad_upd_stmt = parser
        .parse_dml("UPDATE upd_null_tbl SET score = NULL WHERE id = 'u1';")
        .expect("parse bad update")
        .expect("dml");
    let err = dml.execute(bad_upd_stmt).await;
    assert!(
        err.is_err(),
        "UPDATE score=NULL should fail for NOT NULL column"
    );
    let err_msg = err.unwrap_err().to_string();
    assert!(
        err_msg.contains("cannot be NULL") || err_msg.contains("not nullable"),
        "error should mention NULL constraint: {err_msg}"
    );
}

/// CREATE TABLE via `DdlService` appears in `information_schema.tables` and `columns`,
/// and `DmlService` can resolve the table metadata immediately after DDL. Covers T9
/// DDL metadata round-trip: catalog write → introspection read → DML resolve.
#[tokio::test]
async fn ddl_create_table_visible_in_introspection_and_resolvable_by_dml() {
    use crate::services::CatalogIntrospectionService;

    let temp_dir = tempfile::tempdir().expect("tempdir");

    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");

    let ddl = DdlService::new(manager.clone());
    ddl.execute(DdlStatement::CreateNamespace {
        namespace: vec!["default".to_string()],
        if_not_exists: true,
        properties: HashMap::new(),
    })
    .await
    .expect("create namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let create_sql = "CREATE TABLE meta_test (id TEXT NOT NULL, label TEXT, score DECIMAL(10,4), PRIMARY KEY (id));";
    let ddl_stmt = parser
        .parse_ddl(create_sql)
        .expect("parse create table")
        .expect("ddl stmt");
    ddl.execute(ddl_stmt).await.expect("create table");

    // information_schema.tables must include the newly created table.
    let introspection = CatalogIntrospectionService::new(manager.clone());
    let result = introspection
            .execute_select(
                "SELECT table_schema, table_name FROM information_schema.tables WHERE table_name = 'meta_test'",
            )
            .await
            .expect("catalog introspection query")
            .expect("must return a result");
    let tables_result = result
        .rows
        .iter()
        .any(|row| row.iter().any(|v| v.contains("meta_test")));
    assert!(
        tables_result,
        "meta_test must appear in information_schema.tables"
    );

    // information_schema.columns must include all declared columns.
    let col_result = introspection
        .execute_select(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'meta_test'",
        )
        .await
        .expect("columns introspection query")
        .expect("must return columns result");
    let all_values: Vec<&str> = col_result
        .rows
        .iter()
        .flat_map(|row| row.iter().map(|v| v.as_str()))
        .collect();
    assert!(
        all_values.contains(&"id"),
        "id column must appear in information_schema.columns"
    );
    assert!(
        all_values.contains(&"label"),
        "label column must appear in information_schema.columns"
    );
    assert!(
        all_values.contains(&"score"),
        "score column must appear in information_schema.columns"
    );

    // DmlService must be able to resolve the table — verifies catalog → DML integration.
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        Arc::new(ExplainOnlyRecordStore),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );
    let (catalog, table_id) = manager
        .resolve_table("meta_test")
        .await
        .expect("DmlService must resolve DDL-created table");
    let schema = catalog
        .get_table(&table_id)
        .await
        .expect("get table schema");
    assert_eq!(schema.name, "meta_test");
    assert_eq!(schema.primary_key.len(), 1);
    assert_eq!(schema.primary_key[0], "id");

    // Explain a write plan into the table — end-to-end DDL → route planner round-trip.
    let explain_stmt = parser
        .parse_dml("INSERT INTO meta_test SELECT * FROM meta_test;")
        .expect("parse explain dml")
        .expect("dml stmt");
    let explanation = dml
        .explain_table_write(explain_stmt)
        .await
        .expect("explain table write must succeed for DDL-created table");
    assert_eq!(explanation.target_table, "meta_test");
}

#[tokio::test]
async fn ddl_constraints_surface_as_route_metadata_gaps() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    DdlService::new(manager.clone())
        .execute(DdlStatement::CreateNamespace {
            namespace: vec!["default".to_string()],
            if_not_exists: true,
            properties: HashMap::new(),
        })
        .await
        .expect("namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let customers = parser
        .parse_ddl("CREATE TABLE customers (id TEXT NOT NULL, PRIMARY KEY (id));")
        .expect("parse customers")
        .expect("customers ddl");
    DdlService::new(manager.clone())
        .execute(customers)
        .await
        .expect("create customers");

    let orders = parser
        .parse_ddl(
            "CREATE TABLE orders_with_constraints (
                    id TEXT NOT NULL,
                    email TEXT,
                    customer_id TEXT,
                    amount FLOAT,
                    PRIMARY KEY (id),
                    UNIQUE (email),
                    CHECK (amount > 0),
                    FOREIGN KEY (customer_id) REFERENCES customers(id) ON UPDATE CASCADE
                );",
        )
        .expect("parse orders")
        .expect("orders ddl");
    DdlService::new(manager.clone())
        .execute(orders)
        .await
        .expect("create orders");

    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        Arc::new(ExplainOnlyRecordStore),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );
    let explain_stmt = parser
        .parse_dml("INSERT INTO orders_with_constraints SELECT * FROM orders_with_constraints;")
        .expect("parse explain dml")
        .expect("dml stmt");
    let explanation = dml
        .explain_table_write(explain_stmt)
        .await
        .expect("explain table write");

    assert_eq!(explanation.target_table, "orders_with_constraints");
    assert!(
        explanation
            .route_metadata
            .constraint_enforcement
            .starts_with("partial_native_enforced:")
    );
    assert!(
        explanation
            .route_metadata
            .constraint_gaps
            .contains(&"unique_indexes_cataloged_not_enforced".to_string())
    );
    assert!(
        explanation
            .route_metadata
            .constraint_enforcement
            .contains("check")
    );
    assert!(
        explanation
            .route_metadata
            .constraint_enforcement
            .contains("unique_non_null_fail_closed")
    );
    assert!(
        explanation
            .route_metadata
            .constraint_enforcement
            .contains("foreign_key_non_null_fail_closed")
    );
    assert!(
        explanation
            .route_metadata
            .constraint_gaps
            .contains(&"foreign_keys_cataloged_not_enforced".to_string())
    );
}

#[tokio::test]
async fn delete_enforces_referential_actions() {
    // TD-110: ON DELETE referential actions. Deleting a parent row triggers
    // NO ACTION (default → reject), CASCADE (delete children), or SET NULL
    // (clear the child FK) on child tables in the same namespace.
    use crate::query::table_write_executor::PlannedOnlyTableWriteExecutor;
    use crate::services::record_store::{DirectWalTableRecordStore, TableRecordGetRequest};
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    let ddl = DdlService::new(manager.clone());
    ddl.execute(DdlStatement::CreateNamespace {
        namespace: vec!["default".to_string()],
        if_not_exists: true,
        properties: HashMap::new(),
    })
    .await
    .expect("create namespace");
    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    for create in [
        "CREATE TABLE customers (id TEXT NOT NULL, name TEXT, PRIMARY KEY (id));",
        "CREATE TABLE orders_restrict (id TEXT NOT NULL, customer_id TEXT, PRIMARY KEY (id), FOREIGN KEY (customer_id) REFERENCES customers (id));",
        "CREATE TABLE orders_cascade (id TEXT NOT NULL, customer_id TEXT, PRIMARY KEY (id), FOREIGN KEY (customer_id) REFERENCES customers (id) ON DELETE CASCADE);",
        "CREATE TABLE orders_setnull (id TEXT NOT NULL, customer_id TEXT, PRIMARY KEY (id), FOREIGN KEY (customer_id) REFERENCES customers (id) ON DELETE SET NULL);",
    ] {
        let stmt = parser
            .parse_ddl(create)
            .expect("parse create")
            .expect("ddl");
        ddl.execute(stmt).await.expect("create table");
    }

    let wal_appender = Arc::new(
        FramedTableWalAppender::open(temp_dir.path().join("refaction.wal"))
            .await
            .expect("open WAL"),
    );
    let record_store = Arc::new(DirectWalTableRecordStore::new(
        Arc::new(MemtableRecordStorage::new()),
        wal_appender,
    ));
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store.clone(),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );
    let run = |sql: &'static str| parser.parse_dml(sql).expect("parse dml").expect("dml");

    // Sanity: the parser must carry ON DELETE actions into the catalog.
    let (catalog, id) = manager
        .resolve_table("orders_cascade")
        .await
        .expect("resolve cascade table");
    let cascade_table = catalog.get_table(&id).await.expect("cascade schema");
    assert!(
        cascade_table
            .relational_capabilities
            .constraints
            .iter()
            .any(|c| matches!(
                c,
                proximadb_catalog::ColumnConstraint::ForeignKey {
                    on_delete: Some(proximadb_catalog::ReferentialAction::Cascade),
                    ..
                }
            )),
        "ON DELETE CASCADE must survive into the catalog schema"
    );

    for sql in [
        "INSERT INTO customers (id, name) VALUES ('c1', 'Alice');",
        "INSERT INTO customers (id, name) VALUES ('c2', 'Bob');",
        "INSERT INTO customers (id, name) VALUES ('c3', 'Cara');",
        "INSERT INTO orders_restrict (id, customer_id) VALUES ('o1', 'c1');",
        "INSERT INTO orders_cascade (id, customer_id) VALUES ('oc1', 'c2');",
        "INSERT INTO orders_setnull (id, customer_id) VALUES ('os1', 'c3');",
    ] {
        dml.execute(run(sql)).await.expect("seed insert");
    }

    // NO ACTION (default): deleting c1 is rejected — o1 still references it.
    let err = dml
        .execute(run("DELETE FROM customers WHERE id = 'c1';"))
        .await
        .expect_err("NO ACTION must reject deleting a referenced parent");
    assert!(
        err.to_string().contains("ON DELETE NO ACTION"),
        "unexpected error: {err}"
    );

    let get = |table: &'static str, key: &'static str| {
        let store = record_store.clone();
        let manager = manager.clone();
        async move {
            let (catalog, id) = manager.resolve_table(table).await.expect("resolve");
            let schema = catalog.get_table(&id).await.expect("schema");
            store
                .get_by_key(
                    &schema,
                    TableRecordGetRequest {
                        table_id: id.name.clone(),
                        key: key.to_string(),
                        include_vector: false,
                        include_props: true,
                    },
                    None,
                    #[cfg(feature = "abac-policy")]
                    &ReadContext::system(SystemReadReason::Statistics, "dml::tests"),
                )
                .await
                .expect("get_by_key")
        }
    };

    // CASCADE: deleting c2 removes the referencing orders_cascade row.
    assert!(get("orders_cascade", "oc1").await.is_some());
    dml.execute(run("DELETE FROM customers WHERE id = 'c2';"))
        .await
        .expect("CASCADE delete of referenced parent must succeed");
    assert!(
        get("orders_cascade", "oc1").await.is_none(),
        "ON DELETE CASCADE must remove the child row"
    );

    // SET NULL: deleting c3 keeps the orders_setnull row but nulls its FK.
    dml.execute(run("DELETE FROM customers WHERE id = 'c3';"))
        .await
        .expect("SET NULL delete of referenced parent must succeed");
    let child = get("orders_setnull", "os1")
        .await
        .expect("ON DELETE SET NULL must keep the child row");
    assert!(
        matches!(
            child.props.get("customer_id"),
            None | Some(ProximaValue::Null)
        ),
        "ON DELETE SET NULL must clear the child FK column, got {:?}",
        child.props.get("customer_id")
    );
}

/// TD-110 S2: composite (multi-column) FOREIGN KEY — a composite-PK parent
/// + composite-FK child; dangling composite tuple is rejected at insert, and
/// CASCADE removes the child when the parent (matched on the full tuple) is
/// deleted.
#[tokio::test]
async fn delete_enforces_composite_fk() {
    use crate::query::table_write_executor::PlannedOnlyTableWriteExecutor;
    use crate::services::record_store::{DirectWalTableRecordStore, TableRecordGetRequest};
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    let ddl = DdlService::new(manager.clone());
    ddl.execute(DdlStatement::CreateNamespace {
        namespace: vec!["default".to_string()],
        if_not_exists: true,
        properties: HashMap::new(),
    })
    .await
    .expect("create namespace");
    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    for create in [
        "CREATE TABLE cmpar (region TEXT NOT NULL, pid TEXT NOT NULL, name TEXT, PRIMARY KEY (region, pid));",
        "CREATE TABLE cmchl_casc (id TEXT NOT NULL, c_region TEXT, c_pid TEXT, PRIMARY KEY (id), FOREIGN KEY (c_region, c_pid) REFERENCES cmpar (region, pid) ON DELETE CASCADE);",
    ] {
        let stmt = parser
            .parse_ddl(create)
            .expect("parse create")
            .expect("ddl");
        ddl.execute(stmt).await.expect("create table");
    }

    let wal_appender = Arc::new(
        FramedTableWalAppender::open(temp_dir.path().join("composite_fk.wal"))
            .await
            .expect("open WAL"),
    );
    let record_store = Arc::new(DirectWalTableRecordStore::new(
        Arc::new(MemtableRecordStorage::new()),
        wal_appender,
    ));
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store.clone(),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );
    let run = |sql: &'static str| parser.parse_dml(sql).expect("parse dml").expect("dml");

    // Sanity: the composite (2-column) ON DELETE CASCADE FK survives into
    // the catalog.
    let (catalog, id) = manager.resolve_table("cmchl_casc").await.expect("resolve");
    let child_table = catalog.get_table(&id).await.expect("schema");
    assert!(
        child_table
            .relational_capabilities
            .constraints
            .iter()
            .any(|c| matches!(
                c,
                proximadb_catalog::ColumnConstraint::ForeignKey {
                    columns,
                    on_delete: Some(proximadb_catalog::ReferentialAction::Cascade),
                    ..
                } if columns.len() == 2
            )),
        "composite ON DELETE CASCADE FK must survive into the catalog"
    );

    dml.execute(run(
        "INSERT INTO cmpar (region, pid, name) VALUES ('us', 'p1', 'Al');",
    ))
    .await
    .expect("seed composite parent");

    // Dangling composite tuple is rejected.
    let err = dml
        .execute(run(
            "INSERT INTO cmchl_casc (id, c_region, c_pid) VALUES ('c9', 'xx', 'zz');",
        ))
        .await
        .expect_err("a dangling composite FK tuple must be rejected");
    assert!(
        err.to_string().contains("violates reference"),
        "unexpected error: {err}"
    );

    // Valid child, then CASCADE on the composite-PK parent removes it.
    dml.execute(run(
        "INSERT INTO cmchl_casc (id, c_region, c_pid) VALUES ('c1', 'us', 'p1');",
    ))
    .await
    .expect("seed composite child");
    let get = |key: &'static str| {
        let store = record_store.clone();
        let manager = manager.clone();
        async move {
            let (catalog, id) = manager.resolve_table("cmchl_casc").await.expect("resolve");
            let schema = catalog.get_table(&id).await.expect("schema");
            store
                .get_by_key(
                    &schema,
                    TableRecordGetRequest {
                        table_id: id.name.clone(),
                        key: key.to_string(),
                        include_vector: false,
                        include_props: true,
                    },
                    None,
                    #[cfg(feature = "abac-policy")]
                    &ReadContext::system(SystemReadReason::Statistics, "dml::tests"),
                )
                .await
                .expect("get_by_key")
        }
    };
    assert!(get("c1").await.is_some());
    dml.execute(run("DELETE FROM cmpar WHERE region = 'us' AND pid = 'p1';"))
        .await
        .expect("composite CASCADE delete must succeed");
    assert!(
        get("c1").await.is_none(),
        "composite CASCADE must remove the child row"
    );
}

/// TD-110 S3: cross-namespace FOREIGN KEY — a child in namespace B whose FK
/// references a parent in namespace A is found by ON DELETE child discovery
/// (a RESTRICT child blocks the parent delete; a CASCADE child is removed).
#[tokio::test]
async fn delete_enforces_cross_namespace_fk() {
    use crate::query::table_write_executor::PlannedOnlyTableWriteExecutor;
    use crate::services::record_store::{DirectWalTableRecordStore, TableRecordGetRequest};
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    let ddl = DdlService::new(manager.clone());
    for ns in ["xns_a", "xns_b"] {
        ddl.execute(DdlStatement::CreateNamespace {
            namespace: vec![ns.to_string()],
            if_not_exists: true,
            properties: HashMap::new(),
        })
        .await
        .expect("create namespace");
    }
    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    for create in [
        "CREATE TABLE xns_a.xpar (id TEXT NOT NULL, name TEXT, PRIMARY KEY (id));",
        "CREATE TABLE xns_b.xchl_casc (id TEXT NOT NULL, pid TEXT, PRIMARY KEY (id), FOREIGN KEY (pid) REFERENCES xns_a.xpar (id) ON DELETE CASCADE);",
        "CREATE TABLE xns_b.xchl_restr (id TEXT NOT NULL, pid TEXT, PRIMARY KEY (id), FOREIGN KEY (pid) REFERENCES xns_a.xpar (id));",
    ] {
        let stmt = parser
            .parse_ddl(create)
            .expect("parse create")
            .expect("ddl");
        ddl.execute(stmt).await.expect("create table");
    }

    let wal_appender = Arc::new(
        FramedTableWalAppender::open(temp_dir.path().join("xns_fk.wal"))
            .await
            .expect("open WAL"),
    );
    let record_store = Arc::new(DirectWalTableRecordStore::new(
        Arc::new(MemtableRecordStorage::new()),
        wal_appender,
    ));
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store.clone(),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );
    let run = |sql: &'static str| parser.parse_dml(sql).expect("parse dml").expect("dml");

    dml.execute(run(
        "INSERT INTO xns_a.xpar (id, name) VALUES ('p1', 'Al');",
    ))
    .await
    .expect("seed cross-ns parent");
    dml.execute(run(
        "INSERT INTO xns_b.xchl_restr (id, pid) VALUES ('r1', 'p1');",
    ))
    .await
    .expect("seed restrict child");

    // RESTRICT: deleting the parent is rejected — a child in ANOTHER
    // namespace still references it (cross-namespace child discovery).
    let err = dml
        .execute(run("DELETE FROM xns_a.xpar WHERE id = 'p1';"))
        .await
        .expect_err("cross-ns RESTRICT must reject the parent delete");
    assert!(
        err.to_string().contains("ON DELETE NO ACTION"),
        "unexpected error: {err}"
    );

    // Drop the RESTRICT child, add a CASCADE child, then the parent delete
    // cascades across namespaces.
    dml.execute(run("DELETE FROM xns_b.xchl_restr WHERE id = 'r1';"))
        .await
        .expect("remove restrict child");
    dml.execute(run(
        "INSERT INTO xns_b.xchl_casc (id, pid) VALUES ('c1', 'p1');",
    ))
    .await
    .expect("seed cascade child");

    let get_casc = |key: &'static str| {
        let store = record_store.clone();
        let manager = manager.clone();
        async move {
            let (catalog, id) = manager
                .resolve_table("xns_b.xchl_casc")
                .await
                .expect("resolve");
            let schema = catalog.get_table(&id).await.expect("schema");
            store
                .get_by_key(
                    &schema,
                    TableRecordGetRequest {
                        table_id: id.name.clone(),
                        key: key.to_string(),
                        include_vector: false,
                        include_props: true,
                    },
                    None,
                    #[cfg(feature = "abac-policy")]
                    &ReadContext::system(SystemReadReason::Statistics, "dml::tests"),
                )
                .await
                .expect("get_by_key")
        }
    };
    assert!(get_casc("c1").await.is_some());
    dml.execute(run("DELETE FROM xns_a.xpar WHERE id = 'p1';"))
        .await
        .expect("cross-ns CASCADE delete must succeed");
    assert!(
        get_casc("c1").await.is_none(),
        "cross-ns CASCADE must remove the child in the other namespace"
    );
}

/// TD-110 S1: N-level CASCADE recursion, cyclic-FK rejection (no partial
/// deletion), and the bounded-depth guard.
#[tokio::test]
async fn delete_cascade_recurses_and_rejects_cycles_and_depth() {
    use crate::query::table_write_executor::PlannedOnlyTableWriteExecutor;
    use crate::services::record_store::{DirectWalTableRecordStore, TableRecordGetRequest};
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    let ddl = DdlService::new(manager.clone());
    ddl.execute(DdlStatement::CreateNamespace {
        namespace: vec!["default".to_string()],
        if_not_exists: true,
        properties: HashMap::new(),
    })
    .await
    .expect("create namespace");
    let parser = crate::query::sql_frontend::SqlFrontendParser::new();

    // 3-level CASCADE chain: cascade_gp -> cascade_p -> cascade_c.
    for create in [
        "CREATE TABLE cascade_gp (id TEXT NOT NULL, PRIMARY KEY (id));",
        "CREATE TABLE cascade_p (id TEXT NOT NULL, gp_id TEXT, PRIMARY KEY (id), FOREIGN KEY (gp_id) REFERENCES cascade_gp (id) ON DELETE CASCADE);",
        "CREATE TABLE cascade_c (id TEXT NOT NULL, p_id TEXT, PRIMARY KEY (id), FOREIGN KEY (p_id) REFERENCES cascade_p (id) ON DELETE CASCADE);",
        // Cyclic CASCADE: cyc_a <-> cyc_b.
        "CREATE TABLE cyc_a (id TEXT NOT NULL, b_id TEXT, PRIMARY KEY (id), FOREIGN KEY (b_id) REFERENCES cyc_b (id) ON DELETE CASCADE);",
        "CREATE TABLE cyc_b (id TEXT NOT NULL, a_id TEXT, PRIMARY KEY (id), FOREIGN KEY (a_id) REFERENCES cyc_a (id) ON DELETE CASCADE);",
    ] {
        let stmt = parser
            .parse_ddl(create)
            .expect("parse create")
            .expect("ddl");
        ddl.execute(stmt).await.expect("create table");
    }
    // Depth-guard chain: depth_t0 .. depth_t16 (17 tables, linear CASCADE).
    let depth_root = "CREATE TABLE depth_t0 (id TEXT NOT NULL, PRIMARY KEY (id));";
    let stmt = parser
        .parse_ddl(depth_root)
        .expect("parse depth root")
        .expect("ddl");
    ddl.execute(stmt).await.expect("create depth_t0");
    for i in 1u32..=16 {
        let create = format!(
            "CREATE TABLE depth_t{i} (id TEXT NOT NULL, parent_id TEXT, PRIMARY KEY (id), FOREIGN KEY (parent_id) REFERENCES depth_t{} (id) ON DELETE CASCADE);",
            i - 1
        );
        let stmt = parser
            .parse_ddl(&create)
            .expect("parse depth create")
            .expect("ddl");
        ddl.execute(stmt).await.expect("create depth table");
    }

    let wal_appender = Arc::new(
        FramedTableWalAppender::open(temp_dir.path().join("td110s1.wal"))
            .await
            .expect("open WAL"),
    );
    let record_store = Arc::new(DirectWalTableRecordStore::new(
        Arc::new(MemtableRecordStorage::new()),
        wal_appender,
    ));
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        record_store.clone(),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );
    let run = |sql: &'static str| parser.parse_dml(sql).expect("parse dml").expect("dml");
    let get = |table: String, key: String| {
        let store = record_store.clone();
        let manager = manager.clone();
        async move {
            let (catalog, id) = manager.resolve_table(&table).await.expect("resolve");
            let schema = catalog.get_table(&id).await.expect("schema");
            store
                .get_by_key(
                    &schema,
                    TableRecordGetRequest {
                        table_id: id.name.clone(),
                        key,
                        include_vector: false,
                        include_props: true,
                    },
                    None,
                    #[cfg(feature = "abac-policy")]
                    &ReadContext::system(SystemReadReason::Statistics, "dml::tests"),
                )
                .await
                .expect("get_by_key")
        }
    };

    // ── Section 1: 3-level CASCADE recursion (the shipped bug orphaned the
    //    grandchild). ──
    for sql in [
        "INSERT INTO cascade_gp (id) VALUES ('g1');",
        "INSERT INTO cascade_p (id, gp_id) VALUES ('p1', 'g1');",
        "INSERT INTO cascade_c (id, p_id) VALUES ('c1', 'p1');",
    ] {
        dml.execute(run(sql)).await.expect("seed chain insert");
    }
    assert!(get("cascade_p".into(), "p1".into()).await.is_some());
    assert!(get("cascade_c".into(), "c1".into()).await.is_some());
    dml.execute(run("DELETE FROM cascade_gp WHERE id = 'g1';"))
        .await
        .expect("3-level CASCADE delete must succeed");
    assert!(
        get("cascade_p".into(), "p1".into()).await.is_none(),
        "CASCADE must recurse: level-1 child removed"
    );
    assert!(
        get("cascade_c".into(), "c1".into()).await.is_none(),
        "CASCADE must recurse to depth 2: grandchild removed (orphaned pre-S1)"
    );

    // ── Section 2: cyclic CASCADE FK rejected with NO partial deletion. ──
    // Seed with NULL FK values (exempt from the insert-time FK check) — a
    // cyclic CASCADE is a *schema* cycle, so the pre-pass detects it from
    // the FK graph alone; no data cycle is required to trigger it.
    for sql in [
        "INSERT INTO cyc_a (id) VALUES ('a1');",
        "INSERT INTO cyc_b (id) VALUES ('b1');",
    ] {
        dml.execute(run(sql)).await.expect("seed cycle insert");
    }
    let err = dml
        .execute(run("DELETE FROM cyc_a WHERE id = 'a1';"))
        .await
        .expect_err("cyclic CASCADE must be rejected");
    assert!(
        err.to_string().contains("cascade cycle"),
        "expected a cycle error, got: {err}"
    );
    // The cycle is detected before any mutation, so neither row is touched.
    assert!(
        get("cyc_a".into(), "a1".into()).await.is_some(),
        "cyclic-CASCADE rejection must not partially delete cyc_a"
    );
    assert!(
        get("cyc_b".into(), "b1".into()).await.is_some(),
        "cyclic-CASCADE rejection must not partially delete cyc_b"
    );

    // ── Section 3: bounded-depth guard trips on a 17-deep chain. ──
    dml.execute(run("INSERT INTO depth_t0 (id) VALUES ('d0');"))
        .await
        .expect("seed depth_t0");
    for i in 1u32..=16 {
        let sql = format!(
            "INSERT INTO depth_t{i} (id, parent_id) VALUES ('d{i}', 'd{}');",
            i - 1
        );
        let stmt = parser
            .parse_dml(&sql)
            .expect("parse depth insert")
            .expect("dml");
        dml.execute(stmt).await.expect("seed depth insert");
    }
    let err = dml
        .execute(run("DELETE FROM depth_t0 WHERE id = 'd0';"))
        .await
        .expect_err("an over-deep CASCADE chain must trip the depth guard");
    assert!(
        err.to_string().contains("maximum depth"),
        "expected a depth-guard error, got: {err}"
    );
}

#[tokio::test]
async fn dml_enforces_foreign_key_rejecting_missing_parent() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    DdlService::new(manager.clone())
        .execute(DdlStatement::CreateNamespace {
            namespace: vec!["default".to_string()],
            if_not_exists: true,
            properties: HashMap::new(),
        })
        .await
        .expect("namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    for ddl in [
        "CREATE TABLE customers_for_fk (id TEXT NOT NULL, PRIMARY KEY (id));",
        "CREATE TABLE orders_with_fk (id TEXT NOT NULL, customer_id TEXT, PRIMARY KEY (id), FOREIGN KEY (customer_id) REFERENCES customers_for_fk(id));",
    ] {
        let stmt = parser.parse_ddl(ddl).expect("parse ddl").expect("ddl stmt");
        DdlService::new(manager.clone())
            .execute(stmt)
            .await
            .expect("create table");
    }

    let dml = DmlService::with_record_store_and_table_write_executor(
        manager,
        Arc::new(ExplainOnlyRecordStore),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );

    // TD-110: FK references are now enforced (DmlService::enforce_foreign_keys).
    // No parent row exists (the explain-only store has no records), so the
    // child insert is rejected as a reference violation rather than the old
    // "not enforced yet" fail-close.
    let fk_insert = parser
        .parse_dml("INSERT INTO orders_with_fk (id, customer_id) VALUES ('o1', 'c1');")
        .expect("parse fk insert")
        .expect("fk insert");
    let fk_err = dml
        .execute(fk_insert)
        .await
        .expect_err("FK to a non-existent parent must be rejected");
    assert!(
        fk_err.to_string().contains("violates reference"),
        "unexpected error: {fk_err}"
    );
}

/// T15: `validate_record_batch_against_schema` passes for conforming records and returns `Err`
/// when a NOT NULL column receives `ProximaValue::Null` in a fast-lane (non-SQL) batch write.
#[tokio::test]
async fn fast_lane_schema_validation_rejects_null_for_not_null_column() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("fast-lane-schema.wal");

    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    DdlService::new(manager.clone())
        .execute(DdlStatement::CreateNamespace {
            namespace: vec!["default".to_string()],
            if_not_exists: true,
            properties: HashMap::new(),
        })
        .await
        .expect("namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
            .parse_ddl(
                "CREATE TABLE fast_lane_tbl (id TEXT NOT NULL, label TEXT, score FLOAT NOT NULL, PRIMARY KEY (id));",
            )
            .expect("parse ddl")
            .expect("ddl");
    DdlService::new(manager.clone())
        .execute(ddl_stmt)
        .await
        .expect("create table");

    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        Arc::new(DirectWalTableRecordStore::new(
            Arc::new(MemtableRecordStorage::new()),
            Arc::new(
                FramedTableWalAppender::open(&wal_path)
                    .await
                    .expect("open WAL"),
            ),
        )),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );

    // Conforming record: id=text, score=float, label=null (nullable).
    let ok_record = ProximaRecord {
        oid: "v1".to_string(),
        props: proximadb_records::ProximaTree::from([
            (
                "id".to_string(),
                ProximaTreeNode::Value(ProximaValue::String("v1".to_string())),
            ),
            (
                "score".to_string(),
                ProximaTreeNode::Value(ProximaValue::Float32(3.5)),
            ),
            (
                "label".to_string(),
                ProximaTreeNode::Value(ProximaValue::Null),
            ),
        ]),
        ..Default::default()
    };
    dml.validate_record_batch_against_schema("fast_lane_tbl", &[ok_record])
        .await
        .expect("conforming record must pass schema validation");

    // Violating record: score is NOT NULL but receives Null.
    let bad_record = ProximaRecord {
        oid: "v2".to_string(),
        props: proximadb_records::ProximaTree::from([
            (
                "id".to_string(),
                ProximaTreeNode::Value(ProximaValue::String("v2".to_string())),
            ),
            (
                "score".to_string(),
                ProximaTreeNode::Value(ProximaValue::Null),
            ),
        ]),
        ..Default::default()
    };
    let err = dml
        .validate_record_batch_against_schema("fast_lane_tbl", &[bad_record])
        .await;
    assert!(
        err.is_err(),
        "NOT NULL column with Null must fail fast-lane validation"
    );
    let err_val = err.unwrap_err();
    let chain: String = err_val
        .chain()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(": ");
    assert!(
        chain.contains("not nullable") || chain.contains("cannot be NULL"),
        "error chain should name the constraint: {chain}"
    );
}

/// T15: `validate_record_batch_against_schema` silently passes when the collection is not
/// registered as a relational table (non-relational / vector-only collections stay open).
#[tokio::test]
async fn fast_lane_schema_validation_skips_non_relational_collections() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("fast-lane-skip.wal");

    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");

    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        Arc::new(DirectWalTableRecordStore::new(
            Arc::new(MemtableRecordStorage::new()),
            Arc::new(
                FramedTableWalAppender::open(&wal_path)
                    .await
                    .expect("open WAL"),
            ),
        )),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );

    // "unknown_collection" is not in xCatalog — validation must silently pass.
    let any_record = ProximaRecord {
        oid: "x1".to_string(),
        props: proximadb_records::ProximaTree::from([(
            "score".to_string(),
            ProximaTreeNode::Value(ProximaValue::Null),
        )]),
        ..Default::default()
    };
    dml.validate_record_batch_against_schema("unknown_collection", &[any_record])
        .await
        .expect("non-relational collection must skip schema validation");
}

/// T11: `explain_analyze_table_write` executes the write and returns the route explanation
/// enriched with `execution_elapsed_us` and `execution_rows_written`.
#[tokio::test]
async fn explain_analyze_executes_write_and_returns_execution_stats() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("explain-analyze.wal");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    DdlService::new(manager.clone())
        .execute(DdlStatement::CreateNamespace {
            namespace: vec!["default".to_string()],
            if_not_exists: true,
            properties: HashMap::new(),
        })
        .await
        .expect("namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
        .parse_ddl(
            "CREATE TABLE analyze_tbl (id TEXT NOT NULL, val INTEGER NOT NULL, PRIMARY KEY (id));",
        )
        .expect("parse ddl")
        .expect("ddl");
    DdlService::new(manager.clone())
        .execute(ddl_stmt)
        .await
        .expect("create table");

    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        Arc::new(DirectWalTableRecordStore::new(
            Arc::new(MemtableRecordStorage::new()),
            Arc::new(
                FramedTableWalAppender::open(&wal_path)
                    .await
                    .expect("open WAL"),
            ),
        )),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );

    let stmt = parser
        .parse_dml("INSERT INTO analyze_tbl (id, val) VALUES ('a1', 42), ('a2', 99);")
        .expect("parse")
        .expect("dml");
    let explanation = dml
        .explain_analyze_table_write(stmt)
        .await
        .expect("explain analyze must succeed");

    assert_eq!(explanation.target_table, "analyze_tbl");
    assert!(
        explanation.execution_elapsed_us.is_some(),
        "elapsed_us must be populated by EXPLAIN ANALYZE"
    );
    assert_eq!(
        explanation.execution_rows_written,
        Some(2),
        "rows_written must reflect the 2 inserted rows"
    );
}

/// T8: After INSERT, `column_stats[col].min_value` and `max_value` are updated to the
/// lexicographic min/max of the inserted values for String and integer columns.
#[tokio::test]
async fn insert_updates_column_min_max_statistics() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("minmax-stats.wal");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    DdlService::new(manager.clone())
        .execute(DdlStatement::CreateNamespace {
            namespace: vec!["default".to_string()],
            if_not_exists: true,
            properties: HashMap::new(),
        })
        .await
        .expect("namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
            .parse_ddl(
                "CREATE TABLE minmax_tbl (id TEXT NOT NULL, name TEXT NOT NULL, score INTEGER NOT NULL, PRIMARY KEY (id));",
            )
            .expect("parse ddl")
            .expect("ddl");
    DdlService::new(manager.clone())
        .execute(ddl_stmt)
        .await
        .expect("create table");

    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        Arc::new(DirectWalTableRecordStore::new(
            Arc::new(MemtableRecordStorage::new()),
            Arc::new(
                FramedTableWalAppender::open(&wal_path)
                    .await
                    .expect("open WAL"),
            ),
        )),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );

    for sql in [
        "INSERT INTO minmax_tbl (id, name, score) VALUES ('r1', 'charlie', 30);",
        "INSERT INTO minmax_tbl (id, name, score) VALUES ('r2', 'alice', 10);",
        "INSERT INTO minmax_tbl (id, name, score) VALUES ('r3', 'bob', 20);",
    ] {
        let stmt = parser.parse_dml(sql).expect("parse").expect("dml");
        dml.execute(stmt).await.expect("insert");
    }

    let (catalog, table_id) = manager.resolve_table("minmax_tbl").await.expect("resolve");
    let stats = catalog.get_statistics(&table_id).await.unwrap_or_default();

    // name is a TEXT column: min = 'alice', max = 'charlie'
    let name_stats = stats.column_stats.get("name").expect("name col stats");
    assert_eq!(
        name_stats.min_value.as_deref(),
        Some("alice"),
        "name min should be 'alice'"
    );
    assert_eq!(
        name_stats.max_value.as_deref(),
        Some("charlie"),
        "name max should be 'charlie'"
    );

    // score is an INTEGER column: min = +000000000000000000010, max = +000000000000000000030
    let score_stats = stats.column_stats.get("score").expect("score col stats");
    assert!(
        score_stats.min_value.is_some(),
        "score min must be populated"
    );
    assert!(
        score_stats.max_value.is_some(),
        "score max must be populated"
    );
    // The sortable min/max string for integer 10 sorts before 30.
    assert!(
        score_stats.min_value < score_stats.max_value,
        "score min must sort before max: min={:?} max={:?}",
        score_stats.min_value,
        score_stats.max_value
    );
}

#[tokio::test]
async fn insert_updates_column_ndv_statistics() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("ndv-stats.wal");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    DdlService::new(manager.clone())
        .execute(DdlStatement::CreateNamespace {
            namespace: vec!["default".to_string()],
            if_not_exists: true,
            properties: HashMap::new(),
        })
        .await
        .expect("namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
        .parse_ddl(
            "CREATE TABLE ndv_tbl (id TEXT NOT NULL, category TEXT NOT NULL, PRIMARY KEY (id));",
        )
        .expect("parse ddl")
        .expect("ddl");
    DdlService::new(manager.clone())
        .execute(ddl_stmt)
        .await
        .expect("create table");

    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        Arc::new(DirectWalTableRecordStore::new(
            Arc::new(MemtableRecordStorage::new()),
            Arc::new(
                FramedTableWalAppender::open(&wal_path)
                    .await
                    .expect("open WAL"),
            ),
        )),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );

    // Insert 4 rows: 3 distinct categories ('a', 'b', 'c'), 4 distinct ids.
    for sql in [
        "INSERT INTO ndv_tbl (id, category) VALUES ('r1', 'a');",
        "INSERT INTO ndv_tbl (id, category) VALUES ('r2', 'b');",
        "INSERT INTO ndv_tbl (id, category) VALUES ('r3', 'c');",
        "INSERT INTO ndv_tbl (id, category) VALUES ('r4', 'a');",
    ] {
        let stmt = parser.parse_dml(sql).expect("parse").expect("dml");
        dml.execute(stmt).await.expect("insert");
    }

    let (catalog, table_id) = manager.resolve_table("ndv_tbl").await.expect("resolve");
    let stats = catalog.get_statistics(&table_id).await.unwrap_or_default();

    // Verify row count (sanity check for the cap logic)
    assert_eq!(stats.row_count, 4, "row count must be 4");

    // id has 4 distinct values across the 4 single-row batches → additive estimate = 4
    let id_ndv = stats
        .column_stats
        .get("id")
        .and_then(|s| s.distinct_count)
        .expect("id distinct_count must be populated");
    assert!(id_ndv >= 1, "id NDV must be at least 1, got {id_ndv}");
    assert!(
        id_ndv <= stats.row_count,
        "id NDV ({id_ndv}) must not exceed row count ({})",
        stats.row_count
    );

    // category has values within each single-row batch → additive estimate = 4 (one per batch),
    // capped at row_count = 4.
    let cat_ndv = stats
        .column_stats
        .get("category")
        .and_then(|s| s.distinct_count)
        .expect("category distinct_count must be populated");
    assert!(
        cat_ndv >= 1,
        "category NDV must be at least 1, got {cat_ndv}"
    );
    assert!(
        cat_ndv <= stats.row_count,
        "category NDV ({cat_ndv}) must not exceed row count ({})",
        stats.row_count
    );
}

/// T18: cross-surface conformance — DML SQL INSERT and fast-lane
/// `validate_record_batch_against_schema` must make the same accept/reject decision
/// for the same logical row, so REST/gRPC/Arrow Flight callers see the same constraint
/// behavior as SQL clients.
#[tokio::test]
async fn dml_and_fast_lane_agree_on_not_null_constraint() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("conformance.wal");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    DdlService::new(manager.clone())
        .execute(DdlStatement::CreateNamespace {
            namespace: vec!["default".to_string()],
            if_not_exists: true,
            properties: HashMap::new(),
        })
        .await
        .expect("namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
        .parse_ddl(
            "CREATE TABLE conf_tbl (id TEXT NOT NULL, label TEXT NOT NULL, PRIMARY KEY (id));",
        )
        .expect("parse ddl")
        .expect("ddl");
    DdlService::new(manager.clone())
        .execute(ddl_stmt)
        .await
        .expect("create table");

    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        Arc::new(DirectWalTableRecordStore::new(
            Arc::new(MemtableRecordStorage::new()),
            Arc::new(
                FramedTableWalAppender::open(&wal_path)
                    .await
                    .expect("open WAL"),
            ),
        )),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );

    // -------- conforming row: both surfaces must accept --------
    let dml_ok = parser
        .parse_dml("INSERT INTO conf_tbl (id, label) VALUES ('k1', 'present');")
        .expect("parse")
        .expect("dml");
    dml.execute(dml_ok)
        .await
        .expect("DML must accept conforming row");

    let mut ok_props = std::collections::HashMap::new();
    ok_props.insert(
        "id".to_string(),
        ProximaTreeNode::Value(ProximaValue::String("k2".to_string())),
    );
    ok_props.insert(
        "label".to_string(),
        ProximaTreeNode::Value(ProximaValue::String("present".to_string())),
    );
    let ok_record = ProximaRecord {
        oid: "k2".to_string(),
        props: ok_props,
        ..Default::default()
    };
    dml.validate_record_batch_against_schema("conf_tbl", &[ok_record])
        .await
        .expect("fast-lane must accept conforming row");

    // -------- violating row: both surfaces must reject --------
    let dml_bad = parser
        .parse_dml("INSERT INTO conf_tbl (id, label) VALUES ('k3', NULL);")
        .expect("parse")
        .expect("dml");
    let dml_err = dml
        .execute(dml_bad)
        .await
        .expect_err("DML must reject NULL for NOT NULL column");

    let mut bad_props = std::collections::HashMap::new();
    bad_props.insert(
        "id".to_string(),
        ProximaTreeNode::Value(ProximaValue::String("k4".to_string())),
    );
    bad_props.insert(
        "label".to_string(),
        ProximaTreeNode::Value(ProximaValue::Null),
    );
    let bad_record = ProximaRecord {
        oid: "k4".to_string(),
        props: bad_props,
        ..Default::default()
    };
    let fast_lane_err = dml
        .validate_record_batch_against_schema("conf_tbl", &[bad_record])
        .await
        .expect_err("fast-lane must reject NULL for NOT NULL column");

    // Both error chains must reference the constraint that was violated.
    let dml_chain: String = dml_err
        .chain()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(": ");
    let fast_lane_chain: String = fast_lane_err
        .chain()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(": ");
    let mentions_constraint = |s: &str| {
        s.contains("not nullable") || s.contains("cannot be NULL") || s.contains("NOT NULL")
    };
    assert!(
        mentions_constraint(&dml_chain),
        "DML error chain should explain NOT NULL violation: {dml_chain}"
    );
    assert!(
        mentions_constraint(&fast_lane_chain),
        "fast-lane error chain should explain NOT NULL violation: {fast_lane_chain}"
    );
}

/// T18: cross-surface conformance — DML SQL INSERT and fast-lane validation must agree on
/// type mismatches. A string value in an integer column must be rejected by both surfaces
/// regardless of the exact error wording.
#[tokio::test]
async fn dml_and_fast_lane_agree_on_type_mismatch() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("type-conformance.wal");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    DdlService::new(manager.clone())
        .execute(DdlStatement::CreateNamespace {
            namespace: vec!["default".to_string()],
            if_not_exists: true,
            properties: HashMap::new(),
        })
        .await
        .expect("namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
        .parse_ddl(
            "CREATE TABLE type_tbl (id TEXT NOT NULL, score INTEGER NOT NULL, PRIMARY KEY (id));",
        )
        .expect("parse ddl")
        .expect("ddl");
    DdlService::new(manager.clone())
        .execute(ddl_stmt)
        .await
        .expect("create table");

    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        Arc::new(DirectWalTableRecordStore::new(
            Arc::new(MemtableRecordStorage::new()),
            Arc::new(
                FramedTableWalAppender::open(&wal_path)
                    .await
                    .expect("open WAL"),
            ),
        )),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );

    // DML SQL path: 'not-an-int' in an INTEGER column must fail at literal coercion.
    let dml_bad = parser
        .parse_dml("INSERT INTO type_tbl (id, score) VALUES ('k1', 'not-an-int');")
        .expect("parse")
        .expect("dml");
    let dml_err = dml
        .execute(dml_bad)
        .await
        .expect_err("DML must reject string literal for INTEGER column");

    // Fast-lane path: ProximaValue::String in an Int32 column must fail validation.
    let mut bad_props = std::collections::HashMap::new();
    bad_props.insert(
        "id".to_string(),
        ProximaTreeNode::Value(ProximaValue::String("k2".to_string())),
    );
    bad_props.insert(
        "score".to_string(),
        ProximaTreeNode::Value(ProximaValue::String("not-an-int".to_string())),
    );
    let bad_record = ProximaRecord {
        oid: "k2".to_string(),
        props: bad_props,
        ..Default::default()
    };
    let fast_lane_err = dml
        .validate_record_batch_against_schema("type_tbl", &[bad_record])
        .await
        .expect_err("fast-lane must reject ProximaValue::String for INTEGER column");

    // Both error chains must mention the integer column or expected type so callers can
    // diagnose the violation. Exact wording differs between paths; both must be informative.
    let dml_chain: String = dml_err
        .chain()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(": ");
    let fast_lane_chain: String = fast_lane_err
        .chain()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(": ");
    let mentions_type = |s: &str| {
        let lower = s.to_lowercase();
        lower.contains("integer") || lower.contains("int32") || lower.contains("int64")
    };
    assert!(
        mentions_type(&dml_chain),
        "DML error chain should explain integer-type violation: {dml_chain}"
    );
    assert!(
        mentions_type(&fast_lane_chain),
        "fast-lane error chain should explain integer-type violation: {fast_lane_chain}"
    );
}

#[tokio::test]
async fn insert_marks_statistics_with_last_analyzed_timestamp() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let wal_path = temp_dir.path().join("stale-stats.wal");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    DdlService::new(manager.clone())
        .execute(DdlStatement::CreateNamespace {
            namespace: vec!["default".to_string()],
            if_not_exists: true,
            properties: HashMap::new(),
        })
        .await
        .expect("namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl_stmt = parser
        .parse_ddl(
            "CREATE TABLE stale_tbl (id TEXT NOT NULL, val INTEGER NOT NULL, PRIMARY KEY (id));",
        )
        .expect("parse ddl")
        .expect("ddl");
    DdlService::new(manager.clone())
        .execute(ddl_stmt)
        .await
        .expect("create table");

    let dml = DmlService::with_record_store_and_table_write_executor(
        manager.clone(),
        Arc::new(DirectWalTableRecordStore::new(
            Arc::new(MemtableRecordStorage::new()),
            Arc::new(
                FramedTableWalAppender::open(&wal_path)
                    .await
                    .expect("open WAL"),
            ),
        )),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );

    let pre_insert_ms = DmlService::now_unix_ms();
    let stmt = parser
        .parse_dml("INSERT INTO stale_tbl (id, val) VALUES ('r1', 42);")
        .expect("parse")
        .expect("dml");
    dml.execute(stmt).await.expect("insert");
    let post_insert_ms = DmlService::now_unix_ms();

    let (catalog, table_id) = manager.resolve_table("stale_tbl").await.expect("resolve");
    let stats = catalog.get_statistics(&table_id).await.unwrap_or_default();

    // last_analyzed_ms must be set and within the wall-clock window of the INSERT.
    let last_ms = stats
        .last_analyzed_ms
        .expect("last_analyzed_ms must be populated after INSERT");
    assert!(
        last_ms >= pre_insert_ms && last_ms <= post_insert_ms,
        "last_analyzed_ms ({last_ms}) must be within [{pre_insert_ms}, {post_insert_ms}]"
    );

    // Stats are fresh inside a generous TTL window and stale outside it.
    assert!(
        !stats.is_stale(last_ms, 60_000),
        "stats updated at now must not be stale within a 60s TTL"
    );
    assert!(
        stats.is_stale(last_ms + 120_000, 60_000),
        "stats 120s old must be stale under a 60s TTL"
    );
}

#[test]
fn sql_like_fast_paths_match_dp() {
    // Each case is (value, pattern). The fast path and DP must agree.
    let cases: &[(&str, &str)] = &[
        // exact-match fast path
        ("hello", "hello"),
        ("hello", "world"),
        ("", ""),
        // suffix fast path: leading %
        ("hello world", "%world"),
        ("hello world", "%earth"),
        ("", "%"),
        // prefix fast path: trailing %
        ("hello world", "hello%"),
        ("hello world", "world%"),
        // contains fast path: %x%
        ("hello world", "%lo wo%"),
        ("hello world", "%missing%"),
        // single % in middle: DP path
        ("hello world", "he%ld"),
        ("hello world", "ab%cd"),
        // underscore: DP path
        ("abc", "a_c"),
        ("abc", "a_d"),
        ("abc", "a__"),
        // mixed wildcards: DP path
        ("hello world", "h_llo%world"),
        // empty pattern, non-empty value
        ("abc", ""),
        // pattern with multiple percents only (matches everything)
        ("abc", "%%"),
        ("", "%%"),
        // multi-byte (non-ASCII) — must take DP path
        ("héllo", "héllo"),
        ("héllo", "h%o"),
        ("héllo", "h_llo"),
    ];

    for (value, pattern) in cases {
        let fast = DmlService::sql_like_matches(value, pattern);
        let dp = DmlService::sql_like_matches_dp(value, pattern);
        assert_eq!(
            fast, dp,
            "LIKE mismatch for value={value:?} pattern={pattern:?}: fast={fast} dp={dp}"
        );
    }
}

#[tokio::test]
async fn exact_table_vector_search_is_bounded_projection_aware_and_uses_full_primary_key() {
    use crate::services::record_store::DirectWalTableRecordStore;
    use crate::services::{FramedTableWalAppender, MemtableRecordStorage};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
        .await
        .expect("native catalog");
    DdlService::new(manager.clone())
        .execute(DdlStatement::CreateNamespace {
            namespace: vec!["default".to_string()],
            if_not_exists: true,
            properties: HashMap::new(),
        })
        .await
        .expect("namespace");

    let parser = crate::query::sql_frontend::SqlFrontendParser::new();
    let ddl = parser
        .parse_ddl(
            "CREATE TABLE vector_rows (scope TEXT, id TEXT, payload JSONB, \
             embedding VECTOR(2), image_embedding VECTOR(3), PRIMARY KEY (scope, id));",
        )
        .expect("parse ddl")
        .expect("ddl");
    DdlService::new(manager.clone())
        .execute(ddl)
        .await
        .expect("create table");

    let wal_path = temp_dir.path().join("vector-search.wal");
    let dml = DmlService::with_record_store_and_table_write_executor(
        manager,
        Arc::new(DirectWalTableRecordStore::new(
            Arc::new(MemtableRecordStorage::new()),
            Arc::new(
                FramedTableWalAppender::open(&wal_path)
                    .await
                    .expect("open WAL"),
            ),
        )),
        Arc::new(PlannedOnlyTableWriteExecutor::new()),
    );
    for sql in [
        r#"INSERT INTO vector_rows (scope, id, payload, embedding, image_embedding) VALUES ('a', '1', '{"rank":1}'::jsonb, '[1,0]'::vector, '[0,0,1]'::vector)"#,
        r#"INSERT INTO vector_rows (scope, id, payload, embedding, image_embedding) VALUES ('a', '2', '{"rank":2}'::jsonb, '[0.9,0.1]'::vector, '[0,1,0]'::vector)"#,
        r#"INSERT INTO vector_rows (scope, id, payload, embedding, image_embedding) VALUES ('b', '1', '{"rank":3}'::jsonb, '[0,1]'::vector, '[1,0,0]'::vector)"#,
    ] {
        let statement = parser
            .parse_dml(sql)
            .expect("parse insert")
            .expect("insert");
        dml.execute(statement).await.expect("insert row");
    }

    let requested = vec!["payload".to_string()];
    let results = dml
        .search_table_vectors_exact(
            "vector_rows",
            "embedding",
            &[1.0, 0.0],
            2,
            None,
            proximadb_distance_types::DistanceMetric::L2,
            Some(&requested),
            None,
            proximadb_runtime::PortIdentity::default(),
            &ExecutionControls::default(),
        )
        .await
        .expect("search")
        .expect("catalog table");

    assert_eq!(results.len(), 2, "the exact route must retain only top_k");
    assert_eq!(results[0].id, "a\u{1f}1");
    assert_eq!(results[1].id, "a\u{1f}2");
    assert!(results.iter().all(|record| record.vector.is_none()));
    assert!(
        results
            .iter()
            .all(|record| record.metadata.keys().map(String::as_str).eq(["payload"])),
        "only requested metadata columns should be materialized"
    );

    let image_results = dml
        .search_table_vectors_exact(
            "vector_rows",
            "image_embedding",
            &[1.0, 0.0, 0.0],
            1,
            None,
            proximadb_distance_types::DistanceMetric::L2,
            Some(&requested),
            None,
            proximadb_runtime::PortIdentity::default(),
            &ExecutionControls::default(),
        )
        .await
        .expect("second vector-column search")
        .expect("catalog table");
    assert_eq!(
        image_results[0].id, "b\u{1f}1",
        "the named vector column, not the table's first vector, defines ordering"
    );

    let missing = vec!["does_not_exist".to_string()];
    let error = dml
        .search_table_vectors_exact(
            "vector_rows",
            "embedding",
            &[1.0, 0.0],
            1,
            None,
            proximadb_distance_types::DistanceMetric::L2,
            Some(&missing),
            None,
            proximadb_runtime::PortIdentity::default(),
            &ExecutionControls::default(),
        )
        .await
        .expect_err("unknown projection must fail instead of becoming NULL");
    assert!(error.to_string().contains("does not exist"), "{error}");

    let cancellation_flag = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let error = dml
        .search_table_vectors_exact(
            "vector_rows",
            "embedding",
            &[1.0, 0.0],
            1,
            None,
            proximadb_distance_types::DistanceMetric::L2,
            Some(&requested),
            None,
            proximadb_runtime::PortIdentity::default(),
            &ExecutionControls {
                cancellation_flag: Some(cancellation_flag),
                ..Default::default()
            },
        )
        .await
        .expect_err("cancelled scoring must stop cooperatively");
    assert!(error.to_string().contains("cancelled"), "{error}");
}

#[cfg(all(test, feature = "abac-policy"))]
mod abac_relational_enforcement_tests {
    //! FA-c Phase 2 e2e: the ABAC enforcement is wired into `DmlService`. Proves
    //! a real `DmlService` resolves a client subject's authorization via the
    //! injected `AbacEnforcer` — a permitted subject ⇒ `Restricted`; an unbound
    //! subject ⇒ `Denied` (fail-closed); an internal read (no subject) ⇒ no
    //! enforcement; a legacy table lacking stable ids ⇒ `Denied` (fail-closed,
    //! never unfiltered). The substrate test proves the compiled predicate then
    //! filters rows; this module proves the DmlService wiring resolves to the
    //! right `AbacScanResult`.
    use super::*;
    use crate::core::search::{ComparisonOperator, FilterExpression};
    use crate::security::rls::{AbacEnforcer, AbacScanResult};
    use proximadb_abac::{
        AttributeAuthority, AttributeBinding, InMemoryAttributeAuthority, InMemoryPolicyEpochs,
        InMemoryPredicateObjectStore,
    };
    use proximadb_catalog::fc_metamodel::{AttrValue, Effect, PolicyBinding, Scope};
    use serde_json::json;
    use std::sync::Arc;

    const TENANT: u64 = 7;
    const NS: u16 = 3;
    const TABLE: u32 = 200;

    /// alice (dept=eng) + a `dept == eng` predicate (obj 42) + a table-scoped
    /// Permit binding referencing it. Self-contained policy held by the enforcer.
    fn enforcer() -> AbacEnforcer {
        let mut authority = InMemoryAttributeAuthority::new();
        authority.upsert(
            AttributeBinding::new("alice", TENANT).with_attr("dept", AttrValue::Str("eng".into())),
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
        AbacEnforcer::new(
            Arc::new(authority),
            Arc::new(store),
            Arc::new(InMemoryPolicyEpochs::new()),
        )
        .with_bindings(bindings)
    }

    fn schema_with_stable_ids() -> CatalogTableSchema {
        let mut s = update_test_schema();
        s.object_id = Some(TABLE as u64);
        s.stable_namespace_id = Some(NS);
        s
    }

    fn identity(
        subject: Option<&str>,
        stable_id: Option<u64>,
    ) -> proximadb_runtime::PortIdentity<'_> {
        proximadb_runtime::PortIdentity {
            tenant_id: Some("t7"),
            subject,
            tenant_stable_id: stable_id,
            auth_class: if subject.is_some() {
                proximadb_tenant::AuthClass::Authenticated
            } else {
                proximadb_tenant::AuthClass::Anonymous
            },
        }
    }

    fn dml_with_enforcer() -> DmlService {
        DmlService::with_record_store_and_table_write_executor(
            Arc::new(CatalogManager::new()),
            Arc::new(ExplainOnlyRecordStore),
            Arc::new(PlannedOnlyTableWriteExecutor::new()),
        )
        .with_abac_enforcer(Arc::new(enforcer()))
    }

    #[test]
    fn permitted_subject_resolves_restricted() {
        let dml = dml_with_enforcer();
        let res = dml.abac_row_filter(
            identity(Some("alice"), Some(TENANT)),
            &schema_with_stable_ids(),
        );
        assert!(
            matches!(res, Some(AbacScanResult::Restricted(_))),
            "alice (bound, permitted, dept=eng predicate) must resolve to Restricted"
        );
    }

    #[test]
    fn unbound_subject_is_denied_fail_closed() {
        let dml = dml_with_enforcer();
        let res = dml.abac_row_filter(
            identity(Some("mallory"), Some(TENANT)),
            &schema_with_stable_ids(),
        );
        assert!(
            matches!(res, Some(AbacScanResult::Denied(_))),
            "an unbound subject must be Denied (fail-closed), never Restricted"
        );
    }

    #[test]
    fn internal_read_has_no_enforcement() {
        let dml = dml_with_enforcer();
        // No subject ⇒ an internal/system read ⇒ no ABAC enforcement applies.
        let res = dml.abac_row_filter(identity(None, Some(TENANT)), &schema_with_stable_ids());
        assert!(
            res.is_none(),
            "an internal read (no subject) must not enforce"
        );
    }

    #[test]
    fn legacy_table_without_stable_ids_denies_fail_closed() {
        let dml = dml_with_enforcer();
        // `update_test_schema()` has no stable object/namespace ids ⇒ cannot build
        // a Target ⇒ fail-closed (Denied), never an unfiltered read.
        let res = dml.abac_row_filter(identity(Some("alice"), Some(TENANT)), &update_test_schema());
        assert!(
            matches!(res, Some(AbacScanResult::Denied(_))),
            "a legacy table lacking stable ids must deny (fail-closed), not leak"
        );
    }

    #[test]
    fn subject_without_stable_tenant_id_denies_fail_closed() {
        let dml = dml_with_enforcer();
        let res = dml.abac_row_filter(identity(Some("alice"), None), &schema_with_stable_ids());
        assert!(
            matches!(res, Some(AbacScanResult::Denied(_))),
            "a subject without the policy key must be denied"
        );
    }

    #[test]
    fn governed_identity_is_ineligible_for_subject_agnostic_fast_paths() {
        let dml = dml_with_enforcer();
        assert!(dml.relational_abac_required(identity(Some("alice"), Some(TENANT))));
        assert!(dml.relational_abac_required(identity(Some("alice"), None)));
        assert!(!dml.relational_abac_required(identity(None, Some(TENANT))));
    }

    #[tokio::test]
    async fn governed_identity_cannot_scan_a_different_tenant_partition() {
        let dml = dml_with_enforcer();
        let mut other = crate::storage::tenant::context::TenantContext::for_tenant_id("other");
        other.tenant_stable_id = Some(TENANT + 1);
        let error = dml
            .scan_table_relational(
                "any-table",
                None,
                None,
                None,
                Some(&other),
                identity(Some("alice"), Some(TENANT)),
            )
            .await
            .expect_err("identity/partition mismatch must fail before catalog or storage access");
        assert!(
            error
                .to_string()
                .contains("identity/storage tenant mismatch")
        );
    }
}

/// TD-USUB-4: publication must be **additive**. `MATERIALIZE` previously wrote
/// `set_storage_layouts(&id, vec![layout])`, discarding every other representation a
/// table had — which is what made cost-based routing across representations
/// inexpressible, since the catalog had no way to record an alternative.
#[test]
fn upsert_storage_layout_is_additive_and_idempotent() {
    fn layout(
        name: &str,
        format: proximadb_catalog::CatalogPhysicalFormat,
    ) -> CatalogStorageLayout {
        CatalogStorageLayout {
            name: name.to_string(),
            physical_format: format,
            location: Some(format!("file:///{name}")),
            ..Default::default()
        }
    }

    // A genuinely new layout is APPENDED, not substituted for what is there. This is
    // the whole point: a table can now hold a native landing layout AND a published
    // Parquet projection at the same time.
    let native = layout(
        "proxima-native",
        proximadb_catalog::CatalogPhysicalFormat::ProximaBlock,
    );
    let parquet = layout(
        "parquet-snapshot",
        proximadb_catalog::CatalogPhysicalFormat::Parquet,
    );
    let merged = upsert_storage_layout(vec![native.clone()], parquet.clone());
    assert_eq!(merged.len(), 2, "a new layout must be added, not replace");
    assert_eq!(merged[0].name, "proxima-native", "existing layout survives");
    assert_eq!(merged[1].name, "parquet-snapshot");

    // Re-publishing the SAME layout name replaces in place — no duplicate rows from a
    // repeated MATERIALIZE, and the updated location/properties win.
    let mut republished = parquet.clone();
    republished.location = Some("file:///new-location".to_string());
    republished
        .properties
        .insert("snapshot_lsn".to_string(), "42".to_string());
    let merged = upsert_storage_layout(merged, republished);
    assert_eq!(merged.len(), 2, "re-publishing must not duplicate");
    assert_eq!(
        merged[1].location.as_deref(),
        Some("file:///new-location"),
        "the re-published layout's fields must win"
    );
    assert_eq!(
        merged[1].properties.get("snapshot_lsn").map(String::as_str),
        Some("42"),
        "per-layout freshness must be carried by the upsert"
    );

    // Order is stable across republication: the replaced layout keeps its position, so
    // a reader treating the first match as the default sees no churn.
    assert_eq!(merged[0].name, "proxima-native");

    // Degenerate case: an empty starting list behaves like a plain insert (the
    // pre-TD-USUB-4 shape, so a first publication is unchanged).
    let fresh = upsert_storage_layout(Vec::new(), parquet.clone());
    assert_eq!(fresh.len(), 1);
    assert_eq!(fresh[0].name, "parquet-snapshot");
}
