use super::*;
use crate::catalog::TableIdentifier;
use crate::network::arrow_ipc::file_export::{
    ArrowFileExportHandler, ArrowFileInfo, ArrowFileRequest, ArrowFileTicket, ExportFileFormat,
};
use crate::services::operations::OperationMetrics;
use arrow_schema::DataType;
use proximadb_catalog::{CatalogColumn, CatalogTableSchema};
use proximadb_data_model::ProximaType;

#[test]
fn test_batch_result_app_metadata_uses_rich_shape() {
    let result = BatchOperationResult::success(
        vec!["record-1".to_string()],
        OperationMetrics {
            total_processed: 1,
            successful_count: 1,
            failed_count: 0,
            updated_count: 0,
            processing_time_us: 123,
            wal_write_time_us: 10,
            index_update_time_us: 20,
        },
    );

    let metadata = ProximaFlightService::batch_result_app_metadata(&result).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&metadata).unwrap();

    assert_eq!(value["success"], true);
    assert_eq!(value["vector_ids"], serde_json::json!(["record-1"]));
    assert_eq!(value["metrics"]["successful_count"], 1);
    assert!(value.get("operation").is_none());
    assert!(value.get("error_message").is_none());
}

#[test]
fn test_table_fqn_from_descriptor_requires_relational_model() {
    let relational_path =
        FlightDescriptor::new_path(vec!["relational".to_string(), "events".to_string()]);
    assert_eq!(
        ProximaFlightService::table_fqn_from_descriptor(&relational_path).unwrap(),
        Some("events".to_string())
    );

    let relational_cmd = FlightDescriptor::new_cmd(
        serde_json::to_vec(&serde_json::json!({
            "model_type": "relational",
            "table_fqn": "analytics.events"
        }))
        .unwrap(),
    );
    assert_eq!(
        ProximaFlightService::table_fqn_from_descriptor(&relational_cmd).unwrap(),
        Some("analytics.events".to_string())
    );

    let vector_path = FlightDescriptor::new_path(vec!["vectors".to_string()]);
    assert_eq!(
        ProximaFlightService::table_fqn_from_descriptor(&vector_path).unwrap(),
        None
    );
}

#[tokio::test]
async fn test_catalog_arrow_schema_for_descriptor_uses_xcatalog_schema() {
    let manager = Arc::new(CatalogManager::new());
    let temp_dir = tempfile::tempdir().unwrap();
    let catalog = manager
        .create_native_catalog("default", &format!("file://{}", temp_dir.path().display()))
        .await
        .unwrap();
    let _ = catalog
        .create_namespace(&["default".to_string()], Default::default())
        .await;

    let table_id = TableIdentifier::new(vec!["default".to_string()], "events".to_string());
    let mut embedding = CatalogColumn::new(
        3,
        "embedding",
        ProximaType::DenseVector {
            element: proximadb_data_model::VectorElement::Float32,
            dim: 0,
        },
    );
    embedding
        .properties
        .insert("dimension".to_string(), "3".to_string());
    let table_schema = CatalogTableSchema::new("events")
        .with_column(CatalogColumn::new(1, "id", ProximaType::Int64).nullable(false))
        .with_column(CatalogColumn::new(2, "payload", ProximaType::Json))
        .with_column(embedding);
    catalog.create_table(&table_id, table_schema).await.unwrap();

    let descriptor =
        FlightDescriptor::new_path(vec!["relational".to_string(), "events".to_string()]);
    let schema = ProximaFlightService::catalog_arrow_schema_for_descriptor(
        Some(&manager),
        &descriptor,
        proximadb_tenant::DEFAULT_TENANT,
    )
    .await
    .unwrap()
    .expect("catalog schema");

    assert_eq!(schema.fields().len(), 3);
    assert_eq!(schema.field(0).name(), "id");
    assert_eq!(*schema.field(0).data_type(), DataType::Int64);
    assert!(!schema.field(0).is_nullable());
    assert_eq!(schema.field(1).name(), "payload");
    assert_eq!(*schema.field(1).data_type(), DataType::Utf8);
    assert_eq!(
        *schema.field(2).data_type(),
        DataType::List(Box::new(arrow_schema::Field::new("item", DataType::Float32, true)).into())
    );
}

#[tokio::test]
async fn test_records_for_write_batches_uses_catalog_bulk_validation_for_tables() {
    let manager = Arc::new(CatalogManager::new());
    let temp_dir = tempfile::tempdir().unwrap();
    let catalog = manager
        .create_native_catalog("default", &format!("file://{}", temp_dir.path().display()))
        .await
        .unwrap();
    let _ = catalog
        .create_namespace(&["default".to_string()], Default::default())
        .await;

    let table_id = TableIdentifier::new(vec!["default".to_string()], "events".to_string());
    let table_schema = CatalogTableSchema::new("events")
        .with_column(CatalogColumn::new(1, "id", ProximaType::String).nullable(false))
        .with_column(CatalogColumn::new(2, "payload", ProximaType::String));
    catalog.create_table(&table_id, table_schema).await.unwrap();

    let schema = Arc::new(Schema::new(vec![
        arrow_schema::Field::new("id", DataType::Utf8, false),
        arrow_schema::Field::new("payload", DataType::Utf8, true),
    ]));
    let batch = arrow_array::RecordBatch::try_new(
        schema,
        vec![
            Arc::new(arrow_array::StringArray::from(vec!["event-1"])),
            Arc::new(arrow_array::StringArray::from(vec!["loaded"])),
        ],
    )
    .unwrap();

    let (records, catalog_result) = ProximaFlightService::records_for_write_batches(
        Some(&manager),
        Some("events"),
        None,
        FlightWriteOperation::Upsert,
        WriteMode::WAL,
        &[batch],
    )
    .await
    .unwrap();

    assert_eq!(records.len(), 1);
    assert_eq!(records[0].oid, "event-1");
    let catalog_result = catalog_result.expect("catalog preparation result");
    assert_eq!(catalog_result.records_written, 1);
    assert!(!catalog_result.table_created);
}

#[test]
fn test_batch_progress_metadata_is_record_oriented() {
    let result =
        BatchOperationResult::failure("bad record".to_string(), "VALIDATION_FAILED".to_string());

    let metadata = ProximaFlightService::batch_progress_app_metadata(
        FlightWriteOperation::Delete,
        2,
        10,
        7,
        &result,
    )
    .unwrap();
    let value: serde_json::Value = serde_json::from_slice(&metadata).unwrap();

    assert_eq!(value["type"], "progress");
    assert_eq!(value["operation"], "delete");
    assert_eq!(value["batch"], 2);
    assert_eq!(value["batch_rows"], 10);
    assert_eq!(value["total_records"], 7);
    assert_eq!(value["successful_count"], 0);
    assert_eq!(value["failed_count"], 1);
    assert_eq!(value["errors"], serde_json::json!(["bad record"]));
    assert!(value.get("total_vectors").is_none());
}

#[test]
fn test_bulk_completion_metadata_is_operation_tagged() {
    let metadata = ProximaFlightService::bulk_insert_complete_app_metadata(
        FlightWriteOperation::Upsert,
        3,
        25,
        2,
        false,
    )
    .unwrap();
    let value: serde_json::Value = serde_json::from_slice(&metadata).unwrap();

    assert_eq!(value["type"], "complete");
    assert_eq!(value["operation"], "upsert");
    assert_eq!(value["total_batches"], 3);
    assert_eq!(value["total_records"], 25);
    assert_eq!(value["total_failed"], 2);
    assert_eq!(value["success"], false);
    assert!(value.get("total_vectors").is_none());
}

/// TD-TENANT-3 **behaviour change**: the canonical header now wins.
///
/// This test previously asserted the opposite — that `x-proximadb-tenant-id`
/// took precedence over `x-tenant-id`. That precedence pointed away from the
/// standard spelling, so a client sending both would converge on the alias
/// slated for removal (S4). Inverting it converges on the canonical name
/// instead. Only a client sending *both* headers with *different* tenant
/// values is affected, and the resolved tenant remains subject to the same
/// `HeaderTrustPolicy` gate either way.
#[test]
fn test_tenant_id_from_flight_metadata_prefers_canonical_header() {
    let mut metadata = tonic::metadata::MetadataMap::new();
    metadata.insert("x-tenant-id", "tenant-b".parse().unwrap());
    metadata.insert("x-proximadb-tenant-id", "tenant-a".parse().unwrap());

    assert_eq!(
        ProximaFlightService::tenant_id_from_metadata(&metadata),
        Some("tenant-b".to_string())
    );
}

/// TD-TENANT-3 S4: the retirement gate is now METRIC-based, not log-based —
/// every deprecated use increments `proximadb_deprecated_claim_uses_total`
/// even though the warn fires once per process. This drives the REAL
/// `tenant_id_from_metadata` twice and asserts the counter moved by exactly
/// two: without this, a long-lived pod goes log-silent while clients keep
/// using an alias, and "quiet warnings" would falsely read as "migrated".
#[test]
fn test_deprecated_alias_use_is_counted_per_use() {
    let label = |v: f64| v;
    let before = crate::network::middleware::claim_metrics::DEPRECATED_CLAIM_USES
        .get_metric_with_label_values(&["flight", "tenant_id"])
        .map(|c| c.get())
        .unwrap_or(0.0);

    for _ in 0..2 {
        // Canonical carries the grant; the legacy name is present alongside
        // purely as the DETECTED attempt (honoring removed, S4 item 1).
        let mut metadata = tonic::metadata::MetadataMap::new();
        metadata.insert("x-tenant-id", "tenant-a".parse().unwrap());
        metadata.insert("tenant_id", "legacy-attempt".parse().unwrap());
        assert_eq!(
            ProximaFlightService::tenant_id_from_metadata(&metadata),
            Some("tenant-a".to_string())
        );
    }

    let after = crate::network::middleware::claim_metrics::DEPRECATED_CLAIM_USES
        .get_metric_with_label_values(&["flight", "tenant_id"])
        .map(|c| c.get())
        .unwrap_or(0.0);
    assert_eq!(label(after - before), 2.0, "counter must be per-use");
}

/// S4 item 1 (2026-08-29): HONORING REMOVED. A legacy alias alone no longer
/// resolves a tenant — the client falls to its credential/default tenant —
/// but the attempt is still detected (warn + counter), so the change is
/// observable rather than silent. The names are deleted at the next release
/// boundary.
#[test]
fn test_flight_legacy_aliases_no_longer_grant_a_tenant() {
    for alias in proximadb_tenant::DEPRECATED_TENANT_CLAIM_ALIASES {
        let mut metadata = tonic::metadata::MetadataMap::new();
        metadata.insert(
            tonic::metadata::MetadataKey::from_static(alias),
            "tenant-a".parse().unwrap(),
        );

        assert_eq!(
            ProximaFlightService::tenant_id_from_metadata(&metadata),
            None,
            "{alias} must not grant a tenant (honoring removed, S4 item 1)"
        );
    }
}

#[test]
fn test_tenant_id_from_flight_metadata_ignores_empty_header() {
    let mut metadata = tonic::metadata::MetadataMap::new();
    metadata.insert("x-proximadb-tenant-id", "".parse().unwrap());

    assert_eq!(
        ProximaFlightService::tenant_id_from_metadata(&metadata),
        None
    );
}

/// TD-TENANT-3 S2: Arrow Flight had no tier surface at all, so a Flight client
/// always ran at the default tier — on the highest cache-pressure path in the
/// system. These pin the gate's two outcomes with the same drop-not-error
/// semantics REST and gRPC use.
#[test]
fn test_flight_tier_claim_is_stamped_when_the_policy_trusts_it() {
    let tenant = "flight-tier-open-tenant";
    let mut metadata = tonic::metadata::MetadataMap::new();
    metadata.insert("x-tenant-tier", "enterprise".parse().unwrap());

    ProximaFlightService::stamp_gated_tier_claim(
        &metadata,
        tenant,
        None,
        proximadb_tenant::HeaderTrustPolicy::Open,
    );

    assert_eq!(
        crate::services::record_store::tenant_tier(tenant).as_deref(),
        Some("enterprise")
    );
}

#[test]
fn test_flight_tier_claim_is_dropped_for_an_unauthenticated_caller() {
    let tenant = "flight-tier-strict-tenant";
    let mut metadata = tonic::metadata::MetadataMap::new();
    metadata.insert("x-tenant-tier", "enterprise".parse().unwrap());

    // No binding + a strict policy = the escalation vector ADR-0053 W8 closed.
    ProximaFlightService::stamp_gated_tier_claim(
        &metadata,
        tenant,
        None,
        proximadb_tenant::HeaderTrustPolicy::AuthenticatedOnly,
    );

    assert_eq!(crate::services::record_store::tenant_tier(tenant), None);
}

#[test]
fn test_flight_absent_tier_claim_stamps_nothing() {
    let tenant = "flight-tier-absent-tenant";
    let metadata = tonic::metadata::MetadataMap::new();

    ProximaFlightService::stamp_gated_tier_claim(
        &metadata,
        tenant,
        None,
        proximadb_tenant::HeaderTrustPolicy::Open,
    );

    assert_eq!(crate::services::record_store::tenant_tier(tenant), None);
}

#[test]
fn test_flight_tenant_resolution_rejects_missing_multi_tenant_identity() {
    let status = ProximaFlightService::resolve_tenant_for_mode(
        None,
        &proximadb_tenant::TenantDeploymentMode::MultiTenant,
    )
    .unwrap_err();

    assert_eq!(status.code(), tonic::Code::Unauthenticated);
    assert_eq!(
        status.message(),
        "tenant id is required in multi-tenant mode"
    );
}

#[test]
fn test_flight_tenant_resolution_defaults_only_in_single_tenant_mode() {
    let tenant = ProximaFlightService::resolve_tenant_for_mode(
        None,
        &proximadb_tenant::TenantDeploymentMode::single_tenant("embedded"),
    )
    .unwrap();

    assert_eq!(tenant, "embedded");
}

#[test]
fn test_auth_data_from_flight_metadata_accepts_api_key_scheme() {
    let mut metadata = tonic::metadata::MetadataMap::new();
    metadata.insert("authorization", "API-Key key-1".parse().unwrap());

    let auth_data = ProximaFlightService::auth_data_from_metadata(&metadata)
        .unwrap()
        .expect("auth data");

    match auth_data {
        AuthenticationData::ApiKey(key) => assert_eq!(key, "key-1"),
        other => panic!("expected API key auth data, got {:?}", other),
    }
}

#[test]
fn test_auth_data_from_flight_metadata_accepts_bearer_jwt() {
    let mut metadata = tonic::metadata::MetadataMap::new();
    metadata.insert("authorization", "Bearer jwt-1".parse().unwrap());

    let auth_data = ProximaFlightService::auth_data_from_metadata(&metadata)
        .unwrap()
        .expect("auth data");

    match auth_data {
        AuthenticationData::JWTToken(token) => assert_eq!(token, "jwt-1"),
        other => panic!("expected JWT auth data, got {:?}", other),
    }
}

#[test]
fn test_auth_data_from_flight_metadata_accepts_x_api_key() {
    let mut metadata = tonic::metadata::MetadataMap::new();
    metadata.insert("x-api-key", "key-2".parse().unwrap());

    let auth_data = ProximaFlightService::auth_data_from_metadata(&metadata)
        .unwrap()
        .expect("auth data");

    match auth_data {
        AuthenticationData::ApiKey(key) => assert_eq!(key, "key-2"),
        other => panic!("expected API key auth data, got {:?}", other),
    }
}

#[test]
fn test_auth_data_from_peer_certificate_der_uses_raw_cert_bytes() {
    let auth_data =
        ProximaFlightService::auth_data_from_peer_certificate_der(&[1, 2, 3]).expect("auth data");

    match auth_data {
        AuthenticationData::ClientCertificate(cert_data) => {
            assert_eq!(cert_data.raw_cert_der, Some(vec![1, 2, 3]));
        }
        other => panic!("expected client certificate auth data, got {:?}", other),
    }
}

#[test]
fn test_auth_data_from_peer_certificate_der_ignores_empty_cert() {
    assert!(
        ProximaFlightService::auth_data_from_peer_certificate_der(&[]).is_none(),
        "empty peer certs should not create auth data"
    );
}

#[test]
fn test_insert_request_conflict_result_rejects_duplicate_ids() {
    let mut seen = HashSet::new();
    let records = vec![
        ProximaRecord {
            oid: "r1".to_string(),
            ..ProximaRecord::default()
        },
        ProximaRecord {
            oid: "r1".to_string(),
            ..ProximaRecord::default()
        },
    ];

    let result = ProximaFlightService::insert_request_conflict_result(&records, &mut seen)
        .expect("duplicate insert should return conflict");

    assert!(!result.success);
    assert_eq!(result.error_code.as_deref(), Some("INSERT_CONFLICT"));
    assert_eq!(result.metrics.successful_count, 0);
    assert!(result.errors[0].contains("appears more than once"));
}

#[test]
fn test_insert_request_conflict_result_tracks_ids_across_batches() {
    let mut seen = HashSet::new();
    let first_batch = vec![ProximaRecord {
        oid: "r1".to_string(),
        ..ProximaRecord::default()
    }];
    let second_batch = vec![ProximaRecord {
        oid: "r1".to_string(),
        ..ProximaRecord::default()
    }];

    assert!(
        ProximaFlightService::insert_request_conflict_result(&first_batch, &mut seen).is_none()
    );
    let result = ProximaFlightService::insert_request_conflict_result(&second_batch, &mut seen)
        .expect("duplicate across batches should return conflict");

    assert!(!result.success);
    assert_eq!(result.error_code.as_deref(), Some("INSERT_CONFLICT"));
}

#[test]
fn test_exchange_descriptor_from_path() {
    let descriptor =
        FlightDescriptor::new_path(vec!["bulk_delete".to_string(), "records".to_string()]);

    let (exchange_type, collection_id, operation) =
        ProximaFlightService::parse_exchange_descriptor(&descriptor).unwrap();

    assert_eq!(exchange_type, "bulk_delete");
    assert_eq!(collection_id, "records");
    assert_eq!(operation, Some(FlightWriteOperation::Delete));
}

#[test]
fn test_exchange_descriptor_from_command() {
    let descriptor = FlightDescriptor::new_cmd(
        serde_json::to_vec(&serde_json::json!({
            "exchange_type": "bulk_upsert",
            "collection_id": "records"
        }))
        .unwrap(),
    );

    let (exchange_type, collection_id, operation) =
        ProximaFlightService::parse_exchange_descriptor(&descriptor).unwrap();

    assert_eq!(exchange_type, "bulk_upsert");
    assert_eq!(collection_id, "records");
    assert_eq!(operation, Some(FlightWriteOperation::Upsert));
}

#[test]
fn test_exchange_descriptor_from_command_operation_alias() {
    let descriptor = FlightDescriptor::new_cmd(
        serde_json::to_vec(&serde_json::json!({
            "operation": "delete",
            "collection": "records"
        }))
        .unwrap(),
    );

    let (exchange_type, collection_id, operation) =
        ProximaFlightService::parse_exchange_descriptor(&descriptor).unwrap();

    assert_eq!(exchange_type, "bulk_delete");
    assert_eq!(collection_id, "records");
    assert_eq!(operation, Some(FlightWriteOperation::Delete));
}

#[test]
fn test_exchange_descriptor_from_command_upsert_alias() {
    let descriptor = FlightDescriptor::new_cmd(
        serde_json::to_vec(&serde_json::json!({
            "operation": "upsert",
            "collection_id": "records"
        }))
        .unwrap(),
    );

    let (exchange_type, collection_id, operation) =
        ProximaFlightService::parse_exchange_descriptor(&descriptor).unwrap();

    assert_eq!(exchange_type, "bulk_upsert");
    assert_eq!(collection_id, "records");
    assert_eq!(operation, Some(FlightWriteOperation::Upsert));
}

#[test]
fn test_arrow_file_ticket_detection() {
    // Test valid arrow file ticket
    let valid_ticket = Ticket {
        ticket: serde_json::to_vec(&serde_json::json!({
            "type": "arrow_file",
            "collection_id": "test_collection",
            "file_path": "/path/to/file.arrow"
        }))
        .unwrap()
        .into(),
    };
    assert!(ArrowFileTicket::is_arrow_file_ticket(&valid_ticket));

    // Test search request ticket (not an arrow file ticket)
    let search_ticket = Ticket {
        ticket: serde_json::to_vec(&serde_json::json!({
            "collection_id": "test_collection",
            "query_vector": [0.1, 0.2, 0.3],
            "top_k": 10
        }))
        .unwrap()
        .into(),
    };
    assert!(!ArrowFileTicket::is_arrow_file_ticket(&search_ticket));

    // Test invalid JSON ticket
    let invalid_ticket = Ticket {
        ticket: b"not json".to_vec().into(),
    };
    assert!(!ArrowFileTicket::is_arrow_file_ticket(&invalid_ticket));
}

#[test]
fn test_arrow_file_ticket_parsing() {
    let ticket = Ticket {
        ticket: serde_json::to_vec(&serde_json::json!({
            "type": "arrow_file",
            "collection_id": "my_collection",
            "file_path": "/data/my_collection/data/block_0.arrow"
        }))
        .unwrap()
        .into(),
    };

    let parsed = ArrowFileTicket::from_ticket(&ticket).unwrap();
    assert_eq!(parsed.ticket_type, "arrow_file");
    assert_eq!(parsed.collection_id, "my_collection");
    assert_eq!(parsed.file_path, "/data/my_collection/data/block_0.arrow");
}

#[test]
fn test_arrow_file_info_serialization() {
    let file_info = ArrowFileInfo {
        path: "/data/test/data/block_0.arrow".to_string(),
        filename: "block_0.arrow".to_string(),
        size_bytes: 1024 * 1024, // 1MB
        num_batches: 10,
        total_records: 10000,
        dimension: 768,
        modified_at: 1704067200, // 2024-01-01 00:00:00 UTC
        format: ExportFileFormat::Arrow,
    };

    let json = serde_json::to_string(&file_info).unwrap();
    let parsed: ArrowFileInfo = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.path, file_info.path);
    assert_eq!(parsed.filename, file_info.filename);
    assert_eq!(parsed.size_bytes, file_info.size_bytes);
    assert_eq!(parsed.num_batches, file_info.num_batches);
    assert_eq!(parsed.total_records, file_info.total_records);
    assert_eq!(parsed.dimension, file_info.dimension);
    assert_eq!(parsed.format, ExportFileFormat::Arrow);
}

#[test]
fn test_arrow_file_export_handler_creation() {
    let storage_locations = vec![
        "file:///tmp/proximadb/d1".to_string(),
        "file:///tmp/proximadb/d2".to_string(),
    ];
    // Handler should be created successfully with storage locations
    let _handler = ArrowFileExportHandler::new(storage_locations);
    // If we get here without panic, the handler was created successfully
}

#[test]
fn test_arrow_file_request_ticket_creation() {
    let request = ArrowFileRequest {
        collection_id: "test_collection".to_string(),
        file_pattern: Some("*.arrow".to_string()),
        limit: Some(100),
        compression: None,
    };

    let ticket = request.create_ticket("/path/to/file.arrow");

    // Verify ticket can be parsed back
    let parsed = ArrowFileTicket::from_ticket(&ticket).unwrap();
    assert_eq!(parsed.collection_id, "test_collection");
    assert_eq!(parsed.file_path, "/path/to/file.arrow");
}

// ── Native embedding dispatch tests (Phase 1) ──────────────────────────

fn init_embedding_singleton() {
    use proximadb_embedding::{
        EmbeddingService,
        config::{EmbedRoute, EmbeddingConfig},
        scheduler::EmbedSchedulerConfig,
    };

    // Idempotent: second call to initialize is a no-op via OnceCell.
    if EmbeddingService::try_global().is_some() {
        return;
    }
    let _ = EmbeddingService::initialize(
        EmbeddingConfig {
            route: EmbedRoute::BgeSmall,
        },
        EmbedSchedulerConfig::default(),
    );
}

/// Records arriving with text but no vector get their `embeddings` field
/// populated by the in-process EmbeddingService at the route's declared
/// dimension (384 for bge-small), which is exactly what the downstream
/// WAL + index paths need to function end-to-end.
///
/// Gated on `--features onnx`: the BGE route requires the real ONNX
/// runtime, and `BgeModel::initialize` deliberately returns
/// `ModelUnavailable` when `onnx` is off (synthetic fallback is forbidden
/// in production paths — see bge.rs). Without the gate this test fails in
/// every default (onnx-off) build.
#[cfg(feature = "onnx")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn embed_text_only_records_populates_empty_embeddings() {
    use proximadb_data_model::ProximaValue;
    use proximadb_records::{ProximaRecord, ProximaTreeNode};

    init_embedding_singleton();

    let mut records = vec![
        ProximaRecord {
            oid: "doc-1".to_string(),
            local_id: Some("doc-1".to_string()),
            tenant_id: "tenant-a".to_string(),
            props: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "text".to_string(),
                    ProximaTreeNode::Value(ProximaValue::String(
                        "API gateway returned 503; check upstream connector health".into(),
                    )),
                );
                m
            },
            ..ProximaRecord::default()
        },
        ProximaRecord {
            oid: "doc-2".to_string(),
            local_id: Some("doc-2".to_string()),
            tenant_id: "tenant-a".to_string(),
            props: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "text".to_string(),
                    ProximaTreeNode::Value(ProximaValue::String("rate limit 429 retry".into())),
                );
                m
            },
            ..ProximaRecord::default()
        },
    ];

    ProximaFlightService::embed_text_only_records(&mut records, Some("tenant-a"))
        .await
        .unwrap();

    assert_eq!(records[0].embeddings.len(), 1);
    assert_eq!(records[1].embeddings.len(), 1);
    assert_eq!(records[0].embeddings[0].dim, 384, "bge-small dimension");
    assert_eq!(records[0].embeddings[0].values.len(), 384);
    assert_eq!(records[0].embeddings[0].modality, "dense_vector");
    // Deterministic: same text → same vector
    assert_ne!(
        records[0].embeddings[0].values,
        records[1].embeddings[0].values
    );
}

/// Records that already have a vector populated should pass through
/// untouched — no embedding inference happens, no extra EmbeddingCell.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn embed_text_only_records_skips_records_with_existing_vector() {
    use proximadb_records::{EmbeddingCell, ProximaRecord};

    init_embedding_singleton();

    let mut records = vec![ProximaRecord {
        oid: "doc-prevector".to_string(),
        local_id: Some("doc-prevector".to_string()),
        tenant_id: "tenant-b".to_string(),
        embeddings: vec![EmbeddingCell::new_fp32(
            "client-provided",
            "dense_vector",
            1536,
            vec![0.1_f32; 1536],
        )],
        ..ProximaRecord::default()
    }];

    ProximaFlightService::embed_text_only_records(&mut records, Some("tenant-b"))
        .await
        .unwrap();

    // Unchanged: still exactly one embedding, still 1536-dim, still the
    // client-provided model id.
    assert_eq!(records[0].embeddings.len(), 1);
    assert_eq!(records[0].embeddings[0].dim, 1536);
    assert_eq!(records[0].embeddings[0].model_id, "client-provided");
}

/// extract_record_text reads from `text` first, then falls back to `body`
/// and `title` so connectors that normalize through AnvaiDocument (which
/// carries title/body separately) still produce embeddings.
#[test]
fn extract_record_text_prefers_text_then_body_then_title() {
    use proximadb_data_model::ProximaValue;
    use proximadb_records::{ProximaRecord, ProximaTreeNode};

    let mk = |key: &str, value: &str| {
        let mut r = ProximaRecord::default();
        r.oid = "r".into();
        r.props.insert(
            key.to_string(),
            ProximaTreeNode::Value(ProximaValue::String(value.into())),
        );
        r
    };

    assert_eq!(
        ProximaFlightService::extract_record_text(&mk("text", "from-text")),
        Some("from-text".to_string())
    );
    assert_eq!(
        ProximaFlightService::extract_record_text(&mk("body", "from-body")),
        Some("from-body".to_string())
    );
    assert_eq!(
        ProximaFlightService::extract_record_text(&mk("title", "from-title")),
        Some("from-title".to_string())
    );
    assert_eq!(
        ProximaFlightService::extract_record_text(&ProximaRecord::default()),
        None
    );
}

// ── Slice 6.2: primary-pod gate ─────────────────────────────────

use crate::cluster::primary_pod_registry::{AssignmentReason, PrimaryPodRegistry};

fn make_gate(registry: Arc<PrimaryPodRegistry>, self_pod_id: &str) -> Option<FlightPrimaryPodGate> {
    Some(FlightPrimaryPodGate {
        registry,
        self_pod_id: self_pod_id.to_string(),
    })
}

#[test]
fn flight_gate_unconfigured_allows_writes() {
    // Backwards-compat: deployments that don't set
    // with_primary_pod_gate (eg. embedded / unit-test
    // construction) must NOT see any new rejections.
    assert!(check_flight_primary_pod_gate(&None, "tenant-a", "coll-1").is_ok());
}

#[test]
fn flight_gate_allows_when_no_binding_exists() {
    let registry = Arc::new(PrimaryPodRegistry::new());
    let g = make_gate(registry, "pod-self");
    assert!(check_flight_primary_pod_gate(&g, "tenant-a", "coll-1").is_ok());
}

#[test]
fn flight_gate_allows_when_binding_matches_self_pod() {
    let registry = Arc::new(PrimaryPodRegistry::new());
    registry.assign("tenant-a", "coll-1", "pod-self", AssignmentReason::Create);
    let g = make_gate(registry, "pod-self");
    assert!(check_flight_primary_pod_gate(&g, "tenant-a", "coll-1").is_ok());
}

#[test]
fn flight_gate_rejects_misrouted_with_failed_precondition_and_metadata() {
    // Locks the wire contract: same Status code + same trailing
    // metadata keys as the gRPC v2 path (covered by
    // record_service tests). A future change that drops one of
    // these headers breaks both at once — that's the point.
    let registry = Arc::new(PrimaryPodRegistry::new());
    registry.assign(
        "tenant-a",
        "coll-1",
        "pod-other",
        AssignmentReason::Operator,
    );
    let g = make_gate(registry, "pod-self");
    let status = check_flight_primary_pod_gate(&g, "tenant-a", "coll-1")
        .expect_err("must reject misrouted write");

    assert_eq!(status.code(), tonic::Code::FailedPrecondition);

    let md = status.metadata();
    assert_eq!(
        md.get("x-primary-pod").unwrap().to_str().unwrap(),
        "pod-other"
    );
    assert_eq!(md.get("x-tenant-id").unwrap().to_str().unwrap(), "tenant-a");
    assert_eq!(
        md.get("x-collection-id").unwrap().to_str().unwrap(),
        "coll-1"
    );
}

#[test]
fn flight_gate_scopes_per_tenant_collection_pair() {
    // Binding on (tenant-a, coll-1) must not leak to other pairs.
    // Same property the gRPC v2 + REST v2 paths enforce.
    let registry = Arc::new(PrimaryPodRegistry::new());
    registry.assign(
        "tenant-a",
        "coll-1",
        "pod-other",
        AssignmentReason::Operator,
    );
    let g = make_gate(registry, "pod-self");

    assert!(check_flight_primary_pod_gate(&g, "tenant-a", "coll-2").is_ok());
    assert!(check_flight_primary_pod_gate(&g, "tenant-b", "coll-1").is_ok());
    assert!(check_flight_primary_pod_gate(&g, "tenant-a", "coll-1").is_err());
}
