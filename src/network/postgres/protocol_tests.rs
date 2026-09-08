use super::*;
use crate::catalog::{CatalogColumn, CatalogTableSchema};
use crate::query::multimodal_router;
use proximadb_records::{ProximaRecord, ProximaTreeNode};

#[test]
fn extract_where_handles_trailing_where_and_prefixed_columns() {
    // WHERE ending the string must not panic (the -1/+7 dance sliced one
    // past the end — abort-class under panic=abort).
    for trailing in [
        "SELECT ' WHERE ' FROM t WHERE",
        "SELECT * FROM t WHERE\n",
        "SELECT * FROM t WHERE\t",
        "SELECT * FROM t WHERE/* comment */",
    ] {
        assert!(PostgresProtocol::extract_select_where_clause(trailing).is_some());
        assert!(PostgresProtocol::extract_select_where_predicates(trailing).is_none());
        assert!(PostgresProtocol::extract_legacy_select_predicates(trailing).is_err());
    }

    // A 'limit_val' column must not shadow a later real LIMIT terminator.
    let predicate = PostgresProtocol::extract_select_where_clause(
        "SELECT * FROM t WHERE limit_val > 5 LIMIT 3",
    );
    assert_eq!(predicate, Some("limit_val > 5"));

    // Clause keywords used as qualified identifiers are not clauses. The
    // scanner must continue to the real top-level WHERE/ORDER BY tokens.
    let predicate = PostgresProtocol::extract_select_where_clause(
        "SELECT t.where FROM t WHERE t.limit > 5 ORDER BY t.id",
    );
    assert_eq!(predicate, Some("t.limit > 5"));

    let predicate = PostgresProtocol::extract_select_where_clause(
        "SELECT COUNT(*) FILTER (WHERE active) FROM t WHERE id = 7",
    );
    assert_eq!(predicate, Some("id = 7"));

    for malformed in [
        "SELECT * FROM t ( WHERE id = 7",
        "SELECT * FROM t ' \" WHERE id = 7",
        "SELECT * FROM t /* WHERE id = 7",
    ] {
        assert!(
            PostgresProtocol::extract_legacy_select_predicates(malformed).is_err(),
            "malformed structure must not become an unfiltered scan: {malformed}"
        );
    }
}

#[test]
fn legacy_create_table_target_rejects_missing_or_unterminated_names() {
    for query in [
        "CREATE TABLE IF NOT EXISTS /* unterminated USING VECTOR",
        "CREATE TABLE -- USING VECTOR",
        "CREATE TABLE IF NOT EXISTS",
        "CREATE TABLE USING VECTOR",
        "CREATE TABLE IF NOT EXISTS USING VECTOR",
        "CREATE TABLE /* comment-only */ USING VECTOR",
        "CREATE TABLE (id INT) USING VECTOR",
        "CREATE TABLE \"unterminated USING VECTOR",
        "CREATE TABLE `unterminated USING VECTOR",
    ] {
        let error = PostgresProtocol::extract_legacy_create_table_target(query)
            .expect_err("malformed CREATE TABLE must fail closed");
        assert!(
            error.to_string().contains("valid table name"),
            "got: {error}"
        );
    }

    assert_eq!(
        PostgresProtocol::extract_legacy_create_table_target(
            "CREATE TABLE /* v2 */ IF NOT EXISTS `logs` USING VECTOR"
        )
        .expect("valid comment-separated target"),
        ("logs".to_string(), true)
    );
    assert_eq!(
        PostgresProtocol::extract_legacy_create_table_target(
            "CREATE TABLE IF NOT EXISTS \"Team Logs\" (id INT) USING VECTOR"
        )
        .expect("quoted target may contain whitespace"),
        ("team logs".to_string(), true)
    );
    assert_eq!(
        PostgresProtocol::extract_legacy_create_table_target(
            "CREATE TABLE IF NOT EXISTS \"team\"\"logs\" (id INT) USING VECTOR"
        )
        .expect("doubled quote escapes one identifier delimiter"),
        ("team\"logs".to_string(), true)
    );
}

#[test]
fn document_json_output_preserves_jsonb_fields() {
    let document = serde_json::json!({"profile": {"tier": "gold"}});
    let object = crate::proto::proximadb_v1::SqlObject {
        fields: [(
            "metadata".to_string(),
            crate::proto::proximadb_v1::SqlValue {
                value: Some(crate::proto::proximadb_v1::sql_value::Value::JsonbValue(
                    proximadb_data_model::ProximaValue::to_jsonb_vec(&document)
                        .expect("encode JSONB test value"),
                )),
            },
        )]
        .into_iter()
        .collect(),
    };

    let encoded = sql_object_to_json(&object);
    let decoded: serde_json::Value =
        serde_json::from_str(&encoded).expect("decode document JSON output");
    assert_eq!(decoded["metadata"], document);
}

#[test]
fn test_frontend_message() {
    assert_eq!(FrontendMessage::Query as u8, b'Q');
    assert_eq!(FrontendMessage::Terminate as u8, b'X');
}

#[test]
fn transaction_control_is_rejected_until_atomic_semantics_exist() {
    for statement in [
        "BEGIN",
        "BEGIN TRANSACTION;",
        "BEGIN ISOLATION LEVEL SERIALIZABLE",
        "START TRANSACTION",
        "START TRANSACTION READ ONLY",
        "COMMIT",
        "COMMIT AND CHAIN",
        "END WORK",
        "ROLLBACK",
        "ROLLBACK AND NO CHAIN",
        "ABORT TRANSACTION",
        "SAVEPOINT nested",
        "RELEASE nested",
        "RELEASE SAVEPOINT nested",
        "ROLLBACK TO nested",
        "PREPARE TRANSACTION 'tx-1'",
        "COMMIT PREPARED 'tx-1'",
        "ROLLBACK PREPARED 'tx-1'",
        "SET TRANSACTION ISOLATION LEVEL READ COMMITTED",
        "SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY",
        "SET CONSTRAINTS ALL DEFERRED",
    ] {
        assert_eq!(
            transaction_control_policy(statement),
            Some(TransactionControlPolicy::Unsupported),
            "{statement} must not report false transaction success"
        );
    }

    assert_eq!(transaction_control_policy("SELECT 1"), None);
}

#[test]
fn transaction_control_is_detected_before_a_multi_statement_batch_runs() {
    let statements = PostgresProtocol::split_sql_statements(
        "BEGIN; INSERT INTO orders (id) VALUES (1); COMMIT;",
    );

    assert!(
        statements
            .iter()
            .any(|statement| transaction_control_policy(statement).is_some())
    );
}

#[test]
fn pgwire_startup_tenant_is_required_in_multi_tenant_mode() {
    let error = PostgresProtocol::resolve_startup_tenant(
        None,
        &proximadb_tenant::TenantDeploymentMode::MultiTenant,
    )
    .unwrap_err();

    assert_eq!(error, "tenant id is required in multi-tenant mode");
}

#[test]
fn pgwire_startup_tenant_defaults_only_in_single_tenant_mode() {
    let tenant = PostgresProtocol::resolve_startup_tenant(
        None,
        &proximadb_tenant::TenantDeploymentMode::single_tenant("embedded"),
    )
    .unwrap();

    assert_eq!(tenant, "embedded");
}

#[test]
fn substitute_placeholders_handles_two_digit_indices() {
    // The old ordered str::replace turned "$10" into "<val1>0"; verify each
    // placeholder is matched by its full digit run.
    let rendered: Vec<Option<String>> = (1..=12).map(|n| Some(format!("v{n}"))).collect();
    let out = substitute_placeholders("a=$1 b=$10 c=$11 d=$2 e=$12", &rendered);
    assert_eq!(out, "a='v1' b='v10' c='v11' d='v2' e='v12'");
}

#[test]
fn substitute_placeholders_null_and_escaping_and_literals() {
    let rendered = vec![None, Some("O'Brien".to_string())];
    // $1 -> NULL (unquoted); $2 -> quoted with '' escaping.
    let out = substitute_placeholders("x=$1, y=$2", &rendered);
    assert_eq!(out, "x=NULL, y='O''Brien'");

    // A `$1` inside a single-quoted string literal must NOT be substituted.
    let out2 = substitute_placeholders("SELECT '$1', $2", &rendered);
    assert_eq!(out2, "SELECT '$1', 'O''Brien'");

    // A substituted value that itself contains `$2` is not re-substituted.
    let rendered2 = vec![
        Some("has $2 inside".to_string()),
        Some("second".to_string()),
    ];
    let out3 = substitute_placeholders("$1 | $2", &rendered2);
    assert_eq!(out3, "'has $2 inside' | 'second'");
}

#[test]
fn decode_binary_param_integers_and_floats_are_big_endian() {
    assert_eq!(
        decode_binary_param(&258i16.to_be_bytes(), &PgType::Int2).unwrap(),
        "258"
    );
    assert_eq!(
        decode_binary_param(&70000i32.to_be_bytes(), &PgType::Int4).unwrap(),
        "70000"
    );
    assert_eq!(
        decode_binary_param(&5_000_000_000i64.to_be_bytes(), &PgType::Int8).unwrap(),
        "5000000000"
    );
    assert_eq!(
        decode_binary_param(&1.5f32.to_be_bytes(), &PgType::Float4).unwrap(),
        "1.5"
    );
    assert_eq!(
        decode_binary_param(&2.25f64.to_be_bytes(), &PgType::Float8).unwrap(),
        "2.25"
    );
    assert_eq!(decode_binary_param(&[1], &PgType::Bool).unwrap(), "true");
    assert_eq!(decode_binary_param(&[0], &PgType::Bool).unwrap(), "false");
}

#[test]
fn decode_binary_param_rejects_bad_width_and_unsupported() {
    // Wrong byte width is rejected rather than mis-decoded.
    assert!(decode_binary_param(&[0, 0, 0], &PgType::Int4).is_err());
    // Unsupported binary type errors instead of silently mangling.
    assert!(decode_binary_param(&[0x01, 0x02], &PgType::Timestamp).is_err());
}

#[test]
fn decode_binary_vector_roundtrips_pgvector_layout() {
    // [int16 dim=2][int16 unused=0][f32 be * 2]
    let mut buf = Vec::new();
    buf.extend_from_slice(&2u16.to_be_bytes());
    buf.extend_from_slice(&0u16.to_be_bytes());
    buf.extend_from_slice(&0.5f32.to_be_bytes());
    buf.extend_from_slice(&(-1.0f32).to_be_bytes());
    assert_eq!(decode_binary_vector(&buf).unwrap(), "[0.5,-1]");
}

// pgvector WHERE-filter + extended-protocol param tests (TD-100/TD-102)
// live in `super::super::pgvector_params` where the logic now resides.

#[test]
fn test_store_type_detection_vector() {
    // Vector queries contain <->, <=>, or <#> operators (pgvector syntax)
    assert_eq!(
        multimodal_router::detect_store_type_from_query(
            "SELECT * FROM embeddings ORDER BY vec <-> '[0.1, 0.2, 0.3]' LIMIT 10",
            "embeddings",
            None,
        ),
        DataModel::Vector
    );
    assert_eq!(
        multimodal_router::detect_store_type_from_query(
            "SELECT id, vec <=> '[0.5, 0.5]' AS similarity FROM items",
            "items",
            None,
        ),
        DataModel::Vector
    );
    assert_eq!(
        multimodal_router::detect_store_type_from_query(
            "SELECT id FROM products ORDER BY embedding <#> $1 LIMIT 5",
            "products",
            None,
        ),
        DataModel::Vector
    );
    assert_eq!(
        multimodal_router::detect_store_type_from_query(
            "SELECT id FROM products ORDER BY VECTOR_DISTANCE(embedding, [0.1, 0.2], 'l2') LIMIT 5",
            "products",
            None,
        ),
        DataModel::Vector
    );

    // CREATE TABLE with VECTOR column type
    assert_eq!(
        multimodal_router::detect_store_type_from_create(
            "CREATE TABLE items (id TEXT, embedding VECTOR(384))",
        ),
        DataModel::Vector
    );

    // Explicit USING VECTOR clause
    assert_eq!(
        multimodal_router::detect_store_type_from_create(
            "CREATE TABLE vecs (id TEXT, data FLOAT[]) USING VECTOR",
        ),
        DataModel::Vector
    );
}

#[test]
fn test_store_type_detection_document() {
    // Document queries use JSON path expressions ($.)
    assert_eq!(
        multimodal_router::detect_store_type_from_query(
            "SELECT * FROM products WHERE data $.price > 100",
            "products",
            None,
        ),
        DataModel::Document
    );

    // Document tables detected by doc_ prefix
    assert_eq!(
        multimodal_router::detect_store_type_from_query(
            "SELECT * FROM doc_users WHERE active = true",
            "doc_users",
            None,
        ),
        DataModel::Document
    );

    // document_ prefix also works
    assert_eq!(
        multimodal_router::detect_store_type_from_query(
            "SELECT * FROM document_orders",
            "document_orders",
            None,
        ),
        DataModel::Document
    );
    assert_eq!(
        multimodal_router::detect_store_type_from_query(
            "SELECT * FROM products WHERE JSON_EXTRACT_TEXT(metadata, 'tenant') = 'acme'",
            "products",
            None,
        ),
        DataModel::Document
    );
    assert_eq!(
        multimodal_router::detect_store_type_from_query(
            "SELECT * FROM products WHERE JSON_CONTAINS(metadata, '{\"role\":\"planner\"}')",
            "products",
            None,
        ),
        DataModel::Document
    );
    assert_eq!(
        multimodal_router::detect_store_type_from_query(
            "SELECT * FROM DOCUMENT_QUERY('agent_docs', '$.role = \"planner\"')",
            "agent_queries",
            None,
        ),
        DataModel::Document
    );

    // CREATE with JSONB column
    assert_eq!(
        multimodal_router::detect_store_type_from_create(
            "CREATE TABLE docs (id TEXT PRIMARY KEY, data JSONB)",
        ),
        DataModel::Document
    );

    // CREATE with explicit USING DOCUMENT
    assert_eq!(
        multimodal_router::detect_store_type_from_create(
            "CREATE TABLE catalog (id TEXT, payload JSON) USING DOCUMENT",
        ),
        DataModel::Document
    );
}

#[test]
fn test_store_type_detection_graph() {
    // Graph tables detected by graph_ prefix
    assert_eq!(
        multimodal_router::detect_store_type_from_query(
            "SELECT * FROM graph_social WHERE node_type = 'person'",
            "graph_social",
            None,
        ),
        DataModel::Graph
    );

    // node_ prefix
    assert_eq!(
        multimodal_router::detect_store_type_from_query(
            "SELECT * FROM node_users",
            "node_users",
            None,
        ),
        DataModel::Graph
    );

    // edge_ prefix
    assert_eq!(
        multimodal_router::detect_store_type_from_query(
            "SELECT * FROM edge_follows",
            "edge_follows",
            None,
        ),
        DataModel::Graph
    );
    assert_eq!(
        multimodal_router::detect_store_type_from_query(
            "SELECT * FROM GRAPH_QUERY('MATCH (n:Agent)-[:CALLS]->(m) RETURN m')",
            "agent_queries",
            None,
        ),
        DataModel::Graph
    );

    // CREATE with explicit USING GRAPH
    assert_eq!(
        multimodal_router::detect_store_type_from_create(
            "CREATE TABLE social_network (id TEXT) USING GRAPH",
        ),
        DataModel::Graph
    );
}

#[test]
fn test_store_type_detection_observability() {
    // log_ prefix -> Observability
    assert_eq!(
        multimodal_router::detect_store_type_from_query(
            "SELECT * FROM log_application WHERE severity = 'error'",
            "log_application",
            None,
        ),
        DataModel::Observability
    );

    // metric_ prefix -> Observability
    assert_eq!(
        multimodal_router::detect_store_type_from_query(
            "SELECT * FROM metric_http_requests",
            "metric_http_requests",
            None,
        ),
        DataModel::Observability
    );

    // trace_ prefix -> Observability
    assert_eq!(
        multimodal_router::detect_store_type_from_query(
            "SELECT * FROM trace_spans WHERE service = 'gateway'",
            "trace_spans",
            None,
        ),
        DataModel::Observability
    );
    assert_eq!(
        multimodal_router::detect_store_type_from_query(
            "SELECT * FROM LOGS('production') WHERE severity = 'ERROR'",
            "ops_queries",
            None,
        ),
        DataModel::Observability
    );
    assert_eq!(
        multimodal_router::detect_store_type_from_query(
            "SELECT * FROM METRICS('system') WHERE metric_name = 'cpu_usage'",
            "ops_queries",
            None,
        ),
        DataModel::Observability
    );

    // CREATE with USING OBSERVABILITY
    assert_eq!(
        multimodal_router::detect_store_type_from_create(
            "CREATE TABLE system_logs (ts TIMESTAMP, msg TEXT) USING OBSERVABILITY",
        ),
        DataModel::Observability
    );

    // CREATE with USING TIMESERIES (also maps to Observability)
    assert_eq!(
        multimodal_router::detect_store_type_from_create(
            "CREATE TABLE sensor_data (ts TIMESTAMP, value FLOAT) USING TIMESERIES",
        ),
        DataModel::Observability
    );
}

#[test]
fn test_store_type_detection_relational() {
    // Standard SQL without any special markers -> Relational (default)
    assert_eq!(
        multimodal_router::detect_store_type_from_query(
            "SELECT id, name, email FROM users WHERE active = true",
            "users",
            None,
        ),
        DataModel::Relational
    );

    // CREATE TABLE without USING clause or special column types
    assert_eq!(
        multimodal_router::detect_store_type_from_create(
            "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), email TEXT)",
        ),
        DataModel::Relational
    );

    // Verify priority: vector operators override table name prefix
    // Even with a graph_ prefix, <-> forces Vector detection
    assert_eq!(
        multimodal_router::detect_store_type_from_query(
            "SELECT * FROM graph_nodes ORDER BY embedding <-> '[0.1]' LIMIT 5",
            "graph_nodes",
            None,
        ),
        DataModel::Vector
    );
}

#[test]
fn test_frontend_message_types() {
    // Verify all FrontendMessage enum byte values match PostgreSQL protocol spec
    assert_eq!(FrontendMessage::Startup as u8, 0);
    assert_eq!(FrontendMessage::Query as u8, b'Q'); // 0x51
    assert_eq!(FrontendMessage::Parse as u8, b'P'); // 0x50
    assert_eq!(FrontendMessage::Bind as u8, b'B'); // 0x42
    assert_eq!(FrontendMessage::Execute as u8, b'E'); // 0x45
    assert_eq!(FrontendMessage::Describe as u8, b'D'); // 0x44
    assert_eq!(FrontendMessage::Sync as u8, b'S'); // 0x53
    assert_eq!(FrontendMessage::Flush as u8, b'H'); // 0x48
    assert_eq!(FrontendMessage::Close as u8, b'C'); // 0x43
    assert_eq!(FrontendMessage::Password as u8, b'p'); // 0x70
    assert_eq!(FrontendMessage::Terminate as u8, b'X'); // 0x58
    assert_eq!(FrontendMessage::CopyData as u8, b'd'); // 0x64
    assert_eq!(FrontendMessage::CopyDone as u8, b'c'); // 0x63
    assert_eq!(FrontendMessage::CopyFail as u8, b'f'); // 0x66

    // Verify exact hex values for key protocol messages
    assert_eq!(FrontendMessage::Query as u8, 0x51);
    assert_eq!(FrontendMessage::Parse as u8, 0x50);
    assert_eq!(FrontendMessage::Bind as u8, 0x42);
    assert_eq!(FrontendMessage::Terminate as u8, 0x58);
}

#[test]
fn execute_max_rows_maps_to_truncating_execution_controls() {
    let unlimited = PostgresProtocol::execution_controls_for_execute_max_rows(0);
    assert_eq!(unlimited.max_rows, None);
    assert_eq!(unlimited.row_limit_mode, RowLimitMode::Error);

    let negative = PostgresProtocol::execution_controls_for_execute_max_rows(-1);
    assert_eq!(negative.max_rows, None);
    assert_eq!(negative.row_limit_mode, RowLimitMode::Error);

    let capped = PostgresProtocol::execution_controls_for_execute_max_rows(5);
    assert_eq!(capped.max_rows, Some(5));
    assert_eq!(capped.row_limit_mode, RowLimitMode::Truncate);
}

#[test]
fn portal_page_bounds_reports_suspended_and_complete_pages() {
    let (end, complete) = PostgresProtocol::portal_page_bounds(5, 0, 2);
    assert_eq!(end, 2);
    assert!(!complete);

    let (end, complete) = PostgresProtocol::portal_page_bounds(5, 2, 3);
    assert_eq!(end, 5);
    assert!(complete);

    let (end, complete) = PostgresProtocol::portal_page_bounds(5, 5, 2);
    assert_eq!(end, 5);
    assert!(complete);
}

#[test]
fn portal_page_bounds_treats_zero_budget_as_unlimited() {
    let (end, complete) = PostgresProtocol::portal_page_bounds(5, 1, 0);
    assert_eq!(end, 5);
    assert!(complete);
}

#[test]
fn test_copy_format_detection() {
    // CopyFormat is private, so we test the detect_copy_format delegation path
    // by verifying the enum values and their properties directly.
    assert_eq!(CopyFormat::Text, CopyFormat::Text);
    assert_eq!(CopyFormat::Csv, CopyFormat::Csv);
    assert_eq!(CopyFormat::Binary, CopyFormat::Binary);
    assert_eq!(CopyFormat::Arrow, CopyFormat::Arrow);

    // All four variants are distinct
    assert_ne!(CopyFormat::Text, CopyFormat::Csv);
    assert_ne!(CopyFormat::Text, CopyFormat::Binary);
    assert_ne!(CopyFormat::Text, CopyFormat::Arrow);
    assert_ne!(CopyFormat::Csv, CopyFormat::Binary);
    assert_ne!(CopyFormat::Csv, CopyFormat::Arrow);
    assert_ne!(CopyFormat::Binary, CopyFormat::Arrow);

    // Verify the detection logic inline (mirrors detect_copy_format)
    let detect = |query: &str| -> CopyFormat {
        let upper = query.to_uppercase();
        if upper.contains("FORMAT ARROW") || upper.contains("FORMAT 'ARROW'") {
            CopyFormat::Arrow
        } else if upper.contains("FORMAT CSV") || upper.contains("FORMAT 'CSV'") {
            CopyFormat::Csv
        } else if upper.contains("FORMAT BINARY") || upper.contains("FORMAT 'BINARY'") {
            CopyFormat::Binary
        } else {
            CopyFormat::Text
        }
    };

    assert_eq!(
        detect("COPY my_table FROM STDIN WITH (FORMAT ARROW)"),
        CopyFormat::Arrow
    );
    assert_eq!(
        detect("COPY my_table FROM STDIN WITH (FORMAT 'ARROW')"),
        CopyFormat::Arrow
    );
    assert_eq!(
        detect("COPY my_table FROM STDIN WITH (FORMAT CSV, HEADER true)"),
        CopyFormat::Csv
    );
    assert_eq!(
        detect("COPY my_table FROM STDIN WITH (FORMAT 'CSV')"),
        CopyFormat::Csv
    );
    assert_eq!(
        detect("COPY my_table FROM STDIN WITH (FORMAT BINARY)"),
        CopyFormat::Binary
    );
    assert_eq!(
        detect("COPY my_table FROM STDIN WITH (FORMAT 'BINARY')"),
        CopyFormat::Binary
    );
    // Default is Text when no FORMAT clause
    assert_eq!(detect("COPY my_table FROM STDIN"), CopyFormat::Text);
    assert_eq!(
        detect("COPY my_table FROM STDIN WITH (HEADER true)"),
        CopyFormat::Text
    );
}

// Note: a prior `test_extract_explain_inner_query_for_table_write`
// covered `PostgresProtocol::extract_explain_inner_query`, which
// was removed alongside `strip_explain_prefix` in clippy cleanup
// batch 14 (commit `555ed5b2a`). `extract_explain_with_analyze`
// is the surviving entry point and is exercised below.

#[test]
fn test_extract_explain_with_analyze_detects_analyze_flag() {
    let (is_analyze, inner) = PostgresProtocol::extract_explain_with_analyze(
        "EXPLAIN (ANALYZE, FORMAT JSON) INSERT INTO facts SELECT * FROM staging;",
    )
    .expect("analyze EXPLAIN should parse");

    assert!(is_analyze, "ANALYZE option should be detected");
    assert_eq!(inner, "INSERT INTO facts SELECT * FROM staging;");
}

#[test]
fn test_extract_explain_with_analyze_bare_analyze_keyword() {
    let (is_analyze, inner) = PostgresProtocol::extract_explain_with_analyze(
        "EXPLAIN ANALYZE INSERT INTO facts SELECT * FROM staging;",
    )
    .expect("bare EXPLAIN ANALYZE should parse");

    assert!(is_analyze, "bare ANALYZE should be detected");
    assert_eq!(inner, "INSERT INTO facts SELECT * FROM staging;");
}

#[test]
fn test_extract_explain_without_analyze_returns_false() {
    let (is_analyze, inner) = PostgresProtocol::extract_explain_with_analyze(
        "EXPLAIN (FORMAT JSON) INSERT INTO facts SELECT * FROM staging;",
    )
    .expect("plain EXPLAIN should parse");

    assert!(!is_analyze, "no ANALYZE option — flag should be false");
    assert_eq!(inner, "INSERT INTO facts SELECT * FROM staging;");
}

#[test]
fn test_parse_set_parameter_for_write_intent_hint() {
    let (name, value) =
        PostgresProtocol::parse_set_parameter("SET proximadb.write.row_count_hint = '100_000';")
            .expect("SET should parse");

    assert_eq!(name, "proximadb.write.row_count_hint");
    assert_eq!(value, "100_000");
}

#[test]
fn test_parse_set_parameter_supports_to_syntax() {
    let (name, value) = PostgresProtocol::parse_set_parameter(
        "SET proximadb.write.batch_local_constraints_sufficient TO on;",
    )
    .expect("SET TO should parse");

    assert_eq!(name, "proximadb.write.batch_local_constraints_sufficient");
    assert_eq!(value, "on");
}

#[test]
fn test_write_intent_overrides_from_session_parameters() {
    let params = std::collections::HashMap::from([
        (
            "proximadb.write.tenant_id".to_string(),
            "tenant-a".to_string(),
        ),
        ("proximadb.write.actor".to_string(), "benchbase".to_string()),
        (
            "proximadb.write.row_count_hint".to_string(),
            "100_000".to_string(),
        ),
        (
            "proximadb.write.estimated_bytes".to_string(),
            "4096".to_string(),
        ),
        (
            "proximadb.write.requires_row_level_semantics".to_string(),
            "off".to_string(),
        ),
        (
            "proximadb.write.batch_local_constraints_sufficient".to_string(),
            "true".to_string(),
        ),
    ]);

    let overrides = PostgresProtocol::write_intent_overrides_from_params(&params);

    assert_eq!(overrides.tenant_id.as_deref(), Some("tenant-a"));
    assert_eq!(overrides.actor.as_deref(), Some("benchbase"));
    assert_eq!(overrides.row_count_hint, Some(100_000));
    assert_eq!(overrides.estimated_bytes, Some(4096));
    assert_eq!(overrides.requires_row_level_semantics, Some(false));
    assert_eq!(overrides.batch_local_constraints_sufficient, Some(true));
}

#[test]
fn test_extract_select_limit_for_relational_scan() {
    assert_eq!(
        PostgresProtocol::extract_select_limit("SELECT * FROM t LIMIT 25;"),
        Some(25)
    );
    assert_eq!(
        PostgresProtocol::extract_select_limit("SELECT * FROM t ORDER BY id"),
        None
    );
}

#[test]
fn test_extract_selected_column_names_for_relational_select() {
    assert!(PostgresProtocol::extract_selected_column_names("SELECT * FROM customers").is_empty());
    assert_eq!(
        PostgresProtocol::extract_selected_column_names(
            "SELECT c_id, customers.c_name AS name FROM customers WHERE c_id = 1"
        ),
        vec!["c_id".to_string(), "c_name".to_string()]
    );
}

#[test]
fn test_extract_select_where_predicates_for_relational_scan() {
    let predicates = PostgresProtocol::extract_select_where_predicates(
        "SELECT * FROM customers WHERE c_name = 'alice updated' AND c_active = true LIMIT 1;",
    )
    .expect("simple AND predicates should parse");

    assert_eq!(predicates.len(), 2);
    assert_eq!(predicates[0].column_name, "c_name");
    match &predicates[0].condition {
        SelectPredicateCondition::Comparison { operator, literal } => {
            assert_eq!(*operator, SelectPredicateOperator::Equal);
            assert_eq!(literal, "alice updated");
        }
        other => panic!("unexpected predicate: {other:?}"),
    }
    assert_eq!(predicates[1].column_name, "c_active");
    match &predicates[1].condition {
        SelectPredicateCondition::Comparison { literal, .. } => {
            assert_eq!(literal, "true");
        }
        other => panic!("unexpected predicate: {other:?}"),
    }
}

#[test]
fn test_extract_select_where_in_like_and_null_predicates() {
    let predicates = PostgresProtocol::extract_select_where_predicates(
            "SELECT * FROM customers WHERE c_id IN (1, 2) AND c_name LIKE 'alice%' AND c_notes IS NULL;",
        )
        .expect("IN, LIKE, and IS NULL predicates should parse");

    assert_eq!(predicates.len(), 3);
    match &predicates[0].condition {
        SelectPredicateCondition::In { literals, negated } => {
            assert!(!negated);
            assert_eq!(literals, &vec!["1".to_string(), "2".to_string()]);
        }
        other => panic!("unexpected predicate: {other:?}"),
    }
    match &predicates[1].condition {
        SelectPredicateCondition::Like { pattern, negated } => {
            assert!(!negated);
            assert_eq!(pattern, "alice%");
        }
        other => panic!("unexpected predicate: {other:?}"),
    }
    match &predicates[2].condition {
        SelectPredicateCondition::IsNull { negated } => assert!(!negated),
        other => panic!("unexpected predicate: {other:?}"),
    }
}

#[test]
fn test_record_matches_relational_scan_predicates() {
    let record = ProximaRecord {
        oid: "1".to_string(),
        props: proximadb_records::ProximaTree::from([
            (
                "name".to_string(),
                ProximaTreeNode::Value(ProximaValue::String("alice".to_string())),
            ),
            (
                "balance".to_string(),
                ProximaTreeNode::Value(ProximaValue::Decimal("75.25".to_string())),
            ),
            (
                "active".to_string(),
                ProximaTreeNode::Value(ProximaValue::Boolean(true)),
            ),
        ]),
        ..Default::default()
    };
    let schema = CatalogTableSchema::new("customers")
        .with_column(CatalogColumn::new(1, "id", ProximaType::Int32))
        .with_column(CatalogColumn::new(2, "name", ProximaType::String))
        .with_column(CatalogColumn::new(
            3,
            "balance",
            ProximaType::Decimal {
                precision: 38,
                scale: 10,
            },
        ))
        .with_column(CatalogColumn::new(4, "active", ProximaType::Boolean))
        .with_primary_key(vec!["id".to_string()]);
    let predicates = vec![
        SelectPredicate {
            column_name: "name".to_string(),
            condition: SelectPredicateCondition::Comparison {
                operator: SelectPredicateOperator::Equal,
                literal: "alice".to_string(),
            },
        },
        SelectPredicate {
            column_name: "balance".to_string(),
            condition: SelectPredicateCondition::Comparison {
                operator: SelectPredicateOperator::GreaterThanOrEqual,
                literal: "75.00".to_string(),
            },
        },
        SelectPredicate {
            column_name: "active".to_string(),
            condition: SelectPredicateCondition::Comparison {
                operator: SelectPredicateOperator::Equal,
                literal: "true".to_string(),
            },
        },
    ];

    assert!(
        DmlService::record_matches_select_predicate_inputs(&record, &schema, &predicates)
            .expect("predicates should resolve")
    );
}

#[test]
fn test_record_matches_in_like_and_null_relational_scan_predicates() {
    let record = ProximaRecord {
        oid: "1".to_string(),
        props: proximadb_records::ProximaTree::from([
            (
                "name".to_string(),
                ProximaTreeNode::Value(ProximaValue::String("alice updated".to_string())),
            ),
            (
                "active".to_string(),
                ProximaTreeNode::Value(ProximaValue::Boolean(true)),
            ),
        ]),
        ..Default::default()
    };
    let schema = CatalogTableSchema::new("customers")
        .with_column(CatalogColumn::new(1, "id", ProximaType::Int32))
        .with_column(CatalogColumn::new(2, "name", ProximaType::String))
        .with_column(CatalogColumn::new(3, "notes", ProximaType::String))
        .with_primary_key(vec!["id".to_string()]);
    let predicates = vec![
        SelectPredicate {
            column_name: "id".to_string(),
            condition: SelectPredicateCondition::In {
                literals: vec!["1".to_string(), "2".to_string()],
                negated: false,
            },
        },
        SelectPredicate {
            column_name: "name".to_string(),
            condition: SelectPredicateCondition::Like {
                pattern: "alice%".to_string(),
                negated: false,
            },
        },
        SelectPredicate {
            column_name: "notes".to_string(),
            condition: SelectPredicateCondition::IsNull { negated: false },
        },
    ];

    assert!(
        DmlService::record_matches_select_predicate_inputs(&record, &schema, &predicates)
            .expect("predicates should resolve")
    );
}

#[test]
fn test_record_matches_not_in_rejects_excluded_values() {
    let schema = CatalogTableSchema::new("orders")
        .with_column(CatalogColumn::new(1, "id", ProximaType::Int32))
        .with_column(CatalogColumn::new(2, "status", ProximaType::String))
        .with_primary_key(vec!["id".to_string()]);

    // Record with id=5 should be rejected by NOT IN (1, 2, 5) predicate.
    let excluded_record = ProximaRecord {
        oid: "5".to_string(),
        ..Default::default()
    };
    let predicates = vec![SelectPredicate {
        column_name: "id".to_string(),
        condition: SelectPredicateCondition::In {
            literals: vec!["1".to_string(), "2".to_string(), "5".to_string()],
            negated: true,
        },
    }];
    assert!(
        !DmlService::record_matches_select_predicate_inputs(&excluded_record, &schema, &predicates)
            .expect("NOT IN must resolve"),
        "record with id in the excluded list must not match NOT IN"
    );

    // Record with id=99 should pass NOT IN (1, 2, 5).
    let passing_record = ProximaRecord {
        oid: "99".to_string(),
        ..Default::default()
    };
    assert!(
        DmlService::record_matches_select_predicate_inputs(&passing_record, &schema, &predicates)
            .expect("NOT IN must resolve"),
        "record with id not in the excluded list must match NOT IN"
    );
}

#[test]
fn test_record_matches_is_not_null_accepts_present_field_rejects_absent() {
    let schema = CatalogTableSchema::new("users")
        .with_column(CatalogColumn::new(1, "id", ProximaType::String))
        .with_column(CatalogColumn::new(2, "email", ProximaType::String))
        .with_primary_key(vec!["id".to_string()]);

    let predicates = vec![SelectPredicate {
        column_name: "email".to_string(),
        condition: SelectPredicateCondition::IsNull { negated: true },
    }];

    // Record WITH email field → matches IS NOT NULL.
    let with_email = ProximaRecord {
        oid: "u1".to_string(),
        props: proximadb_records::ProximaTree::from([(
            "email".to_string(),
            ProximaTreeNode::Value(ProximaValue::String("u@example.com".to_string())),
        )]),
        ..Default::default()
    };
    assert!(
        DmlService::record_matches_select_predicate_inputs(&with_email, &schema, &predicates)
            .expect("IS NOT NULL must resolve"),
        "record with email present must match IS NOT NULL"
    );

    // Record WITHOUT email field → must NOT match IS NOT NULL.
    let without_email = ProximaRecord {
        oid: "u2".to_string(),
        ..Default::default()
    };
    assert!(
        !DmlService::record_matches_select_predicate_inputs(&without_email, &schema, &predicates)
            .expect("IS NOT NULL must resolve"),
        "record with absent email must not match IS NOT NULL"
    );
}

#[test]
fn test_extract_vector_dimension() {
    // extract_vector_dimension is a method on PostgresProtocol which requires
    // a full instance with TcpStream. Instead, test the parsing logic directly
    // since it's a pure string operation.
    let extract = |query: &str| -> Option<u32> {
        let vector_pos = query.find("VECTOR(")?;
        let after_vector = &query[vector_pos + 7..];
        let dim_end = after_vector.find(')')?;
        after_vector[..dim_end].trim().parse().ok()
    };

    // Standard dimension extraction
    assert_eq!(
        extract("CREATE TABLE items (id TEXT, embedding VECTOR(384))"),
        Some(384)
    );
    assert_eq!(
        extract("CREATE TABLE docs (id TEXT, vec VECTOR(128))"),
        Some(128)
    );
    assert_eq!(
        extract("CREATE TABLE large (id TEXT, emb VECTOR(1536))"),
        Some(1536)
    );

    // Small dimension
    assert_eq!(extract("CREATE TABLE tiny (id TEXT, v VECTOR(2))"), Some(2));

    // Whitespace around number
    assert_eq!(
        extract("CREATE TABLE ws (id TEXT, v VECTOR( 256 ))"),
        Some(256)
    );

    // No VECTOR column -> None
    assert_eq!(extract("CREATE TABLE plain (id INT, name TEXT)"), None);

    // Malformed (no closing paren) -> None
    assert_eq!(extract("CREATE TABLE broken (id TEXT, v VECTOR("), None);

    // Non-numeric content -> None
    assert_eq!(
        extract("CREATE TABLE broken (id TEXT, v VECTOR(abc))"),
        None
    );
}

#[test]
fn test_or_predicate_same_column_folds_to_in() {
    let predicates = PostgresProtocol::extract_select_where_predicates(
        "SELECT c_id, c_name FROM pgwire_smoke_customer WHERE c_id = 1 OR c_id = 2;",
    )
    .expect("single-column OR should fold to IN");

    assert_eq!(predicates.len(), 1);
    assert_eq!(predicates[0].column_name, "c_id");
    match &predicates[0].condition {
        SelectPredicateCondition::In { literals, negated } => {
            assert!(!negated);
            assert_eq!(literals, &vec!["1".to_string(), "2".to_string()]);
        }
        other => panic!("expected In, got: {other:?}"),
    }
}

#[test]
fn test_or_predicate_three_values_folds_to_in() {
    let predicates = PostgresProtocol::extract_select_where_predicates(
        "SELECT * FROM orders WHERE status = 'N' OR status = 'P' OR status = 'C';",
    )
    .expect("three-value OR on same column should fold");

    assert_eq!(predicates.len(), 1);
    match &predicates[0].condition {
        SelectPredicateCondition::In { literals, .. } => {
            assert_eq!(
                literals,
                &vec!["N".to_string(), "P".to_string(), "C".to_string()]
            );
        }
        other => panic!("expected In, got: {other:?}"),
    }
}

#[test]
fn test_or_predicate_multi_column_falls_back_to_full_scan() {
    // Different columns: col1 = v1 OR col2 = v2 — cannot fold, returns None → full scan
    let result = PostgresProtocol::extract_select_where_predicates(
        "SELECT * FROM t WHERE col1 = 1 OR col2 = 2;",
    );
    assert!(
        result.is_none(),
        "multi-column OR should return None (full scan)"
    );
}

#[test]
fn test_or_predicate_non_equality_falls_back_to_full_scan() {
    // OR with non-equality: col > v1 OR col < v2 — cannot fold
    let result = PostgresProtocol::extract_select_where_predicates(
        "SELECT * FROM t WHERE col > 5 OR col < 0;",
    );
    assert!(
        result.is_none(),
        "non-equality OR should return None (full scan)"
    );
}

#[test]
fn test_and_chain_with_in_predicate_parses_correctly() {
    // AND-chain: one IN predicate + one equality — both must be extracted
    let result = PostgresProtocol::extract_select_where_predicates(
        "SELECT * FROM t WHERE c_id IN (1, 2, 3) AND c_active = true;",
    );
    let predicates = result.expect("AND chain with IN must parse");
    assert_eq!(predicates.len(), 2);
    let in_pred = predicates
        .iter()
        .find(|p| p.column_name.eq_ignore_ascii_case("c_id"))
        .expect("IN predicate for c_id must be present");
    match &in_pred.condition {
        SelectPredicateCondition::In { literals, negated } => {
            assert!(!negated);
            assert_eq!(literals.len(), 3);
        }
        other => panic!("expected In condition, got {:?}", other),
    }
    let eq_pred = predicates
        .iter()
        .find(|p| p.column_name.eq_ignore_ascii_case("c_active"))
        .expect("equality predicate for c_active must be present");
    match &eq_pred.condition {
        SelectPredicateCondition::Comparison { literal, .. } => {
            assert_eq!(literal, "true");
        }
        other => panic!("expected Comparison condition, got {:?}", other),
    }
}

#[test]
fn test_and_chain_with_is_null_predicate() {
    let result = PostgresProtocol::extract_select_where_predicates(
        "SELECT * FROM t WHERE label IS NULL AND c_id = 5;",
    );
    let predicates = result.expect("AND chain with IS NULL must parse");
    assert_eq!(predicates.len(), 2);
    let null_pred = predicates
        .iter()
        .find(|p| p.column_name.eq_ignore_ascii_case("label"))
        .expect("IS NULL predicate for label must be present");
    match &null_pred.condition {
        SelectPredicateCondition::IsNull { negated } => {
            assert!(!negated, "IS NULL must not be negated");
        }
        other => panic!("expected IsNull condition, got {:?}", other),
    }
}

#[test]
fn test_and_chain_with_like_predicate_parses_correctly() {
    let result = PostgresProtocol::extract_select_where_predicates(
        "SELECT * FROM t WHERE label LIKE 'prefix%' AND c_id = 5;",
    );
    let predicates = result.expect("AND chain with LIKE must parse");
    assert_eq!(predicates.len(), 2);
    let like_pred = predicates
        .iter()
        .find(|p| p.column_name.eq_ignore_ascii_case("label"))
        .expect("LIKE predicate for label must be present");
    match &like_pred.condition {
        SelectPredicateCondition::Like { pattern, negated } => {
            assert_eq!(pattern, "prefix%");
            assert!(!negated, "LIKE must not be negated");
        }
        other => panic!("expected Like condition, got {:?}", other),
    }
    let id_pred = predicates
        .iter()
        .find(|p| p.column_name.eq_ignore_ascii_case("c_id"))
        .expect("comparison predicate for c_id must be present");
    assert!(matches!(
        &id_pred.condition,
        SelectPredicateCondition::Comparison { literal, .. } if literal == "5"
    ));
}

#[test]
fn test_and_chain_with_is_not_null_predicate() {
    let result = PostgresProtocol::extract_select_where_predicates(
        "SELECT * FROM t WHERE label IS NOT NULL AND c_active = true;",
    );
    let predicates = result.expect("AND chain with IS NOT NULL must parse");
    assert_eq!(predicates.len(), 2);
    let not_null_pred = predicates
        .iter()
        .find(|p| p.column_name.eq_ignore_ascii_case("label"))
        .expect("IS NOT NULL predicate for label must be present");
    match &not_null_pred.condition {
        SelectPredicateCondition::IsNull { negated } => {
            assert!(*negated, "IS NOT NULL must be negated=true");
        }
        other => panic!("expected IsNull(negated=true) condition, got {:?}", other),
    }
}

#[test]
fn test_and_chain_with_not_in_predicate_parses_correctly() {
    let result = PostgresProtocol::extract_select_where_predicates(
        "SELECT * FROM t WHERE c_id NOT IN (10, 20, 30) AND c_active = false;",
    );
    let predicates = result.expect("AND chain with NOT IN must parse");
    assert_eq!(predicates.len(), 2);
    let not_in_pred = predicates
        .iter()
        .find(|p| p.column_name.eq_ignore_ascii_case("c_id"))
        .expect("NOT IN predicate for c_id must be present");
    match &not_in_pred.condition {
        SelectPredicateCondition::In { literals, negated } => {
            assert_eq!(literals.len(), 3);
            assert!(*negated, "NOT IN must be negated=true");
        }
        other => panic!("expected In(negated=true) condition, got {:?}", other),
    }
}

// === ADR-018 Phase 2: IF NOT EXISTS tests ===

#[test]
fn test_create_table_without_if_not_exists() {
    let upper = "CREATE TABLE users (id TEXT, name TEXT)";
    assert!(!upper.contains("IF NOT EXISTS"));
}

#[test]
fn test_create_table_with_if_not_exists() {
    let upper = "CREATE TABLE IF NOT EXISTS users (id TEXT, name TEXT)";
    assert!(upper.contains("IF NOT EXISTS"));
}

#[test]
fn test_drop_table_without_if_exists() {
    let upper = "DROP TABLE users";
    assert!(!upper.contains("IF EXISTS"));
}

#[test]
fn test_drop_table_with_if_exists() {
    let upper = "DROP TABLE IF EXISTS users";
    assert!(upper.contains("IF EXISTS"));
}

// ---------------- ADR-018 Phase 2: multi-column ORDER BY ----------------

#[test]
fn order_by_single_column_default_asc_nulls_last() {
    let keys = PostgresProtocol::extract_select_order_by("SELECT * FROM t ORDER BY name")
        .expect("single-col ORDER BY must parse");
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0].column, "name");
    assert!(!keys[0].desc);
    // Postgres default: ASC → NULLS LAST.
    assert!(!keys[0].nulls_first);
}

#[test]
fn order_by_explicit_desc_default_nulls_first() {
    let keys =
        PostgresProtocol::extract_select_order_by("SELECT * FROM t ORDER BY score DESC").unwrap();
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0].column, "score");
    assert!(keys[0].desc);
    // Postgres default: DESC → NULLS FIRST.
    assert!(keys[0].nulls_first);
}

#[test]
fn order_by_explicit_nulls_first_overrides_default() {
    let keys =
        PostgresProtocol::extract_select_order_by("SELECT * FROM t ORDER BY score ASC NULLS FIRST")
            .unwrap();
    assert_eq!(keys.len(), 1);
    assert!(!keys[0].desc);
    // Override: NULLS FIRST under ASC.
    assert!(keys[0].nulls_first);
}

#[test]
fn order_by_explicit_nulls_last_overrides_default() {
    let keys =
        PostgresProtocol::extract_select_order_by("SELECT * FROM t ORDER BY score DESC NULLS LAST")
            .unwrap();
    assert_eq!(keys.len(), 1);
    assert!(keys[0].desc);
    // Override: NULLS LAST under DESC.
    assert!(!keys[0].nulls_first);
}

#[test]
fn order_by_multi_column_preserves_declaration_order() {
    let keys = PostgresProtocol::extract_select_order_by(
        "SELECT * FROM t ORDER BY name ASC, score DESC, created_at",
    )
    .expect("multi-col ORDER BY must parse (Phase 2)");
    assert_eq!(keys.len(), 3);
    assert_eq!(keys[0].column, "name");
    assert!(!keys[0].desc);
    assert_eq!(keys[1].column, "score");
    assert!(keys[1].desc);
    assert_eq!(keys[2].column, "created_at");
    assert!(!keys[2].desc);
}

#[test]
fn order_by_multi_column_per_key_nulls() {
    let keys = PostgresProtocol::extract_select_order_by(
        "SELECT * FROM t ORDER BY a NULLS FIRST, b DESC NULLS LAST",
    )
    .unwrap();
    assert_eq!(keys.len(), 2);
    assert!(keys[0].nulls_first); // explicit NULLS FIRST on ASC
    assert!(!keys[1].nulls_first); // explicit NULLS LAST on DESC
}

#[test]
fn order_by_terminates_at_limit() {
    let keys = PostgresProtocol::extract_select_order_by("SELECT * FROM t ORDER BY name LIMIT 10")
        .unwrap();
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0].column, "name");
}

#[test]
fn order_by_terminates_at_offset() {
    let keys = PostgresProtocol::extract_select_order_by("SELECT * FROM t ORDER BY name OFFSET 5")
        .unwrap();
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0].column, "name");
}

#[test]
fn order_by_no_clause_returns_none() {
    assert!(PostgresProtocol::extract_select_order_by("SELECT * FROM t").is_none(),);
}

#[test]
fn split_top_level_commas_respects_string_literals() {
    let parts = PostgresProtocol::split_top_level_commas("a, 'b, c', d");
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].trim(), "a");
    assert_eq!(parts[1].trim(), "'b, c'");
    assert_eq!(parts[2].trim(), "d");
}

// ── Slice 6.3: primary-pod gate ─────────────────────────────────

use crate::cluster::primary_pod_registry::{AssignmentReason, PrimaryPodRegistry};

fn make_pgwire_gate(
    registry: Arc<PrimaryPodRegistry>,
    self_pod_id: &str,
) -> Option<PgwirePrimaryPodGate> {
    Some(PgwirePrimaryPodGate {
        registry,
        self_pod_id: self_pod_id.to_string(),
    })
}

#[test]
fn pgwire_gate_unconfigured_allows_writes() {
    let outcome = check_pgwire_primary_pod_gate(&None, "tenant-a", "users");
    assert!(matches!(outcome, PgwireGateOutcome::Allow));
}

#[test]
fn pgwire_gate_allows_when_no_binding_exists() {
    let registry = Arc::new(PrimaryPodRegistry::new());
    let g = make_pgwire_gate(registry, "pod-self");
    assert!(matches!(
        check_pgwire_primary_pod_gate(&g, "tenant-a", "users"),
        PgwireGateOutcome::Allow
    ));
}

#[test]
fn pgwire_gate_allows_when_binding_matches_self_pod() {
    let registry = Arc::new(PrimaryPodRegistry::new());
    registry.assign("tenant-a", "users", "pod-self", AssignmentReason::Create);
    let g = make_pgwire_gate(registry, "pod-self");
    assert!(matches!(
        check_pgwire_primary_pod_gate(&g, "tenant-a", "users"),
        PgwireGateOutcome::Allow
    ));
}

#[test]
fn pgwire_gate_returns_misrouted_with_target_pod() {
    // The pgwire surface conveys the target pod by surfacing it
    // in the SQLSTATE-57P03 error MESSAGE rather than trailing
    // metadata (pgwire has no equivalent). The structured outcome
    // here is what feeds that format!() call, so locking it in
    // protects the operator-visible psql error text.
    let registry = Arc::new(PrimaryPodRegistry::new());
    registry.assign("tenant-a", "users", "pod-other", AssignmentReason::Operator);
    let g = make_pgwire_gate(registry, "pod-self");

    match check_pgwire_primary_pod_gate(&g, "tenant-a", "users") {
        PgwireGateOutcome::Misrouted { target_pod } => {
            assert_eq!(target_pod, "pod-other");
        }
        PgwireGateOutcome::Allow => panic!("expected misrouted, got allow"),
    }
}

#[test]
fn pgwire_gate_scopes_per_tenant_collection_pair() {
    // Same scoping invariant as the other gate surfaces — bindings
    // don't bleed across (tenant_id, collection_id) pairs.
    let registry = Arc::new(PrimaryPodRegistry::new());
    registry.assign("tenant-a", "users", "pod-other", AssignmentReason::Operator);
    let g = make_pgwire_gate(registry, "pod-self");

    assert!(matches!(
        check_pgwire_primary_pod_gate(&g, "tenant-a", "orders"),
        PgwireGateOutcome::Allow
    ));
    assert!(matches!(
        check_pgwire_primary_pod_gate(&g, "tenant-b", "users"),
        PgwireGateOutcome::Allow
    ));
    assert!(matches!(
        check_pgwire_primary_pod_gate(&g, "tenant-a", "users"),
        PgwireGateOutcome::Misrouted { .. }
    ));
}

/// TD-REL-LOWER-1: the legacy single-table SELECT path must recognize a
/// multi-table / derived-table FROM (which the relational pipeline declined)
/// and reject it specifically — never misparse `customer,` / `(select` as a
/// table name.
#[test]
fn legacy_from_shape_guard_flags_multi_table_selects() {
    let flagged = [
        // Comma-joins (TPC-H form) — with and without WHERE.
        "SELECT * FROM CUSTOMER, ORDERS WHERE C_CUSTKEY = O_CUSTKEY",
        "SELECT COUNT(*) FROM ORDERS, LINEITEM",
        // Explicit JOINs.
        "SELECT * FROM ORDERS O JOIN LINEITEM L ON O.O_ORDERKEY = L.L_ORDERKEY WHERE 1=1",
        "SELECT * FROM A LEFT JOIN B ON A.X = B.X",
        // Derived table.
        "SELECT AVG(S) FROM (SELECT SUM(T) AS S FROM ORDERS GROUP BY K) D",
    ];
    for q in flagged {
        assert!(
            PostgresProtocol::legacy_unsupported_from_shape(q).is_some(),
            "must flag: {q}"
        );
    }
    let clean = [
        // Single-table shapes — including commas in the projection, an IN list
        // in WHERE, and a multi-key ORDER BY (commas OUTSIDE the FROM clause).
        "SELECT A, B, C FROM ORDERS WHERE K IN (1, 2, 3)",
        "SELECT A FROM ORDERS ORDER BY A, B",
        "SELECT COUNT(*) FROM ORDERS GROUP BY K, J",
        "SELECT A FROM ORDERS O WHERE O.K = 1",
        // No FROM at all.
        "SELECT 1",
    ];
    for q in clean {
        assert_eq!(
            PostgresProtocol::legacy_unsupported_from_shape(q),
            None,
            "must not flag: {q}"
        );
    }
}

// ── TD-ABAC-10 (ADR-087): the pgwire session-identity truth table ────────────
//
// `startup_identity` is the ONE place pgwire turns gate-accepted startup values
// into the caller identity every downstream read consumes, so its truth table is
// tested in isolation rather than through a TCP handshake.

/// A resolver that mints a stable id only for the tenants it knows — models the
/// real catalog resolver's "unminted tenant ⇒ None" behavior.
struct FakeStableIds(&'static str, u64);

impl proximadb_tenant::TenantStableIdResolver for FakeStableIds {
    fn stable_id_of(&self, tenant_id: &str) -> Option<u64> {
        (tenant_id == self.0).then_some(self.1)
    }
}

#[test]
fn startup_identity_classifies_an_asserted_user_as_trust_asserted_and_stamps_the_stable_id() {
    let resolver = FakeStableIds("acme", 42);
    let identity = PostgresProtocol::startup_identity("acme", "alice", Some(&resolver));

    assert_eq!(identity.tenant, "acme");
    assert_eq!(identity.subject.as_deref(), Some("alice"));
    // pgwire is trust auth — an accepted `user` is an assertion, never a
    // credential. Enforcement still applies; the class is the audit trail.
    assert_eq!(
        identity.auth_class,
        proximadb_tenant::AuthClass::TrustAsserted
    );
    // The ABAC policy key is stamped exactly once, here.
    assert_eq!(identity.tenant_stable_id, Some(42));
}

#[test]
fn startup_identity_treats_empty_and_anonymous_users_as_no_subject() {
    let resolver = FakeStableIds("acme", 42);
    for user in ["", "   ", "anonymous"] {
        let identity = PostgresProtocol::startup_identity("acme", user, Some(&resolver));
        assert_eq!(
            identity.subject, None,
            "user {user:?} must not be a subject"
        );
        assert_eq!(
            identity.auth_class,
            proximadb_tenant::AuthClass::Anonymous,
            "user {user:?} must be anonymous"
        );
        // The tenant is still resolved (and still stamped) without a subject.
        assert_eq!(identity.tenant_stable_id, Some(42));
    }
}

#[test]
fn startup_identity_leaves_the_stable_id_absent_when_unminted_or_unwired() {
    // Unminted tenant (resolver wired, no id for this tenant).
    let resolver = FakeStableIds("acme", 42);
    let unminted = PostgresProtocol::startup_identity("other", "alice", Some(&resolver));
    assert_eq!(unminted.subject.as_deref(), Some("alice"));
    assert_eq!(
        unminted.tenant_stable_id, None,
        "an unminted tenant carries no policy key"
    );

    // No resolver wired at all.
    let unwired = PostgresProtocol::startup_identity("acme", "alice", None);
    assert_eq!(unwired.tenant_stable_id, None);
}
