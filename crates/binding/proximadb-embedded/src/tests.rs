#[cfg(test)]
mod tests {
    use crate::*;
    use std::collections::HashMap;

    // ========================================================================
    // StorageLocationConfig Tests
    // ========================================================================

    #[test]
    fn test_storage_location_url_conversion() {
        let loc = StorageLocationConfig::new("/nvme1/proximadb");
        assert_eq!(loc.to_url(), "file:///nvme1/proximadb");

        let loc = StorageLocationConfig::new("file:///already/url");
        assert_eq!(loc.to_url(), "file:///already/url");

        let loc = StorageLocationConfig::new("relative/path");
        assert!(loc.to_url().starts_with("file://"));
        assert!(loc.to_url().contains("relative/path"));
    }

    #[test]
    fn test_storage_location_builder() {
        let loc = StorageLocationConfig::new("/data")
            .with_weight(2)
            .with_tag("hot");

        assert_eq!(loc.weight, 2);
        assert!(loc.tags.contains(&"hot".to_string()));
    }

    #[test]
    fn test_storage_location_default_values() {
        let loc = StorageLocationConfig::new("/data");
        assert_eq!(loc.path, "/data");
        assert_eq!(loc.weight, 1);
        assert!(loc.tags.is_empty());
    }

    #[test]
    fn test_storage_location_multiple_tags() {
        let loc = StorageLocationConfig::new("/fast-storage")
            .with_tag("hot")
            .with_tag("nvme")
            .with_tag("primary");

        assert_eq!(loc.tags.len(), 3);
        assert!(loc.tags.contains(&"hot".to_string()));
        assert!(loc.tags.contains(&"nvme".to_string()));
        assert!(loc.tags.contains(&"primary".to_string()));
    }

    #[test]
    fn test_storage_location_high_weight() {
        let loc = StorageLocationConfig::new("/high-capacity").with_weight(10);
        assert_eq!(loc.weight, 10);
    }

    // ========================================================================
    // EmbeddedConfig Tests
    // ========================================================================

    #[test]
    fn test_embedded_config_default() {
        let config = EmbeddedConfig::default();
        assert_eq!(config.cache_size_mb, 512);
        assert_eq!(config.default_engine, "sst");
        assert!(config.enable_wal);
        assert_eq!(config.wal_sync_mode, "batch");
        assert_eq!(config.block_prune_mode, "sqrt");
        assert!(config.enable_rl_planner);
        assert!(config.rl_policy_path.is_none());
        assert_eq!(config.access_mode, AccessMode::Exclusive);
        assert!(config.node_id.is_none());
    }

    #[test]
    fn test_embedded_config_storage_locations_default() {
        let config = EmbeddedConfig::default();
        assert_eq!(config.storage_locations.len(), 1);
        assert_eq!(config.storage_locations[0].path, "./data");
        assert_eq!(config.storage_locations[0].weight, 1);
    }

    #[test]
    fn explicit_close_releases_ownership_before_rust_drop() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let config = EmbeddedConfig::for_low_memory(temp_dir.path().to_string_lossy().as_ref());
        let db = EmbeddedProximaDB::new(config.clone()).expect("first open");

        let started = std::time::Instant::now();
        let conflict = EmbeddedProximaDB::new(config.clone());
        assert!(conflict.is_err(), "second exclusive open must fail");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(1),
            "ownership conflict must fail immediately"
        );

        db.close();
        db.close();
        assert!(!db.can_write(), "closed handle must be write-fenced");

        let reopened = EmbeddedProximaDB::new(config)
            .expect("close must release ownership before the old handle is dropped");
        reopened.close();
    }

    #[test]
    fn overlapping_secondary_storage_root_is_rejected() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let shared = temp_dir.path().join("shared");
        let mut first = EmbeddedConfig::for_low_memory(
            temp_dir.path().join("first").to_string_lossy().as_ref(),
        );
        first.storage_locations.push(StorageLocationConfig::new(
            shared.to_string_lossy().as_ref(),
        ));
        let mut second = EmbeddedConfig::for_low_memory(
            temp_dir.path().join("second").to_string_lossy().as_ref(),
        );
        second.storage_locations.push(StorageLocationConfig::new(
            shared.to_string_lossy().as_ref(),
        ));

        let db = EmbeddedProximaDB::new(first).expect("first owner");
        let conflict = EmbeddedProximaDB::new(second);
        assert!(
            conflict.is_err(),
            "different primary roots must still conflict on a shared secondary root"
        );
        db.close();
    }

    #[test]
    fn test_embedded_execute_sql_lowers_agentic_ddl_to_catalog() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let config = EmbeddedConfig::for_low_memory(temp_dir.path().to_string_lossy().as_ref());
        let db = EmbeddedProximaDB::new(config).expect("create embedded db");

        let result = db
            .execute_sql(
                "CREATE TABLE IF NOT EXISTS \"agent_store\" (
                    \"record_id\" TEXT NOT NULL,
                    \"tenant_id\" TEXT NOT NULL,
                    \"payload\" JSONB NOT NULL DEFAULT '{}'::jsonb,
                    \"embedding\" VECTOR(64),
                    PRIMARY KEY (\"record_id\")
                ) WITH (
                    storage_engine = 'SST',
                    layout = 'hybrid',
                    xcatalog_namespace = 'agentic.embedded',
                    schema_kind = 'agentic_mixed'
                );",
                None,
                None,
            )
            .expect("execute create table ddl");

        assert_eq!(result.row_count, 1);
        assert_eq!(result.rows[0]["success"], serde_json::json!(true));

        db.execute_sql(
            "CREATE INDEX idx_agent_payload ON agent_store USING GIN (payload);",
            None,
            None,
        )
        .expect("execute gin index ddl");
        db.execute_sql(
            "CREATE INDEX idx_agent_embedding ON agent_store USING HNSW (embedding);",
            None,
            None,
        )
        .expect("execute hnsw index ddl");

        let (catalog, table_id) = db
            .runtime
            .block_on(db.catalog_manager.resolve_table("agent_store"))
            .expect("resolve agent_store");
        let schema = db
            .runtime
            .block_on(catalog.get_table(&table_id))
            .expect("catalog schema");

        assert_eq!(schema.primary_key, vec!["record_id".to_string()]);
        assert_eq!(
            schema
                .properties
                .get("xcatalog_namespace")
                .map(String::as_str),
            Some("agentic.embedded")
        );
        assert_eq!(
            schema.properties.get("schema_kind").map(String::as_str),
            Some("agentic_mixed")
        );
        assert_eq!(
            schema
                .columns
                .iter()
                .find(|column| column.name == "embedding")
                .and_then(|column| column.properties.get("dimension"))
                .map(String::as_str),
            Some("64")
        );

        let indexes = db
            .runtime
            .block_on(catalog.list_indexes(&table_id))
            .expect("catalog indexes");
        assert!(
            indexes
                .iter()
                .any(|index| index.index_type == proximadb::catalog::CatalogIndexType::Gin)
        );
        assert!(
            indexes
                .iter()
                .any(|index| index.index_type == proximadb::catalog::CatalogIndexType::Hnsw)
        );

        let tables = db
            .execute_sql(
                "SELECT * FROM xcatalog.tables WHERE table_name = 'agent_store';",
                None,
                None,
            )
            .expect("query xcatalog tables");
        assert_eq!(tables.row_count, 1);
        assert_eq!(
            tables.rows[0]["table_name"],
            serde_json::json!("agent_store")
        );
        assert_eq!(
            tables.rows[0]["schema_kind"],
            serde_json::json!("agentic_mixed")
        );
        assert_eq!(tables.rows[0]["storage_engine"], serde_json::json!("SST"));
        assert_eq!(
            tables.rows[0]["xcatalog_namespace"],
            serde_json::json!("agentic.embedded")
        );

        let columns = db
            .execute_sql(
                "SELECT * FROM xcatalog.columns WHERE table_name = 'agent_store';",
                None,
                None,
            )
            .expect("query xcatalog columns");
        assert!(columns.rows.iter().any(|row| {
            row["column_name"] == serde_json::json!("payload")
                && row["data_type"] == serde_json::json!("jsonb")
        }));
        assert!(columns.rows.iter().any(|row| {
            row["column_name"] == serde_json::json!("embedding")
                && row["data_type"] == serde_json::json!("vector")
                && row["vector_dimension"] == serde_json::json!("64")
        }));

        let index_rows = db
            .execute_sql(
                "SELECT * FROM xcatalog.indexes WHERE table_name = 'agent_store';",
                None,
                None,
            )
            .expect("query xcatalog indexes");
        assert!(index_rows.rows.iter().any(|row| {
            row["index_name"] == serde_json::json!("idx_agent_payload")
                && row["index_type"] == serde_json::json!("gin")
        }));
        assert!(index_rows.rows.iter().any(|row| {
            row["index_name"] == serde_json::json!("idx_agent_embedding")
                && row["index_type"] == serde_json::json!("hnsw")
        }));

        let insert = db
            .execute_sql(
                "INSERT INTO agent_store (
                    record_id,
                    tenant_id,
                    payload,
                    embedding
                ) VALUES (
                    'record-1',
                    'tenant-1',
                    '{\"kind\":\"memory\",\"score\":7}'::jsonb,
                    '[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                      0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                      1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
                      2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2,
                      3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
                      4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8,
                      4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6,
                      5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4]'
                );",
                None,
                None,
            )
            .expect("insert canonical mixed record through DML");
        assert_eq!(insert.row_count, 1);
        assert_eq!(insert.rows[0]["success"], serde_json::json!(true));
        assert_eq!(insert.rows[0]["rows_affected"], serde_json::json!(1));
        assert_eq!(
            insert.rows[0]["inserted_ids"],
            serde_json::json!(["record-1"])
        );

        let update = db
            .execute_sql(
                "UPDATE agent_store
                 SET payload = '{\"kind\":\"updated\",\"score\":9}'::jsonb,
                     embedding = '[9.9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                       0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                       1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
                       2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2,
                       3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
                       4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8,
                       4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6,
                       5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4]'
                 WHERE record_id = 'record-1';",
                None,
                None,
            )
            .expect("update canonical mixed record through DML");
        assert_eq!(update.row_count, 1);
        assert_eq!(update.rows[0]["success"], serde_json::json!(true));
        assert_eq!(update.rows[0]["rows_affected"], serde_json::json!(1));

        let updated_record = db
            .get_vector("agent_store", "record-1")
            .expect("get updated record")
            .expect("updated record should exist");
        let updated_vec = updated_record
            .embeddings
            .first()
            .map(|e| e.as_fp32_slice())
            .unwrap_or(&[]);
        assert_eq!(updated_vec.len(), 64);
        assert!((updated_vec[0] - 9.9).abs() < 0.001);

        let default_insert = db
            .execute_sql(
                "INSERT INTO agent_store (
                    record_id,
                    tenant_id,
                    embedding
                ) VALUES (
                    'record-2',
                    'tenant-1',
                    '[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                      0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                      1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
                      2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2,
                      3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
                      4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8,
                      4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6,
                      5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4]'
                );",
                None,
                None,
            )
            .expect("insert mixed record with catalog JSONB default");
        assert_eq!(default_insert.row_count, 1);
        assert_eq!(default_insert.rows[0]["success"], serde_json::json!(true));
        assert_eq!(
            default_insert.rows[0]["inserted_ids"],
            serde_json::json!(["record-2"])
        );

        let delete = db
            .execute_sql(
                "DELETE FROM agent_store WHERE record_id = 'record-2';",
                None,
                None,
            )
            .expect("delete canonical mixed record through DML");
        assert_eq!(delete.row_count, 1);
        assert_eq!(delete.rows[0]["success"], serde_json::json!(true));
        assert_eq!(delete.rows[0]["rows_affected"], serde_json::json!(1));
    }

    #[test]
    fn test_embedded_config_for_benchmarks() {
        let config = EmbeddedConfig::for_benchmarks("/tmp/bench");
        assert_eq!(config.cache_size_mb, 1024);
        assert_eq!(config.default_engine, "sst");
        assert!(config.enable_wal);
        assert_eq!(config.wal_sync_mode, "batch");
        assert!(config.enable_rl_planner);
        assert_eq!(config.storage_locations[0].path, "/tmp/bench");
        assert!(
            config.storage_locations[0]
                .tags
                .contains(&"benchmark".to_string())
        );
    }

    #[test]
    fn test_embedded_config_for_low_memory() {
        let config = EmbeddedConfig::for_low_memory("/tmp/lowmem");
        assert_eq!(config.cache_size_mb, 128);
        assert!(!config.enable_rl_planner);
        assert_eq!(config.storage_locations[0].path, "/tmp/lowmem");
    }

    #[test]
    fn test_embedded_config_with_access_mode() {
        let config = EmbeddedConfig::default().with_access_mode(AccessMode::SharedRead);
        assert_eq!(config.access_mode, AccessMode::SharedRead);
    }

    #[test]
    fn test_embedded_config_with_node_id() {
        let config = EmbeddedConfig::default().with_node_id("node-123");
        assert_eq!(config.node_id, Some("node-123".to_string()));
    }

    #[test]
    fn test_embedded_config_chained_builders() {
        let config = EmbeddedConfig::default()
            .with_access_mode(AccessMode::LeaderFollower)
            .with_node_id("leader-node");
        assert_eq!(config.access_mode, AccessMode::LeaderFollower);
        assert_eq!(config.node_id, Some("leader-node".to_string()));
    }

    #[test]
    fn test_collection_engine_name_uses_storage_engine_strings() {
        assert_eq!(
            collection_engine_name(Some(
                proximadb::proto::proximadb_v1::StorageEngine::Tst as i32
            )),
            "tst"
        );
        assert_eq!(
            collection_engine_name(Some(
                proximadb::proto::proximadb_v1::StorageEngine::Sst as i32
            )),
            "sst"
        );
    }

    // ========================================================================
    // GraphNode Tests
    // ========================================================================

    #[test]
    fn test_graph_node_creation() {
        let node = GraphNode::new("node_1");
        assert_eq!(node.id, "node_1");
        assert!(node.labels.is_empty());
        assert!(node.properties.is_empty());
    }

    #[test]
    fn test_graph_node_with_label() {
        let node = GraphNode::new("user_1").with_label("Person");
        assert_eq!(node.labels.len(), 1);
        assert!(node.labels.contains(&"Person".to_string()));
    }

    #[test]
    fn test_graph_node_multiple_labels() {
        let node = GraphNode::new("entity_1")
            .with_label("Person")
            .with_label("Employee")
            .with_label("Manager");
        assert_eq!(node.labels.len(), 3);
        assert!(node.labels.contains(&"Person".to_string()));
        assert!(node.labels.contains(&"Employee".to_string()));
        assert!(node.labels.contains(&"Manager".to_string()));
    }

    #[test]
    fn test_graph_node_with_property() {
        use proximadb::graph::model::PropertyValue;
        let node = GraphNode::new("user_1").with_property("name", "Alice");
        assert_eq!(
            node.properties.get("name"),
            Some(&PropertyValue::from("Alice"))
        );
    }

    #[test]
    fn test_graph_node_multiple_properties() {
        use proximadb::graph::model::PropertyValue;
        // `age` is an i64 now — the typed boundary (#1698) is the point:
        // it must NOT come back as "30".
        let node = GraphNode::new("user_1")
            .with_property("name", "Alice")
            .with_property("email", "alice@example.com")
            .with_property("age", 30i64);
        assert_eq!(node.properties.len(), 3);
        assert_eq!(
            node.properties.get("name"),
            Some(&PropertyValue::from("Alice"))
        );
        assert_eq!(
            node.properties.get("email"),
            Some(&PropertyValue::from("alice@example.com"))
        );
        assert_eq!(
            node.properties.get("age"),
            Some(&PropertyValue::from(30i64))
        );
    }

    #[test]
    fn test_graph_node_builder_chain() {
        let node = GraphNode::new("func_main")
            .with_label("function")
            .with_label("public")
            .with_property("name", "main")
            .with_property("file", "main.rs")
            .with_property("line", 10i64);

        assert_eq!(node.id, "func_main");
        assert_eq!(node.labels.len(), 2);
        assert_eq!(node.properties.len(), 3);
    }

    #[test]
    fn test_graph_node_to_proto() {
        let node = GraphNode::new("test_node")
            .with_label("TestLabel")
            .with_property("key", "value");

        let proto = node.to_proto();
        assert_eq!(proto.id, "test_node");
        assert!(proto.labels.contains(&"TestLabel".to_string()));
        assert!(proto.properties.contains_key("key"));
    }

    #[test]
    fn test_graph_node_clone() {
        let node = GraphNode::new("original")
            .with_label("Label")
            .with_property("prop", "value");

        let cloned = node.clone();
        assert_eq!(cloned.id, node.id);
        assert_eq!(cloned.labels, node.labels);
        assert_eq!(cloned.properties, node.properties);
    }

    // ========================================================================
    // GraphEdge Tests
    // ========================================================================

    #[test]
    fn test_graph_edge_creation() {
        let edge = GraphEdge::new("node_a", "node_b", "KNOWS");
        assert_eq!(edge.from_node_id, "node_a");
        assert_eq!(edge.to_node_id, "node_b");
        assert_eq!(edge.edge_type, "KNOWS");
        assert!(edge.id.is_none());
        assert!(edge.weight.is_none());
        assert!(edge.properties.is_empty());
    }

    #[test]
    fn test_graph_edge_with_id() {
        let edge = GraphEdge::new("a", "b", "REL").with_id("edge_123");
        assert_eq!(edge.id, Some("edge_123".to_string()));
    }

    #[test]
    fn test_graph_edge_with_weight() {
        let edge = GraphEdge::new("a", "b", "WEIGHTED").with_weight(0.75);
        assert_eq!(edge.weight, Some(0.75));
    }

    #[test]
    fn test_graph_edge_with_property() {
        let edge = GraphEdge::new("a", "b", "RELATIONSHIP").with_property("since", "2024");
        assert_eq!(
            edge.properties.get("since"),
            Some(&proximadb::graph::model::PropertyValue::from("2024"))
        );
    }

    #[test]
    fn test_graph_edge_builder_chain() {
        let edge = GraphEdge::new("user_1", "user_2", "FOLLOWS")
            .with_id("follow_edge")
            .with_weight(1.0)
            .with_property("timestamp", "2024-01-01")
            .with_property("source", "web");

        assert_eq!(edge.from_node_id, "user_1");
        assert_eq!(edge.to_node_id, "user_2");
        assert_eq!(edge.edge_type, "FOLLOWS");
        assert_eq!(edge.id, Some("follow_edge".to_string()));
        assert_eq!(edge.weight, Some(1.0));
        assert_eq!(edge.properties.len(), 2);
    }

    #[test]
    fn test_graph_edge_generated_id() {
        let edge = GraphEdge::new("a", "b", "TYPE");
        // Test the generated_id method indirectly via to_proto
        let proto = edge.to_proto();
        assert_eq!(proto.id, "a->b:TYPE");
    }

    #[test]
    fn test_graph_edge_to_proto() {
        let edge = GraphEdge::new("from", "to", "REL")
            .with_weight(0.5)
            .with_property("key", "value");

        let proto = edge.to_proto();
        assert_eq!(proto.from_node_id, "from");
        assert_eq!(proto.to_node_id, "to");
        assert_eq!(proto.edge_type, "REL");
        assert_eq!(proto.weight, Some(0.5));
    }

    #[test]
    fn test_graph_edge_clone() {
        let edge = GraphEdge::new("a", "b", "TYPE")
            .with_weight(0.8)
            .with_property("key", "value");

        let cloned = edge.clone();
        assert_eq!(cloned.from_node_id, edge.from_node_id);
        assert_eq!(cloned.to_node_id, edge.to_node_id);
        assert_eq!(cloned.edge_type, edge.edge_type);
        assert_eq!(cloned.weight, edge.weight);
    }

    // ========================================================================
    // SearchResult Tests
    // ========================================================================

    #[test]
    fn test_search_result_creation() {
        let mut metadata = HashMap::new();
        metadata.insert("category".to_string(), "technology".to_string());

        let result = SearchResult {
            id: "vec_123".to_string(),
            score: 0.95,
            metadata,
        };

        assert_eq!(result.id, "vec_123");
        assert!((result.score - 0.95).abs() < f32::EPSILON);
        assert_eq!(
            result.metadata.get("category"),
            Some(&"technology".to_string())
        );
    }

    #[test]
    fn test_search_result_empty_metadata() {
        let result = SearchResult {
            id: "id".to_string(),
            score: 0.5,
            metadata: HashMap::new(),
        };
        assert!(result.metadata.is_empty());
    }

    #[test]
    fn test_search_result_clone() {
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), "value".to_string());

        let result = SearchResult {
            id: "original".to_string(),
            score: 0.75,
            metadata,
        };

        let cloned = result.clone();
        assert_eq!(cloned.id, result.id);
        assert_eq!(cloned.score, result.score);
        assert_eq!(cloned.metadata, result.metadata);
    }

    // ========================================================================
    // CollectionInfo Tests
    // ========================================================================

    #[test]
    fn test_collection_info_creation() {
        let info = CollectionInfo {
            name: "my_collection".to_string(),
            dimension: 768,
            vector_count: 10000,
            engine: "sst".to_string(),
            disk_usage_bytes: 1024 * 1024 * 100,
        };

        assert_eq!(info.name, "my_collection");
        assert_eq!(info.dimension, 768);
        assert_eq!(info.vector_count, 10000);
        assert_eq!(info.engine, "sst");
        assert_eq!(info.disk_usage_bytes, 104857600);
    }

    #[test]
    fn test_collection_info_clone() {
        let info = CollectionInfo {
            name: "test".to_string(),
            dimension: 256,
            vector_count: 500,
            engine: "helix".to_string(),
            disk_usage_bytes: 1000,
        };

        let cloned = info.clone();
        assert_eq!(cloned.name, info.name);
        assert_eq!(cloned.dimension, info.dimension);
    }

    // ========================================================================
    // StorageStats Tests
    // ========================================================================

    #[test]
    fn test_storage_stats_creation() {
        let stats = StorageStats {
            total_vectors: 50000,
            total_collections: 5,
            disk_usage_bytes: 1024 * 1024 * 500,
            cache_hit_rate: 0.85,
        };

        assert_eq!(stats.total_vectors, 50000);
        assert_eq!(stats.total_collections, 5);
        assert!((stats.cache_hit_rate - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_storage_stats_empty() {
        let stats = StorageStats {
            total_vectors: 0,
            total_collections: 0,
            disk_usage_bytes: 0,
            cache_hit_rate: 0.0,
        };

        assert_eq!(stats.total_vectors, 0);
        assert_eq!(stats.total_collections, 0);
    }

    #[test]
    fn test_storage_stats_clone() {
        let stats = StorageStats {
            total_vectors: 1000,
            total_collections: 2,
            disk_usage_bytes: 5000,
            cache_hit_rate: 0.9,
        };

        let cloned = stats.clone();
        assert_eq!(cloned.total_vectors, stats.total_vectors);
        assert_eq!(cloned.cache_hit_rate, stats.cache_hit_rate);
    }

    // ========================================================================
    // GraphStats Tests
    // ========================================================================

    #[test]
    fn test_graph_stats_creation() {
        let stats = GraphStats {
            total_nodes: 1000,
            total_edges: 5000,
        };

        assert_eq!(stats.total_nodes, 1000);
        assert_eq!(stats.total_edges, 5000);
    }

    #[test]
    fn test_graph_stats_empty() {
        let stats = GraphStats {
            total_nodes: 0,
            total_edges: 0,
        };

        assert_eq!(stats.total_nodes, 0);
        assert_eq!(stats.total_edges, 0);
    }

    #[test]
    fn test_graph_stats_clone() {
        let stats = GraphStats {
            total_nodes: 100,
            total_edges: 500,
        };

        let cloned = stats.clone();
        assert_eq!(cloned.total_nodes, stats.total_nodes);
        assert_eq!(cloned.total_edges, stats.total_edges);
    }

    // ========================================================================
    // AccessMode Tests (from coordination module, but used in EmbeddedConfig)
    // ========================================================================

    #[test]
    fn test_access_mode_equality() {
        assert_eq!(AccessMode::Exclusive, AccessMode::Exclusive);
        assert_eq!(AccessMode::SharedRead, AccessMode::SharedRead);
        assert_eq!(AccessMode::LeaderFollower, AccessMode::LeaderFollower);
        assert_ne!(AccessMode::Exclusive, AccessMode::SharedRead);
    }

    #[test]
    fn test_access_mode_can_write() {
        assert!(AccessMode::Exclusive.can_write());
        assert!(!AccessMode::SharedRead.can_write());
        assert!(AccessMode::LeaderFollower.can_write());
    }

    // ========================================================================
    // CacheStatsSnapshot Tests (internal type)
    // ========================================================================

    #[test]
    fn test_cache_stats_snapshot_creation() {
        let snapshot = CacheStatsSnapshot {
            entries: 1000,
            memory_bytes: 1024 * 1024,
        };

        assert_eq!(snapshot.entries, 1000);
        assert_eq!(snapshot.memory_bytes, 1048576);
    }

    #[test]
    fn test_cache_stats_snapshot_clone() {
        let snapshot = CacheStatsSnapshot {
            entries: 500,
            memory_bytes: 2048,
        };

        let cloned = snapshot.clone();
        assert_eq!(cloned.entries, snapshot.entries);
        assert_eq!(cloned.memory_bytes, snapshot.memory_bytes);
    }

    // ========================================================================
    // Embedded Config Defaults Tests
    // ========================================================================

    #[test]
    fn test_embedded_config_defaults() {
        let config = EmbeddedConfig::default();

        // Core defaults
        assert_eq!(config.metadata_path, "./data/metadata");
        assert_eq!(config.cache_size_mb, 512);
        assert_eq!(config.default_engine, "sst");
        assert!(config.enable_wal);
        assert_eq!(config.wal_sync_mode, "batch");

        // Block pruning defaults
        assert_eq!(config.block_prune_mode, "sqrt");
        assert!((config.block_prune_ratio - 0.2).abs() < f32::EPSILON);
        assert_eq!(config.block_prune_min_keep, 1);
        assert_eq!(config.block_prune_max_keep, 0); // No cap

        // RL planner defaults
        assert!(config.enable_rl_planner);
        assert!(config.rl_policy_path.is_none());

        // Multi-process coordination defaults
        assert_eq!(config.access_mode, AccessMode::Exclusive);
        assert!(config.node_id.is_none());

        // Storage locations default
        assert_eq!(config.storage_locations.len(), 1);
        assert_eq!(config.storage_locations[0].path, "./data");
        assert_eq!(config.storage_locations[0].weight, 1);
        assert!(config.storage_locations[0].tags.is_empty());
    }

    // ========================================================================
    // Embedded Config Builder Tests
    // ========================================================================

    #[test]
    fn test_embedded_config_builder() {
        let config = EmbeddedConfig {
            storage_locations: vec![
                StorageLocationConfig::new("/nvme/data")
                    .with_weight(3)
                    .with_tag("hot"),
                StorageLocationConfig::new("/hdd/data")
                    .with_weight(1)
                    .with_tag("cold"),
            ],
            metadata_path: "/nvme/metadata".to_string(),
            cache_size_mb: 2048,
            default_engine: "viper".to_string(),
            enable_wal: false,
            wal_sync_mode: "immediate".to_string(),
            block_prune_mode: "ratio".to_string(),
            block_prune_ratio: 0.5,
            block_prune_min_keep: 2,
            block_prune_max_keep: 100,
            enable_rl_planner: false,
            rl_policy_path: Some("/custom/rl_policy.json".to_string()),
            access_mode: AccessMode::SharedRead,
            node_id: Some("node-42".to_string()),
            tenant_id: None,
        };

        assert_eq!(config.storage_locations.len(), 2);
        assert_eq!(config.storage_locations[0].weight, 3);
        assert!(
            config.storage_locations[0]
                .tags
                .contains(&"hot".to_string())
        );
        assert_eq!(config.storage_locations[1].weight, 1);
        assert!(
            config.storage_locations[1]
                .tags
                .contains(&"cold".to_string())
        );
        assert_eq!(config.metadata_path, "/nvme/metadata");
        assert_eq!(config.cache_size_mb, 2048);
        assert_eq!(config.default_engine, "viper");
        assert!(!config.enable_wal);
        assert_eq!(config.wal_sync_mode, "immediate");
        assert_eq!(config.block_prune_mode, "ratio");
        assert!((config.block_prune_ratio - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.block_prune_min_keep, 2);
        assert_eq!(config.block_prune_max_keep, 100);
        assert!(!config.enable_rl_planner);
        assert_eq!(
            config.rl_policy_path,
            Some("/custom/rl_policy.json".to_string())
        );
        assert_eq!(config.access_mode, AccessMode::SharedRead);
        assert_eq!(config.node_id, Some("node-42".to_string()));
        assert!(config.tenant_id.is_none());

        // Test chained builder methods
        let config2 = EmbeddedConfig::default()
            .with_access_mode(AccessMode::LeaderFollower)
            .with_node_id("leader-1")
            .with_tenant("tenant-a");
        assert_eq!(config2.access_mode, AccessMode::LeaderFollower);
        assert_eq!(config2.node_id, Some("leader-1".to_string()));
        assert_eq!(config2.tenant_id, Some("tenant-a".to_string()));
    }

    // ========================================================================
    // CollectionInfo Config Creation Tests
    // ========================================================================

    #[test]
    fn test_collection_config_creation() {
        let info = CollectionInfo {
            name: "embeddings".to_string(),
            dimension: 768,
            vector_count: 50000,
            engine: "sst".to_string(),
            disk_usage_bytes: 1024 * 1024 * 100, // 100 MB
        };

        assert_eq!(info.name, "embeddings");
        assert_eq!(info.dimension, 768);
        assert_eq!(info.vector_count, 50000);
        assert_eq!(info.engine, "sst");
        assert_eq!(info.disk_usage_bytes, 104857600);

        // Verify Clone works
        let cloned = info.clone();
        assert_eq!(cloned.name, info.name);
        assert_eq!(cloned.dimension, info.dimension);
        assert_eq!(cloned.vector_count, info.vector_count);
    }

    // ========================================================================
    // SearchResult Defaults Tests
    // ========================================================================

    #[test]
    fn test_search_params_defaults() {
        // Verify SearchResult construction with default-like values
        let result = SearchResult {
            id: "vec_001".to_string(),
            score: 0.0,
            metadata: HashMap::new(),
        };
        assert_eq!(result.id, "vec_001");
        assert!((result.score - 0.0).abs() < f32::EPSILON);
        assert!(result.metadata.is_empty());

        // Verify with populated metadata
        let mut meta = HashMap::new();
        meta.insert("category".to_string(), "test".to_string());
        meta.insert("source".to_string(), "unit_test".to_string());

        let result_with_meta = SearchResult {
            id: "vec_002".to_string(),
            score: 0.95,
            metadata: meta,
        };
        assert_eq!(result_with_meta.metadata.len(), 2);
        assert_eq!(
            result_with_meta.metadata.get("category"),
            Some(&"test".to_string())
        );
    }

    // ========================================================================
    // Embedded Metrics Creation Tests
    // ========================================================================

    #[test]
    fn test_embedded_metrics_creation() {
        let collector = EmbeddedMetricsCollector::new();

        // Take a snapshot with default window
        let metrics = collector.snapshot(RollingWindow::AllTime);

        // All counters should be zero
        assert_eq!(metrics.total_searches, 0);
        assert_eq!(metrics.total_inserts, 0);
        assert_eq!(metrics.total_deletes, 0);
        assert_eq!(metrics.total_flushes, 0);
        assert_eq!(metrics.total_gets, 0);
        assert_eq!(metrics.total_upserts, 0);
        assert_eq!(metrics.total_vectors_inserted, 0);
        assert_eq!(metrics.total_vectors_deleted, 0);
        assert_eq!(metrics.total_bytes_written, 0);
        assert_eq!(metrics.total_bytes_read, 0);
        assert_eq!(metrics.total_errors, 0);

        // Cache stats should be zero
        assert_eq!(metrics.cache_hits, 0);
        assert_eq!(metrics.cache_misses, 0);
        assert_eq!(metrics.cache_entries, 0);
        assert_eq!(metrics.cache_memory_bytes, 0);
        assert_eq!(metrics.cache_evictions, 0);

        // WAL stats should be zero
        assert_eq!(metrics.wal_pending_bytes, 0);
        assert_eq!(metrics.wal_segments_count, 0);

        // Latency stats should have zero count
        assert_eq!(metrics.search_latency.count, 0);
        assert_eq!(metrics.insert_latency.count, 0);
        assert_eq!(metrics.delete_latency.count, 0);

        // Record some operations and verify counters update
        collector.record_search_us(500);
        collector.record_insert_us(200, 10);
        collector.record_error();

        let updated = collector.snapshot(RollingWindow::AllTime);
        assert_eq!(updated.total_searches, 1);
        assert_eq!(updated.total_inserts, 1);
        assert_eq!(updated.total_vectors_inserted, 10);
        assert_eq!(updated.total_errors, 1);
    }

    // ========================================================================
    // Embedded Version Info Tests
    // ========================================================================

    #[test]
    fn test_embedded_version_info() {
        // The crate version is available via env! macro at compile time
        let version = env!("CARGO_PKG_VERSION");

        // Version should be non-empty and follow semver format
        assert!(!version.is_empty());

        let parts: Vec<&str> = version.split('.').collect();
        assert!(
            parts.len() >= 2,
            "Version '{}' should have at least major.minor components",
            version
        );

        // Major version should be parseable as a number
        let major: u32 = parts[0]
            .parse()
            .unwrap_or_else(|_| panic!("Major version '{}' should be numeric", parts[0]));
        // Minor version should be parseable as a number
        let _minor: u32 = parts[1]
            .parse()
            .unwrap_or_else(|_| panic!("Minor version '{}' should be numeric", parts[1]));

        // ProximaDB should be at least v0.x
        assert!(
            major < 100,
            "Major version {} seems unreasonably large",
            major
        );

        // Verify the crate name. Was asserted as "proximadb" — this test moved
        // here in #1021's extraction with the assertion unchanged, so it could
        // never pass in this crate and no CI lane runs these tests (fourth
        // #1021-stranded artifact: wheel config #1675, workflow notes,
        // loopback-guard paths, this).
        let crate_name = env!("CARGO_PKG_NAME");
        assert_eq!(crate_name, "proximadb-embedded");
    }

    // ========================================================================
    // TD-097 (Spark JNI) + TD-099 (cursor) embedded surfaces:
    //   - get_collection_schema
    //   - scan_records (cursor pagination)
    //   - plan_partitions (single-partition fallback)
    // ========================================================================

    /// Helper: spin up a fresh EmbeddedProximaDB on a tempdir. WAL is
    /// explicitly enabled so `insert_proxima_records` writes are
    /// visible to `scan_records` via the canonical WAL/memtable scan.
    fn build_test_db() -> (EmbeddedProximaDB, tempfile::TempDir) {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let mut config = EmbeddedConfig::for_low_memory(temp_dir.path().to_string_lossy().as_ref());
        config.enable_wal = true;
        let db = EmbeddedProximaDB::new(config).expect("create embedded db");
        (db, temp_dir)
    }

    /// Helper: build a minimal ProximaRecord for scan tests. Mirrors
    /// the shape used by the existing fp16/fp32 e2e tests (oid +
    /// local_id + a single dense_vector embedding cell).
    fn make_test_record(oid: &str, updated_at_ns: i64) -> proximadb_records::ProximaRecord {
        proximadb_records::ProximaRecord {
            oid: oid.to_string(),
            local_id: Some(oid.to_string()),
            updated_at_ns,
            embeddings: vec![proximadb_records::EmbeddingCell {
                model_id: "test-model".to_string(),
                modality: "dense_vector".to_string(),
                dim: 4,
                values: proximadb_records::EmbeddingValues::Fp32(vec![0.1, 0.2, 0.3, 0.4]),
                precision: proximadb_records::EmbeddingScalarType::Fp32,
                ..Default::default()
            }],
            ..proximadb_records::ProximaRecord::default()
        }
    }

    #[test]
    fn direct_embedded_record_write_publishes_content_revision() {
        let (db, _temp_dir) = build_test_db();
        db.create_collection("embedded_revision_col", 4, None)
            .expect("create collection");
        let collection = db
            .runtime
            .block_on(
                db.collection_port
                    .get_collection("embedded_revision_col", None),
            )
            .expect("resolve collection")
            .expect("collection exists");
        let before = db.runtime.block_on(
            proximadb::catalog::CorpusVersionRegistry::global()
                .current_content("default", &collection.id),
        );

        db.insert_proxima_records(
            "embedded_revision_col",
            vec![make_test_record("revision-record", 1)],
        )
        .expect("insert record");

        let after = db.runtime.block_on(
            proximadb::catalog::CorpusVersionRegistry::global()
                .current_content("default", &collection.id),
        );
        assert_eq!(after, before + 1);
        db.close();
    }

    #[test]
    fn legacy_direct_and_batch_writes_publish_content_revision() {
        let (db, _temp_dir) = build_test_db();
        db.create_collection("legacy_revision_col", 4, None)
            .expect("create collection");
        let collection = db
            .runtime
            .block_on(
                db.collection_port
                    .get_collection("legacy_revision_col", None),
            )
            .expect("resolve collection")
            .expect("collection exists");
        let registry = proximadb::catalog::CorpusVersionRegistry::global();
        let before = db
            .runtime
            .block_on(registry.current_content("default", &collection.id));

        db.runtime
            .block_on(
                db.shared_services
                    .vector_operations_service
                    .insert_vectors_direct(
                        &collection.id,
                        std::sync::Arc::new(vec![make_test_record("direct", 1)]),
                    ),
            )
            .expect("direct write");
        let after_direct = db
            .runtime
            .block_on(registry.current_content("default", &collection.id));
        assert_eq!(after_direct, before + 1);

        db.runtime
            .block_on(
                db.shared_services
                    .vector_operations_service
                    .insert_vectors_via_batch_pipeline(
                        &collection.id,
                        vec![make_test_record("batch", 2)],
                    ),
            )
            .expect("batch-pipeline write");
        let after_batch = db
            .runtime
            .block_on(registry.current_content("default", &collection.id));
        assert_eq!(after_batch, after_direct + 1);
        db.close();
    }

    #[test]
    fn closed_handle_rejects_single_vector_delete() {
        let (db, _temp_dir) = build_test_db();
        db.create_collection("closed_delete_col", 4, None)
            .expect("create collection");
        db.close();

        let error = db
            .delete_vector("closed_delete_col", "record-a")
            .expect_err("closed handle must reject every write API");
        assert!(
            error.to_string().contains("closed"),
            "write fence should identify the closed database: {error}"
        );
    }

    #[test]
    fn test_get_collection_schema_returns_none_for_missing_collection() {
        let (db, _td) = build_test_db();
        let result = db
            .get_collection_schema("does_not_exist")
            .expect("call must not error");
        assert!(result.is_none(), "missing collection ⇒ None");
    }

    #[test]
    fn test_get_collection_schema_returns_named_collection() {
        let (db, _td) = build_test_db();
        db.create_collection("test_col", 4, None)
            .expect("create_collection");
        let schema = db
            .get_collection_schema("test_col")
            .expect("schema call")
            .expect("collection must exist");
        assert_eq!(schema["name"], serde_json::json!("test_col"));
        assert_eq!(schema["dimension"], serde_json::json!(4));
        assert!(
            schema.get("storage_engine").is_some(),
            "schema must expose storage_engine: {schema}"
        );
    }

    #[test]
    fn test_get_collection_schema_surfaces_vector_count() {
        let (db, _td) = build_test_db();
        db.create_collection("counted_col", 4, None)
            .expect("create");
        let schema = db
            .get_collection_schema("counted_col")
            .expect("schema call")
            .expect("collection exists");
        // Even with zero records, vector_count must be a Number (not
        // missing), so downstream consumers can rely on its presence.
        assert!(
            schema["vector_count"].is_number(),
            "vector_count must be present + numeric: {schema}"
        );
    }

    #[test]
    fn test_scan_records_empty_collection_returns_empty_page() {
        let (db, _td) = build_test_db();
        db.create_collection("empty_col", 4, None).expect("create");
        let (records, next_cursor) = db.scan_records("empty_col", None, 100).expect("scan call");
        assert!(records.is_empty(), "empty collection ⇒ no records");
        assert!(next_cursor.is_none(), "empty collection ⇒ no next cursor");
    }

    #[test]
    fn test_scan_records_returns_inserted_records_in_stable_order() {
        let (db, _td) = build_test_db();
        db.create_collection("scan_col", 4, None).expect("create");

        // Insert 3 records OUT of (updated_at_ns, oid) order.
        let records = vec![
            make_test_record("c", 30),
            make_test_record("a", 10),
            make_test_record("b", 20),
        ];
        db.insert_proxima_records("scan_col", records)
            .expect("insert");

        let (page, next_cursor) = db.scan_records("scan_col", None, 100).expect("scan call");
        // Sort order is (updated_at_ns, oid): a(10), b(20), c(30).
        assert_eq!(page.len(), 3, "all 3 records: {page:?}");
        assert_eq!(page[0].oid, "a");
        assert_eq!(page[1].oid, "b");
        assert_eq!(page[2].oid, "c");
        assert!(next_cursor.is_none(), "short page ⇒ no next cursor");
    }

    #[test]
    fn test_scan_records_paginates_via_cursor_round_trip() {
        let (db, _td) = build_test_db();
        db.create_collection("paged_col", 4, None).expect("create");
        let records = (0..5)
            .map(|i| make_test_record(&format!("r{i:02}"), 100 + i as i64))
            .collect();
        db.insert_proxima_records("paged_col", records)
            .expect("insert");

        // First page (limit=2).
        let (page1, cursor1) = db.scan_records("paged_col", None, 2).expect("p1");
        assert_eq!(page1.len(), 2);
        let cur = cursor1.expect("full page ⇒ next cursor");

        // Second page resumes from cursor.
        let (page2, cursor2) = db.scan_records("paged_col", Some(cur), 2).expect("p2");
        assert_eq!(page2.len(), 2);
        let cur2 = cursor2.expect("still full page ⇒ next cursor");

        // Third page is short ⇒ no next cursor.
        let (page3, cursor3) = db.scan_records("paged_col", Some(cur2), 2).expect("p3");
        assert_eq!(page3.len(), 1, "5 total - 4 returned = 1 remaining");
        assert!(cursor3.is_none(), "short page ⇒ no next cursor");

        // Round-trip: union of (page1, page2, page3) covers all 5
        // records exactly once.
        let mut all_oids: Vec<String> = page1
            .iter()
            .chain(page2.iter())
            .chain(page3.iter())
            .map(|r| r.oid.clone())
            .collect();
        all_oids.sort();
        all_oids.dedup();
        assert_eq!(all_oids.len(), 5, "exactly 5 distinct oids: {all_oids:?}");
    }

    #[test]
    fn scan_cursor_rejects_collection_mutation_between_pages() {
        let (db, _td) = build_test_db();
        db.create_collection("mutable_scan_col", 4, None)
            .expect("create");
        let records = (0..3)
            .map(|i| make_test_record(&format!("r{i:02}"), 100 + i as i64))
            .collect();
        db.insert_proxima_records("mutable_scan_col", records)
            .expect("initial insert");

        let (_, cursor) = db
            .scan_records("mutable_scan_col", None, 2)
            .expect("first page");
        db.insert_proxima_records("mutable_scan_col", vec![make_test_record("r99", 999)])
            .expect("mutating insert");

        let error = db
            .scan_records("mutable_scan_col", cursor, 2)
            .expect_err("a cursor must be fenced after collection content changes");
        assert!(
            error.to_string().contains("changed"),
            "revision error should tell callers to restart: {error}"
        );
        db.close();
    }

    #[test]
    fn scan_cursor_rejects_previous_embedded_incarnation() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let config = EmbeddedConfig::for_low_memory(temp_dir.path().to_string_lossy().as_ref());
        let db = EmbeddedProximaDB::new(config.clone()).expect("first open");
        db.create_collection("restart_scan_col", 4, None)
            .expect("create");
        let records = (0..3)
            .map(|i| make_test_record(&format!("r{i:02}"), 100 + i as i64))
            .collect();
        db.insert_proxima_records("restart_scan_col", records)
            .expect("insert");
        let (_, cursor) = db
            .scan_records("restart_scan_col", None, 2)
            .expect("first page");
        db.close();

        let reopened = EmbeddedProximaDB::new(config).expect("reopen");
        let error = reopened
            .scan_records("restart_scan_col", cursor, 2)
            .expect_err("cursor from a prior embedded incarnation must be rejected");
        assert!(
            error.to_string().contains("incarnation"),
            "restart fence should be explicit: {error}"
        );
        reopened.close();
    }

    #[test]
    fn test_scan_records_rejects_stale_cursor() {
        use proximadb::services::scan_cursor::ScanCursor;
        let (db, _td) = build_test_db();
        db.create_collection("stale_col", 4, None).expect("create");
        let canonical_id = db
            .runtime
            .block_on(db.collection_port.resolve_collection_id("stale_col"))
            .expect("resolve collection")
            .expect("collection exists");

        // Forge a cursor with epoch from > 24h ago.
        let stale = ScanCursor {
            collection_id: canonical_id,
            last_updated_at_ns: 0,
            last_oid: "x".to_string(),
            epoch_ns: 0, // wall clock is now ~> 24h since this
            content_revision: None,
            content_revision_token: None,
        };
        let raw = stale.encode().unwrap();
        let err = db.scan_records("stale_col", Some(raw), 10).unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("expired"),
            "stale cursor error must say 'expired': {err}"
        );
    }

    #[test]
    fn test_scan_records_excludes_ttl_expired_full_path() {
        // TD-099(3d): a TTL-expired record must be dropped end-to-end through the
        // push-down path (embedded → service → WAL scan index), which proves the
        // read-time `now_ns` is threaded correctly.
        let (db, _td) = build_test_db();
        db.create_collection("ttl_scan_col", 4, None)
            .expect("create");
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);

        let live = make_test_record("live", now_ns - 1_000);
        let mut dead = make_test_record("dead", now_ns - 1_000);
        dead.valid_to_ns = Some(now_ns - 1); // expired in the past

        db.insert_proxima_records("ttl_scan_col", vec![live, dead])
            .expect("insert");

        let (page, _next) = db.scan_records("ttl_scan_col", None, 100).expect("scan");
        let got: Vec<&str> = page.iter().map(|r| r.oid.as_str()).collect();
        assert_eq!(
            got,
            vec!["live"],
            "expired record must be excluded: {page:?}"
        );
    }

    #[test]
    fn test_plan_partitions_returns_single_partition_for_existing_collection() {
        let (db, _td) = build_test_db();
        db.create_collection("part_col", 4, None).expect("create");
        let partitions = db.plan_partitions("part_col", 8).expect("plan");
        // Single-partition fallback regardless of requested count.
        assert_eq!(partitions.len(), 1, "fallback: 1 partition: {partitions:?}");
        assert_eq!(partitions[0].partition_id, 0);
        assert_eq!(partitions[0].splits.len(), 1);
        assert_eq!(
            partitions[0].splits[0].file_path, "collection://part_col",
            "split must carry the logical collection URI"
        );
    }

    #[test]
    fn test_plan_partitions_errors_on_missing_collection() {
        let (db, _td) = build_test_db();
        let err = db
            .plan_partitions("does_not_exist", 1)
            .expect_err("missing collection must error");
        assert!(
            err.to_string().contains("does_not_exist"),
            "error must name the missing collection: {err}"
        );
    }

    // ========================================================================
    // TD-DSEFF-2: embedded migration-risk disclosure.
    //
    // Pre-#1743, every embedded collection's flushes were routed through SST
    // regardless of declared engine. `engine_data_mismatch` is the pure
    // decision logic behind the one-shot, log-only warning `EmbeddedProximaDB
    // ::new` emits for a non-SST-declared collection whose data directory
    // holds only legacy `.sst` files.
    // ========================================================================

    #[test]
    fn engine_data_mismatch_detects_sst_only_directory() {
        let files = vec!["part-0.sst".to_string(), "part-1.sst".to_string()];
        assert!(
            engine_data_mismatch(&files),
            "a non-empty directory containing ONLY .sst files must be flagged"
        );
    }

    #[test]
    fn engine_data_mismatch_ignores_empty_directory() {
        // No flushes yet — nothing to migrate, not a mismatch.
        assert!(!engine_data_mismatch(&[]));
    }

    #[test]
    fn engine_data_mismatch_ignores_directory_with_declared_engine_files() {
        // The declared (non-SST) engine's own files are present — no mismatch,
        // regardless of whether some .sst leftovers also exist.
        let files = vec!["part-0.helix".to_string()];
        assert!(!engine_data_mismatch(&files));

        let mixed = vec!["part-0.sst".to_string(), "part-1.helix".to_string()];
        assert!(
            !engine_data_mismatch(&mixed),
            "even one non-.sst file means the declared engine has real data — not a mismatch"
        );
    }

    /// End-to-end: a genuinely mismatched collection (declared HELIX, data
    /// directory containing only a planted `.sst` file — simulating
    /// pre-#1743 residue) must not prevent a fresh `EmbeddedProximaDB::new`
    /// from opening the same directory. This exercises the real scan path
    /// (catalog lookup, engine-type resolution, directory listing) against
    /// live data, not just the pure `engine_data_mismatch` decision — the
    /// crate has no tracing-capture test convention to assert on the
    /// warning's text itself (see TD-DSEFF-2), so this asserts the
    /// best-effort diagnostic never blocks startup.
    #[test]
    fn reopen_does_not_fail_on_mismatched_pre_fix_collection_data() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let config = EmbeddedConfig::for_low_memory(temp_dir.path().to_string_lossy().as_ref());

        {
            let db = EmbeddedProximaDB::new(config.clone()).expect("first open");
            db.create_collection("mismatch_col", 4, Some("helix"))
                .expect("create helix collection");
            db.insert(
                "mismatch_col",
                vec!["v1".to_string()],
                vec![vec![0.1, 0.2, 0.3, 0.4]],
                None,
            )
            .expect("insert");
            db.flush().expect("flush writes the real .helix file(s)");

            // Find the real on-disk data directory the flush just wrote to
            // (walk the tempdir rather than re-deriving the path formula —
            // this test should stay correct even if that formula changes),
            // then replace its contents with a planted .sst file to
            // simulate the pre-#1743 flush-routing bug's residue.
            let mut data_dir = None;
            for entry in walkdir_files(temp_dir.path()) {
                if entry.extension().and_then(|e| e.to_str()) == Some("helix") {
                    data_dir = entry.parent().map(|p| p.to_path_buf());
                    break;
                }
            }
            let data_dir = data_dir.expect("flush must have written at least one .helix file");
            for entry in std::fs::read_dir(&data_dir).unwrap() {
                let entry = entry.unwrap();
                if entry.path().extension().and_then(|e| e.to_str()) == Some("helix") {
                    std::fs::remove_file(entry.path()).unwrap();
                }
            }
            std::fs::write(data_dir.join("legacy.sst"), b"pre-#1743 residue").unwrap();

            db.close();
        }

        // Reopening must succeed — the migration-risk scan is best-effort
        // and log-only, never a hard failure.
        let reopened = EmbeddedProximaDB::new(config);
        assert!(
            reopened.is_ok(),
            "reopening a directory with mismatched pre-fix data must not fail: {:?}",
            reopened.err()
        );
    }

    /// Recursively list every file (not directory) under `root`.
    fn walkdir_files(root: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut out = Vec::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else {
                    out.push(path);
                }
            }
        }
        out
    }

    // ========================================================================
    // ids/vectors count validation (redundancy pass 3)
    // ========================================================================

    #[test]
    fn test_insert_rejects_ids_vectors_count_mismatch() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let config = EmbeddedConfig::for_low_memory(temp_dir.path().to_string_lossy().as_ref());
        let db = EmbeddedProximaDB::new(config).expect("create embedded db");

        db.execute_sql(
            "CREATE TABLE IF NOT EXISTS \"insert_mismatch_t\" (
                \"record_id\" TEXT NOT NULL,
                \"embedding\" VECTOR(4),
                PRIMARY KEY (\"record_id\")
            ) WITH (storage_engine = 'SST');",
            None,
            None,
        )
        .expect("create table");

        // 2 ids, 1 vector: must be rejected, not silently zip-truncated to 1
        let result = db.insert(
            "insert_mismatch_t",
            vec!["a".to_string(), "b".to_string()],
            vec![vec![0.0, 0.1, 0.2, 0.3]],
            None,
        );

        let err = result.expect_err("mismatched ids/vectors must fail, not truncate");
        assert!(
            err.to_string()
                .contains("IDs count (2) must match vectors count (1)"),
            "unexpected error: {err}"
        );
        db.close();
    }
}
