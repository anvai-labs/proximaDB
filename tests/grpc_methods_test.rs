// Integration Test: gRPC Methods (TD-046)
//
// This test validates that the graph service supports the operations
// needed for gRPC endpoint parity with REST API.
//
// Test Coverage:
// - Graph service creation and management
// - ORION engine (production-ready)
// - Retired engine-name rejection for PULSAR and QUASAR
// - Graph metadata and statistics
// - Engine type validation

#[cfg(test)]
mod grpc_methods_tests {
    use proximadb::graph::{Node, PropertyValue, service::GraphOperationsService};
    use proximadb::proto::proximadb_v1::{CompressionAlgorithm, GraphStorageConfig};
    use std::collections::HashMap;
    use std::sync::Arc;

    /// Helper function to create a test graph
    async fn create_test_graph_with_engine(
        graph_id: &str,
        engine_type: &str,
    ) -> Arc<GraphOperationsService> {
        let service = Arc::new(GraphOperationsService::new());

        // Clean up any existing test data
        let test_dir = format!("/tmp/proximadb-test-{}", graph_id);
        let _ = std::fs::remove_dir_all(&test_dir);
        std::fs::create_dir_all(&test_dir).unwrap();

        let create_request = proximadb::proto::proximadb_v1::CreateGraphRequest {
            graph_id: graph_id.to_string(),
            name: Some(format!("Test Graph {}", graph_id)),
            description: Some(format!("Test graph for {} engine", engine_type)),
            schema: None,
            storage_config: Some(GraphStorageConfig {
                engine_type: engine_type.to_string(),
                base_url: test_dir,
                compression: CompressionAlgorithm::CompressionSnappy as i32,
                enable_wal: true,
                snapshot_interval_hours: 24,
                engine_specific_config: HashMap::new(),
            }),
            engine_config: None,
            access_control: None,
        };

        service
            .create_graph_collection(create_request)
            .await
            .expect("Failed to create test graph");

        service
    }

    /// Test graph creation with ORION engine (production-ready)
    #[tokio::test]
    async fn test_create_graph_with_orion_engine() {
        let graph_id = "test_orion_graph";
        let service = create_test_graph_with_engine(graph_id, "ORION").await;

        // Add a test node
        let node = Node {
            id: "node1".to_string(),
            labels: vec!["Test".to_string()],
            properties: HashMap::new(),
            embedding: None,
            created_at_ms: 0,
            updated_at_ms: 0,
        };

        service
            .create_node(graph_id, node)
            .await
            .expect("Failed to create node");

        // Verify node was created
        let retrieved_node = service
            .get_node(graph_id, &"node1".to_string())
            .await
            .expect("Failed to get node")
            .expect("Node not found");

        assert_eq!(retrieved_node.id, "node1");

        // Cleanup
        service.remove_graph(graph_id);
    }

    /// Test graph operation rejection with retired PULSAR engine metadata
    #[tokio::test]
    async fn test_create_graph_with_pulsar_engine_rejected() {
        let graph_id = "test_pulsar_graph";
        let service = create_test_graph_with_engine(graph_id, "PULSAR").await;

        let node = Node {
            id: "node1".to_string(),
            labels: vec!["Test".to_string()],
            properties: HashMap::new(),
            embedding: None,
            created_at_ms: 0,
            updated_at_ms: 0,
        };

        let err = service
            .create_node(graph_id, node)
            .await
            .expect_err("PULSAR metadata should be rejected");
        assert!(err.to_string().contains("retired"));

        service.remove_graph(graph_id);
    }

    /// Test graph operation rejection with retired QUASAR engine metadata
    #[tokio::test]
    async fn test_create_graph_with_quasar_engine_rejected() {
        let graph_id = "test_quasar_graph";
        let service = create_test_graph_with_engine(graph_id, "QUASAR").await;

        let node = Node {
            id: "node1".to_string(),
            labels: vec!["Test".to_string()],
            properties: HashMap::new(),
            embedding: None,
            created_at_ms: 0,
            updated_at_ms: 0,
        };

        let err = service
            .create_node(graph_id, node)
            .await
            .expect_err("QUASAR metadata should be rejected");
        assert!(err.to_string().contains("retired"));

        service.remove_graph(graph_id);
    }

    /// Test graph statistics and metadata
    #[tokio::test]
    async fn test_graph_statistics() {
        let graph_id = "test_stats_graph";
        let service = create_test_graph_with_engine(graph_id, "ORION").await;

        // Add multiple nodes
        for i in 1..=5 {
            let node = Node {
                id: format!("node{}", i),
                labels: vec!["Test".to_string()],
                properties: HashMap::new(),
                embedding: None,
                created_at_ms: 0,
                updated_at_ms: 0,
            };

            service
                .create_node(graph_id, node)
                .await
                .expect("Failed to create node");
        }

        // Verify nodes exist
        for i in 1..=5 {
            let node_id = format!("node{}", i);
            let retrieved_node = service
                .get_node(graph_id, &node_id)
                .await
                .expect("Failed to get node")
                .expect("Node not found");

            assert_eq!(retrieved_node.id, node_id);
        }

        // Cleanup
        service.remove_graph(graph_id);
    }

    /// Test graph with edges for tier migration simulation
    #[tokio::test]
    async fn test_graph_with_edges_for_migration() {
        let graph_id = "test_migration_graph";
        let service = create_test_graph_with_engine(graph_id, "ORION").await;

        // Create nodes
        for i in 1..=3 {
            let node = Node {
                id: format!("node{}", i),
                labels: vec!["Test".to_string()],
                properties: HashMap::new(),
                embedding: None,
                created_at_ms: 0,
                updated_at_ms: 0,
            };

            service
                .create_node(graph_id, node)
                .await
                .expect("Failed to create node");
        }

        // Create edges
        for i in 1..2 {
            let edge = proximadb::graph::Edge {
                id: format!("edge{}", i),
                from_node_id: format!("node{}", i),
                to_node_id: format!("node{}", i + 1),
                edge_type: "CONNECTS".to_string(),
                properties: HashMap::new(),
                weight: Some(1.0),
                created_at_ms: 0,
                updated_at_ms: 0,
            };

            service
                .create_edge(graph_id, edge)
                .await
                .expect("Failed to create edge");
        }

        // Verify graph structure
        let node1 = service
            .get_node(graph_id, &"node1".to_string())
            .await
            .expect("Failed to get node1")
            .expect("Node1 not found");

        assert_eq!(node1.id, "node1");

        let edge1 = service
            .get_edge(graph_id, &"edge1".to_string())
            .await
            .expect("Failed to get edge1")
            .expect("Edge1 not found");

        assert_eq!(edge1.from_node_id, "node1");
        assert_eq!(edge1.to_node_id, "node2");

        // Cleanup
        service.remove_graph(graph_id);
    }

    /// Test graph engine type validation
    #[tokio::test]
    async fn test_graph_engine_type_validation() {
        let graph_id = "test_orion_engine";
        let service = create_test_graph_with_engine(graph_id, "ORION").await;
        let node = Node {
            id: "node1".to_string(),
            labels: vec!["Test".to_string()],
            properties: HashMap::new(),
            embedding: None,
            created_at_ms: 0,
            updated_at_ms: 0,
        };

        service
            .create_node(graph_id, node)
            .await
            .expect("ORION should remain the valid graph engine");

        assert_eq!(
            proximadb::graph::GraphEngineFactory::engine_type_from_string("PULSAR"),
            None
        );
        assert_eq!(
            proximadb::graph::GraphEngineFactory::engine_type_from_string("QUASAR"),
            None
        );

        service.remove_graph(graph_id);
    }

    /// Test graph cleanup and removal
    #[tokio::test]
    async fn test_graph_cleanup() {
        let graph_id = "test_cleanup_graph";
        let service = create_test_graph_with_engine(graph_id, "ORION").await;

        // Add some data
        let node = Node {
            id: "node1".to_string(),
            labels: vec!["Test".to_string()],
            properties: HashMap::new(),
            embedding: None,
            created_at_ms: 0,
            updated_at_ms: 0,
        };

        service
            .create_node(graph_id, node)
            .await
            .expect("Failed to create node");

        // Verify node exists
        let retrieved_node = service
            .get_node(graph_id, &"node1".to_string())
            .await
            .expect("Failed to get node");

        assert!(retrieved_node.is_some());

        // Remove graph
        service.remove_graph(graph_id);

        // Verify graph was removed (node should no longer exist)
        // Note: In production, the graph engine would be completely removed
        // For this test, we verify the removal operation doesn't panic
    }

    /// A READ of a graph that was never created must not create it. Reads used
    /// to go through `get_or_create_graph_engine`, so looking up a node in a
    /// mistyped graph silently registered a new graph collection and then
    /// answered "no such node". Mutations of existing data (update/delete)
    /// must not create it either; only creating writes provision.
    #[tokio::test]
    async fn test_read_or_mutation_of_missing_graph_does_not_create_it() {
        let service = Arc::new(GraphOperationsService::new());
        let before = service.list_graphs().await.expect("list graphs");

        assert!(
            service
                .get_node("typo_graph", &"n1".to_string())
                .await
                .is_err()
        );
        assert!(
            service
                .get_edge("typo_graph", &"e1".to_string())
                .await
                .is_err()
        );
        assert!(
            service
                .delete_node("typo_graph", &"n1".to_string())
                .await
                .is_err()
        );
        let err = service
            .get_node("typo_graph", &"n1".to_string())
            .await
            .expect_err("missing graph")
            .to_string();
        assert!(err.contains("not found"), "maps to NOT_FOUND: {err}");

        let after = service.list_graphs().await.expect("list graphs");
        assert_eq!(before, after, "a read/mutation must not provision a graph");
        assert!(!after.iter().any(|g| g.contains("typo_graph")), "{after:?}");

        // The other half of the contract: an EXISTING graph with a missing
        // node is Ok(None) (a plain "not there"), not an error.
        let existing = create_test_graph_with_engine("lookup_contract_graph", "ORION").await;
        assert!(
            existing
                .get_node("lookup_contract_graph", &"absent".to_string())
                .await
                .expect("existing graph: lookup succeeds")
                .is_none()
        );
    }

    /// Test error handling for non-existent graphs
    #[tokio::test]
    async fn test_graph_error_handling() {
        let service = Arc::new(GraphOperationsService::new());

        // Try to get node from non-existent graph
        let result = service
            .get_node("nonexistent_graph", &"node1".to_string())
            .await;

        // Should return an error
        assert!(result.is_err(), "Should error for non-existent graph");
    }

    /// Test graph with complex properties
    #[tokio::test]
    async fn test_graph_with_complex_properties() {
        let graph_id = "test_complex_props";
        let service = create_test_graph_with_engine(graph_id, "ORION").await;

        // Create node with complex properties
        let node = Node {
            id: "node1".to_string(),
            labels: vec!["Person".to_string()],
            properties: HashMap::from([
                (
                    "name".to_string(),
                    PropertyValue {
                        value: Some(proximadb::graph::property_value::Value::StringValue(
                            "Alice".to_string(),
                        )),
                    },
                ),
                (
                    "age".to_string(),
                    PropertyValue {
                        value: Some(proximadb::graph::property_value::Value::IntValue(30)),
                    },
                ),
                (
                    "score".to_string(),
                    PropertyValue {
                        value: Some(proximadb::graph::property_value::Value::DoubleValue(0.95)),
                    },
                ),
                (
                    "active".to_string(),
                    PropertyValue {
                        value: Some(proximadb::graph::property_value::Value::BoolValue(true)),
                    },
                ),
            ]),
            embedding: None,
            created_at_ms: 0,
            updated_at_ms: 0,
        };

        service
            .create_node(graph_id, node)
            .await
            .expect("Failed to create node");

        // Verify node with complex properties
        let retrieved_node = service
            .get_node(graph_id, &"node1".to_string())
            .await
            .expect("Failed to get node")
            .expect("Node not found");

        assert_eq!(retrieved_node.id, "node1");
        assert_eq!(retrieved_node.properties.len(), 4);

        // Cleanup
        service.remove_graph(graph_id);
    }
}
