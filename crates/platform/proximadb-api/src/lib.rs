//! # ProximaDB API Handlers
//!
//! This crate contains API request/response handling for ProximaDB.
//!
//! ## Protocol Support
//!
//! - **REST** - HTTP/JSON API via Axum
//! - **gRPC** - Protocol Buffers API via Tonic
//! - **PostgreSQL Wire** - pgvector-compatible protocol
//! - **Arrow Flight** - Columnar data exchange
//!
//! ## Architecture
//!
//! The API layer provides protocol adapters that:
//! - Validate and map incoming requests
//! - Route to appropriate service calls
//! - Shape responses for protocol compatibility
//! - Handle authentication and middleware
//!
//! ## Dependencies
//!
//! - `proximadb-query` - Query execution contracts
//! - `proximadb-proto` - Protocol buffer types
//! - `proximadb-kernel` - Core error types

pub mod arrow_flight;
pub mod grpc;
pub mod middleware;
pub mod pgwire;
pub mod rest;

// Re-export Arrow Flight and pgwire types
pub use arrow_flight::ProximaFlightService;
pub use pgwire::{PostgresServer, PostgresSession};

// Re-export common types
pub use grpc::{GrpcApiHandler, GrpcRequest, GrpcResponse};
pub use rest::{RestApiHandler, RestRequest, RestResponse};

pub use rest::canonical::{
    DocumentHandler, DocumentQueryHandler, EntityHandler, GraphHandler, GraphTraversalHandler,
    HybridSearchHandler, LogsHandler, MetricsHandler, ProgressiveSearchHandler, VectorHandler,
};

// Re-export v2 agentic contracts
pub use grpc::v2::{
    AgentCheckpointService, AgentEventService, AgentMemoryService, AgenticGrpcBackend,
    CheckpointServiceRequest, CheckpointServiceResponse, EventAppendServiceRequest,
    EventAppendServiceResponse, MemoryPutServiceRequest, MemoryPutServiceResponse,
    MemorySearchServiceRequest, MemorySearchServiceResponse,
};
pub use rest::v2::{
    AgentCheckpointRequest, AgentCheckpointResponse, AgentEventAppendRequest,
    AgentEventAppendResponse, AgentEventRecord, AgentEventReplayRequest, AgentMemoryItem,
    AgentMemoryPutRequest, AgentMemorySearchRequest, AgentMemorySearchResponse, AgenticApiError,
    AgenticErrorBody, AgenticRestBackend,
};

// Re-export middleware
pub use middleware::{
    AuthMiddleware, CorsMiddleware, RateLimitMiddleware, RequestIdMiddleware, auth::AuthConfig,
    cors::CorsConfig, rate_limit::RateLimitConfig,
};

#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::{Arc, Mutex};

    use anyhow::Result;
    use async_trait::async_trait;
    use proximadb_data_model::ProximaValue;
    use proximadb_proto::v1::{
        Collection, CollectionConfig, CollectionRequest, CollectionResponse, ExecuteQueryResponse,
        HybridSearchRequest, HybridSearchResponse, VectorBatchRequest, VectorOperationResponse,
        VectorSearchRequest,
    };
    use proximadb_runtime::{
        ApiHandlersPort, CollectionPort, CollectionSchemaMetadata, CollectionSchemaUpdate,
        PortIdentity, QueryAdapterPort, SqlExecutionResult, UnifiedHandlers, VectorOpsPort,
    };
    use serde_json::Value as JsonValue;

    #[derive(Debug, Clone, PartialEq)]
    pub(crate) enum ApiCall {
        Collection {
            operation: i32,
            tenant_id: Option<String>,
            collection_id: Option<String>,
        },
        SchemaGet {
            tenant_id: Option<String>,
            collection_id: String,
        },
        SchemaUpdate {
            tenant_id: Option<String>,
            collection_id: String,
            schema_id: String,
        },
        VectorSearch {
            tenant_id: Option<String>,
            collection_id: String,
            tenant_aware: bool,
        },
        VectorBatch {
            tenant_id: Option<String>,
            collection_id: String,
            vector_count: usize,
        },
        VectorGet {
            tenant_id: Option<String>,
            collection_id: String,
            vector_id: String,
            include_vector: bool,
            include_metadata: bool,
        },
        Sql {
            query: String,
            parameter_count: Option<usize>,
            collection: Option<String>,
            tenant_id: Option<String>,
            subject: Option<String>,
            tenant_stable_id: Option<u64>,
            auth_class: proximadb_tenant::AuthClass,
        },
        Hybrid,
    }

    pub(crate) struct RecordingApiPort {
        calls: Mutex<Vec<ApiCall>>,
        pub(crate) collection_response: Mutex<CollectionResponse>,
        pub(crate) schema_metadata_response: Mutex<Option<CollectionSchemaMetadata>>,
        pub(crate) vector_response: Mutex<VectorOperationResponse>,
        pub(crate) sql_response: Mutex<ExecuteQueryResponse>,
    }

    impl RecordingApiPort {
        pub(crate) fn new() -> Arc<Self> {
            Arc::new(Self {
                calls: Mutex::new(Vec::new()),
                collection_response: Mutex::new(CollectionResponse::default()),
                schema_metadata_response: Mutex::new(None),
                vector_response: Mutex::new(VectorOperationResponse::default()),
                sql_response: Mutex::new(ExecuteQueryResponse::default()),
            })
        }

        pub(crate) fn calls(&self) -> Vec<ApiCall> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl ApiHandlersPort for RecordingApiPort {
        async fn handle_collection_operation_for_tenant(
            &self,
            request: CollectionRequest,
            tenant_id: Option<&str>,
        ) -> Result<CollectionResponse> {
            self.calls.lock().unwrap().push(ApiCall::Collection {
                operation: request.operation,
                tenant_id: tenant_id.map(ToOwned::to_owned),
                collection_id: request.collection_id.clone(),
            });
            Ok(self.collection_response.lock().unwrap().clone())
        }

        async fn get_collection_schema_metadata(
            &self,
            collection_id: &str,
            tenant_id: Option<&str>,
        ) -> Result<Option<CollectionSchemaMetadata>> {
            self.calls.lock().unwrap().push(ApiCall::SchemaGet {
                tenant_id: tenant_id.map(ToOwned::to_owned),
                collection_id: collection_id.to_string(),
            });
            Ok(self.schema_metadata_response.lock().unwrap().clone())
        }

        async fn update_collection_schema_metadata(
            &self,
            collection_id: &str,
            update: CollectionSchemaUpdate,
            tenant_id: Option<&str>,
        ) -> Result<CollectionSchemaMetadata> {
            self.calls.lock().unwrap().push(ApiCall::SchemaUpdate {
                tenant_id: tenant_id.map(ToOwned::to_owned),
                collection_id: collection_id.to_string(),
                schema_id: update.schema_id.clone(),
            });
            let metadata = CollectionSchemaMetadata {
                collection_id: collection_id.to_string(),
                schema_id: Some(update.schema_id),
                schema_version: Some(update.schema_version),
                enforcement: Some(update.enforcement),
                auto_evolve: update.auto_evolve,
                enabled: true,
                columns: update.columns,
                ..CollectionSchemaMetadata::default()
            };
            *self.schema_metadata_response.lock().unwrap() = Some(metadata.clone());
            Ok(metadata)
        }

        async fn handle_vector_search_v1_for_tenant(
            &self,
            request: VectorSearchRequest,
            identity: PortIdentity<'_>,
        ) -> Result<VectorOperationResponse> {
            let tenant_id = identity.tenant_id;
            self.calls.lock().unwrap().push(ApiCall::VectorSearch {
                tenant_id: tenant_id.map(ToOwned::to_owned),
                collection_id: request.collection_id,
                tenant_aware: true,
            });
            Ok(self.vector_response.lock().unwrap().clone())
        }

        async fn handle_vector_search_v1(
            &self,
            request: VectorSearchRequest,
        ) -> Result<VectorOperationResponse> {
            self.calls.lock().unwrap().push(ApiCall::VectorSearch {
                tenant_id: None,
                collection_id: request.collection_id,
                tenant_aware: false,
            });
            Ok(self.vector_response.lock().unwrap().clone())
        }

        async fn handle_vector_batch_v1_for_tenant(
            &self,
            request: VectorBatchRequest,
            tenant_id: Option<&str>,
        ) -> Result<VectorOperationResponse> {
            self.calls.lock().unwrap().push(ApiCall::VectorBatch {
                tenant_id: tenant_id.map(ToOwned::to_owned),
                collection_id: request.collection_id,
                vector_count: request.vectors.len(),
            });
            Ok(self.vector_response.lock().unwrap().clone())
        }

        async fn handle_vector_v1_for_tenant(
            &self,
            collection_id: &str,
            vector_id: &str,
            include_vector: bool,
            include_metadata: bool,
            tenant_id: Option<&str>,
        ) -> Result<VectorOperationResponse> {
            self.calls.lock().unwrap().push(ApiCall::VectorGet {
                tenant_id: tenant_id.map(ToOwned::to_owned),
                collection_id: collection_id.to_string(),
                vector_id: vector_id.to_string(),
                include_vector,
                include_metadata,
            });
            Ok(self.vector_response.lock().unwrap().clone())
        }

        async fn execute_hybrid_query(
            &self,
            _request: HybridSearchRequest,
        ) -> Result<HybridSearchResponse> {
            self.calls.lock().unwrap().push(ApiCall::Hybrid);
            Ok(HybridSearchResponse::default())
        }

        async fn execute_sql_v1(
            &self,
            query: String,
            parameters: Option<Vec<ProximaValue>>,
            collection: Option<String>,
            identity: proximadb_runtime::PortIdentity<'_>,
        ) -> Result<ExecuteQueryResponse> {
            self.calls.lock().unwrap().push(ApiCall::Sql {
                query,
                parameter_count: parameters.as_ref().map(Vec::len),
                collection,
                tenant_id: identity.tenant_id.map(ToOwned::to_owned),
                subject: identity.subject.map(ToOwned::to_owned),
                tenant_stable_id: identity.tenant_stable_id,
                auth_class: identity.auth_class,
            });
            Ok(self.sql_response.lock().unwrap().clone())
        }

        async fn execute_sql(
            &self,
            query: String,
            parameters: Option<Vec<ProximaValue>>,
            collection: Option<String>,
            identity: proximadb_runtime::PortIdentity<'_>,
        ) -> Result<SqlExecutionResult> {
            self.calls.lock().unwrap().push(ApiCall::Sql {
                query,
                parameter_count: parameters.as_ref().map(Vec::len),
                collection,
                tenant_id: identity.tenant_id.map(ToOwned::to_owned),
                subject: identity.subject.map(ToOwned::to_owned),
                tenant_stable_id: identity.tenant_stable_id,
                auth_class: identity.auth_class,
            });
            let response = self.sql_response.lock().unwrap().clone();
            Ok(SqlExecutionResult {
                columns: response.columns,
                column_types: response.column_types,
                rows: response
                    .rows
                    .iter()
                    .map(|row| {
                        row.fields
                            .iter()
                            .map(|field| {
                                field.value.as_ref().map_or(ProximaValue::Null, |value| {
                                    proximadb_records::conversions::sql_value_to_proxima(value)
                                })
                            })
                            .collect()
                    })
                    .collect(),
                rows_scanned: response.rows_scanned,
                execution_time_ms: response.execution_time_ms,
                ..Default::default()
            })
        }
    }

    struct NoopCollectionPort;

    #[async_trait]
    impl CollectionPort for NoopCollectionPort {
        async fn get_collection(
            &self,
            _identifier: &str,
            _tenant_id: Option<&str>,
        ) -> Result<Option<Collection>> {
            Ok(None)
        }

        async fn create_collection(
            &self,
            config: CollectionConfig,
            _tenant_id: Option<&str>,
        ) -> Result<Collection> {
            Ok(Collection {
                id: config.name.clone(),
                config: Some(config),
                ..Collection::default()
            })
        }

        async fn update_collection(
            &self,
            id: &str,
            config: CollectionConfig,
            _tenant_id: Option<&str>,
        ) -> Result<Collection> {
            Ok(Collection {
                id: id.to_string(),
                config: Some(config),
                ..Collection::default()
            })
        }

        async fn delete_collection(&self, _id: &str, _tenant_id: Option<&str>) -> Result<bool> {
            Ok(true)
        }

        async fn list_collections(&self, _tenant_id: Option<&str>) -> Result<Vec<Collection>> {
            Ok(Vec::new())
        }

        async fn resolve_collection_id(&self, identifier: &str) -> Result<Option<String>> {
            Ok(Some(identifier.to_string()))
        }
    }

    struct NoopVectorOpsPort;

    #[async_trait]
    impl VectorOpsPort for NoopVectorOpsPort {
        async fn search(
            &self,
            _request: VectorSearchRequest,
            _identity: PortIdentity<'_>,
        ) -> Result<VectorOperationResponse> {
            Ok(VectorOperationResponse::default())
        }

        async fn batch_upsert(
            &self,
            _request: VectorBatchRequest,
            _tenant_id: Option<&str>,
        ) -> Result<VectorOperationResponse> {
            Ok(VectorOperationResponse::default())
        }

        async fn get_vector(
            &self,
            _collection_id: &str,
            _vector_id: &str,
            _include_vector: bool,
            _include_metadata: bool,
            _tenant_id: Option<&str>,
        ) -> Result<VectorOperationResponse> {
            Ok(VectorOperationResponse::default())
        }

        async fn flush_all(&self) -> Result<()> {
            Ok(())
        }

        async fn metrics(&self) -> Result<JsonValue> {
            Ok(JsonValue::Object(Default::default()))
        }
    }

    struct NoopQueryAdapterPort;

    #[async_trait]
    impl QueryAdapterPort for NoopQueryAdapterPort {
        async fn vector_search(
            &self,
            _request: VectorSearchRequest,
        ) -> Result<VectorOperationResponse> {
            Ok(VectorOperationResponse::default())
        }

        async fn execute_hybrid(
            &self,
            _request: HybridSearchRequest,
        ) -> Result<HybridSearchResponse> {
            Ok(HybridSearchResponse::default())
        }

        async fn execute_sql(
            &self,
            _query: String,
            _collection: Option<String>,
            _identity: PortIdentity<'_>,
        ) -> Result<SqlExecutionResult> {
            Ok(SqlExecutionResult::default())
        }
    }

    pub(crate) fn noop_unified_handlers() -> Arc<UnifiedHandlers> {
        Arc::new(UnifiedHandlers::new(
            Arc::new(NoopCollectionPort),
            Arc::new(NoopVectorOpsPort),
            Some(Arc::new(NoopQueryAdapterPort)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use proximadb_proto::v1::{
        CollectionConfig, CollectionOperation, CollectionRequest, HybridSearchRequest,
        VectorBatchRequest, VectorRecord, VectorSearchRequest,
    };
    use proximadb_runtime::{ApiHandlersPort, PortIdentity};

    use super::test_support::{ApiCall, RecordingApiPort, noop_unified_handlers};

    #[test]
    fn test_api_module_imports() {
        // Basic test to verify the module structure is working
        // More comprehensive tests will be added as modules are extracted
    }

    #[tokio::test]
    // Deliberately exercises the deprecated v1 ExecuteHybridQuery dispatch (TD-143).
    #[allow(deprecated)]
    async fn recording_api_port_captures_all_protocol_dispatch_shapes() {
        let port = RecordingApiPort::new();

        port.handle_collection_operation_for_tenant(
            CollectionRequest {
                operation: CollectionOperation::CollectionCreate as i32,
                collection_id: Some("docs".to_string()),
                ..CollectionRequest::default()
            },
            Some("tenant-a"),
        )
        .await
        .unwrap();
        port.handle_vector_search_v1(VectorSearchRequest {
            collection_id: "global_docs".to_string(),
            ..VectorSearchRequest::default()
        })
        .await
        .unwrap();
        port.handle_vector_search_v1_for_tenant(
            VectorSearchRequest {
                collection_id: "tenant_docs".to_string(),
                ..VectorSearchRequest::default()
            },
            PortIdentity::for_tenant("tenant-a"),
        )
        .await
        .unwrap();
        port.handle_vector_batch_v1_for_tenant(
            VectorBatchRequest {
                collection_id: "docs".to_string(),
                vectors: vec![VectorRecord {
                    id: "vec-1".to_string(),
                    vector: vec![0.1, 0.2],
                    ..VectorRecord::default()
                }],
            },
            Some("tenant-a"),
        )
        .await
        .unwrap();
        port.handle_vector_v1_for_tenant("docs", "vec-1", true, false, Some("tenant-a"))
            .await
            .unwrap();
        port.execute_hybrid_query(HybridSearchRequest::default())
            .await
            .unwrap();
        port.execute_sql_v1(
            "select * from docs".to_string(),
            None,
            Some("docs".to_string()),
            proximadb_runtime::PortIdentity::anonymous(),
        )
        .await
        .unwrap();

        assert_eq!(
            port.calls(),
            vec![
                ApiCall::Collection {
                    operation: CollectionOperation::CollectionCreate as i32,
                    tenant_id: Some("tenant-a".to_string()),
                    collection_id: Some("docs".to_string()),
                },
                ApiCall::VectorSearch {
                    tenant_id: None,
                    collection_id: "global_docs".to_string(),
                    tenant_aware: false,
                },
                ApiCall::VectorSearch {
                    tenant_id: Some("tenant-a".to_string()),
                    collection_id: "tenant_docs".to_string(),
                    tenant_aware: true,
                },
                ApiCall::VectorBatch {
                    tenant_id: Some("tenant-a".to_string()),
                    collection_id: "docs".to_string(),
                    vector_count: 1,
                },
                ApiCall::VectorGet {
                    tenant_id: Some("tenant-a".to_string()),
                    collection_id: "docs".to_string(),
                    vector_id: "vec-1".to_string(),
                    include_vector: true,
                    include_metadata: false,
                },
                ApiCall::Hybrid,
                ApiCall::Sql {
                    query: "select * from docs".to_string(),
                    parameter_count: None,
                    collection: Some("docs".to_string()),
                    tenant_id: None,
                    subject: None,
                    tenant_stable_id: None,
                    auth_class: proximadb_tenant::AuthClass::Anonymous,
                },
            ]
        );
    }

    #[tokio::test]
    async fn noop_unified_handlers_exercise_all_injected_ports() {
        let handlers = noop_unified_handlers();

        assert!(
            handlers
                .collection
                .get_collection("docs", Some("tenant-a"))
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(
            handlers
                .collection
                .create_collection(
                    CollectionConfig {
                        name: "docs".to_string(),
                        ..CollectionConfig::default()
                    },
                    Some("tenant-a"),
                )
                .await
                .unwrap()
                .id,
            "docs"
        );
        assert_eq!(
            handlers
                .collection
                .update_collection(
                    "docs",
                    CollectionConfig {
                        name: "docs-v2".to_string(),
                        ..CollectionConfig::default()
                    },
                    None,
                )
                .await
                .unwrap()
                .id,
            "docs"
        );
        assert!(
            handlers
                .collection
                .delete_collection("docs", None)
                .await
                .unwrap()
        );
        assert!(
            handlers
                .collection
                .list_collections(None)
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            handlers
                .collection
                .resolve_collection_id("docs")
                .await
                .unwrap()
                .as_deref(),
            Some("docs")
        );

        assert!(
            handlers
                .vector_ops
                .search(VectorSearchRequest::default(), PortIdentity::anonymous())
                .await
                .unwrap()
                .results
                .is_none()
        );
        assert!(
            handlers
                .vector_ops
                .batch_upsert(VectorBatchRequest::default(), Some("tenant-a"))
                .await
                .unwrap()
                .results
                .is_none()
        );
        assert!(
            handlers
                .vector_ops
                .get_vector("docs", "vec-1", true, true, None)
                .await
                .unwrap()
                .results
                .is_none()
        );
        handlers.vector_ops.flush_all().await.unwrap();
        assert!(handlers.vector_ops.metrics().await.unwrap().is_object());

        let query = handlers.query_adapter.as_ref().unwrap();
        assert!(
            query
                .vector_search(VectorSearchRequest::default())
                .await
                .unwrap()
                .results
                .is_none()
        );
        query
            .execute_hybrid(HybridSearchRequest::default())
            .await
            .unwrap();
        assert!(
            query
                .execute_sql(
                    "select 1".to_string(),
                    Some("docs".to_string()),
                    PortIdentity::anonymous(),
                )
                .await
                .unwrap()
                .rows
                .is_empty()
        );
    }
}
