//! Transactional facade for multi-model stores with executable participant-backed writes.

use std::sync::Arc;

use anyhow::{Result, anyhow};
use proximadb_records::ProximaRecord;

use crate::proto::proximadb_v1::{DocumentUpdate, LogEntry, MetricSample, SqlObject, TraceData};
use crate::storage::multimodel::facade::MultiModelStorageFacade;

use super::coordinator::{TransactionConfig, TransactionCoordinator};
use super::isolation::IsolationLevel;
use super::participants::{
    DocumentStoreParticipant, GraphEdge, GraphNode, GraphStoreParticipant,
    ObservabilityStoreParticipant, VectorStoreParticipant,
};
use super::two_phase_commit::{CommitResult, ParticipantType, PrepareResult, TwoPhaseParticipant};

/// Runtime wrapper that wires real store-backed participants into the
/// multi-model transaction coordinator and exposes staging methods for
/// document, observability, vector, and graph writes.
pub struct TransactionalMultiModelFacade {
    storage: Arc<MultiModelStorageFacade>,
    coordinator: Arc<TransactionCoordinator>,
    document_participant: Option<Arc<DocumentStoreParticipant>>,
    observability_participant: Option<Arc<ObservabilityStoreParticipant>>,
    vector_participant: Option<Arc<VectorStoreParticipant>>,
    graph_participant: Option<Arc<GraphStoreParticipant>>,
}

impl TransactionalMultiModelFacade {
    pub async fn new(storage: Arc<MultiModelStorageFacade>, config: TransactionConfig) -> Self {
        let coordinator = Arc::new(TransactionCoordinator::new(config));

        let document_participant = storage
            .get_document_store()
            .and_then(|store| store.service().map(Arc::clone))
            .map(DocumentStoreParticipant::new)
            .map(Arc::new);
        let observability_participant = storage
            .get_observability_store()
            .and_then(|store| store.service().map(Arc::clone))
            .map(ObservabilityStoreParticipant::new)
            .map(Arc::new);

        // Vector participants are wired when their write-operation adapter is
        // provided via `with_vector_participant`. GraphStore implements the
        // graph write adapter directly so commits flow through the service path.
        let vector_participant: Option<Arc<VectorStoreParticipant>> = None;
        let graph_participant = storage
            .get_graph_store()
            .map(|store| {
                let service: Arc<dyn super::participants::GraphWriteOperations> = store.clone();
                GraphStoreParticipant::new(service)
            })
            .map(Arc::new);

        if let Some(participant) = &document_participant {
            coordinator.register_participant(participant.clone()).await;
        }
        if let Some(participant) = &observability_participant {
            coordinator.register_participant(participant.clone()).await;
        }
        if let Some(participant) = &graph_participant {
            coordinator.register_participant(participant.clone()).await;
        }

        Self {
            storage,
            coordinator,
            document_participant,
            observability_participant,
            vector_participant,
            graph_participant,
        }
    }

    /// Attach a vector store participant backed by a [`VectorWriteOperations`] service.
    pub async fn with_vector_participant(
        mut self,
        service: Arc<dyn super::participants::VectorWriteOperations>,
    ) -> Self {
        let participant = Arc::new(VectorStoreParticipant::new(service));
        self.coordinator
            .register_participant(participant.clone())
            .await;
        self.vector_participant = Some(participant);
        self
    }

    /// Attach a graph store participant backed by a [`GraphWriteOperations`] service.
    pub async fn with_graph_participant(
        mut self,
        service: Arc<dyn super::participants::GraphWriteOperations>,
    ) -> Self {
        let participant = Arc::new(GraphStoreParticipant::new(service));
        self.coordinator
            .register_participant(participant.clone())
            .await;
        self.graph_participant = Some(participant);
        self
    }

    pub fn storage(&self) -> &Arc<MultiModelStorageFacade> {
        &self.storage
    }

    pub fn coordinator(&self) -> Arc<TransactionCoordinator> {
        Arc::clone(&self.coordinator)
    }

    pub async fn begin(&self, isolation_level: Option<IsolationLevel>) -> Result<String> {
        self.coordinator.begin(isolation_level).await
    }

    pub async fn commit(&self, transaction_id: &str) -> Result<()> {
        self.coordinator.commit(transaction_id).await?;
        self.commit_staged_participants(transaction_id).await
    }

    pub async fn rollback(&self, transaction_id: &str) -> Result<()> {
        self.coordinator.rollback(transaction_id).await?;

        // Clear staged operations from all participants
        if let Some(ref participant) = self.document_participant {
            participant.clear_staged(transaction_id).await;
        }
        if let Some(ref participant) = self.observability_participant {
            participant.clear_staged(transaction_id).await;
        }
        if let Some(ref participant) = self.vector_participant {
            participant.clear_staged(transaction_id).await;
        }
        if let Some(ref participant) = self.graph_participant {
            participant.clear_staged(transaction_id).await;
        }

        Ok(())
    }

    async fn commit_staged_participants(&self, transaction_id: &str) -> Result<()> {
        if let Some(ref participant) = self.document_participant {
            Self::commit_participant(participant.as_ref(), transaction_id).await?;
        }
        if let Some(ref participant) = self.observability_participant {
            Self::commit_participant(participant.as_ref(), transaction_id).await?;
        }
        if let Some(ref participant) = self.vector_participant {
            Self::commit_participant(participant.as_ref(), transaction_id).await?;
        }
        if let Some(ref participant) = self.graph_participant {
            Self::commit_participant(participant.as_ref(), transaction_id).await?;
        }
        Ok(())
    }

    async fn commit_participant(
        participant: &(dyn TwoPhaseParticipant + Send + Sync),
        transaction_id: &str,
    ) -> Result<()> {
        match participant.prepare(transaction_id).await {
            PrepareResult::Yes => {}
            PrepareResult::No(reason) => {
                return Err(anyhow!(
                    "{} participant rejected transaction {}: {}",
                    participant.participant_type().name(),
                    transaction_id,
                    reason
                ));
            }
            PrepareResult::Timeout => {
                return Err(anyhow!(
                    "{} participant prepare timed out for transaction {}",
                    participant.participant_type().name(),
                    transaction_id
                ));
            }
            PrepareResult::Error(error) => {
                return Err(anyhow!(
                    "{} participant prepare failed for transaction {}: {}",
                    participant.participant_type().name(),
                    transaction_id,
                    error
                ));
            }
        }

        match participant.commit(transaction_id).await {
            CommitResult::Success => Ok(()),
            CommitResult::Failed(error) => Err(anyhow!(
                "{} participant commit failed for transaction {}: {}",
                participant.participant_type().name(),
                transaction_id,
                error
            )),
            CommitResult::Timeout => Err(anyhow!(
                "{} participant commit timed out for transaction {}",
                participant.participant_type().name(),
                transaction_id
            )),
        }
    }

    pub async fn insert_document(
        &self,
        transaction_id: &str,
        collection: &str,
        id: &str,
        document: SqlObject,
        indexed_paths: Vec<String>,
    ) -> Result<()> {
        let participant = self.document_participant.as_ref().ok_or_else(|| {
            anyhow!("Document transactional participant is not configured on this facade")
        })?;
        self.coordinator
            .involve_store(transaction_id, ParticipantType::Document)
            .await?;
        participant
            .stage_insert(transaction_id, collection, id, document, indexed_paths)
            .await?;
        self.coordinator
            .record_write(transaction_id, "document", id)
            .await
    }

    pub async fn update_document(
        &self,
        transaction_id: &str,
        collection: &str,
        id: &str,
        updates: Vec<DocumentUpdate>,
    ) -> Result<()> {
        let participant = self.document_participant.as_ref().ok_or_else(|| {
            anyhow!("Document transactional participant is not configured on this facade")
        })?;
        self.coordinator
            .involve_store(transaction_id, ParticipantType::Document)
            .await?;
        participant
            .stage_update(transaction_id, collection, id, updates)
            .await?;
        self.coordinator
            .record_write(transaction_id, "document", id)
            .await
    }

    pub async fn delete_document(
        &self,
        transaction_id: &str,
        collection: &str,
        id: &str,
    ) -> Result<()> {
        let participant = self.document_participant.as_ref().ok_or_else(|| {
            anyhow!("Document transactional participant is not configured on this facade")
        })?;
        self.coordinator
            .involve_store(transaction_id, ParticipantType::Document)
            .await?;
        participant
            .stage_delete(transaction_id, collection, id)
            .await?;
        self.coordinator
            .record_delete(transaction_id, "document", id)
            .await
    }

    pub async fn ingest_logs(
        &self,
        transaction_id: &str,
        namespace: &str,
        logs: Vec<LogEntry>,
    ) -> Result<()> {
        let participant = self.observability_participant.as_ref().ok_or_else(|| {
            anyhow!("Observability transactional participant is not configured on this facade")
        })?;
        self.coordinator
            .involve_store(transaction_id, ParticipantType::Observability)
            .await?;
        participant
            .stage_ingest_logs(transaction_id, namespace, logs)
            .await?;
        self.coordinator
            .record_write(
                transaction_id,
                "observability",
                &Self::observability_write_key("logs", namespace),
            )
            .await
    }

    pub async fn ingest_metrics(
        &self,
        transaction_id: &str,
        namespace: &str,
        metrics: Vec<MetricSample>,
    ) -> Result<()> {
        let participant = self.observability_participant.as_ref().ok_or_else(|| {
            anyhow!("Observability transactional participant is not configured on this facade")
        })?;
        self.coordinator
            .involve_store(transaction_id, ParticipantType::Observability)
            .await?;
        participant
            .stage_ingest_metrics(transaction_id, namespace, metrics)
            .await?;
        self.coordinator
            .record_write(
                transaction_id,
                "observability",
                &Self::observability_write_key("metrics", namespace),
            )
            .await
    }

    pub async fn ingest_traces(
        &self,
        transaction_id: &str,
        namespace: &str,
        traces: Vec<TraceData>,
    ) -> Result<()> {
        let participant = self.observability_participant.as_ref().ok_or_else(|| {
            anyhow!("Observability transactional participant is not configured on this facade")
        })?;
        self.coordinator
            .involve_store(transaction_id, ParticipantType::Observability)
            .await?;
        participant
            .stage_ingest_traces(transaction_id, namespace, traces)
            .await?;
        self.coordinator
            .record_write(
                transaction_id,
                "observability",
                &Self::observability_write_key("traces", namespace),
            )
            .await
    }

    // -- Vector staging methods --

    pub async fn insert_vectors(
        &self,
        transaction_id: &str,
        collection: &str,
        vectors: Vec<ProximaRecord>,
    ) -> Result<()> {
        let participant = self.vector_participant.as_ref().ok_or_else(|| {
            anyhow!("Vector transactional participant is not configured on this facade")
        })?;
        self.coordinator
            .involve_store(transaction_id, ParticipantType::Vector)
            .await?;
        participant
            .stage_insert(transaction_id, collection, vectors)
            .await?;
        self.coordinator
            .record_write(transaction_id, "vector", collection)
            .await
    }

    pub async fn delete_vectors(
        &self,
        transaction_id: &str,
        collection: &str,
        ids: Vec<String>,
    ) -> Result<()> {
        let participant = self.vector_participant.as_ref().ok_or_else(|| {
            anyhow!("Vector transactional participant is not configured on this facade")
        })?;
        self.coordinator
            .involve_store(transaction_id, ParticipantType::Vector)
            .await?;
        participant
            .stage_delete(transaction_id, collection, ids)
            .await?;
        self.coordinator
            .record_delete(transaction_id, "vector", collection)
            .await
    }

    // -- Graph staging methods --

    pub async fn create_graph_node(
        &self,
        transaction_id: &str,
        graph_id: &str,
        node: GraphNode,
    ) -> Result<()> {
        let participant = self.graph_participant.as_ref().ok_or_else(|| {
            anyhow!("Graph transactional participant is not configured on this facade")
        })?;
        self.coordinator
            .involve_store(transaction_id, ParticipantType::Graph)
            .await?;
        let node_id = node.id.clone();
        participant
            .stage_create_node(transaction_id, graph_id, node)
            .await?;
        self.coordinator
            .record_write(transaction_id, "graph", &node_id)
            .await
    }

    pub async fn create_graph_edge(
        &self,
        transaction_id: &str,
        graph_id: &str,
        edge: GraphEdge,
    ) -> Result<()> {
        let participant = self.graph_participant.as_ref().ok_or_else(|| {
            anyhow!("Graph transactional participant is not configured on this facade")
        })?;
        self.coordinator
            .involve_store(transaction_id, ParticipantType::Graph)
            .await?;
        let edge_id = edge.id.clone();
        participant
            .stage_create_edge(transaction_id, graph_id, edge)
            .await?;
        self.coordinator
            .record_write(transaction_id, "graph", &edge_id)
            .await
    }

    fn observability_write_key(kind: &str, namespace: &str) -> String {
        format!("{kind}:{namespace}")
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::graph::GraphService;
    use crate::proto::proximadb_v1::{
        DocumentCollectionConfig, LogFilter, ObservabilityNamespaceConfig, Severity, SqlValue,
        sql_value,
    };
    use crate::storage::multimodel::stores::{
        DocumentStore, DocumentStoreConfig, GraphStore, GraphStoreConfig, ObservabilityStore,
        ObservabilityStoreConfig,
    };
    use crate::storage::traits::{
        DocumentCollectionInfo, DocumentRecord, DocumentStorageOperations, IngestResult,
        LogQueryResult, MetricAggregationParams, MetricAggregationResult, NamespaceInfo,
        ObservabilityStorageOperations,
    };
    use proximadb_graph::record::{GraphEdgeKey, GraphNodeKey};
    use proximadb_records::test_support::MemoryRecordStore;
    use proximadb_records::{RecordKey, RecordStore};
    use tokio::sync::RwLock;

    #[derive(Default)]
    struct RecordingDocumentService {
        records: RwLock<HashMap<String, DocumentRecord>>,
    }

    #[async_trait::async_trait]
    impl crate::storage::traits::DocumentStorageOperations for RecordingDocumentService {
        async fn insert_document(
            &self,
            collection: &str,
            id: &str,
            document: SqlObject,
            _indexed_paths: Vec<String>,
        ) -> Result<DocumentRecord> {
            let record = DocumentRecord {
                id: id.to_string(),
                document,
                version: 1,
                created_at_ns: 0,
                updated_at_ns: 0,
            };
            self.records
                .write()
                .await
                .insert(format!("{collection}:{id}"), record.clone());
            Ok(record)
        }

        async fn get_document(&self, collection: &str, id: &str) -> Result<Option<DocumentRecord>> {
            Ok(self
                .records
                .read()
                .await
                .get(&format!("{collection}:{id}"))
                .cloned())
        }

        async fn query_documents(
            &self,
            _collection: &str,
            _filter: Option<crate::proto::proximadb_v1::DocumentFilter>,
            _limit: usize,
            _offset: usize,
        ) -> Result<Vec<DocumentRecord>> {
            Ok(self.records.read().await.values().cloned().collect())
        }

        async fn update_document(
            &self,
            _collection: &str,
            _id: &str,
            _updates: Vec<DocumentUpdate>,
        ) -> Result<DocumentRecord> {
            Err(anyhow!("not implemented in test"))
        }

        async fn delete_document(&self, collection: &str, id: &str) -> Result<bool> {
            Ok(self
                .records
                .write()
                .await
                .remove(&format!("{collection}:{id}"))
                .is_some())
        }

        async fn create_document_collection(
            &self,
            config: DocumentCollectionConfig,
        ) -> Result<String> {
            Ok(config.name)
        }

        async fn list_document_collections(&self) -> Result<Vec<DocumentCollectionInfo>> {
            Ok(vec![])
        }
    }

    #[derive(Default)]
    struct RecordingObservabilityService {
        logs: RwLock<HashMap<String, Vec<LogEntry>>>,
    }

    #[async_trait::async_trait]
    impl ObservabilityStorageOperations for RecordingObservabilityService {
        async fn ingest_logs(&self, namespace: &str, logs: Vec<LogEntry>) -> Result<IngestResult> {
            self.logs
                .write()
                .await
                .entry(namespace.to_string())
                .or_default()
                .extend(logs.clone());
            Ok(IngestResult {
                ingested: logs.len() as u64,
                failed: 0,
                errors: vec![],
                processing_time_ms: 0,
            })
        }

        async fn ingest_metrics(
            &self,
            _namespace: &str,
            metrics: Vec<MetricSample>,
        ) -> Result<IngestResult> {
            Ok(IngestResult {
                ingested: metrics.len() as u64,
                failed: 0,
                errors: vec![],
                processing_time_ms: 0,
            })
        }

        async fn ingest_traces(
            &self,
            _namespace: &str,
            traces: Vec<TraceData>,
        ) -> Result<IngestResult> {
            Ok(IngestResult {
                ingested: traces.len() as u64,
                failed: 0,
                errors: vec![],
                processing_time_ms: 0,
            })
        }

        async fn query_logs(
            &self,
            namespace: &str,
            _start_time_ns: i64,
            _end_time_ns: i64,
            _filter: Option<LogFilter>,
            _limit: u32,
        ) -> Result<LogQueryResult> {
            Ok(LogQueryResult {
                logs: self
                    .logs
                    .read()
                    .await
                    .get(namespace)
                    .cloned()
                    .unwrap_or_default(),
                next_cursor: None,
                total_matched: 0,
                query_time_ms: 0,
            })
        }

        async fn aggregate_metrics(
            &self,
            _namespace: &str,
            _params: MetricAggregationParams,
        ) -> Result<MetricAggregationResult> {
            Ok(MetricAggregationResult {
                series: vec![],
                query_time_ms: 0,
            })
        }

        async fn query_traces(
            &self,
            _namespace: &str,
            _start_time_ns: i64,
            _end_time_ns: i64,
            _trace_id: Option<String>,
            _service: Option<String>,
            _limit: u32,
        ) -> Result<Vec<TraceData>> {
            Ok(vec![])
        }

        async fn create_namespace(&self, config: ObservabilityNamespaceConfig) -> Result<String> {
            Ok(config.name)
        }

        async fn list_namespaces(&self) -> Result<Vec<NamespaceInfo>> {
            Ok(vec![])
        }
    }

    #[tokio::test]
    async fn test_transactional_facade_commits_distributed_document_and_logs() {
        let document_service = Arc::new(RecordingDocumentService::default());
        let observability_service = Arc::new(RecordingObservabilityService::default());

        let storage = Arc::new(
            MultiModelStorageFacade::new()
                .with_document_store(Arc::new(
                    DocumentStore::new(DocumentStoreConfig::default())
                        .with_service(document_service.clone()),
                ))
                .with_observability_store(Arc::new(
                    ObservabilityStore::new(ObservabilityStoreConfig::default())
                        .with_service(observability_service.clone()),
                )),
        );

        let runtime =
            TransactionalMultiModelFacade::new(storage, TransactionConfig::default()).await;
        let transaction_id = runtime.begin(None).await.unwrap();

        runtime
            .insert_document(
                &transaction_id,
                "users",
                "user-1",
                SqlObject {
                    fields: HashMap::from([(
                        "name".to_string(),
                        SqlValue {
                            value: Some(sql_value::Value::StringValue("alice".to_string())),
                        },
                    )]),
                },
                vec!["name".to_string()],
            )
            .await
            .unwrap();
        runtime
            .ingest_logs(
                &transaction_id,
                "prod",
                vec![LogEntry {
                    timestamp_ns: 1,
                    severity: Severity::Info as i32,
                    message: "user created".to_string(),
                    fields: HashMap::new(),
                    source: Some("api".to_string()),
                    service: Some("users".to_string()),
                }],
            )
            .await
            .unwrap();

        runtime.commit(&transaction_id).await.unwrap();

        assert!(
            document_service
                .get_document("users", "user-1")
                .await
                .unwrap()
                .is_some()
        );
        assert_eq!(
            observability_service
                .query_logs("prod", 0, 10, None, 10)
                .await
                .unwrap()
                .logs
                .len(),
            1
        );
        assert_eq!(
            runtime.coordinator.get_state(&transaction_id).await,
            Some(super::super::two_phase_commit::TransactionState::Committed)
        );
        assert_eq!(
            runtime
                .coordinator
                .two_phase_commit()
                .get_state(&transaction_id)
                .await,
            Some(super::super::two_phase_commit::TransactionState::Committed)
        );
    }

    #[tokio::test]
    async fn test_transactional_facade_rollback_clears_single_store_document_write() {
        let document_service = Arc::new(RecordingDocumentService::default());
        let storage = Arc::new(
            MultiModelStorageFacade::new().with_document_store(Arc::new(
                DocumentStore::new(DocumentStoreConfig::default())
                    .with_service(document_service.clone()),
            )),
        );

        let runtime =
            TransactionalMultiModelFacade::new(storage, TransactionConfig::default()).await;
        let transaction_id = runtime.begin(None).await.unwrap();

        runtime
            .insert_document(
                &transaction_id,
                "users",
                "user-2",
                SqlObject {
                    fields: HashMap::new(),
                },
                vec![],
            )
            .await
            .unwrap();

        runtime.rollback(&transaction_id).await.unwrap();

        assert!(
            document_service
                .get_document("users", "user-2")
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(
            runtime.coordinator.get_state(&transaction_id).await,
            Some(super::super::two_phase_commit::TransactionState::Aborted)
        );
        assert_eq!(
            runtime
                .document_participant
                .as_ref()
                .unwrap()
                .staged_operation_count(&transaction_id)
                .await,
            0
        );
    }

    #[tokio::test]
    async fn test_transactional_facade_commits_graph_records_through_service() {
        let graph_id = format!("txn_graph_{}", std::process::id());
        let record_store = Arc::new(MemoryRecordStore::default());
        let graph_service =
            Arc::new(GraphService::new().with_canonical_record_store(record_store.clone()));
        graph_service
            .create_graph_collection(crate::proto::proximadb_v1::CreateGraphRequest {
                graph_id: graph_id.clone(),
                name: None,
                description: None,
                schema: None,
                storage_config: None,
                engine_config: None,
                access_control: None,
            })
            .await
            .unwrap();
        let graph_store = Arc::new(
            GraphStore::new(GraphStoreConfig::default())
                .with_service(graph_service.clone())
                .with_default_graph(graph_id.clone()),
        );
        let storage = Arc::new(MultiModelStorageFacade::new().with_graph_store(graph_store));
        let runtime =
            TransactionalMultiModelFacade::new(storage, TransactionConfig::default()).await;
        let transaction_id = runtime.begin(None).await.unwrap();

        runtime
            .create_graph_node(
                &transaction_id,
                &graph_id,
                GraphNode {
                    id: "n1".to_string(),
                    label: "Person".to_string(),
                    properties: HashMap::from([("name".to_string(), "alice".to_string())]),
                },
            )
            .await
            .unwrap();
        runtime
            .create_graph_node(
                &transaction_id,
                &graph_id,
                GraphNode {
                    id: "n2".to_string(),
                    label: "Person".to_string(),
                    properties: HashMap::from([("name".to_string(), "bob".to_string())]),
                },
            )
            .await
            .unwrap();
        runtime
            .create_graph_edge(
                &transaction_id,
                &graph_id,
                GraphEdge {
                    id: "e1".to_string(),
                    source: "n1".to_string(),
                    target: "n2".to_string(),
                    edge_type: "KNOWS".to_string(),
                    properties: HashMap::from([("since".to_string(), "2026".to_string())]),
                },
            )
            .await
            .unwrap();

        runtime.commit(&transaction_id).await.unwrap();

        assert!(
            graph_service
                .get_node(&graph_id, &"n1".to_string())
                .await
                .unwrap()
                .is_some()
        );
        assert!(
            graph_service
                .get_edge(&graph_id, &"e1".to_string())
                .await
                .unwrap()
                .is_some()
        );
        assert!(
            record_store
                .get_record(&RecordKey::new(
                    GraphNodeKey::new(&graph_id, "n1").canonical_oid()
                ))
                .await
                .unwrap()
                .is_some()
        );
        assert!(
            record_store
                .get_record(&RecordKey::new(
                    GraphEdgeKey::new(&graph_id, "e1").canonical_oid()
                ))
                .await
                .unwrap()
                .is_some()
        );
    }
}
