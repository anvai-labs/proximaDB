//! # Multi-Model Storage Facade
//!
//! Unified entry point for all multi-model storage operations.
//! Routes operations to appropriate specialized stores based on data model.

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;

use crate::catalog::internal::{
    CatalogObject, InformationSchema, InternalSchemaRegistry, ObjectType,
};
use crate::graph::engines::GraphEngine;
use crate::storage::traits::{
    CompactionParameters, DocumentStorageOperations, FlushParameters, MultiModelStats,
    ObservabilityStorageOperations, UnifiedStorageFormat,
};

use super::stores::{
    DocumentStore, DocumentStoreConfig, GraphStore, GraphStoreConfig, ObservabilityStore,
    ObservabilityStoreConfig, RDBMSStore, RDBMSStoreConfig, VectorStore, VectorStoreConfig,
};
use super::traits::{ModelType, MultiModelStorageEngine, StoreCapabilities};

/// Configuration for the multi-model storage facade
#[derive(Debug, Clone, Default)]
pub struct MultiModelFacadeConfig {
    /// Vector store configuration
    pub vector: VectorStoreConfig,
    /// Document store configuration
    pub document: DocumentStoreConfig,
    /// Graph store configuration
    pub graph: GraphStoreConfig,
    /// RDBMS store configuration
    pub rdbms: RDBMSStoreConfig,
    /// Observability store configuration
    pub observability: ObservabilityStoreConfig,
}

/// MultiModelStorageFacade provides unified access to all specialized stores
///
/// ## Architecture
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────┐
/// │               MultiModelStorageFacade                            │
/// │  Unified entry point for all multi-model storage operations     │
/// └─────────────────────────────────────────────────────────────────┘
///                               │
///         ┌─────────────────────┼─────────────────────┐
///         ▼                     ▼                     ▼
/// ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
/// │  VectorStore  │    │  GraphStore   │    │  RDBMSStore   │
/// │  (Helix+SST)  │    │   (Orion)     │    │  (SST+column) │
/// └───────────────┘    └───────────────┘    └───────────────┘
///         │                     │                     │
/// ┌───────────────┐    ┌───────────────┐
/// │ DocumentStore │    │ Observability │
/// │ (SST+column+  │    │ (Partitioned+ │
/// │  Tantivy)     │    │  WAL+Rollups) │
/// └───────────────┘    └───────────────┘
/// ```
pub struct MultiModelStorageFacade {
    /// Vector store (Helix + SST)
    vector_store: Option<Arc<VectorStore>>,
    /// Document store (SST + columnar + Tantivy)
    document_store: Option<Arc<DocumentStore>>,
    /// Graph store (Orion)
    graph_store: Option<Arc<GraphStore>>,
    /// RDBMS store (SST + columnar HTAP)
    rdbms_store: Option<Arc<RDBMSStore>>,
    /// Observability store (Time-partitioned + WAL)
    observability_store: Option<Arc<ObservabilityStore>>,
    /// Internal schema registry for multi-model catalog
    schema_registry: Arc<InternalSchemaRegistry>,
    /// Configuration
    _config: MultiModelFacadeConfig,
}

impl MultiModelStorageFacade {
    /// Create a new facade with default configuration
    pub fn new() -> Self {
        Self::with_config(MultiModelFacadeConfig::default())
    }

    /// Create a new facade with the given configuration
    pub fn with_config(config: MultiModelFacadeConfig) -> Self {
        Self {
            vector_store: None,
            document_store: None,
            graph_store: None,
            rdbms_store: None,
            observability_store: None,
            schema_registry: Arc::new(InternalSchemaRegistry::new()),
            _config: config,
        }
    }

    /// Create a new facade with a custom schema registry
    pub fn with_registry(
        config: MultiModelFacadeConfig,
        registry: Arc<InternalSchemaRegistry>,
    ) -> Self {
        Self {
            vector_store: None,
            document_store: None,
            graph_store: None,
            rdbms_store: None,
            observability_store: None,
            schema_registry: registry,
            _config: config,
        }
    }

    /// Set the vector store
    pub fn with_vector_store(mut self, store: Arc<VectorStore>) -> Self {
        self.vector_store = Some(store);
        self
    }

    /// Set the document store
    pub fn with_document_store(mut self, store: Arc<DocumentStore>) -> Self {
        self.document_store = Some(store);
        self
    }

    /// Set the graph store
    pub fn with_graph_store(mut self, store: Arc<GraphStore>) -> Self {
        self.graph_store = Some(store);
        self
    }

    /// Set the RDBMS store
    pub fn with_rdbms_store(mut self, store: Arc<RDBMSStore>) -> Self {
        self.rdbms_store = Some(store);
        self
    }

    /// Set the observability store
    pub fn with_observability_store(mut self, store: Arc<ObservabilityStore>) -> Self {
        self.observability_store = Some(store);
        self
    }

    /// Get a reference to the vector store
    pub fn get_vector_store(&self) -> Option<&Arc<VectorStore>> {
        self.vector_store.as_ref()
    }

    /// Get a reference to the document store
    pub fn get_document_store(&self) -> Option<&Arc<DocumentStore>> {
        self.document_store.as_ref()
    }

    /// Get a reference to the graph store
    pub fn get_graph_store(&self) -> Option<&Arc<GraphStore>> {
        self.graph_store.as_ref()
    }

    /// Get a reference to the RDBMS store
    pub fn get_rdbms_store(&self) -> Option<&Arc<RDBMSStore>> {
        self.rdbms_store.as_ref()
    }

    /// Get a reference to the observability store
    pub fn get_observability_store(&self) -> Option<&Arc<ObservabilityStore>> {
        self.observability_store.as_ref()
    }

    /// Get a reference to the schema registry
    pub fn schema_registry(&self) -> &Arc<InternalSchemaRegistry> {
        &self.schema_registry
    }

    /// Get an INFORMATION_SCHEMA interface for introspection
    pub fn information_schema(&self) -> InformationSchema {
        InformationSchema::new(self.schema_registry.clone())
    }

    /// Register a vector collection in the schema registry
    pub async fn register_vector_collection(
        &self,
        name: &str,
        dimension: u32,
        distance_metric: &str,
    ) -> Result<Arc<CatalogObject>> {
        self.schema_registry
            .create_vector_collection(name, dimension, distance_metric)
            .await
    }

    /// Register a document collection in the schema registry
    pub async fn register_document_collection(
        &self,
        name: &str,
        json_schema: Option<&str>,
    ) -> Result<Arc<CatalogObject>> {
        self.schema_registry
            .create_document_collection(name, json_schema)
            .await
    }

    /// Register a graph in the schema registry
    pub async fn register_graph(&self, name: &str, directed: bool) -> Result<Arc<CatalogObject>> {
        self.schema_registry.create_graph(name, directed).await
    }

    /// Register a log stream in the schema registry
    pub async fn register_log_stream(
        &self,
        name: &str,
        retention_seconds: u64,
    ) -> Result<Arc<CatalogObject>> {
        self.schema_registry
            .create_log_stream(name, retention_seconds)
            .await
    }

    /// Get a catalog object by fully qualified name
    pub async fn get_catalog_object(&self, fqn: &str) -> Result<Arc<CatalogObject>> {
        self.schema_registry.get(fqn).await
    }

    /// List all catalog objects
    pub async fn list_catalog_objects(&self) -> Vec<Arc<CatalogObject>> {
        self.schema_registry.list_all().await
    }

    /// List catalog objects by type
    pub async fn list_objects_by_type(&self, object_type: ObjectType) -> Vec<Arc<CatalogObject>> {
        self.schema_registry.list_by_type(object_type).await
    }
}

impl Default for MultiModelStorageFacade {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MultiModelStorageEngine for MultiModelStorageFacade {
    fn vector_store(&self) -> Option<Arc<dyn UnifiedStorageFormat>> {
        // Return the primary engine from VectorStore
        self.vector_store
            .as_ref()
            .and_then(|s| s.primary_engine().cloned())
    }

    fn document_store(&self) -> Option<Arc<dyn DocumentStorageOperations>> {
        self.document_store
            .as_ref()
            .map(|s| s.clone() as Arc<dyn DocumentStorageOperations>)
    }

    fn graph_store(&self) -> Option<Arc<dyn GraphEngine>> {
        self.graph_store
            .as_ref()
            .map(|s| s.clone() as Arc<dyn GraphEngine>)
    }

    fn observability_store(&self) -> Option<Arc<dyn ObservabilityStorageOperations>> {
        self.observability_store
            .as_ref()
            .map(|s| s.clone() as Arc<dyn ObservabilityStorageOperations>)
    }

    fn rdbms_store(&self) -> Option<Arc<dyn UnifiedStorageFormat>> {
        // Return the primary engine from RDBMSStore
        self.rdbms_store
            .as_ref()
            .and_then(|s| s.primary_engine().cloned())
    }

    fn supported_models(&self) -> Vec<ModelType> {
        let mut models = Vec::new();

        if self.vector_store.is_some() {
            models.push(ModelType::Vector);
        }
        if self.document_store.is_some() {
            models.push(ModelType::Document);
        }
        if self.graph_store.is_some() {
            models.push(ModelType::Graph);
        }
        if self.rdbms_store.is_some() {
            models.push(ModelType::Relational);
        }
        if self.observability_store.is_some() {
            models.push(ModelType::Observability);
        }

        models
    }

    fn get_capabilities(&self, model_type: ModelType) -> Option<StoreCapabilities> {
        match model_type {
            ModelType::Vector => self.vector_store.as_ref().map(|s| s.capabilities()),
            ModelType::Document => self.document_store.as_ref().map(|s| s.capabilities()),
            ModelType::Graph => self.graph_store.as_ref().map(|s| s.capabilities()),
            ModelType::Relational => self.rdbms_store.as_ref().map(|s| s.capabilities()),
            ModelType::Observability => self.observability_store.as_ref().map(|s| s.capabilities()),
            ModelType::TimeSeries => None, // Not yet implemented
            ModelType::Event => None,      // Not yet implemented
        }
    }

    async fn get_multi_model_stats(&self) -> Result<MultiModelStats> {
        let mut stats = MultiModelStats::default();

        // Collect vector stats from primary engine
        if let Some(store) = &self.vector_store
            && let Some(engine) = store.primary_engine()
            && let Ok(engine_stats) = engine.get_engine_stats().await
        {
            stats.total_storage_bytes += engine_stats.total_storage_bytes;
        }

        // Collect RDBMS stats
        if let Some(store) = &self.rdbms_store
            && let Some(engine) = store.primary_engine()
            && let Ok(engine_stats) = engine.get_engine_stats().await
        {
            stats.total_storage_bytes += engine_stats.total_storage_bytes;
        }

        // Collect graph stats
        if let Some(store) = &self.graph_store {
            stats.graph_node_count = store.node_count() as u64;
            stats.graph_edge_count = store.edge_count() as u64;
        }

        // Document and observability stats would come from their services
        // These are typically async and would be populated on demand

        Ok(stats)
    }

    async fn get_storage_size(&self, model_type: ModelType) -> Result<u64> {
        match model_type {
            ModelType::Vector => {
                if let Some(store) = &self.vector_store
                    && let Some(engine) = store.primary_engine()
                    && let Ok(stats) = engine.get_engine_stats().await
                {
                    return Ok(stats.total_storage_bytes);
                }
                Ok(0)
            }
            ModelType::Relational => {
                if let Some(store) = &self.rdbms_store
                    && let Some(engine) = store.primary_engine()
                    && let Ok(stats) = engine.get_engine_stats().await
                {
                    return Ok(stats.total_storage_bytes);
                }
                Ok(0)
            }
            _ => Ok(0), // Other stores would need their own size tracking
        }
    }

    async fn flush_all(&self) -> Result<()> {
        let flush_params = FlushParameters {
            collection_id: None, // Flush all collections
            force: true,
            synchronous: true,
            ..Default::default()
        };

        if let Some(store) = &self.vector_store
            && let Some(engine) = store.primary_engine()
        {
            engine.flush(flush_params.clone()).await?;
        }
        if let Some(store) = &self.rdbms_store
            && let Some(engine) = store.primary_engine()
        {
            engine.flush(flush_params).await?;
        }

        // Document and observability would have their own flush methods

        Ok(())
    }

    async fn compact_all(&self) -> Result<()> {
        let compact_params = CompactionParameters::default();

        if let Some(store) = &self.vector_store
            && let Some(engine) = store.primary_engine()
        {
            engine.compact(compact_params.clone()).await?;
        }
        if let Some(store) = &self.rdbms_store
            && let Some(engine) = store.primary_engine()
        {
            engine.compact(compact_params).await?;
        }

        Ok(())
    }

    async fn stop(&self) -> Result<()> {
        // Flush all stores before stopping
        self.flush_all().await?;

        // Additional cleanup would go here

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::internal::ObjectType;

    #[test]
    fn test_facade_default() {
        let facade = MultiModelStorageFacade::new();
        assert!(facade.supported_models().is_empty());
    }

    #[test]
    fn test_facade_config() {
        let config = MultiModelFacadeConfig::default();
        let facade = MultiModelStorageFacade::with_config(config);
        assert!(facade.vector_store.is_none());
    }

    #[tokio::test]
    async fn test_facade_stats_empty() {
        let facade = MultiModelStorageFacade::new();
        let stats = facade.get_multi_model_stats().await.unwrap();
        assert_eq!(stats.vector_count, 0);
        assert_eq!(stats.graph_node_count, 0);
    }

    #[tokio::test]
    async fn test_facade_schema_registry() {
        let facade = MultiModelStorageFacade::new();

        // Register a vector collection
        let obj = facade
            .register_vector_collection("embeddings", 768, "cosine")
            .await
            .unwrap();
        assert_eq!(obj.name, "embeddings");
        assert_eq!(obj.object_type, ObjectType::VectorCollection);

        // Get it back
        let retrieved = facade
            .get_catalog_object("default.public.embeddings")
            .await
            .unwrap();
        assert_eq!(retrieved.name, "embeddings");
    }

    #[tokio::test]
    async fn test_facade_register_multiple_types() {
        let facade = MultiModelStorageFacade::new();

        // Register different object types
        facade
            .register_vector_collection("vectors", 128, "l2")
            .await
            .unwrap();
        facade
            .register_document_collection("documents", None)
            .await
            .unwrap();
        facade.register_graph("social", true).await.unwrap();
        facade.register_log_stream("logs", 86400).await.unwrap();

        // List all objects
        let all = facade.list_catalog_objects().await;
        assert_eq!(all.len(), 4);

        // List by type
        let vectors = facade
            .list_objects_by_type(ObjectType::VectorCollection)
            .await;
        assert_eq!(vectors.len(), 1);

        let graphs = facade.list_objects_by_type(ObjectType::Graph).await;
        assert_eq!(graphs.len(), 1);
    }

    #[tokio::test]
    async fn test_facade_information_schema() {
        let facade = MultiModelStorageFacade::new();

        facade
            .register_vector_collection("embeddings", 768, "cosine")
            .await
            .unwrap();
        facade.register_graph("knowledge", true).await.unwrap();

        let info_schema = facade.information_schema();

        let tables = info_schema.tables().await;
        assert_eq!(tables.len(), 2);

        let vectors = info_schema.vector_collections().await;
        assert_eq!(vectors.len(), 1);
        assert_eq!(vectors[0].dimension, 768);

        let graphs = info_schema.graphs().await;
        assert_eq!(graphs.len(), 1);
        assert_eq!(graphs[0].graph_type, "directed");
    }
}
