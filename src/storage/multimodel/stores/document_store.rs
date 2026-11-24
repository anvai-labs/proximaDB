//! # Document Store
//!
//! Combines SST (hot tier), columnar Parquet (cold tier), and Tantivy (full-text search)
//! for MongoDB-like JSON document storage.
//!
//! ## Storage Strategy (from `/src/storage/document/mod.rs`)
//!
//! - **Hot tier**: SST engine with document-optimized blocks
//! - **Cold tier**: Parquet columnar storage
//! - **Indexes**: JSON path (B+ tree), Array (inverted), Full-text (Tantivy)

use async_trait::async_trait;
use std::sync::Arc;

use anyhow::Result;

use crate::proto::proximadb_v1::{
    DocumentCollectionConfig, DocumentFilter, DocumentUpdate, SqlObject,
};
use crate::storage::traits::{DocumentCollectionInfo, DocumentRecord, DocumentStorageOperations};

use super::super::traits::{ModelType, StoreCapabilities};

/// Configuration for the document store
#[derive(Debug, Clone)]
pub struct DocumentStoreConfig {
    /// Maximum documents in hot tier before migration
    pub hot_tier_max_documents: usize,
    /// Enable full-text search indexing
    pub enable_fulltext: bool,
    /// Default schema validation mode
    pub schema_validation: SchemaValidationMode,
}

/// Schema validation mode for documents
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaValidationMode {
    /// No validation (MongoDB-like)
    None,
    /// Validate on read only
    OnRead,
    /// Validate on write (strict)
    OnWrite,
    /// Warn but don't reject invalid documents
    WarnOnly,
}

impl Default for DocumentStoreConfig {
    fn default() -> Self {
        Self {
            hot_tier_max_documents: 100_000,
            enable_fulltext: true,
            schema_validation: SchemaValidationMode::None,
        }
    }
}

/// DocumentStore wraps the existing DocumentService for multi-model integration
///
/// ## Architecture
///
/// ```text
/// ┌─────────────────────────────────────────┐
/// │           DocumentStore                  │
/// │  ┌─────────────────────────────────────┐│
/// │  │      DocumentService                ││
/// │  │  (existing implementation)          ││
/// │  └─────────────────────────────────────┘│
/// │              │                │          │
/// │    ┌─────────▼───────┐ ┌─────▼────────┐ │
/// │    │   SST Engine    │ │ Columnar Eng │ │
/// │    │  (Hot Tier)     │ │ (Cold Tier)  │ │
/// │    │  - Real-time    │ │ - Columnar   │ │
/// │    │  - Low-latency  │ │ - Analytics  │ │
/// │    └─────────────────┘ └──────────────┘ │
/// │              │                           │
/// │    ┌─────────▼───────────────────────┐  │
/// │    │     Tantivy Index               │  │
/// │    │  (Full-text Search)             │  │
/// │    └─────────────────────────────────┘  │
/// └─────────────────────────────────────────┘
/// ```
pub struct DocumentStore {
    /// The underlying document storage operations service
    service: Option<Arc<dyn DocumentStorageOperations>>,
    /// Configuration
    config: DocumentStoreConfig,
}

impl DocumentStore {
    /// Create a new DocumentStore with the given configuration
    pub fn new(config: DocumentStoreConfig) -> Self {
        Self {
            service: None,
            config,
        }
    }

    /// Set the underlying document service
    pub fn with_service(mut self, service: Arc<dyn DocumentStorageOperations>) -> Self {
        self.service = Some(service);
        self
    }

    /// Get store capabilities
    pub fn capabilities(&self) -> StoreCapabilities {
        StoreCapabilities {
            model_type: ModelType::Document,
            supports_transactions: false, // Future: add transaction support
            supports_secondary_indexes: true, // JSON path, array, full-text indexes
            supports_acid: false,
            supports_streaming: true,
            max_recommended_records: Some(10_000_000), // 10M documents
            description:
                "MongoDB-like JSON documents with SST (hot) + Parquet (cold) + Tantivy (full-text)"
                    .to_string(),
        }
    }

    /// Get the underlying service
    pub fn service(&self) -> Option<&Arc<dyn DocumentStorageOperations>> {
        self.service.as_ref()
    }

    /// Get configuration
    pub fn config(&self) -> &DocumentStoreConfig {
        &self.config
    }

    /// Check if store is operational
    pub fn is_operational(&self) -> bool {
        self.service.is_some()
    }
}

#[async_trait]
impl DocumentStorageOperations for DocumentStore {
    async fn insert_document(
        &self,
        collection: &str,
        id: &str,
        document: SqlObject,
        indexed_paths: Vec<String>,
    ) -> Result<DocumentRecord> {
        let service = self
            .service
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Document service not configured"))?;
        service
            .insert_document(collection, id, document, indexed_paths)
            .await
    }

    async fn get_document(&self, collection: &str, id: &str) -> Result<Option<DocumentRecord>> {
        let service = self
            .service
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Document service not configured"))?;
        service.get_document(collection, id).await
    }

    async fn query_documents(
        &self,
        collection: &str,
        filter: Option<DocumentFilter>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<DocumentRecord>> {
        let service = self
            .service
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Document service not configured"))?;
        service
            .query_documents(collection, filter, limit, offset)
            .await
    }

    async fn update_document(
        &self,
        collection: &str,
        id: &str,
        updates: Vec<DocumentUpdate>,
    ) -> Result<DocumentRecord> {
        let service = self
            .service
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Document service not configured"))?;
        service.update_document(collection, id, updates).await
    }

    async fn delete_document(&self, collection: &str, id: &str) -> Result<bool> {
        let service = self
            .service
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Document service not configured"))?;
        service.delete_document(collection, id).await
    }

    async fn create_document_collection(&self, config: DocumentCollectionConfig) -> Result<String> {
        let service = self
            .service
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Document service not configured"))?;
        service.create_document_collection(config).await
    }

    async fn list_document_collections(&self) -> Result<Vec<DocumentCollectionInfo>> {
        let service = self
            .service
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Document service not configured"))?;
        service.list_document_collections().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_store_config_default() {
        let config = DocumentStoreConfig::default();
        assert_eq!(config.hot_tier_max_documents, 100_000);
        assert!(config.enable_fulltext);
        assert_eq!(config.schema_validation, SchemaValidationMode::None);
    }

    #[test]
    fn test_document_store_capabilities() {
        let store = DocumentStore::new(DocumentStoreConfig::default());
        let caps = store.capabilities();

        assert_eq!(caps.model_type, ModelType::Document);
        assert!(caps.supports_secondary_indexes);
        assert!(caps.supports_streaming);
    }
}
