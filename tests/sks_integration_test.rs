/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! Integration tests for Semantic Knowledge Store (SKS)

#[cfg(test)]
mod sks_integration_tests {
    use anyhow::Result;
    use chrono::Utc;
    use proximadb::proto::proximadb_v1::{
        EmbeddingVersion, Entity, Modality, Provenance, Relation, TypedField, TypedMetadata,
    };
    use proximadb::storage::entity_store::CsrRelationsStore as InMemoryRelationsStore;
    use proximadb::storage::entity_store::InMemoryProvenanceRegistry;
    use proximadb::storage::entity_store::{EntityStore, ProximaEntityStore};
    use proximadb::storage::traits::StorageEngineStrategy;
    use proximadb::storage::traits::{
        CompactionParameters, CompactionResult, FlushParameters, FlushResult,
    };
    use proximadb_records::ProximaRecord;
    use std::collections::HashMap;
    use std::sync::Arc;

    /// Create a test entity store
    async fn create_test_store() -> Arc<ProximaEntityStore> {
        // Create mock storage engine
        let storage_engine = create_mock_storage_engine();

        // Create relations store (CsrRelationsStore is in-memory; the
        // older signature took a storage_engine arg but the canonical
        // type is zero-arg — entity_store_legacy uses RwLock<HashMap>).
        let relations_store = Arc::new(InMemoryRelationsStore::new());

        // Create provenance registry (also zero-arg now).
        let provenance_registry = Arc::new(InMemoryProvenanceRegistry::new());

        // Create entity store
        Arc::new(ProximaEntityStore::new(
            storage_engine,
            relations_store,
            provenance_registry,
        ))
    }

    /// Create a mock storage engine for testing
    fn create_mock_storage_engine() -> Arc<dyn proximadb::storage::traits::UnifiedStorageEngine> {
        struct MockStorageEngine;

        #[async_trait::async_trait]
        impl proximadb::storage::traits::UnifiedStorageEngine for MockStorageEngine {
            fn engine_name(&self) -> &'static str {
                "mock"
            }
            fn engine_version(&self) -> &'static str {
                "1.0.0"
            }
            fn strategy(&self) -> StorageEngineStrategy {
                StorageEngineStrategy::Nova
            }

            async fn do_flush(&self, _params: &FlushParameters) -> Result<FlushResult> {
                Ok(FlushResult {
                    success: true,
                    collections_affected: vec![],
                    entries_flushed: Some(0),
                    bytes_written: Some(0),
                    files_created: Some(0),
                    file_paths: vec![],
                    duration_ms: Some(0),
                    completed_at: Utc::now(),
                    engine_metrics: HashMap::new(),
                    compaction_triggered: false,
                    compaction_error: None,
                    flushed_batch_ids: vec![],
                })
            }

            async fn do_compact(&self, _params: &CompactionParameters) -> Result<CompactionResult> {
                Ok(CompactionResult {
                    success: true,
                    collections_affected: vec![],
                    entries_processed: Some(0),
                    entries_removed: Some(0),
                    bytes_read: Some(0),
                    bytes_written: Some(0),
                    input_files: Some(0),
                    output_files: Some(0),
                    duration_ms: Some(0),
                    completed_at: Utc::now(),
                    engine_metrics: HashMap::new(),
                })
            }

            async fn collect_engine_metrics(&self) -> Result<HashMap<String, serde_json::Value>> {
                Ok(HashMap::new())
            }

            async fn vector_by_id(
                &self,
                _collection_id: &str,
                _base_path: &str,
                _vector_id: &str,
            ) -> Result<Option<ProximaRecord>> {
                Ok(None)
            }

            async fn search_vectors_unified(
                &self,
                _ctx: &proximadb::storage::traits::StorageQueryContext,
            ) -> Result<Vec<proximadb::core::search::results::OptimizedSearchRecord>> {
                Ok(vec![])
            }
        }

        Arc::new(MockStorageEngine)
    }

    /// Create a test entity with embeddings
    fn create_test_entity(id: &str) -> Entity {
        let embedding = EmbeddingVersion {
            model_id: "test-model".to_string(),
            model_version: "v1".to_string(),
            vector: vec![0.1, 0.2, 0.3, 0.4, 0.5],
            dimension: 5,
            created_at_ms: 1609459200000, // Unix epoch milliseconds
            model_params: Default::default(),
            modality: Modality::Text as i32,
        };

        let provenance = Provenance {
            source_id: "test-source".to_string(),
            chunk_id: "chunk-1".to_string(),
            chunk_position: 0,
            extraction_method: "test-extraction".to_string(),
            extracted_at_ms: 1609459200000, // Unix epoch milliseconds
            metadata: Default::default(),
        };

        Entity {
            id: id.to_string(),
            embeddings: vec![embedding],
            typed_metadata: None,
            flexible_metadata: HashMap::new(),
            provenance: Some(provenance),
            relations: vec![],
            temporal: None,
            collection_id: "test-collection".to_string(),
        }
    }

    #[tokio::test]
    async fn test_entity_upsert_and_retrieve() {
        let store = create_test_store().await;
        let entity = create_test_entity("test-entity-1");

        // Upsert entity
        let result = store.upsert_entity("test-collection", entity.clone()).await;
        assert!(result.is_ok());
        let entity_id = result.unwrap();
        assert_eq!(entity_id, "test-entity-1");

        // Retrieve entity
        let retrieved = store
            .get_entity(
                "test-collection",
                &entity_id,
                true,  // include_embeddings
                false, // include_relations
            )
            .await;

        assert!(retrieved.is_ok());
        let retrieved_entity = retrieved.unwrap();
        assert!(retrieved_entity.is_some());

        let retrieved_entity = retrieved_entity.unwrap();
        assert_eq!(retrieved_entity.id, entity_id);
        assert_eq!(retrieved_entity.embeddings.len(), 1);
    }

    #[tokio::test]
    async fn test_entity_with_relations() {
        let store = create_test_store().await;

        // Create two entities
        let entity1 = create_test_entity("entity-1");
        let entity2 = create_test_entity("entity-2");

        // Add a relation
        let relation = Relation {
            source_entity_id: "entity-1".to_string(),
            target_entity_id: "entity-2".to_string(),
            relation_type: "cites".to_string(),
            weight: 1.0,
            created_at_ms: 1609459200000, // Unix epoch milliseconds
            properties: Default::default(),
        };

        let mut entity1_with_relation = entity1;
        entity1_with_relation.relations.push(relation);

        // Upsert entities
        let _ = store
            .upsert_entity("test-collection", entity1_with_relation)
            .await
            .unwrap();
        let _ = store
            .upsert_entity("test-collection", entity2)
            .await
            .unwrap();

        // Retrieve entity with relations
        let retrieved = store
            .get_entity(
                "test-collection",
                "entity-1",
                false, // include_embeddings
                true,  // include_relations
            )
            .await
            .unwrap();

        assert!(retrieved.is_some());
        let entity = retrieved.unwrap();
        assert_eq!(entity.relations.len(), 1);
        assert_eq!(entity.relations[0].relation_type, "cites");
    }

    #[tokio::test]
    async fn test_entity_search() {
        let store = create_test_store().await;

        // Create and upsert multiple entities
        for i in 0..5 {
            let entity = create_test_entity(&format!("entity-{}", i));
            let _ = store
                .upsert_entity("test-collection", entity)
                .await
                .unwrap();
        }

        // Search entities
        let query_vector = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let results = store
            .search_entities(
                "test-collection",
                Some(query_vector),
                None, // metadata_filter
                // None, // temporal_filter
                3, // top_k
            )
            .await;

        assert!(results.is_ok());
        let search_results = results.unwrap();
        assert!(search_results.len() <= 3);
    }

    #[tokio::test]
    async fn test_entity_deletion() {
        let store = create_test_store().await;
        let entity = create_test_entity("entity-to-delete");

        // Upsert entity
        let _ = store
            .upsert_entity("test-collection", entity)
            .await
            .unwrap();

        // Verify it exists
        let exists = store
            .get_entity("test-collection", "entity-to-delete", false, false)
            .await
            .unwrap();
        assert!(exists.is_some());

        // Delete entity
        let deleted = store
            .delete_entity(
                "test-collection",
                "entity-to-delete",
                true, // hard_delete
            )
            .await
            .unwrap();
        assert!(deleted);

        // Verify it's deleted
        let after_delete = store
            .get_entity("test-collection", "entity-to-delete", false, false)
            .await
            .unwrap();
        assert!(after_delete.is_none());
    }

    #[tokio::test]
    async fn test_entity_with_typed_metadata() {
        let store = create_test_store().await;

        // Create entity with typed metadata
        let mut entity = create_test_entity("entity-with-metadata");

        let mut fields = std::collections::HashMap::new();
        fields.insert(
            "author".to_string(),
            TypedField {
                value: Some(
                    proximadb::proto::proximadb_v1::typed_field::Value::StringValue(
                        "John Doe".to_string(),
                    ),
                ),
                indexed: true,
                filterable: true,
            },
        );
        fields.insert(
            "year".to_string(),
            TypedField {
                value: Some(proximadb::proto::proximadb_v1::typed_field::Value::IntValue(2024)),
                indexed: true,
                filterable: true,
            },
        );

        entity.typed_metadata = Some(TypedMetadata { fields });

        // Upsert entity
        let _ = store
            .upsert_entity("test-collection", entity.clone())
            .await
            .unwrap();

        // Retrieve and verify metadata
        let retrieved = store
            .get_entity("test-collection", "entity-with-metadata", false, false)
            .await
            .unwrap()
            .unwrap();

        assert!(retrieved.typed_metadata.is_some());
        let metadata = retrieved.typed_metadata.unwrap();
        assert_eq!(metadata.fields.len(), 2);
        assert!(metadata.fields.contains_key("author"));
        assert!(metadata.fields.contains_key("year"));
    }
}
