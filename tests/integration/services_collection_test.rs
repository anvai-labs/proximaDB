//! Unit tests for CollectionService

use anyhow::Result;
use std::sync::Arc;
use tempfile::TempDir;

use proximadb::core::config::StorageConfig;
use proximadb::proto::proximadb_v1::{CollectionConfig, DistanceMetric, StorageEngine};
use proximadb::services::collection_service::CollectionService;
use proximadb::storage::tenant::{TenantConfig, TenantContext, TenantManager};

/// Create test collection service
async fn create_test_service() -> Result<(Arc<CollectionService>, TempDir)> {
    let temp_dir = TempDir::new()?;

    // Create collection service with storage config rooted at the temp dir
    let mut storage_config = StorageConfig::default();
    storage_config.metadata_url = format!("file://{}/metadata", temp_dir.path().display());
    let service = Arc::new(CollectionService::new(storage_config).await?);

    Ok((service, temp_dir))
}

async fn create_test_service_with_tenant_manager(
    max_collections: u32,
) -> Result<(
    Arc<CollectionService>,
    TempDir,
    TenantContext,
    TenantContext,
)> {
    let temp_dir = TempDir::new()?;

    let tenant_manager = Arc::new(TenantManager::new());
    let mut tenant_config = TenantConfig::default();
    tenant_config.resource_limits.max_collections = max_collections;

    let tenant_a = tenant_manager
        .create_tenant("tenant_a".to_string(), tenant_config.clone())
        .await?;
    let tenant_b = tenant_manager
        .create_tenant("tenant_b".to_string(), tenant_config)
        .await?;

    let mut storage_config = StorageConfig::default();
    storage_config.metadata_url = format!("file://{}/metadata", temp_dir.path().display());
    let service = Arc::new(
        CollectionService::new(storage_config)
            .await?
            .with_tenant_manager(tenant_manager),
    );

    Ok((service, temp_dir, tenant_a, tenant_b))
}

#[tokio::test]
async fn test_create_collection() -> Result<()> {
    let (service, _temp_dir) = create_test_service().await?;

    // Create collection config
    let config = CollectionConfig {
        name: "test_collection".to_string(),
        embedding_config: None,
        dimension: 384,
        distance_metric: Some(DistanceMetric::Cosine as i32),
        storage_engine: Some(StorageEngine::Viper as i32),
        filterable_columns: vec![],
        index_configs: vec![],
        quantization: None,
        primary_index: Some("HNSW".to_string()),
        auto_index_selection: Some(false),
        description: Some("Test collection".to_string()),
        tags: vec!["test".to_string()],
        owner: Some("test_user".to_string()),
        embedding_models: vec![],
        storage_config: None,
        record_schema: None,
        enable_proxima_record: None,
        text_columns: vec![],
        text_storage_configs: vec![],
        enable_dual_use_embeddings: None,
        canonical_embedding_precision: None,
        permitted_principals: vec![],
        index_policy: None,
        pax_vector_quant: None,
    };

    let response = service.create_collection(&config).await?;

    assert!(response.success);
    assert!(response.storage_path.is_some());

    Ok(())
}

#[tokio::test]
async fn test_get_collection() -> Result<()> {
    let (service, _temp_dir) = create_test_service().await?;

    // Create a collection first
    let config = CollectionConfig {
        name: "test_get".to_string(),
        embedding_config: None,
        dimension: 256,
        distance_metric: Some(DistanceMetric::Euclidean as i32),
        storage_engine: Some(StorageEngine::Sst as i32),
        filterable_columns: vec![],
        index_configs: vec![],
        quantization: None,
        primary_index: Some("FLAT".to_string()),
        auto_index_selection: Some(false),
        description: None,
        tags: vec![],
        owner: None,
        embedding_models: vec![],
        storage_config: None,
        record_schema: None,
        enable_proxima_record: None,
        text_columns: vec![],
        text_storage_configs: vec![],
        enable_dual_use_embeddings: None,
        canonical_embedding_precision: None,
        permitted_principals: vec![],
        index_policy: None,
        pax_vector_quant: None,
    };

    let create_response = service.create_collection(&config).await?;
    assert!(create_response.success);

    // Get by name
    let collection = service
        .get_collection_with_tenant_context("test_get", None)
        .await?;
    assert!(collection.is_some());

    let collection = collection.unwrap();
    assert_eq!(collection.config.as_ref().unwrap().name, "test_get");

    Ok(())
}

#[tokio::test]
async fn test_list_collections() -> Result<()> {
    let (service, _temp_dir) = create_test_service().await?;

    // Create multiple collections
    for i in 0..3 {
        let config = CollectionConfig {
            name: format!("collection_{}", i),
            embedding_config: None,
            dimension: 128,
            distance_metric: Some(DistanceMetric::Cosine as i32),
            storage_engine: Some(StorageEngine::Viper as i32),
            filterable_columns: vec![],
            index_configs: vec![],
            quantization: None,
            primary_index: Some("HNSW".to_string()),
            auto_index_selection: Some(false),
            description: None,
            tags: vec![],
            owner: None,
            embedding_models: vec![],
            storage_config: None,
            record_schema: None,
            enable_proxima_record: None,
            text_columns: vec![],
            text_storage_configs: vec![],
            enable_dual_use_embeddings: None,
            canonical_embedding_precision: None,
            permitted_principals: vec![],
            index_policy: None,
            pax_vector_quant: None,
        };

        let response = service.create_collection(&config).await?;
        assert!(response.success);
    }

    // List all collections
    let collections = service.list_collections().await?;
    assert_eq!(collections.len(), 3);

    Ok(())
}

#[tokio::test]
async fn test_delete_collection() -> Result<()> {
    let (service, _temp_dir) = create_test_service().await?;

    // Create a collection
    let config = CollectionConfig {
        name: "test_delete".to_string(),
        embedding_config: None,
        dimension: 64,
        distance_metric: Some(DistanceMetric::Manhattan as i32),
        storage_engine: Some(StorageEngine::Sst as i32),
        filterable_columns: vec![],
        index_configs: vec![],
        quantization: None,
        primary_index: Some("FLAT".to_string()),
        auto_index_selection: Some(false),
        description: None,
        tags: vec![],
        owner: None,
        embedding_models: vec![],
        storage_config: None,
        record_schema: None,
        enable_proxima_record: None,
        text_columns: vec![],
        text_storage_configs: vec![],
        enable_dual_use_embeddings: None,
        canonical_embedding_precision: None,
        permitted_principals: vec![],
        index_policy: None,
        pax_vector_quant: None,
    };

    let create_response = service.create_collection(&config).await?;
    assert!(create_response.success);

    // Delete the collection
    let delete_response = service.delete_collection("test_delete").await?;
    assert!(delete_response.success);

    // Verify it's gone
    let collection = service
        .get_collection_with_tenant_context("test_delete", None)
        .await?;
    assert!(collection.is_none());

    Ok(())
}

#[tokio::test]
async fn test_tenant_scoped_collection_access_and_delete() -> Result<()> {
    let (service, _temp_dir, tenant_a, tenant_b) =
        create_test_service_with_tenant_manager(10).await?;

    let config = CollectionConfig {
        name: "tenant_coll_one".to_string(),
        embedding_config: None,
        dimension: 128,
        distance_metric: Some(DistanceMetric::Cosine as i32),
        storage_engine: Some(StorageEngine::Sst as i32),
        filterable_columns: vec![],
        index_configs: vec![],
        quantization: None,
        primary_index: Some("HNSW".to_string()),
        auto_index_selection: Some(false),
        description: Some("Tenant scoped collection".to_string()),
        tags: vec!["purpose:test".to_string()],
        owner: None,
        embedding_models: vec![],
        storage_config: None,
        record_schema: None,
        enable_proxima_record: None,
        text_columns: vec![],
        text_storage_configs: vec![],
        enable_dual_use_embeddings: None,
        canonical_embedding_precision: None,
        permitted_principals: vec![],
        index_policy: None,
        pax_vector_quant: None,
    };

    let create_response = service
        .create_collection_with_tenant_context(&config, Some(&tenant_a))
        .await?;
    assert!(create_response.success);

    let own_collection = service
        .get_collection_with_tenant_context("tenant_coll_one", Some(&tenant_a))
        .await?;
    assert!(own_collection.is_some());

    let denied_collection = service
        .get_collection_with_tenant_context("tenant_coll_one", Some(&tenant_b))
        .await?;
    assert!(denied_collection.is_none());

    let denied_delete = service
        .delete_collection_with_tenant_context("tenant_coll_one", Some(&tenant_b))
        .await?;
    assert!(!denied_delete.success);
    assert!(
        denied_delete
            .error_code
            .as_deref()
            .unwrap_or_default()
            .contains("TENANT_ACCESS_DENIED")
    );

    let allowed_delete = service
        .delete_collection_with_tenant_context("tenant_coll_one", Some(&tenant_a))
        .await?;
    assert!(allowed_delete.success);

    let deleted = service
        .get_collection_with_tenant_context("tenant_coll_one", Some(&tenant_a))
        .await?;
    assert!(deleted.is_none());

    Ok(())
}

/// TD-064 S2: the pgwire vector-search routing gate resolves its target through
/// `CollectionPort::get_collection(name, tenant_scope)` and denies the search on
/// `Ok(None)` or `Err`. This pins that exact trait-method contract so the
/// structural tenant isolation the pgwire search path now relies on cannot
/// silently regress:
///   * owner tenant                        → Ok(Some)  (search allowed)
///   * cross tenant                        → Ok(None)  (gate → "relation does not exist")
///   * missing tenant in multi-tenant mode → Err       (gate fails closed)
#[tokio::test]
async fn test_collection_port_get_collection_is_tenant_scoped() -> Result<()> {
    use proximadb_runtime::CollectionPort;

    let (service, _temp_dir, tenant_a, tenant_b) =
        create_test_service_with_tenant_manager(10).await?;

    let config = CollectionConfig {
        name: "scoped_vectors".to_string(),
        embedding_config: None,
        dimension: 8,
        distance_metric: Some(DistanceMetric::Cosine as i32),
        storage_engine: Some(StorageEngine::Sst as i32),
        filterable_columns: vec![],
        index_configs: vec![],
        quantization: None,
        primary_index: Some("HNSW".to_string()),
        auto_index_selection: Some(false),
        description: Some("Tenant scoped vectors".to_string()),
        tags: vec!["purpose:test".to_string()],
        owner: None,
        embedding_models: vec![],
        storage_config: None,
        record_schema: None,
        enable_proxima_record: None,
        text_columns: vec![],
        text_storage_configs: vec![],
        enable_dual_use_embeddings: None,
        canonical_embedding_precision: None,
        permitted_principals: vec![],
        index_policy: None,
        pax_vector_quant: None,
    };

    service
        .create_collection_with_tenant_context(&config, Some(&tenant_a))
        .await?;

    // Owner tenant resolves the collection → pgwire gate allows the search.
    let owned = CollectionPort::get_collection(
        service.as_ref(),
        "scoped_vectors",
        Some(&tenant_a.tenant_id),
    )
    .await?;
    assert!(
        owned.is_some(),
        "owner tenant must resolve its own collection"
    );

    // Cross-tenant resolves to None → pgwire gate denies (relation does not exist).
    let cross = CollectionPort::get_collection(
        service.as_ref(),
        "scoped_vectors",
        Some(&tenant_b.tenant_id),
    )
    .await?;
    assert!(
        cross.is_none(),
        "cross-tenant vector-search access must be denied (structural isolation)"
    );

    // Missing tenant under multi-tenant mode fails closed → pgwire gate denies.
    let missing = CollectionPort::get_collection(service.as_ref(), "scoped_vectors", None).await;
    assert!(
        missing.is_err(),
        "missing tenant in multi-tenant mode must fail closed, not resolve unscoped"
    );

    Ok(())
}

#[tokio::test]
async fn test_tenant_collection_limit_enforced() -> Result<()> {
    let (service, _temp_dir, tenant_a, _tenant_b) =
        create_test_service_with_tenant_manager(1).await?;

    let config_one = CollectionConfig {
        name: "tenant_lim_one".to_string(),
        embedding_config: None,
        dimension: 64,
        distance_metric: Some(DistanceMetric::Cosine as i32),
        storage_engine: Some(StorageEngine::Sst as i32),
        filterable_columns: vec![],
        index_configs: vec![],
        quantization: None,
        primary_index: Some("FLAT".to_string()),
        auto_index_selection: Some(false),
        description: None,
        tags: vec![],
        owner: None,
        embedding_models: vec![],
        storage_config: None,
        record_schema: None,
        enable_proxima_record: None,
        text_columns: vec![],
        text_storage_configs: vec![],
        enable_dual_use_embeddings: None,
        canonical_embedding_precision: None,
        permitted_principals: vec![],
        index_policy: None,
        pax_vector_quant: None,
    };

    let config_two = CollectionConfig {
        name: "tenant_lim_two".to_string(),
        ..config_one.clone()
    };

    let first = service
        .create_collection_with_tenant_context(&config_one, Some(&tenant_a))
        .await?;
    assert!(first.success);

    let second = service
        .create_collection_with_tenant_context(&config_two, Some(&tenant_a))
        .await?;
    assert!(!second.success);
    assert!(
        second
            .error_code
            .as_deref()
            .unwrap_or_default()
            .contains("TENANT_COLLECTION_LIMIT_EXCEEDED")
    );

    Ok(())
}

#[tokio::test]
async fn test_tenant_scoped_collection_listing() -> Result<()> {
    let (service, _temp_dir, tenant_a, tenant_b) =
        create_test_service_with_tenant_manager(10).await?;

    let config_a = CollectionConfig {
        name: "tenant_list_a".to_string(),
        embedding_config: None,
        dimension: 32,
        distance_metric: Some(DistanceMetric::Cosine as i32),
        storage_engine: Some(StorageEngine::Sst as i32),
        filterable_columns: vec![],
        index_configs: vec![],
        quantization: None,
        primary_index: Some("FLAT".to_string()),
        auto_index_selection: Some(false),
        description: None,
        tags: vec![],
        owner: None,
        embedding_models: vec![],
        storage_config: None,
        record_schema: None,
        enable_proxima_record: None,
        text_columns: vec![],
        text_storage_configs: vec![],
        enable_dual_use_embeddings: None,
        canonical_embedding_precision: None,
        permitted_principals: vec![],
        index_policy: None,
        pax_vector_quant: None,
    };

    let config_b = CollectionConfig {
        name: "tenant_list_b".to_string(),
        ..config_a.clone()
    };

    assert!(
        service
            .create_collection_with_tenant_context(&config_a, Some(&tenant_a))
            .await?
            .success
    );
    assert!(
        service
            .create_collection_with_tenant_context(&config_b, Some(&tenant_b))
            .await?
            .success
    );

    let tenant_a_collections = service
        .list_collections_with_tenant_context(Some(&tenant_a))
        .await?;
    assert_eq!(tenant_a_collections.len(), 1);
    assert_eq!(
        tenant_a_collections[0]
            .config
            .as_ref()
            .map(|cfg| cfg.name.as_str()),
        Some("tenant_list_a")
    );

    let tenant_b_collections = service
        .list_collections_with_tenant_context(Some(&tenant_b))
        .await?;
    assert_eq!(tenant_b_collections.len(), 1);
    assert_eq!(
        tenant_b_collections[0]
            .config
            .as_ref()
            .map(|cfg| cfg.name.as_str()),
        Some("tenant_list_b")
    );

    Ok(())
}
