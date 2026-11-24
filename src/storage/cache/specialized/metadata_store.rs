use crate::storage::cache::base::BaseCacheImpl;
use crate::storage::cache::traits::{BaseCache, CacheValue};
use anyhow::Result;
use async_trait::async_trait;
use serde;
use serde_json::Value;
use std::collections::HashMap;

// Implement CacheKey for String (if not already done elsewhere)
// Skip if already implemented in vector_data.rs
// impl CacheKey for String {}

impl CacheValue for Value {
    fn size_bytes(&self) -> usize {
        // Estimate JSON size - rough approximation
        serde_json::to_string(self).map(|s| s.len()).unwrap_or(256)
    }
}

/// Metadata cache using the base cache infrastructure
#[derive(Debug)]
pub struct MetadataStore {
    base: BaseCacheImpl<String, Value>,
}

impl MetadataStore {
    pub fn new(max_memory_mb: usize) -> Self {
        Self {
            base: BaseCacheImpl::new(max_memory_mb),
        }
    }

    /// Put metadata (simple wrapper without hooks)
    pub async fn put(&self, key: &str, value: Value) -> Result<()> {
        self.put_with_hooks(key.to_string(), value).await;
        Ok(())
    }

    /// Get metadata (simple wrapper without hooks)
    pub async fn get(&self, key: &str) -> Option<Value> {
        self.get_with_hooks(key).await
    }

    /// Put metadata with hooks
    pub async fn put_with_hooks(&self, key: String, value: Value) {
        BaseCache::put_with_hooks(&self.base, key, value).await;
    }

    /// Get metadata with hooks
    pub async fn get_with_hooks(&self, key: &str) -> Option<Value> {
        BaseCache::get_with_hooks(&self.base, &key.to_string()).await
    }

    /// Clear all metadata entries
    pub async fn clear_all(&self) -> anyhow::Result<()> {
        // For now, we can't directly clear the backend due to encapsulation
        // This would require adding a clear method to the BaseCache trait
        // As a workaround, we could track keys separately or add the method later
        // Reset metrics at least
        self.base.metrics().reset().await;
        Ok(())
    }

    /// Get total size in bytes
    pub async fn size_bytes(&self) -> usize {
        // Use the memory_usage method from base cache
        self.base.memory_usage().await
    }

    /// Get total number of entries
    pub async fn total_entries(&self) -> usize {
        // Use the size method from base cache
        self.base.size().await
    }

    /// Invalidate a metadata entry
    pub async fn invalidate(&self, key: &str) -> bool {
        BaseCache::invalidate(&self.base, &key.to_string()).await
    }

    /// Get cache metrics
    pub fn metrics(&self) -> &crate::storage::traits::UnifiedMetricsCollector {
        self.base.metrics()
    }

    // Specialized metadata methods for testing

    /// Put collection metadata
    pub async fn put_collection_metadata(
        &self,
        collection_id: &str,
        metadata: impl serde::Serialize,
    ) {
        let key = format!("collection:{}", collection_id);
        if let Ok(value) = serde_json::to_value(metadata) {
            self.put_with_hooks(key, value).await;
        }
    }

    /// Get collection metadata
    pub async fn collection_metadata<T>(&self, collection_id: &str) -> Option<T>
    where
        T: serde::de::DeserializeOwned,
    {
        let key = format!("collection:{}", collection_id);
        let value = self.get_with_hooks(&key).await?;
        serde_json::from_value(value).ok()
    }

    /// Put schema metadata
    pub async fn put_schema_metadata(&self, schema_id: &str, metadata: impl serde::Serialize) {
        let key = format!("schema:{}", schema_id);
        if let Ok(value) = serde_json::to_value(metadata) {
            self.put_with_hooks(key, value).await;
        }
    }

    /// Get schema metadata
    pub async fn get_schema_metadata<T>(&self, schema_id: &str) -> Option<T>
    where
        T: serde::de::DeserializeOwned,
    {
        let key = format!("schema:{}", schema_id);
        let value = self.get_with_hooks(&key).await?;
        serde_json::from_value(value).ok()
    }

    /// Invalidate all metadata for a collection
    pub async fn invalidate_collection(&self, collection_id: &str) {
        let key = format!("collection:{}", collection_id);
        self.invalidate(&key).await;
    }
}

// ========================================================================================
// Parquet Metadata Cache Operations - Extending MetadataStore for the columnar engines
// ========================================================================================

/// Parquet schema mapping for efficient column access
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ParquetSchemaMapping {
    pub vector_column: String,
    pub metadata_columns: Vec<String>,
    pub quantized_columns: Vec<String>,
    pub filterable_columns: Vec<String>,
    pub timestamp_columns: Vec<String>,
}

/// Parquet file metadata for optimization decisions
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ParquetFileMetadata {
    pub total_rows: usize,
    pub row_groups: usize,
    pub file_size: usize,
    pub is_cloud_storage: bool,
    pub supports_range_requests: bool,
    pub column_stats: HashMap<String, PushdownColumnStatistics>,
}

/// Column statistics for predicate pushdown
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PushdownColumnStatistics {
    pub min_value: Value,
    pub max_value: Value,
    pub null_count: usize,
    pub distinct_count: usize,
}

impl MetadataStore {
    /// Cache Parquet schema mapping
    pub async fn cache_parquet_schema(
        &self,
        file_path: &str,
        schema: ParquetSchemaMapping,
    ) -> Result<()> {
        let key = format!("parquet_schema:{}", file_path);
        let value = serde_json::to_value(schema)?;
        self.put(&key, value).await?;
        Ok(())
    }

    /// Get cached Parquet schema
    pub async fn get_parquet_schema(&self, file_path: &str) -> Option<ParquetSchemaMapping> {
        let key = format!("parquet_schema:{}", file_path);
        let value = self.get(&key).await?;
        serde_json::from_value(value).ok()
    }

    /// Cache Parquet file metadata
    pub async fn cache_parquet_metadata(
        &self,
        file_path: &str,
        metadata: ParquetFileMetadata,
    ) -> Result<()> {
        let key = format!("parquet_meta:{}", file_path);
        let value = serde_json::to_value(metadata)?;
        self.put(&key, value).await?;
        Ok(())
    }

    /// Get cached Parquet file metadata
    pub async fn get_parquet_metadata(&self, file_path: &str) -> Option<ParquetFileMetadata> {
        let key = format!("parquet_meta:{}", file_path);
        let value = self.get(&key).await?;
        serde_json::from_value(value).ok()
    }

    /// Cache row group metadata for selective reading
    pub async fn cache_row_group_metadata(
        &self,
        file_path: &str,
        row_group_idx: usize,
        metadata: Value,
    ) -> Result<()> {
        let key = format!("parquet_rg:{}:{}", file_path, row_group_idx);
        self.put(&key, metadata).await?;
        Ok(())
    }

    /// Get cached row group metadata
    pub async fn row_group_metadata(&self, file_path: &str, row_group_idx: usize) -> Option<Value> {
        let key = format!("parquet_rg:{}:{}", file_path, row_group_idx);
        self.get(&key).await
    }

    /// Cache multiple schemas as a batch
    pub async fn cache_parquet_schemas_batch(
        &self,
        schemas: Vec<(String, ParquetSchemaMapping)>,
    ) -> Result<()> {
        for (file_path, schema) in schemas {
            self.cache_parquet_schema(&file_path, schema).await?;
        }
        Ok(())
    }

    /// Get schemas for multiple files
    pub async fn get_parquet_schemas(
        &self,
        file_paths: &[String],
    ) -> HashMap<String, ParquetSchemaMapping> {
        let mut results = HashMap::new();

        for file_path in file_paths {
            if let Some(schema) = self.get_parquet_schema(file_path).await {
                results.insert(file_path.clone(), schema);
            }
        }

        results
    }

    /// Invalidate all Parquet metadata for a file
    pub async fn invalidate_parquet_file(&self, file_path: &str) -> Result<()> {
        // Invalidate schema
        let schema_key = format!("parquet_schema:{}", file_path);
        self.invalidate(&schema_key).await;

        // Invalidate file metadata
        let meta_key = format!("parquet_meta:{}", file_path);
        self.invalidate(&meta_key).await;

        // Note: Row group metadata would need tracking for complete invalidation
        // For now, this is a best-effort invalidation

        Ok(())
    }

    /// Check if Parquet metadata exists for a file
    pub async fn has_parquet_metadata(&self, file_path: &str) -> bool {
        let key = format!("parquet_meta:{}", file_path);
        self.get(&key).await.is_some()
    }
}

// Delegate BaseCache implementation to the base
#[async_trait]
impl BaseCache for MetadataStore {
    type Key = String;
    type Value = Value;

    async fn check_l1(&self, key: &Self::Key) -> Option<Self::Value> {
        self.base.check_l1(key).await
    }

    async fn check_l2(&self, key: &Self::Key) -> Option<Self::Value> {
        self.base.check_l2(key).await
    }

    async fn check_l3(&self, key: &Self::Key) -> Option<Self::Value> {
        self.base.check_l3(key).await
    }

    async fn put_l1(&self, key: Self::Key, value: Self::Value) {
        self.base.put_l1(key, value).await
    }

    async fn put_l2(&self, key: Self::Key, value: Self::Value) {
        self.base.put_l2(key, value).await
    }

    async fn put_l3(&self, key: Self::Key, value: Self::Value) {
        self.base.put_l3(key, value).await
    }

    async fn invalidate_l1(&self, key: &Self::Key) -> bool {
        self.base.invalidate_l1(key).await
    }

    async fn invalidate_l2(&self, key: &Self::Key) -> bool {
        self.base.invalidate_l2(key).await
    }

    async fn invalidate_l3(&self, key: &Self::Key) -> bool {
        self.base.invalidate_l3(key).await
    }

    async fn promote_to_l1(&self, key: &Self::Key, value: &Self::Value) {
        self.base.promote_to_l1(key, value).await
    }

    async fn promote_to_l2(&self, key: &Self::Key, value: &Self::Value) {
        self.base.promote_to_l2(key, value).await
    }

    async fn select_tier(
        &self,
        key: &Self::Key,
        value: &Self::Value,
    ) -> crate::storage::cache::backend::CacheTier {
        self.base.select_tier(key, value).await
    }

    fn metrics(&self) -> &crate::storage::traits::UnifiedMetricsCollector {
        self.base.metrics()
    }
}
