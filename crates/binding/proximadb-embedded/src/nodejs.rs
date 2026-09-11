//! Node.js NAPI-RS Bindings for ProximaDB Embedded Mode
//!
//! This module provides Node.js bindings using NAPI-RS for using
//! ProximaDB as an embedded database in Node.js applications.

use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::collections::HashMap;
use std::sync::Arc;

use super::{EmbeddedConfig, EmbeddedProximaDB, StorageLocationConfig};

/// Disk configuration for multi-disk setups
#[napi(object)]
pub struct DiskConfig {
    /// Path to the storage directory
    pub path: String,
    /// Weight for data distribution (higher = more data)
    pub weight: Option<u32>,
    /// Tags for storage tier identification
    pub tags: Option<Vec<String>>,
}

/// Database configuration options
#[napi(object)]
pub struct ProximaDBConfig {
    /// Data directories (single path or array of DiskConfig)
    pub data_dirs: Option<Vec<DiskConfig>>,
    /// Single data directory path
    pub data_dir: Option<String>,
    /// Metadata directory path
    pub metadata_dir: Option<String>,
    /// Cache size in megabytes
    pub cache_size_mb: Option<i32>,
    /// Default storage engine
    pub default_engine: Option<String>,
    /// Enable write-ahead logging
    pub enable_wal: Option<bool>,
    /// WAL sync mode: "immediate", "batch", "async"
    pub wal_sync_mode: Option<String>,
}

/// Search result
#[napi(object)]
pub struct NodejsSearchResult {
    /// Vector ID
    pub id: String,
    /// Similarity score
    pub score: f64,
    /// Metadata
    pub metadata: Option<HashMap<String, String>>,
}

/// Collection information
#[napi(object)]
pub struct NodejsCollectionInfo {
    /// Collection name
    pub name: String,
    /// Vector dimension
    pub dimension: i32,
    /// Number of vectors
    pub vector_count: i64,
    /// Storage engine type
    pub engine: String,
}

/// Backwards-compat alias for [`NodejsStorageStats`].
pub type StorageStats = NodejsStorageStats;

/// Storage statistics
#[napi(object)]
pub struct NodejsStorageStats {
    /// Total vectors
    pub total_vectors: i64,
    /// Total collections
    pub total_collections: i64,
    /// Disk usage in bytes
    pub disk_usage_bytes: i64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

/// ProximaDB embedded database for Node.js
// Keep the public acronym spelling stable across napi-derive case-conversion
// changes. The JavaScript shim and published type declarations export
// `ProximaDB`, not the v3-derived `ProximaDb` spelling.
#[napi(js_name = "ProximaDB")]
pub struct ProximaDB {
    inner: Arc<EmbeddedProximaDB>,
}

#[napi]
impl ProximaDB {
    /// Create a new ProximaDB instance
    ///
    /// @param config - Configuration options
    /// @returns ProximaDB instance
    ///
    /// @example
    /// ```javascript
    /// const db = new ProximaDB({
    ///   dataDir: './my_database',
    ///   cacheSizeMb: 1024,
    ///   defaultEngine: 'sst'
    /// });
    /// ```
    #[napi(constructor)]
    pub fn new(config: Option<ProximaDBConfig>) -> Result<Self> {
        let config = config.unwrap_or_else(|| ProximaDBConfig {
            data_dirs: None,
            data_dir: Some("./data".to_string()),
            metadata_dir: None,
            cache_size_mb: Some(512),
            default_engine: Some("sst".to_string()),
            enable_wal: Some(true),
            wal_sync_mode: Some("batch".to_string()),
        });

        // Build storage locations
        let storage_locations = if let Some(disks) = config.data_dirs {
            disks
                .into_iter()
                .map(|d| StorageLocationConfig {
                    path: d.path,
                    weight: d.weight.unwrap_or(1),
                    tags: d.tags.unwrap_or_default(),
                })
                .collect()
        } else if let Some(path) = config.data_dir {
            vec![StorageLocationConfig::new(path)]
        } else {
            vec![StorageLocationConfig::new("./data")]
        };

        let metadata_path = config
            .metadata_dir
            .unwrap_or_else(|| format!("{}/metadata", storage_locations[0].path));

        let embedded_config = EmbeddedConfig {
            storage_locations,
            metadata_path,
            cache_size_mb: config.cache_size_mb.unwrap_or(512) as usize,
            default_engine: config.default_engine.unwrap_or_else(|| "sst".to_string()),
            enable_wal: config.enable_wal.unwrap_or(true),
            wal_sync_mode: config.wal_sync_mode.unwrap_or_else(|| "batch".to_string()),
            ..EmbeddedConfig::default()
        };

        let db = EmbeddedProximaDB::new(embedded_config)
            .map_err(|e| Error::from_reason(format!("Failed to create database: {}", e)))?;

        Ok(Self {
            inner: Arc::new(db),
        })
    }

    /// Create a new collection
    ///
    /// @param name - Collection name
    /// @param dimension - Vector dimension
    /// @param engine - Storage engine type (optional)
    #[napi]
    pub fn create_collection(
        &self,
        name: String,
        dimension: i32,
        engine: Option<String>,
    ) -> Result<()> {
        self.inner
            .create_collection(&name, dimension as u32, engine.as_deref())
            .map_err(|e| Error::from_reason(format!("Failed to create collection: {}", e)))
    }

    /// Delete a collection
    ///
    /// @param name - Collection name
    #[napi]
    pub fn delete_collection(&self, name: String) -> Result<()> {
        self.inner
            .delete_collection(&name)
            .map_err(|e| Error::from_reason(format!("Failed to delete collection: {}", e)))
    }

    /// Get collection information
    ///
    /// @param name - Collection name
    /// @returns Collection info or null if not found
    #[napi]
    pub fn get_collection(&self, name: String) -> Result<Option<NodejsCollectionInfo>> {
        self.inner
            .get_collection(&name)
            .map(|opt| {
                opt.map(|info| NodejsCollectionInfo {
                    name: info.name,
                    dimension: info.dimension as i32,
                    vector_count: info.vector_count as i64,
                    engine: info.engine,
                })
            })
            .map_err(|e| Error::from_reason(format!("Failed to get collection: {}", e)))
    }

    /// List all collections
    ///
    /// @returns Array of collection info
    #[napi]
    pub fn list_collections(&self) -> Result<Vec<NodejsCollectionInfo>> {
        self.inner
            .list_collections()
            .map(|collections| {
                collections
                    .into_iter()
                    .map(|info| NodejsCollectionInfo {
                        name: info.name,
                        dimension: info.dimension as i32,
                        vector_count: info.vector_count as i64,
                        engine: info.engine,
                    })
                    .collect()
            })
            .map_err(|e| Error::from_reason(format!("Failed to list collections: {}", e)))
    }

    /// Insert vectors into a collection
    ///
    /// @param collection - Collection name
    /// @param ids - Array of vector IDs
    /// @param vectors - 2D array of vectors
    /// @param metadata - Optional array of metadata objects
    /// @returns Number of vectors inserted
    ///
    /// @example
    /// ```javascript
    /// const ids = ['vec_0', 'vec_1'];
    /// const vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
    /// const metadata = [{ category: 'A' }, { category: 'B' }];
    /// const count = db.insert('embeddings', ids, vectors, metadata);
    /// ```
    #[napi]
    pub fn insert(
        &self,
        collection: String,
        ids: Vec<String>,
        vectors: Vec<Vec<f64>>,
        metadata: Option<Vec<HashMap<String, String>>>,
    ) -> Result<i32> {
        // Convert f64 to f32
        let f32_vectors: Vec<Vec<f32>> = vectors
            .into_iter()
            .map(|v| v.into_iter().map(|x| x as f32).collect())
            .collect();

        // Convert metadata to serde_json::Value if present
        let json_metadata = metadata.map(|meta_list| {
            meta_list
                .into_iter()
                .map(|meta| {
                    meta.into_iter()
                        .map(|(k, v)| (k, serde_json::Value::String(v)))
                        .collect()
                })
                .collect()
        });

        self.inner
            .insert(&collection, ids, f32_vectors, json_metadata)
            .map(|count| count as i32)
            .map_err(|e| Error::from_reason(format!("Insert failed: {}", e)))
    }

    /// Search for similar vectors
    ///
    /// @param collection - Collection name
    /// @param query - Query vector
    /// @param topK - Number of results (default: 10)
    /// @param filter - Optional filter expression
    /// @returns Array of search results
    ///
    /// @example
    /// ```javascript
    /// const query = [0.1, 0.2, 0.3];
    /// const results = db.search('embeddings', query, 10);
    /// for (const r of results) {
    ///   console.log(`${r.id}: ${r.score}`);
    /// }
    /// ```
    #[napi]
    pub fn search(
        &self,
        collection: String,
        query: Vec<f64>,
        top_k: Option<i32>,
        filter: Option<String>,
    ) -> Result<Vec<NodejsSearchResult>> {
        let f32_query: Vec<f32> = query.into_iter().map(|x| x as f32).collect();
        let k = top_k.unwrap_or(10) as usize;

        self.inner
            .search(&collection, f32_query, k, filter.as_deref())
            .map(|results| {
                results
                    .into_iter()
                    .map(|r| NodejsSearchResult {
                        id: r.id,
                        score: r.score as f64,
                        metadata: Some(r.metadata),
                    })
                    .collect()
            })
            .map_err(|e| Error::from_reason(format!("Search failed: {}", e)))
    }

    /// Flush all pending writes to disk
    #[napi]
    pub fn flush(&self) -> Result<()> {
        self.inner
            .flush()
            .map_err(|e| Error::from_reason(format!("Flush failed: {}", e)))
    }

    /// Get storage statistics
    ///
    /// @returns Storage statistics
    #[napi]
    pub fn stats(&self) -> Result<NodejsStorageStats> {
        self.inner
            .stats()
            .map(|s| NodejsStorageStats {
                total_vectors: s.total_vectors as i64,
                total_collections: s.total_collections as i64,
                disk_usage_bytes: s.disk_usage_bytes as i64,
                cache_hit_rate: s.cache_hit_rate,
            })
            .map_err(|e| Error::from_reason(format!("Failed to get stats: {}", e)))
    }
}

/// Get ProximaDB version
#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
