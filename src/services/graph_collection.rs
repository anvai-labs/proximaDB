/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! # Graph Collection Service
//!
//! Manages graph collections metadata, schemas, and configurations following
//! the same pattern as vector collections service.

use crate::proto::proximadb_v1::{CreateGraphRequest, GraphCollection, GraphSchema};
use dashmap::DashMap;
use proximadb_kernel::error::ProximaDBError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Convenience result type alias using ProximaDBError.
type Result<T> = std::result::Result<T, ProximaDBError>;

/// Serializable graph collection metadata for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GraphCollectionMetadata {
    /// Unique identifier for this graph collection
    graph_id: String,
    /// Human-readable name for the graph collection
    name: String,
    /// User-provided description of the graph collection
    description: String,
    /// Unix timestamp when the graph was created
    created_at: i64,
    /// Unix timestamp when the graph was last modified
    updated_at: i64,
    /// Serialized graph schema as JSON for flexibility
    schema_json: Option<String>,
    /// Serialized storage configuration as JSON
    storage_config_json: Option<String>,
    /// Serialized engine configuration as JSON
    engine_config_json: Option<String>,
    /// Serialized access control rules as JSON
    access_control_json: Option<String>,
}

impl GraphCollectionMetadata {
    /// Convert a `GraphCollection` into its serializable metadata form.
    fn from_collection(collection: &GraphCollection) -> Self {
        Self {
            graph_id: collection.graph_id.clone(),
            name: collection.name.clone(),
            description: collection.description.clone(),
            created_at: collection.created_at,
            updated_at: collection.updated_at,
            schema_json: collection
                .schema
                .as_ref()
                .map(|s| serde_json::to_string(s).unwrap_or_default()),
            storage_config_json: collection
                .storage_config
                .as_ref()
                .map(|s| serde_json::to_string(s).unwrap_or_default()),
            engine_config_json: collection
                .engine_config
                .as_ref()
                .map(|s| serde_json::to_string(s).unwrap_or_default()),
            access_control_json: collection
                .access_control
                .as_ref()
                .map(|s| serde_json::to_string(s).unwrap_or_default()),
        }
    }

    /// Reconstruct a `GraphCollection` from this persisted metadata.
    fn to_collection(&self) -> GraphCollection {
        GraphCollection {
            graph_id: self.graph_id.clone(),
            name: self.name.clone(),
            description: self.description.clone(),
            created_at: self.created_at,
            updated_at: self.updated_at,
            schema: self
                .schema_json
                .as_ref()
                .and_then(|s| serde_json::from_str(s).ok()),
            storage_config: self
                .storage_config_json
                .as_ref()
                .and_then(|s| serde_json::from_str(s).ok()),
            engine_config: self
                .engine_config_json
                .as_ref()
                .and_then(|s| serde_json::from_str(s).ok()),
            access_control: self
                .access_control_json
                .as_ref()
                .and_then(|s| serde_json::from_str(s).ok()),
            stats: None, // Stats are runtime-computed
        }
    }
}

/// Persistent metadata store for all graph collections
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct GraphMetadataStore {
    /// Version for format compatibility
    version: u32,
    /// All graph collection metadata
    collections: HashMap<String, GraphCollectionMetadata>,
}

/// Graph Collection Service - manages graph metadata and configurations
pub struct GraphCollectionService {
    /// In-memory cache of graph collections metadata
    collections: Arc<DashMap<String, Arc<GraphCollection>>>,

    /// Path to persistence file
    persistence_path: PathBuf,

    /// Object-store filesystem for URL-backed metadata. `None` preserves the
    /// local temp-file + rename implementation used by embedded/local mode.
    persistence_filesystem: Option<Arc<dyn crate::storage::persistence::filesystem::FileSystem>>,

    /// Scheme-qualified object path when `persistence_filesystem` is set.
    persistence_url: Option<String>,

    /// Lock for persistence operations
    persistence_lock: Arc<RwLock<()>>,

    /// Configuration
    max_graphs: usize,
    /// Maximum number of metadata entries to cache in memory
    #[allow(dead_code)]
    metadata_cache_size: usize,
}

impl GraphCollectionService {
    /// Default persistence path
    const DEFAULT_PERSISTENCE_PATH: &'static str = "/tmp/proximadb/metadata/graph_collections.json";

    /// Create a new graph collection service with default persistence path
    pub fn new() -> Self {
        Self::new_with_path(PathBuf::from(Self::DEFAULT_PERSISTENCE_PATH))
    }

    /// Create a new graph collection service with a custom persistence path
    pub fn new_with_path(persistence_path: PathBuf) -> Self {
        Self {
            collections: Arc::new(DashMap::new()),
            persistence_path,
            persistence_filesystem: None,
            persistence_url: None,
            persistence_lock: Arc::new(RwLock::new(())),
            max_graphs: 1000,
            metadata_cache_size: 100,
        }
    }

    /// Create a new graph collection service with auto-recovery from disk
    ///
    /// This is the recommended constructor for production use. It automatically
    /// loads persisted graph collection metadata from disk, ensuring data
    /// survives restarts.
    ///
    /// # Example
    /// ```ignore
    /// let service = GraphCollectionService::new_with_recovery().await?;
    /// ```
    pub async fn new_with_recovery() -> Result<Self> {
        Self::new_with_recovery_at(PathBuf::from(Self::DEFAULT_PERSISTENCE_PATH)).await
    }

    /// Create a new graph collection service with auto-recovery at a custom path
    ///
    /// Loads persisted graph collection metadata from the specified path.
    pub async fn new_with_recovery_at(persistence_path: PathBuf) -> Result<Self> {
        let service = Self::new_with_path(persistence_path);

        // Load persisted collections from disk
        match service.load_from_disk().await {
            Ok(count) => {
                if count > 0 {
                    info!(
                        "Graph collection service recovered {} collections from disk",
                        count
                    );
                } else {
                    debug!("Graph collection service started with no persisted collections");
                }
            }
            Err(e) => {
                warn!(
                    "Failed to load graph collections from disk: {}. Starting with empty state.",
                    e
                );
                // Continue with empty state - don't fail initialization
            }
        }

        Ok(service)
    }

    /// Create a graph collection service whose sidecar lives under an object-
    /// store metadata URL. All reads/writes go through the unified filesystem;
    /// no URL-shaped path reaches `std::fs`/`tokio::fs`.
    pub async fn new_with_recovery_at_url(
        persistence_url: String,
        filesystem_factory: Arc<crate::storage::persistence::filesystem::FilesystemFactory>,
    ) -> Result<Self> {
        let filesystem = filesystem_factory
            .get_filesystem(&persistence_url)
            .map_err(|e| {
                ProximaDBError::Internal(format!(
                    "Failed to open graph metadata filesystem at {persistence_url}: {e}"
                ))
            })?;
        let service = Self {
            collections: Arc::new(DashMap::new()),
            // Never used by the URL-backed branch; retained to keep the local
            // constructor and its tests byte-identical.
            persistence_path: PathBuf::new(),
            persistence_filesystem: Some(filesystem),
            persistence_url: Some(persistence_url),
            persistence_lock: Arc::new(RwLock::new(())),
            max_graphs: 1000,
            metadata_cache_size: 100,
        };
        service.load_from_disk().await?;
        Ok(service)
    }

    /// Load graph collections from persistent storage
    pub async fn load_from_disk(&self) -> Result<usize> {
        let _lock = self.persistence_lock.read().await;

        if let (Some(filesystem), Some(url)) = (&self.persistence_filesystem, &self.persistence_url)
        {
            if !filesystem.exists(url).await.map_err(|e| {
                ProximaDBError::Internal(format!("Failed to probe graph metadata at {url}: {e}"))
            })? {
                debug!("Graph metadata object does not exist yet: {}", url);
                return Ok(0);
            }
            let contents = filesystem.read(url).await.map_err(|e| {
                ProximaDBError::Internal(format!("Failed to read graph metadata object {url}: {e}"))
            })?;
            return self.load_serialized_store(&contents);
        }

        if !self.persistence_path.exists() {
            debug!(
                "Graph metadata file does not exist yet: {:?}",
                self.persistence_path
            );
            return Ok(0);
        }

        let contents = fs::read_to_string(&self.persistence_path)
            .await
            .map_err(|e| {
                ProximaDBError::Internal(format!("Failed to read graph metadata file: {}", e))
            })?;

        self.load_serialized_store(contents.as_bytes())
    }

    fn load_serialized_store(&self, contents: &[u8]) -> Result<usize> {
        let store: GraphMetadataStore = serde_json::from_slice(contents).map_err(|e| {
            ProximaDBError::Internal(format!("Failed to parse graph metadata: {}", e))
        })?;
        let count = store.collections.len();
        for (graph_id, metadata) in store.collections {
            let collection = Arc::new(metadata.to_collection());
            self.collections.insert(graph_id, collection);
        }

        info!("Loaded {} graph collections from persistent storage", count);
        Ok(count)
    }

    /// Save all graph collections to persistent storage
    async fn save_to_disk(&self) -> Result<()> {
        let _lock = self.persistence_lock.write().await;

        // Ensure directory exists
        if let Some(parent) = self.persistence_path.parent() {
            fs::create_dir_all(parent).await.map_err(|e| {
                ProximaDBError::Internal(format!(
                    "Failed to create graph metadata directory: {}",
                    e
                ))
            })?;
        }

        // Build the store from in-memory collections
        let mut store = GraphMetadataStore {
            version: 1,
            collections: HashMap::new(),
        };

        for entry in self.collections.iter() {
            let collection = entry.value();
            let metadata = GraphCollectionMetadata::from_collection(collection);
            store.collections.insert(entry.key().clone(), metadata);
        }

        // Write to temporary file, then rename for atomicity
        let temp_path = self.persistence_path.with_extension("json.tmp");
        let contents = serde_json::to_string_pretty(&store).map_err(|e| {
            ProximaDBError::Internal(format!("Failed to serialize graph metadata: {}", e))
        })?;

        if let (Some(filesystem), Some(url)) = (&self.persistence_filesystem, &self.persistence_url)
        {
            filesystem
                .write_atomic(url, contents.as_bytes(), None)
                .await
                .map_err(|e| {
                    ProximaDBError::Internal(format!(
                        "Failed to persist graph metadata object {url}: {e}"
                    ))
                })?;
            debug!(
                "Saved {} graph collections to object storage at {}",
                store.collections.len(),
                url
            );
            return Ok(());
        }

        fs::write(&temp_path, &contents).await.map_err(|e| {
            ProximaDBError::Internal(format!("Failed to write graph metadata: {}", e))
        })?;

        fs::rename(&temp_path, &self.persistence_path)
            .await
            .map_err(|e| {
                ProximaDBError::Internal(format!("Failed to finalize graph metadata file: {}", e))
            })?;

        debug!(
            "Saved {} graph collections to persistent storage",
            store.collections.len()
        );
        Ok(())
    }

    /// Create a new graph collection
    pub async fn create_graph(&self, request: CreateGraphRequest) -> Result<Arc<GraphCollection>> {
        let graph_id = &request.graph_id;

        // Check if graph already exists
        if self.collections.contains_key(graph_id) {
            return Err(ProximaDBError::InvalidInput(format!(
                "Graph collection '{}' already exists",
                graph_id
            )));
        }

        // Check limits
        if self.collections.len() >= self.max_graphs {
            return Err(ProximaDBError::InvalidInput(format!(
                "Maximum number of graphs ({}) exceeded",
                self.max_graphs
            )));
        }

        // Create graph collection metadata
        let now = chrono::Utc::now().timestamp();
        let collection = GraphCollection {
            graph_id: graph_id.clone(),
            name: request.name.unwrap_or_else(|| graph_id.clone()),
            description: request.description.unwrap_or_default(),
            schema: request.schema,
            storage_config: request.storage_config,
            engine_config: request.engine_config,
            access_control: request.access_control,
            stats: None, // Will be populated by operations service
            created_at: now,
            updated_at: now,
        };

        let collection_arc = Arc::new(collection);
        self.collections
            .insert(graph_id.clone(), collection_arc.clone());

        // Persist to disk
        if let Err(e) = self.save_to_disk().await {
            warn!("Failed to persist graph metadata: {}", e);
            // Continue - metadata is in memory, just not persisted yet
        }

        info!("Created graph collection: {}", graph_id);
        Ok(collection_arc)
    }

    /// Get graph collection metadata
    pub async fn get_graph(&self, graph_id: &str) -> Result<Option<Arc<GraphCollection>>> {
        Ok(self
            .collections
            .get(graph_id)
            .map(|entry| Arc::clone(&entry)))
    }

    /// Delete a graph collection
    pub async fn delete_graph(&self, graph_id: &str) -> Result<()> {
        match self.collections.remove(graph_id) {
            Some(_) => {
                // Persist to disk
                if let Err(e) = self.save_to_disk().await {
                    warn!("Failed to persist graph metadata after delete: {}", e);
                }
                info!("Deleted graph collection: {}", graph_id);
                Ok(())
            }
            None => Err(ProximaDBError::NotFound {
                resource_type: "graph collection".to_string(),
                id: graph_id.to_string(),
            }),
        }
    }

    /// List all graph collections
    pub async fn list_graphs(&self) -> Result<Vec<Arc<GraphCollection>>> {
        Ok(self
            .collections
            .iter()
            .map(|entry| Arc::clone(&entry))
            .collect())
    }

    /// Update graph schema
    pub async fn update_schema(&self, graph_id: &str, schema: GraphSchema) -> Result<()> {
        if let Some(mut collection_ref) = self.collections.get_mut(graph_id) {
            // Deferred: Implement proper schema validation
            // Deferred: Handle schema migration if data exists

            let mut collection = (**collection_ref).clone();
            collection.schema = Some(schema);
            collection.updated_at = chrono::Utc::now().timestamp();

            *collection_ref = Arc::new(collection);

            // Persist to disk
            drop(collection_ref); // Release the lock before async call
            if let Err(e) = self.save_to_disk().await {
                warn!(
                    "Failed to persist graph metadata after schema update: {}",
                    e
                );
            }

            info!("Updated schema for graph: {}", graph_id);
            Ok(())
        } else {
            Err(ProximaDBError::NotFound {
                resource_type: "graph collection".to_string(),
                id: graph_id.to_string(),
            })
        }
    }

    /// Validate that a graph exists before operations
    pub async fn ensure_graph_exists(&self, graph_id: &str) -> Result<Arc<GraphCollection>> {
        self.get_graph(graph_id)
            .await?
            .ok_or_else(|| ProximaDBError::NotFound {
                resource_type: "graph collection".to_string(),
                id: graph_id.to_string(),
            })
    }

    /// Get the graph collection for `graph_id`, lazily provisioning it with defaults (ORION
    /// engine, the service's default storage root) if it does not yet exist — the get-or-create
    /// counterpart of [`ensure_graph_exists`].
    ///
    /// This lets a tenant-scoped first write auto-register its collection under the SCOPED key
    /// (`{tenant}/{graph_id}`), so named-tenant graph create→use works without an out-of-band bare
    /// create (the v2 gRPC surface has no create RPC, and the REST v1 / pgwire create paths are not
    /// tenant-scoped — TD-GRAPH-TENANT-1). The default tenant stays bare, so its existing
    /// explicitly-created collections are found on the first branch and never double-provisioned.
    /// Idempotent and race-safe: if a concurrent caller wins the insert, we pick it up on re-fetch.
    pub async fn ensure_or_create_graph(&self, graph_id: &str) -> Result<Arc<GraphCollection>> {
        if let Some(existing) = self.get_graph(graph_id).await? {
            return Ok(existing);
        }
        match self
            .create_graph(CreateGraphRequest {
                graph_id: graph_id.to_string(),
                ..Default::default()
            })
            .await
        {
            Ok(created) => Ok(created),
            // Lost a create race (another task registered it first) — re-fetch the winner.
            Err(_) => self.ensure_graph_exists(graph_id).await,
        }
    }

    /// Get graph statistics
    pub async fn get_graph_stats(
        &self,
        graph_id: &str,
    ) -> Result<Option<crate::proto::proximadb_v1::GraphStats>> {
        if let Some(collection) = self.get_graph(graph_id).await? {
            Ok(collection.stats.clone())
        } else {
            Err(ProximaDBError::NotFound {
                resource_type: "graph collection".to_string(),
                id: graph_id.to_string(),
            })
        }
    }
}

impl Default for GraphCollectionService {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn test_path(name: &str) -> PathBuf {
        let mut path = env::temp_dir();
        path.push(format!("proximadb_test_{}", name));
        path.push("graph_metadata.json");
        path
    }

    #[tokio::test]
    async fn test_create_graph() {
        let path = test_path("create");
        let _ = fs::remove_file(&path).await;

        let service = GraphCollectionService::new_with_path(path.clone());

        let request = CreateGraphRequest {
            graph_id: "test_graph".to_string(),
            name: Some("Test Graph".to_string()),
            description: Some("A test graph".to_string()),
            schema: None,
            storage_config: None,
            engine_config: None,
            access_control: None,
        };

        let result = service.create_graph(request).await;
        assert!(result.is_ok());

        let graph = result.unwrap();
        assert_eq!(graph.graph_id, "test_graph");
        assert_eq!(graph.name, "Test Graph");

        // Cleanup
        let _ = fs::remove_file(&path).await;
    }

    #[tokio::test]
    async fn test_duplicate_graph() {
        let path = test_path("duplicate");
        let _ = fs::remove_file(&path).await;

        let service = GraphCollectionService::new_with_path(path.clone());

        let request = CreateGraphRequest {
            graph_id: "test_graph".to_string(),
            name: Some("Test Graph".to_string()),
            description: None,
            schema: None,
            storage_config: None,
            engine_config: None,
            access_control: None,
        };

        // First creation should succeed
        assert!(service.create_graph(request.clone()).await.is_ok());

        // Second creation should fail
        assert!(service.create_graph(request).await.is_err());

        // Cleanup
        let _ = fs::remove_file(&path).await;
    }

    #[tokio::test]
    async fn test_persistence_across_instances() {
        let path = test_path("persistence");
        let _ = fs::remove_file(&path).await;

        // Create a graph with first instance
        {
            let service = GraphCollectionService::new_with_path(path.clone());

            let request = CreateGraphRequest {
                graph_id: "persistent_graph".to_string(),
                name: Some("Persistent Graph".to_string()),
                description: Some("This should persist".to_string()),
                schema: None,
                storage_config: None,
                engine_config: None,
                access_control: None,
            };

            service.create_graph(request).await.unwrap();
        }

        // Create new instance and load from disk
        {
            let service = GraphCollectionService::new_with_path(path.clone());
            let count = service.load_from_disk().await.unwrap();
            assert_eq!(count, 1, "Should load 1 graph from disk");

            let graph = service.get_graph("persistent_graph").await.unwrap();
            assert!(graph.is_some(), "Graph should exist after loading");

            let graph = graph.unwrap();
            assert_eq!(graph.name, "Persistent Graph");
            assert_eq!(graph.description, "This should persist");
        }

        // Cleanup
        let _ = fs::remove_file(&path).await;
    }

    #[tokio::test]
    async fn test_delete_persists() {
        let path = test_path("delete");
        let _ = fs::remove_file(&path).await;

        // Create and then delete a graph
        {
            let service = GraphCollectionService::new_with_path(path.clone());

            let request = CreateGraphRequest {
                graph_id: "to_delete".to_string(),
                name: Some("To Delete".to_string()),
                description: None,
                schema: None,
                storage_config: None,
                engine_config: None,
                access_control: None,
            };

            service.create_graph(request).await.unwrap();
            service.delete_graph("to_delete").await.unwrap();
        }

        // Reload and verify deletion persisted
        {
            let service = GraphCollectionService::new_with_path(path.clone());
            let count = service.load_from_disk().await.unwrap();
            assert_eq!(count, 0, "Deleted graph should not be loaded");
        }

        // Cleanup
        let _ = fs::remove_file(&path).await;
    }
}
