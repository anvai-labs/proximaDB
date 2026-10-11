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

//! SST Engine Collections Module
//!
//! Contains collection management utilities for the SST engine.
//! This module provides:
//! - Collection metadata and statistics
//! - File management and cleanup operations
//! - Vector existence checking with bloom filters
//! - Collection scanning and enumeration

use anyhow::Result;
use std::collections::HashMap;
use tracing::{debug, info, warn};

use crate::proto::proximadb_v1::{Collection, VectorRecord};
use crate::storage::engines::sst::{SstEngine, SstError};
use crate::storage::traits::CompactionResult;

impl SstEngine {
    /// Get collection information
    pub async fn collection(&self, collection_id: &str) -> Result<Collection> {
        debug!("📂 SST: Getting collection info for {}", collection_id);

        // Create a basic collection structure
        // In a real implementation, this would load from metadata storage
        use crate::proto::proximadb_v1::{CollectionConfig, CollectionStats};

        let config = CollectionConfig {
            name: collection_id.to_string(),
            embedding_config: None,
            dimension: 1536,          // Default dimension
            distance_metric: Some(0), // Default metric
            storage_engine: Some(0),  // SST engine
            tags: vec![],
            description: None,
            filterable_columns: vec![],
            index_configs: vec![],
            quantization: None,
            storage_config: None,
            primary_index: Some("default".to_string()),
            auto_index_selection: Some(true),
            owner: None,
            embedding_models: vec![],
            // ProximaRecord schema configuration (NEW)
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

        let stats = CollectionStats {
            vector_count: 0,
            index_size_bytes: 0,
            data_size_bytes: 0,
        };

        Ok(Collection {
            id: collection_id.to_string(),
            config: Some(config),
            stats: Some(stats),
            created_at: 0, // Would be loaded from metadata
            updated_at: 0, // Would be loaded from metadata
            storage_assignment: None,
        })
    }

    /// List all files for a collection
    pub async fn list_collection_files(&self, collection_id: &str) -> Result<Vec<String>> {
        debug!("📋 SST: Listing files for collection {}", collection_id);

        let storage_url = self.get_collection_storage_url(collection_id).await?;
        let fs = self.filesystem().get_filesystem(&storage_url)?;

        let mut files = Vec::new();
        match fs.list(&storage_url).await {
            Ok(entries) => {
                for entry in entries {
                    if !entry.metadata.is_directory {
                        files.push(entry.url);
                    }
                }
            }
            Err(e) => {
                warn!(
                    "Failed to list files for collection {}: {}",
                    collection_id, e
                );
            }
        }

        debug!(
            "📋 Found {} files for collection {}",
            files.len(),
            collection_id
        );
        Ok(files)
    }

    /// Get collection statistics
    pub fn collection_stats(&self, collection_id: &str) -> Result<serde_json::Value> {
        debug!("📊 SST: Getting stats for collection {}", collection_id);

        // In a real implementation, this would:
        // 1. Scan SST file metadata to get vector counts
        // 2. Calculate storage sizes
        // 3. Analyze distribution across levels
        let vector_count = 0; // Placeholder

        let stats = serde_json::json!({
            "collection_id": collection_id,
            "engine": "sst",
            "vector_count": vector_count,
            "file_count": 0,
            "total_size_bytes": 0,
            "levels": {
                "0": { "files": 0, "size_bytes": 0 },
                "1": { "files": 0, "size_bytes": 0 },
                "2": { "files": 0, "size_bytes": 0 }
            },
            "bloom_filter_stats": {
                "total_keys": 0,
                "false_positive_rate": 0.01
            }
        });

        Ok(stats)
    }

    /// Get collection metadata
    pub fn collection_metadata(&self, collection_id: &str) -> Result<serde_json::Value> {
        debug!("📋 SST: Getting metadata for collection {}", collection_id);

        let metadata = serde_json::json!({
            "collection_id": collection_id,
            "engine": "sst",
            "storage_format": "sstable",
            "compression": "lz4",
            "bloom_filters_enabled": true,
            "compaction_enabled": true,
            "three_stage_filtering": true,
            "created_at": null,
            "updated_at": null,
            "schema_version": "1.0"
        });

        Ok(metadata)
    }

    /// Check if a vector exists in the collection using bloom filters
    pub async fn contains_vector(&self, collection_id: &str, id: &str) -> Result<bool> {
        debug!(
            "🔍 SST: Checking if vector {} exists in collection {}",
            id, collection_id
        );

        let _storage_url = self.get_collection_storage_url(collection_id).await?;

        // Get unified filesystem for this collection
        let unified_fs = self
            .unified_fs()
            .ok_or_else(|| SstError::Internal("Unified filesystem not initialized".to_string()))?;

        // List SST files and check bloom filters
        let files = match unified_fs.list("/").await {
            Ok(files) => files,
            Err(_) => return Ok(false),
        };

        for entry in files.iter().filter(|f| f.name.ends_with(".sst")) {
            // Check if the file exists and is accessible
            if unified_fs.exists(&entry.name).await.unwrap_or(false) {
                // In production, this would:
                // 1. Read the bloom filter from SST file footer
                // 2. Check if the vector ID hash is in the bloom filter
                // 3. Return false for definite non-existence, true for possible existence

                debug!("🔍 Checking bloom filter in {}", entry.name);

                // For now, use a conservative approach
                if self.check_bloom_filter(&entry.name, id).await? {
                    return Ok(true);
                }
            }
        }

        debug!(
            "❌ Vector {} not found in any SST file for collection {}",
            id, collection_id
        );
        Ok(false)
    }

    /// Check bloom filter for a specific vector ID in an SST file
    async fn check_bloom_filter(&self, file_path: &str, vector_id: &str) -> Result<bool> {
        debug!(
            "🔍 Checking bloom filter in {} for vector {}",
            file_path, vector_id
        );

        // In a real implementation, this would:
        // 1. Read the SST file footer to get bloom filter offset
        // 2. Read the bloom filter data
        // 3. Hash the vector ID and check against the bloom filter
        // 4. Return false for definite non-existence, true for possible existence

        // For now, return true (conservative - assume it might exist)
        Ok(true)
    }

    /// Clean up all files for a collection
    pub async fn cleanup_collection_files(&self, collection_id: &str) -> Result<()> {
        info!("🧹 SST: Cleaning up files for collection {}", collection_id);

        let _storage_url = self.get_collection_storage_url(collection_id).await?;
        let unified_fs = self
            .unified_fs()
            .ok_or_else(|| SstError::Internal("Unified filesystem not initialized".to_string()))?;

        // List all files for the collection
        let files = match unified_fs.list("/").await {
            Ok(files) => files,
            Err(e) => {
                warn!("Failed to list collection files for cleanup: {}", e);
                return Ok(()); // Don't fail cleanup
            }
        };

        let mut deleted_count = 0;
        let mut error_count = 0;

        // Clean up SST files, bloom filters, and metadata files
        for entry in &files {
            if entry.name.ends_with(".sst")
                || entry.name.ends_with(".bloom")
                || entry.name.ends_with(".meta")
                || entry.name.ends_with(".idx")
            {
                match unified_fs.delete(&entry.name).await {
                    Ok(_) => {
                        deleted_count += 1;
                        debug!("🗑️ Deleted file: {}", entry.name);
                    }
                    Err(e) => {
                        error_count += 1;
                        warn!("Failed to delete collection file {}: {}", entry.name, e);
                        // Continue with other files
                    }
                }
            }
        }

        info!(
            "🧹 Cleanup completed for collection {}: {} files deleted, {} errors",
            collection_id, deleted_count, error_count
        );

        Ok(())
    }

    /// Scan all vectors in a collection with pagination
    pub async fn scan_all_vectors(
        &self,
        collection_id: &str,
        offset: usize,
        limit: Option<usize>,
    ) -> Result<Vec<VectorRecord>> {
        debug!(
            "📖 SST: Scanning vectors in collection {} (offset: {}, limit: {:?})",
            collection_id, offset, limit
        );

        let _storage_url = self.get_collection_storage_url(collection_id).await?;

        // In a real implementation, this would:
        // 1. List all SST files for the collection
        // 2. Open each file and scan records
        // 3. Apply offset and limit across all files
        // 4. Return paginated results

        let results = Vec::new();
        let _effective_limit = limit.unwrap_or(1000); // Default limit

        // For now, return empty results as this is a complex operation
        // that requires implementing SST file reading
        debug!(
            "📖 Scan completed for collection {}: {} vectors returned",
            collection_id,
            results.len()
        );

        Ok(results)
    }

    /// Get storage URL for a collection — resolved via the injected
    /// `CollectionPathResolver` (catalog → config fallback; the single source of
    /// truth), retiring the `/data/collections/{id}` placeholder. The resolver
    /// returns the collection ROOT (`{base}/{collection_id}`); `/data` is the
    /// segment subdirectory (matches `StoragePath::collection_data_path` + the
    /// flush's `get_collection_storage_url_from_params`). When no resolver is
    /// injected (un-wired deployment / test), a `ConfigFallbackResolver` is
    /// default-constructed so the URL is still well-formed. Production wires the
    /// catalog-coupled resolver via `SharedServices::new`.
    async fn get_collection_storage_url(&self, collection_id: &str) -> Result<String> {
        use crate::storage::trait_components::path_resolver::CollectionPathResolver;
        let root = match self.path_resolver.as_ref() {
            Some(resolver) => resolver.resolve_base_location(collection_id).await?,
            None => {
                crate::storage::trait_components::path_resolver::ConfigFallbackResolver::default()
                    .resolve_base_location(collection_id)
                    .await?
            }
        };
        let storage_url = format!("{root}/data");
        debug!(
            "📂 Storage URL for collection {}: {}",
            collection_id, storage_url
        );
        Ok(storage_url)
    }

    /// Compact a specific collection
    pub async fn compact_collection(
        &self,
        collection_id: &str,
        _target_level: Option<u8>,
    ) -> Result<CompactionResult> {
        info!(
            "🔄 SST: Starting compaction for collection {}",
            collection_id
        );

        let _storage_url = self.get_collection_storage_url(collection_id).await?;

        // In a real implementation, this would:
        // 1. Identify files that need compaction at the target level
        // 2. Read and merge overlapping SST files
        // 3. Write new compacted files
        // 4. Update metadata and delete old files

        let result = CompactionResult {
            success: true,
            collections_affected: vec![collection_id.to_string()],
            entries_processed: Some(0),
            entries_removed: Some(0),
            bytes_read: Some(0),
            bytes_written: Some(0),
            input_files: Some(0),
            output_files: Some(0),
            duration_ms: Some(0),
            completed_at: chrono::Utc::now(),
            engine_metrics: HashMap::new(),
        };

        info!("✅ Compaction completed for collection {}", collection_id);
        Ok(result)
    }

    /// Get collection size statistics
    pub async fn get_collection_size(&self, collection_id: &str) -> Result<CollectionSizeInfo> {
        debug!("📏 SST: Getting size info for collection {}", collection_id);

        let files = self.list_collection_files(collection_id).await?;
        let mut total_size = 0u64;
        let mut file_sizes = HashMap::new();

        for file_path in &files {
            if let Ok(fs) = self.filesystem().get_filesystem(file_path)
                && let Ok(metadata) = fs.metadata(file_path).await
            {
                total_size += metadata.size;
                file_sizes.insert(file_path.clone(), metadata.size);
            }
        }

        Ok(CollectionSizeInfo {
            collection_id: collection_id.to_string(),
            total_size_bytes: total_size,
            file_count: files.len(),
            file_sizes,
            estimated_vector_count: 0, // Would be calculated from SST metadata
        })
    }
}

// Using CompactionResult from storage::traits instead of local definition

/// Information about collection storage size
#[derive(Debug, Clone)]
pub struct CollectionSizeInfo {
    pub collection_id: String,
    pub total_size_bytes: u64,
    pub file_count: usize,
    pub file_sizes: HashMap<String, u64>,
    pub estimated_vector_count: u64,
}

#[cfg(test)]
#[cfg_attr(test, path = "collections_tests.rs")]
mod tests;
