// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Background Flush Context - Pre-computed Collection Metadata
//!
//! Eliminates redundant collection service calls by pre-computing all needed metadata
//! for background flush and compaction operations.

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::proto::proximadb_v1::FilterableColumnSpec;
use proximadb_distance_kernel::DistanceMetric;
use proximadb_storage_ports::CollectionMetadataPort;

/// Storage engine types supported by ProximaDB
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageEngineType {
    /// VIPER - Columnar Parquet-based engine for analytics workloads
    Viper,
    /// SST - Hybrid columnar SSTable engine for OLTP workloads (ProximaBlocks format)
    Sst,
    /// TST - Time-series optimized storage engine
    Tst,
}

impl TryFrom<i32> for StorageEngineType {
    type Error = anyhow::Error;

    fn try_from(value: i32) -> Result<Self> {
        match value {
            x if x == crate::proto::proximadb_v1::StorageEngine::Viper as i32 => {
                Ok(StorageEngineType::Viper)
            }
            x if x == crate::proto::proximadb_v1::StorageEngine::Sst as i32 => {
                Ok(StorageEngineType::Sst)
            }
            x if x == crate::proto::proximadb_v1::StorageEngine::Tst as i32 => {
                Ok(StorageEngineType::Tst)
            }
            _ => Err(anyhow!("Unknown storage engine type: {}", value)),
        }
    }
}

/// Backwards-compat alias for [`BackgroundFlushCompressionConfig`].
pub type CompressionConfig = BackgroundFlushCompressionConfig;

/// Compression configuration for storage engines
#[derive(Debug, Clone)]
pub struct BackgroundFlushCompressionConfig {
    pub enabled: bool,
    pub compression_type: String, // "zstd", "snappy", "lz4", "gzip"
    pub level: i32,
}

impl Default for BackgroundFlushCompressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            compression_type: "zstd".to_string(),
            level: 3, // Balanced CPU/IO performance
        }
    }
}

/// Quantization configuration for vector compression during background flush.
///
/// Distinct from the canonical `proximadb_quantization_types::QuantizationConfig`
/// (which models quantization type/level/asymmetry/cache): this is the flush-context
/// shape (enabled + string type + bit width + PQ subspaces). Renamed from the former
/// `QuantizationConfig` to remove the cross-module name collision (LLD duplication watch).
#[derive(Debug, Clone)]
pub struct FlushQuantizationConfig {
    pub enabled: bool,
    pub quantization_type: String, // "product", "scalar", "binary"
    pub bits_per_component: u8,
    pub subspaces: Option<u8>,
}

/// Operation priority for background tasks
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OperationPriority {
    Low,
    #[default]
    Normal,
    High,
    Critical,
}

/// Pre-computed context containing ALL metadata needed for background flush/compaction
///
/// This eliminates the need for background threads to make collection service calls,
/// improving performance and reliability.
#[derive(Debug, Clone)]
pub struct BackgroundFlushContext {
    // === Core Identification ===
    /// Collection ID being processed
    pub collection_id: String,

    // === Engine Configuration ===
    /// Storage engine type (VIPER vs SST)
    pub storage_engine: StorageEngineType,
    /// Base storage location path (e.g., "file:///data/disk1")
    pub base_location: String,

    // === Vector Configuration ===
    /// Vector dimension size
    pub dimension: usize,
    /// Distance metric for similarity calculations
    pub distance_metric: DistanceMetric,

    // === Storage Configuration ===
    /// Compression settings for storage engine
    pub compression_config: BackgroundFlushCompressionConfig,

    // === Schema Configuration ===
    /// Filterable metadata columns with their types
    pub filterable_columns: Vec<FilterableColumnSpec>,

    // === Performance Configuration ===
    /// Vector quantization settings (if enabled)
    pub quantization: Option<FlushQuantizationConfig>,
    /// Suggested batch size for operations
    pub batch_size_hint: Option<usize>,

    // === Operational Configuration ===
    /// Priority level for background operations
    pub priority: OperationPriority,
    /// Timeout for operations in milliseconds
    pub timeout_ms: Option<u64>,
    /// Additional metadata for future extensions
    pub extra_metadata: HashMap<String, String>,

    // === Isolation ===
    /// Owning tenant for the collection, resolved once when the context is built
    /// (D3: amortized, not per row). Consumed by the flush coordinator's
    /// `resolve_flush_tenant` so the A6 storage-write fence can check
    /// `(tenant, collection)` ownership. `None` when the collection is not
    /// tenant-scoped ⇒ the fence fails open.
    pub tenant_id: Option<String>,

    /// The catalog collection meta this context was built from, when known.
    /// TD-READFID-2 review: compaction resolves the output encoding (base
    /// quant, rg layout, arming) from its tags + `pax_vector_quant`, which the
    /// synthesized proto below cannot carry. `None` (test contexts) marks the
    /// meta as incomplete so the SST engine defers instead of guessing.
    pub collection_meta: Option<std::sync::Arc<crate::proto::proximadb_v1::Collection>>,
}

impl BackgroundFlushContext {
    /// Convert internal DistanceMetric to proto DistanceMetric
    /// Centralizes distance metric conversion to ensure all 13 supported metrics are handled
    pub fn distance_metric_to_proto(metric: &DistanceMetric) -> i32 {
        use crate::proto::proximadb_v1::DistanceMetric as ProtoDistanceMetric;

        match metric {
            // Core metrics
            DistanceMetric::Cosine => ProtoDistanceMetric::Cosine as i32,
            DistanceMetric::Euclidean => ProtoDistanceMetric::Euclidean as i32,
            DistanceMetric::DotProduct => ProtoDistanceMetric::DotProduct as i32,

            // Extended metrics
            DistanceMetric::Manhattan => ProtoDistanceMetric::Manhattan as i32,
            DistanceMetric::Hamming => ProtoDistanceMetric::Hamming as i32,
            DistanceMetric::Jaccard => ProtoDistanceMetric::Jaccard as i32,
            DistanceMetric::Chebyshev => ProtoDistanceMetric::Chebyshev as i32,
            DistanceMetric::Canberra => ProtoDistanceMetric::Canberra as i32,
            DistanceMetric::Minkowski => ProtoDistanceMetric::Minkowski as i32,
            DistanceMetric::Angular => ProtoDistanceMetric::Angular as i32,
            DistanceMetric::BrayCurtis => ProtoDistanceMetric::BrayCurtis as i32,
            DistanceMetric::Hellinger => ProtoDistanceMetric::Hellinger as i32,
            DistanceMetric::Custom => ProtoDistanceMetric::Custom as i32,

            // Handle unspecified - default to Cosine as the most common metric
            DistanceMetric::Unspecified => ProtoDistanceMetric::Cosine as i32,
        }
    }

    /// Convert internal StorageEngineType to proto StorageEngine
    /// Centralizes storage engine conversion for code reuse
    pub fn storage_engine_to_proto(engine: &StorageEngineType) -> i32 {
        use crate::proto::proximadb_v1::StorageEngine as ProtoStorageEngine;

        match engine {
            StorageEngineType::Viper => ProtoStorageEngine::Viper as i32,
            StorageEngineType::Sst => ProtoStorageEngine::Sst as i32,
            StorageEngineType::Tst => ProtoStorageEngine::Tst as i32,
        }
    }

    /// Create a complete Collection proto from the background context
    /// This provides all necessary information for flush and compaction operations
    /// without requiring additional service calls
    pub fn to_collection_proto(&self) -> crate::proto::proximadb_v1::Collection {
        // The real catalog meta, when known, is the authority (tags,
        // `pax_vector_quant`, precision, typed identity all intact).
        if let Some(meta) = &self.collection_meta {
            return meta.as_ref().clone();
        }
        use crate::proto::proximadb_v1::{
            Collection, CollectionConfig, CollectionStats, StorageAssignment,
        };

        let storage_assignment = StorageAssignment {
            primary_path: self.base_location.clone(),
            backup_paths: vec![],
            engine: 0, // Default engine
            engine_config: std::collections::HashMap::new(),
            base_location: self.base_location.clone(),
            assigned_at: chrono::Utc::now().timestamp_millis(),
            ..Default::default()
        };

        let config = CollectionConfig {
            name: self.collection_id.clone(),
            dimension: self.dimension as u32,
            distance_metric: Some(Self::distance_metric_to_proto(&self.distance_metric)),
            storage_engine: Some(Self::storage_engine_to_proto(&self.storage_engine)),
            filterable_columns: self.filterable_columns.clone(),
            quantization: self.quantization.as_ref().map(|qc| {
                crate::proto::proximadb_v1::QuantizationConfig {
                    enabled: Some(qc.enabled),
                    ..Default::default()
                }
            }),
            ..Default::default()
        };

        let stats = CollectionStats {
            vector_count: 0,     // Unknown from context
            index_size_bytes: 0, // Unknown from context
            data_size_bytes: 0,  // Unknown from context
        };

        Collection {
            id: self.collection_id.clone(),
            config: Some(config),
            stats: Some(stats),
            created_at: chrono::Utc::now().timestamp_millis(),
            updated_at: chrono::Utc::now().timestamp_millis(),
            storage_assignment: Some(storage_assignment),
        }
    }

    /// Create context from collection service (eliminates future service calls)
    pub async fn from_collection_service(
        service: &dyn CollectionMetadataPort,
        collection_id: &str,
    ) -> Result<Self> {
        // Single collection service call - all subsequent operations use this context
        let collection = service
            .collection(collection_id)
            .await
            .context("Failed to fetch collection from service")?
            .ok_or_else(|| anyhow!("Collection '{}' not found", collection_id))?;
        Self::from_collection(collection_id, collection)
    }

    /// Build the context from an already-fetched catalog collection meta. The
    /// meta itself is retained so compaction resolves from the real tags.
    pub fn from_collection(
        collection_id: &str,
        collection: crate::proto::proximadb_v1::Collection,
    ) -> Result<Self> {
        let config = collection
            .config
            .as_ref()
            .ok_or_else(|| anyhow!("Collection '{}' has no configuration", collection_id))?;

        let storage_assignment = collection
            .storage_assignment
            .as_ref()
            .ok_or_else(|| anyhow!("Collection '{}' has no storage assignment", collection_id))?;

        // Parse storage engine type
        let storage_engine = StorageEngineType::try_from(config.storage_engine.unwrap_or(0))
            .context("Failed to parse storage engine type")?;

        // Parse distance metric
        let distance_metric = DistanceMetric::try_from(config.distance_metric.unwrap_or(0))
            .context("Failed to parse distance metric")?;

        // Create compression config based on storage engine defaults
        let compression_config = match storage_engine {
            StorageEngineType::Viper => BackgroundFlushCompressionConfig {
                enabled: true,
                compression_type: "zstd".to_string(),
                level: 3, // Balanced for Parquet
            },
            StorageEngineType::Sst => BackgroundFlushCompressionConfig {
                enabled: true,
                compression_type: "snappy".to_string(),
                level: 1, // Fast compression for OLTP
            },
            StorageEngineType::Tst => BackgroundFlushCompressionConfig {
                enabled: true,
                compression_type: "zstd".to_string(),
                level: 3, // Balanced archival compression for time-series workloads
            },
        };

        // Parse quantization config if present
        let quantization = config.quantization.as_ref().map(|qc| {
            // Extract quantization type from the new QuantizationConfig
            let quantization_type = match qc.strategy() {
                crate::proto::proximadb_v1::quantization_config::Strategy::SmartDefaults => {
                    "smart_defaults"
                }
                crate::proto::proximadb_v1::quantization_config::Strategy::CustomLevels => {
                    "custom_levels"
                }
                crate::proto::proximadb_v1::quantization_config::Strategy::Minimal => "minimal",
                crate::proto::proximadb_v1::quantization_config::Strategy::Aggressive => {
                    "aggressive"
                }
            };

            FlushQuantizationConfig {
                enabled: qc.enabled.unwrap_or(false),
                quantization_type: quantization_type.to_string(),
                bits_per_component: 8, // Default - could be extracted from specific quantization types
                subspaces: Some(8), // Default - could be extracted from ProductQuantization config
            }
        });

        // Determine batch size hint based on dimension and engine
        let batch_size_hint = match storage_engine {
            StorageEngineType::Viper => Some(1000.min(10000 / (config.dimension / 100).max(1))),
            StorageEngineType::Sst => Some(500.min(5000 / (config.dimension / 100).max(1))),
            StorageEngineType::Tst => Some(2000.min(20000 / (config.dimension / 100).max(1))),
        };

        // Resolve the owning tenant once, here, from the same collection proto we
        // already fetched (D3: one amortized lookup, never per row). Reuses the
        // canonical resolver so tag/owner precedence matches the network gates.
        let tenant_id = proximadb_tenant::tenant_id_of(&collection);

        Ok(Self {
            collection_id: collection_id.to_string(),
            storage_engine,
            base_location: storage_assignment.base_location.clone(),
            dimension: config.dimension as usize,
            distance_metric,
            compression_config,
            filterable_columns: config.filterable_columns.clone(),
            quantization,
            batch_size_hint: batch_size_hint.map(|s| s as usize),
            priority: OperationPriority::Normal,
            timeout_ms: Some(300_000), // 5 minutes default
            extra_metadata: HashMap::new(),
            tenant_id,
            collection_meta: Some(std::sync::Arc::new(collection)),
        })
    }

    /// Create context for testing with minimal required fields
    pub fn for_testing(collection_id: &str, storage_engine: StorageEngineType) -> Self {
        Self {
            collection_id: collection_id.to_string(),
            storage_engine,
            base_location: "file:///tmp/test_data".to_string(),
            dimension: 384,
            distance_metric: DistanceMetric::Cosine,
            compression_config: BackgroundFlushCompressionConfig::default(),
            filterable_columns: Vec::new(),
            quantization: None,
            batch_size_hint: Some(1000),
            priority: OperationPriority::Normal,
            timeout_ms: Some(60_000),
            extra_metadata: HashMap::new(),
            tenant_id: None,
            collection_meta: None,
        }
    }

    /// Get engine name as string (for compatibility with existing code)
    pub fn engine_name(&self) -> &str {
        match self.storage_engine {
            StorageEngineType::Viper => "viper",
            StorageEngineType::Sst => "sst",
            StorageEngineType::Tst => "tst",
        }
    }

    /// Get suggested row group size for Parquet (VIPER engine)
    pub fn row_group_size(&self) -> usize {
        match self.storage_engine {
            StorageEngineType::Viper => {
                // Balance between compression efficiency and memory usage
                let base_size = 10_000;
                let dimension_factor = (self.dimension / 100).max(1);
                (base_size / dimension_factor).clamp(1_000, 50_000)
            }
            StorageEngineType::Sst => 1_000, // Smaller for OLTP workloads
            StorageEngineType::Tst => {
                let base_size = 20_000;
                let dimension_factor = (self.dimension / 100).max(1);
                (base_size / dimension_factor).clamp(2_000, 100_000)
            }
        }
    }

    /// Get suggested flush threshold in number of vectors
    pub fn flush_threshold(&self) -> usize {
        match self.storage_engine {
            StorageEngineType::Viper => {
                // VIPER benefits from larger batches for columnar compression
                let base_threshold = 50_000;
                let dimension_factor = (self.dimension / 100).max(1);
                (base_threshold / dimension_factor).clamp(10_000, 100_000)
            }
            StorageEngineType::Sst => {
                // SST optimized for smaller, frequent flushes
                let base_threshold = 10_000;
                let dimension_factor = (self.dimension / 100).max(1);
                (base_threshold / dimension_factor).clamp(1_000, 25_000)
            }
            StorageEngineType::Tst => {
                // TST favors larger time-windowed flushes for partition efficiency.
                let base_threshold = 100_000;
                let dimension_factor = (self.dimension / 100).max(1);
                (base_threshold / dimension_factor).clamp(5_000, 250_000)
            }
        }
    }
}
