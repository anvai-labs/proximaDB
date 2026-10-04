//! Codebook Metadata Storage for Quantized Vectors
//!
//! This module provides unified codebook serialization and storage for all engines.
//! - ProximaBlock engines (SST, SWIFT, HELIX): Store as footer metadata
//! - Parquet engines (VIPER, NOVA): Store as sidecar files
//!
//! Integrates with existing UnifiedQuantizationEngine and GlobalQuantizationCache.

use anyhow::{Context, Result};
use bytes::{Bytes, BytesMut};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::storage::compute_bridge::global_cache::{GlobalQuantizationCache, QuantizationCacheKey};
use proximadb_quantization_kernel::quantization_engine::{
    Codebook, CodebookData, QuantizationLevel, UnifiedQuantizationEngine,
};
use proximadb_quantization_kernel::storage_engine::StorageQuantizationEngine;

/// Codebook metadata stored at file level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationCodebookMetadata {
    /// Collection ID for this codebook
    pub collection_id: String,

    /// Binary quantization codebook (if present)
    pub binary_codebook: Option<BinaryCodebook>,

    /// INT8 quantization codebook (if present)
    pub int8_codebook: Option<Int8Codebook>,

    /// PQ quantization codebooks (different configurations)
    pub pq_codebooks: HashMap<String, PqCodebook>,

    /// Creation timestamp
    pub created_at: i64,

    /// Number of vectors used for training
    pub training_samples: usize,

    /// Schema version for forward compatibility
    pub schema_version: u32,
}

/// Binary quantization codebook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryCodebook {
    /// Threshold value for binary quantization
    pub threshold: f32,

    /// Mean vector (for centering)
    pub mean: Option<Vec<f32>>,

    /// Dimension
    pub dimension: usize,
}

/// INT8 quantization codebook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Int8Codebook {
    /// Scale factor for quantization
    pub scale: f32,

    /// Zero point (offset)
    pub zero_point: i8,

    /// Min value in original data
    pub min_value: f32,

    /// Max value in original data
    pub max_value: f32,

    /// Dimension
    pub dimension: usize,
}

/// PQ quantization codebook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqCodebook {
    /// Number of subvectors
    pub num_subvectors: u32,

    /// Bits per code
    pub bits_per_code: u8,

    /// Centroids for each subvector
    /// Shape: [num_subvectors][num_centroids][subvector_dim]
    pub centroids: Vec<Vec<Vec<f32>>>,

    /// Dimension of original vectors
    pub dimension: usize,

    /// Subvector dimension
    pub subvector_dim: usize,

    /// Number of centroids per subvector (2^bits_per_code)
    pub num_centroids: usize,

    /// Training configuration
    pub training_config: PqTrainingConfig,
}

/// PQ training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqTrainingConfig {
    /// Number of iterations for k-means
    pub num_iterations: usize,

    /// Random seed for reproducibility
    pub seed: Option<u64>,

    /// Distance metric used
    pub distance_metric: String,
}

/// Codebook serializer for different storage formats
pub struct CodebookSerializer {
    /// Reference to global quantization cache
    cache: Option<Arc<GlobalQuantizationCache>>,

    /// Reference to storage quantization engine
    storage_engine: Option<Arc<StorageQuantizationEngine>>,
}

impl CodebookSerializer {
    /// Create new codebook serializer
    pub fn new() -> Self {
        Self {
            cache: None,
            storage_engine: None,
        }
    }

    /// Create with cache integration
    pub fn with_cache(cache: Arc<GlobalQuantizationCache>) -> Self {
        Self {
            cache: Some(cache),
            storage_engine: None,
        }
    }

    /// Create with storage engine integration
    pub fn with_storage_engine(engine: Arc<StorageQuantizationEngine>) -> Self {
        Self {
            cache: None,
            storage_engine: Some(engine),
        }
    }

    /// Serialize codebook metadata to bytes for footer storage
    pub fn serialize_for_footer(&self, metadata: &QuantizationCodebookMetadata) -> Result<Bytes> {
        // Use bincode for efficient binary serialization
        let bytes =
            bincode::serialize(metadata).context("Failed to serialize codebook metadata")?;
        Ok(Bytes::from(bytes))
    }

    /// Deserialize codebook metadata from footer bytes
    pub fn deserialize_from_footer(&self, bytes: &[u8]) -> Result<QuantizationCodebookMetadata> {
        let metadata =
            bincode::deserialize(bytes).context("Failed to deserialize codebook metadata")?;
        Ok(metadata)
    }

    /// Serialize codebook metadata to JSON for sidecar files
    pub fn serialize_for_sidecar(&self, metadata: &QuantizationCodebookMetadata) -> Result<String> {
        let json = serde_json::to_string_pretty(metadata)
            .context("Failed to serialize codebook metadata to JSON")?;
        Ok(json)
    }

    /// Deserialize codebook metadata from sidecar JSON
    pub fn deserialize_from_sidecar(&self, json: &str) -> Result<QuantizationCodebookMetadata> {
        let metadata = serde_json::from_str(json)
            .context("Failed to deserialize codebook metadata from JSON")?;
        Ok(metadata)
    }

    /// Extract codebook metadata from UnifiedQuantizationEngine
    pub async fn extract_from_engine(
        &self,
        _engine: &UnifiedQuantizationEngine,
        collection_id: &str,
    ) -> Result<QuantizationCodebookMetadata> {
        let mut metadata = QuantizationCodebookMetadata {
            collection_id: collection_id.to_string(),
            binary_codebook: None,
            int8_codebook: None,
            pq_codebooks: HashMap::new(),
            created_at: chrono::Utc::now().timestamp(),
            training_samples: 0,
            schema_version: 1,
        };

        // Try to get from global cache if available
        if let Some(ref cache) = self.cache {
            // Check for binary codebook
            let binary_key = QuantizationCacheKey::binary(collection_id);
            if let Some(_codebook) = cache.get_codebook(&binary_key).await {
                metadata.binary_codebook = Some(BinaryCodebook {
                    threshold: 0.0, // Default threshold
                    mean: None,
                    dimension: 0, // Will be set during actual quantization
                });
            }

            // Check for INT8 codebook
            let int8_key = QuantizationCacheKey::int8(collection_id);
            if let Some(_codebook) = cache.get_codebook(&int8_key).await {
                metadata.int8_codebook = Some(Int8Codebook {
                    scale: 1.0,
                    zero_point: 0,
                    min_value: -1.0,
                    max_value: 1.0,
                    dimension: 0,
                });
            }

            // Check for PQ codebooks (common configurations)
            for bits in [4, 8, 16] {
                for subvectors in [8, 16, 32] {
                    let pq_key = QuantizationCacheKey::pq(collection_id, bits, subvectors);
                    if let Some(codebook) = cache.get_codebook(&pq_key).await {
                        let key = format!("pq{bits}_{subvectors}");
                        metadata
                            .pq_codebooks
                            .insert(key, self.convert_to_pq_codebook(&codebook)?);
                    }
                }
            }
        }

        // Try to get from storage engine if available
        if let Some(ref engine) = self.storage_engine {
            // Storage engine has cached codebooks
            for codebook_id in engine.list_cached_codebooks() {
                if codebook_id.starts_with(&format!("{}_", collection_id)) {
                    // Parse codebook type from ID
                    if codebook_id.contains("_pq_") {
                        // Extract PQ parameters from ID
                        // Format: {collection_id}_pq_{bits}_{subvectors}
                        if let Some(codebook) = engine.get_cached_codebook(&codebook_id) {
                            let key = codebook_id.replace(&format!("{collection_id}_"), "");
                            metadata
                                .pq_codebooks
                                .insert(key, self.convert_cached_to_pq_codebook(&codebook)?);
                        }
                    }
                }
            }
        }

        Ok(metadata)
    }

    /// Convert unified Codebook to PqCodebook
    fn convert_to_pq_codebook(&self, codebook: &Arc<Codebook>) -> Result<PqCodebook> {
        // Extract PQ data from the codebook
        if let CodebookData::ProductQuantization {
            centroids,
            _subvector_dim,
        } = &codebook.data
        {
            // Extract quantization parameters from the level
            let (num_subvectors, bits_per_code) =
                if let Some(QuantizationLevel::Pq(pq)) = &codebook.quantization_level.level_type {
                    (pq.num_subvectors, pq.bits_per_code)
                } else {
                    return Err(anyhow::anyhow!("Codebook is not PQ type"));
                };

            let num_centroids = 1 << bits_per_code;
            let subvector_dim = *_subvector_dim;

            // Calculate original dimension
            let dimension = subvector_dim * num_subvectors as usize;

            Ok(PqCodebook {
                num_subvectors: num_subvectors as u32,
                bits_per_code: bits_per_code as u8,
                centroids: centroids.clone(),
                dimension,
                subvector_dim,
                num_centroids,
                training_config: PqTrainingConfig {
                    num_iterations: codebook.training_config.iterations,
                    seed: codebook.training_config.seed,
                    distance_metric: "euclidean".to_string(),
                },
            })
        } else {
            Err(anyhow::anyhow!("Codebook is not ProductQuantization type"))
        }
    }

    /// Convert cached codebook to PqCodebook
    fn convert_cached_to_pq_codebook(&self, codebook: &Arc<Vec<Vec<f32>>>) -> Result<PqCodebook> {
        // The cached codebook is already in nested format: [subvectors][centroids * dim]
        let num_subvectors = codebook.len() as u32;
        if num_subvectors == 0 {
            return Err(anyhow::anyhow!("Empty codebook"));
        }

        // Determine bits per code from number of centroids
        let total_centroid_data = codebook[0].len();
        let num_centroids = 256; // Default to 8-bit
        let subvector_dim = total_centroid_data / num_centroids;
        let bits_per_code = 8u8;

        // Reshape to nested structure
        let mut nested_centroids = Vec::with_capacity(num_subvectors as usize);
        for subvec_data in codebook.iter() {
            let mut subvec_centroids = Vec::with_capacity(num_centroids);
            for centroid_idx in 0..num_centroids {
                let start = centroid_idx * subvector_dim;
                let end = start + subvector_dim;
                if end <= subvec_data.len() {
                    subvec_centroids.push(subvec_data[start..end].to_vec());
                } else {
                    subvec_centroids.push(vec![0.0; subvector_dim]);
                }
            }
            nested_centroids.push(subvec_centroids);
        }

        Ok(PqCodebook {
            num_subvectors,
            bits_per_code,
            centroids: nested_centroids,
            dimension: num_subvectors as usize * subvector_dim,
            subvector_dim,
            num_centroids,
            training_config: PqTrainingConfig {
                num_iterations: 100,
                seed: None,
                distance_metric: "euclidean".to_string(),
            },
        })
    }

    /// Create codebook metadata from quantization configuration
    pub fn create_from_config(
        &self,
        collection_id: &str,
        config: &crate::proto::proximadb_v1::QuantizationConfig,
        dimension: usize,
    ) -> QuantizationCodebookMetadata {
        let mut metadata = QuantizationCodebookMetadata {
            collection_id: collection_id.to_string(),
            binary_codebook: None,
            int8_codebook: None,
            pq_codebooks: HashMap::new(),
            created_at: chrono::Utc::now().timestamp(),
            training_samples: 0,
            schema_version: 1,
        };

        // Add binary codebook if configured
        if config.enable_binary.unwrap_or(false) {
            metadata.binary_codebook = Some(BinaryCodebook {
                threshold: if config.binary_threshold.unwrap_or(0.0) != 0.0 {
                    config.binary_threshold.unwrap_or(0.0)
                } else {
                    0.0
                },
                mean: None,
                dimension,
            });
        }

        // Add INT8 codebook if configured (INT8 doesn't have a specific enable flag)
        // Use enable_progressive_search as a proxy for INT8 support
        if config.enable_progressive_search.unwrap_or(false) {
            metadata.int8_codebook = Some(Int8Codebook {
                scale: 1.0,
                zero_point: 0,
                min_value: -1.0,
                max_value: 1.0,
                dimension,
            });
        }

        // Add PQ codebooks if configured
        if config.enable_pq.unwrap_or(false) {
            let bits = config.pq_bits.unwrap_or(8);
            let subvectors = config.pq_segments.unwrap_or(8);
            let key = format!("pq{}_{}", bits, subvectors);

            // Create empty PQ codebook structure
            let subvector_dim = dimension.div_ceil(subvectors as usize);
            let num_centroids = 1 << bits;

            metadata.pq_codebooks.insert(
                key,
                PqCodebook {
                    num_subvectors: subvectors,
                    bits_per_code: bits as u8,
                    centroids: vec![
                        vec![vec![0.0; subvector_dim]; num_centroids];
                        subvectors as usize
                    ],
                    dimension,
                    subvector_dim,
                    num_centroids,
                    training_config: PqTrainingConfig {
                        num_iterations: 100,
                        seed: None,
                        distance_metric: "euclidean".to_string(),
                    },
                },
            );
        }

        metadata
    }
}

impl Default for CodebookSerializer {
    fn default() -> Self {
        Self::new()
    }
}

/// ProximaBlock footer with codebook metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProximaBlockFooter {
    /// Magic number for validation
    pub magic: [u8; 8],

    /// Block metadata offset
    pub metadata_offset: u64,

    /// Block metadata size
    pub metadata_size: u64,

    /// Codebook metadata offset (optional)
    pub codebook_offset: Option<u64>,

    /// Codebook metadata size (optional)
    pub codebook_size: Option<u64>,

    /// Block index offset
    pub index_offset: u64,

    /// Block index size
    pub index_size: u64,

    /// Footer checksum
    pub checksum: u32,
}

impl ProximaBlockFooter {
    pub const MAGIC: [u8; 8] = *b"PROXBLK1";
    pub const FOOTER_SIZE: usize = 60; // Fixed size footer (8+8+8+8+8+8+8+4)

    /// Write footer to buffer
    pub fn write_to_buffer(&self, buffer: &mut BytesMut) {
        buffer.extend_from_slice(&self.magic);
        buffer.extend_from_slice(&self.metadata_offset.to_le_bytes());
        buffer.extend_from_slice(&self.metadata_size.to_le_bytes());

        if let Some(offset) = self.codebook_offset {
            buffer.extend_from_slice(&offset.to_le_bytes());
        } else {
            buffer.extend_from_slice(&0u64.to_le_bytes());
        }

        if let Some(size) = self.codebook_size {
            buffer.extend_from_slice(&size.to_le_bytes());
        } else {
            buffer.extend_from_slice(&0u64.to_le_bytes());
        }

        buffer.extend_from_slice(&self.index_offset.to_le_bytes());
        buffer.extend_from_slice(&self.index_size.to_le_bytes());
        buffer.extend_from_slice(&self.checksum.to_le_bytes());
    }

    /// Read footer from buffer
    pub fn read_from_buffer(buffer: &[u8]) -> Result<Self> {
        if buffer.len() < Self::FOOTER_SIZE {
            return Err(anyhow::anyhow!("Buffer too small for footer"));
        }

        let mut magic = [0u8; 8];
        magic.copy_from_slice(&buffer[0..8]);

        if magic != Self::MAGIC {
            return Err(anyhow::anyhow!("Invalid magic number in footer"));
        }

        let metadata_offset = u64::from_le_bytes([
            buffer[8], buffer[9], buffer[10], buffer[11], buffer[12], buffer[13], buffer[14],
            buffer[15],
        ]);

        let metadata_size = u64::from_le_bytes([
            buffer[16], buffer[17], buffer[18], buffer[19], buffer[20], buffer[21], buffer[22],
            buffer[23],
        ]);

        let codebook_offset_raw = u64::from_le_bytes([
            buffer[24], buffer[25], buffer[26], buffer[27], buffer[28], buffer[29], buffer[30],
            buffer[31],
        ]);

        let codebook_size_raw = u64::from_le_bytes([
            buffer[32], buffer[33], buffer[34], buffer[35], buffer[36], buffer[37], buffer[38],
            buffer[39],
        ]);

        let codebook_offset = if codebook_offset_raw == 0 {
            None
        } else {
            Some(codebook_offset_raw)
        };
        let codebook_size = if codebook_size_raw == 0 {
            None
        } else {
            Some(codebook_size_raw)
        };

        let index_offset = u64::from_le_bytes([
            buffer[40], buffer[41], buffer[42], buffer[43], buffer[44], buffer[45], buffer[46],
            buffer[47],
        ]);

        let index_size = u64::from_le_bytes([
            buffer[48], buffer[49], buffer[50], buffer[51], buffer[52], buffer[53], buffer[54],
            buffer[55],
        ]);

        let checksum = u32::from_le_bytes([buffer[56], buffer[57], buffer[58], buffer[59]]);

        Ok(Self {
            magic,
            metadata_offset,
            metadata_size,
            codebook_offset,
            codebook_size,
            index_offset,
            index_size,
            checksum,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codebook_serialization() {
        let metadata = QuantizationCodebookMetadata {
            collection_id: "test_collection".to_string(),
            binary_codebook: Some(BinaryCodebook {
                threshold: 0.5,
                mean: Some(vec![0.1, 0.2, 0.3]),
                dimension: 3,
            }),
            int8_codebook: Some(Int8Codebook {
                scale: 0.01,
                zero_point: 0,
                min_value: -1.0,
                max_value: 1.0,
                dimension: 3,
            }),
            pq_codebooks: HashMap::new(),
            created_at: 1234567890,
            training_samples: 1000,
            schema_version: 1,
        };

        let serializer = CodebookSerializer::new();

        // Test footer serialization
        let bytes = serializer.serialize_for_footer(&metadata).unwrap();
        let deserialized = serializer.deserialize_from_footer(&bytes).unwrap();
        assert_eq!(deserialized.collection_id, metadata.collection_id);
        assert_eq!(deserialized.training_samples, metadata.training_samples);

        // Test sidecar serialization
        let json = serializer.serialize_for_sidecar(&metadata).unwrap();
        let deserialized = serializer.deserialize_from_sidecar(&json).unwrap();
        assert_eq!(deserialized.collection_id, metadata.collection_id);
        assert_eq!(deserialized.schema_version, metadata.schema_version);
    }

    #[test]
    fn test_proxima_block_footer() {
        let footer = ProximaBlockFooter {
            magic: ProximaBlockFooter::MAGIC,
            metadata_offset: 1024,
            metadata_size: 512,
            codebook_offset: Some(2048),
            codebook_size: Some(256),
            index_offset: 4096,
            index_size: 1024,
            checksum: 0x12345678,
        };

        let mut buffer = BytesMut::new();
        footer.write_to_buffer(&mut buffer);

        assert_eq!(buffer.len(), ProximaBlockFooter::FOOTER_SIZE);

        let deserialized = ProximaBlockFooter::read_from_buffer(&buffer).unwrap();
        assert_eq!(deserialized.metadata_offset, footer.metadata_offset);
        assert_eq!(deserialized.codebook_offset, footer.codebook_offset);
        assert_eq!(deserialized.checksum, footer.checksum);
    }
}
