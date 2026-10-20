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

//! AXIS Index Serialization
//!
//! Provides efficient serialization and deserialization for all AXIS index types
//! with support for incremental updates and delta management.

use crate::index::axis::filterable_metadata::FilterableHnswMetadata;
use crate::index::axis::{
    AxisHnswConfig, AxisHnswIndex, SerializableIvfColdTier, SerializableIvfState,
    SerializableIvfStateV1, SerializableIvfWarmTier, UnifiedIvfIndex,
};
// pub-use so dual_store_ivf.rs's own tests can import ColdPathLoadPolicy
// from this module path (TD: that test should import from its own
// module instead — the indirection is the other-agent's WIP).
pub use crate::index::axis::ColdPathLoadPolicy;
use crate::storage::persistence::filesystem::{FileSystem, FilesystemError};
use bincode;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

/// Magic bytes for index format identification
const AXIS_MAGIC: &[u8; 4] = b"AXIS";
/// File-layout version. v1 = single `SerializableIvfState` body. v2 (ADR-023
/// T-C) = cold-first two-blob body `[COLD][WARM]` (WARM = one bincode tier).
/// v3 (ADR-023 R3 (b)) = same `[COLD][WARM]` framing but WARM is concatenated
/// per-cluster chunks with a byte-directory in the header, so a single cluster's
/// fp32 is range-readable. v1/v2 files still load via their respective paths.
const VERSION: u16 = 3;
/// IVF tier-payload version (shares the value used by `SerializableIvf*`).
const IVF_TIER_VERSION: u32 = 2;

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

/// Decode an IVF body, tolerating both the ADR-023 v2 layout (with the COLD
/// `binary_tier`) and the legacy v1 layout (no binary tier).
///
/// v2 is attempted first. This is sound because a v1 payload has no trailing
/// bytes for `binary_tier`, so the v2 decode hits EOF and we fall through to v1;
/// a v2 payload always decodes as v2 before the fallback is reached. A v1 body
/// yields an empty `binary_tier` (the index reconstructs binary codes from fp32
/// on restore, as it did before ADR-023).
fn decode_ivf_state(index_data: &[u8]) -> Result<SerializableIvfState> {
    match bincode::deserialize::<SerializableIvfState>(index_data) {
        Ok(state) => Ok(state),
        Err(_) => {
            let v1: SerializableIvfStateV1 = bincode::deserialize(index_data)?;
            Ok(SerializableIvfState {
                version: v1.version,
                vector_count: v1.vector_count,
                config: v1.config,
                centroids: v1.centroids,
                vectors: v1.vectors,
                binary_tier: Vec::new(),
            })
        }
    }
}

/// Split a framed index blob into its `IndexHeader` and the trailing body bytes
/// (after the `[u32 header_len][header]` prefix). Validates magic + that the
/// file-layout version is not newer than this build understands.
fn split_header(data: &[u8]) -> Result<(IndexHeader, &[u8])> {
    if data.len() < 4 {
        return Err(SerializationError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "Data too short",
        )));
    }
    let header_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    if data.len() < 4 + header_len {
        return Err(SerializationError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "Header incomplete",
        )));
    }
    let header: IndexHeader = bincode::deserialize(&data[4..4 + header_len])?;
    if header.magic != *AXIS_MAGIC {
        return Err(SerializationError::InvalidMagic);
    }
    if header.version > VERSION {
        return Err(SerializationError::UnsupportedVersion(header.version));
    }
    Ok((header, &data[4 + header_len..]))
}

fn unix_now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

/// Serialization error types
#[derive(Debug, thiserror::Error)]
pub enum SerializationError {
    /// Underlying I/O error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Bincode encode/decode failure
    #[error("Bincode error: {0}")]
    Bincode(#[from] bincode::Error),

    /// Apache Avro encode/decode failure
    #[error("Avro error: {0}")]
    Avro(String),

    /// Compression or decompression failure
    #[error("Compression error: {0}")]
    Compression(String),

    /// The file does not start with the expected `AXIS` magic bytes
    #[error("Invalid magic bytes")]
    InvalidMagic,

    /// The format version is newer than this build can handle
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u16),

    /// Stored CRC32 does not match the computed checksum
    #[error("Checksum mismatch")]
    ChecksumMismatch,

    /// Unrecognised index type discriminant
    #[error("Unknown index type: {0}")]
    UnknownIndex(String),

    /// Serialization is not implemented for the given index type
    #[error("Serialization not supported for index type: {0}")]
    NotSupported(String),
}

/// Convenience `Result` alias using [`SerializationError`] as the error type
pub type Result<T> = std::result::Result<T, SerializationError>;

/// Index types that can be serialized
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Index {
    /// Hierarchical Navigable Small World graph index
    Hnsw,
    /// Inverted File index with optional product quantization
    Ivf,
    /// Locality Sensitive Hashing index
    Lsh,
    /// Annoy tree-based approximate nearest neighbour index
    Annoy,
    /// Product Quantization index
    Pq,
    /// Flat brute-force index (exact search)
    Flat,
}

/// Backwards-compat alias for [`AxisSerializedIndexMetadata`].
pub type IndexMetadata = AxisSerializedIndexMetadata;

/// Metadata for serialized index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisSerializedIndexMetadata {
    /// Index type
    pub index_type: Index,

    /// Collection ID
    pub collection_id: String,

    /// Number of vectors in index
    pub num_vectors: usize,

    /// Vector dimension
    pub dimension: usize,

    /// Timestamp of serialization
    pub timestamp: u64,

    /// Checksum of the data
    pub checksum: u32,

    /// Is this a delta/incremental update
    pub is_delta: bool,

    /// Base checkpoint ID if this is a delta
    pub base_checkpoint_id: Option<String>,

    /// Custom metadata
    pub custom_metadata: Option<Vec<u8>>,
}

/// Cold-path tier profile (ADR-023 / F2 T-A), carried in
/// [`AxisSerializedIndexMetadata::custom_metadata`] as a bincode blob. Lets the
/// cold-load policy decide binary-first ordering and lets operators see the
/// cold/warm split (success criterion #1: cold tier should be ≈ 1/32 of warm).
/// Held in `custom_metadata` (an opaque `Option<Vec<u8>>`) rather than as new
/// header fields so pre-ADR-023 serialized indexes still deserialize unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ColdPathProfile {
    /// Whether the 1-bit COLD tier is populated.
    pub has_binary_tier: bool,
    /// Serialized bytes of the COLD tier (config + centroids + binary codes) — the
    /// leading blob; also the byte offset of the WARM blob (ADR-023 T-C layout).
    pub cold_tier_bytes: u64,
    /// Serialized bytes of the WARM tier (fp32 vectors) — the trailing blob.
    pub warm_tier_bytes: u64,
    /// CRC32 of the WARM blob (the COLD blob's CRC is `metadata.checksum`, so a
    /// cold-first read validates the COLD tier without touching the WARM bytes).
    pub warm_checksum: u32,
}

/// One cluster's extent within the WARM blob (ADR-023 R3 (b) byte-directory).
/// Offsets are relative to the WARM blob start; the absolute file offset is
/// `4 + header_len + profile.cold_tier_bytes + offset`. Lets a loader
/// `read_range` exactly one cluster's fp32 without downloading the whole tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WarmExtent {
    /// IVF cluster this extent holds the fp32 vectors for.
    pub cluster_id: u32,
    /// Byte offset of the cluster's bincode chunk, relative to the WARM blob start.
    pub offset: u64,
    /// Byte length of the cluster's bincode chunk.
    pub len: u64,
}

/// v3 cold-path metadata (ADR-023 R3 (b)): the v2 [`ColdPathProfile`] plus a
/// per-cluster WARM byte-directory. Serialized into
/// [`AxisSerializedIndexMetadata::custom_metadata`]. Wrapping (rather than adding
/// a `Vec` field to `ColdPathProfile`) keeps `ColdPathProfile` `Copy` and keeps
/// v2 blobs (bare `ColdPathProfile`) deserializing unchanged.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ColdPathMetaV3 {
    /// The v2 cold/warm split profile (cold offset, warm size + CRC).
    pub profile: ColdPathProfile,
    /// Per-cluster WARM extents, in WARM-blob order.
    pub warm_directory: Vec<WarmExtent>,
}

impl AxisSerializedIndexMetadata {
    /// Decode the [`ColdPathProfile`] from `custom_metadata`, if present and in
    /// the ADR-023 format. Tries the v3 [`ColdPathMetaV3`] wrapper first, then a
    /// bare v2 `ColdPathProfile`. Returns `None` for indexes serialized before
    /// ADR-023 (whose `custom_metadata` held a 1-byte marker or was absent). The
    /// 21-byte v2 blob never mis-decodes as the ≥29-byte v3 wrapper.
    pub fn cold_path_profile(&self) -> Option<ColdPathProfile> {
        let bytes = self.custom_metadata.as_ref()?;
        if let Ok(meta) = bincode::deserialize::<ColdPathMetaV3>(bytes) {
            return Some(meta.profile);
        }
        bincode::deserialize::<ColdPathProfile>(bytes).ok()
    }

    /// Decode the full v3 cold-path metadata (profile + WARM byte-directory).
    /// `None` for v1/v2 indexes (no directory) — callers fall back to a
    /// whole-WARM-blob decode.
    pub fn cold_path_meta(&self) -> Option<ColdPathMetaV3> {
        let bytes = self.custom_metadata.as_ref()?;
        bincode::deserialize::<ColdPathMetaV3>(bytes).ok()
    }

    /// The per-cluster WARM byte-directory, if this is a v3 index.
    pub fn warm_directory(&self) -> Option<Vec<WarmExtent>> {
        self.cold_path_meta().map(|m| m.warm_directory)
    }
}

/// Result of an ADR-023 T-C cold-load. Carries the (possibly `ColdBinaryOnly`)
/// index plus, for `BinaryFirstThenRerank`, the deferred WARM bytes to apply once
/// the fp32 tier is fetched (`IndexSerializer::decode_warm_tier` +
/// `UnifiedIvfIndex::restore_warm_tier`).
pub struct ColdLoadResult {
    /// The loaded index — `ColdBinaryOnly` (Stage-1-only) when `warm` is `Some`,
    /// else `FullTwoStage`.
    pub index: UnifiedIvfIndex,
    /// The deserialized header metadata (cold-path profile, vector count, …).
    pub metadata: AxisSerializedIndexMetadata,
    /// Deferred WARM fp32 blob (`Some` only for `BinaryFirstThenRerank`).
    pub warm: Option<Vec<u8>>,
}

/// Result of a byte-RANGE cold load (ADR-023 R3 (b),
/// [`IndexSerializer::cold_load_ranged`]). Carries the `ColdBinaryOnly` index
/// plus what's needed to range-fetch WARM clusters on demand — without ever
/// downloading the whole WARM tier.
pub struct RangedColdLoad {
    /// The `ColdBinaryOnly` index — serves Stage-1 immediately.
    pub index: UnifiedIvfIndex,
    /// Header metadata (cold-path profile + WARM byte-directory).
    pub metadata: AxisSerializedIndexMetadata,
    /// Absolute file offset of the WARM blob: `4 + header_len + cold_tier_bytes`.
    /// Add an extent's `offset` to range-read that cluster's fp32.
    pub warm_base: u64,
    /// Per-cluster WARM byte-directory. Empty for v1/v2 files (no per-cluster
    /// framing) — callers then fall back to a whole-WARM read.
    pub directory: Vec<WarmExtent>,
}

/// Map a [`FilesystemError`] into a [`SerializationError`] for the ranged loader.
fn fs_err(e: FilesystemError) -> SerializationError {
    SerializationError::Io(std::io::Error::other(e.to_string()))
}

/// Header for serialized index file
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IndexHeader {
    /// Magic bytes (AXIS)
    pub magic: [u8; 4],

    /// Format version
    pub version: u16,

    /// Metadata
    pub metadata: AxisSerializedIndexMetadata,
}

/// Checkpoint for incremental updates
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IndexCheckpoint {
    /// Unique checkpoint ID
    pub checkpoint_id: String,

    /// Timestamp
    pub timestamp: u64,

    /// Full index data
    pub index_data: Vec<u8>,

    /// Metadata
    pub metadata: AxisSerializedIndexMetadata,
}

/// Delta update for incremental changes
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IndexDelta {
    /// Base checkpoint this delta applies to
    pub base_checkpoint_id: String,

    /// Delta ID
    pub delta_id: String,

    /// Timestamp
    pub timestamp: u64,

    /// Operations in this delta
    pub operations: Vec<DeltaOperation>,
}

/// Operations that can be applied as deltas
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum DeltaOperation {
    /// Add new vectors
    AddVectors {
        /// List of `(external_id, vector_data)` pairs to insert
        vectors: Vec<(String, Vec<f32>)>,
    },

    /// Remove vectors by ID
    RemoveVectors {
        /// External IDs of vectors to delete
        vector_ids: Vec<String>,
    },

    /// Update existing vectors
    UpdateVectors {
        /// List of `(external_id, new_vector_data)` pairs
        updates: Vec<(String, Vec<f32>)>,
    },

    /// Rebuild specific parts
    RebuildPartial {
        /// Internal node IDs that need to be rebuilt
        affected_nodes: Vec<usize>,
    },
}

/// Main serialization handler for AXIS indexes
pub struct IndexSerializer;

impl IndexSerializer {
    /// Serialize HNSW index to bytes
    pub fn serialize_hnsw(index: &AxisHnswIndex, collection_id: &str) -> Result<Vec<u8>> {
        info!("Serializing HNSW index for collection {}", collection_id);

        // Create metadata
        let metadata = AxisSerializedIndexMetadata {
            index_type: crate::index::axis::storage::serialization::Index::Hnsw,
            collection_id: collection_id.to_string(),
            num_vectors: index.len(),
            dimension: index.dimension(),
            timestamp: unix_now_secs(),
            checksum: 0, // Will be calculated after serialization
            is_delta: false,
            base_checkpoint_id: None,
            custom_metadata: None,
        };

        // Serialize index data
        let index_data = index.serialize_internal()?;

        // Calculate checksum
        let checksum = proximadb_kernel::checksum::crc32_fast(&index_data);

        // Create header with updated checksum
        let mut final_metadata = metadata;
        final_metadata.checksum = checksum;

        let header = IndexHeader {
            magic: *AXIS_MAGIC,
            version: VERSION,
            metadata: final_metadata,
        };

        // Combine header and data
        let mut result = Vec::new();
        let header_bytes = bincode::serialize(&header)?;
        result.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
        result.extend_from_slice(&header_bytes);
        result.extend_from_slice(&index_data);

        info!("Serialized HNSW index: {} bytes", result.len());
        Ok(result)
    }

    /// Deserialize HNSW index from bytes
    pub fn deserialize_hnsw(
        data: &[u8],
        config: &AxisHnswConfig,
    ) -> Result<(AxisHnswIndex, AxisSerializedIndexMetadata)> {
        info!("Deserializing HNSW index");

        // Read header length
        if data.len() < 4 {
            return Err(SerializationError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Data too short",
            )));
        }

        let header_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

        // Read header
        if data.len() < 4 + header_len {
            return Err(SerializationError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Header incomplete",
            )));
        }

        let header: IndexHeader = bincode::deserialize(&data[4..4 + header_len])?;

        // Validate magic and version
        if header.magic != *AXIS_MAGIC {
            return Err(SerializationError::InvalidMagic);
        }

        if header.version > VERSION {
            return Err(SerializationError::UnsupportedVersion(header.version));
        }

        // Validate checksum
        let index_data = &data[4 + header_len..];
        let checksum = proximadb_kernel::checksum::crc32_fast(index_data);

        if checksum != header.metadata.checksum {
            return Err(SerializationError::ChecksumMismatch);
        }

        // Deserialize index
        let index = AxisHnswIndex::deserialize_internal(index_data, config)?;

        info!(
            "Deserialized HNSW index with {} vectors",
            header.metadata.num_vectors
        );
        Ok((index, header.metadata))
    }

    /// Serialize a trained IVF index to bytes (TD-087 Slice B). Async because
    /// the IVF stores (`AdaptiveStore` posting lists + vector store) are async.
    /// The payload is the `SerializableIvfState` (config essentials + centroids +
    /// raw vectors); the header records vector count, dimension, and a binary-tier
    /// marker in `custom_metadata`.
    pub async fn serialize_ivf(index: &UnifiedIvfIndex, collection_id: &str) -> Result<Vec<u8>> {
        info!("Serializing IVF index for collection {}", collection_id);

        let state = index
            .export_state()
            .await
            .map_err(|e| SerializationError::Io(std::io::Error::other(e.to_string())))?;

        // ADR-023 T-C: the cold-first (v2) layout carries posting-list membership
        // only inside the binary tier `(id, code, cluster_id)`. A collection with
        // no binary tier (use_binary off) has no membership to range-read, so
        // cold-first can't rebuild its posting lists and there's no Stage-1
        // benefit anyway — serialize it in the legacy v1 single-blob layout
        // (full restore via `add_vector` replay rebuilds membership).
        if state.binary_tier.is_empty() {
            return Self::serialize_ivf_v1_legacy(&state, collection_id, index);
        }

        // ADR-023 T-C cold-first layout: split the body into a COLD blob
        // (config + centroids + 1-bit codes) written FIRST, then a WARM blob
        // (fp32 vectors). A cold-first loader range-reads `[header][COLD]` and
        // serves Stage-1 without the fp32. `metadata.checksum` covers the COLD
        // blob; `profile.warm_checksum` covers the WARM blob.
        // ADR-023 R3: group the WARM fp32 vectors by IVF cluster (cluster_id from
        // the binary tier) so warm-apply can install one cluster at a time.
        let warm_clusters = {
            let id_cluster: std::collections::HashMap<&str, u32> = state
                .binary_tier
                .iter()
                .map(|(id, _, c)| (id.as_str(), *c))
                .collect();
            let mut by_cluster: std::collections::BTreeMap<u32, Vec<(String, Vec<f32>)>> =
                std::collections::BTreeMap::new();
            for (id, v) in state.vectors {
                let c = id_cluster.get(id.as_str()).copied().unwrap_or(0);
                by_cluster.entry(c).or_default().push((id, v));
            }
            by_cluster.into_iter().collect::<Vec<_>>()
        };
        let cold = SerializableIvfColdTier {
            version: IVF_TIER_VERSION,
            config: state.config.clone(),
            centroids: state.centroids.clone(),
            binary_tier: state.binary_tier,
        };
        let cold_bytes = bincode::serialize(&cold)?;
        let cold_checksum = proximadb_kernel::checksum::crc32_fast(&cold_bytes);

        // ADR-023 R3 (b): write each cluster's fp32 as an independently
        // range-readable bincode chunk, concatenated into the WARM blob, with a
        // byte-directory recording each cluster's [offset, len]. A loader can then
        // `read_range` exactly one cluster's fp32 (offset relative to the WARM
        // blob start). The chunk payload is the cluster's `Vec<(id, fp32)>`; the
        // cluster_id lives in the directory.
        let mut warm_bytes = Vec::new();
        let mut warm_directory = Vec::with_capacity(warm_clusters.len());
        for (cluster_id, vectors) in &warm_clusters {
            let chunk = bincode::serialize(vectors)?;
            warm_directory.push(WarmExtent {
                cluster_id: *cluster_id,
                offset: warm_bytes.len() as u64,
                len: chunk.len() as u64,
            });
            warm_bytes.extend_from_slice(&chunk);
        }
        let warm_checksum = proximadb_kernel::checksum::crc32_fast(&warm_bytes);

        let profile = ColdPathProfile {
            has_binary_tier: !cold.binary_tier.is_empty(),
            cold_tier_bytes: cold_bytes.len() as u64,
            warm_tier_bytes: warm_bytes.len() as u64,
            warm_checksum,
        };
        // v3 metadata: profile + the WARM byte-directory (rides in custom_metadata
        // so one small header range-read yields both).
        let meta = ColdPathMetaV3 {
            profile,
            warm_directory,
        };

        let metadata = AxisSerializedIndexMetadata {
            index_type: Index::Ivf,
            collection_id: collection_id.to_string(),
            num_vectors: index.len(),
            dimension: index.dimension(),
            timestamp: unix_now_secs(),
            checksum: cold_checksum, // COLD blob CRC (validated on cold-first read)
            is_delta: false,
            base_checkpoint_id: None,
            // ADR-023 R3 (b): cold-path tier profile + WARM byte-directory.
            custom_metadata: Some(bincode::serialize(&meta)?),
        };

        let header = IndexHeader {
            magic: *AXIS_MAGIC,
            version: VERSION,
            metadata,
        };

        let mut result = Vec::new();
        let header_bytes = bincode::serialize(&header)?;
        result.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
        result.extend_from_slice(&header_bytes);
        result.extend_from_slice(&cold_bytes); // COLD first (range-readable)
        result.extend_from_slice(&warm_bytes); // WARM second (deferrable)

        info!("Serialized IVF index: {} bytes", result.len());
        Ok(result)
    }

    /// Legacy v1 single-blob layout (ADR-023 T-C): the whole `SerializableIvfState`
    /// as one bincode body, restored via `add_vector` replay (which rebuilds
    /// posting-list membership). Used for collections with no binary tier, whose
    /// COLD tier carries no membership to range-read.
    fn serialize_ivf_v1_legacy(
        state: &SerializableIvfState,
        collection_id: &str,
        index: &UnifiedIvfIndex,
    ) -> Result<Vec<u8>> {
        let body = bincode::serialize(state)?;
        let checksum = proximadb_kernel::checksum::crc32_fast(&body);
        let profile = ColdPathProfile {
            has_binary_tier: false,
            cold_tier_bytes: 0,
            warm_tier_bytes: body.len() as u64,
            warm_checksum: 0,
        };
        let metadata = AxisSerializedIndexMetadata {
            index_type: Index::Ivf,
            collection_id: collection_id.to_string(),
            num_vectors: index.len(),
            dimension: index.dimension(),
            timestamp: unix_now_secs(),
            checksum,
            is_delta: false,
            base_checkpoint_id: None,
            custom_metadata: Some(bincode::serialize(&profile)?),
        };
        let header = IndexHeader {
            magic: *AXIS_MAGIC,
            version: 1, // legacy single-blob layout
            metadata,
        };
        let mut result = Vec::new();
        let header_bytes = bincode::serialize(&header)?;
        result.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
        result.extend_from_slice(&header_bytes);
        result.extend_from_slice(&body);
        Ok(result)
    }

    /// Deserialize a trained IVF index from bytes (TD-087 Slice B). The payload
    /// is self-describing (embedded config), so no external config is needed:
    /// the centroids are installed and `add_vector` is replayed to rebuild
    /// posting lists, binary codes, PQ codes, and the vector store.
    pub async fn deserialize_ivf(
        data: &[u8],
    ) -> Result<(UnifiedIvfIndex, AxisSerializedIndexMetadata)> {
        info!("Deserializing IVF index");
        let (header, body) = split_header(data)?;

        if header.version >= 2 {
            // ADR-023 T-C cold-first layout: `[COLD][WARM]`. Load both for a full
            // (FullTwoStage) index — restore COLD then install WARM.
            let (cold, warm_bytes) = Self::read_cold_blob(&header, body)?;
            // Validate the WARM blob too (full read).
            let profile = Self::cold_profile(&header)?;
            if proximadb_kernel::checksum::crc32_fast(warm_bytes) != profile.warm_checksum {
                return Err(SerializationError::ChecksumMismatch);
            }
            // v3: directory-framed per-cluster chunks; v2: one bincode tier.
            let warm_flat = Self::decode_warm_flat(&header, warm_bytes)?;
            let mut index = Self::new_cold_index(&header, cold).await?;
            index
                .restore_warm_tier(warm_flat)
                .map_err(|e| SerializationError::Io(std::io::Error::other(e.to_string())))?;
            info!(
                "Deserialized IVF index with {} vectors",
                header.metadata.num_vectors
            );
            return Ok((index, header.metadata));
        }

        // Legacy v1 single-blob layout.
        if proximadb_kernel::checksum::crc32_fast(body) != header.metadata.checksum {
            return Err(SerializationError::ChecksumMismatch);
        }
        let state = decode_ivf_state(body)?;
        let config = state.config.to_config();
        let mut index = UnifiedIvfIndex::new(header.metadata.collection_id.clone(), config)
            .map_err(|e| SerializationError::Io(std::io::Error::other(e.to_string())))?;
        index
            .restore_state(state)
            .await
            .map_err(|e| SerializationError::Io(std::io::Error::other(e.to_string())))?;
        info!(
            "Deserialized IVF index with {} vectors",
            header.metadata.num_vectors
        );
        Ok((index, header.metadata))
    }

    /// Read + validate the cold-path profile from a v2 header.
    fn cold_profile(header: &IndexHeader) -> Result<ColdPathProfile> {
        header.metadata.cold_path_profile().ok_or_else(|| {
            SerializationError::Io(std::io::Error::other(
                "cold-first (v2) IVF index missing its cold-path profile",
            ))
        })
    }

    /// Split a v2 body into its COLD blob (validated against `metadata.checksum`)
    /// and the trailing WARM bytes (validated by the caller when needed).
    fn read_cold_blob<'a>(
        header: &IndexHeader,
        body: &'a [u8],
    ) -> Result<(SerializableIvfColdTier, &'a [u8])> {
        let profile = Self::cold_profile(header)?;
        let cold_len = profile.cold_tier_bytes as usize;
        if body.len() < cold_len {
            return Err(SerializationError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "cold blob truncated",
            )));
        }
        let cold_bytes = &body[..cold_len];
        if proximadb_kernel::checksum::crc32_fast(cold_bytes) != header.metadata.checksum {
            return Err(SerializationError::ChecksumMismatch);
        }
        let cold: SerializableIvfColdTier = bincode::deserialize(cold_bytes)?;
        Ok((cold, &body[cold_len..]))
    }

    /// Construct a `ColdBinaryOnly` index from a decoded COLD tier.
    async fn new_cold_index(
        header: &IndexHeader,
        cold: SerializableIvfColdTier,
    ) -> Result<UnifiedIvfIndex> {
        let config = cold.config.to_config();
        let mut index = UnifiedIvfIndex::new(header.metadata.collection_id.clone(), config)
            .map_err(|e| SerializationError::Io(std::io::Error::other(e.to_string())))?;
        index
            .restore_cold_only(cold)
            .await
            .map_err(|e| SerializationError::Io(std::io::Error::other(e.to_string())))?;
        Ok(index)
    }

    /// ADR-023 T-C cold-first load: read ONLY the COLD blob (range
    /// `[header][COLD]`), returning a `ColdBinaryOnly` index that serves Stage-1
    /// immediately plus the deferred WARM bytes. Apply the WARM tier later with
    /// [`decode_warm_tier`](Self::decode_warm_tier) + `restore_warm_tier` (T-E).
    /// v1 (legacy) files have no separable COLD tier, so they fall back to a full
    /// eager load with `warm = None`.
    pub async fn deserialize_ivf_cold_only(data: &[u8]) -> Result<ColdLoadResult> {
        let (header, body) = split_header(data)?;
        if header.version < 2 {
            let (index, metadata) = Self::deserialize_ivf(data).await?;
            return Ok(ColdLoadResult {
                index,
                metadata,
                warm: None,
            });
        }
        let (cold, warm_bytes) = Self::read_cold_blob(&header, body)?;
        let warm = Some(warm_bytes.to_vec());
        let index = Self::new_cold_index(&header, cold).await?;
        info!(
            "Cold-first load: {} vectors served Stage-1 ({} cold bytes, {} warm deferred)",
            header.metadata.num_vectors,
            header
                .metadata
                .cold_path_profile()
                .map(|p| p.cold_tier_bytes)
                .unwrap_or(0),
            warm_bytes.len(),
        );
        Ok(ColdLoadResult {
            index,
            metadata: header.metadata,
            warm,
        })
    }

    /// Load an IVF index under a [`ColdPathLoadPolicy`] (ADR-023 T-C). `FullEager`
    /// loads both tiers; `BinaryFirstThenRerank` loads the COLD tier first and
    /// defers the WARM bytes in the result.
    pub async fn load_ivf_with_policy(
        data: &[u8],
        policy: ColdPathLoadPolicy,
    ) -> Result<ColdLoadResult> {
        match policy {
            ColdPathLoadPolicy::FullEager => {
                let (index, metadata) = Self::deserialize_ivf(data).await?;
                Ok(ColdLoadResult {
                    index,
                    metadata,
                    warm: None,
                })
            }
            ColdPathLoadPolicy::BinaryFirstThenRerank => {
                Self::deserialize_ivf_cold_only(data).await
            }
        }
    }

    /// Decode the deferred WARM bytes into a flat fp32 list (ADR-023 T-E
    /// FullEager / whole-tier `restore_warm_tier`).
    pub fn decode_warm_tier(warm_bytes: &[u8]) -> Result<Vec<(String, Vec<f32>)>> {
        let warm: SerializableIvfWarmTier = bincode::deserialize(warm_bytes)?;
        Ok(warm.into_flat())
    }

    /// Decode the deferred WARM bytes into per-cluster fp32 extents (ADR-023 R3):
    /// `(cluster_id, [(id, fp32)])`. The background warm-apply installs these one
    /// cluster at a time so the fill interleaves with serving. v2 layout (whole
    /// bincode tier); for v3 directory-framed bytes use [`decode_warm_clusters_dir`].
    pub fn decode_warm_clusters(warm_bytes: &[u8]) -> Result<Vec<(u32, Vec<(String, Vec<f32>)>)>> {
        let warm: SerializableIvfWarmTier = bincode::deserialize(warm_bytes)?;
        Ok(warm.clusters)
    }

    /// Decode one cluster's WARM chunk (ADR-023 R3 (b)): the bincode payload at a
    /// [`WarmExtent`] is the cluster's `Vec<(id, fp32)>`. Used by the ranged
    /// loader after a single-extent `read_range`.
    pub fn decode_warm_cluster_chunk(chunk: &[u8]) -> Result<Vec<(String, Vec<f32>)>> {
        Ok(bincode::deserialize(chunk)?)
    }

    /// Decode v3 directory-framed WARM bytes into per-cluster extents by slicing
    /// each [`WarmExtent`] out of the concatenated blob. Pairs the directory's
    /// `cluster_id`s with the decoded chunks.
    pub fn decode_warm_clusters_dir(
        warm_bytes: &[u8],
        directory: &[WarmExtent],
    ) -> Result<Vec<(u32, Vec<(String, Vec<f32>)>)>> {
        let mut out = Vec::with_capacity(directory.len());
        for ext in directory {
            let (start, end) = (ext.offset as usize, (ext.offset + ext.len) as usize);
            let chunk = warm_bytes.get(start..end).ok_or_else(|| {
                SerializationError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "warm extent out of bounds",
                ))
            })?;
            out.push((ext.cluster_id, Self::decode_warm_cluster_chunk(chunk)?));
        }
        Ok(out)
    }

    /// Decode the WARM blob into a flat fp32 list, handling both v3 (directory-
    /// framed chunks) and v2 (one bincode tier). Used by the full (`FullEager`)
    /// deserialize path.
    fn decode_warm_flat(
        header: &IndexHeader,
        warm_bytes: &[u8],
    ) -> Result<Vec<(String, Vec<f32>)>> {
        match header.metadata.warm_directory() {
            Some(directory) => {
                let mut flat = Vec::new();
                for (_cluster_id, vectors) in
                    Self::decode_warm_clusters_dir(warm_bytes, &directory)?
                {
                    flat.extend(vectors);
                }
                Ok(flat)
            }
            None => Ok(Self::decode_warm_tier(warm_bytes)?),
        }
    }

    /// Persist a trained IVF index to `path` via the canonical [`FileSystem`]
    /// trait — the SAME abstraction the cold-load path reads through, so
    /// persistence is object-store-aware (`s3://`/`adls://`/`gs://`/`file://`)
    /// rather than local-only (TD-087 Slice B; ADR-023 R3 Slice 4 convergence).
    /// `path` is a backend-relative URI/path; `create_dirs` lets the local
    /// backend make parent dirs (object stores ignore it).
    pub async fn persist_ivf_index(
        index: &UnifiedIvfIndex,
        collection_id: &str,
        path: &str,
        fs: &Arc<dyn FileSystem>,
    ) -> Result<()> {
        let bytes = Self::serialize_ivf(index, collection_id).await?;
        fs.write(
            path,
            &bytes,
            Some(crate::storage::persistence::filesystem::FileOptions {
                create_dirs: true,
                overwrite: true,
                ..Default::default()
            }),
        )
        .await
        .map_err(fs_err)?;
        Ok(())
    }

    /// Load a trained IVF index from `path`. Disk wrapper around `deserialize_ivf`.
    pub async fn load_ivf_index(
        path: &Path,
    ) -> Result<(UnifiedIvfIndex, AxisSerializedIndexMetadata)> {
        let bytes = tokio::fs::read(path).await?;
        Self::deserialize_ivf(&bytes).await
    }

    /// ADR-023 T-E cold-path auto-load: pick `BinaryFirstThenRerank` when the file
    /// has a populated COLD tier (Stage-1 can serve while the WARM tier is
    /// deferred), else `FullEager` (v1 / non-binary collections — no Stage-1
    /// benefit, and their posting-list membership lives only in the full body).
    pub async fn load_ivf_cold_path(data: &[u8]) -> Result<ColdLoadResult> {
        let (header, _) = split_header(data)?;
        let cold_first = header.version >= 2
            && header
                .metadata
                .cold_path_profile()
                .map(|p| p.has_binary_tier)
                .unwrap_or(false);
        let policy = if cold_first {
            ColdPathLoadPolicy::BinaryFirstThenRerank
        } else {
            ColdPathLoadPolicy::FullEager
        };
        Self::load_ivf_with_policy(data, policy).await
    }

    /// ADR-023 R3 (b): cold-load an IVF index via byte-RANGE reads from any
    /// [`FileSystem`] (local, S3, …). Reads only `[header]` + `[COLD]` to start
    /// serving Stage-1 — the WARM fp32 is never downloaded here. Each cluster's
    /// fp32 is fetched on demand with
    /// [`fetch_warm_cluster_ranged`](Self::fetch_warm_cluster_ranged), so an
    /// object-store load never pulls clusters a query doesn't probe. This is the
    /// true I/O-bound version of the cold path (vs. `load_ivf_cold_path`, which
    /// takes the whole file in memory).
    ///
    /// v3 files carry the per-cluster WARM byte-directory (`directory` non-empty).
    /// v1/v2 files load cold here too (empty `directory`); their WARM tier is not
    /// per-cluster range-readable, so callers fall back to a whole-WARM read.
    pub async fn cold_load_ranged(fs: &Arc<dyn FileSystem>, path: &str) -> Result<RangedColdLoad> {
        let (header, header_len) = Self::read_header_ranged(fs, path).await?;
        Self::cold_from_header(fs, path, header, header_len).await
    }

    /// Like [`cold_load_ranged`](Self::cold_load_ranged) but returns `None` for
    /// indexes that aren't cold-first eligible (v1 / non-binary collections —
    /// their COLD tier carries no posting-list membership to range-read). The
    /// manager uses this to choose ranged I/O for binary indexes and a whole-file
    /// load for the rest, reading only the cheap header first to decide.
    pub async fn try_cold_load_ranged(
        fs: &Arc<dyn FileSystem>,
        path: &str,
    ) -> Result<Option<RangedColdLoad>> {
        let (header, header_len) = Self::read_header_ranged(fs, path).await?;
        let eligible = header.version >= 2
            && header
                .metadata
                .cold_path_profile()
                .map(|p| p.has_binary_tier)
                .unwrap_or(false);
        if !eligible {
            return Ok(None);
        }
        Ok(Some(
            Self::cold_from_header(fs, path, header, header_len).await?,
        ))
    }

    /// Range-read the `[u32 header_len][header]` prefix and validate magic/version.
    async fn read_header_ranged(
        fs: &Arc<dyn FileSystem>,
        path: &str,
    ) -> Result<(IndexHeader, u64)> {
        let head4 = fs.read_range(path, 0, 4).await.map_err(fs_err)?;
        if head4.len() < 4 {
            return Err(SerializationError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "missing header length prefix",
            )));
        }
        let header_len = u32::from_le_bytes([head4[0], head4[1], head4[2], head4[3]]) as u64;
        let header_bytes = fs.read_range(path, 4, header_len).await.map_err(fs_err)?;
        let header: IndexHeader = bincode::deserialize(&header_bytes)?;
        if header.magic != *AXIS_MAGIC {
            return Err(SerializationError::InvalidMagic);
        }
        if header.version > VERSION {
            return Err(SerializationError::UnsupportedVersion(header.version));
        }
        Ok((header, header_len))
    }

    /// Read ONLY the COLD blob (validate its CRC), build the `ColdBinaryOnly`
    /// index, and return the WARM base offset + per-cluster directory.
    async fn cold_from_header(
        fs: &Arc<dyn FileSystem>,
        path: &str,
        header: IndexHeader,
        header_len: u64,
    ) -> Result<RangedColdLoad> {
        let profile = Self::cold_profile(&header)?;
        let cold_off = 4 + header_len;
        let cold_bytes = fs
            .read_range(path, cold_off, profile.cold_tier_bytes)
            .await
            .map_err(fs_err)?;
        if proximadb_kernel::checksum::crc32_fast(&cold_bytes) != header.metadata.checksum {
            return Err(SerializationError::ChecksumMismatch);
        }
        let cold: SerializableIvfColdTier = bincode::deserialize(&cold_bytes)?;
        let index = Self::new_cold_index(&header, cold).await?;

        let warm_base = cold_off + profile.cold_tier_bytes;
        let directory = header.metadata.warm_directory().unwrap_or_default();
        info!(
            "Ranged cold-load: served Stage-1 after {} bytes ([header]+[COLD]); {} warm clusters deferred",
            warm_base,
            directory.len(),
        );
        Ok(RangedColdLoad {
            index,
            metadata: header.metadata,
            warm_base,
            directory,
        })
    }

    /// ADR-023 R3 (b): range-fetch and decode ONE cluster's fp32 from the WARM
    /// blob — the only bytes read are that cluster's chunk
    /// (`[warm_base + extent.offset, +extent.len]`).
    pub async fn fetch_warm_cluster_ranged(
        fs: &Arc<dyn FileSystem>,
        path: &str,
        warm_base: u64,
        extent: &WarmExtent,
    ) -> Result<Vec<(String, Vec<f32>)>> {
        let chunk = fs
            .read_range(path, warm_base + extent.offset, extent.len)
            .await
            .map_err(fs_err)?;
        Self::decode_warm_cluster_chunk(&chunk)
    }

    /// Create a checkpoint from current index state
    pub fn create_checkpoint(
        index_type: Index,
        index_data: Vec<u8>,
        collection_id: &str,
    ) -> Result<IndexCheckpoint> {
        let checkpoint_id = format!("chk_{}_{}", collection_id, unix_now_millis());

        let timestamp = unix_now_secs();

        let metadata = AxisSerializedIndexMetadata {
            index_type,
            collection_id: collection_id.to_string(),
            num_vectors: 0, // Will be updated by specific index
            dimension: 0,   // Will be updated by specific index
            timestamp,
            checksum: proximadb_kernel::checksum::crc32_fast(&index_data),
            is_delta: false,
            base_checkpoint_id: None,
            custom_metadata: None,
        };

        Ok(IndexCheckpoint {
            checkpoint_id,
            timestamp,
            index_data,
            metadata,
        })
    }

    /// Apply delta operations to an index
    pub fn apply_delta(checkpoint: &IndexCheckpoint, delta: &IndexDelta) -> Result<Vec<u8>> {
        if delta.base_checkpoint_id != checkpoint.checkpoint_id {
            return Err(SerializationError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Delta base does not match checkpoint",
            )));
        }

        info!(
            "Applying {} delta operations to checkpoint {}",
            delta.operations.len(),
            checkpoint.checkpoint_id
        );

        // This would need index-specific implementation
        // For now, return the original data
        warn!("Delta application not yet implemented for production");
        Ok(checkpoint.index_data.clone())
    }

    /// Serialize any supported index type
    pub fn serialize_generic(
        index: &dyn SerializableIndex,
        collection_id: &str,
    ) -> Result<Vec<u8>> {
        index.serialize_to_bytes(collection_id)
    }
}

/// Trait for indexes that can be serialized
pub trait SerializableIndex: Send + Sync {
    /// Get the index type
    fn index_type(&self) -> Index;

    /// Serialize to bytes
    fn serialize_to_bytes(&self, collection_id: &str) -> Result<Vec<u8>>;

    /// Get number of vectors
    fn len(&self) -> usize;

    /// Get dimension
    fn dimension(&self) -> usize;

    /// Serialize internal data (without header)
    fn serialize_internal(&self) -> Result<Vec<u8>>;

    /// Check if index is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Serializable HNSW configuration (mirrors AxisHnswConfig)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableHnswConfig {
    /// Number of bidirectional links per node (M parameter)
    pub m: usize,
    /// Size of the candidate list during index construction
    pub ef_construction: usize,
    /// Size of the candidate list during search
    pub ef: usize,
    /// Maximum number of graph layers
    pub max_layers: usize,
    /// Distance metric code: 0=L2, 1=Cosine, 2=DotProduct
    pub distance_metric: u8,
}

/// Serializable ID mapping data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableIdMapping {
    /// External ID -> Internal ID pairs
    pub external_to_internal: Vec<(String, usize)>,
    /// Next available internal ID
    pub next_id: usize,
}

/// Serializable vector data (ID + raw bytes)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableVector {
    /// External vector identifier
    pub id: String,
    /// Raw vector bytes (layout depends on quantization settings)
    pub data: Vec<u8>,
}

/// Serializable collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableCollectionConfig {
    /// Vector dimensionality
    pub dimension: usize,
    /// Whether quantization is enabled for this collection
    pub is_quantized: bool,
    /// Quantization method code: 0=INT8, 1=PQ8, 2=PQ4, 3=Binary
    pub quantization_method: Option<u8>,
}

/// V1 HNSW state — pre-TD-064 snapshot layout, no filterable metadata.
///
/// Used only for backward-compatible deserialization of legacy snapshots.
/// New writes always produce `SerializableHnswState` (v2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableHnswStateV1 {
    pub version: u32,
    pub config: SerializableHnswConfig,
    pub collection_config: SerializableCollectionConfig,
    pub id_mapping: SerializableIdMapping,
    pub layers: Vec<((usize, usize), Vec<usize>)>,
    pub max_layer: usize,
    pub entry_point: Option<usize>,
    pub vectors: Vec<SerializableVector>,
    pub quantized_vectors: Vec<(String, Vec<u8>)>,
    pub dimension: usize,
}

/// Complete serializable HNSW state (v2 — current).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableHnswState {
    /// Version for forward compatibility
    pub version: u32,
    /// HNSW configuration
    pub config: SerializableHnswConfig,
    /// Collection configuration for vector interpretation
    pub collection_config: SerializableCollectionConfig,
    /// ID mappings
    pub id_mapping: SerializableIdMapping,
    /// Graph layers: (layer, node_id) -> connections
    pub layers: Vec<((usize, usize), Vec<usize>)>,
    /// Maximum layer in use
    pub max_layer: usize,
    /// Entry point node ID (if any)
    pub entry_point: Option<usize>,
    /// Vector data
    pub vectors: Vec<SerializableVector>,
    /// Quantized vectors (ID -> quantized bytes)
    pub quantized_vectors: Vec<(String, Vec<u8>)>,
    /// Dimension (for validation)
    pub dimension: usize,
    /// TD-064: Filterable metadata cached per external_id for predicate-aware
    /// search. Empty when the index hasn't observed any `add_with_metadata`.
    pub filterable_metadata: Vec<(String, FilterableHnswMetadata)>,
}

impl SerializableHnswState {
    /// Current serialization format version.
    ///
    /// * v1 — vectors + graph + id mapping (legacy)
    /// * v2 — adds `filterable_metadata` (TD-064 predicate-aware search)
    pub const CURRENT_VERSION: u32 = 2;
}

impl From<SerializableHnswStateV1> for SerializableHnswState {
    fn from(v1: SerializableHnswStateV1) -> Self {
        Self {
            version: SerializableHnswState::CURRENT_VERSION,
            config: v1.config,
            collection_config: v1.collection_config,
            id_mapping: v1.id_mapping,
            layers: v1.layers,
            max_layer: v1.max_layer,
            entry_point: v1.entry_point,
            vectors: v1.vectors,
            quantized_vectors: v1.quantized_vectors,
            dimension: v1.dimension,
            filterable_metadata: Vec::new(),
        }
    }
}

/// Extension trait for HNSW serialization
impl SerializableIndex for AxisHnswIndex {
    fn index_type(&self) -> Index {
        Index::Hnsw
    }

    fn serialize_to_bytes(&self, collection_id: &str) -> Result<Vec<u8>> {
        IndexSerializer::serialize_hnsw(self, collection_id)
    }

    fn len(&self) -> usize {
        // Access the actual vector count via id_mapping
        self.id_mapping_len()
    }

    fn dimension(&self) -> usize {
        // Access dimension from vectors collection config
        self.get_dimension()
    }

    fn serialize_internal(&self) -> Result<Vec<u8>> {
        info!("Starting HNSW serialize_internal");

        // 1. Serialize config
        let config = SerializableHnswConfig {
            m: self.get_config_m(),
            ef_construction: self.get_config_ef_construction(),
            ef: self.get_config_ef(),
            max_layers: self.get_config_max_layers(),
            distance_metric: self.get_config_distance_metric_code(),
        };

        // 2. Get collection config
        let (dimension, is_quantized, quant_method) = self.get_collection_config_details();
        let collection_config = SerializableCollectionConfig {
            dimension,
            is_quantized,
            quantization_method: quant_method,
        };

        // 3. Serialize ID mapping
        let id_mapping = self.serialize_id_mapping();

        // 4. Serialize graph layers
        let layers = self.serialize_layers();

        // 5. Get max layer and entry point
        let max_layer = self.get_max_layer();
        let entry_point = self.get_entry_point();

        // 6. Serialize vectors
        let vectors = self.serialize_vectors();

        // 7. Serialize quantized vectors
        let quantized_vectors = self.serialize_quantized_vectors();

        // 8. TD-064: serialize cached filterable metadata
        let filterable_metadata = self.serialize_filterable_metadata();

        // Create complete state
        let state = SerializableHnswState {
            version: SerializableHnswState::CURRENT_VERSION,
            config,
            collection_config,
            id_mapping,
            layers,
            max_layer,
            entry_point,
            vectors,
            quantized_vectors,
            dimension,
            filterable_metadata,
        };

        // Serialize with bincode
        let bytes = bincode::serialize(&state).map_err(SerializationError::Bincode)?;

        info!(
            "HNSW serialize_internal complete: {} bytes, {} vectors, {} layers",
            bytes.len(),
            state.vectors.len(),
            state.layers.len()
        );

        Ok(bytes)
    }
}

/// Extension trait for HNSW deserialization
impl AxisHnswIndex {
    /// Deserialize HNSW index from bytes.
    ///
    /// Tries the current (v2) layout first; falls back to v1 when the
    /// snapshot predates TD-064 (no `filterable_metadata` field), upgrading
    /// in-memory to v2 with empty cached metadata.
    pub fn deserialize_internal(data: &[u8], config: &AxisHnswConfig) -> Result<Self> {
        info!("Starting HNSW deserialize_internal: {} bytes", data.len());

        // Try v2 first
        let state: SerializableHnswState = match bincode::deserialize::<SerializableHnswState>(data)
        {
            Ok(state) => state,
            Err(v2_err) => {
                // Fall back to v1 (no filterable_metadata field)
                match bincode::deserialize::<SerializableHnswStateV1>(data) {
                    Ok(v1) => {
                        warn!(
                            "HNSW snapshot is v1; upgrading in-memory to v2 with empty filterable metadata"
                        );
                        SerializableHnswState::from(v1)
                    }
                    Err(_) => return Err(SerializationError::Bincode(v2_err)),
                }
            }
        };

        // Validate version
        if state.version > SerializableHnswState::CURRENT_VERSION {
            return Err(SerializationError::UnsupportedVersion(state.version as u16));
        }

        // Create new index with deserialized config
        let index = match AxisHnswIndex::new(config.clone(), state.dimension) {
            Ok(idx) => idx,
            Err(e) => {
                return Err(SerializationError::Io(std::io::Error::other(e.to_string())));
            }
        };

        // Capture counts before moving
        let vector_count = state.id_mapping.external_to_internal.len();
        let metadata_count = state.filterable_metadata.len();

        // Restore the index state using public reconstruction method
        if let Err(e) = index.restore_from_state(
            state.id_mapping,
            state.layers,
            state.max_layer,
            state.entry_point,
            state.vectors,
            state.quantized_vectors,
            state.collection_config,
        ) {
            return Err(SerializationError::Io(std::io::Error::other(format!(
                "Failed to restore HNSW state: {}",
                e
            ))));
        }

        // TD-064: restore filterable metadata cache (no-op for upgraded v1 snapshots).
        index.restore_filterable_metadata(state.filterable_metadata);

        info!(
            "HNSW deserialize: restored {} filterable metadata entries",
            metadata_count
        );

        info!(
            "HNSW deserialize_internal complete: {} vectors restored",
            vector_count
        );

        Ok(index)
    }
}

// NOTE (TD-087 Slice B): IVF no longer implements the sync `SerializableIndex`
// trait — its stores are async, so serialization goes through the async
// `IndexSerializer::serialize_ivf`/`deserialize_ivf` + `UnifiedIvfIndex::
// export_state`/`restore_state` instead. The previous trait impl was a stub
// (empty bytes / empty index) and had no callers.

/// Manager for delta updates
pub struct DeltaManager {
    /// Base checkpoint
    base_checkpoint: Option<IndexCheckpoint>,

    /// Accumulated deltas since checkpoint
    deltas: Vec<IndexDelta>,

    /// Maximum deltas before forcing new checkpoint
    max_deltas_before_checkpoint: usize,
}

impl DeltaManager {
    /// Create a new delta manager with the given maximum delta count before a checkpoint is forced
    pub fn new(max_deltas: usize) -> Self {
        Self {
            base_checkpoint: None,
            deltas: Vec::new(),
            max_deltas_before_checkpoint: max_deltas,
        }
    }

    /// Add a delta operation
    pub fn add_delta(&mut self, operation: DeltaOperation) -> Option<IndexCheckpoint> {
        if let Some(ref checkpoint) = self.base_checkpoint {
            let delta = IndexDelta {
                base_checkpoint_id: checkpoint.checkpoint_id.clone(),
                delta_id: format!("delta_{}", unix_now_millis()),
                timestamp: unix_now_secs(),
                operations: vec![operation],
            };

            self.deltas.push(delta);

            // Check if we need a new checkpoint
            if self.deltas.len() >= self.max_deltas_before_checkpoint {
                debug!("Delta threshold reached, signaling for new checkpoint");
                // Return signal that new checkpoint is needed
                return Some(checkpoint.clone());
            }
        }

        None
    }

    /// Set new base checkpoint
    pub fn set_checkpoint(&mut self, checkpoint: IndexCheckpoint) {
        info!("Setting new base checkpoint: {}", checkpoint.checkpoint_id);
        self.base_checkpoint = Some(checkpoint);
        self.deltas.clear();
    }

    /// Get current deltas
    pub fn get_deltas(&self) -> &[IndexDelta] {
        &self.deltas
    }

    /// Apply all deltas to get current state
    pub fn reconstruct_current_state(&self) -> Result<Option<Vec<u8>>> {
        if let Some(ref checkpoint) = self.base_checkpoint {
            let mut current_data = checkpoint.index_data.clone();

            for delta in &self.deltas {
                current_data = IndexSerializer::apply_delta(checkpoint, delta)?;
            }

            Ok(Some(current_data))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::index::axis::storage::serialization::AxisSerializedIndexMetadata;
    use crate::index::axis::*;
    use serde::{Deserialize, Serialize};

    /// Index type enum for testing
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[allow(dead_code)]
    enum Index {
        Hnsw,
        Ivf,
        Flat,
    }

    #[test]
    fn test_metadata_serialization() {
        let metadata = AxisSerializedIndexMetadata {
            index_type: crate::index::axis::storage::serialization::Index::Hnsw,
            collection_id: "test_collection".to_string(),
            num_vectors: 1000,
            dimension: 128,
            timestamp: 1234567890,
            checksum: 0xDEADBEEF,
            is_delta: false,
            base_checkpoint_id: None,
            custom_metadata: None,
        };

        let serialized = bincode::serialize(&metadata).unwrap();
        let deserialized: AxisSerializedIndexMetadata = bincode::deserialize(&serialized).unwrap();

        // The format_strategy arms (Avro/Compression) folded into this enum —
        // keep their renderings stable across the merge.
        assert_eq!(
            crate::index::axis::storage::serialization::SerializationError::Avro(
                "schema".to_string()
            )
            .to_string(),
            "Avro error: schema"
        );
        assert_eq!(
            crate::index::axis::storage::serialization::SerializationError::Compression(
                "zstd".to_string()
            )
            .to_string(),
            "Compression error: zstd"
        );

        assert_eq!(metadata.index_type, deserialized.index_type);
        assert_eq!(metadata.collection_id, deserialized.collection_id);
        assert_eq!(metadata.num_vectors, deserialized.num_vectors);
        assert_eq!(metadata.checksum, deserialized.checksum);
    }

    #[tokio::test]
    async fn decode_ivf_state_tolerates_v1_and_v2() {
        // ADR-023 T-A: the v2 decoder reads v2 payloads (with the COLD binary
        // tier) and falls back to legacy v1 payloads (without it).
        let _ = proximadb_hardware::hardware_capabilities();
        let config = UnifiedIvfConfig {
            dimension: 4,
            n_clusters: 2,
            min_train_size: 2,
            use_binary: true,
            ..Default::default()
        };
        let mut index = UnifiedIvfIndex::new("c_v1".to_string(), config).unwrap();
        let data = [
            vec![1.0f32, -1.0, 1.0, -1.0],
            vec![-1.0, 1.0, -1.0, 1.0],
            vec![1.0, 1.0, -1.0, -1.0],
            vec![-1.0, -1.0, 1.0, 1.0],
        ];
        index.train(data.to_vec()).await.unwrap();
        for (i, v) in data.iter().enumerate() {
            index
                .add_vector(format!("v{i}"), v.clone(), None)
                .await
                .unwrap();
        }
        let v2 = index.export_state().await.unwrap();
        assert!(!v2.binary_tier.is_empty(), "v2 state carries a COLD tier");

        // A v2 body decodes as v2 — the COLD tier survives.
        let v2_bytes = bincode::serialize(&v2).unwrap();
        let decoded_v2 = super::decode_ivf_state(&v2_bytes).unwrap();
        assert_eq!(decoded_v2.binary_tier.len(), v2.binary_tier.len());

        // A v1 body (struct without `binary_tier`) decodes via fallback to an
        // empty COLD tier — pre-ADR-023 indexes still load.
        let v1 = SerializableIvfStateV1 {
            version: 1,
            vector_count: v2.vector_count,
            config: v2.config.clone(),
            centroids: v2.centroids.clone(),
            vectors: v2.vectors.clone(),
        };
        let v1_bytes = bincode::serialize(&v1).unwrap();
        let decoded_v1 = super::decode_ivf_state(&v1_bytes).unwrap();
        assert!(
            decoded_v1.binary_tier.is_empty(),
            "v1 fallback yields an empty COLD tier"
        );
        assert_eq!(decoded_v1.vectors.len(), v2.vectors.len());
    }

    #[test]
    fn test_delta_manager() {
        let mut manager = DeltaManager::new(5);

        let checkpoint = IndexCheckpoint {
            checkpoint_id: "test_checkpoint".to_string(),
            timestamp: 1234567890,
            index_data: vec![1, 2, 3, 4, 5],
            metadata: AxisSerializedIndexMetadata {
                index_type: crate::index::axis::storage::serialization::Index::Hnsw,
                collection_id: "test".to_string(),
                num_vectors: 100,
                dimension: 128,
                timestamp: 1234567890,
                checksum: 12345,
                is_delta: false,
                base_checkpoint_id: None,
                custom_metadata: None,
            },
        };

        manager.set_checkpoint(checkpoint);

        // Add some deltas
        for i in 0..3 {
            let op = DeltaOperation::AddVectors {
                vectors: vec![(format!("vec_{}", i), vec![0.1; 128])],
            };
            assert!(manager.add_delta(op).is_none());
        }

        assert_eq!(manager.get_deltas().len(), 3);
    }
}
