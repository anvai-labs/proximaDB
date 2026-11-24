//! Metadata serialization traits for engine-specific implementations
//!
//! This module defines the contract that storage engines must implement
//! to provide their own metadata serialization logic to the unified filesystem.
//!
//! Each engine owns its metadata format and provides a serializer implementation.

use anyhow::Result;
use bytes::Bytes;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::fmt::Debug;

/// Trait for engine-specific metadata serialization
///
/// Each storage engine implements this trait to handle its own metadata format.
/// This keeps the filesystem layer decoupled from engine-specific details.
pub trait EngineMetadataSerializer: Send + Sync + Debug {
    /// Serialize metadata to bytes
    fn serialize(&self, metadata: &dyn Any) -> Result<Bytes>;

    /// Deserialize metadata from bytes
    fn deserialize(&self, bytes: &[u8]) -> Result<Box<dyn Any + Send + Sync>>;

    /// Get the engine type this serializer handles
    fn engine_type(&self) -> &str;

    /// Extract cacheable metadata components (e.g., Parquet footer)
    /// Returns None if no special caching is needed
    fn extract_cacheable_component(&self, _data: &[u8], _file_path: &str) -> Option<Bytes> {
        None
    }

    /// Check if a file path should have metadata cached
    fn should_cache_metadata(&self, _file_path: &str) -> bool {
        true
    }
}

pub fn serialize_typed_metadata<T>(metadata: &dyn Any, expected_type: &str) -> Result<Bytes>
where
    T: Serialize + Send + Sync + 'static,
{
    if let Some(typed_metadata) = metadata.downcast_ref::<T>() {
        let bytes = bincode::serialize(typed_metadata)?;
        Ok(Bytes::from(bytes))
    } else {
        anyhow::bail!("Expected {expected_type} type for serializer")
    }
}

pub fn deserialize_typed_metadata<T>(bytes: &[u8]) -> Result<Box<dyn Any + Send + Sync>>
where
    T: DeserializeOwned + Send + Sync + 'static,
{
    let typed_metadata: T = bincode::deserialize(bytes)?;
    Ok(Box::new(typed_metadata))
}

/// Generic metadata that can be cached by the filesystem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheableMetadata {
    /// File path
    pub file_path: String,

    /// File size in bytes
    pub file_size: u64,

    /// Last modified timestamp
    pub last_modified: i64,

    /// Engine-specific serialized metadata (stored as Vec<u8> for serialization)
    pub engine_metadata: Option<Vec<u8>>,

    /// Extracted cacheable component (e.g., Parquet footer, stored as Vec<u8> for serialization)
    pub cached_component: Option<Vec<u8>>,
}

/// Default serializer for unknown/generic engines
#[derive(Debug)]
pub struct GenericMetadataSerializer;

impl EngineMetadataSerializer for GenericMetadataSerializer {
    fn serialize(&self, metadata: &dyn Any) -> Result<Bytes> {
        // For generic engines, we just store basic file info
        if let Some(file_meta) =
            metadata.downcast_ref::<crate::storage::persistence::filesystem::FileMetadata>()
        {
            let cacheable = CacheableMetadata {
                file_path: file_meta.path.clone(),
                file_size: file_meta.size,
                last_modified: file_meta.modified.map_or(0, |dt| dt.timestamp()),
                engine_metadata: None,
                cached_component: None,
            };
            let bytes = bincode::serialize(&cacheable)?;
            Ok(Bytes::from(bytes))
        } else {
            anyhow::bail!("Cannot serialize unknown metadata type")
        }
    }

    fn deserialize(&self, bytes: &[u8]) -> Result<Box<dyn Any + Send + Sync>> {
        let cacheable: CacheableMetadata = bincode::deserialize(bytes)?;
        Ok(Box::new(cacheable))
    }

    fn engine_type(&self) -> &str {
        "generic"
    }
}

