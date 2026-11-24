// Copyright 2025 Vijaykumar Singh
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Shared metadata serializer helpers for storage engines.
//!
//! This module consolidates the duplicated `unified_metadata_serializer.rs` files
//! found across multiple engines (raptor, nova, sst, swift, helix) into
//! a single shared implementation. Engines should migrate to using these helpers
//! and remove their per-engine copies.
//!
//! ## Migration Status (TD-DRY-METADATA)
//!
//! All previously duplicated per-engine `unified_metadata_serializer.rs` files have
//! been migrated to shared implementations under `src/storage/engines/core`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Common metadata header shared across all storage engines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedMetadataHeader {
    /// Engine that produced this metadata
    pub engine_type: String,
    /// Collection this metadata belongs to
    pub collection_id: String,
    /// Schema version for forward compatibility
    pub schema_version: u32,
    /// Timestamp of last flush (epoch millis)
    pub last_flush_timestamp_ms: u64,
    /// Number of records in this segment
    pub record_count: usize,
    /// Arbitrary key-value engine-specific metadata
    pub extensions: HashMap<String, String>,
}

/// Helper to build a default metadata file path for a collection segment.
pub fn metadata_path(base_dir: &Path, collection_id: &str, segment_id: &str) -> std::path::PathBuf {
    base_dir
        .join(collection_id)
        .join("metadata")
        .join(format!("{}_meta.json", segment_id))
}

/// Serialize metadata header to JSON bytes.
pub fn serialize_header(header: &UnifiedMetadataHeader) -> anyhow::Result<Vec<u8>> {
    Ok(serde_json::to_vec_pretty(header)?)
}

/// Deserialize metadata header from JSON bytes.
pub fn deserialize_header(data: &[u8]) -> anyhow::Result<UnifiedMetadataHeader> {
    Ok(serde_json::from_slice(data)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_header() {
        let header = UnifiedMetadataHeader {
            engine_type: "helix".to_string(),
            collection_id: "col_123".to_string(),
            schema_version: 1,
            last_flush_timestamp_ms: 1700000000000,
            record_count: 42,
            extensions: HashMap::new(),
        };
        let bytes = serialize_header(&header).unwrap();
        let restored = deserialize_header(&bytes).unwrap();
        assert_eq!(restored.engine_type, "helix");
        assert_eq!(restored.record_count, 42);
    }

    #[test]
    fn metadata_path_format() {
        let path = metadata_path(Path::new("/data"), "col_abc", "seg_001");
        assert_eq!(
            path.to_str().unwrap(),
            "/data/col_abc/metadata/seg_001_meta.json"
        );
    }
}
