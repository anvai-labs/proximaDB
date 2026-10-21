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

//! SST Engine Record Types (Compatibility Layer)
//!
//! ## Status: Deprecated for Production Use
//!
//! Production code uses `ProximaRecord` and ProximaBlocks directly. This module
//! is kept only for test compatibility. Legacy block-type history is recoverable
//! from git; the uncompiled `blocks_archive.rs` file was removed in PCX-010.
//!
//! ## Migration Note (TD-001)
//! Block types (ProximaDataBlock, ProximaBlockMetadata, etc.) have been migrated to:
//! - `storage::engines::core::formats::proximablocks`
//!
//! ## Future Work
//! Migrate tests in `tests/sst/` to use canonical storage blocks directly, then
//! archive this module.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::core::search::sql_value_filter::proxima_tree_to_json_map;
use proximadb_records::ProximaRecord;

/// SST record representation with LSM-tree metadata
///
/// ## Deprecation Notice
/// Production code has been optimized to use `ProximaRecord`/ProximaBlocks
/// directly. This type is only used in tests and will be archived once tests
/// are updated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SstRecord {
    /// Unique identifier
    pub id: String,
    /// Vector data
    pub vector: Option<Vec<f32>>,
    /// Metadata as JSON value
    pub metadata: Option<serde_json::Value>,
    /// Sequence number for ordering
    pub sequence_number: u64,
    /// LSM tree level
    pub level: u8,
    /// Deletion marker
    pub is_tombstone: bool,
    /// Timestamp
    pub timestamp: u64,
}

impl SstRecord {
    /// Create from the canonical record envelope.
    pub fn from_proxima_record(record: ProximaRecord, sequence_number: u64, level: u8) -> Self {
        let metadata_json = if !record.props.is_empty() {
            Some(serde_json::Value::Object(
                proxima_tree_to_json_map(&record.props)
                    .into_iter()
                    .collect(),
            ))
        } else {
            None
        };

        let vector = record
            .embeddings
            .first()
            .map(|embedding| embedding.values.to_fp32_owned());

        SstRecord {
            id: record.oid,
            vector,
            metadata: metadata_json,
            sequence_number,
            level,
            is_tombstone: false,
            timestamp: (record.created_at_ns / 1_000_000).max(0) as u64,
        }
    }

    /// Create a tombstone record for deletion
    pub fn tombstone(id: String, sequence_number: u64, level: u8) -> Self {
        SstRecord {
            id,
            vector: None,
            metadata: None,
            sequence_number,
            level,
            is_tombstone: true,
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
        }
    }

    /// Convert to OptimizedSearchRecord
    pub fn to_optimized_search_result(
        &self,
        score: f32,
    ) -> crate::core::search::results::OptimizedSearchRecord {
        use proximadb_data_model::ProximaValue;
        let metadata = self
            .metadata
            .as_ref()
            .map(|json_value| {
                if let serde_json::Value::Object(map) = json_value {
                    map.iter()
                        .map(|(k, v)| {
                            let pv = match v {
                                serde_json::Value::String(s) => ProximaValue::String(s.clone()),
                                serde_json::Value::Number(n) => {
                                    ProximaValue::Float64(n.as_f64().unwrap_or(0.0))
                                }
                                serde_json::Value::Bool(b) => ProximaValue::Boolean(*b),
                                serde_json::Value::Null => ProximaValue::Null,
                                other => ProximaValue::Json(serde_json::json!(other)),
                            };
                            (k.clone(), pv)
                        })
                        .collect()
                } else {
                    std::collections::HashMap::new()
                }
            })
            .unwrap_or_default();
        crate::core::search::results::OptimizedSearchRecord {
            id: self.id.clone(),
            score,
            vector: self.vector.as_ref().map(|v| Arc::new(v.clone())),
            metadata,
            timestamp: Some(self.timestamp as i64),
            version: Some(self.sequence_number as u32),
            ..Default::default()
        }
    }

    /// Serialize record to bytes
    pub fn serialize(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(self)?)
    }

    /// Deserialize record from bytes
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        Ok(serde_json::from_slice(data)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_data_model::ProximaValue;
    use proximadb_records::{EmbeddingCell, ProximaTreeNode};

    fn canonical_record() -> ProximaRecord {
        ProximaRecord {
            oid: "row-1".to_string(),
            created_at_ns: 12_345_000_000,
            embeddings: vec![EmbeddingCell::new_fp32(
                "model-a",
                "text",
                3,
                vec![1.0, 2.0, 3.0],
            )],
            props: [
                (
                    "category".to_string(),
                    ProximaTreeNode::Value(ProximaValue::String("books".to_string())),
                ),
                (
                    "price".to_string(),
                    ProximaTreeNode::Value(ProximaValue::Float64(12.5)),
                ),
                (
                    "active".to_string(),
                    ProximaTreeNode::Value(ProximaValue::Boolean(true)),
                ),
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn from_proxima_record_projects_identity_vector_metadata_and_time() {
        let record = SstRecord::from_proxima_record(canonical_record(), 42, 3);

        assert_eq!(record.id, "row-1");
        assert_eq!(record.vector, Some(vec![1.0, 2.0, 3.0]));
        assert_eq!(record.sequence_number, 42);
        assert_eq!(record.level, 3);
        assert!(!record.is_tombstone);
        assert_eq!(record.timestamp, 12_345);

        let metadata = record
            .metadata
            .as_ref()
            .and_then(|v| v.as_object())
            .unwrap();
        assert_eq!(metadata["category"], serde_json::json!("books"));
        assert_eq!(metadata["price"], serde_json::json!(12.5));
        assert_eq!(metadata["active"], serde_json::json!(true));
    }

    #[test]
    fn from_proxima_record_handles_missing_embedding_and_negative_time() {
        let record = ProximaRecord {
            oid: "row-empty".to_string(),
            created_at_ns: -99,
            ..Default::default()
        };

        let sst = SstRecord::from_proxima_record(record, 7, 0);
        assert_eq!(sst.id, "row-empty");
        assert_eq!(sst.vector, None);
        assert_eq!(sst.metadata, None);
        assert_eq!(sst.timestamp, 0);
        assert_eq!(sst.sequence_number, 7);
    }

    #[test]
    fn tombstone_and_search_result_projection_are_stable() {
        let tombstone = SstRecord::tombstone("delete-me".to_string(), 99, 2);
        assert_eq!(tombstone.id, "delete-me");
        assert!(tombstone.is_tombstone);
        assert_eq!(tombstone.vector, None);
        assert_eq!(tombstone.sequence_number, 99);
        assert_eq!(tombstone.level, 2);

        let record = SstRecord::from_proxima_record(canonical_record(), 42, 3);
        let result = record.to_optimized_search_result(0.75);
        assert_eq!(result.id, "row-1");
        assert_eq!(result.score, 0.75);
        assert_eq!(result.timestamp, Some(12_345));
        assert_eq!(result.version, Some(42));
        assert_eq!(
            result.vector.as_ref().map(|vector| vector.as_slice()),
            Some(&[1.0, 2.0, 3.0][..])
        );
        assert_eq!(
            result.metadata.get("category"),
            Some(&ProximaValue::String("books".to_string()))
        );
        assert_eq!(
            result.metadata.get("price"),
            Some(&ProximaValue::Float64(12.5))
        );
        assert_eq!(
            result.metadata.get("active"),
            Some(&ProximaValue::Boolean(true))
        );
    }

    #[test]
    fn serialize_round_trip_preserves_compatibility_shape() {
        let record = SstRecord::from_proxima_record(canonical_record(), 42, 3);
        let bytes = record.serialize().unwrap();
        let decoded = SstRecord::deserialize(&bytes).unwrap();

        assert_eq!(decoded.id, record.id);
        assert_eq!(decoded.vector, record.vector);
        assert_eq!(decoded.metadata, record.metadata);
        assert_eq!(decoded.sequence_number, record.sequence_number);
        assert_eq!(decoded.level, record.level);
        assert_eq!(decoded.timestamp, record.timestamp);
    }
}
