//! Search execution utilities for vector operations.
//!
//! Provides helper types and functions for executing vector searches
//! across different storage engines with progressive quantization support.

use std::collections::HashMap;

use proximadb_data_model::ProximaValue;

/// Search result with similarity scores.
#[derive(Debug, Clone)]
pub struct VectorSearchExecutorResult {
    /// Vector ID
    pub id: String,
    /// Vector data
    pub vector: Option<Vec<f32>>,
    /// Associated metadata
    pub metadata: HashMap<String, ProximaValue>,
    /// Similarity score
    pub score: f32,
}

/// Convert proto search results to ProximaRecord format.
pub fn proto_results_to_vector_records(
    search_results: Vec<crate::proto::proximadb_v1::SearchResult>,
) -> Vec<proximadb_records::ProximaRecord> {
    let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
    let mut records = Vec::new();
    for search_result in search_results {
        for result in search_result.results {
            let dim = result.vector.len() as u32;
            records.push(proximadb_records::ProximaRecord {
                oid: result.id,
                created_at_ns: now_ns,
                updated_at_ns: now_ns,
                origin: Some("search_result".to_string()),
                embeddings: if !result.vector.is_empty() {
                    vec![proximadb_records::EmbeddingCell {
                        model_id: "default".to_string(),
                        modality: "vector".to_string(),
                        values: proximadb_records::EmbeddingValues::Fp32(result.vector),
                        dim,
                        ..Default::default()
                    }]
                } else {
                    vec![]
                },
                props: result
                    .metadata
                    .into_iter()
                    .map(|(k, v)| {
                        use proximadb_records::ProximaTreeNode;
                        // Round 16: delegate the whole lowering to the
                        // canonical converter — 8 hand-copied arms diverged
                        // (top-level legacy magic bytes stayed Binary while
                        // nested ones were sniffed; nesting-dependent types).
                        let node = ProximaTreeNode::Value(
                            proximadb_records::conversions::sql_value_to_proxima(&v),
                        );
                        (k, node)
                    })
                    .collect(),
                ..Default::default()
            });
        }
    }
    records
}

/// Calculate candidate counts for progressive search stages.
///
/// Uses the formula: k_stage = k · Π(1/r_i) for all subsequent stages
/// where r_i is the recall target for stage i.
///
/// # Arguments
///
/// * `k` - Final top-k result count
/// * `recalls` - Recall targets for each progressive stage
///
/// # Returns
///
/// Vector of candidate counts for each stage
pub fn calculate_progressive_candidates(k: usize, recalls: &[f32]) -> Vec<usize> {
    let mut candidates = Vec::with_capacity(recalls.len());
    let mut recall_product = 1.0;

    for &recall in recalls {
        recall_product *= recall;
        let stage_k = ((k as f32) / recall_product).ceil() as usize;
        candidates.push(stage_k.max(k));
    }

    candidates
}

/// Build default progressive stages configuration.
pub fn default_progressive_stages() -> Vec<String> {
    vec!["binary".into(), "int8".into(), "pq".into(), "full".into()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_progressive_candidates() {
        let k = 10;
        let recalls = vec![0.5, 0.8]; // Binary→INT8→Full

        let candidates = calculate_progressive_candidates(k, &recalls);

        // First stage: 10 / 0.5 = 20
        assert_eq!(candidates[0], 20);
        // Second stage: 10 / (0.5 * 0.8) = 25
        assert_eq!(candidates[1], 25);
    }

    #[test]
    fn test_default_progressive_stages() {
        let stages = default_progressive_stages();
        assert_eq!(stages.len(), 4);
        assert_eq!(stages[0], "binary");
        assert_eq!(stages[3], "full");
    }

    #[test]
    fn test_proto_results_to_vector_records() {
        use crate::proto::proximadb_v1::{
            SearchResult as ProtoSearchResult, SearchVectorRecord, SqlArray, SqlObject, SqlValue,
            sql_value,
        };

        let proto_result = ProtoSearchResult {
            collection_id: Some("test".to_string()),
            total_found: 1,
            results: vec![SearchVectorRecord {
                id: "vec1".to_string(),
                score: 0.95,
                vector: vec![1.0, 2.0, 3.0],
                metadata: HashMap::from([
                    (
                        "tags".to_string(),
                        SqlValue {
                            value: Some(sql_value::Value::ArrayValue(SqlArray {
                                values: vec![SqlValue {
                                    value: Some(sql_value::Value::StringValue("red".to_string())),
                                }],
                            })),
                        },
                    ),
                    (
                        "profile".to_string(),
                        SqlValue {
                            value: Some(sql_value::Value::ObjectValue(SqlObject {
                                fields: HashMap::from([(
                                    "deleted".to_string(),
                                    SqlValue { value: None },
                                )]),
                            })),
                        },
                    ),
                    ("null_value".to_string(), SqlValue { value: None }),
                ]),
                version: None,
                similarity: None,
                timestamp: None,
                source: None,
                expanded_context: Vec::new(),
                semantic_similarity: None,
                quantization_info: None,
                engine_stats: HashMap::new(),
                index_path: None,
            }],
        };

        let records = proto_results_to_vector_records(vec![proto_result]);

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].oid, "vec1");
        assert_eq!(records[0].origin, Some("search_result".to_string()));
        assert_eq!(
            records[0].props.get("tags"),
            Some(&proximadb_records::ProximaTreeNode::Value(
                ProximaValue::Array(vec![ProximaValue::String("red".to_string())])
            ))
        );
        assert_eq!(
            records[0].props.get("profile"),
            Some(&proximadb_records::ProximaTreeNode::Value(
                ProximaValue::Map(HashMap::from([("deleted".to_string(), ProximaValue::Null)]))
            ))
        );
        assert_eq!(
            records[0].props.get("null_value"),
            Some(&proximadb_records::ProximaTreeNode::Value(
                ProximaValue::Null
            ))
        );
    }
}
