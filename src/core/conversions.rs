/*
 * Copyright 2025 ProximaDB
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

//! Unified conversion utilities for ProximaDB
//!
//! This module consolidates all conversion logic between:
//! - REST and Proto types
//! - Native types and Proto messages  
//! - JSON and Proto formats
//!
//! Eliminates duplication across REST, gRPC, and internal handlers.

use anyhow::Result;

use crate::core::search::results::OptimizedSearchRecord;
use crate::proto::proximadb_v1::{
    CollectionConfig, CollectionOperation, DistanceMetric, IndexingAlgorithm, MetadataItem,
    SearchQuery, SearchResult as ProtoSearchResult, SearchVectorRecord, SqlValue, StorageEngine,
    VectorOperation, VectorSearchRequest,
};
use std::collections::HashMap;

// ============================================================================
// REST to Proto Conversions
// ============================================================================

/// Convert REST collection operation string to proto enum
pub fn parse_collection_operation(op: &str) -> Result<CollectionOperation, String> {
    match op.to_lowercase().as_str() {
        "create" => Ok(CollectionOperation::CollectionCreate),
        "get" => Ok(CollectionOperation::CollectionGet),
        "list" => Ok(CollectionOperation::CollectionList),
        "update" => Ok(CollectionOperation::CollectionUpdate),
        "delete" => Ok(CollectionOperation::CollectionDelete),
        _ => Err(format!("Invalid operation: {}", op)),
    }
}

/// Convert REST vector operation string to proto enum
pub fn parse_vector_operation(op: &str) -> Result<VectorOperation, String> {
    match op.to_lowercase().as_str() {
        "insert" | "upsert" | "update" | "delete" => Ok(VectorOperation::VectorBatch),
        "search" => Ok(VectorOperation::VectorSearch),
        "get" => Ok(VectorOperation::VectorGet),
        _ => Err(format!("Invalid operation: {}", op)),
    }
}

/// Convert distance metric string to proto enum
pub fn parse_distance_metric(metric: &str) -> Result<DistanceMetric> {
    match metric.to_lowercase().as_str() {
        "cosine" => Ok(DistanceMetric::Cosine),
        "euclidean" | "l2" => Ok(DistanceMetric::Euclidean),
        "manhattan" | "l1" => Ok(DistanceMetric::Manhattan),
        "dot" | "dot_product" | "inner_product" => Ok(DistanceMetric::DotProduct),
        _ => Err(anyhow::anyhow!("Invalid distance metric: {}", metric)),
    }
}

/// Convert storage engine string to proto enum
pub fn parse_storage_engine(engine: &str) -> Result<StorageEngine> {
    match engine.to_lowercase().as_str() {
        "sst" => Ok(StorageEngine::Sst),
        "helix" => Ok(StorageEngine::Helix),
        "viper" => Ok(StorageEngine::Viper),
        "nova" => Ok(StorageEngine::Nova),
        "swift" => Ok(StorageEngine::Swift),
        "raptor" => Ok(StorageEngine::Raptor),
        "mmap" => Ok(StorageEngine::Mmap),
        "hybrid" => Ok(StorageEngine::Hybrid),
        "tst" => Ok(StorageEngine::Tst),
        _ => Err(anyhow::anyhow!("Invalid storage engine: {}", engine)),
    }
}

/// Convert indexing algorithm string to proto enum
pub fn parse_indexing_algorithm(algo: &str) -> Result<IndexingAlgorithm> {
    match algo.to_lowercase().as_str() {
        "hnsw" => Ok(IndexingAlgorithm::Hnsw),
        "ivf" => Ok(IndexingAlgorithm::Ivf),
        "flat" => Ok(IndexingAlgorithm::Flat),
        "pq" => Ok(IndexingAlgorithm::Pq),
        "annoy" => Ok(IndexingAlgorithm::Annoy),
        "lsh" => Ok(IndexingAlgorithm::Lsh),
        _ => Err(anyhow::anyhow!("Invalid indexing algorithm: {}", algo)),
    }
}

// ============================================================================
// Proto to String Conversions
// ============================================================================

/// Convert proto distance metric to string
pub fn distance_metric_to_string(metric: i32) -> &'static str {
    match DistanceMetric::try_from(metric) {
        Ok(DistanceMetric::Cosine) => "cosine",
        Ok(DistanceMetric::Euclidean) => "euclidean",
        Ok(DistanceMetric::Manhattan) => "manhattan",
        Ok(DistanceMetric::DotProduct) => "dot_product",
        _ => "unknown",
    }
}

/// Convert proto storage engine to string
pub fn storage_engine_to_string(engine: i32) -> &'static str {
    match StorageEngine::try_from(engine) {
        Ok(StorageEngine::Sst) => "sst",
        Ok(StorageEngine::Helix) => "helix",
        Ok(StorageEngine::Viper) => "viper",
        Ok(StorageEngine::Nova) => "nova",
        Ok(StorageEngine::Swift) => "swift",
        Ok(StorageEngine::Raptor) => "raptor",
        Ok(StorageEngine::Mmap) => "mmap",
        Ok(StorageEngine::Hybrid) => "hybrid",
        Ok(StorageEngine::Tst) => "tst",
        _ => "unknown",
    }
}

/// Convert proto collection operation to string
pub fn collection_operation_to_string(op: i32) -> &'static str {
    match CollectionOperation::try_from(op) {
        Ok(CollectionOperation::CollectionCreate) => "create",
        Ok(CollectionOperation::CollectionGet) => "get",
        Ok(CollectionOperation::CollectionList) => "list",
        Ok(CollectionOperation::CollectionUpdate) => "update",
        Ok(CollectionOperation::CollectionDelete) => "delete",
        _ => "unknown",
    }
}

/// Convert proto vector operation to string
pub fn vector_operation_to_string(op: i32) -> &'static str {
    match VectorOperation::try_from(op) {
        Ok(VectorOperation::VectorBatch) => "batch",
        Ok(VectorOperation::VectorSearch) => "search",
        Ok(VectorOperation::VectorGet) => "get",
        _ => "unknown",
    }
}

/// Convert serde_json map to SqlValue HashMap (for SearchVectorRecord)
pub fn convert_serde_json_to_sql_value_map(
    metadata: HashMap<String, serde_json::Value>,
) -> HashMap<String, SqlValue> {
    metadata
        .into_iter()
        .map(|(k, v)| {
            let sql_value = match v {
                serde_json::Value::String(s) => SqlValue {
                    value: Some(crate::proto::proximadb_v1::sql_value::Value::StringValue(s)),
                },
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        SqlValue {
                            value: Some(crate::proto::proximadb_v1::sql_value::Value::Int64Value(
                                i,
                            )),
                        }
                    } else if let Some(f) = n.as_f64() {
                        SqlValue {
                            value: Some(crate::proto::proximadb_v1::sql_value::Value::NumberValue(
                                f,
                            )),
                        }
                    } else {
                        SqlValue {
                            value: Some(crate::proto::proximadb_v1::sql_value::Value::StringValue(
                                n.to_string(),
                            )),
                        }
                    }
                }
                serde_json::Value::Bool(b) => SqlValue {
                    value: Some(crate::proto::proximadb_v1::sql_value::Value::BoolValue(b)),
                },
                serde_json::Value::Array(arr) => SqlValue {
                    value: Some(crate::proto::proximadb_v1::sql_value::Value::StringValue(
                        serde_json::to_string(&arr).unwrap_or_else(|_| "[]".to_string()),
                    )),
                },
                serde_json::Value::Object(obj) => SqlValue {
                    value: Some(crate::proto::proximadb_v1::sql_value::Value::StringValue(
                        serde_json::to_string(&obj).unwrap_or_else(|_| "{}".to_string()),
                    )),
                },
                serde_json::Value::Null => SqlValue {
                    value: Some(crate::proto::proximadb_v1::sql_value::Value::StringValue(
                        "null".to_string(),
                    )),
                },
            };
            (k, sql_value)
        })
        .collect()
}

// ============================================================================
// Native to Proto Conversions
// ============================================================================

/// Convert optimized native search results into the proto search-result envelope.
pub fn optimized_results_to_proto_search_result(
    results: Vec<crate::core::search::results::OptimizedSearchRecord>,
) -> ProtoSearchResult {
    let total_found = results.len() as i64;
    ProtoSearchResult {
        results: results.into_iter().map(SearchVectorRecord::from).collect(),
        total_found,
        collection_id: None,
    }
}

/// Convert a vector of native search results directly to proto SearchVectorRecord
pub fn convert_search_results(
    native_results: Vec<OptimizedSearchRecord>,
    include_vectors: bool,
    include_metadata: bool,
) -> Vec<SearchVectorRecord> {
    native_results
        .into_iter()
        .map(|mut result| {
            // Apply include flags
            if !include_vectors {
                result.vector = None;
            }
            if !include_metadata {
                result.metadata.clear();
            }
            SearchVectorRecord::from(result)
        })
        .collect()
}

// ============================================================================
// JSON to Proto Conversions
// ============================================================================

/// Convert JSON metadata to proto MetadataItem vector
pub fn convert_metadata_to_proto(
    metadata: serde_json::Map<String, serde_json::Value>,
) -> Vec<MetadataItem> {
    metadata
        .into_iter()
        .map(|(key, value)| {
            let metadata_value = match value {
                serde_json::Value::String(s) => {
                    Some(crate::proto::proximadb_v1::metadata_item::Value::StringValue(s))
                }
                serde_json::Value::Number(n) => {
                    if let Some(f) = n.as_f64() {
                        Some(crate::proto::proximadb_v1::metadata_item::Value::NumberValue(f))
                    } else {
                        Some(
                            crate::proto::proximadb_v1::metadata_item::Value::StringValue(
                                n.to_string(),
                            ),
                        )
                    }
                }
                serde_json::Value::Bool(b) => Some(
                    crate::proto::proximadb_v1::metadata_item::Value::BoolValue(b),
                ),
                _ => Some(
                    crate::proto::proximadb_v1::metadata_item::Value::StringValue(
                        value.to_string(),
                    ),
                ),
            };
            MetadataItem {
                key,
                value: metadata_value,
            }
        })
        .collect()
}

/// Convert proto MetadataItem vector to JSON object
pub fn convert_metadata_from_proto(
    items: Vec<MetadataItem>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut map = serde_json::Map::new();
    for item in items {
        if let Some(value) = item.value {
            let json_value = match value {
                crate::proto::proximadb_v1::metadata_item::Value::StringValue(s) => {
                    serde_json::Value::String(s)
                }
                crate::proto::proximadb_v1::metadata_item::Value::NumberValue(n) => {
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(n).unwrap_or(serde_json::Number::from(0)),
                    )
                }
                crate::proto::proximadb_v1::metadata_item::Value::BoolValue(b) => {
                    serde_json::Value::Bool(b)
                } // Note: ListValue and MapValue were removed from proto
                  // Arrays and objects can be stored as JSON strings if needed
            };
            map.insert(item.key, json_value);
        }
    }
    map
}

// ============================================================================
// Collection Config Conversions
// ============================================================================

/// Convert a ProximaValue metadata map to JSON values.
pub fn proxima_values_to_json_map(
    items: HashMap<String, proximadb_data_model::ProximaValue>,
) -> HashMap<String, serde_json::Value> {
    items
        .into_iter()
        .map(|(k, v)| {
            let json = proximadb_embedded_common::proxima_value_to_json(v);
            (k, json)
        })
        .collect()
}

/// Convert a JSON map to v1 SqlValue map
pub fn json_map_to_sql_values(
    items: HashMap<String, serde_json::Value>,
) -> HashMap<String, crate::proto::proximadb_v1::SqlValue> {
    let mut out = HashMap::new();
    for (k, v) in items {
        let sql = match v {
            serde_json::Value::String(s) => crate::proto::proximadb_v1::SqlValue {
                value: Some(crate::proto::proximadb_v1::sql_value::Value::StringValue(s)),
            },
            serde_json::Value::Number(n) => {
                let f = n.as_f64().unwrap_or(0.0);
                crate::proto::proximadb_v1::SqlValue {
                    value: Some(crate::proto::proximadb_v1::sql_value::Value::NumberValue(f)),
                }
            }
            serde_json::Value::Bool(b) => crate::proto::proximadb_v1::SqlValue {
                value: Some(crate::proto::proximadb_v1::sql_value::Value::BoolValue(b)),
            },
            other => crate::proto::proximadb_v1::SqlValue {
                value: Some(crate::proto::proximadb_v1::sql_value::Value::StringValue(
                    other.to_string(),
                )),
            },
        };
        out.insert(k, sql);
    }
    out
}

/// Build a CollectionConfig from JSON parameters
pub fn build_collection_config(
    name: String,
    dimension: usize,
    distance_metric: Option<String>,
    storage_engine: Option<String>,
    indexing_algorithm: Option<String>,
    _metadata: Option<serde_json::Map<String, serde_json::Value>>,
) -> Result<CollectionConfig> {
    let mut config = CollectionConfig {
        name: name.clone(),
        embedding_config: None,
        dimension: dimension as u32,
        distance_metric: distance_metric
            .map(|m| parse_distance_metric(&m))
            .transpose()?
            .map(|m| m as i32),
        storage_engine: storage_engine
            .map(|e| parse_storage_engine(&e))
            .transpose()?
            .map(|e| e as i32),
        storage_config: None,
        index_configs: indexing_algorithm
            .map(|a| parse_indexing_algorithm(&a))
            .transpose()?
            .map(|algo| {
                vec![crate::proto::proximadb_v1::IndexConfig {
                    algorithm: algo as i32,
                    ..Default::default()
                }]
            })
            .unwrap_or_default(),
        filterable_columns: vec![],
        quantization: None,
        primary_index: None,
        auto_index_selection: None,
        embedding_models: vec![],
        description: None,
        tags: vec![],
        owner: None,
        // ProximaRecord schema configuration (NEW)
        record_schema: None,
        enable_proxima_record: None,
        text_columns: vec![],
        text_storage_configs: vec![],
        enable_dual_use_embeddings: None,
        canonical_embedding_precision: None,
        permitted_principals: vec![],
        // Generic JSON builder carries no routing policy (REST v2 build is the
        // typed surface); default to auto (None).
        index_policy: None,
        pax_vector_quant: None,
    };

    // Apply smart defaults from proto comments
    crate::proto::defaults::apply_collection_config_defaults(&mut config);

    Ok(config)
}

// ============================================================================
// Search Request Conversions
// ============================================================================

/// Convert JSON search request to proto VectorSearchRequest
pub fn build_vector_search_request(
    collection_id: String,
    vector: Vec<f32>,
    top_k: u32,
    metadata_filter: Option<serde_json::Map<String, serde_json::Value>>,
    include_vector: bool,
    include_metadata: bool,
) -> VectorSearchRequest {
    let query = SearchQuery {
        vector,
        filters: std::collections::HashMap::new(), // Deferred: Convert metadata_filter to filters
        advanced_filter: metadata_filter.map(|f| {
            // Convert JSON map to MetadataFilter with filter conditions
            let mut conditions = Vec::new();

            // Convert each key-value pair to a filter clause
            for (key, value) in f {
                let clause = match value {
                    serde_json::Value::String(s) => crate::proto::proximadb_v1::FilterClause {
                        field: key,
                        op: crate::proto::proximadb_v1::ComparisonOp::Eq as i32,
                        value: Some(
                            crate::proto::proximadb_v1::filter_clause::Value::StringValue(s),
                        ),
                    },
                    serde_json::Value::Number(n) => crate::proto::proximadb_v1::FilterClause {
                        field: key,
                        op: crate::proto::proximadb_v1::ComparisonOp::Eq as i32,
                        value: Some(
                            crate::proto::proximadb_v1::filter_clause::Value::DoubleValue(
                                n.as_f64().unwrap_or(0.0),
                            ),
                        ),
                    },
                    serde_json::Value::Bool(b) => crate::proto::proximadb_v1::FilterClause {
                        field: key,
                        op: crate::proto::proximadb_v1::ComparisonOp::Eq as i32,
                        value: Some(crate::proto::proximadb_v1::filter_clause::Value::BoolValue(
                            b,
                        )),
                    },
                    _ => continue, // Skip complex types for now
                };
                conditions.push(clause);
            }

            crate::proto::proximadb_v1::MetadataFilter {
                clauses: conditions, // Renamed from conditions to clauses
                op: crate::proto::proximadb_v1::LogicalOp::And as i32, // Use LogicalOp enum
            }
        }),
    };

    VectorSearchRequest {
        collection_id,
        queries: vec![query],
        top_k,
        distance_metric_override: None,
        search_params: None,
        include_fields: Some(crate::proto::proximadb_v1::IncludeFields {
            vector: include_vector,
            metadata: include_metadata,
            score: true,
            rank: false,
            source: false,
            source_options: std::collections::HashMap::new(),
        }),
        search_optimization: Some(crate::proto::proximadb_v1::SearchOptimization {
            top_k: Some(top_k),
            accuracy_threshold: None,
            filters: std::collections::HashMap::new(),
        }),
    }
}

// ============================================================================
// v1 Metadata Conversions (MetadataItem <-> SqlValue)
// ============================================================================

/// Convert MetadataItem vec to SqlValue map (both v1 types)
pub fn metadata_items_to_sql_values(
    meta: Vec<crate::proto::proximadb_v1::MetadataItem>,
) -> std::collections::HashMap<String, crate::proto::proximadb_v1::SqlValue> {
    let mut out = std::collections::HashMap::new();
    for item in meta {
        let val = match item.value {
            Some(crate::proto::proximadb_v1::metadata_item::Value::StringValue(s)) => {
                crate::proto::proximadb_v1::SqlValue {
                    value: Some(crate::proto::proximadb_v1::sql_value::Value::StringValue(s)),
                }
            }
            Some(crate::proto::proximadb_v1::metadata_item::Value::NumberValue(n)) => {
                crate::proto::proximadb_v1::SqlValue {
                    value: Some(crate::proto::proximadb_v1::sql_value::Value::NumberValue(n)),
                }
            }
            Some(crate::proto::proximadb_v1::metadata_item::Value::BoolValue(b)) => {
                crate::proto::proximadb_v1::SqlValue {
                    value: Some(crate::proto::proximadb_v1::sql_value::Value::BoolValue(b)),
                }
            }
            None => crate::proto::proximadb_v1::SqlValue { value: None },
        };
        out.insert(item.key, val);
    }
    out
}

/// Convert SqlValue map to MetadataItem vec (both v1 types)
pub fn sql_values_to_metadata_items(
    meta: std::collections::HashMap<String, crate::proto::proximadb_v1::SqlValue>,
) -> Vec<crate::proto::proximadb_v1::MetadataItem> {
    meta.into_iter()
        .map(|(k, v)| {
            let val = match v.value {
                Some(crate::proto::proximadb_v1::sql_value::Value::StringValue(s)) => {
                    Some(crate::proto::proximadb_v1::metadata_item::Value::StringValue(s))
                }
                Some(crate::proto::proximadb_v1::sql_value::Value::NumberValue(n)) => {
                    Some(crate::proto::proximadb_v1::metadata_item::Value::NumberValue(n))
                }
                Some(crate::proto::proximadb_v1::sql_value::Value::BoolValue(b)) => Some(
                    crate::proto::proximadb_v1::metadata_item::Value::BoolValue(b),
                ),
                Some(crate::proto::proximadb_v1::sql_value::Value::Int64Value(i)) => {
                    Some(crate::proto::proximadb_v1::metadata_item::Value::NumberValue(i as f64))
                }
                Some(crate::proto::proximadb_v1::sql_value::Value::BytesValue(_)) => Some(
                    crate::proto::proximadb_v1::metadata_item::Value::StringValue(
                        "[Binary Data]".to_string(),
                    ),
                ),
                // MetadataItem's oneof carries only String/Number/Bool — the
                // canonical JSON text is the best lossy mapping (consistent
                // with the sibling map fn), not a byte-length placeholder.
                Some(crate::proto::proximadb_v1::sql_value::Value::JsonbValue(b)) => Some(
                    crate::proto::proximadb_v1::metadata_item::Value::StringValue(
                        proximadb_data_model::ProximaValue::jsonb_to_json_string_lossy(&b),
                    ),
                ),
                Some(crate::proto::proximadb_v1::sql_value::Value::NullValue(_)) => None,
                Some(crate::proto::proximadb_v1::sql_value::Value::ArrayValue(_)) => Some(
                    crate::proto::proximadb_v1::metadata_item::Value::StringValue(
                        "[Array]".to_string(),
                    ),
                ),
                Some(crate::proto::proximadb_v1::sql_value::Value::ObjectValue(_)) => Some(
                    crate::proto::proximadb_v1::metadata_item::Value::StringValue(
                        "[Object]".to_string(),
                    ),
                ),
                None => None,
            };
            crate::proto::proximadb_v1::MetadataItem { key: k, value: val }
        })
        .collect()
}

// ============================================================================
// Zero-Cost HashMap Metadata Access Patterns (Compile-Time Optimized)
// ============================================================================

/// Zero-cost inline helper for HashMap metadata iteration with .key/.value pattern
/// Compiler optimizes this away to direct tuple destructuring
#[inline(always)]
pub fn iter_metadata_optimized<'a>(
    metadata: &'a std::collections::HashMap<String, crate::proto::proximadb_v1::SqlValue>,
) -> impl Iterator<Item = (&'a String, &'a crate::proto::proximadb_v1::SqlValue)> + 'a {
    metadata.iter()
}

/// Zero-cost inline key access from HashMap tuple
#[inline(always)]
pub fn get_key<'a>(tuple: (&'a String, &'a crate::proto::proximadb_v1::SqlValue)) -> &'a String {
    tuple.0
}

/// Zero-cost inline value access from HashMap tuple  
#[inline(always)]
pub fn get_value<'a>(
    tuple: (&'a String, &'a crate::proto::proximadb_v1::SqlValue),
) -> &'a crate::proto::proximadb_v1::SqlValue {
    tuple.1
}

/// Zero-cost inline metadata lookup (O(1) vs O(n) scan)
#[inline(always)]
pub fn get_metadata_value<'a>(
    metadata: &'a std::collections::HashMap<String, crate::proto::proximadb_v1::SqlValue>,
    key: &str,
) -> Option<&'a crate::proto::proximadb_v1::SqlValue> {
    metadata.get(key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_operations() {
        assert_eq!(
            parse_collection_operation("create").unwrap(),
            CollectionOperation::CollectionCreate
        );
        assert_eq!(
            parse_vector_operation("search").unwrap(),
            VectorOperation::VectorSearch
        );
        assert!(parse_collection_operation("invalid").is_err());
    }

    #[test]
    fn test_metadata_conversion() {
        let mut json_metadata = serde_json::Map::new();
        json_metadata.insert("key1".to_string(), json!("value1"));
        json_metadata.insert("key2".to_string(), json!(42.0));
        json_metadata.insert("key3".to_string(), json!(true));

        let proto_metadata = convert_metadata_to_proto(json_metadata.clone());
        assert_eq!(proto_metadata.len(), 3);

        let back_to_json = convert_metadata_from_proto(proto_metadata);
        assert_eq!(back_to_json.len(), 3);
        assert_eq!(back_to_json.get("key1").unwrap(), &json!("value1"));
    }
}
