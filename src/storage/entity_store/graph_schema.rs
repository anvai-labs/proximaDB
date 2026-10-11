// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Schema mappers for SKS Graph-First Architecture
//!
//! This module provides bidirectional conversion between SKS Entity/Relation types
//! and Orion graph Node/Edge types.
//!
//! ## Design Principles
//! - **Lossless Conversion**: Round-trip Entity → Node → Entity preserves all data
//! - **Type Safety**: Use Rust type system to prevent invalid mappings
//! - **Performance**: Minimize allocations where possible
//!
//! ## Schema Mapping
//! ```text
//! Entity (SKS)                    Node (Orion)
//! ├── id                       →  id
//! ├── collection_id            →  labels[0] (primary label)
//! ├── embeddings[]             →  embedding (first version only for now)
//! ├── typed_metadata           →  properties["__typed_metadata"]
//! ├── flexible_metadata        →  properties["__flexible_metadata"]
//! ├── provenance               →  properties["__provenance"]
//! └── temporal                 →  properties["__temporal"]
//!
//! Relation (SKS)                  Edge (Orion)
//! ├── source_entity_id         →  from_node_id
//! ├── target_entity_id         →  to_node_id
//! ├── relation_type            →  edge_type
//! ├── weight                   →  weight
//! ├── created_at_ms            →  properties["created_at_ms"]
//! └── properties               →  properties (merged)
//! ```

use anyhow::{Context, Result};
use std::collections::HashMap;

use proximadb_data_model::ProximaValue;
use proximadb_records::conversions::{proxima_to_json, sql_value_to_proxima};

use crate::graph::{Edge, Node, PropertyValue, property_value};
use crate::proto::proximadb_v1::{EmbeddingVersion, Entity, Relation};

/// Special property keys for storing SKS metadata in Orion nodes
const TYPED_METADATA_KEY: &str = "__typed_metadata";
const FLEXIBLE_METADATA_KEY: &str = "__flexible_metadata";
const PROVENANCE_KEY: &str = "__provenance";
const TEMPORAL_KEY: &str = "__temporal";
const EMBEDDINGS_KEY: &str = "__embeddings";

/// Maps Entity to Node and vice versa
pub struct EntityNodeMapper;

/// JSON text becomes a StringValue property — except the literal `null`
/// document, which is the property model's null form (round-12 rule at this
/// third seam).
/// Serialize with SORTED object keys — stable across HashMap iteration
/// orders so exact-string property comparisons are write-order-independent
/// (serde_json's preserve_order is active via utoipa; unsorted
/// serialization is not canonical).
pub(crate) fn canonical_json_string(value: &serde_json::Value) -> String {
    fn sort_rec(v: &serde_json::Value) -> serde_json::Value {
        match v {
            serde_json::Value::Object(map) => {
                let mut entries: Vec<(String, serde_json::Value)> =
                    map.iter().map(|(k, v)| (k.clone(), sort_rec(v))).collect();
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                entries.into_iter().collect()
            }
            serde_json::Value::Array(values) => {
                serde_json::Value::Array(values.iter().map(sort_rec).collect())
            }
            _ => v.clone(),
        }
    }
    sort_rec(value).to_string()
}

/// Project a canonical value into the direct graph-property subset used by
/// legacy entity metadata filtering. Structured values use sorted-key JSON
/// text because this legacy filter compares strings exactly. JSON null remains
/// a present property with an unset inner oneof. Values outside the historical
/// direct-property subset remain available in the compatibility blob only.
fn proxima_value_to_direct_property(value: &ProximaValue) -> Option<PropertyValue> {
    let value = match value {
        ProximaValue::String(value) => Some(property_value::Value::StringValue(value.clone())),
        ProximaValue::Float64(value) => Some(property_value::Value::DoubleValue(*value)),
        ProximaValue::Int64(value) => Some(property_value::Value::IntValue(*value)),
        ProximaValue::Boolean(value) => Some(property_value::Value::BoolValue(*value)),
        ProximaValue::Json(_)
        | ProximaValue::Jsonb(_)
        | ProximaValue::Array(_)
        | ProximaValue::Map(_)
        | ProximaValue::Struct(_)
        | ProximaValue::SparseVector { .. } => {
            let json = proxima_to_json(value);
            if json.is_null() {
                None
            } else {
                Some(property_value::Value::StringValue(canonical_json_string(
                    &json,
                )))
            }
        }
        _ => return None,
    };
    Some(PropertyValue { value })
}

impl EntityNodeMapper {
    /// Convert SKS Entity to Orion Node
    ///
    /// ## Implementation Details
    /// - Entity ID becomes Node ID
    /// - collection_id becomes primary label (labels[0])
    /// - First embedding version stored in node.embedding
    /// - All metadata preserved in node properties with special prefixes
    pub fn entity_to_node(&self, entity: &Entity) -> Result<Node> {
        let mut properties = HashMap::new();

        // Store typed metadata as serialized JSON in properties
        if let Some(typed_metadata) = &entity.typed_metadata {
            let metadata_json = serde_json::to_string(typed_metadata)
                .context("Failed to serialize typed_metadata")?;
            properties.insert(
                TYPED_METADATA_KEY.to_string(),
                PropertyValue {
                    value: Some(property_value::Value::StringValue(metadata_json)),
                },
            );
        }

        // Store flexible metadata (convert SqlValue to PropertyValue)
        if !entity.flexible_metadata.is_empty() {
            // Store JSON blob for backward compatibility and full restoration
            let flexible_json = serde_json::to_string(&entity.flexible_metadata)
                .context("Failed to serialize flexible_metadata")?;
            properties.insert(
                FLEXIBLE_METADATA_KEY.to_string(),
                PropertyValue {
                    value: Some(property_value::Value::StringValue(flexible_json)),
                },
            );

            // Also store individual fields as direct node properties for efficient filtering
            // This enables matches_metadata_filter to work without deserializing JSON
            for (key, sql_value) in &entity.flexible_metadata {
                let canonical = sql_value_to_proxima(sql_value);
                if let Some(property) = proxima_value_to_direct_property(&canonical) {
                    properties.insert(key.clone(), property);
                }
            }
        }

        // Store provenance
        if let Some(provenance) = &entity.provenance {
            let provenance_json =
                serde_json::to_string(provenance).context("Failed to serialize provenance")?;
            properties.insert(
                PROVENANCE_KEY.to_string(),
                PropertyValue {
                    value: Some(property_value::Value::StringValue(provenance_json)),
                },
            );
        }

        // Store temporal info
        if let Some(temporal) = &entity.temporal {
            let temporal_json =
                serde_json::to_string(temporal).context("Failed to serialize temporal")?;
            properties.insert(
                TEMPORAL_KEY.to_string(),
                PropertyValue {
                    value: Some(property_value::Value::StringValue(temporal_json)),
                },
            );
        }

        // Store additional embedding versions (beyond the first) as JSON
        if entity.embeddings.len() > 1 {
            let embeddings_json = serde_json::to_string(&entity.embeddings[1..])
                .context("Failed to serialize additional embeddings")?;
            properties.insert(
                EMBEDDINGS_KEY.to_string(),
                PropertyValue {
                    value: Some(property_value::Value::StringValue(embeddings_json)),
                },
            );
        }

        Ok(Node {
            id: entity.id.clone(),
            labels: vec![entity.collection_id.clone()],
            properties,
            embedding: entity.embeddings.first().cloned().map(Into::into),
            created_at_ms: chrono::Utc::now().timestamp_millis(),
            updated_at_ms: chrono::Utc::now().timestamp_millis(),
        })
    }

    /// Convert Orion Node back to SKS Entity
    ///
    /// ## Implementation Details
    /// - Node ID becomes Entity ID
    /// - Primary label (labels[0]) becomes collection_id
    /// - Node.embedding becomes first embedding version
    /// - Deserialize all metadata fields from properties
    pub fn node_to_entity(&self, node: &Node) -> Result<Entity> {
        // Extract collection_id from primary label
        let collection_id = node
            .labels
            .first()
            .context("Node must have at least one label")?
            .clone();

        // Start with node.embedding if present
        let mut embeddings = Vec::new();
        if let Some(embedding) = &node.embedding {
            embeddings.push(embedding.clone().into());
        }

        // Add additional embeddings if stored in properties
        if let Some(embeddings_prop) = node.properties.get(EMBEDDINGS_KEY)
            && let Some(property_value::Value::StringValue(json)) = &embeddings_prop.value
        {
            let additional: Vec<EmbeddingVersion> = serde_json::from_str(json)
                .context("Failed to deserialize additional embeddings")?;
            embeddings.extend(additional);
        }

        // Extract typed metadata
        let typed_metadata = if let Some(metadata_prop) = node.properties.get(TYPED_METADATA_KEY) {
            if let Some(property_value::Value::StringValue(json)) = &metadata_prop.value {
                Some(serde_json::from_str(json).context("Failed to deserialize typed_metadata")?)
            } else {
                None
            }
        } else {
            None
        };

        // Extract flexible metadata
        let flexible_metadata =
            if let Some(flexible_prop) = node.properties.get(FLEXIBLE_METADATA_KEY) {
                if let Some(property_value::Value::StringValue(json)) = &flexible_prop.value {
                    serde_json::from_str(json).context("Failed to deserialize flexible_metadata")?
                } else {
                    HashMap::new()
                }
            } else {
                HashMap::new()
            };

        // Extract provenance
        let provenance = if let Some(provenance_prop) = node.properties.get(PROVENANCE_KEY) {
            if let Some(property_value::Value::StringValue(json)) = &provenance_prop.value {
                Some(serde_json::from_str(json).context("Failed to deserialize provenance")?)
            } else {
                None
            }
        } else {
            None
        };

        // Extract temporal
        let temporal = if let Some(temporal_prop) = node.properties.get(TEMPORAL_KEY) {
            if let Some(property_value::Value::StringValue(json)) = &temporal_prop.value {
                Some(serde_json::from_str(json).context("Failed to deserialize temporal")?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Entity {
            id: node.id.clone(),
            collection_id,
            embeddings,
            typed_metadata,
            flexible_metadata,
            provenance,
            temporal,
            relations: Vec::new(), // Relations stored separately as edges
        })
    }
}

/// Maps Relation to Edge and vice versa
pub struct RelationEdgeMapper;

impl RelationEdgeMapper {
    /// Convert SKS Relation to Orion Edge
    pub fn relation_to_edge(&self, relation: &Relation) -> Result<Edge> {
        let mut properties = HashMap::new();

        // Store created_at_ms
        properties.insert(
            "created_at_ms".to_string(),
            PropertyValue {
                value: Some(property_value::Value::IntValue(relation.created_at_ms)),
            },
        );

        // Merge relation properties (convert String to PropertyValue)
        for (key, string_value) in &relation.properties {
            properties.insert(
                key.clone(),
                PropertyValue {
                    value: Some(property_value::Value::StringValue(string_value.clone())),
                },
            );
        }

        Ok(Edge {
            id: uuid::Uuid::new_v4().to_string(),
            from_node_id: relation.source_entity_id.clone(),
            to_node_id: relation.target_entity_id.clone(),
            edge_type: relation.relation_type.clone(),
            properties,
            weight: Some(relation.weight as f64),
            created_at_ms: relation.created_at_ms,
            updated_at_ms: relation.created_at_ms,
        })
    }

    /// Convert Orion Edge back to SKS Relation
    pub fn edge_to_relation(&self, edge: &Edge) -> Result<Relation> {
        // Extract created_at_ms
        let created_at_ms = edge
            .properties
            .get("created_at_ms")
            .and_then(|v| {
                if let Some(property_value::Value::IntValue(i)) = &v.value {
                    Some(*i)
                } else {
                    None
                }
            })
            .unwrap_or(edge.created_at_ms);

        // Extract relation properties (skip created_at_ms, convert PropertyValue to String)
        let mut properties = HashMap::new();
        for (key, prop_value) in &edge.properties {
            if key != "created_at_ms" {
                // Convert PropertyValue to String (best effort)
                if let Some(property_value::Value::StringValue(s)) = &prop_value.value {
                    properties.insert(key.clone(), s.clone());
                } else if let Some(value) = &prop_value.value {
                    // For non-string values, serialize to JSON string
                    properties.insert(key.clone(), format!("{:?}", value));
                }
            }
        }

        Ok(Relation {
            source_entity_id: edge.from_node_id.clone(),
            target_entity_id: edge.to_node_id.clone(),
            relation_type: edge.edge_type.clone(),
            weight: edge.weight.unwrap_or(1.0) as f32,
            created_at_ms,
            properties,
        })
    }
}

// ============================================================================
// Helper Functions: SqlValue ↔ PropertyValue Conversion
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::proximadb_v1::Modality;
    use crate::proto::proximadb_v1::{
        SqlArray, SqlObject, SqlValue, TypedField, TypedMetadata, sql_value, typed_field,
    };

    #[test]
    fn canonical_json_sorts_objects_nested_inside_arrays() {
        let left = serde_json::json!([{"z": 1, "a": {"y": 2, "b": 3}}]);
        let right = serde_json::json!([{"a": {"b": 3, "y": 2}, "z": 1}]);

        assert_eq!(canonical_json_string(&left), canonical_json_string(&right));
    }

    #[test]
    fn flexible_structured_metadata_uses_canonical_json_and_preserves_null() {
        let json_null = proximadb_data_model::ProximaValue::to_jsonb_vec(&serde_json::Value::Null)
            .expect("encode JSON null test value");
        let entity = Entity {
            id: "entity-structured".to_string(),
            collection_id: "collection".to_string(),
            embeddings: Vec::new(),
            typed_metadata: None,
            flexible_metadata: HashMap::from([
                (
                    "document".to_string(),
                    SqlValue {
                        value: Some(sql_value::Value::ObjectValue(SqlObject {
                            fields: HashMap::from([(
                                "rank".to_string(),
                                SqlValue {
                                    value: Some(sql_value::Value::Int64Value(7)),
                                },
                            )]),
                        })),
                    },
                ),
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
                    "deleted_value".to_string(),
                    SqlValue {
                        value: Some(sql_value::Value::JsonbValue(json_null)),
                    },
                ),
            ]),
            provenance: None,
            temporal: None,
            relations: Vec::new(),
        };

        let node = EntityNodeMapper
            .entity_to_node(&entity)
            .expect("convert structured entity");

        assert_eq!(
            node.properties.get("document"),
            Some(&PropertyValue {
                value: Some(property_value::Value::StringValue(
                    serde_json::json!({"rank": 7}).to_string(),
                )),
            }),
            "proto wrapper fields must not leak into canonical JSON text"
        );
        assert_eq!(
            node.properties.get("tags"),
            Some(&PropertyValue {
                value: Some(property_value::Value::StringValue(
                    serde_json::json!(["red"]).to_string(),
                )),
            })
        );
        assert_eq!(
            node.properties.get("deleted_value"),
            Some(&PropertyValue { value: None }),
            "JSON null is a present null property, not an absent field"
        );
    }

    #[test]
    fn test_entity_node_round_trip() {
        // Create test entity
        let mut entity = Entity {
            id: "test-entity-1".to_string(),
            collection_id: "test-collection".to_string(),
            embeddings: vec![EmbeddingVersion {
                model_id: "test-model".to_string(),
                model_version: "v1".to_string(),
                vector: vec![0.1, 0.2, 0.3],
                dimension: 3,
                created_at_ms: 1234567890,
                model_params: HashMap::new(),
                modality: Modality::Text as i32,
            }],
            typed_metadata: None,
            flexible_metadata: HashMap::new(),
            provenance: None,
            temporal: None,
            relations: Vec::new(),
        };

        // Add typed metadata
        let mut fields = HashMap::new();
        fields.insert(
            "name".to_string(),
            TypedField {
                indexed: true,
                filterable: true,
                value: Some(typed_field::Value::StringValue("Test Entity".to_string())),
            },
        );
        entity.typed_metadata = Some(TypedMetadata { fields });

        // Convert to node
        let mapper = EntityNodeMapper;
        let node = mapper
            .entity_to_node(&entity)
            .expect("Failed to convert to node");

        // Verify node structure
        assert_eq!(node.id, entity.id);
        assert_eq!(node.labels[0], entity.collection_id);
        assert!(node.embedding.is_some());
        assert!(node.properties.contains_key(TYPED_METADATA_KEY));

        // Convert back to entity
        let entity_restored = mapper
            .node_to_entity(&node)
            .expect("Failed to convert back to entity");

        // Verify round-trip correctness
        assert_eq!(entity_restored.id, entity.id);
        assert_eq!(entity_restored.collection_id, entity.collection_id);
        assert_eq!(entity_restored.embeddings.len(), entity.embeddings.len());
        assert_eq!(
            entity_restored.embeddings[0].vector,
            entity.embeddings[0].vector
        );
        assert!(entity_restored.typed_metadata.is_some());
    }

    #[test]
    fn test_relation_edge_round_trip() {
        // Create test relation
        let relation = Relation {
            source_entity_id: "entity-1".to_string(),
            target_entity_id: "entity-2".to_string(),
            relation_type: "related_to".to_string(),
            weight: 0.85,
            created_at_ms: 1234567890,
            properties: HashMap::new(),
        };

        // Convert to edge
        let mapper = RelationEdgeMapper;
        let edge = mapper
            .relation_to_edge(&relation)
            .expect("Failed to convert to edge");

        // Verify edge structure
        assert_eq!(edge.from_node_id, relation.source_entity_id);
        assert_eq!(edge.to_node_id, relation.target_entity_id);
        assert_eq!(edge.edge_type, relation.relation_type);
        assert_eq!(edge.weight, Some(relation.weight as f64));

        // Convert back to relation
        let relation_restored = mapper
            .edge_to_relation(&edge)
            .expect("Failed to convert back to relation");

        // Verify round-trip correctness
        assert_eq!(
            relation_restored.source_entity_id,
            relation.source_entity_id
        );
        assert_eq!(
            relation_restored.target_entity_id,
            relation.target_entity_id
        );
        assert_eq!(relation_restored.relation_type, relation.relation_type);
        assert_eq!(relation_restored.weight, relation.weight);
    }
}
