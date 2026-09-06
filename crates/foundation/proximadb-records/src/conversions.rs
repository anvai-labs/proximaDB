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

//! Bidirectional conversions between modality-specific proto records and
//! [`ProximaRecord`] (spec §3 — Phase B).
//!
//! These `From` impls are the seam between the legacy per-modality proto
//! types and the unified envelope. They live here (not in the root crate)
//! to keep the dependency direction clean: `proximadb-records` → `proximadb-proto`.

use std::collections::HashMap;

use proximadb_data_model::ProximaValue;
use proximadb_proto::proximadb_v1::{
    Edge, EmbeddingVersion, Node, PropertyValue, SqlArray, SqlObject, SqlValue, VectorRecord,
    property_value, sql_value,
};

use crate::{
    EdgeDirection, EdgeShape, EmbeddingCell, LabelSet, ProximaRecord, ProximaTree, ProximaTreeNode,
    TypedRef, schema_version,
};

// ---------------------------------------------------------------------------
// Value conversion helpers
// ---------------------------------------------------------------------------

/// Convert a proto `SqlValue` to a `ProximaValue`.
///
/// Used for metadata fields in `VectorRecord`.
pub fn sql_value_to_proxima(sql: &SqlValue) -> ProximaValue {
    match &sql.value {
        Some(sql_value::Value::StringValue(s)) => ProximaValue::String(s.clone()),
        Some(sql_value::Value::NumberValue(f)) => ProximaValue::Float64(*f),
        Some(sql_value::Value::BoolValue(b)) => ProximaValue::Boolean(*b),
        Some(sql_value::Value::Int64Value(i)) => ProximaValue::Int64(*i),
        Some(sql_value::Value::BytesValue(b)) => {
            // Round 8: strip the legacy magic prefix FIRST — feeding it to
            // rmp-serde "succeeds" (0xff is a negative fixint; trailing bytes
            // ignored) and silently replaces the document with Number(-1).
            // This matches search-types' decoder for magic-prefixed bytes
            // (magic + JSON text); only the magic BYTES are shared via the
            // data-model const — the codec bodies remain line-copies pending
            // the tracked consolidation.
            // The legacy payload after the magic is JSON TEXT (matching
            // search-types' decoder) — round 8 mistakenly msgpack-decoded it,
            // which "succeeds" on short payloads as a garbage number and
            // mangles documents into Binary (round-9 empirical finding).
            if let Some(payload) = b.strip_prefix(proximadb_data_model::JSONB_LEGACY_MAGIC) {
                match serde_json::from_slice(payload) {
                    Ok(v) => return ProximaValue::Jsonb(v),
                    Err(_) => return ProximaValue::Binary(b.clone()),
                }
            }
            ProximaValue::from_jsonb_or_binary(b)
        }
        // types.proto tag 9: JSONB by declaration — canonical decode-or-binary
        // via the shared helper.
        Some(sql_value::Value::JsonbValue(b)) => ProximaValue::from_jsonb_or_binary(b),
        Some(sql_value::Value::NullValue(_)) => ProximaValue::Null,
        Some(sql_value::Value::ArrayValue(arr)) => {
            let items: Vec<ProximaValue> = arr.values.iter().map(sql_value_to_proxima).collect();
            ProximaValue::Array(items)
        }
        Some(sql_value::Value::ObjectValue(obj)) => {
            let map: HashMap<String, ProximaValue> = obj
                .fields
                .iter()
                .map(|(k, v)| (k.clone(), sql_value_to_proxima(v)))
                .collect();
            ProximaValue::Map(map)
        }
        None => ProximaValue::Null,
    }
}

/// Canonical JSON *rendering* of a proto `SqlValue` — THE one converter for
/// API-facing surfaces (REST rows, pgwire, Arrow, hybrid, document adapter,
/// metadata helper). TD-PROTO-2 consolidation: these sites carried seven
/// hand-rolled copies whose bytes/NaN/Jsonb renderings had drifted (hex vs
/// base64 vs int-array; NaN→0 vs Null). Rendering contract:
///
/// - bytes → base64 string (data-preserving, matches pgwire/Arrow)
/// - non-finite floats → null (JSON has no NaN/∞)
/// - JsonbValue → canonical MessagePack decode (jsonb_to_json_lossy)
/// - unset oneof / NullValue → null
///
/// Distinct from `sql_value_filter::sql_val_to_json` (search-types), the
/// FILTER LOWERING — it renders bytes as an int-array for comparison
/// semantics and must not be conflated with rendering.
pub fn sql_value_to_json(value: &SqlValue) -> serde_json::Value {
    match value.value.as_ref() {
        None | Some(sql_value::Value::NullValue(_)) => serde_json::Value::Null,
        Some(sql_value::Value::BoolValue(b)) => serde_json::Value::Bool(*b),
        Some(sql_value::Value::Int64Value(i)) => serde_json::Value::Number((*i).into()),
        Some(sql_value::Value::NumberValue(f)) => serde_json::Number::from_f64(*f)
            .map_or(serde_json::Value::Null, serde_json::Value::Number),
        Some(sql_value::Value::StringValue(s)) => serde_json::Value::String(s.clone()),
        Some(sql_value::Value::BytesValue(b)) => {
            serde_json::Value::String(proximadb_proto::utils::encoding::base64_encode(b))
        }
        Some(sql_value::Value::JsonbValue(b)) => ProximaValue::jsonb_to_json_lossy(b),
        Some(sql_value::Value::ArrayValue(arr)) => {
            serde_json::Value::Array(arr.values.iter().map(sql_value_to_json).collect())
        }
        Some(sql_value::Value::ObjectValue(obj)) => serde_json::Value::Object(
            obj.fields
                .iter()
                .map(|(k, v)| (k.clone(), sql_value_to_json(v)))
                .collect(),
        ),
    }
}

/// Canonical JSON *rendering* of a `ProximaValue` for API-facing surfaces —
/// the ProximaValue-side twin of [`sql_value_to_json`]: Binary/BinaryVector
/// as base64 (proxima_to_json alone renders them per-byte int-arrays,
/// INCLUDING nested inside Array/Map/Struct), Uuid/ULID dashed-hex (the
/// cross-surface text convention). Not inlined into proxima_to_json because
/// persisted graph canonical-text seams rely on its exact current output.
pub fn proxima_value_to_json_canonical(value: &ProximaValue) -> serde_json::Value {
    match value {
        ProximaValue::Binary(b) | ProximaValue::BinaryVector(b) => {
            serde_json::Value::String(proximadb_proto::utils::encoding::base64_encode(b))
        }
        ProximaValue::Uuid(u) => {
            serde_json::Value::String(proximadb_kernel::uuid::Uuid::from_bytes(*u).to_string())
        }
        ProximaValue::ULID(u) => {
            serde_json::Value::String(u.iter().map(|x| format!("{x:02x}")).collect::<String>())
        }
        ProximaValue::Array(items) => {
            serde_json::Value::Array(items.iter().map(proxima_value_to_json_canonical).collect())
        }
        ProximaValue::Map(fields) | ProximaValue::Struct(fields) => serde_json::Value::Object(
            fields
                .iter()
                .map(|(k, v)| (k.clone(), proxima_value_to_json_canonical(v)))
                .collect(),
        ),
        other => proxima_to_json(other),
    }
}

/// Convert a `serde_json::Value` into a `ProximaValue` (natural-JSON mapping).
///
/// Numbers split into `Int64`/`Float64`. This matches the composition
/// `json -> SqlValue -> sql_value_to_proxima`, so internal compute migrating off a
/// jsonpath/`serde_json` bridge (e.g. document filter path extraction) keeps identical
/// value semantics. Canonical home for the previously-scattered `json_to_proxima_value`
/// copies (arrow_ipc, websocket, rest/v2) to converge onto later.
pub fn json_to_proxima(value: &serde_json::Value) -> ProximaValue {
    match value {
        serde_json::Value::Null => ProximaValue::Null,
        serde_json::Value::Bool(b) => ProximaValue::Boolean(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                ProximaValue::Int64(i)
            } else {
                ProximaValue::Float64(n.as_f64().unwrap_or(0.0))
            }
        }
        serde_json::Value::String(s) => ProximaValue::String(s.clone()),
        serde_json::Value::Array(arr) => {
            ProximaValue::Array(arr.iter().map(json_to_proxima).collect())
        }
        serde_json::Value::Object(obj) => ProximaValue::Map(
            obj.iter()
                .map(|(k, v)| (k.clone(), json_to_proxima(v)))
                .collect(),
        ),
    }
}

/// Convert a canonical [`ProximaValue`] to its natural JSON representation.
///
/// This is a wire-edge projection for JSON transports. Internal execution
/// must retain `ProximaValue`; callers should invoke this only while building
/// an HTTP/document payload. Non-finite floats become JSON `null`, binary
/// values become byte arrays, and vectors retain their numeric structure.
pub fn proxima_to_json(value: &ProximaValue) -> serde_json::Value {
    use serde_json::{Map, Number, Value};

    fn float(value: f64) -> Value {
        Number::from_f64(value).map_or(Value::Null, Value::Number)
    }

    match value {
        ProximaValue::Null => Value::Null,
        ProximaValue::Boolean(value) => Value::Bool(*value),
        ProximaValue::Int8(value) => Value::Number((*value as i64).into()),
        ProximaValue::Int16(value) => Value::Number((*value as i64).into()),
        ProximaValue::Int32(value) => Value::Number((*value as i64).into()),
        ProximaValue::Int64(value) => Value::Number((*value).into()),
        ProximaValue::UInt8(value) => Value::Number((*value as u64).into()),
        ProximaValue::UInt16(value) => Value::Number((*value as u64).into()),
        ProximaValue::UInt32(value) => Value::Number((*value as u64).into()),
        ProximaValue::UInt64(value) => Value::Number((*value).into()),
        ProximaValue::Float16(value) | ProximaValue::Float32(value) => float(*value as f64),
        ProximaValue::Float64(value) => float(*value),
        ProximaValue::Decimal(value)
        | ProximaValue::String(value)
        | ProximaValue::Symbol(value) => Value::String(value.clone()),
        ProximaValue::Binary(value) | ProximaValue::BinaryVector(value) => Value::Array(
            value
                .iter()
                .map(|byte| Value::Number((*byte as u64).into()))
                .collect(),
        ),
        ProximaValue::Date(value) => Value::Number((*value as i64).into()),
        ProximaValue::Time(value, _)
        | ProximaValue::Timestamp(value, _)
        | ProximaValue::TimestampTz(value, _) => Value::Number((*value).into()),
        ProximaValue::Uuid(value) | ProximaValue::ULID(value) => Value::String(
            value
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>(),
        ),
        ProximaValue::Json(value) | ProximaValue::Jsonb(value) => value.clone(),
        ProximaValue::Array(values) => Value::Array(values.iter().map(proxima_to_json).collect()),
        ProximaValue::Map(values) | ProximaValue::Struct(values) => Value::Object(
            values
                .iter()
                .map(|(key, value)| (key.clone(), proxima_to_json(value)))
                .collect::<Map<_, _>>(),
        ),
        ProximaValue::DenseVector(values) => {
            Value::Array(values.iter().map(|value| float(*value as f64)).collect())
        }
        ProximaValue::SparseVector { indices, values } => Value::Object(
            [
                (
                    "indices".to_string(),
                    Value::Array(
                        indices
                            .iter()
                            .map(|value| Value::Number((*value as u64).into()))
                            .collect(),
                    ),
                ),
                (
                    "values".to_string(),
                    Value::Array(values.iter().map(|value| float(*value as f64)).collect()),
                ),
            ]
            .into_iter()
            .collect(),
        ),
    }
}

/// Convert a canonical `ProximaValue` into the legacy v1 `SqlValue`.
///
/// This exists only as a storage-service adapter while the vector operations
/// service still accepts `VectorRecord`. Public and internal v2 APIs should use
/// `ProximaValue`/`ProximaRecord` directly.
pub fn proxima_to_sql_value(value: &ProximaValue) -> SqlValue {
    let inner = match value {
        ProximaValue::Boolean(v) => sql_value::Value::BoolValue(*v),
        ProximaValue::Int8(v) => sql_value::Value::Int64Value(*v as i64),
        ProximaValue::Int16(v) => sql_value::Value::Int64Value(*v as i64),
        ProximaValue::Int32(v) => sql_value::Value::Int64Value(*v as i64),
        ProximaValue::Int64(v) => sql_value::Value::Int64Value(*v),
        ProximaValue::UInt8(v) => sql_value::Value::Int64Value(*v as i64),
        ProximaValue::UInt16(v) => sql_value::Value::Int64Value(*v as i64),
        ProximaValue::UInt32(v) => sql_value::Value::Int64Value(*v as i64),
        ProximaValue::UInt64(v) => sql_value::Value::StringValue(v.to_string()),
        ProximaValue::Float16(v) | ProximaValue::Float32(v) => {
            sql_value::Value::NumberValue(*v as f64)
        }
        ProximaValue::Float64(v) => sql_value::Value::NumberValue(*v),
        ProximaValue::Decimal(v) => sql_value::Value::StringValue(v.clone()),
        ProximaValue::String(v) | ProximaValue::Symbol(v) => {
            sql_value::Value::StringValue(v.clone())
        }
        ProximaValue::Binary(v) | ProximaValue::BinaryVector(v) => {
            sql_value::Value::BytesValue(v.clone())
        }
        ProximaValue::Date(v) => sql_value::Value::Int64Value(*v as i64),
        ProximaValue::Time(v, _) => sql_value::Value::Int64Value(*v),
        ProximaValue::Timestamp(v, _) | ProximaValue::TimestampTz(v, _) => {
            sql_value::Value::Int64Value(*v)
        }
        ProximaValue::Uuid(v) | ProximaValue::ULID(v) => sql_value::Value::BytesValue(v.to_vec()),
        ProximaValue::Json(v) => sql_value::Value::StringValue(v.to_string()),
        // TD-PROTO-2: emit the declared tag-9 variant so JSONB is typed on the
        // wire and the canonical decode path (jsonb_to_json_lossy / filter
        // lowering) is reachable. Transitional mixed-read state, stated
        // precisely: segments written before this change carry the same
        // MessagePack bytes under tag 8 (BytesValue). Readers still DECODE
        // those (BytesValue arm → opaque bytes/binary), so no data is lost,
        // but canonical JSON rendering and structured filters only engage for
        // tag-9 values — legacy-tagged JSONB filters exactly as it did before
        // this PR (byte-rendering miss), until rewritten. The tag-9 variant
        // did not exist before this PR.
        ProximaValue::Jsonb(v) => match ProximaValue::to_jsonb_vec(v) {
            Ok(bytes) => sql_value::Value::JsonbValue(bytes),
            Err(_) => {
                // Round 7: fall back to the legacy tag-8 magic+JSON-text
                // encoding (matching search-types' writer) — the existing
                // magic-stripping decoder recovers it losslessly. Plain
                // JSON-text under tag 9 would violate tag 9's MessagePack
                // contract and silently change type on read.
                let mut bytes = proximadb_data_model::JSONB_LEGACY_MAGIC.to_vec();
                bytes.extend_from_slice(v.to_string().as_bytes());
                sql_value::Value::BytesValue(bytes)
            }
        },
        ProximaValue::Array(values) => sql_value::Value::ArrayValue(SqlArray {
            values: values.iter().map(proxima_to_sql_value).collect(),
        }),
        ProximaValue::Map(values) | ProximaValue::Struct(values) => {
            sql_value::Value::ObjectValue(SqlObject {
                fields: values
                    .iter()
                    .map(|(k, v)| (k.clone(), proxima_to_sql_value(v)))
                    .collect(),
            })
        }
        ProximaValue::DenseVector(values) => sql_value::Value::ArrayValue(SqlArray {
            values: values
                .iter()
                .map(|v| SqlValue {
                    value: Some(sql_value::Value::NumberValue(*v as f64)),
                })
                .collect(),
        }),
        ProximaValue::SparseVector { indices, values } => {
            sql_value::Value::ObjectValue(SqlObject {
                fields: HashMap::from([
                    (
                        "indices".to_string(),
                        proxima_to_sql_value(&ProximaValue::Array(
                            indices.iter().map(|v| ProximaValue::UInt32(*v)).collect(),
                        )),
                    ),
                    (
                        "values".to_string(),
                        proxima_to_sql_value(&ProximaValue::DenseVector(values.clone())),
                    ),
                ]),
            })
        }
        ProximaValue::Null => sql_value::Value::NullValue(0),
    };

    SqlValue { value: Some(inner) }
}

/// Flatten a `ProximaTree` into legacy metadata for the remaining v1 storage path.
pub fn proxima_tree_to_sql_metadata(tree: &ProximaTree) -> HashMap<String, SqlValue> {
    tree.iter()
        .map(|(key, node)| {
            let value = match node {
                ProximaTreeNode::Value(value) => proxima_to_sql_value(value),
                ProximaTreeNode::Object(subtree) => {
                    proxima_to_sql_value(&ProximaValue::Map(proxima_tree_to_value_map(subtree)))
                }
            };
            (key.clone(), value)
        })
        .collect()
}

/// Convert a canonical nested property tree to nested `ProximaValue` maps.
pub fn proxima_tree_to_value_map(tree: &ProximaTree) -> HashMap<String, ProximaValue> {
    tree.iter()
        .map(|(key, node)| {
            let value = match node {
                ProximaTreeNode::Value(value) => value.clone(),
                ProximaTreeNode::Object(subtree) => {
                    ProximaValue::Map(proxima_tree_to_value_map(subtree))
                }
            };
            (key.clone(), value)
        })
        .collect()
}

/// Convert the canonical envelope to `VectorRecord` for the current vector
/// operations service. This should be deleted once storage accepts envelopes.
pub fn proxima_record_to_vector(record: &ProximaRecord) -> VectorRecord {
    // INT-2.5b: extract a fp32 view regardless of underlying variant.
    // VectorRecord is a v1 proto with `vector: Vec<f32>`, so non-Fp32
    // variants get a one-shot promote here.
    let vector: Vec<f32> = record
        .embeddings
        .first()
        .map(|embedding| embedding.values.to_fp32_owned())
        .unwrap_or_default();

    VectorRecord {
        id: record.oid.clone(),
        vector,
        metadata: proxima_tree_to_sql_metadata(&record.props),
        timestamp: Some(record.created_at_ns / 1_000_000),
        updated_at: Some(record.updated_at_ns / 1_000_000),
        expires_at: record.valid_to_ns.map(|ns| ns / 1_000_000),
        version: Some(record.record_version as u32),
        source: record.origin.clone(),
    }
}

// ---------------------------------------------------------------------------
// ProximaRecord → VectorRecord
// ---------------------------------------------------------------------------

impl From<ProximaRecord> for VectorRecord {
    fn from(r: ProximaRecord) -> Self {
        proxima_record_to_vector(&r)
    }
}

impl From<&ProximaRecord> for VectorRecord {
    fn from(r: &ProximaRecord) -> Self {
        proxima_record_to_vector(r)
    }
}

/// Convert a proto `PropertyValue` to a `ProximaValue`.
///
/// Used for property maps in graph `Node` and `Edge`.
pub fn property_value_to_proxima(pv: &PropertyValue) -> ProximaValue {
    match &pv.value {
        Some(property_value::Value::StringValue(s)) => ProximaValue::String(s.clone()),
        Some(property_value::Value::IntValue(i)) => ProximaValue::Int64(*i),
        Some(property_value::Value::DoubleValue(f)) => ProximaValue::Float64(*f),
        Some(property_value::Value::BoolValue(b)) => ProximaValue::Boolean(*b),
        Some(property_value::Value::BytesValue(b)) => ProximaValue::Binary(b.clone()),
        Some(property_value::Value::ArrayValue(arr)) => {
            let items: Vec<ProximaValue> =
                arr.values.iter().map(property_value_to_proxima).collect();
            ProximaValue::Array(items)
        }
        Some(property_value::Value::ObjectValue(obj)) => {
            let map: HashMap<String, ProximaValue> = obj
                .fields
                .iter()
                .map(|(k, v)| (k.clone(), property_value_to_proxima(v)))
                .collect();
            ProximaValue::Map(map)
        }
        Some(property_value::Value::VectorValue(vd)) => {
            ProximaValue::DenseVector(vd.values.clone())
        }
        None => ProximaValue::Null,
    }
}

/// Convert an `EmbeddingVersion` proto to an `EmbeddingCell`.
pub fn embedding_version_to_cell(ev: &EmbeddingVersion) -> EmbeddingCell {
    EmbeddingCell::new_fp32(
        ev.model_id.clone(),
        format!("{}", ev.modality), // stored as i32 in proto
        ev.dimension,
        ev.vector.clone(),
    )
}

fn ms_to_ns(ms: i64) -> i64 {
    ms.saturating_mul(1_000_000)
}

// ---------------------------------------------------------------------------
// VectorRecord → ProximaRecord
// ---------------------------------------------------------------------------

impl From<VectorRecord> for ProximaRecord {
    fn from(v: VectorRecord) -> Self {
        ProximaRecord::from(&v)
    }
}

impl From<&VectorRecord> for ProximaRecord {
    fn from(v: &VectorRecord) -> Self {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as i64;

        let created_at_ns = v.timestamp.map(ms_to_ns).unwrap_or(now_ns);
        let updated_at_ns = v.updated_at.map(ms_to_ns).unwrap_or(created_at_ns);

        // Convert metadata map to ProximaTree
        let props: ProximaTree = v
            .metadata
            .iter()
            .map(|(k, sv)| (k.clone(), ProximaTreeNode::Value(sql_value_to_proxima(sv))))
            .collect();

        // The vector becomes a default embedding cell
        let embeddings = if !v.vector.is_empty() {
            vec![EmbeddingCell::new_fp32(
                "default",
                "dense_vector",
                v.vector.len() as u32,
                v.vector.clone(),
            )]
        } else {
            vec![]
        };

        ProximaRecord {
            oid: v.id.clone(),
            local_id: None,
            tid: None,
            variation_id: None,
            record_version: v.version.unwrap_or(0) as u64,
            spec_version: 1,
            tenant_id: String::new(),
            permitted_principals: Vec::new(),
            rls_policy_id: None,
            created_at_ns,
            updated_at_ns,
            valid_from_ns: None,
            valid_to_ns: v.expires_at.map(ms_to_ns),
            origin: v.source.clone(),
            actor: None,
            method: Some("vector_insert".to_string()),
            memory_type: None,
            props,
            refs: Vec::new(),
            edge: None,
            embeddings,
            sequence: None,
            labels: LabelSet::new(),
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Node → ProximaRecord
// ---------------------------------------------------------------------------

impl From<&Node> for ProximaRecord {
    fn from(n: &Node) -> Self {
        let created_at_ns = ms_to_ns(n.created_at_ms);
        let updated_at_ns = ms_to_ns(n.updated_at_ms);

        let props: ProximaTree = n
            .properties
            .iter()
            .map(|(k, pv)| {
                (
                    k.clone(),
                    ProximaTreeNode::Value(property_value_to_proxima(pv)),
                )
            })
            .collect();

        let labels = LabelSet::from(n.labels.clone());

        let embeddings = if let Some(ev) = &n.embedding {
            vec![embedding_version_to_cell(ev)]
        } else {
            vec![]
        };

        // Add a ref back to the graph node origin so cross-model queries can find it
        let refs = vec![TypedRef::GraphEdge {
            edge_id: n.id.clone(),
            direction: EdgeDirection::Outgoing,
        }];

        ProximaRecord {
            schema_version: schema_version::default_schema_version(),
            oid: n.id.clone(),
            local_id: None,
            tid: None,
            variation_id: None,
            record_version: 0,
            spec_version: 1,
            tenant_id: String::new(),
            permitted_principals: Vec::new(),
            rls_policy_id: None,
            created_at_ns,
            updated_at_ns,
            valid_from_ns: None,
            valid_to_ns: None,
            origin: None,
            actor: None,
            method: Some("graph_node".to_string()),
            memory_type: None,
            props,
            refs,
            edge: None,
            embeddings,
            sequence: None,
            labels,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Edge → ProximaRecord
// ---------------------------------------------------------------------------

impl From<&Edge> for ProximaRecord {
    fn from(e: &Edge) -> Self {
        let created_at_ns = ms_to_ns(e.created_at_ms);
        let updated_at_ns = ms_to_ns(e.updated_at_ms);

        let props: ProximaTree = e
            .properties
            .iter()
            .map(|(k, pv)| {
                (
                    k.clone(),
                    ProximaTreeNode::Value(property_value_to_proxima(pv)),
                )
            })
            .collect();

        let edge = Some(EdgeShape {
            source_id: e.from_node_id.clone(),
            target_id: e.to_node_id.clone(),
            edge_type: e.edge_type.clone(),
            weight: e.weight,
        });

        ProximaRecord {
            schema_version: schema_version::default_schema_version(),
            oid: e.id.clone(),
            local_id: None,
            tid: None,
            variation_id: None,
            record_version: 0,
            spec_version: 1,
            tenant_id: String::new(),
            permitted_principals: Vec::new(),
            rls_policy_id: None,
            created_at_ns,
            updated_at_ns,
            valid_from_ns: None,
            valid_to_ns: None,
            origin: None,
            actor: None,
            method: Some("graph_edge".to_string()),
            memory_type: None,
            props,
            refs: Vec::new(),
            edge,
            embeddings: Vec::new(),
            sequence: None,
            labels: LabelSet::new(),
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Tests (TDD — written before impl)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_data_model::ProximaValue;
    use proximadb_proto::proximadb_v1::{
        PropertyValue, SqlValue, VectorRecord, property_value, sql_value,
    };
    use std::collections::HashMap;

    fn make_sql_string(s: &str) -> SqlValue {
        SqlValue {
            value: Some(sql_value::Value::StringValue(s.to_string())),
        }
    }

    fn make_sql_int(i: i64) -> SqlValue {
        SqlValue {
            value: Some(sql_value::Value::Int64Value(i)),
        }
    }

    fn make_sql_float(f: f64) -> SqlValue {
        SqlValue {
            value: Some(sql_value::Value::NumberValue(f)),
        }
    }

    fn make_prop_string(s: &str) -> PropertyValue {
        PropertyValue {
            value: Some(property_value::Value::StringValue(s.to_string())),
        }
    }

    fn make_prop_int(i: i64) -> PropertyValue {
        PropertyValue {
            value: Some(property_value::Value::IntValue(i)),
        }
    }

    #[test]
    fn proxima_to_json_is_a_natural_wire_edge_projection() {
        let value = ProximaValue::Map(HashMap::from([
            ("count".to_string(), ProximaValue::Int64(7)),
            (
                "items".to_string(),
                ProximaValue::Array(vec![ProximaValue::Boolean(true), ProximaValue::Null]),
            ),
        ]));
        assert_eq!(
            proxima_to_json(&value),
            serde_json::json!({"count": 7, "items": [true, null]})
        );
    }

    #[test]
    fn proxima_to_json_preserves_unsigned_and_vector_values() {
        assert_eq!(
            proxima_to_json(&ProximaValue::UInt64(u64::MAX)),
            serde_json::json!(u64::MAX)
        );
        assert_eq!(
            proxima_to_json(&ProximaValue::DenseVector(vec![0.25, 0.5])),
            serde_json::json!([0.25, 0.5])
        );
    }

    // --- SqlValue → ProximaValue ---

    #[test]
    fn test_sql_string_to_proxima() {
        let sv = make_sql_string("hello");
        assert!(matches!(sql_value_to_proxima(&sv), ProximaValue::String(s) if s == "hello"));
    }

    #[test]
    fn test_sql_int_to_proxima() {
        let sv = make_sql_int(42);
        assert!(matches!(sql_value_to_proxima(&sv), ProximaValue::Int64(42)));
    }

    #[test]
    fn test_sql_float_to_proxima() {
        let expected = 3.125;
        let sv = make_sql_float(expected);
        match sql_value_to_proxima(&sv) {
            ProximaValue::Float64(f) => assert!((f - expected).abs() < 1e-9),
            other => panic!("expected Float64, got {:?}", other),
        }
    }

    #[test]
    fn test_sql_null_to_proxima() {
        let sv = SqlValue {
            value: Some(sql_value::Value::NullValue(0)),
        };
        assert!(matches!(sql_value_to_proxima(&sv), ProximaValue::Null));
    }

    #[test]
    fn test_sql_none_to_proxima_null() {
        let sv = SqlValue { value: None };
        assert!(matches!(sql_value_to_proxima(&sv), ProximaValue::Null));
    }

    #[test]
    fn test_sql_jsonb_to_proxima() {
        let original = serde_json::json!({"foo": "bar"});
        let bytes = ProximaValue::to_jsonb_vec(&original).unwrap();
        let sv = SqlValue {
            value: Some(sql_value::Value::BytesValue(bytes)),
        };
        match sql_value_to_proxima(&sv) {
            ProximaValue::Jsonb(v) => assert_eq!(v, original),
            other => panic!("expected Jsonb, got {:?}", other),
        }
    }

    #[test]
    fn legacy_magic_json_text_decodes_to_jsonb() {
        let original = serde_json::json!({"memory": {"type": "fact"}, "rank": 7});
        let mut bytes = proximadb_data_model::JSONB_LEGACY_MAGIC.to_vec();
        bytes.extend_from_slice(original.to_string().as_bytes());
        let sql = SqlValue {
            value: Some(sql_value::Value::BytesValue(bytes)),
        };

        assert_eq!(
            sql_value_to_proxima(&sql),
            ProximaValue::Jsonb(original),
            "the legacy magic payload is JSON text, not MessagePack"
        );
    }

    #[test]
    fn invalid_legacy_magic_payload_stays_binary() {
        let mut bytes = proximadb_data_model::JSONB_LEGACY_MAGIC.to_vec();
        bytes.extend_from_slice(b"{not-json");
        let sql = SqlValue {
            value: Some(sql_value::Value::BytesValue(bytes.clone())),
        };

        assert_eq!(sql_value_to_proxima(&sql), ProximaValue::Binary(bytes));
    }

    #[test]
    fn sql_value_conversion_covers_binary_bool_array_and_object_shapes() {
        let binary = SqlValue {
            value: Some(sql_value::Value::BytesValue(vec![0xc1])),
        };
        assert_eq!(
            sql_value_to_proxima(&binary),
            ProximaValue::Binary(vec![0xc1])
        );
        assert_eq!(
            sql_value_to_proxima(&SqlValue {
                value: Some(sql_value::Value::BoolValue(true)),
            }),
            ProximaValue::Boolean(true)
        );
        assert_eq!(
            sql_value_to_proxima(&SqlValue {
                value: Some(sql_value::Value::ArrayValue(SqlArray {
                    values: vec![make_sql_string("x"), make_sql_int(7)],
                })),
            }),
            ProximaValue::Array(vec![
                ProximaValue::String("x".to_string()),
                ProximaValue::Int64(7),
            ])
        );
        assert_eq!(
            sql_value_to_proxima(&SqlValue {
                value: Some(sql_value::Value::ObjectValue(SqlObject {
                    fields: HashMap::from([("k".to_string(), make_sql_float(1.5))]),
                })),
            }),
            ProximaValue::Map(HashMap::from([(
                "k".to_string(),
                ProximaValue::Float64(1.5),
            )]))
        );
    }

    #[test]
    fn canonical_uuid_is_dashed() {
        // Cross-surface text convention (pgwire pins dashed; /v2/records
        // dashes) — the canonical wrapper must not drift from it.
        let u = ProximaValue::Uuid([
            0x55, 0x0e, 0x84, 0x00, 0xe2, 0x9b, 0x41, 0xd4, 0xa7, 0x16, 0x44, 0x65, 0x54, 0x40,
            0x00, 0x00,
        ]);
        assert_eq!(
            proxima_value_to_json_canonical(&u),
            serde_json::json!("550e8400-e29b-41d4-a716-446554400000")
        );
    }

    #[test]
    fn canonical_rendering_contract_is_pinned() {
        // TD-PROTO-2 consolidation: the ONE API-facing rendering — bytes are
        // base64, non-finite floats null, Jsonb decodes, unset oneof null.
        use proximadb_proto::proximadb_v1::sql_value::Value as V;
        let mk = |v: Option<V>| SqlValue { value: v };
        // All padding branches pinned with literals (a same-function loop
        // cannot fail).
        assert_eq!(
            sql_value_to_json(&mk(Some(V::BytesValue(vec![0, 1, 255])))),
            serde_json::json!("AAH/")
        );
        assert_eq!(
            sql_value_to_json(&mk(Some(V::BytesValue(vec![1, 2])))),
            serde_json::json!("AQI=")
        );
        assert_eq!(
            sql_value_to_json(&mk(Some(V::BytesValue(vec![7])))),
            serde_json::json!("Bw==")
        );
        assert_eq!(
            sql_value_to_json(&mk(Some(V::BytesValue(vec![])))),
            serde_json::json!("")
        );
        assert_eq!(
            sql_value_to_json(&mk(Some(V::BytesValue(vec![0, 1, 255])))),
            serde_json::json!("AAH/")
        );
        assert_eq!(
            sql_value_to_json(&mk(Some(V::NumberValue(f64::NAN)))),
            serde_json::json!(null)
        );
        assert_eq!(sql_value_to_json(&mk(None)), serde_json::json!(null));
        let doc = serde_json::json!({"a": 1});
        assert_eq!(
            sql_value_to_json(&mk(Some(V::JsonbValue(
                ProximaValue::to_jsonb_vec(&doc).unwrap()
            )))),
            doc
        );
    }

    #[test]
    fn jsonb_writes_the_declared_tag9_variant() {
        // TD-PROTO-2 review round 2: the writer must emit the declared
        // JsonbValue variant (tag 9) — encoding JSONB under BytesValue made
        // the canonical filter decode unreachable for production data.
        let doc = serde_json::json!({"memory": {"type": "fact"}});
        let sql = proxima_to_sql_value(&ProximaValue::Jsonb(doc.clone()));
        let Some(sql_value::Value::JsonbValue(bytes)) = &sql.value else {
            panic!(
                "Jsonb must serialize to the tag-9 variant, got {:?}",
                sql.value
            );
        };
        // Round-trips through the canonical decode-or-binary helper.
        assert_eq!(
            ProximaValue::from_jsonb_or_binary(bytes),
            ProximaValue::Jsonb(doc)
        );
    }

    #[test]
    fn proxima_to_sql_value_covers_lossy_legacy_adapter_shapes() {
        let cases = vec![
            (
                ProximaValue::Boolean(false),
                sql_value::Value::BoolValue(false),
            ),
            (ProximaValue::Int8(-8), sql_value::Value::Int64Value(-8)),
            (ProximaValue::Int16(-16), sql_value::Value::Int64Value(-16)),
            (ProximaValue::Int32(32), sql_value::Value::Int64Value(32)),
            (ProximaValue::Int64(64), sql_value::Value::Int64Value(64)),
            (ProximaValue::UInt8(8), sql_value::Value::Int64Value(8)),
            (ProximaValue::UInt16(16), sql_value::Value::Int64Value(16)),
            (ProximaValue::UInt32(32), sql_value::Value::Int64Value(32)),
            (
                ProximaValue::UInt64(u64::MAX),
                sql_value::Value::StringValue(u64::MAX.to_string()),
            ),
            (
                ProximaValue::Float32(1.25),
                sql_value::Value::NumberValue(1.25),
            ),
            (
                ProximaValue::Float64(2.5),
                sql_value::Value::NumberValue(2.5),
            ),
            (
                ProximaValue::Decimal("99.01".to_string()),
                sql_value::Value::StringValue("99.01".to_string()),
            ),
            (
                ProximaValue::Symbol("hot".to_string()),
                sql_value::Value::StringValue("hot".to_string()),
            ),
            (
                ProximaValue::Binary(vec![1, 2]),
                sql_value::Value::BytesValue(vec![1, 2]),
            ),
            (
                ProximaValue::BinaryVector(vec![0b1010]),
                sql_value::Value::BytesValue(vec![0b1010]),
            ),
            (
                ProximaValue::Date(20_000),
                sql_value::Value::Int64Value(20_000),
            ),
            (
                ProximaValue::Time(123, proximadb_data_model::TimeUnit::Microsecond),
                sql_value::Value::Int64Value(123),
            ),
            (
                ProximaValue::Timestamp(456, proximadb_data_model::TimeUnit::Microsecond),
                sql_value::Value::Int64Value(456),
            ),
            (
                ProximaValue::TimestampTz(789, proximadb_data_model::TimeUnit::Microsecond),
                sql_value::Value::Int64Value(789),
            ),
            (
                ProximaValue::Uuid([3; 16]),
                sql_value::Value::BytesValue(vec![3; 16]),
            ),
            (
                ProximaValue::ULID([4; 16]),
                sql_value::Value::BytesValue(vec![4; 16]),
            ),
            (
                ProximaValue::Json(serde_json::json!({"a": 1})),
                sql_value::Value::StringValue(r#"{"a":1}"#.to_string()),
            ),
            (ProximaValue::Null, sql_value::Value::NullValue(0)),
        ];

        for (value, expected) in cases {
            assert_eq!(proxima_to_sql_value(&value).value, Some(expected));
        }

        assert!(matches!(
            proxima_to_sql_value(&ProximaValue::Array(vec![
                ProximaValue::Int64(1),
                ProximaValue::String("x".to_string())
            ]))
            .value,
            Some(sql_value::Value::ArrayValue(SqlArray { values })) if values.len() == 2
        ));
        assert!(matches!(
            proxima_to_sql_value(&ProximaValue::Map(HashMap::from([(
                "k".to_string(),
                ProximaValue::Boolean(true)
            )])))
            .value,
            Some(sql_value::Value::ObjectValue(SqlObject { fields })) if fields.contains_key("k")
        ));
        assert!(matches!(
            proxima_to_sql_value(&ProximaValue::DenseVector(vec![0.1, 0.2])).value,
            Some(sql_value::Value::ArrayValue(SqlArray { values })) if values.len() == 2
        ));
        assert!(matches!(
            proxima_to_sql_value(&ProximaValue::SparseVector {
                indices: vec![1, 3],
                values: vec![0.5, 0.9],
            })
            .value,
            Some(sql_value::Value::ObjectValue(SqlObject { fields }))
                if fields.contains_key("indices") && fields.contains_key("values")
        ));
    }

    // --- PropertyValue → ProximaValue ---

    #[test]
    fn test_property_string_to_proxima() {
        let pv = make_prop_string("alice");
        assert!(matches!(property_value_to_proxima(&pv), ProximaValue::String(s) if s == "alice"));
    }

    #[test]
    fn test_property_int_to_proxima() {
        let pv = make_prop_int(99);
        assert!(matches!(
            property_value_to_proxima(&pv),
            ProximaValue::Int64(99)
        ));
    }

    #[test]
    fn property_value_conversion_covers_all_graph_property_shapes() {
        assert_eq!(
            property_value_to_proxima(&PropertyValue {
                value: Some(property_value::Value::DoubleValue(1.5)),
            }),
            ProximaValue::Float64(1.5)
        );
        assert_eq!(
            property_value_to_proxima(&PropertyValue {
                value: Some(property_value::Value::BoolValue(true)),
            }),
            ProximaValue::Boolean(true)
        );
        assert_eq!(
            property_value_to_proxima(&PropertyValue {
                value: Some(property_value::Value::BytesValue(vec![1, 2])),
            }),
            ProximaValue::Binary(vec![1, 2])
        );
        assert_eq!(
            property_value_to_proxima(&PropertyValue {
                value: Some(property_value::Value::ArrayValue(
                    proximadb_proto::proximadb_v1::PropertyArray {
                        values: vec![make_prop_string("x"), make_prop_int(3)],
                    },
                )),
            }),
            ProximaValue::Array(vec![
                ProximaValue::String("x".to_string()),
                ProximaValue::Int64(3),
            ])
        );
        assert_eq!(
            property_value_to_proxima(&PropertyValue {
                value: Some(property_value::Value::ObjectValue(
                    proximadb_proto::proximadb_v1::PropertyObject {
                        fields: HashMap::from([(
                            "flag".to_string(),
                            PropertyValue {
                                value: Some(property_value::Value::BoolValue(false)),
                            }
                        )]),
                    },
                )),
            }),
            ProximaValue::Map(HashMap::from([(
                "flag".to_string(),
                ProximaValue::Boolean(false),
            )]))
        );
        assert_eq!(
            property_value_to_proxima(&PropertyValue {
                value: Some(property_value::Value::VectorValue(
                    proximadb_proto::proximadb_v1::VectorData {
                        values: vec![0.1, 0.2],
                    },
                )),
            }),
            ProximaValue::DenseVector(vec![0.1, 0.2])
        );
        assert_eq!(
            property_value_to_proxima(&PropertyValue { value: None }),
            ProximaValue::Null
        );
    }

    #[test]
    fn proxima_tree_and_record_to_legacy_vector_preserve_nested_props_and_temporal_fields() {
        let nested = ProximaTree::from([(
            "city".to_string(),
            ProximaTreeNode::Value(ProximaValue::String("Chicago".to_string())),
        )]);
        let tree = ProximaTree::from([
            (
                "score".to_string(),
                ProximaTreeNode::Value(ProximaValue::Float64(0.9)),
            ),
            ("address".to_string(), ProximaTreeNode::Object(nested)),
        ]);

        let values = proxima_tree_to_value_map(&tree);
        assert!(matches!(values.get("address"), Some(ProximaValue::Map(_))));

        let metadata = proxima_tree_to_sql_metadata(&tree);
        assert!(matches!(
            metadata.get("address").and_then(|v| v.value.as_ref()),
            Some(sql_value::Value::ObjectValue(_))
        ));

        let mut record = ProximaRecord {
            oid: "rec-1".to_string(),
            props: tree,
            created_at_ns: 10_000_000,
            updated_at_ns: 20_000_000,
            valid_to_ns: Some(30_000_000),
            record_version: 7,
            origin: Some("canonical".to_string()),
            ..ProximaRecord::default()
        };
        record
            .embeddings
            .push(EmbeddingCell::new_fp32("model", "text", 2, vec![0.1, 0.2]));

        let vector = proxima_record_to_vector(&record);
        assert_eq!(vector.id, "rec-1");
        assert_eq!(vector.vector, vec![0.1, 0.2]);
        assert_eq!(vector.timestamp, Some(10));
        assert_eq!(vector.updated_at, Some(20));
        assert_eq!(vector.expires_at, Some(30));
        assert_eq!(vector.version, Some(7));
        assert_eq!(vector.source.as_deref(), Some("canonical"));

        let vector_from_ref = VectorRecord::from(&record);
        let vector_from_owned = VectorRecord::from(record);
        assert_eq!(vector_from_ref.id, vector_from_owned.id);
    }

    #[test]
    fn embedding_version_and_node_embedding_lower_into_canonical_envelope() {
        let embedding = EmbeddingVersion {
            model_id: "e5".to_string(),
            model_version: "1".to_string(),
            vector: vec![0.4, 0.5],
            dimension: 2,
            created_at_ms: 100,
            model_params: HashMap::new(),
            modality: proximadb_proto::proximadb_v1::Modality::Image as i32,
        };

        let cell = embedding_version_to_cell(&embedding);
        assert_eq!(cell.model_id, "e5");
        assert_eq!(cell.modality, "1");
        assert_eq!(cell.dim, 2);
        assert_eq!(cell.as_fp32_slice(), &[0.4, 0.5]);

        let node = Node {
            id: "node-img".to_string(),
            labels: vec!["Asset".to_string()],
            properties: HashMap::from([("kind".to_string(), make_prop_string("image"))]),
            embedding: Some(embedding),
            created_at_ms: 5,
            updated_at_ms: 6,
        };
        let record = ProximaRecord::from(&node);
        assert_eq!(record.embeddings.len(), 1);
        assert_eq!(record.embeddings[0].model_id, "e5");
        assert!(matches!(
            record.refs.as_slice(),
            [TypedRef::GraphEdge {
                edge_id,
                direction: EdgeDirection::Outgoing,
            }] if edge_id == "node-img"
        ));
        assert_eq!(record.updated_at_ns, ms_to_ns(6));
    }

    #[test]
    fn vector_record_owned_conversion_and_expiry_fields_are_preserved() {
        let record = VectorRecord {
            id: "vec-expiring".to_string(),
            vector: vec![0.7],
            metadata: HashMap::new(),
            timestamp: Some(11),
            updated_at: Some(12),
            expires_at: Some(13),
            version: None,
            source: None,
        };

        let canonical = ProximaRecord::from(record);
        assert_eq!(canonical.oid, "vec-expiring");
        assert_eq!(canonical.created_at_ns, ms_to_ns(11));
        assert_eq!(canonical.updated_at_ns, ms_to_ns(12));
        assert_eq!(canonical.valid_to_ns, Some(ms_to_ns(13)));
        assert_eq!(canonical.record_version, 0);
        assert_eq!(canonical.method.as_deref(), Some("vector_insert"));
    }

    // --- VectorRecord → ProximaRecord ---

    #[test]
    fn test_vector_record_to_proxima() {
        let mut meta = HashMap::new();
        meta.insert("category".to_string(), make_sql_string("tech"));
        meta.insert("score".to_string(), make_sql_float(0.95));

        let vr = VectorRecord {
            id: "vec_001".to_string(),
            vector: vec![0.1, 0.2, 0.3],
            metadata: meta,
            timestamp: Some(1_704_067_200_000), // 2024-01-01 ms
            updated_at: None,
            expires_at: None,
            version: Some(2),
            source: Some("test_source".to_string()),
        };

        let pr = ProximaRecord::from(&vr);

        assert_eq!(pr.oid, "vec_001");
        assert_eq!(pr.record_version, 2);
        assert_eq!(pr.origin.as_deref(), Some("test_source"));

        // Embedding cell from vector
        assert_eq!(pr.embeddings.len(), 1);
        assert_eq!(pr.embeddings[0].dim, 3);
        assert_eq!(pr.embeddings[0].modality, "dense_vector");

        // Props from metadata
        assert!(pr.props.contains_key("category"));
        assert!(pr.props.contains_key("score"));

        // Temporal
        assert!(pr.created_at_ns > 0);
        assert!(pr.valid_to_ns.is_none());
    }

    #[test]
    fn test_vector_record_empty_vector() {
        let vr = VectorRecord {
            id: "empty_vec".to_string(),
            vector: vec![],
            metadata: HashMap::new(),
            timestamp: None,
            updated_at: None,
            expires_at: None,
            version: None,
            source: None,
        };

        let pr = ProximaRecord::from(&vr);
        assert!(pr.embeddings.is_empty(), "empty vector → no embedding cell");
    }

    // --- Node → ProximaRecord ---

    #[test]
    fn test_node_to_proxima() {
        use proximadb_proto::proximadb_v1::Node;
        let mut props = HashMap::new();
        props.insert("name".to_string(), make_prop_string("Alice"));
        props.insert("age".to_string(), make_prop_int(30));

        let node = Node {
            id: "node_alice".to_string(),
            labels: vec!["Person".to_string(), "Employee".to_string()],
            properties: props,
            embedding: None,
            created_at_ms: 1_000_000,
            updated_at_ms: 2_000_000,
        };

        let pr = ProximaRecord::from(&node);

        assert_eq!(pr.oid, "node_alice");
        assert!(pr.labels.contains("Person"));
        assert!(pr.labels.contains("Employee"));
        assert!(pr.props.contains_key("name"));
        assert!(pr.props.contains_key("age"));
        assert!(pr.edge.is_none(), "nodes have no edge topology");
        assert_eq!(pr.created_at_ns, ms_to_ns(1_000_000));
    }

    // --- Edge → ProximaRecord ---

    #[test]
    fn test_edge_to_proxima() {
        use proximadb_proto::proximadb_v1::Edge;
        let edge = Edge {
            id: "edge_001".to_string(),
            from_node_id: "node_a".to_string(),
            to_node_id: "node_b".to_string(),
            edge_type: "KNOWS".to_string(),
            properties: HashMap::new(),
            weight: Some(0.75),
            created_at_ms: 1_000_000,
            updated_at_ms: 1_000_000,
        };

        let pr = ProximaRecord::from(&edge);

        assert_eq!(pr.oid, "edge_001");

        let es = pr.edge.as_ref().expect("edge record must have EdgeShape");
        assert_eq!(es.source_id, "node_a");
        assert_eq!(es.target_id, "node_b");
        assert_eq!(es.edge_type, "KNOWS");
        assert!((es.weight.unwrap() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_edge_without_weight() {
        use proximadb_proto::proximadb_v1::Edge;
        let edge = Edge {
            id: "edge_002".to_string(),
            from_node_id: "a".to_string(),
            to_node_id: "b".to_string(),
            edge_type: "LINKED".to_string(),
            properties: HashMap::new(),
            weight: None,
            created_at_ms: 0,
            updated_at_ms: 0,
        };

        let pr = ProximaRecord::from(&edge);
        assert!(pr.edge.as_ref().unwrap().weight.is_none());
    }
}
