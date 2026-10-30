//! Shared plumbing for the per-data-model `AqlSource` implementations:
//! the proto-SQL-value converter (one policy per crate — every source spells
//! bytes/JSONB decoding identically so a cross-source predicate sees the
//! same value everywhere) and the completed-execution audit-frame tail.

pub mod document;
pub mod graph;
pub mod memory;
pub mod observability;
pub mod vector;

use std::collections::HashMap;

use crate::proto::proximadb_v1::sql_value::Value as SqlValueData;
use crate::query::aql::{AqlResult, AqlValue, AuditContext, AuditFrame, AuditOp, DataModel};

/// Convert a proto SQL value to an [`AqlValue`].
pub(super) fn sql_value_to_aql(val: &SqlValueData) -> AqlValue {
    match val {
        SqlValueData::StringValue(s) => AqlValue::String(s.clone()),
        SqlValueData::Int64Value(i) => AqlValue::Int(*i),
        SqlValueData::NumberValue(f) => AqlValue::Float(*f),
        SqlValueData::BoolValue(b) => AqlValue::Bool(*b),
        SqlValueData::BytesValue(b) => {
            // Generic bytes may carry raw JSON from legacy producers — the
            // heuristic stays; JSONB-declared fields (tag 9) decode through
            // jsonb_to_json_lossy below.
            if let Ok(json) = serde_json::from_slice(b) {
                AqlValue::Jsonb(json)
            } else {
                AqlValue::Null
            }
        }
        // types.proto tag 9: JSONB by declaration — same JSON heuristic as
        // BytesValue so a JSONB field is not silently nulled by the `_`
        // wildcard.
        SqlValueData::JsonbValue(b) => {
            // TD-PROTO-2: canonical MessagePack JSONB decode (the one
            // sanctioned JSONB->JSON representation) — not a raw-JSON
            // heuristic that diverges from the API-facing surfaces.
            AqlValue::Jsonb(proximadb_data_model::ProximaValue::jsonb_to_json_lossy(b))
        }
        SqlValueData::ObjectValue(obj) => {
            let mut map = serde_json::Map::new();
            for (k, v) in &obj.fields {
                if let Some(inner_val) = &v.value {
                    map.insert(k.clone(), aql_to_json_value(&sql_value_to_aql(inner_val)));
                }
            }
            AqlValue::Jsonb(serde_json::Value::Object(map))
        }
        SqlValueData::ArrayValue(arr) => {
            let values = arr
                .values
                .iter()
                .map(|value| {
                    value
                        .value
                        .as_ref()
                        .map(sql_value_to_aql)
                        .map(|value| aql_to_json_value(&value))
                        .unwrap_or(serde_json::Value::Null)
                })
                .collect();
            AqlValue::Jsonb(serde_json::Value::Array(values))
        }
        _ => AqlValue::Null,
    }
}

/// Convert an [`AqlValue`] to plain JSON (nested object/array payloads).
pub(super) fn aql_to_json_value(aql: &AqlValue) -> serde_json::Value {
    match aql {
        AqlValue::String(s) => serde_json::Value::String(s.clone()),
        AqlValue::Int(i) => serde_json::json!(i),
        AqlValue::Float(f) => serde_json::json!(f),
        AqlValue::Bool(b) => serde_json::Value::Bool(*b),
        AqlValue::Json(j) | AqlValue::Jsonb(j) => j.clone(),
        _ => serde_json::Value::Null,
    }
}

/// Emit the completed-execution audit frame and wrap the rows into the
/// [`AqlResult`]: scanned == returned, no pushed/post filters (none of the
/// document/graph/observability sources push predicates yet).
pub(super) fn audited_ok(
    ctx: &mut AuditContext,
    model: DataModel,
    op: AuditOp,
    rows: Vec<HashMap<String, AqlValue>>,
    wall_time_us: u64,
) -> AqlResult {
    let frame = AuditFrame {
        frame_id: 0, // Set by ctx.push_frame
        source: model,
        op,
        filters_pushed: Vec::new(), // TODO: Serialize predicates
        filters_post: Vec::new(),
        records_scanned: rows.len() as u64,
        records_returned: rows.len() as u64,
        wall_time_us,
        error: None,
        redaction_count: 0,
    };
    let frame_id = ctx.push_frame(frame);
    AqlResult { rows, frame_id }
}
