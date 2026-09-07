//! Type-safe metadata filtering for SqlValue-based metadata
//!
//! This module provides efficient, type-preserving filter evaluation for ProximaDB's
//! SqlValue-based metadata system. Unlike JSON-based filtering, this approach:
//!
//! - **Preserves type precision**: No lossy conversions (e.g., integers stay integers)
//! - **Faster execution**: Zero conversion overhead, direct type comparisons
//! - **Type-safe**: Leverages Rust's type system for correctness
//!
//! ## Usage
//!
//! ```rust,ignore
//! use proximadb::core::search::sql_value_filter::evaluate_filter;
//! use proximadb::core::search::FilterExpression;
//!
//! let metadata: HashMap<String, SqlValue> = /* ... */;
//! let filter = FilterExpression::Comparison {
//!     field: "category".to_string(),
//!     operator: ComparisonOperator::Equals,
//!     value: serde_json::Value::String("A".to_string()),
//! };
//!
//! if evaluate_filter(&filter, &metadata) {
//!     // Record matches filter
//! }
//! ```
//!
//! ## Implementation Note
//!
//! The filter value comes from the query (as serde_json::Value), while the metadata
//! is stored as SqlValue. We compare them type-safely without lossy conversions.

use std::borrow::Cow;
use std::collections::HashMap;

use proximadb_data_model::ProximaValue;
use proximadb_filter_expression::{ComparisonOperator, FilterExpression};
use proximadb_proto::proximadb_v1::SqlValue;
use proximadb_proto::proximadb_v1::sql_value::Value as SqlVal;
use proximadb_records::{ProximaTree, ProximaTreeNode};

/// Fail-loud error from the strict v2 metadata-filter evaluators (TD-FILT-1). The legacy
/// [`evaluate_filter_resolved`] silently folds an unresolved field into `false` (dropping the row);
/// the strict variants surface it so a caller can reject the result instead of accepting a silently
/// wrong set. v2-canonical: this lives on the `ProximaValue`/`ProximaTree` seam (ADR-024), not the
/// deprecated v1 `MetadataValue` path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterEvalError {
    /// A value comparison referenced a field the resolver could not lower to a comparable value —
    /// either absent from the record, or a non-scalar leaf. (`IsNull`/`IsNotNull` are NOT errors:
    /// field absence is their domain, per SQL semantics.)
    MissingField { field: String },
}

impl std::fmt::Display for FilterEvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FilterEvalError::MissingField { field } => write!(
                f,
                "filter field {field:?} could not be resolved to a comparable value (absent or non-scalar leaf)"
            ),
        }
    }
}

impl std::error::Error for FilterEvalError {}

/// Evaluate a filter expression against proto `SqlValue` (wire-format) metadata.
///
/// Thin adapter over the canonical [`evaluate_filter_resolved`] seam: each field's
/// `SqlValue` is lowered to `serde_json::Value` via [`sql_val_to_json`] (numbers
/// stay numeric so integer precision is preserved by `compare_json_numbers`), so
/// this path supports the full operator set with semantics identical to
/// [`evaluate_filter_proxima`]. Previously this evaluator handled only
/// equality + numeric ordering; it now matches the canonical behavior.
///
/// * `expr` - The filter expression to evaluate
/// * `metadata` - The record's metadata as a proto `SqlValue` map
pub fn evaluate_filter(expr: &FilterExpression, metadata: &HashMap<String, SqlValue>) -> bool {
    evaluate_filter_resolved(expr, &|field| {
        if let Some(sql_value) = metadata.get(field) {
            // Round 12: a top-level unset oneof is the SAME wire form of null
            // as the nested one the round-11 fix handles — `a = null` must
            // match, not drop the row.
            return Some(sql_value_to_filter_literal(sql_value));
        }
        // Dot traversal for JSON(B) roots — mirrors `resolve_proxima_value`
        // so the SqlValue and ProximaTree paths agree for nested fields:
        // `memory.type` resolves inside JsonbValue({"memory":{"type":…}}).
        // Two phases (round 8: the three-cursor form enforced its own
        // invariants by hand): walk SqlObject segments by reference until the
        // path ends or a JSONB leaf appears, then (re)use the shared JSON
        // walker for the remainder. Only the final leaf is cloned.
        let (head, tail) = field.split_once('.')?;
        let mut segments = tail.split('.');
        // A null HEAD (unset oneof at depth 1) is untraversable — no object
        // behind it, so a dotted path dead-ends: `a.b` over {"a": null}
        // resolves to nothing (consistent with its IS NULL answer).
        let mut current = metadata.get(head)?.value.as_ref()?;
        for segment in segments.by_ref() {
            match current {
                SqlVal::ObjectValue(obj) => match obj.fields.get(segment) {
                    Some(child) => {
                        static NULL_SENTINEL: SqlVal = SqlVal::NullValue(0);
                        current = child.value.as_ref().unwrap_or(&NULL_SENTINEL);
                    }
                    None => return None,
                },
                // A NullValue sentinel mid-path is terminal (the wildcard
                // below returns None); as the final leaf the post-loop
                // lowering renders it as JSON null.
                // Canonical JSONB decodes once and continues as JSON; legacy
                // tag-8 bytes stay opaque (pre-PR behavior).
                SqlVal::JsonbValue(bytes) => {
                    let root = ProximaValue::jsonb_to_json_lossy(bytes);
                    // `segment` was consumed from the iterator but not yet
                    // applied — it leads the JSON-side path.
                    return json_get_path(&root, std::iter::once(segment).chain(segments)).cloned();
                }
                // A non-object mid-path cannot be traversed further —
                // `segment` was consumed as a lookup key and found nothing
                // object-shaped behind it. Returning the PARENT's value here
                // (round-10 finding) admitted rows on `user.name = <user's
                // own value>` and flipped IS NULL semantics; the legitimate
                // leaf case is the post-loop match.
                _ => return None,
            }
        }
        // Path ended on a node: lower the leaf (sql_val_to_json's
        // ObjectValue arm already maps unset-oneof children to null —
        // obj_to_json was a duplicate of it).
        Some(sql_val_to_json(current))
    })
}

/// Walk dotted path segments through JSON objects by reference.
///
/// This object-only behavior is shared by query filtering and OIDC claim
/// resolution. Array indexes are deliberately not interpreted as path
/// segments; changing that contract requires review of the authorization
/// callers as well as the query callers.
pub fn json_get_path<'a, 'p>(
    root: &'a serde_json::Value,
    segments: impl Iterator<Item = &'p str>,
) -> Option<&'a serde_json::Value> {
    let mut current = root;
    for segment in segments {
        current = match current {
            serde_json::Value::Object(object) => object.get(segment)?,
            _ => return None,
        };
    }
    Some(current)
}

/// Lower a whole `SqlValue` (including its unset-oneof null wire form) to
/// JSON. Single authority for "unset oneof == JSON null" — rounds 11/12 were
/// each this rule drifting by depth; do not re-derive it per call site.
pub fn sql_value_to_filter_literal(sql: &SqlValue) -> serde_json::Value {
    match sql.value.as_ref() {
        Some(inner) => sql_val_to_json(inner),
        None => serde_json::Value::Null,
    }
}

/// Lower a proto `SqlValue` payload to `serde_json::Value` so the wire/`SqlValue`
/// metadata path shares the canonical operator semantics (the seam compares on
/// `serde_json::Value`). Numbers stay numeric (so `compare_json_numbers` keeps
/// integer precision); bytes become a JSON array of byte values.
fn sql_val_to_json(value: &SqlVal) -> serde_json::Value {
    match value {
        SqlVal::StringValue(s) => serde_json::Value::String(s.clone()),
        SqlVal::NumberValue(n) => serde_json::Number::from_f64(*n)
            .map_or(serde_json::Value::Null, serde_json::Value::Number),
        SqlVal::BoolValue(b) => serde_json::Value::Bool(*b),
        SqlVal::Int64Value(i) => serde_json::Value::Number((*i).into()),
        SqlVal::BytesValue(bytes) => serde_json::Value::Array(
            bytes
                .iter()
                .map(|byte| serde_json::Value::Number((*byte).into()))
                .collect(),
        ),
        // JSONB decodes to the JSON document itself so structured filters
        // (`$.field op literal`) can actually match — a byte-array rendering
        // would make every comparison against a JSONB column miss.
        SqlVal::JsonbValue(bytes) => ProximaValue::jsonb_to_json_lossy(bytes),
        SqlVal::NullValue(_) => serde_json::Value::Null,
        SqlVal::ArrayValue(array) => serde_json::Value::Array(
            array
                .values
                .iter()
                .map(|v| {
                    v.value
                        .as_ref()
                        .map_or(serde_json::Value::Null, sql_val_to_json)
                })
                .collect(),
        ),
        SqlVal::ObjectValue(object) => serde_json::Value::Object(
            object
                .fields
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        v.value
                            .as_ref()
                            .map_or(serde_json::Value::Null, sql_val_to_json),
                    )
                })
                .collect(),
        ),
    }
}

/// Convert a `ProximaValue` leaf to a `serde_json::Value` for filter evaluation.
pub fn proxima_value_to_filter_literal(pv: &ProximaValue) -> serde_json::Value {
    match pv {
        ProximaValue::Null => serde_json::Value::Null,
        ProximaValue::Boolean(b) => serde_json::Value::Bool(*b),
        ProximaValue::Int8(n) => serde_json::Value::Number((*n as i64).into()),
        ProximaValue::Int16(n) => serde_json::Value::Number((*n as i64).into()),
        ProximaValue::Int32(n) => serde_json::Value::Number((*n as i64).into()),
        ProximaValue::Int64(n) => serde_json::Value::Number((*n).into()),
        ProximaValue::UInt8(n) => serde_json::Value::Number((*n as u64).into()),
        ProximaValue::UInt16(n) => serde_json::Value::Number((*n as u64).into()),
        ProximaValue::UInt32(n) => serde_json::Value::Number((*n as u64).into()),
        ProximaValue::UInt64(n) => serde_json::Value::Number((*n).into()),
        ProximaValue::Float16(f) | ProximaValue::Float32(f) => {
            serde_json::Number::from_f64(*f as f64)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null)
        }
        ProximaValue::Float64(f) => serde_json::Number::from_f64(*f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        ProximaValue::String(s) | ProximaValue::Symbol(s) | ProximaValue::Decimal(s) => {
            serde_json::Value::String(s.clone())
        }
        ProximaValue::Json(v) => v.clone(),
        ProximaValue::Jsonb(v) => v.clone(),
        ProximaValue::Array(items) => {
            serde_json::Value::Array(items.iter().map(proxima_value_to_filter_literal).collect())
        }
        ProximaValue::Map(map) | ProximaValue::Struct(map) => {
            let obj: serde_json::Map<String, serde_json::Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), proxima_value_to_filter_literal(v)))
                .collect();
            serde_json::Value::Object(obj)
        }
        // Temporal types — represent as milliseconds integer
        ProximaValue::Timestamp(ns, _) | ProximaValue::TimestampTz(ns, _) => {
            serde_json::Value::Number((*ns).into())
        }
        ProximaValue::Date(d) => serde_json::Value::Number((*d as i64).into()),
        ProximaValue::Time(t, _) => serde_json::Value::Number((*t).into()),
        // UUID/ULID — string representation. SEAM NOTE: this dashed spelling
        // holds on the ProximaTree seam only; when a Uuid crosses the v1
        // `SqlValue` bridge (`proxima_to_sql_value` → BytesValue) the type is
        // lost and the SqlValue seam renders raw bytes (int-array), so the
        // same stored uuid needs a different literal per seam until the bridge
        // is type-preserving (tracked in TD-PROTO-2).
        ProximaValue::Uuid(b) => serde_json::Value::String(
            proximadb_kernel::uuid::Uuid::from_bytes(*b).to_hyphenated_string(),
        ),
        ProximaValue::ULID(b) => {
            // Single-pass hex (16 bytes → 32 chars): the per-byte format!
            // form allocated one intermediate String per byte on the
            // per-record render path.
            use std::fmt::Write as _;
            let mut hex = String::with_capacity(b.len() * 2);
            for byte in b {
                let _ = write!(hex, "{byte:02x}");
            }
            serde_json::Value::String(hex)
        }
        // Binary — the int-array spelling the SQL-side filter literal lowers
        // to — the old '[binary:N]' placeholder could never equal any literal,
        // so bytes equality filters silently matched nothing (round 5). The
        // render allocates one serde_json::Value per byte per record on scans;
        // an allocation-free native bytes comparison needs the resolved-field
        // contract to carry bytes natively (tracked in TD-PROTO-2). The SAME
        // tracked item covers bytes op-semantics: rendered as an array, bytes
        // flow into compare_json_op's array arms, so IN/NOT-IN/CONTAINS take
        // element-membership semantics (byte-level) that can contradict
        // Equals — op-aware bytes routing needs the same contract change.
        ProximaValue::Binary(b) | ProximaValue::BinaryVector(b) => serde_json::Value::Array(
            b.iter()
                .map(|x| serde_json::Value::Number((*x as u64).into()))
                .collect(),
        ),
        ProximaValue::DenseVector(v) => serde_json::Value::Array(
            v.iter()
                .map(|f| {
                    serde_json::Number::from_f64(*f as f64)
                        .map(serde_json::Value::Number)
                        .unwrap_or(serde_json::Value::Null)
                })
                .collect(),
        ),
        // Canonical {indices, values} object — the same shape records'
        // proxima_to_json renders (and the read path returns), so an object
        // literal CAN match. The old "[sparse_vector]" placeholder never
        // equaled any literal, and a string literal of the placeholder itself
        // degenerately matched every sparse value.
        ProximaValue::SparseVector { indices, values } => serde_json::json!({
            "indices": indices,
            "values": values,
        }),
    }
}

/// Flatten a `ProximaTree` to a `HashMap<String, serde_json::Value>` for filter evaluation.
///
/// Nested `Object` nodes are serialised as JSON objects. This matches the shape
/// expected by `json_comparison::evaluate_filter` and `MetadataQueryEngine::evaluate`.
pub fn proxima_tree_to_json_map(props: &ProximaTree) -> HashMap<String, serde_json::Value> {
    fn node_to_json(node: &ProximaTreeNode) -> serde_json::Value {
        match node {
            ProximaTreeNode::Value(pv) => proxima_value_to_filter_literal(pv),
            ProximaTreeNode::Object(subtree) => serde_json::Value::Object(
                subtree
                    .iter()
                    .map(|(k, n)| (k.clone(), node_to_json(n)))
                    .collect(),
            ),
        }
    }

    props
        .iter()
        .map(|(key, node)| (key.clone(), node_to_json(node)))
        .collect()
}

/// Serialise `props` to a canonical JSON string: object keys sorted at every level.
///
/// Key order must be canonical here, not incidental: `serde_json::Map` is
/// order-preserving whenever some dependency enables the `preserve_order`
/// feature (datafusion-physical-plan 55+ does, via `serde_json/preserve_order`)
/// and a plain `BTreeMap` otherwise, so insertion-order serialisation changes
/// byte-for-byte with the dependency graph. Sorting explicitly pins the
/// serialised form — the same bytes `BTreeMap` semantics produced before.
pub fn proxima_tree_to_canonical_json_string(props: &ProximaTree) -> String {
    fn node_to_canonical_json(node: &ProximaTreeNode) -> serde_json::Value {
        match node {
            ProximaTreeNode::Value(pv) => proxima_value_to_filter_literal(pv),
            ProximaTreeNode::Object(subtree) => {
                let mut entries: Vec<(String, serde_json::Value)> = subtree
                    .iter()
                    .map(|(k, n)| (k.clone(), node_to_canonical_json(n)))
                    .collect();
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                serde_json::Value::Object(entries.into_iter().collect())
            }
        }
    }

    let mut entries: Vec<(String, serde_json::Value)> = props
        .iter()
        .map(|(key, node)| (key.clone(), node_to_canonical_json(node)))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    serde_json::to_string(&serde_json::Value::Object(entries.into_iter().collect()))
        .unwrap_or_default()
}

/// Flatten a `ProximaTree` into the canonical `OptimizedSearchRecord` metadata map.
///
/// Nested objects are preserved as `ProximaValue::Struct` so storage engines can return
/// canonical metadata without detouring through the deprecated v1 `SqlValue` envelope.
pub fn proxima_tree_to_value_map(props: &ProximaTree) -> HashMap<String, ProximaValue> {
    fn node_to_value(node: &ProximaTreeNode) -> ProximaValue {
        match node {
            ProximaTreeNode::Value(value) => value.clone(),
            ProximaTreeNode::Object(subtree) => ProximaValue::Struct(
                subtree
                    .iter()
                    .map(|(key, child)| (key.clone(), node_to_value(child)))
                    .collect(),
            ),
        }
    }

    props
        .iter()
        .map(|(key, node)| (key.clone(), node_to_value(node)))
        .collect()
}

/// Apply a single comparison operator to a field value already lowered to
/// `serde_json::Value` and the filter literal.
///
/// This is the **single source of operator semantics** for every
/// `FilterExpression` evaluator in the crate. Concrete evaluators differ only in
/// how they resolve a field name to a `serde_json::Value`; once resolved, they
/// must all funnel through here so a given `(operator, value, literal)` triple
/// behaves identically regardless of the storage representation it came from.
pub fn compare_json_op(
    operator: &ComparisonOperator,
    json_val: &serde_json::Value,
    value: &serde_json::Value,
) -> bool {
    // Null tests are total — decided once, here (round 17: they were
    // previously decided in these guards AND again in the main match, three
    // encodings of one rule).
    match operator {
        ComparisonOperator::IsNull => return json_val.is_null(),
        ComparisonOperator::IsNotNull => return !json_val.is_null(),
        _ => {}
    }
    // Null LITERAL semantics: `= null` matches a null field (pinned);
    // `!= null` / NOT IN(null) are the Mongo-style exclude-nulls idiom
    // (admit non-null rows — the live REST neq path); In(null) is false;
    // Contains null matches arrays holding a null element (develop's
    // structural semantics). Everything else with a null literal: false.
    if value.is_null() {
        return match operator {
            ComparisonOperator::Equals => json_val.is_null(),
            ComparisonOperator::NotEquals | ComparisonOperator::NotIn => !json_val.is_null(),
            ComparisonOperator::In => false,
            ComparisonOperator::Contains => {
                matches!(json_val, serde_json::Value::Array(items) if items.iter().any(|i| i.is_null()))
            }
            _ => false,
        };
    }
    // Null FIELD with a non-null literal: no value comparison matches.
    if json_val.is_null() {
        return false;
    }
    match operator {
        // Unreachable — null tests return above; kept for exhaustiveness so
        // a future operator fails compilation here rather than silently
        // falling through.
        ComparisonOperator::IsNull | ComparisonOperator::IsNotNull => false,
        ComparisonOperator::Equals => json_eq(json_val, value),
        ComparisonOperator::NotEquals => !json_eq(json_val, value),
        ComparisonOperator::LessThan => compare_json_lt(json_val, value),
        ComparisonOperator::LessThanOrEqual => compare_json_lte(json_val, value),
        ComparisonOperator::GreaterThan => compare_json_gt(json_val, value),
        ComparisonOperator::GreaterThanOrEqual => compare_json_gte(json_val, value),
        ComparisonOperator::In => match json_val {
            // Array-valued prop (e.g. `member_oids`): match when
            // the prop set intersects the query list.
            serde_json::Value::Array(items) => value.as_array().is_some_and(|values| {
                items
                    .iter()
                    .any(|item| values.iter().any(|v| json_eq(item, v)))
            }),
            // Scalar prop: membership in the query list.
            _ => value
                .as_array()
                .is_some_and(|values| values.iter().any(|v| json_eq(json_val, v))),
        },
        ComparisonOperator::NotIn => {
            // SQL: a null element in the list makes NOT IN UNKNOWN ⇒ deny
            // (round 15; the security walker delegates here too).
            let list_has_null = value
                .as_array()
                .is_some_and(|vs| vs.iter().any(|v| v.is_null()));
            if list_has_null {
                false
            } else {
                match json_val {
                    // Array-valued prop: pass when the prop set is
                    // disjoint from the query list.
                    serde_json::Value::Array(items) => value.as_array().is_none_or(|values| {
                        !items
                            .iter()
                            .any(|item| values.iter().any(|v| json_eq(item, v)))
                    }),
                    // Scalar prop: excluded when it equals any list element.
                    _ => value
                        .as_array()
                        .is_none_or(|values| values.iter().all(|v| !json_eq(json_val, v))),
                }
            }
        }
        ComparisonOperator::Contains => match json_val {
            // Array-valued prop: element membership
            // (e.g. `member_oids` contains `"u1"`).
            serde_json::Value::Array(items) => items.iter().any(|item| json_eq(item, value)),
            // Scalar string prop: substring match.
            _ => json_val
                .as_str()
                .zip(value.as_str())
                .is_some_and(|(haystack, needle)| haystack.contains(needle)),
        },
        ComparisonOperator::StartsWith => json_val
            .as_str()
            .zip(value.as_str())
            .is_some_and(|(haystack, prefix)| haystack.starts_with(prefix)),
        ComparisonOperator::EndsWith => json_val
            .as_str()
            .zip(value.as_str())
            .is_some_and(|(haystack, suffix)| haystack.ends_with(suffix)),
        ComparisonOperator::Between => value.as_array().is_some_and(|bounds| {
            // (A null FIELD never reaches here — the centralized guards
            // above return first; the bounds checks are the live ones.)
            bounds.len() == 2
                && !bounds[0].is_null()
                && !bounds[1].is_null()
                && compare_json_gte(json_val, &bounds[0])
                && compare_json_lte(json_val, &bounds[1])
        }),
        // Null tests on a value that the resolver already produced: present and
        // JSON-null ⇒ null. Field *absence* is handled in `evaluate_filter_resolved`
        // (absent ⇒ IS NULL), which is the only place that can observe absence.
        // Full SQL LIKE: `%` = any run, `_` = exactly one char, anywhere in the
        // pattern. Shared with every evaluator via `json_comparison`.
        ComparisonOperator::Like => {
            json_val
                .as_str()
                .zip(value.as_str())
                .is_some_and(|(haystack, pattern)| {
                    crate::json_comparison::like_pattern_match(haystack, pattern)
                })
        }
    }
}

/// Walk a `FilterExpression`, resolving each comparison's field to a
/// `serde_json::Value` via `resolve`, and apply [`compare_json_op`].
///
/// `resolve` returns `None` when the field is absent or is not a scalar leaf the
/// evaluator handles. For every operator except the null tests, an unresolved
/// field makes the comparison `false`. The null tests follow SQL semantics on
/// absence: a *missing* field IS NULL (`IsNull` ⇒ `true`, `IsNotNull` ⇒ `false`),
/// distinct from a field that is present-and-null (also null).
///
/// This is the shared spine; concrete evaluators are thin adapters that supply a
/// representation-specific `resolve` (see [`evaluate_filter_proxima`]).
pub fn evaluate_filter_resolved<F>(expr: &FilterExpression, resolve: &F) -> bool
where
    F: Fn(&str) -> Option<serde_json::Value>,
{
    match expr {
        FilterExpression::And(exprs) => exprs.iter().all(|e| evaluate_filter_resolved(e, resolve)),
        FilterExpression::Or(exprs) => exprs.iter().any(|e| evaluate_filter_resolved(e, resolve)),
        FilterExpression::Not(e) => {
            // SQL 3VL: NOT(UNKNOWN) = UNKNOWN ⇒ deny. The two-valued flip
            // would ADMIT null/absent-field rows whose inner comparison
            // denied (rounds 18-19 vs develop). Nullness is probed only when
            // the inner comparison DENIED (single resolution in the common
            // match case — the eager probe double-resolved every record).
            let inner = evaluate_filter_resolved(e, resolve);
            if !inner
                && let FilterExpression::Comparison {
                    field, operator, ..
                } = &**e
                && !matches!(
                    operator,
                    ComparisonOperator::IsNull | ComparisonOperator::IsNotNull
                )
                && resolve(field).is_none_or(|v| v.is_null())
            {
                false
            } else {
                !inner
            }
        }
        FilterExpression::Comparison {
            field,
            operator,
            value,
        } => match operator {
            // Null tests are the only operators whose result depends on field
            // *absence*, so they are decided here (where the resolver's `Option`
            // distinguishes absent from present). SQL semantics: an absent field
            // IS NULL. `compare_json_op` cannot see absence, so it must never be
            // reached for these.
            ComparisonOperator::IsNull => resolve(field).is_none_or(|json_val| json_val.is_null()),
            ComparisonOperator::IsNotNull => {
                resolve(field).is_some_and(|json_val| !json_val.is_null())
            }
            _ => match resolve(field) {
                Some(json_val) => compare_json_op(operator, &json_val, value),
                None => false,
            },
        },
    }
}

/// Fail-loud variant of [`evaluate_filter_resolved`] (TD-FILT-1, v2-canonical). Returns
/// `Err(MissingField)` for a value comparison whose field the resolver cannot lower (absent or a
/// non-scalar leaf), instead of silently returning `false`. `IsNull`/`IsNotNull` keep SQL absence
/// semantics (`Ok`, never an error). The comparison layer is unchanged (it total-orders by JSON
/// type precedence, so there is no per-comparison type mismatch to surface on this path).
///
/// Callers wanting ADR-043 fail-loud semantics migrate here; the legacy `evaluate_filter*` stay the
/// silent default until call-sites are migrated slice-by-slice (default-off / behavior-preserving).
pub fn evaluate_filter_resolved_strict<F>(
    expr: &FilterExpression,
    resolve: &F,
) -> Result<bool, FilterEvalError>
where
    F: Fn(&str) -> Option<serde_json::Value>,
{
    match expr {
        FilterExpression::And(exprs) => {
            // Non-short-circuit: evaluate every child so a MissingField in a later arm surfaces
            // (fail-loud) rather than being hidden by an earlier Ok(false).
            let mut acc = true;
            for e in exprs {
                acc &= evaluate_filter_resolved_strict(e, resolve)?;
            }
            Ok(acc)
        }
        FilterExpression::Or(exprs) => {
            let mut pending_err: Option<FilterEvalError> = None;
            for e in exprs {
                match evaluate_filter_resolved_strict(e, resolve) {
                    Ok(true) => return Ok(true),
                    Ok(false) => {}
                    Err(e) => pending_err = pending_err.or(Some(e)),
                }
            }
            // No arm was true; surface a deferred error if any arm failed (fail-loud), else false.
            match pending_err {
                Some(e) => Err(e),
                None => Ok(false),
            }
        }
        FilterExpression::Not(e) => {
            // Round 19: same SQL NOT(UNKNOWN)=deny rule as the permissive
            // walker (round 18 fixed only that one — strict/legacy modes
            // disagreed on null-field rows).
            if let FilterExpression::Comparison {
                field, operator, ..
            } = &**e
                && !matches!(
                    operator,
                    ComparisonOperator::IsNull | ComparisonOperator::IsNotNull
                )
                && resolve(field).is_none_or(|v| v.is_null())
            {
                Ok(false)
            } else {
                Ok(!evaluate_filter_resolved_strict(e, resolve)?)
            }
        }
        FilterExpression::Comparison {
            field,
            operator,
            value,
        } => match operator {
            // Null tests are decided by presence (SQL: an absent field IS NULL) — never an error.
            ComparisonOperator::IsNull => {
                Ok(resolve(field).is_none_or(|json_val| json_val.is_null()))
            }
            ComparisonOperator::IsNotNull => {
                Ok(resolve(field).is_some_and(|json_val| !json_val.is_null()))
            }
            _ => match resolve(field) {
                Some(json_val) => Ok(compare_json_op(operator, &json_val, value)),
                None => Err(FilterEvalError::MissingField {
                    field: field.clone(),
                }),
            },
        },
    }
}

/// Evaluate a filter expression against a `ProximaTree` (canonical v2 path).
///
/// Thin adapter over [`evaluate_filter_resolved`]: each field resolves to its
/// scalar leaf lowered via [`proxima_value_to_filter_literal`]. Dot-separated fields walk
/// both native [`ProximaTreeNode::Object`] values and JSON/JSONB scalar values,
/// matching PostgreSQL's `payload->>'key'` semantics without flattening stored
/// documents into a second metadata representation.
pub fn evaluate_filter_proxima(expr: &FilterExpression, props: &ProximaTree) -> bool {
    evaluate_filter_resolved(expr, &|field| {
        resolve_proxima_value(props, field)
            .map(|value| proxima_value_to_filter_literal(value.as_ref()))
    })
}

fn resolve_proxima_value<'a>(props: &'a ProximaTree, field: &str) -> Option<Cow<'a, ProximaValue>> {
    if let Some(value) = proximadb_records::tree_get(props, field) {
        return Some(Cow::Borrowed(value));
    }
    let (head, tail) = field.split_once('.')?;
    let root = match props.get(head)? {
        ProximaTreeNode::Value(ProximaValue::Json(value))
        | ProximaTreeNode::Value(ProximaValue::Jsonb(value)) => value,
        _ => return None,
    };
    json_get_path(root, tail.split('.'))
        .map(|value| Cow::Owned(proximadb_records::conversions::json_to_proxima(value)))
}

/// Evaluate an **authorization** filter against a `ProximaTree` (canonical v2 path).
///
/// The type-strict twin of [`evaluate_filter_proxima`], and the adapter a
/// security predicate should be evaluated with: it resolves on the
/// `ProximaValue`/`ProximaTree` seam (ADR-024) — never the deprecated v1
/// `SqlValue` envelope — and compares via [`compare_json_op_type_strict`], so a
/// numeric policy threshold cannot be satisfied by a string-valued field
/// (TD-FOUNDATION-3 slice FA-a2 / TF-2 S3).
///
/// Note the two independent `strict` axes on this path, which are easy to
/// conflate:
///
/// * [`evaluate_filter_proxima_strict`] is **fail-loud on an unresolved field**
///   (TD-FILT-1) — it surfaces a missing field as an error instead of dropping
///   the row.
/// * this function is **type-strict** — it refuses cross-class comparisons.
///   Absence is handled exactly as the default walker handles it.
pub fn evaluate_filter_proxima_type_strict(expr: &FilterExpression, props: &ProximaTree) -> bool {
    evaluate_filter_resolved_type_strict(expr, &|field| {
        resolve_proxima_value(props, field)
            .map(|value| proxima_value_to_filter_literal(value.as_ref()))
    })
}

/// Native, allocation-free ordering for the filter's common scalar types, matching
/// [`crate::json_comparison::compare_json_values`] over the JSON-lowered value. Returns `None`
/// (→ JSON-path fallback) for numbers, `Decimal`, and exotic types: `compare_json_numbers` uses an
/// epsilon for float/huge-int *equality* that a single ordering can't express, so a native numeric
/// ordering would risk diverging from the JSON path (breaking the strict≡default parity this slice
/// guarantees). The real, parity-safe win here is `String`/`Symbol` (avoids the per-field clone).
/// (TD-FILT-1 slice 3.)
fn native_scalar_order(
    pv: &ProximaValue,
    literal: &serde_json::Value,
) -> Option<std::cmp::Ordering> {
    use serde_json::Value as J;
    match (pv, literal) {
        (ProximaValue::Boolean(b), J::Bool(jb)) => Some(b.cmp(jb)),
        (ProximaValue::String(s) | ProximaValue::Symbol(s), J::String(js)) => {
            Some(s.as_str().cmp(js.as_str()))
        }
        _ => None,
    }
}

/// Map an ordering result onto a comparison operator's boolean outcome.
fn apply_op(op: &ComparisonOperator, ord: std::cmp::Ordering) -> bool {
    use std::cmp::Ordering::*;
    match op {
        ComparisonOperator::Equals => ord == Equal,
        ComparisonOperator::NotEquals => ord != Equal,
        ComparisonOperator::LessThan => ord == Less,
        ComparisonOperator::LessThanOrEqual => matches!(ord, Less | Equal),
        ComparisonOperator::GreaterThan => ord == Greater,
        ComparisonOperator::GreaterThanOrEqual => matches!(ord, Greater | Equal),
        // Unreachable: compare_proxima_op only routes the six ordering/equality ops here.
        _ => false,
    }
}

/// Compare a stored [`ProximaValue`] against a query literal (`serde_json::Value`). Native
/// (allocation-free) for `Boolean` and `String`/`Symbol` — matching the JSON-path semantics
/// exactly — and a JSON-path fallback for everything else. Guarantees strict ≡ default on which
/// rows match (native cases match by construction; fallback cases ARE the JSON path).
/// (TD-FILT-1 slice 3.)
fn compare_proxima_op(
    pv: &ProximaValue,
    op: &ComparisonOperator,
    literal: &serde_json::Value,
) -> bool {
    use ComparisonOperator::*;
    match op {
        Equals | NotEquals | LessThan | LessThanOrEqual | GreaterThan | GreaterThanOrEqual => {
            match native_scalar_order(pv, literal) {
                Some(ord) => apply_op(op, ord),
                None => compare_json_op(op, &proxima_value_to_filter_literal(pv), literal),
            }
        }
        // Operators that don't reduce to a single ordering always use the JSON path.
        _ => compare_json_op(op, &proxima_value_to_filter_literal(pv), literal),
    }
}

/// Fail-loud variant of [`evaluate_filter_proxima`] (TD-FILT-1) on the canonical `ProximaTree`
/// path. Resolves each field through the same nested-field resolver as
/// [`evaluate_filter_proxima`] and compares via [`compare_proxima_op`]. Native tree
/// values stay borrowed; a leaf nested inside JSON/JSONB is converted to an owned
/// [`ProximaValue`]. Surfaces an unresolved field as [`FilterEvalError::MissingField`]
/// instead of silently dropping the row.
pub fn evaluate_filter_proxima_strict(
    expr: &FilterExpression,
    props: &ProximaTree,
) -> Result<bool, FilterEvalError> {
    match expr {
        FilterExpression::And(exprs) => {
            // Non-short-circuit: evaluate every child so a MissingField in a later arm surfaces.
            let mut acc = true;
            for e in exprs {
                acc &= evaluate_filter_proxima_strict(e, props)?;
            }
            Ok(acc)
        }
        FilterExpression::Or(exprs) => {
            let mut pending: Option<FilterEvalError> = None;
            for e in exprs {
                match evaluate_filter_proxima_strict(e, props) {
                    Ok(true) => return Ok(true),
                    Ok(false) => {}
                    Err(e) => pending = pending.or(Some(e)),
                }
            }
            match pending {
                Some(e) => Err(e),
                None => Ok(false),
            }
        }
        FilterExpression::Not(e) => {
            // Round 19: NOT(UNKNOWN)=deny — see the resolved-strict arm.
            if let FilterExpression::Comparison {
                field, operator, ..
            } = &**e
                && !matches!(
                    operator,
                    ComparisonOperator::IsNull | ComparisonOperator::IsNotNull
                )
                && proximadb_records::tree_get(props, field)
                    .is_none_or(|pv| matches!(pv, ProximaValue::Null))
            {
                Ok(false)
            } else {
                Ok(!evaluate_filter_proxima_strict(e, props)?)
            }
        }
        FilterExpression::Comparison {
            field,
            operator,
            value,
        } => match operator {
            // Null tests are decided by presence (SQL: an absent field IS NULL) — never an error.
            ComparisonOperator::IsNull => Ok(resolve_proxima_value(props, field)
                .is_none_or(|pv| matches!(pv.as_ref(), ProximaValue::Null))),
            ComparisonOperator::IsNotNull => Ok(resolve_proxima_value(props, field)
                .is_some_and(|pv| !matches!(pv.as_ref(), ProximaValue::Null))),
            _ => match resolve_proxima_value(props, field) {
                Some(pv) => Ok(compare_proxima_op(pv.as_ref(), operator, value)),
                None => Err(FilterEvalError::MissingField {
                    field: field.clone(),
                }),
            },
        },
    }
}

/// Numeric-aware equality for JSON values — the single equality primitive for
/// every `FilterExpression` evaluator.
///
/// `serde_json::Value`'s derived `PartialEq` compares numbers by their internal
/// representation, so an integer `2` and a float `2.0` are NOT equal, and a large
/// `i64` beyond `f64` exact range can mis-compare. Numbers are compared via the
/// shared, integer-first [`compare_json_numbers`] (precise for `i64`/`u64`, epsilon
/// for floats, `NaN == NaN`); all other JSON values fall back to structural
/// equality (correct for strings, bools, arrays, objects, null).
fn json_eq(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    use crate::json_comparison::compare_json_numbers;
    match (a, b) {
        (serde_json::Value::Number(n1), serde_json::Value::Number(n2)) => {
            compare_json_numbers(n1, n2)
        }
        _ => a == b,
    }
}

/// Total-order comparison shared by every evaluator's ordering operators
/// (`<`, `<=`, `>`, `>=`, `Between`). Delegates to [`compare_json_values`], which
/// orders numbers precisely, strings/bools/arrays by value, and otherwise by JSON
/// type precedence (`Null < Bool < Number < String < Array < Object`).
fn compare_json_lt(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    use std::cmp::Ordering;
    crate::json_comparison::compare_json_values(a, b) == Ordering::Less
}
fn compare_json_lte(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    use std::cmp::Ordering;
    matches!(
        crate::json_comparison::compare_json_values(a, b),
        Ordering::Less | Ordering::Equal
    )
}
fn compare_json_gt(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    use std::cmp::Ordering;
    crate::json_comparison::compare_json_values(a, b) == Ordering::Greater
}
fn compare_json_gte(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    use std::cmp::Ordering;
    matches!(
        crate::json_comparison::compare_json_values(a, b),
        Ordering::Greater | Ordering::Equal
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_sql_value(value: SqlVal) -> SqlValue {
        SqlValue { value: Some(value) }
    }

    #[test]
    fn exotic_filter_literals_render_matchably() {
        // Pins the literal spellings the ProximaTree seam renders for typed
        // exotics: bytes as the int-array, Uuid dashed, ULID plain hex,
        // sparse as the canonical {indices, values} object. Each previously
        // rendered a placeholder or Rust Debug string no literal could ever
        // match (TD-PROTO-2 rounds 5-9).
        let mut props: ProximaTree = HashMap::new();
        props.insert(
            "blob".to_string(),
            ProximaTreeNode::Value(ProximaValue::Binary(vec![0x00, 0x01, 0x02])),
        );
        props.insert(
            "correlation_id".to_string(),
            ProximaTreeNode::Value(ProximaValue::Uuid([
                0x55, 0x0e, 0x84, 0x00, 0xe2, 0x9b, 0x41, 0xd4, 0xa7, 0x16, 0x44, 0x65, 0x54, 0x40,
                0x00, 0x00,
            ])),
        );
        props.insert(
            "trace_id".to_string(),
            ProximaTreeNode::Value(ProximaValue::ULID([
                0x01, 0xab, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00,
            ])),
        );
        props.insert(
            "embedding_sparse".to_string(),
            ProximaTreeNode::Value(ProximaValue::SparseVector {
                indices: vec![1, 3],
                values: vec![0.5, 2.5],
            }),
        );

        let eval = |field: &str, value: serde_json::Value| {
            evaluate_filter_proxima(
                &FilterExpression::Comparison {
                    field: field.to_string(),
                    operator: ComparisonOperator::Equals,
                    value,
                },
                &props,
            )
        };

        assert!(eval("blob", json!([0, 1, 2])));
        assert!(!eval("blob", json!("AAEC"))); // base64 is the RENDERING spelling, not the filter's
        assert!(eval(
            "correlation_id",
            json!("550e8400-e29b-41d4-a716-446554400000")
        ));
        assert!(eval("trace_id", json!("01ab0000000000000000000000000000")));
        assert!(eval(
            "embedding_sparse",
            json!({"indices": [1, 3], "values": [0.5, 2.5]})
        ));
    }

    #[test]
    fn jsonb_metadata_decodes_to_json_for_filtering() {
        // Canonical MessagePack JSONB (types.proto tag 9) must lower to the
        // JSON document itself, not a byte rendering — the per-byte-array and
        // hex renderings made every structured filter against a JSONB column
        // silently miss (TD-PROTO-2 review finding).
        let doc = json!({"memory": {"type": "fact"}});
        let bytes = ProximaValue::to_jsonb_vec(&doc).unwrap();
        assert_eq!(sql_val_to_json(&SqlVal::JsonbValue(bytes)), doc);

        // End-to-end: a scalar JSONB document is filterable via the metadata path.
        let mut metadata = HashMap::new();
        metadata.insert(
            "score".to_string(),
            make_sql_value(SqlVal::JsonbValue(
                ProximaValue::to_jsonb_vec(&json!(42)).unwrap(),
            )),
        );
        let filter = FilterExpression::Comparison {
            field: "score".to_string(),
            operator: ComparisonOperator::Equals,
            value: json!(42),
        };
        assert!(evaluate_filter(&filter, &metadata));

        // Malformed JSONB falls back to an observable hex string, never a
        // drop. (0xc1 is msgpack's never-used marker — guaranteed invalid.)
        assert_eq!(
            sql_val_to_json(&SqlVal::JsonbValue(vec![0xc1])),
            serde_json::Value::String("c1".to_string())
        );
    }

    #[test]
    fn null_valued_field_satisfies_no_range_predicate() {
        // Round 13 (SQL three-valued logic): a null-valued field must not
        // leak into range results while an absent field does not.
        let mut metadata = HashMap::new();
        metadata.insert("rank".to_string(), SqlValue { value: None });
        let lt = FilterExpression::Comparison {
            field: "rank".to_string(),
            operator: ComparisonOperator::LessThan,
            value: json!(10),
        };
        assert!(!evaluate_filter(&lt, &metadata));
        let gte = FilterExpression::Comparison {
            field: "rank".to_string(),
            operator: ComparisonOperator::GreaterThanOrEqual,
            value: json!(0),
        };
        assert!(!evaluate_filter(&gte, &metadata));
        // Equality with an explicit null literal still matches (pinned in
        // round 12) — the JSON-consistent choice, documented here.
        let eq = FilterExpression::Comparison {
            field: "rank".to_string(),
            operator: ComparisonOperator::Equals,
            value: json!(null),
        };
        assert!(evaluate_filter(&eq, &metadata));

        metadata.insert("rank".to_string(), make_sql_value(SqlVal::Int64Value(5)));
        let between_null_bound = FilterExpression::Comparison {
            field: "rank".to_string(),
            operator: ComparisonOperator::Between,
            value: json!([null, 10]),
        };
        assert!(
            !evaluate_filter(&between_null_bound, &metadata),
            "BETWEEN is a range predicate and a null bound is unordered"
        );
    }

    #[test]
    fn null_valued_field_satisfies_no_not_equals_or_not_in() {
        // Round 14's centralized guard: a null on either side makes every
        // comparison false except IS NULL / IS NOT NULL and both-null `=`.
        // Pins the !=/NOT-IN half the round-13 guards missed — a null field
        // is not "different from" a literal.
        let mut metadata = HashMap::new();
        metadata.insert("status".to_string(), SqlValue { value: None });

        let ne_literal = FilterExpression::Comparison {
            field: "status".to_string(),
            operator: ComparisonOperator::NotEquals,
            value: json!("deleted"),
        };
        assert!(
            !evaluate_filter(&ne_literal, &metadata),
            "null != 'deleted' is UNKNOWN, not TRUE — the row must not match"
        );

        let not_in = FilterExpression::Comparison {
            field: "status".to_string(),
            operator: ComparisonOperator::NotIn,
            value: json!(["archived", "deleted"]),
        };
        assert!(
            !evaluate_filter(&not_in, &metadata),
            "null NOT IN (...) is UNKNOWN, not TRUE"
        );

        // Positive control: a non-null field keeps matching both forms.
        metadata.insert(
            "status".to_string(),
            make_sql_value(SqlVal::StringValue("active".to_string())),
        );
        assert!(evaluate_filter(&ne_literal, &metadata));
        assert!(evaluate_filter(&not_in, &metadata));

        // The deliberate stricter carve-out: `!= null` is FALSE for every
        // field (either-null short-circuit) — IS NOT NULL is the operator
        // that speaks about null.
        metadata.insert("status".to_string(), SqlValue { value: None });
        let ne_null = FilterExpression::Comparison {
            field: "status".to_string(),
            operator: ComparisonOperator::NotEquals,
            value: json!(null),
        };
        assert!(!evaluate_filter(&ne_null, &metadata));
        // Round 15 DELIBERATELY diverges from an earlier pin here: `!= null`
        // is the Mongo-style exclude-nulls idiom (live on the REST neq path),
        // and develop admitted non-null rows — zero rows was the regression
        // the round-15 review flagged. Use IS NOT NULL for null-testing.
        metadata.insert(
            "status".to_string(),
            make_sql_value(SqlVal::StringValue("active".to_string())),
        );
        assert!(evaluate_filter(&ne_null, &metadata));
    }

    #[test]
    fn top_level_none_valued_field_resolves_to_json_null() {
        // Round 12: the same wire form of null at the TOP level — `a = null`
        // must match, not drop the row.
        let mut metadata = HashMap::new();
        metadata.insert("a".to_string(), SqlValue { value: None });
        let eq_null = FilterExpression::Comparison {
            field: "a".to_string(),
            operator: ComparisonOperator::Equals,
            value: json!(null),
        };
        assert!(evaluate_filter(&eq_null, &metadata));
    }

    #[test]
    fn none_valued_child_resolves_to_json_null() {
        // Round 11: {"a": {"b": {"c": null}}} — the writer emits
        // SqlValue{value: None} for the nested null; `a.b.c = null` and
        // `a.b = {"c": null}` must both match as they did pre-rewrite.
        let mut c_leaf = HashMap::new();
        c_leaf.insert("c".to_string(), SqlValue { value: None });
        let mut a_root = HashMap::new();
        a_root.insert(
            "b".to_string(),
            make_sql_value(SqlVal::ObjectValue(
                proximadb_proto::proximadb_v1::SqlObject { fields: c_leaf },
            )),
        );
        let mut metadata = HashMap::new();
        metadata.insert(
            "a".to_string(),
            make_sql_value(SqlVal::ObjectValue(
                proximadb_proto::proximadb_v1::SqlObject { fields: a_root },
            )),
        );
        let eq_null = FilterExpression::Comparison {
            field: "a.b.c".to_string(),
            operator: ComparisonOperator::Equals,
            value: json!(null),
        };
        assert!(evaluate_filter(&eq_null, &metadata));

        let obj_eq = FilterExpression::Comparison {
            field: "a".to_string(),
            operator: ComparisonOperator::Equals,
            value: json!({"b": {"c": null}}),
        };
        assert!(evaluate_filter(&obj_eq, &metadata));
    }

    #[test]
    fn dotted_field_on_a_scalar_parent_never_resolves() {
        // Round 10: the traversal once returned the scalar PARENT's value
        // for `user.name` over {"user": "admin"}, admitting rows on
        // `user.name = "admin"` and flipping IS NULL semantics.
        let mut metadata = HashMap::new();
        metadata.insert(
            "user".to_string(),
            make_sql_value(SqlVal::StringValue("admin".to_string())),
        );
        let filter = FilterExpression::Comparison {
            field: "user.name".to_string(),
            operator: ComparisonOperator::Equals,
            value: json!("admin"),
        };
        assert!(!evaluate_filter(&filter, &metadata));

        let null_test = FilterExpression::Comparison {
            field: "user.name".to_string(),
            operator: ComparisonOperator::IsNull,
            value: json!(null),
        };
        // The field cannot resolve — SQL semantics treat an absent field as
        // NULL (the strict evaluator's is_none_or; the round-10 bug returned
        // the parent's non-null value and flipped this to false).
        assert!(evaluate_filter(&null_test, &metadata));

        metadata.insert(
            "payload".to_string(),
            make_sql_value(SqlVal::ObjectValue(
                proximadb_proto::proximadb_v1::SqlObject {
                    fields: HashMap::from([(
                        "rank".to_string(),
                        make_sql_value(SqlVal::Int64Value(7)),
                    )]),
                },
            )),
        );
        let nested = FilterExpression::Comparison {
            field: "payload.rank.extra".to_string(),
            operator: ComparisonOperator::Equals,
            value: json!(7),
        };
        assert!(
            !evaluate_filter(&nested, &metadata),
            "a nested scalar cannot satisfy a remaining path segment"
        );
    }

    #[test]
    fn jsonb_dot_path_filtering_matches_the_proxima_tree_semantics() {
        // Round 5: the SqlValue path must dot-traverse JSON(B) roots exactly
        // like `resolve_proxima_value` does on the ProximaTree path —
        // otherwise the same data filters differently per engine.
        let mut metadata = HashMap::new();
        metadata.insert(
            "payload".to_string(),
            make_sql_value(SqlVal::JsonbValue(
                ProximaValue::to_jsonb_vec(&json!({"memory": {"type": "fact"}})).unwrap(),
            )),
        );
        let filter = FilterExpression::Comparison {
            field: "payload.memory.type".to_string(),
            operator: ComparisonOperator::Equals,
            value: json!("fact"),
        };
        assert!(evaluate_filter(&filter, &metadata));

        let miss = FilterExpression::Comparison {
            field: "payload.memory.type".to_string(),
            operator: ComparisonOperator::Equals,
            value: json!("fiction"),
        };
        assert!(!evaluate_filter(&miss, &metadata));
    }

    fn proxima_array_props(field: &str, values: &[&str]) -> ProximaTree {
        let mut props = ProximaTree::new();
        props.insert(
            field.to_string(),
            ProximaTreeNode::Value(ProximaValue::Array(
                values
                    .iter()
                    .map(|v| ProximaValue::String(v.to_string()))
                    .collect(),
            )),
        );
        props
    }

    #[test]
    fn proxima_jsonb_dot_path_resolves_without_flattening() {
        let props = ProximaTree::from([(
            "payload".to_string(),
            ProximaTreeNode::Value(ProximaValue::Jsonb(json!({
                "memory": {"type": "fact"}
            }))),
        )]);
        let filter = FilterExpression::Comparison {
            field: "payload.memory.type".to_string(),
            operator: ComparisonOperator::Equals,
            value: json!("fact"),
        };

        assert!(evaluate_filter_proxima(&filter, &props));
        assert!(evaluate_filter_proxima_type_strict(&filter, &props));
        assert_eq!(evaluate_filter_proxima_strict(&filter, &props), Ok(true));
    }

    #[test]
    fn proxima_scalar_typed_props_compare_by_value() {
        // Covers the string/int/float/bool prop envelope through the canonical
        // ProximaTree evaluator: each typed ProximaValue must round-trip to a
        // JSON value the comparators handle by value (not by representation).
        let mut props = ProximaTree::new();
        props.insert(
            "account_id".to_string(),
            ProximaTreeNode::Value(ProximaValue::String("acctA".to_string())),
        );
        props.insert(
            "tier".to_string(),
            ProximaTreeNode::Value(ProximaValue::Int64(2)),
        );
        props.insert(
            "score".to_string(),
            ProximaTreeNode::Value(ProximaValue::Float64(0.5)),
        );
        props.insert(
            "active".to_string(),
            ProximaTreeNode::Value(ProximaValue::Boolean(true)),
        );

        let cases: Vec<(&str, ComparisonOperator, serde_json::Value, bool)> = vec![
            (
                "account_id",
                ComparisonOperator::Equals,
                json!("acctA"),
                true,
            ),
            (
                "account_id",
                ComparisonOperator::Equals,
                json!("acctB"),
                false,
            ),
            // int prop vs int literal AND float-typed literal (numeric-aware).
            ("tier", ComparisonOperator::Equals, json!(2), true),
            ("tier", ComparisonOperator::Equals, json!(2.0), true),
            ("tier", ComparisonOperator::GreaterThan, json!(1), true),
            ("tier", ComparisonOperator::LessThan, json!(2), false),
            (
                "score",
                ComparisonOperator::LessThanOrEqual,
                json!(0.5),
                true,
            ),
            ("score", ComparisonOperator::GreaterThan, json!(0.9), false),
            ("active", ComparisonOperator::Equals, json!(true), true),
            ("active", ComparisonOperator::NotEquals, json!(false), true),
        ];
        for (field, operator, value, expected) in cases {
            let label = format!("{field} {operator:?} {value} should be {expected}");
            let filter = FilterExpression::Comparison {
                field: field.to_string(),
                operator,
                value,
            };
            assert_eq!(
                evaluate_filter_proxima(&filter, &props),
                expected,
                "{label}"
            );
        }
    }

    #[test]
    fn proxima_array_prop_contains_membership() {
        let props = proxima_array_props("member_oids", &["u1", "u2"]);

        let hit = FilterExpression::Comparison {
            field: "member_oids".to_string(),
            operator: ComparisonOperator::Contains,
            value: json!("u1"),
        };
        let miss = FilterExpression::Comparison {
            field: "member_oids".to_string(),
            operator: ComparisonOperator::Contains,
            value: json!("u9"),
        };
        assert!(evaluate_filter_proxima(&hit, &props));
        assert!(!evaluate_filter_proxima(&miss, &props));
    }

    #[test]
    fn proxima_array_prop_in_intersection() {
        let props = proxima_array_props("member_oids", &["u1", "u2"]);

        let intersects = FilterExpression::Comparison {
            field: "member_oids".to_string(),
            operator: ComparisonOperator::In,
            value: json!(["u2", "u3"]),
        };
        let disjoint = FilterExpression::Comparison {
            field: "member_oids".to_string(),
            operator: ComparisonOperator::In,
            value: json!(["u7", "u8"]),
        };
        assert!(evaluate_filter_proxima(&intersects, &props));
        assert!(!evaluate_filter_proxima(&disjoint, &props));

        // NotIn is the negation: disjoint passes, intersecting fails.
        let not_in_disjoint = FilterExpression::Comparison {
            field: "member_oids".to_string(),
            operator: ComparisonOperator::NotIn,
            value: json!(["u7", "u8"]),
        };
        assert!(evaluate_filter_proxima(&not_in_disjoint, &props));
    }

    /// Golden conformance + cross-representation parity. Pins the canonical
    /// operator semantics (the approved "richest union") AND asserts the
    /// `ProximaTree` adapter and a json-map resolver agree on every case — the
    /// safety net for converging the other evaluators onto this seam.
    #[test]
    fn golden_operator_semantics_parity() {
        use std::collections::HashMap;

        // The same record in both representations.
        let mut props = ProximaTree::new();
        props.insert(
            "account_id".to_string(),
            ProximaTreeNode::Value(ProximaValue::String("acctA".to_string())),
        );
        props.insert(
            "tier".to_string(),
            ProximaTreeNode::Value(ProximaValue::Int64(2)),
        );
        props.insert(
            "score".to_string(),
            ProximaTreeNode::Value(ProximaValue::Float64(0.5)),
        );
        props.insert(
            "active".to_string(),
            ProximaTreeNode::Value(ProximaValue::Boolean(true)),
        );
        props.insert(
            "member_oids".to_string(),
            ProximaTreeNode::Value(ProximaValue::Array(vec![
                ProximaValue::String("u1".to_string()),
                ProximaValue::String("u2".to_string()),
            ])),
        );
        props.insert(
            "maybe".to_string(),
            ProximaTreeNode::Value(ProximaValue::Null),
        );

        let jmap: HashMap<String, serde_json::Value> = HashMap::from([
            ("account_id".to_string(), json!("acctA")),
            ("tier".to_string(), json!(2)),
            ("score".to_string(), json!(0.5)),
            ("active".to_string(), json!(true)),
            ("member_oids".to_string(), json!(["u1", "u2"])),
            ("maybe".to_string(), serde_json::Value::Null),
        ]);

        use ComparisonOperator::*;
        // (field, operator, literal, expected)
        let cases: Vec<(&str, ComparisonOperator, serde_json::Value, bool)> = vec![
            // equality — numeric-aware (int literal AND float-typed literal)
            ("account_id", Equals, json!("acctA"), true),
            ("account_id", NotEquals, json!("acctB"), true),
            ("tier", Equals, json!(2), true),
            ("tier", Equals, json!(2.0), true),
            ("score", Equals, json!(0.5), true),
            // ordering — numeric
            ("tier", GreaterThan, json!(1), true),
            ("tier", LessThan, json!(2), false),
            ("tier", GreaterThanOrEqual, json!(2), true),
            ("score", LessThanOrEqual, json!(0.5), true),
            // ordering — STRING (union: lexicographic; was false on old proxima)
            ("account_id", LessThan, json!("acctB"), true),
            ("account_id", GreaterThan, json!("acctB"), false),
            // In / NotIn — scalar membership
            ("tier", In, json!([1, 2, 3]), true),
            ("tier", NotIn, json!([1, 3]), true),
            // In / Contains / NotIn — array-valued prop (intersection / membership)
            ("member_oids", In, json!(["u2", "u9"]), true),
            ("member_oids", In, json!(["u7", "u8"]), false),
            ("member_oids", Contains, json!("u1"), true),
            ("member_oids", Contains, json!("u9"), false),
            ("member_oids", NotIn, json!(["u7", "u8"]), true),
            // affix
            ("account_id", StartsWith, json!("acc"), true),
            ("account_id", EndsWith, json!("tA"), true),
            ("account_id", StartsWith, json!("xyz"), false),
            // Between (numeric, inclusive)
            ("tier", Between, json!([1, 3]), true),
            ("tier", Between, json!([3, 4]), false),
            // LIKE — full engine: leading/trailing %, internal %, and `_`
            ("account_id", Like, json!("acctA"), true),
            ("account_id", Like, json!("ac%tA"), true),
            ("account_id", Like, json!("%cct%"), true),
            ("account_id", Like, json!("_cctA"), true),
            ("account_id", Like, json!("ac_A"), false),
            // IsNull / IsNotNull — present-and-null
            ("maybe", IsNull, serde_json::Value::Null, true),
            ("maybe", IsNotNull, serde_json::Value::Null, false),
            // present-and-non-null
            ("account_id", IsNull, serde_json::Value::Null, false),
            ("account_id", IsNotNull, serde_json::Value::Null, true),
            // ABSENT field — SQL semantics: absent IS NULL
            ("ghost", IsNull, serde_json::Value::Null, true),
            ("ghost", IsNotNull, serde_json::Value::Null, false),
            ("ghost", Equals, json!("x"), false),
        ];

        for (field, operator, value, expected) in cases {
            let filter = FilterExpression::Comparison {
                field: field.to_string(),
                operator: operator.clone(),
                value,
            };
            let via_proxima = evaluate_filter_proxima(&filter, &props);
            let via_jmap = evaluate_filter_resolved(&filter, &|f| jmap.get(f).cloned());
            assert_eq!(
                via_proxima, expected,
                "proxima: {field} {operator:?} expected {expected}"
            );
            assert_eq!(
                via_jmap, via_proxima,
                "representation parity broke for {field} {operator:?}"
            );
        }
    }

    #[test]
    fn test_string_equality() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "name".to_string(),
            make_sql_value(SqlVal::StringValue("Alice".to_string())),
        );

        let filter = FilterExpression::Comparison {
            field: "name".to_string(),
            operator: ComparisonOperator::Equals,
            value: json!("Alice"),
        };

        assert!(evaluate_filter(&filter, &metadata));
    }

    #[test]
    fn test_number_comparisons() {
        let mut metadata = HashMap::new();
        metadata.insert("age".to_string(), make_sql_value(SqlVal::NumberValue(25.0)));

        // Less than
        let filter = FilterExpression::Comparison {
            field: "age".to_string(),
            operator: ComparisonOperator::LessThan,
            value: json!(30),
        };
        assert!(evaluate_filter(&filter, &metadata));

        // Greater than
        let filter = FilterExpression::Comparison {
            field: "age".to_string(),
            operator: ComparisonOperator::GreaterThan,
            value: json!(20),
        };
        assert!(evaluate_filter(&filter, &metadata));
    }

    #[test]
    fn test_bool_equality() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "active".to_string(),
            make_sql_value(SqlVal::BoolValue(true)),
        );

        let filter = FilterExpression::Comparison {
            field: "active".to_string(),
            operator: ComparisonOperator::Equals,
            value: json!(true),
        };

        assert!(evaluate_filter(&filter, &metadata));
    }

    #[test]
    fn test_and_or_not() {
        let mut metadata = HashMap::new();
        metadata.insert("age".to_string(), make_sql_value(SqlVal::NumberValue(25.0)));
        metadata.insert(
            "active".to_string(),
            make_sql_value(SqlVal::BoolValue(true)),
        );

        // AND
        let filter = FilterExpression::And(vec![
            FilterExpression::Comparison {
                field: "age".to_string(),
                operator: ComparisonOperator::GreaterThan,
                value: json!(20),
            },
            FilterExpression::Comparison {
                field: "active".to_string(),
                operator: ComparisonOperator::Equals,
                value: json!(true),
            },
        ]);
        assert!(evaluate_filter(&filter, &metadata));
    }

    // ---- TD-FILT-1: evaluate_filter_proxima_strict (fail-loud, v2 ProximaValue path) ----

    fn proxima_props_one(field: &str, pv: ProximaValue) -> ProximaTree {
        let mut props = ProximaTree::new();
        props.insert(field.to_string(), ProximaTreeNode::Value(pv));
        props
    }

    #[test]
    fn strict_clean_match_agrees_with_legacy() {
        let props = proxima_props_one("tier", ProximaValue::Int64(2));
        let filter = FilterExpression::Comparison {
            field: "tier".to_string(),
            operator: ComparisonOperator::Equals,
            value: json!(2),
        };
        assert!(evaluate_filter_proxima(&filter, &props));
        assert!(evaluate_filter_proxima_strict(&filter, &props).unwrap());
    }

    #[test]
    fn strict_missing_field_surfaces_error_legacy_silently_drops() {
        // No "tier" field in the (empty) props.
        let props = ProximaTree::new();
        let filter = FilterExpression::Comparison {
            field: "tier".to_string(),
            operator: ComparisonOperator::Equals,
            value: json!(2),
        };
        // The bug (TD-FILT-1): the legacy path silently drops the row (`None => false`).
        assert!(
            !evaluate_filter_proxima(&filter, &props),
            "legacy evaluate_filter_proxima silently drops the unresolved-field row"
        );
        // The fix: strict surfaces it as a typed error.
        assert_eq!(
            evaluate_filter_proxima_strict(&filter, &props),
            Err(FilterEvalError::MissingField {
                field: "tier".to_string()
            })
        );
    }

    #[test]
    fn strict_isnull_on_missing_is_ok_not_error() {
        // Absent field IS NULL (SQL semantics) — not a MissingField error.
        let props = ProximaTree::new();
        let filter = FilterExpression::Comparison {
            field: "tier".to_string(),
            operator: ComparisonOperator::IsNull,
            value: json!(null),
        };
        assert!(evaluate_filter_proxima_strict(&filter, &props).unwrap());
    }

    #[test]
    fn strict_and_surfaces_missing_field_even_when_an_earlier_arm_is_false() {
        // arm1 (a==2) is false (a is 1); arm2 references a missing field. Non-short-circuit strict
        // must surface the MissingField rather than hide it behind arm1's Ok(false).
        let mut props = ProximaTree::new();
        props.insert(
            "a".to_string(),
            ProximaTreeNode::Value(ProximaValue::Int64(1)),
        );
        let filter = FilterExpression::And(vec![
            FilterExpression::Comparison {
                field: "a".to_string(),
                operator: ComparisonOperator::Equals,
                value: json!(2),
            },
            FilterExpression::Comparison {
                field: "missing".to_string(),
                operator: ComparisonOperator::Equals,
                value: json!(1),
            },
        ]);
        assert!(matches!(
            evaluate_filter_proxima_strict(&filter, &props),
            Err(FilterEvalError::MissingField { .. })
        ));
    }

    // ---- TD-FILT-1 slice 3: native comparator parity (strict ≡ default on matches) ----

    #[test]
    fn native_compare_matches_json_path_for_supported_scalars() {
        // For every (pv, literal, op) the native path handles (Boolean↔Bool, String/Symbol↔String),
        // compare_proxima_op MUST equal the JSON path — so strict and default agree on matches.
        let pvs = [
            ProximaValue::Boolean(true),
            ProximaValue::Boolean(false),
            ProximaValue::String("apple".to_string()),
            ProximaValue::String("banana".to_string()),
            ProximaValue::Symbol("sym".to_string()),
            ProximaValue::String("".to_string()),
        ];
        let lits = [
            json!(true),
            json!(false),
            json!("apple"),
            json!("mango"),
            json!("banana"),
            json!("sym"),
            json!(""),
        ];
        let ops = [
            ComparisonOperator::Equals,
            ComparisonOperator::NotEquals,
            ComparisonOperator::LessThan,
            ComparisonOperator::LessThanOrEqual,
            ComparisonOperator::GreaterThan,
            ComparisonOperator::GreaterThanOrEqual,
        ];
        for pv in &pvs {
            for lit in &lits {
                if native_scalar_order(pv, lit).is_some() {
                    for op in &ops {
                        let native = compare_proxima_op(pv, op, lit);
                        let json_path =
                            compare_json_op(op, &proxima_value_to_filter_literal(pv), lit);
                        assert_eq!(
                            native, json_path,
                            "parity break: pv={pv:?} lit={lit} op={op:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn numbers_and_exotic_types_fall_back_to_json_path() {
        // Numbers / Decimal / exotic / type-mismatched pairs are not handled natively (the
        // epsilon-equality semantics make a native numeric ordering unsafe for parity), so
        // compare_proxima_op IS the JSON path for them.
        let cases = [
            (ProximaValue::Int64(5), json!(5)),
            (ProximaValue::Int64(2), json!(2.0)),
            (ProximaValue::Float64(2.5), json!(2.5)),
            (ProximaValue::Boolean(true), json!(5)),
        ];
        for (pv, lit) in cases {
            assert!(
                native_scalar_order(&pv, &lit).is_none(),
                "expected JSON-path fallback: {pv:?} vs {lit}"
            );
            let op = ComparisonOperator::Equals;
            assert_eq!(
                compare_proxima_op(&pv, &op, &lit),
                compare_json_op(&op, &proxima_value_to_filter_literal(&pv), &lit),
                "fallback parity: {pv:?} vs {lit}"
            );
        }
    }
}

// ===========================================================================
// Type-strict operator evaluation (TD-FOUNDATION-3 slice FA-a2 / TF-2 S3)
// ===========================================================================

/// [`compare_json_op`], but on a **3-valued, type-strict substrate**: returns
/// `None` when the operands are *incomparable* (different JSON classes), so a
/// caller can propagate that as SQL UNKNOWN rather than collapsing it to a
/// boolean that `Not` would flip into an admit.
///
/// ## Why three-valued, not a guarded boolean
///
/// An earlier version returned `bool`, guarding each negative operator with a
/// `same_class` check and letting `Equals`/`In` fall through to `json_eq`. That
/// is correct for *positive* comparisons but inverts under `Not`: a strict
/// comparison returns `false` for an incomparable pair, and `Not(false)` **admits**
/// — so `NOT(owner == "u/alice")` admitted `{owner: 42}` while `owner != "u/alice"`
/// denied it. The same flip re-opens every guarded operator under negation
/// (`Not(GreaterThan)`, `Not(NotEquals)`, …). Returning `None` for incomparable,
/// and propagating `None` through `Not` as `None` (UNKNOWN → deny), closes the
/// whole class at once. This is the "3-valued substrate" TF-2 §1.4 S2 prescribes.
///
/// Integers and floats share the `Number` class, so `1` and `1.5` remain
/// comparable. The string operators require both operands to be strings; the
/// null tests are decided by presence and never return `None`.
///
/// Additive: [`compare_json_op`] is untouched.
pub fn compare_json_op_type_strict(
    operator: &ComparisonOperator,
    json_val: &serde_json::Value,
    value: &serde_json::Value,
) -> Option<bool> {
    use crate::json_comparison::comparable_class;

    // Null literals: the class gate below would make every null literal
    // cross-class (Null is its own class) and deny BEFORE the centralized
    // null-literal idioms apply — a security predicate using the documented
    // `!= null` exclude-nulls idiom would silently match zero rows. Route
    // null literals through compare_json_op first (round 17); the null
    // FIELD case still falls through to the class gate as deny.
    if value.is_null() {
        return Some(compare_json_op(operator, json_val, value));
    }

    let same_class =
        |other: &serde_json::Value| comparable_class(json_val) == comparable_class(other);

    let require_same = || if same_class(value) { Some(()) } else { None };

    match operator {
        // Null tests are decided by presence — handled by the walker, never None.
        ComparisonOperator::IsNull | ComparisonOperator::IsNotNull => {
            Some(compare_json_op(operator, json_val, value))
        }

        // String operators are meaningful only between two strings.
        ComparisonOperator::Contains
        | ComparisonOperator::StartsWith
        | ComparisonOperator::EndsWith
        | ComparisonOperator::Like => {
            if json_val.is_string() && value.is_string() {
                Some(compare_json_op(operator, json_val, value))
            } else {
                None
            }
        }

        // `In`/`NotIn` need at least one list element the field could actually
        // have compared against, else the question is incomparable (and `NotIn`
        // must not be satisfied merely by being incomparable).
        ComparisonOperator::In | ComparisonOperator::NotIn => match value.as_array() {
            Some(values) if values.iter().any(same_class) => {
                Some(compare_json_op(operator, json_val, value))
            }
            _ => None,
        },

        // `Between` needs both bounds comparable.
        ComparisonOperator::Between => match value.as_array() {
            Some(bounds)
                if bounds.len() == 2 && same_class(&bounds[0]) && same_class(&bounds[1]) =>
            {
                Some(compare_json_op(operator, json_val, value))
            }
            _ => None,
        },

        // Equals/NotEquals use EXACT numeric equality, not json_eq's epsilon.
        // (Red-team Finding 5: json_eq uses a relative epsilon for floats, so a
        // predicate `clearance == 3` would admit `clearance = 3.0000001` within
        // epsilon — a false admit in a security context. The ordered operators
        // already use exact `partial_cmp`; this makes Equals consistent with
        // them under the strict path.)
        ComparisonOperator::Equals => require_same().map(|_| exact_json_eq(json_val, value)),
        ComparisonOperator::NotEquals => require_same().map(|_| !exact_json_eq(json_val, value)),

        // The ordered operators require the same class.
        _ => require_same().map(|_| compare_json_op(operator, json_val, value)),
    }
}

/// Exact JSON equality — like `json_eq` but WITHOUT the float epsilon.
///
/// `json_eq` uses a relative-epsilon comparison for floats (`compare_json_numbers`),
/// which is defensible for user-facing metadata search but **wrong for a security
/// predicate**: a policy `clearance == 3` should not admit `3.0000001`. This
/// function is the strict-path alternative — exact for numbers, structural for
/// everything else.
fn exact_json_eq(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    match (a, b) {
        (serde_json::Value::Number(n1), serde_json::Value::Number(n2)) => {
            // Exact integer first (preserves precision), then exact float.
            match (n1.as_i64(), n2.as_i64()) {
                (Some(i1), Some(i2)) => i1 == i2,
                _ => match (n1.as_u64(), n2.as_u64()) {
                    (Some(u1), Some(u2)) => u1 == u2,
                    _ => n1
                        .as_f64()
                        .zip(n2.as_f64())
                        .is_some_and(|(f1, f2)| f1 == f2),
                },
            }
        }
        // Non-numbers: structural equality, same as json_eq.
        _ => a == b,
    }
}
#[cfg(test)]
mod type_strict_tests {
    use super::*;
    use serde_json::json;

    fn permissive(
        op: ComparisonOperator,
        field: serde_json::Value,
        lit: serde_json::Value,
    ) -> bool {
        compare_json_op(&op, &field, &lit)
    }
    fn strict(op: ComparisonOperator, field: serde_json::Value, lit: serde_json::Value) -> bool {
        // UNKNOWN ⇒ deny, matching the security walker's contract.
        compare_json_op_type_strict(&op, &field, &lit).unwrap_or(false)
    }

    #[test]
    fn a_numeric_threshold_no_longer_admits_a_string_clearance() {
        // TF-2 §1.4's case, end to end: `clearance >= 3` against a record whose
        // clearance is a string.
        assert!(
            permissive(
                ComparisonOperator::GreaterThanOrEqual,
                json!("TOP_SECRET"),
                json!(3)
            ),
            "characterizing the defect: the permissive dispatch admits"
        );
        assert!(!strict(
            ComparisonOperator::GreaterThanOrEqual,
            json!("TOP_SECRET"),
            json!(3)
        ));
        // …including the string that merely looks like a smaller number.
        assert!(!strict(
            ComparisonOperator::GreaterThanOrEqual,
            json!("2"),
            json!(3)
        ));
    }

    #[test]
    fn ordered_operators_still_work_within_a_class() {
        assert!(strict(ComparisonOperator::GreaterThan, json!(5), json!(3)));
        assert!(!strict(ComparisonOperator::GreaterThan, json!(2), json!(3)));
        assert!(strict(ComparisonOperator::LessThan, json!("a"), json!("b")));
        // Integer field against a float literal must keep working.
        assert!(strict(
            ComparisonOperator::GreaterThan,
            json!(1),
            json!(0.8)
        ));
    }

    #[test]
    fn not_equals_is_not_satisfied_by_being_incomparable() {
        assert!(
            permissive(ComparisonOperator::NotEquals, json!("eng"), json!(5)),
            "characterizing the defect: incomparable satisfies !="
        );
        assert!(!strict(
            ComparisonOperator::NotEquals,
            json!("eng"),
            json!(5)
        ));
        // Same-class inequality is unaffected.
        assert!(strict(
            ComparisonOperator::NotEquals,
            json!("eng"),
            json!("hr")
        ));
        assert!(!strict(
            ComparisonOperator::NotEquals,
            json!("eng"),
            json!("eng")
        ));
    }

    #[test]
    fn not_in_requires_a_comparable_list_element() {
        assert!(
            permissive(ComparisonOperator::NotIn, json!("eng"), json!([1, 2])),
            "characterizing the defect: a wholly incomparable list excludes nothing"
        );
        assert!(!strict(
            ComparisonOperator::NotIn,
            json!("eng"),
            json!([1, 2])
        ));
        // A same-class list behaves normally.
        assert!(strict(
            ComparisonOperator::NotIn,
            json!("eng"),
            json!(["hr", "legal"])
        ));
        assert!(!strict(
            ComparisonOperator::NotIn,
            json!("eng"),
            json!(["eng", "hr"])
        ));
    }

    #[test]
    fn between_requires_both_bounds_to_be_comparable() {
        assert!(strict(
            ComparisonOperator::Between,
            json!(5),
            json!([1, 10])
        ));
        assert!(!strict(
            ComparisonOperator::Between,
            json!(5),
            json!([1, "10"])
        ));
        assert!(!strict(
            ComparisonOperator::Between,
            json!("5"),
            json!([1, 10])
        ));
    }

    #[test]
    fn already_exact_operators_are_unchanged() {
        // Equals, In, the string operators and the null tests are type-exact
        // already; strict mode must not alter them.
        for (op, field, lit) in [
            (ComparisonOperator::Equals, json!("eng"), json!("eng")),
            (ComparisonOperator::Equals, json!("eng"), json!(5)),
            (ComparisonOperator::In, json!("eng"), json!(["eng", "hr"])),
            (ComparisonOperator::In, json!("eng"), json!([1, 2])),
            (
                ComparisonOperator::StartsWith,
                json!("engineering"),
                json!("eng"),
            ),
            (ComparisonOperator::StartsWith, json!(5), json!("eng")),
            (
                ComparisonOperator::Contains,
                json!("engineering"),
                json!("gin"),
            ),
            (
                ComparisonOperator::Like,
                json!("engineering"),
                json!("eng%"),
            ),
            (ComparisonOperator::IsNull, json!(null), json!(null)),
            (ComparisonOperator::IsNotNull, json!("eng"), json!(null)),
        ] {
            // For already-exact operators over same-class operands, strict (collapsed
            // to bool) must agree with permissive. unwrap_or(false) = UNKNOWN⇒deny.
            assert_eq!(
                compare_json_op_type_strict(&op, &field, &lit).unwrap_or(false),
                compare_json_op(&op, &field, &lit),
                "strict mode changed an already-exact operator: {op:?} {field} {lit}"
            );
        }
    }

    #[test]
    fn strict_equals_uses_exact_numeric_comparison_not_epsilon() {
        // Finding 5: json_eq uses relative epsilon for floats, so a security
        // predicate clearance == 3 would admit 3.0000001 within epsilon.
        // The strict path must use exact equality.
        let near = json!(1.0_f64 + f64::EPSILON); // within json_eq's epsilon of 1.0
        let one = json!(1.0);
        assert!(
            compare_json_op(&ComparisonOperator::Equals, &near, &one),
            "precondition: the permissive json_eq admits (epsilon absorbs the diff)"
        );
        assert!(
            !compare_json_op_type_strict(&ComparisonOperator::Equals, &near, &one).unwrap_or(false),
            "strict Equals must NOT admit a near-but-not-equal float (no epsilon)"
        );
        // And exact equality still works.
        assert!(
            compare_json_op_type_strict(&ComparisonOperator::Equals, &json!(3), &json!(3))
                .unwrap_or(false)
        );
    }

    #[test]
    fn strict_not_equals_uses_exact_numeric_comparison() {
        let near = json!(1.0_f64 + f64::EPSILON);
        assert!(
            compare_json_op_type_strict(&ComparisonOperator::NotEquals, &near, &json!(1.0))
                .unwrap_or(false),
            "strict NotEquals must deny a near-but-not-equal float (they are NOT equal exactly)"
        );
    }

    #[test]
    fn strict_never_admits_more_than_permissive() {
        // The deny-biased property, over the operator × value grid: strict mode
        // may only ever tighten.
        let values = [
            json!(3),
            json!(3.5),
            json!("3"),
            json!("eng"),
            json!(true),
            json!(null),
            json!([1, 2]),
        ];
        let ops = [
            ComparisonOperator::Equals,
            ComparisonOperator::NotEquals,
            ComparisonOperator::LessThan,
            ComparisonOperator::LessThanOrEqual,
            ComparisonOperator::GreaterThan,
            ComparisonOperator::GreaterThanOrEqual,
            ComparisonOperator::In,
            ComparisonOperator::NotIn,
            ComparisonOperator::Between,
            ComparisonOperator::Contains,
            ComparisonOperator::StartsWith,
            ComparisonOperator::EndsWith,
            ComparisonOperator::Like,
        ];
        for op in &ops {
            for field in &values {
                for lit in &values {
                    if compare_json_op_type_strict(op, field, lit).unwrap_or(false) {
                        assert!(
                            compare_json_op(op, field, lit),
                            "strict admitted what permissive denies: {op:?} {field} {lit}"
                        );
                    }
                }
            }
        }
    }
}

/// [`evaluate_filter_resolved`], but evaluating every comparison through
/// [`compare_json_op_type_strict`] — the walker an **authorization** expression
/// is evaluated with (TD-FOUNDATION-3 slice FA-a2 / TF-2 S3).
///
/// # Why a separate walker rather than a strict mode on the shared one
///
/// A security predicate and a user's own filter end up ANDed together, but they
/// do not want the same semantics: the user's `clearance >= 3` may legitimately
/// use the permissive total order, while the policy's must not, or a
/// string-valued `clearance` walks through the gate. A single tree evaluated by a
/// single walker cannot give two subtrees two semantics without a marker node in
/// `FilterExpression` — a foundation type used everywhere.
///
/// `AND` associativity makes that unnecessary. Evaluating the two expressions
/// **separately** and requiring both is exactly equivalent to evaluating their
/// conjunction, so the security expression can be walked strictly here while the
/// user's is walked by [`evaluate_filter_resolved`] as before. No shared type
/// changes, and no live query changes meaning.
///
/// 3-valued evaluation of an expression under type-strict comparison.
///
/// `None` is SQL UNKNOWN — produced by an incomparable comparison (different
/// JSON classes; see [`compare_json_op_type_strict`]) **or an absent field**.
/// It propagates through the logical connectives by Kleene semantics and through
/// `Not` as itself, so a negated incomparable/absent comparison stays UNKNOWN
/// (→ deny) rather than flipping to admit. This is the security walker: a row is
/// admitted only on a definite `Some(true)`.
pub fn evaluate_filter_resolved_type_strict_tri<F>(
    expr: &FilterExpression,
    resolve: &F,
) -> Option<bool>
where
    F: Fn(&str) -> Option<serde_json::Value>,
{
    match expr {
        FilterExpression::And(exprs) => {
            let mut saw_unknown = false;
            for e in exprs {
                match evaluate_filter_resolved_type_strict_tri(e, resolve) {
                    Some(false) => return Some(false),
                    None => saw_unknown = true,
                    Some(true) => {}
                }
            }
            if saw_unknown { None } else { Some(true) }
        }
        FilterExpression::Or(exprs) => {
            let mut saw_unknown = false;
            for e in exprs {
                match evaluate_filter_resolved_type_strict_tri(e, resolve) {
                    Some(true) => return Some(true),
                    None => saw_unknown = true,
                    Some(false) => {}
                }
            }
            if saw_unknown { None } else { Some(false) }
        }
        // The fix: Not(UNKNOWN) = UNKNOWN (deny), not `!false` = true (admit).
        FilterExpression::Not(e) => {
            evaluate_filter_resolved_type_strict_tri(e, resolve).map(|b| !b)
        }
        FilterExpression::Comparison {
            field,
            operator,
            value,
        } => match operator {
            ComparisonOperator::IsNull => {
                Some(resolve(field).is_none_or(|json_val| json_val.is_null()))
            }
            ComparisonOperator::IsNotNull => {
                Some(resolve(field).is_some_and(|json_val| !json_val.is_null()))
            }
            _ => match resolve(field) {
                // Absent field → UNKNOWN (None), not Some(false). This is the
                // fix for the absence-axis `Not` leak: a value comparison over a
                // missing field is UNKNOWN, so `Not(UNKNOWN) = UNKNOWN → deny`
                // (via the wrapper's `unwrap_or(false)`), not `Not(false) = admit`.
                // Positive comparisons are unaffected — UNKNOWN also denies. Null
                // tests are handled above (presence-decided) and never reach here.
                None => None,
                Some(json_val) => compare_json_op_type_strict(operator, &json_val, value),
            },
        },
    }
}

/// Type-strict evaluation collapsed to a boolean: UNKNOWN ⇒ **deny**. This is the
/// security-walker contract — see [`evaluate_filter_resolved_type_strict_tri`].
pub fn evaluate_filter_resolved_type_strict<F>(expr: &FilterExpression, resolve: &F) -> bool
where
    F: Fn(&str) -> Option<serde_json::Value>,
{
    evaluate_filter_resolved_type_strict_tri(expr, resolve).unwrap_or(false)
}
#[cfg(test)]
mod type_strict_walker_tests {
    use super::*;
    use serde_json::json;

    fn cmp(
        field: &str,
        operator: ComparisonOperator,
        value: serde_json::Value,
    ) -> FilterExpression {
        FilterExpression::Comparison {
            field: field.to_string(),
            operator,
            value,
        }
    }

    fn admits(expr: &FilterExpression, row: &serde_json::Value) -> bool {
        evaluate_filter_resolved_type_strict(expr, &|f| row.get(f).cloned())
    }
    fn admits_permissive(expr: &FilterExpression, row: &serde_json::Value) -> bool {
        evaluate_filter_resolved(expr, &|f| row.get(f).cloned())
    }

    #[test]
    fn a_clearance_gate_no_longer_admits_a_string_valued_record() {
        // The end-to-end shape of TF-2 §1.4's ClassificationBased case.
        let policy = cmp(
            "clearance",
            ComparisonOperator::GreaterThanOrEqual,
            json!(3),
        );
        let row = json!({ "clearance": "TOP_SECRET" });

        assert!(
            admits_permissive(&policy, &row),
            "characterizing the defect: the permissive walker admits"
        );
        assert!(!admits(&policy, &row));
        // A genuinely-cleared record still passes.
        assert!(admits(&policy, &json!({ "clearance": 5 })));
        assert!(!admits(&policy, &json!({ "clearance": 1 })));
    }

    #[test]
    fn absence_is_handled_exactly_as_in_the_permissive_walker() {
        let row = json!({ "other": 1 });
        for expr in [
            cmp("dept", ComparisonOperator::Equals, json!("eng")),
            cmp("dept", ComparisonOperator::GreaterThan, json!(3)),
            cmp("dept", ComparisonOperator::IsNull, json!(null)),
            cmp("dept", ComparisonOperator::IsNotNull, json!(null)),
        ] {
            assert_eq!(
                admits(&expr, &row),
                admits_permissive(&expr, &row),
                "absence handling diverged for {expr:?}"
            );
        }
    }

    #[test]
    fn the_null_guard_from_fa_a_composes_with_type_strictness() {
        // FA-a lowers `Not(P)` as `And([Not(P), IsNotNull(f)])`. Together with
        // strict comparison that closes both the absence axis and the type axis.
        let guarded = FilterExpression::And(vec![
            FilterExpression::Not(Box::new(cmp(
                "clearance",
                ComparisonOperator::Equals,
                json!(3),
            ))),
            cmp("clearance", ComparisonOperator::IsNotNull, json!(null)),
        ]);

        assert!(!admits(&guarded, &json!({})), "absent field excluded");
        assert!(
            !admits(&guarded, &json!({ "clearance": null })),
            "null field excluded"
        );
        assert!(admits(&guarded, &json!({ "clearance": 5 })));
        assert!(!admits(&guarded, &json!({ "clearance": 3 })));
    }

    // ---- Red-team (2026-07-28): the Not(comparison) fail-open class ----

    #[test]
    fn not_of_equals_over_a_cross_class_field_no_longer_admits() {
        // Finding 2: `Not(Equals(owner, "u/alice"))` admitted {owner: 42} because
        // Equals (cross-class) was false and `!false` admitted. The permissive
        // walker still admits; the strict walker must deny.
        let policy = FilterExpression::Not(Box::new(cmp(
            "owner",
            ComparisonOperator::Equals,
            json!("u/alice"),
        )));
        let row = json!({ "owner": 42 });

        assert!(
            admits_permissive(&policy, &row),
            "characterizing the leak: permissive Not(Equals) admits cross-class"
        );
        assert!(
            !admits(&policy, &row),
            "strict Not(Equals) over a cross-class field must deny (UNKNOWN, not admit)"
        );
    }

    #[test]
    fn not_of_an_ordered_comparison_over_a_cross_class_field_denies() {
        // The leak generalized beyond Equals: ANY strict comparison returns false
        // for cross-class, and a bare `!` flipped it to admit.
        for op in [
            ComparisonOperator::GreaterThan,
            ComparisonOperator::GreaterThanOrEqual,
            ComparisonOperator::LessThan,
            ComparisonOperator::NotEquals,
        ] {
            let policy = FilterExpression::Not(Box::new(cmp("clearance", op.clone(), json!(3))));
            assert!(
                !admits(&policy, &json!({ "clearance": "TS" })),
                "strict Not({op:?}) over a cross-class field must deny"
            );
        }
    }

    #[test]
    fn the_tri_walker_carries_incomparable_as_unknown_not_false() {
        // The mechanism: compare_json_op_type_strict returns None for cross-class,
        // and Not(None) = None (UNKNOWN), which collapses to deny — not `!false`.
        let resolve = |f: &str| match f {
            "owner" => Some(json!(42)),
            _ => None,
        };
        let eq = cmp("owner", ComparisonOperator::Equals, json!("u/alice"));
        // The bare comparison is incomparable, not false.
        assert_eq!(
            evaluate_filter_resolved_type_strict_tri(&eq, &resolve),
            None,
            "a cross-class comparison is UNKNOWN, not false"
        );
        // And negating UNKNOWN stays UNKNOWN (→ deny), never admits.
        assert_eq!(
            evaluate_filter_resolved_type_strict_tri(
                &FilterExpression::Not(Box::new(eq)),
                &resolve
            ),
            None
        );
    }

    #[test]
    fn not_of_in_over_a_wholly_incomparable_list_denies() {
        // Not(In(owner, ["u/alice","u/bob"])) over {owner: 42}: the list has no
        // Number element, so the In question is incomparable → Not stays UNKNOWN → deny.
        let policy = FilterExpression::Not(Box::new(cmp(
            "owner",
            ComparisonOperator::In,
            json!(["u/alice", "u/bob"]),
        )));
        assert!(!admits(&policy, &json!({ "owner": 42 })));
        // Positive In over a comparable list element still works.
        assert!(admits(
            &cmp("owner", ComparisonOperator::In, json!([42, 7])),
            &json!({ "owner": 42 })
        ));
    }

    #[test]
    fn and_associativity_makes_separate_evaluation_equivalent_to_conjunction() {
        // The property the two-walker design rests on: evaluating a conjunction
        // is the same as evaluating each conjunct and requiring both. Verified
        // over a grid so the design note is not just prose.
        let security = cmp("dept", ComparisonOperator::Equals, json!("eng"));
        let user = cmp("score", ComparisonOperator::GreaterThan, json!(0.5));
        let conjunction = FilterExpression::And(vec![security.clone(), user.clone()]);

        for row in [
            json!({ "dept": "eng", "score": 0.9 }),
            json!({ "dept": "eng", "score": 0.1 }),
            json!({ "dept": "hr", "score": 0.9 }),
            json!({}),
            json!({ "dept": "eng" }),
            json!({ "score": 0.9 }),
        ] {
            assert_eq!(
                admits(&conjunction, &row),
                admits(&security, &row) && admits(&user, &row),
                "separate evaluation diverged from the conjunction for {row}"
            );
        }
    }

    #[test]
    fn the_strict_walker_never_admits_more_than_the_permissive_one() {
        // Deny-biased over a grid of expression shapes × rows.
        let exprs = vec![
            cmp("a", ComparisonOperator::GreaterThanOrEqual, json!(3)),
            cmp("a", ComparisonOperator::NotEquals, json!(3)),
            cmp("a", ComparisonOperator::NotIn, json!([1, 2])),
            cmp("a", ComparisonOperator::Between, json!([1, 10])),
            FilterExpression::Not(Box::new(cmp("a", ComparisonOperator::Equals, json!(3)))),
            FilterExpression::And(vec![
                cmp("a", ComparisonOperator::GreaterThan, json!(1)),
                cmp("b", ComparisonOperator::Equals, json!("x")),
            ]),
            FilterExpression::Or(vec![
                cmp("a", ComparisonOperator::LessThan, json!(1)),
                cmp("b", ComparisonOperator::NotEquals, json!("x")),
            ]),
        ];
        let rows = vec![
            json!({}),
            json!({ "a": 5 }),
            json!({ "a": "5" }),
            json!({ "a": null }),
            json!({ "a": true }),
            json!({ "a": 5, "b": "x" }),
            json!({ "a": "5", "b": 7 }),
        ];
        for e in &exprs {
            for row in &rows {
                if admits(e, row) {
                    assert!(
                        admits_permissive(e, row),
                        "strict walker admitted what permissive denies: {e:?} {row}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod type_strict_proxima_tests {
    use super::*;
    use proximadb_data_model::ProximaValue;
    use serde_json::json;

    fn tree(pairs: Vec<(&str, ProximaValue)>) -> ProximaTree {
        let mut t = ProximaTree::new();
        for (k, v) in pairs {
            t.insert(k.to_string(), ProximaTreeNode::Value(v));
        }
        t
    }

    #[test]
    fn the_canonical_seam_denies_a_string_clearance_against_a_numeric_gate() {
        let policy = FilterExpression::Comparison {
            field: "clearance".to_string(),
            operator: ComparisonOperator::GreaterThanOrEqual,
            value: json!(3),
        };

        // ProximaValue path — no SqlValue envelope anywhere.
        let string_clearance = tree(vec![(
            "clearance",
            ProximaValue::String("TOP_SECRET".to_string()),
        )]);
        assert!(
            evaluate_filter_proxima(&policy, &string_clearance),
            "characterizing the defect on the canonical seam"
        );
        assert!(!evaluate_filter_proxima_type_strict(
            &policy,
            &string_clearance
        ));

        // A genuinely-cleared record still passes.
        assert!(evaluate_filter_proxima_type_strict(
            &policy,
            &tree(vec![("clearance", ProximaValue::Int64(5))])
        ));
        assert!(!evaluate_filter_proxima_type_strict(
            &policy,
            &tree(vec![("clearance", ProximaValue::Int64(1))])
        ));
    }

    #[test]
    fn the_canonical_seam_matches_the_default_walker_within_a_class() {
        let policy = FilterExpression::Comparison {
            field: "dept".to_string(),
            operator: ComparisonOperator::Equals,
            value: json!("eng"),
        };
        for props in [
            tree(vec![("dept", ProximaValue::String("eng".to_string()))]),
            tree(vec![("dept", ProximaValue::String("hr".to_string()))]),
            tree(vec![]),
        ] {
            assert_eq!(
                evaluate_filter_proxima_type_strict(&policy, &props),
                evaluate_filter_proxima(&policy, &props),
                "type strictness changed a same-class comparison"
            );
        }
    }
}
