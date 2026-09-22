//! # proxima_arrow — canonical `ProximaRecord` → Arrow `RecordBatch` converter (F1.5)
//!
//! The single, schema-driven home for turning canonical [`ProximaRecord`]s into an Arrow
//! [`RecordBatch`]. It is the data-plane companion to [`ProximaSchema::to_arrow_schema`]
//! (the *schema* plane): the Arrow schema is taken verbatim from the [`ProximaSchema`], and
//! each column's values are materialized into the exact Arrow array type that
//! [`ProximaColumn::to_arrow_type`] declares — so the produced batch always satisfies
//! `RecordBatch::try_new` against `schema.to_arrow_schema()`.
//!
//! ## Why this exists
//!
//! Three ad-hoc converters predate this one and each loses information:
//! - `src/connectors/spark.rs` (`proxima_records_to_record_batch`) — **id + vector only**;
//!   per-record `props` were an explicit "follow-up".
//! - `src/storage/formats/arrow_conversion.rs` — `VectorRecord`-based (legacy edge type).
//! - `codec.rs::vector_records_to_batch` — collapses all props into one `StructArray`.
//!
//! This module fixes the prop-collapse class at the source: every non-deleted
//! [`ProximaColumn`] becomes its own typed, null-masked Arrow column, read from
//! `record.props` (via [`tree_get`]) for scalar columns and from `record.embeddings` for the
//! vector column. It unblocks the warehouse base-tier write path (F2
//! `write_records_to_parquet` = this converter + a parquet write over `ProximaObjectStore`)
//! and read-side schema/row fidelity (F5).
//!
//! ## Mapping contract
//!
//! Columns are emitted in `schema.columns` order, skipping tombstoned (`is_deleted`) ones,
//! matching `to_arrow_schema`. For each column:
//! - **Scalar columns** read `tree_get(&record.props, &col.name)`. An absent key, a
//!   [`ProximaValue::Null`], or a value that cannot be coerced to the column's family yields
//!   an Arrow null (NOT an error — sparse props are normal). Cross-family coercions are
//!   modest and lossless-widening only (any integer → a wider int; any int/float → float).
//! - **Vector columns** ([`ProximaType::DenseVector`]) read `record.embeddings.first()` and
//!   encode the fp32 values as little-endian bytes into a `FixedSizeBinary(dimension * 4)`
//!   array — exactly the Arrow type `to_arrow_type` declares. A record with no embedding is
//!   null; a dimension mismatch is a hard error (it would corrupt the fixed-width layout).
//!
//! Complex/temporal/decimal column types that have no scalar `props` representation yet
//! return a [`StorageError::Serialization`] (with an "unsupported column type" message)
//! rather than silently emitting nulls, so the gap is visible to the caller.

use std::sync::Arc;

use arrow_array::builder::{
    BinaryBuilder, BooleanBuilder, Date32Builder, FixedSizeBinaryBuilder, Float32Builder,
    Float64Builder, Int8Builder, Int16Builder, Int32Builder, Int64Builder, StringBuilder,
    UInt8Builder, UInt16Builder, UInt32Builder, UInt64Builder,
};
use std::collections::HashMap;

use arrow_array::{
    Array, ArrayRef, BinaryArray, BooleanArray, Date32Array, FixedSizeBinaryArray, Float32Array,
    Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, LargeStringArray, RecordBatch,
    StringArray, TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
use proximadb_kernel::error::StorageError;
use proximadb_records::{
    EmbeddingCell, EmbeddingValues, ProximaRecord, ProximaTreeNode, ProximaValue, tree_get,
};

use proximadb_data_model::{ProximaType, TimeUnit as DmTimeUnit, VectorElement as DmVectorElement};

use crate::proxima_schema::{ProximaColumn, ProximaSchema};

/// Convert a slice of canonical [`ProximaRecord`]s into an Arrow [`RecordBatch`] whose schema
/// is `schema.to_arrow_schema()` and whose columns are materialized per the mapping contract
/// documented at the module level.
///
/// Returns a [`StorageError::Serialization`] for column types without a scalar `props`
/// materialization yet, and [`StorageError::InvalidDimension`] when an embedding's length
/// does not match its declared vector dimension.
pub fn proxima_records_to_record_batch(
    records: &[ProximaRecord],
    schema: &ProximaSchema,
) -> Result<RecordBatch, StorageError> {
    let arrow_schema = schema.to_arrow_schema();

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(arrow_schema.fields().len());
    for col in schema.columns.iter().filter(|c| !c.is_deleted) {
        columns.push(build_column(records, col)?);
    }

    RecordBatch::try_new(arrow_schema, columns).map_err(|e| {
        StorageError::Serialization(format!("proxima_arrow: RecordBatch::try_new failed: {e}"))
    })
}

// ---------------------------------------------------------------------------
// TD-USUB-11 — reserved system columns for a FAITHFUL canonical-record round trip
// ---------------------------------------------------------------------------
//
// `proxima_records_to_record_batch` above materializes **only declared schema
// columns**, reading each value out of `props`. Every field that lives on
// `ProximaRecord` itself rather than in `props` — identity and lifecycle — is
// therefore absent from the Parquet copy. `reconstruct_oid` on the record-store
// read path exists precisely because of that gap.
//
// That is survivable while Parquet is only an *export* of rows already filtered
// live: nothing downstream needs the lifecycle fields back. It is NOT survivable
// once Parquet is a **landing** format (ADR-094 Decision 1), because a landed
// segment is immutable — deleting a row already written can no longer remove it,
// so deletion must become a `valid_to_ns` tombstone, and a tombstone that does
// not round-trip is a **resurrected row**: the silent wrong answer mandate #1
// forbids. A faithful encoding is therefore a prerequisite for durable
// relational storage (TD-USUB-1 slice 2), not an optimisation.
//
// **Mixed-read-safe (mandate #8).** These columns are additive and opt-in: the
// legacy writer is untouched and still byte-identical, and the reader lifts a
// system column only when present, falling back to exactly today's behaviour
// otherwise. No flag day and no version byte to coordinate — presence of the
// column IS the marker.

/// Canonical record `oid` (identity). Without it, identity depends on the table
/// declaring a primary key, which `reconstruct_oid` can only infer.
pub const SYS_COL_OID: &str = "__proxima_oid";
/// Canonical record `tenant_id`. Isolation is structural (path/partition), so
/// this is fidelity rather than enforcement — but a record that returns with an
/// empty tenant is not the record that was written.
pub const SYS_COL_TENANT_ID: &str = "__proxima_tenant_id";
/// Canonical record `variation_id`.
pub const SYS_COL_VARIATION_ID: &str = "__proxima_variation_id";
/// Canonical record `created_at_ns`.
pub const SYS_COL_CREATED_AT_NS: &str = "__proxima_created_at_ns";
/// Canonical record `updated_at_ns`.
pub const SYS_COL_UPDATED_AT_NS: &str = "__proxima_updated_at_ns";
/// Canonical record `valid_from_ns`.
pub const SYS_COL_VALID_FROM_NS: &str = "__proxima_valid_from_ns";
/// Canonical record `valid_to_ns` — **the load-bearing one**. This is the
/// tombstone/TTL field the canonical dead-record predicate reads
/// (`ProximaRecord::is_dead`, mandate #16a). Losing it turns a deleted or expired
/// row into a live one on read-back.
pub const SYS_COL_VALID_TO_NS: &str = "__proxima_valid_to_ns";

/// Every reserved system column, in the order they are appended to a batch.
pub const SYSTEM_COLUMNS: &[&str] = &[
    SYS_COL_OID,
    SYS_COL_TENANT_ID,
    SYS_COL_VARIATION_ID,
    SYS_COL_CREATED_AT_NS,
    SYS_COL_UPDATED_AT_NS,
    SYS_COL_VALID_FROM_NS,
    SYS_COL_VALID_TO_NS,
];

/// True iff `name` is a reserved system column.
///
/// Callers that surface user-visible column lists (projection, `SELECT *`, schema
/// introspection) must exclude these so the encoding stays invisible to SQL.
pub fn is_system_column(name: &str) -> bool {
    SYSTEM_COLUMNS.contains(&name)
}

/// Like [`proxima_records_to_record_batch`], but appends the reserved system
/// columns so the batch round-trips a canonical record **faithfully** — identity
/// and lifecycle included.
///
/// Use this wherever the Parquet object is authoritative for live state (a
/// landing segment). The plain encoder stays correct, and byte-identical, for an
/// export of rows that were already filtered live.
pub fn proxima_records_to_record_batch_with_system_columns(
    records: &[ProximaRecord],
    schema: &ProximaSchema,
) -> Result<RecordBatch, StorageError> {
    let user_schema = schema.to_arrow_schema();

    // A declared user column colliding with a reserved name would put two Arrow
    // fields of the same name in one batch, making read-back ambiguous. Fail
    // closed and name the offender rather than silently shadowing one of them.
    if let Some(clash) = user_schema
        .fields()
        .iter()
        .find(|f| is_system_column(f.name()))
    {
        return Err(StorageError::Serialization(format!(
            "proxima_arrow: user column '{}' collides with a reserved system column; \
             reserved names are {SYSTEM_COLUMNS:?}",
            clash.name()
        )));
    }

    let mut fields: Vec<ArrowField> = user_schema.fields().iter().map(|f| (**f).clone()).collect();
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(fields.len() + SYSTEM_COLUMNS.len());
    for col in schema.columns.iter().filter(|c| !c.is_deleted) {
        columns.push(build_column(records, col)?);
    }

    let str_col = |values: Vec<Option<String>>| -> ArrayRef {
        let mut b = StringBuilder::new();
        for v in values {
            match v {
                Some(s) => b.append_value(s),
                None => b.append_null(),
            }
        }
        Arc::new(b.finish()) as ArrayRef
    };
    let i64_col = |values: Vec<Option<i64>>| -> ArrayRef {
        let mut b = Int64Builder::with_capacity(values.len());
        for v in values {
            match v {
                Some(x) => b.append_value(x),
                None => b.append_null(),
            }
        }
        Arc::new(b.finish()) as ArrayRef
    };

    let sys: Vec<(&str, ArrowDataType, ArrayRef)> = vec![
        (
            SYS_COL_OID,
            ArrowDataType::Utf8,
            str_col(records.iter().map(|r| Some(r.oid.clone())).collect()),
        ),
        (
            SYS_COL_TENANT_ID,
            ArrowDataType::Utf8,
            str_col(records.iter().map(|r| Some(r.tenant_id.clone())).collect()),
        ),
        (
            SYS_COL_VARIATION_ID,
            ArrowDataType::Utf8,
            str_col(records.iter().map(|r| r.variation_id.clone()).collect()),
        ),
        (
            SYS_COL_CREATED_AT_NS,
            ArrowDataType::Int64,
            i64_col(records.iter().map(|r| Some(r.created_at_ns)).collect()),
        ),
        (
            SYS_COL_UPDATED_AT_NS,
            ArrowDataType::Int64,
            i64_col(records.iter().map(|r| Some(r.updated_at_ns)).collect()),
        ),
        (
            SYS_COL_VALID_FROM_NS,
            ArrowDataType::Int64,
            i64_col(records.iter().map(|r| r.valid_from_ns).collect()),
        ),
        (
            SYS_COL_VALID_TO_NS,
            ArrowDataType::Int64,
            i64_col(records.iter().map(|r| r.valid_to_ns).collect()),
        ),
    ];
    for (name, dt, array) in sys {
        fields.push(ArrowField::new(name, dt, true));
        columns.push(array);
    }

    RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), columns).map_err(|e| {
        StorageError::Serialization(format!(
            "proxima_arrow: RecordBatch::try_new (with system columns) failed: {e}"
        ))
    })
}

/// Build one Arrow array for `col` over every record, in row order.
fn build_column(records: &[ProximaRecord], col: &ProximaColumn) -> Result<ArrayRef, StorageError> {
    let n = records.len();
    macro_rules! scalar_col {
        ($builder:ident, $extract:expr) => {{
            let mut b = $builder::with_capacity(n);
            for r in records {
                match tree_get(&r.props, &col.name) {
                    Some(v) => match ($extract)(v) {
                        Some(x) => b.append_value(x),
                        None => b.append_null(),
                    },
                    None => b.append_null(),
                }
            }
            Ok(Arc::new(b.finish()) as ArrayRef)
        }};
    }

    match &col.data_type {
        ProximaType::Boolean => scalar_col!(BooleanBuilder, pv_as_bool),
        ProximaType::Int8 => scalar_col!(Int8Builder, |v| pv_as_i64(v).map(|x| x as i8)),
        ProximaType::Int16 => scalar_col!(Int16Builder, |v| pv_as_i64(v).map(|x| x as i16)),
        ProximaType::Int32 => scalar_col!(Int32Builder, |v| pv_as_i64(v).map(|x| x as i32)),
        ProximaType::Int64 => scalar_col!(Int64Builder, pv_as_i64),
        ProximaType::UInt8 => scalar_col!(UInt8Builder, |v| pv_as_u64(v).map(|x| x as u8)),
        ProximaType::UInt16 => scalar_col!(UInt16Builder, |v| pv_as_u64(v).map(|x| x as u16)),
        ProximaType::UInt32 => scalar_col!(UInt32Builder, |v| pv_as_u64(v).map(|x| x as u32)),
        ProximaType::UInt64 => scalar_col!(UInt64Builder, pv_as_u64),
        ProximaType::Float32 => scalar_col!(Float32Builder, |v| pv_as_f64(v).map(|x| x as f32)),
        ProximaType::Float64 => scalar_col!(Float64Builder, pv_as_f64),
        ProximaType::Date => scalar_col!(Date32Builder, pv_as_date32),
        // Temporal: build in the column's DECLARED unit, converting each stored
        // value from its own unit (INSERT lowers DATE/TIMESTAMP literals to ms).
        // TimestampTz carries a "UTC" zone so the Arrow type matches `to_arrow_type`.
        ProximaType::Timestamp(unit) => build_timestamp_column(records, col, *unit, false),
        ProximaType::TimestampTz(unit) => build_timestamp_column(records, col, *unit, true),
        // Utf8-backed types per `to_arrow_type` (String, Json).
        ProximaType::String | ProximaType::Json => {
            let mut b = StringBuilder::new();
            for r in records {
                match tree_get(&r.props, &col.name).and_then(pv_as_string) {
                    Some(s) => b.append_value(s),
                    None => b.append_null(),
                }
            }
            Ok(Arc::new(b.finish()) as ArrayRef)
        }
        ProximaType::Binary => {
            let mut b = BinaryBuilder::new();
            for r in records {
                match tree_get(&r.props, &col.name).and_then(pv_as_binary) {
                    Some(bytes) => b.append_value(bytes),
                    None => b.append_null(),
                }
            }
            Ok(Arc::new(b.finish()) as ArrayRef)
        }
        ProximaType::DenseVector { dim, .. } => build_vector_column(records, col, *dim as u32),
        other => Err(StorageError::Serialization(format!(
            "proxima_arrow: column `{}` has type {:?}, which has no canonical ProximaValue→Arrow \
             materialization yet",
            col.name, other
        ))),
    }
}

/// Build the dense-vector column as `FixedSizeBinary(dimension * 4)`, encoding each record's
/// first embedding as little-endian fp32 bytes (the layout `to_arrow_type` declares for
/// [`ProximaType::DenseVector`]). Absent embedding → null; length mismatch → hard error.
///
/// NOTE (ADR-024 Step 2b / TD-111 follow-up): this data-plane writer always emits fp32
/// (`dimension * 4`), while `ProximaType::DenseVector::to_arrow_type` now declares
/// `dimension * element.byte_width()`. For Float32 (the only element produced in practice)
/// the declared and written widths agree; a non-f32 element would declare a wider schema than
/// the bytes written. Latent, not a live regression — typed-element encoding is the follow-up.
fn build_vector_column(
    records: &[ProximaRecord],
    col: &ProximaColumn,
    dimension: u32,
) -> Result<ArrayRef, StorageError> {
    let byte_width = (dimension as usize) * 4;
    let mut b = FixedSizeBinaryBuilder::with_capacity(records.len(), byte_width as i32);
    for r in records {
        match r.embeddings.first() {
            Some(cell) => {
                let vals = cell.values.to_fp32_owned();
                if vals.len() != dimension as usize {
                    return Err(StorageError::InvalidDimension {
                        expected: dimension as usize,
                        actual: vals.len(),
                    });
                }
                let mut bytes = Vec::with_capacity(byte_width);
                for f in &vals {
                    bytes.extend_from_slice(&f.to_le_bytes());
                }
                b.append_value(&bytes).map_err(|e| {
                    StorageError::Serialization(format!(
                        "proxima_arrow: vector column `{}` append failed: {e}",
                        col.name
                    ))
                })?;
            }
            None => b.append_null(),
        }
    }
    Ok(Arc::new(b.finish()) as ArrayRef)
}

/// Nanoseconds per one tick of a [`DmTimeUnit`].
fn time_unit_nanos(u: DmTimeUnit) -> i64 {
    match u {
        DmTimeUnit::Second => 1_000_000_000,
        DmTimeUnit::Millisecond => 1_000_000,
        DmTimeUnit::Microsecond => 1_000,
        DmTimeUnit::Nanosecond => 1,
    }
}

/// Convert a tick count from `from` unit to `to` unit (via nanoseconds).
fn convert_ticks(val: i64, from: DmTimeUnit, to: DmTimeUnit) -> i64 {
    val.saturating_mul(time_unit_nanos(from)) / time_unit_nanos(to)
}

/// Build a Timestamp(unit[, UTC]) column, converting each stored value into the
/// column's declared `unit`. A bare `Int64` is assumed already in `unit`.
fn build_timestamp_column(
    records: &[ProximaRecord],
    col: &ProximaColumn,
    unit: DmTimeUnit,
    tz: bool,
) -> Result<ArrayRef, StorageError> {
    let vals: Vec<Option<i64>> = records
        .iter()
        .map(|r| {
            tree_get(&r.props, &col.name).and_then(|v| match v {
                ProximaValue::Timestamp(t, u) | ProximaValue::TimestampTz(t, u) => {
                    Some(convert_ticks(*t, *u, unit))
                }
                ProximaValue::Int64(t) => Some(*t),
                _ => None,
            })
        })
        .collect();
    let tzopt: Option<String> = if tz { Some("UTC".to_string()) } else { None };
    let arr: ArrayRef = match unit {
        DmTimeUnit::Second => Arc::new(TimestampSecondArray::from(vals).with_timezone_opt(tzopt)),
        DmTimeUnit::Millisecond => {
            Arc::new(TimestampMillisecondArray::from(vals).with_timezone_opt(tzopt))
        }
        DmTimeUnit::Microsecond => {
            Arc::new(TimestampMicrosecondArray::from(vals).with_timezone_opt(tzopt))
        }
        DmTimeUnit::Nanosecond => {
            Arc::new(TimestampNanosecondArray::from(vals).with_timezone_opt(tzopt))
        }
    };
    Ok(arr)
}

// ---------------------------------------------------------------------------
// ProximaValue extractors (None = emit Arrow null; modest, widening-only coercions)
// ---------------------------------------------------------------------------

fn pv_as_bool(v: &ProximaValue) -> Option<bool> {
    match v {
        ProximaValue::Boolean(b) => Some(*b),
        _ => None,
    }
}

/// Any signed/unsigned integer variant widened to `i64`. (`UInt64` is reinterpreted; values
/// above `i64::MAX` are vanishingly rare for relational keys and would only matter for a true
/// `UInt64` column, which routes through [`pv_as_u64`] instead.)
fn pv_as_i64(v: &ProximaValue) -> Option<i64> {
    match v {
        ProximaValue::Int8(x) => Some(*x as i64),
        ProximaValue::Int16(x) => Some(*x as i64),
        ProximaValue::Int32(x) => Some(*x as i64),
        ProximaValue::Int64(x) => Some(*x),
        ProximaValue::UInt8(x) => Some(*x as i64),
        ProximaValue::UInt16(x) => Some(*x as i64),
        ProximaValue::UInt32(x) => Some(*x as i64),
        ProximaValue::UInt64(x) => Some(*x as i64),
        _ => None,
    }
}

fn pv_as_u64(v: &ProximaValue) -> Option<u64> {
    match v {
        ProximaValue::UInt8(x) => Some(*x as u64),
        ProximaValue::UInt16(x) => Some(*x as u64),
        ProximaValue::UInt32(x) => Some(*x as u64),
        ProximaValue::UInt64(x) => Some(*x),
        ProximaValue::Int8(x) if *x >= 0 => Some(*x as u64),
        ProximaValue::Int16(x) if *x >= 0 => Some(*x as u64),
        ProximaValue::Int32(x) if *x >= 0 => Some(*x as u64),
        ProximaValue::Int64(x) if *x >= 0 => Some(*x as u64),
        _ => None,
    }
}

/// Any float or integer variant widened to `f64`.
fn pv_as_f64(v: &ProximaValue) -> Option<f64> {
    match v {
        ProximaValue::Float16(x) | ProximaValue::Float32(x) => Some(*x as f64),
        ProximaValue::Float64(x) => Some(*x),
        ProximaValue::Int8(x) => Some(*x as f64),
        ProximaValue::Int16(x) => Some(*x as f64),
        ProximaValue::Int32(x) => Some(*x as f64),
        ProximaValue::Int64(x) => Some(*x as f64),
        ProximaValue::UInt8(x) => Some(*x as f64),
        ProximaValue::UInt16(x) => Some(*x as f64),
        ProximaValue::UInt32(x) => Some(*x as f64),
        ProximaValue::UInt64(x) => Some(*x as f64),
        _ => None,
    }
}

fn pv_as_string(v: &ProximaValue) -> Option<String> {
    match v {
        ProximaValue::String(s) | ProximaValue::Symbol(s) | ProximaValue::Decimal(s) => {
            Some(s.clone())
        }
        ProximaValue::Json(j) | ProximaValue::Jsonb(j) => Some(j.to_string()),
        ProximaValue::Uuid(b) | ProximaValue::ULID(b) => Some(format_uuid(b)),
        _ => None,
    }
}

fn pv_as_binary(v: &ProximaValue) -> Option<&[u8]> {
    match v {
        ProximaValue::Binary(b) | ProximaValue::BinaryVector(b) => Some(b.as_slice()),
        _ => None,
    }
}

fn pv_as_date32(v: &ProximaValue) -> Option<i32> {
    match v {
        ProximaValue::Date(d) => Some(*d),
        ProximaValue::Int32(d) => Some(*d),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Schema inference (for schema-less write paths, e.g. ObjectStoreBridge)
// ---------------------------------------------------------------------------

/// Column name synthesized for the dense-vector column when inferring a schema.
///
/// [`proxima_records_to_record_batch`] reads the vector from `record.embeddings.first()`
/// regardless of the column's *name*, so the exact name is arbitrary — this constant just
/// keeps it stable across a write→read round-trip.
pub const INFERRED_VECTOR_COLUMN: &str = "vector";

/// Map a concrete [`ProximaValue`] to the [`ProximaType`] the canonical converter can
/// materialize. Returns `None` for `Null` and for variants without a scalar Arrow column in
/// [`build_column`] (Decimal — needs precision/scale; Time/Timestamp; Array/Map/Struct;
/// identifier (Uuid/ULID) and sparse/quantized vectors) so inference never produces a column
/// the converter would reject.
fn proxima_value_to_data_type(v: &ProximaValue) -> Option<ProximaType> {
    Some(match v {
        ProximaValue::Boolean(_) => ProximaType::Boolean,
        ProximaValue::Int8(_) => ProximaType::Int8,
        ProximaValue::Int16(_) => ProximaType::Int16,
        ProximaValue::Int32(_) => ProximaType::Int32,
        ProximaValue::Int64(_) => ProximaType::Int64,
        ProximaValue::UInt8(_) => ProximaType::UInt8,
        ProximaValue::UInt16(_) => ProximaType::UInt16,
        ProximaValue::UInt32(_) => ProximaType::UInt32,
        ProximaValue::UInt64(_) => ProximaType::UInt64,
        ProximaValue::Float16(_) | ProximaValue::Float32(_) => ProximaType::Float32,
        ProximaValue::Float64(_) => ProximaType::Float64,
        ProximaValue::String(_) | ProximaValue::Symbol(_) => ProximaType::String,
        ProximaValue::Binary(_) => ProximaType::Binary,
        ProximaValue::Date(_) => ProximaType::Date,
        ProximaValue::Json(_) | ProximaValue::Jsonb(_) => ProximaType::Json,
        _ => return None,
    })
}

/// Best-effort [`ProximaSchema`] inference from a batch of records — for schema-less write
/// paths such as `ObjectStoreBridge::write_records_to_parquet`, which take only `&[ProximaRecord]`
/// (the records are self-describing).
///
/// Rules (conservative — emits only column types [`proxima_records_to_record_batch`] can
/// materialize, so inference + conversion never disagree):
/// - Scalar columns are the union of top-level `props` **leaf** keys, in first-seen order; a
///   column's type is taken from the first non-`Null`, materializable [`ProximaValue`] observed
///   for that key. Keys that are only ever `Null`/complex/unsupported are skipped (documented
///   loss). Nested `Object` props are not flattened in v1.
/// - Every column is nullable (a key absent from some record is a null there).
/// - If any record carries an embedding, one dense-vector column named [`INFERRED_VECTOR_COLUMN`]
///   is appended, its dimension = the first embedding's length, element type `Float32`.
pub fn infer_proxima_schema(records: &[ProximaRecord]) -> ProximaSchema {
    let mut order: Vec<String> = Vec::new();
    let mut types: HashMap<String, ProximaType> = HashMap::new();

    for r in records {
        for (key, node) in &r.props {
            if types.contains_key(key) {
                continue;
            }
            // Null / unsupported leaf: leave unlocked so a later record's concrete value
            // for the same key can still define the column.
            let ProximaTreeNode::Value(v) = node else {
                continue;
            };
            let Some(dt) = proxima_value_to_data_type(v) else {
                continue;
            };
            order.push(key.clone());
            types.insert(key.clone(), dt);
        }
    }

    let mut columns: Vec<ProximaColumn> = Vec::with_capacity(order.len() + 1);
    let mut next_id = 1;
    let mut push = |name: String, data_type: ProximaType, id: &mut i32| {
        columns.push(ProximaColumn {
            id: *id,
            object_id: None,
            name,
            data_type,
            nullable: true,
            default_value: None,
            comment: None,
            properties: HashMap::new(),
            is_deleted: false,
            original_id: None,
        });
        *id += 1;
    };

    for name in order {
        let dt = types.remove(&name).expect("type recorded for ordered key");
        push(name, dt, &mut next_id);
    }

    if let Some(dim) = records
        .iter()
        .find_map(|r| r.embeddings.first().map(|c| c.values.len()))
    {
        push(
            INFERRED_VECTOR_COLUMN.to_string(),
            ProximaType::DenseVector {
                element: DmVectorElement::Float32,
                dim,
            },
            &mut next_id,
        );
    }

    ProximaSchema::from_columns("inferred".to_string(), columns, Vec::new())
}

/// Render a 16-byte UUID/ULID as the canonical hyphenated hex form.
fn format_uuid(b: &[u8; 16]) -> String {
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        b[0],
        b[1],
        b[2],
        b[3],
        b[4],
        b[5],
        b[6],
        b[7],
        b[8],
        b[9],
        b[10],
        b[11],
        b[12],
        b[13],
        b[14],
        b[15]
    )
}

// ---------------------------------------------------------------------------
// Reverse conversion: Arrow RecordBatch -> ProximaRecord (S7 canonical inverse)
// ---------------------------------------------------------------------------

/// Map a single Arrow cell to a [`ProximaValue`] — the scalar inverse of the
/// per-family encoders in [`proxima_records_to_record_batch`]. Null cells and
/// array types without a scalar `ProximaValue` (lists/structs/the vector column)
/// yield `None`, so the key is simply omitted from `props` (sparse props are
/// normal). This is the single canonical Arrow→ProximaValue mapper; consumers
/// reading Proxima-written batches must reuse it rather than re-deriving one.
pub fn arrow_cell_to_proxima_value(array: &ArrayRef, row: usize) -> Option<ProximaValue> {
    if array.is_null(row) {
        return None;
    }
    let any = array.as_any();
    if let Some(a) = any.downcast_ref::<StringArray>() {
        Some(ProximaValue::String(a.value(row).to_string()))
    } else if let Some(a) = any.downcast_ref::<LargeStringArray>() {
        Some(ProximaValue::String(a.value(row).to_string()))
    } else if let Some(a) = any.downcast_ref::<BooleanArray>() {
        Some(ProximaValue::Boolean(a.value(row)))
    } else if let Some(a) = any.downcast_ref::<Int8Array>() {
        Some(ProximaValue::Int8(a.value(row)))
    } else if let Some(a) = any.downcast_ref::<Int16Array>() {
        Some(ProximaValue::Int16(a.value(row)))
    } else if let Some(a) = any.downcast_ref::<Int32Array>() {
        Some(ProximaValue::Int32(a.value(row)))
    } else if let Some(a) = any.downcast_ref::<Int64Array>() {
        Some(ProximaValue::Int64(a.value(row)))
    } else if let Some(a) = any.downcast_ref::<UInt8Array>() {
        Some(ProximaValue::UInt8(a.value(row)))
    } else if let Some(a) = any.downcast_ref::<UInt16Array>() {
        Some(ProximaValue::UInt16(a.value(row)))
    } else if let Some(a) = any.downcast_ref::<UInt32Array>() {
        Some(ProximaValue::UInt32(a.value(row)))
    } else if let Some(a) = any.downcast_ref::<UInt64Array>() {
        Some(ProximaValue::UInt64(a.value(row)))
    } else if let Some(a) = any.downcast_ref::<Float32Array>() {
        Some(ProximaValue::Float32(a.value(row)))
    } else if let Some(a) = any.downcast_ref::<Float64Array>() {
        Some(ProximaValue::Float64(a.value(row)))
    } else if let Some(a) = any.downcast_ref::<Date32Array>() {
        Some(ProximaValue::Date(a.value(row)))
    } else {
        any.downcast_ref::<BinaryArray>()
            .map(|a| ProximaValue::Binary(a.value(row).to_vec()))
    }
}

/// Reverse of [`proxima_records_to_record_batch`]: materialize the rows of an
/// Arrow [`RecordBatch`] back into canonical [`ProximaRecord`]s. Scalar columns
/// become `props` (via [`arrow_cell_to_proxima_value`]); a `FixedSizeBinary`
/// column is decoded as the little-endian fp32 dense vector (the layout
/// `proxima_records_to_record_batch` writes for [`ProximaType::DenseVector`]) into
/// `record.embeddings`.
///
/// **Reserved system columns are lifted onto the record, not into `props`**
/// (TD-USUB-11). A batch written by
/// [`proxima_records_to_record_batch_with_system_columns`] therefore restores
/// identity and lifecycle — `oid`, `tenant_id`, `variation_id`, the timestamps
/// and, critically, `valid_to_ns` — so a tombstoned or TTL-expired row does not
/// come back looking live.
///
/// When those columns are ABSENT this behaves exactly as before: identity
/// (`oid`) and `variation_id` are not inferred, because a plain self-describing
/// Arrow batch does not carry them (the canonical write path stores the key as a
/// PK column value), and the catalog-aware caller stamps them after this returns.
/// That fallback is what makes the encoding mixed-read-safe — presence of the
/// column is the marker, so old and new objects coexist with no flag day.
pub fn record_batch_to_proxima_records(batch: &RecordBatch) -> Vec<ProximaRecord> {
    let n = batch.num_rows();
    let mut out: Vec<ProximaRecord> = (0..n).map(|_| ProximaRecord::default()).collect();
    let arrow_schema = batch.schema();

    for (col_idx, field) in arrow_schema.fields().iter().enumerate() {
        let array = batch.column(col_idx);
        if let Some(fsb) = array.as_any().downcast_ref::<FixedSizeBinaryArray>() {
            for (row, record) in out.iter_mut().enumerate() {
                if fsb.is_null(row) {
                    continue;
                }
                let bytes = fsb.value(row);
                let values: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let dim = values.len() as u32;
                record.embeddings.push(EmbeddingCell {
                    values: EmbeddingValues::Fp32(values),
                    dim,
                    ..Default::default()
                });
            }
            continue;
        }
        let name = field.name();

        // TD-USUB-11: a reserved system column restores a RECORD FIELD; it must
        // never land in `props`, or it would reappear as a user-visible column.
        if is_system_column(name) {
            for (row, record) in out.iter_mut().enumerate() {
                let Some(value) = arrow_cell_to_proxima_value(array, row) else {
                    continue; // NULL — leave the field at its default.
                };
                match name.as_str() {
                    SYS_COL_OID => {
                        if let ProximaValue::String(s) = value {
                            record.oid = s;
                        }
                    }
                    SYS_COL_TENANT_ID => {
                        if let ProximaValue::String(s) = value {
                            record.tenant_id = s;
                        }
                    }
                    SYS_COL_VARIATION_ID => {
                        if let ProximaValue::String(s) = value {
                            record.variation_id = Some(s);
                        }
                    }
                    SYS_COL_CREATED_AT_NS => {
                        if let Some(n) = pv_as_i64(&value) {
                            record.created_at_ns = n;
                        }
                    }
                    SYS_COL_UPDATED_AT_NS => {
                        if let Some(n) = pv_as_i64(&value) {
                            record.updated_at_ns = n;
                        }
                    }
                    SYS_COL_VALID_FROM_NS => record.valid_from_ns = pv_as_i64(&value),
                    SYS_COL_VALID_TO_NS => record.valid_to_ns = pv_as_i64(&value),
                    _ => {}
                }
            }
            continue;
        }

        for (row, record) in out.iter_mut().enumerate() {
            if let Some(value) = arrow_cell_to_proxima_value(array, row) {
                record
                    .props
                    .insert(name.clone(), ProximaTreeNode::Value(value));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn col(id: i32, name: &str, dt: ProximaType, nullable: bool) -> ProximaColumn {
        ProximaColumn {
            id,
            object_id: None,
            name: name.to_string(),
            data_type: dt,
            nullable,
            default_value: None,
            comment: None,
            properties: HashMap::new(),
            is_deleted: false,
            original_id: None,
        }
    }

    fn leaf(v: ProximaValue) -> ProximaTreeNode {
        ProximaTreeNode::Value(v)
    }

    fn record_with_props(oid: &str, props: Vec<(&str, ProximaValue)>) -> ProximaRecord {
        let mut r = ProximaRecord {
            oid: oid.to_string(),
            ..Default::default()
        };
        for (k, v) in props {
            r.props.insert(k.to_string(), leaf(v));
        }
        r
    }

    // --- TD-USUB-11: faithful canonical-record round trip -------------------

    fn sys_schema() -> ProximaSchema {
        ProximaSchema::from_columns(
            "orders".to_string(),
            vec![col(1, "status", ProximaType::String, true)],
            vec![1],
        )
    }

    /// **The load-bearing test.** A tombstoned row (`valid_to_ns == Some(0)`) must
    /// come back tombstoned. Without the system columns it returns with
    /// `valid_to_ns == None` — i.e. LIVE — which is how a deleted row resurrects
    /// out of an immutable landing segment.
    #[test]
    fn tombstone_survives_the_round_trip_and_is_lost_without_system_columns() {
        let mut deleted =
            record_with_props("o1", vec![("status", ProximaValue::String("x".into()))]);
        deleted.valid_to_ns = Some(0);
        let schema = sys_schema();

        // WITH system columns: the tombstone survives.
        let batch = proxima_records_to_record_batch_with_system_columns(
            std::slice::from_ref(&deleted),
            &schema,
        )
        .expect("encode");
        let back = record_batch_to_proxima_records(&batch);
        assert_eq!(back.len(), 1);
        assert_eq!(
            back[0].valid_to_ns,
            Some(0),
            "tombstone must round-trip, or a deleted row comes back live"
        );
        assert!(
            back[0].is_dead(1),
            "the canonical dead-record predicate must still see it as dead"
        );

        // WITHOUT them: the tombstone is LOST. This is the defect the system
        // columns exist to close — asserted so the gap cannot silently reappear.
        let legacy = proxima_records_to_record_batch(std::slice::from_ref(&deleted), &schema)
            .expect("legacy encode");
        let legacy_back = record_batch_to_proxima_records(&legacy);
        assert_eq!(
            legacy_back[0].valid_to_ns, None,
            "legacy encoding drops the tombstone — this is why the system columns exist"
        );
        assert!(
            !legacy_back[0].is_dead(1),
            "legacy round trip resurrects a deleted row (documented, not endorsed)"
        );
    }

    /// Identity and lifecycle all survive, and the reserved columns stay OUT of
    /// `props` so they never surface as user-visible columns.
    #[test]
    fn system_columns_restore_identity_and_stay_out_of_props() {
        let mut r = record_with_props(
            "order-7",
            vec![("status", ProximaValue::String("open".into()))],
        );
        r.tenant_id = "tenant-a".into();
        r.variation_id = Some("orders".into());
        r.created_at_ns = 111;
        r.updated_at_ns = 222;
        r.valid_from_ns = Some(333);
        r.valid_to_ns = Some(444);

        let batch = proxima_records_to_record_batch_with_system_columns(
            std::slice::from_ref(&r),
            &sys_schema(),
        )
        .expect("encode");
        let back = &record_batch_to_proxima_records(&batch)[0];

        assert_eq!(back.oid, "order-7");
        assert_eq!(back.tenant_id, "tenant-a");
        assert_eq!(back.variation_id.as_deref(), Some("orders"));
        assert_eq!(back.created_at_ns, 111);
        assert_eq!(back.updated_at_ns, 222);
        assert_eq!(back.valid_from_ns, Some(333));
        assert_eq!(back.valid_to_ns, Some(444));

        // The user column is still a prop; no reserved name leaked in.
        assert!(back.props.contains_key("status"));
        for sys in SYSTEM_COLUMNS {
            assert!(
                !back.props.contains_key(*sys),
                "reserved column {sys} must not surface as a user prop"
            );
        }
    }

    /// Mixed-read safety (mandate #8): a batch with no system columns decodes
    /// EXACTLY as it did before this change — same props, no fabricated identity.
    #[test]
    fn legacy_batches_decode_unchanged() {
        let r = record_with_props(
            "ignored",
            vec![("status", ProximaValue::String("open".into()))],
        );
        let batch = proxima_records_to_record_batch(std::slice::from_ref(&r), &sys_schema())
            .expect("encode");
        let back = &record_batch_to_proxima_records(&batch)[0];

        assert_eq!(
            back.oid, "",
            "identity is still not inferred for legacy batches"
        );
        assert_eq!(back.tenant_id, "");
        assert_eq!(back.valid_to_ns, None);
        assert!(back.props.contains_key("status"));
    }

    /// The legacy encoder must remain byte-identical — adding the opt-in variant
    /// must not perturb existing Parquet output (MATERIALIZE included).
    #[test]
    fn legacy_encoder_output_is_unchanged_by_the_new_variant() {
        let r = record_with_props("o", vec![("status", ProximaValue::String("open".into()))]);
        let schema = sys_schema();
        let batch =
            proxima_records_to_record_batch(std::slice::from_ref(&r), &schema).expect("encode");
        assert_eq!(
            batch.num_columns(),
            1,
            "legacy encoder must emit ONLY declared user columns"
        );
        assert_eq!(batch.schema().field(0).name(), "status");
    }

    /// A user column colliding with a reserved name must fail closed and name the
    /// column — two same-named Arrow fields would make read-back ambiguous.
    #[test]
    fn reserved_name_collision_fails_closed() {
        let schema = ProximaSchema::from_columns(
            "orders".to_string(),
            vec![col(1, SYS_COL_VALID_TO_NS, ProximaType::Int64, true)],
            vec![1],
        );
        let r = record_with_props("o", vec![]);
        let err =
            proxima_records_to_record_batch_with_system_columns(std::slice::from_ref(&r), &schema)
                .expect_err("a reserved-name collision must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains(SYS_COL_VALID_TO_NS) && msg.contains("collides"),
            "error must name the colliding column, got: {msg}"
        );
    }

    #[test]
    fn typed_columns_with_null_masking() {
        // Schema: name:Utf8, age:Int64, score:Float64, active:Boolean.
        let schema = ProximaSchema::from_columns(
            "test".to_string(),
            vec![
                col(1, "name", ProximaType::String, true),
                col(2, "age", ProximaType::Int64, true),
                col(3, "score", ProximaType::Float64, true),
                col(4, "active", ProximaType::Boolean, true),
            ],
            vec![1],
        );

        // Row 0: all present. Row 1: `age` and `active` ABSENT (must become Arrow null).
        let records = vec![
            record_with_props(
                "r0",
                vec![
                    ("name", ProximaValue::String("alice".into())),
                    ("age", ProximaValue::Int64(30)),
                    ("score", ProximaValue::Float64(9.5)),
                    ("active", ProximaValue::Boolean(true)),
                ],
            ),
            record_with_props(
                "r1",
                vec![
                    ("name", ProximaValue::String("bob".into())),
                    ("score", ProximaValue::Float64(1.25)),
                ],
            ),
        ];

        let batch = proxima_records_to_record_batch(&records, &schema).unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 4);

        let names = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(names.value(0), "alice");
        assert_eq!(names.value(1), "bob");

        let ages = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(ages.value(0), 30);
        assert!(ages.is_null(1), "absent prop must be a null, not 0");

        let scores = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(scores.value(0), 9.5);
        assert_eq!(scores.value(1), 1.25);

        let active = batch
            .column(3)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert!(active.value(0));
        assert!(active.is_null(1), "absent bool must be a null");
    }

    #[test]
    fn vector_column_encodes_fp32_le_and_nulls() {
        let dim = 3u32;
        let schema = ProximaSchema::from_columns(
            "vec".to_string(),
            vec![
                col(1, "id", ProximaType::String, false),
                col(
                    2,
                    "embedding",
                    ProximaType::DenseVector {
                        element: DmVectorElement::Float32,
                        dim: dim as usize,
                    },
                    true,
                ),
            ],
            vec![1],
        );

        let mut r0 = record_with_props("r0", vec![("id", ProximaValue::String("r0".into()))]);
        r0.embeddings.push(EmbeddingCell {
            values: EmbeddingValues::Fp32(vec![1.0, 2.0, 3.0]),
            ..Default::default()
        });
        // r1 has NO embedding -> null vector row.
        let r1 = record_with_props("r1", vec![("id", ProximaValue::String("r1".into()))]);

        let batch = proxima_records_to_record_batch(&[r0, r1], &schema).unwrap();
        let vecs = batch
            .column(1)
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .unwrap();
        assert_eq!(vecs.value_length(), (dim * 4) as i32);

        let raw = vecs.value(0);
        let decoded: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(decoded, vec![1.0, 2.0, 3.0]);
        assert!(vecs.is_null(1), "record without embedding must be null");
    }

    #[test]
    fn vector_dimension_mismatch_is_error() {
        let schema = ProximaSchema::from_columns(
            "vec".to_string(),
            vec![col(
                1,
                "embedding",
                ProximaType::DenseVector {
                    element: DmVectorElement::Float32,
                    dim: 4,
                },
                true,
            )],
            vec![],
        );
        let mut r0 = ProximaRecord {
            oid: "r0".into(),
            ..Default::default()
        };
        r0.embeddings.push(EmbeddingCell {
            values: EmbeddingValues::Fp32(vec![1.0, 2.0, 3.0]), // 3 != 4
            ..Default::default()
        });
        let err = proxima_records_to_record_batch(&[r0], &schema).unwrap_err();
        assert!(matches!(err, StorageError::InvalidDimension { .. }));
    }

    #[test]
    fn integer_widening_coercion() {
        // Column is Int64 but the prop is an Int32 — widening must succeed, not null out.
        let schema = ProximaSchema::from_columns(
            "coerce".to_string(),
            vec![col(1, "n", ProximaType::Int64, true)],
            vec![],
        );
        let r = record_with_props("r0", vec![("n", ProximaValue::Int32(7))]);
        let batch = proxima_records_to_record_batch(&[r], &schema).unwrap();
        let ns = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(ns.value(0), 7);
    }

    #[test]
    fn infer_schema_unions_props_and_appends_vector() {
        // r0 defines name(String)+age(Int64) and an embedding; r1 adds score(Float64) and a
        // value `tags` that is only ever Null in r0 -> tags must NOT become a column from r0,
        // but r1 gives it a concrete String -> it should appear, after the first-seen columns.
        let mut r0 = record_with_props(
            "r0",
            vec![
                ("name", ProximaValue::String("a".into())),
                ("age", ProximaValue::Int64(1)),
                ("tags", ProximaValue::Null),
            ],
        );
        r0.embeddings.push(EmbeddingCell {
            values: EmbeddingValues::Fp32(vec![0.0, 1.0]),
            ..Default::default()
        });
        let r1 = record_with_props(
            "r1",
            vec![
                ("name", ProximaValue::String("b".into())),
                ("score", ProximaValue::Float64(2.0)),
                ("tags", ProximaValue::String("x".into())),
            ],
        );

        let schema = infer_proxima_schema(&[r0, r1]);
        let by_name: HashMap<&str, &ProximaColumn> = schema
            .columns
            .iter()
            .map(|c| (c.name.as_str(), c))
            .collect();

        assert_eq!(by_name["name"].data_type, ProximaType::String);
        assert_eq!(by_name["age"].data_type, ProximaType::Int64);
        assert_eq!(by_name["score"].data_type, ProximaType::Float64);
        assert_eq!(by_name["tags"].data_type, ProximaType::String);
        assert!(by_name.values().all(|c| c.nullable));

        let vec_col = by_name[INFERRED_VECTOR_COLUMN];
        assert_eq!(
            vec_col.data_type,
            ProximaType::DenseVector {
                element: DmVectorElement::Float32,
                dim: 2,
            }
        );

        // The inferred schema must round-trip the records through the canonical converter.
        let r0b = {
            let mut r = record_with_props("r0", vec![("name", ProximaValue::String("a".into()))]);
            r.embeddings.push(EmbeddingCell {
                values: EmbeddingValues::Fp32(vec![0.0, 1.0]),
                ..Default::default()
            });
            r
        };
        let batch = proxima_records_to_record_batch(&[r0b], &schema).unwrap();
        assert_eq!(batch.num_columns(), schema.columns.len());
    }

    #[test]
    fn infer_schema_skips_only_null_or_complex_columns() {
        // `blob` is an Array (unsupported) and `nothing` is always Null -> neither becomes a column.
        let r = record_with_props(
            "r0",
            vec![
                ("ok", ProximaValue::Int64(1)),
                ("blob", ProximaValue::Array(vec![ProximaValue::Int64(1)])),
                ("nothing", ProximaValue::Null),
            ],
        );
        let schema = infer_proxima_schema(&[r]);
        let names: Vec<&str> = schema.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["ok"]);
    }

    /// S7 inverse identity: forward-encode records then reverse-decode them and
    /// confirm scalar props + the dense vector survive (the vector column round-
    /// trips through `FixedSizeBinary`; an absent embedding decodes to none).
    #[test]
    fn record_batch_reverse_round_trips_scalars_and_vector() {
        let schema = ProximaSchema::from_columns(
            "rt".to_string(),
            vec![
                col(1, "name", ProximaType::String, true),
                col(2, "age", ProximaType::Int64, true),
                col(
                    3,
                    "vector",
                    ProximaType::DenseVector {
                        element: DmVectorElement::Float32,
                        dim: 3,
                    },
                    true,
                ),
            ],
            vec![1],
        );

        let mut r0 = record_with_props(
            "r0",
            vec![
                ("name", ProximaValue::String("alice".into())),
                ("age", ProximaValue::Int64(30)),
            ],
        );
        r0.embeddings.push(EmbeddingCell {
            values: EmbeddingValues::Fp32(vec![1.0, 2.0, 3.0]),
            dim: 3,
            ..Default::default()
        });
        let r1 = record_with_props(
            "r1",
            vec![
                ("name", ProximaValue::String("bob".into())),
                ("age", ProximaValue::Int64(41)),
            ],
        );

        let batch = proxima_records_to_record_batch(&[r0, r1], &schema).unwrap();
        let back = record_batch_to_proxima_records(&batch);

        assert_eq!(back.len(), 2);
        assert_eq!(
            tree_get(&back[0].props, "name"),
            Some(&ProximaValue::String("alice".into()))
        );
        assert_eq!(
            tree_get(&back[0].props, "age"),
            Some(&ProximaValue::Int64(30))
        );
        assert_eq!(
            back[0].embeddings.first().map(|e| e.values.to_fp32_owned()),
            Some(vec![1.0, 2.0, 3.0]),
            "dense vector must survive the FixedSizeBinary round-trip"
        );
        assert!(
            back[1].embeddings.is_empty(),
            "a record without an embedding must decode to no embedding"
        );
        assert_eq!(
            tree_get(&back[1].props, "name"),
            Some(&ProximaValue::String("bob".into()))
        );
    }
}
