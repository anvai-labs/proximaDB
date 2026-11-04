//! # Schema Inference for ProximaDB Collections
//!
//! Extracts Arrow schemas and statistics from ProximaDB collections
//! for DataFusion query optimization.

use std::sync::Arc;

use datafusion::arrow::datatypes::{
    DataType as ArrowDataType, Field, Schema as ArrowSchema, TimeUnit as ArrowTimeUnit,
};
use datafusion::common::{ColumnStatistics, ScalarValue as DfScalarValue, Statistics};
use datafusion::error::Result as DFResult;
use datafusion_common::stats::Precision;

use crate::storage::formats::FormatStatistics;
use proximadb_data_model::{ProximaType, TimeUnit as DmTimeUnit, VectorElement};
use proximadb_storage_common::format_splits::FileSplit;

use crate::storage::schema::{ProximaColumn, ProximaSchema};

/// Infer Arrow schema from a ProximaDB collection.
///
/// This uses the collection's ProximaSchema to generate a compatible Arrow schema.
pub fn infer_schema_from_collection(proxima_schema: &ProximaSchema) -> DFResult<Arc<ArrowSchema>> {
    // to_arrow_schema() already returns Arc<ArrowSchema>
    Ok(proxima_schema.to_arrow_schema())
}

/// Extract DataFusion Statistics from ProximaDB format statistics.
pub fn extract_statistics(format_stats: &FormatStatistics, schema: &ArrowSchema) -> Statistics {
    let num_rows = if format_stats.row_count > 0 {
        Precision::Exact(format_stats.row_count as usize)
    } else {
        Precision::Absent
    };

    let total_byte_size = if format_stats.size_bytes > 0 {
        Precision::Exact(format_stats.size_bytes as usize)
    } else {
        Precision::Absent
    };

    // Create column statistics for each field (schema-width — DataFusion 54
    // indexes `column_statistics[col.index()]`, see `proxima_scan_exec.rs`).
    let column_statistics: Vec<ColumnStatistics> = schema
        .fields()
        .iter()
        .map(|field| {
            // Try to get column stats from format_stats
            if let Some(col_stats) = format_stats.column_stats.get(field.name()) {
                // Build column statistics from stored values
                let mut cs = ColumnStatistics::new_unknown();
                // null_count is u64 (not Option)
                cs.null_count = Precision::Exact(col_stats.null_count as usize);
                // distinct_count is Option<u64>
                if let Some(distinct) = col_stats.distinct_count {
                    cs.distinct_count = Precision::Exact(distinct as usize);
                }
                if let Some(min) = col_stats
                    .min
                    .as_ref()
                    .and_then(|v| json_bound_to_scalar(v, field.data_type()))
                {
                    cs.min_value = Precision::Exact(min);
                }
                if let Some(max) = col_stats
                    .max
                    .as_ref()
                    .and_then(|v| json_bound_to_scalar(v, field.data_type()))
                {
                    cs.max_value = Precision::Exact(max);
                }
                cs
            } else {
                ColumnStatistics::new_unknown()
            }
        })
        .collect();

    Statistics {
        num_rows,
        total_byte_size,
        column_statistics,
    }
}

/// Project the Parquet-footer split statistics of a table into DataFusion
/// [`Statistics`] (ADR-050 D2 / TD-DFR-1): min-of-mins / max-of-maxes across
/// splits, summed null counts, summed rows/bytes.
///
/// Everything is [`Precision::Inexact`] — per ADR-050 D3 statistics are
/// freshness-bounded *hints, not truth* (post-materialize deltas make the
/// Parquet snapshot conservative). The vector is always **schema-width**
/// (DataFusion 54 indexes `column_statistics[col.index()]` from parent
/// operators); columns without footer bounds stay `new_unknown()`.
///
/// When `collection` is given and the ADR-037 statistics registry holds a
/// resident summary for it, its HLL `distinct_estimate` fills
/// `distinct_count` — Parquet footers rarely carry NDV, the sketch does.
///
/// `include_value_bounds` gates min/max projection (see
/// [`df_stats_value_bounds_enabled`]): typed value bounds feed DataFusion's
/// recursive interval/selectivity analysis, deepening planner recursion on
/// join+subquery-heavy plans (TPC-H Q2 class), whose debug-build stack use is
/// already near the limit on some environments (a Q2 stack overflow reproduces
/// on WSL2 debug builds even WITHOUT bounds; `RUST_MIN_STACK=16777216` clears
/// it). Bounds therefore stay opt-in (default-OFF, storage-migration mandate
/// posture) until that recursion depth is characterized. Row/byte counts, null
/// counts, and NDV — the inputs that drive join ordering — always flow.
pub fn statistics_from_splits(
    splits: &[FileSplit],
    schema: &ArrowSchema,
    collection: Option<&str>,
    include_value_bounds: bool,
    stats_trust: proximadb_data_model::StatsTrust,
) -> Statistics {
    // ADR-058 D5/§9.A: Untrusted (external/federated) parquet ⇒ do NOT surface
    // footer stats as Exact. The writer's footer may be stale or absent, so
    // eliding COUNT/MIN/MAX from it could return a WRONG aggregate. Return all
    // Absent so DataFusion's AggregateStatistics declines to elide (scans real
    // column data instead). Trusted (ProximaDB-generated) ⇒ the footer is
    // authoritative ⇒ fall through to Exact elision.
    if matches!(stats_trust, proximadb_data_model::StatsTrust::Untrusted) {
        return Statistics::new_unknown(schema);
    }
    let mut num_rows = 0usize;
    let mut any_rows = false;
    let mut total_bytes = 0usize;
    let mut any_bytes = false;
    for split in splits {
        if let Some(rows) = split.statistics.row_count {
            num_rows = num_rows.saturating_add(rows as usize);
            any_rows = true;
        }
        if let Some(bytes) = split.statistics.byte_size {
            total_bytes = total_bytes.saturating_add(bytes as usize);
            any_bytes = true;
        }
    }

    let registry_fields = collection
        .and_then(|c| crate::core::statistics::statistics_registry().envelope(c))
        .map(|env| env.fields)
        .unwrap_or_default();

    let column_statistics: Vec<ColumnStatistics> = schema
        .fields()
        .iter()
        .map(|field| {
            let mut cs = ColumnStatistics::new_unknown();
            let mut min: Option<DfScalarValue> = None;
            let mut max: Option<DfScalarValue> = None;
            let mut nulls = 0usize;
            let mut covered = 0usize;
            for split in splits {
                let Some(bounds) = split.statistics.column_stats.get(field.name()) else {
                    continue;
                };
                covered += 1;
                nulls = nulls.saturating_add(bounds.null_count as usize);
                if let Some(v) = bounds
                    .min
                    .as_ref()
                    .and_then(|v| json_bound_to_scalar(v, field.data_type()))
                {
                    min = Some(match min.take() {
                        Some(cur) => scalar_min(cur, v),
                        None => v,
                    });
                }
                if let Some(v) = bounds
                    .max
                    .as_ref()
                    .and_then(|v| json_bound_to_scalar(v, field.data_type()))
                {
                    max = Some(match max.take() {
                        Some(cur) => scalar_max(cur, v),
                        None => v,
                    });
                }
            }
            // Only claim bounds/nulls when EVERY split carries footer stats for
            // this column — a partially-covered column would understate.
            if covered == splits.len() && covered > 0 {
                if include_value_bounds {
                    // Parquet min/max are EXACT for numeric columns but TRUNCATED
                    // for BYTE_ARRAY (string/binary) — a truncated string bound is a
                    // prefix, not the true value. Report `Exact` only for numeric
                    // types so DataFusion's `AggregateStatistics` rule can elide
                    // `MIN`/`MAX` from the footer (e.g. ClickBench q07's
                    // `MIN/MAX(EventDate)`, a 10 s scan today); keep string bounds
                    // `Inexact` (zone-map pruning only). A `WHERE` inserts a
                    // `FilterExec` that re-marks stats Inexact, so filtered
                    // aggregates still scan; the ADR-025 delta-merge path uses its
                    // own MemTable, never this provider.
                    let numeric = field.data_type().is_numeric();
                    if let Some(min) = min {
                        cs.min_value = if numeric {
                            Precision::Exact(min)
                        } else {
                            Precision::Inexact(min)
                        };
                    }
                    if let Some(max) = max {
                        cs.max_value = if numeric {
                            Precision::Exact(max)
                        } else {
                            Precision::Inexact(max)
                        };
                    }
                }
                // `null_count` from the footer is exact metadata (not a truncatable
                // bound), so report `Exact` unconditionally — this lets the rule
                // answer `COUNT(col)` as `num_rows − null_count` with no scan,
                // independent of the value-bounds gate above.
                cs.null_count = Precision::Exact(nulls);
            }
            if let Some(ndv) = registry_fields
                .iter()
                .find(|f| f.name == *field.name())
                .and_then(|f| f.distinct_estimate)
            {
                cs.distinct_count = Precision::Inexact(ndv as usize);
            }
            cs
        })
        .collect();

    Statistics {
        // Parquet row-group `num_rows` is EXACT metadata (unlike the min/max value
        // bounds below, which are Inexact zone-maps). Reporting it Exact lets
        // DataFusion's `AggregateStatistics` rule answer an unfiltered `COUNT(*)`
        // from the footer alone — zero column scan (TD-OLAP-4). Safe: a `WHERE`
        // inserts a `FilterExec` that re-marks stats Inexact, so filtered counts
        // still scan; and only footer-accurate providers use this (the ADR-025
        // delta-merge path builds its own MemTable when a WAL delta exists).
        num_rows: if any_rows {
            Precision::Exact(num_rows)
        } else {
            Precision::Absent
        },
        total_byte_size: if any_bytes {
            Precision::Inexact(total_bytes)
        } else {
            Precision::Absent
        },
        column_statistics,
    }
}

proximadb_config::env_gate! {
    /// Opt-in gate for projecting typed min/max value bounds into DataFusion
    /// `ColumnStatistics` (default **off**, per the default-OFF mandate for
    /// planner-behavior changes): bounds enable DataFusion 54's recursive
    /// interval-based filter-selectivity analysis, deepening plan-time recursion
    /// on join+subquery-heavy shapes (TPC-H Q2 class) whose debug-build stack use
    /// is already borderline on some environments. Enable with
    /// `PROXIMADB_DF_STATS_VALUE_BOUNDS=1` once that recursion depth is
    /// characterized; row/null/NDV statistics flow regardless.
    ///
    /// When ON, numeric bounds are reported `Exact` (see `statistics_from_splits`),
    /// which additionally unlocks the *non-recursive* `AggregateStatistics` rule to
    /// elide unfiltered `MIN`/`MAX` from the footer (e.g. ClickBench q07's
    /// `MIN/MAX(EventDate)` — a 10 s full scan today answered from metadata). That
    /// aggregate-elision path is unrelated to the filter-selectivity recursion that
    /// keeps the default OFF; `COUNT(col)` elision (via exact `null_count`) is wired
    /// unconditionally and needs no gate.
    pub fn df_stats_value_bounds_enabled() { gate: "PROXIMADB_DF_STATS_VALUE_BOUNDS", default: false }
}

/// Narrow a table-wide [`Statistics`] to a projection, preserving the
/// schema-width contract for the projected schema.
pub fn project_statistics(stats: &Statistics, projection: Option<&[usize]>) -> Statistics {
    match projection {
        None => stats.clone(),
        Some(indices) => Statistics {
            num_rows: stats.num_rows,
            total_byte_size: stats.total_byte_size.to_inexact(),
            column_statistics: indices
                .iter()
                .map(|&i| {
                    stats
                        .column_statistics
                        .get(i)
                        .cloned()
                        .unwrap_or_else(ColumnStatistics::new_unknown)
                })
                .collect(),
        },
    }
}

/// Convert a JSON zone-map bound into a typed DataFusion scalar for `field`.
///
/// Footer bounds are stored as JSON (`i64`/`f64`/`bool`/string — see
/// `column_bounds_from_parquet`); only types with a lossless mapping are
/// projected, everything else stays `Absent`.
fn json_bound_to_scalar(
    value: &serde_json::Value,
    data_type: &ArrowDataType,
) -> Option<DfScalarValue> {
    match data_type {
        ArrowDataType::Boolean => value.as_bool().map(|v| DfScalarValue::Boolean(Some(v))),
        ArrowDataType::Int8 => value
            .as_i64()
            .and_then(|v| i8::try_from(v).ok())
            .map(|v| DfScalarValue::Int8(Some(v))),
        ArrowDataType::Int16 => value
            .as_i64()
            .and_then(|v| i16::try_from(v).ok())
            .map(|v| DfScalarValue::Int16(Some(v))),
        ArrowDataType::Int32 => value
            .as_i64()
            .and_then(|v| i32::try_from(v).ok())
            .map(|v| DfScalarValue::Int32(Some(v))),
        ArrowDataType::Int64 => value.as_i64().map(|v| DfScalarValue::Int64(Some(v))),
        ArrowDataType::UInt8 => value
            .as_u64()
            .and_then(|v| u8::try_from(v).ok())
            .map(|v| DfScalarValue::UInt8(Some(v))),
        ArrowDataType::UInt16 => value
            .as_u64()
            .and_then(|v| u16::try_from(v).ok())
            .map(|v| DfScalarValue::UInt16(Some(v))),
        ArrowDataType::UInt32 => value
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .map(|v| DfScalarValue::UInt32(Some(v))),
        ArrowDataType::UInt64 => value.as_u64().map(|v| DfScalarValue::UInt64(Some(v))),
        ArrowDataType::Float32 => value
            .as_f64()
            .map(|v| DfScalarValue::Float32(Some(v as f32))),
        ArrowDataType::Float64 => value.as_f64().map(|v| DfScalarValue::Float64(Some(v))),
        ArrowDataType::Utf8 => value
            .as_str()
            .map(|v| DfScalarValue::Utf8(Some(v.to_string()))),
        ArrowDataType::LargeUtf8 => value
            .as_str()
            .map(|v| DfScalarValue::LargeUtf8(Some(v.to_string()))),
        _ => None,
    }
}

fn scalar_min(a: DfScalarValue, b: DfScalarValue) -> DfScalarValue {
    match a.partial_cmp(&b) {
        Some(std::cmp::Ordering::Greater) => b,
        _ => a,
    }
}

fn scalar_max(a: DfScalarValue, b: DfScalarValue) -> DfScalarValue {
    match a.partial_cmp(&b) {
        Some(std::cmp::Ordering::Less) => b,
        _ => a,
    }
}

/// Convert ProximaType to Arrow DataType.
pub fn proxima_to_arrow_type(proxima_type: &ProximaType) -> ArrowDataType {
    match proxima_type {
        ProximaType::Boolean => ArrowDataType::Boolean,
        ProximaType::Int8 => ArrowDataType::Int8,
        ProximaType::Int16 => ArrowDataType::Int16,
        ProximaType::Int32 => ArrowDataType::Int32,
        ProximaType::Int64 => ArrowDataType::Int64,
        ProximaType::UInt8 => ArrowDataType::UInt8,
        ProximaType::UInt16 => ArrowDataType::UInt16,
        ProximaType::UInt32 => ArrowDataType::UInt32,
        ProximaType::UInt64 => ArrowDataType::UInt64,
        ProximaType::Float16 => ArrowDataType::Float16,
        ProximaType::Float32 => ArrowDataType::Float32,
        ProximaType::Float64 => ArrowDataType::Float64,
        ProximaType::Decimal { precision, scale } => {
            ArrowDataType::Decimal128(*precision, *scale as i8)
        }
        ProximaType::String | ProximaType::Symbol => ArrowDataType::Utf8,
        ProximaType::Binary => ArrowDataType::Binary,
        ProximaType::Date => ArrowDataType::Date32,
        ProximaType::Time(unit) => ArrowDataType::Time64(time_unit_to_arrow(*unit)),
        ProximaType::Timestamp(unit) => ArrowDataType::Timestamp(time_unit_to_arrow(*unit), None),
        ProximaType::TimestampTz(unit) => {
            ArrowDataType::Timestamp(time_unit_to_arrow(*unit), Some(Arc::from("UTC")))
        }
        ProximaType::Uuid | ProximaType::ULID => ArrowDataType::FixedSizeBinary(16),
        ProximaType::Json | ProximaType::Jsonb => ArrowDataType::Utf8, // JSON as string
        ProximaType::Array(element) => {
            let element_type = proxima_to_arrow_type(element);
            ArrowDataType::List(Arc::new(Field::new("item", element_type, true)))
        }
        ProximaType::Map { key, value } => {
            let key_type = proxima_to_arrow_type(key);
            let value_type = proxima_to_arrow_type(value);
            let struct_field = ArrowDataType::Struct(
                vec![
                    Field::new("key", key_type, false),
                    Field::new("value", value_type, true),
                ]
                .into(),
            );
            ArrowDataType::Map(Arc::new(Field::new("entries", struct_field, false)), false)
        }
        ProximaType::Struct { fields } => {
            let arrow_fields: Vec<Field> = fields
                .iter()
                .map(|(name, ty)| Field::new(name, proxima_to_arrow_type(ty), true))
                .collect();
            ArrowDataType::Struct(arrow_fields.into())
        }

        // Vector types
        ProximaType::DenseVector { element, dim } => {
            let elem_arrow = match element {
                VectorElement::Float32 => ArrowDataType::Float32,
                VectorElement::Float64 => ArrowDataType::Float64,
                VectorElement::Float16 => ArrowDataType::Float16,
                VectorElement::BFloat16 => ArrowDataType::Float32, // BFloat16 approximated as Float32
                VectorElement::Int8 => ArrowDataType::Int8,
            };
            ArrowDataType::FixedSizeList(
                Arc::new(Field::new("value", elem_arrow, true)),
                *dim as i32,
            )
        }
        ProximaType::SparseVector { .. } => {
            // Sparse vector as struct of indices and values
            ArrowDataType::Struct(
                vec![
                    Field::new(
                        "indices",
                        ArrowDataType::List(Arc::new(Field::new(
                            "item",
                            ArrowDataType::UInt32,
                            false,
                        ))),
                        false,
                    ),
                    Field::new(
                        "values",
                        ArrowDataType::List(Arc::new(Field::new(
                            "item",
                            ArrowDataType::Float32,
                            false,
                        ))),
                        false,
                    ),
                ]
                .into(),
            )
        }
        ProximaType::BinaryVector { dim } => {
            // Binary vector as fixed-size binary
            ArrowDataType::FixedSizeBinary(dim.div_ceil(8) as i32)
        }
        // Interval/Duration/geo/null have no distinct DataFusion projection here.
        ProximaType::Interval(_) => {
            ArrowDataType::Interval(datafusion::arrow::datatypes::IntervalUnit::MonthDayNano)
        }
        ProximaType::Duration(unit) => ArrowDataType::Duration(time_unit_to_arrow(*unit)),
        ProximaType::Point | ProximaType::GeographyPoint => ArrowDataType::FixedSizeBinary(16),
        ProximaType::Null => ArrowDataType::Null,
    }
}

/// Convert the canonical [`DmTimeUnit`] to Arrow's time unit.
fn time_unit_to_arrow(unit: DmTimeUnit) -> ArrowTimeUnit {
    match unit {
        DmTimeUnit::Second => ArrowTimeUnit::Second,
        DmTimeUnit::Millisecond => ArrowTimeUnit::Millisecond,
        DmTimeUnit::Microsecond => ArrowTimeUnit::Microsecond,
        DmTimeUnit::Nanosecond => ArrowTimeUnit::Nanosecond,
    }
}

/// Convert ProximaColumn to Arrow Field.
pub fn proxima_column_to_arrow_field(column: &ProximaColumn) -> Field {
    let arrow_type = proxima_to_arrow_type(&column.data_type);
    Field::new(&column.name, arrow_type, column.nullable)
}

/// Merge multiple schemas (e.g., from different files).
pub fn merge_schemas(schemas: &[&ArrowSchema]) -> DFResult<ArrowSchema> {
    if schemas.is_empty() {
        return Ok(ArrowSchema::empty());
    }

    if schemas.len() == 1 {
        return Ok(schemas[0].clone());
    }

    // Start with first schema
    let mut merged_fields = schemas[0].fields().to_vec();

    // Merge fields from other schemas
    for schema in schemas.iter().skip(1) {
        for field in schema.fields() {
            if !merged_fields.iter().any(|f| f.name() == field.name()) {
                merged_fields.push(field.clone());
            }
        }
    }

    Ok(ArrowSchema::new(merged_fields))
}

/// Estimate statistics from file metadata.
pub fn estimate_statistics_from_files(
    file_sizes: &[u64],
    row_counts: &[Option<u64>],
) -> Statistics {
    let total_bytes: u64 = file_sizes.iter().sum();
    let total_rows: Option<u64> = row_counts
        .iter()
        .try_fold(0u64, |acc, count| count.map(|c| acc + c));

    let num_rows = match total_rows {
        Some(rows) => datafusion_common::stats::Precision::Exact(rows as usize),
        None => datafusion_common::stats::Precision::Absent,
    };

    Statistics {
        num_rows,
        total_byte_size: datafusion_common::stats::Precision::Exact(total_bytes as usize),
        column_statistics: vec![],
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_schema_from_collection() {
        let schema = ProximaSchema::vector_record_schema(768);
        let arrow_schema = infer_schema_from_collection(&schema).unwrap();

        assert!(arrow_schema.field_with_name("id").is_ok());
        assert!(arrow_schema.field_with_name("vector").is_ok());
    }

    #[test]
    fn test_proxima_to_arrow_type() {
        assert_eq!(
            proxima_to_arrow_type(&ProximaType::Int64),
            ArrowDataType::Int64
        );
        assert_eq!(
            proxima_to_arrow_type(&ProximaType::String),
            ArrowDataType::Utf8
        );
        assert_eq!(
            proxima_to_arrow_type(&ProximaType::Boolean),
            ArrowDataType::Boolean
        );
    }

    #[test]
    fn test_vector_type_conversion() {
        let vector_type = ProximaType::DenseVector {
            element: VectorElement::Float32,
            dim: 128,
        };

        let arrow_type = proxima_to_arrow_type(&vector_type);

        match arrow_type {
            ArrowDataType::FixedSizeList(_, size) => {
                assert_eq!(size, 128);
            }
            _ => panic!("Expected FixedSizeList"),
        }
    }

    use proximadb_storage_common::format_splits::{ColumnBounds, SplitStatistics};

    fn split_with_bounds(
        rows: u64,
        bytes: u64,
        bounds: &[(&str, serde_json::Value, serde_json::Value, u64)],
    ) -> FileSplit {
        let mut split = FileSplit::new_row_group("f.parquet".to_string(), 0, 0, bytes, rows as i64);
        let mut stats = SplitStatistics {
            row_count: Some(rows),
            byte_size: Some(bytes),
            ..Default::default()
        };
        for (name, min, max, nulls) in bounds {
            stats.column_stats.insert(
                (*name).to_string(),
                ColumnBounds {
                    min: Some(min.clone()),
                    max: Some(max.clone()),
                    null_count: *nulls,
                    distinct_count: None,
                },
            );
        }
        split.statistics = stats;
        split
    }

    fn two_col_schema() -> ArrowSchema {
        ArrowSchema::new(vec![
            Field::new("x", ArrowDataType::Int64, false),
            Field::new("name", ArrowDataType::Utf8, true),
        ])
    }

    #[test]
    fn statistics_from_splits_merges_bounds_schema_width() {
        let schema = two_col_schema();
        let splits = vec![
            split_with_bounds(
                2,
                100,
                &[
                    ("x", serde_json::json!(1), serde_json::json!(10), 0),
                    ("name", serde_json::json!("a"), serde_json::json!("m"), 1),
                ],
            ),
            split_with_bounds(
                3,
                200,
                &[
                    ("x", serde_json::json!(-5), serde_json::json!(7), 2),
                    ("name", serde_json::json!("c"), serde_json::json!("z"), 0),
                ],
            ),
        ];

        let stats = statistics_from_splits(
            &splits,
            &schema,
            None,
            true,
            proximadb_data_model::StatsTrust::Trusted,
        );

        assert_eq!(stats.num_rows, Precision::Exact(5));
        assert_eq!(stats.total_byte_size, Precision::Inexact(300));
        // Schema-width contract (DataFusion 54 indexes by column position).
        assert_eq!(stats.column_statistics.len(), schema.fields().len());

        // Numeric column: parquet min/max are exact → `Exact` (elidable MIN/MAX);
        // null_count is exact footer metadata → `Exact`.
        let x = &stats.column_statistics[0];
        assert_eq!(
            x.min_value,
            Precision::Exact(DfScalarValue::Int64(Some(-5)))
        );
        assert_eq!(
            x.max_value,
            Precision::Exact(DfScalarValue::Int64(Some(10)))
        );
        assert_eq!(x.null_count, Precision::Exact(2));

        // String column: parquet BYTE_ARRAY min/max are truncated → stay `Inexact`
        // (zone-map pruning only); null_count is still exact → `Exact`.
        let name = &stats.column_statistics[1];
        assert_eq!(
            name.min_value,
            Precision::Inexact(DfScalarValue::Utf8(Some("a".to_string())))
        );
        assert_eq!(
            name.max_value,
            Precision::Inexact(DfScalarValue::Utf8(Some("z".to_string())))
        );
        assert_eq!(name.null_count, Precision::Exact(1));
    }

    #[test]
    fn statistics_from_splits_partial_coverage_stays_unknown() {
        let schema = two_col_schema();
        // Second split carries NO bounds for `x` — claiming merged bounds
        // would understate the true range, so the column must stay unknown.
        let splits = vec![
            split_with_bounds(
                2,
                10,
                &[("x", serde_json::json!(1), serde_json::json!(2), 0)],
            ),
            split_with_bounds(2, 10, &[]),
        ];

        let stats = statistics_from_splits(
            &splits,
            &schema,
            None,
            true,
            proximadb_data_model::StatsTrust::Trusted,
        );
        let x = &stats.column_statistics[0];
        assert_eq!(x.min_value, Precision::Absent);
        assert_eq!(x.max_value, Precision::Absent);
        assert_eq!(x.null_count, Precision::Absent);
    }

    #[test]
    fn statistics_from_splits_overlays_registry_ndv() {
        let collection = "schema_inference_ndv_overlay_test";
        let int64_type = serde_json::to_value(ProximaType::Int64).expect("serde ProximaType");
        crate::core::statistics::statistics_registry().update(collection, |summary| {
            for i in 0..64 {
                summary.observe_field("x", &int64_type, Some(&serde_json::json!(i)));
            }
        });

        let schema = two_col_schema();
        let splits = vec![split_with_bounds(
            64,
            10,
            &[("x", serde_json::json!(0), serde_json::json!(63), 0)],
        )];

        let stats = statistics_from_splits(
            &splits,
            &schema,
            Some(collection),
            true,
            proximadb_data_model::StatsTrust::Trusted,
        );
        crate::core::statistics::statistics_registry().remove(collection);

        match stats.column_statistics[0].distinct_count {
            Precision::Inexact(ndv) => {
                assert!(
                    (32..=128).contains(&ndv),
                    "HLL NDV estimate for 64 distinct values, got {ndv}"
                );
            }
            ref other => panic!("expected Inexact NDV from the registry, got {other:?}"),
        }
        // Column without a registry field stays absent.
        assert_eq!(stats.column_statistics[1].distinct_count, Precision::Absent);
    }

    #[test]
    fn statistics_from_splits_gates_value_bounds_off() {
        let schema = two_col_schema();
        let splits = vec![split_with_bounds(
            2,
            10,
            &[("x", serde_json::json!(1), serde_json::json!(2), 3)],
        )];

        let stats = statistics_from_splits(
            &splits,
            &schema,
            None,
            false,
            proximadb_data_model::StatsTrust::Trusted,
        );
        let x = &stats.column_statistics[0];
        // Bounds suppressed (the recursive interval-analysis trigger)…
        assert_eq!(x.min_value, Precision::Absent);
        assert_eq!(x.max_value, Precision::Absent);
        // …but the join-ordering inputs still flow — null_count is exact footer
        // metadata (COUNT(col) elision is independent of the value-bounds gate).
        assert_eq!(x.null_count, Precision::Exact(3));
        assert_eq!(stats.num_rows, Precision::Exact(2));
    }

    #[test]
    fn statistics_from_splits_numeric_bounds_exact_string_inexact() {
        // The MIN/MAX metadata-elision contract: DataFusion's `AggregateStatistics`
        // rule elides `MIN`/`MAX` only when the column bound is `Exact`. Parquet
        // numeric stats are exact, so those must be `Exact` (killing ClickBench
        // q07's `MIN/MAX(EventDate)` scan); BYTE_ARRAY (string) stats are truncated
        // prefixes, so they must stay `Inexact` — an elided string MIN/MAX off a
        // truncated bound would be WRONG.
        let schema = ArrowSchema::new(vec![
            Field::new("f", ArrowDataType::Float64, false),
            Field::new("s", ArrowDataType::Utf8, true),
        ]);
        let splits = vec![split_with_bounds(
            4,
            40,
            &[
                ("f", serde_json::json!(1.5), serde_json::json!(9.5), 0),
                ("s", serde_json::json!("aaa"), serde_json::json!("zzz"), 2),
            ],
        )];

        let stats = statistics_from_splits(
            &splits,
            &schema,
            None,
            true,
            proximadb_data_model::StatsTrust::Trusted,
        );

        let f = &stats.column_statistics[0];
        assert_eq!(
            f.min_value,
            Precision::Exact(DfScalarValue::Float64(Some(1.5)))
        );
        assert_eq!(
            f.max_value,
            Precision::Exact(DfScalarValue::Float64(Some(9.5)))
        );
        assert_eq!(f.null_count, Precision::Exact(0));

        let s = &stats.column_statistics[1];
        assert!(matches!(s.min_value, Precision::Inexact(_)));
        assert!(matches!(s.max_value, Precision::Inexact(_)));
        // null_count is exact metadata for every column type.
        assert_eq!(s.null_count, Precision::Exact(2));
    }

    #[test]
    fn project_statistics_narrows_to_projection() {
        let schema = two_col_schema();
        let splits = vec![split_with_bounds(
            2,
            10,
            &[
                ("x", serde_json::json!(1), serde_json::json!(2), 0),
                ("name", serde_json::json!("a"), serde_json::json!("b"), 0),
            ],
        )];
        let full = statistics_from_splits(
            &splits,
            &schema,
            None,
            true,
            proximadb_data_model::StatsTrust::Trusted,
        );

        let projected = project_statistics(&full, Some(&[1]));
        assert_eq!(projected.column_statistics.len(), 1);
        assert_eq!(
            projected.column_statistics[0].min_value,
            Precision::Inexact(DfScalarValue::Utf8(Some("a".to_string())))
        );

        let unprojected = project_statistics(&full, None);
        assert_eq!(unprojected.column_statistics.len(), 2);
    }

    #[test]
    fn json_bounds_convert_per_arrow_type() {
        assert_eq!(
            json_bound_to_scalar(&serde_json::json!(7), &ArrowDataType::Int32),
            Some(DfScalarValue::Int32(Some(7)))
        );
        assert_eq!(
            json_bound_to_scalar(&serde_json::json!(1.5), &ArrowDataType::Float64),
            Some(DfScalarValue::Float64(Some(1.5)))
        );
        assert_eq!(
            json_bound_to_scalar(&serde_json::json!(true), &ArrowDataType::Boolean),
            Some(DfScalarValue::Boolean(Some(true)))
        );
        // No lossless mapping → None, never a wrong-typed scalar.
        assert_eq!(
            json_bound_to_scalar(&serde_json::json!("s"), &ArrowDataType::Date32),
            None
        );
    }
}
