//! # stats — Parquet footer → Iceberg manifest column metrics
//!
//! Iceberg engines plan scans from the manifest alone: a data file whose
//! `lower_bounds`/`upper_bounds` exclude a predicate is never opened. This
//! module derives those metrics (plus `column_sizes`, `value_counts`,
//! `null_value_counts`) from the Parquet footer of a data file ProximaDB just
//! wrote, so a published snapshot is prunable by Spark / Trino / DuckDB /
//! PyIceberg — and by ProximaDB's own manifest-driven planning.
//!
//! Bounds use the Iceberg spec's *single-value binary serialization*
//! (Appendix D): `int`/`date` 4-byte LE, `long` 8-byte LE, `float`/`double`
//! IEEE-754 LE, `boolean` one byte, `string` UTF-8, `binary` raw. Conservative:
//! a bound is emitted only when every row group carries min/max for the column
//! (all-null row groups excepted); strings and binaries longer than the
//! default `truncate(16)` metrics mode keep a truncated lower bound and drop the
//! upper bound (a truncated upper bound would need incrementing to stay valid).
//! Floating-point bounds are dropped if NaN.

use parquet::file::metadata::ParquetMetaData;
use parquet::file::statistics::Statistics;

use crate::iceberg::{ColumnFileStats, IcebergField};

/// Iceberg's default `write.metadata.metrics.default` = `truncate(16)`.
const TRUNCATE_LEN: usize = 16;

/// A bound in comparable native form, folded across row groups.
#[derive(Clone, Debug, PartialEq, PartialOrd)]
enum Bound {
    Int(i64),
    /// A big-endian two's-complement fixed-length value (wide decimals), widened.
    Wide(i128),
    Float(f64),
    Bool(bool),
    Bytes(Vec<u8>),
}

fn bounds_of(stats: &Statistics) -> Option<(Bound, Bound)> {
    Some(match stats {
        Statistics::Boolean(s) => (Bound::Bool(*s.min_opt()?), Bound::Bool(*s.max_opt()?)),
        Statistics::Int32(s) => (
            Bound::Int(*s.min_opt()? as i64),
            Bound::Int(*s.max_opt()? as i64),
        ),
        Statistics::Int64(s) => (Bound::Int(*s.min_opt()?), Bound::Int(*s.max_opt()?)),
        Statistics::Float(s) => (
            Bound::Float(*s.min_opt()? as f64),
            Bound::Float(*s.max_opt()? as f64),
        ),
        Statistics::Double(s) => (Bound::Float(*s.min_opt()?), Bound::Float(*s.max_opt()?)),
        Statistics::ByteArray(s) => (
            Bound::Bytes(s.min_opt()?.data().to_vec()),
            Bound::Bytes(s.max_opt()?.data().to_vec()),
        ),
        // FIXED_LEN_BYTE_ARRAY carries wide decimals (precision > 18) as big-endian
        // two's complement; only decimal columns encode a bound from it.
        Statistics::FixedLenByteArray(s) => (
            Bound::Wide(be_signed(s.min_opt()?.data())?),
            Bound::Wide(be_signed(s.max_opt()?.data())?),
        ),
        _ => return None,
    })
}

/// Sign-extend big-endian two's-complement bytes (at most 16) to `i128`.
fn be_signed(bytes: &[u8]) -> Option<i128> {
    if bytes.is_empty() || bytes.len() > 16 {
        return None;
    }
    let fill = if bytes[0] & 0x80 != 0 { 0xFF } else { 0x00 };
    let mut buf = [fill; 16];
    buf[16 - bytes.len()..].copy_from_slice(bytes);
    Some(i128::from_be_bytes(buf))
}

/// Iceberg's decimal bound: the unscaled value as the *minimum* number of
/// big-endian two's-complement bytes (Appendix D).
fn decimal_bound(v: i128) -> Vec<u8> {
    let bytes = v.to_be_bytes();
    let mut start = 0;
    while start < 15 {
        let (b, next) = (bytes[start], bytes[start + 1]);
        if (b == 0x00 && next & 0x80 == 0) || (b == 0xFF && next & 0x80 != 0) {
            start += 1;
        } else {
            break;
        }
    }
    bytes[start..].to_vec()
}

/// Serialize a folded bound per the Iceberg type, or `None` when the type is
/// not covered (no bound is always valid).
fn encode(type_name: &str, bound: &Bound, is_upper: bool) -> Option<Vec<u8>> {
    if type_name.starts_with("decimal(") {
        return match bound {
            Bound::Int(v) => Some(decimal_bound(*v as i128)),
            Bound::Wide(v) => Some(decimal_bound(*v)),
            _ => None,
        };
    }
    match (type_name, bound) {
        ("int" | "date", Bound::Int(v)) => i32::try_from(*v).ok().map(|v| v.to_le_bytes().to_vec()),
        ("long", Bound::Int(v)) => Some(v.to_le_bytes().to_vec()),
        ("float", Bound::Float(v)) if !v.is_nan() => Some((*v as f32).to_le_bytes().to_vec()),
        ("double", Bound::Float(v)) if !v.is_nan() => Some(v.to_le_bytes().to_vec()),
        ("boolean", Bound::Bool(b)) => Some(vec![*b as u8]),
        ("string", Bound::Bytes(b)) => {
            let s = std::str::from_utf8(b).ok()?;
            if s.chars().count() <= TRUNCATE_LEN {
                Some(b.clone())
            } else if is_upper {
                None
            } else {
                Some(
                    s.chars()
                        .take(TRUNCATE_LEN)
                        .collect::<String>()
                        .into_bytes(),
                )
            }
        }
        ("binary", Bound::Bytes(b)) => {
            if b.len() <= TRUNCATE_LEN {
                Some(b.clone())
            } else if is_upper {
                None
            } else {
                Some(b[..TRUNCATE_LEN].to_vec())
            }
        }
        _ => None,
    }
}

/// Iceberg column metrics for every `fields` entry that maps to a top-level
/// Parquet leaf column of the same name.
pub fn iceberg_column_stats(
    metadata: &ParquetMetaData,
    fields: &[IcebergField],
) -> Vec<ColumnFileStats> {
    let schema = metadata.file_metadata().schema_descr();
    let mut out = Vec::new();
    for field in fields {
        let Some(leaf) =
            (0..schema.num_columns()).find(|&i| schema.column(i).path().string() == field.name)
        else {
            continue;
        };
        let mut column_size = 0i64;
        let mut value_count = 0i64;
        let mut null_count: Option<i64> = Some(0);
        let mut lower: Option<Bound> = None;
        let mut upper: Option<Bound> = None;
        let mut bounds_valid = true;
        for rg in metadata.row_groups() {
            let chunk = rg.column(leaf);
            column_size += chunk.compressed_size();
            value_count += chunk.num_values();
            let Some(stats) = chunk.statistics() else {
                null_count = None;
                bounds_valid = false;
                continue;
            };
            let chunk_nulls = stats.null_count_opt().map(|n| n as i64);
            null_count = null_count.zip(chunk_nulls).map(|(a, b)| a + b);
            match bounds_of(stats) {
                Some((lo, hi)) => {
                    lower = Some(match lower {
                        Some(cur) if cur <= lo => cur,
                        _ => lo,
                    });
                    upper = Some(match upper {
                        Some(cur) if cur >= hi => cur,
                        _ => hi,
                    });
                }
                // All-null chunk: contributes no values, bounds stay valid.
                None if chunk_nulls == Some(chunk.num_values()) => {}
                None => bounds_valid = false,
            }
        }
        // A decimal bound is the column's UNSCALED value, so it is only meaningful
        // when the Parquet column really is DECIMAL-annotated (same scale);
        // otherwise a plain integer would be misread as unscaled.
        let decimal_ok = !field.type_name.starts_with("decimal(")
            || matches!(
                schema.column(leaf).logical_type_ref(),
                Some(parquet::basic::LogicalType::Decimal { .. })
            );
        let (lower_bound, upper_bound) = if bounds_valid && decimal_ok {
            (
                lower
                    .as_ref()
                    .and_then(|b| encode(&field.type_name, b, false)),
                upper
                    .as_ref()
                    .and_then(|b| encode(&field.type_name, b, true)),
            )
        } else {
            (None, None)
        };
        out.push(ColumnFileStats {
            field_id: field.id,
            column_size: Some(column_size),
            value_count: Some(value_count),
            null_value_count: null_count,
            lower_bound,
            upper_bound,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use bytes::Bytes;
    use parquet::arrow::ArrowWriter;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::file::properties::WriterProperties;
    use std::sync::Arc;

    fn field(id: i32, name: &str, t: &str) -> IcebergField {
        IcebergField {
            id,
            name: name.to_string(),
            type_name: t.to_string(),
            required: false,
        }
    }

    fn footer() -> Arc<ParquetMetaData> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("d", DataType::Int32, false),
            Field::new("k", DataType::Int64, true),
            Field::new("s", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from((10..20).collect::<Vec<i32>>())),
                Arc::new(Int64Array::from(vec![
                    Some(-5),
                    None,
                    Some(7),
                    Some(3),
                    None,
                    Some(100),
                    Some(1),
                    Some(2),
                    Some(3),
                    Some(4),
                ])),
                Arc::new(StringArray::from(vec![
                    Some("apple"),
                    Some("pear"),
                    None,
                    Some("zebra-with-a-very-long-name"),
                    Some("fig"),
                    Some("kiwi"),
                    Some("lime"),
                    Some("date"),
                    Some("plum"),
                    Some("aardvark-long-string-value"),
                ])),
            ],
        )
        .unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(4))
            .build();
        let mut buf = Vec::new();
        let mut w = ArrowWriter::try_new(&mut buf, schema, Some(props)).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
        ParquetRecordBatchReaderBuilder::try_new(Bytes::from(buf))
            .unwrap()
            .metadata()
            .clone()
    }

    #[test]
    fn folds_bounds_and_counts_across_row_groups() {
        let meta = footer();
        assert_eq!(meta.num_row_groups(), 3);
        let stats = iceberg_column_stats(
            &meta,
            &[
                field(1, "d", "int"),
                field(2, "k", "long"),
                field(3, "s", "string"),
            ],
        );
        let d = &stats[0];
        assert_eq!(d.field_id, 1);
        assert_eq!(d.value_count, Some(10));
        assert_eq!(d.null_value_count, Some(0));
        assert_eq!(d.lower_bound.as_deref(), Some(&10i32.to_le_bytes()[..]));
        assert_eq!(d.upper_bound.as_deref(), Some(&19i32.to_le_bytes()[..]));

        let k = &stats[1];
        assert_eq!(k.null_value_count, Some(2));
        assert_eq!(k.lower_bound.as_deref(), Some(&(-5i64).to_le_bytes()[..]));
        assert_eq!(k.upper_bound.as_deref(), Some(&100i64.to_le_bytes()[..]));

        // Strings: the min ("aardvark-long-string-value") is truncated to 16
        // chars; the max ("zebra-with-…", > 16 chars) is dropped.
        let s = &stats[2];
        assert_eq!(s.lower_bound.as_deref(), Some(&b"aardvark-long-st"[..]));
        assert_eq!(s.upper_bound, None);
        assert_eq!(s.null_value_count, Some(1));
    }

    #[test]
    fn unknown_columns_and_types_are_skipped() {
        let meta = footer();
        let stats = iceberg_column_stats(
            &meta,
            &[field(9, "missing", "long"), field(1, "d", "decimal(9,2)")],
        );
        assert_eq!(stats.len(), 1);
        assert_eq!(
            stats[0].lower_bound, None,
            "decimal over a non-DECIMAL Parquet column → no bound"
        );
        assert_eq!(stats[0].value_count, Some(10));
    }

    #[test]
    fn decimal_bounds_use_minimal_big_endian_twos_complement() {
        assert_eq!(decimal_bound(0), vec![0x00]);
        assert_eq!(decimal_bound(1250), vec![0x04, 0xE2]);
        assert_eq!(decimal_bound(127), vec![0x7F]);
        assert_eq!(decimal_bound(128), vec![0x00, 0x80]);
        assert_eq!(decimal_bound(-1), vec![0xFF]);
        assert_eq!(decimal_bound(-129), vec![0xFF, 0x7F]);
        assert_eq!(be_signed(&[0xFF, 0x7F]), Some(-129));
        assert_eq!(be_signed(&[0x04, 0xE2]), Some(1250));
        assert_eq!(
            encode("decimal(10,2)", &Bound::Int(1250), false),
            Some(vec![0x04, 0xE2])
        );
        assert_eq!(
            encode("decimal(30,2)", &Bound::Wide(-129), true),
            Some(vec![0xFF, 0x7F])
        );
    }

    #[test]
    fn decimal_column_gets_manifest_bounds() {
        use arrow_array::Decimal128Array;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "amt",
            DataType::Decimal128(10, 2),
            false,
        )]));
        let values = Decimal128Array::from(vec![1250_i128, -99, 100_000])
            .with_precision_and_scale(10, 2)
            .unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(values)]).unwrap();
        let mut buf = Vec::new();
        let mut w = ArrowWriter::try_new(&mut buf, schema, None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
        let meta = ParquetRecordBatchReaderBuilder::try_new(Bytes::from(buf))
            .unwrap()
            .metadata()
            .clone();
        let stats = iceberg_column_stats(&meta, &[field(1, "amt", "decimal(10,2)")]);
        assert_eq!(stats[0].lower_bound, Some(decimal_bound(-99)));
        assert_eq!(stats[0].upper_bound, Some(decimal_bound(100_000)));
    }
}
