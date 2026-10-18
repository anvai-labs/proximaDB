//! # parquet_page_prune — engine-neutral Parquet page-index pruning
//!
//! Turns the Parquet page index (`ColumnIndex` per-page min/max/null-count +
//! `OffsetIndex` page row offsets, parquet-format spec) into a [`RowSelection`]
//! for one row group, so a reader decodes only the data pages whose bounds can
//! satisfy the scan's conjunctive predicates. This is the level *below* the
//! row-group zone map: on clustered data a row group that survives min/max can
//! still have most of its pages skipped.
//!
//! Engine-neutral by construction (ADR-039 / ADR-054): predicates are the shared
//! [`ScalarPredicate`] the split-pruning layer already uses, and page bounds are
//! evaluated with the *same* [`ColumnBounds::can_prune`] semantics as row-group
//! pruning — so the DataFusion provider and the native Parquet scan prune
//! identically. Conservative everywhere: a missing page index, an unknown column,
//! an unsupported type, or a page without statistics keeps the rows. Callers must
//! still apply the predicate row-by-row (the selection is a superset).

use arrow_array::cast::AsArray;
use arrow_array::types::*;
use arrow_array::{Array, ArrayRef, UInt64Array};
use arrow_schema::{DataType, Schema};
use parquet::arrow::arrow_reader::statistics::StatisticsConverter;
use parquet::arrow::arrow_reader::{RowSelection, RowSelector};
use parquet::file::metadata::ParquetMetaData;

use crate::format_splits::{ColumnBounds, ScalarPredicate};

/// Page-pruning outcome for one row group.
#[derive(Debug, Clone, Default)]
pub struct PagePruneOutcome {
    /// Rows to decode; `None` = read the whole row group (nothing prunable).
    pub selection: Option<RowSelection>,
    /// Rows in the row group.
    pub rows_total: u64,
    /// Rows the selection skips (elided before decode).
    pub rows_skipped: u64,
    /// Data pages evaluated across the predicate columns.
    pub pages_total: u64,
    /// Of those, pages whose bounds excluded them.
    pub pages_pruned: u64,
}

/// Compute the page-level [`RowSelection`] for `row_group` under the conjunction
/// of `predicates` (`(column, predicate)` pairs). `metadata` must have been
/// loaded with the page index (`ArrowReaderOptions::with_page_index_policy`);
/// without it this returns "read everything".
pub fn page_row_selection(
    metadata: &ParquetMetaData,
    arrow_schema: &Schema,
    row_group: usize,
    predicates: &[(String, ScalarPredicate)],
) -> PagePruneOutcome {
    let Some(rg_meta) = metadata.row_groups().get(row_group) else {
        return PagePruneOutcome::default();
    };
    let rows_total = rg_meta.num_rows().max(0) as u64;
    let mut outcome = PagePruneOutcome {
        rows_total,
        ..Default::default()
    };
    let (Some(column_index), Some(offset_index)) =
        (metadata.column_index(), metadata.offset_index())
    else {
        return outcome;
    };
    let parquet_schema = metadata.file_metadata().schema_descr();

    let mut combined: Option<RowSelection> = None;
    for (column, predicate) in predicates {
        let Ok(converter) = StatisticsConverter::try_new(column, arrow_schema, parquet_schema)
        else {
            continue;
        };
        let rgs = [row_group];
        let (Ok(mins), Ok(maxes), Ok(nulls), Ok(Some(row_counts))) = (
            converter.data_page_mins(column_index, offset_index, &rgs),
            converter.data_page_maxes(column_index, offset_index, &rgs),
            converter.data_page_null_counts(column_index, offset_index, &rgs),
            converter.data_page_row_counts(offset_index, metadata.row_groups(), &rgs),
        ) else {
            continue;
        };
        let Some(keep) = page_keep_mask(&mins, &maxes, &nulls, predicate) else {
            continue;
        };
        if keep.len() != row_counts.len() {
            continue;
        }
        outcome.pages_total += keep.len() as u64;
        outcome.pages_pruned += keep.iter().filter(|k| !**k).count() as u64;
        if keep.iter().all(|k| *k) {
            continue;
        }
        let selection = selection_from_pages(&keep, &row_counts);
        combined = Some(match combined {
            Some(prev) => prev.intersection(&selection),
            None => selection,
        });
    }

    if let Some(selection) = combined {
        let selected = selection.row_count() as u64;
        outcome.rows_skipped = rows_total.saturating_sub(selected);
        outcome.selection = Some(selection);
    }
    outcome
}

/// Can row group `row_group` be skipped entirely under the conjunction of
/// `predicates`, judged from its footer (chunk-level) statistics? Same
/// [`ColumnBounds::can_prune`] semantics as split pruning and page pruning, so
/// every engine prunes identically. Conservative: an unknown column, an
/// unsupported type, or missing statistics never prunes.
pub fn row_group_prunable(
    metadata: &ParquetMetaData,
    arrow_schema: &Schema,
    row_group: usize,
    predicates: &[(String, ScalarPredicate)],
) -> bool {
    if row_group >= metadata.num_row_groups() {
        return false;
    }
    let parquet_schema = metadata.file_metadata().schema_descr();
    let groups = &metadata.row_groups()[row_group..=row_group];
    predicates.iter().any(|(column, predicate)| {
        let Ok(converter) = StatisticsConverter::try_new(column, arrow_schema, parquet_schema)
        else {
            return false;
        };
        let (Ok(mins), Ok(maxes), Ok(nulls)) = (
            converter.row_group_mins(groups),
            converter.row_group_maxes(groups),
            converter.row_group_null_counts(groups),
        ) else {
            return false;
        };
        page_keep_mask(&mins, &maxes, &nulls, predicate)
            .is_some_and(|keep| keep.len() == 1 && !keep[0])
    })
}

/// Per-page "may contain a match" mask, or `None` when the column type is not
/// representable in the shared bound model (then no page is pruned).
fn page_keep_mask(
    mins: &ArrayRef,
    maxes: &ArrayRef,
    nulls: &UInt64Array,
    predicate: &ScalarPredicate,
) -> Option<Vec<bool>> {
    if mins.len() != maxes.len() {
        return None;
    }
    let mut keep = Vec::with_capacity(mins.len());
    for page in 0..mins.len() {
        let min = json_value_at(mins, page)?;
        let max = json_value_at(maxes, page)?;
        // IS NULL prunes on null_count == 0; an unknown null count must not
        // masquerade as zero.
        if nulls.is_null(page) && matches!(predicate, ScalarPredicate::IsNull) {
            keep.push(true);
            continue;
        }
        // A page with no recorded min/max (all-null page or stats missing) is kept
        // unless the shared bounds logic can decide on null-count alone.
        let bounds = ColumnBounds {
            min,
            max,
            null_count: if nulls.is_null(page) {
                0
            } else {
                nulls.value(page)
            },
            distinct_count: None,
        };
        let prunable = bounds.min.is_some() && bounds.max.is_some() && bounds.can_prune(predicate);
        keep.push(!prunable);
    }
    Some(keep)
}

/// Element `i` of a page-statistics array as the shared JSON bound encoding
/// (mirrors the row-group mapping: integers and dates widen to i64 — dates as
/// days since epoch — floats to f64, strings verbatim). `Some(None)` = the page
/// has no value for this bound; outer `None` = unsupported type.
fn json_value_at(array: &ArrayRef, i: usize) -> Option<Option<serde_json::Value>> {
    if array.is_null(i) {
        return Some(None);
    }
    let v = match array.data_type() {
        DataType::Int8 => serde_json::json!(array.as_primitive::<Int8Type>().value(i) as i64),
        DataType::Int16 => serde_json::json!(array.as_primitive::<Int16Type>().value(i) as i64),
        DataType::Int32 => serde_json::json!(array.as_primitive::<Int32Type>().value(i) as i64),
        DataType::Int64 => serde_json::json!(array.as_primitive::<Int64Type>().value(i)),
        DataType::UInt8 => serde_json::json!(array.as_primitive::<UInt8Type>().value(i) as i64),
        DataType::UInt16 => serde_json::json!(array.as_primitive::<UInt16Type>().value(i) as i64),
        DataType::UInt32 => serde_json::json!(array.as_primitive::<UInt32Type>().value(i) as i64),
        DataType::Date32 => serde_json::json!(array.as_primitive::<Date32Type>().value(i) as i64),
        DataType::Float32 => serde_json::json!(array.as_primitive::<Float32Type>().value(i) as f64),
        DataType::Float64 => serde_json::json!(array.as_primitive::<Float64Type>().value(i)),
        DataType::Boolean => serde_json::json!(array.as_boolean().value(i)),
        DataType::Utf8 => serde_json::json!(array.as_string::<i32>().value(i)),
        DataType::LargeUtf8 => serde_json::json!(array.as_string::<i64>().value(i)),
        DataType::Utf8View => serde_json::json!(array.as_string_view().value(i)),
        _ => return None,
    };
    Some(Some(v))
}

/// Contiguous select/skip runs from a per-page keep mask.
fn selection_from_pages(keep: &[bool], row_counts: &UInt64Array) -> RowSelection {
    let mut selectors: Vec<RowSelector> = Vec::new();
    for (page, &k) in keep.iter().enumerate() {
        let rows = if row_counts.is_null(page) {
            0
        } else {
            row_counts.value(page) as usize
        };
        if rows == 0 {
            continue;
        }
        match selectors.last_mut() {
            Some(last) if last.skip != k => last.row_count += rows,
            _ => selectors.push(if k {
                RowSelector::select(rows)
            } else {
                RowSelector::skip(rows)
            }),
        }
    }
    RowSelection::from(selectors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format_splits::ScalarValue;
    use crate::parquet_write_policy::{ParquetLayoutHints, ParquetWritePolicy};
    use crate::proxima_parquet::record_batches_to_parquet_bytes;
    use arrow_array::{Int64Array, RecordBatch, StringArray};
    use arrow_schema::Field;
    use bytes::Bytes;
    use parquet::arrow::arrow_reader::{ArrowReaderOptions, ParquetRecordBatchReaderBuilder};
    use parquet::file::metadata::PageIndexPolicy;
    use std::sync::Arc;

    /// 10k rows in one row group, 1k-row pages, `k` sorted 0..10k.
    fn sorted_file() -> (Bytes, Arc<Schema>) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("s", DataType::Utf8, false),
        ]));
        let k: Vec<i64> = (0..10_000).collect();
        let s: Vec<String> = (0..10_000).map(|i| format!("v{:05}", i)).collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(k)),
                Arc::new(StringArray::from(s)),
            ],
        )
        .unwrap();
        let policy = ParquetWritePolicy {
            row_group_rows: 10_000,
            data_page_rows: 1_000,
            ..ParquetWritePolicy::default()
        };
        let props = policy.writer_properties(&batch, &ParquetLayoutHints::default());
        let bytes = record_batches_to_parquet_bytes(&[batch], schema.clone(), Some(props)).unwrap();
        (Bytes::from(bytes), schema)
    }

    fn metadata(bytes: &Bytes) -> Arc<ParquetMetaData> {
        let opts = ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required);
        ParquetRecordBatchReaderBuilder::try_new_with_options(bytes.clone(), opts)
            .unwrap()
            .metadata()
            .clone()
    }

    fn rows_read(bytes: &Bytes, selection: Option<RowSelection>) -> Vec<i64> {
        let opts = ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required);
        let mut b =
            ParquetRecordBatchReaderBuilder::try_new_with_options(bytes.clone(), opts).unwrap();
        if let Some(sel) = selection {
            b = b.with_row_selection(sel);
        }
        let mut out = Vec::new();
        for batch in b.build().unwrap() {
            let batch = batch.unwrap();
            out.extend(
                batch
                    .column(0)
                    .as_primitive::<Int64Type>()
                    .values()
                    .iter()
                    .copied(),
            );
        }
        out
    }

    #[test]
    fn range_predicate_skips_pages_outside_bounds() {
        let (bytes, schema) = sorted_file();
        let meta = metadata(&bytes);
        let preds = vec![(
            "k".to_string(),
            ScalarPredicate::Between(ScalarValue::Int64(2_500), ScalarValue::Int64(3_200)),
        )];
        let out = page_row_selection(&meta, &schema, 0, &preds);
        assert_eq!(out.pages_total, 10);
        assert_eq!(
            out.pages_pruned, 8,
            "only pages [2000,3000) and [3000,4000) survive"
        );
        assert_eq!(out.rows_skipped, 8_000);
        let rows = rows_read(&bytes, out.selection);
        assert_eq!(rows.len(), 2_000);
        // No false negatives: every matching row is still read.
        assert!((2_500..=3_200).all(|v| rows.contains(&v)));
    }

    #[test]
    fn conjunction_intersects_and_string_bounds_prune() {
        let (bytes, schema) = sorted_file();
        let meta = metadata(&bytes);
        let preds = vec![
            (
                "k".to_string(),
                ScalarPredicate::GreaterThanOrEqual(ScalarValue::Int64(5_000)),
            ),
            (
                "s".to_string(),
                ScalarPredicate::Equal(ScalarValue::String("v07123".into())),
            ),
        ];
        let out = page_row_selection(&meta, &schema, 0, &preds);
        assert_eq!(out.rows_skipped, 9_000);
        let rows = rows_read(&bytes, out.selection);
        assert!(rows.contains(&7_123));
        assert_eq!(rows.len(), 1_000);
    }

    #[test]
    fn unprunable_or_unknown_predicates_read_everything() {
        let (bytes, schema) = sorted_file();
        let meta = metadata(&bytes);
        let preds = vec![
            (
                "k".to_string(),
                ScalarPredicate::GreaterThanOrEqual(ScalarValue::Int64(0)),
            ),
            (
                "missing".to_string(),
                ScalarPredicate::Equal(ScalarValue::Int64(1)),
            ),
        ];
        let out = page_row_selection(&meta, &schema, 0, &preds);
        assert!(out.selection.is_none());
        assert_eq!(out.rows_skipped, 0);
    }

    #[test]
    fn row_group_prunable_uses_footer_bounds() {
        let (bytes, schema) = sorted_file();
        let meta = metadata(&bytes);
        let outside = vec![(
            "k".to_string(),
            ScalarPredicate::GreaterThan(ScalarValue::Int64(20_000)),
        )];
        let inside = vec![(
            "k".to_string(),
            ScalarPredicate::Equal(ScalarValue::Int64(42)),
        )];
        let unknown = vec![(
            "nope".to_string(),
            ScalarPredicate::Equal(ScalarValue::Int64(1)),
        )];
        assert!(row_group_prunable(&meta, &schema, 0, &outside));
        assert!(!row_group_prunable(&meta, &schema, 0, &inside));
        assert!(!row_group_prunable(&meta, &schema, 0, &unknown));
        assert!(
            !row_group_prunable(&meta, &schema, 7, &outside),
            "out-of-range group never prunes"
        );
    }

    #[test]
    fn without_page_index_nothing_is_pruned() {
        let (bytes, schema) = sorted_file();
        let meta = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())
            .unwrap()
            .metadata()
            .clone();
        let preds = vec![(
            "k".to_string(),
            ScalarPredicate::Equal(ScalarValue::Int64(42)),
        )];
        let out = page_row_selection(&meta, &schema, 0, &preds);
        assert!(out.selection.is_none());
    }
}
