//! # parquet_write_policy — spec-grounded Parquet writer policy for the warehouse tier
//!
//! One place that decides *how* ProximaDB lays out a Parquet data file, so every
//! warehouse writer (MATERIALIZE snapshots, Iceberg data files, bulk append) emits
//! the same reader-prunable geometry instead of the Arrow defaults. Every knob maps
//! to a structure defined by the Apache Parquet format spec (`parquet-format`), so
//! any spec-compliant reader — DataFusion, DuckDB, Spark, Trino, PyIceberg — can use
//! it, not just ProximaDB:
//!
//! | Knob | Parquet-format structure | Reader benefit |
//! |---|---|---|
//! | row-group cap | `RowGroup` | split granularity for zone-map / runtime-filter skip |
//! | page statistics | `ColumnIndex` + `OffsetIndex` (page index) | page-level skip inside a row group |
//! | data-page row limit | `PageLocation.first_row_index` | page-index skip granularity |
//! | Bloom filters | `BloomFilterHeader` — Split Block Bloom Filter, XXH64 seed 0 | equality / IN skip where min/max cannot |
//! | sort order | `RowGroup.sorting_columns` | readers may assume order (merge / limit / skip) |
//!
//! The Bloom filter is the spec's Split Block Bloom Filter (SBBF: 256-bit blocks of
//! eight 32-bit words, eight salted probes, XXH64 seed 0) as implemented by the
//! `parquet` crate — *not* the in-house `proximadb-bloom` k-hash filter, which no
//! external engine can read.
//!
//! All knobs are environment-overridable so a single build can A/B the layout
//! (the TPC perf ledger runs both arms). Layout is reader geometry, not a format
//! change: every file written under any setting is a plain, mixed-read-safe Parquet
//! file.

use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType, Schema};
use parquet::file::metadata::SortingColumn;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet::schema::types::ColumnPath;
use std::collections::HashSet;

/// Default rows per row group (see [`ParquetWritePolicy::row_group_rows`]).
pub const DEFAULT_ROW_GROUP_ROWS: usize = 65_536;
/// Default rows per data page — the page-index skip granularity.
pub const DEFAULT_DATA_PAGE_ROWS: usize = 8_192;
/// Default row cap per data file of a multi-part publication
/// (see [`ParquetWritePolicy::target_file_rows`]).
pub const DEFAULT_TARGET_FILE_ROWS: usize = 16 * 1024 * 1024;
/// Default byte target per data file of a multi-part publication — the Iceberg
/// `write.target-file-size-bytes` order of magnitude, so parts stay parallel
/// read units without degenerating into many small objects.
pub const DEFAULT_TARGET_FILE_BYTES: usize = 256 * 1024 * 1024;
/// Default Bloom false-positive probability (1%, ≈9.6 bits per distinct value).
pub const DEFAULT_BLOOM_FPP: f64 = 0.01;

/// Writer-side hints the caller knows and the data cannot reveal.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParquetLayoutHints {
    /// Columns the rows were sorted by (ascending, nulls last), outermost first.
    /// Declared in `RowGroup.sorting_columns`; the leading sort column is also
    /// excluded from Bloom filters because its min/max is already exact.
    pub sort_columns: Vec<String>,
}

/// Bloom-filter policy (SBBF per the Parquet spec).
#[derive(Debug, Clone, PartialEq)]
pub struct BloomPolicy {
    /// Master switch (default on; `PROXIMADB_PARQUET_WRITE_BLOOM_DISABLE=1` turns it off).
    pub enabled: bool,
    /// Target false-positive probability (`PROXIMADB_PARQUET_WRITE_BLOOM_FPP`).
    pub fpp: f64,
    /// Explicit column allow-list (`PROXIMADB_PARQUET_WRITE_BLOOM_COLUMNS`, comma
    /// separated). `None` = automatic selection by type (see [`bloom_eligible`]).
    pub columns: Option<Vec<String>>,
    /// Files with fewer rows than this get no Bloom filters: a file that fits a
    /// single data page is read whole anyway.
    pub min_rows: usize,
}

/// The warehouse Parquet layout policy.
#[derive(Debug, Clone, PartialEq)]
pub struct ParquetWritePolicy {
    /// Max rows per row group (`PROXIMADB_MATERIALIZE_ROW_GROUP_ROWS`). The Arrow
    /// default (1,048,576) packs any table under ~1M rows into a single split whose
    /// whole-domain min/max can never prune (measured 0% skip in
    /// `TPC_PERF_GATE_EVIDENCE_2026_07_04`).
    pub row_group_rows: usize,
    /// Max rows per data page (`PROXIMADB_PARQUET_WRITE_PAGE_ROWS`).
    pub data_page_rows: usize,
    /// Write page-level statistics → `ColumnIndex` (default on;
    /// `PROXIMADB_PARQUET_WRITE_PAGE_INDEX_DISABLE=1` leaves the Arrow default).
    pub page_statistics: bool,
    /// Declare `sorting_columns` from [`ParquetLayoutHints`] (default on;
    /// `PROXIMADB_PARQUET_WRITE_SORTING_DISABLE=1` turns it off).
    pub declare_sorting: bool,
    pub bloom: BloomPolicy,
    /// A multi-part writer starts a new data file once the current one holds this
    /// many rows (`PROXIMADB_MATERIALIZE_TARGET_FILE_ROWS`). Exact: a batch that
    /// crosses the cap is split.
    pub target_file_rows: usize,
    /// A multi-part writer starts a new data file once the current one reaches
    /// about this many bytes, encoded plus buffered
    /// (`PROXIMADB_MATERIALIZE_TARGET_FILE_BYTES`). Checked between batches, so a
    /// part may overshoot by one batch.
    pub target_file_bytes: usize,
}

impl Default for ParquetWritePolicy {
    fn default() -> Self {
        Self {
            row_group_rows: DEFAULT_ROW_GROUP_ROWS,
            data_page_rows: DEFAULT_DATA_PAGE_ROWS,
            page_statistics: true,
            declare_sorting: true,
            bloom: BloomPolicy {
                enabled: true,
                fpp: DEFAULT_BLOOM_FPP,
                columns: None,
                min_rows: DEFAULT_DATA_PAGE_ROWS,
            },
            target_file_rows: DEFAULT_TARGET_FILE_ROWS,
            target_file_bytes: DEFAULT_TARGET_FILE_BYTES,
        }
    }
}

/// Kill-switch convention (ENV_GATE_REGISTRY rule 3): `…_DISABLE` set to
/// `1|true|on|yes` turns the shipped-default-ON feature off.
fn env_disabled(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| {
        v == "1"
            || v.eq_ignore_ascii_case("true")
            || v.eq_ignore_ascii_case("on")
            || v.eq_ignore_ascii_case("yes")
    })
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
}

impl ParquetWritePolicy {
    /// The default policy with environment overrides applied.
    /// `PROXIMADB_PARQUET_WRITE_POLICY=legacy` restores the exact pre-policy layout
    /// (row-group cap only) for A/B measurement.
    pub fn from_env() -> Self {
        let d = Self::default();
        // File sizing is publication geometry, not page/row-group layout, so it
        // applies under the legacy layout too.
        let target_file_rows =
            env_usize("PROXIMADB_MATERIALIZE_TARGET_FILE_ROWS").unwrap_or(d.target_file_rows);
        let target_file_bytes =
            env_usize("PROXIMADB_MATERIALIZE_TARGET_FILE_BYTES").unwrap_or(d.target_file_bytes);
        if std::env::var("PROXIMADB_PARQUET_WRITE_POLICY")
            .is_ok_and(|v| v.eq_ignore_ascii_case("legacy"))
        {
            return Self {
                target_file_rows,
                target_file_bytes,
                ..Self::legacy(
                    env_usize("PROXIMADB_MATERIALIZE_ROW_GROUP_ROWS").unwrap_or(d.row_group_rows),
                )
            };
        }
        Self {
            target_file_rows,
            target_file_bytes,
            row_group_rows: env_usize("PROXIMADB_MATERIALIZE_ROW_GROUP_ROWS")
                .unwrap_or(d.row_group_rows),
            data_page_rows: env_usize("PROXIMADB_PARQUET_WRITE_PAGE_ROWS")
                .unwrap_or(d.data_page_rows),
            page_statistics: d.page_statistics
                && !env_disabled("PROXIMADB_PARQUET_WRITE_PAGE_INDEX_DISABLE"),
            declare_sorting: d.declare_sorting
                && !env_disabled("PROXIMADB_PARQUET_WRITE_SORTING_DISABLE"),
            bloom: BloomPolicy {
                enabled: d.bloom.enabled && !env_disabled("PROXIMADB_PARQUET_WRITE_BLOOM_DISABLE"),
                fpp: std::env::var("PROXIMADB_PARQUET_WRITE_BLOOM_FPP")
                    .ok()
                    .and_then(|v| v.parse::<f64>().ok())
                    .filter(|p| *p > 0.0 && *p < 1.0)
                    .unwrap_or(d.bloom.fpp),
                columns: std::env::var("PROXIMADB_PARQUET_WRITE_BLOOM_COLUMNS")
                    .ok()
                    .map(|v| {
                        v.split(',')
                            .map(|c| c.trim().to_string())
                            .filter(|c| !c.is_empty())
                            .collect()
                    }),
                min_rows: env_usize("PROXIMADB_PARQUET_WRITE_BLOOM_MIN_ROWS")
                    .unwrap_or(d.bloom.min_rows),
            },
        }
    }

    /// The legacy pre-policy layout: row-group cap only, everything else the Arrow
    /// default. Kept so the perf ledger can measure the policy against it.
    pub fn legacy(row_group_rows: usize) -> Self {
        Self {
            row_group_rows,
            data_page_rows: 0,
            page_statistics: false,
            declare_sorting: false,
            bloom: BloomPolicy {
                enabled: false,
                ..Self::default().bloom
            },
            ..Self::default()
        }
    }

    /// Columns that receive a Bloom filter, each with its per-row-group NDV
    /// estimate (the SBBF is sized from it). Deterministic order (schema order).
    pub fn bloom_columns(
        &self,
        batch: &RecordBatch,
        hints: &ParquetLayoutHints,
    ) -> Vec<(String, u64)> {
        if !self.bloom.enabled || batch.num_rows() < self.bloom.min_rows {
            return Vec::new();
        }
        let leading_sort = hints.sort_columns.first().map(String::as_str);
        let schema = batch.schema();
        let mut out = Vec::new();
        for (i, field) in schema.fields().iter().enumerate() {
            let name = field.name().as_str();
            let selected = match &self.bloom.columns {
                Some(list) => list.iter().any(|c| c == name),
                None => bloom_eligible(field.data_type()) && Some(name) != leading_sort,
            };
            if !selected {
                continue;
            }
            let ndv = estimate_row_group_ndv(batch.column(i).as_ref(), self.row_group_rows);
            if ndv > 0 {
                out.push((name.to_string(), ndv));
            }
        }
        out
    }

    /// Build the [`WriterProperties`] for one data file holding `batch`.
    pub fn writer_properties(
        &self,
        batch: &RecordBatch,
        hints: &ParquetLayoutHints,
    ) -> WriterProperties {
        let mut builder =
            WriterProperties::builder().set_max_row_group_row_count(Some(self.row_group_rows));
        if self.data_page_rows > 0 {
            // The row limit is only checked between write mini-batches, so a
            // mini-batch that divides the page size keeps page boundaries exact.
            let mini_batch = if self.data_page_rows.is_multiple_of(1024) {
                1024
            } else {
                self.data_page_rows.min(1024)
            };
            builder = builder
                .set_data_page_row_count_limit(self.data_page_rows)
                .set_write_batch_size(mini_batch);
        }
        if self.page_statistics {
            builder = builder.set_statistics_enabled(EnabledStatistics::Page);
        }
        if self.declare_sorting
            && let Some(sorting) = sorting_columns(&batch.schema(), &hints.sort_columns)
        {
            builder = builder.set_sorting_columns(Some(sorting));
        }
        for (column, ndv) in self.bloom_columns(batch, hints) {
            let path = ColumnPath::from(column);
            builder = builder
                .set_column_bloom_filter_enabled(path.clone(), true)
                .set_column_bloom_filter_fpp(path.clone(), self.bloom.fpp)
                .set_column_bloom_filter_max_ndv(path, ndv);
        }
        builder.build()
    }
}

/// Types whose equality probes a Bloom filter can answer and min/max usually
/// cannot: integers, dates, and strings/binary. Floats (equality is rare and
/// NaN-ambiguous), booleans and other two-valued domains (min/max is exact), and
/// nested types are excluded.
pub fn bloom_eligible(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Date32
            | DataType::Date64
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::BinaryView
            | DataType::FixedSizeBinary(_)
    )
}

/// `sorting_columns` for the leaf columns named in `sort_columns` (ascending,
/// nulls last — the order MATERIALIZE's cluster sort produces). Stops at the
/// first name not found so a declared prefix is never wrong.
fn sorting_columns(schema: &Schema, sort_columns: &[String]) -> Option<Vec<SortingColumn>> {
    let mut out = Vec::new();
    for name in sort_columns {
        let Ok(idx) = schema.index_of(name) else {
            break;
        };
        out.push(SortingColumn {
            column_idx: idx as i32,
            descending: false,
            nulls_first: false,
        });
    }
    (!out.is_empty()).then_some(out)
}

/// Distinct non-null values per row group, estimated as the maximum exact
/// distinct count over three contiguous row-group-sized windows (head, middle,
/// tail). Row groups are contiguous row ranges, so a contiguous window is an
/// unbiased stand-in for a row group — including on sorted data, where a strided
/// sample would overstate per-row-group NDV. Exact within each window.
pub fn estimate_row_group_ndv(array: &dyn Array, row_group_rows: usize) -> u64 {
    let len = array.len();
    if len == 0 {
        return 0;
    }
    let window = row_group_rows.min(len);
    let starts = [0, (len - window) / 2, len - window];
    let mut best = 0u64;
    for start in starts {
        let slice = array.slice(start, window);
        best = best.max(exact_distinct(slice.as_ref()));
    }
    best
}

fn exact_distinct(array: &dyn Array) -> u64 {
    use arrow_array::cast::AsArray;
    use arrow_array::types::*;
    macro_rules! prim {
        ($t:ty) => {{
            let a = array.as_primitive::<$t>();
            let set: HashSet<_> = a.iter().flatten().map(|v| v.to_le_bytes()).collect();
            set.len() as u64
        }};
    }
    match array.data_type() {
        DataType::Int8 => prim!(Int8Type),
        DataType::Int16 => prim!(Int16Type),
        DataType::Int32 => prim!(Int32Type),
        DataType::Int64 => prim!(Int64Type),
        DataType::UInt8 => prim!(UInt8Type),
        DataType::UInt16 => prim!(UInt16Type),
        DataType::UInt32 => prim!(UInt32Type),
        DataType::UInt64 => prim!(UInt64Type),
        DataType::Date32 => prim!(Date32Type),
        DataType::Date64 => prim!(Date64Type),
        DataType::Utf8 => array
            .as_string::<i32>()
            .iter()
            .flatten()
            .collect::<HashSet<_>>()
            .len() as u64,
        DataType::LargeUtf8 => array
            .as_string::<i64>()
            .iter()
            .flatten()
            .collect::<HashSet<_>>()
            .len() as u64,
        DataType::Utf8View => array
            .as_string_view()
            .iter()
            .flatten()
            .collect::<HashSet<_>>()
            .len() as u64,
        DataType::Binary => array
            .as_binary::<i32>()
            .iter()
            .flatten()
            .collect::<HashSet<_>>()
            .len() as u64,
        DataType::LargeBinary => array
            .as_binary::<i64>()
            .iter()
            .flatten()
            .collect::<HashSet<_>>()
            .len() as u64,
        DataType::BinaryView => array
            .as_binary_view()
            .iter()
            .flatten()
            .collect::<HashSet<_>>()
            .len() as u64,
        DataType::FixedSizeBinary(_) => array
            .as_fixed_size_binary()
            .iter()
            .flatten()
            .collect::<HashSet<_>>()
            .len() as u64,
        // Unsupported type: size for a full window (never undersized).
        _ => (array.len() - array.null_count()) as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proxima_parquet::record_batches_to_parquet_bytes;
    use arrow_array::{Float64Array, Int64Array, StringArray};
    use arrow_schema::Field;
    use bytes::Bytes;
    use parquet::file::properties::ReaderProperties;
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use parquet::file::serialized_reader::ReadOptionsBuilder;
    use std::sync::Arc;

    fn fact_batch(rows: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("date_sk", DataType::Int64, false),
            Field::new("item_sk", DataType::Int64, false),
            Field::new("state", DataType::Utf8, true),
            Field::new("price", DataType::Float64, true),
        ]));
        let date: Vec<i64> = (0..rows as i64).map(|i| i / 100).collect();
        let item: Vec<i64> = (0..rows as i64).map(|i| (i * 7919) % 5_000).collect();
        let states = ["CA", "TX", "NY", "WA"];
        let state: Vec<Option<&str>> = (0..rows).map(|i| Some(states[i % 4])).collect();
        let price: Vec<f64> = (0..rows).map(|i| i as f64 * 0.5).collect();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(date)),
                Arc::new(Int64Array::from(item)),
                Arc::new(StringArray::from(state)),
                Arc::new(Float64Array::from(price)),
            ],
        )
        .unwrap()
    }

    fn hints() -> ParquetLayoutHints {
        ParquetLayoutHints {
            sort_columns: vec!["date_sk".to_string()],
        }
    }

    #[test]
    fn auto_selection_skips_leading_sort_column_and_floats() {
        let policy = ParquetWritePolicy::default();
        let cols: Vec<String> = policy
            .bloom_columns(&fact_batch(20_000), &hints())
            .into_iter()
            .map(|(c, _)| c)
            .collect();
        assert_eq!(cols, vec!["item_sk".to_string(), "state".to_string()]);
    }

    #[test]
    fn ndv_estimate_is_per_row_group_on_sorted_data() {
        // 20k rows, date_sk = i/100 → 200 distinct overall, but a 1,000-row
        // row group holds only 10 distinct dates.
        let batch = fact_batch(20_000);
        assert_eq!(estimate_row_group_ndv(batch.column(0).as_ref(), 1_000), 10);
        assert_eq!(estimate_row_group_ndv(batch.column(2).as_ref(), 1_000), 4);
    }

    #[test]
    fn small_files_and_disabled_policy_get_no_bloom() {
        let policy = ParquetWritePolicy::default();
        assert!(policy.bloom_columns(&fact_batch(100), &hints()).is_empty());
        let legacy = ParquetWritePolicy::legacy(65_536);
        assert!(
            legacy
                .bloom_columns(&fact_batch(20_000), &hints())
                .is_empty()
        );
    }

    #[test]
    fn explicit_column_list_overrides_auto_selection() {
        let mut policy = ParquetWritePolicy::default();
        policy.bloom.columns = Some(vec!["date_sk".to_string()]);
        let cols: Vec<String> = policy
            .bloom_columns(&fact_batch(20_000), &hints())
            .into_iter()
            .map(|(c, _)| c)
            .collect();
        assert_eq!(cols, vec!["date_sk".to_string()]);
    }

    /// The written file carries every spec structure the policy promises:
    /// multiple row groups, a page index (column + offset index) with several
    /// pages per row group, declared sorting columns, and SBBF Bloom filters
    /// that answer membership with no false negatives.
    #[test]
    fn written_file_carries_page_index_sorting_and_sbbf() {
        let policy = ParquetWritePolicy {
            row_group_rows: 10_000,
            data_page_rows: 1_000,
            ..ParquetWritePolicy::default()
        };
        let batch = fact_batch(30_000);
        let props = policy.writer_properties(&batch, &hints());
        let bytes =
            record_batches_to_parquet_bytes(&[batch.clone()], batch.schema(), Some(props)).unwrap();

        let opts = ReadOptionsBuilder::new()
            .with_page_index()
            .with_reader_properties(
                ReaderProperties::builder()
                    .set_read_bloom_filter(true)
                    .build(),
            )
            .build();
        let reader = SerializedFileReader::new_with_options(Bytes::from(bytes), opts).unwrap();
        let meta = reader.metadata();
        assert_eq!(meta.num_row_groups(), 3);

        let rg0 = meta.row_group(0);
        let sorting = rg0.sorting_columns().expect("sorting_columns declared");
        assert_eq!(sorting[0].column_idx, 0);
        assert!(!sorting[0].descending);

        let offset_index = meta.offset_index().expect("offset index written");
        assert!(
            offset_index[0][0].page_locations().len() >= 10,
            "≈1k-row pages"
        );
        assert!(meta.column_index().is_some(), "column index written");

        let rg_reader = reader.get_row_group(0).unwrap();
        assert!(
            rg_reader.get_column_bloom_filter(0).is_none(),
            "sort key: no bloom"
        );
        let item_bloom = rg_reader.get_column_bloom_filter(1).expect("item_sk bloom");
        let state_bloom = rg_reader.get_column_bloom_filter(2).expect("state bloom");
        assert!(
            rg_reader.get_column_bloom_filter(3).is_none(),
            "float: no bloom"
        );
        // No false negatives for values present in row group 0 …
        for i in 0..10_000i64 {
            assert!(item_bloom.check(&((i * 7919) % 5_000)));
        }
        assert!(state_bloom.check(&"TX"));
        // … and an absent value is (with fpp 1%) rejected.
        assert!(!state_bloom.check(&"ZZ-absent-state"));
    }

    #[test]
    fn legacy_policy_writes_no_page_index_or_bloom() {
        let policy = ParquetWritePolicy::legacy(10_000);
        let batch = fact_batch(30_000);
        let props = policy.writer_properties(&batch, &hints());
        let bytes =
            record_batches_to_parquet_bytes(&[batch.clone()], batch.schema(), Some(props)).unwrap();
        let opts = ReadOptionsBuilder::new()
            .with_reader_properties(
                ReaderProperties::builder()
                    .set_read_bloom_filter(true)
                    .build(),
            )
            .build();
        let reader = SerializedFileReader::new_with_options(Bytes::from(bytes), opts).unwrap();
        assert!(reader.metadata().row_group(0).sorting_columns().is_none());
        assert!(
            reader
                .get_row_group(0)
                .unwrap()
                .get_column_bloom_filter(1)
                .is_none()
        );
    }
}
