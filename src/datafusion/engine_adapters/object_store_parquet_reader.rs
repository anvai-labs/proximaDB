//! Parquet-over-object_store TableProvider for the warehouse base tier.
//!
//! This is the DataFusion read leaf for Parquet/Iceberg publications. It uses
//! `IcebergObjectStoreBridge` for base-prefix discovery and Parquet's
//! `ParquetObjectReader` for range-based footer and row-group reads.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use arrow_schema::{Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::arrow::array::{
    Array, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
    LargeStringArray, StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::record_batch::{RecordBatch, RecordBatchOptions};
use datafusion::catalog::Session;
use datafusion::common::Statistics;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::logical_expr::{Expr, Operator, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::prelude::SessionContext;
use futures::StreamExt;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::path::Path;
use parquet::arrow::ProjectionMask;
// arrow-rs 59.2 deprecated ParquetObjectReader (wants direct AsyncFileReader impls,
// apache/arrow-rs#10308). Contained here; the migration is a tracked follow-up.
#[allow(deprecated)]
use parquet::arrow::async_reader::{ParquetObjectReader, ParquetRecordBatchStreamBuilder};
use parquet::bloom_filter::Sbbf;
use parquet::file::metadata::ParquetMetaData;
use parquet::file::statistics::Statistics as ParquetStatistics;
use proximadb_bloom::{BloomFilterBuilder, BloomFilterStrategy, CoreBloomFilterConfig};
use proximadb_iceberg_engine::IcebergObjectStoreBridge;
use proximadb_storage_common::format_splits::{
    ColumnBounds, FileSplit, ScalarPredicate, ScalarValue, SplitStatistics, SplitType,
};
use proximadb_storage_common::object_store_bridge::ObjectStoreBridge;
use url::Url;

use super::super::physical_filter_translate::df_scalar_value;
use super::super::proxima_scan_exec::{ProximaScanExec, SplitReader};
use super::super::proxima_table_provider::EngineType;
use super::super::schema_inference::{
    df_stats_value_bounds_enabled, project_statistics, statistics_from_splits,
};
use crate::observability::object_store_trace::TracingObjectStore;

/// True iff `proj` selects every column in ascending order — a no-op projection
/// where pushing a mask (and reordering) buys nothing, so the reader reads all.
fn is_identity(proj: &[usize]) -> bool {
    proj.iter().enumerate().all(|(i, &c)| i == c)
}

fn project_schema(full: &SchemaRef, projection: Option<&[usize]>) -> SchemaRef {
    match projection {
        Some(proj) => {
            let fields: Vec<_> = proj
                .iter()
                .filter_map(|&i| full.fields().get(i))
                .map(|f| f.as_ref().clone())
                .collect();
            Arc::new(Schema::new(fields))
        }
        None => full.clone(),
    }
}

/// Footer prefetch size (bytes). Without a hint the parquet reader pulls a
/// whole-file suffix to find the footer on a small object — measured ~116 MB
/// "open" I/O per query at 1M ClickBench (TD-OLAP-4). 512 KiB covers a
/// wide-table (105-column) footer in one ranged read; a larger real footer just
/// costs one extra ranged read (rare). Overridable via
/// `PROXIMADB_PARQUET_FOOTER_HINT_KB` for tuning at very wide/large scales.
fn footer_size_hint() -> usize {
    std::env::var("PROXIMADB_PARQUET_FOOTER_HINT_KB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&kb| kb > 0)
        .map(|kb| kb * 1024)
        .unwrap_or(512 * 1024)
}

// arrow-rs 59.2 deprecated ParquetObjectReader (wants direct AsyncFileReader impls,
// apache/arrow-rs#10308). Contained here; the migration is a tracked follow-up.
#[allow(deprecated)]
fn parquet_reader(
    store: Arc<dyn ObjectStore>,
    path: Path,
    file_size: Option<u64>,
) -> ParquetObjectReader {
    let reader = ParquetObjectReader::new(store, path).with_footer_size_hint(footer_size_hint());
    match file_size {
        Some(size) => reader.with_file_size(size),
        None => reader,
    }
}

fn df_err(context: &str, err: impl std::fmt::Display) -> DataFusionError {
    DataFusionError::Execution(format!("{context}: {err}"))
}

/// Build the per-query traced object store for `location` WITHOUT any discovery
/// I/O (no LIST/HEAD/footer). Used on a table-OPEN cache hit: the store must be
/// re-created per query because [`TracingObjectStore::wrap`] captures the
/// io-trace handle at construction, so a shared/cached store would misattribute
/// later queries' `bytes_read`. Mirrors the store creation inside
/// `open_direct_object` / `open_bridge_base`.
fn build_traced_store(location: &str) -> DFResult<Arc<dyn ObjectStore>> {
    if location.ends_with(".parquet") {
        let parsed = Url::parse(location).map_err(|e| df_err("parse object-store url", e))?;
        let (store, _path) =
            object_store::parse_url(&parsed).map_err(|e| df_err("object_store parse_url", e))?;
        Ok(TracingObjectStore::wrap(Arc::from(store)))
    } else {
        let bridge = IcebergObjectStoreBridge::from_url(location)
            .map_err(|e| df_err(&format!("object-store bridge({location})"), e))?;
        Ok(TracingObjectStore::wrap(bridge.inner_store()))
    }
}

/// Runtime-filter (TD-OLAP-3) bloom semi-join gate. Default **ON** (ADR-056
/// MVP-1, 2026-07-08): measured 30–57% `bytes_read` reduction on TPC-DS star
/// joins where the fact table has no direct predicate and is reached only via
/// the dimension (evidence v5/v6 in the TD-OLAP-3 ledger). Correctness-neutral:
/// the bloom has no false negatives (a matching split key always reads present;
/// the `FilterExec` above re-checks exactly), so promotion is gated only on the
/// qa-gate `bytes_read` ratchet — a regression fails CI. Opt out with
/// `PROXIMADB_DF_BLOOM_SEMIJOIN=0`. `min_keys` is the in-list cardinality above
/// which min/max zone-maps collapse (a fact probe against thousands of
/// dimension keys) and reading a split's key column to bloom-test it pays for
/// itself (`PROXIMADB_DF_BLOOM_SEMIJOIN_MIN_KEYS`, default 256).
#[derive(Clone, Copy, Debug)]
struct SemijoinConfig {
    enabled: bool,
    min_keys: usize,
}

impl SemijoinConfig {
    const DEFAULT_MIN_KEYS: usize = 256;

    fn from_env() -> Self {
        let enabled = std::env::var("PROXIMADB_DF_BLOOM_SEMIJOIN")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true);
        let min_keys = std::env::var("PROXIMADB_DF_BLOOM_SEMIJOIN_MIN_KEYS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(Self::DEFAULT_MIN_KEYS);
        Self { enabled, min_keys }
    }
}

/// Canonical key family a split's keys were encoded with. Stored so the in-list
/// bloom encodes its values into the *same* byte layout — a cross-family
/// encoding mismatch must never silently drop a matching key (a false-negative
/// prune = wrong results), so on mismatch [`bloom_prunes_split`] declines to
/// prune rather than guess.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KeyFamily {
    Int,
    Float,
    Str,
    Bool,
}

/// The distinct join-key values of one split (row group), encoded into the
/// canonical byte layout of the column's [`KeyFamily`]. Built lazily from a
/// projected read of just that column and cached per (file, row-group, column).
///
/// Semi-join direction (load-bearing): the bloom is built over the *in-list*
/// (dimension) keys and each *split* key is tested against it — not the reverse.
/// Testing thousands of in-list keys against a per-split bloom collapses to
/// "never prune" (≈`fpr` false positives per split keep it, and the target case
/// is thousands of keys); testing a split's bounded key set against a low-`fpr`
/// in-list bloom prunes reliably (`P(prune | disjoint) = (1-fpr)^K`). Both
/// directions are false-negative-free; only this one actually prunes.
#[derive(Clone, Debug)]
struct SplitKeys {
    family: KeyFamily,
    keys: Vec<Vec<u8>>,
}

/// A bloom over the probe-side in-list (dimension) keys for one column, plus the
/// key family its values were encoded with. Built once per query.
struct InlistBloom {
    family: KeyFamily,
    filter: Box<dyn BloomFilterStrategy>,
}

/// Native Parquet row-group bloom filter for one join-key column.
#[derive(Debug)]
struct RowGroupBloom {
    family: KeyFamily,
    filter: Sbbf,
}

/// Cap on distinct keys cached per split. Above it a split is not bloom-pruned
/// (cached as `None`): the memory no longer pays off and a very large key set
/// prunes with low probability anyway. min/max zone-maps still apply.
const MAX_SPLIT_DISTINCT_KEYS: usize = 1_000_000;

/// Bits per key for the in-list bloom → ≈6.6e-5 false-positive rate
/// (`0.6185^20`), low enough that testing a split's key set against it prunes
/// reliably rather than being defeated by a single false positive.
const INLIST_BLOOM_BITS_PER_KEY: u32 = 20;

/// Lazily-built per-split key sets, keyed by (file_path, row_group_index,
/// column). `None` records an attempted-but-unavailable build (error, nested
/// type, or over the distinct-key cap) so pruning falls back to min/max without
/// retrying.
type SplitKeyCache = HashMap<(String, usize, String), Option<Arc<SplitKeys>>>;

/// Lazily-read native Parquet row-group bloom filters keyed by
/// (file_path, row_group_index, column). `None` records an absent/unreadable
/// bloom filter and allows the older key-column fallback.
type RowGroupBloomCache = HashMap<(String, usize, String), Option<Arc<RowGroupBloom>>>;

/// Read-only view handed to the (synchronous) split-pruning pass once the async
/// prefetch has populated the split-key cache: the per-column in-list blooms
/// built from this query's predicates, plus the cached split key sets.
struct BloomPruneView<'a> {
    min_keys: usize,
    inlist_blooms: &'a HashMap<String, InlistBloom>,
    row_group_blooms: &'a RowGroupBloomCache,
    split_keys: &'a SplitKeyCache,
}

/// Map an Arrow key-column type to the canonical bloom key family, or `None`
/// for unsupported/nested types (the bloom is then skipped, min/max is used).
fn column_key_family(dt: &DataType) -> Option<KeyFamily> {
    match dt {
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => Some(KeyFamily::Int),
        DataType::Float32 | DataType::Float64 => Some(KeyFamily::Float),
        DataType::Utf8 | DataType::LargeUtf8 => Some(KeyFamily::Str),
        DataType::Boolean => Some(KeyFamily::Bool),
        _ => None,
    }
}

/// Canonical key bytes for a build-side (split column) value at `row`, or
/// `None` for null / unsupported types. Integers widen to `i64` LE and floats
/// to `f64` LE so the encoding is stable across Arrow's fixed-width variants
/// and matches [`probe_key_bytes`].
fn array_key_bytes(array: &dyn Array, row: usize) -> Option<Vec<u8>> {
    if array.is_null(row) {
        return None;
    }
    let any = array.as_any();
    let i64_le = |v: i64| v.to_le_bytes().to_vec();
    let f64_le = |v: f64| v.to_le_bytes().to_vec();
    match array.data_type() {
        DataType::Int8 => any
            .downcast_ref::<Int8Array>()
            .map(|a| i64_le(a.value(row) as i64)),
        DataType::Int16 => any
            .downcast_ref::<Int16Array>()
            .map(|a| i64_le(a.value(row) as i64)),
        DataType::Int32 => any
            .downcast_ref::<Int32Array>()
            .map(|a| i64_le(a.value(row) as i64)),
        DataType::Int64 => any
            .downcast_ref::<Int64Array>()
            .map(|a| i64_le(a.value(row))),
        DataType::UInt8 => any
            .downcast_ref::<UInt8Array>()
            .map(|a| i64_le(a.value(row) as i64)),
        DataType::UInt16 => any
            .downcast_ref::<UInt16Array>()
            .map(|a| i64_le(a.value(row) as i64)),
        DataType::UInt32 => any
            .downcast_ref::<UInt32Array>()
            .map(|a| i64_le(a.value(row) as i64)),
        DataType::UInt64 => any
            .downcast_ref::<UInt64Array>()
            .map(|a| i64_le(a.value(row) as i64)),
        DataType::Float32 => any
            .downcast_ref::<Float32Array>()
            .map(|a| f64_le(a.value(row) as f64)),
        DataType::Float64 => any
            .downcast_ref::<Float64Array>()
            .map(|a| f64_le(a.value(row))),
        DataType::Utf8 => any
            .downcast_ref::<StringArray>()
            .map(|a| a.value(row).as_bytes().to_vec()),
        DataType::LargeUtf8 => any
            .downcast_ref::<LargeStringArray>()
            .map(|a| a.value(row).as_bytes().to_vec()),
        DataType::Boolean => any
            .downcast_ref::<BooleanArray>()
            .map(|a| vec![a.value(row) as u8]),
        _ => None,
    }
}

/// Canonical key bytes for a probe-side (in-list) value, encoded into the split
/// bloom's key family. Returns `None` when the value cannot be encoded into
/// that family — the caller then treats the value as *possibly present* (no
/// prune), never a false negative.
fn probe_key_bytes(family: KeyFamily, value: &ScalarValue) -> Option<Vec<u8>> {
    match (family, value) {
        (KeyFamily::Int, ScalarValue::Int64(i)) => Some(i.to_le_bytes().to_vec()),
        (KeyFamily::Float, ScalarValue::Float64(f)) => Some(f.to_le_bytes().to_vec()),
        // Int literal against a float key: SQL compares 5 == 5.0, so widen to
        // the column's f64 layout rather than miss the match.
        (KeyFamily::Float, ScalarValue::Int64(i)) => Some((*i as f64).to_le_bytes().to_vec()),
        (KeyFamily::Str, ScalarValue::String(s)) => Some(s.as_bytes().to_vec()),
        (KeyFamily::Bool, ScalarValue::Bool(b)) => Some(vec![*b as u8]),
        _ => None,
    }
}

/// Build a bloom over the probe-side in-list `values` for a column of the given
/// key `family`, sized for a low false-positive rate so split-key testing prunes
/// reliably. Values that cannot be encoded into the family are skipped (they
/// cannot match a key of that family anyway).
fn build_inlist_bloom(family: KeyFamily, values: &[ScalarValue]) -> InlistBloom {
    let config = CoreBloomFilterConfig {
        bits_per_key: INLIST_BLOOM_BITS_PER_KEY,
        expected_items: values.len().max(1),
        ..Default::default()
    };
    let mut builder = BloomFilterBuilder::new(config);
    for value in values {
        if let Some(bytes) = probe_key_bytes(family, value) {
            builder.add(&bytes);
        }
    }
    InlistBloom {
        family,
        filter: builder.build(),
    }
}

/// Prune a split iff *none* of its distinct keys can be in the probe-side
/// in-list — i.e. every split key misses the in-list bloom. The bloom has no
/// false negatives, so a split key that really is in the in-list always reports
/// `might_contain == true` and the split is kept (never a false-negative prune);
/// a false positive merely keeps a split the hash join then filters exactly.
fn bloom_prunes_split(inlist: &InlistBloom, split_keys: &SplitKeys) -> bool {
    // Families must match (both derive from the same column type); if not, stay
    // conservative and do not prune.
    if inlist.family != split_keys.family || split_keys.keys.is_empty() {
        return false;
    }
    split_keys
        .keys
        .iter()
        .all(|k| !inlist.filter.might_contain(k))
}

/// Prune using a native Parquet row-group bloom iff every probe-side in-list
/// key is definitely absent from the split's persisted key bloom. This is
/// false-negative-free: an unsupported value or a bloom hit keeps the split.
fn parquet_bloom_prunes_split(row_group_bloom: &RowGroupBloom, values: &[ScalarValue]) -> bool {
    let mut checked_any = false;
    for value in values {
        let Some(bytes) = probe_key_bytes(row_group_bloom.family, value) else {
            return false;
        };
        checked_any = true;
        if row_group_bloom.filter.check(&bytes) {
            return false;
        }
    }
    checked_any
}

fn row_group_index(split: &FileSplit) -> Option<usize> {
    match &split.split_type {
        SplitType::RowGroup {
            row_group_index, ..
        } => Some(*row_group_index),
        _ => None,
    }
}

/// Collect positive in-lists whose cardinality is at/above the semi-join
/// threshold, as (column, literal values) — the candidates for a per-split
/// bloom prefilter.
fn collect_semijoin_inlists(
    expr: &Expr,
    min_keys: usize,
    out: &mut Vec<(String, Vec<ScalarValue>)>,
) {
    match expr {
        Expr::BinaryExpr(binary) if matches!(binary.op, Operator::And | Operator::Or) => {
            collect_semijoin_inlists(&binary.left, min_keys, out);
            collect_semijoin_inlists(&binary.right, min_keys, out);
        }
        Expr::InList(list) if !list.negated && list.list.len() >= min_keys => {
            if let Some(column) = column_name(&list.expr) {
                let values = list
                    .list
                    .iter()
                    .filter_map(literal_value)
                    .collect::<Vec<_>>();
                if !values.is_empty() {
                    out.push((column, values));
                }
            }
        }
        _ => {}
    }
}

/// Reads a selected Parquet row group through object_store range requests.
#[derive(Debug)]
pub struct ObjectStoreParquetSplitReader {
    schema: SchemaRef,
    store: Arc<dyn ObjectStore>,
    file_sizes: HashMap<String, u64>,
}

#[async_trait]
impl SplitReader for ObjectStoreParquetSplitReader {
    async fn read_split(
        &self,
        split: &FileSplit,
        projection: Option<&[usize]>,
        batch_size: usize,
    ) -> DFResult<SendableRecordBatchStream> {
        let row_group_index = match &split.split_type {
            SplitType::RowGroup {
                row_group_index, ..
            } => *row_group_index,
            other => {
                return Err(df_err(
                    "ObjectStoreParquetSplitReader expects RowGroup splits",
                    format!("{other:?}"),
                ));
            }
        };
        let path = Path::from(split.file_path.as_str());
        let file_size = self.file_sizes.get(&split.file_path).copied();
        let reader = parquet_reader(self.store.clone(), path, file_size);
        let builder = ParquetRecordBatchStreamBuilder::new(reader)
            .await
            .map_err(|e| df_err("parquet open", e))?;

        let out_schema = project_schema(&self.schema, projection);

        // TRUE projection PUSHDOWN: fetch only the projected column chunks, not
        // all columns (TD-OLAP-4 co-design — a query touching 3 of 105 columns
        // was reading every column chunk, ~97% wasted bytes). The reader emits
        // the selected columns in ascending FILE order; `reorder` maps them back
        // to the requested projection order so the batch matches `out_schema`.
        // For a flat schema, root columns == leaf columns, so `roots` is exact.
        let (builder, reorder) = match projection {
            Some(proj) if proj.len() != self.schema.fields().len() || !is_identity(proj) => {
                let mask = ProjectionMask::roots(builder.parquet_schema(), proj.iter().copied());
                let mut file_order: Vec<usize> = proj.to_vec();
                file_order.sort_unstable();
                file_order.dedup();
                // Each requested column is in `file_order` by construction (it is
                // built from `proj`); map fallibly rather than panic (panic policy).
                let reorder: Vec<usize> = proj
                    .iter()
                    .map(|c| {
                        file_order.iter().position(|x| x == c).ok_or_else(|| {
                            df_err("projection", "requested column missing from selected set")
                        })
                    })
                    .collect::<DFResult<Vec<_>>>()?;
                (builder.with_projection(mask), Some(reorder))
            }
            // Full/absent projection: read every column, no reorder.
            _ => (builder, None),
        };

        let mut stream = builder
            .with_row_groups(vec![row_group_index])
            .with_batch_size(batch_size)
            .build()
            .map_err(|e| df_err("parquet reader build", e))?;

        let mut batches = Vec::new();
        while let Some(batch) = stream.next().await {
            let batch = batch.map_err(|e| df_err("parquet decode", e))?;
            let batch = match &reorder {
                Some(order) => {
                    let cols: Vec<_> = order.iter().map(|&i| batch.column(i).clone()).collect();
                    // `with_row_count` preserves the count for an EMPTY projection
                    // (e.g. `COUNT(*)`, which reads zero columns) — a 0-column
                    // batch is otherwise rejected without an explicit row count.
                    let opts = RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
                    RecordBatch::try_new_with_options(out_schema.clone(), cols, &opts)
                        .map_err(|e| df_err("projection reorder", e))?
                }
                None => batch,
            };
            batches.push(Ok(batch));
        }
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            out_schema,
            futures::stream::iter(batches),
        )))
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn engine_type(&self) -> EngineType {
        EngineType::Viper
    }

    fn supports_projection_pushdown(&self) -> bool {
        true
    }
}

/// DataFusion table backed by one or more Parquet objects in object_store.
#[derive(Debug)]
pub struct ObjectStoreParquetTable {
    schema: SchemaRef,
    store: Arc<dyn ObjectStore>,
    splits: Vec<FileSplit>,
    file_sizes: HashMap<String, u64>,
    location: String,
    /// Registered table/collection name — keys the ADR-037 statistics
    /// registry for the NDV overlay in [`TableProvider::statistics`].
    table_name: Option<String>,
    /// TD-OLAP-3: lazily-built, cached per-split join-key sets for the high-NDV
    /// in-list semi-join case (see `prepare_split_keys`).
    split_key_cache: Arc<Mutex<SplitKeyCache>>,
    /// Native Parquet row-group bloom filters, preferred over query-time
    /// key-column materialization because they let pruning happen from metadata
    /// without reading the data column first.
    row_group_bloom_cache: Arc<Mutex<RowGroupBloomCache>>,
    /// ADR-058 D5/§9.A: trust level for this table's parquet footer stats.
    /// `Trusted` (default — ProximaDB wrote the footer) ⇒ `statistics_from_splits`
    /// may surface footer row_count/null_count/min-max as `Precision::Exact` for
    /// DataFusion's `AggregateStatistics` elision. `Untrusted` (external/federated)
    /// ⇒ stats are surfaced as `Precision::Absent` so COUNT/MIN/MAX scan.
    stats_trust: proximadb_data_model::StatsTrust,
}

impl ObjectStoreParquetTable {
    /// Open either a ProximaDB base-prefix location (`data/*.parquet`) or a
    /// direct external `.parquet` object.
    pub async fn open(location: &str) -> DFResult<Self> {
        if location.ends_with(".parquet") {
            return Self::open_direct_object(location).await;
        }
        Self::open_bridge_base(location).await
    }

    async fn open_bridge_base(location: &str) -> DFResult<Self> {
        let bridge = Arc::new(
            IcebergObjectStoreBridge::from_url(location)
                .map_err(|e| df_err(&format!("object-store bridge({location})"), e))?,
        );
        // ADR-059 first-principles: scan the location's IMMEDIATE parquet files
        // (Hive/Unity/Polaris model — location = the directory where parquet is
        // immediate, skipping commit-style `_`/`.` files via the `.parquet`
        // extension filter). Mixed-read-safe fallback: if no immediate parquet,
        // try the legacy `data/` subpath (old materialized tables where location
        // = table base dir, parquet in `{location}/data/`).
        let mut parquet_paths = bridge
            .list_objects(&Path::from(""))
            .await
            .unwrap_or_default()
            .into_iter()
            .filter(|path| path.as_ref().ends_with(".parquet"))
            .collect::<Vec<_>>();
        if parquet_paths.is_empty() {
            parquet_paths = bridge
                .list_objects(&Path::from("data"))
                .await
                .map_err(|e| df_err(&format!("list parquet data/ fallback {location}"), e))?
                .into_iter()
                .filter(|path| path.as_ref().ends_with(".parquet"))
                .collect::<Vec<_>>();
        }
        parquet_paths.sort_by(|a, b| a.as_ref().cmp(b.as_ref()));
        if parquet_paths.is_empty() {
            return Err(df_err(
                "open object-store parquet base",
                format!("no parquet objects under {location} (immediate or data/)"),
            ));
        }

        // Wrap the store so footer + row-group reads land in the per-query
        // I/O trace (ADR-030/TD-158) — the DataFusion route's bytes-scanned
        // was previously invisible to the co-design cost model.
        let store = TracingObjectStore::wrap(bridge.inner_store());
        let mut files = Vec::with_capacity(parquet_paths.len());
        for relative in parquet_paths {
            let full = bridge.full_object_path(&relative);
            let size = store
                .head(&full)
                .await
                .map_err(|e| df_err(&format!("head {full}"), e))?
                .size;
            files.push((full, Some(size)));
        }
        Self::open_files(store, files, location.to_string()).await
    }

    async fn open_direct_object(location: &str) -> DFResult<Self> {
        let parsed = Url::parse(location).map_err(|e| df_err("parse object-store url", e))?;
        let (store, path) =
            object_store::parse_url(&parsed).map_err(|e| df_err("object_store parse_url", e))?;
        let store: Arc<dyn ObjectStore> = TracingObjectStore::wrap(Arc::from(store));
        let size = store
            .head(&path)
            .await
            .map_err(|e| df_err(&format!("head {path}"), e))?
            .size;
        Self::open_files(store, vec![(path, Some(size))], location.to_string()).await
    }

    async fn open_files(
        store: Arc<dyn ObjectStore>,
        files: Vec<(Path, Option<u64>)>,
        location: String,
    ) -> DFResult<Self> {
        let mut schema = None;
        let mut splits = Vec::new();
        let mut file_sizes = HashMap::new();
        for (path, size) in files {
            let reader = parquet_reader(store.clone(), path.clone(), size);
            let builder = ParquetRecordBatchStreamBuilder::new(reader)
                .await
                .map_err(|e| df_err(&format!("parquet metadata {path}"), e))?;
            let file_schema = builder.schema().clone();
            if schema.is_none() {
                schema = Some(Self::normalize_schema_for_planning(file_schema));
            }
            if let Some(size) = size {
                file_sizes.insert(path.as_ref().to_string(), size);
            }
            splits.extend(row_group_splits(path, builder.metadata()));
        }

        Ok(Self {
            schema: schema
                .ok_or_else(|| df_err("open object-store parquet", "no parquet files"))?,
            store,
            splits,
            file_sizes,
            location,
            table_name: None,
            split_key_cache: Arc::new(Mutex::new(HashMap::new())),
            row_group_bloom_cache: Arc::new(Mutex::new(HashMap::new())),
            stats_trust: proximadb_data_model::StatsTrust::Trusted,
        })
    }

    /// The immutable footer-derived discovery result (schema + splits + file
    /// sizes), for seeding the table-OPEN cache. The object store is not part of
    /// it — see [`table_open_cache`](super::table_open_cache).
    fn discovery(&self) -> super::table_open_cache::TableOpenDiscovery {
        super::table_open_cache::TableOpenDiscovery {
            schema: self.schema.clone(),
            splits: self.splits.clone(),
            file_sizes: self.file_sizes.clone(),
        }
    }

    /// TD-185a: strip per-field metadata from a footer-derived schema before it
    /// becomes the table's planning schema. The materializer bakes the catalog's
    /// field tags (`proxima_column_id`, `comment`) into the Parquet Arrow schema;
    /// nothing on the DataFusion route reads them, and DataFusion's recursive-CTE
    /// planning compares the recursive term's fields against the SQL-declared
    /// CTE schema INCLUDING metadata — a tagged scan field can never match a
    /// SQL-declared column, which failed every recursive CTE over materialized
    /// tables at planning time (`Cannot project plan column … field metadata
    /// differs`). Schema-level (file) metadata is kept: it does not participate
    /// in field comparison.
    fn normalize_schema_for_planning(schema: SchemaRef) -> SchemaRef {
        let fields: Vec<arrow_schema::Field> = schema
            .fields()
            .iter()
            .map(|f| {
                let field = f.as_ref().clone();
                if field.metadata().is_empty() {
                    field
                } else {
                    field.with_metadata(Default::default())
                }
            })
            .collect();
        Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()))
    }

    /// Assemble a table from a cached [`TableOpenDiscovery`](super::table_open_cache::TableOpenDiscovery)
    /// plus a freshly-built (per-query) traced store — the table-OPEN cache-hit
    /// path, which skips LIST + HEAD + footer discovery entirely.
    fn from_cached(
        disc: &super::table_open_cache::TableOpenDiscovery,
        location: &str,
    ) -> DFResult<Self> {
        let store = build_traced_store(location)?;
        Ok(Self {
            schema: disc.schema.clone(),
            store,
            splits: disc.splits.clone(),
            file_sizes: disc.file_sizes.clone(),
            location: location.to_string(),
            table_name: None,
            split_key_cache: Arc::new(Mutex::new(HashMap::new())),
            // Per-query caches start empty (#742) — the table-OPEN cache reuses the
            // immutable footer discovery, not these query-time bloom/key caches.
            row_group_bloom_cache: Arc::new(Mutex::new(HashMap::new())),
            stats_trust: proximadb_data_model::StatsTrust::Trusted,
        })
    }

    /// TD-OLAP-4 shadow probe: the raw inputs a native
    /// [`ParquetScanOperator`](crate::query::execution::native_parquet_scan::ParquetScanOperator)
    /// needs to read the SAME external parquet this DataFusion table reads — the
    /// traced object store, the distinct files (with footer byte sizes), and the
    /// file Arrow schema. Whole-file granularity (every row group); the native scan
    /// does not yet do row-group zone-map pruning, so a native sample built from
    /// this is an upper bound on selective-filter queries (documented caveat for
    /// the engine-dimension trace).
    pub fn native_scan_inputs(
        &self,
    ) -> (Arc<dyn ObjectStore>, Vec<(Path, Option<u64>)>, SchemaRef) {
        let mut seen = std::collections::BTreeSet::new();
        let mut files: Vec<(Path, Option<u64>)> = Vec::new();
        for split in &self.splits {
            if seen.insert(split.file_path.clone()) {
                let size = self.file_sizes.get(&split.file_path).copied();
                files.push((Path::from(split.file_path.as_str()), size));
            }
        }
        (self.store.clone(), files, self.schema.clone())
    }

    /// Attach the registered table name (keys the ADR-037 statistics registry
    /// so [`TableProvider::statistics`] can overlay the HLL NDV estimate).
    pub fn with_table_name(mut self, name: impl Into<String>) -> Self {
        self.table_name = Some(name.into());
        self
    }

    /// ADR-058 D5/§9.A: set this table's footer-stats trust level. Called at
    /// registration with the per-table trust derived from the catalog authority
    /// (`Trusted` for ProximaDB-generated parquet, `Untrusted` for
    /// external/federated). Controls whether `statistics_from_splits` may
    /// surface footer stats as `Precision::Exact` for AggregateStatistics elision.
    pub fn with_stats_trust(mut self, trust: proximadb_data_model::StatsTrust) -> Self {
        self.stats_trust = trust;
        self
    }

    /// Number of row-group splits before runtime filter pruning.
    pub fn split_count(&self) -> usize {
        self.splits.len()
    }

    /// Total rows across all row-group splits (from Parquet footer statistics),
    /// or `None` when no split carries a row count. Footer-`fresh` — used by the
    /// read-route EXPLAIN to disclose the table's estimated scan cardinality.
    pub fn estimated_rows(&self) -> Option<u64> {
        let mut total = 0u64;
        let mut any = false;
        for split in &self.splits {
            if let Some(rows) = split.statistics.row_count {
                total = total.saturating_add(rows);
                any = true;
            }
        }
        any.then_some(total)
    }

    /// Total on-disk bytes across all row-group splits (from Parquet footer
    /// statistics), or `None` when no split carries a byte size.
    pub fn estimated_bytes(&self) -> Option<u64> {
        let mut total = 0u64;
        let mut any = false;
        for split in &self.splits {
            if let Some(bytes) = split.statistics.byte_size {
                total = total.saturating_add(bytes);
                any = true;
            }
        }
        any.then_some(total)
    }

    /// TD-OLAP-4 metadata elision: the footer-derived numeric `(min, max,
    /// null_count)` for a column, reduced (min-of-mins / max-of-maxes / Σ nulls)
    /// across ALL row-group splits — the input for eliding unfiltered
    /// `MIN`/`MAX`/`COUNT(col)` without a scan. Returns `None` unless EVERY split
    /// carries numeric bounds for the column (a partially-covered column would
    /// understate the range), or the bounds are non-numeric (truncated string
    /// stats → the caller scans, matching DataFusion's `Inexact` guard).
    pub fn aggregate_numeric_bounds(&self, col: &str) -> Option<(f64, f64, u64)> {
        if self.splits.is_empty() {
            return None;
        }
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        let mut nulls: u64 = 0;
        for split in &self.splits {
            let cb = split.statistics.column_stats.get(col)?;
            nulls = nulls.saturating_add(cb.null_count);
            let mn = cb.min.as_ref()?.as_f64()?;
            let mx = cb.max.as_ref()?.as_f64()?;
            lo = lo.min(mn);
            hi = hi.max(mx);
        }
        Some((lo, hi, nulls))
    }

    /// Count row-group splits that survive the given filters. Uses min/max
    /// zone-maps plus (bloom semi-join default-ON; disable with
    /// `PROXIMADB_DF_BLOOM_SEMIJOIN=0`) any split key
    /// sets already prefetched into the cache; it does not itself prefetch, so
    /// EXPLAIN-style callers see the min/max result unless `scan()` — or a
    /// test — has run the async prefetch first.
    pub fn pruned_split_count(&self, filters: &[Expr]) -> usize {
        self.prune_splits(filters, SemijoinConfig::from_env()).len()
    }

    fn reader(&self) -> Arc<ObjectStoreParquetSplitReader> {
        Arc::new(ObjectStoreParquetSplitReader {
            schema: self.schema.clone(),
            store: self.store.clone(),
            file_sizes: self.file_sizes.clone(),
        })
    }

    fn prune_splits(&self, filters: &[Expr], cfg: SemijoinConfig) -> Vec<FileSplit> {
        // Build this query's per-column in-list blooms once (cheap; reused across
        // every split), then hold the split-key cache for the whole pass.
        let inlist_blooms = if cfg.enabled {
            self.build_inlist_blooms(filters, cfg)
        } else {
            HashMap::new()
        };
        let guard = if inlist_blooms.is_empty() {
            None
        } else {
            self.split_key_cache
                .lock()
                .ok()
                .zip(self.row_group_bloom_cache.lock().ok())
        };
        let view = guard
            .as_ref()
            .map(|(split_keys, row_group_blooms)| BloomPruneView {
                min_keys: cfg.min_keys,
                inlist_blooms: &inlist_blooms,
                row_group_blooms,
                split_keys,
            });
        self.splits
            .iter()
            .filter(|split| {
                !filters
                    .iter()
                    .any(|filter| filter_prunes_split(split, filter, view.as_ref()))
            })
            .cloned()
            .collect()
    }

    /// Resolve a column name to its canonical bloom key family via the table
    /// schema, or `None` for unsupported/nested types.
    fn column_family(&self, column: &str) -> Option<KeyFamily> {
        let index = self.schema.index_of(column).ok()?;
        column_key_family(self.schema.field(index).data_type())
    }

    /// Build one low-`fpr` bloom per high-NDV in-list column from this query's
    /// predicates (the probe-side/dimension key set the split keys are tested
    /// against). When a column carries more than one qualifying in-list, the
    /// bloom is built over the *union* of their values: a union bloom stays
    /// false-negative-free for every constituent predicate (it can only
    /// over-retain a split, never wrongly prune one).
    fn build_inlist_blooms(
        &self,
        filters: &[Expr],
        cfg: SemijoinConfig,
    ) -> HashMap<String, InlistBloom> {
        let mut inlists: Vec<(String, Vec<ScalarValue>)> = Vec::new();
        for filter in filters {
            collect_semijoin_inlists(filter, cfg.min_keys, &mut inlists);
        }
        let mut per_column: HashMap<String, Vec<ScalarValue>> = HashMap::new();
        for (column, values) in inlists {
            per_column.entry(column).or_default().extend(values);
        }
        let mut blooms = HashMap::new();
        for (column, values) in per_column {
            let Some(family) = self.column_family(&column) else {
                continue;
            };
            blooms.insert(column, build_inlist_bloom(family, &values));
        }
        blooms
    }

    /// TD-OLAP-3 bloom semi-join prefetch: for high-NDV positive in-lists whose
    /// min/max zone-maps fail to prune a split, prefer native Parquet row-group
    /// blooms; only files without those blooms fall back to materializing that
    /// split's distinct join-key set via a projected read of just that column.
    /// Splits the zone-map already prunes are skipped, so extra metadata/key
    /// reads are paid only where the bloom can actually help.
    async fn prepare_split_keys(&self, filters: &[Expr], cfg: SemijoinConfig) {
        if !cfg.enabled {
            return;
        }
        let mut inlists: Vec<(String, Vec<ScalarValue>)> = Vec::new();
        for filter in filters {
            collect_semijoin_inlists(filter, cfg.min_keys, &mut inlists);
        }
        if inlists.is_empty() {
            return;
        }

        let mut missing_native: HashMap<(String, String), Vec<usize>> = HashMap::new();
        for split in &self.splits {
            let Some(rg) = row_group_index(split) else {
                continue;
            };
            for (column, values) in &inlists {
                // Where min/max already prunes, skip the read entirely.
                if split.can_prune_scalar(column, &ScalarPredicate::In(values.clone())) {
                    continue;
                }
                let key = (split.file_path.clone(), rg, column.clone());
                let known = self
                    .row_group_bloom_cache
                    .lock()
                    .ok()
                    .map(|cache| cache.contains_key(&key))
                    .unwrap_or(true);
                if !known {
                    missing_native
                        .entry((split.file_path.clone(), column.clone()))
                        .or_default()
                        .push(rg);
                }
            }
        }

        for ((file_path, column), mut row_groups) in missing_native {
            row_groups.sort_unstable();
            row_groups.dedup();
            let blooms = self
                .read_row_group_blooms(&file_path, &column, &row_groups)
                .await;
            if let Ok(mut cache) = self.row_group_bloom_cache.lock() {
                for (rg, bloom) in blooms {
                    cache.insert((file_path.clone(), rg, column.clone()), bloom);
                }
            }
        }

        for split in &self.splits {
            let Some(rg) = row_group_index(split) else {
                continue;
            };
            for (column, values) in &inlists {
                if split.can_prune_scalar(column, &ScalarPredicate::In(values.clone())) {
                    continue;
                }
                let key = (split.file_path.clone(), rg, column.clone());
                let has_native = self
                    .row_group_bloom_cache
                    .lock()
                    .ok()
                    .and_then(|cache| cache.get(&key).cloned())
                    .flatten()
                    .is_some();
                if has_native {
                    continue;
                }
                let already = self
                    .split_key_cache
                    .lock()
                    .map(|cache| cache.contains_key(&key))
                    .unwrap_or(true);
                if already {
                    continue;
                }
                let keys = self.build_split_keys(&split.file_path, rg, column).await;
                if let Ok(mut cache) = self.split_key_cache.lock() {
                    cache.insert(key, keys.map(Arc::new));
                }
            }
        }
    }

    /// Read native Parquet row-group bloom filters for one join-key column.
    /// This is the preferred semi-join pruning substrate: the bloom lives in
    /// row-group metadata, so the scan can skip a data row group without first
    /// materializing its key column.
    async fn read_row_group_blooms(
        &self,
        file_path: &str,
        column: &str,
        row_groups: &[usize],
    ) -> Vec<(usize, Option<Arc<RowGroupBloom>>)> {
        let Some(field_index) = self.schema.index_of(column).ok() else {
            return row_groups.iter().map(|&rg| (rg, None)).collect();
        };
        let Some(family) = column_key_family(self.schema.field(field_index).data_type()) else {
            return row_groups.iter().map(|&rg| (rg, None)).collect();
        };
        let path = Path::from(file_path);
        let file_size = self.file_sizes.get(file_path).copied();
        let reader = parquet_reader(self.store.clone(), path, file_size);
        let Ok(mut stream_builder) = ParquetRecordBatchStreamBuilder::new(reader).await else {
            return row_groups.iter().map(|&rg| (rg, None)).collect();
        };
        let mut out = Vec::with_capacity(row_groups.len());
        for &rg_index in row_groups {
            let bloom = stream_builder
                .get_row_group_column_bloom_filter(rg_index, field_index)
                .await
                .ok()
                .flatten()
                .map(|filter| Arc::new(RowGroupBloom { family, filter }));
            out.push((rg_index, bloom));
        }
        out
    }

    /// Materialize the distinct key bytes of one join-key column of one row group
    /// via a projected object_store read (only that column's pages are fetched,
    /// and through the tracing store so its bytes land in the I/O trace). Returns
    /// `None` on any error, unsupported/nested key type, or when the distinct-key
    /// count exceeds [`MAX_SPLIT_DISTINCT_KEYS`] — the caller caches the `None`
    /// and falls back to min/max (never a false negative).
    async fn build_split_keys(
        &self,
        file_path: &str,
        rg_index: usize,
        column: &str,
    ) -> Option<SplitKeys> {
        let field_index = self.schema.index_of(column).ok()?;
        let family = column_key_family(self.schema.field(field_index).data_type())?;
        let path = Path::from(file_path);
        let file_size = self.file_sizes.get(file_path).copied();
        let reader = parquet_reader(self.store.clone(), path, file_size);
        let stream_builder = ParquetRecordBatchStreamBuilder::new(reader).await.ok()?;
        let mask = ProjectionMask::roots(stream_builder.parquet_schema(), [field_index]);
        let mut stream = stream_builder
            .with_row_groups(vec![rg_index])
            .with_projection(mask)
            .with_batch_size(8192)
            .build()
            .ok()?;
        let mut seen: HashSet<Vec<u8>> = HashSet::new();
        while let Some(batch) = stream.next().await {
            let batch = batch.ok()?;
            // Projected to exactly the key column; refuse anything else so a
            // mask/index mismatch can never seed keys from the wrong column
            // (which would risk a false-negative prune).
            if batch.num_columns() != 1 || batch.schema().field(0).name() != column {
                return None;
            }
            let array = batch.column(0);
            for row in 0..array.len() {
                if let Some(bytes) = array_key_bytes(array.as_ref(), row) {
                    if seen.len() >= MAX_SPLIT_DISTINCT_KEYS && !seen.contains(&bytes) {
                        return None; // too many distinct keys — fall back to min/max
                    }
                    seen.insert(bytes);
                }
            }
        }
        Some(SplitKeys {
            family,
            keys: seen.into_iter().collect(),
        })
    }
}

#[async_trait]
impl TableProvider for ObjectStoreParquetTable {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        // TD-OLAP-3 slice A: without this override DataFusion's optimizer
        // keeps every WHERE predicate in a FilterExec and hands `scan()` an
        // EMPTY filter list — the split-pruning machinery below never fires.
        // `Inexact` = predicates are passed to `scan()` for best-effort
        // row-group pruning AND the FilterExec stays above for exact
        // row-level filtering, so results cannot regress.
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }

    fn statistics(&self) -> Option<Statistics> {
        // ADR-050 D2 / TD-DFR-1: footer-derived, zero-new-scan column
        // statistics so the planner stops planning blind. Typed min/max value
        // bounds are opt-in (default OFF) — see `df_stats_value_bounds_enabled`.
        Some(statistics_from_splits(
            &self.splits,
            &self.schema,
            self.table_name.as_deref(),
            df_stats_value_bounds_enabled(),
            self.stats_trust,
        ))
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let target_partitions = state.config().target_partitions();
        let semijoin_cfg = SemijoinConfig::from_env();
        // TD-OLAP-3 bloom semi-join (default-OFF): prefetch per-split join-key
        // sets for high-NDV in-lists so `prune_splits` can skip fact splits whose
        // collapsed min/max window cannot. No-op when disabled/absent.
        self.prepare_split_keys(filters, semijoin_cfg).await;
        let pruned = self.prune_splits(filters, semijoin_cfg);
        // Skip-ratio evidence for the runtime-filter promotion gate
        // (TD-OLAP-3/TD-OLAP-4): candidate vs pruned splits, per query.
        crate::observability::io_trace::record_splits(
            self.splits.len() as u64,
            (self.splits.len() - pruned.len()) as u64,
        );
        // Post-prune stats, narrowed to the projection: `partition_statistics`
        // requires the vector to match the exec's PROJECTED schema width.
        let scan_stats = project_statistics(
            &statistics_from_splits(
                &pruned,
                &self.schema,
                self.table_name.as_deref(),
                df_stats_value_bounds_enabled(),
                self.stats_trust,
            ),
            projection.map(|p| p.as_slice()),
        );
        let exec = ProximaScanExec::builder()
            .schema(self.schema.clone())
            .splits(pruned)
            .reader(self.reader())
            .projection(projection.cloned())
            .filters(filters.to_vec())
            .limit(limit)
            .collection_name(self.location.clone())
            .batch_size(8192)
            .target_partitions(target_partitions)
            .statistics(scan_stats)
            .build()?;
        Ok(Arc::new(exec))
    }
}

fn row_group_splits(path: Path, metadata: &ParquetMetaData) -> Vec<FileSplit> {
    let file_path = path.as_ref().to_string();
    let mut splits = Vec::with_capacity(metadata.num_row_groups());
    for i in 0..metadata.num_row_groups() {
        let rg = metadata.row_group(i);
        let mut split = FileSplit::new_row_group(
            file_path.clone(),
            i,
            0,
            rg.total_byte_size().max(0) as u64,
            rg.num_rows(),
        );
        split.statistics = row_group_statistics(metadata, i);
        splits.push(split);
    }
    splits
}

fn row_group_statistics(metadata: &ParquetMetaData, row_group_index: usize) -> SplitStatistics {
    let rg = metadata.row_group(row_group_index);
    let mut stats = SplitStatistics {
        row_count: Some(rg.num_rows().max(0) as u64),
        byte_size: Some(rg.total_byte_size().max(0) as u64),
        ..Default::default()
    };
    for column in rg.columns() {
        let Some(bounds) = column.statistics().and_then(column_bounds_from_parquet) else {
            continue;
        };
        stats
            .column_stats
            .insert(column.column_path().string(), bounds);
    }
    stats
}

fn column_bounds_from_parquet(stats: &ParquetStatistics) -> Option<ColumnBounds> {
    let (min, max) = match stats {
        ParquetStatistics::Boolean(s) => (
            s.min_opt().map(|v| serde_json::json!(*v)),
            s.max_opt().map(|v| serde_json::json!(*v)),
        ),
        ParquetStatistics::Int32(s) => (
            s.min_opt().map(|v| serde_json::json!(*v as i64)),
            s.max_opt().map(|v| serde_json::json!(*v as i64)),
        ),
        ParquetStatistics::Int64(s) => (
            s.min_opt().map(|v| serde_json::json!(*v)),
            s.max_opt().map(|v| serde_json::json!(*v)),
        ),
        ParquetStatistics::Float(s) => (
            s.min_opt().map(|v| serde_json::json!(*v as f64)),
            s.max_opt().map(|v| serde_json::json!(*v as f64)),
        ),
        ParquetStatistics::Double(s) => (
            s.min_opt().map(|v| serde_json::json!(*v)),
            s.max_opt().map(|v| serde_json::json!(*v)),
        ),
        ParquetStatistics::ByteArray(s) => (
            s.min_opt()
                .and_then(|v| v.as_utf8().ok())
                .map(|v| serde_json::json!(v)),
            s.max_opt()
                .and_then(|v| v.as_utf8().ok())
                .map(|v| serde_json::json!(v)),
        ),
        _ => return None,
    };
    Some(ColumnBounds {
        min,
        max,
        null_count: stats.null_count_opt().unwrap_or(0),
        distinct_count: stats.distinct_count_opt(),
    })
}

fn filter_prunes_split(split: &FileSplit, expr: &Expr, view: Option<&BloomPruneView>) -> bool {
    match expr {
        Expr::BinaryExpr(binary) => match binary.op {
            Operator::And => {
                filter_prunes_split(split, &binary.left, view)
                    || filter_prunes_split(split, &binary.right, view)
            }
            Operator::Or => {
                filter_prunes_split(split, &binary.left, view)
                    && filter_prunes_split(split, &binary.right, view)
            }
            _ => comparison_predicate(&binary.left, binary.op, &binary.right)
                .map(|(column, predicate)| split.can_prune_scalar(&column, &predicate))
                .unwrap_or(false),
        },
        Expr::IsNull(inner) => column_name(inner)
            .map(|column| split.can_prune_scalar(&column, &ScalarPredicate::IsNull))
            .unwrap_or(false),
        Expr::IsNotNull(inner) => column_name(inner)
            .map(|column| split.can_prune_scalar(&column, &ScalarPredicate::IsNotNull))
            .unwrap_or(false),
        Expr::InList(list) if !list.negated => {
            let Some(column) = column_name(&list.expr) else {
                return false;
            };
            let values = list
                .list
                .iter()
                .filter_map(literal_value)
                .collect::<Vec<_>>();
            if values.is_empty() {
                return false;
            }
            // Min/max zone-map first (cheap; exact for splits with a narrow key
            // range).
            if split.can_prune_scalar(&column, &ScalarPredicate::In(values.clone())) {
                return true;
            }
            // High-NDV semi-join fallback (TD-OLAP-3): the zone-map collapses
            // when the key set spans the split's [min,max]. Test the split's
            // distinct keys against the low-fpr in-list bloom and prune only when
            // *none* can be present; the FilterExec above re-checks exactly.
            if let Some(view) = view
                && values.len() >= view.min_keys
                && let (Some(inlist), Some(rg)) =
                    (view.inlist_blooms.get(&column), row_group_index(split))
            {
                let key = (split.file_path.clone(), rg, column.clone());
                if let Some(Some(row_group_bloom)) = view.row_group_blooms.get(&key) {
                    return parquet_bloom_prunes_split(row_group_bloom, &values);
                }
                if let Some(Some(split_keys)) = view.split_keys.get(&key) {
                    return bloom_prunes_split(inlist, split_keys);
                }
            }
            false
        }
        Expr::Between(between) if !between.negated => {
            let Some(column) = column_name(&between.expr) else {
                return false;
            };
            match (literal_value(&between.low), literal_value(&between.high)) {
                (Some(low), Some(high)) => {
                    split.can_prune_scalar(&column, &ScalarPredicate::Between(low, high))
                }
                _ => false,
            }
        }
        _ => false,
    }
}

fn comparison_predicate(
    left: &Expr,
    op: Operator,
    right: &Expr,
) -> Option<(String, ScalarPredicate)> {
    if let (Some(column), Some(value)) = (column_name(left), literal_value(right)) {
        return scalar_predicate(op, value).map(|predicate| (column, predicate));
    }
    if let (Some(value), Some(column)) = (literal_value(left), column_name(right)) {
        return scalar_predicate(reverse_operator(op), value).map(|predicate| (column, predicate));
    }
    None
}

fn scalar_predicate(op: Operator, value: ScalarValue) -> Option<ScalarPredicate> {
    match op {
        Operator::Eq => Some(ScalarPredicate::Equal(value)),
        Operator::NotEq => Some(ScalarPredicate::NotEqual(value)),
        Operator::Lt => Some(ScalarPredicate::LessThan(value)),
        Operator::LtEq => Some(ScalarPredicate::LessThanOrEqual(value)),
        Operator::Gt => Some(ScalarPredicate::GreaterThan(value)),
        Operator::GtEq => Some(ScalarPredicate::GreaterThanOrEqual(value)),
        _ => None,
    }
}

fn reverse_operator(op: Operator) -> Operator {
    match op {
        Operator::Lt => Operator::Gt,
        Operator::LtEq => Operator::GtEq,
        Operator::Gt => Operator::Lt,
        Operator::GtEq => Operator::LtEq,
        other => other,
    }
}

fn column_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Column(column) => Some(column.name.clone()),
        Expr::Cast(cast) => column_name(&cast.expr),
        Expr::TryCast(cast) => column_name(&cast.expr),
        _ => None,
    }
}

fn literal_value(expr: &Expr) -> Option<ScalarValue> {
    match expr {
        Expr::Literal(value, _) => df_scalar_value(value),
        Expr::Cast(cast) => literal_value(&cast.expr),
        Expr::TryCast(cast) => literal_value(&cast.expr),
        _ => None,
    }
}

/// Register an object_store-backed Parquet location as a DataFusion table.
pub async fn register_object_store_parquet_location(
    ctx: &SessionContext,
    name: &str,
    location: &str,
    tenant: Option<&str>,
    stats_trust: proximadb_data_model::StatsTrust,
) -> DFResult<Arc<ObjectStoreParquetTable>> {
    // Table-OPEN cache (TD-OLAP-4): on a hit, skip LIST + HEAD + footer discovery
    // and rebuild the table from cached metadata + a fresh per-query traced store.
    // Gated default-OFF; the cache is a pure function of the immutable snapshot,
    // invalidated on re-MATERIALIZE.
    if let Some(disc) = super::table_open_cache::get(tenant, location) {
        let table = Arc::new(
            ObjectStoreParquetTable::from_cached(&disc, location)?
                .with_table_name(name)
                .with_stats_trust(stats_trust),
        );
        ctx.register_table(name, table.clone())?;
        crate::observability::io_trace::record_table_open(true);
        return Ok(table);
    }

    let table = Arc::new(
        ObjectStoreParquetTable::open(location)
            .await?
            .with_table_name(name)
            .with_stats_trust(stats_trust),
    );
    super::table_open_cache::insert(tenant, location, table.discovery());
    crate::observability::io_trace::record_table_open(false);
    ctx.register_table(name, table.clone())?;
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::common::{Column, ScalarValue as DfScalarValue};
    use datafusion::logical_expr::BinaryExpr;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use parquet::schema::types::ColumnPath;

    fn write_two_row_group_parquet(path: &std::path::Path) -> SchemaRef {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Utf8, false),
            Field::new("x", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "b", "b"])),
                Arc::new(Int64Array::from(vec![1_i64, 2, 100, 101])),
            ],
        )
        .unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(2))
            .build();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        schema
    }

    fn gt_filter(column: &str, value: i64) -> Expr {
        Expr::BinaryExpr(BinaryExpr::new(
            Box::new(Expr::Column(Column::from_name(column))),
            Operator::Gt,
            Box::new(Expr::Literal(DfScalarValue::Int64(Some(value)), None)),
        ))
    }

    #[test]
    fn normalize_schema_for_planning_strips_field_metadata_only() {
        let mut field_meta = std::collections::HashMap::new();
        field_meta.insert("proxima_column_id".to_string(), "2".to_string());
        let mut file_meta = std::collections::HashMap::new();
        file_meta.insert("format".to_string(), "proximadb".to_string());
        let schema = Arc::new(Schema::new_with_metadata(
            vec![
                Field::new("id", DataType::Int64, true).with_metadata(field_meta),
                Field::new("name", DataType::Utf8, false),
            ],
            file_meta.clone(),
        ));

        let normalized = ObjectStoreParquetTable::normalize_schema_for_planning(schema);
        // Field metadata (the catalog's proxima_column_id / comment tags) is
        // stripped — it can never match a SQL-declared CTE column in DataFusion's
        // recursive-CTE field comparison.
        assert!(normalized.field(0).metadata().is_empty());
        assert!(normalized.field(1).metadata().is_empty());
        // Names / types / nullability are preserved.
        assert_eq!(normalized.field(0).name(), "id");
        assert_eq!(normalized.field(0).data_type(), &DataType::Int64);
        assert!(normalized.field(0).is_nullable());
        assert_eq!(normalized.field(1).name(), "name");
        // Schema-level (file) metadata is kept.
        assert_eq!(normalized.metadata(), &file_meta);
    }

    #[tokio::test]
    async fn opens_bridge_base_prefix_and_discovers_data_parquet() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        write_two_row_group_parquet(&data_dir.join("part-0.parquet"));

        let location = format!("file://{}", tmp.path().display());
        let table = ObjectStoreParquetTable::open(&location).await.unwrap();
        assert_eq!(table.split_count(), 2);
        assert_eq!(table.schema().fields().len(), 2);
    }

    #[tokio::test]
    async fn aggregate_numeric_bounds_reduces_min_max_across_row_groups() {
        // Two row groups: x = [1,2] and [100,101]. Footer elision must reduce to
        // min-of-mins=1, max-of-maxes=101 across BOTH splits (not just the first).
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        write_two_row_group_parquet(&data_dir.join("part-0.parquet"));

        let location = format!("file://{}", tmp.path().display());
        let table = ObjectStoreParquetTable::open(&location).await.unwrap();
        assert_eq!(table.split_count(), 2, "two row groups");

        let (min, max, nulls) = table.aggregate_numeric_bounds("x").expect("numeric bounds");
        assert_eq!(min, 1.0, "min-of-mins across both row groups");
        assert_eq!(max, 101.0, "max-of-maxes across both row groups");
        assert_eq!(nulls, 0);
        // A truncated-stats / string column reduces to numeric `None` (→ scan).
        assert!(
            table.aggregate_numeric_bounds("k").is_none()
                || table.aggregate_numeric_bounds("k").unwrap().0.is_nan(),
            "string column has no numeric bounds"
        );
        // A missing column → None (never a wrong bound).
        assert!(table.aggregate_numeric_bounds("nope").is_none());
    }

    #[tokio::test]
    async fn native_scan_inputs_yields_distinct_files_with_sizes_and_schema() {
        // Two row groups live in ONE file → the native scan (whole-file) must see a
        // single distinct file, with a footer byte size and the full 2-column schema.
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        write_two_row_group_parquet(&data_dir.join("part-0.parquet"));

        let location = format!("file://{}", tmp.path().display());
        let table = ObjectStoreParquetTable::open(&location).await.unwrap();
        assert_eq!(table.split_count(), 2, "two row groups");

        let (_store, files, schema) = table.native_scan_inputs();
        assert_eq!(
            files.len(),
            1,
            "two row groups collapse to one distinct file"
        );
        assert!(files[0].0.as_ref().ends_with("part-0.parquet"));
        assert!(
            files[0].1.is_some_and(|sz| sz > 0),
            "footer byte size present"
        );
        assert_eq!(schema.fields().len(), 2, "full file schema (k, x)");
    }

    #[tokio::test]
    async fn estimated_rows_and_bytes_sum_row_group_footer_stats() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        // Two row groups of 2 rows each → 4 rows total.
        write_two_row_group_parquet(&data_dir.join("part-0.parquet"));

        let location = format!("file://{}", tmp.path().display());
        let table = ObjectStoreParquetTable::open(&location).await.unwrap();
        assert_eq!(table.split_count(), 2);
        assert_eq!(table.estimated_rows(), Some(4));
        assert!(
            table.estimated_bytes().unwrap_or(0) > 0,
            "row-group byte sizes are present in the footer"
        );
    }

    #[tokio::test]
    async fn table_provider_statistics_projects_footer_bounds() {
        use datafusion_common::stats::Precision;

        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        write_two_row_group_parquet(&data_dir.join("part-0.parquet"));

        let location = format!("file://{}", tmp.path().display());
        let table = ObjectStoreParquetTable::open(&location).await.unwrap();

        // Provider path — value bounds are gated default-OFF
        // (`df_stats_value_bounds_enabled`), but rows/bytes/nulls always flow.
        let stats = TableProvider::statistics(&table).expect("footer-backed statistics");
        assert_eq!(stats.num_rows, Precision::Exact(4));
        // Schema-width contract: one entry per table column.
        assert_eq!(stats.column_statistics.len(), 2);
        // `null_count` is exact footer metadata → `Exact` (gate-independent; enables
        // `COUNT(col)` elision).
        assert_eq!(stats.column_statistics[1].null_count, Precision::Exact(0));
        assert_eq!(
            stats.column_statistics[1].min_value,
            Precision::Absent,
            "value bounds stay Absent while the gate is off (default)"
        );

        // With bounds enabled: min-of-mins / max-of-maxes across row groups.
        let full = statistics_from_splits(
            &table.splits,
            &table.schema,
            None,
            true,
            proximadb_data_model::StatsTrust::Trusted,
        );
        let k = &full.column_statistics[0];
        assert_eq!(
            k.min_value,
            Precision::Inexact(DfScalarValue::Utf8(Some("a".to_string())))
        );
        assert_eq!(
            k.max_value,
            Precision::Inexact(DfScalarValue::Utf8(Some("b".to_string())))
        );
        // Numeric column: parquet min/max are exact → `Exact` (elidable MIN/MAX);
        // null_count exact for every type. (The string column `k` above stays
        // Inexact — parquet truncates BYTE_ARRAY stats.)
        let x = &full.column_statistics[1];
        assert_eq!(x.min_value, Precision::Exact(DfScalarValue::Int64(Some(1))));
        assert_eq!(
            x.max_value,
            Precision::Exact(DfScalarValue::Int64(Some(101)))
        );
        assert_eq!(x.null_count, Precision::Exact(0));
    }

    /// TD-OLAP-3 slice A: with `supports_filters_pushdown` (Inexact), a WHERE
    /// predicate planned through a real DataFusion session reaches `scan()`
    /// and prunes row-group splits — previously the optimizer kept it in a
    /// FilterExec and `scan()` saw an empty filter list (dormant pruning).
    #[tokio::test]
    async fn where_predicate_reaches_scan_and_prunes_splits() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        write_two_row_group_parquet(&data_dir.join("part-0.parquet"));

        let location = format!("file://{}", tmp.path().display());
        let ctx = SessionContext::new();
        register_object_store_parquet_location(
            &ctx,
            "t",
            &location,
            None,
            proximadb_data_model::StatsTrust::Trusted,
        )
        .await
        .unwrap();

        // Row group 1 holds x ∈ {1,2}; row group 2 holds x ∈ {100,101}.
        let df = ctx.sql("SELECT x FROM t WHERE x > 50").await.unwrap();
        let plan = df.create_physical_plan().await.unwrap();

        fn find_scan(plan: &Arc<dyn ExecutionPlan>) -> Option<usize> {
            if let Some(scan) = plan.downcast_ref::<ProximaScanExec>() {
                return Some(scan.total_split_count());
            }
            plan.children().iter().find_map(|c| find_scan(c))
        }
        let splits = find_scan(&plan).expect("plan contains a ProximaScanExec");
        assert_eq!(
            splits, 1,
            "the WHERE predicate must reach scan() and prune the disjoint row group"
        );

        // Correctness: the FilterExec above still applies exact semantics.
        let rows = ctx
            .sql("SELECT x FROM t WHERE x > 50")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let total: usize = rows.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2, "exactly the two rows with x > 50");
    }

    #[tokio::test]
    async fn prunes_row_groups_from_parquet_min_max_statistics() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        write_two_row_group_parquet(&data_dir.join("part-0.parquet"));

        let location = format!("file://{}", tmp.path().display());
        let table = ObjectStoreParquetTable::open(&location).await.unwrap();
        let filters = vec![gt_filter("x", 50)];

        assert_eq!(table.split_count(), 2);
        assert_eq!(table.pruned_split_count(&filters), 1);
    }

    // ---- TD-OLAP-3 bloom semi-join --------------------------------------

    /// Two row groups, one Int64 key column, contents controlled per group via
    /// an explicit `flush()` between the two write batches.
    fn write_int_key_parquet(path: &std::path::Path, rg0: &[i64], rg1: &[i64]) -> SchemaRef {
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), None).unwrap();
        let b0 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(rg0.to_vec()))],
        )
        .unwrap();
        writer.write(&b0).unwrap();
        writer.flush().unwrap(); // seal row group 0
        let b1 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(rg1.to_vec()))],
        )
        .unwrap();
        writer.write(&b1).unwrap();
        writer.close().unwrap();
        schema
    }

    fn in_list_i64(column: &str, values: &[i64]) -> Expr {
        Expr::InList(datafusion::logical_expr::expr::InList::new(
            Box::new(Expr::Column(Column::from_name(column))),
            values
                .iter()
                .map(|&v| Expr::Literal(DfScalarValue::Int64(Some(v)), None))
                .collect(),
            false,
        ))
    }

    /// The headline case: an in-list whose min/max window overlaps a row group
    /// but whose keys are all absent from it. Min/max cannot prune; the bloom
    /// can. Proves the bloom prunes strictly more than the zone-map here.
    #[tokio::test]
    async fn bloom_semijoin_prunes_where_minmax_collapses() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        // rg0 = a few odds spanning [1,199]; rg1 = 1000..=1099.
        let odds: Vec<i64> = vec![1, 51, 99, 149, 199];
        let far: Vec<i64> = (1000..=1099).collect();
        write_int_key_parquet(&data_dir.join("part-0.parquet"), &odds, &far);

        let location = format!("file://{}", tmp.path().display());
        let table = ObjectStoreParquetTable::open(&location).await.unwrap();
        assert_eq!(table.split_count(), 2);

        // Even keys 2..=300: each ∈ [1,199] for rg0 (min/max cannot prune) but
        // none is actually present (rg0 holds only odds); all < 1000 for rg1.
        let evens: Vec<i64> = (2..=300).step_by(2).collect();
        let filters = vec![in_list_i64("k", &evens)];
        let on = SemijoinConfig {
            enabled: true,
            min_keys: 4,
        };
        let off = SemijoinConfig {
            enabled: false,
            min_keys: 4,
        };

        // Baseline (min/max only): rg1 pruned, rg0 survives.
        assert_eq!(table.prune_splits(&filters, off).len(), 1);

        // With split keys prefetched: rg0 is also pruned — each of its odd keys
        // misses the even-keyed in-list bloom.
        table.prepare_split_keys(&filters, on).await;
        assert_eq!(
            table.prune_splits(&filters, on).len(),
            0,
            "bloom prunes the row group whose collapsed min/max window could not"
        );
    }

    /// Correctness invariant (mandate #2/#8): a split that DOES contain a probe
    /// key must never be pruned. A false-negative prune would drop real rows.
    #[tokio::test]
    async fn bloom_semijoin_never_prunes_split_with_a_match() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        let odds: Vec<i64> = (1..=199).step_by(2).collect();
        let far: Vec<i64> = (1000..=1099).collect();
        write_int_key_parquet(&data_dir.join("part-0.parquet"), &odds, &far);

        let location = format!("file://{}", tmp.path().display());
        let table = ObjectStoreParquetTable::open(&location).await.unwrap();

        // In-list of odd keys that ARE present in rg0 (plus some absent evens).
        let mut present: Vec<i64> = (1..=197).step_by(2).collect();
        present.extend([2_i64, 4, 6, 8]);
        let filters = vec![in_list_i64("k", &present)];
        let on = SemijoinConfig {
            enabled: true,
            min_keys: 4,
        };
        table.prepare_split_keys(&filters, on).await;
        let survivors = table.prune_splits(&filters, on);
        assert_eq!(survivors.len(), 1, "rg0 contains matches and must survive");
        assert_eq!(
            row_group_index(&survivors[0]),
            Some(0),
            "the surviving split is the one holding the matching keys"
        );
    }

    /// No-false-negative at the bloom layer: every in-list key must report
    /// `might_contain == true` through the same encoding the split side uses —
    /// so a split key that really is in the in-list can never be pruned away.
    #[test]
    fn inlist_bloom_has_no_false_negative() {
        let values: Vec<ScalarValue> = (1..=500).map(ScalarValue::Int64).collect();
        let bloom = build_inlist_bloom(KeyFamily::Int, &values);
        assert_eq!(bloom.family, KeyFamily::Int);
        for v in &values {
            let bytes = probe_key_bytes(KeyFamily::Int, v).unwrap();
            assert!(
                bloom.filter.might_contain(&bytes),
                "in-list key {v:?} must not be a bloom false negative"
            );
        }
    }

    /// `build_split_keys` reads exactly the distinct values of the key column.
    #[tokio::test]
    async fn build_split_keys_reads_distinct_column_values() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        // rg0 has a duplicate (7 appears twice) → 3 distinct keys.
        write_int_key_parquet(&data_dir.join("part-0.parquet"), &[7, 7, 9, 11], &[1000]);

        let location = format!("file://{}", tmp.path().display());
        let table = ObjectStoreParquetTable::open(&location).await.unwrap();
        let sk = table
            .build_split_keys(&table.splits[0].file_path, 0, "k")
            .await
            .expect("keys read for the Int64 column");
        assert_eq!(sk.family, KeyFamily::Int);
        let mut got: Vec<i64> = sk
            .keys
            .iter()
            .map(|b| i64::from_le_bytes(b.as_slice().try_into().unwrap()))
            .collect();
        got.sort_unstable();
        assert_eq!(got, vec![7, 9, 11]);
    }

    /// `bloom_prunes_split`: prunes on a disjoint key set, keeps on any overlap.
    #[test]
    fn bloom_prunes_split_only_when_disjoint() {
        let evens: Vec<ScalarValue> = (2..=200).step_by(2).map(ScalarValue::Int64).collect();
        let inlist = build_inlist_bloom(KeyFamily::Int, &evens);

        let enc = |vs: &[i64]| SplitKeys {
            family: KeyFamily::Int,
            keys: vs.iter().map(|v| v.to_le_bytes().to_vec()).collect(),
        };
        // All odd → disjoint from the even in-list → prune.
        assert!(bloom_prunes_split(&inlist, &enc(&[1, 51, 99, 149, 199])));
        // Contains an even key (in the in-list) → must NOT prune.
        assert!(!bloom_prunes_split(&inlist, &enc(&[1, 51, 100, 149])));
        // Family mismatch → never prune.
        let str_keys = SplitKeys {
            family: KeyFamily::Str,
            keys: vec![b"x".to_vec()],
        };
        assert!(!bloom_prunes_split(&inlist, &str_keys));
    }

    /// Two different high-NDV in-lists on the SAME column: the per-column bloom
    /// is the union of their values, so a split whose keys match one in-list is
    /// never pruned by the other's bloom. (Keying by column with first-wins
    /// would false-negative-prune here.)
    #[tokio::test]
    async fn bloom_semijoin_union_over_multiple_inlists_is_safe() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        let odds: Vec<i64> = vec![1, 51, 99, 149, 199];
        let far: Vec<i64> = (1000..=1099).collect();
        write_int_key_parquet(&data_dir.join("part-0.parquet"), &odds, &far);

        let location = format!("file://{}", tmp.path().display());
        let table = ObjectStoreParquetTable::open(&location).await.unwrap();

        // `k IN (evens…) AND k IN (rg0 keys…)` — rg0 has rows matching the
        // second list, so it must survive despite missing the first.
        let evens: Vec<i64> = (2..=300).step_by(2).collect();
        let filters = vec![in_list_i64("k", &evens).and(in_list_i64("k", &odds))];
        let on = SemijoinConfig {
            enabled: true,
            min_keys: 4,
        };
        table.prepare_split_keys(&filters, on).await;
        let survivors = table.prune_splits(&filters, on);
        assert_eq!(
            survivors.len(),
            1,
            "rg0 matches the second in-list; keep it"
        );
        assert_eq!(row_group_index(&survivors[0]), Some(0));
    }

    /// Below the cardinality threshold the bloom is neither built nor consulted;
    /// pruning falls back to min/max exactly as before.
    #[tokio::test]
    async fn bloom_semijoin_gated_out_below_threshold() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        let odds: Vec<i64> = vec![1, 51, 99, 149, 199];
        let far: Vec<i64> = (1000..=1099).collect();
        write_int_key_parquet(&data_dir.join("part-0.parquet"), &odds, &far);

        let location = format!("file://{}", tmp.path().display());
        let table = ObjectStoreParquetTable::open(&location).await.unwrap();
        let evens: Vec<i64> = (2..=300).step_by(2).collect(); // 150 keys
        let filters = vec![in_list_i64("k", &evens)];
        let cfg = SemijoinConfig {
            enabled: true,
            min_keys: 1000, // above the 150-key list → gated out
        };
        table.prepare_split_keys(&filters, cfg).await;
        assert_eq!(
            table.prune_splits(&filters, cfg).len(),
            1,
            "rg0 survives via min/max; the bloom path is gated out"
        );
    }

    /// Build-side and probe-side encodings must agree byte-for-byte, and a
    /// cross-family probe must fail to encode (→ conservative keep, never a
    /// false negative).
    #[test]
    fn probe_and_array_key_bytes_agree() {
        let ints = Int64Array::from(vec![5_i64, 42]);
        assert_eq!(
            array_key_bytes(&ints, 0),
            Some(5_i64.to_le_bytes().to_vec())
        );
        assert_eq!(
            probe_key_bytes(KeyFamily::Int, &ScalarValue::Int64(5)),
            array_key_bytes(&ints, 0)
        );

        let strs = StringArray::from(vec!["x", "y"]);
        assert_eq!(
            probe_key_bytes(KeyFamily::Str, &ScalarValue::String("x".to_string())),
            array_key_bytes(&strs, 0)
        );

        // Cross-family probe (int against a string key) cannot encode.
        assert_eq!(
            probe_key_bytes(KeyFamily::Str, &ScalarValue::Int64(5)),
            None
        );
    }

    /// Wide fact table: one Int64 key column + 12 Float64 payload columns, one
    /// row group per `groups` entry (sealed with `flush()`).
    fn write_wide_fact_parquet(path: &std::path::Path, groups: &[Vec<i64>]) -> SchemaRef {
        let mut fields = vec![Field::new("k", DataType::Int64, false)];
        for j in 0..12 {
            fields.push(Field::new(format!("p{j}"), DataType::Float64, false));
        }
        let schema = Arc::new(Schema::new(fields));
        let props = WriterProperties::builder()
            .set_column_bloom_filter_enabled(ColumnPath::from("k"), true)
            .set_column_bloom_filter_fpp(ColumnPath::from("k"), 0.000_001)
            .set_column_bloom_filter_max_ndv(ColumnPath::from("k"), 256)
            .build();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props)).unwrap();
        for (gi, ks) in groups.iter().enumerate() {
            let n = ks.len();
            let mut cols: Vec<arrow_array::ArrayRef> = vec![Arc::new(Int64Array::from(ks.clone()))];
            for j in 0..12 {
                // Distinct-ish floats so the payload does not trivially compress
                // away (we need the payload read to dominate the key-column read).
                let vals: Vec<f64> = (0..n)
                    .map(|i| (i as f64) * 1.000_001 + (j as f64) * 7.0 + gi as f64)
                    .collect();
                cols.push(Arc::new(arrow_array::Float64Array::from(vals)));
            }
            let batch = RecordBatch::try_new(schema.clone(), cols).unwrap();
            writer.write(&batch).unwrap();
            writer.flush().unwrap(); // seal this row group
        }
        writer.close().unwrap();
        schema
    }

    /// Native Parquet row-group blooms are the profitable path: they let the
    /// scan decide whether a row group can match without first reading the key
    /// column from that same row group. The older key-column fallback remains
    /// for files that do not carry native blooms, but must not run when native
    /// blooms are available.
    #[tokio::test]
    async fn bloom_semijoin_prefers_native_parquet_bloom_over_key_scan() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        let rg0: Vec<i64> = (0..300).map(|i| 1 + 2 * (i % 100)).collect();
        let rg1: Vec<i64> = (0..300).map(|i| 1000 + (i % 100)).collect();
        let rg2: Vec<i64> = (0..300).map(|i| 2 + 2 * (i % 10)).collect();
        write_wide_fact_parquet(&data_dir.join("part-0.parquet"), &[rg0, rg1, rg2]);

        let location = format!("file://{}", tmp.path().display());
        let table = ObjectStoreParquetTable::open(&location).await.unwrap();
        let evens: Vec<i64> = (2..=300).step_by(2).collect();
        let filters = vec![in_list_i64("k", &evens)];
        let cfg = SemijoinConfig {
            enabled: true,
            min_keys: 4,
        };

        table.prepare_split_keys(&filters, cfg).await;
        let survivors = table.prune_splits(&filters, cfg);
        assert_eq!(
            survivors.len(),
            1,
            "native row-group bloom prunes rg0; min/max prunes rg1; rg2 survives"
        );
        assert_eq!(row_group_index(&survivors[0]), Some(2));
        assert!(
            table
                .row_group_bloom_cache
                .lock()
                .unwrap()
                .values()
                .any(|entry| entry.is_some()),
            "native Parquet row-group blooms were read"
        );
        assert!(
            table.split_key_cache.lock().unwrap().is_empty(),
            "native blooms must avoid query-time key-column materialization"
        );
    }

    /// TD-OLAP-4 evidence: on a high-NDV `IN`-list semi-join, the bloom prunes a
    /// wide fact split whose collapsed min/max window cannot — a *metered*
    /// bytes-scanned reduction (io_trace `bytes_read`), same rows. Drives the
    /// real `scan()` path; the store is wrapped inside the io_trace scope so
    /// spawned-partition reads are attributed (TD-OLAP-3 handle capture).
    #[tokio::test]
    async fn bloom_semijoin_reduces_metered_bytes_scanned() {
        use crate::observability::io_trace;

        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        // rg0: odds spanning [1,199] (min/max cannot prune vs an even in-list,
        // but the keys are all absent → the bloom can). rg1: 1000..1099 (min/max
        // prunes). rg2: evens 2..20 (present in the in-list → the survivor).
        let rg0: Vec<i64> = (0..300).map(|i| 1 + 2 * (i % 100)).collect();
        let rg1: Vec<i64> = (0..300).map(|i| 1000 + (i % 100)).collect();
        let rg2: Vec<i64> = (0..300).map(|i| 2 + 2 * (i % 10)).collect();
        write_wide_fact_parquet(&data_dir.join("part-0.parquet"), &[rg0, rg1, rg2]);

        let location = format!("file://{}", tmp.path().display());
        // Wide projection so a pruned row group saves a large payload read; the
        // bloom prefetch only reads the narrow key column.
        let evens: Vec<i64> = (2..=300).step_by(2).collect();
        let in_list = evens
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let query =
            format!("SELECT p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11 FROM t WHERE k IN ({in_list})");

        async fn measure(location: &str, on: bool, query: &str) -> (usize, u64, u64) {
            // SAFETY: nextest runs each test in its own process; this test is the
            // only code in it and toggles the flag single-threaded.
            unsafe {
                if on {
                    std::env::set_var("PROXIMADB_DF_BLOOM_SEMIJOIN", "1");
                    std::env::set_var("PROXIMADB_DF_BLOOM_SEMIJOIN_MIN_KEYS", "4");
                } else {
                    // Explicit opt-out: the gate defaults ON (ADR-056 MVP-1), so
                    // the "off" arm must set `=0`, not merely remove the var.
                    std::env::set_var("PROXIMADB_DF_BLOOM_SEMIJOIN", "0");
                }
            }
            io_trace::scope(async move {
                let ctx = SessionContext::new();
                // Register (wraps the store) INSIDE the scope so the io_trace
                // handle is captured for spawned-partition reads.
                register_object_store_parquet_location(
                    &ctx,
                    "t",
                    location,
                    None,
                    proximadb_data_model::StatsTrust::Trusted,
                )
                .await
                .unwrap();
                let batches = ctx.sql(query).await.unwrap().collect().await.unwrap();
                let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
                let snap = io_trace::snapshot().expect("snapshot inside scope");
                (rows, snap.bytes_read, snap.splits_pruned)
            })
            .await
        }

        let (rows_off, bytes_off, pruned_off) = measure(&location, false, &query).await;
        let (rows_on, bytes_on, pruned_on) = measure(&location, true, &query).await;
        unsafe {
            std::env::remove_var("PROXIMADB_DF_BLOOM_SEMIJOIN");
            std::env::remove_var("PROXIMADB_DF_BLOOM_SEMIJOIN_MIN_KEYS");
        }

        let reduction = 100.0 * (bytes_off.saturating_sub(bytes_on)) as f64 / bytes_off as f64;
        eprintln!(
            "TD-OLAP-4 bloom-semijoin evidence: rows off/on = {rows_off}/{rows_on}; \
             bytes_read off/on = {bytes_off}/{bytes_on} ({reduction:.1}% reduction); \
             splits_pruned off/on = {pruned_off}/{pruned_on}"
        );

        // Correctness: identical results with and without the prefilter.
        assert_eq!(rows_off, rows_on, "bloom prefilter must not change results");
        assert_eq!(
            rows_on, 300,
            "only rg2's 300 even-keyed rows match the in-list"
        );
        // The bloom prunes one extra split (rg0) that min/max could not …
        assert!(
            pruned_on > pruned_off,
            "bloom prunes an extra split (off={pruned_off} on={pruned_on})"
        );
        // … and pruning that wide split is a metered bytes-scanned reduction.
        assert!(
            bytes_on < bytes_off,
            "bloom must reduce metered bytes_read (off={bytes_off} on={bytes_on})"
        );
    }
}
