//! Bounded-memory MATERIALIZE (P2.4, TD-USUB-10 / TD-OLAPFMT-3).
//!
//! The pieces `DmlService::materialize_table_to_parquet` composes:
//!
//! * [`sort_batches`]: the cluster-key sort as a DataFusion `SortExec` under a
//!   memory-limited pool, so a table larger than the budget sorts by spilling runs
//!   to local disk instead of holding every row;
//! * [`generation_stem`]: the name every data file of one publication starts
//!   with, unique per publication and ordered by time;
//! * [`gc_older_generations`]: after version `V` commits, delete the data files
//!   no longer referenced by `V` or `V − 1` (a reader that opened `V − 1` keeps
//!   its files).

use std::collections::HashSet;

use anyhow::{Result, anyhow};
use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use futures::stream::BoxStream;
use object_store::path::Path;
use proximadb_storage_common::object_store_bridge::ObjectStoreBridge;

/// Rows per scan chunk (`PROXIMADB_MATERIALIZE_SCAN_CHUNK_ROWS`).
const DEFAULT_SCAN_CHUNK_ROWS: usize = 8_192;
/// Memory budget of the cluster-key sort before it spills
/// (`PROXIMADB_MATERIALIZE_SORT_MEMORY_BYTES`).
const DEFAULT_SORT_MEMORY_BYTES: usize = 256 * 1024 * 1024;

fn env_positive(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
}

pub(super) fn scan_chunk_rows() -> usize {
    env_positive("PROXIMADB_MATERIALIZE_SCAN_CHUNK_ROWS").unwrap_or(DEFAULT_SCAN_CHUNK_ROWS)
}

fn sort_memory_bytes() -> usize {
    env_positive("PROXIMADB_MATERIALIZE_SORT_MEMORY_BYTES").unwrap_or(DEFAULT_SORT_MEMORY_BYTES)
}

/// `g{version:020}-{nonce}`, where `version` is the Iceberg metadata version the
/// publication commits as (its parent's successor, fixed before the scan). The
/// create-only commit makes that a total order across processes, so generation
/// order never depends on a clock: a publisher that has not committed yet can
/// only commit a later version than any committed one. The random nonce keeps
/// two publishers racing for the same version from sharing a part name.
pub(super) fn generation_stem(version: u64) -> String {
    let nonce = uuid::Uuid::new_v4().simple().to_string();
    format!("g{version:020}-{}", &nonce[..8])
}

/// The generation-ordering prefix of a data file name (`g{020}`), if the name
/// has one. Files written before generation naming (`part-0.parquet`) have none.
fn generation_of(file_name: &str) -> Option<&str> {
    let generation = file_name.get(..21)?;
    (generation.starts_with('g') && generation[1..].bytes().all(|b| b.is_ascii_digit()))
        .then_some(generation)
}

/// What a failed `publish_iceberg_files` actually left behind.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum CommitCheck {
    /// The commit landed as this version (its error came after the write, e.g. a
    /// lost response whose retry hit its own create-only precondition).
    Landed(u64),
    /// No version names the parts: they are safe to delete.
    NotLanded,
    /// The metadata could not be read: keep the parts (GC collects them later).
    Unknown,
}

/// After `publish_iceberg_files` returned an error, decide whether the commit
/// landed anyway. The publication can only have committed as `successor`, so
/// that version, if it exists, decides by whether it names `parts`.
pub(super) async fn check_commit(
    bridge: &dyn ObjectStoreBridge,
    metadata_prefix: &str,
    successor: u64,
    parts: &[Path],
) -> CommitCheck {
    let latest = match bridge.latest_metadata_version(metadata_prefix).await {
        Ok(latest) => latest,
        Err(_) => return CommitCheck::Unknown,
    };
    let Some(first) = parts.first() else {
        return CommitCheck::NotLanded;
    };
    if latest.is_none_or(|latest| latest < successor) {
        return CommitCheck::NotLanded;
    }
    match bridge.read_iceberg_table(metadata_prefix, successor).await {
        Ok(files) if files.contains(first) => CommitCheck::Landed(successor),
        Ok(_) => CommitCheck::NotLanded,
        Err(_) => CommitCheck::Unknown,
    }
}

/// Batches in ascending `column` order, NULLs last (TD-OLAP-6 clustering:
/// tight row-group bounds on the cluster key). DataFusion `SortExec` under a memory-limited
/// pool: past the budget it sorts runs, spills them to local disk and merges.
pub(super) fn sort_batches(
    input: BoxStream<'static, Result<RecordBatch>>,
    schema: SchemaRef,
    column: &str,
) -> Result<BoxStream<'static, Result<RecordBatch>>> {
    sort_batches_with_budget(input, schema, column, sort_memory_bytes())
}

#[cfg(feature = "datafusion-integration")]
fn sort_batches_with_budget(
    input: BoxStream<'static, Result<RecordBatch>>,
    schema: SchemaRef,
    column: &str,
    memory_bytes: usize,
) -> Result<BoxStream<'static, Result<RecordBatch>>> {
    use std::sync::Arc;

    use arrow_schema::SortOptions;
    use datafusion::execution::TaskContext;
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;
    use datafusion::physical_expr::{LexOrdering, PhysicalSortExpr};
    use datafusion::physical_plan::ExecutionPlan;
    use datafusion::physical_plan::SendableRecordBatchStream;
    use datafusion::physical_plan::expressions::col;
    use datafusion::physical_plan::sorts::sort::SortExec;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use datafusion::physical_plan::streaming::{PartitionStream, StreamingTableExec};
    use futures::StreamExt;

    /// The scan as a one-shot DataFusion partition.
    struct OnceStream {
        schema: SchemaRef,
        stream: parking_lot::Mutex<Option<SendableRecordBatchStream>>,
    }
    impl std::fmt::Debug for OnceStream {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("MaterializeScan")
        }
    }
    impl PartitionStream for OnceStream {
        fn schema(&self) -> &SchemaRef {
            &self.schema
        }
        fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
            self.stream.lock().take().unwrap_or_else(|| {
                Box::pin(RecordBatchStreamAdapter::new(
                    self.schema.clone(),
                    futures::stream::once(async {
                        Err(datafusion::error::DataFusionError::Internal(
                            "materialize scan executed twice".into(),
                        ))
                    }),
                ))
            })
        }
    }

    let df_input = input
        .map(|batch| batch.map_err(|e| datafusion::error::DataFusionError::External(e.into())));
    let source = OnceStream {
        schema: schema.clone(),
        stream: parking_lot::Mutex::new(Some(Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            df_input,
        )))),
    };
    let scan = StreamingTableExec::try_new(
        schema.clone(),
        vec![Arc::new(source)],
        None,
        Vec::<LexOrdering>::new(),
        false,
        None,
    )?;
    let ordering = LexOrdering::new(vec![PhysicalSortExpr::new(
        col(column, &schema)?,
        SortOptions {
            descending: false,
            nulls_first: false,
        },
    )])
    .ok_or_else(|| anyhow!("materialize: empty sort ordering"))?;
    let sort = Arc::new(SortExec::new(ordering, Arc::new(scan)));
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_limit(memory_bytes, 1.0)
        .build_arc()?;
    // The merge phase reserves its buffers up front; keep that reservation a
    // fraction of the budget so a small budget spills instead of failing.
    let config = datafusion::execution::config::SessionConfig::new()
        .with_sort_spill_reservation_bytes((memory_bytes / 8).min(10 * 1024 * 1024));
    let ctx = Arc::new(
        TaskContext::default()
            .with_session_config(config)
            .with_runtime(runtime),
    );
    let sorted = sort.execute(0, ctx)?;
    Ok(sorted
        .map(|batch| batch.map_err(|e| anyhow!("materialize sort: {e}")))
        .boxed())
}

/// Without DataFusion the sort is in memory: the whole table is held. Builds
/// that publish large tables enable `datafusion-integration` (a default feature).
#[cfg(not(feature = "datafusion-integration"))]
fn sort_batches_with_budget(
    input: BoxStream<'static, Result<RecordBatch>>,
    schema: SchemaRef,
    column: &str,
    _memory_bytes: usize,
) -> Result<BoxStream<'static, Result<RecordBatch>>> {
    use futures::{StreamExt, TryStreamExt};
    let index = schema.index_of(column)?;
    Ok(futures::stream::once(async move {
        let batches: Vec<RecordBatch> = input.try_collect().await?;
        let all = arrow::compute::concat_batches(&schema, &batches)?;
        let order = arrow::compute::sort_to_indices(
            all.column(index),
            Some(arrow_schema::SortOptions {
                descending: false,
                nulls_first: false,
            }),
            None,
        )?;
        Ok(arrow::compute::take_record_batch(&all, &order)?)
    })
    .boxed())
}

/// Delete the data files under `data_prefix` that the just-committed version
/// (`keep_current`) and its predecessor (`keep_previous`) do not reference and
/// that are OLDER than this publication (`current_stem`): an older generation,
/// or a pre-generation name. This generation's other files (a publisher that
/// lost the race for this version) and newer ones (a publisher that has not
/// committed yet) are never touched; see [`generation_stem`]. Returns how many files were
/// deleted. Best-effort per file: the first failure is returned after every
/// candidate was attempted.
pub(super) async fn gc_older_generations(
    bridge: &dyn ObjectStoreBridge,
    data_prefix: &Path,
    keep_current: &HashSet<String>,
    keep_previous: &HashSet<String>,
    current_stem: &str,
) -> Result<usize> {
    let current_generation =
        generation_of(current_stem).ok_or_else(|| anyhow!("bad stem {current_stem}"))?;
    let store = bridge.inner_store();
    let mut deleted = 0;
    let mut first_err = None;
    for key in bridge.list_objects(data_prefix).await? {
        let key_str = key.as_ref();
        if !key_str.ends_with(".parquet")
            || keep_current.contains(key_str)
            || keep_previous.contains(key_str)
        {
            continue;
        }
        let name = key.filename().unwrap_or_default();
        let older = generation_of(name).is_none_or(|g| g < current_generation);
        if !older {
            continue;
        }
        match object_store::ObjectStoreExt::delete(store.as_ref(), &bridge.object_store_path(&key))
            .await
        {
            Ok(()) => deleted += 1,
            Err(e) => {
                first_err.get_or_insert(anyhow!("materialize gc: delete {key_str}: {e}"));
            }
        }
    }
    first_err.map_or(Ok(deleted), Err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_stems_order_by_snapshot_and_parse_back() {
        let a = generation_stem(1_700_000_000_000);
        let b = generation_stem(1_700_000_000_001);
        assert!(a < b, "{a} < {b}");
        assert_ne!(generation_stem(5), generation_stem(5), "nonce");
        assert_eq!(
            generation_of(&format!("{a}-00000.parquet")),
            Some("g00000001700000000000")
        );
        assert_eq!(generation_of("part-0.parquet"), None);
        assert_eq!(generation_of("gx000000000000000000-0.parquet"), None);
    }

    /// A sort over more data than its memory budget completes by spilling (an
    /// in-memory sort would fail with resources exhausted) and orders NULLs last.
    #[cfg(feature = "datafusion-integration")]
    #[tokio::test]
    async fn the_sort_spills_past_its_budget_and_orders_nulls_last() {
        use arrow_array::{Int64Array, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use futures::{StreamExt, TryStreamExt};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("pad", DataType::Utf8, false),
        ]));
        let batches: Vec<Result<RecordBatch>> = (0..64)
            .map(|b| {
                let keys: Vec<Option<i64>> = (0..4096)
                    .map(|i| {
                        let v = (i * 7919 + b * 104_729) % 1_000_003;
                        (v % 97 != 0).then_some(v)
                    })
                    .collect();
                let pad: Vec<String> = (0..4096).map(|i| format!("{i:0>64}")).collect();
                Ok(RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int64Array::from(keys)),
                        Arc::new(StringArray::from(pad)),
                    ],
                )
                .unwrap())
            })
            .collect();
        // ~18 MB of input against an 8 MB budget.
        let sorted: Vec<RecordBatch> = sort_batches_with_budget(
            futures::stream::iter(batches).boxed(),
            schema,
            "k",
            8 * 1024 * 1024,
        )
        .unwrap()
        .try_collect()
        .await
        .expect("sort spills instead of failing");
        let keys: Vec<Option<i64>> = sorted
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .iter()
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(keys.len(), 64 * 4096);
        let first_null = keys
            .iter()
            .position(Option::is_none)
            .expect("nulls present");
        assert!(keys[first_null..].iter().all(Option::is_none), "nulls last");
        assert!(
            keys[..first_null].windows(2).all(|w| w[0] <= w[1]),
            "ascending"
        );
    }

    /// The old generation and the pre-generation file go; the previous
    /// version's files, the current ones and a NEWER generation stay.
    #[tokio::test]
    async fn gc_keeps_current_previous_and_newer_generations() {
        use proximadb_iceberg_engine::IcebergObjectStoreBridge;
        let bridge = IcebergObjectStoreBridge::from_url("memory:///gc-test").unwrap();
        let store = bridge.inner_store();
        let old = generation_stem(1);
        let prev = generation_stem(2);
        let cur = generation_stem(3);
        let newer = generation_stem(4);
        let names = [
            "part-0.parquet".to_string(),
            format!("{old}-00000.parquet"),
            format!("{prev}-00000.parquet"),
            format!("{cur}-00000.parquet"),
            format!("{cur}-00001.parquet"),
            format!("{newer}-00000.parquet"),
            "_SUCCESS".to_string(),
        ];
        for name in &names {
            object_store::ObjectStoreExt::put(
                store.as_ref(),
                &bridge.object_store_path(&Path::from(format!("t/data/{name}"))),
                bytes::Bytes::from_static(b"x").into(),
            )
            .await
            .unwrap();
        }
        let key = |n: &str| format!("t/data/{n}");
        let current: HashSet<String> = [
            key(&format!("{cur}-00000.parquet")),
            key(&format!("{cur}-00001.parquet")),
        ]
        .into();
        let previous: HashSet<String> = [key(&format!("{prev}-00000.parquet"))].into();
        let deleted =
            gc_older_generations(&bridge, &Path::from("t/data"), &current, &previous, &cur)
                .await
                .unwrap();
        assert_eq!(deleted, 2);
        let mut left: Vec<String> = bridge
            .list_objects(&Path::from("t/data"))
            .await
            .unwrap()
            .into_iter()
            .map(|p| p.filename().unwrap().to_string())
            .collect();
        left.sort();
        let mut expected = vec![
            format!("{prev}-00000.parquet"),
            format!("{cur}-00000.parquet"),
            format!("{cur}-00001.parquet"),
            format!("{newer}-00000.parquet"),
            "_SUCCESS".to_string(),
        ];
        expected.sort();
        assert_eq!(left, expected);
    }
}
