// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! # DuckDB-Local OLAP engine (ADR-059)
//!
//! In-process DuckDB behind the `ExecutionEngine` seam, mirroring
//! [`DataFusionLocalEngine`]. Reads the **SAME materialized parquet** ProximaDB
//! wrote (via DuckDB's `read_parquet`), executes the SQL, and returns an
//! [`ExecutionPipelineResult`] built from DuckDB's Arrow output through the same
//! Arrow→`RelationalRow` bridge DataFusion uses (`record_batches_to_pipeline_result`).
//!
//! Used by the perf-ledger harnesses (`tpc_perf_ledger_e2e.rs`,
//! `clickbench_ledger_e2e.rs`) for an **in-process, io-traced** DuckDB baseline,
//! so the DataFusion-vs-DuckDB discriminant is engine behavior only (same data,
//! same SQL, same process) — not the out-of-process CLI subprocess (which carried
//! no io-trace and used a separate data-load path).
//!
//! io-trace is **engine-reported** (ADR-059 / decider-accepted): DuckDB's parquet
//! reader is internal C++ with no `ObjectStore` hook, so per-GET
//! `range_gets`/`splits_pruned` aren't available; `compute_ms` (wall) is recorded.
//! (`bytes_read` from DuckDB's profiling API is a follow-up refinement.)

use super::engine::{
    ExecutionEngine, ExecutionError, ExecutionPipelineResult, QueryExecutionContext,
};
use async_trait::async_trait;

/// ADR-059 rollout step 2 (production routing gate, default OFF): when
/// `PROXIMADB_DUCKDB_ROUTE` is truthy, join/agg-shaped parquet-backed SELECTs
/// are attempted on DuckDB as the PRIMARY engine (DataFusion stays the
/// correctness floor), and `DuckDbCompat` becomes a freshness-safe cost-model
/// candidate so its cells warm through the router. Mirrors
/// [`super::native_engine::native_route_enabled`]; ledger-gated promotion per
/// the ADR-054 progressive-cutover discipline.
pub fn duckdb_route_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("PROXIMADB_DUCKDB_ROUTE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// In-process DuckDB OLAP engine. Stateless — a fresh in-memory DuckDB is opened
/// per `execute_sql` (the harness runs one query at a time against the
/// materialized parquet, registered as views).
pub struct DuckDbLocalEngine;

#[async_trait]
impl ExecutionEngine for DuckDbLocalEngine {
    async fn execute_sql(
        &self,
        sql: &str,
        context: QueryExecutionContext,
    ) -> Result<ExecutionPipelineResult, ExecutionError> {
        // File selection is async object-store I/O, so resolve each view's
        // source here, before the blocking hop.
        let mut views = Vec::with_capacity(context.parquet_tables.len());
        for (name, location) in &context.parquet_tables {
            views.push((name.clone(), read_parquet_source(location).await?));
        }
        // DuckDB is synchronous + CPU-bound (CBO + vectorized joins). Run it on
        // the blocking pool so it never holds a tokio worker thread.
        let sql = sql.to_string();
        let (result, compute_ms) = tokio::task::spawn_blocking(move || run_duckdb(&sql, &views))
            .await
            .map_err(|e| ExecutionError::Execution(format!("duckdb join/task: {e}")))??;
        // Record io-trace HERE (async context), not inside spawn_blocking — the
        // task-local `IO_TRACE` is set in this async scope (the harness wraps the
        // call in an io-trace scope), but does not propagate to the blocking
        // thread. DuckDB's reader is internal C++ (no ObjectStore hook), so only
        // engine-reported compute_ms is available (bytes_read is a follow-up).
        crate::observability::io_trace::record_compute_ms("duckdb", compute_ms);
        Ok(result)
    }
}

/// The `read_parquet` argument for a table `location`: the files its published
/// Iceberg snapshot selects, chosen by the SAME helper as the DataFusion reader
/// so both engines read one generation. Only a table with no metadata log keeps
/// the directory form; unreadable metadata fails the query instead of reading
/// whatever parquet happens to be present.
async fn read_parquet_source(location: &str) -> Result<String, ExecutionError> {
    use crate::datafusion::engine_adapters::ObjectStoreParquetTable;
    let selected = ObjectStoreParquetTable::manifest_selected_uris(location)
        .await
        .map_err(|e| ExecutionError::Execution(format!("duckdb select files {location}: {e}")))?;
    Ok(match selected {
        Some(files) => {
            let quoted: Vec<String> = files
                .iter()
                .map(|f| sql_string_literal(f.as_str()))
                .collect();
            format!("[{}]", quoted.join(", "))
        }
        None => sql_string_literal(location.trim_end_matches('/')),
    })
}

/// A SQL single-quoted string literal (`'` doubled), for object paths/URIs.
fn sql_string_literal(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

/// Open an in-memory DuckDB, register the parquet tables, run the SQL, and
/// convert the Arrow result to [`ExecutionPipelineResult`]. Returns the result
/// + the wall `compute_ms` (recorded by the caller in the async io-trace scope).
fn run_duckdb(
    sql: &str,
    views: &[(String, String)],
) -> Result<(ExecutionPipelineResult, u64), ExecutionError> {
    let conn = duckdb::Connection::open_in_memory()
        .map_err(|e| ExecutionError::Execution(format!("duckdb open: {e}")))?;
    // Register each parquet table as a view over `read_parquet` — DuckDB reads
    // the SAME materialized parquet ProximaDB wrote (same data; no re-load). The
    // source is the snapshot's file list (`read_parquet_source`), or for a table
    // with no metadata log its base directory (the catalog API contract — like
    // Hive/Unity/Polaris: the reader scans immediate parquet files in it,
    // skipping commit-style files `_`/`.` prefixed; partitioned layouts use
    // `colname=value/` subdirs). NOT a recursive glob.
    for (name, source) in views {
        let ddl = format!("CREATE VIEW \"{name}\" AS SELECT * FROM read_parquet({source})");
        conn.execute_batch(&ddl)
            .map_err(|e| ExecutionError::Execution(format!("duckdb register {name}: {e}")))?;
    }
    // Prepare + execute + time. DuckDB's CBO (join reorder + broadcast) runs
    // here — the perf lever that closes the measured 100–700× gap (TD-OLAP-15).
    let started = std::time::Instant::now();
    let mut stmt = conn
        .prepare(sql)
        .map_err(|e| ExecutionError::Execution(format!("duckdb prepare: {e}")))?;
    let arrow = stmt
        .query_arrow(duckdb::params![])
        .map_err(|e| ExecutionError::Execution(format!("duckdb execute: {e}")))?;
    // Keep the result schema: an empty result yields zero batches but is still a
    // valid answer (seven TPC-DS queries return no rows at small scale factors).
    let duck_schema = arrow.get_schema();
    let duck_batches: Vec<duckdb::arrow::array::RecordBatch> = arrow.collect();
    let compute_ms = started.elapsed().as_millis() as u64;
    // Arrow → ExecutionPipelineResult (same bridge DataFusion uses). duckdb
    // vendors its own arrow 58 while the workspace builds against 59, so the
    // RecordBatch types no longer unify — the batches cross the Arrow C-data
    // FFI, whose struct layout is stable across these minors.
    let batches = duck_batches
        .iter()
        .map(ffi_batch_58_to_59)
        .collect::<Result<Vec<_>, ExecutionError>>()?;
    let schema = match batches.first() {
        Some(b) => b.schema(),
        None => ffi_schema_58_to_59(duck_schema.as_ref())?,
    };
    let result = super::native_engine::record_batches_to_pipeline_result(&schema, &batches);
    Ok((result, compute_ms))
}

/// duckdb vendors `arrow ^58` internally; the workspace builds against arrow
/// 59, so the `RecordBatch`/`FFI_*` types are distinct crates and data crosses
/// the Arrow C-data interface at the raw-struct level. `FFI_ArrowArray` and
/// `FFI_ArrowSchema` are `#[repr(C)]` with field-for-field identical
/// definitions in 58.3.0 and 59.3.0 (the C-data-interface layout is frozen),
/// so the exported structs are moved across the two crate versions verbatim;
/// `mem::transmute` compile-errors if the sizes ever diverge. The `release`
/// callback is an `extern "C"` fn owned by the producing (duckdb) side and
/// stays valid in-process; the importing side's `Drop` invokes it identically.
/// Convert a duckdb (arrow 58) schema to the workspace arrow 59 schema across the
/// stable Arrow C-data interface (see [`ffi_batch_58_to_59`]).
fn ffi_schema_58_to_59(
    schema: &duckdb::arrow::datatypes::Schema,
) -> Result<arrow_schema::SchemaRef, ExecutionError> {
    let ffi_schema = duckdb::arrow::ffi::FFI_ArrowSchema::try_from(schema)
        .map_err(|e| ExecutionError::Execution(format!("duckdb arrow schema FFI: {e}")))?;
    // SAFETY: identical `#[repr(C)]` layout in arrow 58.3 and 59.3 (C-data interface).
    let schema_59: arrow::ffi::FFI_ArrowSchema = unsafe { std::mem::transmute(ffi_schema) };
    Ok(std::sync::Arc::new(
        arrow_schema::Schema::try_from(&schema_59)
            .map_err(|e| ExecutionError::Execution(format!("duckdb arrow schema FFI: {e}")))?,
    ))
}

fn ffi_batch_58_to_59(
    batch: &duckdb::arrow::array::RecordBatch,
) -> Result<arrow_array::RecordBatch, ExecutionError> {
    let ffi_schema = duckdb::arrow::ffi::FFI_ArrowSchema::try_from(batch.schema().as_ref())
        .map_err(|e| ExecutionError::Execution(format!("duckdb arrow schema FFI: {e}")))?;
    let schema_59: arrow::ffi::FFI_ArrowSchema = unsafe { std::mem::transmute(ffi_schema) };
    let out_schema = std::sync::Arc::new(
        arrow_schema::Schema::try_from(&schema_59)
            .map_err(|e| ExecutionError::Execution(format!("duckdb arrow schema FFI: {e}")))?,
    );
    let columns = batch
        .columns()
        .iter()
        .enumerate()
        .map(|(i, col)| {
            let ffi_arr = duckdb::arrow::ffi::FFI_ArrowArray::new(&col.to_data());
            // SAFETY: the transmuted `FFI_ArrowArray` wraps buffers of the
            // live in-process `col` (duckdb's arrow) with duckdb's release
            // callback, and `schema_59.child(i)` (the same bytes, read
            // through the identically-laid-out 59 type) is its matching type
            // record; `from_ffi` copies into a workspace arrow-59 `ArrayData`
            // before the FFI wrapper drops and releases.
            let arr_59: arrow::ffi::FFI_ArrowArray = unsafe { std::mem::transmute(ffi_arr) };
            let data = unsafe { arrow::ffi::from_ffi(arr_59, schema_59.child(i)) }
                .map_err(|e| ExecutionError::Execution(format!("duckdb arrow data FFI: {e}")))?;
            Ok(arrow_array::make_array(data))
        })
        .collect::<Result<Vec<_>, ExecutionError>>()?;
    arrow_array::RecordBatch::try_new(out_schema, columns)
        .map_err(|e| ExecutionError::Execution(format!("duckdb arrow batch FFI: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datafusion::engine_adapters::object_store_parquet_reader::tests::publish_snapshot_then_add_stale_part;

    /// The view must read the snapshot's files, not the directory: a stale
    /// parquet beside the manifest-listed one would otherwise be unioned in.
    #[tokio::test]
    async fn view_reads_manifest_files_only() {
        let tmp = tempfile::tempdir().unwrap();
        let base_url = publish_snapshot_then_add_stale_part(tmp.path()).await;
        let context = QueryExecutionContext {
            parquet_tables: vec![("t".to_string(), format!("{base_url}/data"))],
            ..Default::default()
        };
        let result = DuckDbLocalEngine
            .execute_sql("SELECT x FROM t", context)
            .await
            .unwrap();
        // part-0 holds 4 rows; reading the directory would add the stale part's 4.
        assert_eq!(result.rows.len(), 4);
    }

    #[test]
    fn sql_string_literal_doubles_single_quotes() {
        assert_eq!(
            sql_string_literal("file:///w/o'brien/part-0.parquet"),
            "'file:///w/o''brien/part-0.parquet'"
        );
    }
}
