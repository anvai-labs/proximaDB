// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! # Native vectorized execution hook (ADR-054 Phase 0/1)
//!
//! Internal to the `Native` backend — NOT a separate `ComputeBackend` variant.
//! When the native engine receives an analytical query (relational + not
//! Parquet-backed), it checks `native_vectorized_enabled()` and, if set, tries
//! the vectorized pipeline (Phase 1+) before falling back to the Volcano.
//!
//! Phase 1: the Arrow→RelationalRow conversion (the output bridge) is
//! implemented here — self-contained, zero DataFusion dependency. This lets
//! the native vectorized engine produce `ExecutionPipelineResult` from Arrow
//! `RecordBatch`es.
//!
//! Phase 2+: FilterProjectOperator + pipeline execution + LogicalNode lowering.

use crate::query::execution::engine::{
    ExecutionError, ExecutionPipelineResult, ExecutionStreamResult,
};
use proximadb_data_model::ProximaValue;
use proximadb_relational_planner::PhysicalPlan;
use proximadb_relational_types::{ColumnInfo, RelationalRow, RelationalSchema};
use std::sync::Arc;

/// Gate: is the native vectorized execution path opted in? Default OFF — the
/// Volcano (row-at-a-time) serves all native-path queries until the vectorized
/// path is promoted (ledger-gated, per ADR-054 §8).
pub fn native_vectorized_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("PROXIMADB_NATIVE_VECTORIZED")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Gate: should the native hash-join path be used? Default OFF — distinct from
/// `PROXIMADB_NATIVE_VECTORIZED` so FilterProject/HashAgg can serve without
/// forcing the join on. When ON, `lower_physical::Join` wires the #779
/// `HashJoinBuildOperator`/`HashJoinProbeOperator` (ADR-054 Phase 3, TD-OLAP-11).
pub fn native_join_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("PROXIMADB_NATIVE_JOIN")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Gate: parallelize native pipelines with the TD-OLAP-12 morsel scheduler?
/// Default OFF, distinct from `PROXIMADB_NATIVE_VECTORIZED`/`_JOIN`. When on, a
/// splittable source (parquet row-group lanes) is decoded across cores and fanned
/// into the serial downstream operators; a non-splittable source runs serially, so
/// this is additive and never a correctness dependency.
pub fn native_morsel_scheduler_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("PROXIMADB_NATIVE_MORSEL")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Gate: run the native-parquet SHADOW probe alongside DataFusion on the OLAP
/// path? Default OFF — this is a benchmark/measurement instrument, not a product
/// path: when on, a parquet SELECT that DataFusion serves is ALSO re-planned,
/// re-opened, and re-executed on the native vectorized engine purely to record a
/// `native-vectorized` compute sample for the engine-dimension trace (TD-OLAP-4).
/// It roughly doubles the query's work, so it must never be on in production.
pub fn native_shadow_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("PROXIMADB_NATIVE_SHADOW")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Gate: route the operation classes native measurably wins to the native
/// vectorized engine as the PRIMARY backend over external parquet (returning its
/// result), with DataFusion as the correctness floor. Default OFF (TD-OLAP-4
/// "favor native by operation"). Distinct from [`native_shadow_enabled`] (which
/// runs native ALONGSIDE DataFusion and discards the result): when this is on, a
/// footer-elidable / scalar-aggregate unfiltered parquet SELECT is served by
/// native and DataFusion is only consulted when native declines the shape.
/// Correctness is preserved because native's result MUST equal DataFusion's for
/// the routed shapes (guarded by the eligibility gates in `try_native_over_parquet`
/// and the native-vs-DataFusion ratchet tests); any decline falls through to
/// DataFusion.
pub fn native_route_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("PROXIMADB_NATIVE_ROUTE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Try the native vectorized execution path for an already-planned query.
/// Called from `NativeVolcanoEngine::execute_physical` (the single native
/// chokepoint, above the Volcano) when [`native_vectorized_enabled`] is set.
///
/// Returns `Ok(Some(result))` if the vectorized engine handled the query.
/// Returns `Ok(None)` when the path is disabled OR the plan shape / constructs
/// are unsupported (Phase 2: only `Project`/`Filter`/`Limit` over `Values`) OR
/// execution hit an issue — in every such case the caller falls back to the
/// Volcano. The experimental, default-off path never fails a query; correctness
/// is policed by the native-vs-Volcano shadow-comparison harness
/// ([`super::native_shadow`], ADR-054 §7 Phase 0.5), which auto-demotes a query
/// shape after a divergence — demoted shapes decline here.
pub async fn try_vectorized(
    physical: &PhysicalPlan,
    scan_ctx: Option<&super::native_ops::ScanCtx>,
) -> Result<Option<ExecutionPipelineResult>, ExecutionError> {
    if !native_vectorized_enabled() {
        return Ok(None);
    }
    // A prior shadow run may have demoted this query shape after observing a
    // native-vs-Volcano divergence (ADR-054 §7 Phase 0.5). Skip native for it.
    if super::native_shadow::is_shape_demoted(physical) {
        return Ok(None);
    }
    // ADR-058 D1: consult the declared capability descriptor. A `Partial`
    // verdict GUARANTEES the lowering declines (sound one-way filter — every
    // reason mirrors a real decline in `walk`/`lower_join`), so decline here
    // with structured reasons and skip the lowering attempt for clearly-
    // unsupported shapes. `Full` does NOT guarantee success (deeper constraints
    // are checked during lowering); fall through and let lower_physical decide.
    if let super::native_ops::SupportLevel::Partial(reasons) =
        super::native_ops::vectorized_supports(physical)
    {
        let labels: Vec<&'static str> = reasons.iter().map(|r| r.as_label()).collect();
        tracing::debug!(
            target: "proximadb::native_vectorized",
            reasons = ?labels,
            "vectorized path declined (unsupported plan shape); falling back to Volcano"
        );
        return Ok(None);
    }
    // Any decline (unsupported shape) or failure (execution error) → Volcano.
    let lowered = match super::native_ops::lower_physical(physical, scan_ctx) {
        Ok(l) => l,
        Err(reason) => {
            tracing::debug!(
                target: "proximadb::native_vectorized",
                %reason,
                "vectorized path declined; falling back to Volcano"
            );
            return Ok(None);
        }
    };
    // TD-OLAP-4 Slice 0 (engine dimension): time + label the native-vectorized
    // engine DISTINCTLY from the Volcano and DataFusion, so the io-trace carries
    // a per-engine compute sample the route cost model can compare on.
    let started = std::time::Instant::now();
    let batches = match super::native_ops::execute_pipeline(&lowered).await {
        Ok(b) => b,
        Err(reason) => {
            tracing::debug!(
                target: "proximadb::native_vectorized",
                %reason,
                "vectorized path failed mid-execution; falling back to Volcano"
            );
            return Ok(None);
        }
    };
    crate::observability::io_trace::record_compute_ms(
        "native-vectorized",
        started.elapsed().as_millis() as u64,
    );
    crate::observability::io_trace::record_route("vectorized", "NativeVectorized");
    let schema = lowered.pipeline.output_schema.as_ref();
    Ok(Some(record_batches_to_pipeline_result(schema, &batches)))
}

/// TD-OLAP-4 (engine dimension, native-parquet): run `physical` on the native
/// vectorized engine using `scan_source` — a
/// [`super::native_parquet_scan::ParquetScanOperator`] built from the SAME
/// object-store + files + projection the DataFusion adapter reads — as the `Scan`
/// leaf. This is the shadow entry that lets native serve external-parquet queries
/// so the io-trace carries a `native-vectorized` compute sample alongside
/// DataFusion's for the SAME plan and storage.
///
/// Returns `Ok(None)` when the path is disabled OR the plan shape is unsupported
/// (native covers `Filter`/`Project`/`Aggregate`/`Limit` over a `Scan` today) OR
/// execution failed — the caller keeps DataFusion's result in every such case;
/// correctness is policed by the shadow-comparison harness.
pub async fn try_vectorized_over_parquet(
    physical: &PhysicalPlan,
    scan_source: Box<dyn proximadb_execution_contracts::ExecutionOperator>,
) -> Result<Option<ExecutionPipelineResult>, ExecutionError> {
    if !native_vectorized_enabled() {
        return Ok(None);
    }
    run_native_over_parquet(physical, scan_source).await
}

/// Gate-free core of the native-parquet path: lower `physical` over `scan_source`,
/// execute, and record the `native-vectorized` compute + route sample. Shared by
/// [`try_vectorized_over_parquet`] (product-gated) and the shadow probe (which
/// gates on [`native_shadow_enabled`] instead, so a measurement run needs a single
/// env switch). Returns `Ok(None)` on decline (unsupported shape) or failure.
pub(crate) async fn run_native_over_parquet(
    physical: &PhysicalPlan,
    scan_source: Box<dyn proximadb_execution_contracts::ExecutionOperator>,
) -> Result<Option<ExecutionPipelineResult>, ExecutionError> {
    let lowered = match super::native_ops::lower_physical_over_source(physical, scan_source) {
        Ok(l) => l,
        Err(reason) => {
            tracing::debug!(
                target: "proximadb::native_vectorized",
                %reason,
                "native-parquet path declined; keeping DataFusion result"
            );
            return Ok(None);
        }
    };
    let started = std::time::Instant::now();
    // TD-OLAP-12: when the morsel scheduler is enabled AND the shape is a single
    // (non-join, non-limit) pipeline, decode the source's row-groups across cores
    // and fan into the serial downstream operators. Otherwise run serially.
    let use_morsel = native_morsel_scheduler_enabled()
        && lowered.build_pipeline.is_none()
        && lowered.limit.is_none();
    let exec_result = if use_morsel {
        use futures::StreamExt;
        use proximadb_execution_contracts::{BatchStream, MorselScheduler};
        let empty: BatchStream = Box::pin(futures::stream::empty());
        let sched = super::morsel_scheduler::TokioMorselScheduler::new();
        match sched.schedule(&lowered.pipeline, empty).await {
            Ok(mut stream) => {
                let mut out = Vec::new();
                let mut err = None;
                while let Some(b) = stream.next().await {
                    match b {
                        Ok(b) => out.push(b),
                        Err(e) => {
                            err = Some(e);
                            break;
                        }
                    }
                }
                match err {
                    Some(e) => Err(e),
                    None => Ok(out),
                }
            }
            Err(e) => Err(e),
        }
    } else {
        super::native_ops::execute_pipeline(&lowered).await
    };
    let batches = match exec_result {
        Ok(b) => b,
        Err(reason) => {
            tracing::debug!(
                target: "proximadb::native_vectorized",
                %reason,
                "native-parquet path failed mid-execution; keeping DataFusion result"
            );
            return Ok(None);
        }
    };
    crate::observability::io_trace::record_compute_ms(
        "native-vectorized",
        started.elapsed().as_millis() as u64,
    );
    crate::observability::io_trace::record_route("vectorized", "NativeVectorized");
    let schema = lowered.pipeline.output_schema.as_ref();
    Ok(Some(record_batches_to_pipeline_result(schema, &batches)))
}

/// Metadata-elision run path (TD-OLAP-4): execute a single source operator that
/// already emits the final result row (footer `MIN`/`MAX`/`COUNT`), recording the
/// `native-vectorized` compute + route sample like the scan path. No plan lowering
/// — the aggregate is elided, so there is no scan and no HashAggregate.
pub(crate) async fn run_native_source_only(
    source: Box<dyn proximadb_execution_contracts::ExecutionOperator>,
) -> Result<Option<ExecutionPipelineResult>, ExecutionError> {
    let output_schema = source.output_schema();
    let started = std::time::Instant::now();
    let batches = match super::native_ops::execute_source(source).await {
        Ok(b) => b,
        Err(reason) => {
            tracing::debug!(
                target: "proximadb::native_vectorized",
                %reason,
                "native metadata-elision failed; keeping DataFusion result"
            );
            return Ok(None);
        }
    };
    crate::observability::io_trace::record_compute_ms(
        "native-vectorized",
        started.elapsed().as_millis() as u64,
    );
    crate::observability::io_trace::record_route("vectorized", "NativeVectorized");
    Ok(Some(record_batches_to_pipeline_result(
        output_schema.as_ref(),
        &batches,
    )))
}

/// Streaming variant — not yet wired (the materialized path serves Phase 2).
pub async fn try_vectorized_stream(
    _physical: &PhysicalPlan,
) -> Result<Option<ExecutionStreamResult>, ExecutionError> {
    Ok(None)
}

// ---------------------------------------------------------------------------
// Production ScanCtx construction (Phase 2.5 production wire)
// ---------------------------------------------------------------------------

/// Process-global lazy-init FilesystemFactory for the native engine's PAX scan.
/// Initialized on first use with the default config (local file). Acceptable for
/// MVP — the native engine is default-OFF; when opt-in, the factory initializes
/// once. Production hardening: thread the real config from `database.rs`.
static NATIVE_FS: std::sync::OnceLock<
    Arc<crate::storage::persistence::filesystem::FilesystemFactory>,
> = std::sync::OnceLock::new();

/// Get the process-global FilesystemFactory (lazy-init). Returns `None` on init
/// failure — the caller falls back to the Volcano (never fails a query).
async fn native_filesystem()
-> Option<&'static Arc<crate::storage::persistence::filesystem::FilesystemFactory>> {
    if let Some(fs) = NATIVE_FS.get() {
        return Some(fs);
    }
    let config = crate::storage::persistence::filesystem::FilesystemConfig::default();
    match crate::storage::persistence::filesystem::FilesystemFactory::create(config).await {
        Ok(factory) => {
            let arc = Arc::new(factory);
            let _ = NATIVE_FS.set(arc);
            NATIVE_FS.get()
        }
        Err(e) => {
            tracing::warn!(target: "proximadb::native_vectorized", error = %e, "native FS init failed");
            None
        }
    }
}

/// Discover `.pax` segment files under `base_path` and return one `FileSplit` per
/// file. Inlined (not imported from `src/datafusion/`) to avoid feature-gate
/// coupling — the native engine compiles without `datafusion-integration`.
async fn discover_native_pax_segments(
    base_path: &str,
    fs: &crate::storage::persistence::filesystem::FilesystemFactory,
) -> Vec<crate::storage::formats::FileSplit> {
    use crate::storage::persistence::filesystem::FilesystemError;
    let entries = match fs.list(base_path).await {
        Ok(e) => e,
        Err(FilesystemError::NotFound(_)) => return Vec::new(),
        Err(FilesystemError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
            return Vec::new();
        }
        Err(e) => {
            tracing::debug!(target: "proximadb::native_vectorized", base_path, error = %e, "PAX discovery failed");
            return Vec::new();
        }
    };
    entries
        .iter()
        .filter(|e| e.name.ends_with(".pax"))
        .map(|e| {
            // TD-USUB-8: carry the object size the listing already returned, so
            // a reader need not pay a `HEAD` to rediscover it.
            crate::storage::formats::FileSplit::new_block(
                format!("{base_path}/{}", e.name),
                0,
                0,
                e.metadata.size,
                0,
            )
            .with_object_size(e.metadata.size)
        })
        .collect()
}

/// Recursively collect all `PhysicalPlan::Scan` table names from a plan tree.
fn collect_scan_tables(
    plan: &PhysicalPlan,
) -> Vec<(String, &proximadb_relational_types::RelationalSchema)> {
    fn collect<'a>(
        plan: &'a PhysicalPlan,
        out: &mut Vec<(String, &'a proximadb_relational_types::RelationalSchema)>,
    ) {
        match plan {
            PhysicalPlan::Scan {
                table,
                output_schema,
                ..
            } => {
                out.push((table.name.clone(), output_schema));
            }
            PhysicalPlan::Filter { input, .. } | PhysicalPlan::Project { input, .. } => {
                collect(input, out);
            }
            PhysicalPlan::Join { left, right, .. } => {
                collect(left, out);
                collect(right, out);
            }
            PhysicalPlan::Limit { input, .. } => collect(input, out),
            _ => {}
        }
    }
    let mut out = Vec::new();
    collect(plan, &mut out);
    out
}

/// Build a `ScanCtx` for the native engine from a PhysicalPlan + the lazy global
/// FilesystemFactory. Scans the plan for `Scan` nodes, resolves each table to a
/// PAX base_path, discovers segments, and constructs the per-table column mapping.
/// Returns `None` on ANY error → Volcano fallback (never fails a query).
pub(crate) async fn build_scan_ctx(physical: &PhysicalPlan) -> Option<super::native_ops::ScanCtx> {
    use super::native_ops::{ScanCtx, ScanTableInfo};
    use std::collections::HashMap;

    // Only build a ScanCtx if the plan actually has Scan nodes.
    let scan_tables = collect_scan_tables(physical);
    if scan_tables.is_empty() {
        return None; // Values-only plan → no ScanCtx needed
    }

    let fs = native_filesystem().await?;

    // Resolve the data_dir from env (matches the server default).
    let data_dir = std::env::var("PROXIMADB_DATA_DIR").unwrap_or_else(|_| "./data".to_string());

    let mut tables = HashMap::new();
    for (table_name, output_schema) in &scan_tables {
        // MVP base_path: {data_dir}/collections/{table_name}
        let base_path = format!("{data_dir}/collections/{table_name}");
        let splits = discover_native_pax_segments(&base_path, fs).await;
        if splits.is_empty() {
            tracing::debug!(
                target: "proximadb::native_vectorized",
                table_name, base_path,
                "no PAX segments found; table will decline to Volcano"
            );
            // No PAX data → skip this table (the Scan arm will Err → Volcano).
            continue;
        }
        // MVP column mapping: ordinal-based (column ordinal → ordinal as i32).
        // The PAX format stores core columns at fixed col_id constants; for user
        // columns, the ordinal is the column_id. This works for the core fields
        // (created_at, etc.) which are at fixed positions.
        let name_to_col_id: HashMap<String, i32> = output_schema
            .columns
            .iter()
            .enumerate()
            .map(|(i, c)| (c.name.clone(), i as i32))
            .collect();
        tables.insert(
            table_name.clone(),
            ScanTableInfo {
                splits,
                name_to_col_id,
            },
        );
    }

    if tables.is_empty() {
        None // No PAX data for any table → Volcano
    } else {
        Some(ScanCtx {
            filesystem_factory: Arc::clone(fs),
            tables,
        })
    }
}

// ---------------------------------------------------------------------------
// Arrow → RelationalRow conversion (Phase 1 — the output bridge)
// ---------------------------------------------------------------------------

/// Convert Arrow RecordBatches → ExecutionPipelineResult.
/// Self-contained (zero DataFusion dependency). Mirrors the DataFusion adapter's
/// private `record_batches_to_pipeline_result` but lives here so the native
/// engine doesn't depend on the adapter. Consumed by `try_vectorized`.
pub(crate) fn record_batches_to_pipeline_result(
    arrow_schema: &arrow::datatypes::Schema,
    batches: &[arrow_array::RecordBatch],
) -> ExecutionPipelineResult {
    let schema = arrow_schema_to_relational(arrow_schema);
    let mut rows: Vec<RelationalRow> = Vec::new();
    for batch in batches {
        rows.extend(record_batch_to_rows(batch));
    }
    ExecutionPipelineResult { schema, rows }
}

fn arrow_schema_to_relational(schema: &arrow::datatypes::Schema) -> RelationalSchema {
    let columns: Vec<ColumnInfo> = schema
        .fields()
        .iter()
        .map(|f| {
            ColumnInfo::new(
                f.name().clone(),
                arrow_type_to_proxima(f.data_type()),
                f.is_nullable(),
            )
        })
        .collect();
    RelationalSchema::new(columns)
}

fn arrow_type_to_proxima(dt: &arrow::datatypes::DataType) -> proximadb_data_model::ProximaType {
    use arrow::datatypes::DataType as D;
    use proximadb_data_model::ProximaType;
    match dt {
        D::Boolean => ProximaType::Boolean,
        D::Int8 => ProximaType::Int8,
        D::Int16 => ProximaType::Int16,
        D::Int32 => ProximaType::Int32,
        D::Int64 => ProximaType::Int64,
        D::UInt8 => ProximaType::UInt8,
        D::UInt16 => ProximaType::UInt16,
        D::UInt32 => ProximaType::UInt32,
        D::UInt64 => ProximaType::UInt64,
        D::Float32 => ProximaType::Float32,
        D::Float64 => ProximaType::Float64,
        D::Utf8 | D::LargeUtf8 => ProximaType::String,
        D::Binary | D::LargeBinary => ProximaType::Binary,
        D::Date32 | D::Date64 => ProximaType::Date,
        _ => ProximaType::String,
    }
}

fn record_batch_to_rows(batch: &arrow_array::RecordBatch) -> Vec<RelationalRow> {
    let ncols = batch.num_columns();
    let mut rows: Vec<RelationalRow> = Vec::with_capacity(batch.num_rows());
    for r in 0..batch.num_rows() {
        let mut row: RelationalRow = Vec::with_capacity(ncols);
        for c in 0..ncols {
            row.push(arrow_cell_to_proxima(batch.column(c).as_ref(), r));
        }
        rows.push(row);
    }
    rows
}

pub(crate) fn arrow_cell_to_proxima(array: &dyn arrow_array::Array, row: usize) -> ProximaValue {
    use arrow::datatypes::DataType as D;
    use proximadb_data_model::{ProximaValue, TimeUnit};

    if array.is_null(row) {
        return ProximaValue::Null;
    }
    macro_rules! v {
        ($t:ty, $ctor:path) => {
            array
                .as_any()
                .downcast_ref::<$t>()
                .map(|a| $ctor(a.value(row)))
        };
    }
    match array.data_type() {
        D::Boolean => v!(arrow_array::BooleanArray, ProximaValue::Boolean),
        D::Int8 => v!(arrow_array::Int8Array, ProximaValue::Int8),
        D::Int16 => v!(arrow_array::Int16Array, ProximaValue::Int16),
        D::Int32 => v!(arrow_array::Int32Array, ProximaValue::Int32),
        D::Int64 => v!(arrow_array::Int64Array, ProximaValue::Int64),
        D::UInt8 => v!(arrow_array::UInt8Array, ProximaValue::UInt8),
        D::UInt16 => v!(arrow_array::UInt16Array, ProximaValue::UInt16),
        D::UInt32 => v!(arrow_array::UInt32Array, ProximaValue::UInt32),
        D::UInt64 => v!(arrow_array::UInt64Array, ProximaValue::UInt64),
        D::Float32 => v!(arrow_array::Float32Array, ProximaValue::Float32),
        D::Float64 => v!(arrow_array::Float64Array, ProximaValue::Float64),
        D::Utf8 => array
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .map(|a| ProximaValue::String(a.value(row).to_string())),
        D::LargeUtf8 => array
            .as_any()
            .downcast_ref::<arrow_array::LargeStringArray>()
            .map(|a| ProximaValue::String(a.value(row).to_string())),
        D::Binary => array
            .as_any()
            .downcast_ref::<arrow_array::BinaryArray>()
            .map(|a| ProximaValue::Binary(a.value(row).to_vec())),
        D::LargeBinary => array
            .as_any()
            .downcast_ref::<arrow_array::LargeBinaryArray>()
            .map(|a| ProximaValue::Binary(a.value(row).to_vec())),
        D::Date32 => v!(arrow_array::Date32Array, ProximaValue::Date),
        D::Date64 => array
            .as_any()
            .downcast_ref::<arrow_array::Date64Array>()
            .map(|a| ProximaValue::Timestamp(a.value(row), TimeUnit::Millisecond)),
        D::Timestamp(unit, _) => match unit {
            arrow::datatypes::TimeUnit::Second => array
                .as_any()
                .downcast_ref::<arrow_array::TimestampSecondArray>()
                .map(|a| ProximaValue::Timestamp(a.value(row), TimeUnit::Second)),
            arrow::datatypes::TimeUnit::Millisecond => array
                .as_any()
                .downcast_ref::<arrow_array::TimestampMillisecondArray>()
                .map(|a| ProximaValue::Timestamp(a.value(row), TimeUnit::Millisecond)),
            arrow::datatypes::TimeUnit::Microsecond => array
                .as_any()
                .downcast_ref::<arrow_array::TimestampMicrosecondArray>()
                .map(|a| ProximaValue::Timestamp(a.value(row), TimeUnit::Microsecond)),
            arrow::datatypes::TimeUnit::Nanosecond => array
                .as_any()
                .downcast_ref::<arrow_array::TimestampNanosecondArray>()
                .map(|a| ProximaValue::Timestamp(a.value(row), TimeUnit::Nanosecond)),
        },
        _ => None,
    }
    .unwrap_or_else(|| {
        match arrow::util::display::ArrayFormatter::try_new(
            array,
            &arrow::util::display::FormatOptions::default(),
        ) {
            Ok(f) => ProximaValue::String(f.value(row).to_string()),
            Err(_) => ProximaValue::Null,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow_array::{Int64Array, RecordBatch};
    use std::sync::Arc;

    #[test]
    fn convert_int64_batch_to_pipeline_result() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(arrow_array::StringArray::from(vec![
                    Some("a"),
                    None,
                    Some("c"),
                ])),
            ],
        )
        .unwrap();

        let result = record_batches_to_pipeline_result(&schema, &[batch]);
        assert_eq!(result.rows.len(), 3);
        assert!(matches!(result.rows[0][0], ProximaValue::Int64(1)));
        assert!(matches!(result.rows[0][1], ProximaValue::String(ref s) if s == "a"));
        assert!(matches!(result.rows[1][0], ProximaValue::Int64(2)));
        assert!(matches!(result.rows[1][1], ProximaValue::Null));
        assert!(matches!(result.rows[2][0], ProximaValue::Int64(3)));
        assert!(matches!(result.rows[2][1], ProximaValue::String(ref s) if s == "c"));
        assert_eq!(result.schema.columns.len(), 2);
        assert_eq!(result.schema.columns[0].name, "k");
        assert_eq!(result.schema.columns[1].name, "v");
    }

    #[test]
    fn convert_empty_batch() {
        let schema = Arc::new(Schema::empty());
        let batch = RecordBatch::new_empty(schema.clone());
        let result = record_batches_to_pipeline_result(&schema, &[batch]);
        assert!(result.rows.is_empty());
    }
}
