// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! # Correct `documents(collection)` PAX pushdown provider (TD-DOC-PUSHDOWN-1, final phase)
//!
//! A raw `documents()` → `PaxTableProvider` delegation is **incorrect** (mandate #16): PAX
//! segments hold FLUSHED data only, so it misses unflushed writes, and `PaxSplitReader` does
//! no dead-filtering, so it surfaces tombstoned/TTL-expired documents. This provider is the
//! storage-inclusive merge that makes a `documents()` scan correct **and** prunes the flushed
//! portion off the wire:
//!
//! 1. **Flushed, pruned** — `PaxSplitReader` (ranged, `PROXIMADB_DF_PAX_RANGED`) over the
//!    collection's `.pax` segments, with the query's WHERE predicates translated to physical
//!    filters so whole blocks are skipped (the byte win; emits the `fetch_pax_ranged` /
//!    `bytes_read` / `range_gets` io_trace). Props are read as the raw msgpack tail (col 8).
//! 2. **Unflushed delta** — `RecordRoutePort::unflushed_records` returns the raw WAL/memtable
//!    records INCLUDING tombstones and TTL-expired rows (NOT dead-filtered), so an unflushed
//!    delete can suppress a still-flushed live copy (#16d).
//! 3. **Merge** — by `oid` (= the document id, since the converged write stamps `oid = doc_id`),
//!    freshest `updated_at_ns` winning, WAL taking priority on ties (dedup-by-oid), then the
//!    canonical `is_record_dead(valid_to_ns, now_ns)` pass (#16a) drops the survivors that are
//!    tombstoned/expired — this is where a WAL tombstone suppresses the flushed live copy.
//! 4. **Freshness (#16c)** — the provider always recomputes the WAL delta and never consults the
//!    query-result cache, so it is `Strong` by construction (no cached result can leak).
//!
//! The result rows are byte-identical to the `(id, props)` MemTable path (same
//! `proxima_tree_to_json_map` transform, same reserved-key stripping, same id derivation), plus
//! the shredded promoted columns as typed columns — Option A schema
//! `(id, props, <props__key typed cols>)`. `supports_filters_pushdown = Inexact`, so DataFusion
//! keeps the residual `FilterExec` above the scan and pushdown only reduces I/O.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::{BooleanBuilder, Float64Builder, Int64Builder, StringBuilder};
use arrow_array::{Array, ArrayRef, BinaryArray, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::DFSchema;
use datafusion::datasource::{MemTable, TableProvider, TableType};
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::ExecutionPlan;
use futures::StreamExt;
use proximadb_block_format::col_id;
use proximadb_data_model::ProximaValue;
use proximadb_document::{DOCUMENT_COLLECTION_PROP, DOCUMENT_TYPE_PROP};
use proximadb_records::{ProximaRecord, ProximaTree, ProximaTreeNode, is_record_dead};
use proximadb_runtime::{PaxColumnDesc, PaxScanInputs, RecordRoutePort};
use proximadb_storage_common::proxima_arrow::arrow_cell_to_proxima_value;

use crate::datafusion::engine_adapters::pax_adapter::PaxSplitReader;
use crate::datafusion::engine_adapters::pax_segment_locator::discover_pax_segments;
use crate::datafusion::proxima_scan_exec::SplitReader;
use crate::datafusion::schema_inference::proxima_to_arrow_type;
use crate::storage::persistence::filesystem::FilesystemFactory;

/// A merged, projected document row keyed by its id, before dead-filtering.
struct MergeEntry {
    updated_at_ns: i64,
    valid_to_ns: Option<i64>,
    /// The stripped document tree (reserved keys removed); `None` for a pure tombstone /
    /// non-document record kept only to suppress a flushed live copy of the same id.
    tree: Option<ProximaTree>,
    from_wal: bool,
}

/// Storage-inclusive, PAX-pruned `documents(collection)` table provider.
pub struct DocumentPaxPushdownProvider {
    record_route: Arc<dyn RecordRoutePort>,
    filesystem_factory: Arc<FilesystemFactory>,
    /// Raw connection tenant (None ⇒ default) — passed structurally to the port.
    tenant: Option<String>,
    /// Tenant-clean collection name.
    collection: String,
    /// Object-store prefix under which the collection's `.pax` segments live.
    segments_base: String,
    /// The collection's shredded promoted columns, exposed as typed output columns.
    shredded: Vec<PaxColumnDesc>,
    /// Option-A output schema: `(id, props, <shredded typed>)`.
    output_schema: SchemaRef,
    /// Schema the PAX reader decodes: id + updated_at + valid_to + props(msgpack) + shredded,
    /// so predicates can prune on canonical AND shredded columns.
    pax_read_schema: SchemaRef,
    /// SQL/field name → PAX `column_id` for the reader (canonical + shredded).
    name_to_col_id: HashMap<String, i32>,
}

impl std::fmt::Debug for DocumentPaxPushdownProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DocumentPaxPushdownProvider")
            .field("tenant", &self.tenant)
            .field("collection", &self.collection)
            .field("segments_base", &self.segments_base)
            .field("shredded", &self.shredded.len())
            .finish_non_exhaustive()
    }
}

impl DocumentPaxPushdownProvider {
    /// Build the provider from resolved [`PaxScanInputs`]. `base_path` (from the record port,
    /// via `DrPathBuilder`) is where the collection's segments directory lives.
    pub fn new(
        record_route: Arc<dyn RecordRoutePort>,
        filesystem_factory: Arc<FilesystemFactory>,
        tenant: Option<String>,
        collection: impl Into<String>,
        inputs: PaxScanInputs,
    ) -> Self {
        let shredded = inputs.columns;

        // Option-A output schema: id, props (JSON), then each shredded promoted column typed.
        let mut output_fields = vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("props", DataType::Utf8, true),
        ];
        for c in &shredded {
            output_fields.push(Field::new(
                &c.sql_name,
                proxima_to_arrow_type(&c.data_type),
                true,
            ));
        }
        let output_schema = Arc::new(Schema::new(output_fields));

        // Reader schema: canonical columns needed for the merge (id/updated_at/valid_to) + the
        // raw msgpack props tail + the shredded columns (so a predicate on any of them prunes).
        let mut read_fields = vec![
            Field::new("id", DataType::Utf8, true),
            Field::new("updated_at", DataType::Int64, true),
            Field::new("valid_to", DataType::Int64, true),
            Field::new("props", DataType::Binary, true),
        ];
        let mut name_to_col_id = HashMap::from([
            ("id".to_string(), col_id::OID),
            ("updated_at".to_string(), col_id::UPDATED_AT),
            ("valid_to".to_string(), col_id::VALID_TO),
            ("props".to_string(), col_id::PROPS),
        ]);
        for c in &shredded {
            read_fields.push(Field::new(
                &c.sql_name,
                proxima_to_arrow_type(&c.data_type),
                true,
            ));
            name_to_col_id.insert(c.sql_name.clone(), c.col_id);
        }
        let pax_read_schema = Arc::new(Schema::new(read_fields));

        // `base_path` ends with `/` (DrPathBuilder); segments live under `<base>segments`.
        let segments_base = format!("{}segments", inputs.base_path);

        Self {
            record_route,
            filesystem_factory,
            tenant,
            collection: collection.into(),
            segments_base,
            shredded,
            output_schema,
            pax_read_schema,
            name_to_col_id,
        }
    }

    /// Read the flushed `.pax` segments through the ranged/pruned reader, translating the
    /// query's WHERE predicates to physical filters so whole blocks are skipped off the wire.
    async fn read_flushed(
        &self,
        state: &dyn Session,
        filters: &[Expr],
    ) -> DFResult<Vec<FlushedRow>> {
        let splits = discover_pax_segments(&self.segments_base, &self.filesystem_factory)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        if splits.is_empty() {
            return Ok(Vec::new());
        }

        // Translate logical WHERE predicates → physical exprs against the reader schema, so
        // `PaxSplitReader` can prune blocks on the referenced (canonical or shredded) columns.
        // A predicate we cannot lower is simply dropped (correctness-safe: the residual
        // `FilterExec` still applies it exactly; only pruning is skipped for it).
        let df_schema = DFSchema::try_from(self.pax_read_schema.as_ref().clone())?;
        let physical_filters: Vec<Arc<dyn PhysicalExpr>> = filters
            .iter()
            .filter_map(|e| state.create_physical_expr(e.clone(), &df_schema).ok())
            .collect();

        let reader = PaxSplitReader::new(
            self.pax_read_schema.clone(),
            self.filesystem_factory.clone(),
            self.name_to_col_id.clone(),
            physical_filters,
            None, // tenant_id
            None, // time_range
        );

        let id_idx = self.pax_read_schema.index_of("id")?;
        let updated_idx = self.pax_read_schema.index_of("updated_at")?;
        let valid_to_idx = self.pax_read_schema.index_of("valid_to")?;
        let props_idx = self.pax_read_schema.index_of("props")?;

        let mut rows = Vec::new();
        for split in &splits {
            let mut stream = reader.read_split(split, None, 8192).await?;
            while let Some(batch) = stream.next().await {
                let batch = batch?;
                collect_flushed_rows(
                    &batch,
                    id_idx,
                    updated_idx,
                    valid_to_idx,
                    props_idx,
                    &self.shredded,
                    &mut rows,
                )?;
            }
        }
        Ok(rows)
    }

    /// Merge flushed + unflushed rows and build the Option-A output batch.
    fn merge_to_batch(
        &self,
        flushed: Vec<FlushedRow>,
        unflushed: Vec<ProximaRecord>,
        now_ns: i64,
    ) -> DFResult<RecordBatch> {
        let mut by_id: HashMap<String, MergeEntry> = HashMap::new();

        // Flushed baseline (segment-resident). WAL overrides below on freshness.
        for row in flushed {
            consider(
                &mut by_id,
                row.id,
                MergeEntry {
                    updated_at_ns: row.updated_at_ns,
                    valid_to_ns: row.valid_to_ns,
                    tree: Some(row.tree),
                    from_wal: false,
                },
            );
        }

        // Unflushed delta (raw WAL, incl. tombstones). A document record yields its stripped
        // tree; a tombstone / non-document record is kept tree-less, keyed by oid, purely to
        // suppress the flushed live copy of the same id (#16d).
        for record in unflushed {
            let (key, tree) =
                match crate::storage::document::canonical_adapter::proxima_record_to_legacy_document(
                    &record,
                ) {
                    Some(doc) => (doc.id, Some(doc.props)),
                    None => (record.oid.clone(), None),
                };
            consider(
                &mut by_id,
                key,
                MergeEntry {
                    updated_at_ns: record.updated_at_ns,
                    valid_to_ns: record.valid_to_ns,
                    tree,
                    from_wal: true,
                },
            );
        }

        // Dead-filter the merged set (#16a) and keep only live document rows. Sort by id for a
        // deterministic scan order (the MemTable delegation still honors ORDER BY above).
        let mut live: Vec<(String, ProximaTree)> = by_id
            .into_iter()
            .filter(|(_, e)| !is_record_dead(e.valid_to_ns, now_ns))
            .filter_map(|(id, e)| e.tree.map(|t| (id, t)))
            .collect();
        live.sort_by(|a, b| a.0.cmp(&b.0));

        self.build_batch(&live)
    }

    /// Assemble the `(id, props, <shredded>)` Arrow batch from the merged live rows.
    fn build_batch(&self, rows: &[(String, ProximaTree)]) -> DFResult<RecordBatch> {
        let ids: Vec<&str> = rows.iter().map(|(id, _)| id.as_str()).collect();
        let props: Vec<String> = rows
            .iter()
            .map(|(_, tree)| {
                crate::core::search::sql_value_filter::proxima_tree_to_canonical_json_string(tree)
            })
            .collect();

        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(2 + self.shredded.len());
        arrays.push(Arc::new(StringArray::from(ids)));
        arrays.push(Arc::new(StringArray::from(props)));

        for c in &self.shredded {
            let field = self
                .output_schema
                .field_with_name(&c.sql_name)
                .map_err(|e| {
                    DataFusionError::Execution(format!(
                        "documents: output field {}: {e}",
                        c.sql_name
                    ))
                })?;
            arrays.push(build_shredded_array(field.data_type(), &c.sql_name, rows));
        }

        RecordBatch::try_new(self.output_schema.clone(), arrays)
            .map_err(|e| DataFusionError::Execution(format!("documents batch build failed: {e}")))
    }
}

#[async_trait]
impl TableProvider for DocumentPaxPushdownProvider {
    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    /// `Inexact`: WHERE predicates are pushed to the flushed PAX reader for block pruning, but
    /// the residual `FilterExec` stays above so the merged result is filtered exactly.
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let flushed = self.read_flushed(state, filters).await?;
        let unflushed = self
            .record_route
            .unflushed_records(&self.collection, self.tenant.as_deref())
            .await
            .map_err(|e| DataFusionError::Execution(format!("documents unflushed scan: {e}")))?;

        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);
        let batch = self.merge_to_batch(flushed, unflushed, now_ns)?;

        // Delegate to a MemTable so projection/filter/limit are honored uniformly (the residual
        // FilterExec guarantees exactness; the pushdown above only reduced I/O).
        let mem = MemTable::try_new(self.output_schema.clone(), vec![vec![batch]])?;
        mem.scan(state, projection, filters, limit).await
    }
}

/// One decoded flushed document row (id + merge keys + stripped tree).
struct FlushedRow {
    id: String,
    updated_at_ns: i64,
    valid_to_ns: Option<i64>,
    tree: ProximaTree,
}

/// Extract flushed document rows from one PAX reader batch (id/updated_at/valid_to/props cols).
fn collect_flushed_rows(
    batch: &RecordBatch,
    id_idx: usize,
    updated_idx: usize,
    valid_to_idx: usize,
    props_idx: usize,
    shredded: &[PaxColumnDesc],
    out: &mut Vec<FlushedRow>,
) -> DFResult<()> {
    let ids = batch
        .column(id_idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| DataFusionError::Execution("documents: id column not Utf8".into()))?;
    let updated = batch
        .column(updated_idx)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| DataFusionError::Execution("documents: updated_at not Int64".into()))?;
    let valid_to = batch
        .column(valid_to_idx)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| DataFusionError::Execution("documents: valid_to not Int64".into()))?;
    let props = batch
        .column(props_idx)
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or_else(|| DataFusionError::Execution("documents: props not Binary".into()))?;

    for i in 0..batch.num_rows() {
        if ids.is_null(i) {
            continue;
        }
        let mut tree: ProximaTree = if props.is_null(i) {
            ProximaTree::default()
        } else {
            // A row whose msgpack fails to parse yields an empty tree (defensive; never panics).
            rmp_serde::from_slice::<ProximaTree>(props.value(i)).unwrap_or_default()
        };
        // TD-USUB-6 2b: the msgpack tail may be a RESIDUAL — a shredded prop whose
        // typed stripe reproduces it exactly is no longer stored there. This
        // column is the document BODY, so a missing key is a missing field, not
        // a missing optimisation. Fill from the typed columns the batch already
        // carries.
        //
        // `or_insert`-style: the tail WINS wherever it still has the key, mirroring
        // `FlatRow::into_record`. A key still in the tail is one the stripe could
        // not reproduce, so its (lossy) stripe value must never overwrite it.
        for col in shredded {
            if tree.contains_key(&col.sql_name) {
                continue;
            }
            let Ok(idx) = batch.schema().index_of(&col.sql_name) else {
                continue;
            };
            if let Some(value) = arrow_cell_to_proxima_value(batch.column(idx), i) {
                tree.insert(col.sql_name.clone(), ProximaTreeNode::Value(value));
            }
        }

        // Match the MemTable path: the document body is props minus the reserved facade keys.
        tree.remove(DOCUMENT_COLLECTION_PROP);
        tree.remove(DOCUMENT_TYPE_PROP);
        out.push(FlushedRow {
            id: ids.value(i).to_string(),
            updated_at_ns: if updated.is_null(i) {
                0
            } else {
                updated.value(i)
            },
            valid_to_ns: if valid_to.is_null(i) {
                None
            } else {
                Some(valid_to.value(i))
            },
            tree,
        });
    }
    Ok(())
}

/// Insert `entry` for `key`, keeping the freshest version (WAL priority on `updated_at` ties).
fn consider(by_id: &mut HashMap<String, MergeEntry>, key: String, entry: MergeEntry) {
    match by_id.get(&key) {
        Some(existing)
            if existing.updated_at_ns > entry.updated_at_ns
                || (existing.updated_at_ns == entry.updated_at_ns && !entry.from_wal) => {}
        _ => {
            by_id.insert(key, entry);
        }
    }
}

/// Build one shredded output column, extracting `key` from each row's tree into the Arrow type.
fn build_shredded_array(
    data_type: &DataType,
    key: &str,
    rows: &[(String, ProximaTree)],
) -> ArrayRef {
    let value_of = |tree: &ProximaTree| -> Option<ProximaValue> {
        match tree.get(key) {
            Some(ProximaTreeNode::Value(v)) => Some(v.clone()),
            _ => None,
        }
    };
    match data_type {
        DataType::Int64 => {
            let mut b = Int64Builder::with_capacity(rows.len());
            for (_, tree) in rows {
                b.append_option(value_of(tree).and_then(proxima_value_as_i64));
            }
            Arc::new(b.finish())
        }
        DataType::Float64 => {
            let mut b = Float64Builder::with_capacity(rows.len());
            for (_, tree) in rows {
                b.append_option(value_of(tree).and_then(proxima_value_as_f64));
            }
            Arc::new(b.finish())
        }
        DataType::Boolean => {
            let mut b = BooleanBuilder::with_capacity(rows.len());
            for (_, tree) in rows {
                b.append_option(match value_of(tree) {
                    Some(ProximaValue::Boolean(v)) => Some(v),
                    _ => None,
                });
            }
            Arc::new(b.finish())
        }
        // Utf8 and any type we don't specialize: render as string (null when absent/mismatched).
        _ => {
            let mut b = StringBuilder::new();
            for (_, tree) in rows {
                b.append_option(value_of(tree).and_then(proxima_value_as_string));
            }
            Arc::new(b.finish())
        }
    }
}

fn proxima_value_as_i64(v: ProximaValue) -> Option<i64> {
    match v {
        ProximaValue::Int8(x) => Some(x as i64),
        ProximaValue::Int16(x) => Some(x as i64),
        ProximaValue::Int32(x) => Some(x as i64),
        ProximaValue::Int64(x) => Some(x),
        ProximaValue::UInt8(x) => Some(x as i64),
        ProximaValue::UInt16(x) => Some(x as i64),
        ProximaValue::UInt32(x) => Some(x as i64),
        ProximaValue::UInt64(x) => i64::try_from(x).ok(),
        _ => None,
    }
}

fn proxima_value_as_f64(v: ProximaValue) -> Option<f64> {
    match v {
        ProximaValue::Float16(x) | ProximaValue::Float32(x) => Some(x as f64),
        ProximaValue::Float64(x) => Some(x),
        ProximaValue::Int8(x) => Some(x as f64),
        ProximaValue::Int16(x) => Some(x as f64),
        ProximaValue::Int32(x) => Some(x as f64),
        ProximaValue::Int64(x) => Some(x as f64),
        _ => None,
    }
}

fn proxima_value_as_string(v: ProximaValue) -> Option<String> {
    match v {
        ProximaValue::String(s) | ProximaValue::Symbol(s) => Some(s),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::prelude::SessionContext;

    use crate::storage::document::DocumentRecord;
    use crate::storage::document::canonical_adapter::legacy_document_to_proxima_record;

    /// A record route whose flushed segments are empty (no `.pax` under `base`) and whose
    /// unflushed delta is a fixed set — exercises the storage-inclusive merge (read-after-write,
    /// dead-filter, dedup, tombstone-suppress) end-to-end through the real `scan()` + MemTable.
    struct FakeRoute {
        base: String,
        columns: Vec<PaxColumnDesc>,
        unflushed: Vec<ProximaRecord>,
    }

    #[async_trait]
    impl RecordRoutePort for FakeRoute {
        async fn insert_records(
            &self,
            _c: &str,
            _r: Vec<ProximaRecord>,
            _t: Option<&str>,
        ) -> anyhow::Result<usize> {
            Ok(0)
        }
        async fn get_record(
            &self,
            _c: &str,
            _r: &str,
            _t: Option<&str>,
        ) -> anyhow::Result<Option<ProximaRecord>> {
            Ok(None)
        }
        async fn scan_records(
            &self,
            _c: &str,
            _l: usize,
            _t: Option<&str>,
        ) -> anyhow::Result<Vec<ProximaRecord>> {
            Ok(Vec::new())
        }
        async fn delete_records(
            &self,
            _c: &str,
            _r: Vec<String>,
            _t: Option<&str>,
        ) -> anyhow::Result<usize> {
            Ok(0)
        }
        async fn collection_exists(&self, _c: &str, _t: Option<&str>) -> bool {
            true
        }
        async fn ensure_collection(
            &self,
            _c: &str,
            _d: u32,
            _t: Option<&str>,
            _p: &[String],
        ) -> anyhow::Result<()> {
            Ok(())
        }
        async fn pax_scan_inputs(&self, _c: &str, _t: Option<&str>) -> Option<PaxScanInputs> {
            Some(PaxScanInputs {
                base_path: self.base.clone(),
                columns: self.columns.clone(),
            })
        }
        async fn unflushed_records(
            &self,
            _c: &str,
            _t: Option<&str>,
        ) -> anyhow::Result<Vec<ProximaRecord>> {
            Ok(self.unflushed.clone())
        }
    }

    /// Test adapter whose unflushed half is the production memtable raw-read
    /// contract, rather than a fixed vector returned by `FakeRoute`.
    struct WalMemtableRoute {
        base: String,
        memtable: Arc<crate::storage::memtable::implementations::global_partitioned::GlobalPartitionedMemtable>,
    }

    #[async_trait]
    impl RecordRoutePort for WalMemtableRoute {
        async fn insert_records(
            &self,
            _c: &str,
            _r: Vec<ProximaRecord>,
            _t: Option<&str>,
        ) -> anyhow::Result<usize> {
            unreachable!("read-only test route")
        }
        async fn get_record(
            &self,
            _c: &str,
            _r: &str,
            _t: Option<&str>,
        ) -> anyhow::Result<Option<ProximaRecord>> {
            unreachable!("read-only test route")
        }
        async fn scan_records(
            &self,
            _c: &str,
            _l: usize,
            _t: Option<&str>,
        ) -> anyhow::Result<Vec<ProximaRecord>> {
            unreachable!("read-only test route")
        }
        async fn delete_records(
            &self,
            _c: &str,
            _r: Vec<String>,
            _t: Option<&str>,
        ) -> anyhow::Result<usize> {
            unreachable!("read-only test route")
        }
        async fn collection_exists(&self, _c: &str, _t: Option<&str>) -> bool {
            true
        }
        async fn ensure_collection(
            &self,
            _c: &str,
            _d: u32,
            _t: Option<&str>,
            _p: &[String],
        ) -> anyhow::Result<()> {
            unreachable!("read-only test route")
        }
        async fn pax_scan_inputs(&self, _c: &str, _t: Option<&str>) -> Option<PaxScanInputs> {
            Some(PaxScanInputs {
                base_path: self.base.clone(),
                columns: Vec::new(),
            })
        }
        async fn unflushed_records(
            &self,
            collection: &str,
            _t: Option<&str>,
        ) -> anyhow::Result<Vec<ProximaRecord>> {
            Ok(self.memtable.get_collection_vectors_raw(collection).await?)
        }
    }

    /// Build a document `ProximaRecord` the way the converged write path does: labeled `document`,
    /// `oid = local_id = id`, props carrying the reserved collection key + the user fields.
    fn doc(
        id: &str,
        coll: &str,
        fields: &[(&str, ProximaValue)],
        updated_at_ns: i64,
        valid_to_ns: Option<i64>,
    ) -> ProximaRecord {
        let tree: ProximaTree = fields
            .iter()
            .map(|(k, v)| (k.to_string(), ProximaTreeNode::Value(v.clone())))
            .collect();
        let dr = DocumentRecord::from_tree(id.to_string(), tree, coll.to_string(), None, None);
        let mut pr = legacy_document_to_proxima_record(&dr);
        pr.oid = id.to_string();
        pr.updated_at_ns = updated_at_ns;
        pr.valid_to_ns = valid_to_ns;
        pr
    }

    /// A bare tombstone that does NOT carry the document facade label — keyed only by oid so it
    /// can suppress a flushed live copy (#16d).
    fn tombstone(id: &str, updated_at_ns: i64) -> ProximaRecord {
        ProximaRecord {
            oid: id.to_string(),
            updated_at_ns,
            valid_to_ns: Some(0),
            ..Default::default()
        }
    }

    fn cols() -> Vec<PaxColumnDesc> {
        use proximadb_data_model::ProximaType;
        vec![
            PaxColumnDesc {
                sql_name: "region".into(),
                col_id: col_id::USER_BASE,
                data_type: ProximaType::String,
            },
            PaxColumnDesc {
                sql_name: "amount".into(),
                col_id: col_id::USER_BASE + 1,
                data_type: ProximaType::Int64,
            },
        ]
    }

    async fn provider_with(unflushed: Vec<ProximaRecord>) -> DocumentPaxPushdownProvider {
        // A base path with no `.pax` segments ⇒ the flushed read is empty (discover returns []),
        // so the scan is driven purely by the unflushed delta — the read-after-write path.
        let base = "file:///nonexistent-doc-pax-test-dir/".to_string();
        let ff = crate::storage::persistence::filesystem::FilesystemFactory::create_default_arc()
            .await
            .expect("filesystem factory");
        let route: Arc<dyn RecordRoutePort> = Arc::new(FakeRoute {
            base: "file:///nonexistent-doc-pax-test-dir/".to_string(),
            columns: cols(),
            unflushed,
        });
        DocumentPaxPushdownProvider::new(
            route,
            ff,
            None,
            "docs",
            PaxScanInputs {
                base_path: base,
                columns: cols(),
            },
        )
    }

    async fn ids(provider: DocumentPaxPushdownProvider, sql: &str) -> Vec<String> {
        let ctx = SessionContext::new();
        ctx.register_table("documents", Arc::new(provider)).unwrap();
        let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
        let mut out = Vec::new();
        for b in &batches {
            if let Some(c) = b.column(0).as_any().downcast_ref::<StringArray>() {
                for i in 0..b.num_rows() {
                    out.push(c.value(i).to_string());
                }
            }
        }
        out
    }

    /// Gate 1 — read-after-write: an inserted-but-unflushed document is visible.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unflushed_write_is_visible() {
        let p = provider_with(vec![
            doc(
                "d1",
                "docs",
                &[("region", ProximaValue::String("US".into()))],
                10,
                None,
            ),
            doc(
                "d2",
                "docs",
                &[("region", ProximaValue::String("EU".into()))],
                20,
                None,
            ),
        ])
        .await;
        assert_eq!(
            ids(p, "SELECT id FROM documents ORDER BY id").await,
            vec!["d1", "d2"]
        );
    }

    /// Gate 2 — dead-record ratchet: a TTL-expired doc and a tombstone are both absent.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dead_and_tombstoned_records_absent() {
        let p = provider_with(vec![
            doc(
                "live",
                "docs",
                &[("region", ProximaValue::String("US".into()))],
                10,
                None,
            ),
            // TTL-expired (valid_to in the past).
            doc(
                "expired",
                "docs",
                &[("region", ProximaValue::String("US".into()))],
                10,
                Some(1),
            ),
            // Tombstone (valid_to == 0).
            tombstone("deleted", 30),
        ])
        .await;
        assert_eq!(
            ids(p, "SELECT id FROM documents ORDER BY id").await,
            vec!["live"]
        );
    }

    /// Invariant 16d production-delta ratchet: a live PAX row followed by an
    /// unflushed WAL tombstone must not resurrect through `documents()`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flushed_live_plus_raw_memtable_delete_returns_zero_rows() {
        use proximadb_block_format::{BlockCompression, BlockMode};
        use proximadb_storage_common::pax_block::PaxSegmentWriter;

        let dir = tempfile::tempdir().expect("tempdir");
        let segments = dir.path().join("segments");
        std::fs::create_dir_all(&segments).expect("segments dir");
        let mut writer = PaxSegmentWriter::new(
            segments.join("data.pax"),
            BlockMode::Pax,
            BlockCompression::None,
            "docs",
            0,
            1,
            None,
        );
        writer
            .add_record(&doc(
                "d1",
                "docs",
                &[("region", ProximaValue::String("US".into()))],
                10,
                None,
            ))
            .expect("write flushed document");
        writer.finish().expect("finish PAX segment");

        let memtable = Arc::new(
            crate::storage::memtable::implementations::global_partitioned::GlobalPartitionedMemtable::new(),
        );
        memtable
            .add_wal_batch(
                "docs",
                crate::storage::memtable::specialized::wal_behavior::WALVectorBatch {
                    batch_id: crate::storage::persistence::write_ahead_log::BatchId::new(),
                    vector_records: Arc::new(vec![tombstone("d1", 20)]),
                    timestamp: std::time::SystemTime::now(),
                    total_size_bytes: 0,
                    is_flushed: false,
                    metadata_bloom_filter: None,
                },
            )
            .await
            .expect("append tombstone to memtable");

        let base = format!("file://{}/", dir.path().display());
        let route: Arc<dyn RecordRoutePort> = Arc::new(WalMemtableRoute {
            base: base.clone(),
            memtable,
        });
        let ff = crate::storage::persistence::filesystem::FilesystemFactory::create_default_arc()
            .await
            .expect("filesystem factory");
        let provider = DocumentPaxPushdownProvider::new(
            route,
            ff,
            None,
            "docs",
            PaxScanInputs {
                base_path: base,
                columns: Vec::new(),
            },
        );

        assert!(
            ids(provider, "SELECT id FROM documents").await.is_empty(),
            "an unflushed tombstone must suppress the flushed live copy"
        );
    }

    /// Gate 3 — dedup-by-oid, freshest wins: two versions of the same id collapse to the newest.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dedup_keeps_freshest_version() {
        let p = provider_with(vec![
            doc(
                "d1",
                "docs",
                &[("region", ProximaValue::String("OLD".into()))],
                10,
                None,
            ),
            doc(
                "d1",
                "docs",
                &[("region", ProximaValue::String("NEW".into()))],
                20,
                None,
            ),
        ])
        .await;
        let ctx = SessionContext::new();
        ctx.register_table("documents", Arc::new(p)).unwrap();
        let batches = ctx
            .sql("SELECT region FROM documents")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
        let region = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0);
        assert_eq!(region, "NEW");
    }

    /// Shredded promoted fields are exposed as typed columns and predicable in SQL.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shredded_columns_are_typed_and_filterable() {
        let p = provider_with(vec![
            doc(
                "d1",
                "docs",
                &[
                    ("region", ProximaValue::String("US".into())),
                    ("amount", ProximaValue::Int64(100)),
                ],
                10,
                None,
            ),
            doc(
                "d2",
                "docs",
                &[
                    ("region", ProximaValue::String("EU".into())),
                    ("amount", ProximaValue::Int64(50)),
                ],
                20,
                None,
            ),
        ])
        .await;
        // Typed predicate over shredded columns (residual FilterExec applies it exactly).
        assert_eq!(
            ids(
                p,
                "SELECT id FROM documents WHERE region = 'US' AND amount > 60 ORDER BY id"
            )
            .await,
            vec!["d1"]
        );
    }

    /// A `DocumentScanPort` serving a fixed record set through the EXACT transform the production
    /// `DocumentServiceScan` applies after `query_documents` (dead-filter, canonical document
    /// rebuild, `proxima_tree_to_json_map` → JSON string) — the MemTable-path A/B twin of
    /// `FakeRoute::unflushed_records` over the same `Vec<ProximaRecord>`.
    struct RecordSetScan {
        records: Vec<ProximaRecord>,
    }

    #[async_trait]
    impl crate::datafusion::cross_modal::DocumentScanPort for RecordSetScan {
        async fn scan(
            &self,
            _tenant: &str,
            _collection: &str,
        ) -> anyhow::Result<Vec<(String, String)>> {
            let now_ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as i64)
                .unwrap_or(0);
            Ok(self
                .records
                .iter()
                .filter(|r| !is_record_dead(r.valid_to_ns, now_ns))
                .filter_map(
                    crate::storage::document::canonical_adapter::proxima_record_to_legacy_document,
                )
                .map(|doc| {
                    (
                        doc.id,
                        crate::core::search::sql_value_filter::proxima_tree_to_canonical_json_string(
                            &doc.props,
                        ),
                    )
                })
                .collect())
        }
    }

    /// Collect the full `(id, props)` projection of a provider, in id order.
    async fn id_props(provider: Arc<dyn TableProvider>) -> Vec<(String, String)> {
        let ctx = SessionContext::new();
        ctx.register_table("documents", provider).unwrap();
        let batches = ctx
            .sql("SELECT id, props FROM documents ORDER BY id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let mut out = Vec::new();
        for b in &batches {
            let ids = b
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("id column");
            let props = b
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("props column");
            for i in 0..b.num_rows() {
                out.push((ids.value(i).to_string(), props.value(i).to_string()));
            }
        }
        out
    }

    /// A/B parity gate — the pushdown provider's `(id, props)` output is byte-identical to the
    /// existing MemTable path (`DocumentTableProvider` over a `DocumentScanPort`, ADR-055
    /// P-DFSource) on the SAME record set, running the same
    /// `SELECT id, props FROM documents ORDER BY id` through BOTH providers end-to-end. This is
    /// the literal side-by-side proof of the "results identical to the MemTable path" claim, not
    /// shared-transform inference — including agreement on what is EXCLUDED (TTL-expired,
    /// tombstoned).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pushdown_output_is_byte_identical_to_memtable_path() {
        let records = vec![
            doc(
                "d1",
                "docs",
                &[
                    ("region", ProximaValue::String("US".into())),
                    ("amount", ProximaValue::Int64(100)),
                    ("active", ProximaValue::Boolean(true)),
                ],
                10,
                None,
            ),
            doc(
                "d2",
                "docs",
                &[
                    ("region", ProximaValue::String("EU".into())),
                    ("score", ProximaValue::Float64(2.5)),
                ],
                20,
                None,
            ),
            doc(
                "d3",
                "docs",
                &[("note", ProximaValue::String("unicode ✓ \"quoted\"".into()))],
                30,
                None,
            ),
            // Both paths must also agree on exclusions: TTL-expired + tombstone.
            doc(
                "expired",
                "docs",
                &[("region", ProximaValue::String("US".into()))],
                10,
                Some(1),
            ),
            tombstone("gone", 40),
        ];

        let pushdown = provider_with(records.clone()).await;
        let memtable = crate::datafusion::cross_modal::DocumentTableProvider::new(
            Arc::new(RecordSetScan { records }),
            proximadb_tenant::DEFAULT_TENANT,
            "docs",
        );

        let a = id_props(Arc::new(pushdown)).await;
        let b = id_props(Arc::new(memtable)).await;
        assert_eq!(
            a.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
            vec!["d1", "d2", "d3"],
            "live rows only, in id order"
        );
        assert_eq!(
            a, b,
            "pushdown (id, props) must be byte-identical to the MemTable path"
        );
    }

    /// Gate 4 — the measured pushdown byte gate (ADR-052 invariant 4). A real multi-block PAX
    /// segment with a shredded `amount` column is scanned through the provider under
    /// `PROXIMADB_DF_PAX_RANGED`; a selective `amount >= N` predicate must (a) issue ranged GETs
    /// and read materially FEWER bytes than the whole segment (blocks pruned off the wire), and
    /// (b) return exactly the matching rows (the residual `FilterExec` guarantees exactness). Run
    /// under nextest so the `PROXIMADB_DF_PAX_RANGED` OnceLock is process-isolated.
    /// TD-USUB-6 2b: with the residual tail ON, a shredded prop is no longer in
    /// the msgpack `PROPS` blob — and this provider decodes that blob as the
    /// DOCUMENT BODY. Without reinstating the typed columns, `SELECT props`
    /// would silently return documents missing those fields.
    ///
    /// This is the flip precondition recorded in TD-USUB-6, exercised end to end
    /// through SQL rather than asserted at the block layer.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn residual_tail_props_projection_still_returns_the_whole_document() {
        use proximadb_block_format::{BlockCompression, BlockMode, ShredColumn};
        use proximadb_data_model::ProximaType;
        use proximadb_storage_common::pax_block::PaxSegmentWriter;

        // SAFETY (edition 2024): under nextest each test owns its process, so this
        // env mutation is single-threaded and read before any other
        // `residual_tail_enabled()` call in-process.
        unsafe { std::env::set_var("PROXIMADB_PAX_RESIDUAL_TAIL", "1") };

        let dir = tempfile::tempdir().expect("tempdir");
        let seg_dir = dir.path().join("segments");
        std::fs::create_dir_all(&seg_dir).expect("segments dir");
        let seg_path = seg_dir.join("data.pax");

        let mut writer = PaxSegmentWriter::new(
            &seg_path,
            BlockMode::Pax,
            BlockCompression::None,
            "docs",
            0,
            1,
            None,
        )
        // EXACT declared type — the only shape for which anything is dropped
        // from the tail.
        .with_shred_columns(vec![ShredColumn::typed(
            "amount",
            col_id::USER_BASE,
            ProximaType::Int64,
        )]);

        for i in 0..8usize {
            let mut props: ProximaTree = HashMap::new();
            props.insert(
                "amount".to_string(),
                ProximaTreeNode::Value(ProximaValue::Int64(i as i64 * 10)),
            );
            // A second, undeclared prop that stays in the tail either way — so the
            // test distinguishes "tail present" from "tail complete".
            props.insert(
                "note".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(format!("n{i}"))),
            );
            let ts = 1_700_000_000_000_000_000 + i as i64;
            writer
                .add_record(&ProximaRecord {
                    oid: format!("d{i}"),
                    tenant_id: "t".into(),
                    created_at_ns: ts,
                    updated_at_ns: ts,
                    valid_to_ns: Some(i64::MAX),
                    props,
                    ..Default::default()
                })
                .expect("add record");
        }
        writer.finish().expect("finish segment");

        let base = format!("file://{}/", dir.path().display());
        let ff = crate::storage::persistence::filesystem::FilesystemFactory::create_default_arc()
            .await
            .expect("filesystem factory");
        let columns = vec![PaxColumnDesc {
            sql_name: "amount".into(),
            col_id: col_id::USER_BASE,
            data_type: ProximaType::Int64,
        }];
        let route: Arc<dyn RecordRoutePort> = Arc::new(FakeRoute {
            base: base.clone(),
            columns: columns.clone(),
            unflushed: Vec::new(),
        });
        let provider = DocumentPaxPushdownProvider::new(
            route,
            ff,
            None,
            "docs",
            PaxScanInputs {
                base_path: base,
                columns,
            },
        );

        let got = id_props(Arc::new(provider)).await;
        assert_eq!(got.len(), 8, "all rows must come back");
        for (id, props) in &got {
            assert!(
                props.contains("\"amount\""),
                "document {id} lost its shredded prop under the residual tail: {props}"
            );
            assert!(
                props.contains("\"note\""),
                "document {id} lost its residual-tail prop: {props}"
            );
        }
        // Spot-check a value, not just key presence.
        let (_, first) = got.first().expect("a row");
        assert!(
            first.contains("\"amount\":0") || first.contains("\"amount\": 0"),
            "amount value must survive reconstruction: {first}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ranged_pushdown_prunes_bytes_and_matches() {
        use proximadb_block_format::{BlockCompression, BlockMode};
        use proximadb_data_model::ProximaType;
        use proximadb_records::{EmbeddingCell, EmbeddingValues};
        use proximadb_storage_common::pax_block::PaxSegmentWriter;

        // SAFETY (edition 2024): under nextest each test owns its process, so this env mutation is
        // single-threaded and read before any other `pax_ranged_read_enabled()` call in-process.
        unsafe { std::env::set_var("PROXIMADB_DF_PAX_RANGED", "1") };

        let dir = tempfile::tempdir().expect("tempdir");
        let seg_dir = dir.path().join("segments");
        std::fs::create_dir_all(&seg_dir).expect("segments dir");
        let seg_path = seg_dir.join("data.pax");

        // 600 doc-shaped records with a monotonically increasing `amount`, shredded at USER_BASE,
        // packed into many small blocks (2 KiB target) so each block owns a contiguous amount
        // range that Stage-B footer pruning can exclude.
        let n = 600usize;
        let mut writer = PaxSegmentWriter::new(
            &seg_path,
            BlockMode::Pax,
            BlockCompression::None,
            "docs",
            0,
            1,
            Some(2048),
        )
        .with_shred_spec(vec![("amount".to_string(), col_id::USER_BASE)]);
        for i in 0..n {
            let mut props: ProximaTree = HashMap::new();
            props.insert(
                "amount".to_string(),
                ProximaTreeNode::Value(ProximaValue::Int64(i as i64)),
            );
            let ts = 1_700_000_000_000_000_000 + i as i64;
            let mut r = ProximaRecord {
                oid: format!("v{i:04}"),
                tenant_id: "t".into(),
                created_at_ns: ts,
                updated_at_ns: ts,
                valid_to_ns: Some(i64::MAX),
                props,
                ..Default::default()
            };
            r.embeddings.push(EmbeddingCell {
                modality: "dense".into(),
                dim: 8,
                values: EmbeddingValues::Fp32(vec![i as f32 * 0.001; 8]),
                ..Default::default()
            });
            writer.add_record(&r).expect("add record");
        }
        let meta = writer.finish().expect("finish segment");
        assert!(
            meta.block_count >= 2,
            "need multiple blocks to prune; got {}",
            meta.block_count
        );
        let file_len = std::fs::metadata(&seg_path).expect("seg meta").len();

        let base = format!("file://{}/", dir.path().display());
        let ff = crate::storage::persistence::filesystem::FilesystemFactory::create_default_arc()
            .await
            .expect("filesystem factory");
        let columns = vec![PaxColumnDesc {
            sql_name: "amount".into(),
            col_id: col_id::USER_BASE,
            data_type: ProximaType::Int64,
        }];
        let route: Arc<dyn RecordRoutePort> = Arc::new(FakeRoute {
            base: base.clone(),
            columns: columns.clone(),
            unflushed: Vec::new(),
        });
        let provider = DocumentPaxPushdownProvider::new(
            route,
            ff,
            None,
            "docs",
            PaxScanInputs {
                base_path: base,
                columns,
            },
        );

        let ctx = SessionContext::new();
        ctx.register_table("documents", Arc::new(provider)).unwrap();

        let (rows, snap) = crate::observability::io_trace::scope(async {
            let batches = ctx
                .sql("SELECT id FROM documents WHERE amount >= 580 ORDER BY id")
                .await
                .unwrap()
                .collect()
                .await
                .unwrap();
            let mut ids = Vec::new();
            for b in &batches {
                if let Some(c) = b.column(0).as_any().downcast_ref::<StringArray>() {
                    for i in 0..b.num_rows() {
                        ids.push(c.value(i).to_string());
                    }
                }
            }
            (
                ids,
                crate::observability::io_trace::snapshot().expect("io_trace scope active"),
            )
        })
        .await;

        // Correctness/parity: exactly amounts 580..=599 survive the residual exact filter.
        assert_eq!(rows.len(), 20, "amount>=580 ⇒ 20 rows, got {rows:?}");
        assert_eq!(rows.first().map(String::as_str), Some("v0580"));
        // Byte win: ranged GETs issued AND fewer bytes read than the whole segment.
        assert!(
            snap.range_gets > 0,
            "expected ranged GETs, got {}",
            snap.range_gets
        );
        assert!(
            snap.bytes_read < file_len,
            "pruned bytes_read {} must be < whole segment {file_len}",
            snap.bytes_read
        );
    }
}
