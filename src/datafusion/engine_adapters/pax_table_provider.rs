// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! # PAX-native OLAP TableProvider (TD-OLAP-1 slice 2)
//!
//! Wraps [`PaxSplitReader`] (the `SplitReader` that bridges PAX segments → Arrow)
//! as a DataFusion [`TableProvider`], so PAX-backed analytical tables can scan
//! through the co-designed PAX→Arrow→DataFusion path instead of the Volcano.
//! Mirrors `ObjectStoreParquetTable` (the Parquet-backed provider) but simpler —
//! no bloom-semijoin caches (TD-OLAP-3's concern; PaxSplitReader does its own
//! per-block pruning via `PaxSegmentScanner`).
//!
//! Splits are discovered once at construction via [`discover_pax_segments`]
//! (TD-097); `scan()` creates a `ProximaScanExec` with a `PaxSplitReader` +
//! the discovered splits — the same exec that ObjectStoreParquetTable uses, so
//! filter/projection pushdown + runtime filters work identically.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::DFSchema;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionContext;

use crate::datafusion::engine_adapters::pax_adapter::PaxSplitReader;
use crate::datafusion::engine_adapters::pax_segment_locator::discover_pax_segments;
use crate::datafusion::proxima_scan_exec::{ProximaScanExec, SplitReader};
use crate::storage::formats::FileSplit;
use crate::storage::persistence::filesystem::FilesystemFactory;

/// DataFusion table backed by PAX segment files (`.pax`). The PAX-native OLAP
/// scan provider — the "make it live" step for TD-OLAP-1's PaxSplitReader.
#[derive(Debug)]
pub struct PaxTableProvider {
    schema: SchemaRef,
    splits: Vec<FileSplit>,
    filesystem_factory: Arc<FilesystemFactory>,
    name_to_col_id: HashMap<String, i32>,
    collection_name: String,
    /// TD-OLAP-1 Test 2.3: Tenant ID for predicate filtering (optional).
    tenant_id: Option<String>,
    /// TD-OLAP-1 Test 2.3: Time range for predicate filtering (optional).
    time_range: Option<(i64, i64)>,
}

impl PaxTableProvider {
    /// Discover `.pax` segments under `base_path` and construct the provider.
    /// The schema + `name_to_col_id` are the caller's responsibility (from the
    /// catalog schema — system columns via `col_id::*`, user columns via their
    /// assigned PAX column IDs).
    ///
    /// TD-OLAP-1 Test 2.3: `tenant_id` and `time_range` enable tenant/time
    /// predicate filtering at the storage layer. When `None`, no filtering is
    /// applied (backward-compatible with `ScanPredicate::default()`).
    pub async fn new(
        schema: SchemaRef,
        base_path: &str,
        filesystem_factory: Arc<FilesystemFactory>,
        name_to_col_id: HashMap<String, i32>,
        collection_name: String,
        tenant_id: Option<String>,
        time_range: Option<(i64, i64)>,
    ) -> DFResult<Self> {
        let splits = discover_pax_segments(base_path, &filesystem_factory)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        Ok(Self {
            schema,
            splits,
            filesystem_factory,
            name_to_col_id,
            collection_name,
            tenant_id,
            time_range,
        })
    }

    /// Construct the `SplitReader` that decodes PAX segments → Arrow batches,
    /// seeded with the caller-resolved physical filters so `PaxSegmentScanner`
    /// can prune blocks off the decode path (TD-OLAP-1 Test 2.6).
    fn reader(&self, physical_filters: Vec<Arc<dyn PhysicalExpr>>) -> Arc<dyn SplitReader> {
        Arc::new(PaxSplitReader::new(
            self.schema.clone(),
            self.filesystem_factory.clone(),
            self.name_to_col_id.clone(),
            physical_filters,
            self.tenant_id.clone(),
            self.time_range,
        ))
    }
}

#[async_trait]
impl TableProvider for PaxTableProvider {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    /// `Inexact` so DataFusion passes WHERE predicates to `scan()` —
    /// PaxSplitReader's per-block column-stat pruning (via PaxSegmentScanner)
    /// uses them to skip blocks. The FilterExec stays above for exact filtering.
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&[usize]>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        // TD-OLAP-1 Test 2.6: lower the pushed WHERE predicates to physical exprs
        // and hand them to the split reader, so the PAX prune stack (zone maps +
        // column stats via `pruning_predicates`) can skip blocks pre-decode.
        // Fail loudly on lowering errors — a silently dropped filter would only
        // cost bytes on the FilterExec path but hides translator gaps.
        let df_schema = DFSchema::try_from(self.schema.as_ref().clone())?;
        let mut physical_filters = Vec::with_capacity(filters.len());
        for expr in filters {
            physical_filters.push(state.create_physical_expr(expr.clone(), &df_schema)?);
        }
        let exec = ProximaScanExec::builder()
            .schema(self.schema.clone())
            .splits(self.splits.clone())
            .reader(self.reader(physical_filters))
            .projection(projection.map(<[usize]>::to_vec))
            .filters(filters.to_vec())
            .limit(limit)
            .collection_name(self.collection_name.clone())
            .batch_size(8192)
            .target_partitions(state.config().target_partitions())
            .build()?;
        Ok(Arc::new(exec))
    }
}

/// Convenience: discover PAX segments under `base_path`, construct a
/// [`PaxTableProvider`], and register it with the DataFusion `SessionContext`.
/// Mirrors `register_object_store_parquet_location`.
///
/// TD-OLAP-1 Test 2.3: `tenant_id` and `time_range` enable tenant/time
/// predicate filtering at the storage layer. When `None`, no filtering is
/// applied (backward-compatible with `ScanPredicate::default()`).
pub async fn register_pax_location(
    ctx: &SessionContext,
    name: &str,
    base_path: &str,
    schema: SchemaRef,
    name_to_col_id: HashMap<String, i32>,
    filesystem_factory: Arc<FilesystemFactory>,
    tenant_id: Option<String>,
    time_range: Option<(i64, i64)>,
) -> DFResult<Arc<PaxTableProvider>> {
    let table = Arc::new(
        PaxTableProvider::new(
            schema,
            base_path,
            filesystem_factory,
            name_to_col_id,
            name.into(),
            tenant_id,
            time_range,
        )
        .await?,
    );
    ctx.register_table(name, table.clone())?;
    Ok(table)
}
