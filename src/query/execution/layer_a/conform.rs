// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! Catalog conformance: turn a spill-segment batch (or canonical records) into the
//! exact rows the materialized native path serves.
//!
//! The materialized path builds every row the same way: decode the canonical
//! record, project it per catalog column (`props`, else the primary key from the
//! oid, else the first embedding), and encode with the catalog schema. This module
//! produces **the same arrays**, by one of two routes per row:
//!
//! * **Typed (fast).** The segment holds the column under the catalog's own Arrow
//!   type, so its array already IS what the encoder would emit: the spill writer
//!   encoded the value from that schema, and any value that would not read back
//!   identically went to the residual instead. Liveness and table membership are
//!   evaluated on the system columns, vectorized.
//! * **Records (slow).** Everything the typed route cannot vouch for goes through
//!   the canonical record and `DmlService::relational_record_for` — the very
//!   projection the materialized path uses: a row with a residual, a projected
//!   primary key that is NULL in its column (its value is the oid), a column
//!   stored under another type, a projected vector column, and every row when the
//!   caller's ABAC access is a row predicate (it is defined on records).
//!
//! A conformed batch keeps **every** row, visible or not, with its oid: a dead or
//! foreign newer copy must still shadow a live older one in the merge.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBuilder, Int64Array, StringArray, UInt32Array,
    new_null_array,
};
use arrow::compute::{interleave, take_record_batch};
use arrow::datatypes::{DataType, Schema, SchemaRef};
use arrow::record_batch::{RecordBatch, RecordBatchOptions};
use proximadb_catalog_schema::CatalogTableSchema;
use proximadb_data_model::ProximaType;
use proximadb_records::ProximaRecord;
use proximadb_storage_common::proxima_arrow::{
    SYS_COL_OID, SYS_COL_RESIDUAL, SYS_COL_VALID_FROM_NS, SYS_COL_VALID_TO_NS,
    SYS_COL_VARIATION_ID, proxima_records_to_record_batch, record_batch_to_proxima_records,
};

use crate::services::dml::DmlService;
use crate::services::layer_a_snapshot::LayerARowAccess;

/// What one scan of one table conforms to. Built once per scan.
pub(crate) struct ConformContext {
    pub schema: CatalogTableSchema,
    /// `schema` with every column nullable — what the record route encodes
    /// with, since it also encodes rows that are never emitted (see `internal`).
    encode_schema: CatalogTableSchema,
    /// `schema`'s column names, in order (the projection helper's input).
    col_names: Vec<String>,
    /// Indices into the catalog Arrow schema (every live column, as the
    /// encoder emits it) this scan emits, in order.
    pub projection: Vec<usize>,
    /// The catalog Arrow schema projected — the scan's output schema.
    pub output: SchemaRef,
    /// `output` with every field nullable: the shape of a conformed batch,
    /// which also holds rows that are never emitted (a persisted tombstone has
    /// no props, so it is NULL even in a NOT NULL column).
    internal: SchemaRef,
    /// Position of the primary-key column in `projection`, if projected.
    projected_pk: Option<usize>,
    /// Every row goes through records — see the module note.
    pub records_only: bool,
    access: Arc<LayerARowAccess>,
    table_name: String,
    /// The read instant liveness is evaluated at (one per scan, as the
    /// materialized path takes one per scan).
    now_ns: i64,
}

impl ConformContext {
    pub fn new(
        schema: CatalogTableSchema,
        projection: Option<&[usize]>,
        access: Arc<LayerARowAccess>,
        now_ns: i64,
    ) -> Result<Self> {
        let full = schema.to_arrow_schema();
        let projection: Vec<usize> = match projection {
            Some(p) => p.to_vec(),
            None => (0..full.fields().len()).collect(),
        };
        if let Some(bad) = projection.iter().find(|&&i| i >= full.fields().len()) {
            return Err(anyhow!(
                "layer A: projection index {bad} out of range for table '{}'",
                schema.name
            ));
        }
        let output = Arc::new(full.project(&projection)?);
        let internal = Arc::new(Schema::new(
            output
                .fields()
                .iter()
                .map(|f| f.as_ref().clone().with_nullable(true))
                .collect::<Vec<_>>(),
        ));
        let live: Vec<_> = schema.columns.iter().filter(|c| !c.is_deleted).collect();
        let pk = schema.primary_key.first();
        let projected_pk = pk.and_then(|pk| {
            projection
                .iter()
                .position(|&i| live[i].name.eq_ignore_ascii_case(pk))
        });
        let projects_vector = projection.iter().any(|&i| {
            matches!(
                live[i].data_type,
                ProximaType::DenseVector { .. }
                    | ProximaType::SparseVector { .. }
                    | ProximaType::BinaryVector { .. }
            )
        });
        let records_only = projects_vector || matches!(*access, LayerARowAccess::Restricted(_));
        Ok(Self {
            col_names: schema.columns.iter().map(|c| c.name.clone()).collect(),
            table_name: schema.name.clone(),
            encode_schema: {
                let mut nullable = schema.clone();
                for column in &mut nullable.columns {
                    column.nullable = true;
                }
                nullable
            },
            schema,
            projection,
            output,
            internal,
            projected_pk,
            records_only,
            access,
            now_ns,
        })
    }

    /// Columns a segment read must fetch: every column when rows go through
    /// records, else the projected ones present in `segment`, plus the system
    /// columns. Names, matched exactly as the record decoder keys `props`.
    pub fn segment_columns(&self, segment: &Schema) -> Option<Vec<usize>> {
        if self.records_only {
            return None;
        }
        let wanted: Vec<&str> = self
            .output
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .chain([
                SYS_COL_OID,
                SYS_COL_VARIATION_ID,
                SYS_COL_VALID_FROM_NS,
                SYS_COL_VALID_TO_NS,
                SYS_COL_RESIDUAL,
            ])
            .collect();
        Some(
            segment
                .fields()
                .iter()
                .enumerate()
                .filter(|(_, f)| wanted.contains(&f.name().as_str()))
                .map(|(i, _)| i)
                .collect(),
        )
    }

    /// Is a decoded record visible to this scan (ignoring shadowing)?
    fn record_visible(&self, record: &ProximaRecord) -> bool {
        !record.is_dead(self.now_ns)
            && record
                .variation_id
                .as_deref()
                .is_none_or(|v| v == self.table_name)
            && match self.access.as_ref() {
                LayerARowAccess::Unrestricted => true,
                LayerARowAccess::Restricted(predicate) => predicate(record),
            }
    }

    /// Records → their projected relational arrays plus visibility. A row whose
    /// projection fails is invisible (all-NULL values), as the materialized
    /// scan drops it.
    fn conform_records(&self, records: &[ProximaRecord]) -> Result<(Vec<ArrayRef>, BooleanArray)> {
        let mut visible = BooleanBuilder::with_capacity(records.len());
        let relational: Vec<ProximaRecord> = records
            .iter()
            .map(|record| {
                match DmlService::relational_record_for(record, &self.schema, &self.col_names) {
                    Some(row) => {
                        visible.append_value(self.record_visible(record));
                        row
                    }
                    None => {
                        visible.append_value(false);
                        ProximaRecord::default()
                    }
                }
            })
            .collect();
        let batch = proxima_records_to_record_batch(&relational, &self.encode_schema)
            .context("layer A: encode conformed rows")?;
        let columns = self
            .projection
            .iter()
            .map(|&i| batch.column(i).clone())
            .collect();
        Ok((columns, visible.finish()))
    }
}

/// Rows in the scan's output shape, each with its oid and whether it is
/// visible once it wins the merge.
#[derive(Debug, Clone)]
pub(crate) struct Conformed {
    pub batch: RecordBatch,
    pub oids: StringArray,
    pub visible: BooleanArray,
}

impl Conformed {
    pub fn len(&self) -> usize {
        self.oids.len()
    }

    fn new(
        ctx: &ConformContext,
        columns: Vec<ArrayRef>,
        oids: StringArray,
        visible: BooleanArray,
    ) -> Result<Self> {
        let options = RecordBatchOptions::new().with_row_count(Some(oids.len()));
        let batch = RecordBatch::try_new_with_options(ctx.internal.clone(), columns, &options)
            .context("layer A: assemble conformed batch")?;
        Ok(Self {
            batch,
            oids,
            visible,
        })
    }

    /// Canonical records — resident rows, or a segment that cannot be streamed.
    pub fn from_records(ctx: &ConformContext, records: &[ProximaRecord]) -> Result<Self> {
        let (columns, visible) = ctx.conform_records(records)?;
        let oids = StringArray::from_iter_values(records.iter().map(|r| r.oid.as_str()));
        Self::new(ctx, columns, oids, visible)
    }
}

/// How one segment's columns map onto the scan's output. Built once per segment
/// from the schema its stream yields.
pub(crate) struct SegmentShape {
    oid: usize,
    variation: Option<usize>,
    valid_from: Option<usize>,
    valid_to: Option<usize>,
    residual: Option<usize>,
    /// Per output column: the segment column holding it under the identical type.
    columns: Vec<Option<usize>>,
    /// Every row goes through records (the context says so, or a projected
    /// column — the primary key included — is stored under another type, or
    /// the projected primary key has no column at all).
    records_only: bool,
}

impl SegmentShape {
    pub fn of(ctx: &ConformContext, segment: &Schema) -> Result<Self> {
        let find = |name: &str| segment.index_of(name).ok();
        let oid = find(SYS_COL_OID)
            .ok_or_else(|| anyhow!("layer A: segment batch has no {SYS_COL_OID} column"))?;
        let mut records_only = ctx.records_only;
        let columns: Vec<Option<usize>> = ctx
            .output
            .fields()
            .iter()
            .map(|field| match find(field.name()) {
                Some(i) if segment.field(i).data_type() == field.data_type() => Some(i),
                Some(_) => {
                    records_only = true;
                    None
                }
                None => None,
            })
            .collect();
        if let Some(pk) = ctx.projected_pk
            && columns[pk].is_none()
        {
            records_only = true;
        }
        let i64_col = |name: &str| -> Result<Option<usize>> {
            match find(name) {
                Some(i) if segment.field(i).data_type() != &DataType::Int64 => Err(anyhow!(
                    "layer A: segment column {name} is {}, not Int64",
                    segment.field(i).data_type()
                )),
                other => Ok(other),
            }
        };
        Ok(Self {
            oid,
            variation: find(SYS_COL_VARIATION_ID),
            valid_from: i64_col(SYS_COL_VALID_FROM_NS)?,
            valid_to: i64_col(SYS_COL_VALID_TO_NS)?,
            residual: find(SYS_COL_RESIDUAL),
            columns,
            records_only,
        })
    }
}

fn utf8<'a>(batch: &'a RecordBatch, idx: usize, name: &str) -> Result<&'a StringArray> {
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow!("layer A: segment column {name} is not Utf8"))
}

fn int64(batch: &RecordBatch, idx: Option<usize>) -> Option<&Int64Array> {
    idx.and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>())
}

/// Conform one segment batch (see the module note for the per-row route).
pub(crate) fn conform_segment_batch(
    ctx: &ConformContext,
    shape: &SegmentShape,
    batch: &RecordBatch,
) -> Result<Conformed> {
    let n = batch.num_rows();
    let oids = utf8(batch, shape.oid, SYS_COL_OID)?.clone();
    if oids.null_count() > 0 {
        return Err(anyhow!("layer A: segment row with a NULL {SYS_COL_OID}"));
    }

    // Which rows need the record route.
    let residual = shape.residual.map(|i| batch.column(i));
    let pk = ctx
        .projected_pk
        .and_then(|p| shape.columns[p])
        .map(|i| batch.column(i));
    let slow: Vec<u32> = if shape.records_only {
        (0..n as u32).collect()
    } else {
        (0..n)
            .filter(|&r| {
                residual.is_some_and(|a| a.is_valid(r)) || pk.is_some_and(|a| a.is_null(r))
            })
            .map(|r| r as u32)
            .collect()
    };

    // Typed visibility: membership and liveness from the system columns.
    let variation = shape
        .variation
        .map(|i| utf8(batch, i, SYS_COL_VARIATION_ID))
        .transpose()?;
    let valid_from = int64(batch, shape.valid_from);
    let valid_to = int64(batch, shape.valid_to);
    let now = ctx.now_ns;
    let mut visible: Vec<bool> = (0..n)
        .map(|r| {
            variation.is_none_or(|v| v.is_null(r) || v.value(r) == ctx.table_name)
                && valid_from.is_none_or(|a| a.is_null(r) || a.value(r) <= now)
                && valid_to.is_none_or(|a| a.is_null(r) || now < a.value(r))
        })
        .collect();

    let typed: Vec<ArrayRef> = ctx
        .output
        .fields()
        .iter()
        .zip(&shape.columns)
        .map(|(field, column)| match column {
            Some(i) => batch.column(*i).clone(),
            None => new_null_array(field.data_type(), n),
        })
        .collect();
    if slow.is_empty() {
        return Conformed::new(ctx, typed, oids, BooleanArray::from(visible));
    }

    let slow_batch = take_record_batch(batch, &UInt32Array::from(slow.clone()))?;
    let records = record_batch_to_proxima_records(&slow_batch);
    let (slow_columns, slow_visible) = ctx.conform_records(&records)?;
    for (k, &r) in slow.iter().enumerate() {
        visible[r as usize] = slow_visible.value(k);
    }
    if slow.len() == n {
        return Conformed::new(ctx, slow_columns, oids, BooleanArray::from(visible));
    }
    // Splice: row r comes from the typed arrays unless it took the record route.
    let mut pick: Vec<(usize, usize)> = (0..n).map(|r| (0, r)).collect();
    for (k, &r) in slow.iter().enumerate() {
        pick[r as usize] = (1, k);
    }
    let columns = typed
        .iter()
        .zip(&slow_columns)
        .map(|(fast, records)| interleave(&[fast.as_ref(), records.as_ref()], &pick))
        .collect::<Result<Vec<_>, _>>()?;
    Conformed::new(ctx, columns, oids, BooleanArray::from(visible))
}
