// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! # PAX-native OLAP split reader (TD-OLAP-1 slice 1)
//!
//! Bridges ProximaDB's canonical PAX block reader ([`PaxBlockReader`], with its
//! hierarchical prune stack — zone maps, bloom, tenant/time/column-stat) to
//! DataFusion's vectorized execution by emitting Arrow `RecordBatch`es through
//! the [`SplitReader`] seam. This is the structural differentiator ADR-052
//! invariant 2 names: the **PAX-native scan** that co-designs ANN and OLAP on
//! one geometry (the same `PaxBlockReader` the vector path uses).
//!
//! ## Scope (slice 1) — and the architectural boundary it respects
//!
//! Per the block-format crate mandate and ADR-033, **pure relational OLAP stays
//! on Parquet/Iceberg** (the team's TD-OLAP-3/6 path). This reader is **not** a
//! replacement for that — it is the scan for the **hybrid vector+OLAP wedge**
//! (vectors force PAX; scan the relational stripes alongside) and for
//! LSM-resident PAX data not yet materialized to Parquet. Slice 1 delivers the
//! relational-stripe → Arrow bridge + the canonical prune stack; vector-stripe
//! decode (`decode_f32_vec_stripe`) and the hybrid-query integration are slice 2
//! — at which point the wedge value materializes.
//!
//! ## SOLID / co-design
//! * **OCP** — the 7th `SplitReader` impl behind the frozen seam; no existing
//!   path is modified (the ad-hoc `SstBlockStream` stub is left untouched;
//!   consolidating it onto `PaxBlockReader` is slice 3).
//! * **SRP** — the prune stack STAYS in `PaxBlockReader`; this reader owns only
//!   byte-load → open → conservative prune check → typed-stripe decode → Arrow.
//! * **Default-off** — gated by `PROXIMADB_DF_PAX_READER` (mirrors
//!   `PROXIMADB_DF_RUNTIME_FILTER_PRUNE`). NOT wired into any TableProvider /
//!   `route_select` (route flip is slice 2, gated on the TD-OLAP-4 ledger, per
//!   ADR-052 observe→ingest→act).

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use arrow_array::builder::{Float32Builder, ListBuilder};
use arrow_array::{
    ArrayRef, BinaryArray, FixedSizeListArray, Float32Array, Float64Array, Int64Array, RecordBatch,
    StringArray,
};
use arrow_schema::{DataType, Field, SchemaRef};
use async_trait::async_trait;
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::stream;
use tracing::trace;

use proximadb_block_format::{
    BLOCK_FOOTER_SIZE, BlockFooter, BlockLayout, BlockZoneSource, ColumnRole, PaxBlockReader,
    metadata_ranges,
};
use proximadb_storage_common::format_splits::{ScalarPredicate, ScalarValue};
use proximadb_storage_common::pax_block::{
    BlockIndexEntry, PaxSegmentScanner, ScanPredicate, SegmentIndex,
};
use proximadb_storage_common::segment_layout::{
    SEG_HEADER_PREFIX_LEN, SEG_LAYOUT_VERSION, SegmentFooterIndex, SegmentHeaderPrefix,
    is_coalesced_segment,
};

use crate::observability::io_trace;

use crate::datafusion::physical_filter_translate::pruning_predicates;
use crate::datafusion::proxima_scan_exec::{EmptyRecordBatchStream, SplitReader};
use crate::datafusion::proxima_table_provider::EngineType;
use crate::storage::formats::FileSplit;
use crate::storage::persistence::filesystem::FilesystemFactory;

/// `true` when the PAX-native OLAP scan is opted in (TD-OLAP-1 slice 1).
/// Default OFF — **do not enable in production until slice 2 dispatch is
/// wired** (see the TD-OLAP-1 slice-2 design record): with the gate ON the
/// scheduler STAMPS `DataFusionLocal` for pax-backed OLAP shapes, but the
/// dispatch block still requires `parquet_backed`, so Volcano serves while
/// the cost model attributes Volcano's measured costs to DataFusion cells
/// (attribution poisoning). Flipping this default is gated on the slice-2
/// requirements (WAL-delta reconciliation, path reconciliation,
/// `pax_backed` predicate refinement), not on the storage layout (which
/// defaulted to row-group Region D on 2026-08-28).
pub fn pax_reader_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("PROXIMADB_DF_PAX_READER")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// `true` when the PAX-native reader should fetch a segment via **ranged reads**
/// (index-zone prune → `read_ranges` of surviving blocks only) instead of one
/// whole-file read (TD-DOC-PUSHDOWN-1). Default OFF — mixed-read-safe: falls back
/// to the whole-file path for any segment without a locatable v2 zone index, and
/// the whole-file path stays the default until the ranged path is baked (ADR-052
/// observe→flip). Gate: `PROXIMADB_DF_PAX_RANGED`.
pub fn pax_ranged_read_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("PROXIMADB_DF_PAX_RANGED")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Smallest possible segment tail (`[body_len u32][PAXZ][SEGMENT_MAGIC 8]` at
/// minimum) — below this a file cannot carry a locatable index, so skip ranged.
const SEGMENT_INDEX_TRAILER_MIN: u64 = 16;

/// Initial tail suffix GET size when locating the segment index; grown ×4 on a
/// short suffix. 8 KiB holds a typical segment's index (~93 B/block ⇒ ~85 blocks)
/// in one GET while staying a small fraction of a modest segment — a large,
/// many-block segment grows one extra step (rare, cheap vs the body reads).
const RANGED_INITIAL_SUFFIX: u64 = 8 * 1024;

/// PAX-native OLAP split reader. Loads a PAX **segment** and iterates its blocks
/// via the canonical [`PaxSegmentScanner`] (each yielded block is a
/// [`PaxBlockReader`]), applies zone-map/bloom pruning from the pushed filters,
/// then decodes the projected relational stripes into one Arrow `RecordBatch`
/// per surviving block.
#[derive(Debug)]
pub struct PaxSplitReader {
    schema: SchemaRef,
    filesystem_factory: Arc<FilesystemFactory>,
    /// Column NAME (DataFusion/SQL-facing) → PAX `column_id`. The prune +
    /// decode APIs of `PaxBlockReader` key on `column_id: i32`.
    name_to_col_id: HashMap<String, i32>,
    /// Translated once at construction from the resolved physical filters.
    /// Empty ⇒ no pruning (correctness-safe). AND-semantics at `block_pruned`.
    prune_predicates: Vec<(String, ScalarPredicate)>,
    /// TD-OLAP-1 Test 2.3: Tenant ID for predicate filtering (optional).
    /// When set, blocks are pruned at scan time if tenant hash doesn't match.
    tenant_id: Option<String>,
    /// TD-OLAP-1 Test 2.3: Time range for predicate filtering (optional).
    /// When set, blocks are pruned if they don't overlap with [from_ns, to_ns].
    time_range: Option<(i64, i64)>,
}

impl PaxSplitReader {
    /// Construct a reader for `schema`, resolving filter→column-id pruning at
    /// build time. `filters` are the resolved physical filters (logical `Expr`s
    /// must be lowered to physical first — the caller's job in slice 2 routing;
    /// pass empty for the decode-only / test path).
    ///
    /// TD-OLAP-1 Test 2.3: `tenant_id` and `time_range` enable tenant/time
    /// predicate filtering at the storage layer. When `None`, no filtering is
    /// applied (backward-compatible with `ScanPredicate::default()`).
    pub fn new(
        schema: SchemaRef,
        filesystem_factory: Arc<FilesystemFactory>,
        name_to_col_id: HashMap<String, i32>,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        tenant_id: Option<String>,
        time_range: Option<(i64, i64)>,
    ) -> Self {
        let mut prune_predicates = Vec::new();
        for f in &filters {
            prune_predicates.extend(pruning_predicates(f));
        }
        Self {
            schema,
            filesystem_factory,
            name_to_col_id,
            prune_predicates,
            tenant_id,
            time_range,
        }
    }

    /// Whole-file read of the split's PAX bytes — the default path and the
    /// mixed-read-safe fallback for segments without a locatable v2 zone index.
    /// The pruning ranged path (fetch only surviving blocks) is [`Self::load_ranged`],
    /// engaged by `PROXIMADB_DF_PAX_RANGED` (TD-DOC-PUSHDOWN-1).
    async fn load_bytes(&self, split: &FileSplit) -> DFResult<Vec<u8>> {
        self.filesystem_factory
            .read(&split.file_path)
            .await
            .map_err(|e| {
                DataFusionError::Execution(format!(
                    "PAX scan: read of {} failed: {e}",
                    split.file_path
                ))
            })
    }

    /// Ranged read (TD-DOC-PUSHDOWN-1): locate the segment index from a small tail
    /// suffix, prune blocks against the **index zone summary** (no block body GET),
    /// then `read_ranges` only the SURVIVING blocks and decode each to Arrow.
    ///
    /// NOTE: `read_ranges` issues one physical GET per surviving block unless
    /// the filesystem carries a range-coalescing policy (default: none). The
    /// byte win below is real; the request count is not reduced by batching
    /// alone. This
    /// is the byte-read win the whole-file [`Self::load_bytes`] path leaves on the
    /// table: a temporal/canonical-column predicate prunes whole blocks off the
    /// wire (`bytes_read` < segment, `range_gets` > 0), reusing the SAME
    /// [`BlockZoneSource`] prune logic as the decode path.
    ///
    /// Returns `Ok(None)` — signalling the caller to fall back to the whole-file
    /// path — when the segment has no locatable/zone-bearing index (legacy v1
    /// segments, tiny single-block segments), so it is mixed-read-safe with no
    /// flag-day.
    ///
    /// Two prune stages (TD-DOC-PUSHDOWN-1): **Stage A** prunes against the index
    /// zone summary (canonical columns, no per-block GET); **Stage B** — engaged only
    /// when a predicate targets a shredded/user column the summary can't evaluate —
    /// range-reads each surviving block's footer/metadata (NOT its body) into a
    /// [`BlockLayout`] and prunes on ALL columns, so a `props__<key>`-pruned block
    /// costs only its footer, never its body.
    pub(crate) async fn load_ranged(
        &self,
        split: &FileSplit,
        out_schema: &SchemaRef,
    ) -> DFResult<Option<Vec<RecordBatch>>> {
        let path = &split.file_path;
        // TD-USUB-8: prefer the object size the PLANNER already knows. The
        // segment locators get it free from `list()`, so the `HEAD` below is a
        // dependent round trip spent rediscovering a number we were handed —
        // and it is the first of the four in the cold-read chain, which nothing
        // else can start without. `None` ⇒ fall back to the HEAD (external or
        // hand-built splits that never carried a size).
        let len = match split.object_size {
            Some(size) => size,
            None => match self.filesystem_factory.metadata(path).await {
                Ok(m) => m.size,
                Err(_) => return Ok(None), // no cheap size → fall back
            },
        };
        if len < SEGMENT_INDEX_TRAILER_MIN {
            return Ok(None);
        }
        // TD-PAXRG-1: a v4 row-group segment DECLARES its index (the coalesced
        // footer + per-RG MinMax stats) in the header-prefix — no PAXZ tail
        // probe. One small GET decides; the footer renders as the layout-neutral
        // `SegmentIndex` the prune/fetch stages below already consume.
        let prefix_len = (SEG_HEADER_PREFIX_LEN as u64).min(len);
        let prefix = self
            .filesystem_factory
            .read_range(path, 0, prefix_len)
            .await
            .map_err(|e| DataFusionError::Execution(format!("PAX ranged header read: {e}")))?;
        let v4_index: Option<SegmentIndex> = if is_coalesced_segment(&prefix) {
            match SegmentHeaderPrefix::parse(&prefix) {
                Ok(h) if h.layout_version == SEG_LAYOUT_VERSION && h.footer_len > 0 => {
                    let body = self
                        .filesystem_factory
                        .read_range(path, h.footer_off, h.footer_len)
                        .await
                        .map_err(|e| {
                            DataFusionError::Execution(format!("PAX ranged v4 footer read: {e}"))
                        })?;
                    SegmentFooterIndex::parse(&body).ok().map(|footer| {
                        let blocks = footer
                            .blocks
                            .iter()
                            .zip(&footer.block_stats)
                            .map(|(entry, stats)| BlockIndexEntry {
                                offset: entry.offset,
                                size: entry.size,
                                zone: stats.as_ref().map(|s| s.zone.clone()),
                                footer_stats: None,
                            })
                            .collect();
                        SegmentIndex { blocks }
                    })
                }
                _ => None,
            }
        } else {
            None
        };
        if let Some(v4_index) = v4_index {
            // Fall through to the shared prune + ranged-fetch stages with the
            // v4 index (boxed to keep the async-block sizing uniform).
            return self
                .load_ranged_with_index(path, v4_index, out_schema)
                .await;
        }

        // Probe the tail for the segment index, growing ×4 on a short suffix. Every
        // physical read (here and below) is accounted by the filesystem layer's own
        // io_trace hooks (`record_range_gets` + `record_bytes_read`), so this method
        // records none itself — it would double-count.
        let mut probe = RANGED_INITIAL_SUFFIX.min(len);
        let index = loop {
            let start = len - probe;
            let suffix = self
                .filesystem_factory
                .read_range(path, start, probe)
                .await
                .map_err(|e| DataFusionError::Execution(format!("PAX ranged tail read: {e}")))?;
            match SegmentIndex::locate_in_suffix(&suffix) {
                Ok(Some(idx)) => break idx,
                Ok(None) if probe < len => probe = (probe.saturating_mul(4)).min(len),
                // No index found even reading the whole file, or a corrupt tail:
                // fall back to the whole-file path (which surfaces real errors).
                Ok(None) | Err(_) => return Ok(None),
            }
        };
        self.load_ranged_with_index(path, index, out_schema).await
    }

    /// Shared ranged stages (TD-PAXRG-1): prune + fetch, layout-agnostic —
    /// `index` is either the PAXZ v2 trailer (v1/v3 segments) or the v4
    /// footer's block table rendered with per-RG zone summaries.
    async fn load_ranged_with_index(
        &self,
        path: &str,
        index: SegmentIndex,
        out_schema: &SchemaRef,
    ) -> DFResult<Option<Vec<RecordBatch>>> {
        // Stage A — prune against the index zone summary (canonical columns; v1
        // entries carry no zone ⇒ conservatively keep). No per-block GET.
        let mut stage_a: Vec<&BlockIndexEntry> = Vec::new();
        for entry in &index.blocks {
            let pruned = match &entry.zone {
                Some(zone) => self.pruned_by(zone as &dyn BlockZoneSource),
                None => false,
            };
            if !pruned {
                stage_a.push(entry);
            }
        }
        io_trace::record_op_str("fetch_pax_ranged");

        // Stage B — shredded/user-column prune (Layer 2). Only when a predicate targets
        // a column the index summary can't evaluate: range-read each survivor's footer +
        // metadata into a `BlockLayout` (carries every column's bounds) and prune on ALL
        // columns before its body is fetched. A pruned block costs only its footer.
        // (io_trace `bytes_read`/`range_gets` accrue automatically per `read_range`.)
        let kept: Vec<&BlockIndexEntry> = if self.needs_footer_prune() {
            let mut kept = Vec::new();
            for entry in stage_a {
                match self
                    .block_layout_ranged(path, entry.offset, entry.size as u64)
                    .await
                {
                    Ok(layout) => {
                        if self.pruned_by(&layout as &dyn BlockZoneSource) {
                            continue; // pruned on a user column — never fetch the body
                        }
                        kept.push(entry);
                    }
                    // Footer read/parse failed ⇒ conservatively keep (decode re-prunes).
                    Err(_) => kept.push(entry),
                }
            }
            kept
        } else {
            stage_a
        };

        if kept.is_empty() {
            return Ok(Some(Vec::new()));
        }
        let ranges: Vec<std::ops::Range<u64>> = kept
            .iter()
            .map(|e| e.offset..e.offset + e.size as u64)
            .collect();
        let bufs = self
            .filesystem_factory
            .read_ranges(path, ranges)
            .await
            .map_err(|e| DataFusionError::Execution(format!("PAX ranged block read: {e}")))?;
        let mut batches = Vec::with_capacity(bufs.len());
        for buf in &bufs {
            let reader = PaxBlockReader::open(buf)
                .map_err(|e| DataFusionError::Execution(format!("PAX ranged block open: {e}")))?;
            // Defence-in-depth: the index summary prunes on canonical columns only;
            // re-run the full prune on the decoded block (idempotent, cheap).
            if self.block_pruned(&reader) {
                continue;
            }
            let arrays: Vec<ArrayRef> = out_schema
                .fields()
                .iter()
                .map(|f| self.decode_column(&reader, f))
                .collect::<DFResult<_>>()?;
            let batch = RecordBatch::try_new(out_schema.clone(), arrays).map_err(|e| {
                DataFusionError::Execution(format!("PAX ranged batch build failed: {e}"))
            })?;
            batches.push(batch);
        }
        Ok(Some(batches))
    }

    /// `true` when any prune predicate targets a column the index zone summary does
    /// NOT carry (a shredded/user column, or a non-summarized canonical column) — the
    /// case where a per-block footer read (Stage B) can prune beyond the index summary.
    /// When every predicate is on a summarized canonical column, Stage A suffices and
    /// no footer GETs are issued.
    fn needs_footer_prune(&self) -> bool {
        self.prune_predicates.iter().any(|(name, _)| {
            self.name_to_col_id
                .get(name)
                .is_some_and(|&cid| !is_canonical_zone_col(cid))
        })
    }

    /// Range-read one block's footer + metadata extent (NO stripe/body bytes) and
    /// assemble a [`BlockLayout`] — per-block statistics carrying EVERY column's bounds
    /// (incl. shredded user columns), so a body-free prune can run. Mirrors
    /// `RangedSegmentReader::build_block_layout` in the `FilesystemFactory` world:
    /// one GET for the trailing footer, one for the contiguous metadata extent (both
    /// accounted by the filesystem layer's io_trace hooks).
    async fn block_layout_ranged(
        &self,
        path: &str,
        block_offset: u64,
        block_size: u64,
    ) -> DFResult<BlockLayout> {
        let footer_start = block_size - BLOCK_FOOTER_SIZE as u64;
        let tail = self
            .filesystem_factory
            .read_range(path, block_offset + footer_start, BLOCK_FOOTER_SIZE as u64)
            .await
            .map_err(|e| DataFusionError::Execution(format!("PAX footer read: {e}")))?;
        let footer = BlockFooter::from_bytes(&tail)
            .map_err(|e| DataFusionError::Execution(format!("PAX footer parse: {e}")))?;
        // The col-meta / vparam / rgdir regions are contiguous below the footer; one
        // ranged read of [meta_start, footer_start) covers them (no stripe bytes).
        let mr = metadata_ranges(&footer, block_size);
        let mut meta_start = mr.col_meta.start;
        if let Some(r) = &mr.vparam {
            meta_start = meta_start.min(r.start);
        }
        if let Some(r) = &mr.rgdir {
            meta_start = meta_start.min(r.start);
        }
        let meta_buf = if footer_start > meta_start {
            self.filesystem_factory
                .read_range(path, block_offset + meta_start, footer_start - meta_start)
                .await
                .map_err(|e| DataFusionError::Execution(format!("PAX meta read: {e}")))?
        } else {
            Vec::new()
        };
        let col_meta = slice_meta(&meta_buf, meta_start, &mr.col_meta)?;
        let vparam = match &mr.vparam {
            Some(r) => Some(slice_meta(&meta_buf, meta_start, r)?),
            None => None,
        };
        let rgdir = match &mr.rgdir {
            Some(r) => Some(slice_meta(&meta_buf, meta_start, r)?),
            None => None,
        };
        BlockLayout::assemble(footer, col_meta, vparam, rgdir)
            .map_err(|e| DataFusionError::Execution(format!("PAX assemble layout: {e}")))
    }

    /// Core decode (no I/O): segment bytes → [`PaxSegmentScanner`] → per-block
    /// prune → decode → one `RecordBatch` per surviving block. Pure / SRP
    /// boundary so the bridge + prune stack can be exercised without a filesystem.
    ///
    /// Real `.pax` files are *segments* (`[blocks][index][SEGMENT_MAGIC]`); a bare
    /// `PaxBlockReader::open(whole_file)` CRC-fails on them, so the segment
    /// scanner is mandatory (slice 1.5 fix to #706).
    fn decode_segment(&self, bytes: &[u8], out_schema: &SchemaRef) -> DFResult<Vec<RecordBatch>> {
        // TD-OLAP-1 Test 2.3: Wire tenant/time predicates instead of default
        let mut predicate = ScanPredicate::default();
        if let Some(ref tenant_id) = self.tenant_id {
            predicate = predicate.with_tenant(tenant_id);
        }
        if let Some((from_ns, to_ns)) = self.time_range {
            predicate = predicate.with_time_range(from_ns, to_ns);
        }

        let mut scanner = PaxSegmentScanner::from_bytes(bytes.to_vec(), predicate)
            .map_err(|e| DataFusionError::Execution(format!("PAX segment open failed: {e}")))?;
        let mut batches = Vec::new();
        while let Some(reader) = scanner.next_block() {
            if self.block_pruned(&reader) {
                trace!("PAX scan: block pruned by zone map");
                continue;
            }
            let arrays: Vec<ArrayRef> = out_schema
                .fields()
                .iter()
                .map(|f| self.decode_column(&reader, f))
                .collect::<DFResult<_>>()?;
            let batch = RecordBatch::try_new(out_schema.clone(), arrays)
                .map_err(|e| DataFusionError::Execution(format!("PAX batch build failed: {e}")))?;
            batches.push(batch);
        }
        Ok(batches)
    }

    /// AND-semantics prune over any [`BlockZoneSource`] (a decoded [`PaxBlockReader`]
    /// or the index [`BlockZoneSummary`]): returns `true` if ANY conjunctive
    /// predicate provably excludes the block. Unknown column/operator ⇒ no prune.
    fn pruned_by(&self, src: &dyn BlockZoneSource) -> bool {
        self.prune_predicates.iter().any(|(name, pred)| {
            let Some(&cid) = self.name_to_col_id.get(name) else {
                return false; // unknown column → cannot prune
            };
            !block_may_contain(src, cid, pred)
        })
    }

    /// AND-semantics block prune on a decoded block. Unrecognized column/operator
    /// ⇒ no prune.
    fn block_pruned(&self, reader: &PaxBlockReader) -> bool {
        self.pruned_by(reader)
    }

    /// Decode one column by its DataFusion field (name → PAX column_id → typed stripe).
    /// The target Arrow type is type-directed for the `props` tail: a `Utf8` field
    /// reconstructs the document JSON object from the msgpack tail (the `documents()`
    /// surface, TD-DOC-PUSHDOWN-1); a `Binary` field emits the raw msgpack bytes.
    ///
    /// TD-OLAP-1 Test 2.4: Vector stripe decode for f32 vectors (ColumnRole::Vector).
    fn decode_column(&self, reader: &PaxBlockReader, field: &Field) -> DFResult<ArrayRef> {
        let name = field.name();
        let cid = *self
            .name_to_col_id
            .get(name)
            .ok_or_else(|| DataFusionError::Execution(format!("PAX: no column_id for {name}")))?;
        let meta = reader
            .column_metas()
            .iter()
            .find(|m| m.column_id == cid)
            .ok_or_else(|| {
                DataFusionError::Execution(format!("PAX: column {name} (id {cid}) not in block"))
            })?;
        // TypeId bytes (proximadb_codec): I64=0x03, F64=0x02, F32=0x01;
        // 0xff = variable-length (string/bytes).
        Ok(match (meta.data_type_id, meta.role) {
            (0x03, _) => Arc::new(Int64Array::from(decode_i64(reader, cid, name)?)),
            (0x02, _) => Arc::new(Float64Array::from(decode_f64(reader, cid, name)?)),
            (0xff, ColumnRole::Props) if field.data_type() == &DataType::Utf8 => {
                Arc::new(StringArray::from(decode_props_json(reader, cid, name)?))
            }
            (0xff, ColumnRole::Props) => {
                Arc::new(BinaryArray::from_iter(decode_bytes(reader, cid, name)?))
            }
            (0xff, ColumnRole::Vector) => {
                // TD-OLAP-1 Test 2.4: f32 vector stripe decode
                let vectors = reader.decode_f32_vec_stripe(cid).ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "PAX: vector column {name} (id {cid}) decode failed"
                    ))
                })?;
                Arc::new(Self::decode_f32_vectors_to_arrow(
                    vectors,
                    field.data_type(),
                )?)
            }
            (0xff, _) => Arc::new(StringArray::from(decode_str(reader, cid, name)?)),
            (other, _) => {
                return Err(DataFusionError::NotImplemented(format!(
                    "PAX data_type_id {other:#x} for {name} (slice 1 covers i64/f64/str/bytes; \
                     f32 vector decode is now implemented)"
                )));
            }
        })
    }

    /// TD-OLAP-1 Test 2.4: Convert f32 vector data to Arrow arrays.
    ///
    /// Takes `Vec<Option<Vec<f32>>>` from `PaxBlockReader::decode_f32_vec_stripe`
    /// and converts it to the appropriate Arrow array type (List<Float32> or
    /// FixedSizeList<Float32>).
    fn decode_f32_vectors_to_arrow(
        vectors: Vec<Option<Vec<f32>>>,
        target_type: &DataType,
    ) -> DFResult<ArrayRef> {
        match target_type {
            DataType::List(field) if matches!(*field.data_type(), DataType::Float32) => {
                // Variable-length vectors: List<Float32>
                let mut builder = ListBuilder::new(Float32Builder::new());
                for vec_opt in vectors {
                    if let Some(vec) = vec_opt {
                        builder.values().append_slice(&vec);
                        builder.append(true);
                    } else {
                        builder.append(false);
                    }
                }
                Ok(Arc::new(builder.finish()))
            }
            DataType::FixedSizeList(field, size)
                if matches!(*field.data_type(), DataType::Float32) =>
            {
                // Fixed-size vectors: FixedSizeList<Float32, N>
                let mut values = Vec::new();

                for vec_opt in &vectors {
                    if let Some(vec) = vec_opt {
                        if vec.len() != *size as usize {
                            return Err(DataFusionError::Execution(format!(
                                "Vector size mismatch: expected {}, got {}",
                                size,
                                vec.len()
                            )));
                        }
                        values.extend_from_slice(vec);
                    } else {
                        return Err(DataFusionError::Execution(
                            "Null vectors not supported in FixedSizeList".to_string(),
                        ));
                    }
                }

                let values_array = Arc::new(Float32Array::from(values));
                Ok(Arc::new(FixedSizeListArray::new(
                    field.clone(),
                    *size,
                    values_array,
                    None, // nulls
                )))
            }
            _ => Err(DataFusionError::Execution(format!(
                "Unsupported target type for vectors: {}",
                target_type
            ))),
        }
    }
}

#[async_trait]
impl SplitReader for PaxSplitReader {
    async fn read_split(
        &self,
        split: &FileSplit,
        projection: Option<&[usize]>,
        _batch_size: usize,
    ) -> DFResult<SendableRecordBatchStream> {
        let out_schema = match projection {
            Some(idx) => Arc::new(
                self.schema
                    .project(idx)
                    .map_err(|e| DataFusionError::Execution(format!("projection: {e}")))?,
            ),
            None => self.schema.clone(),
        };
        // Ranged path (default-OFF): prune whole blocks off the wire via the index
        // zone summary, fetch only survivors. Falls back to the whole-file read for
        // any segment without a locatable v2 zone index (mixed-read-safe).
        let batches = match if pax_ranged_read_enabled() {
            self.load_ranged(split, &out_schema).await?
        } else {
            None
        } {
            Some(batches) => batches,
            None => {
                let bytes = self.load_bytes(split).await?;
                self.decode_segment(&bytes, &out_schema)?
            }
        };
        if batches.is_empty() {
            Ok(Box::pin(EmptyRecordBatchStream::new(out_schema)))
        } else {
            Ok(Box::pin(RecordBatchStreamAdapter::new(
                out_schema,
                stream::iter(batches.into_iter().map(Ok)),
            )))
        }
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn engine_type(&self) -> EngineType {
        // No `Pax` variant exists (PAX is the SST block format); slice 2 may add one.
        EngineType::Sst
    }

    fn supports_filter_pushdown(&self) -> bool {
        true
    }

    fn supports_projection_pushdown(&self) -> bool {
        true
    }
}

// ---- free helpers (no `&self`, reusable / testable in isolation) ----

fn decode_i64(reader: &PaxBlockReader, cid: i32, name: &str) -> DFResult<Vec<Option<i64>>> {
    reader.decode_i64_stripe(cid).ok_or_else(|| {
        DataFusionError::Execution(format!("PAX: i64 decode failed for {name} (id {cid})"))
    })
}

fn decode_f64(reader: &PaxBlockReader, cid: i32, name: &str) -> DFResult<Vec<Option<f64>>> {
    reader.decode_f64_stripe(cid).ok_or_else(|| {
        DataFusionError::Execution(format!("PAX: f64 decode failed for {name} (id {cid})"))
    })
}

fn decode_str(reader: &PaxBlockReader, cid: i32, name: &str) -> DFResult<Vec<Option<String>>> {
    reader.decode_str_stripe(cid).ok_or_else(|| {
        DataFusionError::Execution(format!("PAX: str decode failed for {name} (id {cid})"))
    })
}

fn decode_bytes(reader: &PaxBlockReader, cid: i32, name: &str) -> DFResult<Vec<Option<Vec<u8>>>> {
    reader.decode_bytes_stripe(cid).ok_or_else(|| {
        DataFusionError::Execution(format!("PAX: bytes decode failed for {name} (id {cid})"))
    })
}

/// Decode the `props` tail (msgpack-serialized [`proximadb_records::ProximaTree`] per row)
/// into JSON-object strings — the `documents()` surface (TD-DOC-PUSHDOWN-1). Reuses the SAME
/// `ProximaTree → JSON` mapping the in-memory document scan uses (`proxima_tree_to_json_map`),
/// so a document read is identical whether served by the PAX scan or the MemTable path. A row
/// whose msgpack fails to parse yields an empty object (defensive; never panics).
fn decode_props_json(
    reader: &PaxBlockReader,
    cid: i32,
    name: &str,
) -> DFResult<Vec<Option<String>>> {
    use proximadb_records::ProximaTree;
    Ok(decode_bytes(reader, cid, name)?
        .into_iter()
        .map(|opt| {
            opt.map(|bytes| match rmp_serde::from_slice::<ProximaTree>(&bytes) {
                Ok(tree) => {
                    let map: serde_json::Map<String, serde_json::Value> =
                        crate::core::search::sql_value_filter::proxima_tree_to_json_map(&tree)
                            .into_iter()
                            .collect();
                    serde_json::to_string(&serde_json::Value::Object(map)).unwrap_or_default()
                }
                Err(_) => "{}".to_string(),
            })
        })
        .collect())
}

/// `true` iff the block MAY contain a row satisfying `pred` (i.e. NOT pruned).
/// Conservative: any unrecognized shape returns `true` (cannot prune). Operates on
/// any [`BlockZoneSource`] so the SAME logic prunes a decoded block AND the
/// index-only zone summary of the ranged read path.
fn block_may_contain(src: &dyn BlockZoneSource, cid: i32, pred: &ScalarPredicate) -> bool {
    use ScalarPredicate as P;
    use ScalarValue as V;
    match pred {
        // Point equality → i64 zone point / f64 degenerate range / string bloom+hash.
        P::Equal(V::Int64(v)) => src.may_contain_i64(cid, *v),
        P::Equal(V::Float64(v)) => src.range_overlaps_f64(cid, *v, *v),
        P::Equal(V::String(s)) => src.may_contain_str(cid, s),
        // i64 ranges: Gt/GtEq ⇒ [v, MAX]; Lt/LtEq ⇒ [MIN, v]; Between ⇒ [lo, hi].
        P::GreaterThan(V::Int64(v)) | P::GreaterThanOrEqual(V::Int64(v)) => {
            src.range_overlaps_i64(cid, *v, i64::MAX)
        }
        P::LessThan(V::Int64(v)) | P::LessThanOrEqual(V::Int64(v)) => {
            src.range_overlaps_i64(cid, i64::MIN, *v)
        }
        P::Between(V::Int64(lo), V::Int64(hi)) => src.range_overlaps_i64(cid, *lo, *hi),
        // f64 ranges.
        P::GreaterThan(V::Float64(v)) | P::GreaterThanOrEqual(V::Float64(v)) => {
            src.range_overlaps_f64(cid, *v, f64::INFINITY)
        }
        P::LessThan(V::Float64(v)) | P::LessThanOrEqual(V::Float64(v)) => {
            src.range_overlaps_f64(cid, f64::NEG_INFINITY, *v)
        }
        P::Between(V::Float64(lo), V::Float64(hi)) => src.range_overlaps_f64(cid, *lo, *hi),
        // NotEqual / IsNull / IsNotNull / In / Bool / Null / type-mismatched ⇒ cannot prune.
        _ => true,
    }
}

/// `true` for the canonical columns whose bounds the index [`BlockZoneSummary`]
/// carries inline (so Stage A can prune them with no per-block GET). Predicates on
/// any other column need the Stage-B footer read to prune.
fn is_canonical_zone_col(cid: i32) -> bool {
    use proximadb_block_format::col_id;
    matches!(
        cid,
        col_id::CREATED_AT
            | col_id::UPDATED_AT
            | col_id::VALID_FROM
            | col_id::VALID_TO
            | col_id::EDGE_WEIGHT
    )
}

/// Slice a block-relative metadata `range` from a buffer that begins at block-relative
/// `base`. Bounds-checked so a corrupt footer offset is a clean error, not a panic
/// (mirrors `RangedSegmentReader::slice_meta`).
fn slice_meta<'a>(buf: &'a [u8], base: u64, range: &std::ops::Range<u64>) -> DFResult<&'a [u8]> {
    let start = (range.start.saturating_sub(base)) as usize;
    let end = (range.end.saturating_sub(base)) as usize;
    buf.get(start..end).ok_or_else(|| {
        DataFusionError::Execution(format!(
            "PAX meta slice {range:?} outside buffer (base {base}, len {})",
            buf.len()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::expressions::{BinaryExpr, col, lit};
    use proximadb_block_format::{VectorQuant, col_id};
    use proximadb_records::ProximaRecord;

    use crate::storage::engines::sst::segment_format::write_pax_segment;

    fn record(oid: &str, tenant: &str, ts: i64) -> ProximaRecord {
        ProximaRecord {
            oid: oid.into(),
            tenant_id: tenant.into(),
            created_at_ns: ts,
            updated_at_ns: ts,
            ..Default::default()
        }
    }

    /// Write a REAL `.pax` segment (`[blocks][index][SEGMENT_MAGIC]`) with two
    /// records and return its bytes. This is the live SST format — a bare
    /// `PaxBlockReader::open(whole_file)` CRC-fails on it (the #706 latent bug
    /// this slice fixes).
    fn segment_bytes(r1: i64, r2: i64) -> Vec<u8> {
        let records = vec![record("r1", "t", r1), record("r2", "t", r2)];
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("seg.pax");
        write_pax_segment(&path, &records, "col", 0, VectorQuant::Auto, None, None)
            .expect("write_pax_segment");
        std::fs::read(&path).expect("read segment back")
    }

    /// `[created_at]` schema + its name→column_id map.
    fn created_at_schema() -> (SchemaRef, HashMap<String, i32>) {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "created_at",
            DataType::Int64,
            true,
        )]));
        let map = HashMap::from([(String::from("created_at"), col_id::CREATED_AT)]);
        (schema, map)
    }

    async fn new_reader(
        schema: SchemaRef,
        map: HashMap<String, i32>,
        filters: Vec<Arc<dyn PhysicalExpr>>,
    ) -> PaxSplitReader {
        let fs = Arc::new(FilesystemFactory::create_default().await.unwrap());
        PaxSplitReader::new(schema, fs, map, filters, None, None)
    }

    #[tokio::test]
    async fn decodes_real_segment_to_arrow() {
        // The segment's block has a created_at zone map of [1000, 3000].
        let bytes = segment_bytes(1000, 3000);
        let (schema, map) = created_at_schema();
        let reader = new_reader(schema.clone(), map, vec![]).await;

        let batches = reader.decode_segment(&bytes, &schema).unwrap();
        assert_eq!(batches.len(), 1, "one block ⇒ one batch");
        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 2);
        let ca = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(ca.value(0), 1000);
        assert_eq!(ca.value(1), 3000);
    }

    /// Flatten the `created_at` column of a batch set into a sorted `Vec<i64>` for
    /// order-independent row-set comparison.
    fn collect_created_at(batches: &[RecordBatch]) -> Vec<i64> {
        let mut v = Vec::new();
        for b in batches {
            let ca = b
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("created_at is Int64");
            for i in 0..b.num_rows() {
                v.push(ca.value(i));
            }
        }
        v.sort_unstable();
        v
    }

    /// Write a MULTI-block `.pax` segment (`n` rows, `created_at = (i+1)*1000`, a
    /// tiny target block so each block spans a small disjoint `created_at` range)
    /// and return its discovered [`FileSplit`] + on-disk size.
    async fn multiblock_split(
        dir: &std::path::Path,
        n: i64,
        target_block: usize,
        fs: &Arc<FilesystemFactory>,
    ) -> (crate::storage::formats::FileSplit, u64) {
        let records: Vec<_> = (0..n)
            .map(|i| record(&format!("r{i:04}"), "t", (i + 1) * 1000))
            .collect();
        let path = dir.join("seg.pax");
        write_pax_segment(
            &path,
            &records,
            "col",
            0,
            VectorQuant::Auto,
            Some(target_block),
            None, // destination_url
        )
        .expect("write_pax_segment");
        let file_len = std::fs::metadata(&path).unwrap().len();
        let base = format!("{}", dir.display());
        let splits =
            crate::datafusion::engine_adapters::pax_segment_locator::discover_pax_segments(
                &base, fs,
            )
            .await
            .unwrap();
        let split = splits
            .into_iter()
            .find(|s| s.file_path.ends_with("seg.pax"))
            .expect("discovered seg.pax split");
        (split, file_len)
    }

    /// The P-Pushdown gate (TD-DOC-PUSHDOWN-1): a selective predicate makes the
    /// ranged reader fetch only surviving blocks — `bytes_read < whole segment`
    /// with `range_gets > 0` — AND returns EXACTLY the rows the whole-file path
    /// would (block-level pruning parity).
    #[tokio::test]
    async fn ranged_read_prunes_blocks_off_the_wire() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let fs = Arc::new(FilesystemFactory::create_default().await.unwrap());
        let (split, file_len) = multiblock_split(tmp.path(), 200, 400, &fs).await;

        let (schema, map) = created_at_schema();
        // created_at >= 150_000 ⇒ only the upper ~quarter of blocks survive; the
        // lower blocks are pruned from the index zone summary before any body GET.
        let filter: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            col("created_at", schema.as_ref()).unwrap(),
            Operator::GtEq,
            lit(150_000i64),
        ));
        let reader = PaxSplitReader::new(schema.clone(), fs.clone(), map, vec![filter], None, None);

        let (ranged_rows, snap) = io_trace::scope(async {
            let batches = reader
                .load_ranged(&split, &schema)
                .await
                .unwrap()
                .expect("v2 zone index present ⇒ ranged path taken (not a fallback)");
            let rows = collect_created_at(&batches);
            (rows, io_trace::snapshot().expect("io_trace scope active"))
        })
        .await;

        assert!(
            snap.range_gets > 0,
            "ranged GETs issued: {}",
            snap.range_gets
        );
        assert!(
            snap.bytes_read < file_len,
            "ranged bytes_read {} must be < whole segment {file_len}",
            snap.bytes_read
        );

        // Parity vs the whole-file path on the SAME filter.
        let disk_path = split
            .file_path
            .strip_prefix("file://")
            .unwrap_or(&split.file_path);
        let whole = std::fs::read(disk_path).expect("read segment back");
        let baseline_rows = collect_created_at(&reader.decode_segment(&whole, &schema).unwrap());
        assert!(
            !ranged_rows.is_empty(),
            "some upper blocks survive the filter"
        );
        assert_eq!(
            ranged_rows, baseline_rows,
            "ranged rows must equal whole-file rows (block-prune parity)"
        );
    }

    /// Write a MULTI-block SHREDDED `.pax` segment: each record carries a props
    /// `age = (i+1)*1000` shredded to a typed i64 user column (`USER_BASE`), tiny target
    /// block ⇒ blocks span disjoint age ranges. Returns the discovered split + size.
    async fn multiblock_shredded_split(
        dir: &std::path::Path,
        n: i64,
        target_block: usize,
        fs: &Arc<FilesystemFactory>,
    ) -> (crate::storage::formats::FileSplit, u64) {
        use proximadb_block_format::{BlockCompression, BlockMode};
        use proximadb_data_model::ProximaValue as PV;
        use proximadb_records::ProximaTreeNode as Node;
        use proximadb_storage_common::pax_block::PaxSegmentWriter;

        let path = dir.join("shred.pax");
        let mut w = PaxSegmentWriter::new(
            &path,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            0, // embedding_count — pure document/relational segment (no vectors)
            Some(target_block),
        )
        .with_shred_spec(vec![("age".to_string(), col_id::USER_BASE)]);
        for i in 0..n {
            let mut rec = record(&format!("r{i:04}"), "t", (i + 1) * 1000);
            rec.props
                .insert("age".into(), Node::Value(PV::Int64((i + 1) * 1000)));
            w.add_record(&rec).unwrap();
        }
        w.finish().unwrap();
        let file_len = std::fs::metadata(&path).unwrap().len();
        let base = format!("{}", dir.display());
        let splits =
            crate::datafusion::engine_adapters::pax_segment_locator::discover_pax_segments(
                &base, fs,
            )
            .await
            .unwrap();
        let split = splits
            .into_iter()
            .find(|s| s.file_path.ends_with("shred.pax"))
            .expect("discovered shred.pax split");
        (split, file_len)
    }

    /// Layer 2 (TD-DOC-PUSHDOWN-1): a predicate on a SHREDDED user column — which the
    /// index zone summary cannot evaluate — prunes block BODIES via the per-block footer
    /// read (Stage B). `bytes_read < whole segment` with `range_gets > 0`, rows identical
    /// to the whole-file path.
    #[tokio::test]
    async fn ranged_footer_prune_skips_bodies_on_user_column() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let fs = Arc::new(FilesystemFactory::create_default().await.unwrap());
        // Realistically LARGE blocks (~130 rows/block): per-block metadata is a small
        // fraction of the body, so pruning a block's body off the wire is a clear net
        // win over the footer read — the regime real document segments live in.
        let (split, file_len) = multiblock_shredded_split(tmp.path(), 2000, 16384, &fs).await;

        let schema = Arc::new(Schema::new(vec![Field::new("age", DataType::Int64, true)]));
        let map = HashMap::from([(String::from("age"), col_id::USER_BASE)]);
        // age = (i+1)*1000 ∈ [1000, 2_000_000]; keep only the top ~25% of blocks.
        let filter: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            col("age", schema.as_ref()).unwrap(),
            Operator::GtEq,
            lit(1_500_000i64),
        ));
        let reader = PaxSplitReader::new(schema.clone(), fs.clone(), map, vec![filter], None, None);
        assert!(
            reader.needs_footer_prune(),
            "a shredded user-column predicate must engage Stage B footer pruning"
        );

        let (rows, snap) = io_trace::scope(async {
            let batches = reader
                .load_ranged(&split, &schema)
                .await
                .unwrap()
                .expect("v2 zone index present ⇒ ranged path taken");
            (collect_created_at(&batches), io_trace::snapshot().unwrap())
        })
        .await;

        assert!(
            snap.range_gets > 0,
            "ranged GETs issued: {}",
            snap.range_gets
        );
        assert!(
            snap.bytes_read < file_len,
            "footer-prune bytes_read {} must be < whole segment {file_len} (user-column pruning)",
            snap.bytes_read
        );

        let disk = split
            .file_path
            .strip_prefix("file://")
            .unwrap_or(&split.file_path);
        let whole = std::fs::read(disk).expect("read segment back");
        let baseline = collect_created_at(&reader.decode_segment(&whole, &schema).unwrap());
        assert!(!rows.is_empty(), "some upper age blocks survive the filter");
        assert_eq!(
            rows, baseline,
            "footer-prune rows must equal whole-file rows (user-column prune parity)"
        );
    }

    #[tokio::test]
    async fn zone_map_prunes_disjoint_block() {
        let bytes = segment_bytes(1000, 3000);
        let (schema, map) = created_at_schema();

        // `created_at >= 4000` — disjoint from [1000, 3000] ⇒ block prunes ⇒ no batches.
        let ge_4000: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            col("created_at", schema.as_ref()).unwrap(),
            Operator::GtEq,
            lit(4000i64),
        ));
        let reader = new_reader(schema.clone(), map, vec![ge_4000]).await;
        assert!(
            reader.decode_segment(&bytes, &schema).unwrap().is_empty(),
            "block [1000,3000] must prune under created_at >= 4000"
        );

        // `created_at <= 500` — disjoint ⇒ prune.
        let le_500: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            col("created_at", schema.as_ref()).unwrap(),
            Operator::LtEq,
            lit(500i64),
        ));
        let reader = new_reader(schema.clone(), created_at_schema().1, vec![le_500]).await;
        assert!(
            reader.decode_segment(&bytes, &schema).unwrap().is_empty(),
            "block [1000,3000] must prune under created_at <= 500"
        );

        // `created_at >= 2000` — overlaps [1000, 3000] ⇒ NOT pruned ⇒ ≥1 batch.
        let ge_2000: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            col("created_at", schema.as_ref()).unwrap(),
            Operator::GtEq,
            lit(2000i64),
        ));
        let reader = new_reader(schema.clone(), created_at_schema().1, vec![ge_2000]).await;
        assert!(
            !reader.decode_segment(&bytes, &schema).unwrap().is_empty(),
            "block [1000,3000] must NOT prune under created_at >= 2000 (overlaps)"
        );
    }

    #[tokio::test]
    async fn empty_filter_set_never_prunes() {
        let bytes = segment_bytes(1000, 3000);
        let (schema, map) = created_at_schema();
        let reader = new_reader(schema.clone(), map, vec![]).await;
        // No predicates ⇒ never pruned, regardless of data.
        assert!(!reader.decode_segment(&bytes, &schema).unwrap().is_empty());
    }
}
