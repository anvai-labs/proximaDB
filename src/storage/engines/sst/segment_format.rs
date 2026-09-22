//! Mixed-format vector-segment read primitives (P3 Phase A).
//!
//! ProximaDB's vector engines (SST/HELIX/SWIFT) historically persist segments as
//! row-based [`ProximaDataBlock`] (raw f32). The RaBitQ path (P3) stores vectors in
//! the columnar PAX segment format, which carries quantized codes and enables ~30×
//! ANN. To migrate additively — mixed-read-safe and default-OFF — a reader must
//! recognise both formats straight from the on-disk bytes.
//!
//! This module is the read-side foundation: a byte-based [`SegmentFormat::detect`]
//! and a [`read_segment_records`] router that decodes either format back to
//! `Vec<ProximaRecord>`, reusing the canonical PAX inverse
//! ([`PaxSegmentScanner::read_records`]). It writes nothing and changes no live read
//! path. The wiring into the cold-search / compaction / recovery sites (Phase A.2)
//! and the PAX write side (Phase B) build on top of this primitive.
//!
//! Detection is unambiguous by magic: a PAX segment starts with the PAX block magic
//! `PBLK` (`0x50…`) and ends with the segment magic `PAXSEG01`; a `ProximaDataBlock`
//! starts with the columnar version byte `0x01` or a compression marker
//! `0x02..=0x0E` — disjoint from `PBLK`, so the legacy path is never mis-routed.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use proximadb_block_format::{BlockCompression, BlockMode, RankMetric, VectorQuant, col_id};
use proximadb_cache::{CacheKind, L2CacheStats, L2Class, PersistentByteStore};
use proximadb_records::ProximaRecord;
use proximadb_storage_common::pax_block::{
    PaxSegmentScanner, PaxSegmentWriter, SEGMENT_MAGIC, ScanPredicate, SegmentMeta,
};
// These are now used only by this module's `#[cfg(test)]` suite — the
// detection (`SegmentFormat` / `is_pax_segment`) and the PAX-decode body they
// served moved down to `proximadb-storage-common`. Gated to test so the non-test
// lib build stays warning-clean under clippy `-D warnings` (the CI gate).
#[cfg(test)]
use proximadb_block_format::BLOCK_MAGIC;
use proximadb_storage_common::segment_layout::SEG_HEADER_PREFIX_LEN;
#[cfg(test)]
use proximadb_storage_common::segment_layout::{
    SEG_LAYOUT_VERSION, SegmentHeaderPrefix, is_coalesced_segment,
};

use crate::storage::engines::core::formats::proximablocks::ProximaDataBlock;
use crate::storage::engines::sst::survivor_range_cache::SurvivorRangeCache;

// `SegmentFormat` (detection + `is_pax_segment`) now lives in the format layer —
// `proximadb_storage_common::segment_layout` — so the mixed-format detection
// primitive is unit-testable without linking the root crate. Re-exported here so
// every existing `crate::storage::engines::sst::segment_format::SegmentFormat`
// reference (and the router below) resolves unchanged (behavior-neutral move).
pub use proximadb_storage_common::segment_layout::SegmentFormat;

// TD-USUB-5 / ADR-094 §4: the decode seam. `RecordReader` and its two
// format-layer implementations live beside the formats they read; re-exported
// here so the router and its callers keep one import path.
pub use proximadb_storage_common::segment_reader::{
    ParquetRecordReader, PaxRecordReader, RecordReader, SegmentReadContext,
};

/// Reads legacy row-based [`ProximaDataBlock`] segments (raw f32).
///
/// This implementation lives in the root crate rather than beside its two
/// siblings in `proximadb-storage-common` because [`ProximaDataBlock`] is defined
/// in `proximadb-engine-core`, which sits *above* the format layer. The trait is
/// the seam that lets all three readers be selected uniformly anyway.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProximaBlocksRecordReader;

impl RecordReader for ProximaBlocksRecordReader {
    fn format(&self) -> SegmentFormat {
        SegmentFormat::ProximaBlocks
    }

    fn read_records(
        &self,
        bytes: &[u8],
        _ctx: &SegmentReadContext<'_>,
    ) -> Result<Vec<ProximaRecord>> {
        // The legacy block is self-describing — it carries its own column names,
        // embeddings and tenant — so the catalog-derived context is unused.
        Ok(ProximaDataBlock::deserialize(bytes, None)?.records)
    }
}

/// The dispatch table: one [`RecordReader`] per [`SegmentFormat`].
///
/// This is the substance of TD-USUB-5's compute-seam change — the decode rule for
/// a format is now addressable on its own, so adding a format is adding an arm
/// here plus a magic in `SegmentFormat::detect`, never editing a decode body.
///
/// The `match` is deliberately exhaustive (no `_` arm) so a future
/// [`SegmentFormat`] variant is a compile error here rather than a silent
/// fall-back to the legacy decoder.
pub fn record_reader_for(format: SegmentFormat) -> Box<dyn RecordReader> {
    match format {
        SegmentFormat::Pax => Box::new(PaxRecordReader),
        SegmentFormat::Parquet => Box::new(ParquetRecordReader),
        SegmentFormat::ProximaBlocks => Box::new(ProximaBlocksRecordReader),
    }
}

/// Decode a persisted vector segment back to records, routing on the detected format.
///
/// Detection ([`SegmentFormat::detect`]) picks the format from the bytes' magic;
/// [`record_reader_for`] picks the reader. PAX decodes via the format-layer
/// inverse (`read_pax_segment_records`), Parquet via the canonical
/// `parquet_bytes_to_record_batches` + Arrow→records mapping, legacy via
/// [`ProximaDataBlock::deserialize`]. This root-level router is the single
/// mixed-read entry called from the cold-search, compaction, and recovery paths.
///
/// Its signature and its behaviour on `PBLK`/`PXH1`/legacy bytes are unchanged by
/// the move to trait dispatch — the same detection feeds the same decoders. The
/// one behavioural difference is that Parquet bytes, which previously fell to the
/// legacy default and were handed to a decoder that cannot read them, now reach a
/// reader that can.
///
/// `embedding_model_ids` / `user_column_keys` are the collection's schema keys used
/// to reconstruct PAX records positionally (empty slices = best-effort defaults);
/// they are ignored for the legacy and Parquet formats, which are self-describing.
///
/// `tenant_ctx` is the segment's owning tenant (from the catalog/path); it is
/// stamped onto rows whose tenant column was dropped (catalog-resolution) and is
/// ignored when the column is present. `None` keeps stored values verbatim.
pub fn read_segment_records(
    bytes: &[u8],
    embedding_model_ids: &[String],
    user_column_keys: &[String],
    tenant_ctx: Option<&str>,
) -> Result<Vec<ProximaRecord>> {
    let ctx = SegmentReadContext::new(embedding_model_ids, user_column_keys, tenant_ctx);
    record_reader_for(SegmentFormat::detect(bytes)).read_records(bytes, &ctx)
}

/// Write `records` as a PAX vector segment at `path` — the write-side inverse of
/// [`read_segment_records`] for the PAX format, via the canonical [`PaxSegmentWriter`]
/// (no hand-rolled encoder, per the storage-format-migration mandate). Phase B's
/// flag-gated SST flush arm calls this against the local staging path; the resulting
/// file is detected as [`SegmentFormat::Pax`] and reads back through the same router.
///
/// `embedding_count` is the collection's embedding-modality count (≥1). `quant` selects
/// the vector quantization strategy (P3 Phase D): `VectorQuant::Auto` keeps the env
/// default (SQ8 unless `PROXIMADB_VECTOR_RABITQ_ENABLE`), `RaBitQ` writes ~30× binary codes.
///
/// `target_block` is the optional target block size in bytes (TD-156 / ADR-026
/// configurable geometry). `None` keeps the writer default; a larger value (e.g.
/// 8-16 MiB for object storage) coalesces rows into fewer blocks, cutting the
/// per-block ranged-GET count — the fragmentation lever measured by the
/// footer-cache economics harness. Mixed-read-safe: block size is a per-segment
/// write choice and is irrelevant to the magic-detected read router.
pub fn write_pax_segment(
    path: &Path,
    records: &[ProximaRecord],
    collection_id: &str,
    embedding_count: usize,
    quant: VectorQuant,
    target_block: Option<usize>,
    destination_url: Option<&str>,
) -> Result<SegmentMeta> {
    write_pax_segment_with_f32_tier(
        path,
        records,
        collection_id,
        embedding_count,
        quant,
        false,
        target_block,
        destination_url,
    )
}

/// Like [`write_pax_segment`] but optionally emits the exact-f32 tier (P3 Phase
/// D). When `f32_tier` is true, each embedding also gets a co-located raw-f32
/// stripe at `col_id::F32_TIER_BASE + i` for an exact
/// final rerank / `include_vectors`. The flush path calls this with the resolved
/// `pax_f32_tier` opt-in; compaction/tests use [`write_pax_segment`] (no tier).
pub fn write_pax_segment_with_f32_tier(
    path: &Path,
    records: &[ProximaRecord],
    collection_id: &str,
    embedding_count: usize,
    quant: VectorQuant,
    f32_tier: bool,
    target_block: Option<usize>,
    destination_url: Option<&str>,
) -> Result<SegmentMeta> {
    write_pax_segment_full(
        path,
        records,
        collection_id,
        embedding_count,
        quant,
        VectorQuant::Sq8,
        f32_tier,
        target_block,
        destination_url,
    )
}

/// Full PAX segment write with configurable tier-1 quant, tier-2 rerank quant,
/// and optional f32 tier. This is the canonical write entry point; the legacy
/// `write_pax_segment_with_f32_tier` delegates with `Sq8` rerank (the default).
#[allow(clippy::too_many_arguments)]
pub fn write_pax_segment_full(
    path: &Path,
    records: &[ProximaRecord],
    collection_id: &str,
    embedding_count: usize,
    quant: VectorQuant,
    rerank_quant: VectorQuant,
    f32_tier: bool,
    target_block: Option<usize>,
    destination_url: Option<&str>,
) -> Result<SegmentMeta> {
    write_pax_segment_full_with_layout(
        path,
        records,
        collection_id,
        embedding_count,
        quant,
        rerank_quant,
        f32_tier,
        target_block,
        rg_layout_enabled(),
        &[],
        destination_url,
    )
}

/// Like [`write_pax_segment_full`] with an explicit row-group Region D switch
/// (TD-PAXRG-1) — the flush seam resolves the per-collection tag + env and
/// passes the result here. No-ops for non-RaBitQ-coalesced writes.
#[allow(clippy::too_many_arguments)]
pub fn write_pax_segment_full_with_layout(
    path: &Path,
    records: &[ProximaRecord],
    collection_id: &str,
    embedding_count: usize,
    quant: VectorQuant,
    rerank_quant: VectorQuant,
    f32_tier: bool,
    target_block: Option<usize>,
    rg_layout: bool,
    shred_spec: &[(String, i32)],
    destination_url: Option<&str>,
) -> Result<SegmentMeta> {
    Ok(write_pax_segment_full_internal(
        path,
        records,
        collection_id,
        embedding_count,
        quant,
        rerank_quant,
        f32_tier,
        target_block,
        rg_layout,
        None,
        shred_spec,
        destination_url,
    )?
    .meta)
}

/// Cache-seed variant with an explicit row-group Region D switch (TD-PAXRG-1)
/// — the flush seam resolves the per-collection tag + env and passes it here.
#[allow(clippy::too_many_arguments)]
pub fn write_pax_segment_full_with_cache_seed_and_layout(
    path: &Path,
    records: &[ProximaRecord],
    collection_id: &str,
    embedding_count: usize,
    quant: VectorQuant,
    rerank_quant: VectorQuant,
    f32_tier: bool,
    target_block: Option<usize>,
    rg_layout: bool,
    include_sq8: bool,
    shred_spec: &[(String, i32)],
    destination_url: Option<&str>,
) -> Result<proximadb_storage_common::pax_block::PaxSegmentWrite> {
    write_pax_segment_full_internal(
        path,
        records,
        collection_id,
        embedding_count,
        quant,
        rerank_quant,
        f32_tier,
        target_block,
        rg_layout,
        Some(include_sq8),
        shred_spec,
        destination_url,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn write_pax_segment_full_with_cache_seed(
    path: &Path,
    records: &[ProximaRecord],
    collection_id: &str,
    embedding_count: usize,
    quant: VectorQuant,
    rerank_quant: VectorQuant,
    f32_tier: bool,
    target_block: Option<usize>,
    include_sq8: bool,
    destination_url: Option<&str>,
) -> Result<proximadb_storage_common::pax_block::PaxSegmentWrite> {
    write_pax_segment_full_with_cache_seed_and_layout(
        path,
        records,
        collection_id,
        embedding_count,
        quant,
        rerank_quant,
        f32_tier,
        target_block,
        rg_layout_enabled(),
        include_sq8,
        &[],
        destination_url,
    )
}

#[allow(clippy::too_many_arguments)]
fn write_pax_segment_full_internal(
    path: &Path,
    records: &[ProximaRecord],
    collection_id: &str,
    embedding_count: usize,
    quant: VectorQuant,
    rerank_quant: VectorQuant,
    f32_tier: bool,
    target_block: Option<usize>,
    rg_layout: bool,
    capture_sq8: Option<bool>,
    shred_spec: &[(String, i32)],
    destination_url: Option<&str>,
) -> Result<proximadb_storage_common::pax_block::PaxSegmentWrite> {
    // TD-RDSTRAT-5 S1 / TD-WLP-4 (default ON, kill-switch
    // `PROXIMADB_PAX_BLOCK_CLUSTER=0`): reorder records by the model-free
    // sign-code bootstrap so spatially-close vectors co-locate into the same
    // block, and compute each block's centroid+radius for the Vector Object
    // Economy directory. Reordering is result-preserving (the reader
    // ranks/dedups by distance + OID). Compaction upgrades the ordering to
    // PCA+IVF via `write_pax_segment_compacted`.
    let cluster = crate::storage::engines::sst::block_cluster::block_cluster_enabled();
    // ADR-065 Region B: the coalesced layout hoists SQ8 into a region whose
    // survivor fetch is row-granular — it needs a locality-preserving order so the
    // RaBitQ survivors (and top-k rows) co-locate. Use Morton/Z-order over the
    // segment-level SQ8 codes (denoised, projection-free, flush-safe). The
    // non-coalesced path keeps the sign-bit bootstrap (block granularity absorbs
    // scattering, so it doesn't need the stronger order).
    let coalesced = quant == VectorQuant::RaBitQ && coalesced_rabitq_enabled();
    let plan = if cluster {
        // TD-WLP-4/WLP-9 eval opt-in (`PROXIMADB_PAX_FLUSH_CLUSTER=ivf`): apply
        // the compaction-grade PCA+IVF re-cluster at flush instead of the
        // bootstrap, so clustering quality is measurable without the (unwired)
        // flush→compaction scheduler. Default OFF ⇒ bootstrap.
        if crate::storage::engines::sst::block_cluster::flush_cluster_ivf() {
            crate::storage::engines::sst::block_cluster::cluster_plan_pca_ivf(
                records,
                0,
                destination_url,
            )
        } else if coalesced {
            crate::storage::engines::sst::block_cluster::cluster_order_sq8_morton(records, 0).map(
                |order| crate::storage::engines::sst::block_cluster::ClusterPlan {
                    order,
                    runs: Vec::new(),
                },
            )
        } else {
            crate::storage::engines::sst::block_cluster::cluster_plan(records, 0)
        }
    } else {
        None
    };
    write_pax_segment_ordered(
        path,
        records,
        collection_id,
        embedding_count,
        quant,
        rerank_quant,
        f32_tier,
        target_block,
        rg_layout,
        cluster,
        plan,
        None, // two-level is compaction-only (TD-RDSTRAT-8)
        capture_sq8,
        shred_spec, // TD-FPRUNE-1 M1: SST flush shreds declared filterable tags
    )
}

/// TD-WLP-4 (ADR-061 D3): the **compaction** write entry point — identical to
/// [`write_pax_segment_full`] except the record ordering is the PCA+IVF
/// re-cluster (`cluster_order_pca_ivf`) instead of the L0 sign-code bootstrap.
/// Compaction is the re-cluster event: the merged batch is large enough to
/// train a write-time PCA, and same-cell rows co-locate into blocks whose
/// centroids+radii make the read-side prune effective. Honors the same
/// kill-switch (`PROXIMADB_PAX_BLOCK_CLUSTER=0` ⇒ insertion order, no
/// centroids). Gating by profile happens at compaction *arming*
/// (`resolve_compaction_armed`): Churn collections never schedule compaction,
/// so this path only serves AppendBulk (or explicit `compaction:on`) work.
#[allow(clippy::too_many_arguments)]
pub fn write_pax_segment_compacted(
    path: &Path,
    records: &[ProximaRecord],
    collection_id: &str,
    embedding_count: usize,
    quant: VectorQuant,
    rerank_quant: VectorQuant,
    f32_tier: bool,
    target_block: Option<usize>,
    destination_url: Option<&str>,
) -> Result<SegmentMeta> {
    // TD-PAXRG-1: compaction-side per-collection plumbing is a recorded
    // follow-up; the env gate governs compacted output for now.
    write_pax_segment_compacted_internal(
        path,
        records,
        collection_id,
        embedding_count,
        quant,
        rerank_quant,
        f32_tier,
        target_block,
        rg_layout_enabled(),
        None,
        &[],
        destination_url,
    )
    .map(|write| write.meta)
}

#[allow(clippy::too_many_arguments)]
pub fn write_pax_segment_compacted_with_cache_seed(
    path: &Path,
    records: &[ProximaRecord],
    collection_id: &str,
    embedding_count: usize,
    quant: VectorQuant,
    rerank_quant: VectorQuant,
    f32_tier: bool,
    target_block: Option<usize>,
    include_sq8: bool,
    destination_url: Option<&str>,
) -> Result<proximadb_storage_common::pax_block::PaxSegmentWrite> {
    write_pax_segment_compacted_internal(
        path,
        records,
        collection_id,
        embedding_count,
        quant,
        rerank_quant,
        f32_tier,
        target_block,
        rg_layout_enabled(),
        Some(include_sq8),
        &[],
        destination_url,
    )
}

/// TD-FPRUNE-1 (A1): [`write_pax_segment_compacted`] plus a P-Shred spec so
/// compaction preserves shredded filterable-tag columns (see
/// [`write_pax_segment_compacted_with_cache_seed_shredded`]).
#[allow(clippy::too_many_arguments)]
pub fn write_pax_segment_compacted_shredded(
    path: &Path,
    records: &[ProximaRecord],
    collection_id: &str,
    embedding_count: usize,
    quant: VectorQuant,
    rerank_quant: VectorQuant,
    f32_tier: bool,
    target_block: Option<usize>,
    shred_spec: &[(String, i32)],
    destination_url: Option<&str>,
) -> Result<SegmentMeta> {
    Ok(write_pax_segment_compacted_internal(
        path,
        records,
        collection_id,
        embedding_count,
        quant,
        rerank_quant,
        f32_tier,
        target_block,
        rg_layout_enabled(),
        None,
        shred_spec,
        destination_url,
    )?
    .meta)
}

/// TD-FPRUNE-1 (A1): [`write_pax_segment_compacted_with_cache_seed`] plus a
/// P-Shred spec so COMPACTION preserves the shredded filterable-tag user-columns
/// (min/max + bloom + self-describing footer field-map) — otherwise a compacted
/// L1/L2 segment drops them and filtered footer-pruning goes dark for it. The
/// spec is rebuilt from an input segment's self-describing footer `shred_field_map`
/// (no catalog/schema needed); merged records keep their `props` (ADR-055), so
/// re-shred reproduces the columns. Empty spec ⇒ byte-identical (legacy inputs).
#[allow(clippy::too_many_arguments)]
pub fn write_pax_segment_compacted_with_cache_seed_shredded(
    path: &Path,
    records: &[ProximaRecord],
    collection_id: &str,
    embedding_count: usize,
    quant: VectorQuant,
    rerank_quant: VectorQuant,
    f32_tier: bool,
    target_block: Option<usize>,
    include_sq8: bool,
    shred_spec: &[(String, i32)],
    destination_url: Option<&str>,
) -> Result<proximadb_storage_common::pax_block::PaxSegmentWrite> {
    write_pax_segment_compacted_internal(
        path,
        records,
        collection_id,
        embedding_count,
        quant,
        rerank_quant,
        f32_tier,
        target_block,
        rg_layout_enabled(),
        Some(include_sq8),
        shred_spec,
        destination_url,
    )
}

#[allow(clippy::too_many_arguments)]
fn write_pax_segment_compacted_internal(
    path: &Path,
    records: &[ProximaRecord],
    collection_id: &str,
    embedding_count: usize,
    quant: VectorQuant,
    rerank_quant: VectorQuant,
    f32_tier: bool,
    target_block: Option<usize>,
    rg_layout: bool,
    capture_sq8: Option<bool>,
    shred_spec: &[(String, i32)],
    destination_url: Option<&str>,
) -> Result<proximadb_storage_common::pax_block::PaxSegmentWrite> {
    use crate::storage::engines::sst::block_cluster;
    let cluster = crate::storage::engines::sst::block_cluster::block_cluster_enabled();
    // TD-RDSTRAT-8 rev 3: compaction is the ONLY write path that emits the
    // persisted-IVF-probe (v3) layout. Training is default ON;
    // `PROXIMADB_PAX_WRITE_A0_TRAIN=0` is the kill-switch. Production flush stays
    // sign-bit L0 (IVF-at-flush measured ~80× flush cost, dropped), and large
    // corpora reach v3 on their normal compaction cadence with no migration event.
    // The probe plan persists the SAME IOP-derived PCA/IVF plan the single-level
    // path computes (no second quantizer); it falls back to the plain single-level
    // plan whenever the model can't be trained (fail-safe, never a worse segment).
    let coalesced = quant == VectorQuant::RaBitQ && coalesced_rabitq_enabled();
    let t_cluster_plan = std::time::Instant::now();
    let mut probe_model = None;
    let plan = if cluster {
        if coalesced && block_cluster::ivf_probe_enabled() {
            // TD-FPRUNE-1 M2 (opt-in): when the segment carries a shredded
            // filterable tag and the partition-layout gate is on, cluster
            // partition-first (cells become partition-homogeneous → uncorrelated
            // tags prune via the M3 probe). Falls back to the global probe plan
            // when the tag is absent/degenerate — same A0 format either way.
            let two_level = block_cluster::partition_layout_enabled()
                .then(|| shred_spec.first())
                .flatten()
                .and_then(|(tag, _)| {
                    block_cluster::cluster_plan_ivf_probe_partitioned(
                        records,
                        0,
                        tag,
                        destination_url,
                    )
                })
                .or_else(|| block_cluster::cluster_plan_ivf_probe(records, 0, destination_url));
            match two_level {
                Some(tl) => {
                    probe_model = Some(tl.model);
                    Some(tl.plan)
                }
                None => block_cluster::cluster_plan_pca_ivf(records, 0, destination_url),
            }
        } else {
            block_cluster::cluster_plan_pca_ivf(records, 0, destination_url)
        }
    } else {
        None
    };
    // TD-COMPACT-1 S1: cluster-plan wall clock (encode/finalize are timed inside
    // `write_pax_segment_ordered` under the same gate).
    if pax_write_trace() {
        eprintln!(
            "[PAX write] n={n} kind=compaction | cluster_plan {ms:.0} ms",
            n = records.len(),
            ms = t_cluster_plan.elapsed().as_secs_f64() * 1e3,
        );
    }
    write_pax_segment_ordered(
        path,
        records,
        collection_id,
        embedding_count,
        quant,
        rerank_quant,
        f32_tier,
        target_block,
        rg_layout,
        cluster,
        plan,
        probe_model,
        capture_sq8,
        shred_spec, // TD-FPRUNE-1 A1: preserve shred columns across compaction
    )
}

/// Construct the PAX writer used after the external spill pipeline has already
/// resolved MVCC and produced final IVF cell order. This is the same writer
/// policy as [`write_pax_segment_compacted_internal`] without re-materializing
/// records or fitting a second model.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pax_spill_compaction_writer(
    path: &Path,
    scratch_root: &Path,
    collection_id: &str,
    embedding_count: usize,
    rerank_quant: VectorQuant,
    f32_tier: bool,
    rg_layout: bool,
    target_block: Option<usize>,
    expected_rows: usize,
    two_level: Option<proximadb_storage_common::coarse_directory::CoarseModel>,
) -> Result<PaxSegmentWriter> {
    if !coalesced_rabitq_enabled() {
        anyhow::bail!("local-spill compaction requires the coalesced RaBitQ PAX layout");
    }
    let cluster = crate::storage::engines::sst::block_cluster::block_cluster_enabled();
    let mut writer = PaxSegmentWriter::new(
        path,
        BlockMode::Pax,
        BlockCompression::Zstd,
        collection_id,
        0,
        embedding_count.max(1),
        target_block,
    )
    .with_quant(VectorQuant::RaBitQ)
    // TD-PAXRG-1: exact tier from the INTERSECTION rule — the caller resolves
    // `pax_inputs_have_f32_tier(&task.input_files)`; under rg_layout the tier
    // emits as Region C on the spill twin (f32 spool + header/copy).
    .with_f32_tier(f32_tier)
    .with_rerank_quant(rerank_quant)
    .with_lossless_clustered(lossless_clustered_enabled() && two_level.is_some())
    .with_lossless_scalar(lossless_scalar_enabled())
    // The spill path is itself default-off. Preserve canonical MVCC sequence
    // in its output so every later compaction has an exact version authority.
    .with_record_version(true)
    .with_block_centroids(cluster)
    .with_coalesced_rabitq(true)
    // TD-PAXRG-1: the spill twin mirrors the in-memory writer's rg flag
    // (caller-resolved: per-collection tag > env > default) so compaction
    // output matches flush output for the same gate state.
    .with_rg_layout(rg_layout)
    .with_expected_rows(expected_rows);
    // TD-PAXRG-1: the OID resolver capture is UNCONDITIONAL — it is additive
    // (a small footer region; absent ⇒ legacy behavior) and row-group
    // Region D's fail-safe requires it for point lookups. Gating it behind a
    // rollout feature made default-ON rg unwritable in feature-off builds.
    writer = writer.with_oid_resolver(true);
    if let Some(model) = two_level {
        writer = writer.with_two_level(model);
    }
    writer.with_local_spill(scratch_root)
}

/// Shared writer loop for the flush (bootstrap-ordered) and compaction
/// (IVF-ordered) entry points.
#[allow(clippy::too_many_arguments)]
fn write_pax_segment_ordered(
    path: &Path,
    records: &[ProximaRecord],
    collection_id: &str,
    embedding_count: usize,
    quant: VectorQuant,
    rerank_quant: VectorQuant,
    f32_tier: bool,
    target_block: Option<usize>,
    rg_layout: bool,
    cluster: bool,
    plan: Option<crate::storage::engines::sst::block_cluster::ClusterPlan>,
    two_level: Option<proximadb_storage_common::coarse_directory::CoarseModel>,
    capture_sq8: Option<bool>,
    shred_spec: &[(String, i32)],
) -> Result<proximadb_storage_common::pax_block::PaxSegmentWrite> {
    let lossless_clustered = lossless_clustered_enabled() && plan.is_some();
    let lossless_scalar = lossless_scalar_enabled();
    let mut writer = PaxSegmentWriter::new(
        path,
        BlockMode::Pax,
        BlockCompression::Zstd,
        collection_id,
        0, // schema_fingerprint — derived from the catalog schema in a later phase
        embedding_count.max(1),
        target_block,
    )
    .with_quant(quant)
    .with_f32_tier(f32_tier)
    .with_rerank_quant(rerank_quant)
    .with_lossless_clustered(lossless_clustered)
    .with_lossless_scalar(lossless_scalar)
    .with_shred_spec(shred_spec.to_vec())
    .with_block_centroids(cluster)
    // ADR-062 / TD-RDSTRAT-6: hoist the RaBitQ binary tier into a coalesced
    // file-level header region for RaBitQ-quantized writes (default ON per the
    // ADR-061 pre-GA in-place amendment; `PROXIMADB_PAX_COALESCED_RABITQ=0` opts
    // out to the legacy in-block RaBitQ layout for mixed-read / measurement).
    .with_coalesced_rabitq(quant == VectorQuant::RaBitQ && coalesced_rabitq_enabled())
    // TD-PAXRG-1: row-group Region D — requires the coalesced layout
    // (Regions A/B are the premise of the wedge); resolved upstream
    // (per-collection tag > env > default OFF).
    .with_rg_layout(rg_layout && quant == VectorQuant::RaBitQ && coalesced_rabitq_enabled())
    .with_expected_rows(records.len());
    // TD-DELVEC-1 WI-3a: capture the OID→position resolver (footer region, WI-2c)
    // so a cold-resident delete (WI-3b) can set a deletion-vector bit without
    // rewriting the segment. Capture is UNCONDITIONAL (additive + mixed-read-
    // safe) — row-group Region D's fail-safe requires it in every build.
    writer = writer.with_oid_resolver(true);
    // TD-RDSTRAT-8 rev 3: the persisted IVF probe directory (v3 layout) — the
    // plan's runs are its IOP-derived cells, so the writer pads blocks at the
    // same boundaries the model's cell_rows describe (a cell = whole D-blocks).
    if let Some(model) = two_level {
        writer = writer.with_two_level(model);
    }
    let trace = pax_write_trace();
    let t_encode = std::time::Instant::now();
    // Consume the ordering plan here so its row-order/runs/model allocations are
    // released before the writer's Region A/B finalization peak.
    match plan {
        Some(plan) => {
            let mut next_run = 1usize;
            for (ordered_row, i) in plan.order.into_iter().enumerate() {
                if plan
                    .runs
                    .get(next_run)
                    .is_some_and(|run| run.start_row == ordered_row)
                {
                    writer.start_cluster_run();
                    next_run += 1;
                }
                writer.add_record(&records[i])?;
            }
        }
        None => {
            for record in records {
                writer.add_record(record)?;
            }
        }
    }
    let encode_ms = t_encode.elapsed().as_secs_f64() * 1e3;
    let t_finalize = std::time::Instant::now();
    let result = match capture_sq8 {
        Some(include_sq8) => writer.finish_with_cache_seed(include_sq8),
        None => writer.finish().map(
            |meta| proximadb_storage_common::pax_block::PaxSegmentWrite {
                meta,
                cache_seed: None,
            },
        ),
    };
    let finalize_ms = t_finalize.elapsed().as_secs_f64() * 1e3;
    // TD-COMPACT-1 S1: encode = per-record RaBitQ/SQ8 quantize; finalize =
    // serialize regions + footer. Combined with `cluster_plan` (logged by the
    // compaction entry under the same gate) and `PROXIMADB_TRACE_IVF_FLUSH`'s
    // cluster internals, this accounts the full re-cluster wall clock.
    if trace {
        eprintln!(
            "[PAX write] n={n} quant={q:?} coalesced={c} | encode {enc:.0} ms  finalize {fin:.0} ms  | total {tot:.0} ms",
            n = records.len(),
            q = quant,
            c = quant == VectorQuant::RaBitQ && coalesced_rabitq_enabled(),
            enc = encode_ms,
            fin = finalize_ms,
            tot = encode_ms + finalize_ms,
        );
    }
    result
}

fn lossless_clustered_enabled() -> bool {
    matches!(
        std::env::var("PROXIMADB_PAX_LOSSLESS_CLUSTERED")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(|value| value.to_ascii_lowercase())
            .as_deref(),
        Some("1") | Some("true") | Some("on") | Some("yes")
    )
}

fn lossless_scalar_enabled() -> bool {
    matches!(
        std::env::var("PROXIMADB_PAX_LOSSLESS_SCALAR")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(|value| value.to_ascii_lowercase())
            .as_deref(),
        Some("1") | Some("true") | Some("on") | Some("yes")
    )
}

/// TD-COMPACT-1 S1: env-gated (`PROXIMADB_TRACE_PAX_WRITE`) phase timer for the
/// PAX segment WRITE path. Logs `cluster_plan` (the re-cluster ordering wall
/// clock) + `encode` (per-record RaBitQ/SQ8 quantize) + `finalize` (serialize
/// regions + footer to disk). Complements `PROXIMADB_TRACE_IVF_FLUSH`, which
/// breaks the cluster plan into pca/project/kmeans/assign/order internals — the
/// two together localize the L0→L1 re-cluster wall clock (observed 666s on SIFT
/// 1M) to cluster-vs-encode-vs-finalize, which is the data needed to choose the
/// fix (S3 preserve the IVF model vs S4 parallelize the re-encode). Off in prod
/// (env unset): a single `var_os` check per write, zero formatting cost.
fn pax_write_trace() -> bool {
    std::env::var_os("PROXIMADB_TRACE_PAX_WRITE").is_some()
}

/// True only when every vector row in every input `.pax` segment has an exact
/// f32 authority. Compaction uses this to decide whether its rewritten RaBitQ
/// output may retain an exact tier. A raw-f32 `EMBED_BASE` is authoritative;
/// otherwise every non-null base vector must have a matching raw
/// `F32_TIER_BASE` row. One exact input never upgrades lossy siblings.
///
/// Legacy ProximaBlocks (`.sst` or no extension) are exact authorities. Other
/// physical formats are not assumed exact without an explicit contract. At
/// least one PAX input must request/preserve exactness; this keeps the optional
/// tier default-OFF for legacy-only compactions. Read/parse/decode failures fail
/// closed to `false`.
pub fn pax_inputs_have_f32_tier(input_files: &[std::path::PathBuf]) -> bool {
    let mut saw_pax = false;
    for f in input_files {
        match f.extension().and_then(|e| e.to_str()) {
            Some("pax") => {}
            Some("sst") | None => continue,
            Some(_) => return false,
        }
        saw_pax = true;
        let Ok(bytes) = std::fs::read(f) else {
            return false;
        };
        if !pax_segment_has_exact_vector_authority(&bytes) {
            return false;
        }
    }
    saw_pax
}

/// True only when every vector row in this PAX segment has an authoritative
/// raw-f32 value, either in `EMBED_BASE` or the co-located exact tier. Query
/// execution and compaction share this pure predicate so neither can relabel a
/// lossy SQ8/RaBitQ reconstruction as exact. Parse/decode failure and empty or
/// non-PAX input fail closed.
pub fn pax_segment_has_exact_vector_authority(bytes: &[u8]) -> bool {
    if SegmentFormat::detect(bytes) != SegmentFormat::Pax {
        return false;
    }
    // ADR-065 hoists the lossy RaBitQ/SQ8 columns out of blocks. The block-level
    // predicate would otherwise see zero base embedding columns and return true
    // vacuously. The segment footer is the authoritative capability declaration
    // for this layout; only an explicitly emitted f32 tier satisfies exactness.
    if proximadb_storage_common::segment_layout::is_coalesced_segment(bytes) {
        return proximadb_storage_common::segment_layout::SegmentFooterIndex::locate_in_segment(
            bytes,
        )
        .ok()
        .flatten()
        .is_some_and(|footer| footer.has_f32_tier);
    }
    let Ok(mut scanner) = PaxSegmentScanner::from_bytes(bytes.to_vec(), ScanPredicate::default())
    else {
        return false;
    };
    let mut saw_block = false;
    while let Some(reader) = scanner.next_block() {
        saw_block = true;
        if !reader.has_exact_vector_authority() {
            return false;
        }
    }
    saw_block
}

/// True iff `bytes` is a `.pax` SEGMENT whose `EMBED_BASE` column is RaBitQ-coded
/// AND no `F32_TIER_BASE` column is present — i.e. the segment carries NO exact
/// vectors, so an AXIS index rebuilt from it (via [`read_segment_records`] →
/// `decode_rabitq_reconstruct`) would be COARSE (low recall). The reopen-rebuild
/// path (`ensure_axis_index_from_sst`) uses this to SKIP rebuilding a lossy AXIS
/// for default RaBitQ-PAX collections, so cold reads fall through to the RaBitQ
/// cascade (`try_pax_cascade`, which ranks the RaBitQ codes properly, ~0.93
/// recall) instead of a coarse rebuilt index (~0.46). Returns `false` for non-Pax,
/// non-RaBitQ quants (RawF32/SQ8/FP16 = exact/near-exact → rebuild), RaBitQ WITH
/// an f32 tier (exact via the tier → rebuild), or any parse error (→ rebuild).
///
/// Pure over `bytes` so it's unit-testable; the engine reads bytes via the
/// filesystem trait (object-store-safe) and calls this. Mirrors
/// [`pax_inputs_have_f32_tier`]'s `PaxSegmentScanner` block-0 metadata read.
pub fn pax_segment_is_coarse_rabitq_without_f32_tier(bytes: &[u8]) -> bool {
    use proximadb_block_format::vparam::QUANT_RABITQ;
    if SegmentFormat::detect(bytes) != SegmentFormat::Pax {
        return false;
    }
    // A `.pax` SEGMENT is `[block(s)][segment_index][SEGMENT_MAGIC]`; the scanner
    // parses the trailing index + magic and yields block 0 (see
    // [`pax_inputs_have_f32_tier`]).
    let Ok(mut scanner) = PaxSegmentScanner::from_bytes(bytes.to_vec(), ScanPredicate::default())
    else {
        return false;
    };
    let Some(reader) = scanner.next_block() else {
        return false;
    };
    let vparams = reader.vector_params();
    let Some(embed) = vparams.get(col_id::EMBED_BASE) else {
        return false;
    };
    embed.quant_kind == QUANT_RABITQ && vparams.get(col_id::F32_TIER_BASE).is_none()
}

/// Detect the tier-2 rerank quant strategy of the source `.pax` segments, so
/// compaction PRESERVES it when re-encoding (mirrors [`pax_inputs_have_f32_tier`]
/// for the f32 tier). Returns the first detected strategy from the co-located
/// `RERANK_BASE` column; defaults to [`VectorQuant::Sq8`] (the validated
/// tier-2) when no `.pax` input carries a rerank column. Correct for both env-
/// and tag-opt-in: the source segment reflects whatever the collection wrote,
/// so preserving it keeps a configured FP16/f32 rerank from being silently
/// downgraded to SQ8 on the first compaction.
pub fn pax_inputs_rerank_quant(
    input_files: &[std::path::PathBuf],
) -> proximadb_block_format::VectorQuant {
    use proximadb_block_format::vparam::{QUANT_FP16, QUANT_RAW_F32};
    for f in input_files {
        if f.extension().and_then(|e| e.to_str()) != Some("pax") {
            continue;
        }
        let Ok(bytes) = std::fs::read(f) else {
            continue;
        };
        // Read block 0's column metadata via the segment scanner — a `.pax`
        // SEGMENT file is `[block(s)][segment_index][SEGMENT_MAGIC]`, so opening
        // the whole file as a single block misreads the footer (see
        // [`pax_inputs_have_f32_tier`]). The rerank column is co-located in every
        // block, so block 0 suffices to detect the segment's tier-2 strategy.
        let Ok(mut scanner) = PaxSegmentScanner::from_bytes(bytes, ScanPredicate::default()) else {
            continue;
        };
        let Some(reader) = scanner.next_block() else {
            continue;
        };
        if let Some(entry) = reader.vector_params().get(col_id::RERANK_BASE) {
            return match entry.quant_kind {
                QUANT_FP16 => VectorQuant::Fp16,
                QUANT_RAW_F32 => VectorQuant::RawF32,
                // SQ8 (the default) or any unknown/reserved kind → Sq8.
                _ => VectorQuant::Sq8,
            };
        }
    }
    VectorQuant::Sq8
}

/// One scored hit from the coalesced RaBitQ segment scan, reranked against the
/// co-located SQ8 column. `distance` carries the canonical public metric value:
/// Euclidean distance, cosine distance, or positive dot-product similarity.
#[derive(Debug, Clone)]
pub struct CascadeHit {
    pub oid: String,
    pub distance: f32,
    /// Exact f32 vector (from the opt-in f32 tier) for `include_vectors`
    /// materialization. `None` when no f32 tier is present (the caller gets
    /// id+score only, as before). Populated lazily for the top-k rows only.
    pub vector: Option<Vec<f32>>,
    /// TD-DELVEC-1 WI-4 slice 2: the hit's global row position in the segment
    /// (the index `read_segment_records` reconstructs positionally — the same
    /// space the deletion vector keys on). Set at coalesced-scan hit construction
    /// from the ranked row index `g`; consumed by `try_pax_cascade`'s merge-on-read
    /// filter so a cold delete is invisible on the RaBitQ ANN path too.
    pub position: u32,
}

/// Canonical resolver from a filter field name to its PAX column id for
/// block/row-group pruning — the single mapper shared by every PAX prune path
/// (e.g. the relational ranged `read_records_pruned` scan). Only the fixed
/// canonical columns carry zone-map stripes; user metadata lives in opaque
/// `props` (not prunable), so unknown fields return `None` and the pruner
/// conservatively keeps the block (no false negatives).
///
/// A plain `fn` (not a closure) so it coerces to both the field→column-mapper
/// trait object form and the `Sync` form the object-storage ranged path requires.
pub(crate) fn pax_field_to_col(field: &str) -> Option<i32> {
    match field {
        "id" | "oid" => Some(col_id::OID),
        "tenant_id" => Some(col_id::TENANT_ID),
        "created_at" | "created_at_ns" => Some(col_id::CREATED_AT),
        "updated_at" | "updated_at_ns" => Some(col_id::UPDATED_AT),
        "valid_from" | "valid_from_ns" => Some(col_id::VALID_FROM),
        "valid_to" | "valid_to_ns" => Some(col_id::VALID_TO),
        _ => None,
    }
}

/// ADR-089 / TD-FPRUNE-1: the filter-aware cascade gate. **FLIPPED
/// default-ON 2026-08-31** — evidence (the promotion criterion the
/// ENV_GATE_REGISTRY row records, discharged NON-vacuously through the real
/// `search_vectors_unified` dispatch): the NEW filtered SIFT ratchet leg
/// `sift_pax_filtered_cascade_recall_ratchet` measures filtered recall@10 =
/// **0.996** vs filtered brute-force ground truth (floor 0.90) over 100
/// queries at N=10k / 8 partitions, with every hit satisfying the predicate
/// AND carrying rehydrated metadata (the top-k rehydration keeps the exact
/// path's response shape). The unfiltered SIFT ratchet holds 0.989 with the
/// P2 write layout active. Fail-safe on every axis (non-coalesced → exact
/// fallback; stage-F error → exact fallback; empty allow-set → empty result);
/// `PROXIMADB_PAX_FILTERED_CASCADE=0|off|false|no` is the emergency
/// kill-switch back to the whole-object exact scan.
pub(crate) fn pax_filtered_cascade_enabled() -> bool {
    !matches!(
        std::env::var("PROXIMADB_PAX_FILTERED_CASCADE")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("0" | "off" | "false" | "no")
    )
}

/// Stage-F observability counters (ADR-089 P1): what the metadata pre-stage
/// pruned and matched, per segment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct FilteredRowAllowStats {
    pub blocks_total: usize,
    /// Blocks skipped by conservative `ColumnMeta` stats (`evaluate_block`)
    /// without decoding a row — but AFTER the block body was fetched (P1 prunes
    /// DECODE, not fetch).
    pub blocks_pruned: usize,
    /// TD-FPRUNE-1 P2: blocks pruned from the FETCH plan by footer-resident
    /// stats — no body GET at all (the zero-GET win). A subset-by-mechanism of
    /// `blocks_total`, disjoint from `blocks_pruned`.
    pub blocks_pruned_footer: usize,
    pub rows_matched: usize,
}

/// ADR-089 / TD-FPRUNE-1 P1 Stage F: build the predicate row allow-set for a
/// coalesced PAX segment from its Region-D metadata, without touching Regions
/// A/B and without decoding any vector data.
///
/// Reads: header prefix + footer + the D-block bodies (coalesced ranged GETs).
/// Per block: conservative stats pruning via the shared [`evaluate_block`]
/// kernel (skips decode; never skips a matching row — `prune.rs` soundness
/// contract), then a row-accurate `evaluate_filter_proxima` over each row's
/// props — the SAME evaluator the exact fallback path uses, so match semantics
/// are identical by construction. Returns `Ok(None)` for a non-coalesced
/// segment (caller falls back to the exact path, fail-safe).
pub(crate) async fn pax_filtered_row_allow(
    fs: &dyn proximadb_storage_filesystem_types::FileSystem,
    path: &str,
    filter: &proximadb_filter_expression::FilterExpression,
) -> Result<Option<(proximadb_block_format::RowAllow, FilteredRowAllowStats)>> {
    use proximadb_block_format::{PaxBlockReader, RowAllow, evaluate_block, record::FlatRow};
    use proximadb_storage_common::segment_layout::{SegmentFooterIndex, SegmentHeaderPrefix};

    // 1. Header prefix → coalesced? (mirrors the cascade's detection).
    let size = fs
        .metadata(path)
        .await
        .map_err(|e| anyhow::anyhow!("filtered stage-F stat {path}: {e}"))?
        .size;
    if size < (SEG_HEADER_PREFIX_LEN as u64 + SEGMENT_MAGIC.len() as u64) {
        return Ok(None);
    }
    // TD-PAXRG-1: floor covers v4 (88 B) as well; parse branches on version.
    let read_len = (coalesced_header_prefetch_floor() as u64).min(size);
    let header_bytes = fs
        .read_range(path, 0, read_len)
        .await
        .map_err(|e| anyhow::anyhow!("filtered stage-F header {path}: {e}"))?;
    let header = match SegmentHeaderPrefix::parse(&header_bytes) {
        Ok(h) => h,
        Err(_) => return Ok(None), // not coalesced → exact fallback
    };

    // 2. Footer → block table (+ cumulative start ordinals).
    let footer_bytes = fs
        .read_range(path, header.footer_off, header.footer_len)
        .await
        .map_err(|e| anyhow::anyhow!("filtered stage-F footer {path}: {e}"))?;
    let footer = SegmentFooterIndex::parse(&footer_bytes)?;
    // TD-FPRUNE-1 P2 engagement: resolve filter field names → PAX col-ids using
    // canonical fields first, then the segment's self-describing shred field-map
    // (written from the P-Shred spec). This lets a USER-TAG filter (e.g.
    // `partition`) prune blocks — without a catalog lookup. Owned map so the
    // resolver closure can be a per-call `&dyn Fn` temporary at each
    // `evaluate_block` (a `&dyn Fn` held across `.await` would make the future
    // `!Send`); the `HashMap` itself is `Send`.
    let user_field_map: std::collections::HashMap<String, i32> =
        footer.shred_field_map.iter().cloned().collect();
    let resolve_field = |field: &str| -> Option<i32> {
        pax_field_to_col(field).or_else(|| user_field_map.get(field).copied())
    };
    let mut block_start: Vec<u64> = Vec::with_capacity(footer.blocks.len());
    let mut acc = 0u64;
    for b in &footer.blocks {
        block_start.push(acc);
        acc += b.row_count as u64;
    }
    let n_rows = acc as usize;
    let mut allow = RowAllow::new(n_rows);
    let mut stats = FilteredRowAllowStats {
        blocks_total: footer.blocks.len(),
        ..Default::default()
    };

    // 3. Fetch ALL D-block bodies via the same IOP-aligned coalescing the
    //    cascade's OID resolve uses. (P1 reads the metadata blocks and prunes
    //    the DECODE via stats; pruning the FETCH needs footer-resident stats —
    //    TD-FPRUNE-1 P2.) Coalesced D blocks carry no vector stripes (ADR-065),
    //    so these bytes are scalar/metadata only.
    let iop_target =
        proximadb_storage_common::iops_budget::IopsBudget::for_path(path).target_block_bytes();
    let defaults =
        crate::storage::engines::sst::readers::sst_query_engine::ObjectRangeCoalescePolicy {
            max_gap_bytes: (iop_target / 4).max(64 * 1024),
            max_range_bytes: iop_target,
        };
    // TD-FPRUNE-1 P2: prune blocks from the FETCH plan using footer-resident
    // stats — a block whose footer `FooterBlockStats` provably cannot match is
    // dropped BEFORE any body GET (the zero-GET win). The payloads live in the
    // optional `SECTION_BLOCK_FILTERABLE_STATS` footer section (they share the
    // sparse index-tagged map the footer parse already built). Blocks with no
    // footer stats (legacy segments, or footer stats gated OFF at write) stay
    // in, conservatively fetched. The footer verdict uses the SAME
    // `evaluate_block` kernel as the post-fetch decode prune, so it can never
    // drop a matching block (prune.rs soundness contract).
    let filterable_stats: std::collections::HashMap<u32, &[u8]> = footer
        .block_filterable_stats
        .iter()
        .map(|(idx, payload)| (*idx, payload.as_slice()))
        .collect();
    let candidate_blocks: Vec<usize> = (0..footer.blocks.len())
        .filter(|&bi| {
            let Some(payload) = filterable_stats.get(&(bi as u32)).copied() else {
                return true; // no footer stats → must fetch
            };
            let fstats = proximadb_block_format::FooterBlockStats::from_bytes(
                payload,
                footer.blocks[bi].row_count,
            );
            let keep = evaluate_block(&fstats, filter, &resolve_field)
                != proximadb_block_format::PruneResult::Skip;
            if !keep {
                stats.blocks_pruned_footer += 1;
            }
            keep
        })
        .collect();
    let policy = resolve_adaptive_range_policy(
        path,
        "region_d_filter",
        "PROXIMADB_PAX_COALESCE_GAP",
        "PROXIMADB_PAX_COALESCE_RANGE",
        defaults,
        |candidate| {
            estimate_coalesced_byte_ranges(
                candidate_blocks.iter().filter_map(|&index| {
                    footer
                        .blocks
                        .get(index)
                        .map(|block| (block.offset, block.offset + block.size as u64))
                }),
                candidate,
            )
        },
    );
    let fetches = plan_coalesced_block_ranges(&footer, &candidate_blocks, &policy);
    for fetch in &fetches {
        let buf = fs
            .read_range(path, fetch.start, fetch.end - fetch.start)
            .await
            .map_err(|e| anyhow::anyhow!("filtered stage-F blocks {path}: {e}"))?;
        for &bi in &fetch.blocks {
            let Some(b) = footer.blocks.get(bi) else {
                continue;
            };
            let Some(rel) = b.offset.checked_sub(fetch.start).map(|r| r as usize) else {
                continue;
            };
            let end = rel + b.size as usize;
            let Some(block_bytes) = buf.get(rel..end) else {
                continue;
            };
            let reader = PaxBlockReader::open(block_bytes)
                .map_err(|e| anyhow::anyhow!("filtered stage-F block open {path}: {e}"))?;
            // Conservative stats prune: provably-no-match blocks skip decode.
            // NOTE: `&resolve_field` is a per-call temporary — a `&dyn Fn`
            // binding held across the fetch `.await` would make this future (and
            // every async caller up to the REST handlers) `!Send`. `resolve_field`
            // borrows the owned `user_field_map` (Send), which is fine to hold.
            if evaluate_block(&reader, filter, &resolve_field)
                == proximadb_block_format::PruneResult::Skip
            {
                stats.blocks_pruned += 1;
                continue;
            }
            let start_ordinal = block_start[bi] as usize;
            for (local, flat) in FlatRow::from_block_reader(&reader)
                .map_err(|e| anyhow::anyhow!("filtered stage-F rows {path}: {e}"))?
                .into_iter()
                .enumerate()
            {
                let props = flat
                    .props_tree()
                    .map_err(|e| anyhow::anyhow!("filtered stage-F props {path}: {e}"))?;
                if crate::core::search::sql_value_filter::evaluate_filter_proxima(filter, &props) {
                    allow.insert(start_ordinal + local);
                    stats.rows_matched += 1;
                }
            }
        }
    }
    tracing::debug!(
        path,
        blocks_total = stats.blocks_total,
        blocks_pruned = stats.blocks_pruned,
        blocks_pruned_footer = stats.blocks_pruned_footer,
        rows_matched = stats.rows_matched,
        n_rows,
        "ADR-089 stage-F row allow-set built (P2: blocks_pruned_footer = zero-GET)"
    );
    // TD-FPRUNE-1 P2 attribution: expose the zero-GET prune engagement in the
    // query-scoped io_trace (the physical savings already land in the universal
    // get_ops/bytes_read as unfetched reads).
    crate::observability::io_trace::record_pax_footer_prune(
        stats.blocks_total as u64,
        stats.blocks_pruned_footer as u64,
    );
    Ok(Some((allow, stats)))
}

/// TD-FPRUNE-1 cascade top-k rehydration: fetch FULL records — props included —
/// for specific global row positions. The cascade path ranks from codes and
/// returns id+score (+reconstructed vector); the exact path it replaces returns
/// complete metadata. This closes that gap for the top-k winners only: positions
/// map to blocks via the footer's cumulative row table and ONLY the blocks
/// holding requested positions are fetched (coalesced, IOP-aligned), so the cost
/// is bounded by the hit set (k), not the segment. Result is index-aligned with
/// `positions`; `None` = position out of range or row not found (never an error).
pub(crate) async fn read_records_by_positions(
    fs: &dyn proximadb_storage_filesystem_types::FileSystem,
    path: &str,
    positions: &[u32],
    embedding_model_ids: &[String],
    user_column_keys: &[String],
    tenant_ctx: Option<&str>,
    // TD-RDSTRAT-13 PR-B 2b: the invariants cache the cascade just warmed —
    // the rehydrate's header/footer reads become cache hits (zero GETs)
    // instead of 3 duplicate IO ops per segment per query.
    cache: Option<&SegmentInvariantsCache>,
) -> Result<Vec<Option<proximadb_records::ProximaRecord>>> {
    use proximadb_block_format::{PaxBlockReader, record::FlatRow};
    use proximadb_storage_common::segment_layout::{SegmentFooterIndex, SegmentHeaderPrefix};

    // TD-RDSTRAT-13 PR-B: attribute the whole rehydration pass (metadata +
    // header/footer + block fetches + decode) — the second Region-D read.
    let rehydrate_started = std::time::Instant::now();
    let mut out: Vec<Option<proximadb_records::ProximaRecord>> = vec![None; positions.len()];
    if positions.is_empty() {
        return Ok(out);
    }

    // 1-2. Header prefix + footer → block table. Cache-first (TD-RDSTRAT-13
    // 2b): the probe warmed the invariants cache EARLIER IN THIS QUERY, so the
    // rehydrate's lookups are L1 hits — no HEAD, no header GET, no footer GET.
    // On a miss (cold cache), fall back to the exact stat+read path.
    let (_header, footer) = if let Some(cache) = cache
        && let Some(inv) = cache.get_or_promote(path).await.value()
    {
        match (
            SegmentHeaderPrefix::parse(&inv.header_bytes),
            SegmentFooterIndex::parse(&inv.footer_bytes),
        ) {
            (Ok(h), Ok(f)) => (h, f),
            (Err(e), _) | (_, Err(e)) => {
                anyhow::bail!("rehydrate cached layout {path}: {e}")
            }
        }
    } else {
        let size = fs
            .metadata(path)
            .await
            .map_err(|e| anyhow::anyhow!("rehydrate stat {path}: {e}"))?
            .size;
        if size < (SEG_HEADER_PREFIX_LEN as u64 + SEGMENT_MAGIC.len() as u64) {
            anyhow::bail!("rehydrate {path}: not a coalesced segment");
        }
        let read_len = (coalesced_header_prefetch_floor() as u64).min(size);
        let header_bytes = fs
            .read_range(path, 0, read_len)
            .await
            .map_err(|e| anyhow::anyhow!("rehydrate header {path}: {e}"))?;
        let header = SegmentHeaderPrefix::parse(&header_bytes)
            .map_err(|e| anyhow::anyhow!("rehydrate header parse {path}: {e}"))?;
        let footer_bytes = fs
            .read_range(path, header.footer_off, header.footer_len)
            .await
            .map_err(|e| anyhow::anyhow!("rehydrate footer {path}: {e}"))?;
        let footer = SegmentFooterIndex::parse(&footer_bytes)?;
        (header, footer)
    };
    let mut block_start: Vec<u64> = Vec::with_capacity(footer.blocks.len());
    let mut acc = 0u64;
    for b in &footer.blocks {
        block_start.push(acc);
        acc += b.row_count as u64;
    }

    // 3. Map positions → (block, local row). Sorted block set dedups fetches.
    let mut wanted: Vec<(usize, u32, usize)> = Vec::with_capacity(positions.len()); // (block, local, out_idx)
    for (out_idx, &pos) in positions.iter().enumerate() {
        let pos = pos as u64;
        let block = match block_start.binary_search(&pos) {
            Ok(b) => b,
            Err(0) => continue, // before the first block — out of range
            Err(insert) => insert - 1,
        };
        let local = pos - block_start[block];
        let row_count = footer.blocks[block].row_count as u64;
        if local >= row_count {
            continue;
        }
        wanted.push((block, local as u32, out_idx));
    }
    if wanted.is_empty() {
        return Ok(out);
    }
    wanted.sort_unstable();
    let mut blocks: Vec<usize> = wanted.iter().map(|w| w.0).collect();
    blocks.dedup();

    // 4. Coalesced fetch of just those blocks (same policy plumbing as stage F).
    let iop_target =
        proximadb_storage_common::iops_budget::IopsBudget::for_path(path).target_block_bytes();
    let defaults =
        crate::storage::engines::sst::readers::sst_query_engine::ObjectRangeCoalescePolicy {
            max_gap_bytes: (iop_target / 4).max(64 * 1024),
            max_range_bytes: iop_target,
        };
    let policy = resolve_adaptive_range_policy(
        path,
        "region_d_filter",
        "PROXIMADB_PAX_COALESCE_GAP",
        "PROXIMADB_PAX_COALESCE_RANGE",
        defaults,
        |candidate| {
            estimate_coalesced_byte_ranges(
                blocks.iter().filter_map(|&index| {
                    footer
                        .blocks
                        .get(index)
                        .map(|block| (block.offset, block.offset + block.size as u64))
                }),
                candidate,
            )
        },
    );
    let fetches = plan_coalesced_block_ranges(&footer, &blocks, &policy);

    // 5. Decode the requested rows.
    for fetch in &fetches {
        let buf = fs
            .read_range(path, fetch.start, fetch.end - fetch.start)
            .await
            .map_err(|e| anyhow::anyhow!("rehydrate blocks {path}: {e}"))?;
        for &bi in &fetch.blocks {
            let Some(b) = footer.blocks.get(bi) else {
                continue;
            };
            let Some(rel) = b.offset.checked_sub(fetch.start).map(|r| r as usize) else {
                continue;
            };
            let end = rel + b.size as usize;
            let Some(block_bytes) = buf.get(rel..end) else {
                continue;
            };
            let reader = PaxBlockReader::open(block_bytes)
                .map_err(|e| anyhow::anyhow!("rehydrate block open {path}: {e}"))?;
            let rows = FlatRow::from_block_reader(&reader)
                .map_err(|e| anyhow::anyhow!("rehydrate rows {path}: {e}"))?;
            // Rows for THIS block from `wanted` (sorted ⇒ contiguous slice).
            let local_base = block_start[bi] as u32;
            for &(_wb, wlocal, out_idx) in wanted.iter().filter(|w| w.0 == bi) {
                let _ = local_base;
                if let Some(flat) = rows.get(wlocal as usize) {
                    out[out_idx] = Some(flat.clone().into_record(
                        embedding_model_ids,
                        user_column_keys,
                        tenant_ctx,
                    )?);
                }
            }
        }
    }
    crate::observability::io_trace::record_rehydrate_us(
        rehydrate_started.elapsed().as_micros() as u64
    );
    crate::storage::engines::sst::metrics::record_rehydrate_us(
        rehydrate_started.elapsed().as_micros() as u64,
    );
    Ok(out)
}

/// Whether the coalesced-RaBitQ layout is engaged for new RaBitQ writes
/// (ADR-062 / TD-RDSTRAT-6). **Default ON** — coalesced scan-then-rerank is the
/// canonical PAX RaBitQ path (pre-GA: no serialized legacy data, so no back-compat
/// — per the ADR-061 in-place amendment). The reader handles BOTH layouts via the
/// `SEG_HEADER_MAGIC` presence-field. `PROXIMADB_PAX_COALESCED_RABITQ=0|off|false`
/// is an emergency kill-switch back to the legacy in-block RaBitQ layout.
pub fn coalesced_rabitq_enabled() -> bool {
    !matches!(
        std::env::var("PROXIMADB_PAX_COALESCED_RABITQ")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("0" | "off" | "false" | "no")
    )
}

/// TD-PAXRG-1: write the **row-group Region D** layout — Region D granules
/// are Parquet-style row groups (Olap-framed, RowDirectory + in-block rgdir
/// suppressed), per-RG stats ride the footer's MinMax payload, the RG target
/// is floored at 256 KiB, and the OID resolver auto-captures (point-lookup
/// authority; a format invariant of the layout, not a caller obligation).
///
/// DEFAULT ON (FLIPPED 2026-08-28): the TD-PAXRG-1 Phase-G evidence is
/// complete — SIFT1M 100k recall@10 = 0.9896 identical to the legacy-block
/// baseline (floor 0.90); at-rest 80.3% of legacy bytes (RowDirectory
/// removal); micro-granule amplification closed 13.4×; OID-chunk top-k −34%
/// bytes at GET parity+1; round-trip/twin/gate purity all pinned. Read side
/// is unconditional. `PROXIMADB_PAX_WRITE_RG_LAYOUT=0|off|false|no` is the
/// kill-switch back to the legacy block framing; the per-collection
/// `pax_rg_layout:on|off` tag (flush) outranks the env.
pub fn rg_layout_enabled() -> bool {
    !matches!(
        std::env::var("PROXIMADB_PAX_WRITE_RG_LAYOUT")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("0" | "off" | "false" | "no")
    )
}

/// Parse a `u64` env override, or return `default`. Used for the coalesced
/// read-path tuning knobs (survivor coalesce gap/range, pool mult/min/rate).
fn env_u64_or(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

/// Backend-derived Region-A probe coalescing policy.
///
/// Region A previously defaulted to gap-zero even though Regions B/D already
/// used the backend I/O budget. That paid one operation per selected IVF cell
/// on Azure. Selected logical slices are retained separately, so bridging a
/// bounded gap changes only bytes transferred, never the rows passed to the
/// ranker or recall.
fn default_probe_range_policy(
    path: &str,
) -> crate::storage::engines::sst::readers::sst_query_engine::ObjectRangeCoalescePolicy {
    let target =
        proximadb_storage_common::iops_budget::IopsBudget::for_path(path).target_block_bytes();
    crate::storage::engines::sst::readers::sst_query_engine::ObjectRangeCoalescePolicy {
        max_gap_bytes: (target / 4).max(64 * 1024),
        max_range_bytes: target,
    }
}

/// Resolve a physical range cap from exact, query-local GET/byte plans.
///
/// This is deliberately a physical-I/O decision made after logical cells/rows
/// have been selected. It cannot change nprobe, survivors, ranking, or recall.
/// Explicit diagnostic overrides retain precedence; the default-OFF chooser
/// gate preserves the fixed policy byte-for-byte when unset.
fn resolve_adaptive_range_policy<F>(
    path: &str,
    region: &str,
    gap_env: &str,
    range_env: &str,
    defaults: crate::storage::engines::sst::readers::sst_query_engine::ObjectRangeCoalescePolicy,
    estimate: F,
) -> crate::storage::engines::sst::readers::sst_query_engine::ObjectRangeCoalescePolicy
where
    F: Fn(
        &crate::storage::engines::sst::readers::sst_query_engine::ObjectRangeCoalescePolicy,
    ) -> crate::storage::engines::core::coalesce_strategy::RangePlanEstimate,
{
    use crate::storage::engines::core::coalesce_strategy::{
        RangePlanCandidate, ReadPlanCostProfile, choose_range_plan, read_strategy_chooser_enabled,
    };

    // Presence, including a malformed value, is treated as an explicit
    // diagnostic request. A malformed value fails closed to the corresponding
    // default rather than silently engaging a different adaptive policy.
    if std::env::var_os(gap_env).is_some() || std::env::var_os(range_env).is_some() {
        return crate::storage::engines::sst::readers::sst_query_engine::ObjectRangeCoalescePolicy {
            max_gap_bytes: env_u64_or(gap_env, defaults.max_gap_bytes),
            max_range_bytes: env_u64_or(range_env, defaults.max_range_bytes),
        };
    }
    if !read_strategy_chooser_enabled() {
        return defaults;
    }

    // Keep the measured gap fixed in this slice. Cap candidates are exact and
    // independently auditable; adaptive gap selection needs its own evidence
    // because it can introduce substantially more bystander bytes.
    let mut policies = vec![defaults];
    for max_range_bytes in proximadb_storage_common::iops_budget::read_range_cap_candidates(path) {
        if max_range_bytes != defaults.max_range_bytes {
            policies.push(
                crate::storage::engines::sst::readers::sst_query_engine::ObjectRangeCoalescePolicy {
                    max_gap_bytes: defaults.max_gap_bytes,
                    max_range_bytes,
                },
            );
        }
    }
    let candidates: Vec<RangePlanCandidate> = policies
        .iter()
        .map(|policy| RangePlanCandidate {
            max_gap_bytes: policy.max_gap_bytes,
            max_range_bytes: policy.max_range_bytes,
            estimate: estimate(policy),
        })
        .collect();
    let profile = ReadPlanCostProfile::for_path(path, None);
    let Some(decision) = choose_range_plan(&candidates, 0, profile) else {
        tracing::warn!(
            target: "pax.range_estimator",
            path,
            region,
            "adaptive range estimator failed closed to fixed policy"
        );
        return defaults;
    };
    let Some(chosen) = policies.get(decision.chosen_index).copied() else {
        return defaults;
    };
    tracing::debug!(
        target: "pax.range_estimator",
        path,
        region,
        cost_class = ?profile.class,
        candidates = candidates.len(),
        admissible = decision.admissible_candidates,
        baseline_gets = decision.baseline.estimate.get_requests,
        baseline_bytes = decision.baseline.estimate.physical_bytes,
        baseline_cap = decision.baseline.max_range_bytes,
        chosen_gets = decision.chosen.estimate.get_requests,
        chosen_bytes = decision.chosen.estimate.physical_bytes,
        chosen_cap = decision.chosen.max_range_bytes,
        chosen_gap = decision.chosen.max_gap_bytes,
        baseline_score = decision.baseline_score,
        chosen_score = decision.chosen_score,
        "selected exact cold-miss GET/byte range-plan knee"
    );
    chosen
}

/// Estimate a coalesced plan without allocating its payload-bearing fetch
/// structures. `ranges` must be ordered by start offset and contain the exact
/// logical byte extents that the executable planner will receive.
fn estimate_coalesced_byte_ranges<I>(
    ranges: I,
    policy: &crate::storage::engines::sst::readers::sst_query_engine::ObjectRangeCoalescePolicy,
) -> crate::storage::engines::core::coalesce_strategy::RangePlanEstimate
where
    I: IntoIterator<Item = (u64, u64)>,
{
    let mut get_requests = 0_u64;
    let mut physical_bytes = 0_u64;
    let mut current: Option<(u64, u64)> = None;
    for (start, end) in ranges {
        if end <= start {
            continue;
        }
        match current {
            Some((current_start, current_end)) => {
                let gap = start.saturating_sub(current_end);
                let merged_end = current_end.max(end);
                let merged_len = merged_end.saturating_sub(current_start);
                if gap <= policy.max_gap_bytes
                    && (policy.max_range_bytes == 0 || merged_len <= policy.max_range_bytes)
                {
                    current = Some((current_start, merged_end));
                } else {
                    get_requests = get_requests.saturating_add(1);
                    physical_bytes =
                        physical_bytes.saturating_add(current_end.saturating_sub(current_start));
                    current = Some((start, end));
                }
            }
            None => current = Some((start, end)),
        }
    }
    if let Some((start, end)) = current {
        get_requests = get_requests.saturating_add(1);
        physical_bytes = physical_bytes.saturating_add(end.saturating_sub(start));
    }
    crate::storage::engines::core::coalesce_strategy::RangePlanEstimate {
        get_requests,
        physical_bytes,
    }
}

/// Mirror Region B's executable whole-region fallback in candidate scoring.
/// Once scattered survivor ranges would move at least the complete SQ8 region,
/// one canonical GET is both fewer requests and no more bytes.
fn estimate_with_whole_region_fallback(
    estimate: crate::storage::engines::core::coalesce_strategy::RangePlanEstimate,
    region_bytes: u64,
) -> crate::storage::engines::core::coalesce_strategy::RangePlanEstimate {
    if estimate.physical_bytes >= region_bytes {
        crate::storage::engines::core::coalesce_strategy::RangePlanEstimate {
            get_requests: 1,
            physical_bytes: region_bytes,
        }
    } else {
        estimate
    }
}

/// A coalesced ranged GET over one or more survivor data blocks.
struct CoalescedFetch {
    start: u64,
    end: u64,
    blocks: Vec<usize>,
}

/// TD-RDSTRAT-12 §3 round 4 (opt-in wave-split): aligned chunks covering
/// `[region_off, region_off + region_len)`, last chunk carrying the remainder.
/// Chunk size comes from the site's `IopsBudget` target so the split ranges
/// match the backend's coalescing profile (4 MiB on azure://).
fn plan_wave_split_ranges(
    region_off: u64,
    region_len: u64,
    chunk: u64,
) -> Vec<std::ops::Range<u64>> {
    let chunk = chunk.max(1);
    let mut ranges = Vec::with_capacity(region_len.div_ceil(chunk) as usize);
    let mut start = region_off;
    let end = region_off.saturating_add(region_len);
    while start < end {
        let stop = start.saturating_add(chunk).min(end);
        ranges.push(start..stop);
        start = stop;
    }
    ranges
}

/// TD-RDSTRAT-12 §3 round 4: `PROXIMADB_READ_RANGES_WAVE_SPLIT_MB` — minimum
/// Region-B whole-region-fallback size (MiB) eligible for an opt-in wave split.
/// Unset ⇒ OFF (the fallback issues its single whole-region GET unchanged, the
/// TD-RDSTRAT-8 GET-economical default). Composes with
/// `PROXIMADB_READ_RANGES_INFLIGHT`: with the cap ≤ 1 the wave runs
/// sequentially, so an armed threshold without an armed cap is inert by
/// construction (the wave metrics make that visible).
fn wave_split_threshold_mb() -> Option<u64> {
    static WAVE_SPLIT_MB: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *WAVE_SPLIT_MB.get_or_init(|| {
        std::env::var("PROXIMADB_READ_RANGES_WAVE_SPLIT_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
    })
}

/// Plan coalesced ranged GETs over a set of survivor block indices using the
/// shared `ObjectRangeCoalescePolicy` thresholds (ADR-062 D3 — the GET win is the
/// *policy* planning merged ranges; `FileSystem::read_ranges` only executes them).
/// Adjacent blocks within `max_gap_bytes` (and under `max_range_bytes`) merge into
/// one GET, so cluster-adjacent survivors fetch in a handful of coalesced reads.
fn plan_coalesced_block_ranges(
    footer: &proximadb_storage_common::segment_layout::SegmentFooterIndex,
    block_indices: &[usize],
    policy: &crate::storage::engines::sst::readers::sst_query_engine::ObjectRangeCoalescePolicy,
) -> Vec<CoalescedFetch> {
    let mut sorted: Vec<usize> = block_indices.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut out: Vec<CoalescedFetch> = Vec::new();
    for bi in sorted {
        let Some(b) = footer.blocks.get(bi) else {
            continue;
        };
        let start = b.offset;
        let end = b.offset + b.size as u64;
        let merge = match out.last_mut() {
            Some(last) => {
                let gap = start.saturating_sub(last.end);
                let merged_len = end - last.start;
                let within_gap = gap <= policy.max_gap_bytes;
                let within_max =
                    policy.max_range_bytes == 0 || merged_len <= policy.max_range_bytes;
                if within_gap && within_max {
                    last.end = last.end.max(end);
                    last.blocks.push(bi);
                    true
                } else {
                    false
                }
            }
            None => false,
        };
        if !merge {
            out.push(CoalescedFetch {
                start,
                end,
                blocks: vec![bi],
            });
        }
    }
    out
}

/// One coalesced survivor byte-range into Region B (the rows it covers).
struct RowFetch {
    start: u64,
    end: u64,
    rows: Vec<usize>,
}

/// Plan coalesced ranged GETs over sorted, deduplicated survivor SQ8 row runs
/// in Region B (ADR-065). Each survivor `g` maps to
/// `[codes_base + g·dim, +dim]`; adjacent runs within `policy` merge into one
/// GET. The caller normalizes once so adaptive candidates and the executable
/// plan share one ordering pass.
fn plan_coalesced_row_ranges(
    survivors: &[usize],
    dim: usize,
    codes_base: u64,
    policy: &crate::storage::engines::sst::readers::sst_query_engine::ObjectRangeCoalescePolicy,
) -> Vec<RowFetch> {
    debug_assert!(survivors.windows(2).all(|pair| pair[0] < pair[1]));
    let dim64 = dim as u64;
    let mut out: Vec<RowFetch> = Vec::new();
    for &g in survivors {
        let start = codes_base + (g as u64) * dim64;
        let end = start + dim64;
        let merge = match out.last_mut() {
            Some(last) => {
                let gap = start.saturating_sub(last.end);
                let merged_len = end - last.start;
                let within_gap = gap <= policy.max_gap_bytes;
                let within_max =
                    policy.max_range_bytes == 0 || merged_len <= policy.max_range_bytes;
                if within_gap && within_max {
                    last.end = last.end.max(end);
                    last.rows.push(g);
                    true
                } else {
                    false
                }
            }
            None => false,
        };
        if !merge {
            out.push(RowFetch {
                start,
                end,
                rows: vec![g],
            });
        }
    }
    out
}

/// One selected IVF cell's exact Region-A code slice. A coalesced physical
/// range may contain gaps or unselected cells, but only these slices are ever
/// handed to the RaBitQ ranker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProbeCellSlice {
    start: u64,
    end: u64,
    row_start: usize,
}

/// One physical Region-A read containing one or more selected cell slices.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ProbeRangeFetch {
    start: u64,
    end: u64,
    selected: Vec<ProbeCellSlice>,
}

/// Merge selected Region-A cell extents according to the provider-bounded
/// range policy. The returned `selected` slices preserve the logical nprobe
/// set, so gap over-read changes only physical I/O, never ranking semantics.
fn plan_probe_cell_ranges(
    cells: &[ProbeCellSlice],
    policy: &crate::storage::engines::sst::readers::sst_query_engine::ObjectRangeCoalescePolicy,
) -> Vec<ProbeRangeFetch> {
    let mut sorted = cells.to_vec();
    sorted.sort_by_key(|cell| cell.start);
    let mut out: Vec<ProbeRangeFetch> = Vec::new();
    for cell in sorted {
        if cell.end <= cell.start {
            continue;
        }
        let merged = match out.last_mut() {
            Some(last) => {
                let gap = cell.start.saturating_sub(last.end);
                let merged_len = cell.end.saturating_sub(last.start);
                let within_gap = gap <= policy.max_gap_bytes;
                let within_max =
                    policy.max_range_bytes == 0 || merged_len <= policy.max_range_bytes;
                if within_gap && within_max {
                    last.end = last.end.max(cell.end);
                    last.selected.push(cell);
                    true
                } else {
                    false
                }
            }
            None => false,
        };
        if !merged {
            out.push(ProbeRangeFetch {
                start: cell.start,
                end: cell.end,
                selected: vec![cell],
            });
        }
    }
    out
}

/// Score one fixed-stride SQ8 row without materializing a decoded `Vec<f32>`.
fn rerank_sq8_distance(
    metric: RankMetric,
    query: &[f32],
    codes: &[u8],
    params: &proximadb_codec::Sq8Params,
    query_norm_squared: f32,
) -> Option<f32> {
    use proximadb_codec::functions::sq8;

    match metric {
        RankMetric::L2 => sq8::l2_squared(codes, query, params),
        RankMetric::DotProduct => sq8::dot_product(codes, query, params).map(|dot| -dot),
        RankMetric::Cosine => {
            let (dot, norm_squared) = sq8::dot_and_norm_squared(codes, query, params)?;
            Some(1.0 - dot / ((query_norm_squared * norm_squared).sqrt() + 1e-12))
        }
    }
}

/// Convert the coalesced reranker's lower-is-better ordering score into the
/// canonical metric value consumed by the public score conversion. This is a
/// one-way boundary: sorting happens before it, and callers must not negate or
/// square-root the value again.
fn canonical_score_from_rank_score(metric: RankMetric, rank_score: f32) -> f32 {
    match metric {
        RankMetric::L2 => rank_score.max(0.0).sqrt(),
        RankMetric::Cosine => rank_score,
        RankMetric::DotProduct => -rank_score,
    }
}

/// Adaptive RaBitQ candidate-pool size for top-`k` over a segment of `n` rows.
///
/// `M = max(k · mult, ceil(n · rate), min)` — the survivor pool scales with the
/// segment size so it stays a roughly constant *fraction* of the corpus (rate,
/// default 1%) instead of a fixed 1000 that starves recall at scale (0.1% of 1M
/// → measured recall 0.968; 1% of 100k → 0.991). Env-overridable:
/// `PROXIMADB_PAX_RABITQ_POOL_MULT` (default 100), `..._MIN` (1000),
/// `PROXIMADB_PAX_RABITQ_POOL_RATE` (default 0.01).
pub fn pax_rabitq_pool_for_top_k(k: usize, n: usize) -> usize {
    // Direct override for the survivor-pool sweep (eval knob): force M to a
    // fixed value to map the GETs-vs-recall curve and the M-scaling law
    // (linear fraction vs N^e vs log). Must stay ≥ k (need ≥ top-k survivors).
    if let Some(pool) = std::env::var("PROXIMADB_PAX_RABITQ_POOL")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|m| *m >= k)
    {
        return pool;
    }
    static C: std::sync::OnceLock<(usize, usize, f64)> = std::sync::OnceLock::new();
    let (mult, min, rate) = C.get_or_init(|| {
        let mult = std::env::var("PROXIMADB_PAX_RABITQ_POOL_MULT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(100);
        let min = std::env::var("PROXIMADB_PAX_RABITQ_POOL_MIN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1000);
        let rate = std::env::var("PROXIMADB_PAX_RABITQ_POOL_RATE")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|f| *f > 0.0)
            .unwrap_or(0.01);
        (mult, min, rate)
    });
    let by_k = k.saturating_mul(*mult);
    let by_n = ((n as f64) * rate).ceil() as usize;
    by_k.max(by_n).max(*min)
}

/// Cached per-file-invariant bytes for a coalesced segment: header-prefix,
/// RaBitQ region, footer-index. Hot queries skip the 3 `read_range` calls →
/// 3 GETs → 0 (the GET counter fires inside `read_range`, so eliding the call
/// is the only way to reduce it — caching parsed structs downstream doesn't help).
pub struct SegmentInvariants {
    pub header_bytes: Vec<u8>,
    pub region_bytes: Option<Arc<[u8]>>,
    pub footer_bytes: Vec<u8>,
    pub a0_bytes: Option<Arc<[u8]>>,
    pub rabitq_header_bytes: Option<Arc<[u8]>>,
}

/// Per-segment invariants cache, **byte-budgeted** (ADR-065 cache-co-design #35).
/// The old entry-count cap was wrong for Region A entries (~24 MB each → 64
/// entries = 1.5 GB unbounded). The cache caps **total bytes** (dominated by
/// Region A `region_bytes`).
///
/// TD-CACHE-2 S1: eviction is **priority-then-recency** — victims are chosen
/// by lowest [`CacheTier::evict_priority`] first (meta-only entries before
/// control-bearing before Region-A-bearing), ties broken by least-recent
/// touch. This enforces the ADR-065 intent the enum documents: a hot
/// segment's Region A (a hit saves the full-region GET, re-read every query)
/// is the last thing shed under budget pressure.
///
/// TD-CACHE-2 S2a (concurrency): the map is **sharded** (16 shards, per-shard
/// `RwLock`) with a shared `AtomicUsize` byte budget and `AtomicU64` recency
/// ticks — a cache HIT takes only a shard *read* lock plus one relaxed store,
/// so the 100%-hit hot path never serializes on a global mutex (the c=16
/// contention suspect in TD-SEARCH-2). Victim selection scans per-shard
/// minima under read locks then confirms under the victim shard's write lock
/// — exact (priority, recency) policy at segment-count cardinality, no global
/// lock ever held. Eviction races are benign: each entry's bytes are credited
/// exactly once by whichever thread removes it.
pub struct SegmentInvariantsCache {
    shards: Box<[std::sync::RwLock<std::collections::HashMap<String, CacheEntry>>]>,
    bytes_used: std::sync::atomic::AtomicUsize,
    /// Monotonic touch counter for recency (relaxed; ordering slack between
    /// racing touches is irrelevant to eviction quality).
    tick: std::sync::atomic::AtomicU64,
    byte_budget: usize,
    l2_store: Option<Arc<PersistentByteStore>>,
    l2_hits: std::sync::atomic::AtomicU64,
    l2_misses: std::sync::atomic::AtomicU64,
}

const INVARIANTS_CACHE_SHARDS: usize = 16;

struct CacheEntry {
    inv: Arc<SegmentInvariants>,
    /// Tick of the last `get`/`put` touch (recency for eviction ties). Atomic
    /// so a hit updates it under the shard READ lock.
    last_hit: std::sync::atomic::AtomicU64,
}

/// The eviction tier of a cached entry, derived from which byte classes it
/// carries: Region-A-bearing entries are `InvariantIndex` (pinned hardest),
/// control-prefix entries (`a0`/RaBitQ params) are `SearchControl`, and
/// header+footer-only entries are `InvariantMeta` (cheapest to refetch).
fn entry_tier(inv: &SegmentInvariants) -> CacheTier {
    if inv.region_bytes.is_some() {
        CacheTier::InvariantIndex
    } else if inv.a0_bytes.is_some() || inv.rabitq_header_bytes.is_some() {
        CacheTier::SearchControl
    } else {
        CacheTier::InvariantMeta
    }
}

/// Bytes an invariants entry contributes. Whole-scan entries are dominated by
/// Region A; probe-only entries contain only small control metadata.
fn inv_bytes(inv: &SegmentInvariants) -> usize {
    inv.region_bytes.as_ref().map_or(0, |bytes| bytes.len())
        + inv.header_bytes.len()
        + inv.footer_bytes.len()
        + inv.a0_bytes.as_ref().map_or(0, |bytes| bytes.len())
        + inv
            .rabitq_header_bytes
            .as_ref()
            .map_or(0, |bytes| bytes.len())
}

enum InvariantLookup {
    L1(Arc<SegmentInvariants>),
    L2(Arc<SegmentInvariants>),
    Miss,
}

impl InvariantLookup {
    fn value(&self) -> Option<Arc<SegmentInvariants>> {
        match self {
            Self::L1(value) | Self::L2(value) => Some(value.clone()),
            Self::Miss => None,
        }
    }
}

const INVARIANT_CONTROL_MAGIC: &[u8; 8] = b"PXINV001";
const OPTIONAL_BYTES_NONE: u64 = u64::MAX;

fn encode_invariant_control(inv: &SegmentInvariants) -> Vec<u8> {
    let a0_len = inv
        .a0_bytes
        .as_ref()
        .map_or(OPTIONAL_BYTES_NONE, |bytes| bytes.len() as u64);
    let rabitq_header_len = inv
        .rabitq_header_bytes
        .as_ref()
        .map_or(OPTIONAL_BYTES_NONE, |bytes| bytes.len() as u64);
    let mut out = Vec::with_capacity(
        40 + inv.header_bytes.len()
            + inv.footer_bytes.len()
            + inv.a0_bytes.as_ref().map_or(0, |bytes| bytes.len())
            + inv
                .rabitq_header_bytes
                .as_ref()
                .map_or(0, |bytes| bytes.len()),
    );
    out.extend_from_slice(INVARIANT_CONTROL_MAGIC);
    out.extend_from_slice(&(inv.header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&(inv.footer_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&a0_len.to_le_bytes());
    out.extend_from_slice(&rabitq_header_len.to_le_bytes());
    out.extend_from_slice(&inv.header_bytes);
    out.extend_from_slice(&inv.footer_bytes);
    if let Some(bytes) = &inv.a0_bytes {
        out.extend_from_slice(bytes);
    }
    if let Some(bytes) = &inv.rabitq_header_bytes {
        out.extend_from_slice(bytes);
    }
    out
}

fn decode_invariant_control(bytes: &[u8]) -> Option<SegmentInvariants> {
    if bytes.len() < 40 || &bytes[..8] != INVARIANT_CONTROL_MAGIC {
        return None;
    }
    let read_len = |start: usize| -> Option<u64> {
        Some(u64::from_le_bytes(
            bytes.get(start..start + 8)?.try_into().ok()?,
        ))
    };
    let header_len = usize::try_from(read_len(8)?).ok()?;
    let footer_len = usize::try_from(read_len(16)?).ok()?;
    let a0_len_raw = read_len(24)?;
    let rabitq_len_raw = read_len(32)?;
    let a0_len = (a0_len_raw != OPTIONAL_BYTES_NONE)
        .then(|| usize::try_from(a0_len_raw).ok())
        .flatten();
    let rabitq_len = (rabitq_len_raw != OPTIONAL_BYTES_NONE)
        .then(|| usize::try_from(rabitq_len_raw).ok())
        .flatten();
    if (a0_len_raw != OPTIONAL_BYTES_NONE && a0_len.is_none())
        || (rabitq_len_raw != OPTIONAL_BYTES_NONE && rabitq_len.is_none())
    {
        return None;
    }
    let mut cursor = 40usize;
    let take = |cursor: &mut usize, len: usize| -> Option<&[u8]> {
        let end = cursor.checked_add(len)?;
        let slice = bytes.get(*cursor..end)?;
        *cursor = end;
        Some(slice)
    };
    let header_bytes = take(&mut cursor, header_len)?.to_vec();
    let footer_bytes = take(&mut cursor, footer_len)?.to_vec();
    let a0_bytes = match a0_len {
        Some(len) => Some(Arc::from(take(&mut cursor, len)?)),
        None => None,
    };
    let rabitq_header_bytes = match rabitq_len {
        Some(len) => Some(Arc::from(take(&mut cursor, len)?)),
        None => None,
    };
    if cursor != bytes.len() {
        return None;
    }
    Some(SegmentInvariants {
        header_bytes,
        region_bytes: None,
        footer_bytes,
        a0_bytes,
        rabitq_header_bytes,
    })
}

impl SegmentInvariantsCache {
    /// `byte_budget` caps the total cached bytes (Region A region_bytes dominate).
    pub fn new(byte_budget: usize) -> Self {
        let shards = (0..INVARIANTS_CACHE_SHARDS)
            .map(|_| std::sync::RwLock::new(std::collections::HashMap::new()))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            shards,
            bytes_used: std::sync::atomic::AtomicUsize::new(0),
            tick: std::sync::atomic::AtomicU64::new(0),
            byte_budget,
            l2_store: None,
            l2_hits: std::sync::atomic::AtomicU64::new(0),
            l2_misses: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub fn with_l2(byte_budget: usize, store: Arc<PersistentByteStore>) -> Self {
        let mut cache = Self::new(byte_budget);
        cache.l2_store = Some(store);
        cache
    }

    fn l2_control_key(path: &str) -> String {
        format!("invariants/control/{path}")
    }

    fn l2_region_key(path: &str) -> String {
        format!("invariants/region-a/{path}")
    }

    /// DRAM lookup followed by persistent-L2 promotion.
    async fn get_or_promote(&self, path: &str) -> InvariantLookup {
        if let Some(value) = self.get(path) {
            return InvariantLookup::L1(value);
        }
        let Some(store) = &self.l2_store else {
            return InvariantLookup::Miss;
        };
        let control_key = Self::l2_control_key(path);
        let control = match store.get(&control_key).await {
            Ok(Some(bytes)) => match decode_invariant_control(&bytes) {
                Some(control) => control,
                None => {
                    store.remove(&control_key);
                    self.l2_misses
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return InvariantLookup::Miss;
                }
            },
            Ok(None) | Err(_) => {
                self.l2_misses
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return InvariantLookup::Miss;
            }
        };
        // A0-backed segments probe bounded Region-A ranges. Rehydrating their
        // complete Region A here would make a process restart consume memory
        // proportional to the largest compacted segment, defeating the spill
        // writer's bound. One-level segments have no ranged probe path and
        // therefore retain the existing whole-region promotion behavior.
        let region_bytes = if control.a0_bytes.is_some() {
            None
        } else {
            store.get(&Self::l2_region_key(path)).await.ok().flatten()
        };
        let value = Arc::new(SegmentInvariants {
            region_bytes,
            ..control
        });
        self.put(path.to_string(), value.clone());
        self.l2_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        InvariantLookup::L2(value)
    }

    /// Write through to L2, then admit into DRAM. Disk-cache errors are
    /// fail-open because object storage remains authoritative.
    pub async fn put_with_l2(&self, path: String, inv: Arc<SegmentInvariants>) {
        if let Some(store) = &self.l2_store {
            let control = Arc::from(encode_invariant_control(&inv));
            let _ = store
                .put(Self::l2_control_key(&path), L2Class::Invariants, control)
                .await;
            if let Some(region) = &inv.region_bytes {
                let _ = store
                    .put(
                        Self::l2_region_key(&path),
                        L2Class::Invariants,
                        region.clone(),
                    )
                    .await;
            }
        }
        self.put(path, inv);
    }

    /// Persist control bytes plus a file-backed Region A without first
    /// materializing the full region in DRAM. The control entry is admitted to
    /// L1 immediately; probed Region-A ranges are served from L2 through
    /// [`Self::get_region_range`] after restart or under DRAM pressure.
    pub async fn seed_control_and_region_from_file(
        &self,
        path: String,
        control: Arc<SegmentInvariants>,
        local_path: &std::path::Path,
        region_off: u64,
        region_len: u64,
    ) -> std::io::Result<bool> {
        let Some(store) = &self.l2_store else {
            self.put(path, control);
            return Ok(false);
        };
        let encoded = Arc::from(encode_invariant_control(&control));
        store
            .put(Self::l2_control_key(&path), L2Class::Invariants, encoded)
            .await?;
        store
            .put_file_range(
                Self::l2_region_key(&path),
                L2Class::Invariants,
                local_path,
                region_off,
                region_len,
            )
            .await?;
        self.put(path, control);
        Ok(true)
    }

    /// Read one Region-A subrange relative to the region start. DRAM is checked
    /// first; persistent L2 is range-verified without hydrating the full region.
    async fn get_region_range(&self, path: &str, relative_off: u64, len: u64) -> Option<Arc<[u8]>> {
        if let Some(region) = self.get(path).and_then(|entry| entry.region_bytes.clone()) {
            let start = usize::try_from(relative_off).ok()?;
            let len = usize::try_from(len).ok()?;
            let end = start.checked_add(len)?;
            return region.get(start..end).map(Arc::from);
        }
        let store = self.l2_store.as_ref()?;
        match store
            .get_range(&Self::l2_region_key(path), relative_off, len)
            .await
        {
            Ok(Some(bytes)) => {
                self.l2_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                crate::observability::io_trace::record_l2s(1, 0);
                Some(bytes)
            }
            Ok(None) | Err(_) => {
                self.l2_misses
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                crate::observability::io_trace::record_l2s(0, 1);
                None
            }
        }
    }

    pub fn l2_stats(&self) -> L2CacheStats {
        L2CacheStats {
            hits: self.l2_hits.load(std::sync::atomic::Ordering::Relaxed),
            misses: self.l2_misses.load(std::sync::atomic::Ordering::Relaxed),
            resident_bytes: self
                .l2_store
                .as_ref()
                .map_or(0, |l2| l2.resident_bytes_for(L2Class::Invariants)),
        }
    }

    fn shard_for(
        &self,
        path: &str,
    ) -> &std::sync::RwLock<std::collections::HashMap<String, CacheEntry>> {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        path.hash(&mut h);
        &self.shards[(h.finish() as usize) % self.shards.len()]
    }

    fn next_tick(&self) -> u64 {
        self.tick.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1
    }

    /// Credit bytes back without underflow (saturating; relaxed — the budget
    /// check tolerates transient slack).
    fn sub_bytes(&self, n: usize) {
        use std::sync::atomic::Ordering;
        let _ = self
            .bytes_used
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(n))
            });
    }

    /// TD-CACHE-2 S2b: internal key of the Region-A companion entry for
    /// `path` (`\u{1}` cannot occur in a segment path). Control bytes and the
    /// Region-A body are stored as SEPARATE entries so budget pressure can
    /// shed a 24 MB region while keeping the few-KB control plane that saves
    /// 3 GETs per cold query — per-class eviction without changing the read
    /// contract (`get` recomposes).
    fn region_entry_key(path: &str) -> String {
        format!("{path}\u{1}A")
    }

    /// On hit, return the cached invariants (recomposed from the control
    /// entry + optional Region-A companion) and stamp recency. Takes only
    /// shard READ locks (TD-CACHE-2 S2a).
    pub fn get(&self, path: &str) -> Option<Arc<SegmentInvariants>> {
        let Some(control) = self.get_entry(path) else {
            // Control anchor gone: a resident Region-A companion is an orphan
            // (nothing can recompose it) — self-heal by freeing its bytes.
            self.invalidate_entry(&Self::region_entry_key(path));
            return None;
        };
        if control.region_bytes.is_some() {
            return Some(control); // legacy merged entry (pre-split put)
        }
        match self.get_entry(&Self::region_entry_key(path)) {
            Some(region) if region.region_bytes.is_some() => Some(Arc::new(SegmentInvariants {
                header_bytes: control.header_bytes.clone(),
                region_bytes: region.region_bytes.clone(),
                footer_bytes: control.footer_bytes.clone(),
                a0_bytes: control.a0_bytes.clone(),
                rabitq_header_bytes: control.rabitq_header_bytes.clone(),
            })),
            _ => Some(control),
        }
    }

    fn get_entry(&self, path: &str) -> Option<Arc<SegmentInvariants>> {
        let shard = self.shard_for(path).read().ok()?;
        let entry = shard.get(path)?;
        entry
            .last_hit
            .store(self.next_tick(), std::sync::atomic::Ordering::Relaxed);
        Some(entry.inv.clone())
    }

    /// Resident bytes currently held (TD-METRICS-1 gauge source).
    pub fn bytes_used(&self) -> usize {
        self.bytes_used.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Insert; while over the byte budget, evict the entry with the lowest
    /// `evict_priority`, ties by least-recent touch (TD-CACHE-2 S1). Victim
    /// search takes per-shard READ locks to find the global (priority, recency)
    /// minimum, then confirms-and-removes under that shard's WRITE lock — no
    /// lock is held across shards, and a racing removal simply retries.
    pub fn put(&self, path: String, inv: Arc<SegmentInvariants>) {
        // TD-CACHE-2 S2b: split Region A into its own entry (tier
        // InvariantIndex) so eviction operates per class; the control entry
        // (tier SearchControl/InvariantMeta) keeps everything else.
        if let Some(region) = inv.region_bytes.clone() {
            let control = Arc::new(SegmentInvariants {
                header_bytes: inv.header_bytes.clone(),
                region_bytes: None,
                footer_bytes: inv.footer_bytes.clone(),
                a0_bytes: inv.a0_bytes.clone(),
                rabitq_header_bytes: inv.rabitq_header_bytes.clone(),
            });
            self.put_entry(
                Self::region_entry_key(&path),
                Arc::new(SegmentInvariants {
                    header_bytes: Vec::new(),
                    region_bytes: Some(region),
                    footer_bytes: Vec::new(),
                    a0_bytes: None,
                    rabitq_header_bytes: None,
                }),
            );
            self.put_entry(path, control);
        } else {
            self.put_entry(path, inv);
        }
    }

    fn put_entry(&self, path: String, inv: Arc<SegmentInvariants>) {
        use std::sync::atomic::Ordering;
        let entry_bytes = inv_bytes(&inv);
        // Replacing an existing path: credit the old entry's bytes back.
        if let Ok(mut shard) = self.shard_for(&path).write()
            && let Some(old) = shard.remove(&path)
        {
            self.sub_bytes(inv_bytes(&old.inv));
        }
        // Evict by (priority asc, last_hit asc) until the new entry fits.
        // Bounded retries guard against pathological races.
        let mut attempts = 0;
        while self.bytes_used().saturating_add(entry_bytes) > self.byte_budget && attempts < 64 {
            attempts += 1;
            // Phase 1: find the global minimum under per-shard read locks.
            let mut victim: Option<(u8, u64, usize, String)> = None;
            for (idx, shard) in self.shards.iter().enumerate() {
                if let Ok(guard) = shard.read() {
                    for (key, entry) in guard.iter() {
                        // S2b: zero-byte entries (e.g. an anchor whose bytes
                        // live in the companion) free nothing — skip them.
                        if inv_bytes(&entry.inv) == 0 {
                            continue;
                        }
                        let prio = entry_tier(&entry.inv).evict_priority();
                        let hit = entry.last_hit.load(Ordering::Relaxed);
                        if victim.as_ref().is_none_or(|v| (prio, hit) < (v.0, v.1)) {
                            victim = Some((prio, hit, idx, key.clone()));
                        }
                    }
                }
            }
            // Phase 2: confirm-and-remove under the victim shard's write lock.
            match victim {
                Some((_, _, idx, key)) => {
                    if let Ok(mut guard) = self.shards[idx].write()
                        && let Some(removed) = guard.remove(&key)
                    {
                        self.sub_bytes(inv_bytes(&removed.inv));
                    }
                    // Removed elsewhere in a race → loop re-checks the budget.
                }
                None => break, // cache empty; admit regardless (budget < entry)
            }
        }
        let tick = self.next_tick();
        if let Ok(mut shard) = self.shard_for(&path).write() {
            self.bytes_used.fetch_add(entry_bytes, Ordering::Relaxed);
            shard.insert(
                path,
                CacheEntry {
                    inv,
                    last_hit: std::sync::atomic::AtomicU64::new(tick),
                },
            );
        }
    }

    /// Remove a path from DRAM.
    ///
    /// Call [`Self::invalidate_all`] when the backing immutable segment has
    /// been retired and persistent entries must be removed as well.
    pub fn invalidate(&self, path: &str) {
        self.invalidate_entry(path);
        self.invalidate_entry(&Self::region_entry_key(path));
    }

    /// Remove both DRAM and persistent entries for a retired segment.
    ///
    /// The persistent sweep shares the store's publication mutex, so an
    /// invalidation racing a cache seed cannot leave a persistent-only entry.
    pub async fn invalidate_all(&self, path: &str) {
        self.invalidate(path);
        if let Some(store) = &self.l2_store {
            let control_key = Self::l2_control_key(path);
            let region_key = Self::l2_region_key(path);
            store
                .remove_where(|key| key == control_key || key == region_key)
                .await;
        }
    }

    /// Remove every invariant entry below a retired collection directory.
    ///
    /// The trailing separator prevents collection `12` from matching `123`.
    /// This is resource reclamation rather than row-level coherence: PAX
    /// segments are immutable, but a deleted collection must not retain dead
    /// DRAM/local-disk capacity until ordinary eviction.
    pub async fn invalidate_prefix_all(&self, path_prefix: &str) -> usize {
        let prefix = format!("{}/", path_prefix.trim_end_matches('/'));
        let mut removed = 0;
        for shard in &self.shards {
            let Ok(mut entries) = shard.write() else {
                continue;
            };
            let victims: Vec<String> = entries
                .keys()
                .filter(|path| path.starts_with(&prefix))
                .cloned()
                .collect();
            for path in victims {
                if let Some(entry) = entries.remove(&path) {
                    self.sub_bytes(inv_bytes(&entry.inv));
                    removed += 1;
                }
            }
        }
        if let Some(store) = &self.l2_store {
            let control_prefix = format!("invariants/control/{prefix}");
            let region_prefix = format!("invariants/region-a/{prefix}");
            removed += store
                .remove_where(|key| {
                    key.starts_with(&control_prefix) || key.starts_with(&region_prefix)
                })
                .await;
        }
        removed
    }

    fn invalidate_entry(&self, key: &str) {
        if let Ok(mut shard) = self.shard_for(key).write()
            && let Some(removed) = shard.remove(key)
        {
            self.sub_bytes(inv_bytes(&removed.inv));
        }
    }
}

/// Install writer-retained PAX regions only after the caller has atomically
/// published `path`. A failed cache write never invalidates the committed
/// segment; the next query falls back to normal read-through.
pub async fn install_pax_cache_seed(
    path: &str,
    seed: proximadb_storage_common::pax_block::PaxCacheSeed,
    invariants: Option<&SegmentInvariantsCache>,
    survivor: Option<&SurvivorRangeCache>,
) {
    if let Some(cache) = invariants {
        cache
            .put_with_l2(
                path.to_string(),
                Arc::new(SegmentInvariants {
                    header_bytes: seed.header_bytes,
                    region_bytes: Some(seed.rabitq_bytes),
                    footer_bytes: seed.footer_bytes,
                    a0_bytes: seed.a0_bytes,
                    rabitq_header_bytes: Some(seed.rabitq_header_bytes),
                }),
            )
            .await;
    }
    if let (Some(cache), Some(sq8)) = (survivor, seed.sq8_bytes)
        && let Err(error) = cache.seed_parent_region(path, seed.sq8_off, sq8).await
    {
        tracing::warn!("post-publication SQ8 cache seed failed for {path}: {error}");
    }
}

fn read_local_segment_range(
    file: &mut std::fs::File,
    path: &Path,
    offset: u64,
    len: u64,
) -> anyhow::Result<Vec<u8>> {
    let len = usize::try_from(len)
        .map_err(|_| anyhow::anyhow!("local PAX range exceeds address space: {len}"))?;
    file.seek(SeekFrom::Start(offset)).map_err(|error| {
        anyhow::anyhow!("seek local PAX {} at {offset}: {error}", path.display())
    })?;
    let mut bytes = vec![0u8; len];
    file.read_exact(&mut bytes).map_err(|error| {
        anyhow::anyhow!(
            "read local PAX {} range {offset}+{}: {error}",
            path.display(),
            len
        )
    })?;
    Ok(bytes)
}

/// Promote immutable Region A/B bytes from a disk-backed writer's local PAX
/// after its final object-store path has been atomically published.
///
/// Large regions stream into persistent L2 in fixed-size chunks. Only the
/// header, A0, RaBitQ parameters, and footer are materialized, so enabling
/// cache-on-write does not undo local-spill compaction's memory bound.
pub async fn install_pax_cache_seed_from_local_file(
    published_path: &str,
    local_path: &Path,
    include_sq8: bool,
    invariants: Option<&SegmentInvariantsCache>,
    survivor: Option<&SurvivorRangeCache>,
) -> anyhow::Result<bool> {
    use proximadb_storage_common::segment_layout::{
        SEG_HEADER_PREFIX_LEN, SegmentFooterIndex, SegmentHeaderPrefix,
    };

    let mut file = std::fs::File::open(local_path)
        .map_err(|error| anyhow::anyhow!("open local PAX {}: {error}", local_path.display()))?;
    let file_len = file
        .metadata()
        .map_err(|error| anyhow::anyhow!("stat local PAX {}: {error}", local_path.display()))?
        .len();
    let header_len = (SEG_HEADER_PREFIX_LEN as u64).min(file_len);
    let header_bytes = read_local_segment_range(&mut file, local_path, 0, header_len)?;
    let header = SegmentHeaderPrefix::parse(&header_bytes)
        .map_err(|error| anyhow::anyhow!("parse local PAX header: {error}"))?;
    let footer_bytes =
        read_local_segment_range(&mut file, local_path, header.footer_off, header.footer_len)?;
    let footer = SegmentFooterIndex::parse(&footer_bytes)
        .map_err(|error| anyhow::anyhow!("parse local PAX footer: {error}"))?;
    let a0_bytes = if header.a0_len > 0 {
        Some(Arc::from(read_local_segment_range(
            &mut file,
            local_path,
            header.a0_off,
            header.a0_len,
        )?))
    } else {
        None
    };
    let rabitq_header_len =
        u64::try_from(proximadb_block_format::region_header_len(footer.embed_dim))
            .map_err(|_| anyhow::anyhow!("RaBitQ header length exceeds u64"))?
            .min(header.rabitq_len);
    let rabitq_header_bytes = Arc::from(read_local_segment_range(
        &mut file,
        local_path,
        header.rabitq_off,
        rabitq_header_len,
    )?);
    let control = Arc::new(SegmentInvariants {
        header_bytes,
        region_bytes: None,
        footer_bytes,
        a0_bytes,
        rabitq_header_bytes: Some(rabitq_header_bytes),
    });

    let mut seeded = false;
    if let Some(cache) = invariants {
        seeded |= cache
            .seed_control_and_region_from_file(
                published_path.to_string(),
                control,
                local_path,
                header.rabitq_off,
                header.rabitq_len,
            )
            .await?;
    }
    if include_sq8 && let Some(cache) = survivor {
        seeded |= cache
            .seed_parent_region_from_file(
                published_path,
                header.sq8_off,
                header.sq8_len,
                local_path,
            )
            .await
            .map_err(|error| anyhow::anyhow!("seed local PAX SQ8 region: {error}"))?;
    }
    Ok(seeded)
}

/// TD-CACHE-1 S1: prefill a segment's CONTROL-plane invariants (header prefix,
/// footer index, A0 coarse directory) into the invariants cache WITHOUT
/// searching — the exact bytes the first query would otherwise fetch as 3
/// ranged GETs. Region A/B payloads are deliberately NOT prefetched (they
/// enrich lazily via the search path's cache merge), keeping warming cost to
/// a few KB per segment. Used by tier-gated boot warming and manifest replay;
/// never called speculatively for idle collections (co-design: warming is
/// demand-proven or contract-driven, not a boot sweep).
///
/// Best-effort: any error leaves the cache unchanged (first query pays the
/// legacy path). A non-coalesced/legacy segment (no `PXH1` prefix) is skipped.
pub async fn prefetch_segment_invariants(
    fs: &dyn proximadb_storage_filesystem_types::FileSystem,
    path: &str,
    cache: &SegmentInvariantsCache,
) -> anyhow::Result<bool> {
    use proximadb_storage_common::segment_layout::SegmentHeaderPrefix;
    if cache.get(path).is_some() {
        return Ok(false); // already warm
    }
    let size = fs
        .metadata(path)
        .await
        .map_err(|e| anyhow::anyhow!("prefetch stat {path}: {e}"))?
        .size;
    if size < (SEG_HEADER_PREFIX_LEN as u64 + SEGMENT_MAGIC.len() as u64) {
        return Ok(false);
    }
    let read_len = (SEG_HEADER_PREFIX_LEN as u64).min(size);
    let header_bytes = fs
        .read_range(path, 0, read_len)
        .await
        .map_err(|e| anyhow::anyhow!("prefetch header {path}: {e}"))?;
    let Ok(header) = SegmentHeaderPrefix::parse(&header_bytes) else {
        return Ok(false); // legacy/non-coalesced segment — nothing to warm
    };
    let footer_bytes = fs
        .read_range(path, header.footer_off, header.footer_len)
        .await
        .map_err(|e| anyhow::anyhow!("prefetch footer {path}: {e}"))?;
    let a0_bytes = if header.a0_len > 0 {
        Some(Arc::from(
            fs.read_range(path, header.a0_off, header.a0_len)
                .await
                .map_err(|e| anyhow::anyhow!("prefetch a0 {path}: {e}"))?
                .as_slice(),
        ))
    } else {
        None
    };
    cache
        .put_with_l2(
            path.to_string(),
            Arc::new(SegmentInvariants {
                header_bytes,
                region_bytes: None,
                footer_bytes,
                a0_bytes,
                rabitq_header_bytes: None,
            }),
        )
        .await;
    Ok(true)
}

/// ADR-062 / TD-RDSTRAT-6 **scan-then-rerank** over a coalesced-RaBitQ segment
/// on the engine's own filesystem — the ranged analogue of
/// [`rabitq_search_segment`] for the new layout:
///
/// 1. **Scan** the coalesced RaBitQ header region (one ranged GET, header-prefix
///    coalesced in) — `keep=100%`: rank ALL codes → approximate distance for
///    every vector (zero prune loss).
/// 2. **Select** the top-M survivors (M = `pax_rabitq_pool_for_top_k(k, n)`,
///    scaled with the segment's row count so it stays ~1% of the corpus).
/// 3. **Rerank** survivors: because the segment is cluster-ordered
///    (`cluster_order_pca_ivf`), survivors fall in few adjacent blocks → the
///    `ObjectRangeCoalescePolicy` plans merged/coalesced ranged GETs → fetch the
///    survivor blocks, decode the UNCHANGED SQ8 `EMBED_BASE` stripe, score.
/// 4. **Finalize** top-k.
///
/// Net: ~1 RaBitQ GET + 1 footer GET + a few coalesced survivor GETs — vs the
/// legacy per-block cascade's ~4-5 GETs × every selected block. Returns
/// `Ok(None)` when `path` is not a coalesced segment (the caller falls back to
/// the legacy in-block RaBitQ path); physical bytes/GETs are traced by the fs
/// layer's `read_range` calls.
///
/// ADR-065 cache-co-design: the read-type taxonomy. One enum drives BOTH (a) the
/// per-tier GET trace (billing/latency accounting) AND (b) the cache's per-tier
/// admission/eviction policy (`evict_priority`). Region A is the hottest, largest
/// read — a cache hit saves a full ~24 MB GET, so it gets the highest priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheTier {
    /// Region A (RaBitQ index scan): ~24 MB, re-read every query (hottest).
    InvariantIndex,
    /// Header / A0 / RaBitQ parameters: the mandatory ANN control prefix.
    SearchControl,
    /// Tail footer / block map / SQ8 parameters: tiny, always-useful.
    InvariantMeta,
    /// Selected Region-A IVF-cell code runs.
    ProbeIndex,
    /// Region B survivor SQ8 ranges: variable, query-dependent.
    SurvivorPayload,
    /// Region D OID ranges: variable, query-dependent.
    ResultPayload,
}

impl CacheTier {
    /// A short label for the trace output.
    pub fn label(self) -> &'static str {
        match self {
            Self::InvariantIndex => "IdxA",
            Self::SearchControl => "Ctl",
            Self::InvariantMeta => "Meta",
            Self::ProbeIndex => "ProbeA",
            Self::SurvivorPayload => "Surv",
            Self::ResultPayload => "OID",
        }
    }
    /// Eviction priority (lower = evict first under memory pressure). Region A is
    /// pinned (a hit saves the most); query-dependent ranges are evicted first.
    pub fn evict_priority(self) -> u8 {
        match self {
            Self::ProbeIndex | Self::SurvivorPayload => 0,
            Self::ResultPayload => 1,
            Self::InvariantMeta => 2,
            Self::InvariantIndex | Self::SearchControl => 3,
        }
    }
}

/// ADR-065 co-design diagnostic: per-GET (tier, size) trace. `record_get` pushes
/// each read_range's tier + on-disk length when `PROXIMADB_TRACE_GETS` is set;
/// `drain_get_trace` returns + clears them so the eval prints the per-tier
/// distribution (GET count = same-region billing; sizes = latency). Zero cost off.
static GET_TRACE: std::sync::Mutex<Vec<(CacheTier, u64)>> = std::sync::Mutex::new(Vec::new());

fn record_get(tier: CacheTier, len: u64) {
    if let Ok(mut v) = GET_TRACE.lock() {
        v.push((tier, len));
    }
}

/// TD-RDSTRAT-13 C1: always-on per-tier physical-GET attribution — the
/// per-query io_trace counters plus the aggregate Prometheus surface — plus
/// the env-gated diagnostic `record_get` push when `trace_on`. One call per
/// physical ranged GET keeps the 1:1 recording invariant the wave code
/// documents (batch-time recording for waved ranges).
fn record_get_physical(tier: CacheTier, len: u64, trace_on: bool) {
    crate::observability::io_trace::record_tier_get(tier as usize, len);
    crate::storage::engines::sst::metrics::record_tier_get(tier as usize, len);
    if trace_on {
        record_get(tier, len);
    }
}

/// Drain the recorded per-GET (tier, size) pairs (empty when trace is off).
pub fn drain_get_trace() -> Vec<(CacheTier, u64)> {
    GET_TRACE
        .lock()
        .map(|mut v| std::mem::take(&mut *v))
        .unwrap_or_default()
}

/// TD-RDSTRAT-8 PR-B diagnostic: per-query coarse-probe counters
/// `(cells_total, cells_probed, probed_rows, fetch_rounds)`. The probe path
/// pushes them when `PROXIMADB_TRACE_GETS` is set; the eval/recall harness drains
/// them to prove `cells_probed << cells_total` and the GET/byte reduction. Zero
/// cost off. (Promoting these into the durable Prometheus io_trace surface —
/// `ivf_cells_total/probed`, `probed_rows`, `fetch_rounds`,
/// `whole_region_fallback` — is a tracked follow-up.)
static PROBE_TRACE: std::sync::Mutex<Vec<(u64, u64, u64, u64)>> = std::sync::Mutex::new(Vec::new());

fn record_probe_trace(cells_total: u64, cells_probed: u64, probed_rows: u64, fetch_rounds: u64) {
    if let Ok(mut v) = PROBE_TRACE.lock() {
        v.push((cells_total, cells_probed, probed_rows, fetch_rounds));
    }
}

/// Record a coarse-probe outcome to BOTH durable surfaces (TD-RDSTRAT-8):
/// the per-query io_trace → warehouse `VectorAnnPayload` (per-tenant, durable)
/// and the aggregate Prometheus operator view. Call whenever the probe engages
/// or armed-but-missed falls back to the whole region scan.
fn record_ivf_probe_durable(
    cells_total: u64,
    cells_probed: u64,
    probed_rows: u64,
    fetch_rounds: u64,
    whole_region_fallback: bool,
) {
    crate::observability::io_trace::record_ivf_coarse_probe(
        cells_total,
        cells_probed,
        probed_rows,
        fetch_rounds,
        whole_region_fallback,
    );
    crate::storage::engines::sst::metrics::record_ivf_coarse_probe(
        cells_total,
        cells_probed,
        probed_rows,
        fetch_rounds,
        whole_region_fallback,
    );
}

/// Drain the recorded coarse-probe counters (empty when trace off / no probe).
pub fn drain_probe_trace() -> Vec<(u64, u64, u64, u64)> {
    PROBE_TRACE
        .lock()
        .map(|mut v| std::mem::take(&mut *v))
        .unwrap_or_default()
}

// ── Coarse-probe (IVF) settings: TOML (`SstConfig.coarse_probe`) → OnceLock ──
// The env-only free fns below consult this static as the FALLBACK default; env
// vars still win (read at call time). Initialized once at SstEngine boot from
// the parsed TOML (src/storage/engines/sst/core.rs), mirroring
// SHARED_INVARIANTS_CACHE. COGS arc: IVF is the master cost lever (~4× per-tenant
// COGS cut vs full-scan, recall ~0.98).

/// Resolved coarse-probe settings (TOML-derived; env overrides applied at call
/// sites, not here). Defaults replicate `CoarseProbeConfig::default()`.
#[derive(Debug, Clone, Copy)]
pub struct CoarseProbeSettings {
    pub enable_read_probe: bool,
    pub enable_write_train: bool,
    pub nprobe_multiplier: f32,
    pub nprobe_min: usize,
    pub nprobe_max: usize,
    /// TD-IVF-3 write-side floor on the coarse-PCA projection width; 0 = legacy
    /// formula. Read by `block_cluster::ivf_ncomp_floor()`.
    pub ncomp_floor: usize,
}

// TD-COMPACT-9: overwritable, NOT first-wins. Every production `SstEngine::new()`
// (shared_services.rs, factory.rs, legacy.rs) is built with `SstConfig::default()`
// (coarse_probe = None), so an OnceLock first-wins seeded the DEFAULT and the
// real `[storage.sst_config.coarse_probe]` TOML was silently ignored (measured:
// nprobe_multiplier=1.5 had no effect while env nprobe DID). An RwLock lets a
// REAL (Some) config — set authoritatively from the loaded server Config in
// `ProximaDB::start` — override the default regardless of construction order.
static SHARED_COARSE_PROBE: std::sync::RwLock<Option<CoarseProbeSettings>> =
    std::sync::RwLock::new(None);

/// Seed the coarse-probe settings. A real (`Some`) config ALWAYS wins (the
/// authoritative call from server startup); a `None` call (a default-config
/// `SstEngine::new()`) only fills the default if nothing real was set yet.
/// Poison-safe (no-panic mandate #4).
pub fn init_coarse_probe_settings(cfg: Option<&crate::core::config::CoarseProbeConfig>) {
    if let Ok(mut g) = SHARED_COARSE_PROBE.write() {
        match cfg {
            Some(c) => *g = Some(c.clone().into()),
            None => {
                g.get_or_insert_with(CoarseProbeSettings::default);
            }
        }
    }
}

impl Default for CoarseProbeSettings {
    fn default() -> Self {
        crate::core::config::CoarseProbeConfig::default().into()
    }
}

impl From<crate::core::config::CoarseProbeConfig> for CoarseProbeSettings {
    fn from(c: crate::core::config::CoarseProbeConfig) -> Self {
        Self {
            enable_read_probe: c.enable_read_probe,
            enable_write_train: c.enable_write_train,
            nprobe_multiplier: c.nprobe_multiplier,
            nprobe_min: c.nprobe_min,
            nprobe_max: c.nprobe_max,
            ncomp_floor: c.ncomp_floor,
        }
    }
}

pub(crate) fn coarse_probe_settings() -> CoarseProbeSettings {
    SHARED_COARSE_PROBE
        .read()
        .ok()
        .and_then(|g| *g)
        .unwrap_or_default()
}

/// Geometric nprobe: `ceil(sqrt(k_c) × multiplier)`, clamped to `[min, max]`
/// then to `k_c`. Sub-linear in corpus (V^0.25): ~56× fewer ranks than full-scan
/// at 1M. The default multiplier 2.0 is recall-ratcheted by the settled
/// one-segment SIFT1M geometry (11/30 cells, recall@10 >= 0.98); operators may
/// trade quality for fewer bytes via TOML. The max-then-min ordering avoids the
/// `clamp(min, k_c)` panic when `k_c < min` (no-panic mandate #4).
fn geometric_nprobe(k_c: usize, s: CoarseProbeSettings) -> usize {
    let raw = ((k_c as f32).sqrt() * s.nprobe_multiplier).ceil() as usize;
    let floored = raw.max(s.nprobe_min);
    let bounded = if s.nprobe_max > 0 {
        floored.min(s.nprobe_max)
    } else {
        floored
    };
    bounded.min(k_c.max(1))
}

/// TD-RDSTRAT-8 PR-B gate: engage the Region-A0 coarse probe on v3 segments.
/// Default **ON** since 2026-07-26 — the flip precondition was met by the
/// nprobe sweep (fixed-slice recall@10 0.9860–0.9870 ≥ the 0.984 ratchet at
/// the default nprobe, with 81 GETs / 92 ms vs 108 GETs / 144 ms unprobed:
/// strictly better on every axis; ledger claim `nprobe_sweep_trained_1m`).
/// Precedence: env kill-switch (`PROXIMADB_PAX_READ_COARSE_PROBE=0|off|false|no`)
/// → else TOML `[storage.sst_config.coarse_probe] enable_read_probe`.
/// Mixed-safe: v1 / A0-less segments never probe regardless.
/// (Retired alias name: `PROXIMADB_IVF2_PROBE` — TD-ENVGATE-1.)
pub fn coarse_probe_enabled() -> bool {
    match std::env::var("PROXIMADB_PAX_READ_COARSE_PROBE") {
        // Env set: ON unless it's an explicit kill-switch value.
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
        // Env unset: fall back to the TOML/config default.
        Err(_) => coarse_probe_settings().enable_read_probe,
    }
}

/// Number of coarse cells to probe. Precedence: env
/// `PROXIMADB_PAX_READ_COARSE_NPROBE` (explicit) → else the geometric default
/// `ceil(sqrt(k_c) × multiplier)` from the TOML config
/// (`[storage.sst_config.coarse_probe] nprobe_multiplier/min/max`), clamped to
/// `k_c`. `>= k_c` is exact mode (every cell).
fn coarse_probe_nprobe(k_c: usize) -> usize {
    std::env::var("PROXIMADB_PAX_READ_COARSE_NPROBE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or_else(|| geometric_nprobe(k_c, coarse_probe_settings()))
        .min(k_c.max(1))
}

/// Outcome of a coarse probe: the global survivor rows plus the io_trace
/// counters (`ivf_cells_total/probed`, `probed_rows`, `fetch_rounds`).
struct CoarseProbeResult {
    survivors: Vec<usize>,
    cells_total: u64,
    cells_probed: u64,
    probed_rows: u64,
    fetch_rounds: u64,
    a0_bytes: Arc<[u8]>,
    rabitq_header_bytes: Arc<[u8]>,
    /// TD-RDSTRAT-13 C4: footer bytes if this probe fetched them (rode the
    /// Region-A wave or overlapped the rank). `None` = the caller owns the
    /// footer (cache hit classified Ready) or the probe never ran.
    footer_bytes: Option<Vec<u8>>,
}

/// TD-RDSTRAT-13 C4: the cascade's footer classification — `Ready` (served
/// from cache/prefix; the probe must not touch it) or `Pending` (cold; ride
/// the probe's Region-A wave, or overlap the rank when there is no wave).
enum FooterPlan {
    Ready(Vec<u8>),
    Pending { off: u64, len: u64 },
}

fn prefetched_slice(prefix: &[u8], offset: u64, len: u64) -> Option<&[u8]> {
    let start = usize::try_from(offset).ok()?;
    let len = usize::try_from(len).ok()?;
    let end = start.checked_add(len)?;
    prefix.get(start..end)
}

const DEFAULT_PREFIX_PREFETCH_BYTES: u64 = 1024 * 1024;
const MAX_PREFIX_PREFETCH_BYTES: u64 = 8 * 1024 * 1024;

fn prefix_prefetch_bytes() -> Option<u64> {
    let configured = std::env::var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_PREFIX_PREFETCH_BYTES);
    (configured > 0).then_some(configured.min(MAX_PREFIX_PREFETCH_BYTES))
}

/// Largest known coalesced header-prefix length across layout versions
/// (TD-PAXRG-1: v4 = 88 B). Prefix GETs fetch AT LEAST this many bytes
/// (clamped to file size) so one read covers v1/v3/v4; `SegmentHeaderPrefix::parse`
/// branches on the version byte.
pub(crate) fn coalesced_header_prefetch_floor() -> usize {
    SEG_HEADER_PREFIX_LEN
}

fn split_probe_metadata_cache_enabled() -> bool {
    // Metadata population is the shipped behavior: the coarse-probe path
    // already fetched header/A0/RaBitQ params/footer, and failing to retain
    // those immutable bytes makes every subsequent query repay four GETs.
    // Preserve the pre-GA positive gate as an emergency opt-out for operators
    // that explicitly set a false value; unset must remain ON.
    std::env::var("PROXIMADB_PAX_SPLIT_PROBE_META_CACHE").map_or(true, |value| {
        !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        )
    })
}

/// TD-RDSTRAT-8 PR-B: compute survivors from ONLY the `nprobe` nearest coarse
/// cells — rank the persisted `k_c` centroids in RAM, then ranged-read just those
/// cells' Region-A code extents (`a_off/a_len`). The whole Region A is never
/// fetched, so cold-query GETs/bytes drop from O(N) to O(nprobe). Returns `None`
/// to fall back to the single-level whole-region scan (fail-safe: A0
/// missing/corrupt, dim mismatch, or degenerate directory).
#[allow(clippy::too_many_arguments)]
async fn coarse_probe_survivors(
    fs: &dyn proximadb_storage_filesystem_types::FileSystem,
    path: &str,
    header: &proximadb_storage_common::segment_layout::SegmentHeaderPrefix,
    query: &[f32],
    metric: RankMetric,
    k: usize,
    trace_on: bool,
    cached: Option<&SegmentInvariants>,
    prefix: &[u8],
    invariants_cache: Option<&SegmentInvariantsCache>,
    survivor_cache: Option<&SurvivorRangeCache>,
    // TD-FPRUNE-1 M3: when a filter is active, the probe becomes a FILTERED ANN —
    // adaptive nprobe by matching-row count + rank restricted to allowed rows.
    row_allow: Option<&proximadb_block_format::RowAllow>,
    // TD-RDSTRAT-13 C4: a cold footer rides this probe's wave (or overlaps the
    // rank when every probe range is warm) instead of serializing after it.
    footer: FooterPlan,
) -> Result<Option<CoarseProbeResult>> {
    use proximadb_storage_common::coarse_directory::{CoarseDirectory, project_with_model};

    let dim = query.len();
    if dim == 0 || header.a0_len == 0 {
        return Ok(None);
    }

    // 1. Region A0 (small, immediately after the header) — one ranged GET.
    let a0_bytes: Arc<[u8]> = if let Some(bytes) = cached.and_then(|inv| inv.a0_bytes.clone()) {
        bytes
    } else if let Some(bytes) = prefetched_slice(prefix, header.a0_off, header.a0_len) {
        Arc::from(bytes.to_vec())
    } else {
        let bytes = fs
            .read_range(path, header.a0_off, header.a0_len)
            .await
            .map_err(|e| anyhow::anyhow!("coarse-probe A0 {path}: {e}"))?;
        record_get_physical(CacheTier::SearchControl, header.a0_len, trace_on);
        Arc::from(bytes)
    };
    let Ok(dir) = CoarseDirectory::parse(&a0_bytes) else {
        return Ok(None);
    };
    let k_c = dir.model.k_c();
    let n_comp = dir.model.n_comp as usize;
    if k_c == 0 || dir.cells.len() != k_c || n_comp == 0 || dir.model.dim as usize != dim {
        return Ok(None);
    }

    // 2. Rank the k_c centroids in RAM; select the nprobe nearest NON-EMPTY cells
    //    (project the query with the SAME persisted f32 model the writer used).
    let q_proj = project_with_model(
        &dir.model.pca_mean,
        &dir.model.pca_components,
        n_comp,
        query,
    );
    if q_proj.len() != n_comp {
        return Ok(None);
    }
    let mut cell_dist: Vec<(usize, f32)> = (0..k_c)
        .map(|c| {
            let cen = &dir.model.centroids[c * n_comp..(c + 1) * n_comp];
            let d: f32 = q_proj.iter().zip(cen).map(|(a, b)| (a - b) * (a - b)).sum();
            (c, d)
        })
        .collect();
    cell_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let nprobe = coarse_probe_nprobe(k_c);
    let mut probed: Vec<usize> = match row_allow {
        // TD-FPRUNE-1 M3: FILTERED ANN — walk cells nearest-first and accumulate
        // PREDICATE-MATCHING rows (allow ∩ cell row-range, computed in RAM from
        // the cell directory + the allow-set — no extra I/O). Probe until at least
        // the geometric `nprobe` cells AND enough matching candidates to fill the
        // survivor pool are covered: a selective filter needs MORE cells to reach
        // k matches (the recall fix that the disarmed-probe path traded away).
        // Cells with zero matching rows are skipped (cell ∩ filter empty ⇒ no GET).
        Some(allow) => {
            let target = pax_rabitq_pool_for_top_k(k, allow.len().max(1)).max(k);
            let mut selected = Vec::new();
            let mut matched = 0usize;
            for (c, _) in cell_dist.iter() {
                let cell = &dir.cells[*c];
                if cell.row_end <= cell.row_begin {
                    continue;
                }
                let cell_matched =
                    allow.count_in_range(cell.row_begin as usize..cell.row_end as usize);
                if cell_matched == 0 {
                    continue;
                }
                selected.push(*c);
                matched += cell_matched;
                if selected.len() >= nprobe && matched >= target {
                    break;
                }
            }
            selected
        }
        None => cell_dist
            .iter()
            .filter(|(c, _)| dir.cells[*c].row_end > dir.cells[*c].row_begin)
            .take(nprobe)
            .map(|(c, _)| *c)
            .collect(),
    };
    if probed.is_empty() {
        return Ok(None);
    }
    // File (a_off) order so physically-adjacent probed cells coalesce.
    probed.sort_by_key(|&c| dir.cells[c].a_off);

    // 3. Region-A header (params + centroid) — one small ranged GET.
    let hdr_len = proximadb_block_format::region_header_len(dim as u32) as u64;
    let hdr_bytes: Arc<[u8]> =
        if let Some(bytes) = cached.and_then(|inv| inv.rabitq_header_bytes.clone()) {
            bytes
        } else if let Some(bytes) = prefetched_slice(prefix, header.rabitq_off, hdr_len) {
            Arc::from(bytes.to_vec())
        } else {
            let bytes = fs
                .read_range(path, header.rabitq_off, hdr_len)
                .await
                .map_err(|e| anyhow::anyhow!("coarse-probe region header {path}: {e}"))?;
            record_get_physical(CacheTier::SearchControl, hdr_len, trace_on);
            Arc::from(bytes)
        };
    let Ok(rq_header) = proximadb_block_format::CoalescedRaBitQHeader::parse(&hdr_bytes) else {
        return Ok(None);
    };

    // 4. Ranged-read the probed cells' Region-A code extents. The default
    //    remains gap-zero. The proof gate can admit bounded gap over-read, but
    //    the ranker receives only the selected slices so nprobe never changes.
    let stride = proximadb_block_format::code_stride(dim as u32) as u64;
    let selected: Vec<ProbeCellSlice> = probed
        .iter()
        .map(|&cell_index| {
            let cell = &dir.cells[cell_index];
            ProbeCellSlice {
                start: cell.a_off,
                end: cell.a_off + cell.a_len,
                row_start: cell.row_begin as usize,
            }
        })
        .collect();
    let defaults = default_probe_range_policy(path);
    let policy = resolve_adaptive_range_policy(
        path,
        "region_a",
        "PROXIMADB_PAX_VECTOR_COALESCE_GAP",
        "PROXIMADB_PAX_VECTOR_COALESCE_RANGE",
        defaults,
        |candidate| {
            estimate_coalesced_byte_ranges(
                selected.iter().map(|cell| (cell.start, cell.end)),
                candidate,
            )
        },
    );
    let fetches = plan_probe_cell_ranges(&selected, &policy);
    let mut fetched = Vec::with_capacity(fetches.len());
    let probed_rows: u64 = selected
        .iter()
        .map(|cell| (cell.end - cell.start) / stride)
        .sum();

    // TD-RDSTRAT-12 §3 rewiring: batch the COLD TAIL of probed-cell ranged reads
    // through `FileSystem::read_ranges_prefetch`, whose bounded-concurrency gate
    // (`PROXIMADB_READ_RANGES_INFLIGHT`) turns cold latency from GETs × RTT into
    // rounds × RTT. Cache dynamics are preserved byte-for-byte:
    //   classify pass resolves layers exactly once (region Arc / prefix /
    //   invariants L1+L2); for anything past those, a side-effect-free
    //   survivor-RAM probe distinguishes "would be served without its loader"
    //   (consume via `get_or_fetch` — counters/track_hot_key fire as usual, the
    //   bundled bytes never run) from "true cold" (batch member — ONE physical
    //   GET issued up front, handed to the loader so L1/L2 population and
    //   metrics stay identical to the sequential baseline).
    enum Slot {
        Ready(Vec<u8>),
        /// Survivor-RAM resident: no physical read; consumed via `get_or_fetch`
        /// for counter/tracking parity (its loader will not run).
        SurvivorCached,
        /// True cold: index into the pending batch queue.
        Cold(usize),
        /// TD-RDSTRAT-13 C4: the cold footer rides the wave at queue index 0
        /// (never enters the survivor cache — Meta tier).
        Footer(usize),
    }
    let mut slots: Vec<Slot> = Vec::with_capacity(fetches.len());
    let mut cold_ranges: Vec<std::ops::Range<u64>> = Vec::new();
    // TD-RDSTRAT-13 C4: the classified footer outcome. `Ready` is served
    // as-is; a `Pending` footer is queued at the FRONT of the cold wave when
    // other cold ranges exist (ceil((n+1)/width) adds a round only when
    // n ≡ 0 (mod width)), or held for the rank-overlap join when this probe
    // is otherwise fully warm — a 1-range wave would spend the RTT standing
    // alone in front of the rank rather than hiding under it.
    let mut footer_out: Option<Vec<u8>> = None;
    let mut footer_queued = false;
    let mut footer_pending: Option<(u64, u64)> = match footer {
        FooterPlan::Ready(bytes) => {
            footer_out = Some(bytes);
            None
        }
        FooterPlan::Pending { off, len } => Some((off, len)),
    };
    for fetch in &fetches {
        let len = fetch.end - fetch.start;
        let relative_off = fetch.start.saturating_sub(header.rabitq_off);
        if let Some(region) = cached
            .and_then(|inv| inv.region_bytes.as_ref())
            .and_then(|r| {
                let start = usize::try_from(fetch.start.checked_sub(header.rabitq_off)?).ok()?;
                let end = start.checked_add(usize::try_from(len).ok()?)?;
                r.get(start..end)
            })
        {
            slots.push(Slot::Ready(region.to_vec()));
            continue;
        }
        if let Some(bytes) = prefetched_slice(prefix, fetch.start, len) {
            slots.push(Slot::Ready(bytes.to_vec()));
            continue;
        }
        if let Some(bytes) = match invariants_cache {
            Some(cache) => cache.get_region_range(path, relative_off, len).await,
            None => None,
        } {
            slots.push(Slot::Ready(bytes.to_vec()));
            continue;
        }
        match survivor_cache {
            Some(sc)
                if sc
                    .peek_memory_exact(CacheKind::Other, path, fetch.start, len)
                    .await
                    .is_some() =>
            {
                slots.push(Slot::SurvivorCached);
            }
            _ => {
                slots.push(Slot::Cold(cold_ranges.len()));
                cold_ranges.push(fetch.start..fetch.end);
            }
        }
    }
    // C4 (cont.): queue the pending footer now that the cold set is known.
    if let Some((off, len)) = footer_pending.take() {
        if cold_ranges.is_empty() {
            footer_pending = Some((off, len)); // held for the rank-overlap join
        } else {
            for slot in slots.iter_mut() {
                if let Slot::Cold(i) = slot {
                    *i += 1;
                }
            }
            cold_ranges.insert(0, off..off + len);
            slots.insert(0, Slot::Footer(0));
            footer_queued = true;
        }
    }

    // One bounded-concurrent I/O round for every true-cold range. `INFLIGHT`
    // caps in-flight GETs here; sequential behavior when unset/≤1 is unchanged
    // from the baseline loop (the cap defers to PARALLEL).
    //
    // Metrics parity: every buffer below corresponds to exactly one physical
    // GET the sequential path would have issued inside its loader, so the
    // per-buffer `record_pax_region_bytes` / `record_get` fires HERE, once,
    // at batch time.
    let mut cold_bytes: Vec<Option<Vec<u8>>> = vec![None; cold_ranges.len()];
    if !cold_ranges.is_empty() {
        let batched = match fs.read_ranges_prefetch(path, cold_ranges.clone()).await {
            Ok(bufs) => bufs,
            Err(_) => {
                // Defensive fallback to the sequential baseline shape.
                let mut seq = Vec::with_capacity(cold_ranges.len());
                for range in &cold_ranges {
                    let b = fs
                        .read_range(path, range.start, range.end - range.start)
                        .await
                        .map_err(|err| {
                            anyhow::anyhow!("coarse-probe Region A fallback {path}: {err}")
                        })?;
                    seq.push(b);
                }
                seq
            }
        };
        for ((range, buf), slot) in cold_ranges.iter().zip(batched).zip(cold_bytes.iter_mut()) {
            crate::observability::io_trace::record_pax_region_bytes(buf.len() as u64, 0);
            record_get_physical(CacheTier::ProbeIndex, range.end - range.start, trace_on);
            *slot = Some(buf);
        }
        // C4: the queued footer rode this wave at index 0 — Meta tier, not a
        // Region-A payload (the region-bytes recorder above is per-range and
        // the footer's own tier record happens at extraction below).
        if footer_queued {
            let buf = cold_bytes[0]
                .take()
                .ok_or_else(|| anyhow::anyhow!("coarse-probe {path}: footer wave slot missing"))?;
            record_get_physical(CacheTier::InvariantMeta, buf.len() as u64, trace_on);
            footer_out = Some(buf);
        }
        // TD-RDSTRAT-12 §2: fold this wave's FS single-slot metrics into the
        // per-query io_trace additively (`fetch_add`/`fetch_max`) — a query
        // issuing several waves (probe tail + Region-B + Region-D) would
        // otherwise lose all but the last to the single-overwrite slot.
        crate::observability::io_trace::drain_and_forward_read_ranges_metrics();
    }

    // C4: the footer slot (always index 0 when present) is already consumed —
    // pop it so the remaining slots align 1:1 with the fetch list.
    if footer_queued {
        slots.remove(0);
    }
    // Consume pass: materialize each fetch's bytes IN ORDER, running the exact
    // same consumer the sequential path used (so tenant-keyed tracking, LRU
    // admission, io_trace survivor counters and error text are unchanged). A
    // Cold loader that finds its entry already populated (cross-query race)
    // simply leaves its pre-batched buffer unconsumed — that GET was ours
    // either way, and the cache wins.
    for (fetch, slot) in fetches.iter().zip(slots.iter()) {
        let len = fetch.end - fetch.start;
        // TD-RDSTRAT-13 C2: fetch buffers are shared `Arc<[u8]>` so the probed
        // runs can move into morsel workers without copying region bytes.
        let bytes: std::sync::Arc<[u8]> = match slot {
            // C4: the footer slot is extracted before this loop (Meta tier,
            // never a Region-A payload); a leftover here is an internal
            // invariant break — fail safe to the baseline loader, never panic.
            Slot::Footer(_) => {
                return Err(anyhow::anyhow!(
                    "coarse-probe Region A {path}: footer slot reached the payload consume pass"
                ));
            }
            Slot::Ready(b) => std::sync::Arc::from(b.clone()),
            Slot::SurvivorCached => {
                // peek_memory_exact said resident, but an LRU race could have
                // evicted before we reach the consume call; fail safe to a real
                // ranged read (the baseline shape), never panic.
                let sc = match survivor_cache {
                    Some(sc) => sc,
                    None => {
                        return Err(anyhow::anyhow!(
                            "coarse-probe Region A {path}: internal slot/cache mismatch"
                        ));
                    }
                };
                std::sync::Arc::from(
                    sc.get_or_fetch(CacheKind::Other, path, fetch.start, len, || async {
                        let b = fs
                            .read_range(path, fetch.start, len)
                            .await
                            .map_err(std::io::Error::other)?;
                        crate::observability::io_trace::record_pax_region_bytes(b.len() as u64, 0);
                        record_get_physical(CacheTier::ProbeIndex, len, trace_on);
                        Ok(b)
                    })
                    .await
                    .map_err(|err| anyhow::anyhow!("coarse-probe Region A {path}: {err}"))?
                    .to_vec(),
                )
            }
            Slot::Cold(i) => {
                let mut buf = cold_bytes[*i].take().ok_or_else(|| {
                    anyhow::anyhow!("coarse-probe Region A {path}: cold fetch {i} missing")
                })?;
                match survivor_cache {
                    Some(sc) => std::sync::Arc::from(
                        sc.get_or_fetch(CacheKind::Other, path, fetch.start, len, || async {
                            Ok::<_, proximadb_storage_filesystem_types::FilesystemError>(
                                std::mem::take(&mut buf),
                            )
                        })
                        .await
                        .map_err(|err| anyhow::anyhow!("coarse-probe Region A {path}: {err}"))?
                        .to_vec(),
                    ),
                    None => std::sync::Arc::from(buf),
                }
            }
        };
        fetched.push((fetch.clone(), bytes));
    }
    let mut runs: Vec<proximadb_block_format::ProbedRun> = Vec::with_capacity(selected.len());
    for (fetch, bytes) in &fetched {
        for cell in &fetch.selected {
            let rel = usize::try_from(cell.start.saturating_sub(fetch.start))
                .map_err(|_| anyhow::anyhow!("coarse-probe slice offset exceeds usize"))?;
            let len = usize::try_from(cell.end.saturating_sub(cell.start))
                .map_err(|_| anyhow::anyhow!("coarse-probe slice length exceeds usize"))?;
            let end = rel
                .checked_add(len)
                .ok_or_else(|| anyhow::anyhow!("coarse-probe slice end overflow"))?;
            if bytes.get(rel..end).is_none() {
                return Err(anyhow::anyhow!("coarse-probe selected slice outside fetch"));
            }
            runs.push(proximadb_block_format::ProbedRun {
                row_start: cell.row_start,
                bytes: bytes.clone(),
                byte_start: rel,
                byte_len: len,
            });
        }
    }
    let pool = pax_rabitq_pool_for_top_k(k, probed_rows as usize);
    // TD-FPRUNE-1 M3: rank the probed cells' codes RESTRICTED to allowed rows
    // (`row_allow`) so the survivor pool fills with predicate-matching candidates;
    // `None` is the unfiltered whole-cell rank (unchanged).
    // TD-RDSTRAT-13 C2: the rank runs morsel-parallel under the documented
    // PROXIMADB_SEARCH_MORSEL_DEGREE adaptive gate; degree ≤ 1 and small
    // regions take the untouched sequential path (byte-identical survivors).
    // TD-RDSTRAT-13 C4: a held cold footer overlaps the rank — the morsels
    // freed the async worker, so the GET hides under the compute instead of
    // serializing after it.
    let survivors = match footer_pending.take() {
        Some((off, len)) => {
            let footer_fut = async {
                let bytes = fs
                    .read_range(path, off, len)
                    .await
                    .map_err(|e| anyhow::anyhow!("coarse-probe footer {path}: {e}"))?;
                record_get_physical(CacheTier::InvariantMeta, bytes.len() as u64, trace_on);
                Ok::<_, anyhow::Error>(bytes)
            };
            let (rank_res, footer_res) = tokio::join!(
                rank_probed_rows_morsels(
                    std::sync::Arc::new(rq_header),
                    runs,
                    query,
                    metric,
                    pool.max(k),
                    row_allow.map(|allow| std::sync::Arc::new(allow.clone())),
                ),
                footer_fut
            );
            match footer_res {
                Ok(bytes) => footer_out = Some(bytes),
                // Footer failure must not discard a good rank — the cascade's
                // sequential footer arm re-reads it (one extra GET on a rare
                // error path).
                Err(e) => tracing::warn!("coarse-probe footer overlap read failed: {e}"),
            }
            rank_res?
        }
        None => {
            rank_probed_rows_morsels(
                std::sync::Arc::new(rq_header),
                runs,
                query,
                metric,
                pool.max(k),
                row_allow.map(|allow| std::sync::Arc::new(allow.clone())),
            )
            .await?
        }
    };

    Ok(Some(CoarseProbeResult {
        survivors,
        cells_total: k_c as u64,
        cells_probed: probed.len() as u64,
        probed_rows,
        fetch_rounds: fetched.len() as u64,
        a0_bytes,
        rabitq_header_bytes: hdr_bytes,
        footer_bytes: footer_out,
    }))
}

/// TD-SEARCH-2 S2: multi-core Stage-A rank. The whole-region RaBitQ scan is
/// the dominant cold-CPU term (~hundreds of ms at 500k rows/segment); split
/// the rows into `morsel_degree()` chunks ranked on `spawn_blocking` threads
/// (real cores — the tokio workers stay free for I/O), then merge the
/// per-chunk `(row, score)` lists by score. Exactly equivalent to a full
/// sequential rank (each chunk keeps its own top-`pool`; the global
/// top-`pool` is a subset of their union). Small regions and degree 1 keep
/// the sequential path — no task overhead where there is nothing to win.
const MORSEL_MIN_ROWS: usize = 65_536;

async fn rank_region_morsels(
    region: proximadb_block_format::RaBitQRegion,
    query: &[f32],
    metric: RankMetric,
    pool: usize,
) -> Vec<usize> {
    let degree = crate::storage::engines::sst::search::morsel_degree();
    let n_rows = region.n_rows();
    if degree <= 1 || n_rows < MORSEL_MIN_ROWS {
        return region.rank(query, metric, pool);
    }
    let region = std::sync::Arc::new(region);
    let chunk = n_rows.div_ceil(degree);
    let tasks: Vec<_> = (0..degree)
        .map(|i| {
            let region = region.clone();
            let query = query.to_vec();
            let rows = (i * chunk)..(((i + 1) * chunk).min(n_rows));
            tokio::task::spawn_blocking(move || {
                region.rank_range_scored(&query, metric, pool, rows)
            })
        })
        .collect();
    let mut merged: Vec<(usize, f32)> = Vec::with_capacity(pool * degree);
    for t in tasks {
        match t.await {
            Ok(part) => merged.extend(part),
            // A cancelled/panicked worker degrades to a partial pool — the
            // remaining chunks still rank; never fail the query for it.
            Err(e) => tracing::warn!("morsel rank worker failed (partial pool): {e}"),
        }
    }
    merged.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    merged.truncate(pool);
    merged.into_iter().map(|(i, _)| i).collect()
}

/// TD-RDSTRAT-13 C2: morsel-parallel COARSE-PROBE rank — the probe path's
/// counterpart to [`rank_region_morsels`]. At 370k-545k probed rows the
/// single-threaded rank is the dominant cold-query compute term (~0.9-1.2 s,
/// S3/MinIO campaign #1795); chunking it across `morsel_degree()` workers
/// moves the Compute term of `T ≈ Σ rounds×RTT + bytes/BW + Compute` without
/// touching GETs or bytes (billed-read parity, TD-RDSTRAT-13).
///
/// Determinism: each morsel keeps its local top-`pool` under the SEQUENTIAL
/// (score, ordinal) total order and the parent re-applies the identical
/// finalizer to their union — byte-identical survivors to
/// `rank_probed_rows_allowed` for ANY degree (the coalesced_rabitq exactness
/// argument applies because every chunk uses the same total order). Degree ≤ 1
/// and small regions route to the UNTOUCHED sequential function, so degree-1
/// byte-identity is structural, not tested-into-existence.
///
/// Failure policy (deliberately stricter than [`rank_region_morsels`], whose
/// partial-pool degrade is fine for an approximate fallback): a probe worker
/// panic degrades to a FULL sequential re-rank — a partial survivor pool would
/// silently change recall.
async fn rank_probed_rows_morsels(
    header: std::sync::Arc<proximadb_block_format::CoalescedRaBitQHeader>,
    runs: Vec<proximadb_block_format::ProbedRun>,
    query: &[f32],
    metric: RankMetric,
    pool: usize,
    allow: Option<std::sync::Arc<proximadb_block_format::row_allow::RowAllow>>,
) -> Result<Vec<usize>> {
    let started = std::time::Instant::now();
    let result = rank_probed_rows_morsels_inner(&header, runs, query, metric, pool, allow).await;
    let elapsed_us = started.elapsed().as_micros() as u64;
    crate::observability::io_trace::record_ivf_probe_rank_us(elapsed_us);
    crate::storage::engines::sst::metrics::record_ivf_probe_rank_us(elapsed_us);
    result
}

async fn rank_probed_rows_morsels_inner(
    header: &std::sync::Arc<proximadb_block_format::CoalescedRaBitQHeader>,
    runs: Vec<proximadb_block_format::ProbedRun>,
    query: &[f32],
    metric: RankMetric,
    pool: usize,
    allow: Option<std::sync::Arc<proximadb_block_format::row_allow::RowAllow>>,
) -> Result<Vec<usize>> {
    // Mirror the sequential validation so the parallel path fails identically.
    if header.dim == 0 {
        anyhow::bail!("coarse-probe rank: region dim is 0");
    }
    if query.len() != header.dim as usize {
        anyhow::bail!(
            "coarse-probe query dimension {} does not match region dimension {}",
            query.len(),
            header.dim
        );
    }
    let total_rows: usize = runs.iter().try_fold(0usize, |total, run| {
        total
            .checked_add(run.byte_len / proximadb_block_format::code_stride(header.dim))
            .ok_or_else(|| anyhow::anyhow!("coarse-probe row count overflow"))
    })?;
    if total_rows == 0 || pool == 0 {
        return Ok(Vec::new());
    }
    let degree = crate::storage::engines::sst::search::morsel_degree();
    if degree <= 1 || total_rows < MORSEL_MIN_ROWS {
        let borrowed: Vec<(usize, &[u8])> = runs
            .iter()
            .map(|r| {
                (
                    r.row_start,
                    r.bytes
                        .get(r.byte_start..r.byte_start + r.byte_len)
                        .unwrap_or(&[]),
                )
            })
            .collect();
        return proximadb_block_format::rank_probed_rows_allowed(
            header,
            &borrowed,
            query,
            metric,
            pool,
            allow.as_deref(),
        );
    }
    let prepared = proximadb_block_format::PreparedProbeRank::build(header, query, metric)?;
    let groups = proximadb_block_format::plan_rank_morsels(&runs, degree);
    let mut tasks = Vec::with_capacity(groups.len());
    for group in groups {
        let prepared = prepared.clone();
        let runs = runs[group.clone()].to_vec();
        let allow = allow.clone();
        tasks.push(tokio::task::spawn_blocking(move || {
            proximadb_block_format::rank_run_group_top(&prepared, &runs, allow.as_deref(), pool)
        }));
    }
    let mut merged: Vec<(usize, f32)> = Vec::new();
    for task in tasks {
        match task.await {
            Ok(Ok(part)) => merged.extend(part),
            // A worker returned a real scoring error — the sequential rank
            // over the same runs would hit the identical error, so propagating
            // is the deterministic outcome.
            Ok(Err(e)) => return Err(e),
            // Worker panic: the parallel result would be a partial pool that
            // silently changes survivors — degrade to the FULL sequential
            // oracle instead (exact, deterministic, rare).
            Err(join_err) => {
                tracing::warn!(
                    "coarse-probe morsel worker failed; sequential fallback: {join_err}"
                );
                let borrowed: Vec<(usize, &[u8])> = runs
                    .iter()
                    .map(|r| {
                        (
                            r.row_start,
                            r.bytes
                                .get(r.byte_start..r.byte_start + r.byte_len)
                                .unwrap_or(&[]),
                        )
                    })
                    .collect();
                return proximadb_block_format::rank_probed_rows_allowed(
                    header,
                    &borrowed,
                    query,
                    metric,
                    pool,
                    allow.as_deref(),
                );
            }
        }
    }
    let keep = pool.min(merged.len());
    if keep < merged.len() {
        merged.select_nth_unstable_by(keep, proximadb_block_format::probed_score_compare);
        merged.truncate(keep);
    }
    merged.sort_unstable_by(proximadb_block_format::probed_score_compare);
    Ok(merged.into_iter().map(|(row, _)| row).collect())
}

pub async fn rabitq_search_segment_coalesced(
    fs: &dyn proximadb_storage_filesystem_types::FileSystem,
    path: &str,
    query: &[f32],
    k: usize,
    metric: RankMetric,
    cache: Option<&SegmentInvariantsCache>,
    survivor_cache: Option<&SurvivorRangeCache>,
) -> Result<Option<Vec<CascadeHit>>> {
    rabitq_search_segment_coalesced_allowed(
        fs,
        path,
        query,
        k,
        metric,
        cache,
        survivor_cache,
        None,
        None,
    )
    .await
}

/// ADR-089 / TD-FPRUNE-1 P1: [`rabitq_search_segment_coalesced`] restricted to
/// an optional predicate row allow-set (global row ordinals). When `Some`,
/// Stage-1 ranks ONLY allowed rows — the survivor pool holds
/// predicate-matching candidates, so filtered recall equals the RaBitQ
/// approximation over the matching rows (same quality bar as the unfiltered
/// whole-region scan). The geometric coarse probe is bypassed under a filter
/// for P1 — probe∩filter composition (nprobe policy under selectivity) is the
/// TD-FPRUNE-1 P3 concern.
#[allow(clippy::too_many_arguments)]
pub async fn rabitq_search_segment_coalesced_allowed(
    fs: &dyn proximadb_storage_filesystem_types::FileSystem,
    path: &str,
    query: &[f32],
    k: usize,
    metric: RankMetric,
    cache: Option<&SegmentInvariantsCache>,
    survivor_cache: Option<&SurvivorRangeCache>,
    row_allow: Option<&proximadb_block_format::RowAllow>,
    // TD-RDSTRAT-13 C3: size reported by the segment listing. `Some` skips the
    // per-segment HEAD (a billed metadata op); `None` stats exactly as before.
    known_size: Option<u64>,
) -> Result<Option<Vec<CascadeHit>>> {
    use proximadb_block_format::{PaxBlockReader, RaBitQRegion, coalesced_sq8, col_id};
    use proximadb_storage_common::segment_layout::{
        SEG_HEADER_PREFIX_LEN, SegmentFooterIndex, SegmentHeaderPrefix,
    };

    // ADR-065 co-design diagnostic: per-GET size trace. When PROXIMADB_TRACE_GETS
    // is set, every read_range in this path records its on-disk byte length; the
    // eval drains it to print the distribution (GET count = the same-region
    // billing metric; GET sizes = the latency metric). Off by default (zero cost).
    let trace_on = std::env::var_os("PROXIMADB_TRACE_GETS").is_some()
        || std::env::var_os("PROXIMADB_TRACE_PAX_STAGES").is_some();

    // PR2: check the per-segment invariants cache. On hit, skip the 3
    // read_range calls (header + region + footer) → 3 GETs → 0 (hot path).
    let invariant_lookup = match cache {
        Some(cache) => cache.get_or_promote(path).await,
        None => InvariantLookup::Miss,
    };
    let cached = invariant_lookup.value();
    // TD-METRICS-1: this lookup is the cache's only read boundary — count
    // hit/miss here (a hit means 3 GETs avoided) and sample resident bytes.
    if cache.is_some() {
        match invariant_lookup {
            InvariantLookup::L1(_) => {
                crate::metrics::operational_metrics::SEGMENT_INVARIANTS_CACHE_HITS_TOTAL.inc();
            }
            InvariantLookup::L2(_) => {}
            InvariantLookup::Miss => {
                crate::metrics::operational_metrics::SEGMENT_INVARIANTS_CACHE_MISSES_TOTAL.inc();
            }
        }
        if let Some(c) = cache {
            crate::metrics::operational_metrics::SEGMENT_INVARIANTS_CACHE_BYTES
                .set(c.bytes_used() as i64);
            crate::metrics::operational_metrics::sync_local_disk_stats("invariants", c.l2_stats());
        }
    }

    // 1. Header-prefix → region/footer extents. From cache (hot) or read (cold).
    let header_bytes: Vec<u8> = if let Some(inv) = cached.as_ref() {
        inv.header_bytes.clone()
    } else {
        // TD-RDSTRAT-13 C3: the listing already reported the object size —
        // trust it as a hint and skip the HEAD (a billed metadata op per
        // segment per query). Correctness is unaffected: the header bytes are
        // still magic-validated by `SegmentHeaderPrefix::parse`, and the
        // footer extents are bounds-checked against this size below (a
        // stale/foreign size fails closed to the generic scan path).
        let size = match known_size {
            Some(s) => s,
            None => {
                crate::observability::io_trace::record_metadata_op();
                crate::storage::engines::sst::metrics::record_metadata_op();
                fs.metadata(path)
                    .await
                    .map_err(|e| anyhow::anyhow!("coalesced scan stat {path}: {e}"))?
                    .size
            }
        };
        if size < (SEG_HEADER_PREFIX_LEN as u64 + SEGMENT_MAGIC.len() as u64) {
            return Ok(None);
        }
        // TD-RDSTRAT-8: fetch the v3 prefix length unconditionally (clamped to
        // file size) — the 16 extra bytes ride the same GET and cover both the
        // v1 (56 B) and v3 (72 B) forms; `parse` branches on the version byte.
        // TD-PAXRG-1: the floor is now v4 (88 B), covering the row-group
        // layout's Region C extent too.
        let floor = coalesced_header_prefetch_floor() as u64;
        let read_len = prefix_prefetch_bytes()
            .unwrap_or(floor)
            .max(floor)
            .min(size);
        let bytes = fs
            .read_range(path, 0, read_len)
            .await
            .map_err(|e| anyhow::anyhow!("coalesced scan header {path}: {e}"))?;
        record_get_physical(CacheTier::SearchControl, bytes.len() as u64, trace_on);
        bytes
    };
    // A v3 (two-level) segment parses here too and falls through to the same
    // single-level whole-region scan: Regions A/B are byte-identical in format
    // (`rabitq_off/sq8_off` skip past A0), and a full scan over coarse-ordered
    // rows is order-agnostic — correct results, v2 GET budget. The A0
    // coarse-probe branch (nprobe-scoped sub-reads) is PR-B of TD-RDSTRAT-8.
    let header = match SegmentHeaderPrefix::parse(&header_bytes) {
        Ok(h) => h,
        Err(_) => return Ok(None), // not coalesced → don't cache
    };
    // TD-RDSTRAT-13 C3 stale-size guard: when the size came from the listing
    // (no HEAD was paid), validate it against the header's footer extents
    // before anything trusts it. Segments are immutable — a size that cannot
    // even contain the footer means the listing described a different object
    // generation, so fail over to the generic scan path (which re-stats) rather
    // than serve extents clamped against a foreign length. Header bytes
    // themselves are already magic-validated above.
    if let Some(listed) = known_size {
        let footer_end = header.footer_off.saturating_add(header.footer_len);
        if footer_end > listed {
            tracing::warn!(
                path,
                listed,
                footer_off = header.footer_off,
                footer_len = header.footer_len,
                "stale listed size for coalesced segment; failing over to generic scan"
            );
            return Ok(None);
        }
    }
    //    probe miss falls through, fail-safe, to the whole-region path.
    // TD-PAXRG-1 collapse: A0 presence is declared by its extent, not a
    // version byte. TD-FPRUNE-1 M3: the probe is armed UNDER A FILTER too —
    // `coarse_probe_survivors` becomes a filtered ANN (adaptive nprobe by
    // matching count + allowed rank), restricting the whole fetch to relevant
    // cells instead of the whole-region scan the disarmed path traded for. A
    // probe miss still falls through to the whole-region allowed rank below
    // (fail-safe).
    let probe_armed = header.a0_len > 0 && coarse_probe_enabled();
    // TD-RDSTRAT-13 C4: classify the footer BEFORE the probe. The footer
    // extents are header fields (ADR-062 guarantees the RaBitQ scan never
    // depends on the footer), so a cold footer can ride the probe's ranged
    // wave — one round instead of a serialized GET after the ~1 s rank.
    // Check order matches the sequential step-3 arms (cache hit, prefix
    // slice) exactly, so a warm footer is never re-batched.
    let footer_plan = if let Some(inv) = cached.as_ref() {
        FooterPlan::Ready(inv.footer_bytes.clone())
    } else if let Some(bytes) =
        prefetched_slice(&header_bytes, header.footer_off, header.footer_len)
    {
        FooterPlan::Ready(bytes.to_vec())
    } else {
        FooterPlan::Pending {
            off: header.footer_off,
            len: header.footer_len,
        }
    };
    let probe = if probe_armed {
        coarse_probe_survivors(
            fs,
            path,
            &header,
            query,
            metric,
            k,
            trace_on,
            cached.as_deref(),
            &header_bytes,
            cache,
            survivor_cache,
            row_allow,
            footer_plan,
        )
        .await
        .ok()
        .flatten()
    } else {
        None
    };
    let (survivors, region_bytes): (Vec<usize>, Option<Arc<[u8]>>) = if let Some(r) = &probe {
        if trace_on {
            record_probe_trace(r.cells_total, r.cells_probed, r.probed_rows, r.fetch_rounds);
        }
        // TD-RDSTRAT-8: durable per-query probe outcome → warehouse `VectorAnnPayload`
        // (always-on when an io_trace scope is active; no-ops otherwise). The dominant
        // cloud cost term is GET round-trips — recording the probe's cut per query is
        // what makes nprobe/spill tuning evidence-led ("trace before you tune").
        record_ivf_probe_durable(
            r.cells_total,
            r.cells_probed,
            r.probed_rows,
            r.fetch_rounds,
            false,
        );
        (r.survivors.clone(), None)
    } else {
        // Armed probe that missed → whole-Region-A fallback (the GET budget the probe
        // exists to avoid). Record the miss so it is observable; a non-v3 / probe-off
        // read records nothing (no IVF occurred).
        if probe_armed {
            record_ivf_probe_durable(0, 0, 0, 0, true);
        }
        let region_bytes: Arc<[u8]> =
            if let Some(bytes) = cached.as_ref().and_then(|inv| inv.region_bytes.clone()) {
                bytes
            } else {
                let bytes = fs
                    .read_range(path, header.rabitq_off, header.rabitq_len)
                    .await
                    .map_err(|e| anyhow::anyhow!("coalesced scan region {path}: {e}"))?;
                crate::observability::io_trace::record_pax_region_bytes(bytes.len() as u64, 0);
                // Trace the whole-region cost so the coarse probe's Region-A saving is
                // observable (co-design: trace before you tune). Diagnostic-only.
                record_get_physical(CacheTier::InvariantIndex, header.rabitq_len, trace_on);
                Arc::from(bytes)
            };
        let region = RaBitQRegion::from_bytes(&region_bytes)?;
        let survivors = match row_allow {
            // ADR-089 P1: rank only predicate-matching rows. The pool scales
            // with the ALLOWED row count (not the segment total), so selective
            // filters don't over-fetch SQ8 survivor ranges. Sequential rank is
            // deliberate: the allowed count is typically well below the morsel
            // threshold; morsel-parallel allowed rank is a P3 follow-up.
            Some(allow) => {
                let pool = pax_rabitq_pool_for_top_k(k, allow.len().max(1));
                region.rank_allowed(query, metric, pool.max(k), allow)
            }
            None => {
                // ADR-062 PR2: adaptive survivor pool — scale M with the
                // segment's rows.
                let pool = pax_rabitq_pool_for_top_k(k, region.n_rows());
                rank_region_morsels(region, query, metric, pool.max(k)).await
            }
        };
        (survivors, Some(region_bytes))
    };
    if survivors.is_empty() {
        return Ok(Some(Vec::new()));
    }

    // 3. Footer-index → block table. Cache hit (hot), else the probe wave
    // fetched it alongside the Region-A ranges (C4), else the sequential
    // baseline GET (probe disarmed/missed — identical shape to before C4).
    let footer_bytes: Vec<u8> = if let Some(inv) = cached.as_ref() {
        inv.footer_bytes.clone()
    } else if let Some(bytes) = probe.as_ref().and_then(|r| r.footer_bytes.clone()) {
        bytes
    } else {
        let bytes = fs
            .read_range(path, header.footer_off, header.footer_len)
            .await
            .map_err(|e| anyhow::anyhow!("coalesced scan footer {path}: {e}"))?;
        record_get_physical(CacheTier::InvariantMeta, bytes.len() as u64, trace_on);
        bytes
    };
    let footer = SegmentFooterIndex::parse(&footer_bytes)?;

    // Cache tiny ANN control metadata independently from the whole Region A.
    // Probe queries therefore warm header/A0/RaBitQ params/footer without
    // admitting a 24+ MiB invariant body they intentionally did not read.
    let split_probe_meta = split_probe_metadata_cache_enabled();
    let probe_a0 = if split_probe_meta {
        probe.as_ref().map(|result| result.a0_bytes.clone())
    } else {
        None
    };
    let probe_rabitq_header = if split_probe_meta {
        probe
            .as_ref()
            .map(|result| result.rabitq_header_bytes.clone())
    } else {
        None
    };
    let cached_region = cached.as_ref().and_then(|entry| entry.region_bytes.clone());
    let cached_a0 = cached.as_ref().and_then(|entry| entry.a0_bytes.clone());
    let cached_rabitq_header = cached
        .as_ref()
        .and_then(|entry| entry.rabitq_header_bytes.clone());
    let cache_changed = (probe.is_none() && cached.is_none())
        || (cached_region.is_none() && region_bytes.is_some())
        || (cached_a0.is_none() && probe_a0.is_some())
        || (cached_rabitq_header.is_none() && probe_rabitq_header.is_some());
    if cache_changed && let Some(c) = cache {
        c.put_with_l2(
            path.to_string(),
            Arc::new(SegmentInvariants {
                header_bytes,
                region_bytes: region_bytes.or(cached_region),
                footer_bytes,
                a0_bytes: probe_a0.or(cached_a0),
                rabitq_header_bytes: probe_rabitq_header.or(cached_rabitq_header),
            }),
        )
        .await;
    }

    // 4. ADR-065 Region B: rerank survivors via the coalesced SQ8 region (pure,
    //    dense — no bystander props/fp32). The dequant key (min + scale) is
    //    mirrored in the footer (already read), so there is NO separate 24 B
    //    Region-B-header GET — reconstruct the params + codes_base from the footer.
    // TD-RDSTRAT-13 PR-B: this section (fetch + rerank) is one of the four
    // attributed tail segments; the wall is recorded at the `scored` boundary.
    let region_b_started = std::time::Instant::now();
    let dim = footer.embed_dim as usize;
    let sq8_params = coalesced_sq8::params_from_min_scale(footer.sq8_min, footer.sq8_scale);
    let query_norm_squared = if metric == RankMetric::Cosine {
        query.iter().map(|value| value * value).sum()
    } else {
        0.0
    };
    let codes_base = header.sq8_off + coalesced_sq8::codes_offset(footer.row_count as usize) as u64;
    // Coalesce policy IOP-aligned to the backend (ADR-065 cache-co-design).
    // The Azure 4 MiB value is a conservative planner default, not a billing
    // quantum or proven SDK split boundary. TD-SEARCH-3 compares the issued
    // range count with Azurite/Azure wire requests before changing it.
    let iop_target =
        proximadb_storage_common::iops_budget::IopsBudget::for_path(path).target_block_bytes();
    let defaults =
        crate::storage::engines::sst::readers::sst_query_engine::ObjectRangeCoalescePolicy {
            max_gap_bytes: (iop_target / 4).max(64 * 1024),
            max_range_bytes: iop_target,
        };
    let mut sorted_survivors = survivors.clone();
    sorted_survivors.sort_unstable();
    sorted_survivors.dedup();
    let policy = resolve_adaptive_range_policy(
        path,
        "region_b",
        "PROXIMADB_PAX_COALESCE_GAP",
        "PROXIMADB_PAX_COALESCE_RANGE",
        defaults,
        |candidate| {
            let estimate = estimate_coalesced_byte_ranges(
                sorted_survivors.iter().map(|&row| {
                    let start = codes_base + (row as u64) * dim as u64;
                    (start, start + dim as u64)
                }),
                candidate,
            );
            estimate_with_whole_region_fallback(estimate, header.sq8_len)
        },
    );
    let sq8_fetches = plan_coalesced_row_ranges(&sorted_survivors, dim, codes_base, &policy);
    let mut scored: Vec<(usize, f32)> = Vec::with_capacity(survivors.len());
    let ranges_bytes: u64 = sq8_fetches.iter().map(|f| f.end - f.start).sum();
    if ranges_bytes >= header.sq8_len {
        // Scattered survivors (sign-bit / no-IVF order): the coalesced ranges would
        // over-read >= the whole Region B — fetch it in one GET instead (fewer GETs,
        // no more bytes). decode_row extracts only the survivors. (Tight survivor
        // ranges — the full ~5x bytes win — need IVF/Hilbert locality: follow-up.)
        // TD-CACHE-6: the whole-Region-B fallback is ONE well-known range —
        // cache it through the seam so hot repeats stop re-paying the
        // largest GET on the path (admission/floors decide residency).
        //
        // TD-RDSTRAT-12 §3 round 4 (STRICTLY OPT-IN): under a high-RTT backend
        // that single mega-GET is the dominant cold-latency term
        // (1×(RTT + region/BW) ≈ seconds), so when BOTH gates are armed —
        // `PROXIMADB_READ_RANGES_INFLIGHT > 1` and
        // `PROXIMADB_READ_RANGES_WAVE_SPLIT_MB` ≥ region size — split it into
        // `IopsBudget`-sized chunks and wave them. This DELIBERATELY trades
        // +N−1 billed GETs for ceil(N/width)×RTT latency (TD-RDSTRAT-8
        // GET-economy vs RTT-economy; the default-OFF gate keeps the
        // GET-economical baseline untouched). Accepted cache trade: per-chunk
        // loaders populate exact-key L1/L2 entries — the search-path parent L2
        // entry never forms; warm serving flows through per-chunk exact tiers
        // (equal total bytes, finer LRU granularity, demanded-chunk reads),
        // and write-time parent seeding still serves via peek step 3.
        let wave_split_active = matches!(wave_split_threshold_mb(), Some(threshold_mb) if
            header.sq8_len >= threshold_mb.saturating_mul(1024 * 1024));
        if wave_split_active {
            let chunk = iop_target;
            let split_ranges = plan_wave_split_ranges(header.sq8_off, header.sq8_len, chunk);
            let mut chunk_bytes: Vec<Option<Vec<u8>>> = vec![None; split_ranges.len()];
            let batched = match fs.read_ranges_prefetch(path, split_ranges.clone()).await {
                Ok(bufs) => bufs,
                Err(_) => {
                    // Defensive fallback to the sequential baseline shape.
                    let mut seq = Vec::with_capacity(split_ranges.len());
                    for range in &split_ranges {
                        let b = fs
                            .read_range(path, range.start, range.end - range.start)
                            .await
                            .map_err(|err| {
                                anyhow::anyhow!(
                                    "coalesced scan Region B split fallback {path}: {err}"
                                )
                            })?;
                        seq.push(b);
                    }
                    seq
                }
            };
            for ((range, buf), slot) in split_ranges.iter().zip(batched).zip(chunk_bytes.iter_mut())
            {
                crate::observability::io_trace::record_pax_region_bytes(0, buf.len() as u64);
                record_get_physical(
                    CacheTier::SurvivorPayload,
                    range.end - range.start,
                    trace_on,
                );
                *slot = Some(buf);
            }
            crate::observability::io_trace::drain_and_forward_read_ranges_metrics();
            tracing::debug!(
                path,
                region_bytes = header.sq8_len,
                chunks = split_ranges.len(),
                "Region-B whole-region fallback wave-split"
            );
            // Feed each chunk through the SAME parent seam (W1) so per-chunk
            // exact L1/L2 entries, hot-key tracking and counters behave
            // exactly like the clustered path's ranges.
            let mut region_buf = vec![0u8; header.sq8_len as usize];
            for (range, slot) in split_ranges.iter().zip(chunk_bytes.iter_mut()) {
                let chunk_len = (range.end - range.start) as usize;
                let mut pre = slot.take().ok_or_else(|| {
                    anyhow::anyhow!("coalesced scan Region B split {path}: chunk missing")
                })?;
                let chunk_off = range.start;
                let rel = (chunk_off - header.sq8_off) as usize;
                if let Some(sc) = survivor_cache {
                    let fed: Arc<[u8]> = sc
                        .get_or_fetch_in_parent(
                            CacheKind::QuantizedCodes,
                            path,
                            chunk_off,
                            chunk_len as u64,
                            header.sq8_off,
                            header.sq8_len,
                            || async {
                                Ok::<_, proximadb_storage_filesystem_types::FilesystemError>(
                                    std::mem::take(&mut pre),
                                )
                            },
                        )
                        .await
                        .map_err(|e| {
                            anyhow::anyhow!("coalesced scan Region B split feed {path}: {e}")
                        })?;
                    region_buf[rel..rel + chunk_len].copy_from_slice(&fed);
                } else {
                    region_buf[rel..rel + chunk_len].copy_from_slice(&pre);
                }
            }
            let region = coalesced_sq8::Sq8Region::from_bytes(&region_buf)?;
            for &g in &survivors {
                if let Some(codes) = region.row_codes(g)
                    && let Some(score) =
                        rerank_sq8_distance(metric, query, codes, &sq8_params, query_norm_squared)
                {
                    scored.push((g, score));
                }
            }
        } else {
            let region_arc: Arc<[u8]> = if let Some(sc) = survivor_cache {
                sc.get_or_fetch_in_parent(
                    CacheKind::QuantizedCodes,
                    path,
                    header.sq8_off,
                    header.sq8_len,
                    header.sq8_off,
                    header.sq8_len,
                    || async {
                        let b = fs
                            .read_range(path, header.sq8_off, header.sq8_len)
                            .await
                            .map_err(std::io::Error::other)?;
                        crate::observability::io_trace::record_pax_region_bytes(0, b.len() as u64);
                        record_get_physical(CacheTier::SurvivorPayload, header.sq8_len, trace_on);
                        Ok(b)
                    },
                )
                .await
                .map_err(|e| anyhow::anyhow!("coalesced scan Region B fetch {path}: {e}"))?
            } else {
                let b = fs
                    .read_range(path, header.sq8_off, header.sq8_len)
                    .await
                    .map_err(|e| anyhow::anyhow!("coalesced scan Region B fetch {path}: {e}"))?;
                crate::observability::io_trace::record_pax_region_bytes(0, b.len() as u64);
                record_get_physical(CacheTier::SurvivorPayload, header.sq8_len, trace_on);
                Arc::from(b)
            };
            let region = coalesced_sq8::Sq8Region::from_bytes(&region_arc)?;
            for &g in &survivors {
                if let Some(codes) = region.row_codes(g)
                    && let Some(score) =
                        rerank_sq8_distance(metric, query, codes, &sq8_params, query_norm_squared)
                {
                    scored.push((g, score));
                }
            }
        }
    } else {
        // Clustered survivors (IVF/Hilbert locality): fetch the few tight coalesced
        // ranges — pure dense SQ8, the minimal-bytes path.
        let dim64 = dim as u64;
        // TD-RDSTRAT-12 §3 round 4: peek·batch·feed (mirror of the probe site).
        // Classify: survivor-RAM/persistent residency (exact L1, parent L1
        // slice, parent L2, exact L2 — all side-effect-free) splits each
        // planned range into SiteServed (the seam will serve it; loader can
        // only run on an LRU race) vs Cold (the loader WILL run → batch it).
        enum RegionBSlot {
            SiteServed,
            Cold(usize),
        }
        let mut slots: Vec<RegionBSlot> = Vec::with_capacity(sq8_fetches.len());
        let mut cold_ranges: Vec<std::ops::Range<u64>> = Vec::new();
        for fetch in &sq8_fetches {
            match survivor_cache {
                Some(sc)
                    if sc
                        .peek_parent_residency(
                            CacheKind::QuantizedCodes,
                            path,
                            fetch.start,
                            fetch.end - fetch.start,
                            header.sq8_off,
                            header.sq8_len,
                        )
                        .await
                        .is_some() =>
                {
                    slots.push(RegionBSlot::SiteServed);
                }
                _ => {
                    slots.push(RegionBSlot::Cold(cold_ranges.len()));
                    cold_ranges.push(fetch.start..fetch.end);
                }
            }
        }
        // One bounded-concurrent wave for every true-cold range (INFLIGHT cap;
        // unset/≤1 ⇒ the trait runs it sequentially — byte-for-byte baseline).
        // D4 memory bound: wave transient = min(INFLIGHT, N) × max_range_bytes
        // ≤ 32 × 4 MiB = 128 MiB; one bounded read_ranges_prefetch, never an
        // unbounded join_all.
        //
        // Metrics parity: every buffer corresponds to exactly one physical GET
        // the sequential loop would have issued inside its loader, so the
        // per-buffer `record_pax_region_bytes` / `record_get` fires HERE, once,
        // at batch time (Cold loaders below stay silent).
        let mut cold_bytes: Vec<Option<Vec<u8>>> = vec![None; cold_ranges.len()];
        if !cold_ranges.is_empty() {
            let batched = match fs.read_ranges_prefetch(path, cold_ranges.clone()).await {
                Ok(bufs) => bufs,
                Err(_) => {
                    // Defensive fallback to the sequential baseline shape.
                    let mut seq = Vec::with_capacity(cold_ranges.len());
                    for range in &cold_ranges {
                        let b = fs
                            .read_range(path, range.start, range.end - range.start)
                            .await
                            .map_err(|err| {
                                anyhow::anyhow!(
                                    "coalesced scan SQ8 survivor fallback {path}: {err}"
                                )
                            })?;
                        seq.push(b);
                    }
                    seq
                }
            };
            for ((range, buf), slot) in cold_ranges.iter().zip(batched).zip(cold_bytes.iter_mut()) {
                crate::observability::io_trace::record_pax_region_bytes(0, buf.len() as u64);
                record_get_physical(
                    CacheTier::SurvivorPayload,
                    range.end - range.start,
                    trace_on,
                );
                *slot = Some(buf);
            }
            crate::observability::io_trace::drain_and_forward_read_ranges_metrics();
        }
        // Consume in plan order through the ORIGINAL seam (counters, hot-key
        // tracking, LRU admission unchanged); then the verbatim per-fetch
        // rerank so `scored` insertion order — and the stable top-k sort's
        // tie-breaking — stays identical to the sequential baseline.
        for (fetch_i, fetch) in sq8_fetches.iter().enumerate() {
            let start = fetch.start;
            let range_len = fetch.end - fetch.start;
            let pre = match &slots[fetch_i] {
                RegionBSlot::Cold(i) => Some(cold_bytes[*i].take().ok_or_else(|| {
                    anyhow::anyhow!(
                        "coalesced scan SQ8 survivor fetch {path}: cold fetch {i} missing"
                    )
                })?),
                RegionBSlot::SiteServed => None,
            };
            // ADR-065 Q3: survivor-range cache. On a hit the loader never runs,
            // so the billed GET fires only on a true miss — bytes-not-billed
            // for free, via the existing backend seam.
            let buf: Arc<[u8]> = if let Some(sc) = survivor_cache {
                match pre {
                    Some(mut pre) => sc
                        .get_or_fetch_in_parent(
                            CacheKind::QuantizedCodes,
                            path,
                            start,
                            range_len,
                            header.sq8_off,
                            header.sq8_len,
                            || async {
                                Ok::<_, proximadb_storage_filesystem_types::FilesystemError>(
                                    std::mem::take(&mut pre),
                                )
                            },
                        )
                        .await
                        .map_err(|e| {
                            anyhow::anyhow!("coalesced scan SQ8 survivor fetch {path}: {e}")
                        })?,
                    None => {
                        // SiteServed (or a peek/consume race): the original
                        // baseline loader — records fire iff it truly runs.
                        sc.get_or_fetch_in_parent(
                            CacheKind::QuantizedCodes,
                            path,
                            start,
                            range_len,
                            header.sq8_off,
                            header.sq8_len,
                            || async move {
                                let b = fs.read_range(path, start, range_len).await?;
                                crate::observability::io_trace::record_pax_region_bytes(
                                    0,
                                    b.len() as u64,
                                );
                                record_get_physical(
                                    CacheTier::SurvivorPayload,
                                    range_len,
                                    trace_on,
                                );
                                Ok(b)
                            },
                        )
                        .await
                        .map_err(|e| {
                            anyhow::anyhow!("coalesced scan SQ8 survivor fetch {path}: {e}")
                        })?
                    }
                }
            } else {
                match pre {
                    Some(b) => Arc::from(b),
                    None => {
                        // survivor_cache is None ⇒ every slot is Cold ⇒ pre is
                        // Some; this arm is unreachable-defense only.
                        let b = fs.read_range(path, start, range_len).await.map_err(|e| {
                            anyhow::anyhow!("coalesced scan SQ8 survivor fetch {path}: {e}")
                        })?;
                        crate::observability::io_trace::record_pax_region_bytes(0, b.len() as u64);
                        record_get_physical(CacheTier::SurvivorPayload, range_len, trace_on);
                        Arc::from(b)
                    }
                }
            };
            for &g in &fetch.rows {
                let rel = (codes_base + (g as u64) * dim64).saturating_sub(fetch.start) as usize;
                if rel + dim > buf.len() {
                    continue;
                }
                if let Some(score) = rerank_sq8_distance(
                    metric,
                    query,
                    &buf[rel..rel + dim],
                    &sq8_params,
                    query_norm_squared,
                ) {
                    scored.push((g, score));
                }
            }
        }
    }
    if scored.is_empty() {
        crate::observability::io_trace::record_region_b_rerank_us(
            region_b_started.elapsed().as_micros() as u64,
        );
        crate::storage::engines::sst::metrics::record_region_b_rerank_us(
            region_b_started.elapsed().as_micros() as u64,
        );
        return Ok(Some(Vec::new()));
    }
    {
        let us = region_b_started.elapsed().as_micros() as u64;
        crate::observability::io_trace::record_region_b_rerank_us(us);
        crate::storage::engines::sst::metrics::record_region_b_rerank_us(us);
    }
    // Global top-k survivor rows (nearest-first; lower score = nearer).
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_k_rows: Vec<usize> = scored.iter().take(k).map(|(g, _)| *g).collect();

    // 5. ADR-065 Region D: fetch ONLY the top-k OIDs from the row blocks (≤k
    //    coalesced GETs — vs PR2's M-survivor block fetches). Map top-k rows →
    //    blocks via cumulative row counts.
    // TD-RDSTRAT-13 PR-B: fetch+decode wall attributed here (both arms).
    let region_d_started = std::time::Instant::now();
    let mut block_start: Vec<u64> = Vec::with_capacity(footer.blocks.len());
    let mut acc = 0u64;
    for b in &footer.blocks {
        block_start.push(acc);
        acc += b.row_count as u64;
    }
    let mut block_rows: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for &g in &top_k_rows {
        if block_start.is_empty() {
            break;
        }
        let bi = match block_start.binary_search(&(g as u64)) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        let local = g as u64 - block_start[bi];
        block_rows.entry(bi).or_default().push(local as usize);
    }
    let topk_blocks: Vec<usize> = block_rows.keys().copied().collect();
    let mut oid_of: std::collections::HashMap<usize, String> = std::collections::HashMap::new();
    // TD-PAXRG-1 Phase D (v4): the footer's per-RG stats payload addresses each
    // RG's OID chunk, so the top-k OIDs are fetched as CHUNKS — the same GET
    // count as whole-RG fetches (request economy parity) at a fraction of the
    // bytes (an RG is up to the multi-MiB granule; the OID chunk is ~tens of
    // bytes per row). Non-v4 segments keep the TD-RDSTRAT-12 §3 r4
    // peek·batch·feed whole-block tail, verbatim.
    // TD-PAXRG-1 collapse: the RG-stats payload IS the v4 marker — non-RG
    // segments carry no stats payloads, so presence alone gates the chunk arm
    // (single layout version — no version check remains).
    let v4_chunk_fetch = topk_blocks.iter().all(|&bi| {
        footer
            .block_stats
            .get(bi)
            .is_some_and(|s| s.as_ref().is_some_and(|stats| stats.oid_chunk_len > 0))
    });
    if v4_chunk_fetch {
        // TD-PAXRG-1 Phase D + TD-RDSTRAT-13 PR-B 2b: per-RG OID-chunk fetch,
        // WAVED — the chunks were previously fetched strictly sequentially
        // (up to ~top-k serial GETs; on object stores that is ~top-k × RTT).
        // Same peek·batch·feed discipline as the legacy whole-block tail:
        // classify via the side-effect-free peek, ONE bounded wave for the
        // true-cold chunks (D4 bound: min(INFLIGHT,N) × chunk_len ≤ small —
        // chunks are ~tens of bytes per row), then feed pre-fetched bytes
        // through the SAME get_or_fetch seam so cache semantics, LRU and
        // counters stay byte-identical; serial fallback on wave Err.
        struct ChunkDesc {
            start: u64,
            range_len: u64,
            row_count: usize,
            encoding_id: u8,
            is_lz4: bool,
        }
        let mut chunk_descs: Vec<ChunkDesc> = Vec::with_capacity(block_rows.len());
        for &bi in block_rows.keys() {
            let b = footer.blocks.get(bi).ok_or_else(|| {
                anyhow::anyhow!("coalesced scan OID chunk block {bi} is missing from footer")
            })?;
            let stats = footer
                .block_stats
                .get(bi)
                .and_then(Option::as_ref)
                .ok_or_else(|| {
                    anyhow::anyhow!("coalesced scan OID chunk stats missing for block {bi}")
                })?;
            let relative_start = u64::from(stats.oid_chunk_rel_off);
            let range_len = stats.oid_chunk_len as u64;
            let relative_end = relative_start.checked_add(range_len).ok_or_else(|| {
                anyhow::anyhow!("coalesced scan OID chunk extent overflows for block {bi}")
            })?;
            if relative_end > u64::from(b.size) {
                anyhow::bail!(
                    "coalesced scan OID chunk [{relative_start},{relative_end}) is outside row group {bi} size {}",
                    b.size
                );
            }
            let start = b.offset.checked_add(relative_start).ok_or_else(|| {
                anyhow::anyhow!("coalesced scan OID chunk absolute offset overflows for block {bi}")
            })?;
            chunk_descs.push(ChunkDesc {
                start,
                range_len,
                row_count: b.row_count as usize,
                encoding_id: stats.oid_encoding_id,
                is_lz4: stats.oid_is_lz4,
            });
        }

        enum ChunkSlot {
            SiteServed,
            Cold(usize),
        }
        let mut chunk_slots: Vec<ChunkSlot> = Vec::with_capacity(chunk_descs.len());
        let mut chunk_cold_ranges: Vec<std::ops::Range<u64>> = Vec::new();
        for (desc_i, (_bi, _locals)) in block_rows.iter().enumerate() {
            let desc = &chunk_descs[desc_i];
            match survivor_cache {
                // TD-IOBUDGET-2 review: L2-aware classification — a chunk
                // resident in the persistent exact-key L2 would be served by
                // the consume call's `get_or_fetch` with ZERO object-store
                // GETs; classifying it Cold (DRAM-only peek) turned that free
                // serve into a billed wave GET whose bytes were then dropped.
                Some(sc)
                    if sc
                        .peek_exact_residency(CacheKind::Other, path, desc.start, desc.range_len)
                        .await
                        .is_some() =>
                {
                    chunk_slots.push(ChunkSlot::SiteServed);
                }
                _ => {
                    chunk_slots.push(ChunkSlot::Cold(chunk_cold_ranges.len()));
                    chunk_cold_ranges.push(desc.start..desc.start + desc.range_len);
                }
            }
        }
        let mut chunk_cold_bytes: Vec<Option<Vec<u8>>> = vec![None; chunk_cold_ranges.len()];
        if !chunk_cold_ranges.is_empty() {
            let batched = match fs
                .read_ranges_prefetch(path, chunk_cold_ranges.clone())
                .await
            {
                Ok(bufs) => bufs,
                Err(_) => {
                    // Defensive fallback to the sequential baseline shape.
                    let mut seq = Vec::with_capacity(chunk_cold_ranges.len());
                    for range in &chunk_cold_ranges {
                        let b = fs
                            .read_range(path, range.start, range.end - range.start)
                            .await
                            .map_err(|err| {
                                anyhow::anyhow!("coalesced scan OID-chunk fallback {path}: {err}")
                            })?;
                        seq.push(b);
                    }
                    seq
                }
            };
            // Metrics parity: one record per physical GET, at batch time.
            for ((range, buf), slot) in chunk_cold_ranges
                .iter()
                .zip(batched)
                .zip(chunk_cold_bytes.iter_mut())
            {
                record_get_physical(CacheTier::ResultPayload, range.end - range.start, trace_on);
                *slot = Some(buf);
            }
            crate::observability::io_trace::drain_and_forward_read_ranges_metrics();
        }

        for (desc_i, (bi, locals)) in block_rows.iter().enumerate() {
            let desc = &chunk_descs[desc_i];
            let pre = match &chunk_slots[desc_i] {
                ChunkSlot::Cold(i) => Some(chunk_cold_bytes[*i].take().ok_or_else(|| {
                    anyhow::anyhow!("coalesced scan OID-chunk fetch {path}: cold chunk {i} missing")
                })?),
                ChunkSlot::SiteServed => None,
            };
            // ADR-065 Q3: same read-through cache class as the legacy OID path.
            let buf: Arc<[u8]> = if let Some(sc) = survivor_cache {
                match pre {
                    Some(mut pre) => sc
                        .get_or_fetch(
                            CacheKind::Other,
                            path,
                            desc.start,
                            desc.range_len,
                            || async {
                                Ok::<_, proximadb_storage_filesystem_types::FilesystemError>(
                                    std::mem::take(&mut pre),
                                )
                            },
                        )
                        .await
                        .map_err(|e| {
                            anyhow::anyhow!("coalesced scan OID-chunk fetch {path}: {e}")
                        })?,
                    None => {
                        // SiteServed (or a peek/consume race): the original
                        // baseline loader — records fire iff it truly runs.
                        sc.get_or_fetch(
                            CacheKind::Other,
                            path,
                            desc.start,
                            desc.range_len,
                            || async move {
                                let b = fs.read_range(path, desc.start, desc.range_len).await?;
                                record_get_physical(
                                    CacheTier::ResultPayload,
                                    desc.range_len,
                                    trace_on,
                                );
                                Ok(b)
                            },
                        )
                        .await
                        .map_err(|e| {
                            anyhow::anyhow!("coalesced scan OID-chunk fetch {path}: {e}")
                        })?
                    }
                }
            } else {
                match pre {
                    Some(b) => Arc::from(b),
                    None => {
                        let b = fs
                            .read_range(path, desc.start, desc.range_len)
                            .await
                            .map_err(|e| {
                                anyhow::anyhow!("coalesced scan OID-chunk fetch {path}: {e}")
                            })?;
                        record_get_physical(CacheTier::ResultPayload, desc.range_len, trace_on);
                        Arc::from(b)
                    }
                }
            };
            if buf.len() as u64 != desc.range_len {
                anyhow::bail!(
                    "coalesced scan OID chunk fetch {path} block {bi} returned {} bytes, expected {}",
                    buf.len(),
                    desc.range_len
                );
            }
            let oids = proximadb_block_format::decode_str_chunk_checked(
                &buf,
                desc.encoding_id,
                desc.is_lz4,
                desc.row_count,
            )
            .map_err(|err| {
                anyhow::anyhow!("coalesced scan OID chunk decode {path} block {bi}: {err}")
            })?;
            for &local in locals {
                let bi = *bi;
                let oid = oids
                    .get(local)
                    .and_then(Option::as_ref)
                    .filter(|oid| !oid.is_empty())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "coalesced scan OID chunk decode {path} block {bi} row {local} produced no identity"
                        )
                    })?;
                let g = block_start[bi] as usize + local;
                if oid_of.insert(g, oid.clone()).is_some() {
                    anyhow::bail!("coalesced scan OID row {g} resolved more than once");
                }
            }
        }
    } else {
        // TD-RDSTRAT-12 §3 round 4 (develop, verbatim): peek·batch·feed for the
        // legacy whole-block OID tail.
        let oid_policy = resolve_adaptive_range_policy(
            path,
            "region_d_topk",
            "PROXIMADB_PAX_COALESCE_GAP",
            "PROXIMADB_PAX_COALESCE_RANGE",
            defaults,
            |candidate| {
                estimate_coalesced_byte_ranges(
                    topk_blocks.iter().filter_map(|&index| {
                        footer
                            .blocks
                            .get(index)
                            .map(|block| (block.offset, block.offset + block.size as u64))
                    }),
                    candidate,
                )
            },
        );
        let oid_fetches = plan_coalesced_block_ranges(&footer, &topk_blocks, &oid_policy);
        // TD-RDSTRAT-12 §3 round 4: peek·batch·feed for the OID tail (mirror of
        // W1; plain exact-key seam — no parent tier here). Classification via the
        // existing side-effect-free `peek_memory_exact`; one bounded-concurrent
        // wave for the true-cold block ranges (D4 bound: min(INFLIGHT,N) ×
        // max_range_bytes ≤ 128 MiB transient).
        enum OidSlot {
            SiteServed,
            Cold(usize),
        }
        let mut oid_slots: Vec<OidSlot> = Vec::with_capacity(oid_fetches.len());
        let mut oid_cold_ranges: Vec<std::ops::Range<u64>> = Vec::new();
        for fetch in &oid_fetches {
            match survivor_cache {
                Some(sc)
                    if sc
                        .peek_memory_exact(
                            CacheKind::Other,
                            path,
                            fetch.start,
                            fetch.end - fetch.start,
                        )
                        .await
                        .is_some() =>
                {
                    oid_slots.push(OidSlot::SiteServed);
                }
                _ => {
                    oid_slots.push(OidSlot::Cold(oid_cold_ranges.len()));
                    oid_cold_ranges.push(fetch.start..fetch.end);
                }
            }
        }
        let mut oid_cold_bytes: Vec<Option<Vec<u8>>> = vec![None; oid_cold_ranges.len()];
        if !oid_cold_ranges.is_empty() {
            let batched = match fs.read_ranges_prefetch(path, oid_cold_ranges.clone()).await {
                Ok(bufs) => bufs,
                Err(_) => {
                    // Defensive fallback to the sequential baseline shape.
                    let mut seq = Vec::with_capacity(oid_cold_ranges.len());
                    for range in &oid_cold_ranges {
                        let b = fs
                            .read_range(path, range.start, range.end - range.start)
                            .await
                            .map_err(|err| {
                                anyhow::anyhow!("coalesced scan OID fallback {path}: {err}")
                            })?;
                        seq.push(b);
                    }
                    seq
                }
            };
            // Metrics parity: the baseline loader records ONLY
            // `record_get(ResultPayload, …)` (never region bytes) — fire it here,
            // once per physical GET, and keep the Cold loaders silent.
            for ((range, buf), slot) in oid_cold_ranges
                .iter()
                .zip(batched)
                .zip(oid_cold_bytes.iter_mut())
            {
                record_get_physical(CacheTier::ResultPayload, range.end - range.start, trace_on);
                *slot = Some(buf);
            }
            crate::observability::io_trace::drain_and_forward_read_ranges_metrics();
        }
        for (fetch_i, fetch) in oid_fetches.iter().enumerate() {
            let start = fetch.start;
            let range_len = fetch.end - fetch.start;
            let pre = match &oid_slots[fetch_i] {
                OidSlot::Cold(i) => Some(oid_cold_bytes[*i].take().ok_or_else(|| {
                    anyhow::anyhow!("coalesced scan OID fetch {path}: cold fetch {i} missing")
                })?),
                OidSlot::SiteServed => None,
            };
            // ADR-065 Q3: same read-through cache as survivors (OID ranges are also
            // immutable per segment + repeat across hot queries). `CacheKind::Other`
            // separates OID stats from survivor (QuantizedCodes) stats.
            let buf: Arc<[u8]> = if let Some(sc) = survivor_cache {
                match pre {
                    Some(mut pre) => sc
                        .get_or_fetch(CacheKind::Other, path, start, range_len, || async {
                            Ok::<_, proximadb_storage_filesystem_types::FilesystemError>(
                                std::mem::take(&mut pre),
                            )
                        })
                        .await
                        .map_err(|e| anyhow::anyhow!("coalesced scan OID fetch {path}: {e}"))?,
                    None => {
                        // SiteServed (or a peek/consume race): the original
                        // baseline loader — records fire iff it truly runs.
                        sc.get_or_fetch(CacheKind::Other, path, start, range_len, || async move {
                            let b = fs.read_range(path, start, range_len).await?;
                            record_get_physical(CacheTier::ResultPayload, range_len, trace_on);
                            Ok(b)
                        })
                        .await
                        .map_err(|e| anyhow::anyhow!("coalesced scan OID fetch {path}: {e}"))?
                    }
                }
            } else {
                match pre {
                    Some(b) => Arc::from(b),
                    None => {
                        let b = fs
                            .read_range(path, start, range_len)
                            .await
                            .map_err(|e| anyhow::anyhow!("coalesced scan OID fetch {path}: {e}"))?;
                        record_get_physical(CacheTier::ResultPayload, range_len, trace_on);
                        Arc::from(b)
                    }
                }
            };
            for &bi in &fetch.blocks {
                let b = footer.blocks.get(bi).ok_or_else(|| {
                    anyhow::anyhow!("coalesced scan OID block {bi} is missing from footer")
                })?;
                let rel = b.offset.checked_sub(fetch.start).ok_or_else(|| {
                    anyhow::anyhow!(
                        "coalesced scan OID block decode {path} block {bi}: offset precedes fetch"
                    )
                })? as usize;
                let end = rel.checked_add(b.size as usize).ok_or_else(|| {
                    anyhow::anyhow!(
                        "coalesced scan OID block decode {path} block {bi}: extent overflows"
                    )
                })?;
                if end > buf.len() {
                    anyhow::bail!(
                        "coalesced scan OID block decode {path} block {bi}: extent {rel}..{end} exceeds fetched {} bytes",
                        buf.len()
                    );
                }
                let reader = PaxBlockReader::open(&buf[rel..end]).map_err(|err| {
                    anyhow::anyhow!("coalesced scan OID block decode {path} block {bi}: {err}")
                })?;
                let oids = reader
                    .decode_str_stripe_checked(col_id::OID)
                    .map_err(|err| {
                        anyhow::anyhow!("coalesced scan OID block decode {path} block {bi}: {err}")
                    })?
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "coalesced scan OID block decode {path} block {bi}: OID stripe missing"
                        )
                    })?;
                let locals = block_rows.get(&bi).ok_or_else(|| {
                    anyhow::anyhow!("coalesced scan OID block {bi} has no selected rows")
                })?;
                for &local in locals {
                    let oid = oids
                        .get(local)
                        .and_then(Option::as_ref)
                        .filter(|oid| !oid.is_empty())
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "coalesced scan OID block decode {path} block {bi} row {local} produced no identity"
                            )
                        })?;
                    let g = block_start[bi] as usize + local;
                    if oid_of.insert(g, oid.clone()).is_some() {
                        anyhow::bail!("coalesced scan OID row {g} resolved more than once");
                    }
                }
            }
        }
    }

    // TD-RDSTRAT-13 PR-B: Region-D fetch+decode wall recorded at the boundary
    // where all top-k OIDs are resolved.
    {
        let us = region_d_started.elapsed().as_micros() as u64;
        crate::observability::io_trace::record_region_d_fetch_decode_us(us);
        crate::storage::engines::sst::metrics::record_region_d_fetch_decode_us(us);
    }

    // 6. Build the top-k hits in nearest-first order (from `scored`); step 7
    //    canonicalizes the rank scores.
    let mut hits: Vec<CascadeHit> = Vec::with_capacity(top_k_rows.len());
    for (g, dist) in scored.iter().take(k) {
        let oid = oid_of
            .get(g)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("coalesced scan top-k row {g} has no resolved OID"))?;
        hits.push(CascadeHit {
            oid,
            distance: *dist,
            vector: None,
            position: *g as u32,
        });
    }

    // 7. Finalize global top-k (nearest-first).
    hits.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.truncate(k);
    for hit in &mut hits {
        hit.distance = canonical_score_from_rank_score(metric, hit.distance);
    }
    Ok(Some(hits))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TD-PAXRG-1 Phase A: the coalesced header-prefix GET floor covers the v4
    /// (row-group Region D) prefix length, so one ranged read serves every
    /// layout version and `parse` branches on the version byte.
    #[test]
    fn v4_header_get_fetches_88_bytes() {
        assert_eq!(coalesced_header_prefetch_floor(), SEG_HEADER_PREFIX_LEN);
        assert_eq!(coalesced_header_prefetch_floor(), 88);
        assert!(coalesced_header_prefetch_floor() > 72, "floor grew past v3");
    }

    /// TD-RDSTRAT-13 C1: `record_get_physical` passes `tier as usize` into the
    /// io_trace/SST per-tier arrays, whose index order is a documented
    /// cross-crate contract (0=IdxA, 1=Ctl, 2=Meta, 3=ProbeA, 4=Surv, 5=OID).
    /// Reordering the enum would silently fold one tier's GETs into another's
    /// counters — pin it.
    #[test]
    fn cache_tier_discriminant_matches_tier_counter_contract() {
        assert_eq!(CacheTier::InvariantIndex as usize, 0);
        assert_eq!(CacheTier::SearchControl as usize, 1);
        assert_eq!(CacheTier::InvariantMeta as usize, 2);
        assert_eq!(CacheTier::ProbeIndex as usize, 3);
        assert_eq!(CacheTier::SurvivorPayload as usize, 4);
        assert_eq!(CacheTier::ResultPayload as usize, 5);
        let labels = crate::storage::engines::sst::metrics::TIER_LABELS;
        for (tier, label) in [
            CacheTier::InvariantIndex,
            CacheTier::SearchControl,
            CacheTier::InvariantMeta,
            CacheTier::ProbeIndex,
            CacheTier::SurvivorPayload,
            CacheTier::ResultPayload,
        ]
        .into_iter()
        .zip(labels)
        {
            assert_eq!(tier.label(), label);
        }
    }

    /// TD-PAXRG-1 Phase C: a v4 segment with the hoisted exact tier (Region C)
    /// still satisfies the segment-level exact-authority predicate — the
    /// footer's `has_f32_tier` capability flag keeps its meaning when the tier
    /// moves out of the blocks.
    #[test]
    fn exact_authority_holds_for_region_c_segment() {
        use proximadb_block_format::{BlockCompression, BlockMode, VectorQuant};
        use proximadb_records::{EmbeddingCell, EmbeddingValues};
        use proximadb_storage_common::pax_block::PaxSegmentWriter;

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("rg-c-authority.pax");
        let mut writer = PaxSegmentWriter::new(
            &path,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            1,
            Some(256 * 1024),
        )
        .with_quant(VectorQuant::RaBitQ)
        .with_coalesced_rabitq(true)
        .with_f32_tier(true)
        .with_rg_layout(true)
        .with_oid_resolver(true);
        for row in 0..32 {
            let mut record = ProximaRecord {
                oid: format!("oid-{row}"),
                tenant_id: "t".into(),
                created_at_ns: 1_000 + row as i64,
                updated_at_ns: 1_000 + row as i64,
                ..Default::default()
            };
            record.embeddings.push(EmbeddingCell {
                modality: "dense".into(),
                dim: 16,
                values: EmbeddingValues::Fp32(vec![0.5; 16]),
                ..Default::default()
            });
            writer.add_record(&record).expect("add record");
        }
        writer.finish().expect("finish v4+C segment");

        let segment = std::fs::read(&path).expect("read segment");
        assert!(
            pax_segment_has_exact_vector_authority(&segment),
            "Region C must satisfy the exact-authority predicate"
        );
    }

    /// TD-PAXRG-1 Phase D: on a v4 segment the top-k OIDs come from the
    /// footer-addressed OID CHUNKS, not whole row groups. Evidence: (a) result
    /// parity with the v3 whole-block fetch on the identical corpus/order
    /// (same Region A/B encodes ⇒ identical distances), and (b) strictly fewer
    /// ranged bytes at no more GETs — request economy at parity, byte win.
    #[tokio::test]
    async fn v4_oid_chunk_fetch_parity_and_byte_win_vs_v3() {
        enable_coalesced_rabitq();
        use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
        use proximadb_storage_common::pax_block::RG_TARGET_MIN_BYTES;
        use proximadb_storage_filesystem_types::FileSystem;
        use proximadb_storage_filesystem_types::counting::{CountingFileSystem, global_counters};

        const DIM: usize = 64;
        const N: usize = 1_500;
        const K: usize = 10;

        let corpus: Vec<Vec<f32>> = (0..N)
            .map(|i| {
                (0..DIM)
                    .map(|d| (((i * 131 + d * 17) % 251) as f32) * 0.01)
                    .collect()
            })
            .collect();

        let build = |path: &std::path::Path, rg: bool| {
            let mut writer = PaxSegmentWriter::new(
                path,
                BlockMode::Pax,
                BlockCompression::None,
                "col",
                0,
                1,
                // v3: small blocks (multi-block segment); v4: floor clamps to
                // one big RG so the whole-RG-vs-chunk contrast is maximal.
                Some(if rg { RG_TARGET_MIN_BYTES } else { 96 * 1024 }),
            )
            .with_quant(VectorQuant::RaBitQ)
            .with_coalesced_rabitq(true)
            .with_oid_resolver(true);
            if rg {
                writer = writer.with_rg_layout(true);
            }
            for (i, v) in corpus.iter().enumerate() {
                writer.add_record(&rec(&format!("r{i}"), 1000 + i as i64, v.clone()))?;
            }
            writer.finish()
        };

        let dir = tempfile::tempdir().unwrap();
        let v3_path = dir.path().join("v3.pax");
        let v4_path = dir.path().join("v4.pax");
        build(&v3_path, false).unwrap();
        build(&v4_path, true).unwrap();

        // Sanity: the v4 file really is the v4 layout.
        let v4_bytes = std::fs::read(&v4_path).unwrap();
        let header = SegmentHeaderPrefix::parse(&v4_bytes[..SEG_HEADER_PREFIX_LEN]).unwrap();
        assert_eq!(header.layout_version, SEG_LAYOUT_VERSION);

        let inner = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
            .await
            .unwrap();
        let counters = global_counters();
        let fs: Arc<dyn FileSystem> =
            Arc::new(CountingFileSystem::new(Arc::new(inner), counters.clone()));

        let query = corpus[137].clone();
        let run = |path: &std::path::Path| {
            let fs = fs.clone();
            let counters = counters.clone();
            let query = query.clone();
            let path = path.to_path_buf();
            async move {
                counters.reset();
                let hits = rabitq_search_segment_coalesced(
                    &*fs,
                    path.to_str().unwrap(),
                    &query,
                    K,
                    RankMetric::L2,
                    None,
                    None,
                )
                .await
                .unwrap()
                .expect("cascade must serve the segment");
                (
                    hits,
                    counters
                        .bytes_read
                        .load(std::sync::atomic::Ordering::Relaxed),
                    counters.total_gets(),
                )
            }
        };

        let (v3_hits, v3_bytes, v3_gets) = run(&v3_path).await;
        let (v4_hits, v4_bytes, v4_gets) = run(&v4_path).await;

        // (a) Parity: same hits, same order, same scores (identical A/B
        // encodes; only Region D's fetch strategy differs).
        assert_eq!(v3_hits.len(), v4_hits.len(), "top-k count parity");
        for (a, b) in v3_hits.iter().zip(&v4_hits) {
            assert_eq!(a.oid, b.oid, "OID parity: {} vs {}", a.oid, b.oid);
            assert_eq!(a.distance.to_bits(), b.distance.to_bits(), "score parity");
        }

        // (b) The byte win: the v4 OID-chunk fetch reads strictly less — the
        // delta is Region D (whole RG bodies vs ~k OID strings). Assert a
        // savings floor so a dead chunk path (falling back to whole RGs)
        // cannot pass on noise.
        eprintln!(
            "[oid-chunk] v3: bytes={v3_bytes} gets={v3_gets}; v4: bytes={v4_bytes} \
             gets={v4_gets}; bytes_saved={}",
            v3_bytes.saturating_sub(v4_bytes)
        );
        assert!(
            v4_bytes < v3_bytes && v3_bytes.saturating_sub(v4_bytes) >= 32 * 1024,
            "chunk fetch must save Region-D bytes: v3={v3_bytes} v4={v4_bytes}"
        );
        // GET economy: chunks are fetched per touched RG and deliberately NOT
        // coalesced across RGs (merging would span the inter-RG gap and
        // re-read the bodies the chunk fetch exists to skip). v3's ADJACENT
        // whole blocks can merge into fewer GETs, so v4 may use a small
        // constant more requests — the deliberate trade is bytes-dominant.
        // The TD's flip criterion is therefore: GET count within a small
        // constant of v3, bytes strictly down.
        assert!(
            v4_gets <= v3_gets + 2,
            "request economy within a small constant: v3={v3_gets} v4={v4_gets}"
        );
    }

    fn write_oid_failure_fixture(path: &std::path::Path, rg_layout: bool) -> Vec<Vec<f32>> {
        use proximadb_block_format::{BlockCompression, BlockMode, VectorQuant};
        use proximadb_storage_common::pax_block::PaxSegmentWriter;

        const DIM: usize = 16;
        let corpus: Vec<Vec<f32>> = (0..32)
            .map(|row| {
                (0..DIM)
                    .map(|dim| ((row * 37 + dim * 11) % 101) as f32 * 0.01)
                    .collect()
            })
            .collect();
        let mut writer = PaxSegmentWriter::new(
            path,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            1,
            Some(256 * 1024),
        )
        .with_quant(VectorQuant::RaBitQ)
        .with_coalesced_rabitq(true)
        .with_oid_resolver(true);
        if rg_layout {
            writer = writer.with_rg_layout(true);
        }
        for (row, vector) in corpus.iter().enumerate() {
            writer
                .add_record(&rec(
                    &format!("oid-{row}"),
                    1_000 + row as i64,
                    vector.clone(),
                ))
                .expect("add fixture row");
        }
        writer.finish().expect("finish fixture segment");
        corpus
    }

    fn rewrite_coalesced_footer(
        path: &std::path::Path,
        mutate: impl FnOnce(&mut proximadb_storage_common::segment_layout::SegmentFooterIndex),
    ) {
        let mut bytes = std::fs::read(path).expect("read fixture segment");
        let header = SegmentHeaderPrefix::parse(&bytes[..SEG_HEADER_PREFIX_LEN])
            .expect("parse fixture header");
        let footer_start = header.footer_off as usize;
        let footer_end = footer_start + header.footer_len as usize;
        let mut footer = proximadb_storage_common::segment_layout::SegmentFooterIndex::parse(
            &bytes[footer_start..footer_end],
        )
        .expect("parse fixture footer");
        mutate(&mut footer);
        let encoded = footer.to_bytes().expect("encode mutated footer");
        assert_eq!(
            encoded.len(),
            header.footer_len as usize,
            "test mutation must preserve footer geometry"
        );
        bytes[footer_start..footer_end].copy_from_slice(&encoded);
        std::fs::write(path, bytes).expect("rewrite fixture segment");
    }

    #[derive(Debug)]
    struct PrefetchErrorFileSystem {
        inner: Arc<dyn proximadb_storage_filesystem_types::FileSystem>,
    }

    #[async_trait::async_trait]
    impl proximadb_storage_filesystem_types::FileSystem for PrefetchErrorFileSystem {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        async fn read(&self, path: &str) -> proximadb_storage_filesystem_types::FsResult<Vec<u8>> {
            self.inner.read(path).await
        }

        async fn read_range(
            &self,
            path: &str,
            offset: u64,
            length: u64,
        ) -> proximadb_storage_filesystem_types::FsResult<Vec<u8>> {
            self.inner.read_range(path, offset, length).await
        }

        async fn read_ranges_prefetch(
            &self,
            _path: &str,
            _ranges: Vec<std::ops::Range<u64>>,
        ) -> proximadb_storage_filesystem_types::FsResult<Vec<Vec<u8>>> {
            Err(proximadb_storage_filesystem_types::FilesystemError::Io(
                std::io::Error::other("injected prefetch failure"),
            ))
        }

        async fn write(
            &self,
            path: &str,
            data: &[u8],
            options: Option<proximadb_storage_filesystem_types::FileOptions>,
        ) -> proximadb_storage_filesystem_types::FsResult<()> {
            self.inner.write(path, data, options).await
        }

        async fn append(
            &self,
            path: &str,
            data: &[u8],
        ) -> proximadb_storage_filesystem_types::FsResult<()> {
            self.inner.append(path, data).await
        }

        async fn delete(&self, path: &str) -> proximadb_storage_filesystem_types::FsResult<()> {
            self.inner.delete(path).await
        }

        async fn exists(&self, path: &str) -> proximadb_storage_filesystem_types::FsResult<bool> {
            self.inner.exists(path).await
        }

        async fn metadata(
            &self,
            path: &str,
        ) -> proximadb_storage_filesystem_types::FsResult<
            proximadb_storage_filesystem_types::FsFileMetadata,
        > {
            self.inner.metadata(path).await
        }

        async fn list(
            &self,
            path: &str,
        ) -> proximadb_storage_filesystem_types::FsResult<
            Vec<proximadb_storage_filesystem_types::DirEntry>,
        > {
            self.inner.list(path).await
        }

        async fn create_dir(&self, path: &str) -> proximadb_storage_filesystem_types::FsResult<()> {
            self.inner.create_dir(path).await
        }

        async fn create_dir_all(
            &self,
            path: &str,
        ) -> proximadb_storage_filesystem_types::FsResult<()> {
            self.inner.create_dir_all(path).await
        }

        async fn copy(
            &self,
            from: &str,
            to: &str,
        ) -> proximadb_storage_filesystem_types::FsResult<()> {
            self.inner.copy(from, to).await
        }

        async fn move_file(
            &self,
            from: &str,
            to: &str,
        ) -> proximadb_storage_filesystem_types::FsResult<()> {
            self.inner.move_file(from, to).await
        }

        fn filesystem_type(&self) -> &'static str {
            "prefetch-error-test"
        }

        async fn sync(&self) -> proximadb_storage_filesystem_types::FsResult<()> {
            self.inner.sync().await
        }

        async fn open_file(
            &self,
            path: &str,
            create: bool,
        ) -> proximadb_storage_filesystem_types::FsResult<
            Box<dyn proximadb_storage_filesystem_types::FilesystemFile>,
        > {
            self.inner.open_file(path, create).await
        }
    }

    #[tokio::test]
    async fn v4_oid_chunk_decode_failure_fails_closed() {
        use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bad-oid-encoding.pax");
        let corpus = write_oid_failure_fixture(&path, true);
        rewrite_coalesced_footer(&path, |footer| {
            for stats in footer.block_stats.iter_mut().flatten() {
                stats.oid_encoding_id = u8::MAX;
            }
        });
        let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
            .await
            .expect("local filesystem");

        let err = rabitq_search_segment_coalesced(
            &fs,
            path.to_str().expect("utf8 path"),
            &corpus[7],
            5,
            RankMetric::L2,
            None,
            None,
        )
        .await
        .expect_err("a malformed OID chunk must not produce empty-ID hits");
        assert!(
            err.to_string().contains("OID chunk decode"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn v4_oid_chunk_extent_outside_row_group_fails_closed() {
        use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bad-oid-extent.pax");
        let corpus = write_oid_failure_fixture(&path, true);
        rewrite_coalesced_footer(&path, |footer| {
            for (block, stats) in footer.blocks.iter().zip(&mut footer.block_stats) {
                if let Some(stats) = stats {
                    stats.oid_chunk_rel_off = block.size;
                }
            }
        });
        let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
            .await
            .expect("local filesystem");

        let err = rabitq_search_segment_coalesced(
            &fs,
            path.to_str().expect("utf8 path"),
            &corpus[7],
            5,
            RankMetric::L2,
            None,
            None,
        )
        .await
        .expect_err("an out-of-range OID chunk must fail before I/O");
        assert!(
            err.to_string().contains("outside row group"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn legacy_oid_block_corruption_fails_closed() {
        use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bad-legacy-oid-block.pax");
        let corpus = write_oid_failure_fixture(&path, false);
        let mut bytes = std::fs::read(&path).expect("read fixture segment");
        let footer =
            proximadb_storage_common::segment_layout::SegmentFooterIndex::locate_in_segment(&bytes)
                .expect("locate footer")
                .expect("coalesced footer");
        let block = footer.blocks.first().expect("one data block");
        let corrupt_at = block.offset as usize + block.size as usize / 2;
        bytes[corrupt_at] ^= 0x5a;
        std::fs::write(&path, bytes).expect("rewrite fixture segment");
        let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
            .await
            .expect("local filesystem");

        let err = rabitq_search_segment_coalesced(
            &fs,
            path.to_str().expect("utf8 path"),
            &corpus[7],
            5,
            RankMetric::L2,
            None,
            None,
        )
        .await
        .expect_err("a corrupt legacy OID block must not produce empty-ID hits");
        assert!(
            err.to_string().contains("OID block decode"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn oid_prefetch_fallback_records_each_physical_get_once() {
        use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

        unsafe { std::env::set_var("PROXIMADB_TRACE_GETS", "1") };
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("oid-prefetch-fallback.pax");
        let corpus = write_oid_failure_fixture(&path, false);
        let local = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
            .await
            .expect("local filesystem");
        let fs = PrefetchErrorFileSystem {
            inner: Arc::new(local),
        };
        let _ = drain_get_trace();

        let (hits, snapshot) = crate::observability::io_trace::scope(async {
            let hits = rabitq_search_segment_coalesced(
                &fs,
                path.to_str().expect("utf8 path"),
                &corpus[7],
                5,
                RankMetric::L2,
                None,
                None,
            )
            .await
            .expect("sequential fallback succeeds")
            .expect("coalesced segment");
            let snapshot = crate::observability::io_trace::snapshot().expect("trace scope");
            (hits, snapshot)
        })
        .await;
        assert!(hits.iter().all(|hit| !hit.oid.is_empty()));
        let trace = drain_get_trace();
        let oid_gets: Vec<_> = trace
            .iter()
            .copied()
            .filter(|(tier, _)| *tier == CacheTier::ResultPayload)
            .collect();
        assert_eq!(
            oid_gets.len(),
            1,
            "one fallback block read must produce one OID GET record: {oid_gets:?}"
        );
        let survivor_bytes: u64 = trace
            .iter()
            .filter(|(tier, _)| *tier == CacheTier::SurvivorPayload)
            .map(|(_, bytes)| *bytes)
            .sum();
        assert_eq!(
            snapshot.ivf_region_b_bytes, survivor_bytes,
            "fallback Region-B bytes must equal the physical survivor GET trace"
        );
        unsafe { std::env::remove_var("PROXIMADB_TRACE_GETS") };
    }

    fn cp_cfg(mult: f32) -> crate::core::config::CoarseProbeConfig {
        crate::core::config::CoarseProbeConfig {
            enable_read_probe: true,
            enable_write_train: true,
            nprobe_multiplier: mult,
            nprobe_min: 3,
            nprobe_max: 0,
            ncomp_floor: 0,
        }
    }

    #[test]
    fn default_probe_quality_and_backend_ranges_match_sift_acceptance() {
        let defaults: CoarseProbeSettings =
            crate::core::config::CoarseProbeConfig::default().into();
        assert_eq!(
            geometric_nprobe(30, defaults),
            11,
            "the one-segment SIFT1M geometry needs 11/30 cells for recall@10 >= 0.98"
        );

        let azure = default_probe_range_policy("azure://container/segment.pax");
        assert_eq!(azure.max_gap_bytes, 1024 * 1024);
        assert_eq!(azure.max_range_bytes, 4 * 1024 * 1024);

        let local = default_probe_range_policy("file:///data/segment.pax");
        assert_eq!(local.max_gap_bytes, 256 * 1024);
        assert_eq!(local.max_range_bytes, 1024 * 1024);
    }

    #[test]
    fn invariant_control_l2_codec_round_trips_optional_regions() {
        let original = SegmentInvariants {
            header_bytes: vec![1, 2, 3],
            region_bytes: Some(Arc::from(&b"region-a"[..])),
            footer_bytes: vec![4, 5],
            a0_bytes: Some(Arc::from(&b"a0"[..])),
            rabitq_header_bytes: Some(Arc::from(&b"params"[..])),
        };
        let decoded =
            decode_invariant_control(&encode_invariant_control(&original)).expect("valid codec");
        assert_eq!(decoded.header_bytes, original.header_bytes);
        assert_eq!(decoded.footer_bytes, original.footer_bytes);
        assert_eq!(decoded.a0_bytes.as_deref(), original.a0_bytes.as_deref());
        assert_eq!(
            decoded.rabitq_header_bytes.as_deref(),
            original.rabitq_header_bytes.as_deref()
        );
        assert!(
            decoded.region_bytes.is_none(),
            "Region A has its own persistent entry"
        );

        let control_only = SegmentInvariants {
            header_bytes: vec![9],
            region_bytes: None,
            footer_bytes: vec![8],
            a0_bytes: None,
            rabitq_header_bytes: None,
        };
        let decoded = decode_invariant_control(&encode_invariant_control(&control_only))
            .expect("valid codec");
        assert!(decoded.a0_bytes.is_none());
        assert!(decoded.rabitq_header_bytes.is_none());
        assert!(decode_invariant_control(b"truncated").is_none());
    }

    #[tokio::test]
    async fn invariant_l2_survives_dram_and_store_restart() {
        let dir = tempfile::tempdir().expect("test tempdir");
        let path = "file:///tenant-a/collection-a/segment-a.px";
        let original = Arc::new(SegmentInvariants {
            header_bytes: vec![1, 2, 3],
            region_bytes: Some(Arc::from(&b"region-a"[..])),
            footer_bytes: vec![4, 5],
            a0_bytes: Some(Arc::from(&b"a0"[..])),
            rabitq_header_bytes: Some(Arc::from(&b"params"[..])),
        });
        let store =
            Arc::new(PersistentByteStore::open(dir.path(), 1 << 20).expect("open L2 store"));
        let cache = SegmentInvariantsCache::with_l2(1 << 20, store);
        cache.put_with_l2(path.to_string(), original.clone()).await;
        drop(cache);

        let reopened_store =
            Arc::new(PersistentByteStore::open(dir.path(), 1 << 20).expect("reopen L2 store"));
        let reopened = SegmentInvariantsCache::with_l2(1 << 20, reopened_store);
        let lookup = reopened.get_or_promote(path).await;
        let promoted = lookup.value().expect("persistent L2 hit");
        assert!(matches!(lookup, InvariantLookup::L2(_)));
        assert_eq!(promoted.header_bytes, original.header_bytes);
        assert_eq!(promoted.footer_bytes, original.footer_bytes);
        assert!(
            promoted.region_bytes.is_none(),
            "A0-backed segments must not hydrate complete Region A after restart"
        );
        assert_eq!(
            reopened
                .get_region_range(path, 1, b"region-a".len() as u64 - 1)
                .await
                .expect("bounded Region A range")
                .as_ref(),
            &b"egion-a"[..]
        );
        assert_eq!(promoted.a0_bytes.as_deref(), original.a0_bytes.as_deref());
        assert_eq!(
            promoted.rabitq_header_bytes.as_deref(),
            original.rabitq_header_bytes.as_deref()
        );
        assert!(
            matches!(reopened.get_or_promote(path).await, InvariantLookup::L1(_)),
            "the persistent value must be promoted into DRAM"
        );
        reopened.invalidate_all(path).await;
        assert!(reopened.get(path).is_none(), "L1 entries were invalidated");
        assert!(matches!(
            reopened.get_or_promote(path).await,
            InvariantLookup::Miss
        ));

        let one_level_path = "file:///tenant-a/collection-a/segment-one-level.px";
        let one_level = Arc::new(SegmentInvariants {
            header_bytes: vec![1],
            region_bytes: Some(Arc::from(&b"whole-region"[..])),
            footer_bytes: vec![2],
            a0_bytes: None,
            rabitq_header_bytes: Some(Arc::from(&b"params"[..])),
        });
        reopened
            .put_with_l2(one_level_path.to_string(), one_level.clone())
            .await;
        drop(reopened);
        let one_level_store =
            Arc::new(PersistentByteStore::open(dir.path(), 1 << 20).expect("reopen one-level L2"));
        let one_level_cache = SegmentInvariantsCache::with_l2(1 << 20, one_level_store);
        let promoted = one_level_cache
            .get_or_promote(one_level_path)
            .await
            .value()
            .expect("one-level persistent hit");
        assert_eq!(
            promoted.region_bytes.as_deref(),
            one_level.region_bytes.as_deref(),
            "one-level scans still require their complete Region A"
        );
    }

    #[tokio::test]
    async fn local_pax_seed_streams_regions_into_persistent_cache() {
        use proximadb_records::{EmbeddingCell, EmbeddingValues};
        use proximadb_storage_common::coarse_directory::CoarseModel;

        const DIM: usize = 8;
        let dir = tempfile::tempdir().expect("test tempdir");
        let segment_path = dir.path().join("spill-output.pax");
        let model = CoarseModel {
            dim: DIM as u32,
            n_comp: 1,
            pca_mean: vec![0.0; DIM],
            pca_components: vec![0.1; DIM],
            centroids: vec![-1.0, 1.0],
            radii: vec![1.0, 1.0],
            cell_rows: vec![2, 2],
            seed: 7,
            trained_on: 4,
        };
        let mut writer = PaxSegmentWriter::new(
            &segment_path,
            BlockMode::Pax,
            BlockCompression::None,
            "collection",
            0,
            1,
            None,
        )
        .with_quant(VectorQuant::RaBitQ)
        .with_rerank_quant(VectorQuant::Sq8)
        .with_coalesced_rabitq(true)
        .with_two_level(model);
        for row in 0..4usize {
            let mut record = rec(&format!("row-{row}"), row as i64, vec![row as f32; DIM]);
            record.embeddings[0] = EmbeddingCell {
                modality: "dense".to_string(),
                dim: DIM as u32,
                values: EmbeddingValues::Fp32(vec![row as f32; DIM]),
                ..EmbeddingCell::default()
            };
            writer.add_record(&record).expect("append record");
        }
        writer.finish().expect("finish PAX");
        let segment = std::fs::read(&segment_path).expect("read PAX");
        let header = SegmentHeaderPrefix::parse(&segment).expect("parse PAX header");

        let cache_path = dir.path().join("cache");
        let store = Arc::new(
            PersistentByteStore::open(&cache_path, 1 << 20).expect("open persistent cache"),
        );
        let invariants = SegmentInvariantsCache::with_l2(1 << 20, store.clone());
        let survivor = SurvivorRangeCache::with_resolver_and_l2(1 << 20, None, Some(store));
        let published = "az://container/data/1/L2_output.pax";
        assert!(
            install_pax_cache_seed_from_local_file(
                published,
                &segment_path,
                true,
                Some(&invariants),
                Some(&survivor),
            )
            .await
            .expect("install local PAX seed")
        );
        let control = invariants.get(published).expect("control cached in DRAM");
        assert!(
            control.region_bytes.is_none(),
            "streaming seed must not materialize Region A in DRAM"
        );
        let region_slice = invariants
            .get_region_range(published, 1, header.rabitq_len - 1)
            .await
            .expect("Region A range from L2");
        assert_eq!(
            region_slice.as_ref(),
            &segment[(header.rabitq_off + 1) as usize
                ..(header.rabitq_off + header.rabitq_len) as usize]
        );
        let sq8_slice = survivor
            .get_or_fetch_in_parent(
                CacheKind::QuantizedCodes,
                published,
                header.sq8_off + 1,
                header.sq8_len - 1,
                header.sq8_off,
                header.sq8_len,
                || async {
                    Err(proximadb_storage_filesystem_types::FilesystemError::Io(
                        std::io::Error::other("object loader must not run"),
                    ))
                },
            )
            .await
            .expect("Region B range from L2");
        assert_eq!(
            sq8_slice.as_ref(),
            &segment[(header.sq8_off + 1) as usize..(header.sq8_off + header.sq8_len) as usize]
        );
    }

    /// TD-COMPACT-9: a REAL (Some) coarse-probe config must WIN and must NOT be
    /// clobbered by a later default-config `SstEngine::new()` init(None) — the
    /// bug that made `[storage.sst_config.coarse_probe] nprobe_multiplier`
    /// silently dead in production. Also verifies the multiplier reaches
    /// geometric_nprobe. (nextest = process isolation; the static won't leak.)
    #[test]
    fn toml_coarse_probe_config_wins_over_default_init() {
        // Simulate a default-config engine seeding first...
        init_coarse_probe_settings(None);
        // ...then the authoritative TOML config (server startup) — must override.
        init_coarse_probe_settings(Some(&cp_cfg(1.5)));
        assert!((coarse_probe_settings().nprobe_multiplier - 1.5).abs() < 1e-6);
        // A later default-config engine init(None) must NOT clobber it.
        init_coarse_probe_settings(None);
        assert!((coarse_probe_settings().nprobe_multiplier - 1.5).abs() < 1e-6);
        // The multiplier actually reaches nprobe: sqrt(64)*1.5 = 12 vs *1.0 = 8.
        assert_eq!(geometric_nprobe(64, coarse_probe_settings()), 12);
        assert_eq!(geometric_nprobe(64, cp_cfg(1.0).into()), 8);
    }

    /// TD-CACHE-2 S2b: Region A and control bytes are separate eviction
    /// units — a put with a region splits into two entries; invalidate
    /// removes both; a region-only shed leaves control serving (recomposed
    /// get degrades to control-only, never a full miss).
    #[test]
    fn invariants_split_region_from_control() {
        let cache = SegmentInvariantsCache::new(1024 * 1024);
        let region: Arc<[u8]> = Arc::from(vec![7u8; 64 * 1024].as_slice());
        cache.put(
            "seg.pax".to_string(),
            Arc::new(SegmentInvariants {
                header_bytes: vec![1; 72],
                region_bytes: Some(region),
                footer_bytes: vec![2; 128],
                a0_bytes: None,
                rabitq_header_bytes: None,
            }),
        );
        // Recomposed hit carries the region.
        let inv = cache.get("seg.pax").expect("hit");
        assert!(
            inv.region_bytes.is_some(),
            "recomposed get carries Region A"
        );
        assert_eq!(inv.header_bytes.len(), 72);
        // Shed ONLY the region companion (simulating per-class eviction).
        cache.invalidate(&SegmentInvariantsCache::region_entry_key("seg.pax"));
        let inv = cache.get("seg.pax").expect("control must still serve");
        assert!(
            inv.region_bytes.is_none(),
            "region shed, control-only hit (3 control GETs still saved)"
        );
        assert_eq!(inv.footer_bytes.len(), 128);
        // Full invalidate removes both and frees all bytes.
        cache.put(
            "seg.pax".to_string(),
            Arc::new(SegmentInvariants {
                header_bytes: vec![1; 72],
                region_bytes: Some(Arc::from(vec![7u8; 1024].as_slice())),
                footer_bytes: vec![2; 128],
                a0_bytes: None,
                rabitq_header_bytes: None,
            }),
        );
        cache.invalidate("seg.pax");
        assert!(cache.get("seg.pax").is_none());
        assert_eq!(cache.bytes_used(), 0, "all bytes credited back");
    }

    /// TD-CACHE-2 S1 verification: over-budget insertion mixing tiers must
    /// shed meta-only entries before Region-A-bearing ones, ties by recency —
    /// never an arbitrary victim.
    #[test]
    fn invariants_cache_evicts_priority_then_recency() {
        fn region_entry(bytes: usize) -> Arc<SegmentInvariants> {
            Arc::new(SegmentInvariants {
                header_bytes: Vec::new(),
                region_bytes: Some(Arc::from(vec![0u8; bytes].as_slice())),
                footer_bytes: Vec::new(),
                a0_bytes: None,
                rabitq_header_bytes: None,
            })
        }
        fn meta_entry(bytes: usize) -> Arc<SegmentInvariants> {
            Arc::new(SegmentInvariants {
                header_bytes: vec![0u8; bytes],
                region_bytes: None,
                footer_bytes: Vec::new(),
                a0_bytes: None,
                rabitq_header_bytes: None,
            })
        }

        let cache = SegmentInvariantsCache::new(1000);
        cache.put("regionA1".into(), region_entry(400));
        cache.put("regionA2".into(), region_entry(400));
        cache.put("meta1".into(), meta_entry(150));
        assert_eq!(cache.bytes_used(), 950);

        // Touch regionA1 so it is more recent than regionA2.
        assert!(cache.get("regionA1").is_some());

        // Over budget by 100: the meta entry (lower priority) must be the
        // victim — NOT either Region A entry.
        cache.put("meta2".into(), meta_entry(150));
        assert!(cache.get("meta1").is_none(), "meta-only evicted first");
        assert!(cache.get("regionA1").is_some());
        assert!(cache.get("regionA2").is_some());
        assert!(cache.get("meta2").is_some());

        // Re-touch regionA1 (the assertion `get`s above stamped recency too,
        // leaving regionA2 fresher) so regionA2 is deterministically the
        // least-recent Region A entry.
        assert!(cache.get("regionA1").is_some());

        // Force a Region-A eviction: meta2 goes first (priority), then the
        // LEAST-RECENT Region A entry (regionA2 — regionA1 was touched).
        cache.put("regionA3".into(), region_entry(400));
        assert!(
            cache.get("meta2").is_none(),
            "meta evicted before any region"
        );
        // S2b per-class semantics: the least-recent Region A COMPANION is the
        // victim — its control anchor still hits (control-only, region shed).
        assert!(
            cache
                .get("regionA2")
                .is_none_or(|inv| inv.region_bytes.is_none()),
            "least-recent Region A shed on priority tie"
        );
        assert!(
            cache
                .get("regionA1")
                .is_some_and(|inv| inv.region_bytes.is_some()),
            "recency-protected survivor keeps its region"
        );
        assert!(
            cache
                .get("regionA3")
                .is_some_and(|inv| inv.region_bytes.is_some())
        );
        assert!(cache.bytes_used() <= 1000);
    }

    #[test]
    fn pax_write_trace_defaults_off() {
        // TD-COMPACT-1 S1: the write-path phase timer MUST be off in prod (env
        // unset) so the eprintln + Instant bookkeeping never fires on the hot
        // write path. Default-off is the safety property; the on-path is
        // exercised end-to-end by setting PROXIMADB_TRACE_PAX_WRITE at the bench.
        // No test in this binary sets the var, so the default read is deterministic.
        assert!(
            !pax_write_trace(),
            "PROXIMADB_TRACE_PAX_WRITE must be unset by default"
        );
    }

    #[test]
    fn probe_range_coalescing_keeps_only_selected_cell_slices() {
        let selected = vec![
            ProbeCellSlice {
                start: 100,
                end: 200,
                row_start: 0,
            },
            ProbeCellSlice {
                start: 250,
                end: 350,
                row_start: 10,
            },
            ProbeCellSlice {
                start: 900,
                end: 1_000,
                row_start: 20,
            },
        ];
        let policy =
            crate::storage::engines::sst::readers::sst_query_engine::ObjectRangeCoalescePolicy {
                max_gap_bytes: 64,
                max_range_bytes: 512,
            };

        let planned = plan_probe_cell_ranges(&selected, &policy);

        assert_eq!(planned.len(), 2, "the 50-byte gap should merge");
        assert_eq!((planned[0].start, planned[0].end), (100, 350));
        assert_eq!(planned[0].selected, selected[..2]);
        assert_eq!(planned[1].selected, selected[2..]);
        let fetched: u64 = planned.iter().map(|range| range.end - range.start).sum();
        let useful: u64 = planned
            .iter()
            .flat_map(|range| &range.selected)
            .map(|cell| cell.end - cell.start)
            .sum();
        assert_eq!(useful, 300);
        assert_eq!(fetched - useful, 50, "only the merged gap is over-read");

        let estimate = estimate_coalesced_byte_ranges(
            selected.iter().map(|cell| (cell.start, cell.end)),
            &policy,
        );
        assert_eq!(estimate.get_requests, planned.len() as u64);
        assert_eq!(estimate.physical_bytes, fetched);

        let fallback = estimate_with_whole_region_fallback(estimate, 200);
        assert_eq!(fallback.get_requests, 1);
        assert_eq!(fallback.physical_bytes, 200);
        assert_eq!(estimate_with_whole_region_fallback(estimate, 400), estimate);
    }

    #[test]
    fn adaptive_range_policy_obeys_gate_knee_and_explicit_override_precedence() {
        use crate::storage::engines::core::coalesce_strategy::RangePlanEstimate;

        let mib = 1024 * 1024;
        let defaults =
            crate::storage::engines::sst::readers::sst_query_engine::ObjectRangeCoalescePolicy {
                max_gap_bytes: mib,
                max_range_bytes: 4 * mib,
            };
        let measured_1m = |policy: &crate::storage::engines::sst::readers::sst_query_engine::ObjectRangeCoalescePolicy| {
            let (gets, bytes_mib) = match policy.max_range_bytes / mib {
                4 => (10_077, 28_561),
                8 => (8_177, 33_111),
                12 => (7_796, 33_691),
                16 => (7_657, 33_713),
                24 => (7_647, 33_713),
                _ => (10_077, 28_561),
            };
            RangePlanEstimate {
                get_requests: gets,
                physical_bytes: bytes_mib * mib,
            }
        };

        unsafe {
            std::env::remove_var("PROXIMADB_STORAGE_READ_STRATEGY_CHOOSER");
            std::env::remove_var("PROXIMADB_PAX_VECTOR_COALESCE_RANGE");
        }
        let gated_off = resolve_adaptive_range_policy(
            "az://container/segment.pax",
            "region_a",
            "PROXIMADB_PAX_VECTOR_COALESCE_GAP",
            "PROXIMADB_PAX_VECTOR_COALESCE_RANGE",
            defaults,
            measured_1m,
        );
        assert_eq!(gated_off, defaults, "unset gate preserves behavior");

        unsafe { std::env::set_var("PROXIMADB_STORAGE_READ_STRATEGY_CHOOSER", "1") };
        let adaptive = resolve_adaptive_range_policy(
            "az://container/segment.pax",
            "region_a",
            "PROXIMADB_PAX_VECTOR_COALESCE_GAP",
            "PROXIMADB_PAX_VECTOR_COALESCE_RANGE",
            defaults,
            measured_1m,
        );
        assert_eq!(adaptive.max_range_bytes, 16 * mib, "measured GET knee");

        unsafe {
            std::env::set_var(
                "PROXIMADB_PAX_VECTOR_COALESCE_RANGE",
                (24 * mib).to_string(),
            )
        };
        let overridden = resolve_adaptive_range_policy(
            "az://container/segment.pax",
            "region_a",
            "PROXIMADB_PAX_VECTOR_COALESCE_GAP",
            "PROXIMADB_PAX_VECTOR_COALESCE_RANGE",
            defaults,
            measured_1m,
        );
        assert_eq!(overridden.max_range_bytes, 24 * mib);

        unsafe {
            std::env::remove_var("PROXIMADB_STORAGE_READ_STRATEGY_CHOOSER");
            std::env::remove_var("PROXIMADB_PAX_VECTOR_COALESCE_RANGE");
        }
    }
    use crate::storage::engines::core::formats::proximablocks::BlockCompressionConfig;
    use proximadb_records::{EmbeddingCell, EmbeddingValues};

    fn rec(oid: &str, ts: i64, vec: Vec<f32>) -> ProximaRecord {
        let mut r = ProximaRecord {
            oid: oid.into(),
            tenant_id: "t".into(),
            created_at_ns: ts,
            updated_at_ns: ts,
            ..Default::default()
        };
        let dim = vec.len();
        r.embeddings.push(EmbeddingCell {
            modality: "dense".into(),
            dim: dim as u32,
            values: EmbeddingValues::Fp32(vec),
            ..Default::default()
        });
        r
    }

    /// A PAX segment written by `PaxSegmentWriter` is detected as `Pax` and round-trips
    /// back to the same records through the format-routing reader.
    #[test]
    fn pax_segment_detected_and_read_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seg.pax");

        let mut w = PaxSegmentWriter::new(
            &path,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            1, // embedding_count
            None,
        );
        w.add_record(&rec("a", 1_700_000_000_000_000_000, vec![1.0, 2.0, 3.0]))
            .unwrap();
        w.add_record(&rec("b", 1_700_000_000_000_000_001, vec![4.0, 5.0, 6.0]))
            .unwrap();
        w.finish().unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(
            SegmentFormat::detect(&bytes),
            SegmentFormat::Pax,
            "PAX segment must be detected as Pax"
        );

        let records = read_segment_records(&bytes, &[], &[], None).unwrap();
        assert_eq!(records.len(), 2);
        let mut oids: Vec<&str> = records.iter().map(|r| r.oid.as_str()).collect();
        oids.sort_unstable();
        assert_eq!(oids, vec!["a", "b"]);
    }

    /// A legacy `ProximaDataBlock` is detected as `ProximaBlocks` and round-trips back
    /// to the same records — proving the router preserves the existing path.
    #[test]
    fn proxima_block_detected_and_read_back() {
        let records = vec![
            rec("x", 1_700_000_000_000_000_000, vec![1.0, 2.0, 3.0, 4.0]),
            rec("y", 1_700_000_000_000_000_001, vec![5.0, 6.0, 7.0, 8.0]),
        ];
        let bytes = ProximaDataBlock::new(records, BlockCompressionConfig::default())
            .serialize()
            .unwrap();

        assert_eq!(
            SegmentFormat::detect(&bytes),
            SegmentFormat::ProximaBlocks,
            "ProximaDataBlock must be detected as ProximaBlocks"
        );

        let back = read_segment_records(&bytes, &[], &[], None).unwrap();
        let mut oids: Vec<&str> = back.iter().map(|r| r.oid.as_str()).collect();
        oids.sort_unstable();
        assert_eq!(oids, vec!["x", "y"]);
    }

    /// Detection never mis-routes the legacy path: version/compression-marker first
    /// bytes, empty input, and a PBLK prefix without the trailing segment magic all
    /// resolve to `ProximaBlocks`.
    #[test]
    fn detection_defaults_to_legacy_and_guards_false_positives() {
        assert_eq!(SegmentFormat::detect(&[]), SegmentFormat::ProximaBlocks);
        assert_eq!(
            SegmentFormat::detect(&[0x01, 0, 0]),
            SegmentFormat::ProximaBlocks
        );
        assert_eq!(
            SegmentFormat::detect(&[0x05, 0, 0]),
            SegmentFormat::ProximaBlocks
        );
        // Leading PBLK but no trailing PAXSEG01 → not a PAX segment.
        let mut faux = BLOCK_MAGIC.to_vec();
        faux.extend_from_slice(&[0u8; 16]);
        assert_eq!(SegmentFormat::detect(&faux), SegmentFormat::ProximaBlocks);
    }

    /// The default format is the legacy one (serde-absent manifests recover unchanged).
    #[test]
    fn default_is_proxima_blocks() {
        assert_eq!(SegmentFormat::default(), SegmentFormat::ProximaBlocks);
    }

    /// `write_pax_segment` is the write-side inverse: what it writes is detected as PAX
    /// and reads back through `read_segment_records` (the Phase B flush↔read round-trip,
    /// exercised without the flush harness).
    #[test]
    fn write_pax_segment_round_trips_through_reader() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("written.pax");
        let records = vec![
            rec("w1", 1_700_000_000_000_000_000, vec![1.0, 2.0, 3.0]),
            rec("w2", 1_700_000_000_000_000_001, vec![4.0, 5.0, 6.0]),
        ];
        let meta =
            write_pax_segment(&path, &records, "col", 1, VectorQuant::Auto, None, None).unwrap();
        assert!(
            meta.block_count >= 1,
            "segment should have at least one block"
        );

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(SegmentFormat::detect(&bytes), SegmentFormat::Pax);
        let back = read_segment_records(&bytes, &[], &[], None).unwrap();
        let mut oids: Vec<&str> = back.iter().map(|r| r.oid.as_str()).collect();
        oids.sort_unstable();
        assert_eq!(oids, vec!["w1", "w2"]);
    }

    /// M1 (ADR-049): a `VectorQuant::RawF32` PAX segment round-trips to EXACT f32
    /// vectors (recall 1.0) — proving the foundation of the exact PAX scan path
    /// (`search_pax_file_exact`, which materialises records via
    /// `read_segment_records`). RawF32 decodes to the exact input vectors, not an
    /// approximation (unlike RaBitQ/SQ8 reconstruction).
    #[test]
    fn rawf32_pax_segment_round_trips_exact() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rawf32.pax");
        let records: Vec<ProximaRecord> = (0..8)
            .map(|i| {
                rec(
                    &format!("v{i}"),
                    1_700_000_000_000_000_000 + i,
                    (0..16).map(|d| (i as f32 + d as f32) * 0.1).collect(),
                )
            })
            .collect();
        write_pax_segment(&path, &records, "col", 1, VectorQuant::RawF32, None, None).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(SegmentFormat::detect(&bytes), SegmentFormat::Pax);

        // Round-trip: RawF32 decodes to the EXACT input vectors (not an
        // approximation, unlike RaBitQ/SQ8 reconstruction).
        let back = read_segment_records(&bytes, &[], &[], None).unwrap();
        assert_eq!(back.len(), records.len());
        for (got, want) in back.iter().zip(records.iter()) {
            assert_eq!(got.oid, want.oid, "oid must round-trip");
            let got_vec = got
                .embeddings
                .first()
                .expect("RawF32 segment must reconstruct its embedding");
            let want_vec = want.embeddings.first().expect("input embedding");
            assert_eq!(
                got_vec.as_fp32_slice(),
                want_vec.as_fp32_slice(),
                "RawF32 PAX must reconstruct the exact vector for {}",
                want.oid
            );
        }
    }

    /// Mixed-read-safe coexistence (storage-format mandate #8). A store rolling
    /// PAX quantization on will have BOTH legacy `ProximaBlocks` segments (written
    /// before the flip) and new PAX segments side by side. The magic-byte router
    /// must read BOTH back through the single `read_segment_records` entry —
    /// disjoint oids, no mis-route. This is the property that makes the staged
    /// default-on flip safe (default OFF in v0.2, flip gated by the recall ratchet).
    #[test]
    fn mixed_format_legacy_and_pax_segments_coexist_and_read_back() {
        let legacy_records = vec![
            rec(
                "legacy_a",
                1_700_000_000_000_000_000,
                vec![1.0, 2.0, 3.0, 4.0],
            ),
            rec(
                "legacy_b",
                1_700_000_000_000_000_001,
                vec![5.0, 6.0, 7.0, 8.0],
            ),
        ];
        let legacy_bytes = ProximaDataBlock::new(legacy_records, BlockCompressionConfig::default())
            .serialize()
            .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let pax_path = dir.path().join("coexist.pax");
        let pax_records = vec![
            rec(
                "pax_c",
                1_700_000_000_000_000_002,
                vec![9.0, 10.0, 11.0, 12.0],
            ),
            rec(
                "pax_d",
                1_700_000_000_000_000_003,
                vec![13.0, 14.0, 15.0, 16.0],
            ),
        ];
        write_pax_segment(
            &pax_path,
            &pax_records,
            "col",
            1,
            VectorQuant::RaBitQ,
            None,
            None,
        )
        .unwrap();
        let pax_bytes = std::fs::read(&pax_path).unwrap();

        // The two segments are detected as DIFFERENT formats (never mis-routed).
        assert_eq!(
            SegmentFormat::detect(&legacy_bytes),
            SegmentFormat::ProximaBlocks
        );
        assert_eq!(SegmentFormat::detect(&pax_bytes), SegmentFormat::Pax);

        // ...and BOTH read back through the single mixed-format router.
        let legacy_back = read_segment_records(&legacy_bytes, &[], &[], None).unwrap();
        let pax_back = read_segment_records(&pax_bytes, &[], &[], None).unwrap();

        let mut legacy_oids: Vec<&str> = legacy_back.iter().map(|r| r.oid.as_str()).collect();
        legacy_oids.sort_unstable();
        let mut pax_oids: Vec<&str> = pax_back.iter().map(|r| r.oid.as_str()).collect();
        pax_oids.sort_unstable();

        assert_eq!(
            legacy_oids,
            vec!["legacy_a", "legacy_b"],
            "legacy segment oids"
        );
        assert_eq!(
            pax_oids,
            vec!["pax_c", "pax_d"],
            "PAX segment oids (disjoint from legacy)"
        );
    }

    /// TD-USUB-5 / ADR-094 §4 — converting [`read_segment_records`] from an
    /// inline `match` to [`RecordReader`] dispatch must be **behaviour-neutral**
    /// for the formats that already existed.
    ///
    /// The assertion is equality against the canonical decoders called
    /// DIRECTLY — `ProximaDataBlock::deserialize` for the legacy format and
    /// `read_pax_segment_records` for PAX — on whole `ProximaRecord`s, not just
    /// oids. So the test fails if the trait indirection perturbs any field
    /// (embeddings, props, timestamps, tenant), which a shape-only assertion
    /// would miss.
    #[test]
    fn trait_dispatch_is_byte_identical_for_existing_formats() {
        let legacy_records = vec![
            rec(
                "legacy_a",
                1_700_000_000_000_000_000,
                vec![1.0, 2.0, 3.0, 4.0],
            ),
            rec(
                "legacy_b",
                1_700_000_000_000_000_001,
                vec![5.0, 6.0, 7.0, 8.0],
            ),
        ];
        let legacy_bytes = ProximaDataBlock::new(legacy_records, BlockCompressionConfig::default())
            .serialize()
            .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let pax_path = dir.path().join("dispatch.pax");
        let pax_records = vec![
            rec(
                "pax_c",
                1_700_000_000_000_000_002,
                vec![9.0, 10.0, 11.0, 12.0],
            ),
            rec(
                "pax_d",
                1_700_000_000_000_000_003,
                vec![13.0, 14.0, 15.0, 16.0],
            ),
        ];
        write_pax_segment(
            &pax_path,
            &pax_records,
            "col",
            1,
            VectorQuant::RaBitQ,
            None,
            None,
        )
        .unwrap();
        let pax_bytes = std::fs::read(&pax_path).unwrap();

        // Legacy: router output == `ProximaDataBlock::deserialize` output.
        let via_router = read_segment_records(&legacy_bytes, &[], &[], Some("t1")).unwrap();
        let direct = ProximaDataBlock::deserialize(&legacy_bytes, None)
            .unwrap()
            .records;
        assert_eq!(via_router, direct, "legacy decode changed under dispatch");

        // PAX: router output == `read_pax_segment_records` output, with the
        // catalog-derived context threaded through unchanged.
        let via_router = read_segment_records(&pax_bytes, &[], &[], Some("t1")).unwrap();
        let direct = proximadb_storage_common::pax_block::read_pax_segment_records(
            &pax_bytes,
            &[],
            &[],
            Some("t1"),
        )
        .unwrap();
        assert_eq!(via_router, direct, "PAX decode changed under dispatch");

        // The dispatch table agrees with detection for every format — the
        // invariant that keeps "detected as X" and "decoded as X" from drifting.
        for bytes in [&legacy_bytes[..], &pax_bytes[..]] {
            let format = SegmentFormat::detect(bytes);
            assert_eq!(record_reader_for(format).format(), format);
        }
    }

    /// TD-USUB-5 — a Parquet object reaching the router is now DECODED rather
    /// than handed to the legacy decoder that cannot read it.
    ///
    /// The second half is the part that matters for mandate #1: even without the
    /// Parquet arm the old code did not return wrong rows here, it returned an
    /// error — but it was an error about the wrong format. Naming the format
    /// turns a mis-route into a decode.
    #[test]
    fn parquet_segment_routes_to_the_parquet_reader() {
        use arrow_array::{RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};

        let schema = Arc::new(Schema::new(vec![Field::new("city", DataType::Utf8, true)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(vec!["pune", "austin"]))],
        )
        .unwrap();
        let bytes = proximadb_storage_common::proxima_parquet::record_batches_to_parquet_bytes(
            &[batch],
            schema,
            None,
        )
        .unwrap();

        assert_eq!(SegmentFormat::detect(&bytes), SegmentFormat::Parquet);
        assert_eq!(
            record_reader_for(SegmentFormat::Parquet).format(),
            SegmentFormat::Parquet
        );

        let records = read_segment_records(&bytes, &[], &[], Some("tenant-9")).unwrap();
        assert_eq!(records.len(), 2);
        assert!(
            records.iter().all(|r| r.tenant_id == "tenant-9"),
            "tenant context must be stamped on the Parquet arm too"
        );

        // Non-vacuity: the legacy decoder genuinely cannot read these bytes, so
        // the new arm is doing real work rather than shadowing a path that
        // already happened to succeed.
        assert!(
            ProximaDataBlock::deserialize(&bytes, None).is_err(),
            "legacy decoder must reject Parquet bytes — otherwise this test proves nothing"
        );
    }

    /// TD-156 / ADR-026: block geometry is the fragmentation lever. A larger
    /// `target_block` packs rows into fewer blocks — so the per-block ranged-GET
    /// count on the object-store read path drops (the +29,900% GET fragmentation
    /// measured by the footer-cache economics harness shrinks as block count
    /// falls). Mixed-read-safe: block size is a per-segment write choice, opaque
    /// to the magic-detected read router.
    #[test]
    fn larger_target_block_yields_fewer_blocks() {
        // TD-PAXRG-1: post-flip the default write is the row-group layout,
        // whose 256 KiB floor clamps sub-floor targets — this test pins the
        // LEGACY geometry lever, so it pins the kill-switch (nextest isolates
        // the process).
        unsafe { std::env::set_var("PROXIMADB_PAX_WRITE_RG_LAYOUT", "0") };
        let records: Vec<ProximaRecord> = (0..512)
            .map(|i| rec(&format!("r{i}"), 1000 + i as i64, vec![i as f32 * 0.1; 64]))
            .collect();
        let blocks_for = |target: Option<usize>| -> usize {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("geo.pax");
            write_pax_segment(&path, &records, "col", 1, VectorQuant::RaBitQ, target, None)
                .unwrap()
                .block_count as usize
        };
        let small = blocks_for(Some(8 * 1024));
        let large = blocks_for(Some(256 * 1024));
        eprintln!(
            "[geometry] 512x64 vectors: 8KiB target -> {small} blocks, 256KiB target -> {large} blocks"
        );
        assert!(
            large < small,
            "larger target_block must yield fewer blocks: {large} >= {small}"
        );
        assert!(large >= 1, "at least one block");
    }

    // ── ADR-062 / TD-RDSTRAT-6 coalesced-RaBitQ layout ───────────────────────

    /// Opt into the coalesced-RaBitQ layout for one test. SAFETY: nextest runs
    /// process-per-test, so the env mutation is isolated to this test process.
    fn enable_coalesced_rabitq() {
        unsafe {
            std::env::set_var("PROXIMADB_PAX_COALESCED_RABITQ", "1");
        }
    }

    fn disable_coalesced_rabitq() {
        unsafe {
            std::env::set_var("PROXIMADB_PAX_COALESCED_RABITQ", "0");
        }
    }

    #[test]
    fn coalesced_requested_f32_tier_is_emitted_and_materializes_exact() {
        enable_coalesced_rabitq();
        // Post-flip the default write is the row-group layout, where the exact
        // tier HOISTS to Region C (covered by
        // `v4_f32_tier_writes_region_c_not_block_stripes` + the parity oracle).
        // THIS test pins the legacy in-block stripe contract, so it pins the
        // legacy framing explicitly (env-independent).
        unsafe { std::env::set_var("PROXIMADB_PAX_WRITE_RG_LAYOUT", "0") };
        use proximadb_block_format::col_id;
        use proximadb_storage_common::segment_layout::SegmentFooterIndex;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("coalesced-exact.pax");
        let records = vec![
            rec("a", 1, vec![-7.25, 0.123_456_7, 3.141_592_7]),
            rec("b", 2, vec![2.75, 8.765_432, -0.333_333_34]),
            rec("c", 3, vec![11.125, -4.567_891, 9.999_991]),
        ];

        write_pax_segment_with_f32_tier(
            &path,
            &records,
            "col",
            1,
            VectorQuant::RaBitQ,
            true,
            Some(1024),
            None,
        )
        .unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let footer = SegmentFooterIndex::locate_in_segment(&bytes)
            .unwrap()
            .expect("coalesced footer");
        assert!(footer.has_f32_tier, "footer must declare the exact tier");
        assert!(
            pax_segment_has_exact_vector_authority(&bytes),
            "coalesced footer capability must admit its emitted exact tier"
        );

        let mut scanner =
            PaxSegmentScanner::from_bytes(bytes.clone(), ScanPredicate::default()).unwrap();
        while let Some(block) = scanner.next_block() {
            assert!(
                block.vector_params().get(col_id::F32_TIER_BASE).is_some(),
                "every coalesced block must emit the requested exact tier"
            );
            assert!(
                block.has_exact_vector_authority(),
                "an exact-only coalesced block must prove non-vacuous authority"
            );
        }

        // The canonical materializer must prefer the block-local exact tier over
        // Region B's SQ8 rerank values. Merely checking that a vector exists would
        // allow Exact search to rank a lossy reconstruction.
        let materialized = read_segment_records(&bytes, &[], &[], None).unwrap();
        assert_eq!(materialized.len(), records.len());
        let expected: std::collections::HashMap<_, _> = records
            .iter()
            .map(|record| (record.oid.as_str(), &record.embeddings))
            .collect();
        for got in &materialized {
            let want = expected.get(got.oid.as_str()).expect("known record oid");
            assert_eq!(
                got.embeddings.first().map(|cell| cell.as_fp32_cow()),
                want.first().map(|cell| cell.as_fp32_cow()),
                "exact tier must round-trip bitwise for {}",
                got.oid
            );
        }
    }

    #[test]
    fn mixed_exact_and_lossy_inputs_do_not_claim_exact_output() {
        disable_coalesced_rabitq();
        let dir = tempfile::tempdir().unwrap();
        let exact = dir.path().join("exact.pax");
        let lossy = dir.path().join("lossy.pax");
        let records = vec![
            rec("a", 1, vec![-1.0, 0.123_456_7, 7.0]),
            rec("b", 2, vec![3.0, 4.765_432, -2.0]),
        ];

        write_pax_segment_with_f32_tier(
            &exact,
            &records,
            "col",
            1,
            VectorQuant::RaBitQ,
            true,
            None,
            None,
        )
        .unwrap();
        write_pax_segment_with_f32_tier(
            &lossy,
            &records,
            "col",
            1,
            VectorQuant::RaBitQ,
            false,
            None,
            None,
        )
        .unwrap();

        assert!(pax_inputs_have_f32_tier(std::slice::from_ref(&exact)));
        assert!(pax_segment_has_exact_vector_authority(
            &std::fs::read(&exact).unwrap()
        ));
        assert!(
            !pax_inputs_have_f32_tier(&[exact.clone(), lossy.clone()]),
            "one exact input must not make a mixed compacted output exact"
        );
        assert!(
            !pax_segment_has_exact_vector_authority(&std::fs::read(&lossy).unwrap()),
            "lossy PAX must not satisfy an exact-search contract"
        );
        assert!(
            !pax_inputs_have_f32_tier(&[exact, dir.path().join("unknown.arrow")]),
            "an unrelated physical format must not be assumed exact"
        );
    }

    #[test]
    fn coalesced_lossy_segment_does_not_claim_exact_authority_vacuously() {
        enable_coalesced_rabitq();
        let dir = tempfile::tempdir().unwrap();
        let lossy = dir.path().join("coalesced-lossy.pax");
        let records = vec![
            rec("a", 1, vec![-1.0, 0.123_456_7, 7.0]),
            rec("b", 2, vec![3.0, 4.765_432, -2.0]),
        ];
        write_pax_segment_with_f32_tier(
            &lossy,
            &records,
            "col",
            1,
            VectorQuant::RaBitQ,
            false,
            None,
            None,
        )
        .unwrap();

        let bytes = std::fs::read(&lossy).unwrap();
        assert!(
            !pax_segment_has_exact_vector_authority(&bytes),
            "hoisted vectors with no f32 tier are lossy, not vacuously exact"
        );
        let mut scanner = PaxSegmentScanner::from_bytes(bytes, ScanPredicate::default()).unwrap();
        while let Some(block) = scanner.next_block() {
            assert!(
                !block.has_exact_vector_authority(),
                "a coalesced block with neither base nor exact stripes must not pass vacuously"
            );
        }
        assert!(
            !pax_inputs_have_f32_tier(&[lossy]),
            "compaction must not upgrade a lossy coalesced input to exact"
        );
    }

    #[test]
    fn coalesced_rank_scores_cross_the_public_metric_boundary_once() {
        assert_eq!(
            canonical_score_from_rank_score(RankMetric::L2, 25.0),
            5.0,
            "squared L2 rank score must become canonical Euclidean distance"
        );
        assert_eq!(
            canonical_score_from_rank_score(RankMetric::DotProduct, -6.0),
            6.0,
            "negated-dot rank score must become public positive-dot similarity"
        );
        assert_eq!(
            canonical_score_from_rank_score(RankMetric::Cosine, 0.25),
            0.25,
            "cosine distance is already canonical"
        );
    }

    #[test]
    fn fused_sq8_rerank_matches_decode_then_score_for_every_metric() {
        let params = proximadb_codec::Sq8Params {
            scale: 0.0625,
            offset: -2.0,
            vmin: -2.0,
            vmax: 13.9375,
        };
        let codes = (0..128)
            .map(|i| ((i * 47 + 11) % 256) as u8)
            .collect::<Vec<_>>();
        let query = (0..128)
            .map(|i| ((i * 31 + 3) as f32 * 0.013).cos() * 3.0)
            .collect::<Vec<_>>();
        let decoded = proximadb_codec::functions::sq8::decode(&codes, &params);
        let query_norm_squared = query.iter().map(|value| value * value).sum::<f32>();
        for metric in [RankMetric::L2, RankMetric::Cosine, RankMetric::DotProduct] {
            let expected = match metric {
                RankMetric::L2 => query
                    .iter()
                    .zip(&decoded)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>(),
                RankMetric::Cosine => {
                    let dot = query.iter().zip(&decoded).map(|(a, b)| a * b).sum::<f32>();
                    let decoded_norm = decoded.iter().map(|value| value * value).sum::<f32>();
                    1.0 - dot / ((query_norm_squared * decoded_norm).sqrt() + 1e-12)
                }
                RankMetric::DotProduct => {
                    -query.iter().zip(&decoded).map(|(a, b)| a * b).sum::<f32>()
                }
            };
            let actual = rerank_sq8_distance(metric, &query, &codes, &params, query_norm_squared)
                .expect("matching SQ8/query dimensions");
            let tolerance = expected.abs().max(1.0) * 1e-5;
            assert!(
                (actual - expected).abs() <= tolerance,
                "{metric:?}: fused={actual} decoded={expected}"
            );
        }
        assert!(rerank_sq8_distance(RankMetric::L2, &query[..127], &codes, &params, 0.0).is_none());
    }

    /// A coalesced-RaBitQ segment (opt-in) is detected as Pax, carries a non-zero
    /// RaBitQ region, and round-trips back to the same records through the
    /// mixed-format reader. The RaBitQ binary tier lives in the header region;
    /// the block's EMBED_BASE is SQ8 (the rerank data), which the reader
    /// reconstructs via the unchanged SQ8 decode path.
    #[test]
    fn coalesced_rabitq_segment_round_trips_and_detects_as_pax() {
        enable_coalesced_rabitq();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("coalesced.pax");
        let records: Vec<ProximaRecord> = (0..64)
            .map(|i| {
                rec(
                    &format!("v{i}"),
                    1_700_000_000_000_000_000 + i,
                    (0..48).map(|d| (i as f32 + d as f32) * 0.1).collect(),
                )
            })
            .collect();
        let meta =
            write_pax_segment(&path, &records, "col", 1, VectorQuant::RaBitQ, None, None).unwrap();
        // Coalesced layout: a non-empty RaBitQ region pinned right after the
        // 40 B header-prefix.
        assert!(
            meta.rabitq_len > 0,
            "coalesced segment must carry a RaBitQ region"
        );
        assert_eq!(
            meta.rabitq_off,
            proximadb_storage_common::segment_layout::SEG_HEADER_PREFIX_LEN as u64
        );

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(SegmentFormat::detect(&bytes), SegmentFormat::Pax);
        assert!(
            is_coalesced_segment(&bytes),
            "opt-in write must produce the coalesced presence-field"
        );

        // Region B (ADR-065): vectors live in the coalesced SQ8 region, so
        // read_segment_records fails closed — verify the OID round-trip via the
        // scanner's block read (Region D row data) instead.
        let mut scan =
            PaxSegmentScanner::from_bytes(bytes.clone(), ScanPredicate::default()).unwrap();
        let back = scan.read_records(&[], &[], None).unwrap();
        assert_eq!(back.len(), records.len());
        let got: std::collections::HashSet<String> = back.iter().map(|r| r.oid.clone()).collect();
        let want: std::collections::HashSet<String> = (0..64).map(|i| format!("v{i}")).collect();
        assert_eq!(got, want, "round-tripped oids must match the input set");
    }

    /// The presence-field cleanly distinguishes a coalesced segment from a legacy
    /// one (mixed-read): opt-out writes carry no RaBitQ region and are NOT
    /// coalesced; opt-in writes are. Both detect as Pax.
    #[test]
    fn coalesced_presence_field_distinguishes_legacy_from_coalesced() {
        let mk = |coalesced: bool, name: &str| -> (Vec<u8>, u64) {
            if coalesced {
                enable_coalesced_rabitq();
            } else {
                disable_coalesced_rabitq();
            }
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join(name);
            let records: Vec<ProximaRecord> = (0..16)
                .map(|i| {
                    rec(
                        &format!("v{i}"),
                        1_700_000_000_000_000_000 + i,
                        vec![i as f32 * 0.1; 32],
                    )
                })
                .collect();
            let meta =
                write_pax_segment(&path, &records, "col", 1, VectorQuant::RaBitQ, None, None)
                    .unwrap();
            (std::fs::read(&path).unwrap(), meta.rabitq_len)
        };

        let (legacy_bytes, legacy_len) = mk(false, "legacy.pax");
        assert_eq!(legacy_len, 0, "legacy write has no coalesced region");
        assert!(
            !is_coalesced_segment(&legacy_bytes),
            "legacy write must not carry the coalesced presence-field"
        );
        assert_eq!(SegmentFormat::detect(&legacy_bytes), SegmentFormat::Pax);

        let (coalesced_bytes, coalesced_len) = mk(true, "coalesced.pax");
        assert!(coalesced_len > 0, "opt-in write carries a RaBitQ region");
        assert!(is_coalesced_segment(&coalesced_bytes));
        assert_eq!(SegmentFormat::detect(&coalesced_bytes), SegmentFormat::Pax);
    }

    /// The coalesced region + footer-index of a real written segment parse and the
    /// region ranks: the RaBitQ region decodes (Slice 2a codec over writer output)
    /// and the footer's block table + rabitq mirror match the SegmentMeta. This
    /// ties the writer (2d), scanner footer parse (2e), and region codec (2a).
    #[test]
    fn coalesced_region_and_footer_parse_from_written_segment() {
        enable_coalesced_rabitq();
        use proximadb_block_format::{RaBitQRegion, col_id};
        use proximadb_storage_common::segment_layout::SegmentFooterIndex;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("coalesced.pax");
        let records: Vec<ProximaRecord> = (0..128)
            .map(|i| {
                rec(
                    &format!("v{i}"),
                    1_700_000_000_000_000_000 + i,
                    (0..64).map(|d| (i as f32 + d as f32) * 0.1).collect(),
                )
            })
            .collect();
        let meta = write_pax_segment(
            &path,
            &records,
            "col",
            1,
            VectorQuant::RaBitQ,
            Some(8 * 1024),
            None,
        )
        .unwrap();
        assert!(meta.block_count >= 1);

        let bytes = std::fs::read(&path).unwrap();

        // Footer-index: block table + rabitq mirror match the SegmentMeta.
        let footer = SegmentFooterIndex::locate_in_segment(&bytes)
            .unwrap()
            .expect("coalesced footer located");
        assert_eq!(footer.blocks.len(), meta.block_count as usize);
        assert_eq!(footer.rabitq_off, meta.rabitq_off);
        assert_eq!(footer.rabitq_len, meta.rabitq_len);
        assert_eq!(footer.row_count, meta.row_count);
        assert_eq!(footer.embed_quant_tag, 1, "block tier-1 is SQ8");

        // RaBitQ region: decodes + ranks nearest-first; n_rows == record count.
        let off = meta.rabitq_off as usize;
        let region = &bytes[off..off + meta.rabitq_len as usize];
        let parsed = RaBitQRegion::from_bytes(region).unwrap();
        assert_eq!(parsed.n_rows(), records.len());

        // ADR-065 Region B: the SQ8 rerank tier is hoisted out of the block into a
        // coalesced region — the block carries NO EMBED_BASE vector stripe (pure
        // row data, Region D). Region B parses and holds the segment vectors.
        let mut scanner =
            PaxSegmentScanner::from_bytes(bytes.clone(), ScanPredicate::default()).unwrap();
        let block = scanner.next_block().expect("segment has a block");
        let embed = block.vector_params().get(col_id::EMBED_BASE);
        assert!(
            embed.is_none(),
            "coalesced block must carry NO EMBED_BASE (SQ8 hoisted to Region B)"
        );
        assert!(
            meta.sq8_len > 0,
            "coalesced segment carries a Region B SQ8 region"
        );
        assert_eq!(footer.sq8_off, meta.sq8_off);
        assert_eq!(footer.sq8_len, meta.sq8_len);
        let sq8_off = meta.sq8_off as usize;
        let sq8_region = &bytes[sq8_off..sq8_off + meta.sq8_len as usize];
        let sq8_hdr = proximadb_block_format::coalesced_sq8::CoalescedSq8Header::parse(sq8_region)
            .expect("Region B header parses");
        assert_eq!(sq8_hdr.n_rows as usize, records.len());
        assert_eq!(sq8_hdr.dim as usize, 64);

        // Rank a query that is one of the records; the top-1 survivor is present.
        let query = records[10]
            .embeddings
            .first()
            .unwrap()
            .as_fp32_slice()
            .to_vec();
        let ranked = parsed.rank(&query, RankMetric::L2, records.len());
        assert!(!ranked.is_empty());
        assert!(parsed.code(ranked[0]).is_some());
    }

    /// ADR-062 scan-then-rerank over a coalesced segment via the filesystem:
    /// recall@k holds vs brute-force ground truth, and the path returns the
    /// nearest neighbours. (The GET-count win is quantified by the SIFT ratchet;
    /// this proves correctness of the new read path end-to-end.)
    #[tokio::test]
    async fn coalesced_scan_then_rerank_recall_vs_bruteforce() {
        enable_coalesced_rabitq();
        use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

        const DIM: usize = 64;
        const N: usize = 400;
        const K: usize = 10;

        let corpus: Vec<Vec<f32>> = (0..N)
            .map(|i| {
                (0..DIM)
                    .map(|d| (((i * 131 + d * 17) % 251) as f32) * 0.01)
                    .collect()
            })
            .collect();
        let records: Vec<ProximaRecord> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| rec(&format!("r{i}"), 1000 + i as i64, v.clone()))
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seg.pax");
        // Small target block ⇒ multi-block segment (exercises survivor→block mapping).
        write_pax_segment(
            &path,
            &records,
            "col",
            1,
            VectorQuant::RaBitQ,
            Some(16 * 1024),
            None,
        )
        .unwrap();
        let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
            .await
            .unwrap();

        let query = corpus[137].clone();
        let hits = rabitq_search_segment_coalesced(
            &fs,
            path.to_str().unwrap(),
            &query,
            K,
            RankMetric::L2,
            None,
            None,
        )
        .await
        .unwrap()
        .expect("coalesced scan-then-rerank must return hits");
        assert!(!hits.is_empty(), "cascade must return top-k");

        // Brute-force ground-truth top-k over the f32 corpus.
        let l2 =
            |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>();
        let mut idx: Vec<usize> = (0..N).collect();
        idx.sort_by(|&a, &b| {
            l2(&corpus[a], &query)
                .partial_cmp(&l2(&corpus[b], &query))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let truth: std::collections::HashSet<String> =
            idx.iter().take(K).map(|i| format!("r{i}")).collect();
        let got: std::collections::HashSet<String> = hits.iter().map(|h| h.oid.clone()).collect();
        let recall = truth.iter().filter(|o| got.contains(*o)).count() as f32 / K as f32;
        assert!(
            recall >= 0.90,
            "coalesced scan-then-rerank recall@{K} = {recall:.2} (N={N})"
        );

        // The nearest neighbour (r137 is the query itself) must be ranked first.
        assert_eq!(
            hits[0].oid, "r137",
            "the query vector itself must be the top hit"
        );
    }

    /// ADR-089 P1 test helper: a record carrying a `partition` prop (`p{i%8}`),
    /// mirroring the context-corridor benchmark's filter shape.
    fn rec_with_partition(i: usize, vec: Vec<f32>) -> ProximaRecord {
        let mut r = rec(&format!("r{i}"), 1000 + i as i64, vec);
        r.props.insert(
            "partition".to_string(),
            proximadb_records::ProximaTreeNode::Value(proximadb_data_model::ProximaValue::String(
                format!("p{}", i % 8),
            )),
        );
        r
    }

    fn partition_filter(value: &str) -> proximadb_filter_expression::FilterExpression {
        proximadb_filter_expression::FilterExpression::Comparison {
            field: "partition".to_string(),
            operator: proximadb_filter_expression::ComparisonOperator::Equals,
            value: serde_json::Value::String(value.to_string()),
        }
    }

    /// ADR-089 / TD-FPRUNE-1 P1: the Stage-F row allow-set is row-accurate —
    /// exactly the segment rows whose props match the predicate, in global
    /// (cluster-ordered) row-ordinal space, verified by an independent decode.
    #[tokio::test]
    async fn stage_f_row_allow_set_is_exact() {
        enable_coalesced_rabitq();
        use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
        use proximadb_block_format::{PaxBlockReader, record::FlatRow};
        use proximadb_storage_common::segment_layout::{SegmentFooterIndex, SegmentHeaderPrefix};

        const DIM: usize = 32;
        const N: usize = 300;
        let records: Vec<ProximaRecord> = (0..N)
            .map(|i| {
                let v: Vec<f32> = (0..DIM)
                    .map(|d| (((i * 131 + d * 17) % 251) as f32) * 0.01)
                    .collect();
                rec_with_partition(i, v)
            })
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seg.pax");
        // Small blocks ⇒ multi-block segment (exercises cross-block ordinals).
        write_pax_segment(
            &path,
            &records,
            "col",
            1,
            VectorQuant::RaBitQ,
            Some(16 * 1024),
            None,
        )
        .unwrap();
        let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
            .await
            .unwrap();
        let p = path.to_str().unwrap();

        let filter = partition_filter("p3");
        let (allow, stats) = pax_filtered_row_allow(&fs, p, &filter)
            .await
            .unwrap()
            .expect("coalesced segment must produce a stage-F allow-set");

        // Independent verification: decode every D-block row in file order and
        // recompute the expected ordinal set from each row's props.
        let bytes = std::fs::read(&path).unwrap();
        let header = SegmentHeaderPrefix::parse(&bytes).unwrap();
        let footer_off = header.footer_off as usize;
        let footer =
            SegmentFooterIndex::parse(&bytes[footer_off..footer_off + header.footer_len as usize])
                .unwrap();
        let mut expected = std::collections::BTreeSet::new();
        let mut g = 0usize;
        for b in &footer.blocks {
            let reader = PaxBlockReader::open(
                &bytes[b.offset as usize..(b.offset + b.size as u64) as usize],
            )
            .unwrap();
            for flat in FlatRow::from_block_reader(&reader).unwrap() {
                let oid_index: usize = flat.oid.trim_start_matches('r').parse().unwrap();
                if oid_index % 8 == 3 {
                    expected.insert(g);
                }
                g += 1;
            }
        }
        assert_eq!(g, N, "independent decode covers every row");
        let got: std::collections::BTreeSet<usize> =
            (0..N).filter(|&row| allow.contains(row)).collect();
        assert_eq!(got, expected, "allow-set == filter-matching rows, exactly");
        assert_eq!(allow.len(), (0..N).filter(|i| i % 8 == 3).count());
        assert_eq!(stats.rows_matched, allow.len());
        assert_eq!(stats.blocks_total, footer.blocks.len());

        // A predicate matching nothing yields an empty (but Some) allow-set.
        let (empty_allow, _) = pax_filtered_row_allow(&fs, p, &partition_filter("p999"))
            .await
            .unwrap()
            .expect("coalesced segment");
        assert!(empty_allow.is_empty());
    }

    /// ADR-089 / TD-FPRUNE-1 P2: with footer stats armed, a filtered query drops
    /// provably-empty blocks from the FETCH plan (zero body GET). Exercised on a
    /// CANONICAL column (`created_at_ns`) — the only fields Stage-F's
    /// `pax_field_to_col` resolves today. Footer-pruning of USER TAGS
    /// (partition/lang) is dark until the P-Shred-arming slice shreds filterable
    /// props into columns AND `pax_field_to_col` resolves them (see TD note): the
    /// footer-prune (`evaluate_block`/`pax_field_to_col` = canonical columns) and
    /// the row-match (`evaluate_filter_proxima` = props) currently read disjoint
    /// field namespaces. Soundness (never drops a matching block) is proven by
    /// the block-format `footer_prune_matches_body_prune_for_same_metas` unit.
    #[tokio::test]
    async fn stage_f_footer_stats_prune_blocks_from_fetch() {
        enable_coalesced_rabitq();
        // Arm P2 footer-stats population at write. Process-global env is safe
        // under nextest's process-per-test isolation (the repo test standard).
        unsafe {
            std::env::set_var("PROXIMADB_PAX_FOOTER_STATS", "1");
        }
        use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

        const DIM: usize = 32;
        const N: usize = 1200;
        // Two groups DISJOINT in both vector space (near 0 vs near 10) AND
        // created_at (1000..1599 vs 2000..2599): the writer's centroid clustering
        // keeps blocks group-homogeneous, so their footer created_at [min,max]
        // ranges don't overlap — a `created_at >= 2000` query provably skips the
        // low-group blocks from the fetch. The small target block + larger N
        // guarantee MULTIPLE blocks (at the 16 KiB default this corpus packed
        // into a single spanning block, which no footer stat can prune).
        let records: Vec<ProximaRecord> = (0..N)
            .map(|i| {
                let low = i < N / 2;
                // ORTHOGONAL anchor dims (+x vs +y): direction-aware orderings
                // (sign-code / Morton / PCA all agree) separate the groups, so
                // blocks stay group-homogeneous and the footer created_at
                // [min,max] ranges are disjoint. (Equal-direction groups
                // interleave under every ordering, which no stat can prune.)
                let anchor = if low { 0 } else { 1 };
                let v: Vec<f32> = (0..DIM)
                    .map(|d| {
                        if d == anchor {
                            50.0
                        } else {
                            (((i * 131 + d * 17) % 17) as f32) * 0.001
                        }
                    })
                    .collect();
                let ts = if low {
                    1000 + i as i64
                } else {
                    2000 + i as i64
                };
                rec(&format!("r{i}"), ts, v)
            })
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seg.pax");
        write_pax_segment(
            &path,
            &records,
            "col",
            1,
            VectorQuant::RaBitQ,
            Some(4 * 1024),
            None,
        )
        .unwrap();
        let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
            .await
            .unwrap();
        let p = path.to_str().unwrap();

        // created_at_ns >= 2000 → provably excludes every low-group block.
        let filter = proximadb_filter_expression::FilterExpression::Comparison {
            field: "created_at_ns".to_string(),
            operator: proximadb_filter_expression::ComparisonOperator::GreaterThanOrEqual,
            value: serde_json::json!(2000),
        };
        let (_allow, stats) = pax_filtered_row_allow(&fs, p, &filter)
            .await
            .unwrap()
            .expect("coalesced segment must produce a stage-F allow-set");

        // The zero-GET win: ≥1 block pruned from the FETCH plan by footer stats,
        // and never all of them (the high-group blocks survive).
        assert!(
            stats.blocks_pruned_footer > 0,
            "footer stats must prune ≥1 block from the fetch (got {} of {} blocks)",
            stats.blocks_pruned_footer,
            stats.blocks_total,
        );
        assert!(
            stats.blocks_pruned_footer < stats.blocks_total,
            "must not prune every block ({} of {})",
            stats.blocks_pruned_footer,
            stats.blocks_total,
        );
        unsafe {
            std::env::remove_var("PROXIMADB_PAX_FOOTER_STATS");
        }
    }

    /// ADR-089 / TD-FPRUNE-1 P2 ENGAGEMENT: a USER-TAG filter (`partition`)
    /// footer-prunes blocks end-to-end — the segment self-describes its shredded
    /// columns (footer `shred_field_map`), so the reader resolves the tag name →
    /// col-id with NO catalog lookup. This is the piece that makes P2 move the
    /// benchmark (partition/lang filters), and the allow-set stays EXACT
    /// (row-match via props is unchanged).
    #[tokio::test]
    async fn stage_f_user_tag_footer_prune_via_self_describing_segment() {
        enable_coalesced_rabitq();
        unsafe {
            std::env::set_var("PROXIMADB_PAX_FOOTER_STATS", "1");
        }
        use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
        use proximadb_block_format::{PaxBlockReader, record::FlatRow};
        use proximadb_storage_common::segment_layout::{SegmentFooterIndex, SegmentHeaderPrefix};

        const DIM: usize = 32;
        const N: usize = 400;
        // Two groups disjoint in vector space AND partition: pA (near 0) / pB
        // (near 10). Clustering keeps blocks partition-homogeneous, so a "pA"
        // query provably skips the pB-only blocks — once `partition` resolves to
        // its shredded col-id via the footer field-map.
        let records: Vec<ProximaRecord> = (0..N)
            .map(|i| {
                let low = i < N / 2;
                let base = if low { 0.0 } else { 10.0 };
                let v: Vec<f32> = (0..DIM)
                    .map(|d| base + (((i * 131 + d * 17) % 17) as f32) * 0.001)
                    .collect();
                let mut r = rec(&format!("r{i}"), 1000 + i as i64, v);
                r.props.insert(
                    "partition".to_string(),
                    proximadb_records::ProximaTreeNode::Value(
                        proximadb_data_model::ProximaValue::String(
                            if low { "pA" } else { "pB" }.to_string(),
                        ),
                    ),
                );
                r
            })
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seg.pax");
        // Shred "partition" into user col-id 100 (the P-Shred spec the production
        // flush path builds from the schema). write_pax_segment_ordered is the
        // module-internal writer that threads the spec.
        let shred = vec![("partition".to_string(), 100i32)];
        let cluster = crate::storage::engines::sst::block_cluster::block_cluster_enabled();
        let plan = if cluster {
            crate::storage::engines::sst::block_cluster::cluster_order_sq8_morton(&records, 0).map(
                |order| crate::storage::engines::sst::block_cluster::ClusterPlan {
                    order,
                    runs: Vec::new(),
                },
            )
        } else {
            None
        };
        write_pax_segment_ordered(
            &path,
            &records,
            "col",
            1,
            VectorQuant::RaBitQ,
            VectorQuant::Sq8,
            false,
            Some(16 * 1024),
            false,
            cluster,
            plan,
            None,
            None,
            &shred,
        )
        .unwrap();
        let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
            .await
            .unwrap();
        let p = path.to_str().unwrap();

        // Precondition: the segment self-describes "partition" → col 100.
        let bytes = std::fs::read(&path).unwrap();
        let header = SegmentHeaderPrefix::parse(&bytes).unwrap();
        let foff = header.footer_off as usize;
        let footer =
            SegmentFooterIndex::parse(&bytes[foff..foff + header.footer_len as usize]).unwrap();
        assert!(
            footer
                .shred_field_map
                .iter()
                .any(|(n, c)| n == "partition" && *c == 100),
            "footer must self-describe the shredded partition column, got {:?}",
            footer.shred_field_map
        );

        let (allow, stats) = pax_filtered_row_allow(&fs, p, &partition_filter("pA"))
            .await
            .unwrap()
            .expect("coalesced segment must produce a stage-F allow-set");

        // Correctness (robust to clustering reorder): allow-set == exactly the
        // pA rows, verified by independent decode of each row's partition prop.
        let mut expected = std::collections::BTreeSet::new();
        let mut g = 0usize;
        for b in &footer.blocks {
            let reader = PaxBlockReader::open(
                &bytes[b.offset as usize..(b.offset + b.size as u64) as usize],
            )
            .unwrap();
            for flat in FlatRow::from_block_reader(&reader).unwrap() {
                if flat.oid.trim_start_matches('r').parse::<usize>().unwrap() < N / 2 {
                    expected.insert(g);
                }
                g += 1;
            }
        }
        let got: std::collections::BTreeSet<usize> =
            (0..N).filter(|&row| allow.contains(row)).collect();
        assert_eq!(
            got, expected,
            "allow-set == pA rows exactly (user-tag prune sound)"
        );
        assert_eq!(allow.len(), N / 2);

        // The zero-GET win on a USER TAG: ≥1 (but not all) block pruned from the
        // fetch — proving the footer field-map resolved "partition".
        assert!(
            stats.blocks_pruned_footer > 0 && stats.blocks_pruned_footer < stats.blocks_total,
            "user-tag footer prune must engage (got {} of {} blocks)",
            stats.blocks_pruned_footer,
            stats.blocks_total,
        );
        unsafe {
            std::env::remove_var("PROXIMADB_PAX_FOOTER_STATS");
        }
    }

    /// ADR-089 / TD-FPRUNE-1 P1: the row-restricted cascade returns ONLY
    /// predicate-matching hits, at recall parity with a filtered brute-force
    /// ground truth (same bar as the unfiltered cascade's recall test).
    #[tokio::test]
    async fn filtered_cascade_matches_filtered_bruteforce() {
        enable_coalesced_rabitq();
        use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

        const DIM: usize = 64;
        const N: usize = 400;
        const K: usize = 10;
        let corpus: Vec<Vec<f32>> = (0..N)
            .map(|i| {
                (0..DIM)
                    .map(|d| (((i * 131 + d * 17) % 251) as f32) * 0.01)
                    .collect()
            })
            .collect();
        let records: Vec<ProximaRecord> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| rec_with_partition(i, v.clone()))
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seg.pax");
        write_pax_segment(
            &path,
            &records,
            "col",
            1,
            VectorQuant::RaBitQ,
            Some(16 * 1024),
            None,
        )
        .unwrap();
        let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
            .await
            .unwrap();
        let p = path.to_str().unwrap();

        let filter = partition_filter("p3");
        let (allow, _) = pax_filtered_row_allow(&fs, p, &filter)
            .await
            .unwrap()
            .expect("stage-F allow-set");
        let query = corpus[137].clone();
        let hits = rabitq_search_segment_coalesced_allowed(
            &fs,
            p,
            &query,
            K,
            RankMetric::L2,
            None,
            None,
            Some(&allow),
            None,
        )
        .await
        .unwrap()
        .expect("restricted cascade returns hits");
        assert!(!hits.is_empty());

        // Every hit satisfies the predicate (row-accurate restriction).
        for h in &hits {
            let i: usize = h.oid.trim_start_matches('r').parse().unwrap();
            assert_eq!(i % 8, 3, "hit {} violates the filter", h.oid);
        }

        // Recall vs the FILTERED brute-force ground truth.
        let l2 =
            |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>();
        let mut matching: Vec<usize> = (0..N).filter(|i| i % 8 == 3).collect();
        matching.sort_by(|&a, &b| {
            l2(&corpus[a], &query)
                .partial_cmp(&l2(&corpus[b], &query))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let truth: std::collections::HashSet<String> =
            matching.iter().take(K).map(|i| format!("r{i}")).collect();
        let got: std::collections::HashSet<String> = hits.iter().map(|h| h.oid.clone()).collect();
        let recall = truth.iter().filter(|o| got.contains(*o)).count() as f32 / K as f32;
        assert!(
            recall >= 0.90,
            "filtered cascade recall@{K} = {recall:.2} vs filtered brute force"
        );

        // An empty allow-set yields an empty result (not an error).
        let empty = proximadb_block_format::RowAllow::new(N);
        let none_hits = rabitq_search_segment_coalesced_allowed(
            &fs,
            p,
            &query,
            K,
            RankMetric::L2,
            None,
            None,
            Some(&empty),
            None,
        )
        .await
        .unwrap()
        .expect("cascade runs");
        assert!(none_hits.is_empty());
    }

    /// ADR-089 P1: the filter-aware cascade is DEFAULT-OFF (mixed-read-safe).
    #[test]
    fn filtered_cascade_gate_defaults_on() {
        // FLIPPED default-ON 2026-08-31 (see the gate's doc comment). Pinned
        // explicitly so this assertion cannot race another test's env pinning
        // under plain `cargo test`'s shared process.
        unsafe {
            std::env::remove_var("PROXIMADB_PAX_FILTERED_CASCADE");
        }
        assert!(pax_filtered_cascade_enabled());
    }

    /// ADR-089 / TD-FPRUNE-1 P1 EVIDENCE HARNESS (engine-level A/B on a real
    /// segment). Not part of the suite — run explicitly, in release, for the
    /// before/after timing that gates the phase:
    ///
    /// ```text
    /// cargo test --release -p proximadb --lib -- --ignored p1_evidence --nocapture
    /// ```
    ///
    /// A = today's filtered path over a flushed segment: whole-object read +
    ///     `read_segment_records` (decode ALL rows) + per-record
    ///     `evaluate_filter_proxima` + exact scoring of survivors.
    /// B = P1: Stage-F row allow-set + row-restricted RaBitQ/SQ8 cascade.
    ///
    /// Engine-level by design: the end-to-end server path cannot serve segments
    /// for REST-created collections until the TD-FLUSH-8 catalog-identity skip
    /// is fixed, and this seam is exactly what ADR-089 P1 changes.
    #[tokio::test]
    #[ignore = "evidence harness — run explicitly in release for timing"]
    async fn p1_evidence_filtered_exact_vs_cascade() {
        enable_coalesced_rabitq();
        use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
        use std::time::Instant;

        const DIM: usize = 128;
        const N: usize = 80_000;
        const K: usize = 10;
        const QUERIES: usize = 15;

        let corpus: Vec<Vec<f32>> = (0..N)
            .map(|i| {
                (0..DIM)
                    .map(|d| (((i * 131 + d * 17) % 251) as f32) * 0.01)
                    .collect()
            })
            .collect();
        let records: Vec<ProximaRecord> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| rec_with_partition(i, v.clone()))
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seg.pax");
        write_pax_segment(&path, &records, "col", 1, VectorQuant::RaBitQ, None, None).unwrap();
        let seg_bytes = std::fs::metadata(&path).unwrap().len();
        let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
            .await
            .unwrap();
        let p = path.to_str().unwrap();
        let filter = partition_filter("p3");

        // --- A: exact-path emulation (mirrors search_pax_file_exact's shape).
        let mut a_times = Vec::new();
        for q in 0..QUERIES {
            let query = &corpus[(q * 5119) % N];
            let t = Instant::now();
            let bytes = std::fs::read(&path).unwrap();
            let recs = read_segment_records(&bytes, &[], &[], None).unwrap();
            let mut scored: Vec<(f32, &ProximaRecord)> = recs
                .iter()
                .filter(|r| {
                    crate::core::search::sql_value_filter::evaluate_filter_proxima(
                        &filter, &r.props,
                    )
                })
                .filter_map(|r| {
                    r.embeddings.first().map(|e| {
                        let v = e.as_fp32_slice();
                        let d: f32 = v.iter().zip(query).map(|(x, y)| (x - y) * (x - y)).sum();
                        (d, r)
                    })
                })
                .collect();
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(K);
            a_times.push(t.elapsed());
            assert!(!scored.is_empty());
        }

        // --- B: P1 Stage-F + restricted cascade (allow-set rebuilt per query,
        //     the worst case — a per-(segment, filter) cache is a P2 lever).
        let mut b_times = Vec::new();
        for q in 0..QUERIES {
            let query = &corpus[(q * 5119) % N];
            let t = Instant::now();
            let (allow, _stats) = pax_filtered_row_allow(&fs, p, &filter)
                .await
                .unwrap()
                .expect("stage-F allow-set");
            let hits = rabitq_search_segment_coalesced_allowed(
                &fs,
                p,
                query,
                K,
                RankMetric::L2,
                None,
                None,
                Some(&allow),
                None,
            )
            .await
            .unwrap()
            .expect("restricted cascade");
            b_times.push(t.elapsed());
            assert!(!hits.is_empty());
            for h in &hits {
                let i: usize = h.oid.trim_start_matches('r').parse().unwrap();
                assert_eq!(i % 8, 3);
            }
        }

        let ms = |d: &std::time::Duration| d.as_secs_f64() * 1e3;
        let mean = |v: &[std::time::Duration]| {
            ms(&(v.iter().sum::<std::time::Duration>())) / v.len() as f64
        };
        let min = |v: &[std::time::Duration]| v.iter().map(ms).fold(f64::INFINITY, f64::min);
        println!(
            "P1 EVIDENCE  N={N} dim={DIM} K={K} queries={QUERIES} segment={:.1}MB",
            seg_bytes as f64 / 1e6
        );
        println!(
            "  A exact whole-object : mean={:.2}ms  min={:.2}ms",
            mean(&a_times),
            min(&a_times)
        );
        println!(
            "  B stage-F + cascade  : mean={:.2}ms  min={:.2}ms",
            mean(&b_times),
            min(&b_times)
        );
        println!("  speedup: {:.1}x (mean)", mean(&a_times) / mean(&b_times));
    }

    /// TD-CACHE-1 S1: `prefetch_segment_invariants` warms the CONTROL plane
    /// (header/footer/A0) so a subsequent cold search issues fewer control
    /// GETs than a fully cold one — without touching Region A/B payloads.
    #[tokio::test]
    async fn prefetch_invariants_reduces_first_search_control_gets() {
        enable_coalesced_rabitq();
        use crate::storage::persistence::filesystem::FileSystem;
        use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
        use proximadb_storage_filesystem_types::counting::{CountingFileSystem, global_counters};
        use std::sync::Arc;
        use std::sync::atomic::Ordering;

        const DIM: usize = 64;
        const N: usize = 300;
        let corpus: Vec<Vec<f32>> = (0..N)
            .map(|i| {
                (0..DIM)
                    .map(|d| (((i * 37 + d * 13) % 199) as f32) * 0.01)
                    .collect()
            })
            .collect();
        let records: Vec<ProximaRecord> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| rec(&format!("p{i}"), 2000 + i as i64, v.clone()))
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prefetch.pax");
        write_pax_segment(
            &path,
            &records,
            "col",
            1,
            VectorQuant::RaBitQ,
            Some(16 * 1024),
            None,
        )
        .unwrap();

        let local = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
            .await
            .unwrap();
        let counters = global_counters();
        let fs: Arc<dyn FileSystem> =
            Arc::new(CountingFileSystem::new(Arc::new(local), counters.clone()));
        let path_str = path.to_str().unwrap();

        // Cold search WITHOUT any cache: baseline GET count.
        let before = counters.range_reads.load(Ordering::Relaxed);
        rabitq_search_segment_coalesced(
            fs.as_ref(),
            path_str,
            &corpus[42],
            5,
            RankMetric::L2,
            None,
            None,
        )
        .await
        .unwrap()
        .expect("cold search hits");
        let cold_gets = counters.range_reads.load(Ordering::Relaxed) - before;

        // Prefetch into a fresh invariants cache (control plane only).
        let cache = SegmentInvariantsCache::new(8 * 1024 * 1024);
        let warmed = prefetch_segment_invariants(fs.as_ref(), path_str, &cache)
            .await
            .unwrap();
        assert!(warmed, "coalesced segment should prefill");
        let inv = cache.get(path_str).expect("invariants cached");
        assert!(!inv.header_bytes.is_empty());
        assert!(!inv.footer_bytes.is_empty());
        assert!(inv.region_bytes.is_none(), "payload must NOT be prefetched");
        // Idempotent: second prefetch is a no-op.
        assert!(
            !prefetch_segment_invariants(fs.as_ref(), path_str, &cache)
                .await
                .unwrap()
        );

        // First search WITH the prefilled cache: fewer GETs than fully cold.
        let before = counters.range_reads.load(Ordering::Relaxed);
        rabitq_search_segment_coalesced(
            fs.as_ref(),
            path_str,
            &corpus[42],
            5,
            RankMetric::L2,
            Some(&cache),
            None,
        )
        .await
        .unwrap()
        .expect("warm search hits");
        let warm_gets = counters.range_reads.load(Ordering::Relaxed) - before;
        assert!(
            warm_gets < cold_gets,
            "prefetch must elide control GETs (cold={cold_gets}, warm={warm_gets})"
        );
    }

    /// ADR-065 Q3: the survivor-range cache turns a hot repeat query into cache
    /// hits. The second identical search issues strictly fewer ranged GETs — the
    /// survivor (Region B) + OID (Region D) ranges are served from RAM, never
    /// reaching the counting FS wrapper — and returns identical results. The
    /// invariants cache is intentionally NOT injected, so the GET reduction is
    /// attributable solely to the survivor cache.
    #[tokio::test]
    async fn survivor_cache_reduces_gets_on_hot_repeat_query() {
        enable_coalesced_rabitq();
        use crate::storage::persistence::filesystem::FileSystem;
        use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
        use proximadb_storage_filesystem_types::counting::{CountingFileSystem, global_counters};
        use std::sync::Arc;
        use std::sync::atomic::Ordering;

        const DIM: usize = 64;
        const N: usize = 400;
        const K: usize = 10;

        let corpus: Vec<Vec<f32>> = (0..N)
            .map(|i| {
                (0..DIM)
                    .map(|d| (((i * 131 + d * 17) % 251) as f32) * 0.01)
                    .collect()
            })
            .collect();
        let records: Vec<ProximaRecord> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| rec(&format!("r{i}"), 1000 + i as i64, v.clone()))
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seg.pax");
        write_pax_segment(
            &path,
            &records,
            "col",
            1,
            VectorQuant::RaBitQ,
            Some(16 * 1024),
            None,
        )
        .unwrap();

        let local = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
            .await
            .unwrap();
        let counters = global_counters();
        let fs: Arc<dyn FileSystem> =
            Arc::new(CountingFileSystem::new(Arc::new(local), counters.clone()));
        let path_str = path.to_str().unwrap();
        let cache = SurvivorRangeCache::new(8 * 1024 * 1024);

        let query = corpus[137].clone();

        // Search 1 (cold): every survivor + OID range misses → fetched + cached.
        let before1 = counters.range_reads.load(Ordering::Relaxed);
        let hits1 = rabitq_search_segment_coalesced(
            fs.as_ref(),
            path_str,
            &query,
            K,
            RankMetric::L2,
            None,
            Some(&cache),
        )
        .await
        .unwrap()
        .expect("first search returns hits");
        let gets1 = counters.range_reads.load(Ordering::Relaxed) - before1;

        // Search 2 (hot): identical query → survivor + OID ranges hit RAM.
        let before2 = counters.range_reads.load(Ordering::Relaxed);
        let hits2 = rabitq_search_segment_coalesced(
            fs.as_ref(),
            path_str,
            &query,
            K,
            RankMetric::L2,
            None,
            Some(&cache),
        )
        .await
        .unwrap()
        .expect("second search returns hits");
        let gets2 = counters.range_reads.load(Ordering::Relaxed) - before2;

        assert!(
            gets2 < gets1,
            "hot repeat query must issue fewer ranged GETs (got {gets2} >= {gets1}; \
             the survivor+OID ranges should have hit the cache)"
        );
        // Recall is identical — the cache returns the same bytes the backend would.
        let ids = |h: &[CascadeHit]| {
            let mut s: Vec<String> = h.iter().map(|x| x.oid.clone()).collect();
            s.sort();
            s
        };
        assert_eq!(
            ids(&hits1),
            ids(&hits2),
            "the survivor cache must not change search results"
        );
    }

    /// TD-RDSTRAT-8 PR-A (rev 3): `write_pax_segment_compacted` emits the
    /// persisted-IVF-probe (v3) layout when write training is enabled (default
    /// ON, compaction-only). The current binary probes v3 by default; the
    /// read-side kill-switch serves it through the same single-level scan path
    /// with full parity: search recall holds, and `read_segment_records`
    /// reconstructs every row WITH its Region-B vectors (the compaction/recovery
    /// inverse — a v3 segment must never silently drop vectors when re-compacted).
    /// TD-CACHE-6: the PROBE-ARMED path must serve hot repeats from the
    /// survivor-cache seam — an identical second search issues strictly
    /// fewer ranged GETs (probed cells + any Region-B fetch cached) and
    /// returns identical results. Before the fix, probe fetches read the fs
    /// directly and every repeat re-paid the full GET chain.
    #[tokio::test]
    async fn probe_path_hot_repeat_served_from_cache_seam() {
        enable_coalesced_rabitq();
        use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
        const DIM: usize = 64;
        const N: usize = 400;
        unsafe {
            std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
            std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "1");
            std::env::set_var("PROXIMADB_IVF_K", "4");
        }
        let corpus: Vec<Vec<f32>> = (0..N)
            .map(|i| {
                (0..DIM)
                    .map(|d| (((i * 131 + d * 17) % 251) as f32) * 0.01)
                    .collect()
            })
            .collect();
        let records: Vec<ProximaRecord> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| rec(&format!("r{i}"), 1000 + i as i64, v.clone()))
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let seg = dir.path().join("v3probe.pax");
        write_pax_segment_compacted(
            &seg,
            &records,
            "col",
            1,
            VectorQuant::RaBitQ,
            VectorQuant::Sq8,
            false,
            Some(16 * 1024),
            None,
        )
        .unwrap();

        use proximadb_storage_filesystem_types::counting::{CountingFileSystem, global_counters};
        let counters = global_counters();
        let counting = CountingFileSystem::new(
            std::sync::Arc::new(LocalFileSystem::new(LocalConfig::default()).await.unwrap()),
            counters.clone(),
        );
        let survivor = SurvivorRangeCache::new(64 * 1024 * 1024);
        let path = seg.to_string_lossy().to_string();
        let query = corpus[7].clone();

        let before = counters
            .range_reads
            .load(std::sync::atomic::Ordering::Relaxed);
        let first = rabitq_search_segment_coalesced(
            &counting,
            &path,
            &query,
            5,
            RankMetric::L2,
            None,
            Some(&survivor),
        )
        .await
        .unwrap()
        .expect("probe search hits");
        let cold_gets = counters
            .range_reads
            .load(std::sync::atomic::Ordering::Relaxed)
            - before;

        let mid = counters
            .range_reads
            .load(std::sync::atomic::Ordering::Relaxed);
        let second = rabitq_search_segment_coalesced(
            &counting,
            &path,
            &query,
            5,
            RankMetric::L2,
            None,
            Some(&survivor),
        )
        .await
        .unwrap()
        .expect("repeat search hits");
        let hot_gets = counters
            .range_reads
            .load(std::sync::atomic::Ordering::Relaxed)
            - mid;

        assert_eq!(
            first.iter().map(|h| &h.oid).collect::<Vec<_>>(),
            second.iter().map(|h| &h.oid).collect::<Vec<_>>(),
            "repeat must return identical results"
        );
        assert!(
            hot_gets < cold_gets,
            "hot repeat must be served from the cache seam (cold={cold_gets}, hot={hot_gets})"
        );
        unsafe {
            std::env::remove_var("PROXIMADB_PAX_WRITE_A0_TRAIN");
            std::env::remove_var("PROXIMADB_PAX_READ_COARSE_PROBE");
            std::env::remove_var("PROXIMADB_IVF_K");
        }
    }

    /// TD-RDSTRAT-12 §3 positive control: with `PROXIMADB_READ_RANGES_INFLIGHT`
    /// armed, the coarse-probe COLD tail must flow through
    /// `FileSystem::read_ranges_prefetch`'s bounded-concurrent executor — i.e.
    /// the §2 `read_ranges` metrics get RECORDED (`Some`) where the sequential
    /// baseline records nothing. This is the exact defect class §3 falsified:
    /// the gate existed, tests passed, and NO production caller ever reached
    /// the gated function. Runs under nextest process-per-test isolation so the
    /// fn-local `OnceLock` env snapshots stay fresh per test.
    #[tokio::test]
    async fn probe_cold_tail_records_read_ranges_metrics_when_inflight_armed() {
        enable_coalesced_rabitq();
        use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
        use proximadb_storage_filesystem_types::counting::{CountingFileSystem, global_counters};
        use proximadb_storage_filesystem_types::drain_read_ranges_metrics;
        const DIM: usize = 64;
        const N: usize = 400;
        unsafe {
            std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
            std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "1");
            std::env::set_var("PROXIMADB_IVF_K", "8");
            std::env::set_var("PROXIMADB_READ_RANGES_INFLIGHT", "4");
            // The 400-row fixture fits inside the default 1 MiB file-prefix
            // prefetch AND one gap-coalesced range — both would satisfy every
            // probed cell above the cold queue before batching sees it.
            // Disarm both so cells reach the TRUE-cold tail this test gates.
            std::env::set_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES", "1");
            std::env::set_var("PROXIMADB_PAX_VECTOR_COALESCE_GAP", "0");
        }
        let corpus: Vec<Vec<f32>> = (0..N)
            .map(|i| {
                (0..DIM)
                    .map(|d| (((i * 131 + d * 17) % 251) as f32) * 0.01)
                    .collect()
            })
            .collect();
        let records: Vec<ProximaRecord> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| rec(&format!("r{i}"), 1000 + i as i64, v.clone()))
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let seg = dir.path().join("v3prefetch.pax");
        write_pax_segment_compacted(
            &seg,
            &records,
            "col",
            1,
            VectorQuant::RaBitQ,
            VectorQuant::Sq8,
            false,
            Some(16 * 1024),
            None,
        )
        .unwrap();

        let counting = CountingFileSystem::new(
            std::sync::Arc::new(LocalFileSystem::new(LocalConfig::default()).await.unwrap()),
            global_counters(),
        );
        let survivor = SurvivorRangeCache::new(64 * 1024 * 1024);
        let path = seg.to_string_lossy().to_string();

        // Clear any metric a previous call left in the single-slot buffer.
        let _ = drain_read_ranges_metrics();

        let hits = rabitq_search_segment_coalesced(
            &counting,
            &path,
            &corpus[7],
            5,
            RankMetric::L2,
            None,
            Some(&survivor),
        )
        .await
        .unwrap()
        .expect("probe search hits");

        let metrics = drain_read_ranges_metrics().expect(
            "armed INFLIGHT must route the probe cold tail through \
                     read_ranges_prefetch and record its concurrency metrics",
        );
        assert!(
            metrics.max_inflight >= 1 && metrics.fetch_rounds >= 1,
            "implausible metrics payload: {metrics:?}"
        );
        assert_eq!(hits.len(), 5);

        // Identical repeat stays served by the survivor seam — its consume
        // pass adds NO further physical reads, so the metric slot is NOT
        // rewritten by this query.
        let again = rabitq_search_segment_coalesced(
            &counting,
            &path,
            &corpus[7],
            5,
            RankMetric::L2,
            None,
            Some(&survivor),
        )
        .await
        .unwrap()
        .expect("repeat search hits");
        assert_eq!(again.len(), 5);
        assert!(
            drain_read_ranges_metrics().is_none(),
            "hot repeat must not issue additional batched reads"
        );

        unsafe {
            std::env::remove_var("PROXIMADB_PAX_WRITE_A0_TRAIN");
            std::env::remove_var("PROXIMADB_PAX_READ_COARSE_PROBE");
            std::env::remove_var("PROXIMADB_IVF_K");
            std::env::remove_var("PROXIMADB_READ_RANGES_INFLIGHT");
            std::env::remove_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES");
            std::env::remove_var("PROXIMADB_PAX_VECTOR_COALESCE_GAP");
        }
    }

    /// TD-RDSTRAT-13 C4: a cold footer rides the probe wave (or the rank
    /// overlap) and is billed EXACTLY ONCE per query — never double-billed by
    /// a wave slot plus a sequential re-read. Tier counters (C1) are the
    /// teeth: `tier_get_ops[InvariantMeta] == 1` per query, and the wave
    /// observability still fires on the cold query.
    #[tokio::test]
    async fn footer_rides_probe_wave_and_is_never_double_billed() {
        enable_coalesced_rabitq();
        use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
        use proximadb_storage_filesystem_types::counting::{CountingFileSystem, global_counters};
        const DIM: usize = 64;
        const N: usize = 400;
        unsafe {
            std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
            std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "1");
            std::env::set_var("PROXIMADB_IVF_K", "8");
            std::env::set_var("PROXIMADB_READ_RANGES_INFLIGHT", "4");
            // Disarm prefix-prefetch and gap-coalescing so the footer AND the
            // probed cells are genuinely cold (the 400-row fixture otherwise
            // fits inside the 1 MiB prefix).
            std::env::set_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES", "1");
            std::env::set_var("PROXIMADB_PAX_VECTOR_COALESCE_GAP", "0");
        }
        let corpus: Vec<Vec<f32>> = (0..N)
            .map(|i| {
                (0..DIM)
                    .map(|d| (((i * 131 + d * 17) % 251) as f32) * 0.01)
                    .collect()
            })
            .collect();
        let records: Vec<ProximaRecord> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| rec(&format!("r{i}"), 1000 + i as i64, v.clone()))
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let seg = dir.path().join("v3footer.pax");
        write_pax_segment_compacted(
            &seg,
            &records,
            "col",
            1,
            VectorQuant::RaBitQ,
            VectorQuant::Sq8,
            false,
            Some(16 * 1024),
            None,
        )
        .unwrap();

        let counting = CountingFileSystem::new(
            std::sync::Arc::new(LocalFileSystem::new(LocalConfig::default()).await.unwrap()),
            global_counters(),
        );
        let survivor = SurvivorRangeCache::new(64 * 1024 * 1024);
        let path = seg.to_string_lossy().to_string();

        let (hits_res, snap) = crate::observability::io_trace::scope(async {
            let hits = rabitq_search_segment_coalesced(
                &counting,
                &path,
                &corpus[7],
                5,
                RankMetric::L2,
                None,
                Some(&survivor),
            )
            .await
            .unwrap()
            .expect("probe search hits");
            (hits, crate::observability::io_trace::snapshot())
        })
        .await;
        let snap = snap.expect("io_trace scope active");
        assert_eq!(hits_res.len(), 5);
        assert_eq!(
            snap.tier_get_ops[CacheTier::InvariantMeta as usize],
            1,
            "cold query bills the footer EXACTLY once (Meta tier): {:?}",
            snap.tier_get_ops
        );
        assert_eq!(
            snap.metadata_ops, 1,
            "exactly one HEAD on the cold path (C3 removes it when sizes thread)"
        );

        // Repeat query (cache=None wrapper ⇒ nothing cached engine-side):
        // the probe ranges now come from the survivor seam and the footer
        // overlaps the rank via the join path — still exactly one footer GET,
        // and no wave runs (nothing else is cold).
        let (hits_res2, snap2) = crate::observability::io_trace::scope(async {
            let hits = rabitq_search_segment_coalesced(
                &counting,
                &path,
                &corpus[7],
                5,
                RankMetric::L2,
                None,
                Some(&survivor),
            )
            .await
            .unwrap()
            .expect("repeat search hits");
            (hits, crate::observability::io_trace::snapshot())
        })
        .await;
        let snap2 = snap2.expect("io_trace scope active");
        assert_eq!(hits_res2.len(), 5);
        assert_eq!(
            snap2.tier_get_ops[CacheTier::InvariantMeta as usize],
            1,
            "repeat query still bills the footer exactly once (join path)"
        );

        unsafe {
            std::env::remove_var("PROXIMADB_PAX_WRITE_A0_TRAIN");
            std::env::remove_var("PROXIMADB_PAX_READ_COARSE_PROBE");
            std::env::remove_var("PROXIMADB_IVF_K");
            std::env::remove_var("PROXIMADB_READ_RANGES_INFLIGHT");
            std::env::remove_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES");
            std::env::remove_var("PROXIMADB_PAX_VECTOR_COALESCE_GAP");
        }
    }

    /// TD-RDSTRAT-12 §3 round 4: `plan_wave_split_ranges` — aligned chunks of
    /// `chunk` bytes covering `[off, off+len)`, last chunk carrying the
    /// remainder. Pure-function table test (the split planner's only math).
    #[test]
    fn plan_wave_split_ranges_chunks_and_remainder() {
        // Exact multiples: 3 chunks of 4.
        assert_eq!(
            plan_wave_split_ranges(100, 12, 4),
            vec![100..104, 104..108, 108..112]
        );
        // Remainder: 4 MiB chunk + 1 B remainder.
        assert_eq!(
            plan_wave_split_ranges(0, 4 * 1024 * 1024 + 1, 4 * 1024 * 1024),
            vec![0..4 * 1024 * 1024, 4 * 1024 * 1024..4 * 1024 * 1024 + 1]
        );
        // Region smaller than one chunk: single range.
        assert_eq!(plan_wave_split_ranges(7, 3, 16), vec![7..10]);
        // Degenerate: empty region ⇒ no ranges; zero chunk is clamped to 1
        // (defensive: always terminates, one byte per range).
        assert_eq!(
            plan_wave_split_ranges(5, 0, 4),
            Vec::<std::ops::Range<u64>>::new()
        );
        assert_eq!(
            plan_wave_split_ranges(5, 4, 0),
            vec![5..6, 6..7, 7..8, 8..9]
        );
    }

    /// TD-RDSTRAT-12 §3 round 4: with INFLIGHT armed, the Region-B clustered
    /// cold tail AND the Region-D OID tail flow through
    /// `read_ranges_prefetch` (the FS slot holds the LAST wave's metrics ⇒
    /// Some after the cold search, None on a hot repeat that everything
    /// serves from the cache seam), and the search results are identical to
    /// the repeat arm. On the tiny fixture the probe tail may already be
    /// RAM-resident by later waves — the assertion is that SOME true-cold
    /// wave fired and the cache seam serves the repeat with no further
    /// physical reads.
    #[tokio::test]
    async fn region_b_and_oid_cold_tails_batch_when_inflight_armed() {
        enable_coalesced_rabitq();
        use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
        use proximadb_storage_filesystem_types::counting::{CountingFileSystem, global_counters};
        use proximadb_storage_filesystem_types::drain_read_ranges_metrics;
        const DIM: usize = 64;
        const N: usize = 400;
        unsafe {
            std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
            std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "1");
            std::env::set_var("PROXIMADB_IVF_K", "8");
            std::env::set_var("PROXIMADB_READ_RANGES_INFLIGHT", "4");
            std::env::set_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES", "1");
            std::env::set_var("PROXIMADB_PAX_VECTOR_COALESCE_GAP", "0");
        }
        let corpus: Vec<Vec<f32>> = (0..N)
            .map(|i| {
                (0..DIM)
                    .map(|d| (((i * 131 + d * 17) % 251) as f32) * 0.01)
                    .collect()
            })
            .collect();
        let records: Vec<ProximaRecord> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| rec(&format!("r{i}"), 1000 + i as i64, v.clone()))
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let seg = dir.path().join("v3regionbd.pax");
        write_pax_segment_compacted(
            &seg,
            &records,
            "col",
            1,
            VectorQuant::RaBitQ,
            VectorQuant::Sq8,
            false,
            Some(16 * 1024),
            None,
        )
        .unwrap();

        let counting = CountingFileSystem::new(
            std::sync::Arc::new(LocalFileSystem::new(LocalConfig::default()).await.unwrap()),
            global_counters(),
        );
        let survivor = SurvivorRangeCache::new(64 * 1024 * 1024);
        let path = seg.to_string_lossy().to_string();

        let _ = drain_read_ranges_metrics();
        let cold = rabitq_search_segment_coalesced(
            &counting,
            &path,
            &corpus[7],
            5,
            RankMetric::L2,
            None,
            Some(&survivor),
        )
        .await
        .unwrap()
        .expect("cold search hits");

        let metrics = drain_read_ranges_metrics()
            .expect("armed INFLIGHT must route the cold tails through read_ranges_prefetch");
        assert!(metrics.fetch_rounds >= 1 && metrics.max_inflight >= 1);

        let warm = rabitq_search_segment_coalesced(
            &counting,
            &path,
            &corpus[7],
            5,
            RankMetric::L2,
            None,
            Some(&survivor),
        )
        .await
        .unwrap()
        .expect("repeat search hits");
        assert_eq!(
            cold.iter().map(|h| h.oid.clone()).collect::<Vec<_>>(),
            warm.iter().map(|h| h.oid.clone()).collect::<Vec<_>>(),
            "wave arm must not change results"
        );
        assert!(
            drain_read_ranges_metrics().is_none(),
            "hot repeat must be fully served by the cache seams"
        );

        unsafe {
            std::env::remove_var("PROXIMADB_PAX_WRITE_A0_TRAIN");
            std::env::remove_var("PROXIMADB_PAX_READ_COARSE_PROBE");
            std::env::remove_var("PROXIMADB_IVF_K");
            std::env::remove_var("PROXIMADB_READ_RANGES_INFLIGHT");
            std::env::remove_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES");
            std::env::remove_var("PROXIMADB_PAX_VECTOR_COALESCE_GAP");
        }
    }

    #[tokio::test]
    async fn ivf_probe_compaction_gate_and_v3_read_compat() {
        enable_coalesced_rabitq();
        use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
        use proximadb_storage_common::coarse_directory::CoarseDirectory;
        use proximadb_storage_common::segment_layout::{SEG_LAYOUT_VERSION, SegmentHeaderPrefix};

        const DIM: usize = 64;
        const N: usize = 400;
        const K: usize = 10;
        // Small k (the shared IOP-derived override) so the tiny fixture still
        // forms multiple cells — the natural N·dim/IOP count would be ~2 here.
        unsafe {
            std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
            std::env::set_var("PROXIMADB_IVF_K", "4");
        }

        let corpus: Vec<Vec<f32>> = (0..N)
            .map(|i| {
                (0..DIM)
                    .map(|d| (((i * 131 + d * 17) % 251) as f32) * 0.01)
                    .collect()
            })
            .collect();
        let records: Vec<ProximaRecord> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| rec(&format!("r{i}"), 1000 + i as i64, v.clone()))
            .collect();
        let dir = tempfile::tempdir().unwrap();

        // Flag ON ⇒ compaction writes v3 (header + A0 + footer mirror agree).
        let v3_path = dir.path().join("v3.pax");
        write_pax_segment_compacted(
            &v3_path,
            &records,
            "col",
            1,
            VectorQuant::RaBitQ,
            VectorQuant::Sq8,
            false,
            Some(16 * 1024),
            None,
        )
        .unwrap();
        let v3_bytes = std::fs::read(&v3_path).unwrap();
        let h = SegmentHeaderPrefix::parse(&v3_bytes).unwrap();
        assert_eq!(
            h.layout_version, SEG_LAYOUT_VERSION,
            "PROXIMADB_PAX_WRITE_A0_TRAIN=1 compaction must emit the v3 layout"
        );
        let a0 =
            CoarseDirectory::parse(&v3_bytes[h.a0_off as usize..(h.a0_off + h.a0_len) as usize])
                .expect("Region A0 parses from the compacted segment");
        assert_eq!(a0.model.rows_covered(), N as u64);
        assert_eq!(a0.model.dim as usize, DIM);

        // Flag OFF ⇒ same records compact to the v1 (single-level) layout.
        // #1234: PROXIMADB_PAX_WRITE_A0_TRAIN is now default-ON; explicitly
        // set "0" to test the OFF -> v1 path (remove_var would fall back to
        // the CoarseProbeConfig default, which is ON).
        unsafe {
            std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "0");
        }
        let v1_path = dir.path().join("v1.pax");
        write_pax_segment_compacted(
            &v1_path,
            &records,
            "col",
            1,
            VectorQuant::RaBitQ,
            VectorQuant::Sq8,
            false,
            Some(16 * 1024),
            None,
        )
        .unwrap();
        let h1 = SegmentHeaderPrefix::parse(&std::fs::read(&v1_path).unwrap()).unwrap();
        assert_eq!(
            h1.layout_version, SEG_LAYOUT_VERSION,
            "without PROXIMADB_PAX_WRITE_A0_TRAIN the compaction layout is unchanged (v1)"
        );

        // v3 read-compat: the coalesced cascade searches the v3 segment with
        // brute-force-level recall (single-level full scan until PR-B).
        let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
            .await
            .unwrap();
        let query = corpus[137].clone();
        let hits = rabitq_search_segment_coalesced(
            &fs,
            v3_path.to_str().unwrap(),
            &query,
            K,
            RankMetric::L2,
            None,
            None,
        )
        .await
        .unwrap()
        .expect("v3 segment must be searchable by the current binary");
        assert_eq!(hits[0].oid, "r137", "nearest neighbour on the v3 layout");
        let l2 =
            |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>();
        let mut idx: Vec<usize> = (0..N).collect();
        idx.sort_by(|&a, &b| {
            l2(&corpus[a], &query)
                .partial_cmp(&l2(&corpus[b], &query))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let truth: std::collections::HashSet<String> =
            idx.iter().take(K).map(|i| format!("r{i}")).collect();
        let got: std::collections::HashSet<String> = hits.iter().map(|h| h.oid.clone()).collect();
        let recall = truth.iter().filter(|o| got.contains(*o)).count() as f32 / K as f32;
        assert!(recall >= 0.90, "v3 read-compat recall@{K} = {recall:.2}");

        // Compaction/recovery inverse: every row reads back WITH its vector
        // (Region-B overlay must work through the v3 prefix).
        let recs = read_segment_records(&v3_bytes, &[], &[], None).unwrap();
        assert_eq!(recs.len(), N);
        assert!(
            recs.iter().all(|r| r
                .embeddings
                .first()
                .is_some_and(|e| !e.as_fp32_slice().is_empty())),
            "v3 read_segment_records must reconstruct vectors (never drop Region B)"
        );
        unsafe {
            std::env::remove_var("PROXIMADB_IVF_K");
        }
    }

    /// TD-RDSTRAT-8 PR-B (the deferred PR-A recall/GET gate): the coarse probe
    /// (`PROXIMADB_PAX_READ_COARSE_PROBE=1`) ranks the persisted centroids in RAM and reads
    /// only the nprobe nearest cells — holding recall on clustered data while
    /// reading materially fewer bytes than the whole-region single-level scan.
    #[tokio::test]
    async fn coarse_probe_holds_recall_and_cuts_bytes() {
        enable_coalesced_rabitq();
        use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

        const DIM: usize = 32;
        const G: usize = 8; // clusters (each centred on its own axis)
        const PER: usize = 64; // members per cluster
        const N: usize = G * PER;
        const K: usize = 10;

        // Deterministic pseudo-random component in [-1, 1].
        let pseudo = |i: usize, d: usize| -> f32 {
            let mut s = (i as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add((d as u64).wrapping_mul(0x632B_E59B_D9B4_E019));
            s ^= s >> 29;
            s = s.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            s ^= s >> 27;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        // Widely-separated clusters (each centred at 50 on its own axis) with
        // members at a monotonically increasing radius — so the true top-K are
        // DISTINGUISHABLE (non-degenerate full-scan recall) AND concentrated in
        // one cluster → a few probed cells capture them.
        let corpus: Vec<Vec<f32>> = (0..N)
            .map(|i| {
                let (g, j) = (i / PER, i % PER);
                let radius = 0.1 + j as f32 * 0.03;
                (0..DIM)
                    .map(|d| (if d == g { 50.0 } else { 0.0 }) + radius * pseudo(i, d))
                    .collect()
            })
            .collect();
        let records: Vec<ProximaRecord> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| rec(&format!("r{i}"), 1000 + i as i64, v.clone()))
            .collect();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("probe.pax");
        unsafe {
            std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
            std::env::set_var("PROXIMADB_IVF_K", "16");
        }
        write_pax_segment_compacted(
            &path,
            &records,
            "col",
            1,
            VectorQuant::RaBitQ,
            VectorQuant::Sq8,
            false,
            Some(16 * 1024),
            None,
        )
        .unwrap();
        unsafe {
            std::env::remove_var("PROXIMADB_PAX_WRITE_A0_TRAIN");
        }

        let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
            .await
            .unwrap();
        let p = path.to_str().unwrap();
        // The exact centre of cluster 0 — its nearest members are the smallest
        // radii (r0, r1, …), a distinguishable top-K all inside one cluster.
        let query: Vec<f32> = (0..DIM).map(|d| if d == 0 { 50.0 } else { 0.0 }).collect();

        // Brute-force truth (top-K by L2).
        let l2 =
            |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>();
        let mut idx: Vec<usize> = (0..N).collect();
        idx.sort_by(|&a, &b| {
            l2(&corpus[a], &query)
                .partial_cmp(&l2(&corpus[b], &query))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let truth: std::collections::HashSet<String> =
            idx.iter().take(K).map(|i| format!("r{i}")).collect();
        let recall = |hits: &[CascadeHit]| {
            let got: std::collections::HashSet<String> =
                hits.iter().map(|h| h.oid.clone()).collect();
            truth.iter().filter(|o| got.contains(*o)).count() as f32 / K as f32
        };

        unsafe {
            std::env::set_var("PROXIMADB_TRACE_GETS", "1");
            // Establish the exact-read control for the prefix-prefetch proof
            // below; production defaults to a bounded 1 MiB prefix.
            std::env::set_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES", "0");
        }

        // Baseline: probe OFF → whole-region single-level scan over the v3 segment.
        unsafe {
            std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "0");
        }
        let _ = drain_get_trace();
        let _ = drain_probe_trace();
        let base = rabitq_search_segment_coalesced(&fs, p, &query, K, RankMetric::L2, None, None)
            .await
            .unwrap()
            .unwrap();
        let base_bytes: u64 = drain_get_trace().iter().map(|(_, b)| b).sum();
        assert!(drain_probe_trace().is_empty(), "probe OFF must not probe");
        let base_recall = recall(&base);
        assert!(
            base_recall >= 0.5,
            "baseline full-scan recall@{K} = {base_recall:.2} (sanity floor)"
        );

        // Probe ON: nprobe=6 of 16 cells.
        unsafe {
            std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "1");
            std::env::set_var("PROXIMADB_PAX_READ_COARSE_NPROBE", "6");
        }
        let _ = drain_get_trace();
        let _ = drain_probe_trace();
        let (probe, probe_snapshot) = crate::observability::io_trace::scope(async {
            let hits =
                rabitq_search_segment_coalesced(&fs, p, &query, K, RankMetric::L2, None, None)
                    .await
                    .unwrap()
                    .unwrap();
            let snapshot = crate::observability::io_trace::snapshot()
                .expect("reader probe test runs inside an io_trace scope");
            (hits, snapshot)
        })
        .await;
        let probe_gets = drain_get_trace();
        let probe_bytes: u64 = probe_gets.iter().map(|(_, b)| b).sum();
        let ptrace = drain_probe_trace();

        // 1. The probe engaged: cells_probed strictly inside (0, cells_total).
        assert_eq!(ptrace.len(), 1, "exactly one probe recorded");
        let (cells_total, cells_probed, probed_rows, fetch_rounds) = ptrace[0];
        assert_eq!(cells_total, 16, "16 coarse cells");
        assert!(
            cells_probed > 0 && cells_probed <= 6,
            "probed {cells_probed} cells (0 < c <= nprobe=6)"
        );
        assert!(
            probed_rows > 0 && probed_rows < N as u64,
            "probed {probed_rows} of {N} rows (a strict subset)"
        );
        // `fetch_rounds` is the count of coalesced Region-A ranged reads — the
        // request-side term of the cost model, and on Hot-tier object storage
        // requests are the only billed read term.
        //
        // What this pins: one probed cell costs AT MOST one ranged GET.
        // `plan_probe_cell_ranges` either merges a cell into the previous range
        // or emits it 1:1, never splitting one cell across several reads, so a
        // future planner that chunked oversized cells to respect
        // `max_range_bytes` would multiply per-query round-trips — and would
        // trip this.
        //
        // What this does NOT catch, stated so the guard is not mistaken for
        // more than it is: coalescing silently ceasing to merge. That keeps
        // `fetch_rounds <= cells_probed` true while raising it toward the
        // ceiling. The relative check further down (coalesced <= default policy)
        // covers part of that; an absolute per-query ratchet needs a real bed
        // and measured baselines, which is TD-IVF-3's deferred probe-economy
        // ratchet, not something a synthetic 16-cell fixture can stand in for.
        assert!(
            (1..=cells_probed).contains(&fetch_rounds),
            "fetch_rounds={fetch_rounds} must be in 1..={cells_probed}: coalescing \
             may merge probed cells into fewer ranged GETs but must never split \
             one cell across several"
        );
        assert_eq!(probe_snapshot.ivf_cells_total, cells_total);
        assert_eq!(probe_snapshot.ivf_cells_probed, cells_probed);
        assert_eq!(probe_snapshot.ivf_probed_rows, probed_rows);
        assert_eq!(probe_snapshot.ivf_fetch_rounds, fetch_rounds);
        assert_eq!(probe_snapshot.ivf_whole_region_fallback, 0);
        let traced_region_a_bytes: u64 = probe_gets
            .iter()
            .filter(|(tier, _)| matches!(tier, CacheTier::ProbeIndex))
            .map(|(_, bytes)| bytes)
            .sum();
        let traced_region_b_bytes: u64 = probe_gets
            .iter()
            .filter(|(tier, _)| matches!(tier, CacheTier::SurvivorPayload))
            .map(|(_, bytes)| bytes)
            .sum();
        assert!(traced_region_a_bytes > 0, "probe must fetch Region A");
        assert!(traced_region_b_bytes > 0, "rerank must fetch Region B");
        assert_eq!(
            probe_snapshot.ivf_region_a_bytes, traced_region_a_bytes,
            "durable Region-A bytes must equal physical probe reads"
        );
        assert_eq!(
            probe_snapshot.ivf_region_b_bytes, traced_region_b_bytes,
            "durable Region-B bytes must equal physical rerank reads"
        );

        // 2. Materially fewer bytes than the whole-region baseline.
        assert!(
            probe_bytes < base_bytes,
            "probe {probe_bytes} B must read fewer than baseline {base_bytes} B"
        );

        // 3. Recall held vs the full scan: probing the query's cluster loses
        //    essentially nothing (the meaningful PR-B claim — robust to the
        //    absolute RaBitQ recall level, which the corpus/quantizer set).
        let probe_recall = recall(&probe);
        assert!(
            probe_recall >= base_recall - 0.1,
            "probe recall@{K} = {probe_recall:.2} must hold vs baseline {base_recall:.2} (nprobe=6/16)"
        );

        // PR-C2 proof: bounded Region-A gap over-read may reduce physical
        // ranges, but it must rank exactly the same selected cells and rows.
        unsafe {
            std::env::set_var("PROXIMADB_PAX_VECTOR_COALESCE_GAP", "1048576");
            std::env::set_var("PROXIMADB_PAX_VECTOR_COALESCE_RANGE", "4194304");
        }
        let _ = drain_get_trace();
        let _ = drain_probe_trace();
        let (coalesced, coalesced_snapshot) = crate::observability::io_trace::scope(async {
            let hits =
                rabitq_search_segment_coalesced(&fs, p, &query, K, RankMetric::L2, None, None)
                    .await
                    .unwrap()
                    .unwrap();
            let snapshot = crate::observability::io_trace::snapshot()
                .expect("coalescing proof runs inside an io_trace scope");
            (hits, snapshot)
        })
        .await;
        let _ = drain_get_trace();
        let _ = drain_probe_trace();
        assert_eq!(
            coalesced.iter().map(|hit| &hit.oid).collect::<Vec<_>>(),
            probe.iter().map(|hit| &hit.oid).collect::<Vec<_>>(),
            "physical gap over-read must not change logical survivors/results"
        );
        assert_eq!(coalesced_snapshot.ivf_cells_probed, cells_probed);
        assert_eq!(coalesced_snapshot.ivf_probed_rows, probed_rows);
        assert!(
            coalesced_snapshot.ivf_fetch_rounds <= fetch_rounds,
            "cost coalescing cannot add Region-A reads"
        );
        unsafe {
            std::env::remove_var("PROXIMADB_PAX_VECTOR_COALESCE_GAP");
            std::env::remove_var("PROXIMADB_PAX_VECTOR_COALESCE_RANGE");
        }

        // PR-C2 proof: a bounded prefix read must replace the separate header,
        // A0 and RaBitQ-parameter GETs with one physical control GET.
        let baseline_control_gets = probe_gets
            .iter()
            .filter(|(tier, _)| *tier == CacheTier::SearchControl)
            .count();
        assert!(
            baseline_control_gets >= 3,
            "control baseline needs header+A0+params"
        );
        unsafe {
            std::env::set_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES", "65536");
        }
        let _ = drain_get_trace();
        let prefetched =
            rabitq_search_segment_coalesced(&fs, p, &query, K, RankMetric::L2, None, None)
                .await
                .unwrap()
                .unwrap();
        let prefix_gets = drain_get_trace();
        let _ = drain_probe_trace();
        assert_eq!(
            prefetched.iter().map(|hit| &hit.oid).collect::<Vec<_>>(),
            probe.iter().map(|hit| &hit.oid).collect::<Vec<_>>()
        );
        assert_eq!(
            prefix_gets
                .iter()
                .filter(|(tier, _)| *tier == CacheTier::SearchControl)
                .count(),
            1,
            "one prefix GET supplies header+A0+RaBitQ parameters"
        );
        unsafe {
            std::env::remove_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES");
        }

        // PR-C2 proof: metadata warms by default even though the probe never
        // admits the complete Region A into the invariant cache. Production
        // does not set an opt-in gate, so this regression must exercise the
        // unset environment exactly as the server does.
        unsafe {
            std::env::remove_var("PROXIMADB_PAX_SPLIT_PROBE_META_CACHE");
        }
        let invariants = SegmentInvariantsCache::new(8 * 1024 * 1024);
        let _ = drain_get_trace();
        let _ = rabitq_search_segment_coalesced(
            &fs,
            p,
            &query,
            K,
            RankMetric::L2,
            Some(&invariants),
            None,
        )
        .await
        .unwrap()
        .unwrap();
        let first_cached = drain_get_trace();
        let _ = drain_probe_trace();
        let _ = rabitq_search_segment_coalesced(
            &fs,
            p,
            &query,
            K,
            RankMetric::L2,
            Some(&invariants),
            None,
        )
        .await
        .unwrap()
        .unwrap();
        let second_cached = drain_get_trace();
        let _ = drain_probe_trace();
        assert!(
            first_cached.iter().any(|(tier, _)| matches!(
                tier,
                CacheTier::SearchControl | CacheTier::InvariantMeta
            )),
            "cold cache population reads control metadata"
        );
        assert!(
            second_cached.iter().all(|(tier, _)| !matches!(
                tier,
                CacheTier::SearchControl | CacheTier::InvariantMeta
            )),
            "warm probe must not re-read cached control metadata"
        );
        unsafe {
            std::env::remove_var("PROXIMADB_PAX_SPLIT_PROBE_META_CACHE");
        }

        // 4. Corrupt A0 fails closed to the canonical whole-Region-A scan and
        //    makes that fallback visible in the durable query trace.
        let mut corrupt = std::fs::read(&path).unwrap();
        let corrupt_header = SegmentHeaderPrefix::parse(&corrupt).unwrap();
        corrupt[corrupt_header.a0_off as usize] ^= 0x01;
        std::fs::write(&path, corrupt).unwrap();
        let (fallback, fallback_snapshot) = crate::observability::io_trace::scope(async {
            let hits =
                rabitq_search_segment_coalesced(&fs, p, &query, K, RankMetric::L2, None, None)
                    .await
                    .unwrap()
                    .unwrap();
            let snapshot = crate::observability::io_trace::snapshot()
                .expect("reader fallback test runs inside an io_trace scope");
            (hits, snapshot)
        })
        .await;
        assert!(
            !fallback.is_empty(),
            "A0 corruption fails closed, not empty"
        );
        assert_eq!(fallback_snapshot.ivf_cells_total, 0);
        assert_eq!(fallback_snapshot.ivf_cells_probed, 0);
        assert_eq!(fallback_snapshot.ivf_whole_region_fallback, 1);
        assert!(
            fallback_snapshot.ivf_region_a_bytes > 0,
            "whole-region fallback must attribute its physical Region-A bytes"
        );
        assert!(
            fallback_snapshot.ivf_region_b_bytes > 0,
            "fallback rerank must attribute its physical Region-B bytes"
        );
        unsafe {
            std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "0");
            std::env::remove_var("PROXIMADB_PAX_READ_COARSE_NPROBE");
            std::env::remove_var("PROXIMADB_TRACE_GETS");
            std::env::remove_var("PROXIMADB_IVF_K");
            std::env::remove_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES");
            std::env::remove_var("PROXIMADB_PAX_VECTOR_COALESCE_GAP");
            std::env::remove_var("PROXIMADB_PAX_VECTOR_COALESCE_RANGE");
            std::env::remove_var("PROXIMADB_PAX_SPLIT_PROBE_META_CACHE");
        }
    }

    /// TD-FPRUNE-1 M3: the FILTERED coarse probe holds recall AND cuts bytes.
    /// Trace-driven: a filtered query previously DISARMED the geometric probe and
    /// read whole Region-A/B (measured 3–7 MB vs the unfiltered 2 MB). M3 re-arms
    /// the probe under a filter — adaptive nprobe by matching-row count + rank
    /// restricted to allowed rows — so a filtered query becomes a filtered ANN
    /// that touches only relevant cells. This asserts (a) recall parity vs the
    /// FILTERED brute-force truth and (b) the probe engaged (a strict cell subset).
    #[tokio::test]
    async fn filtered_coarse_probe_holds_recall_and_cuts_bytes() {
        enable_coalesced_rabitq();
        use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

        const DIM: usize = 32;
        const G: usize = 8; // clusters (each on its own axis) == partitions
        const PER: usize = 64;
        const N: usize = G * PER;
        const K: usize = 10;

        let pseudo = |i: usize, d: usize| -> f32 {
            let mut s = (i as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add((d as u64).wrapping_mul(0x632B_E59B_D9B4_E019));
            s ^= s >> 29;
            s = s.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            s ^= s >> 27;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        // partition == cluster (correlated with vector space). Each cluster's
        // members sit at increasing radius so the top-K are distinguishable.
        let corpus: Vec<Vec<f32>> = (0..N)
            .map(|i| {
                let (g, j) = (i / PER, i % PER);
                let radius = 0.1 + j as f32 * 0.03;
                (0..DIM)
                    .map(|d| (if d == g { 50.0 } else { 0.0 }) + radius * pseudo(i, d))
                    .collect()
            })
            .collect();
        let records: Vec<ProximaRecord> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let mut r = rec(&format!("r{i}"), 1000 + i as i64, v.clone());
                r.props.insert(
                    "partition".to_string(),
                    proximadb_records::ProximaTreeNode::Value(
                        proximadb_data_model::ProximaValue::String(format!("p{}", i / PER)),
                    ),
                );
                r
            })
            .collect();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fprobe.pax");
        unsafe {
            std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
            std::env::set_var("PROXIMADB_IVF_K", "16");
            // This test targets the GLOBAL single-directory probe; pin it off the
            // now-default-on partition layout (C2) so its cell-count assertions hold.
            std::env::set_var("PROXIMADB_PAX_PARTITION_LAYOUT", "0");
        }
        // Compacted (v3, A0) + shredded so `partition` is a footer-resolvable column.
        write_pax_segment_compacted_shredded(
            &path,
            &records,
            "col",
            1,
            VectorQuant::RaBitQ,
            VectorQuant::Sq8,
            false,
            Some(16 * 1024),
            &[("partition".to_string(), 100)],
            None,
        )
        .unwrap();
        unsafe {
            std::env::remove_var("PROXIMADB_PAX_WRITE_A0_TRAIN");
        }

        let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
            .await
            .unwrap();
        let p = path.to_str().unwrap();
        // Query at cluster 0's centre; filter partition=p0 (== cluster 0).
        let query: Vec<f32> = (0..DIM).map(|d| if d == 0 { 50.0 } else { 0.0 }).collect();

        // FILTERED brute-force truth: top-K among partition==p0 rows (cluster 0).
        let l2 =
            |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>();
        let mut p0: Vec<usize> = (0..N).filter(|&i| i / PER == 0).collect();
        p0.sort_by(|&a, &b| {
            l2(&corpus[a], &query)
                .partial_cmp(&l2(&corpus[b], &query))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let truth: std::collections::HashSet<String> =
            p0.iter().take(K).map(|i| format!("r{i}")).collect();

        // Build the allow-set (partition==p0) from the segment's own footer.
        let (allow, _stats) = pax_filtered_row_allow(&fs, p, &partition_filter("p0"))
            .await
            .unwrap()
            .expect("shredded v3 segment yields a stage-F allow-set");
        assert!(!allow.is_empty(), "partition=p0 must match rows");

        // FILTERED probe ON, nprobe floor small — M3 expands adaptively to cover
        // enough matching rows. Shrink the prefix prefetch (default 1 MiB slurps a
        // small test segment WHOLE, masking the ranged-read win) and disable the
        // cross-call metadata cache so the byte A/B below isolates the region reads
        // — the cloud-GET cost term the co-design mandate targets.
        unsafe {
            std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "1");
            std::env::set_var("PROXIMADB_PAX_READ_COARSE_NPROBE", "2");
            std::env::set_var("PROXIMADB_TRACE_GETS", "1");
            std::env::set_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES", "4096");
            std::env::set_var("PROXIMADB_PAX_SPLIT_PROBE_META_CACHE", "0");
        }
        let _ = drain_get_trace();
        let _ = drain_probe_trace();
        let hits = rabitq_search_segment_coalesced_allowed(
            &fs,
            p,
            &query,
            K,
            RankMetric::L2,
            None,
            None,
            Some(&allow),
            None,
        )
        .await
        .unwrap()
        .unwrap();
        let ptrace = drain_probe_trace();
        let bytes_on: u64 = drain_get_trace().iter().map(|(_, b)| b).sum();

        // (a) recall parity vs the FILTERED truth.
        let got: std::collections::HashSet<String> = hits.iter().map(|h| h.oid.clone()).collect();
        let recall = truth.iter().filter(|o| got.contains(*o)).count() as f32 / K as f32;
        assert!(
            recall >= 0.9,
            "filtered probe recall@{K} = {recall:.2} (want ≥0.9 vs filtered brute-force)"
        );
        // Every hit is a partition=p0 row (no cross-partition leakage).
        for h in &hits {
            let idx: usize = h.oid.trim_start_matches('r').parse().unwrap();
            assert_eq!(idx / PER, 0, "hit {} is not partition p0", h.oid);
        }
        // (b) the probe engaged under the filter — a strict cell subset (not the
        // whole-region fallback the disarmed path took).
        assert_eq!(ptrace.len(), 1, "exactly one filtered probe recorded");
        let (cells_total, cells_probed, probed_rows, _rounds) = ptrace[0];
        assert_eq!(cells_total, 16);
        assert!(
            cells_probed > 0 && cells_probed < cells_total,
            "filtered probe touched {cells_probed} of {cells_total} cells (strict subset)"
        );
        assert!(
            probed_rows < N as u64,
            "filtered probe read {probed_rows} of {N} rows (subset, not whole region)"
        );

        // (c) the byte cut is MEASURED, not asserted: re-run the SAME filtered query
        // with the probe DISARMED (the pre-M3 behavior — whole Region-A/B allowed
        // rank) and confirm the armed probe fetched strictly fewer bytes. With the
        // prefix prefetch shrunk above, this reflects the ranged-GET (cloud) regime.
        unsafe {
            std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "0");
        }
        let _ = drain_get_trace();
        let base = rabitq_search_segment_coalesced_allowed(
            &fs,
            p,
            &query,
            K,
            RankMetric::L2,
            None,
            None,
            Some(&allow),
            None,
        )
        .await
        .unwrap()
        .unwrap();
        let bytes_off: u64 = drain_get_trace().iter().map(|(_, b)| b).sum();
        assert!(
            drain_probe_trace().is_empty(),
            "probe OFF must not record a probe (whole-region baseline)"
        );
        // The disarmed baseline is still correct (whole-region allowed rank) — same
        // filtered top-K — so the armed probe trades no recall for the byte cut.
        let base_got: std::collections::HashSet<String> =
            base.iter().map(|h| h.oid.clone()).collect();
        assert_eq!(got, base_got, "armed probe and whole-region baseline agree");
        assert!(
            bytes_on < bytes_off,
            "filtered probe fetched {bytes_on} B vs the disarmed {bytes_off} B \
             (want strictly fewer — the ranged-read win)"
        );

        unsafe {
            std::env::remove_var("PROXIMADB_PAX_READ_COARSE_PROBE");
            std::env::remove_var("PROXIMADB_PAX_READ_COARSE_NPROBE");
            std::env::remove_var("PROXIMADB_TRACE_GETS");
            std::env::remove_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES");
            std::env::remove_var("PROXIMADB_PAX_SPLIT_PROBE_META_CACHE");
            std::env::remove_var("PROXIMADB_PAX_PARTITION_LAYOUT");
            std::env::remove_var("PROXIMADB_IVF_K");
        }
    }

    /// TD-FPRUNE-1 M2: the tag-aware layout makes an UNCORRELATED tag prune —
    /// the case M3 alone cannot help (every geometric cell holds every value, so
    /// `count_in_range` skips nothing). With partition-by-tag layout the cells
    /// are partition-homogeneous, so the filtered probe accumulates its matching
    /// pool from FEWER cells. This builds the SAME uncorrelated corpus twice
    /// (layout ON vs OFF) and asserts the ON segment reads strictly fewer rows to
    /// serve the same filtered top-K, at recall parity and with no leakage.
    #[tokio::test]
    async fn partition_layout_prunes_uncorrelated_tag() {
        enable_coalesced_rabitq();
        use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

        const DIM: usize = 32;
        const P: usize = 4; // partitions (the tag)
        const GV: usize = 8; // vector clusters (independent of the tag)
        const N: usize = 512;
        const K: usize = 10;

        let pseudo = |i: usize, d: usize| -> f32 {
            let mut s = (i as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add((d as u64).wrapping_mul(0x632B_E59B_D9B4_E019));
            s ^= s >> 29;
            s = s.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            s ^= s >> 27;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        // partition = i % P; vector cluster = (i / P) % GV. Disjoint index ranges
        // ⇒ the tag is UNCORRELATED with vector geometry (every cluster carries
        // every partition). This is the case zone-maps/blooms cannot prune.
        let part = |i: usize| i % P;
        let vclust = |i: usize| (i / P) % GV;
        let corpus: Vec<Vec<f32>> = (0..N)
            .map(|i| {
                let c = vclust(i);
                (0..DIM)
                    .map(|d| (if d == c { 50.0 } else { 0.0 }) + 0.4 * pseudo(i, d))
                    .collect()
            })
            .collect();
        let records: Vec<ProximaRecord> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let mut r = rec(&format!("r{i}"), 1000 + i as i64, v.clone());
                r.props.insert(
                    "partition".to_string(),
                    proximadb_records::ProximaTreeNode::Value(
                        proximadb_data_model::ProximaValue::String(format!("p{}", part(i))),
                    ),
                );
                r
            })
            .collect();

        // Query at vector cluster 0; filter partition == p0 (UNCORRELATED with it).
        let query: Vec<f32> = (0..DIM).map(|d| if d == 0 { 50.0 } else { 0.0 }).collect();
        let l2 =
            |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>();
        let mut p0: Vec<usize> = (0..N).filter(|&i| part(i) == 0).collect();
        p0.sort_by(|&a, &b| {
            l2(&corpus[a], &query)
                .partial_cmp(&l2(&corpus[b], &query))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let truth: std::collections::HashSet<String> =
            p0.iter().take(K).map(|i| format!("r{i}")).collect();

        let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
            .await
            .unwrap();

        // Build one v3 shredded segment with the partition-layout gate `on`, and
        // measure a filtered probe on it: returns (hits, probed_rows).
        async fn build_and_probe(
            fs: &LocalFileSystem,
            records: &[ProximaRecord],
            query: &[f32],
            k: usize,
            layout_on: bool,
        ) -> (Vec<CascadeHit>, u64) {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("m2.pax");
            unsafe {
                std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
                std::env::set_var("PROXIMADB_IVF_K", "16");
                // Explicit "1"/"0" (the layout is now default-on, C2): "0" is the
                // kill-switch to the global layout for the OFF arm.
                std::env::set_var(
                    "PROXIMADB_PAX_PARTITION_LAYOUT",
                    if layout_on { "1" } else { "0" },
                );
            }
            write_pax_segment_compacted_shredded(
                &path,
                records,
                "col",
                1,
                VectorQuant::RaBitQ,
                VectorQuant::Sq8,
                false,
                Some(16 * 1024),
                &[("partition".to_string(), 100)],
                None,
            )
            .unwrap();
            unsafe {
                std::env::remove_var("PROXIMADB_PAX_WRITE_A0_TRAIN");
                std::env::remove_var("PROXIMADB_PAX_PARTITION_LAYOUT");
            }
            let p = path.to_str().unwrap();
            let (allow, _s) = pax_filtered_row_allow(fs, p, &partition_filter("p0"))
                .await
                .unwrap()
                .expect("shredded v3 segment yields an allow-set");
            unsafe {
                std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "1");
                std::env::set_var("PROXIMADB_PAX_READ_COARSE_NPROBE", "2");
                std::env::set_var("PROXIMADB_TRACE_GETS", "1");
                std::env::set_var("PROXIMADB_PAX_SPLIT_PROBE_META_CACHE", "0");
            }
            let _ = drain_probe_trace();
            let hits = rabitq_search_segment_coalesced_allowed(
                fs,
                p,
                query,
                k,
                RankMetric::L2,
                None,
                None,
                Some(&allow),
                None,
            )
            .await
            .unwrap()
            .unwrap();
            let ptrace = drain_probe_trace();
            unsafe {
                std::env::remove_var("PROXIMADB_PAX_READ_COARSE_PROBE");
                std::env::remove_var("PROXIMADB_PAX_READ_COARSE_NPROBE");
                std::env::remove_var("PROXIMADB_TRACE_GETS");
                std::env::remove_var("PROXIMADB_PAX_SPLIT_PROBE_META_CACHE");
                std::env::remove_var("PROXIMADB_IVF_K");
            }
            let probed_rows = ptrace.first().map(|t| t.2).unwrap_or(u64::MAX);
            (hits, probed_rows)
        }

        let (hits_on, rows_on) = build_and_probe(&fs, &records, &query, K, true).await;
        let (hits_off, rows_off) = build_and_probe(&fs, &records, &query, K, false).await;

        // The M2 risk is a RELATIVE one — does per-partition clustering degrade
        // recall vs the global layout at identical settings? So compare the two,
        // not an absolute bar (1-bit RaBitQ over a 512-row toy corpus is coarse by
        // construction; the absolute recall gate is the SIFT ratchet + a real
        // eval, not this unit).
        let recall = |hits: &[CascadeHit]| {
            let got: std::collections::HashSet<String> =
                hits.iter().map(|h| h.oid.clone()).collect();
            truth.iter().filter(|o| got.contains(*o)).count() as f32 / K as f32
        };
        let (r_on, r_off) = (recall(&hits_on), recall(&hits_off));
        eprintln!(
            "[M2] recall on={r_on:.2} off={r_off:.2} | probed_rows on={rows_on} off={rows_off}"
        );
        // No cross-partition leakage under the layout (the allow-set is exact).
        for h in &hits_on {
            let idx: usize = h.oid.trim_start_matches('r').parse().unwrap();
            assert_eq!(part(idx), 0, "hit {} is not partition p0", h.oid);
        }
        // Both layouts find real neighbors (search works at all).
        assert!(r_on > 0.0 && r_off > 0.0, "both layouts return matches");
        // Partition clustering does not MATERIALLY degrade recall vs the global
        // layout at the same settings (within one RaBitQ quantization step).
        assert!(
            r_on >= r_off - 0.15,
            "partition layout must not hurt recall vs global: on={r_on:.2} off={r_off:.2}"
        );
        // THE M2 WIN: partition-homogeneous cells let the filtered probe reach its
        // matching pool from fewer rows than the global (mixed-cell) layout, where
        // every probed cell is only ~1/P matching. This is the uncorrelated-tag
        // prune that M3 alone cannot deliver.
        assert!(
            rows_on < rows_off,
            "partition layout must probe FEWER rows for an uncorrelated tag: \
             on={rows_on} vs off={rows_off}"
        );
    }

    /// TD-FPRUNE-1 C1: integrated full-stack GET/byte breakdown for a filtered PAX
    /// query at realistic multi-block, multi-partition scale — the co-design
    /// "trace before you tune" evidence that Slice A + M3 + M2 actually moved the
    /// dominant cost, and which term is the RESIDUAL. It SPLITS the filtered path
    /// into its two I/O stages — the Stage-F allow-set build (`pax_filtered_row_allow`,
    /// reads footer-surviving Region-D blocks) vs the probe+rerank
    /// (`rabitq_search_segment_coalesced_allowed`, reads A0 + probe cells + Region-B)
    /// — for BOTH a correlated and an uncorrelated tag, with M2 layout off vs on.
    /// The printed breakdown identifies the residual dominant GET/byte term (does
    /// the allow-set build dominate? → P3 per-cell stats is justified). Assertions
    /// are the regression guards; the eprintln is the evidence.
    #[tokio::test]
    async fn filtered_stack_get_byte_breakdown() {
        enable_coalesced_rabitq();
        use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

        const DIM: usize = 64;
        const P: usize = 8; // partitions (the tag)
        const N: usize = 4096;
        const K: usize = 10;

        let pseudo = |i: usize, d: usize| -> f32 {
            let mut s = (i as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add((d as u64).wrapping_mul(0x632B_E59B_D9B4_E019));
            s ^= s >> 29;
            s = s.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            s ^= s >> 27;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };

        // One filtered measurement over a freshly built segment. `correlated`
        // decides whether the tag tracks the vector cluster; `m2_on` toggles the
        // partition layout at compaction. Returns the per-stage GET/byte split.
        struct Measure {
            filesize: u64,
            allow_gets: u64,
            allow_bytes: u64,
            blocks_total: usize,
            blocks_pruned_footer: usize,
            search_gets: u64,
            search_bytes: u64,
            region_a_bytes: u64,
            region_b_bytes: u64,
            whole_region_fallback: u64,
            recall: f32,
        }
        async fn measure(
            fs: &dyn proximadb_storage_filesystem_types::FileSystem,
            pseudo: &dyn Fn(usize, usize) -> f32,
            correlated: bool,
            m2_on: bool,
        ) -> Measure {
            // partition = i%P; vector cluster = correlated ? partition : (i/P)%P.
            let part = |i: usize| i % P;
            let vclust = |i: usize| if correlated { i % P } else { (i / P) % P };
            let corpus: Vec<Vec<f32>> = (0..N)
                .map(|i| {
                    let c = vclust(i);
                    (0..DIM)
                        .map(|d| (if d == c { 50.0 } else { 0.0 }) + 0.4 * pseudo(i, d))
                        .collect()
                })
                .collect();
            let records: Vec<ProximaRecord> = corpus
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let mut r = rec(&format!("r{i}"), 1000 + i as i64, v.clone());
                    r.props.insert(
                        "partition".to_string(),
                        proximadb_records::ProximaTreeNode::Value(
                            proximadb_data_model::ProximaValue::String(format!("p{}", part(i))),
                        ),
                    );
                    r
                })
                .collect();

            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("c1.pax");
            unsafe {
                std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
                std::env::set_var("PROXIMADB_IVF_K", "64");
                // Pin what this measurement depends on explicitly (the
                // cascade gate is default-ON since the 2026-08-31 flip; footer
                // stats remains default-OFF) — removed with the rest at the end
                // of the arm.
                std::env::set_var("PROXIMADB_PAX_FILTERED_CASCADE", "1");
                std::env::set_var("PROXIMADB_PAX_FOOTER_STATS", "1");
                // Explicit "1"/"0" (layout is now default-on, C2): "0" pins the OFF
                // arm to the global layout via the kill-switch.
                std::env::set_var(
                    "PROXIMADB_PAX_PARTITION_LAYOUT",
                    if m2_on { "1" } else { "0" },
                );
            }
            // Small blocks ⇒ many Region-D blocks (footer-pruning needs blocks
            // small enough to be partition-homogeneous after clustering).
            write_pax_segment_compacted_shredded(
                &path,
                &records,
                "col",
                1,
                VectorQuant::RaBitQ,
                VectorQuant::Sq8,
                false,
                Some(8 * 1024),
                &[("partition".to_string(), 100)],
                None,
            )
            .unwrap();
            unsafe {
                std::env::remove_var("PROXIMADB_PAX_WRITE_A0_TRAIN");
                std::env::remove_var("PROXIMADB_PAX_PARTITION_LAYOUT");
            }
            let p = path.to_str().unwrap();
            let filesize = fs.metadata(p).await.unwrap().size;

            // Filtered truth: top-K among partition==p0 rows.
            let query: Vec<f32> = (0..DIM).map(|d| if d == 0 { 50.0 } else { 0.0 }).collect();
            let l2 =
                |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>();
            let mut p0: Vec<usize> = (0..N).filter(|&i| part(i) == 0).collect();
            p0.sort_by(|&a, &b| {
                l2(&corpus[a], &query)
                    .partial_cmp(&l2(&corpus[b], &query))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let truth: std::collections::HashSet<String> =
                p0.iter().take(K).map(|i| format!("r{i}")).collect();

            // Isolate the ranged-GET regime (shrink prefetch so the segment isn't
            // slurped whole; disable the cross-call metadata cache). GET/byte are
            // measured via the FS-leaf `io_trace` accumulator (`read_range` feeds
            // `record_range_gets`/`record_bytes_read`) — the SAME counters the
            // route cost model + KRU meter price — NOT `drain_get_trace` (that
            // `GET_TRACE` is a probe-only counter and misses the Stage-F reads).
            unsafe {
                std::env::set_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES", "4096");
                std::env::set_var("PROXIMADB_PAX_SPLIT_PROBE_META_CACHE", "0");
                std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "1");
                std::env::set_var("PROXIMADB_PAX_READ_COARSE_NPROBE", "4");
            }
            use crate::observability::io_trace;

            // Stage-F allow-set build (Region-D only), under its own trace scope.
            let (allow_res, allow_snap) = io_trace::scope(async {
                let r = pax_filtered_row_allow(fs, p, &partition_filter("p0")).await;
                (r, io_trace::snapshot())
            })
            .await;
            let (allow, stats) = allow_res
                .unwrap()
                .expect("shredded v3 segment yields an allow-set");
            let allow_snap = allow_snap.expect("io_trace scope active");

            // Probe + rerank (A0 + probe cells + Region-B), given the allow-set.
            let (hits_res, search_snap) = io_trace::scope(async {
                let r = rabitq_search_segment_coalesced_allowed(
                    fs,
                    p,
                    &query,
                    K,
                    RankMetric::L2,
                    None,
                    None,
                    Some(&allow),
                    None,
                )
                .await;
                (r, io_trace::snapshot())
            })
            .await;
            let hits = hits_res.unwrap().unwrap();
            let search_snap = search_snap.expect("io_trace scope active");

            unsafe {
                std::env::remove_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES");
                std::env::remove_var("PROXIMADB_PAX_SPLIT_PROBE_META_CACHE");
                std::env::remove_var("PROXIMADB_PAX_READ_COARSE_PROBE");
                std::env::remove_var("PROXIMADB_PAX_READ_COARSE_NPROBE");
                std::env::remove_var("PROXIMADB_IVF_K");
                std::env::remove_var("PROXIMADB_PAX_FILTERED_CASCADE");
                std::env::remove_var("PROXIMADB_PAX_FOOTER_STATS");
            }

            let got: std::collections::HashSet<String> =
                hits.iter().map(|h| h.oid.clone()).collect();
            let recall = truth.iter().filter(|o| got.contains(*o)).count() as f32 / K as f32;
            Measure {
                filesize,
                allow_gets: allow_snap.range_gets,
                allow_bytes: allow_snap.bytes_read,
                blocks_total: stats.blocks_total,
                blocks_pruned_footer: stats.blocks_pruned_footer,
                search_gets: search_snap.range_gets,
                search_bytes: search_snap.bytes_read,
                region_a_bytes: search_snap.ivf_region_a_bytes,
                region_b_bytes: search_snap.ivf_region_b_bytes,
                whole_region_fallback: search_snap.ivf_whole_region_fallback,
                recall,
            }
        }

        let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
            .await
            .unwrap();

        let mut rows = Vec::new();
        for (label, correlated, m2_on) in [
            ("correlated       (M2 off)", true, false),
            ("uncorrelated     (M2 off)", false, false),
            ("uncorrelated     (M2 on )", false, true),
        ] {
            let m = measure(&fs, &pseudo, correlated, m2_on).await;
            let total_bytes = m.allow_bytes + m.search_bytes;
            let total_gets = m.allow_gets + m.search_gets;
            eprintln!(
                "[C1] {label} | file={fk}KB whole-scan | \
                 allow-build: {ag} GETs {ab}KB ({bp}/{bt} D-blocks footer-pruned) | \
                 probe+rerank: {sg} GETs {sb}KB (A={ra}KB B={rb}KB wholeB-fallback={wf}) | \
                 TOTAL {tg} GETs {tk}KB ({pct}% of whole) | recall={r:.2}",
                fk = m.filesize / 1024,
                ag = m.allow_gets,
                ab = m.allow_bytes / 1024,
                bp = m.blocks_pruned_footer,
                bt = m.blocks_total,
                sg = m.search_gets,
                sb = m.search_bytes / 1024,
                ra = m.region_a_bytes / 1024,
                rb = m.region_b_bytes / 1024,
                wf = m.whole_region_fallback,
                tg = total_gets,
                tk = total_bytes / 1024,
                pct = (total_bytes * 100).checked_div(m.filesize).unwrap_or(0),
                r = m.recall,
            );
            // The stack composes end-to-end (finds real neighbors) in every config.
            assert!(m.recall > 0.0, "{label}: filtered search returns matches");
            rows.push(m);
        }
        // The measured residual: for an UNCORRELATED tag the Stage-F allow-set
        // build reads ~all Region-D blocks (footer stats can't prune it), so M2's
        // partition-homogeneous layout — which restores footer-pruning AND tightens
        // survivor ranges — must cut BOTH the allow-build and the probe+rerank
        // bytes. This is the end-to-end proof M2 attacks the dominant filtered term.
        let off = &rows[1]; // uncorrelated M2 off
        let on = &rows[2]; // uncorrelated M2 on
        assert!(
            on.blocks_pruned_footer > off.blocks_pruned_footer,
            "M2 must restore footer-pruning for the uncorrelated tag: on={} off={}",
            on.blocks_pruned_footer,
            off.blocks_pruned_footer
        );
        let (on_total, off_total) = (
            on.allow_bytes + on.search_bytes,
            off.allow_bytes + off.search_bytes,
        );
        assert!(
            on_total < off_total,
            "M2 must cut total filtered bytes for the uncorrelated tag: on={on_total} off={off_total}"
        );
        // And with M2 the filtered stack reads strictly less than a whole-object
        // scan (the cascade-OFF baseline) — the co-design win, restored.
        assert!(
            on_total < on.filesize,
            "M2 uncorrelated total {on_total}B must be < whole-object {}B",
            on.filesize
        );
    }

    /// TD-FPRUNE-1 C2: M2's per-partition clustering must not degrade UNFILTERED
    /// recall vs the global layout — the SILENT boundary-miss risk (a row assigned
    /// to its within-partition centroid can fall outside the nprobe-nearest cells
    /// of a globally-ranked unfiltered query). The SIFT ratchet cannot see this
    /// (SIFT carries no tag, so M2 never engages there). This builds the SAME
    /// uncorrelated-tag corpus twice (M2 off vs on) and runs UNFILTERED recall@10
    /// vs the f32 brute-force truth over many queries, requiring the partition
    /// layout to hold recall within tolerance of the global layout. It is the eval
    /// that gates any M2 default change.
    #[tokio::test]
    async fn m2_unfiltered_recall_neutral_vs_global() {
        enable_coalesced_rabitq();
        use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

        const DIM: usize = 64;
        const P: usize = 8;
        const N: usize = 4096;
        const K: usize = 10;
        const NQ: usize = 32;

        let pseudo = |i: usize, d: usize| -> f32 {
            let mut s = (i as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add((d as u64).wrapping_mul(0x632B_E59B_D9B4_E019));
            s ^= s >> 29;
            s = s.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            s ^= s >> 27;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        // Uncorrelated tag (worst case for M2): partition = i%P, vector cluster =
        // (i/P)%P — disjoint index ranges ⇒ every partition spans every cluster,
        // so per-partition centroids overlap maximally (the boundary-miss surface).
        let part = |i: usize| i % P;
        let vclust = |i: usize| (i / P) % P;
        let corpus: Vec<Vec<f32>> = (0..N)
            .map(|i| {
                let c = vclust(i);
                (0..DIM)
                    .map(|d| (if d == c { 50.0 } else { 0.0 }) + 0.6 * pseudo(i, d))
                    .collect()
            })
            .collect();
        let records: Vec<ProximaRecord> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let mut r = rec(&format!("r{i}"), 1000 + i as i64, v.clone());
                r.props.insert(
                    "partition".to_string(),
                    proximadb_records::ProximaTreeNode::Value(
                        proximadb_data_model::ProximaValue::String(format!("p{}", part(i))),
                    ),
                );
                r
            })
            .collect();

        async fn build(
            records: &[ProximaRecord],
            m2_on: bool,
        ) -> (tempfile::TempDir, std::path::PathBuf) {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("c2.pax");
            unsafe {
                std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
                std::env::set_var("PROXIMADB_IVF_K", "64");
                // Explicit "1"/"0" (not present/absent) so this eval is robust to
                // the default flip — the off case is the GLOBAL layout via the
                // kill-switch, on is the partition layout.
                std::env::set_var(
                    "PROXIMADB_PAX_PARTITION_LAYOUT",
                    if m2_on { "1" } else { "0" },
                );
            }
            write_pax_segment_compacted_shredded(
                &path,
                records,
                "col",
                1,
                VectorQuant::RaBitQ,
                VectorQuant::Sq8,
                false,
                Some(16 * 1024),
                &[("partition".to_string(), 100)],
                None,
            )
            .unwrap();
            unsafe {
                std::env::remove_var("PROXIMADB_PAX_WRITE_A0_TRAIN");
                std::env::remove_var("PROXIMADB_PAX_PARTITION_LAYOUT");
            }
            (dir, path)
        }

        let (_d_off, path_off) = build(&records, false).await;
        let (_d_on, path_on) = build(&records, true).await;
        let (p_off, p_on) = (path_off.to_str().unwrap(), path_on.to_str().unwrap());

        // Guard against a VACUOUS pass: M2 must actually have changed the layout
        // (partition-contiguous rows ⇒ different physical bytes). If the two
        // segments were byte-identical, `cluster_plan_ivf_probe_partitioned` fell
        // back to the global plan and the recall comparison below would be a no-op.
        let bytes_off = std::fs::read(&path_off).unwrap();
        let bytes_on = std::fs::read(&path_on).unwrap();
        assert_ne!(
            bytes_off, bytes_on,
            "M2 partition layout did not engage — segments are byte-identical"
        );

        let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
            .await
            .unwrap();
        unsafe {
            std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "1");
        }

        let l2 =
            |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>();

        let (mut sum_off, mut sum_on) = (0f32, 0f32);
        for qi in (0..N).step_by(N / NQ).take(NQ) {
            let query = &corpus[qi];
            // f32 brute-force ground truth top-K over ALL rows (unfiltered).
            let mut idx: Vec<usize> = (0..N).collect();
            idx.sort_by(|&a, &b| {
                l2(&corpus[a], query)
                    .partial_cmp(&l2(&corpus[b], query))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let gt: std::collections::HashSet<String> =
                idx.iter().take(K).map(|i| format!("r{i}")).collect();
            let recall = |hits: &[CascadeHit]| {
                let got: std::collections::HashSet<String> =
                    hits.iter().map(|h| h.oid.clone()).collect();
                gt.iter().filter(|o| got.contains(*o)).count() as f32 / K as f32
            };
            let h_off =
                rabitq_search_segment_coalesced(&fs, p_off, query, K, RankMetric::L2, None, None)
                    .await
                    .unwrap()
                    .unwrap();
            let h_on =
                rabitq_search_segment_coalesced(&fs, p_on, query, K, RankMetric::L2, None, None)
                    .await
                    .unwrap()
                    .unwrap();
            sum_off += recall(&h_off);
            sum_on += recall(&h_on);
        }
        unsafe {
            std::env::remove_var("PROXIMADB_PAX_READ_COARSE_PROBE");
            std::env::remove_var("PROXIMADB_IVF_K");
        }
        let (r_off, r_on) = (sum_off / NQ as f32, sum_on / NQ as f32);
        eprintln!(
            "[C2] unfiltered recall@{K} over {NQ} queries: \
             global(M2 off)={r_off:.3}  partition(M2 on)={r_on:.3}  Δ={:.3}",
            r_on - r_off
        );
        // The gate: partition clustering must hold unfiltered recall within one
        // quantization step of the global layout. If this ever fails, M2 must stay
        // opt-in and the dual-directory follow-up is justified.
        assert!(
            r_on >= r_off - 0.05,
            "M2 partition layout degrades unfiltered recall vs global: \
             on={r_on:.3} off={r_off:.3} (tol 0.05)"
        );
    }
}
