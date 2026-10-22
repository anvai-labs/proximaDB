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
        // bootstrap, so clustering quality is measurable without waiting on a
        // compaction cadence. (Not "the (unwired) flush→compaction scheduler",
        // as an earlier revision said: TD-WLP-7 / ADR-061 D3 wired it in #1012.)
        // Default OFF ⇒ bootstrap.
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
    rg_layout: Option<bool>,
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
        // TD-PAXRG-1: the producer-resolved `pax_rg_layout` (tag > env >
        // default); `None` follows the env gate.
        rg_layout.unwrap_or_else(rg_layout_enabled),
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
    rg_layout: Option<bool>,
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
        rg_layout.unwrap_or_else(rg_layout_enabled),
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
    // `pax_compaction_input_tiers(&port, &task.input_files)`; under rg_layout the tier
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

/// The vector tiers compaction must carry from its input segments into the
/// merged output (P3 Phase D exact tier + tier-2 rerank quant).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PaxCompactionInputTiers {
    /// True only when every vector row in every input `.pax` segment has an
    /// exact f32 authority (the intersection rule): a raw-f32 `EMBED_BASE`, or a
    /// matching `F32_TIER_BASE` / Region C exact tier. One exact input never
    /// upgrades lossy siblings. Legacy ProximaBlocks (`.sst` or no extension)
    /// are exact authorities; other physical formats are not assumed exact. At
    /// least one PAX input must carry exactness, so the optional tier stays
    /// default-OFF for legacy-only compactions.
    pub f32_tier: bool,
    /// The first `.pax` input's co-located `RERANK_BASE` quant (SQ8/FP16/f32),
    /// so compaction never silently downgrades a configured FP16/f32 rerank to
    /// SQ8. Defaults to [`VectorQuant::Sq8`] when no input carries a block-level
    /// rerank column (coalesced segments hoist it to Region B, which is SQ8).
    pub rerank_quant: VectorQuant,
    /// False only when the probe was skipped because the merged records carry
    /// no embeddings (relational / document): then neither answer applies and
    /// callers must not reason about exactness (no "inputs not exact" warning).
    pub carries_vectors: bool,
}

/// What one `.pax` compaction input declares about its vector tiers.
struct PaxInputTierProbe {
    exact: bool,
    rerank_quant: Option<VectorQuant>,
}

/// Resolve [`PaxCompactionInputTiers`] for a compaction task's inputs.
///
/// TD-READFID-2: inputs arrive as storage URLs (`file://{abs}` from
/// `LocalFileSystem::list`, `s3://…`/`az://…`/`gs://…` from object stores) or
/// bare paths, so every read goes through the filesystem port, never
/// `std::fs::read` (which cannot open a URL and made the intersection false on
/// every real deployment — the exact tier was dropped at the first compaction).
///
/// Co-design (I/O round-trips): a coalesced segment declares `has_f32_tier` in
/// its footer, so it is answered from two small ranged reads (header prefix,
/// footer) instead of a whole-object GET. Legacy block-layout PAX (whose
/// exactness is per block) and anything the ranged probe cannot classify fall
/// back to ONE whole read through the port that answers both questions. (Under
/// local encryption the port's `EncryptedFilesystem::read_range` decrypts the
/// whole object and slices it, so the probe sees plaintext but each ranged read
/// costs a whole read + decrypt; TD-COMPACT-IO-1.)
///
/// An input that cannot be READ is an `Err`, never "not exact": classifying a
/// transient I/O failure as lossy would let compaction write a tierless output
/// and then delete the inputs — the only copies of the exact f32 values. The
/// caller aborts the compaction and keeps its inputs so it retries later.
pub async fn pax_compaction_input_tiers(
    port: &std::sync::Arc<dyn proximadb_storage_ports::FilesystemPort>,
    input_files: &[std::path::PathBuf],
) -> anyhow::Result<PaxCompactionInputTiers> {
    let mut saw_pax = false;
    let mut all_exact = true;
    let mut rerank_quant: Option<VectorQuant> = None;
    for f in input_files {
        match f.extension().and_then(|e| e.to_str()) {
            Some("pax") => {}
            Some("sst") | None => continue,
            Some(_) => {
                all_exact = false;
                continue;
            }
        }
        saw_pax = true;
        // Once the intersection is false and a rerank quant is known, the
        // remaining inputs cannot change the answer — skip their I/O.
        if !all_exact && rerank_quant.is_some() {
            continue;
        }
        let url = f.to_string_lossy();
        let probe = probe_pax_input_tiers(port, &url).await?;
        all_exact &= probe.exact;
        if rerank_quant.is_none() {
            rerank_quant = probe.rerank_quant;
        }
    }
    Ok(PaxCompactionInputTiers {
        f32_tier: saw_pax && all_exact,
        rerank_quant: rerank_quant.unwrap_or(VectorQuant::Sq8),
        carries_vectors: true,
    })
}

/// [`pax_compaction_input_tiers`] for a canonical compaction that already holds
/// the merged `records`. When no record carries an embedding (relational or
/// document collections) neither answer is used — there is no vector column to
/// tier or rerank — so the probe is skipped with zero I/O instead of paying a
/// whole read per block-layout input (TD-COMPACT-IO-1).
pub async fn pax_compaction_input_tiers_for_records(
    port: &std::sync::Arc<dyn proximadb_storage_ports::FilesystemPort>,
    input_files: &[std::path::PathBuf],
    records: &[ProximaRecord],
) -> anyhow::Result<PaxCompactionInputTiers> {
    if !records.iter().any(|record| !record.embeddings.is_empty()) {
        return Ok(PaxCompactionInputTiers {
            f32_tier: false,
            rerank_quant: VectorQuant::Sq8,
            carries_vectors: false,
        });
    }
    pax_compaction_input_tiers(port, input_files).await
}

async fn probe_pax_input_tiers(
    port: &std::sync::Arc<dyn proximadb_storage_ports::FilesystemPort>,
    url: &str,
) -> anyhow::Result<PaxInputTierProbe> {
    match read_coalesced_segment_footer(port, url).await {
        // Coalesced: the footer is the authoritative tier declaration and the
        // rerank column is hoisted out of blocks (Region B, SQ8) — identical to
        // what `pax_segment_has_exact_vector_authority` and a block-0 probe of
        // the whole object would conclude.
        Ok(Some(footer)) => {
            return Ok(PaxInputTierProbe {
                exact: footer.has_f32_tier,
                rerank_quant: None,
            });
        }
        Ok(None) => {}
        Err(error) => {
            tracing::debug!(
                url,
                %error,
                "compaction tier probe: ranged footer read failed; falling back to a whole read"
            );
        }
    }
    let bytes = crate::storage::engines::sst::staged_write::read_object_bytes(port, url)
        .await
        .map_err(|error| {
            tracing::warn!(
                url,
                %error,
                "compaction tier probe: input segment unreadable; aborting this compaction \
                 and keeping its inputs (retried on the next trigger)"
            );
            anyhow::anyhow!("compaction tier probe could not read input {url}: {error}")
        })?;
    Ok(PaxInputTierProbe {
        exact: pax_segment_has_exact_vector_authority(&bytes),
        rerank_quant: pax_segment_rerank_quant(bytes),
    })
}

/// Read a coalesced segment's footer index through the filesystem port with
/// two ranged reads: the fixed-size header prefix (magic + the footer extent
/// mirror, as `pax_filtered_row_allow` uses it), then the footer body. No HEAD
/// and no tail read. `Ok(None)` when the object is not a coalesced segment
/// (legacy block layout, too small), so the caller falls back to a whole read.
pub(crate) async fn read_coalesced_segment_footer(
    port: &std::sync::Arc<dyn proximadb_storage_ports::FilesystemPort>,
    url: &str,
) -> anyhow::Result<Option<proximadb_storage_common::segment_layout::SegmentFooterIndex>> {
    use proximadb_storage_common::segment_layout::{
        SEG_HEADER_PREFIX_LEN, SegmentFooterIndex, SegmentHeaderPrefix, is_coalesced_segment,
    };
    let fs = port
        .get_filesystem(url)
        .map_err(|e| anyhow::anyhow!("filesystem for {url}: {e}"))?;
    let prefix = fs
        .read_range(url, 0, SEG_HEADER_PREFIX_LEN as u64)
        .await
        .map_err(|e| anyhow::anyhow!("header read for {url}: {e}"))?;
    if !is_coalesced_segment(&prefix) {
        return Ok(None);
    }
    let header = SegmentHeaderPrefix::parse(&prefix)
        .map_err(|e| anyhow::anyhow!("header parse for {url}: {e}"))?;
    if header.footer_len == 0 {
        anyhow::bail!("coalesced segment {url} declares an empty footer");
    }
    let footer = fs
        .read_range(url, header.footer_off, header.footer_len)
        .await
        .map_err(|e| anyhow::anyhow!("footer read for {url}: {e}"))?;
    if footer.len() as u64 != header.footer_len {
        anyhow::bail!("coalesced footer extent overruns segment {url}");
    }
    SegmentFooterIndex::parse(&footer)
        .map(Some)
        .map_err(|e| anyhow::anyhow!("footer parse for {url}: {e}"))
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
/// [`pax_segment_rerank_quant`]'s `PaxSegmentScanner` block-0 metadata read.
pub fn pax_segment_is_coarse_rabitq_without_f32_tier(bytes: &[u8]) -> bool {
    use proximadb_block_format::vparam::QUANT_RABITQ;
    if SegmentFormat::detect(bytes) != SegmentFormat::Pax {
        return false;
    }
    // A `.pax` SEGMENT is `[block(s)][segment_index][SEGMENT_MAGIC]`; the scanner
    // parses the trailing index + magic and yields block 0 (see
    // [`pax_segment_rerank_quant`]).
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

/// The tier-2 rerank quant of one `.pax` segment: block 0's co-located
/// `RERANK_BASE` column kind (SQ8/FP16/f32), or `None` when the segment has no
/// block-level rerank column (coalesced layouts hoist it to Region B, SQ8) or
/// does not parse. Pure over `bytes`; [`pax_compaction_input_tiers`] uses it so
/// compaction PRESERVES a configured FP16/f32 rerank instead of silently
/// downgrading it to SQ8.
pub fn pax_segment_rerank_quant(bytes: Vec<u8>) -> Option<VectorQuant> {
    use proximadb_block_format::vparam::{QUANT_FP16, QUANT_RAW_F32};
    // Read block 0's column metadata via the segment scanner — a `.pax`
    // SEGMENT file is `[block(s)][segment_index][SEGMENT_MAGIC]`, so opening
    // the whole file as a single block misreads the footer. The rerank column
    // is co-located in every block, so block 0 suffices.
    let mut scanner = PaxSegmentScanner::from_bytes(bytes, ScanPredicate::default()).ok()?;
    let reader = scanner.next_block()?;
    let entry = reader.vector_params().get(col_id::RERANK_BASE)?;
    Some(match entry.quant_kind {
        QUANT_FP16 => VectorQuant::Fp16,
        QUANT_RAW_F32 => VectorQuant::RawF32,
        // SQ8 (the default) or any unknown/reserved kind → Sq8.
        _ => VectorQuant::Sq8,
    })
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
        // TD-SECSCAN-1 §5: the pool can never usefully exceed the segment's
        // rows, and an unclamped value derived from a client `top_k` used to
        // drive a multi-TB merge allocation downstream.
        return pool.min(n.max(1));
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
    // TD-SECSCAN-1 §5: clamp to the segment rows — with `k = u32::MAX` (an
    // uncapped surface) `by_k` alone was a ~4e11 survivor pool feeding a
    // `Vec::with_capacity(pool * degree)` merge buffer.
    by_k.max(by_n).max(*min).min(n.max(1))
}

/// Morsel merge-buffer capacity for ranking a region of `n_rows` rows into a
/// `pool`-sized survivor set across `degree` workers: the pool is clamped to
/// the rows it can actually select from, and the product saturates instead of
/// overflowing (TD-SECSCAN-1 §5).
pub(crate) fn morsel_merge_capacity(pool: usize, n_rows: usize, degree: usize) -> usize {
    pool.min(n_rows).saturating_mul(degree)
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
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
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

/// TD-PAXDIR-1 D2: cell count for a query planned on the collection-global
/// centroid directory. Precedence: env
/// `PROXIMADB_PAX_READ_CENTROID_DIRECTORY_NPROBE` → else the same geometric
/// default as the per-segment probe, applied to the **total** usable cells
/// (`ceil(sqrt(cells) × multiplier)`), so S segments of k cells probe about
/// `2·sqrt(S·k)` cells instead of `S · 2·sqrt(k)`.
pub(crate) fn centroid_directory_nprobe(total_cells: usize) -> usize {
    std::env::var("PROXIMADB_PAX_READ_CENTROID_DIRECTORY_NPROBE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or_else(|| geometric_nprobe(total_cells, coarse_probe_settings()))
        .min(total_cells.max(1))
}

tokio::task_local! {
    /// TD-PAXDIR-1 D2: the Region A0 cells the collection-global centroid
    /// directory selected for the segment this task is scanning. Set by the
    /// collection fan-out around one segment's scan; the coarse probe uses it
    /// in place of its own per-segment nprobe ranking.
    static DIRECTORY_CELLS: Arc<[u32]>;
}

/// Run `scan` with the directory-selected A0 cells for its segment (or, with
/// `None`, unchanged). The probe runs inline in the scanning task, so the
/// scope reaches it without widening the cascade's signatures.
pub(crate) async fn scope_directory_cells<F: std::future::Future>(
    cells: Option<Arc<[u32]>>,
    scan: F,
) -> F::Output {
    match cells {
        Some(cells) => DIRECTORY_CELLS.scope(cells, scan).await,
        None => scan.await,
    }
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
    // TD-PAXDIR-1 D2: the collection directory already ranked this segment's
    // cells against every other segment's; probe exactly its choice (valid,
    // non-empty, distinct cells only). Filtered queries never carry a
    // directory choice.
    let directory_cells: Option<Vec<usize>> = DIRECTORY_CELLS
        .try_with(|cells| {
            cells
                .iter()
                .map(|&c| c as usize)
                .filter(|&c| c < k_c && dir.cells[c].row_end > dir.cells[c].row_begin)
                .collect::<std::collections::BTreeSet<usize>>()
                .into_iter()
                .collect::<Vec<usize>>()
        })
        .ok()
        .filter(|cells| row_allow.is_none() && !cells.is_empty());
    let mut probed: Vec<usize> = match row_allow {
        // TD-FPRUNE-1 M3: FILTERED ANN — walk cells nearest-first and accumulate
        // PREDICATE-MATCHING rows (allow ∩ cell row-range, computed in RAM from
        // the cell directory + the allow-set — no extra I/O). Probe until at least
        // the geometric `nprobe` cells AND enough matching candidates to fill the
        // survivor pool are covered: a selective filter needs MORE cells to reach
        // k matches (the recall fix that the disarmed-probe path traded away).
        // Cells with zero matching rows are skipped (cell ∩ filter empty ⇒ no GET).
        Some(allow) => {
            // TD-SECSCAN-1 §5: probing stops at the pool (or the allowed
            // rows — there is nothing more to match).
            let target = pax_rabitq_pool_for_top_k(k, allow.len().max(1))
                .max(k)
                .min(allow.len());
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
    if let Some(cells) = directory_cells {
        probed = cells;
    }
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
    // TD-SECSCAN-1 §5: callers may pass `pool.max(k)` with a client-sized `k`;
    // the pool can never select more survivors than the region has rows.
    let pool = pool.min(n_rows);
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
    let mut merged: Vec<(usize, f32)> =
        Vec::with_capacity(morsel_merge_capacity(pool, n_rows, degree));
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
                let pool = pax_rabitq_pool_for_top_k(k, allow.len().max(1))
                    .max(k)
                    .min(allow.len().max(1));
                region.rank_allowed(query, metric, pool, allow)
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
#[path = "segment_format_tests.rs"]
mod tests;
