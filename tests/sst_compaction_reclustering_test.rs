//! TD-WLP-4 (ADR-061 D3) integration gate: compaction re-clusters with
//! PCA+IVF on fp32, measurably improving intra-block coherence over the L0
//! sign-code bootstrap.
//!
//! Coherence is asserted through the per-block RMS radii (TD-WLP-3): the
//! compacted write's mean block radius must undercut the bootstrap write's on
//! magnitude-separated clusters — the exact geometry the sign-bit bootstrap
//! cannot separate (identical sign patterns) but fp32-PCA k-means can.
//! Clustering runs on PCA-projected fp32, never on quantized codes (D3/A3).

use proximadb::storage::engines::sst::segment_format::{
    write_pax_segment_compacted, write_pax_segment_full,
};
use proximadb_block_format::VectorQuant;
use proximadb_records::{EmbeddingCell, EmbeddingValues, ProximaRecord};

const DIM: usize = 8;
const CLUSTERS: usize = 4;
const PER_CLUSTER: usize = 128;
/// `target_block` is a byte threshold the writer compares against
/// `row_count * 1024`, so `64 * 1024` cuts blocks at 64 rows.
const ROWS_PER_BLOCK_TARGET: usize = 64;

fn record(oid: &str, v: Vec<f32>) -> ProximaRecord {
    ProximaRecord {
        oid: oid.to_string(),
        created_at_ns: 1,
        updated_at_ns: 1,
        record_version: 1,
        embeddings: vec![EmbeddingCell::new_fp32(
            "test",
            "dense_vector",
            DIM as u32,
            v,
        )],
        ..ProximaRecord::default()
    }
}

/// Interleaved records from `CLUSTERS` tight, magnitude-separated clusters —
/// all-positive coordinates, so every vector has the SAME sign pattern and the
/// sign-code bootstrap cannot tell clusters apart. Centers 0/10/20/30 with a
/// ±0.05 deterministic jitter keep the clusters far apart relative to spread.
fn interleaved_records() -> Vec<ProximaRecord> {
    let mut recs = Vec::with_capacity(CLUSTERS * PER_CLUSTER);
    for i in 0..PER_CLUSTER {
        for c in 0..CLUSTERS {
            let center = 10.0 * c as f32 + 1.0;
            let jitter = ((i * 7 + c * 3) % 11) as f32 * 0.01 - 0.05;
            recs.push(record(&format!("c{c}_r{i:03}"), vec![center + jitter; DIM]));
        }
    }
    recs
}

fn mean_radius(radii: &[f32]) -> f32 {
    if radii.is_empty() {
        return f32::INFINITY;
    }
    radii.iter().sum::<f32>() / radii.len() as f32
}

/// The compacted (PCA+IVF) write produces non-empty centroids+radii and
/// strictly tighter blocks than the L0 bootstrap on sign-degenerate data.
#[tokio::test]
async fn test_append_compaction_reclusters_and_improves_coherence() {
    unsafe {
        // Clustering is default-ON (TD-WLP-4); pin the kill-switch off in case
        // the ambient env set it.
        std::env::remove_var("PROXIMADB_PAX_BLOCK_CLUSTER");
        // This gate measures *intra-block* vector coherence (block_radii).
        // Coalesced-RaBitQ (ADR-065, default-ON) hoists the vectors out of the
        // blocks into Region B, so block-level radii no longer reflect vector
        // clustering. Disable coalescing here so vectors stay in the blocks —
        // the layout this metric is defined on (and the one develop ran under).
        // Coalesced coherence is covered by the SIFT recall gate instead.
        std::env::set_var("PROXIMADB_PAX_COALESCED_RABITQ", "0");
    }
    let records = interleaved_records();
    let dir = tempfile::tempdir().expect("tempdir");

    let bootstrap_meta = write_pax_segment_full(
        &dir.path().join("l0_bootstrap.pax"),
        &records,
        "wlp4_recluster",
        records.len(),
        VectorQuant::RaBitQ,
        VectorQuant::Sq8,
        false,
        Some(ROWS_PER_BLOCK_TARGET * 1024),
        None,
    )
    .expect("bootstrap write");

    let compacted_meta = write_pax_segment_compacted(
        &dir.path().join("compacted.pax"),
        &records,
        "wlp4_recluster",
        records.len(),
        VectorQuant::RaBitQ,
        VectorQuant::Sq8,
        false,
        Some(ROWS_PER_BLOCK_TARGET * 1024),
        None,
    )
    .expect("compacted write");

    // Both writes carry the VOE inputs (centroids + radii, 1:1).
    for (name, meta) in [
        ("bootstrap", &bootstrap_meta),
        ("compacted", &compacted_meta),
    ] {
        assert!(
            !meta.block_centroids.is_empty(),
            "{name}: default-ON clustering must emit block centroids"
        );
        assert_eq!(
            meta.block_radii.len(),
            meta.block_centroids.len(),
            "{name}: radii 1:1 with centroids"
        );
    }

    // Coherence: the sign-code bootstrap cannot separate magnitude-only
    // clusters (mixed blocks ⇒ radii ~ inter-cluster distance); PCA+IVF can
    // (per-cluster blocks ⇒ radii ~ jitter). Require a decisive 5× margin so
    // the gate fails loudly if the IVF order regresses to the bootstrap.
    let bootstrap_r = mean_radius(&bootstrap_meta.block_radii);
    let compacted_r = mean_radius(&compacted_meta.block_radii);
    assert!(
        compacted_r * 5.0 < bootstrap_r,
        "compacted mean block radius must decisively undercut the bootstrap \
         (bootstrap={bootstrap_r}, compacted={compacted_r})"
    );

    // Result-preservation: the reorder is physical only — same record count.
    assert_eq!(compacted_meta.row_count, bootstrap_meta.row_count);
    assert_eq!(compacted_meta.row_count as usize, records.len());
}

/// The kill-switch restores insertion-order writes with no centroids for BOTH
/// entry points (the one escape hatch — no per-path flag zoo).
#[tokio::test]
async fn test_cluster_kill_switch_disables_centroids() {
    unsafe {
        std::env::set_var("PROXIMADB_PAX_BLOCK_CLUSTER", "0");
    }
    let records = interleaved_records();
    let dir = tempfile::tempdir().expect("tempdir");
    let meta = write_pax_segment_compacted(
        &dir.path().join("killed.pax"),
        &records,
        "wlp4_kill",
        records.len(),
        VectorQuant::RaBitQ,
        VectorQuant::Sq8,
        false,
        Some(ROWS_PER_BLOCK_TARGET * 1024),
        None,
    )
    .expect("write with kill-switch");
    assert!(
        meta.block_centroids.is_empty(),
        "kill-switch must skip clustering + centroid emission"
    );
    unsafe {
        std::env::remove_var("PROXIMADB_PAX_BLOCK_CLUSTER");
    }
}
