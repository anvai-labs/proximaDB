use crate::core::search::BlockPruneConfig;
use crate::storage::engines::core::constants::pruning as pruning_constants;
use crate::storage::engines::sst::IndexEntry;
use proximadb_distance_kernel::{DistanceMetric, engine::UnifiedDistanceCompute};
use tracing::debug;

/// Compute Z-Order code for a query vector using PCA transform.
///
/// This transforms the query to PCA space and encodes it with Z-Order,
/// enabling spatial range-based block pruning.
///
/// # Arguments
/// * `query` - Query vector (original dimension)
/// * `entries` - Index entries with block centroids
///
/// # Returns
/// Z-Order code for the query, or None if insufficient data
pub(crate) fn compute_query_zorder_code(
    query: &[f32],
    _entries: &[IndexEntry],
    collection_id: &str,
) -> Option<crate::storage::engines::core::formats::proximablocks::spatial_encoding::SpatialCode> {
    use crate::storage::engines::core::formats::proximablocks::spatial_clustering::{
        AdaptivePcaConfig, ZOrderEncoder,
    };

    let pca_model = crate::storage::engines::sst::core::get_collection_pca_model(collection_id)?;
    let projected = pca_model.project(query).ok()?;

    let config = AdaptivePcaConfig::for_vector_dim(query.len());
    let n_dimensions = pca_model.n_components.min(config.n_components);
    let encoder = ZOrderEncoder::new(n_dimensions, config.bits_per_dim);

    let coords: Vec<f32> = projected.into_iter().take(n_dimensions).collect();
    let coords = if coords.len() < n_dimensions {
        let mut padded = coords;
        padded.resize(n_dimensions, 0.0);
        padded
    } else {
        coords
    };

    let normalized: Vec<f32> = coords
        .iter()
        .map(|&v| {
            let clamped = v.clamp(-10.0, 10.0);
            (clamped + 10.0) / 20.0
        })
        .collect();

    Some(encoder.encode(&normalized))
}

/// Calculate Z-Order epsilon for pruning range.
pub(crate) fn calculate_zorder_epsilon(
    query_code: &crate::storage::engines::core::formats::proximablocks::spatial_encoding::SpatialCode,
    entries: &[IndexEntry],
) -> crate::storage::engines::core::formats::proximablocks::spatial_encoding::SpatialCode {
    use crate::storage::engines::core::formats::proximablocks::spatial_encoding::SpatialCode;

    let codes: Vec<&SpatialCode> = entries
        .iter()
        .filter_map(|e| e.zorder_code.as_ref())
        .collect();
    if codes.is_empty() {
        return match query_code {
            SpatialCode::Code64(_) => SpatialCode::Code64(u64::MAX),
            SpatialCode::Code128(_) => SpatialCode::Code128(u128::MAX),
            SpatialCode::Code256 { .. } => SpatialCode::Code256 {
                low: u128::MAX,
                high: u128::MAX,
            },
            SpatialCode::Code512(_) => SpatialCode::Code512(
                crate::storage::engines::core::formats::proximablocks::spatial_encoding::U512::MAX,
            ),
        };
    }

    let Some(min_code) = codes.iter().min() else {
        return query_code.clone();
    };
    let Some(max_code) = codes.iter().max() else {
        return query_code.clone();
    };

    max_code.epsilon(min_code, 10.0, 1000)
}

#[allow(dead_code)]
pub(crate) fn filter_blocks_by_zorder(
    query: &[f32],
    entries: &[IndexEntry],
    collection_id: &str,
) -> Option<Vec<usize>> {
    let query_code = compute_query_zorder_code(query, entries, collection_id)?;
    let epsilon = calculate_zorder_epsilon(&query_code, entries);
    let min_code = query_code.saturating_sub(&epsilon);
    let max_code = query_code.saturating_add(&epsilon);

    let filtered_indices: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter(|(_, entry)| {
            if let Some(code) = &entry.zorder_code {
                code.in_range(&min_code, &max_code)
            } else {
                true
            }
        })
        .map(|(idx, _)| idx)
        .collect();

    let pruned_percentage = if !entries.is_empty() {
        100 - (filtered_indices.len() * 100 / entries.len())
    } else {
        0
    };

    debug!(
        "🔬 SST Z-Order Pruning: {} → {} blocks ({}% pruned)",
        entries.len(),
        filtered_indices.len(),
        pruned_percentage
    );

    Some(filtered_indices)
}

#[allow(dead_code)]
pub(crate) fn normalize_coords_for_zorder(coords: &[f32]) -> Vec<f32> {
    if coords.is_empty() {
        return Vec::new();
    }

    let min_val = coords.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = coords.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max_val - min_val;

    if range < 1e-6 {
        return vec![0.5; coords.len()];
    }

    coords
        .iter()
        .map(|&c| ((c - min_val) / range).clamp(0.0, 1.0))
        .collect()
}

pub(crate) fn select_blocks_by_centroid(
    query: &[f32],
    entries: &[IndexEntry],
    metric: DistanceMetric,
    prune: &BlockPruneConfig,
) -> Vec<usize> {
    if prune.force_exact {
        return (0..entries.len()).collect();
    }

    let min_blocks_threshold = prune
        .min_blocks_override
        .unwrap_or(pruning_constants::MIN_BLOCKS_FOR_PRUNING);

    if entries.len() < min_blocks_threshold {
        debug!(
            "Block pruning skipped: {} blocks < {} threshold",
            entries.len(),
            min_blocks_threshold
        );
        return (0..entries.len()).collect();
    }

    let mut scored = Vec::with_capacity(entries.len());

    for (idx, entry) in entries.iter().enumerate() {
        let centroid = crate::storage::engines::sst::get_centroid_fp32(
            &entry.block_centroid_fp16,
            &entry.block_centroid,
        );
        if centroid.len() != query.len() {
            scored.push((f32::INFINITY, idx));
            continue;
        }
        let dist = metric_distance(query, &centroid, metric);
        // TD-RDSTRAT-5 lever-3: rank by the distance LOWER BOUND
        // `d(query, centroid) − k·radius` — the closest a point in this block
        // could be (triangle inequality). `radius_k = 0` ⇒ raw centroid distance
        // (legacy). `> 0` keeps spread-out blocks that a center-only rank would
        // wrongly prune, raising recall at a fixed keep-ratio.
        let score = dist - prune.radius_k * entry.block_radius;
        scored.push((score, idx));
    }

    if scored.is_empty() {
        return Vec::new();
    }

    let mut keep = match prune.mode {
        crate::core::search::BlockPruneMode::Sqrt => (scored.len() as f32).sqrt().ceil() as usize,
        crate::core::search::BlockPruneMode::Ratio => {
            let r = prune.ratio.clamp(0.0, 1.0);
            ((scored.len() as f32) * r).ceil() as usize
        }
        crate::core::search::BlockPruneMode::Fixed(k) => k,
    };

    keep = keep.max(prune.min_keep);
    if prune.max_keep > 0 {
        keep = keep.min(prune.max_keep);
    }
    keep = keep.clamp(1, scored.len());

    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut selected: Vec<usize> = scored.into_iter().take(keep).map(|(_, idx)| idx).collect();
    selected.sort_unstable();
    selected.dedup();
    selected
}

pub(crate) fn metric_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    let distance_compute = UnifiedDistanceCompute::default();
    distance_compute.distance_with_metric(a, b, &metric)
}

#[cfg(test)]
mod td096_block_skip_tests {
    //! TD-096 S1: deterministic proof of the sqrt centroid-pruning block-skip
    //! ratio — the mechanism behind the historical ~5% Exact-mode recall
    //! collapse (now bypassed for `SearchMode::Exact` via `force_exact`; see
    //! `src/storage/engines/sst/search/mod.rs` and the `BlockPruneConfig` docs).
    //! These pin the pruning behavior so a regression that re-introduces the
    //! collapse is caught without a heavyweight recall bench.
    use super::*;
    use crate::core::search::BlockPruneMode;

    /// `n` orthogonal unit basis vectors in `dim` dimensions (e_0..e_{n-1}) —
    /// distinct directions so Cosine ranking is unambiguous (e_i·e_j = δ_ij).
    fn basis_centroids(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                v[i] = 1.0;
                v
            })
            .collect()
    }

    fn entries_with_centroids(centroids: &[Vec<f32>]) -> Vec<IndexEntry> {
        centroids
            .iter()
            .map(|c| IndexEntry {
                block_centroid: c.clone(),
                block_centroid_fp16: None,
                ..Default::default()
            })
            .collect()
    }

    /// At N≥threshold blocks, Sqrt mode keeps only ~sqrt(N) — the aggressive
    /// skip that, pre-`force_exact`, dropped Exact recall to ~5% at scale.
    #[test]
    fn sqrt_pruning_keeps_approximately_sqrt_n_blocks() {
        // 100 orthogonal centroids (Cosine ranking is unambiguous: e_i·e_j=δ).
        let entries = entries_with_centroids(&basis_centroids(100, 100));
        let mut cfg = BlockPruneConfig::for_testing(); // bypass the 100-block threshold
        cfg.mode = BlockPruneMode::Sqrt;
        let mut query = vec![0.0f32; 100];
        query[50] = 1.0; // == e_50, the nearest centroid by Cosine
        let selected = select_blocks_by_centroid(&query, &entries, DistanceMetric::Cosine, &cfg);
        assert_eq!(
            selected.len(),
            10,
            "Sqrt mode must keep sqrt(100)=10 of 100 blocks"
        );
        // The nearest block (centroid e_50) must survive the prune.
        assert!(selected.contains(&50));
    }

    /// `force_exact` bypasses pruning entirely — the override that eliminated
    /// the 5% collapse for `SearchMode::Exact`.
    #[test]
    fn force_exact_keeps_all_blocks() {
        let entries = entries_with_centroids(&basis_centroids(50, 50));
        let mut cfg = BlockPruneConfig::for_testing();
        cfg.force_exact = true;
        // Direction is irrelevant under force_exact (no scoring).
        let query = vec![1.0f32; 50];
        let selected = select_blocks_by_centroid(&query, &entries, DistanceMetric::Cosine, &cfg);
        assert_eq!(
            selected.len(),
            50,
            "force_exact must keep every block (no pruning)"
        );
    }

    /// Below the production MIN_BLOCKS_FOR_PRUNING threshold (and no override),
    /// no pruning happens — the small-collection path is always exact.
    #[test]
    fn below_threshold_no_pruning() {
        let entries = entries_with_centroids(&basis_centroids(10, 10));
        let cfg = BlockPruneConfig::default(); // min_blocks_override=None → threshold=100
        let query = vec![1.0f32; 10];
        let selected = select_blocks_by_centroid(&query, &entries, DistanceMetric::Cosine, &cfg);
        assert_eq!(
            selected.len(),
            10,
            "below MIN_BLOCKS_FOR_PRUNING, all blocks are kept"
        );
    }
}

#[cfg(test)]
mod wlp3_radius_prune_tests {
    //! TD-WLP-3 (ADR-061 D7): deterministic gates for the radius lower-bound
    //! prune score `d(q, centroid) − k·radius`. The default `radius_k = 0.0`
    //! must reproduce the legacy centroid-only selection EXACTLY (default
    //! safety); a positive `k` must rescue a spread-out block that a
    //! center-only rank would wrongly prune (the measured recall lever).
    use super::*;
    use crate::core::search::BlockPruneMode;

    /// Blocks on a line: centroid_i = [i, 0, 0, 0] with the given per-block
    /// radii, so L2 distance from an origin query ranks them by index.
    fn line_entries(radii: &[f32]) -> Vec<IndexEntry> {
        radii
            .iter()
            .enumerate()
            .map(|(i, &radius)| IndexEntry {
                block_centroid: vec![i as f32, 0.0, 0.0, 0.0],
                block_centroid_fp16: None,
                block_radius: radius,
                ..Default::default()
            })
            .collect()
    }

    /// `radius_k = 0.0` (the default) reproduces the legacy centroid-only
    /// selection exactly, even when radii vary wildly — proves the ADR-061
    /// default-OFF safety property at the scoring seam.
    #[test]
    fn radius_k_zero_reproduces_legacy_selection() {
        // Wildly varying radii that WOULD reorder the ranking if consulted.
        let entries = line_entries(&[0.0, 0.0, 100.0, 50.0, 25.0, 12.0, 6.0, 3.0]);
        let mut cfg = BlockPruneConfig::for_testing();
        cfg.mode = BlockPruneMode::Fixed(3);
        assert_eq!(cfg.radius_k, 0.0, "default radius_k must be 0.0 (legacy)");
        let query = vec![0.0f32; 4];
        let selected = select_blocks_by_centroid(&query, &entries, DistanceMetric::Euclidean, &cfg);
        assert_eq!(
            selected,
            vec![0, 1, 2],
            "radius_k=0.0 must rank by raw centroid distance only (legacy \
             selection), ignoring radii entirely"
        );
    }

    /// A positive `radius_k` rescues a far-but-spread-out block whose distance
    /// LOWER BOUND beats a nearer tight block — the exact failure mode behind
    /// the measured 0.99→0.81 pruned-recall drop (ADR-061 §Context).
    #[test]
    fn radius_k_positive_rescues_spread_out_block() {
        // Block 0: d=0, tight. Block 1: d=1, tight. Block 2: d=2, radius 1.8
        // → lower bound 0.2, undercutting block 1's 1.0.
        let entries = line_entries(&[0.0, 0.0, 1.8]);
        let mut cfg = BlockPruneConfig::for_testing();
        cfg.mode = BlockPruneMode::Fixed(2);
        let query = vec![0.0f32; 4];

        // Legacy (k=0): keeps the two nearest centroids {0, 1}.
        let legacy = select_blocks_by_centroid(&query, &entries, DistanceMetric::Euclidean, &cfg);
        assert_eq!(legacy, vec![0, 1], "k=0 keeps the two nearest centroids");

        // k=1: block 2's lower bound (2 − 1.8 = 0.2) outranks block 1 (1.0).
        cfg.radius_k = 1.0;
        let with_radius =
            select_blocks_by_centroid(&query, &entries, DistanceMetric::Euclidean, &cfg);
        assert_eq!(
            with_radius,
            vec![0, 2],
            "k=1 must keep the spread-out block whose distance lower bound \
             undercuts a nearer tight block"
        );
    }
}

#[cfg(test)]
mod zorder_tests {
    //! Z-Order spatial-code coverage, ported from the orphaned (never
    //! compiled) sst/readers/tests.rs before its deletion — these are the
    //! only tests of compute_query_zorder_code / calculate_zorder_epsilon /
    //! filter_blocks_by_zorder / normalize_coords_for_zorder (of these,
    //! sst_query_engine currently exercises compute_query_zorder_code; the
    //! rest are #[allow(dead_code)] pending wider query-side adoption).
    use super::*;
    use crate::storage::engines::core::formats::proximablocks::spatial_encoding::SpatialCode;

    /// Compact entry builder (the port also drops the ~20-line exhaustive
    /// IndexEntry literals the orphaned file repeated per test).
    fn zorder_entry(
        block_id: u32,
        centroid: Vec<f32>,
        zorder_code: Option<SpatialCode>,
    ) -> IndexEntry {
        IndexEntry {
            key: format!("block_{block_id}"),
            block_id,
            block_centroid: centroid,
            zorder_code,
            ..Default::default()
        }
    }

    /// Trains and caches a 2-component PCA model for `collection`, the
    /// precondition for the query-side Z-Order code paths.
    fn cache_pca_model(collection: &str) {
        use crate::proto::proximadb_v1::VectorRecord;
        use crate::storage::engines::sst::pca_manager::EnhancedPCAModel;

        let vectors: Vec<VectorRecord> = (0..100)
            .map(|i| VectorRecord {
                id: format!("vec_{i}"),
                vector: vec![i as f32 / 100.0, i as f32 / 50.0],
                ..Default::default()
            })
            .collect();
        let model = EnhancedPCAModel::train(&vectors, 2).expect("train PCA");
        crate::storage::engines::sst::core::set_collection_pca_model(collection, model);
    }

    #[test]
    fn compute_query_zorder_code_without_pca_model_returns_none() {
        let entries = vec![
            zorder_entry(0, vec![0.0, 0.0], Some(SpatialCode::Code64(100))),
            zorder_entry(1, vec![1.0, 1.0], Some(SpatialCode::Code64(200))),
        ];
        // No cached PCA model → graceful fallback to centroid-only pruning.
        let code = compute_query_zorder_code(&[1.0, 0.5], &entries, "zorder_no_pca");
        assert!(
            code.is_none(),
            "should return None when no PCA model is cached"
        );
    }

    #[test]
    fn compute_query_zorder_code_with_pca_model_returns_some() {
        cache_pca_model("zorder_with_pca");
        let entries = vec![zorder_entry(
            0,
            vec![0.0, 0.0],
            Some(SpatialCode::Code64(100)),
        )];
        let code = compute_query_zorder_code(&[1.0, 0.5], &entries, "zorder_with_pca");
        assert!(
            code.is_some(),
            "should compute a Z-Order code when PCA is cached"
        );
    }

    #[test]
    fn compute_query_zorder_code_empty_entries_returns_none() {
        let code = compute_query_zorder_code(&[1.0, 0.5], &[], "zorder_empty");
        assert!(code.is_none(), "should return None for empty entries");
    }

    #[test]
    fn calculate_zorder_epsilon_is_max_of_ten_percent_and_floor() {
        let entries = vec![
            zorder_entry(0, vec![0.0, 0.0], Some(SpatialCode::Code64(1000))),
            zorder_entry(1, vec![10.0, 10.0], Some(SpatialCode::Code64(10000))),
        ];
        // 10% of range = 900, but the floor is 1000 → 1000 wins.
        let epsilon = calculate_zorder_epsilon(&SpatialCode::Code64(5000), &entries);
        assert_eq!(epsilon, SpatialCode::Code64(1000));
    }

    #[test]
    fn calculate_zorder_epsilon_no_codes_is_max() {
        let entries = vec![zorder_entry(0, vec![0.0, 0.0], None)];
        let epsilon = calculate_zorder_epsilon(&SpatialCode::Code64(0), &entries);
        assert_eq!(epsilon, SpatialCode::Code64(u64::MAX));
    }

    #[test]
    fn filter_blocks_by_zorder_without_pca_returns_none() {
        let entries = vec![
            zorder_entry(0, vec![0.0, 0.0], Some(SpatialCode::Code64(100))),
            zorder_entry(1, vec![1.0, 1.0], Some(SpatialCode::Code64(5000))),
            zorder_entry(2, vec![10.0, 10.0], Some(SpatialCode::Code64(10000))),
        ];
        let filtered = filter_blocks_by_zorder(&[1.0, 1.0], &entries, "zorder_filter_no_pca");
        assert!(
            filtered.is_none(),
            "should fall back to centroid pruning without PCA"
        );
    }

    #[test]
    fn filter_blocks_by_zorder_with_pca_includes_uncoded_blocks() {
        cache_pca_model("zorder_filter_pca");
        let entries = vec![
            zorder_entry(0, vec![0.0, 0.0], Some(SpatialCode::Code64(100))),
            zorder_entry(1, vec![1.0, 1.0], Some(SpatialCode::Code64(5000))),
            zorder_entry(2, vec![10.0, 10.0], Some(SpatialCode::Code64(10000))),
            // No Z-Order code → always included (backward compatibility).
            zorder_entry(3, vec![0.5, 0.5], None),
        ];
        let filtered = filter_blocks_by_zorder(&[1.0, 1.0], &entries, "zorder_filter_pca")
            .expect("cached PCA model enables Z-Order filtering");
        assert!(
            filtered.contains(&3),
            "block without a Z-Order code must be included"
        );
    }

    #[test]
    fn filter_blocks_by_zorder_backward_compat_mixed_entries() {
        cache_pca_model("zorder_backward_compat");
        let entries = vec![
            zorder_entry(0, vec![0.0, 0.0], None),
            zorder_entry(1, vec![1.0, 1.0], Some(SpatialCode::Code64(5000))),
        ];
        let filtered = filter_blocks_by_zorder(&[1.0, 1.0], &entries, "zorder_backward_compat")
            .expect("mixed coded/uncoded blocks are supported");
        assert!(filtered.contains(&0), "uncoded block must be included");
    }

    #[test]
    fn normalize_coords_for_zorder_maps_to_unit_interval() {
        let normalized = normalize_coords_for_zorder(&[-10.0f32, 0.0, 10.0, 20.0]);
        assert_eq!(normalized.len(), 4);
        for &val in &normalized {
            assert!(val >= 0.0 && val <= 1.0, "{val} not in [0, 1]");
        }
        assert!((normalized[0] - 0.0).abs() < 1e-5, "min maps to 0.0");
        assert!((normalized[3] - 1.0).abs() < 1e-5, "max maps to 1.0");
    }

    #[test]
    fn normalize_coords_for_zorder_constant_maps_to_midpoint() {
        let normalized = normalize_coords_for_zorder(&[5.0f32, 5.0, 5.0]);
        for &val in &normalized {
            assert!(
                (val - 0.5).abs() < 1e-5,
                "constant coords map to 0.5, got {val}"
            );
        }
    }
}
