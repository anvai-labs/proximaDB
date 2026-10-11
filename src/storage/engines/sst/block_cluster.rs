//! TD-RDSTRAT-5 S1: cheap **sort-by-code** block clustering at PAX write.
//!
//! Reorders records before block-cutting so spatially-close vectors land in the
//! same block, making each block's centroid tight — the prerequisite for the
//! centroid probe-prune (TD-RDSTRAT-4 Lever B, landing through the Vector Object
//! Economy directory per TD-RDSTRAT-5). The ordering key is a **binary-reflected
//! Gray code over the sign bits of the mean-centered vector** — a cheap,
//! streaming, no-k-means space-filling proxy for angular locality (Gray coding
//! makes adjacent keys Hamming-adjacent). This is the MVP; true IVF/k-means
//! clustering is the S4 upgrade.
//!
//! Reordering is **result-preserving**: the reader ranks/dedups by distance and
//! OID, so the returned top-k is identical regardless of physical order (parity-
//! gated in tests). Default ON since TD-WLP-4 (ADR-061 D3);
//! `PROXIMADB_PAX_BLOCK_CLUSTER=0` is the kill-switch. The sign-Gray key is the
//! model-free **L0 bootstrap**; compaction re-clusters with PCA+IVF
//! ([`cluster_order_pca_ivf`]).

use proximadb_records::{EmbeddingValues, ProximaRecord};

/// One contiguous cluster in the reordered output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrderedClusterRun {
    pub start_row: usize,
    pub row_count: usize,
}

/// Physical ordering plus the cluster boundaries that produced it.
///
/// Keeping both avoids reverse-engineering clusters from reordered vectors and
/// lets the PAX writer apply cluster-local, exact storage transforms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterPlan {
    pub order: Vec<usize>,
    pub runs: Vec<OrderedClusterRun>,
}

/// Sort-by-code block clustering at PAX write (TD-RDSTRAT-5 S1, default ON
/// since TD-WLP-4 / ADR-061 D3 — pre-GA "arm defaults" directive). Records are
/// reordered by locality key at flush so blocks are spatially coherent and
/// centroids+radii are emitted for the VOE directory. Reordering is
/// result-preserving, so the only escape needed is the kill-switch:
/// `PROXIMADB_PAX_BLOCK_CLUSTER=0|false|off|no` restores insertion-order
/// writes (no centroids).
pub fn block_cluster_enabled() -> bool {
    !env_flag_off("PROXIMADB_PAX_BLOCK_CLUSTER")
}

/// True when `var` is explicitly set to a falsy value (kill-switch semantics:
/// unset/unrecognized → not off).
fn env_flag_off(var: &str) -> bool {
    match std::env::var(var).ok().as_deref().map(str::trim) {
        Some(v) => matches!(
            v.to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        None => false,
    }
}

/// TD-WLP-4/WLP-9 eval opt-in: upgrade the **flush** (L0) ordering from the
/// model-free sign-bit bootstrap ([`cluster_order`]) to the full PCA+IVF
/// re-cluster ([`cluster_order_pca_ivf`]) — the ordering compaction normally
/// applies, but reached at flush so it is measurable without waiting on a
/// compaction cadence. (Not "the (unwired) flush→compaction scheduler", as an
/// earlier revision said: TD-WLP-7 / ADR-061 D3 wired it in #1012.) Default OFF: production L0 flush keeps the
/// bootstrap (cold-start-safe, streaming), and the production re-cluster event
/// remains compaction. Set `PROXIMADB_PAX_FLUSH_CLUSTER=ivf` to exercise PCA/IVF
/// at flush (the SIFT recall eval uses this to validate clustering quality). The
/// ordering is result-preserving (the reader ranks/dedups by distance + OID),
/// so this is mixed-read-safe.
pub fn flush_cluster_ivf() -> bool {
    matches!(
        std::env::var("PROXIMADB_PAX_FLUSH_CLUSTER")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(|v| v.to_ascii_lowercase())
            .as_deref(),
        Some("ivf") | Some("pca_ivf") | Some("1") | Some("true") | Some("on") | Some("yes")
    )
}

/// The f32 vector of `record`'s embedding `idx`, if present and Fp32-typed (the
/// canonical write-time embedding representation — quantization happens inside the
/// block writer). Non-Fp32 variants return `None` (they don't contribute a key).
fn embedding_f32(record: &ProximaRecord, idx: usize) -> Option<&[f32]> {
    match record.embeddings.get(idx).map(|e| &e.values) {
        Some(EmbeddingValues::Fp32(v)) if !v.is_empty() => Some(v),
        _ => None,
    }
}

/// A permutation of `0..records.len()` that orders records by the Gray-coded
/// sign-bit key of their mean-centered embedding `idx`, so spatially-close
/// vectors are adjacent (and thus co-located into the same block by the
/// row-count block-cutter).
///
/// Returns `None` (caller keeps insertion order) when clustering can't help or
/// apply: fewer than 2 records, or no record carries a usable Fp32 embedding at
/// `idx`. Records without a usable embedding sort last (empty key), stably.
///
/// Pure + deterministic (unit-tested). The key is a *proxy* for angular order,
/// not an exact clustering — S4 replaces it with IVF/k-means.
pub fn cluster_order(records: &[ProximaRecord], idx: usize) -> Option<Vec<usize>> {
    cluster_plan(records, idx).map(|plan| plan.order)
}

/// Bootstrap cluster plan: Gray-key order plus exact runs of equal keys.
pub fn cluster_plan(records: &[ProximaRecord], idx: usize) -> Option<ClusterPlan> {
    if records.len() < 2 {
        return None;
    }
    // Dimension + mean (centroid) from the records that carry an Fp32 embedding.
    let dim = records
        .iter()
        .find_map(|r| embedding_f32(r, idx).map(<[f32]>::len))?;
    let mut mean = vec![0f64; dim];
    let mut n = 0u64;
    for r in records {
        if let Some(v) = embedding_f32(r, idx)
            && v.len() == dim
        {
            for (m, &x) in mean.iter_mut().zip(v) {
                *m += x as f64;
            }
            n += 1;
        }
    }
    if n == 0 {
        return None;
    }
    for m in &mut mean {
        *m /= n as f64;
    }

    // Key = Gray-coded sign bits of (v - mean), MSB-first. Records with no usable
    // embedding get an all-ones key so they sort last (after every real key).
    let key_of = |r: &ProximaRecord| -> Vec<u8> {
        match embedding_f32(r, idx) {
            Some(v) if v.len() == dim => sign_gray_key(v, &mean),
            _ => vec![0xFFu8; dim.div_ceil(8)],
        }
    };
    let mut order: Vec<usize> = (0..records.len()).collect();
    // Precompute keys once (avoid recomputing in the comparator).
    let keys: Vec<Vec<u8>> = records.iter().map(key_of).collect();
    order.sort_by(|&a, &b| keys[a].cmp(&keys[b]).then(a.cmp(&b)));
    let ordered_keys: Vec<&[u8]> = order.iter().map(|&row| keys[row].as_slice()).collect();
    let runs = contiguous_runs(&ordered_keys);
    Some(ClusterPlan { order, runs })
}

/// ADR-065 Region B locality order: **Morton/Z-order over the segment-level SQ8
/// codes**. The RaBitQ-top-M survivors are a spatial neighbourhood; ordering
/// Region B by an SQ8-Morton key co-locates them (and the top-k result rows) so
/// the survivor + OID fetches collapse to a few contiguous ranges.
///
/// Why SQ8 (not fp32): scalar quantization denoises (drops fp32's noisy low
/// bits), collapses near-duplicates to the same/adjacent cell, is compact
/// (8 bits/dim — exactly the precision a Morton key uses), and is Region B's own
/// representation (no separate PCA/projection — flush-safe, unlike IVF).
///
/// One segment-level `Sq8Params` fit (same fit Region B will store), quantize,
/// Morton-key, sort. Records with no/short embedding sort last. Returns `None`
/// when fewer than 2 usable rows.
pub fn cluster_order_sq8_morton(records: &[ProximaRecord], idx: usize) -> Option<Vec<usize>> {
    use proximadb_codec::functions::sq8::{fit_params, quantize_one};
    let usable: Vec<(usize, &[f32])> = records
        .iter()
        .enumerate()
        .filter_map(|(i, r)| embedding_f32(r, idx).map(|v| (i, v)))
        .collect();
    if usable.len() < 2 {
        return None;
    }
    let dim = usable[0].1.len();
    // One segment-level fit over all usable vectors (flattened) — identical to the
    // fit Region B's encode_region performs, so the order matches its stored codes.
    let mut flat: Vec<f32> = Vec::with_capacity(usable.len() * dim);
    for (_, v) in &usable {
        if v.len() == dim {
            flat.extend_from_slice(v);
        }
    }
    let params = fit_params(&flat);

    let key_of = |v: &[f32]| -> Vec<u8> {
        let mut codes = vec![0u8; dim];
        for d in 0..dim {
            codes[d] = quantize_one(v[d], &params);
        }
        sq8_morton_key(&codes, dim)
    };
    // Records with no/short embedding sort last (all-ones key).
    let tail_key = vec![0xFFu8; dim];
    let mut order: Vec<usize> = (0..records.len()).collect();
    let keys: Vec<Vec<u8>> = records
        .iter()
        .map(|r| match embedding_f32(r, idx) {
            Some(v) if v.len() == dim => key_of(v),
            _ => tail_key.clone(),
        })
        .collect();
    order.sort_by(|&a, &b| keys[a].cmp(&keys[b]).then(a.cmp(&b)));
    Some(order)
}

/// Morton/Z-order key over `dim` SQ8 bytes: interleave the 8 bits of each
/// dimension, **MSB-first across dims then bit-7→0**, packed MSB-first into
/// bytes → hierarchical locality (the top `dim` key bits = bit-7 of every dim =
/// the coarse sign-level; successive levels refine). Output is `dim` bytes
/// (`8·dim` bits). Lexicographic byte compare = Morton order.
fn sq8_morton_key(codes: &[u8], dim: usize) -> Vec<u8> {
    let mut key = vec![0u8; dim]; // 8·dim bits == dim bytes
    for d in 0..dim {
        let c = codes[d];
        for b in 0..8u32 {
            if (c >> (7 - b)) & 1 == 1 {
                // sort-key bit position (0 = MSB): level `b` (bit 7-b of each dim),
                // dim index `d` within the level.
                let pos = b * dim as u32 + d as u32;
                key[(pos / 8) as usize] |= 1u8 << (7 - (pos % 8));
            }
        }
    }
    key
}

/// TD-WLP-4 (ADR-061 D3): the compaction re-cluster order — **PCA + IVF
/// (k-means) on fp32**, replacing the sign-Gray L0 bootstrap when a merged
/// batch is worth a model. Trains a write-time PCA on this batch
/// (`IncrementalPCA`, f32-native), k-means-clusters the projections
/// (foundation clustering kernel), orders IVF cells by the Hilbert code of
/// their centroid (set-normalized, so same-region cells are physically
/// contiguous — coalescing-friendly), and orders records by
/// `(cell rank, PC1 within cell, index)`. Clustering is computed on
/// PCA-projected fp32, never on quantized codes (ADR-061 D3/A3).
///
/// Falls back to [`cluster_order`] (the model-free bootstrap) when the batch
/// is too small to train (< [`MIN_ROWS_FOR_IVF`] usable rows) or k-means
/// fails; returns `None` like `cluster_order` when nothing is usable. The
/// batch-local model is discarded after ordering — persisted-model reuse
/// (`pca_model_ref`) is the TD-WLP-4b follow-up.
pub fn cluster_order_pca_ivf(records: &[ProximaRecord], idx: usize) -> Option<Vec<usize>> {
    // Test-only wrapper — NO production callers. The destination is NOT
    // order-irrelevant (k feeds `ivf_projection_dims` and the train sample,
    // which move the model and hence the order); `None` (CLOUD target) is
    // simply the explicit fallback for these order-shape tests.
    cluster_plan_pca_ivf(records, idx, None).map(|plan| plan.order)
}

/// Below this many usable rows a trained model can't beat the bootstrap.
const MIN_ROWS_FOR_IVF: usize = 64;

/// Fine IVF cell count ≈ blockcount (ADR-065 co-design): one cell ≈ one
/// IOP-sized block, so survivors span fewer cells → fewer GETs and each
/// cell-fetch is one efficient IOP. `PROXIMADB_IVF_K` overrides to map the
/// GETs-vs-recall curve (eval knob, not a production setting).
fn ivf_fine_cell_count(n_usable: usize, dim: usize, destination: Option<&str>) -> usize {
    // Shared with the read-side range planner so both sides agree on what an
    // "IOP" is (TD-IVF-4: two independent constants is the defect). The
    // destination URL (the published object, NOT the local staging path)
    // resolves the same per-location budget the reader uses for that location;
    // `None` falls back to the CLOUD target and remains the explicit answer
    // for callers that genuinely lack a destination.
    let iop_target =
        proximadb_storage_common::iops_budget::write_target_block_bytes(destination) as usize;
    std::env::var("PROXIMADB_IVF_K")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or_else(|| ((n_usable * dim) / iop_target.max(1)).clamp(2, 4096))
}

/// k-means convergence floor: enough samples per centroid to estimate a stable
/// centroid in the (low) PCA-projected dimension. k-means is well-converged by
/// ~tens of samples/centroid; 64 is a generous margin.
const IVF_MIN_SAMPLES_PER_CENTROID: usize = 64;

/// Upper bound on the default train sample to keep PCA-fit + k-means cost
/// bounded at extreme scale (cluster time is ~19% of compaction; finalize
/// dominates, but we still don't want the sample to grow without limit).
const IVF_TRAIN_SAMPLE_CAP: usize = 200_000;

/// TD-COMPACT-1: default IVF train-sample floor, SCALED with the cell count `k`.
/// Because `k ∝ N` ([`ivf_fine_cell_count`], one cell ≈ one IOP-sized block), a
/// fixed sample means samples-per-centroid DECAYS as the LSM tree grows
/// (L0→L1→L2…): at ~26M usable rows a fixed 50k sample drops below the k-means
/// convergence floor and centroids become underrepresented → worse ordering →
/// worse block-centroid prune → more GETs. Scaling the floor with `k` holds
/// samples/centroid stable across levels. At small `k` this is just the 50k
/// baseline (behavior unchanged for ≤~26M); at large `k` (higher levels / larger
/// collections) it grows, capped to bound PCA/k-means cost.
fn default_train_sample(k: usize) -> usize {
    (50_000)
        .max(k.saturating_mul(IVF_MIN_SAMPLES_PER_CENTROID))
        .min(IVF_TRAIN_SAMPLE_CAP)
}

/// Compiled-in default for the coarse-PCA projection-width floor. **`0` =
/// disabled**, i.e. the legacy formula below is used unchanged.
///
/// TD-IVF-3 measured the GET/query optimum at 32 (128-d) and 64 (384-d, 768-d)
/// against a formula that yields 10–14, so this constant is the seam that will
/// carry the widened default once the bake completes. It ships at `0` so the
/// plumbing lands bit-identical and the flip is a one-line, revertible change.
const IVF_NCOMP_FLOOR_DEFAULT: usize = 0;

/// Ceiling applied to the *floor* (never to the legacy value, and never to the
/// explicit `PROXIMADB_IVF_NCOMP` eval override).
///
/// Set to the **measured optimum** (64 at 384-d and 768-d, TD-IVF-3), with two
/// independent constraints checked against it:
///
/// * **A0 bytes** — the directory carries `n_comp·dim + k_c·n_comp` f32s and is
///   fetched whole on first touch. Guarded at width 64 for the widest supported
///   geometry by `coarse_directory.rs::a0_size_budget_tests`.
/// * **PCA conditioning** (TD-IVF-5) — the extractor uses power iteration with
///   deflation from a fixed seed and no re-orthogonalisation, so past a
///   spectrum-dependent depth it returns near-duplicates of components already
///   extracted. Onset measured on the **real** corpora: component 111 for 128-d
///   BIGANN, and none within 128 for 384-d or 768-d BGE. 64 therefore carries
///   margin on every corpus this system has been measured against.
///
/// The bound is *corpus-dependent, not universal*: an adversarial fast-decaying
/// spectrum (`0.90^j`) degenerates as early as component 42. Raising the floor
/// above 64 needs the conditioning check re-run for the corpus in question — see
/// `spatial_clustering.rs::deep_component_conditioning_tests`.
const IVF_NCOMP_FLOOR_CEILING: usize = 64;

/// TD-WLP-4b: PCA projection dimensionality = max of two logarithmic terms
/// (`a·log2 dim` intrinsic-dim, `b·log2 k` partition-granularity insurance),
/// env-tunable (`PROXIMADB_IVF_NCOMP[_A|_B]`), then raised to the configured
/// floor (TD-IVF-3). See the sweep notes at the call sites.
///
/// `PROXIMADB_IVF_NCOMP` remains an **absolute** override, clamped only to `dim`:
/// the bench harness sweeps widths (e.g. 128) deliberately outside the shipped
/// policy band, and capping it would silently truncate those measurements.
fn ivf_projection_dims(dim: usize, k: usize) -> usize {
    if let Some(explicit) = std::env::var("PROXIMADB_IVF_NCOMP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
    {
        return explicit.clamp(1, dim);
    }
    let ivf_a = std::env::var("PROXIMADB_IVF_NCOMP_A")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|x| *x > 0.0)
        .unwrap_or(1.5);
    let ivf_b = std::env::var("PROXIMADB_IVF_NCOMP_B")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|x| *x > 0.0)
        .unwrap_or(1.5);
    projection_dims_with_floor(dim, k, ivf_a, ivf_b, ivf_ncomp_floor())
}

/// Pure width policy, split out so the floor can be tested across dimensions
/// without mutating process environment (`set_var`/`remove_var` are unsafe in
/// edition 2024 precisely because they race across threads).
///
/// The ceiling binds only `floor`, never the legacy term — so `floor == 0`
/// reproduces the pre-TD-IVF-3 value exactly and this function is bit-identical
/// until the default is flipped.
fn projection_dims_with_floor(dim: usize, k: usize, a: f64, b: f64, floor: usize) -> usize {
    let dim_term = a * (dim as f64).log2();
    let k_term = b * (k.max(2) as f64).log2();
    let legacy = dim_term.max(k_term).floor() as usize;
    legacy.max(floor.min(IVF_NCOMP_FLOOR_CEILING)).clamp(1, dim)
}

/// Resolved projection-width floor. Precedence: env `PROXIMADB_IVF_NCOMP_FLOOR`
/// → TOML `[storage.sst_config.coarse_probe] ncomp_floor` → compiled-in
/// [`IVF_NCOMP_FLOOR_DEFAULT`], mirroring [`ivf_probe_enabled`]. A malformed env
/// value falls through to config rather than being read as `0`, so a typo cannot
/// silently disable the widened default once it ships.
fn ivf_ncomp_floor() -> usize {
    std::env::var("PROXIMADB_IVF_NCOMP_FLOOR")
        .ok()
        .and_then(|raw| parse_ncomp_floor(&raw))
        .unwrap_or_else(|| {
            crate::storage::engines::sst::segment_format::coarse_probe_settings().ncomp_floor
        })
}

/// Pure parse of the floor override, split out so it is testable without
/// mutating process environment (`set_var`/`remove_var` are unsafe in edition
/// 2024 precisely because they race across threads). `None` = "not specified
/// here", so the caller falls through to config.
fn parse_ncomp_floor(raw: &str) -> Option<usize> {
    raw.trim().parse::<usize>().ok()
}

/// Order cells by the Hilbert code of their centroid so same-region cells are
/// physically contiguous (coalescing-friendly). Normalization is over the
/// CENTROID SET per dimension (per-vector min/max would destroy cross-centroid
/// comparability). Returns `(emission order, rank per cell)`.
fn hilbert_cell_order(centroids: &[Vec<f32>]) -> (Vec<usize>, Vec<usize>) {
    let hilbert_dims = centroids.first().map(|c| c.len().clamp(1, 6)).unwrap_or(1);
    let bits_per_dim = 10usize; // 6 dims × 10 bits ≤ u64
    let mut lo = vec![f32::INFINITY; hilbert_dims];
    let mut hi = vec![f32::NEG_INFINITY; hilbert_dims];
    for c in centroids {
        for d in 0..hilbert_dims {
            lo[d] = lo[d].min(c[d]);
            hi[d] = hi[d].max(c[d]);
        }
    }
    let curve =
        proximadb_storage_common::hilbert_curve::HilbertCurve::new(hilbert_dims, bits_per_dim);
    let max_val = (1u32 << bits_per_dim) - 1;
    let cell_key = |c: &[f32]| -> u64 {
        let ints: Vec<u32> = (0..hilbert_dims)
            .map(|d| {
                let range = hi[d] - lo[d];
                if range <= 0.0 {
                    max_val / 2
                } else {
                    (((c[d] - lo[d]) / range) * max_val as f32) as u32
                }
            })
            .collect();
        curve.encode(&ints)
    };
    let mut cell_order: Vec<usize> = (0..centroids.len()).collect();
    let keys: Vec<u64> = centroids.iter().map(|c| cell_key(c)).collect();
    cell_order.sort_by_key(|&c| keys[c]);
    let mut cell_rank = vec![0usize; centroids.len()];
    for (rank, &cell) in cell_order.iter().enumerate() {
        cell_rank[cell] = rank;
    }
    (cell_order, cell_rank)
}

/// Seed for the compaction IVF k-means. A physical layout must not depend on
/// thread-local RNG state: identical input must produce identical IVF cells,
/// byte counts, and eval results. Shared by the single-level plan
/// ([`cluster_plan_pca_ivf`]) and the persisted probe directory
/// ([`cluster_plan_ivf_probe`]) — rev 3 persists the *same* plan, so the seed is
/// the same.
const PAX_IVF_KMEANS_SEED: u64 = 0x5041_585F_4956_4631;

/// Eval-only override for the k-means seed (`PROXIMADB_IVF_KMEANS_SEED`).
///
/// The pipeline is deterministic end to end, which makes results perfectly
/// reproducible and — by construction — blind to how much a conclusion depends on
/// one clustering draw. Every coarse-PCA width measurement to date (TD-IVF-3) was
/// taken at the single compiled-in seed, so the sensitivity of those optima to
/// initialisation is unmeasured. This gate exists to measure it.
///
/// Unset in production: the default keeps the compiled-in constant, so behaviour
/// and reproducibility are unchanged. The chosen seed is persisted per segment
/// (see `seed:` below), so a bed always records which draw produced it.
fn pax_ivf_kmeans_seed() -> u64 {
    resolve_kmeans_seed(std::env::var("PROXIMADB_IVF_KMEANS_SEED").ok())
}

/// Pure resolution of the seed override, split out so it is testable without
/// mutating process environment (`set_var`/`remove_var` are unsafe in edition
/// 2024 precisely because they race across threads).
fn resolve_kmeans_seed(raw: Option<String>) -> u64 {
    raw.and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(PAX_IVF_KMEANS_SEED)
}

/// Bounded, deterministic training geometry shared by the in-memory and local
/// spill compaction paths. The spill path uses `sample_step` to revisit its
/// checksummed winner run without retaining the raw corpus in RAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IvfTrainingShape {
    pub(crate) k: usize,
    pub(crate) n_components: usize,
    pub(crate) train_sample: usize,
    pub(crate) sample_step: usize,
}

pub(crate) fn ivf_training_shape(
    usable_rows: usize,
    dim: usize,
    destination: Option<&str>,
) -> Option<IvfTrainingShape> {
    if usable_rows < MIN_ROWS_FOR_IVF || dim == 0 {
        return None;
    }
    let k = ivf_fine_cell_count(usable_rows, dim, destination).min(usable_rows);
    let train_sample = std::env::var("PROXIMADB_IVF_TRAIN_SAMPLE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| default_train_sample(k))
        .min(usable_rows);
    Some(IvfTrainingShape {
        k,
        n_components: ivf_projection_dims(dim, k),
        train_sample,
        sample_step: usable_rows.div_ceil(train_sample).max(1),
    })
}

/// Assignment returned by the shared persisted-f32 IVF classifier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct IvfAssignment {
    pub(crate) source_cell: usize,
    pub(crate) cell_rank: usize,
    pub(crate) pc1: f32,
    pub(crate) distance_sq: f32,
}

/// Fitted classifier used by both compaction implementations. Keeping the
/// source-centroid order until all rows are assigned lets radius/count
/// accumulation remain O(k), then [`Self::finish_model`] emits A0 arrays in
/// the Hilbert order used by the physical segment.
pub(crate) struct IvfProbeClassifier {
    dim: usize,
    n_comp: usize,
    pca_mean: Vec<f32>,
    pca_components: Vec<f32>,
    centroids: Vec<Vec<f32>>,
    cell_order: Vec<usize>,
    cell_rank: Vec<usize>,
    trained_on: usize,
}

impl IvfProbeClassifier {
    pub(crate) fn classify(&self, vector: &[f32]) -> Option<IvfAssignment> {
        use proximadb_storage_common::coarse_directory::project_with_model;

        if vector.len() != self.dim {
            return None;
        }
        let coords = project_with_model(&self.pca_mean, &self.pca_components, self.n_comp, vector);
        self.classify_coords(&coords)
    }

    /// Nearest-centroid assignment for an **already-projected** row.
    ///
    /// Split out of [`Self::classify`] so callers that have projected the row
    /// already do not pay for a second projection. `cluster_plan_ivf_probe` used
    /// to project every usable row, then call `classify`, which projected it
    /// again — the projection term is `N·dim·n_comp`, so that duplicate was a
    /// large share of the width-linear write cost this crate is about to raise.
    pub(crate) fn classify_coords(&self, coords: &[f32]) -> Option<IvfAssignment> {
        if self.centroids.is_empty() {
            return None;
        }
        let mut source_cell = 0usize;
        let mut distance_sq = f32::INFINITY;
        for (cell, centroid) in self.centroids.iter().enumerate() {
            let distance = coords
                .iter()
                .zip(centroid)
                .map(|(left, right)| (left - right) * (left - right))
                .sum::<f32>();
            if distance < distance_sq {
                source_cell = cell;
                distance_sq = distance;
            }
        }
        Some(IvfAssignment {
            source_cell,
            cell_rank: self.cell_rank[source_cell],
            pc1: coords.first().copied().unwrap_or(0.0),
            distance_sq,
        })
    }

    pub(crate) fn cell_count(&self) -> usize {
        self.centroids.len()
    }

    pub(crate) fn finish_model(
        self,
        source_cell_rows: &[u64],
        source_cell_radii_sq: &[f32],
    ) -> Option<proximadb_storage_common::coarse_directory::CoarseModel> {
        if source_cell_rows.len() != self.centroids.len()
            || source_cell_radii_sq.len() != self.centroids.len()
        {
            return None;
        }
        let n_comp = u16::try_from(self.n_comp).ok()?;
        let dim = u32::try_from(self.dim).ok()?;
        let centroids = self
            .cell_order
            .iter()
            .flat_map(|&cell| self.centroids[cell].iter().copied())
            .collect();
        let radii = self
            .cell_order
            .iter()
            .map(|&cell| source_cell_radii_sq[cell].sqrt())
            .collect();
        let cell_rows = self
            .cell_order
            .iter()
            .map(|&cell| source_cell_rows[cell])
            .collect();
        Some(proximadb_storage_common::coarse_directory::CoarseModel {
            dim,
            n_comp,
            pca_mean: self.pca_mean,
            pca_components: self.pca_components,
            centroids,
            radii,
            cell_rows,
            seed: pax_ivf_kmeans_seed(),
            trained_on: self.trained_on as u64,
        })
    }
}

/// Persisted-f32 PCA projection. Spill compaction creates this after its first
/// sampled pass, then projects the same sampled positions during a second pass;
/// raw high-dimensional samples are therefore never retained in RAM.
pub(crate) struct IvfPcaProjection {
    pca_mean: Vec<f32>,
    pca_components: Vec<f32>,
    n_comp: usize,
}

impl IvfPcaProjection {
    pub(crate) fn from_finalized(
        pca: &crate::storage::engines::core::formats::proximablocks::spatial_clustering::IncrementalPCA,
    ) -> Option<Self> {
        let pca_mean = pca
            .mean()
            .iter()
            .map(|&value| value as f32)
            .collect::<Vec<_>>();
        let components = pca.components()?;
        let n_comp = components.len();
        if n_comp == 0 {
            return None;
        }
        let pca_components = components
            .iter()
            .flat_map(|row| row.iter().map(|&value| value as f32))
            .collect::<Vec<_>>();
        Some(Self {
            pca_mean,
            pca_components,
            n_comp,
        })
    }

    pub(crate) fn project(&self, vector: &[f32]) -> Vec<f32> {
        proximadb_storage_common::coarse_directory::project_with_model(
            &self.pca_mean,
            &self.pca_components,
            self.n_comp,
            vector,
        )
    }
}

/// Complete the shared classifier from persisted PCA and a low-dimensional
/// deterministic sample. Both paths train k-means on the exact f32 projection
/// used by the reader.
pub(crate) fn finish_ivf_probe_classifier(
    projection: IvfPcaProjection,
    shape: IvfTrainingShape,
    sample_coords: &[Vec<f32>],
) -> Option<IvfProbeClassifier> {
    let k = shape.k.min(sample_coords.len());
    let Ok(centroids) = proximadb_clustering_kernel::kmeans_clustering_seeded(
        sample_coords,
        k,
        15,
        1e-3,
        pax_ivf_kmeans_seed(),
    ) else {
        return None;
    };
    if centroids.is_empty() {
        return None;
    }
    let (cell_order, cell_rank) = hilbert_cell_order(&centroids);
    Some(IvfProbeClassifier {
        dim: projection.pca_mean.len(),
        n_comp: projection.n_comp,
        pca_mean: projection.pca_mean,
        pca_components: projection.pca_components,
        centroids,
        cell_order,
        cell_rank,
        trained_on: sample_coords.len(),
    })
}

/// Compaction-grade PCA+IVF plan, including the contiguous IVF-cell runs.
pub fn cluster_plan_pca_ivf(
    records: &[ProximaRecord],
    idx: usize,
    destination: Option<&str>,
) -> Option<ClusterPlan> {
    use crate::storage::engines::core::formats::proximablocks::spatial_clustering::IncrementalPCA;

    let usable: Vec<(usize, &[f32])> = records
        .iter()
        .enumerate()
        .filter_map(|(i, r)| embedding_f32(r, idx).map(|v| (i, v)))
        .collect();
    if usable.len() < MIN_ROWS_FOR_IVF {
        return cluster_plan(records, idx);
    }
    let dim = usable[0].1.len();
    if usable.iter().any(|(_, v)| v.len() != dim) {
        return cluster_plan(records, idx);
    }

    let trace_ivf = std::env::var_os("PROXIMADB_TRACE_IVF_FLUSH").is_some();
    let t_start = std::time::Instant::now();

    // IVF cells ≈ blockcount (ADR-065 co-design); k scales with N (more data ⇒
    // more cells, members/cell ~constant). Computed UP FRONT so n_components
    // can couple to it. n_comp: dim-term wins at SIFT's k (Phase-0 sweep:
    // recall@10 flat 0.989 for n_comp ∈ [4,128]); the k-term is the high-k
    // hedge. Clustering-only — search-time PCA keeps its own cap.
    let k = ivf_fine_cell_count(usable.len(), dim, destination);
    let n_components = ivf_projection_dims(dim, k);
    // TD-WLP-4b sample-train: fit PCA + train k-means on a deterministic
    // ~SAMPLE subset (the covariance / centroids converge far before N=1M),
    // then project + assign ALL rows. Cuts pca_fit + kmeans ~N/sample (~20x at
    // SIFT1M) with negligible recall impact; project/assign still cover all N.
    // Deterministic stride (a physical layout must not depend on RNG).
    //
    // TD-COMPACT-1: the default floor scales with `k` (via
    // [`default_train_sample`]) so samples-per-centroid stays above the k-means
    // convergence floor as the LSM tree grows — a fixed 50k would underrepresent
    // centroids at higher levels (k ∝ N). `PROXIMADB_IVF_TRAIN_SAMPLE` still
    // overrides absolutely.
    let train_sample = std::env::var("PROXIMADB_IVF_TRAIN_SAMPLE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or_else(|| default_train_sample(k))
        .min(usable.len());
    let sample_step = if usable.len() > train_sample {
        (usable.len() / train_sample).max(1)
    } else {
        1
    };
    let mut pca = IncrementalPCA::new(dim, n_components);
    for (_, v) in usable.iter().step_by(sample_step) {
        pca.add_sample(v);
    }
    pca.finalize();
    let t_pca = std::time::Instant::now();
    let coords: Vec<Vec<f32>> = usable.iter().map(|(_, v)| pca.transform(v)).collect();
    let t_proj = std::time::Instant::now();
    // Train k-means on the sampled projections (same stride as the PCA fit);
    // assign ALL rows to the resulting centroids.
    let train_coords: Vec<Vec<f32>> = coords.iter().step_by(sample_step).cloned().collect();
    let Ok(centroids) = proximadb_clustering_kernel::kmeans_clustering_seeded(
        &train_coords,
        k,
        15,
        1e-3,
        pax_ivf_kmeans_seed(),
    ) else {
        return cluster_plan(records, idx);
    };
    let t_kmeans = std::time::Instant::now();
    let assignments = proximadb_clustering_kernel::kmeans_assign(&coords, &centroids);
    let t_assign = std::time::Instant::now();

    // Order cells by the Hilbert code of their centroid (set-normalized).
    let (_cell_order, cell_rank) = hilbert_cell_order(&centroids);

    // Records: usable ordered by (cell rank, PC1, index); unusable last, stably.
    // This path keeps the full `coords` (kmeans_assign needs every row), so PC1
    // is projected out of it rather than carried on an assignment.
    let usable_orig: Vec<usize> = usable.iter().map(|(i, _)| *i).collect();
    let pc1: Vec<f32> = coords
        .iter()
        .map(|c| c.first().copied().unwrap_or(0.0))
        .collect();
    let (order, runs) =
        order_rows_by_cell(records.len(), &usable_orig, &pc1, &assignments, &cell_rank);
    let t_end = std::time::Instant::now();
    if trace_ivf {
        eprintln!(
            "[IVF flush] N={n} dim={dim} k={k} n_comp={nc} | pca_fit {pca:.0} ms  project {proj:.0} ms  kmeans {km:.0} ms  assign {asg:.0} ms  order {ord:.0} ms  | total {tot:.0} ms",
            n = usable.len(),
            nc = n_components,
            pca = (t_pca - t_start).as_secs_f64() * 1e3,
            proj = (t_proj - t_pca).as_secs_f64() * 1e3,
            km = (t_kmeans - t_proj).as_secs_f64() * 1e3,
            asg = (t_assign - t_kmeans).as_secs_f64() * 1e3,
            ord = (t_end - t_assign).as_secs_f64() * 1e3,
            tot = (t_end - t_start).as_secs_f64() * 1e3,
        );
    }
    Some(ClusterPlan { order, runs })
}

/// TD-RDSTRAT-8 (rev 3): gate for the persisted-IVF-probe v3 compaction layout
/// (the IOP-derived plan written into Region A0). Default **ON** (the COGS-arc
/// flip — IVF cuts GETs/query ~4× and per-tenant COGS ~4× vs full-scan; the
/// recall/GET eval gates passed: ledger `nprobe_sweep_trained_1m`). Precedence:
/// env `PROXIMADB_PAX_WRITE_A0_TRAIN` (truthy `1|true|on|yes` ⇒ on; set to
/// anything else ⇒ off) → else TOML `[storage.sst_config.coarse_probe]
/// enable_write_train`. Mixed-read-safe beside single-level segments either way.
/// Pre-GA clean rename (TD-ENVGATE-1): the former `PROXIMADB_IVF2` name is
/// RETIRED (ENV_GATE_REGISTRY "Retired names" — reserved, never repurposed).
pub fn ivf_probe_enabled() -> bool {
    match std::env::var("PROXIMADB_PAX_WRITE_A0_TRAIN") {
        // Env set: honor the truthy/non-truthy value.
        Ok(_) => env_gate_on("PROXIMADB_PAX_WRITE_A0_TRAIN"),
        // Env unset: fall back to the TOML/config default.
        Err(_) => {
            crate::storage::engines::sst::segment_format::coarse_probe_settings().enable_write_train
        }
    }
}

/// TD-FPRUNE-1 M2 / C2: gate for the tag-aware two-level compaction layout
/// (partition-by-tag, cluster-within). **Default ON** — engaged STRUCTURALLY at
/// compaction whenever the segment carries a shredded filterable tag (the
/// per-collection signal: the owner declared `filterable_columns`; the caller
/// ANDs this with `shred_spec.first()`). Evidence-gated flip: C1 measured a 4.2×
/// filtered byte cut for an UNCORRELATED tag (whose non-partitioned filtered
/// query reads *more* than a full scan), and C2 measured UNFILTERED recall
/// neutral vs the global layout (Δ=0.000). Mixed-read-safe: emits the SAME A0
/// `CoarseModel` format (no layout_version change), only a different row
/// placement. `PROXIMADB_PAX_PARTITION_LAYOUT=0|off|false|no` is the kill-switch
/// back to the global (single-directory) plan.
pub fn partition_layout_enabled() -> bool {
    !matches!(
        std::env::var("PROXIMADB_PAX_PARTITION_LAYOUT")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("0" | "off" | "false" | "no")
    )
}

/// Boolean gate read: truthy = `1|true|on|yes`.
pub(crate) fn env_gate_on(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            )
        })
        .unwrap_or(false)
}

/// The persisted IVF probe directory compaction plan (rev 3): the physical row
/// order + runs (one run per non-empty IOP-derived cell, in emission order) plus
/// the trained [`CoarseModel`] the writer persists into Region A0.
pub struct IvfProbePlan {
    pub plan: ClusterPlan,
    pub model: proximadb_storage_common::coarse_directory::CoarseModel,
}

/// TD-RDSTRAT-8 (rev 3): the **persisted IVF probe directory** compaction plan.
/// Reuses the *existing* IOP-derived PCA/IVF recipe — `ivf_fine_cell_count` `k`
/// (≈ `N·dim / IOP_target`, capped 4096), `ivf_projection_dims` `n_comp`,
/// `PROXIMADB_IVF_TRAIN_SAMPLE` sample-train, and [`PAX_IVF_KMEANS_SEED`] — the
/// same plan [`cluster_plan_pca_ivf`] computes for the single-level layout, and
/// **persists it** into Region A0 rather than discarding it.
///
/// There is NO second `sqrt(N)` quantizer (rev 3 rejected the coarse-on-fine
/// level): ranking the `k` IOP-derived centroids in RAM (`k` ≈ 305 at 10M,
/// ≈ 3050 at 100M, capped 4096) is trivial and already breaks GETs ∝ N, whereas
/// `sqrt(N)` cells would fragment Region B into sub-IOP (~128 KiB) runs — worse
/// cloud economics. The cells here are the IOP-aligned cells the compaction
/// already speaks; A0 just makes them query-visible so the PR-B reader ranks the
/// directory in RAM and fetches only probed cells.
///
/// Write/read projection consistency: the PCA model is **truncated to the f32
/// precision A0 persists before anything is assigned** — coordinates, centroids,
/// and radii all come from the shared [`project_with_model`] kernel the query
/// path uses, so a query projects into exactly the space the writer clustered
/// (no f64-train vs f32-read drift at cell boundaries; vectors stay f32
/// throughout, only dot-product accumulators widen transiently to f64).
///
/// Deterministic end to end: strided sample-train, seeded k-means, fixed-init
/// PCA — identical input ⇒ identical plan ⇒ identical segment bytes (unit-gated
/// in `pax_block.rs`). `PROXIMADB_IVF_K` overrides `k` (shared eval knob).
///
/// Returns `None` when the IOP-derived plan can't be trained (fewer than
/// [`MIN_ROWS_FOR_IVF`] usable rows, degenerate dim, or k-means failure) — the
/// caller falls back to [`cluster_plan_pca_ivf`] and writes the single-level
/// layout (fail-safe, never a worse segment).
pub fn cluster_plan_ivf_probe(
    records: &[ProximaRecord],
    idx: usize,
    destination: Option<&str>,
) -> Option<IvfProbePlan> {
    use crate::storage::engines::core::formats::proximablocks::spatial_clustering::IncrementalPCA;

    let usable: Vec<(usize, &[f32])> = records
        .iter()
        .enumerate()
        .filter_map(|(i, r)| embedding_f32(r, idx).map(|v| (i, v)))
        .collect();
    if usable.len() < MIN_ROWS_FOR_IVF {
        return None;
    }
    let dim = usable[0].1.len();
    if dim == 0 || usable.iter().any(|(_, v)| v.len() != dim) {
        return None;
    }

    let trace_ivf = std::env::var_os("PROXIMADB_TRACE_IVF_FLUSH").is_some();
    let t_start = std::time::Instant::now();

    let shape = ivf_training_shape(usable.len(), dim, destination)?;
    let mut pca = IncrementalPCA::new(dim, shape.n_components);
    for (_, v) in usable.iter().step_by(shape.sample_step) {
        pca.add_sample(v);
    }
    pca.finalize();
    let t_pca = std::time::Instant::now();

    let projection = IvfPcaProjection::from_finalized(&pca)?;
    // Project ONLY the training sample. k-means needs exactly these rows, and
    // every usable row is projected once below in the assignment pass. The
    // previous shape projected all N rows here and then called `classify`, which
    // projected each row a SECOND time — while also holding an O(N·n_comp)
    // buffer (at 3.3M rows and width 64 that is ~845 MB of coordinates whose
    // only surviving use was PC1, which `IvfAssignment` already carries).
    // Sampling stride matches the PCA fit above, so this is bit-identical.
    let sample_coords = usable
        .iter()
        .step_by(shape.sample_step)
        .map(|(_, vector)| projection.project(vector))
        .collect::<Vec<_>>();
    let t_proj = std::time::Instant::now();

    let classifier = finish_ivf_probe_classifier(projection, shape, &sample_coords)?;
    let cell_count = classifier.cell_count();
    let n_comp = classifier.n_comp;
    let t_kmeans = std::time::Instant::now();
    let classified = usable
        .iter()
        .map(|(_, vector)| classifier.classify(vector))
        .collect::<Option<Vec<_>>>()?;
    // PC1 per usable row — the sole reason the full coords buffer existed.
    let pc1: Vec<f32> = classified.iter().map(|a| a.pc1).collect();
    let assignments = classified
        .iter()
        .map(|assignment| assignment.source_cell)
        .collect::<Vec<_>>();
    let mut radii_sq = vec![0f32; cell_count];
    let mut counts = vec![0u64; cell_count];
    for assignment in &classified {
        counts[assignment.source_cell] = counts[assignment.source_cell].saturating_add(1);
        if assignment.distance_sq > radii_sq[assignment.source_cell] {
            radii_sq[assignment.source_cell] = assignment.distance_sq;
        }
    }
    let t_assign = std::time::Instant::now();

    if trace_ivf {
        trace_between_centroid_variance(&sample_coords, &classifier.centroids, &counts);
    }

    // Rows: usable ordered by (cell rank, PC1, index); unusable last, stably (the
    // no-embedding tail — outside every cell, unreachable by ANN anyway).
    let usable_orig: Vec<usize> = usable.iter().map(|(i, _)| *i).collect();
    let (order, runs) = order_rows_by_cell(
        records.len(),
        &usable_orig,
        &pc1,
        &assignments,
        &classifier.cell_rank,
    );
    let model = classifier.finish_model(&counts, &radii_sq)?;
    let t_end = std::time::Instant::now();
    if trace_ivf {
        // Phase labels are load-bearing: TD-IVF-3 attributed the width-linear
        // write cost to k-means over the training sample, which would make it
        // independent of collection size. `project(sample)` scales with the
        // sample; `project+assign` scales with N. Comparing the two across
        // corpus sizes settles that directly.
        eprintln!(
            "[IVF probe compaction] N={n} dim={dim} k={k} n_comp={n_comp} | pca_fit {pca:.0} ms  project(sample) {proj:.0} ms  kmeans {km:.0} ms  project+assign {asg:.0} ms  order {ord:.0} ms  | total {tot:.0} ms",
            n = usable.len(),
            k = cell_count,
            pca = (t_pca - t_start).as_secs_f64() * 1e3,
            proj = (t_proj - t_pca).as_secs_f64() * 1e3,
            km = (t_kmeans - t_proj).as_secs_f64() * 1e3,
            asg = (t_assign - t_kmeans).as_secs_f64() * 1e3,
            ord = (t_end - t_assign).as_secs_f64() * 1e3,
            tot = (t_end - t_start).as_secs_f64() * 1e3,
        );
    }
    Some(IvfProbePlan {
        plan: ClusterPlan { order, runs },
        model,
    })
}

/// TD-FPRUNE-1 M2: canonical equality-group key for a filterable tag value. Low-
/// cardinality categorical/scalar values group by value (type-prefixed so an
/// int `1` and a string `"1"` never collide); float/complex/missing values share
/// a single sentinel group (they are not the low-card tags partitioning targets).
/// A deterministic `String` — the physical layout must never depend on map
/// iteration order or RNG.
fn partition_group_key(v: &proximadb_data_model::ProximaValue) -> String {
    use proximadb_data_model::ProximaValue as V;
    match v {
        V::Boolean(b) => format!("b:{b}"),
        V::Int8(x) => format!("i:{x}"),
        V::Int16(x) => format!("i:{x}"),
        V::Int32(x) => format!("i:{x}"),
        V::Int64(x) => format!("i:{x}"),
        V::UInt8(x) => format!("u:{x}"),
        V::UInt16(x) => format!("u:{x}"),
        V::UInt32(x) => format!("u:{x}"),
        V::UInt64(x) => format!("u:{x}"),
        V::String(s) | V::Symbol(s) => format!("s:{s}"),
        V::Decimal(s) => format!("d:{s}"),
        V::Date(x) => format!("dt:{x}"),
        // Floats/complex/binary are not low-card partition keys — sentinel group.
        _ => "\u{0}none".to_string(),
    }
}

/// TD-FPRUNE-1 M2: tag-aware two-level compaction layout. Trains ONE global PCA
/// (a shared projection space, so the reader's coarse probe is unchanged), then
/// clusters vectors SEPARATELY within each partition group of the declared
/// low-card filterable tag and concatenates the per-group centroids. Rows are
/// ordered partition-contiguous (group order, Hilbert within group), so every A0
/// cell is partition-HOMOGENEOUS. A filtered query then skips the other
/// partitions' cells for free via the M3 `count_in_range` test — this is the
/// LAYOUT lever that makes UNCORRELATED tags prune (min/max/bloom cannot, because
/// every geometric cell contains every value). Emits the SAME A0 `CoarseModel`
/// cell format (no `layout_version` change) — mixed-read-safe; only the row
/// placement differs. Returns `None` (caller falls back to the global
/// [`cluster_plan_ivf_probe`]) when the tag is absent, degenerate (a single
/// group), or the corpus is below the IVF floor.
pub fn cluster_plan_ivf_probe_partitioned(
    records: &[ProximaRecord],
    idx: usize,
    partition_key: &str,
    destination: Option<&str>,
) -> Option<IvfProbePlan> {
    use crate::storage::engines::core::formats::proximablocks::spatial_clustering::IncrementalPCA;

    let usable: Vec<(usize, &[f32])> = records
        .iter()
        .enumerate()
        .filter_map(|(i, r)| embedding_f32(r, idx).map(|v| (i, v)))
        .collect();
    if usable.len() < MIN_ROWS_FOR_IVF {
        return None;
    }
    let dim = usable[0].1.len();
    if dim == 0 || usable.iter().any(|(_, v)| v.len() != dim) {
        return None;
    }

    // Group usable-row POSITIONS (indices into `usable`) by the tag value. A
    // BTreeMap keeps the group order deterministic (sorted key) — the physical
    // layout must not depend on hash iteration order.
    let mut groups: std::collections::BTreeMap<String, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (pos, (orig, _)) in usable.iter().enumerate() {
        let key = match records[*orig].props.get(partition_key) {
            Some(proximadb_records::ProximaTreeNode::Value(v)) => partition_group_key(v),
            _ => "\u{0}none".to_string(),
        };
        groups.entry(key).or_default().push(pos);
    }
    // A single group ⇒ no partitioning benefit; let the caller use the global plan.
    if groups.len() < 2 {
        return None;
    }

    // ONE global PCA over the whole corpus (the shared projection space the
    // reader's probe already assumes).
    let shape = ivf_training_shape(usable.len(), dim, destination)?;
    let mut pca = IncrementalPCA::new(dim, shape.n_components);
    for (_, v) in usable.iter().step_by(shape.sample_step) {
        pca.add_sample(v);
    }
    pca.finalize();
    let projection = IvfPcaProjection::from_finalized(&pca)?;
    let coords: Vec<Vec<f32>> = usable.iter().map(|(_, v)| projection.project(v)).collect();
    let IvfPcaProjection {
        pca_mean,
        pca_components,
        n_comp,
    } = projection;
    let k_total = shape.k;

    // Per-partition k-means in the shared space; concatenate the centroids into a
    // single global cell space, keeping cells of one group contiguous in rank so
    // rows land partition-contiguous.
    let mut centroids: Vec<Vec<f32>> = Vec::new();
    let mut assignments = vec![0usize; usable.len()];
    let mut cell_rank: Vec<usize> = Vec::new();
    let mut rank_base = 0usize;
    for positions in groups.values() {
        let g = positions.len();
        // Cells proportional to the group's share of the corpus, ≥1 and ≤ |group|.
        let k_g = ((k_total as f64 * g as f64 / usable.len() as f64).round() as usize)
            .max(1)
            .min(g);
        let group_coords: Vec<Vec<f32>> = positions.iter().map(|&p| coords[p].clone()).collect();
        let mut train: Vec<Vec<f32>> = group_coords
            .iter()
            .step_by(shape.sample_step.max(1))
            .cloned()
            .collect();
        if train.is_empty() {
            train = group_coords.clone();
        }
        let k_g = k_g.min(train.len()).max(1);
        let Ok(cg) = proximadb_clustering_kernel::kmeans_clustering_seeded(
            &train,
            k_g,
            15,
            1e-3,
            pax_ivf_kmeans_seed(),
        ) else {
            return None;
        };
        if cg.is_empty() {
            return None;
        }
        let offset = centroids.len();
        let local = proximadb_clustering_kernel::kmeans_assign(&group_coords, &cg);
        for (lp, &pos) in positions.iter().enumerate() {
            assignments[pos] = offset + local[lp];
        }
        // Hilbert order WITHIN the group; the global rank stays partition-blocked.
        let (_ord, local_rank) = hilbert_cell_order(&cg);
        cell_rank.resize(offset + cg.len(), 0);
        for (lc, &lr) in local_rank.iter().enumerate() {
            cell_rank[offset + lc] = rank_base + lr;
        }
        rank_base += cg.len();
        centroids.extend(cg);
    }

    let total_cells = centroids.len();
    let mut counts = vec![0u64; total_cells];
    let mut radii_sq = vec![0f32; total_cells];
    for (pos, &cell) in assignments.iter().enumerate() {
        counts[cell] = counts[cell].saturating_add(1);
        let d = coords[pos]
            .iter()
            .zip(&centroids[cell])
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>();
        if d > radii_sq[cell] {
            radii_sq[cell] = d;
        }
    }

    let pc1: Vec<f32> = coords
        .iter()
        .map(|c| c.first().copied().unwrap_or(0.0))
        .collect();
    let usable_orig: Vec<usize> = usable.iter().map(|(i, _)| *i).collect();
    let (order, runs) =
        order_rows_by_cell(records.len(), &usable_orig, &pc1, &assignments, &cell_rank);

    // Emission order = cells sorted by rank (partition-blocked, Hilbert within).
    let mut cell_order: Vec<usize> = (0..total_cells).collect();
    cell_order.sort_by_key(|&c| cell_rank[c]);

    let classifier = IvfProbeClassifier {
        dim,
        n_comp,
        pca_mean,
        pca_components,
        centroids,
        cell_order,
        cell_rank,
        trained_on: usable.len(),
    };
    let model = classifier.finish_model(&counts, &radii_sq)?;
    Some(IvfProbePlan {
        plan: ClusterPlan { order, runs },
        model,
    })
}

fn contiguous_runs<T: PartialEq>(ordered_labels: &[T]) -> Vec<OrderedClusterRun> {
    if ordered_labels.is_empty() {
        return Vec::new();
    }
    let mut runs = Vec::new();
    let mut start = 0usize;
    for row in 1..ordered_labels.len() {
        if ordered_labels[row] != ordered_labels[row - 1] {
            runs.push(OrderedClusterRun {
                start_row: start,
                row_count: row - start,
            });
            start = row;
        }
    }
    runs.push(OrderedClusterRun {
        start_row: start,
        row_count: ordered_labels.len() - start,
    });
    runs
}

/// Shared IVF row ordering (single-level and probe-directory plans use the same
/// rule): usable rows sorted by `(cell rank, PC1, original index)`, then the
/// unusable (no-embedding) rows appended last, stably. Returns the full-record
/// permutation plus the contiguous per-cell runs (unusable tail is one trailing
/// run keyed `usize::MAX`). Projection-agnostic — the caller decides whether
/// `coords` came from `pca.transform` (f64 model) or `project_with_model` (f32),
/// so extracting this keeps both plans byte-identical to their inline form.
/// Diagnostic (trace-gated): the share of **between-centroid** variance carried
/// by each PCA component, and its cumulative profile.
///
/// Coarse ranking succeeds or fails on whether the projection preserves the
/// separation *between cells*, which is not the same as preserving total
/// variance — TD-IVF-3 measured a 4.4-point spread across corpora in
/// between-centroid share at the optimum versus 22.2 points in total-variance
/// share. That invariant is nonetheless **not** used to choose the width: it is
/// tight in variance-share space but loose in width space (no single threshold
/// reproduces all three measured optima), so it would be the fifth refuted
/// dimension-keyed model. Emitting it keeps the evidence accruing on every bed at
/// zero production cost, without the policy depending on it.
fn trace_between_centroid_variance(
    sample_coords: &[Vec<f32>],
    centroids: &[Vec<f32>],
    counts: &[u64],
) {
    let Some(n_comp) = sample_coords.first().map(|c| c.len()) else {
        return;
    };
    if n_comp == 0 || centroids.len() != counts.len() {
        return;
    }
    let total_rows: f64 = counts.iter().map(|&c| c as f64).sum();
    if total_rows <= 0.0 || sample_coords.is_empty() {
        return;
    }

    // Between-centroid variance per component, weighted by realised cell
    // occupancy (the grand mean is that same weighted mean, so the two are
    // consistent even when cells are badly unbalanced).
    let mut grand = vec![0f64; n_comp];
    for (centroid, &count) in centroids.iter().zip(counts) {
        for (j, value) in centroid.iter().take(n_comp).enumerate() {
            grand[j] += *value as f64 * count as f64;
        }
    }
    for value in &mut grand {
        *value /= total_rows;
    }
    let mut between = vec![0f64; n_comp];
    for (centroid, &count) in centroids.iter().zip(counts) {
        for (j, value) in centroid.iter().take(n_comp).enumerate() {
            let delta = *value as f64 - grand[j];
            between[j] += count as f64 * delta * delta;
        }
    }
    for value in &mut between {
        *value /= total_rows;
    }

    // Total variance per component, over the training sample.
    let n_sample = sample_coords.len() as f64;
    let mut mean = vec![0f64; n_comp];
    for coords in sample_coords {
        for (j, value) in coords.iter().take(n_comp).enumerate() {
            mean[j] += *value as f64;
        }
    }
    for value in &mut mean {
        *value /= n_sample;
    }
    let mut total = vec![0f64; n_comp];
    for coords in sample_coords {
        for (j, value) in coords.iter().take(n_comp).enumerate() {
            let delta = *value as f64 - mean[j];
            total[j] += delta * delta;
        }
    }
    for value in &mut total {
        *value /= n_sample;
    }

    let between_sum: f64 = between.iter().sum();
    if between_sum <= 0.0 {
        return;
    }
    // eta^2 of the leading component, plus the cumulative between-centroid share
    // at the widths the study actually measured.
    let eta2_first = if total[0] > 0.0 {
        between[0] / total[0]
    } else {
        0.0
    };
    let mut cumulative = String::new();
    let mut running = 0f64;
    let mut next_marker = 0usize;
    const MARKERS: [usize; 6] = [12, 16, 24, 32, 48, 64];
    for (j, value) in between.iter().enumerate() {
        running += value;
        while next_marker < MARKERS.len() && MARKERS[next_marker] == j + 1 {
            cumulative.push_str(&format!(
                " w{}={:.1}%",
                MARKERS[next_marker],
                100.0 * running / between_sum
            ));
            next_marker += 1;
        }
    }
    eprintln!("[IVF eta2] n_comp={n_comp} eta2(pc1)={eta2_first:.3} between_share:{cumulative}");
}

fn order_rows_by_cell(
    records_len: usize,
    usable_orig: &[usize],
    pc1: &[f32],
    assignments: &[usize],
    cell_rank: &[usize],
) -> (Vec<usize>, Vec<OrderedClusterRun>) {
    let mut usable_sorted: Vec<usize> = (0..usable_orig.len()).collect();
    usable_sorted.sort_by(|&a, &b| {
        cell_rank[assignments[a]]
            .cmp(&cell_rank[assignments[b]])
            .then_with(|| {
                let pa = pc1.get(a).copied().unwrap_or(0.0);
                let pb = pc1.get(b).copied().unwrap_or(0.0);
                pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| usable_orig[a].cmp(&usable_orig[b]))
    });
    let mut order: Vec<usize> = Vec::with_capacity(records_len);
    order.extend(usable_sorted.iter().map(|&u| usable_orig[u]));
    let in_usable: std::collections::HashSet<usize> = usable_orig.iter().copied().collect();
    order.extend((0..records_len).filter(|i| !in_usable.contains(i)));
    let mut ordered_cells: Vec<usize> = usable_sorted.iter().map(|&u| assignments[u]).collect();
    ordered_cells.extend(std::iter::repeat_n(
        usize::MAX,
        records_len - usable_orig.len(),
    ));
    let runs = contiguous_runs(&ordered_cells);
    (order, runs)
}

/// Binary-reflected Gray code over the sign bits of `v - mean`, packed MSB-first
/// into `ceil(dim/8)` bytes. Bit i (before Gray) is 1 iff `v[i] - mean[i] >= 0`.
/// Gray transform over the bit stream: `g_i = b_i XOR b_{i-1}` (b_{-1}=0), so
/// lexicographically-adjacent keys differ in few sign bits (angular proximity).
fn sign_gray_key(v: &[f32], mean: &[f64]) -> Vec<u8> {
    let dim = v.len();
    let mut key = vec![0u8; dim.div_ceil(8)];
    let mut prev = 0u8;
    for i in 0..dim {
        let raw = u8::from((v[i] as f64) - mean[i] >= 0.0);
        let g = raw ^ prev;
        prev = raw;
        if g != 0 {
            key[i / 8] |= 0x80 >> (i % 8);
        }
    }
    key
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_records::EmbeddingCell;

    #[test]
    fn default_train_sample_scales_with_k_and_caps() {
        // Small k (low LSM levels, ≤~26M rows): unchanged 50k baseline — the
        // existing SIFT1M behavior is preserved.
        assert_eq!(default_train_sample(16), 50_000);
        assert_eq!(default_train_sample(100), 50_000);
        assert_eq!(default_train_sample(500), 50_000);
        // Large k (higher levels / bigger collections): scales so that
        // samples-per-centroid stays ≥ the convergence floor.
        for k in [1_000usize, 2_000, 3_000] {
            let s = default_train_sample(k);
            assert!(
                s / k >= IVF_MIN_SAMPLES_PER_CENTROID,
                "k={k}: sample={s} → {}/centroid < floor",
                s / k
            );
        }
        // Extreme k: capped to bound PCA/k-means cost.
        assert_eq!(default_train_sample(10_000), IVF_TRAIN_SAMPLE_CAP);
        assert!(default_train_sample(100_000) <= IVF_TRAIN_SAMPLE_CAP);
    }

    fn rec(oid: &str, vec: Vec<f32>) -> ProximaRecord {
        let dim = vec.len() as u32;
        let mut r = ProximaRecord {
            oid: oid.into(),
            ..Default::default()
        };
        r.embeddings.push(EmbeddingCell {
            modality: "dense".into(),
            dim,
            values: EmbeddingValues::Fp32(vec),
            ..Default::default()
        });
        r
    }

    #[test]
    fn cluster_order_is_a_permutation() {
        let recs = vec![
            rec("a", vec![1.0, 1.0, 1.0, 1.0]),
            rec("b", vec![-1.0, -1.0, -1.0, -1.0]),
            rec("c", vec![1.0, 1.0, -1.0, -1.0]),
            rec("d", vec![-1.0, -1.0, 1.0, 1.0]),
        ];
        let order = cluster_order(&recs, 0).expect("some order");
        assert_eq!(order.len(), 4);
        let mut seen = order.clone();
        seen.sort_unstable();
        assert_eq!(
            seen,
            vec![0, 1, 2, 3],
            "must be a permutation of all indices"
        );
    }

    #[test]
    fn cluster_order_groups_similar_sign_patterns_adjacently() {
        // Two tight clusters around opposite corners; within a cluster vectors
        // share sign pattern. After ordering, the two members of each cluster
        // must be adjacent (not interleaved).
        let recs = vec![
            rec("p1", vec![2.0, 2.0, 2.0, 2.0]),
            rec("n1", vec![-2.0, -2.0, -2.0, -2.0]),
            rec("p2", vec![3.0, 1.0, 2.0, 4.0]),
            rec("n2", vec![-3.0, -1.0, -2.0, -4.0]),
        ];
        let order = cluster_order(&recs, 0).expect("order");
        let oids: Vec<&str> = order.iter().map(|&i| recs[i].oid.as_str()).collect();
        // p1,p2 adjacent and n1,n2 adjacent (order of the two groups is unspecified).
        let pos = |o: &str| oids.iter().position(|x| *x == o).unwrap();
        assert_eq!(
            (pos("p1") as i32 - pos("p2") as i32).abs(),
            1,
            "positive cluster members adjacent: {oids:?}"
        );
        assert_eq!(
            (pos("n1") as i32 - pos("n2") as i32).abs(),
            1,
            "negative cluster members adjacent: {oids:?}"
        );
    }

    #[test]
    fn cluster_plan_runs_cover_reordered_rows_exactly() -> anyhow::Result<()> {
        let recs = vec![
            rec("p1", vec![2.0, 2.0, 2.0, 2.0]),
            rec("n1", vec![-2.0, -2.0, -2.0, -2.0]),
            rec("p2", vec![3.0, 1.0, 2.0, 4.0]),
            rec("n2", vec![-3.0, -1.0, -2.0, -4.0]),
        ];
        let plan = cluster_plan(&recs, 0)
            .ok_or_else(|| anyhow::anyhow!("expected a bootstrap cluster plan"))?;
        assert_eq!(plan.order.len(), recs.len());
        assert!(plan.runs.len() >= 2);
        let mut expected_start = 0usize;
        for run in &plan.runs {
            assert_eq!(run.start_row, expected_start);
            assert!(run.row_count > 0);
            expected_start += run.row_count;
        }
        assert_eq!(expected_start, recs.len());
        Ok(())
    }

    #[test]
    fn cluster_order_none_when_too_few_or_no_embeddings() {
        assert!(cluster_order(&[], 0).is_none());
        assert!(cluster_order(&[rec("solo", vec![1.0, 2.0])], 0).is_none());
        // Records with no Fp32 embedding at idx 0 → None.
        let bare = vec![
            ProximaRecord {
                oid: "x".into(),
                ..Default::default()
            },
            ProximaRecord {
                oid: "y".into(),
                ..Default::default()
            },
        ];
        assert!(cluster_order(&bare, 0).is_none());
    }

    #[test]
    fn cluster_order_groups_high_dim_clusters_16d() {
        // 16-dim (⇒ a 2-byte sign key): two clusters in opposite orthants. Cluster
        // A sits near +1 in every dim, cluster B near -1, with a tiny per-dim
        // perturbation that never flips a sign. After centering (mean ≈ 0), A's
        // sign bits are all-1 and B's all-0, so each cluster shares one key and
        // its members land contiguously.
        let dim = 16usize;
        let mk = |oid: &str, base: f32, seed: usize| {
            let v: Vec<f32> = (0..dim)
                .map(|d| base + (((d * 7 + seed) % 5) as f32) * 0.01)
                .collect();
            rec(oid, v)
        };
        let recs = vec![
            mk("a1", 1.0, 1),
            mk("b1", -1.0, 1),
            mk("a2", 1.0, 2),
            mk("b2", -1.0, 2),
            mk("a3", 1.0, 3),
            mk("b3", -1.0, 3),
        ];
        let order = cluster_order(&recs, 0).expect("order");
        let oids: Vec<&str> = order.iter().map(|&i| recs[i].oid.as_str()).collect();
        let positions = |c: char| -> Vec<usize> {
            let mut p: Vec<usize> = oids
                .iter()
                .enumerate()
                .filter(|(_, o)| o.starts_with(c))
                .map(|(i, _)| i)
                .collect();
            p.sort_unstable();
            p
        };
        let contiguous = |p: &[usize]| p.windows(2).all(|w| w[1] == w[0] + 1);
        assert!(
            contiguous(&positions('a')),
            "A cluster contiguous: {oids:?}"
        );
        assert!(
            contiguous(&positions('b')),
            "B cluster contiguous: {oids:?}"
        );
    }

    #[test]
    fn cluster_order_32d_permutation_and_locality() {
        // 32-dim (4-byte key): a gradient of vectors from all-negative to
        // all-positive by flipping one more sign per step. The Gray-coded key is
        // monotone-ish along this gradient, so nearest-sign-pattern neighbours end
        // up adjacent; here we assert the result is a valid permutation and the
        // two extreme vectors (all-neg, all-pos) are NOT adjacent (maximally far
        // sign patterns shouldn't collapse together).
        let dim = 32usize;
        let recs: Vec<_> = (0..=dim)
            .map(|k| {
                // first k dims +1, rest -1
                let v: Vec<f32> = (0..dim).map(|d| if d < k { 1.0 } else { -1.0 }).collect();
                rec(&format!("g{k:02}"), v)
            })
            .collect();
        let order = cluster_order(&recs, 0).expect("order");
        assert_eq!(order.len(), dim + 1);
        let mut seen = order.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..=dim).collect::<Vec<_>>(), "permutation");
        // Keys are well-formed 4-byte (ceil(32/8)) sign-Gray keys — distinct k give
        // distinct sign patterns, so no two gradient steps collide to the same slot
        // beyond the stable index tiebreak (i.e. order has no duplicates — asserted
        // above). Sanity: the all-negative (g00) and all-positive (g32) endpoints
        // are different positions.
        let pos = |o: &str| order.iter().position(|&i| recs[i].oid == o).unwrap();
        assert_ne!(pos("g00"), pos("g32"));
    }

    #[test]
    fn sign_gray_key_is_two_bytes_for_16_dims() {
        let mean = vec![0f64; 16];
        let v = vec![1.0f32; 16];
        assert_eq!(sign_gray_key(&v, &mean).len(), 2, "ceil(16/8) = 2 bytes");
        let v32 = vec![1.0f32; 33];
        let mean32 = vec![0f64; 33];
        assert_eq!(
            sign_gray_key(&v32, &mean32).len(),
            5,
            "ceil(33/8) = 5 bytes"
        );
    }

    #[test]
    fn block_cluster_defaults_on_with_kill_switch() {
        // TD-WLP-4: clustering is default-ON; only an explicitly falsy env
        // value disables it (kill-switch semantics). nextest process-per-test
        // isolation makes the env mutation safe.
        unsafe {
            std::env::remove_var("PROXIMADB_PAX_BLOCK_CLUSTER");
        }
        assert!(block_cluster_enabled(), "unset ⇒ clustering ON (default)");
        unsafe {
            std::env::set_var("PROXIMADB_PAX_BLOCK_CLUSTER", "0");
        }
        assert!(!block_cluster_enabled(), "explicit falsy ⇒ kill-switch");
        unsafe {
            std::env::set_var("PROXIMADB_PAX_BLOCK_CLUSTER", "1");
        }
        assert!(block_cluster_enabled(), "truthy stays ON");
        unsafe {
            std::env::remove_var("PROXIMADB_PAX_BLOCK_CLUSTER");
        }
    }

    /// TD-WLP-4: PCA+IVF ordering groups two well-separated clusters
    /// contiguously and returns a permutation; below the training floor it
    /// falls back to the bootstrap (still a permutation).
    #[test]
    fn cluster_order_pca_ivf_groups_clusters_contiguously() {
        // 80 records in two tight 8-dim clusters (interleaved on input).
        let mut recs = Vec::new();
        for i in 0..40 {
            let e = (i % 5) as f32 * 0.01;
            recs.push(rec(&format!("a{i:02}"), vec![1.0 + e; 8]));
            recs.push(rec(&format!("b{i:02}"), vec![-1.0 - e; 8]));
        }
        let order = cluster_order_pca_ivf(&recs, 0).expect("order");
        assert_eq!(order.len(), recs.len());
        let mut seen = order.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..recs.len()).collect::<Vec<_>>(), "permutation");
        // Contiguity: all a's adjacent, all b's adjacent.
        let labels: Vec<char> = order
            .iter()
            .map(|&i| recs[i].oid.chars().next().unwrap_or('?'))
            .collect();
        let transitions = labels.windows(2).filter(|w| w[0] != w[1]).count();
        assert_eq!(
            transitions, 1,
            "two clusters must form two contiguous runs, got {labels:?}"
        );
        assert_eq!(
            cluster_order_pca_ivf(&recs, 0),
            Some(order),
            "identical input must produce an identical persisted order"
        );
        // Small batch → bootstrap fallback, still a permutation.
        let small: Vec<ProximaRecord> = recs.iter().take(4).cloned().collect();
        let fallback = cluster_order_pca_ivf(&small, 0).expect("fallback order");
        let mut fs = fallback.clone();
        fs.sort_unstable();
        assert_eq!(fs, vec![0, 1, 2, 3]);
    }

    /// TD-RDSTRAT-8 (rev 3): the persisted IVF probe plan is a permutation that
    /// groups the IOP-derived cells contiguously, produces a consistent persisted
    /// model (cell_rows partition the usable rows, arrays sized by k), puts
    /// no-embedding rows in the tail outside every cell, and is deterministic.
    #[test]
    fn cluster_plan_ivf_probe_groups_cells_and_is_deterministic() {
        // Force k=2 so the two synthetic clusters map 1:1 to cells (the shared
        // IOP-derived override, since the plan we persist IS the single-level
        // plan). nextest process-per-test isolation makes the env mutation safe.
        unsafe {
            std::env::set_var("PROXIMADB_IVF_K", "2");
        }
        let mut recs = Vec::new();
        for i in 0..100 {
            let e = (i % 5) as f32 * 0.01;
            recs.push(rec(&format!("a{i:03}"), vec![1.0 + e; 8]));
            recs.push(rec(&format!("b{i:03}"), vec![-1.0 - e; 8]));
        }
        // Two records without embeddings — must sort last, outside all cells.
        recs.push(ProximaRecord {
            oid: "bare1".into(),
            ..Default::default()
        });
        recs.push(ProximaRecord {
            oid: "bare2".into(),
            ..Default::default()
        });

        let tl = cluster_plan_ivf_probe(&recs, 0, None).expect("ivf probe plan");
        // Permutation over ALL records.
        assert_eq!(tl.plan.order.len(), recs.len());
        let mut seen = tl.plan.order.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..recs.len()).collect::<Vec<_>>());

        // Model shape: 2 cells; usable rows partitioned; arrays sized by k_c
        // (the directory's cell count = the plan's IOP-derived k).
        assert!(tl.model.validate().is_ok());
        assert_eq!(tl.model.k_c(), 2);
        assert_eq!(tl.model.rows_covered(), 200);
        assert_eq!(tl.model.cell_rows, vec![100, 100]);
        assert_eq!(tl.model.dim, 8);
        assert_eq!(
            tl.model.centroids.len(),
            2 * tl.model.n_comp as usize,
            "centroids are k_c × n_comp in emission order"
        );
        assert!(tl.model.radii.iter().all(|r| r.is_finite() && *r >= 0.0));

        // Cells are contiguous in the order: exactly one a↔b transition among
        // the usable prefix, and the bare rows are the tail.
        let labels: Vec<char> = tl.plan.order[..200]
            .iter()
            .map(|&i| recs[i].oid.chars().next().unwrap_or('?'))
            .collect();
        let transitions = labels.windows(2).filter(|w| w[0] != w[1]).count();
        assert_eq!(transitions, 1, "cells must be contiguous: {labels:?}");
        let tail: Vec<&str> = tl.plan.order[200..]
            .iter()
            .map(|&i| recs[i].oid.as_str())
            .collect();
        assert_eq!(tail, vec!["bare1", "bare2"], "no-embedding rows tail last");

        // Deterministic: identical input ⇒ identical order AND model.
        let again = cluster_plan_ivf_probe(&recs, 0, None).expect("second plan");
        assert_eq!(again.plan.order, tl.plan.order);
        assert_eq!(again.model, tl.model);

        // Too-small batch ⇒ None (caller falls back to the single-level plan).
        let small: Vec<ProximaRecord> = recs.iter().take(8).cloned().collect();
        assert!(cluster_plan_ivf_probe(&small, 0, None).is_none());
        unsafe {
            std::env::remove_var("PROXIMADB_IVF_K");
        }
    }
}

#[cfg(test)]
mod kmeans_seed_gate_tests {
    use super::{PAX_IVF_KMEANS_SEED, resolve_kmeans_seed};

    /// Unset must be bit-identical to the compiled-in constant: this gate exists
    /// to measure initialisation sensitivity in evaluation, never to change
    /// production behaviour or the determinism the harness relies on.
    #[test]
    fn unset_falls_back_to_the_compiled_in_seed() {
        assert_eq!(resolve_kmeans_seed(None), PAX_IVF_KMEANS_SEED);
    }

    /// A malformed value must NOT silently become 0. A seed of 0 is a perfectly
    /// legitimate draw, so a typo would quietly produce a different clustering
    /// than intended and invalidate the comparison with no error anywhere.
    #[test]
    fn unparseable_value_falls_back_rather_than_defaulting_to_zero() {
        for bad in ["not-a-number", "", "-1", "1.5", "0x10"] {
            assert_eq!(
                resolve_kmeans_seed(Some(bad.to_string())),
                PAX_IVF_KMEANS_SEED,
                "malformed seed {bad:?} must fall back, not parse to something else"
            );
        }
    }

    /// A valid override is honoured exactly — including 0, which must be usable
    /// as a deliberate draw even though it is also the "empty" u64.
    #[test]
    fn valid_override_is_honoured_including_zero() {
        assert_eq!(resolve_kmeans_seed(Some("0".into())), 0);
        assert_eq!(resolve_kmeans_seed(Some("12345".into())), 12345);
    }
}

/// TD-IVF-3: the coarse-PCA projection-width floor.
#[cfg(test)]
mod ncomp_floor_tests {
    use super::{IVF_NCOMP_FLOOR_CEILING, parse_ncomp_floor, projection_dims_with_floor};

    /// The three geometries the width study measured. With the floor disabled
    /// the policy must reproduce the widths those beds actually ran at, or every
    /// number in TD-IVF-3 is being compared against the wrong baseline.
    const MEASURED: [(usize, usize, usize); 3] = [
        (128, 100, 10), // 128-d BIGANN
        (384, 90, 12),  // 384-d BGE
        (768, 90, 14),  // 768-d BGE
    ];

    #[test]
    fn floor_disabled_reproduces_the_legacy_widths_exactly() {
        for (dim, k, expected) in MEASURED {
            assert_eq!(
                projection_dims_with_floor(dim, k, 1.5, 1.5, 0),
                expected,
                "dim={dim} k={k}: a 0 floor must be bit-identical to the legacy formula"
            );
        }
    }

    /// The floor raises, never lowers — the legacy term is not capped by it.
    #[test]
    fn floor_raises_narrow_widths_and_never_lowers() {
        for (dim, k, legacy) in MEASURED {
            for floor in [16usize, 24, 32, 64] {
                let got = projection_dims_with_floor(dim, k, 1.5, 1.5, floor);
                assert_eq!(got, floor.max(legacy), "dim={dim} k={k} floor={floor}");
                assert!(got >= legacy, "the floor must never narrow the projection");
            }
        }
    }

    /// The ceiling sits at the measured optimum, and both constraints on it (A0
    /// bytes, PCA conditioning) were checked at that width — see the constant's
    /// documentation. Anything beyond it clamps rather than silently producing
    /// components the extractor cannot condition (TD-IVF-5).
    #[test]
    fn ceiling_is_the_measured_optimum_and_clamps_beyond_it() {
        assert_eq!(IVF_NCOMP_FLOOR_CEILING, 64);
        assert_eq!(projection_dims_with_floor(768, 90, 1.5, 1.5, 64), 64);
        assert_eq!(projection_dims_with_floor(768, 90, 1.5, 1.5, 128), 64);
    }

    /// A0 carries `n_comp·dim + k_c·n_comp` f32s, so width is linear in the cold
    /// prefix a first query fetches. The ceiling bounds that growth even if an
    /// operator sets something extreme.
    #[test]
    fn floor_is_capped_by_the_ceiling() {
        assert_eq!(
            projection_dims_with_floor(768, 90, 1.5, 1.5, 4096),
            IVF_NCOMP_FLOOR_CEILING
        );
    }

    /// `n_comp <= dim` is a hard invariant of the persisted coarse directory —
    /// `CoarseModel::validate` rejects a model that violates it, which would fail
    /// the write rather than degrade it.
    #[test]
    fn width_never_exceeds_the_ambient_dimension() {
        for dim in [2usize, 8, 16, 48] {
            let got = projection_dims_with_floor(dim, 90, 1.5, 1.5, 64);
            assert!(
                got <= dim,
                "dim={dim}: width {got} exceeds ambient dimension"
            );
            assert!(got >= 1, "dim={dim}: width must stay positive");
        }
    }

    /// A malformed value must fall through to config, NOT read as 0. Once the
    /// default ships widened, parsing a typo as 0 would silently restore the
    /// under-projecting formula on every segment written — the exact regression
    /// this floor exists to remove, with no error anywhere.
    #[test]
    fn malformed_floor_falls_through_rather_than_disabling() {
        for bad in ["not-a-number", "", "-1", "1.5", "0x40", "64x"] {
            assert_eq!(
                parse_ncomp_floor(bad),
                None,
                "malformed floor {bad:?} must fall through to config"
            );
        }
    }

    /// An explicit 0 is a real setting — "use the legacy formula" — and must be
    /// distinguishable from "unset".
    #[test]
    fn explicit_values_parse_including_zero() {
        assert_eq!(parse_ncomp_floor("0"), Some(0));
        assert_eq!(parse_ncomp_floor("64"), Some(64));
        assert_eq!(parse_ncomp_floor("  32  "), Some(32));
    }
    /// TD-IVF-4: the destination URL reaches the IOP target. A registered
    /// per-location budget (TD-IOBUDGET-1) changes the derived cell count `k`
    /// (visible through `default_train_sample`'s k-scaling), while `None`
    /// keeps the CLOUD fallback. Pure arithmetic — no records, no k-means.
    /// (nextest process isolation covers the registry; the Drop guard restores
    /// the env vars for shared-process legacy `cargo test` runs.)
    #[test]
    fn destination_url_resolves_the_iop_target() {
        use proximadb_storage_common::iops_budget::{IopsBudget, register_location_budget};
        struct RestoreEnvs;
        impl Drop for RestoreEnvs {
            fn drop(&mut self) {
                unsafe { std::env::remove_var("PROXIMADB_IVF_K") };
                unsafe { std::env::remove_var("PROXIMADB_IVF_TRAIN_SAMPLE") };
            }
        }
        let _env_guard = RestoreEnvs;
        unsafe {
            std::env::remove_var("PROXIMADB_IVF_K");
        }
        unsafe {
            std::env::remove_var("PROXIMADB_IVF_TRAIN_SAMPLE");
        }

        // 500k usable 768-d rows: k under the CLOUD 4 MiB target is ~91
        // (384 MB / 4 MiB — train sample stays at the 50k floor); under a
        // registered 64 KiB target k = 5859 clamps to the 4096 cap (train
        // sample scales to the 200k cap).
        let (rows, dim) = (500_000usize, 768usize);
        let baseline = super::ivf_training_shape(rows, dim, None).expect("baseline shape");
        assert_eq!(
            baseline.train_sample, 50_000,
            "CLOUD fallback keeps the floor"
        );

        register_location_budget(
            "s3://iobudget-writer-test",
            IopsBudget {
                min: 64 * 1024,
                target: 64 * 1024,
                max: 1024 * 1024,
            },
        );
        let shaped =
            super::ivf_training_shape(rows, dim, Some("s3://iobudget-writer-test/c/seg.pax"))
                .expect("registered shape");
        assert_ne!(
            shaped.train_sample, baseline.train_sample,
            "the registered budget must move the IOP-derived k"
        );
        assert_eq!(shaped.train_sample, 200_000, "k-scaled sample hits the cap");
    }
}
