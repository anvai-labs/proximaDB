//! TurboQuant scoring kernels.
//!
//! Public entry point: [`search`] takes a query, an `EncodedBatch`, the
//! rotation matrix, the centroid table, an optional `Calibration`, the
//! desired top-`k`, and an optional packed allowlist bitmap; returns the
//! top-`k` `(score, slot_index)` pairs in descending score order.
//!
//! ## Phase coverage
//!
//! - **P4.2 (this session)**: portable scalar kernel — correctness oracle
//!   for the SIMD kernels in subsequent sessions. Runtime dispatch picks
//!   it unconditionally for now.
//! - **P4.3+ (future sessions)**: NEON / AVX2 / AVX-512BW kernels gated
//!   by the runtime `HardwareCapabilities` detection (Q14 of
//!   `EMBEDDING_PRECISION_LLD_2026_05_22`). The scalar kernel stays as a
//!   correctness fallback for CPUs without SIMD.
//!
//! ## Bit-identical scoring (LLD Q11)
//!
//! All kernels MUST produce element-wise-equal scores for the same
//! `(rotation, calibration, codes, query)` tuple. The scalar kernel is
//! the reference. Subsequent SIMD kernels are written to match it via the
//! `FLUSH_EVERY=256` periodic accumulator flush technique. The Implementation
//! Status table in `TURBOQUANT_LLD_2026_05_30.adoc` is the live tracker
//! for which kernels are wired in.

pub mod scalar;

use super::{Calibration, TurboQuantError, encode::EncodedBatch};

/// One search result: `(score, slot_index)`. Slot indices are positional
/// into the encoded batch — translation to external IDs happens in the
/// `IdMapIndex` layer (planned for P6).
pub type SearchHit = (f32, u32);

/// Run a top-`k` search against an encoded batch.
///
/// # Arguments
///
/// - `query` — raw input vector of length `index.dim`. Not normalised by
///   the caller; the kernel normalises and rotates internally.
/// - `index` — `EncodedBatch` produced by `encode::encode_batch`.
/// - `rotation` — same `dim x dim` row-major matrix used at encode.
/// - `centroids` — same `2^bit_width` centroid table used at encode.
/// - `calibration` — `Some(cal)` when the index was encoded with TQ+;
///   `None` for identity calibration.
/// - `k` — number of top results to return. Bounded by `index.n_vectors`
///   and (when `mask` is `Some`) the number of allowed slots.
/// - `mask` — `Some(bitmap)` for in-kernel allowlist filtering; `None`
///   to scan every slot.
///
/// # Errors
///
/// - `DimNotMultipleOf8` / `BitWidthOutOfRange` — propagated when index
///   shape doesn't match constants.
/// - `VectorBufferNotMultipleOfDim` — `query.len() != index.dim`.
/// - `InvalidInputValue` — query contains NaN / Inf / `|x| >= 1e16`.
///
/// # Routing
///
/// Today dispatches unconditionally to [`scalar::search_scalar`]. When
/// future sessions land SIMD kernels, this function will inspect the
/// runtime `HardwareCapabilities` registry (Q14) and route accordingly,
/// keeping the scalar path as the always-available fallback.
pub fn search(
    query: &[f32],
    index: &EncodedBatch,
    rotation: &[f32],
    centroids: &[f32],
    calibration: Option<&Calibration>,
    k: usize,
    mask: Option<&[u64]>,
) -> Result<Vec<SearchHit>, TurboQuantError> {
    super::check_dim(index.dim)?;
    super::check_bit_width(index.bit_width)?;
    if query.len() != index.dim {
        return Err(TurboQuantError::VectorBufferNotMultipleOfDim {
            vectors_len: query.len(),
            dim: index.dim,
        });
    }
    if let Some((vi, ci, val)) = super::first_invalid_coord(query, index.dim) {
        return Err(TurboQuantError::InvalidInputValue {
            vector_index: vi,
            coord_index: ci,
            value: val,
        });
    }
    debug_assert_eq!(rotation.len(), index.dim * index.dim);
    debug_assert_eq!(centroids.len(), 1 << index.bit_width);
    if let Some(c) = calibration {
        debug_assert_eq!(c.dim(), index.dim);
    }
    if let Some(m) = mask {
        let needed_words = (index.n_vectors + 63) >> 6;
        debug_assert!(
            m.len() >= needed_words,
            "mask too short: {} words, need {} for n_vectors = {}",
            m.len(),
            needed_words,
            index.n_vectors,
        );
    }

    Ok(scalar::search_scalar(
        query,
        index,
        rotation,
        centroids,
        calibration,
        k,
        mask,
    ))
}

#[cfg(test)]
mod tests {
    use super::super::{
        calibration::fit_calibration, codebook::codebook, encode::encode_batch,
        rotation::make_rotation_matrix,
    };
    use super::*;
    use chacha20::ChaCha8Rng;
    use rand::{RngExt, SeedableRng};
    use rand_distr::StandardNormal;

    /// Generate `n` random unit-norm vectors at dim `dim` with the given
    /// seed.
    fn random_unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut v = vec![0.0f32; n * dim];
        for i in 0..n {
            let mut sumsq = 0.0f64;
            for d in 0..dim {
                let x: f64 = rng.sample(StandardNormal);
                v[i * dim + d] = x as f32;
                sumsq += x * x;
            }
            let inv = if sumsq > 1e-30 {
                (1.0 / sumsq.sqrt()) as f32
            } else {
                0.0
            };
            for d in 0..dim {
                v[i * dim + d] *= inv;
            }
        }
        v
    }

    /// Helper: encode + search round-trip with no mask, no calibration.
    fn encode_and_search(
        vectors: &[f32],
        query: &[f32],
        dim: usize,
        bit_width: u8,
        seed: u64,
        k: usize,
    ) -> Vec<SearchHit> {
        let r = make_rotation_matrix(dim, seed);
        let (bnd, c) = codebook(bit_width as usize, dim);
        let batch = encode_batch(vectors, dim, bit_width, &r, &bnd, &c, None).unwrap();
        search(query, &batch, &r, &c, None, k, None).unwrap()
    }

    #[test]
    fn rejects_query_with_wrong_dim() {
        let dim = 16;
        let v = random_unit_vectors(4, dim, 1);
        let r = make_rotation_matrix(dim, 1);
        let (bnd, c) = codebook(2, dim);
        let batch = encode_batch(&v, dim, 2, &r, &bnd, &c, None).unwrap();
        let q = vec![0.5f32; dim - 1];
        let err = search(&q, &batch, &r, &c, None, 1, None).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::VectorBufferNotMultipleOfDim { .. }
        ));
    }

    #[test]
    fn rejects_query_with_nan() {
        let dim = 16;
        let v = random_unit_vectors(4, dim, 1);
        let r = make_rotation_matrix(dim, 1);
        let (bnd, c) = codebook(2, dim);
        let batch = encode_batch(&v, dim, 2, &r, &bnd, &c, None).unwrap();
        let mut q = vec![0.5f32; dim];
        q[3] = f32::NAN;
        let err = search(&q, &batch, &r, &c, None, 1, None).unwrap_err();
        assert!(matches!(err, TurboQuantError::InvalidInputValue { .. }));
    }

    #[test]
    fn empty_index_returns_empty_results() {
        let dim = 16;
        let r = make_rotation_matrix(dim, 1);
        let (bnd, c) = codebook(2, dim);
        let batch = encode_batch(&[], dim, 2, &r, &bnd, &c, None).unwrap();
        let q = vec![0.5f32; dim];
        let hits = search(&q, &batch, &r, &c, None, 5, None).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn k_is_capped_to_n_vectors() {
        let dim = 16;
        let v = random_unit_vectors(3, dim, 1);
        let q = random_unit_vectors(1, dim, 2);
        let hits = encode_and_search(&v, &q, dim, 4, 7, 100);
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn top_k_is_sorted_descending() {
        let dim = 32;
        let v = random_unit_vectors(20, dim, 42);
        let q = random_unit_vectors(1, dim, 43);
        let hits = encode_and_search(&v, &q, dim, 4, 100, 10);
        assert_eq!(hits.len(), 10);
        for w in hits.windows(2) {
            assert!(
                w[0].0 >= w[1].0,
                "results not sorted: {} then {}",
                w[0].0,
                w[1].0,
            );
        }
    }

    #[test]
    fn top_k_indices_are_unique() {
        let dim = 32;
        let v = random_unit_vectors(20, dim, 11);
        let q = random_unit_vectors(1, dim, 12);
        let hits = encode_and_search(&v, &q, dim, 2, 13, 10);
        let mut idxs: Vec<u32> = hits.iter().map(|h| h.1).collect();
        idxs.sort_unstable();
        idxs.dedup();
        assert_eq!(idxs.len(), 10, "duplicate indices in top-k");
    }

    #[test]
    fn self_search_picks_self_at_top_4bit_at_high_dim() {
        // At 4-bit, d=256, searching for a vector that's in the index
        // should usually return that vector as the top hit. We test with
        // 20 random vectors and pick one of them as the query.
        let dim = 256;
        let v = random_unit_vectors(20, dim, 77);
        // Use vector 7 as the query (raw, not yet quantized).
        let q_slice = &v[7 * dim..(7 + 1) * dim];
        let r = make_rotation_matrix(dim, 99);
        let (bnd, c) = codebook(4, dim);
        let batch = encode_batch(&v, dim, 4, &r, &bnd, &c, None).unwrap();
        let hits = search(q_slice, &batch, &r, &c, None, 1, None).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits[0].1, 7,
            "self-query should return self; got slot {} with score {}",
            hits[0].1, hits[0].0,
        );
    }

    #[test]
    fn search_results_are_deterministic() {
        let dim = 32;
        let v = random_unit_vectors(10, dim, 5);
        let q = random_unit_vectors(1, dim, 6);
        let a = encode_and_search(&v, &q, dim, 4, 7, 5);
        let b = encode_and_search(&v, &q, dim, 4, 7, 5);
        assert_eq!(a, b);
    }

    #[test]
    fn mask_zero_returns_empty() {
        let dim = 64;
        let v = random_unit_vectors(64, dim, 1);
        let q = random_unit_vectors(1, dim, 2);
        let r = make_rotation_matrix(dim, 3);
        let (bnd, c) = codebook(4, dim);
        let batch = encode_batch(&v, dim, 4, &r, &bnd, &c, None).unwrap();
        // All-zero mask covering the 64 slots: 1 word.
        let mask = vec![0u64];
        let hits = search(&q, &batch, &r, &c, None, 5, Some(&mask)).unwrap();
        assert!(hits.is_empty(), "all-zero mask must return no hits");
    }

    #[test]
    fn mask_selective_restricts_to_allowed_slots() {
        let dim = 64;
        let n = 64;
        let v = random_unit_vectors(n, dim, 4);
        let q = random_unit_vectors(1, dim, 5);
        let r = make_rotation_matrix(dim, 6);
        let (bnd, c) = codebook(4, dim);
        let batch = encode_batch(&v, dim, 4, &r, &bnd, &c, None).unwrap();
        // Allow only slots 3, 17, 42.
        let allowed = [3u32, 17, 42];
        let mut mask = vec![0u64; (n + 63) >> 6];
        for &s in &allowed {
            mask[(s >> 6) as usize] |= 1u64 << (s & 63);
        }
        let hits = search(&q, &batch, &r, &c, None, 5, Some(&mask)).unwrap();
        assert_eq!(hits.len(), 3);
        let mut idxs: Vec<u32> = hits.iter().map(|h| h.1).collect();
        idxs.sort_unstable();
        assert_eq!(idxs, allowed);
    }

    #[test]
    fn mask_top_k_capped_to_allowed_count() {
        let dim = 64;
        let n = 64;
        let v = random_unit_vectors(n, dim, 14);
        let q = random_unit_vectors(1, dim, 15);
        let r = make_rotation_matrix(dim, 16);
        let (bnd, c) = codebook(4, dim);
        let batch = encode_batch(&v, dim, 4, &r, &bnd, &c, None).unwrap();
        // Allow exactly 2 slots, ask for top-10 → return 2.
        let mut mask = vec![0u64; (n + 63) >> 6];
        mask[0] |= 1u64 << 5;
        mask[0] |= 1u64 << 25;
        let hits = search(&q, &batch, &r, &c, None, 10, Some(&mask)).unwrap();
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn search_with_calibration_round_trip() {
        // Encode with TQ+, search with the same calibration. Self-query
        // must still find self at top.
        let dim = 128;
        let n = 1024;
        let v = random_unit_vectors(n, dim, 21);
        let r = make_rotation_matrix(dim, 22);
        let (bnd, c) = codebook(4, dim);

        // Build calibration from the rotated batch.
        let mut rotated = vec![0.0f32; n * dim];
        for i in 0..n {
            let row = &v[i * dim..(i + 1) * dim];
            for k in 0..dim {
                let r_row = &r[k * dim..(k + 1) * dim];
                let mut acc = 0.0f64;
                for j in 0..dim {
                    acc += (r_row[j] as f64) * (row[j] as f64);
                }
                rotated[i * dim + k] = acc as f32;
            }
        }
        let cal = fit_calibration(&rotated, n, dim).unwrap();

        let batch = encode_batch(&v, dim, 4, &r, &bnd, &c, Some(&cal)).unwrap();
        let q = &v[42 * dim..(42 + 1) * dim];
        let hits = search(q, &batch, &r, &c, Some(&cal), 1, None).unwrap();
        assert_eq!(hits[0].1, 42, "self-query with TQ+ should find self");
    }

    // ------------------------------------------------------------------
    // P9.A — Recall validation + mask correctness
    // ------------------------------------------------------------------
    //
    // These tests exercise the full P1–P7 pipeline against an oracle
    // computed in raw FP32. They're the algorithm-correctness battery
    // future SIMD kernels (P4 NEON/AVX2/AVX-512BW) and algorithm tweaks
    // gate against. The recall thresholds are conservative — far below
    // the turbovec-published numbers on real OpenAI embeddings — because
    // random unit-vector inputs are the hardest case (no structure to
    // exploit) and we run at d=256 (lower than the 1536/3072 used in the
    // paper's R@10 ≥ 0.95 figures). The formal turbovec-numbers
    // reproduction is deferred per LLD §"Test Plan" — it requires real
    // OpenAI DBpedia dataset wiring.

    /// Brute-force FP32 cosine top-k oracle. Same scoring shape as the
    /// kernel: `<v, q>` against unit-normalised database vectors.
    fn brute_force_top_k(
        vectors: &[f32],
        query: &[f32],
        n: usize,
        dim: usize,
        k: usize,
    ) -> Vec<u32> {
        let mut q_unit = query.to_vec();
        let mut sumsq = 0.0f64;
        for &x in query {
            sumsq += (x as f64) * (x as f64);
        }
        let inv = if sumsq > 1e-30 {
            (1.0 / sumsq.sqrt()) as f32
        } else {
            0.0
        };
        for x in q_unit.iter_mut() {
            *x *= inv;
        }
        let mut scored: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let row = &vectors[i * dim..(i + 1) * dim];
                let mut acc = 0.0f64;
                for d in 0..dim {
                    acc += (row[d] as f64) * (q_unit[d] as f64);
                }
                (i, acc as f32)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.iter().take(k).map(|(i, _)| *i as u32).collect()
    }

    /// Compute recall@k: fraction of oracle hits that also appear in the
    /// kernel's top-k.
    fn recall_at_k(oracle: &[u32], kernel: &[(f32, u32)], k: usize) -> f32 {
        let oracle_set: std::collections::HashSet<u32> = oracle.iter().take(k).copied().collect();
        let hits = kernel
            .iter()
            .take(k)
            .filter(|h| oracle_set.contains(&h.1))
            .count();
        hits as f32 / k as f32
    }

    #[test]
    fn recall_at_10_4bit_d256_meets_floor() {
        // 4-bit at d=256, n=500, q=20 random unit vectors. Random unit
        // input is the hardest input case (no structure to exploit). At
        // d=1536 the published R@10 is ~0.95; at d=256 we expect ~0.70+.
        let dim = 256;
        let n = 500;
        let n_queries = 20;
        let vectors = random_unit_vectors(n, dim, 100);
        let queries = random_unit_vectors(n_queries, dim, 200);
        let r = make_rotation_matrix(dim, 300);
        let (bnd, c) = codebook(4, dim);
        let batch = encode_batch(&vectors, dim, 4, &r, &bnd, &c, None).unwrap();

        let mut recalls = Vec::with_capacity(n_queries);
        for qi in 0..n_queries {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let oracle = brute_force_top_k(&vectors, q, n, dim, 10);
            let hits = search(q, &batch, &r, &c, None, 10, None).unwrap();
            recalls.push(recall_at_k(&oracle, &hits, 10));
        }
        let mean = recalls.iter().sum::<f32>() / n_queries as f32;
        // Conservative gate — published numbers are higher at higher d.
        assert!(
            mean >= 0.60,
            "4-bit recall@10 at d=256 below floor: mean = {mean}",
        );
    }

    #[test]
    fn recall_at_10_2bit_d256_meets_floor() {
        // 2-bit is the aggressive-compression case (16x at d=1536).
        // Random unit at d=256 → ~0.40+ R@10 floor.
        let dim = 256;
        let n = 500;
        let n_queries = 20;
        let vectors = random_unit_vectors(n, dim, 101);
        let queries = random_unit_vectors(n_queries, dim, 201);
        let r = make_rotation_matrix(dim, 301);
        let (bnd, c) = codebook(2, dim);
        let batch = encode_batch(&vectors, dim, 2, &r, &bnd, &c, None).unwrap();

        let mut recalls = Vec::with_capacity(n_queries);
        for qi in 0..n_queries {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let oracle = brute_force_top_k(&vectors, q, n, dim, 10);
            let hits = search(q, &batch, &r, &c, None, 10, None).unwrap();
            recalls.push(recall_at_k(&oracle, &hits, 10));
        }
        let mean = recalls.iter().sum::<f32>() / n_queries as f32;
        assert!(
            mean >= 0.30,
            "2-bit recall@10 at d=256 below floor: mean = {mean}",
        );
    }

    #[test]
    fn mask_path_matches_full_scan_on_allowed_subset() {
        // Verifies: searching with a mask covering slots `S` returns the
        // same top-k as searching unrestricted and post-filtering to `S`.
        // This is the correctness contract the kernel's mask early-exit
        // must preserve regardless of how blocks are skipped.
        let dim = 128;
        let n = 200;
        let vectors = random_unit_vectors(n, dim, 400);
        let query = &random_unit_vectors(1, dim, 401);
        let r = make_rotation_matrix(dim, 402);
        let (bnd, c) = codebook(4, dim);
        let batch = encode_batch(&vectors, dim, 4, &r, &bnd, &c, None).unwrap();

        // Pick a 10% allowlist deterministically.
        let allowed_slots: Vec<u32> = (0..n as u32).filter(|s| s % 10 == 0).collect();
        let n_words = (n + 63) >> 6;
        let mut mask = vec![0u64; n_words];
        for &s in &allowed_slots {
            mask[(s >> 6) as usize] |= 1u64 << (s & 63);
        }

        // Mask path: kernel returns top-k restricted to allowed slots.
        let mask_hits = search(query, &batch, &r, &c, None, 5, Some(&mask)).unwrap();
        // Reference path: unrestricted search at k=n, then post-filter.
        let all_hits = search(query, &batch, &r, &c, None, n, None).unwrap();
        let allowed_set: std::collections::HashSet<u32> = allowed_slots.iter().copied().collect();
        let reference: Vec<(f32, u32)> = all_hits
            .into_iter()
            .filter(|h| allowed_set.contains(&h.1))
            .take(5)
            .collect();

        assert_eq!(mask_hits.len(), reference.len());
        for (m, r) in mask_hits.iter().zip(reference.iter()) {
            assert_eq!(m.1, r.1, "mask vs full-scan slot mismatch");
            // Scores must be element-wise equal (mask path computes the
            // same per-vector score; it just skips non-allowed blocks).
            assert!(
                (m.0 - r.0).abs() < 1e-5,
                "mask vs full-scan score drift: {} vs {}",
                m.0,
                r.0,
            );
        }
    }

    #[test]
    fn recall_at_10_4bit_d1536_meets_floor() {
        // Production-scale dim. Paper at d=1536 reports R@10 >= 0.95
        // on real OpenAI embeddings. Random unit input is the hardest
        // case (no structure to exploit), so we assert a conservative
        // floor of R@10 >= 0.75. A regression below this catches
        // catastrophic algorithmic mistakes; the formal turbovec-numbers
        // reproduction on real OpenAI DBpedia datasets lives in P9.C
        // (deferred — requires external dataset wiring).
        let dim = 1536;
        let n = 1000;
        let n_queries = 10;
        let vectors = random_unit_vectors(n, dim, 700);
        let queries = random_unit_vectors(n_queries, dim, 701);
        let r = make_rotation_matrix(dim, 702);
        let (bnd, c) = codebook(4, dim);
        let batch = encode_batch(&vectors, dim, 4, &r, &bnd, &c, None).unwrap();

        let mut recalls = Vec::with_capacity(n_queries);
        for qi in 0..n_queries {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let oracle = brute_force_top_k(&vectors, q, n, dim, 10);
            let hits = search(q, &batch, &r, &c, None, 10, None).unwrap();
            recalls.push(recall_at_k(&oracle, &hits, 10));
        }
        let mean = recalls.iter().sum::<f32>() / n_queries as f32;
        assert!(
            mean >= 0.75,
            "4-bit recall@10 at d=1536 below floor: mean = {mean}",
        );
    }

    #[test]
    fn mask_path_is_faster_than_full_scan_at_10pct_selectivity() {
        // LLD §"Acceptance Criteria" #4 (paraphrased): "kernel-pushed
        // mask beats oversample-post-filter at 10% selectivity by ≥50%".
        // The strict 50% target is for the formal speed bench (P9.C) on
        // real datasets at n=100k. Here we assert the **directional**
        // claim: the in-kernel mask path is meaningfully faster than the
        // unrestricted full-scan path. Margin of 1.3× leaves noise room
        // for CI jitter while still failing on regressions that
        // accidentally disable the block-skip.
        //
        // The mask covers ~10% of slots in a **contiguous** layout (slots
        // 0..n/10). This matches the realistic multi-tenant case where a
        // tenant's vectors cluster together in the index. With a random
        // 10% mask, every 32-vec block would contain ≥3 set bits at
        // expectation → block-skip never fires. Clustered masks are the
        // LLD's intended fast path; the per-slot mask check still applies
        // when blocks contain mixed bits (e.g., tenant boundary blocks).

        use super::super::mask;
        use std::time::Instant;

        let dim = 128;
        let n = 10000;
        let vectors = random_unit_vectors(n, dim, 900);
        let query = random_unit_vectors(1, dim, 901);
        let r = make_rotation_matrix(dim, 902);
        let (bnd, c) = codebook(4, dim);
        let batch = encode_batch(&vectors, dim, 4, &r, &bnd, &c, None).unwrap();

        // Contiguous 10% allowlist: slots 0..1000 of 10000.
        let allowed = n / 10;
        let n_words = (n + 63) >> 6;
        let mut bitmap = vec![0u64; n_words];
        for slot in 0..allowed {
            bitmap[slot >> 6] |= 1u64 << (slot & 63);
        }

        // Reset the block-skip counter so we can confirm the mask path
        // actually exercised the early-exit.
        mask::reset_blocks_skipped_by_mask();
        let before_skip = mask::blocks_skipped_by_mask();

        // Run each path multiple times and take the median. Warm-up the
        // first iteration to absorb code-cache / branch-predictor noise.
        let warmup_full = search(&query, &batch, &r, &c, None, 10, None).unwrap();
        let _ = warmup_full;
        let warmup_mask = search(&query, &batch, &r, &c, None, 10, Some(&bitmap)).unwrap();
        let _ = warmup_mask;

        const REPS: usize = 5;
        let mut full_times = Vec::with_capacity(REPS);
        let mut mask_times = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t0 = Instant::now();
            let _ = search(&query, &batch, &r, &c, None, 10, None).unwrap();
            full_times.push(t0.elapsed());

            let t0 = Instant::now();
            let _ = search(&query, &batch, &r, &c, None, 10, Some(&bitmap)).unwrap();
            mask_times.push(t0.elapsed());
        }
        full_times.sort();
        mask_times.sort();
        let full_median = full_times[REPS / 2];
        let mask_median = mask_times[REPS / 2];

        // Directional claim: mask path is at least 1.3× faster than full
        // scan. The formal LLD 50%-faster gate sits on real datasets
        // (P9.C deferred).
        let speedup = full_median.as_secs_f64() / mask_median.as_secs_f64();
        assert!(
            speedup >= 1.3,
            "mask path not faster: full={:?}, mask={:?}, speedup={speedup}",
            full_median,
            mask_median,
        );

        // Confirm the block-skip counter advanced — proves the
        // early-exit fired and wasn't optimized out by the compiler.
        let after_skip = mask::blocks_skipped_by_mask();
        assert!(
            after_skip > before_skip,
            "BLOCKS_SKIPPED_BY_MASK didn't advance ({before_skip} → {after_skip})",
        );
    }

    #[test]
    fn tq_plus_meets_or_exceeds_identity_recall_on_anisotropic_data() {
        // Construct biased input (coord 0 is shifted), fit TQ+ from it,
        // measure R@10 with and without calibration. TQ+ should not
        // reduce recall on anisotropic data; per the paper it typically
        // increases it at low bit-widths. Conservative assertion: TQ+
        // recall is within 5pp of identity, AND identity recall is above
        // a reasonable floor.
        let dim = 128;
        let n = 1024; // ≥ TQPLUS_MIN_SAMPLES
        let n_queries = 20;

        // Build a batch where coord 0 is biased upward.
        let mut rng = ChaCha8Rng::seed_from_u64(500);
        let mut vectors = vec![0.0f32; n * dim];
        for i in 0..n {
            let mut sumsq = 0.0f64;
            for d in 0..dim {
                let mut x: f64 = rng.sample(StandardNormal);
                if d == 0 {
                    x += 0.7;
                }
                vectors[i * dim + d] = x as f32;
                sumsq += x * x;
            }
            let inv = if sumsq > 1e-30 {
                (1.0 / sumsq.sqrt()) as f32
            } else {
                0.0
            };
            for d in 0..dim {
                vectors[i * dim + d] *= inv;
            }
        }
        let queries = random_unit_vectors(n_queries, dim, 501);

        let r = make_rotation_matrix(dim, 502);
        let (bnd, c) = codebook(4, dim);

        // Identity-cal encode + search.
        let batch_id = encode_batch(&vectors, dim, 4, &r, &bnd, &c, None).unwrap();
        let mut recall_id_sum = 0.0f32;
        for qi in 0..n_queries {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let oracle = brute_force_top_k(&vectors, q, n, dim, 10);
            let hits = search(q, &batch_id, &r, &c, None, 10, None).unwrap();
            recall_id_sum += recall_at_k(&oracle, &hits, 10);
        }
        let recall_id = recall_id_sum / n_queries as f32;

        // Fit TQ+ from the rotated batch.
        let mut rotated = vec![0.0f32; n * dim];
        for i in 0..n {
            let row = &vectors[i * dim..(i + 1) * dim];
            for k in 0..dim {
                let r_row = &r[k * dim..(k + 1) * dim];
                let mut acc = 0.0f64;
                for j in 0..dim {
                    acc += (r_row[j] as f64) * (row[j] as f64);
                }
                rotated[i * dim + k] = acc as f32;
            }
        }
        let cal = fit_calibration(&rotated, n, dim).unwrap();

        // TQ+ encode + search.
        let batch_tq = encode_batch(&vectors, dim, 4, &r, &bnd, &c, Some(&cal)).unwrap();
        let mut recall_tq_sum = 0.0f32;
        for qi in 0..n_queries {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let oracle = brute_force_top_k(&vectors, q, n, dim, 10);
            let hits = search(q, &batch_tq, &r, &c, Some(&cal), 10, None).unwrap();
            recall_tq_sum += recall_at_k(&oracle, &hits, 10);
        }
        let recall_tq = recall_tq_sum / n_queries as f32;

        // Identity floor — anisotropic data is harder than uniform.
        assert!(
            recall_id >= 0.40,
            "4-bit identity recall on anisotropic d=128 below floor: {recall_id}",
        );
        // TQ+ must not degrade recall meaningfully. We don't assert TQ+ >
        // identity strictly because the test set is small (n_queries=20)
        // and the per-query swing is noisy; the strong claim is that
        // calibration doesn't break recall.
        assert!(
            recall_tq + 0.05 >= recall_id,
            "TQ+ recall {recall_tq} far below identity {recall_id} \
             (more than 5pp drop is unexpected — investigate)",
        );
    }
}
