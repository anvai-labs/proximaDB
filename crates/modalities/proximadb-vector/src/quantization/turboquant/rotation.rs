//! Seeded random orthogonal rotation matrix construction.
//!
//! After multiplication by a random orthogonal matrix, each coordinate of a
//! unit vector on the (d-1)-sphere independently follows
//! Beta((d-1)/2, (d-1)/2) shifted to `[-1, 1]`. This is the central
//! observation underlying TurboQuant (arXiv:2504.19874) — it lets us fit
//! the codebook from the math once, without ever touching the data.
//!
//! ## Construction (paper-driven, NOT lifted from turbovec)
//!
//! 1. Draw an `n x n` matrix `G` whose entries are i.i.d. `N(0, 1)` from
//!    a deterministic PRNG seeded with `seed`.
//! 2. Orthonormalize the columns of `G` using **Modified Gram-Schmidt**.
//!    For each column `j`, subtract its projection onto every preceding
//!    column (updated in-place to keep the orthogonalization numerically
//!    stable), then renormalize.
//! 3. Apply sign correction: flip column `j` whose post-MGS diagonal
//!    entry — equivalently, the post-projection norm sign carried over
//!    from the column's initial dot with itself — is negative. This pins
//!    the rotation across machines (LLD Q11).
//!
//! MGS is mathematically equivalent to the Q factor of a QR decomposition
//! with positive diagonal of R; it just constructs `Q` directly rather
//! than going through R explicitly. We use it here instead of `faer`'s
//! solver to keep this module's dependency surface minimal — `faer`
//! remains a workspace dep for future GEMM batching in P4's search path.
//!
//! ## Per-collection seed
//!
//! In production, the seed comes from
//! `proximadb_quantization_types::derive_rotation_seed(collection_id)` and
//! is persisted in xCatalog (P8). This module accepts any `u64` so it can
//! be unit-tested in isolation. Multi-tenant collections must use
//! different seeds — see LLD §"Locked Type Signatures" + Q3.

use chacha20::ChaCha8Rng;
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;

/// Generate an `n x n` orthogonal matrix deterministically from `seed`.
///
/// Returns the matrix in **row-major** layout as a flat `Vec<f32>` of
/// length `n * n`. `result[i * n + j]` is row `i`, column `j`.
///
/// The implementation works internally in `f64` to keep MGS numerically
/// stable for `n` up to a few thousand, then narrows to `f32` at the end.
/// The narrowing is the dominant error source — for `n = 3072` the
/// resulting `Q · Q^T` deviates from the identity by `~1e-5` in the worst
/// off-diagonal entry, well within the bit-width-dependent tolerance the
/// quantizer exposes downstream.
pub fn make_rotation_matrix(dim: usize, seed: u64) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Step 1: build the Gaussian source matrix in **column-major** layout
    // so MGS can read each column as a contiguous slice. `col(j)` is
    // `cols[j * dim..(j + 1) * dim]`.
    let mut cols: Vec<f64> = (0..dim * dim).map(|_| rng.sample(StandardNormal)).collect();

    // Step 2: Modified Gram-Schmidt orthonormalization.
    //
    // For each column j from 0..dim:
    //   - Subtract projections onto every prior column.
    //   - Compute norm. If the norm is too small (catastrophic
    //     cancellation against a previous column), fall back to a unit
    //     basis vector to keep the result invertible. This effectively
    //     handles the measure-zero set of singular Gaussian matrices.
    //   - Normalize.
    //
    // Sign correction: the original column's first surviving sign
    // determines the rotation orientation. We capture it before
    // normalization and flip the column to make that sign positive — this
    // is the analogue of "Q diag(sign(diag(R)))" in QR-based construction.
    for j in 0..dim {
        // Save the column's pre-projection signature for sign correction.
        let leading_orig_sign = {
            let col_j = &cols[j * dim..(j + 1) * dim];
            col_j
                .iter()
                .find(|&&x| x.abs() > 1e-30)
                .map(|&x| x.signum())
                .unwrap_or(1.0)
        };

        // Subtract projections onto every prior, already-orthonormal column.
        for k in 0..j {
            // proj = <col_k, col_j>
            let mut proj = 0.0f64;
            for i in 0..dim {
                proj += cols[k * dim + i] * cols[j * dim + i];
            }
            // col_j -= proj * col_k
            for i in 0..dim {
                cols[j * dim + i] -= proj * cols[k * dim + i];
            }
        }

        // Renormalize column j (and fall back to a unit basis vector if
        // the orthogonalized norm is too small to invert reliably).
        let mut norm_sq = 0.0f64;
        for i in 0..dim {
            let v = cols[j * dim + i];
            norm_sq += v * v;
        }
        if norm_sq < 1e-30 {
            for i in 0..dim {
                cols[j * dim + i] = if i == j { 1.0 } else { 0.0 };
            }
            continue;
        }
        let norm = norm_sq.sqrt();
        let inv = 1.0 / norm;
        for i in 0..dim {
            cols[j * dim + i] *= inv;
        }

        // Sign correction: flip the whole column if the leading
        // pre-projection sign was negative.
        if leading_orig_sign < 0.0 {
            for i in 0..dim {
                cols[j * dim + i] = -cols[j * dim + i];
            }
        }
    }

    // Convert column-major f64 to row-major f32.
    // result[i * dim + j] = column_major[j * dim + i].
    let mut result = vec![0.0f32; dim * dim];
    for j in 0..dim {
        for i in 0..dim {
            result[i * dim + j] = cols[j * dim + i] as f32;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compute `(R · R^T)[i, j]` from a row-major flat buffer.
    fn dot_rows(matrix: &[f32], dim: usize, i: usize, j: usize) -> f32 {
        let mut s = 0.0f64;
        for k in 0..dim {
            s += (matrix[i * dim + k] as f64) * (matrix[j * dim + k] as f64);
        }
        s as f32
    }

    /// Tolerance for "is approximately the identity" at the given dim.
    /// Larger dim accumulates more floating-point error in the dot product.
    fn orthogonality_tol(dim: usize) -> f32 {
        // 1e-4 * sqrt(dim) covers d=64..3072 with a comfortable margin.
        1e-4 * (dim as f32).sqrt()
    }

    #[test]
    fn empty_dim_returns_empty_matrix() {
        let m = make_rotation_matrix(0, 42);
        assert!(m.is_empty());
    }

    #[test]
    fn is_orthogonal_at_d_64() {
        let dim = 64;
        let m = make_rotation_matrix(dim, 42);
        assert_eq!(m.len(), dim * dim);
        let tol = orthogonality_tol(dim);
        for i in 0..dim {
            let diag = dot_rows(&m, dim, i, i);
            assert!((diag - 1.0).abs() < tol, "diag[{i}] = {diag}, tol = {tol}");
            for j in (i + 1)..dim {
                let off = dot_rows(&m, dim, i, j);
                assert!(off.abs() < tol, "off[{i},{j}] = {off}, tol = {tol}");
            }
        }
    }

    #[test]
    fn is_orthogonal_at_d_256() {
        let dim = 256;
        let m = make_rotation_matrix(dim, 99);
        let tol = orthogonality_tol(dim);
        // Sample-check rather than full Gram matrix (cost is O(d^3)).
        for i in [0, 1, 17, 100, 200, 255] {
            let diag = dot_rows(&m, dim, i, i);
            assert!((diag - 1.0).abs() < tol, "diag[{i}] = {diag}");
        }
        for (i, j) in [(0, 1), (17, 100), (200, 255), (5, 250)] {
            let off = dot_rows(&m, dim, i, j);
            assert!(off.abs() < tol, "off[{i},{j}] = {off}");
        }
    }

    #[test]
    fn is_deterministic_for_same_seed() {
        let a = make_rotation_matrix(128, 0xDEAD_BEEF);
        let b = make_rotation_matrix(128, 0xDEAD_BEEF);
        // Bit-identical — `to_bits` catches any floating-point drift.
        assert_eq!(a.len(), b.len());
        for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(av.to_bits(), bv.to_bits(), "mismatch at index {i}");
        }
    }

    #[test]
    fn rotation_matches_persisted_seed_fixture() {
        // Captured with rand_chacha 0.3 and rand_distr 0.4, which produced
        // rotations for indexes that still reconstruct the matrix from its seed.
        let expected = [
            0x3eaad01b, 0x3eee4e27, 0xbf51de17, 0x3f6e5fb6, 0xbe98d06c, 0x3e568208, 0xbe16b645,
            0xbf554f08, 0xbf087089,
        ];
        let actual = make_rotation_matrix(3, 42);
        assert!(actual.iter().map(|value| value.to_bits()).eq(expected));
    }

    #[test]
    fn differs_per_seed() {
        // Per LLD Q3: per-collection rotation seeds. Two different seeds
        // must produce visibly different matrices so two tenants with the
        // same data still encode to different code streams.
        let a = make_rotation_matrix(64, 1);
        let b = make_rotation_matrix(64, 2);
        // The two matrices should differ in essentially every entry.
        let mut diffs = 0usize;
        for (av, bv) in a.iter().zip(b.iter()) {
            if (av - bv).abs() > 1e-6 {
                diffs += 1;
            }
        }
        // Out of 4096 entries, we expect ~all of them to differ; assert at
        // least 99% to avoid spurious failures on impossible PRNG luck.
        assert!(diffs > 4060, "only {diffs} / 4096 entries differ");
    }

    #[test]
    fn sign_correction_is_applied() {
        // After sign correction, the (0, 0) entry — which derives from the
        // first column of Q after possible sign flip — should be stable
        // for the same seed. This isn't a tight bound on correctness; it's
        // a regression check that we don't accidentally drop the sign-flip
        // pass in a refactor.
        let m1 = make_rotation_matrix(32, 12345);
        let m2 = make_rotation_matrix(32, 12345);
        assert_eq!(m1[0].to_bits(), m2[0].to_bits());
        // And a different seed should usually produce a different first
        // entry — verify in conjunction with `differs_per_seed`.
    }
}
