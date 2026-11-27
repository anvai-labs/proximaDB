//! TQ+ per-coordinate calibration (frozen-after-first-batch).
//!
//! After multiplication by a random orthogonal matrix, each coordinate of
//! a unit vector on the (d-1)-sphere *asymptotically* follows Beta((d-1)/2,
//! (d-1)/2) shifted to `[-1, 1]`. At finite d (especially small d, like
//! word-vector embeddings at d=200) individual coordinates drift away from
//! the canonical Beta marginal. TQ+ (arXiv:2504.19874 §"Online"; companion
//! to the base TurboQuant quantizer) closes that gap with two scalars per
//! coordinate — a `shift` and a `scale` — that map each coordinate's
//! empirical 5/95% quantiles onto the canonical Beta marginal's 5/95%
//! quantiles. The Lloyd-Max codebook (`codebook.rs`) is then quantizing
//! against the *target* distribution it was designed for.
//!
//! ## Frozen-after-first-batch invariant
//!
//! Per `TURBOQUANT_LLD_2026_05_30.adoc` §"Decision Index" Q7: TQ+ is fit
//! ONCE from the empirical quantiles of the first qualifying batch
//! (`n >= TQPLUS_MIN_SAMPLES` = 1000) and then frozen for the collection's
//! lifetime. Subsequent `add()` batches re-use the same `Calibration`.
//! Re-fitting only happens on an epoch bump (P8 wires that to the WAL
//! repair source).
//!
//! [`fit_calibration`] is intentionally **pure data** — it returns the
//! fitted struct (or `None` if the batch is too small). Storing it,
//! refusing re-fits, and threading it through the encode pipeline are
//! the caller's responsibility. The encode entry point in P3 accepts
//! `Option<&Calibration>` and propagates it as-is.
//!
//! ## Coordinate transform
//!
//! At encode:
//!   `u_calibrated[d] = (u_rot[d] + shift[d]) * scale_tq[d]`
//!
//! At search (P4):
//!   `q_calibrated[d] = q_rot[d] / scale_tq[d]`
//!   `bias_correction = -<q_rot, shift>`
//!
//! With these transforms applied symmetrically, the kernel reconstructs
//! `<q_rot, x_hat_orig>` from the calibrated-space LUT — no kernel change
//! is required for TQ+.

use serde::{Deserialize, Serialize};
use statrs::distribution::{Beta, ContinuousCDF};
use std::cmp::Ordering;

/// Below this many input vectors, per-coordinate quantile estimates are
/// too noisy to fit reliably — calibration silently falls back to
/// identity. Matches the empirical floor from the TurboQuant paper
/// (paper §"Online"; mirrored in the reference implementation).
pub const TQPLUS_MIN_SAMPLES: usize = 1000;

/// Empirical quantile pair used to fit each coordinate. The 5/95 split
/// is the LLD-locked default; changing it would require ADR amendment
/// because it influences code-book mis-fit risk on anisotropic data.
pub const TQPLUS_P_LO: f64 = 0.05;
pub const TQPLUS_P_HI: f64 = 0.95;

/// Per-coordinate TQ+ calibration vectors. Both vectors have length `dim`
/// when the calibration has been fitted; both are empty when the
/// collection runs in identity mode.
///
/// Serializable so it can be persisted alongside codes in the `.tq` file
/// (LLD §3 wire format). The byte layout there places `shift` first,
/// then `scale_tq`, both as little-endian f32.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    pub shift: Vec<f32>,
    pub scale_tq: Vec<f32>,
}

impl Calibration {
    pub fn dim(&self) -> usize {
        self.shift.len()
    }
}

/// Fit a per-coordinate `(shift, scale_tq)` from a batch of rotated
/// unit-vector coordinates.
///
/// `rotated` is the row-major batch produced by the rotation step:
/// `rotated[i * dim + d]` is the d-th coordinate of the i-th vector after
/// multiplication by the per-collection rotation matrix. `n` is the
/// number of vectors; `dim` is the per-vector length. Caller validates
/// `rotated.len() == n * dim` and that `dim` matches the index's
/// committed dim.
///
/// Returns:
/// - `None` when `n < TQPLUS_MIN_SAMPLES`. Callers using TQ+ should
///   either retry after more data is ingested or fall back to identity
///   calibration (LLD Q7).
/// - `Some(Calibration)` with `dim` entries per vector otherwise.
///
/// Degenerate coordinates (constant or near-constant) fall through to
/// identity `(0, 1)` for that coordinate — `scale_tq[d]` is never zero,
/// which keeps the search-time inverse `q_rot[d] / scale_tq[d]` finite.
pub fn fit_calibration(rotated: &[f32], n: usize, dim: usize) -> Option<Calibration> {
    if n < TQPLUS_MIN_SAMPLES {
        return None;
    }
    debug_assert_eq!(rotated.len(), n * dim);

    let a = (dim as f64 - 1.0) / 2.0;
    let beta = Beta::new(a, a).expect("Beta((d-1)/2, (d-1)/2) is valid for any d >= 2");
    // Beta is on [0, 1]; canonical marginal is shifted to [-1, 1].
    let qc_lo = (2.0 * beta.inverse_cdf(TQPLUS_P_LO) - 1.0) as f32;
    let qc_hi = (2.0 * beta.inverse_cdf(TQPLUS_P_HI) - 1.0) as f32;
    let qc_span = qc_hi - qc_lo;

    let lo_idx = ((n as f64) * TQPLUS_P_LO) as usize;
    let hi_idx = (((n as f64) * TQPLUS_P_HI) as usize).min(n - 1);

    let mut shift = vec![0.0f32; dim];
    let mut scale_tq = vec![1.0f32; dim];

    // One sort per coordinate — `O(d * n log n)`. At d=1536, n=1k this is
    // ~15 ms single-threaded; well under encoding cost. Could be
    // parallelized over coordinates with Rayon when it lands as a workspace
    // dep on this crate; deferred to keep P3 dependency-light.
    let mut coord_buf: Vec<f32> = vec![0.0f32; n];
    for d in 0..dim {
        for i in 0..n {
            coord_buf[i] = rotated[i * dim + d];
        }
        coord_buf.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let qe_lo = coord_buf[lo_idx];
        let qe_hi = coord_buf[hi_idx];
        let qe_span = qe_hi - qe_lo;
        if qe_span > 1e-6 {
            let s = qc_span / qe_span;
            // shift maps qe_lo onto qc_lo:
            //   qc_lo = (qe_lo + shift) * scale_tq
            //   => shift = qc_lo / scale_tq - qe_lo
            scale_tq[d] = s;
            shift[d] = qc_lo / s - qe_lo;
        }
        // else: leave (0, 1) — degenerate coord stays in original space.
    }

    Some(Calibration { shift, scale_tq })
}

/// Apply the calibration to one rotated unit-vector coordinate. Lifts the
/// transform into one place so encode-side and search-side code paths
/// agree on the convention.
///
/// Returns `(u_rot + shift) * scale_tq`. With identity calibration
/// (`shift = 0`, `scale_tq = 1`), this is a no-op.
#[inline(always)]
pub fn apply_at_encode(u_rot: f32, shift_d: f32, scale_tq_d: f32) -> f32 {
    (u_rot + shift_d) * scale_tq_d
}

/// Recover the rotated-space (uncalibrated) centroid value from a code's
/// calibrated centroid. Used in the encode pipeline for the RaBitQ-style
/// length-renorm inner product:
///
///   `x_hat_orig[d] = centroid_calib / scale_tq[d] - shift[d]`
///
/// At identity, this collapses to `centroid_calib`.
#[inline(always)]
pub fn centroid_in_original_space(centroid_calib: f32, shift_d: f32, scale_tq_d: f32) -> f32 {
    centroid_calib / scale_tq_d - shift_d
}

#[cfg(test)]
mod tests {
    use super::*;
    use chacha20::ChaCha8Rng;
    use rand::{RngExt, SeedableRng};
    use rand_distr::StandardNormal;

    /// Generate `n` rotated unit-vector samples in `dim` dimensions by
    /// drawing from `N(0, I)` and normalising — this is statistically
    /// equivalent to uniform on the (d-1)-sphere, which is the
    /// distribution the rotated vectors approximate.
    fn synth_rotated(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut out = vec![0.0f32; n * dim];
        for i in 0..n {
            let mut sumsq = 0.0f64;
            for d in 0..dim {
                let x: f64 = rng.sample(StandardNormal);
                out[i * dim + d] = x as f32;
                sumsq += x * x;
            }
            let inv = if sumsq > 1e-30 {
                (1.0 / sumsq.sqrt()) as f32
            } else {
                0.0
            };
            for d in 0..dim {
                out[i * dim + d] *= inv;
            }
        }
        out
    }

    #[test]
    fn returns_none_below_min_samples() {
        let rotated = synth_rotated(100, 64, 1);
        assert!(fit_calibration(&rotated, 100, 64).is_none());
        let rotated = synth_rotated(999, 64, 1);
        assert!(fit_calibration(&rotated, 999, 64).is_none());
    }

    #[test]
    fn returns_some_at_min_samples() {
        let rotated = synth_rotated(1000, 64, 1);
        let cal = fit_calibration(&rotated, 1000, 64).unwrap();
        assert_eq!(cal.dim(), 64);
        assert_eq!(cal.shift.len(), 64);
        assert_eq!(cal.scale_tq.len(), 64);
    }

    #[test]
    fn scale_tq_is_finite_and_positive() {
        let rotated = synth_rotated(2000, 128, 42);
        let cal = fit_calibration(&rotated, 2000, 128).unwrap();
        for (d, &s) in cal.scale_tq.iter().enumerate() {
            assert!(s.is_finite(), "scale_tq[{d}] not finite: {s}");
            assert!(s > 0.0, "scale_tq[{d}] not positive: {s}");
        }
    }

    #[test]
    fn shift_is_finite() {
        let rotated = synth_rotated(2000, 128, 7);
        let cal = fit_calibration(&rotated, 2000, 128).unwrap();
        for (d, &s) in cal.shift.iter().enumerate() {
            assert!(s.is_finite(), "shift[{d}] not finite: {s}");
        }
    }

    #[test]
    fn deterministic_for_same_input() {
        let rotated = synth_rotated(1500, 64, 99);
        let a = fit_calibration(&rotated, 1500, 64).unwrap();
        let b = fit_calibration(&rotated, 1500, 64).unwrap();
        for (x, y) in a.shift.iter().zip(b.shift.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        for (x, y) in a.scale_tq.iter().zip(b.scale_tq.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    #[test]
    fn degenerate_constant_coord_falls_through_to_identity() {
        // Coordinate 5 is constant across all 1k vectors; quantile span
        // is zero, so the fit should leave (0, 1) for that coord and
        // refuse to divide by zero.
        let mut rotated = synth_rotated(1500, 32, 1);
        for i in 0..1500 {
            rotated[i * 32 + 5] = 0.123;
        }
        let cal = fit_calibration(&rotated, 1500, 32).unwrap();
        assert_eq!(cal.shift[5], 0.0);
        assert_eq!(cal.scale_tq[5], 1.0);
        // Other coords were fit and should usually deviate from identity.
        let any_non_identity = cal
            .shift
            .iter()
            .enumerate()
            .any(|(d, &s)| d != 5 && s != 0.0);
        assert!(any_non_identity, "no non-identity coords found");
    }

    #[test]
    fn apply_and_centroid_roundtrip_at_identity() {
        // Identity calibration → encode/decode helpers are no-ops.
        let u = 0.7321f32;
        let calibrated = apply_at_encode(u, 0.0, 1.0);
        assert_eq!(u, calibrated);
        let recovered = centroid_in_original_space(calibrated, 0.0, 1.0);
        assert_eq!(u, recovered);
    }

    #[test]
    fn apply_and_centroid_roundtrip_at_tqplus() {
        // Round-trip: take a value u_rot, apply at encode, then assume
        // centroid_calib = u_calib (no quantization), recover x_hat_orig.
        // x_hat_orig should equal u_rot.
        let u_rot = -0.42f32;
        let shift = 0.1f32;
        let scale_tq = 1.7f32;
        let u_calib = apply_at_encode(u_rot, shift, scale_tq);
        // Pretend the centroid landed exactly on u_calib (perfect quantizer).
        let recovered = centroid_in_original_space(u_calib, shift, scale_tq);
        assert!(
            (u_rot - recovered).abs() < 1e-5,
            "round-trip drift: {u_rot} vs {recovered}",
        );
    }

    #[test]
    fn min_samples_constant_is_locked() {
        // Tripwire: changing TQPLUS_MIN_SAMPLES is a wire-contract change
        // (alters when calibration fits) and requires LLD amendment.
        assert_eq!(TQPLUS_MIN_SAMPLES, 1000);
        assert_eq!(TQPLUS_P_LO, 0.05);
        assert_eq!(TQPLUS_P_HI, 0.95);
    }
}
