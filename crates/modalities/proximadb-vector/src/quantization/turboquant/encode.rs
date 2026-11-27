//! Encode batch of vectors to TurboQuant codes.
//!
//! Pipeline (P3 scope — adds TQ+ calibration + RaBitQ-style length-renorm
//! scalar; P2 identity-only path retained via `calibration = None`):
//!
//! 1. Normalize: strip `||v||`, store as a single f64.
//! 2. Rotate: `u_rot = R · u` where `R` is the per-collection rotation
//!    matrix.
//! 3. (Optional) calibrate: `u_calib[d] = (u_rot[d] + shift[d]) * scale_tq[d]`.
//! 4. Quantize: code[d] is the number of codebook boundaries strictly
//!    less than `u_calib[d]` (or `u_rot[d]` in identity mode).
//! 5. Per-vector RaBitQ-style length-renorm scalar:
//!
//!        scale_i = ||v_i|| / <u_rot,i, x_hat_orig,i>
//!
//!    where
//!
//!        x_hat_orig[d] = centroids[code[d]] / scale_tq[d] - shift[d]
//!
//!    (At identity, `x_hat_orig[d] = centroids[code[d]]`.) Storing
//!    `scale_i` per vector turns the search-path inner-product estimator
//!    from downward-biased to unbiased at zero scoring cost (RaBitQ,
//!    SIGMOD 2024). Recall gain is largest at 2-bit. Mirrors LLD §"Locked
//!    Type Signatures" `TurboQuantVectorData.scale` semantics.
//!
//! 6. Bit-pack 2- or 4-bit codes into a flat byte buffer, contiguous per
//!    vector. Bit layout matches LLD §3:
//!    - 2-bit: 4 codes per byte, big-endian within the byte.
//!      `byte = (c0 << 6) | (c1 << 4) | (c2 << 2) | c3`.
//!    - 4-bit: 2 codes per byte, big-endian within the byte.
//!      `byte = (c0 << 4) | c1`.
//!
//! Output is `EncodedBatch`: bit-packed `codes`, per-vector RaBitQ
//! `scales`, and shape fields. The `scales` field carries the
//! length-renorm scalar even in identity-calibration mode — that path is
//! just `scale_tq[d] = 1, shift[d] = 0`, so the math collapses cleanly.

use super::{
    Calibration, TurboQuantError, calibration, check_bit_width, check_dim, first_invalid_coord,
};

/// Output of one `encode_batch` call.
///
/// All fields are owned; the struct is meant to be moved into a
/// `TurboQuantVectorData` (root crate) or persisted to a `.tq` file. The
/// wire format itself is locked in LLD §3.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedBatch {
    /// Bit-packed codes, contiguous per vector. Length
    /// `n_vectors * ceil(dim * bit_width / 8)`.
    pub codes: Vec<u8>,
    /// Per-vector RaBitQ-style length-renormalization scalar:
    /// `scale = ||v|| / <u_rot, x_hat_orig>`. Applied at the SIMD kernel's
    /// final multiplication site (P4) to recover an unbiased
    /// inner-product estimator.
    pub scales: Vec<f32>,
    pub dim: usize,
    pub bit_width: u8,
    pub n_vectors: usize,
}

/// Encode a batch of vectors.
///
/// `vectors.len()` must equal `n_vectors * dim`. `dim` must already be
/// validated against the rotation matrix's dim. `rotation` is the
/// `dim x dim` matrix returned by [`super::rotation::make_rotation_matrix`]
/// in row-major layout. `boundaries` / `centroids` come from
/// [`super::codebook::codebook`]. `calibration`, when `Some`, must have
/// `dim` entries in both `shift` and `scale_tq`; pass `None` for identity
/// (no TQ+).
///
/// # Errors
///
/// - `DimNotMultipleOf8` — `dim` is 0 or not a multiple of 8.
/// - `BitWidthOutOfRange` — bit_width is not 2 or 4 in P1+P2+P3.
/// - `VectorBufferNotMultipleOfDim` — `vectors.len()` % `dim != 0`.
/// - `InvalidInputValue` — any coordinate is NaN, Inf, or `|x| >= 1e16`.
pub fn encode_batch(
    vectors: &[f32],
    dim: usize,
    bit_width: u8,
    rotation: &[f32],
    boundaries: &[f32],
    centroids: &[f32],
    calibration: Option<&Calibration>,
) -> Result<EncodedBatch, TurboQuantError> {
    check_dim(dim)?;
    check_bit_width(bit_width)?;
    if !vectors.is_empty() && vectors.len() % dim != 0 {
        return Err(TurboQuantError::VectorBufferNotMultipleOfDim {
            vectors_len: vectors.len(),
            dim,
        });
    }
    if let Some((vi, ci, val)) = first_invalid_coord(vectors, dim) {
        return Err(TurboQuantError::InvalidInputValue {
            vector_index: vi,
            coord_index: ci,
            value: val,
        });
    }
    debug_assert_eq!(rotation.len(), dim * dim);
    debug_assert_eq!(boundaries.len(), (1 << bit_width) - 1);
    debug_assert_eq!(centroids.len(), 1 << bit_width);
    if let Some(c) = calibration {
        debug_assert_eq!(c.dim(), dim, "calibration dim mismatch");
    }

    let n_vectors = vectors.len() / dim;
    let bits_per_vec = dim * bit_width as usize;
    let bytes_per_vec = bits_per_vec.div_ceil(8);
    let mut codes = vec![0u8; n_vectors * bytes_per_vec];
    let mut scales = vec![0.0f32; n_vectors];

    // Per-vector scratch buffers reused across the loop.
    let mut u_rot = vec![0.0f32; dim];
    let mut codes_one = vec![0u8; dim];

    for v_idx in 0..n_vectors {
        let row = &vectors[v_idx * dim..(v_idx + 1) * dim];

        // Step 1: norm + unit direction in f64 for numerical stability.
        let mut sumsq = 0.0f64;
        for &x in row {
            sumsq += (x as f64) * (x as f64);
        }
        let norm_f64 = sumsq.sqrt();
        let inv_norm = if norm_f64 > 1e-10 {
            1.0 / norm_f64
        } else {
            0.0
        };

        // Step 2: rotated unit vector. R stored row-major; `u_rot[i]
        // = sum_j R[i, j] * u[j]`.
        for i in 0..dim {
            let r_row = &rotation[i * dim..(i + 1) * dim];
            let mut acc = 0.0f64;
            for j in 0..dim {
                acc += (r_row[j] as f64) * (row[j] as f64) * inv_norm;
            }
            u_rot[i] = acc as f32;
        }

        // Step 3 + 4: quantize each coordinate. With calibration, the
        // value passed to the boundary scan is `(u_rot + shift) * scale_tq`;
        // without, the rotated value itself.
        match calibration {
            Some(cal) => {
                for d in 0..dim {
                    let u_calib =
                        calibration::apply_at_encode(u_rot[d], cal.shift[d], cal.scale_tq[d]);
                    codes_one[d] = quantize_one(u_calib, boundaries);
                }
            }
            None => {
                for d in 0..dim {
                    codes_one[d] = quantize_one(u_rot[d], boundaries);
                }
            }
        }

        // Step 5: per-vector RaBitQ-style length-renorm scalar.
        //
        // x_hat_orig[d] = centroids[code[d]] / scale_tq[d] - shift[d]
        //              (= centroids[code[d]] when calibration is None)
        // inner = sum_d u_rot[d] * x_hat_orig[d]
        // scale = ||v|| / inner
        //
        // Guard against the rare case where the centroid reconstruction
        // is exactly orthogonal to u_rot (or input was the zero vector,
        // making u_rot identically zero). In both cases we store the
        // bare norm — search will treat the vector as a degenerate
        // candidate with the same magnitude estimate as the identity
        // path.
        let mut inner = 0.0f64;
        match calibration {
            Some(cal) => {
                for d in 0..dim {
                    let x_hat_orig = calibration::centroid_in_original_space(
                        centroids[codes_one[d] as usize],
                        cal.shift[d],
                        cal.scale_tq[d],
                    );
                    inner += (u_rot[d] as f64) * (x_hat_orig as f64);
                }
            }
            None => {
                for d in 0..dim {
                    let x_hat = centroids[codes_one[d] as usize];
                    inner += (u_rot[d] as f64) * (x_hat as f64);
                }
            }
        }
        scales[v_idx] = if inner.abs() > 1e-10 {
            (norm_f64 / inner) as f32
        } else {
            norm_f64 as f32
        };

        // Step 6: bit-pack codes for this vector into the output buffer.
        let dst = &mut codes[v_idx * bytes_per_vec..(v_idx + 1) * bytes_per_vec];
        match bit_width {
            2 => pack_2bit_from_codes(&codes_one, dst),
            4 => pack_4bit_from_codes(&codes_one, dst),
            other => return Err(TurboQuantError::BitWidthOutOfRange(other)),
        }
    }

    Ok(EncodedBatch {
        codes,
        scales,
        dim,
        bit_width,
        n_vectors,
    })
}

/// Quantize one coordinate against the codebook boundaries. The code is
/// the count of boundaries strictly less than `v` — equivalently, the
/// index of the cell containing `v`.
#[inline(always)]
fn quantize_one(v: f32, boundaries: &[f32]) -> u8 {
    let mut code = 0u8;
    for &b in boundaries {
        if v > b {
            code += 1;
        }
    }
    code
}

/// Pack 4 codes per byte (2-bit), big-endian within byte.
fn pack_2bit_from_codes(codes: &[u8], dst: &mut [u8]) {
    let mut i = 0;
    while i < codes.len() {
        let byte = (codes[i] << 6) | (codes[i + 1] << 4) | (codes[i + 2] << 2) | codes[i + 3];
        dst[i / 4] = byte;
        i += 4;
    }
}

/// Pack 2 codes per byte (4-bit), big-endian within byte.
fn pack_4bit_from_codes(codes: &[u8], dst: &mut [u8]) {
    let mut i = 0;
    while i < codes.len() {
        let byte = (codes[i] << 4) | codes[i + 1];
        dst[i / 2] = byte;
        i += 2;
    }
}

#[cfg(test)]
mod tests {
    use super::super::{codebook::codebook, rotation::make_rotation_matrix};
    use super::*;
    use chacha20::ChaCha8Rng;
    use rand::{RngExt, SeedableRng};
    use rand_distr::StandardNormal;

    fn make_rotation(dim: usize, seed: u64) -> Vec<f32> {
        make_rotation_matrix(dim, seed)
    }

    #[test]
    fn rejects_misaligned_dim() {
        let r = make_rotation(8, 1);
        let (bnd, c) = codebook(2, 8);
        let err = encode_batch(&[], 7, 2, &r, &bnd, &c, None).unwrap_err();
        assert!(matches!(err, TurboQuantError::DimNotMultipleOf8(7)));
    }

    #[test]
    fn rejects_unsupported_bit_width() {
        let r = make_rotation(8, 1);
        let (bnd, c) = codebook(2, 8);
        let err = encode_batch(&[], 8, 3, &r, &bnd, &c, None).unwrap_err();
        assert!(matches!(err, TurboQuantError::BitWidthOutOfRange(3)));
    }

    #[test]
    fn rejects_misaligned_buffer_length() {
        let r = make_rotation(8, 1);
        let (bnd, c) = codebook(2, 8);
        let v = vec![0.5f32; 9];
        let err = encode_batch(&v, 8, 2, &r, &bnd, &c, None).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::VectorBufferNotMultipleOfDim {
                vectors_len: 9,
                dim: 8
            }
        ));
    }

    #[test]
    fn rejects_nan_input() {
        let r = make_rotation(8, 1);
        let (bnd, c) = codebook(2, 8);
        let mut v = vec![0.1f32; 16];
        v[10] = f32::NAN;
        let err = encode_batch(&v, 8, 2, &r, &bnd, &c, None).unwrap_err();
        assert!(matches!(err, TurboQuantError::InvalidInputValue { .. }));
    }

    #[test]
    fn empty_batch_returns_empty_codes() {
        let r = make_rotation(8, 1);
        let (bnd, c) = codebook(2, 8);
        let out = encode_batch(&[], 8, 2, &r, &bnd, &c, None).unwrap();
        assert_eq!(out.n_vectors, 0);
        assert!(out.codes.is_empty());
        assert!(out.scales.is_empty());
        assert_eq!(out.dim, 8);
        assert_eq!(out.bit_width, 2);
    }

    #[test]
    fn encodes_one_2bit_vector_with_correct_shape() {
        let dim = 16;
        let r = make_rotation(dim, 7);
        let (bnd, c) = codebook(2, dim);
        let v = vec![0.5f32; dim];
        let out = encode_batch(&v, dim, 2, &r, &bnd, &c, None).unwrap();
        assert_eq!(out.n_vectors, 1);
        assert_eq!(out.codes.len(), dim * 2 / 8);
        assert_eq!(out.scales.len(), 1);
        assert!(out.scales[0].is_finite());
    }

    #[test]
    fn encodes_one_4bit_vector_with_correct_shape() {
        let dim = 16;
        let r = make_rotation(dim, 7);
        let (bnd, c) = codebook(4, dim);
        let v = vec![0.25f32; dim];
        let out = encode_batch(&v, dim, 4, &r, &bnd, &c, None).unwrap();
        assert_eq!(out.n_vectors, 1);
        assert_eq!(out.codes.len(), 8);
        assert_eq!(out.scales.len(), 1);
        assert!(out.scales[0].is_finite());
    }

    #[test]
    fn encodes_two_vectors_independently() {
        let dim = 8;
        let r = make_rotation(dim, 11);
        let (bnd, c) = codebook(2, dim);
        let mut v = vec![0.0f32; 2 * dim];
        for i in 0..dim {
            v[i] = (i as f32) * 0.1;
            v[dim + i] = -((i as f32) * 0.1);
        }
        let out = encode_batch(&v, dim, 2, &r, &bnd, &c, None).unwrap();
        assert_eq!(out.n_vectors, 2);
        assert_eq!(out.codes.len(), 2 * (dim * 2 / 8));
        assert_eq!(out.scales.len(), 2);
        // Both vectors have the same magnitude by symmetry, so their
        // length-renorm scales should be comparable in magnitude (within
        // a few percent — quantization noise differs).
        let r0 = out.scales[0].abs();
        let r1 = out.scales[1].abs();
        assert!(
            ((r0 - r1).abs() / r0.max(r1)) < 0.1,
            "asymmetric scales: {r0} vs {r1}",
        );
    }

    #[test]
    fn zero_vector_handles_gracefully() {
        // ||v|| = 0 → norm_f64 = 0 → inv_norm = 0 → u_rot = 0 → inner = 0.
        // The fallback path stores `norm_f64 = 0`, and all codes round to
        // the centroid closest to 0. No panics, no NaN.
        let dim = 8;
        let r = make_rotation(dim, 1);
        let (bnd, c) = codebook(4, dim);
        let v = vec![0.0f32; dim];
        let out = encode_batch(&v, dim, 4, &r, &bnd, &c, None).unwrap();
        assert_eq!(out.n_vectors, 1);
        assert_eq!(out.scales[0], 0.0);
        let first = out.codes[0];
        assert!(out.codes.iter().all(|&b| b == first));
    }

    #[test]
    fn identity_calibration_matches_none_path() {
        let dim = 16;
        let r = make_rotation(dim, 5);
        let (bnd, c) = codebook(4, dim);
        let v: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1 - 0.7).collect();

        let identity = Calibration {
            shift: vec![0.0f32; dim],
            scale_tq: vec![1.0f32; dim],
        };
        let a = encode_batch(&v, dim, 4, &r, &bnd, &c, None).unwrap();
        let b = encode_batch(&v, dim, 4, &r, &bnd, &c, Some(&identity)).unwrap();
        assert_eq!(
            a.codes, b.codes,
            "identity calibration must match no-calibration"
        );
        // Scales should also match within rounding — the multiplication
        // by 1.0 and add of 0.0 may differ in the last bit. Compare via
        // a tight epsilon.
        for (x, y) in a.scales.iter().zip(b.scales.iter()) {
            assert!((x - y).abs() < 1e-5, "scale drift: {x} vs {y}");
        }
    }

    #[test]
    fn encoding_is_deterministic() {
        let dim = 16;
        let r = make_rotation(dim, 99);
        let (bnd, c) = codebook(2, dim);
        let v: Vec<f32> = (0..dim).map(|i| (i as f32) - 8.0).collect();
        let a = encode_batch(&v, dim, 2, &r, &bnd, &c, None).unwrap();
        let b = encode_batch(&v, dim, 2, &r, &bnd, &c, None).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn different_seeds_produce_different_codes_for_same_input() {
        let dim = 16;
        let r1 = make_rotation(dim, 1);
        let r2 = make_rotation(dim, 2);
        let (bnd, c) = codebook(2, dim);
        let v: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1 - 0.8).collect();
        let a = encode_batch(&v, dim, 2, &r1, &bnd, &c, None).unwrap();
        let b = encode_batch(&v, dim, 2, &r2, &bnd, &c, None).unwrap();
        assert_ne!(a.codes, b.codes);
    }

    #[test]
    fn rabitq_scale_is_finite_and_positive_for_random_input() {
        // Generate 200 random unit-vector samples at d=64; every encode
        // must produce a finite, positive RaBitQ scale (corner cases —
        // exact zero, exact orthogonality — are extremely improbable on
        // random Gaussian-rotated inputs).
        let dim = 64;
        let r = make_rotation(dim, 21);
        let (bnd, c) = codebook(4, dim);
        let mut rng = ChaCha8Rng::seed_from_u64(31);
        let mut v = vec![0.0f32; 200 * dim];
        for i in 0..200 {
            let mut sumsq = 0.0f64;
            for d in 0..dim {
                let x: f64 = rng.sample(StandardNormal);
                v[i * dim + d] = x as f32;
                sumsq += x * x;
            }
            let inv = (1.0 / sumsq.sqrt()) as f32;
            for d in 0..dim {
                v[i * dim + d] *= inv;
            }
        }
        let out = encode_batch(&v, dim, 4, &r, &bnd, &c, None).unwrap();
        for (i, &s) in out.scales.iter().enumerate() {
            assert!(s.is_finite(), "scale[{i}] not finite: {s}");
            assert!(s > 0.0, "scale[{i}] not positive: {s}");
        }
    }

    #[test]
    fn tq_plus_calibration_changes_codes_when_input_is_biased() {
        // Construct a 1000-vector batch in the *unrotated* space whose
        // first coordinate is heavily biased (mean 0.3 instead of 0).
        // After rotation + TQ+ fit, the codes should differ from the
        // identity-calibration path.
        let dim = 64;
        let r = make_rotation(dim, 13);
        let (bnd, c) = codebook(2, dim);

        let mut rng = ChaCha8Rng::seed_from_u64(47);
        let mut v = vec![0.0f32; 1000 * dim];
        for i in 0..1000 {
            let mut sumsq = 0.0f64;
            for d in 0..dim {
                let mut x: f64 = rng.sample(StandardNormal);
                if d == 0 {
                    x += 1.0; // bias the first coord
                }
                v[i * dim + d] = x as f32;
                sumsq += x * x;
            }
            let inv = (1.0 / sumsq.sqrt()) as f32;
            for d in 0..dim {
                v[i * dim + d] *= inv;
            }
        }

        // Compute the rotated batch in the same way encode_batch does so
        // we can fit calibration off it.
        let mut rotated = vec![0.0f32; 1000 * dim];
        for i in 0..1000 {
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
        let cal = super::super::fit_calibration(&rotated, 1000, dim).unwrap();

        let identity = encode_batch(&v, dim, 2, &r, &bnd, &c, None).unwrap();
        let calibrated = encode_batch(&v, dim, 2, &r, &bnd, &c, Some(&cal)).unwrap();
        assert_ne!(
            identity.codes, calibrated.codes,
            "biased input should produce different codes under TQ+ calibration",
        );
    }
}
