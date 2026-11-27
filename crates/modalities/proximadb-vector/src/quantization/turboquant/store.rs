//! High-level TurboQuant store — ties rotation + codebook + calibration +
//! encoded state + kernel into one usable end-to-end API.
//!
//! Per `TURBOQUANT_LLD_2026_05_30.adoc` §"Concurrency Model": the rotation
//! matrix and Lloyd-Max codebook are lazy-init `OnceLock`s (deterministic
//! functions of construction parameters; cache once, share across
//! threads). Encoded state — codes, per-vector RaBitQ scales, the fitted
//! TQ+ calibration — lives behind a single `Mutex`. `search()` takes
//! `&self` and is concurrency-safe; multiple readers serialize through
//! the mutex but each reader runs the kernel from start to finish without
//! interleaving.
//!
//! ## Scope (P6.A)
//!
//! This is the **engine integration shim**, not the AXIS adapter
//! integration itself. It gives both:
//! - **P6 AXIS adapters** a stable construction + add + search surface
//!   they can wrap, when the adapter signature delta lands in a follow-up
//!   session.
//! - **P8 xCatalog + EXPLAIN wiring** an artifact whose state
//!   (`rotation_seed`, `calibration_mode`, `bit_width`, fitted-vs-not
//!   calibration) can be projected into `RoutedExecutionPlan` hints.
//!
//! ## Known suboptimal — addressed in P6 AXIS wiring
//!
//! [`search()`] currently materialises an [`EncodedBatch`] by cloning
//! the stored `codes` + `scales`. This is `O(n * bytes_per_vec)` per
//! search and dominates wall-clock at large `n`. The proper fix is
//! [`kernel::search`] taking the constituent slices directly; deferred
//! to P6 when AXIS adapters get their own memory ownership and can
//! supply the slices without copying.

use std::sync::{Mutex, OnceLock};

use proximadb_quantization_types::CalibrationMode;

use super::{
    Calibration, TQPLUS_MIN_SAMPLES, TurboQuantError, check_bit_width, check_dim,
    codebook::codebook,
    encode::{EncodedBatch, encode_batch},
    fit_calibration,
    io::{self, PersistedStore},
    kernel::{self, SearchHit},
    rotation::make_rotation_matrix,
};

/// Observability snapshot of a [`TurboQuantStore`]'s encoded state.
///
/// Returned by [`TurboQuantStore::stats`]. Designed for operator-facing
/// surfaces (admin endpoints, capacity-planning dashboards, the
/// EXPLAIN hint set in `TurboQuantExplainHints`) so production callers
/// don't need access to the store's private fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StoreStats {
    pub dim: usize,
    pub bit_width: u8,
    pub calibration_mode: CalibrationMode,
    pub rotation_seed: u64,
    pub n_vectors: usize,
    /// Bit-packed code bytes for one vector — `ceil(dim * bit_width / 8)`.
    pub bytes_per_vec_codes: usize,
    /// Total encoded code bytes across all vectors.
    pub codes_bytes: usize,
    /// Per-vector RaBitQ scales — `n_vectors * 4`.
    pub scales_bytes: usize,
    /// TQ+ calibration vectors when fit (`2 * dim * 4`); zero otherwise.
    pub calibration_bytes: usize,
    /// Total in-RAM encoded state. Excludes the rotation matrix and
    /// codebook caches because those are deterministic and shared
    /// across the collection's lifetime.
    pub total_bytes: usize,
    /// Compression ratio per vector vs an FP32 baseline. Includes the
    /// per-vector RaBitQ scale in the compressed-side cost. At
    /// `bit_width=2, dim=1536` this is ~15.83×; at `bit_width=4,
    /// dim=1536` it is ~7.96×.
    pub compression_ratio_vs_fp32: f32,
    /// `true` iff a TQ+ calibration has been fit.
    pub has_calibration: bool,
}

/// Owning end-to-end TurboQuant index. Construction is cheap; the heavy
/// caches (rotation matrix, Lloyd-Max codebook) initialise on first use.
///
/// `Debug` is implemented manually to avoid printing the (potentially
/// huge) rotation matrix and codes buffers — only the configuration
/// fields and counts are surfaced.
pub struct TurboQuantStore {
    dim: usize,
    bit_width: u8,
    calibration_mode: CalibrationMode,
    rotation_seed: u64,

    /// Deterministic from `(dim, rotation_seed)`. Lazy-init `OnceLock`
    /// so two parallel `search()` callers don't pay the cost twice.
    rotation: OnceLock<Vec<f32>>,
    /// Deterministic from `(bit_width, dim)`. Held as
    /// `(boundaries, centroids)`.
    codebook_cache: OnceLock<(Vec<f32>, Vec<f32>)>,

    /// Mutable state: encoded codes/scales appended on every `add()` plus
    /// the fitted TQ+ calibration. Single mutex keeps the invariant that
    /// the calibration matches the codes that were encoded against it.
    inner: Mutex<StoreInner>,
}

struct StoreInner {
    /// Flat bit-packed codes for all encoded vectors, contiguous per
    /// vector. Length = `n_vectors * ceil(dim * bit_width / 8)`.
    codes: Vec<u8>,
    /// Per-vector RaBitQ-style length-renormalization scales. Length =
    /// `n_vectors`.
    scales: Vec<f32>,
    /// Total encoded vector count.
    n_vectors: usize,
    /// `Some` after the first qualifying batch has fit it; `None` for
    /// identity mode or before the threshold is met. Frozen after fit
    /// per LLD §"Decision Index" Q7.
    calibration: Option<Calibration>,
}

impl std::fmt::Debug for TurboQuantStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (n_vectors, has_cal) = self
            .inner
            .lock()
            .map(|i| (i.n_vectors, i.calibration.is_some()))
            .unwrap_or((0, false));
        f.debug_struct("TurboQuantStore")
            .field("dim", &self.dim)
            .field("bit_width", &self.bit_width)
            .field("calibration_mode", &self.calibration_mode)
            .field("rotation_seed", &self.rotation_seed)
            .field("n_vectors", &n_vectors)
            .field("has_calibration", &has_cal)
            .finish()
    }
}

impl TurboQuantStore {
    /// Construct a new store. Validates `dim` and `bit_width` per LLD
    /// §"Algorithm Constants" — `dim` must be a positive multiple of 8;
    /// `bit_width` must be in `{2, 4}` (3-bit deferred per Q10).
    pub fn new(
        dim: usize,
        bit_width: u8,
        calibration_mode: CalibrationMode,
        rotation_seed: u64,
    ) -> Result<Self, TurboQuantError> {
        check_dim(dim)?;
        check_bit_width(bit_width)?;
        Ok(Self {
            dim,
            bit_width,
            calibration_mode,
            rotation_seed,
            rotation: OnceLock::new(),
            codebook_cache: OnceLock::new(),
            inner: Mutex::new(StoreInner {
                codes: Vec::new(),
                scales: Vec::new(),
                n_vectors: 0,
                calibration: None,
            }),
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn bit_width(&self) -> u8 {
        self.bit_width
    }
    pub fn calibration_mode(&self) -> CalibrationMode {
        self.calibration_mode
    }
    pub fn rotation_seed(&self) -> u64 {
        self.rotation_seed
    }

    /// Total number of encoded vectors. Takes the inner lock briefly.
    pub fn len(&self) -> usize {
        self.inner.lock().map(|i| i.n_vectors).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Compute an observability snapshot of the store's encoded state.
    /// Takes the inner lock briefly. Designed for operator endpoints
    /// (`/admin/turboquant/{collection_id}/stats`, capacity planning,
    /// compression-ratio dashboards) so production callers don't need to
    /// reach inside the private state.
    pub fn stats(&self) -> StoreStats {
        let (n_vectors, has_calibration) = self
            .inner
            .lock()
            .map(|i| (i.n_vectors, i.calibration.is_some()))
            .unwrap_or((0, false));

        let bytes_per_vec_codes = (self.dim * self.bit_width as usize).div_ceil(8);
        let codes_bytes = n_vectors * bytes_per_vec_codes;
        let scales_bytes = n_vectors * std::mem::size_of::<f32>();
        // TQ+ calibration is `2 * dim * f32` when fit; zero otherwise.
        // Mirrors LLD §3 wire-format body layout.
        let calibration_bytes = if has_calibration {
            2 * self.dim * std::mem::size_of::<f32>()
        } else {
            0
        };
        // Total in-RAM memory consumed by encoded state. Excludes the
        // OnceLock caches (rotation matrix `dim^2 * 4`, codebook
        // `2^bit_width * 4` floats) because those are deterministic
        // from `(dim, rotation_seed, bit_width)` — the same value
        // operators see in the catalog — and don't scale with `n`.
        let total_bytes = codes_bytes + scales_bytes + calibration_bytes;

        // FP32 baseline: `n * dim * 4`. Compression ratio is the
        // multiplier vs that baseline. At `n = 0` the ratio is the
        // theoretical per-vector ratio (so dashboards don't show NaN on
        // empty collections).
        let fp32_bytes_per_vec = self.dim * std::mem::size_of::<f32>();
        let compressed_per_vec = bytes_per_vec_codes + std::mem::size_of::<f32>();
        let compression_ratio_vs_fp32 = if compressed_per_vec == 0 {
            1.0
        } else {
            fp32_bytes_per_vec as f32 / compressed_per_vec as f32
        };

        StoreStats {
            dim: self.dim,
            bit_width: self.bit_width,
            calibration_mode: self.calibration_mode,
            rotation_seed: self.rotation_seed,
            n_vectors,
            bytes_per_vec_codes,
            codes_bytes,
            scales_bytes,
            calibration_bytes,
            total_bytes,
            compression_ratio_vs_fp32,
            has_calibration,
        }
    }

    /// Whether a TQ+ calibration has been fit yet. `false` for identity
    /// mode (always) and for TQ+ mode before the first ≥1000-vector
    /// batch. Useful for EXPLAIN — when `false` and the configured mode
    /// is `TqPlus`, the routing plan should surface
    /// `calibration_mode="identity"` (LLD Q7).
    pub fn has_calibration(&self) -> bool {
        self.inner
            .lock()
            .map(|i| i.calibration.is_some())
            .unwrap_or(false)
    }

    /// Read-only access to the rotation matrix cache. Initialises on
    /// first call.
    fn rotation(&self) -> &[f32] {
        self.rotation
            .get_or_init(|| make_rotation_matrix(self.dim, self.rotation_seed))
    }

    /// Read-only access to the boundaries + centroids cache.
    fn codebook(&self) -> (&[f32], &[f32]) {
        let (b, c) = self
            .codebook_cache
            .get_or_init(|| codebook(self.bit_width as usize, self.dim));
        (b.as_slice(), c.as_slice())
    }

    /// Add a batch of raw FP32 vectors. `vectors.len()` must equal
    /// `n_vectors * self.dim()`. The first qualifying batch in `TqPlus`
    /// mode fits calibration; subsequent batches reuse it (frozen-after-
    /// first-batch invariant per LLD Q7).
    ///
    /// Note: takes `&self` (not `&mut self`) so external callers can
    /// share an `Arc<TurboQuantStore>` without juggling write locks at
    /// their level. The mutex serialises concurrent `add()`s internally.
    pub fn add(&self, vectors: &[f32]) -> Result<(), TurboQuantError> {
        if vectors.is_empty() {
            return Ok(());
        }
        if vectors.len() % self.dim != 0 {
            return Err(TurboQuantError::VectorBufferNotMultipleOfDim {
                vectors_len: vectors.len(),
                dim: self.dim,
            });
        }
        let rotation = self.rotation();
        let (boundaries, centroids) = self.codebook();

        let mut inner = self
            .inner
            .lock()
            .expect("TurboQuantStore::inner mutex poisoned");

        // Fit calibration on the first qualifying batch in TqPlus mode.
        // Identity mode skips this. Once fit, the calibration is frozen.
        if matches!(self.calibration_mode, CalibrationMode::TqPlus)
            && inner.calibration.is_none()
            && inner.n_vectors == 0
        {
            let n = vectors.len() / self.dim;
            if n >= TQPLUS_MIN_SAMPLES {
                // Rotate the batch (same numerics as encode_batch) so we
                // can fit per-coord quantiles in calibrated space.
                let mut rotated = vec![0.0f32; n * self.dim];
                for i in 0..n {
                    let row = &vectors[i * self.dim..(i + 1) * self.dim];
                    let mut sumsq = 0.0f64;
                    for &x in row {
                        sumsq += (x as f64) * (x as f64);
                    }
                    let inv_norm = if sumsq > 1e-30 {
                        1.0 / sumsq.sqrt()
                    } else {
                        0.0
                    };
                    for k in 0..self.dim {
                        let r_row = &rotation[k * self.dim..(k + 1) * self.dim];
                        let mut acc = 0.0f64;
                        for j in 0..self.dim {
                            acc += (r_row[j] as f64) * (row[j] as f64) * inv_norm;
                        }
                        rotated[i * self.dim + k] = acc as f32;
                    }
                }
                inner.calibration = fit_calibration(&rotated, n, self.dim);
            }
            // If the first batch is too small, calibration stays `None`
            // and this batch — plus all future batches — encode against
            // identity. LLD Q7 silent-fallback semantics.
        }

        let batch = encode_batch(
            vectors,
            self.dim,
            self.bit_width,
            rotation,
            boundaries,
            centroids,
            inner.calibration.as_ref(),
        )?;

        inner.codes.extend_from_slice(&batch.codes);
        inner.scales.extend_from_slice(&batch.scales);
        inner.n_vectors += batch.n_vectors;

        Ok(())
    }

    /// Run a top-`k` search against the store. `mask`, when `Some`, is
    /// the packed-bitmap allowlist consumed by the kernel's block-skip
    /// path (per LLD §"In-Kernel Allowlist").
    ///
    /// Takes `&self` and is safe under concurrent callers. Today
    /// serialises through the inner mutex during the clone; the kernel
    /// run itself drops the lock before scoring. P6 wires the
    /// no-clone path so search becomes lock-free outside the brief
    /// snapshot read.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        mask: Option<&[u64]>,
    ) -> Result<Vec<SearchHit>, TurboQuantError> {
        if query.len() != self.dim {
            return Err(TurboQuantError::VectorBufferNotMultipleOfDim {
                vectors_len: query.len(),
                dim: self.dim,
            });
        }
        let rotation = self.rotation();
        let (_boundaries, centroids) = self.codebook();

        // Snapshot the encoded state under the mutex. The kernel then
        // runs outside the lock — this is the bottleneck that P6
        // optimises away by giving the kernel direct &[u8]/&[f32] slices.
        let (batch, calibration) = {
            let inner = self
                .inner
                .lock()
                .expect("TurboQuantStore::inner mutex poisoned");
            if inner.n_vectors == 0 {
                return Ok(Vec::new());
            }
            (
                EncodedBatch {
                    codes: inner.codes.clone(),
                    scales: inner.scales.clone(),
                    dim: self.dim,
                    bit_width: self.bit_width,
                    n_vectors: inner.n_vectors,
                },
                inner.calibration.clone(),
            )
        };

        kernel::search(
            query,
            &batch,
            rotation,
            centroids,
            calibration.as_ref(),
            k,
            mask,
        )
    }

    /// Remove the vector at the given slot. Semantics match
    /// [`Vec::swap_remove`]: the last vector is moved into the deleted
    /// slot, then the trailing entry is truncated. **Slot indices of
    /// the moved vector change** — any external id-map must rewrite
    /// the moved id to point at `slot`.
    ///
    /// Returns the previous slot of the moved vector (`n_vectors - 1`
    /// before the call). Equals `slot` when the removed entry was
    /// already the last element — in that case nothing was moved and
    /// the caller need not update an id-map.
    ///
    /// # Errors
    ///
    /// Returns [`TurboQuantError::InvalidFileFormat`] (with
    /// `"slot N out of bounds"`) when `slot >= len()`. Out-of-range
    /// removal must NOT panic per the GA contract that every API
    /// misuse returns a typed error.
    pub fn remove_slot(&self, slot: usize) -> Result<usize, TurboQuantError> {
        let mut inner = self
            .inner
            .lock()
            .expect("TurboQuantStore::inner mutex poisoned");
        if slot >= inner.n_vectors {
            return Err(TurboQuantError::InvalidFileFormat(format!(
                "slot {slot} out of bounds (n_vectors = {})",
                inner.n_vectors,
            )));
        }
        let bytes_per_vec = (self.dim * self.bit_width as usize).div_ceil(8);
        let last_slot = inner.n_vectors - 1;
        if slot != last_slot {
            // Move the last vector's packed bytes into `slot`. Use
            // `copy_within` so we don't double-borrow `inner.codes`.
            let src = last_slot * bytes_per_vec;
            let dst = slot * bytes_per_vec;
            inner.codes.copy_within(src..src + bytes_per_vec, dst);
            // Move the last vector's RaBitQ scale.
            inner.scales[slot] = inner.scales[last_slot];
        }
        inner.codes.truncate(last_slot * bytes_per_vec);
        inner.scales.truncate(last_slot);
        inner.n_vectors -= 1;
        Ok(last_slot)
    }

    /// Truncate the encoded state — drops all codes and scales,
    /// resets `n_vectors` to zero. **Preserves** the fitted TQ+
    /// calibration so re-adding vectors stays consistent with any
    /// `.tq` file that may have been saved before the clear (the
    /// calibration is part of the wire contract).
    ///
    /// To truly reset calibration, construct a new store with a fresh
    /// `(rotation_seed, calibration_mode)`. This API is for the common
    /// "drop and re-encode" case where the operator wants to rebuild
    /// the index in place.
    pub fn clear(&self) {
        let mut inner = self
            .inner
            .lock()
            .expect("TurboQuantStore::inner mutex poisoned");
        inner.codes.clear();
        inner.scales.clear();
        inner.n_vectors = 0;
        // calibration intentionally preserved — see method doc.
    }

    /// Snapshot the store's encoded state into a `.tq` file at `path`.
    ///
    /// Writes the LLD §3 byte sequence: 64-byte header + bit-packed codes
    /// + per-vector scales + (if TQ+) calibration. The file is the
    /// canonical persisted form; the rotation matrix is NOT included —
    /// it's a deterministic function of `(dim, rotation_seed)` and the
    /// loader regenerates it on first `search()`.
    ///
    /// The `encoded_epoch` field defaults to 0 in this API. Callers
    /// driving precision-epoch lifecycle (per `EMBEDDING_PRECISION_LLD_
    /// 2026_05_22` Q12) should supply it via [`Self::save_with_epoch`]
    /// instead.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), TurboQuantError> {
        self.save_with_epoch(path, 0)
    }

    /// Snapshot to any `impl Write` (in-memory buffer, network stream,
    /// fs::File wrapped in a BufWriter, …). Useful for the `IdMapIndex`
    /// wrapper which composes the store body with an ID footer.
    /// Behaviour mirrors [`Self::save_with_epoch`] but the caller owns
    /// the destination.
    pub fn save_with_epoch_to_writer<W: std::io::Write>(
        &self,
        w: &mut W,
        encoded_epoch: u64,
    ) -> Result<(), TurboQuantError> {
        let snapshot = {
            let inner = self
                .inner
                .lock()
                .expect("TurboQuantStore::inner mutex poisoned");
            PersistedStore {
                bit_width: self.bit_width,
                calibration_mode: self.calibration_mode,
                rotation_seed: self.rotation_seed,
                dim: self.dim,
                n_vectors: inner.n_vectors,
                encoded_epoch,
                codes: inner.codes.clone(),
                scales: inner.scales.clone(),
                calibration: inner.calibration.clone(),
            }
        };
        io::write_to(w, &snapshot)
    }

    /// Like [`Self::save`] but stamps the file header's
    /// `encoded_epoch` with the supplied value. P8 wires this from the
    /// xCatalog quantization-epoch column when the engine integration
    /// lands.
    pub fn save_with_epoch(
        &self,
        path: impl AsRef<std::path::Path>,
        encoded_epoch: u64,
    ) -> Result<(), TurboQuantError> {
        // Snapshot under the mutex; write to disk outside the lock so
        // concurrent searches aren't stalled by the I/O.
        let snapshot = {
            let inner = self
                .inner
                .lock()
                .expect("TurboQuantStore::inner mutex poisoned");
            PersistedStore {
                bit_width: self.bit_width,
                calibration_mode: self.calibration_mode,
                rotation_seed: self.rotation_seed,
                dim: self.dim,
                n_vectors: inner.n_vectors,
                encoded_epoch,
                codes: inner.codes.clone(),
                scales: inner.scales.clone(),
                calibration: inner.calibration.clone(),
            }
        };
        let file = std::fs::File::create(path.as_ref()).map_err(|e| {
            TurboQuantError::InvalidFileFormat(
                format!("could not create {:?}: {e}", path.as_ref(),),
            )
        })?;
        let mut writer = std::io::BufWriter::new(file);
        io::write_to(&mut writer, &snapshot)?;
        use std::io::Write;
        writer
            .flush()
            .map_err(|e| TurboQuantError::InvalidFileFormat(format!("flush failed: {e}")))?;
        Ok(())
    }

    /// Atomic hot-reload from a `.tq` file. The new file's shape
    /// (`dim`, `bit_width`, `calibration_mode`, `rotation_seed`) MUST
    /// match this store — rejected as `InvalidFileFormat` otherwise so
    /// callers can't accidentally swap a collection's index against a
    /// differently-configured snapshot.
    ///
    /// The atomicity guarantee: under the inner mutex, the encoded
    /// state (codes + scales + calibration + n_vectors) is replaced as
    /// a single critical section. Concurrent `search()` callers see
    /// either the pre-reload state or the post-reload state, never a
    /// partial state. Existing `OnceLock` caches (rotation matrix,
    /// codebook) are preserved because their inputs — `(dim,
    /// rotation_seed)` and `(bit_width, dim)` — are part of the
    /// shape-match contract and therefore unchanged.
    ///
    /// Use case: an out-of-band rebuild (tier-migration job, manual
    /// fix-up, batched re-encode) writes a new `.tq` and signals the
    /// running service to pick it up. The service calls `reload_from()`
    /// without dropping any `Arc<TurboQuantStore>` handles.
    pub fn reload_from(&self, path: impl AsRef<std::path::Path>) -> Result<(), TurboQuantError> {
        let file = std::fs::File::open(path.as_ref()).map_err(|e| {
            TurboQuantError::InvalidFileFormat(format!("could not open {:?}: {e}", path.as_ref(),))
        })?;
        let mut reader = std::io::BufReader::new(file);
        let persisted = io::read_from(&mut reader)?;
        self.swap_in(persisted)
    }

    /// Same as [`Self::reload_from`] but takes an already-deserialized
    /// [`PersistedStore`]. Useful when the new state arrives from
    /// somewhere other than the local filesystem (replica feed, S3
    /// object, in-memory rebuild).
    pub fn swap_in(&self, persisted: PersistedStore) -> Result<(), TurboQuantError> {
        // Shape match — the four fields below are part of the store's
        // identity. Catalog-driven rebuilds preserve them by
        // construction; this check catches accidental cross-collection
        // loads and operator typos.
        if persisted.dim != self.dim {
            return Err(TurboQuantError::InvalidFileFormat(format!(
                "reload dim mismatch: store={} file={}",
                self.dim, persisted.dim,
            )));
        }
        if persisted.bit_width != self.bit_width {
            return Err(TurboQuantError::InvalidFileFormat(format!(
                "reload bit_width mismatch: store={} file={}",
                self.bit_width, persisted.bit_width,
            )));
        }
        if persisted.calibration_mode != self.calibration_mode {
            return Err(TurboQuantError::InvalidFileFormat(format!(
                "reload calibration_mode mismatch: store={:?} file={:?}",
                self.calibration_mode, persisted.calibration_mode,
            )));
        }
        if persisted.rotation_seed != self.rotation_seed {
            return Err(TurboQuantError::InvalidFileFormat(format!(
                "reload rotation_seed mismatch: store={} file={}",
                self.rotation_seed, persisted.rotation_seed,
            )));
        }

        let mut inner = self
            .inner
            .lock()
            .expect("TurboQuantStore::inner mutex poisoned");
        inner.codes = persisted.codes;
        inner.scales = persisted.scales;
        inner.n_vectors = persisted.n_vectors;
        inner.calibration = persisted.calibration;
        Ok(())
    }

    /// Restore a `TurboQuantStore` from a `.tq` file previously written
    /// by [`Self::save`]. The returned store has the same encoded state
    /// and TQ+ calibration; the rotation matrix and Lloyd-Max codebook
    /// caches start empty and re-materialize on first search.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, TurboQuantError> {
        let file = std::fs::File::open(path.as_ref()).map_err(|e| {
            TurboQuantError::InvalidFileFormat(format!("could not open {:?}: {e}", path.as_ref(),))
        })?;
        let mut reader = std::io::BufReader::new(file);
        let persisted = io::read_from(&mut reader)?;
        Self::from_persisted(persisted)
    }

    /// Hydrate a `TurboQuantStore` from an already-deserialized
    /// `PersistedStore`. Useful for tests that round-trip through a
    /// `Vec<u8>` without touching the filesystem.
    pub fn from_persisted(p: PersistedStore) -> Result<Self, TurboQuantError> {
        check_dim(p.dim)?;
        check_bit_width(p.bit_width)?;
        Ok(Self {
            dim: p.dim,
            bit_width: p.bit_width,
            calibration_mode: p.calibration_mode,
            rotation_seed: p.rotation_seed,
            rotation: OnceLock::new(),
            codebook_cache: OnceLock::new(),
            inner: Mutex::new(StoreInner {
                codes: p.codes,
                scales: p.scales,
                n_vectors: p.n_vectors,
                calibration: p.calibration,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chacha20::ChaCha8Rng;
    use rand::{RngExt, SeedableRng};
    use rand_distr::StandardNormal;

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

    #[test]
    fn new_rejects_bad_dim() {
        let err = TurboQuantStore::new(7, 4, CalibrationMode::Identity, 42).unwrap_err();
        assert!(matches!(err, TurboQuantError::DimNotMultipleOf8(7)));
    }

    #[test]
    fn new_rejects_bad_bit_width() {
        let err = TurboQuantStore::new(64, 5, CalibrationMode::Identity, 42).unwrap_err();
        assert!(matches!(err, TurboQuantError::BitWidthOutOfRange(5)));
    }

    #[test]
    fn new_store_is_empty() {
        let s = TurboQuantStore::new(64, 4, CalibrationMode::Identity, 42).unwrap();
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
        assert_eq!(s.dim(), 64);
        assert_eq!(s.bit_width(), 4);
        assert_eq!(s.calibration_mode(), CalibrationMode::Identity);
        assert_eq!(s.rotation_seed(), 42);
        assert!(!s.has_calibration());
    }

    #[test]
    fn empty_add_is_noop() {
        let s = TurboQuantStore::new(64, 4, CalibrationMode::Identity, 1).unwrap();
        s.add(&[]).unwrap();
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn add_rejects_misaligned_buffer() {
        let s = TurboQuantStore::new(8, 4, CalibrationMode::Identity, 1).unwrap();
        let v = vec![0.5f32; 9]; // not a multiple of dim=8
        let err = s.add(&v).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::VectorBufferNotMultipleOfDim {
                vectors_len: 9,
                dim: 8
            }
        ));
    }

    #[test]
    fn add_increments_len() {
        let dim = 16;
        let s = TurboQuantStore::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        let v = random_unit_vectors(5, dim, 100);
        s.add(&v).unwrap();
        assert_eq!(s.len(), 5);
        let v2 = random_unit_vectors(3, dim, 101);
        s.add(&v2).unwrap();
        assert_eq!(s.len(), 8);
    }

    #[test]
    fn search_on_empty_store_returns_empty() {
        let dim = 16;
        let s = TurboQuantStore::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        let q = vec![0.5f32; dim];
        let hits = s.search(&q, 5, None).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn search_rejects_wrong_dim_query() {
        let dim = 16;
        let s = TurboQuantStore::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        let q = vec![0.5f32; dim - 1];
        let err = s.search(&q, 5, None).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::VectorBufferNotMultipleOfDim { .. }
        ));
    }

    #[test]
    fn self_query_recovers_self_at_top_1() {
        // d=128, identity calibration. Pick a vector from the index and
        // query for it; expect to come back as top-1.
        let dim = 128;
        let n = 50;
        let s = TurboQuantStore::new(dim, 4, CalibrationMode::Identity, 7).unwrap();
        let v = random_unit_vectors(n, dim, 200);
        s.add(&v).unwrap();
        let q = &v[17 * dim..(17 + 1) * dim];
        let hits = s.search(q, 1, None).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].1, 17);
    }

    #[test]
    fn add_across_multiple_batches_searchable() {
        let dim = 64;
        let s = TurboQuantStore::new(dim, 4, CalibrationMode::Identity, 8).unwrap();
        // Three batches; total = 30 vectors.
        for batch in 0..3 {
            let v = random_unit_vectors(10, dim, 300 + batch as u64);
            s.add(&v).unwrap();
        }
        assert_eq!(s.len(), 30);
        let q = vec![0.5f32; dim];
        let hits = s.search(&q, 5, None).unwrap();
        assert_eq!(hits.len(), 5);
        // All slot indices must be in range.
        for h in &hits {
            assert!((h.1 as usize) < 30);
        }
    }

    #[test]
    fn tq_plus_mode_does_not_fit_calibration_below_threshold() {
        let dim = 64;
        let s = TurboQuantStore::new(dim, 4, CalibrationMode::TqPlus, 9).unwrap();
        let v = random_unit_vectors(500, dim, 400); // below TQPLUS_MIN_SAMPLES
        s.add(&v).unwrap();
        assert_eq!(s.len(), 500);
        // No calibration was fit — silent identity fallback (LLD Q7).
        assert!(!s.has_calibration());
    }

    #[test]
    fn tq_plus_mode_fits_calibration_at_threshold() {
        let dim = 64;
        let s = TurboQuantStore::new(dim, 4, CalibrationMode::TqPlus, 10).unwrap();
        let v = random_unit_vectors(1024, dim, 500); // above TQPLUS_MIN_SAMPLES
        s.add(&v).unwrap();
        assert_eq!(s.len(), 1024);
        assert!(s.has_calibration());
    }

    #[test]
    fn calibration_is_frozen_after_first_qualifying_batch() {
        let dim = 64;
        let s = TurboQuantStore::new(dim, 4, CalibrationMode::TqPlus, 11).unwrap();
        let v1 = random_unit_vectors(1024, dim, 600);
        s.add(&v1).unwrap();
        assert!(s.has_calibration());
        let snapshot1 = {
            let inner = s.inner.lock().unwrap();
            inner.calibration.clone().unwrap()
        };
        // Add a second batch — calibration must NOT change.
        let v2 = random_unit_vectors(1024, dim, 601);
        s.add(&v2).unwrap();
        let snapshot2 = {
            let inner = s.inner.lock().unwrap();
            inner.calibration.clone().unwrap()
        };
        assert_eq!(
            snapshot1, snapshot2,
            "calibration must be frozen after the first qualifying batch",
        );
    }

    #[test]
    fn search_with_mask_restricts_to_allowed_slots() {
        let dim = 64;
        let s = TurboQuantStore::new(dim, 4, CalibrationMode::Identity, 12).unwrap();
        let v = random_unit_vectors(64, dim, 700);
        s.add(&v).unwrap();

        let allowed = [3u32, 17, 42];
        let mut mask = vec![0u64; ((64 + 63) >> 6) as usize];
        for &slot in &allowed {
            mask[(slot >> 6) as usize] |= 1u64 << (slot & 63);
        }
        let q = vec![0.5f32; dim];
        let hits = s.search(&q, 5, Some(&mask)).unwrap();
        assert_eq!(hits.len(), 3);
        let mut idxs: Vec<u32> = hits.iter().map(|h| h.1).collect();
        idxs.sort();
        assert_eq!(idxs, allowed);
    }

    // ------------------------------------------------------------------
    // reload_from / swap_in — atomic hot-reload
    // ------------------------------------------------------------------

    #[test]
    fn reload_from_swaps_in_new_state_atomically() {
        let dim = 32;
        let bw = 4;
        let seed = 0xfade_face_u64;
        let mode = CalibrationMode::Identity;
        let s = TurboQuantStore::new(dim, bw, mode, seed).unwrap();
        // Initial state: 5 vectors.
        s.add(&random_unit_vectors(5, dim, 100)).unwrap();
        assert_eq!(s.len(), 5);

        // Build a different snapshot (10 vectors) by way of a sibling
        // store and tempfile, then reload `s` from it.
        let donor = TurboQuantStore::new(dim, bw, mode, seed).unwrap();
        donor.add(&random_unit_vectors(10, dim, 200)).unwrap();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        donor.save(tmp.path()).unwrap();

        s.reload_from(tmp.path()).unwrap();
        // After reload, len() reflects the new state, not the old.
        assert_eq!(s.len(), 10);
        // Search against the new state should return slots in 0..10
        // — the old 5 vectors are gone.
        let q = random_unit_vectors(1, dim, 300);
        let hits = s.search(&q, 5, None).unwrap();
        for h in &hits {
            assert!((h.1 as usize) < 10);
        }
    }

    #[test]
    fn reload_from_rejects_dim_mismatch() {
        let s = TurboQuantStore::new(32, 4, CalibrationMode::Identity, 1).unwrap();
        let donor = TurboQuantStore::new(64, 4, CalibrationMode::Identity, 1).unwrap();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        donor.save(tmp.path()).unwrap();
        let err = s.reload_from(tmp.path()).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::InvalidFileFormat(ref msg) if msg.contains("dim mismatch")
        ));
        // Original store unchanged.
        assert_eq!(s.dim(), 32);
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn reload_from_rejects_bit_width_mismatch() {
        let s = TurboQuantStore::new(32, 4, CalibrationMode::Identity, 1).unwrap();
        let donor = TurboQuantStore::new(32, 2, CalibrationMode::Identity, 1).unwrap();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        donor.save(tmp.path()).unwrap();
        let err = s.reload_from(tmp.path()).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::InvalidFileFormat(ref msg) if msg.contains("bit_width mismatch")
        ));
    }

    #[test]
    fn reload_from_rejects_rotation_seed_mismatch() {
        let s = TurboQuantStore::new(32, 4, CalibrationMode::Identity, 1).unwrap();
        let donor = TurboQuantStore::new(32, 4, CalibrationMode::Identity, 2).unwrap();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        donor.save(tmp.path()).unwrap();
        let err = s.reload_from(tmp.path()).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::InvalidFileFormat(ref msg) if msg.contains("rotation_seed mismatch")
        ));
    }

    #[test]
    fn reload_from_rejects_calibration_mode_mismatch() {
        let dim = 32;
        let s = TurboQuantStore::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        // Donor is TqPlus; feed it enough vectors that calibration fits
        // (otherwise io::read_from would reject the file's TqPlus header
        // with no calibration body before reaching the mode-mismatch
        // check we're trying to exercise).
        let donor = TurboQuantStore::new(dim, 4, CalibrationMode::TqPlus, 1).unwrap();
        donor.add(&random_unit_vectors(1024, dim, 100)).unwrap();
        assert!(donor.has_calibration());
        let tmp = tempfile::NamedTempFile::new().unwrap();
        donor.save(tmp.path()).unwrap();
        let err = s.reload_from(tmp.path()).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::InvalidFileFormat(ref msg) if msg.contains("calibration_mode mismatch")
        ));
    }

    #[test]
    fn reload_from_under_concurrent_readers_keeps_state_coherent() {
        // While one thread repeatedly reloads, several reader threads
        // search in a loop. Every snapshot they see must be coherent —
        // either entirely-old or entirely-new — never a partial state.
        // Coherence proxy: every returned slot must be < the store's
        // current len() at the moment of search.
        use std::sync::Arc;
        use std::thread;
        use std::time::{Duration, Instant};

        let dim = 32;
        let seed = 0x1234;
        let s = Arc::new(TurboQuantStore::new(dim, 4, CalibrationMode::Identity, seed).unwrap());
        s.add(&random_unit_vectors(5, dim, 100)).unwrap();

        // Build two donor files: one with 20 vectors, one with 50.
        // The writer alternates between them.
        let donor_a = TurboQuantStore::new(dim, 4, CalibrationMode::Identity, seed).unwrap();
        donor_a.add(&random_unit_vectors(20, dim, 201)).unwrap();
        let donor_b = TurboQuantStore::new(dim, 4, CalibrationMode::Identity, seed).unwrap();
        donor_b.add(&random_unit_vectors(50, dim, 202)).unwrap();
        let tmp_a = tempfile::NamedTempFile::new().unwrap();
        let tmp_b = tempfile::NamedTempFile::new().unwrap();
        donor_a.save(tmp_a.path()).unwrap();
        donor_b.save(tmp_b.path()).unwrap();

        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut handles = Vec::new();

        // Writer: alternating reload_from, finite count.
        {
            let s = Arc::clone(&s);
            let stop = Arc::clone(&stop);
            let path_a = tmp_a.path().to_path_buf();
            let path_b = tmp_b.path().to_path_buf();
            handles.push(thread::spawn(move || {
                for i in 0..6 {
                    let p = if i % 2 == 0 { &path_a } else { &path_b };
                    s.reload_from(p).unwrap();
                    thread::sleep(Duration::from_millis(2));
                }
                stop.store(true, std::sync::atomic::Ordering::SeqCst);
            }));
        }

        // 3 readers asserting coherence.
        let deadline = Instant::now() + Duration::from_secs(10);
        for r in 0..3 {
            let s = Arc::clone(&s);
            let stop = Arc::clone(&stop);
            handles.push(thread::spawn(move || {
                let q = random_unit_vectors(1, dim, 300 + r as u64);
                let mut searches = 0usize;
                while !stop.load(std::sync::atomic::Ordering::SeqCst) && Instant::now() < deadline {
                    let len_before = s.len();
                    let hits = s.search(&q, 5, None).unwrap();
                    // Every returned slot must be < the len observed
                    // **before** the search ran. If reload landed
                    // mid-search and shrank the store, we'd see slots
                    // ≥ len_before — that would prove a partial state.
                    // (We allow len to grow between observations.)
                    let len_after = s.len();
                    let coherent_bound = len_before.max(len_after);
                    for h in &hits {
                        assert!(
                            (h.1 as usize) < coherent_bound,
                            "reader saw partial state: slot {} ≥ coherent_bound {}",
                            h.1,
                            coherent_bound,
                        );
                    }
                    searches += 1;
                }
                assert!(searches > 0, "reader {r} got 0 searches");
            }));
        }

        for h in handles {
            h.join().expect("worker panicked");
        }
    }

    #[test]
    fn swap_in_directly_with_persisted_skips_filesystem() {
        // swap_in must work without ever touching the disk — useful
        // when the new state arrives from an in-memory rebuild or a
        // network feed.
        let dim = 16;
        let bw = 4;
        let seed = 7;
        let s = TurboQuantStore::new(dim, bw, CalibrationMode::Identity, seed).unwrap();
        s.add(&random_unit_vectors(3, dim, 100)).unwrap();

        // Construct a PersistedStore in memory by saving a donor to a
        // buffer.
        let donor = TurboQuantStore::new(dim, bw, CalibrationMode::Identity, seed).unwrap();
        donor.add(&random_unit_vectors(8, dim, 200)).unwrap();
        let mut buf = Vec::new();
        donor.save_with_epoch_to_writer(&mut buf, 0).unwrap();
        let persisted = io::read_from(&mut std::io::Cursor::new(buf)).unwrap();
        s.swap_in(persisted).unwrap();
        assert_eq!(s.len(), 8);
    }

    // ------------------------------------------------------------------
    // remove_slot + clear (delete support — GA contract)
    // ------------------------------------------------------------------

    #[test]
    fn remove_slot_out_of_range_returns_typed_error() {
        let s = TurboQuantStore::new(8, 4, CalibrationMode::Identity, 1).unwrap();
        let err = s.remove_slot(0).unwrap_err();
        assert!(matches!(err, TurboQuantError::InvalidFileFormat(_)));
    }

    #[test]
    fn remove_slot_last_element_is_truncate_only() {
        let dim = 16;
        let s = TurboQuantStore::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        let v = random_unit_vectors(3, dim, 100);
        s.add(&v).unwrap();
        // Removing the last slot returns the same slot (nothing moved).
        let moved = s.remove_slot(2).unwrap();
        assert_eq!(moved, 2);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn remove_slot_middle_element_moves_last_into_slot() {
        let dim = 16;
        let s = TurboQuantStore::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        let v = random_unit_vectors(5, dim, 100);
        s.add(&v).unwrap();
        // Capture the codes + scale for slot 4 (the last one) before
        // the swap so we can verify they landed at slot 1 after.
        let bytes_per_vec = (dim * 4) / 8;
        let (last_codes, last_scale) = {
            let inner = s.inner.lock().unwrap();
            let codes = inner.codes[4 * bytes_per_vec..5 * bytes_per_vec].to_vec();
            let scale = inner.scales[4];
            (codes, scale)
        };
        let moved = s.remove_slot(1).unwrap();
        assert_eq!(moved, 4, "should return previous slot of moved vector");
        assert_eq!(s.len(), 4);
        // The vector formerly at slot 4 must now live at slot 1.
        let (now_at_1_codes, now_at_1_scale) = {
            let inner = s.inner.lock().unwrap();
            let codes = inner.codes[1 * bytes_per_vec..2 * bytes_per_vec].to_vec();
            let scale = inner.scales[1];
            (codes, scale)
        };
        assert_eq!(last_codes, now_at_1_codes, "moved codes mismatch");
        assert_eq!(last_scale.to_bits(), now_at_1_scale.to_bits());
    }

    #[test]
    fn remove_slot_then_search_returns_only_remaining_slots() {
        let dim = 32;
        let s = TurboQuantStore::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        let v = random_unit_vectors(20, dim, 100);
        s.add(&v).unwrap();
        s.remove_slot(7).unwrap();
        s.remove_slot(3).unwrap();
        let q = random_unit_vectors(1, dim, 200);
        let hits = s.search(&q, 5, None).unwrap();
        // After two removes, len = 18. Each hit must be a valid slot.
        assert_eq!(s.len(), 18);
        for h in &hits {
            assert!(
                (h.1 as usize) < 18,
                "search returned slot {} but only 18 vectors remain",
                h.1,
            );
        }
    }

    #[test]
    fn remove_slot_supports_full_drain() {
        let dim = 16;
        let s = TurboQuantStore::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        let v = random_unit_vectors(3, dim, 100);
        s.add(&v).unwrap();
        // Remove until empty — must never panic.
        s.remove_slot(0).unwrap();
        s.remove_slot(0).unwrap();
        s.remove_slot(0).unwrap();
        assert_eq!(s.len(), 0);
        // Removing again must error, not panic.
        let err = s.remove_slot(0).unwrap_err();
        assert!(matches!(err, TurboQuantError::InvalidFileFormat(_)));
    }

    #[test]
    fn clear_truncates_state_but_preserves_calibration() {
        let dim = 64;
        let s = TurboQuantStore::new(dim, 4, CalibrationMode::TqPlus, 1).unwrap();
        let v = random_unit_vectors(1024, dim, 100);
        s.add(&v).unwrap();
        assert!(
            s.has_calibration(),
            "TQ+ should be fit after a 1024-vec batch"
        );

        s.clear();

        assert_eq!(s.len(), 0);
        assert!(
            s.has_calibration(),
            "clear() must preserve calibration (wire-contract requirement)",
        );
        // Stats should report zero codes + zero scales but the
        // calibration_bytes is still populated.
        let st = s.stats();
        assert_eq!(st.n_vectors, 0);
        assert_eq!(st.codes_bytes, 0);
        assert_eq!(st.scales_bytes, 0);
        assert_eq!(st.calibration_bytes, 2 * dim * 4);
    }

    #[test]
    fn add_after_clear_reuses_existing_calibration() {
        // Frozen-after-first-batch invariant: a new add() after clear()
        // must NOT re-fit calibration. We check this by snapshotting
        // the calibration before clear and after the subsequent add.
        let dim = 64;
        let s = TurboQuantStore::new(dim, 4, CalibrationMode::TqPlus, 1).unwrap();
        let v1 = random_unit_vectors(1024, dim, 100);
        s.add(&v1).unwrap();
        let cal_before = {
            let inner = s.inner.lock().unwrap();
            inner.calibration.clone().unwrap()
        };

        s.clear();
        let v2 = random_unit_vectors(1024, dim, 200);
        s.add(&v2).unwrap();
        let cal_after = {
            let inner = s.inner.lock().unwrap();
            inner.calibration.clone().unwrap()
        };
        assert_eq!(
            cal_before, cal_after,
            "calibration must stay frozen across clear()+add()",
        );
    }

    #[test]
    fn clear_then_add_reuses_round_trips_through_save() {
        // The wire contract requires .tq files to carry the same
        // calibration the encoded codes were quantized against. After
        // clear()+add(), saving and reloading must give us back a
        // working store that produces sensible search results.
        let dim = 32;
        let s = TurboQuantStore::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        s.add(&random_unit_vectors(10, dim, 100)).unwrap();
        s.clear();
        s.add(&random_unit_vectors(5, dim, 101)).unwrap();
        assert_eq!(s.len(), 5);

        let tmp = tempfile::NamedTempFile::new().unwrap();
        s.save(tmp.path()).unwrap();
        let restored = TurboQuantStore::load(tmp.path()).unwrap();
        assert_eq!(restored.len(), 5);
        let q = random_unit_vectors(1, dim, 200);
        let original = s.search(&q, 3, None).unwrap();
        let after = restored.search(&q, 3, None).unwrap();
        assert_eq!(original.len(), after.len());
        for (a, b) in original.iter().zip(after.iter()) {
            assert_eq!(a.1, b.1, "post-clear save/load slot mismatch");
        }
    }

    // ------------------------------------------------------------------
    // StoreStats observability
    // ------------------------------------------------------------------

    #[test]
    fn stats_on_empty_store_reports_zero_counts_but_meaningful_ratio() {
        let dim = 1536;
        let s = TurboQuantStore::new(dim, 2, CalibrationMode::Identity, 1).unwrap();
        let st = s.stats();
        assert_eq!(st.dim, 1536);
        assert_eq!(st.bit_width, 2);
        assert_eq!(st.n_vectors, 0);
        assert_eq!(st.codes_bytes, 0);
        assert_eq!(st.scales_bytes, 0);
        assert_eq!(st.calibration_bytes, 0);
        assert_eq!(st.total_bytes, 0);
        // 2-bit @ d=1536: codes = ceil(1536*2/8) = 384 bytes; scale = 4
        // bytes; per-vec total = 388. FP32 baseline = 6144. Ratio
        // ≈ 15.83×.
        assert!(
            (st.compression_ratio_vs_fp32 - 15.83).abs() < 0.05,
            "ratio = {}",
            st.compression_ratio_vs_fp32,
        );
        assert!(!st.has_calibration);
    }

    #[test]
    fn stats_4bit_compression_ratio_matches_lld() {
        let dim = 1536;
        let s = TurboQuantStore::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        let st = s.stats();
        // 4-bit @ d=1536: codes = 768 bytes; scale = 4 bytes; total per
        // vec = 772. FP32 = 6144. Ratio ≈ 7.96×.
        assert!(
            (st.compression_ratio_vs_fp32 - 7.96).abs() < 0.05,
            "ratio = {}",
            st.compression_ratio_vs_fp32,
        );
    }

    #[test]
    fn stats_total_bytes_grows_with_added_vectors() {
        let dim = 64;
        let s = TurboQuantStore::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        let before = s.stats().total_bytes;
        assert_eq!(before, 0);

        let v = random_unit_vectors(10, dim, 100);
        s.add(&v).unwrap();
        let after = s.stats().total_bytes;
        // 10 vectors × (ceil(64 * 4 / 8) + 4) = 10 × (32 + 4) = 360 bytes.
        assert_eq!(after, 360);
    }

    #[test]
    fn stats_records_calibration_bytes_when_tq_plus_fit() {
        let dim = 64;
        let s = TurboQuantStore::new(dim, 4, CalibrationMode::TqPlus, 1).unwrap();
        let v = random_unit_vectors(1024, dim, 100);
        s.add(&v).unwrap();
        let st = s.stats();
        assert!(st.has_calibration);
        // TQ+ calibration = 2 * dim * f32 = 2 * 64 * 4 = 512 bytes.
        assert_eq!(st.calibration_bytes, 512);
        // Codes + scales + calibration.
        let expected_codes = 1024 * 32; // ceil(64*4/8) = 32 per vec
        let expected_scales = 1024 * 4;
        assert_eq!(st.codes_bytes, expected_codes);
        assert_eq!(st.scales_bytes, expected_scales);
        assert_eq!(st.total_bytes, expected_codes + expected_scales + 512,);
    }

    #[test]
    fn stats_reports_no_calibration_when_below_threshold() {
        let dim = 64;
        let s = TurboQuantStore::new(dim, 4, CalibrationMode::TqPlus, 1).unwrap();
        let v = random_unit_vectors(500, dim, 100); // below TQPLUS_MIN_SAMPLES
        s.add(&v).unwrap();
        let st = s.stats();
        assert!(!st.has_calibration);
        assert_eq!(st.calibration_bytes, 0);
    }

    #[test]
    fn stats_mirrors_accessor_methods() {
        let s = TurboQuantStore::new(128, 2, CalibrationMode::TqPlus, 42).unwrap();
        let st = s.stats();
        assert_eq!(st.dim, s.dim());
        assert_eq!(st.bit_width, s.bit_width());
        assert_eq!(st.calibration_mode, s.calibration_mode());
        assert_eq!(st.rotation_seed, s.rotation_seed());
        assert_eq!(st.n_vectors, s.len());
        assert_eq!(st.has_calibration, s.has_calibration());
    }

    #[test]
    fn save_load_round_trip_identity_preserves_search_results() {
        // Build a store, run a query, save to a Vec<u8>, restore from
        // the Vec<u8>, run the same query, expect identical hits.
        let dim = 64;
        let n = 50;
        let s = TurboQuantStore::new(dim, 4, CalibrationMode::Identity, 1234).unwrap();
        let vectors = random_unit_vectors(n, dim, 1000);
        s.add(&vectors).unwrap();

        let q = random_unit_vectors(1, dim, 1001);
        let original = s.search(&q, 5, None).unwrap();

        // Round-trip through PersistedStore + write_to/read_from.
        let snapshot = {
            let inner = s.inner.lock().unwrap();
            PersistedStore {
                bit_width: 4,
                calibration_mode: CalibrationMode::Identity,
                rotation_seed: 1234,
                dim,
                n_vectors: inner.n_vectors,
                encoded_epoch: 0,
                codes: inner.codes.clone(),
                scales: inner.scales.clone(),
                calibration: None,
            }
        };
        let mut buf = Vec::new();
        io::write_to(&mut buf, &snapshot).unwrap();
        let mut cur = std::io::Cursor::new(buf);
        let restored = io::read_from(&mut cur).unwrap();
        let s2 = TurboQuantStore::from_persisted(restored).unwrap();

        assert_eq!(s2.len(), n);
        assert_eq!(s2.dim(), dim);
        assert_eq!(s2.bit_width(), 4);
        assert_eq!(s2.rotation_seed(), 1234);
        assert_eq!(s2.calibration_mode(), CalibrationMode::Identity);

        let restored_hits = s2.search(&q, 5, None).unwrap();
        assert_eq!(original.len(), restored_hits.len());
        for (a, b) in original.iter().zip(restored_hits.iter()) {
            assert_eq!(a.1, b.1, "slot mismatch after round-trip");
            assert!(
                (a.0 - b.0).abs() < 1e-5,
                "score drift after round-trip: {} vs {}",
                a.0,
                b.0,
            );
        }
    }

    #[test]
    fn save_load_round_trip_tq_plus_preserves_calibration() {
        let dim = 64;
        let n = 1024; // above TQPLUS_MIN_SAMPLES
        let s = TurboQuantStore::new(dim, 4, CalibrationMode::TqPlus, 5678).unwrap();
        let vectors = random_unit_vectors(n, dim, 2000);
        s.add(&vectors).unwrap();
        assert!(s.has_calibration());

        let q = random_unit_vectors(1, dim, 2001);
        let original = s.search(&q, 5, None).unwrap();

        // Round-trip — calibration must survive.
        let snapshot = {
            let inner = s.inner.lock().unwrap();
            PersistedStore {
                bit_width: 4,
                calibration_mode: CalibrationMode::TqPlus,
                rotation_seed: 5678,
                dim,
                n_vectors: inner.n_vectors,
                encoded_epoch: 11,
                codes: inner.codes.clone(),
                scales: inner.scales.clone(),
                calibration: inner.calibration.clone(),
            }
        };
        let mut buf = Vec::new();
        io::write_to(&mut buf, &snapshot).unwrap();
        let mut cur = std::io::Cursor::new(buf);
        let restored = io::read_from(&mut cur).unwrap();
        let s2 = TurboQuantStore::from_persisted(restored).unwrap();

        assert_eq!(s2.calibration_mode(), CalibrationMode::TqPlus);
        assert!(s2.has_calibration());

        let restored_hits = s2.search(&q, 5, None).unwrap();
        assert_eq!(original.len(), restored_hits.len());
        for (a, b) in original.iter().zip(restored_hits.iter()) {
            assert_eq!(a.1, b.1, "TQ+ round-trip slot mismatch");
            assert!(
                (a.0 - b.0).abs() < 1e-5,
                "TQ+ round-trip score drift: {} vs {}",
                a.0,
                b.0,
            );
        }
    }

    #[test]
    fn save_and_load_via_tempfile_round_trip() {
        // Same round-trip as above but through the filesystem.
        let dim = 32;
        let n = 20;
        let s = TurboQuantStore::new(dim, 4, CalibrationMode::Identity, 9001).unwrap();
        let vectors = random_unit_vectors(n, dim, 3000);
        s.add(&vectors).unwrap();
        let q = random_unit_vectors(1, dim, 3001);
        let original = s.search(&q, 3, None).unwrap();

        let tmp = tempfile::NamedTempFile::new().unwrap();
        s.save_with_epoch(tmp.path(), 42).unwrap();
        let s2 = TurboQuantStore::load(tmp.path()).unwrap();

        assert_eq!(s2.len(), n);
        assert_eq!(s2.dim(), dim);
        assert_eq!(s2.rotation_seed(), 9001);
        let restored = s2.search(&q, 3, None).unwrap();
        assert_eq!(original.len(), restored.len());
        for (a, b) in original.iter().zip(restored.iter()) {
            assert_eq!(a.1, b.1);
        }
    }

    #[test]
    fn load_rejects_nonexistent_file() {
        let err = TurboQuantStore::load("/tmp/turboquant-does-not-exist-xyz.tq").unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::InvalidFileFormat(ref s) if s.contains("could not open")
        ));
    }

    #[test]
    fn concurrent_search_returns_consistent_results() {
        // Multiple threads searching the same store must each get a
        // valid top-k. This exercises the mutex + OnceLock cache.
        use std::sync::Arc;
        use std::thread;

        let dim = 64;
        let s = Arc::new(TurboQuantStore::new(dim, 4, CalibrationMode::Identity, 13).unwrap());
        let v = random_unit_vectors(100, dim, 800);
        s.add(&v).unwrap();

        let mut handles = Vec::new();
        for t in 0..4 {
            let s = Arc::clone(&s);
            let q = random_unit_vectors(1, dim, 900 + t as u64);
            handles.push(thread::spawn(move || s.search(&q, 5, None).unwrap()));
        }
        for h in handles {
            let hits = h.join().unwrap();
            assert_eq!(hits.len(), 5);
            for hit in &hits {
                assert!((hit.1 as usize) < 100);
            }
        }
    }
}
