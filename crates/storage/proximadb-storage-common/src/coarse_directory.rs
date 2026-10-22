//! Region A0 — the persisted IVF probe directory (TD-RDSTRAT-8 rev 3).
//!
//! Compaction already trains an IOP-derived PCA/IVF plan (`cells ≈ N·dim/IOP`)
//! to order rows, then discards the model, centroids, and cell runs — so the
//! reader re-ranks every row and plans ranges blind to cell boundaries. The v3
//! layout **persists that existing plan** as a query-visible directory: rows are
//! ordered by IVF cell, every region (A/B/D) is cell-contiguous, and this
//! directory maps each cell to **direct ranged-GET byte extents** into every
//! region. The read path ranks the `k_c` centroids in RAM (trivial even at
//! k ∝ N — ~305 at 10M, ~3050 at 100M, capped 4096) and probes `nprobe` cells,
//! trading the whole-region O(N)-byte scan for O(nprobe) ranged GETs — the
//! dimensional cost term this layout moves. There is NO second `sqrt(N)`
//! quantizer (rev 3): the cells ARE the IOP-derived cells compaction computes.
//!
//! A0 is small (~120–300 KB: `k_c ≤ 4096` cells × 72 B + the PCA model) and sits
//! **immediately after the header-prefix**, so a cold query fetches
//! `[0, a0_off + a0_len)` in one ranged GET and every later wave is
//! nprobe-scoped. It rides the per-segment invariants cache thereafter.
//!
//! The directory persists the **write-time PCA model** (mean + components — the
//! TD-WLP-4b deferred persistence, required here): query-time coarse ranking
//! must project with the exact model the writer clustered with.
//!
//! ## Byte layout (fixed, little-endian, deterministic)
//!
//! ```text
//! [magic "PXA0" 4][version u8][flags u8][n_comp u16]
//! [k_c u32][dim u32]
//! [seed u64][trained_on u64][rows_covered u64]
//! [pca_mean:       dim × f32]
//! [pca_components: n_comp × dim × f32]   (row-major)
//! [centroids:      k_c × n_comp × f32]   (PCA space, emission/Hilbert order)
//! [radii:          k_c × f32]            (PCA space, max member→centroid)
//! [cells:          k_c × 72 B]           (see CoarseCellEntry)
//! [checksum u64]                          (FNV-1a over all preceding bytes)
//! ```
//!
//! Rows NOT covered by any cell (records without a usable embedding) sort to
//! the segment tail, after `rows_covered`; they carry no vector so the coarse
//! probe can never miss them.

#![forbid(unsafe_code)]

use anyhow::{Result, bail};

/// Region A0 magic — first 4 bytes of the serialized directory.
pub const A0_MAGIC: &[u8; 4] = b"PXA0";

/// A0 serde version. Single version pre-GA (in-place evolution — no versioned
/// files on disk); versioning re-engages at GA.
pub const A0_VERSION: u8 = 1;

/// Fixed head: magic(4) + version(1) + flags(1) + n_comp(2) + k_c(4) + dim(4)
/// + seed(8) + trained_on(8) + rows_covered(8).
const A0_FIXED_HEAD_LEN: usize = 40;

/// One per-cell entry: row range + direct byte extents into Regions A/B/C + the
/// Region D block range.
const A0_CELL_ENTRY_LEN: usize = 8 * 8 + 4 * 2;

/// Trailing checksum length.
const A0_CHECKSUM_LEN: usize = 8;

/// The trained coarse model — everything the clustering step produces. The
/// writer combines this with the per-cell byte extents (which only it knows,
/// at region-assembly time) into the serialized [`CoarseDirectory`].
#[derive(Debug, Clone, PartialEq)]
pub struct CoarseModel {
    /// Original embedding dimensionality.
    pub dim: u32,
    /// PCA projection dimensionality (`n_comp ≤ dim`).
    pub n_comp: u16,
    /// PCA per-dimension mean (`dim` values).
    pub pca_mean: Vec<f32>,
    /// PCA components, row-major (`n_comp × dim` values).
    pub pca_components: Vec<f32>,
    /// Coarse centroids in PCA space, emission (Hilbert) order
    /// (`k_c × n_comp` values).
    pub centroids: Vec<f32>,
    /// Per-cell max member→centroid distance in PCA space (`k_c` values) — the
    /// escalation lower bound `LB(c) = d_pca(q, centroid_c) − radius_c`.
    pub radii: Vec<f32>,
    /// Rows per cell in emission order (`k_c` values; empty cells are 0). The
    /// writer turns the prefix sums into cell row ranges and block boundaries.
    pub cell_rows: Vec<u64>,
    /// Deterministic k-means seed (diagnostics + reproducibility).
    pub seed: u64,
    /// Number of training samples the PCA/k-means saw (diagnostics).
    pub trained_on: u64,
}

impl CoarseModel {
    /// Coarse cell count.
    pub fn k_c(&self) -> usize {
        self.radii.len()
    }

    /// Total rows covered by cells (rows past this are the no-embedding tail).
    pub fn rows_covered(&self) -> u64 {
        self.cell_rows.iter().sum()
    }

    /// Structural validation — every array length must agree with
    /// `k_c/dim/n_comp`. Fail-closed: a malformed model must never serialize.
    pub fn validate(&self) -> Result<()> {
        let k_c = self.k_c();
        if k_c == 0 {
            bail!("coarse directory requires at least one cell");
        }
        if self.dim == 0 || self.n_comp == 0 || self.n_comp as u32 > self.dim {
            bail!(
                "coarse directory: invalid dims (dim={}, n_comp={})",
                self.dim,
                self.n_comp
            );
        }
        if self.pca_mean.len() != self.dim as usize {
            bail!("coarse directory: pca_mean length mismatch");
        }
        if self.pca_components.len() != self.n_comp as usize * self.dim as usize {
            bail!("coarse directory: pca_components length mismatch");
        }
        if self.centroids.len() != k_c * self.n_comp as usize {
            bail!("coarse directory: centroids length mismatch");
        }
        if self.cell_rows.len() != k_c {
            bail!("coarse directory: cell_rows length mismatch");
        }
        Ok(())
    }
}

/// One coarse cell's addressing entry: the global row range plus **direct
/// ranged-GET byte extents** into every region (absolute file offsets), so a
/// probe never derives offsets from strides at read time — the writer, which
/// owns the layout, records them exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CoarseCellEntry {
    /// Global row range `[row_begin, row_end)` in segment (cluster) order.
    pub row_begin: u64,
    pub row_end: u64,
    /// Region A (RaBitQ codes) byte extent for this cell's rows.
    pub a_off: u64,
    pub a_len: u64,
    /// Region B (SQ8 codes) byte extent for this cell's rows.
    pub b_off: u64,
    pub b_len: u64,
    /// Region C (optional exact fp32) byte extent — 0/0 until a hoisted fp32
    /// region exists (today the exact tier lives in Region D blocks).
    pub c_off: u64,
    pub c_len: u64,
    /// Region D block-ordinal range `[d_block_begin, d_block_end)` — block
    /// boundaries never straddle a coarse cell (the writer pads/flushes at
    /// every cell boundary).
    pub d_block_begin: u32,
    pub d_block_end: u32,
}

impl CoarseCellEntry {
    pub(crate) fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.row_begin.to_le_bytes());
        out.extend_from_slice(&self.row_end.to_le_bytes());
        out.extend_from_slice(&self.a_off.to_le_bytes());
        out.extend_from_slice(&self.a_len.to_le_bytes());
        out.extend_from_slice(&self.b_off.to_le_bytes());
        out.extend_from_slice(&self.b_len.to_le_bytes());
        out.extend_from_slice(&self.c_off.to_le_bytes());
        out.extend_from_slice(&self.c_len.to_le_bytes());
        out.extend_from_slice(&self.d_block_begin.to_le_bytes());
        out.extend_from_slice(&self.d_block_end.to_le_bytes());
    }

    pub(crate) fn read_from(input: &[u8], p: &mut usize) -> Result<Self> {
        Ok(Self {
            row_begin: read_u64(input, p)?,
            row_end: read_u64(input, p)?,
            a_off: read_u64(input, p)?,
            a_len: read_u64(input, p)?,
            b_off: read_u64(input, p)?,
            b_len: read_u64(input, p)?,
            c_off: read_u64(input, p)?,
            c_len: read_u64(input, p)?,
            d_block_begin: read_u32(input, p)?,
            d_block_end: read_u32(input, p)?,
        })
    }
}

/// The serialized Region A0: the trained model plus per-cell addressing.
#[derive(Debug, Clone, PartialEq)]
pub struct CoarseDirectory {
    pub model: CoarseModel,
    /// Per-cell addressing, 1:1 with the model's emission-ordered cells.
    pub cells: Vec<CoarseCellEntry>,
}

impl CoarseDirectory {
    /// Exact serialized byte length for a `(k_c, dim, n_comp)` directory — the
    /// writer sizes the region (and therefore every downstream offset) from
    /// this **before** serializing, so region offsets and A0 contents can be
    /// computed in one pass without placeholder rewrites.
    pub fn serialized_len(k_c: usize, dim: usize, n_comp: usize) -> usize {
        A0_FIXED_HEAD_LEN
            + dim * 4                    // pca_mean
            + n_comp * dim * 4           // pca_components
            + k_c * n_comp * 4           // centroids
            + k_c * 4                    // radii
            + k_c * A0_CELL_ENTRY_LEN    // cells
            + A0_CHECKSUM_LEN
    }

    /// Serialize (deterministic — identical input ⇒ identical bytes; a physical
    /// layout must never depend on ambient state).
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.model.validate()?;
        if self.cells.len() != self.model.k_c() {
            bail!("coarse directory: cells length disagrees with model k_c");
        }
        let m = &self.model;
        let k_c = m.k_c();
        let expected = Self::serialized_len(k_c, m.dim as usize, m.n_comp as usize);
        let mut buf = Vec::with_capacity(expected);
        buf.extend_from_slice(A0_MAGIC);
        buf.push(A0_VERSION);
        buf.push(0); // flags (reserved)
        buf.extend_from_slice(&m.n_comp.to_le_bytes());
        let k_c_u32 =
            u32::try_from(k_c).map_err(|_| anyhow::anyhow!("coarse cell count exceeds u32"))?;
        buf.extend_from_slice(&k_c_u32.to_le_bytes());
        buf.extend_from_slice(&m.dim.to_le_bytes());
        buf.extend_from_slice(&m.seed.to_le_bytes());
        buf.extend_from_slice(&m.trained_on.to_le_bytes());
        buf.extend_from_slice(&m.rows_covered().to_le_bytes());
        for &v in &m.pca_mean {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &m.pca_components {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &m.centroids {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &m.radii {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for cell in &self.cells {
            cell.write_to(&mut buf);
        }
        let checksum = fnv1a64(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        if buf.len() != expected {
            bail!(
                "coarse directory serialized length {} != computed {expected}",
                buf.len()
            );
        }
        Ok(buf)
    }

    /// Parse + verify. Fail-closed on magic/version/truncation/checksum — a
    /// corrupt directory must degrade to an error (the caller falls back to the
    /// single-level scan), never a silent mis-probe.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < A0_FIXED_HEAD_LEN + A0_CHECKSUM_LEN {
            bail!("coarse directory too short: {}", bytes.len());
        }
        if &bytes[..4] != A0_MAGIC {
            bail!("not a coarse directory (bad magic)");
        }
        let version = bytes[4];
        if version != A0_VERSION {
            bail!("unsupported coarse directory version {version}");
        }
        let mut p = 6usize; // past magic + version + flags
        let n_comp = read_u16(bytes, &mut p)?;
        let k_c = read_u32(bytes, &mut p)? as usize;
        let dim = read_u32(bytes, &mut p)?;
        let seed = read_u64(bytes, &mut p)?;
        let trained_on = read_u64(bytes, &mut p)?;
        let rows_covered = read_u64(bytes, &mut p)?;
        let expected = Self::serialized_len(k_c, dim as usize, n_comp as usize);
        if bytes.len() != expected {
            bail!(
                "coarse directory length {} != expected {expected} (k_c={k_c}, dim={dim}, n_comp={n_comp})",
                bytes.len()
            );
        }
        let body_len = bytes.len() - A0_CHECKSUM_LEN;
        let stored = u64::from_le_bytes(bytes[body_len..].try_into()?);
        let computed = fnv1a64(&bytes[..body_len]);
        if stored != computed {
            bail!("coarse directory checksum mismatch");
        }
        let pca_mean = read_f32s(bytes, &mut p, dim as usize)?;
        let pca_components = read_f32s(bytes, &mut p, n_comp as usize * dim as usize)?;
        let centroids = read_f32s(bytes, &mut p, k_c * n_comp as usize)?;
        let radii = read_f32s(bytes, &mut p, k_c)?;
        let mut cells = Vec::with_capacity(k_c);
        for _ in 0..k_c {
            cells.push(CoarseCellEntry::read_from(bytes, &mut p)?);
        }
        let model = CoarseModel {
            dim,
            n_comp,
            pca_mean,
            pca_components,
            centroids,
            radii,
            cell_rows: cells.iter().map(|c| c.row_end - c.row_begin).collect(),
            seed,
            trained_on,
        };
        model.validate()?;
        if model.rows_covered() != rows_covered {
            bail!("coarse directory rows_covered disagrees with cell ranges");
        }
        Ok(Self { model, cells })
    }

    /// Project `query` into PCA space with the persisted model — the exact
    /// projection the writer clustered with (query-time coarse ranking, PR-B).
    pub fn project(&self, query: &[f32]) -> Vec<f32> {
        let m = &self.model;
        project_with_model(&m.pca_mean, &m.pca_components, m.n_comp as usize, query)
    }
}

/// Project `v` with an f32-persisted PCA model (`mean` len `dim`, `components`
/// row-major `n_comp × dim`). The **single shared projection kernel**: the
/// write path assigns/radius-bounds with it and the read path ranks with it, so
/// both sides compute bit-identical coordinates from the persisted f32 model
/// (no write-f64 vs read-f32 drift at cell boundaries). Vectors stay f32 —
/// only the dot-product accumulator widens to f64 (standard cancellation-safe
/// accumulation; transient, never stored).
pub fn project_with_model(mean: &[f32], components: &[f32], n_comp: usize, v: &[f32]) -> Vec<f32> {
    let dim = mean.len();
    let mut out = vec![0f32; n_comp];
    for (c, slot) in out.iter_mut().enumerate() {
        let row = &components[c * dim..(c + 1) * dim];
        let mut dot = 0f64;
        for j in 0..dim.min(v.len()) {
            dot += (v[j] as f64 - mean[j] as f64) * row[j] as f64;
        }
        *slot = dot as f32;
    }
    out
}

/// FNV-1a 64-bit over raw bytes (checksum — deterministic, dependency-free).
pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

fn ensure_remaining(input: &[u8], position: usize, len: usize) -> Result<()> {
    let end = position
        .checked_add(len)
        .ok_or_else(|| anyhow::anyhow!("coarse directory: length overflow"))?;
    if end > input.len() {
        bail!("coarse directory truncated");
    }
    Ok(())
}

pub(crate) fn read_u16(input: &[u8], p: &mut usize) -> Result<u16> {
    ensure_remaining(input, *p, 2)?;
    let v = u16::from_le_bytes(input[*p..*p + 2].try_into()?);
    *p += 2;
    Ok(v)
}

pub(crate) fn read_u32(input: &[u8], p: &mut usize) -> Result<u32> {
    ensure_remaining(input, *p, 4)?;
    let v = u32::from_le_bytes(input[*p..*p + 4].try_into()?);
    *p += 4;
    Ok(v)
}

pub(crate) fn read_u64(input: &[u8], p: &mut usize) -> Result<u64> {
    ensure_remaining(input, *p, 8)?;
    let v = u64::from_le_bytes(input[*p..*p + 8].try_into()?);
    *p += 8;
    Ok(v)
}

pub(crate) fn read_f32s(input: &[u8], p: &mut usize, count: usize) -> Result<Vec<f32>> {
    ensure_remaining(input, *p, count * 4)?;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let off = *p + i * 4;
        out.push(f32::from_le_bytes(input[off..off + 4].try_into()?));
    }
    *p += count * 4;
    Ok(out)
}

#[cfg(test)]
mod a0_size_budget_tests {
    use super::CoarseDirectory;

    /// A0 is fetched **whole** on the first touch of a segment, before any cell
    /// is probed, so its size is cold-start latency paid on the critical path.
    /// The co-design invariant (ADR-065) is that one fetch ≈ one IOP-sized
    /// block, so the directory must fit inside a single 4 MiB IOP even at the
    /// worst supported geometry.
    const IOP_TARGET_BYTES: usize = 4 * 1024 * 1024;

    /// Widest supported corner: `k_c` is clamped to 4096 by `ivf_fine_cell_count`
    /// and 3072 is the largest embedding dimension we ship against (OpenAI
    /// text-embedding-3-large).
    const WIDEST_K_C: usize = 4096;
    const WIDEST_DIM: usize = 3072;

    /// A0 grows linearly in `n_comp` on *two* terms — `n_comp·dim` components
    /// and `k_c·n_comp` centroids — so raising the projection-width floor
    /// (TD-IVF-3) inflates the cold prefix everywhere, not just on wide corpora.
    /// This is the guard that keeps a future floor increase from silently
    /// turning the first query of every segment into a multi-IOP fetch.
    #[test]
    fn a0_fits_one_iop_at_the_widest_supported_geometry() {
        for n_comp in [14usize, 32, 64] {
            let len = CoarseDirectory::serialized_len(WIDEST_K_C, WIDEST_DIM, n_comp);
            assert!(
                len <= IOP_TARGET_BYTES,
                "A0 at k_c={WIDEST_K_C} dim={WIDEST_DIM} n_comp={n_comp} is {len} bytes, \
                 over the {IOP_TARGET_BYTES}-byte IOP target: the cold prefix would cost \
                 more than one round-trip before any cell is read"
            );
        }
    }

    /// Pins the shape of the growth so a regression in `serialized_len` (an
    /// added per-cell or per-component field, say) shows up here rather than as
    /// a quietly larger cold read.
    #[test]
    fn a0_growth_is_affine_in_projection_width() {
        let at = |n| CoarseDirectory::serialized_len(WIDEST_K_C, WIDEST_DIM, n);
        // Equal steps in width must produce equal increments — the two
        // width-linear terms are `n_comp·dim` and `k_c·n_comp`, with no
        // higher-order term in `n_comp`.
        let step_a = at(32) - at(16);
        let step_b = at(48) - at(32);
        let step_c = at(64) - at(48);
        assert_eq!(
            step_a, step_b,
            "width growth must be affine, not accelerating"
        );
        assert_eq!(
            step_b, step_c,
            "width growth must be affine, not accelerating"
        );
        // And the width-independent terms (mean, radii, cell entries) must be a
        // real intercept, so doubling the width less than doubles the directory.
        assert!(
            at(64) < 2 * at(32),
            "expected a non-zero width-independent term"
        );
    }
}

/// TD-IVF-3 mixed-read safety: raising the projection-width default changes the
/// *value* of `n_comp`, never the byte layout. Old and new segments must coexist
/// in one collection, each probed at its own persisted width, with no version
/// bump and no reader branch.
#[cfg(test)]
mod mixed_width_read_tests {
    use super::*;

    fn directory_at_width(n_comp: u16) -> CoarseDirectory {
        let dim = 64u32;
        let k_c = 4usize;
        let model = CoarseModel {
            dim,
            n_comp,
            pca_mean: (0..dim).map(|i| i as f32 * 0.5).collect(),
            pca_components: (0..n_comp as usize * dim as usize)
                .map(|i| (i as f32).sin())
                .collect(),
            centroids: (0..k_c * n_comp as usize)
                .map(|i| i as f32 * 0.25)
                .collect(),
            radii: vec![1.0, 2.0, 0.0, 3.5],
            cell_rows: vec![100, 50, 0, 25],
            seed: 0xABCD_1234_5678_9ABC,
            trained_on: 175,
        };
        let mut row = 0u64;
        let mut cells = Vec::new();
        for (i, &rows) in model.cell_rows.iter().enumerate() {
            cells.push(CoarseCellEntry {
                row_begin: row,
                row_end: row + rows,
                a_off: 1000 + row * 24,
                a_len: rows * 24,
                b_off: 9000 + row * 8,
                b_len: rows * 8,
                c_off: 0,
                c_len: 0,
                d_block_begin: i as u32,
                d_block_end: i as u32 + u32::from(rows > 0),
            });
            row += rows;
        }
        CoarseDirectory { model, cells }
    }

    /// The load-bearing property: a directory written at the legacy width and one
    /// written at the widened floor both parse, independently, each reporting its
    /// own `n_comp`. Nothing recomputes the width from `dim`/`k_c` at read time,
    /// so there is no flag-day and no mis-probe path.
    #[test]
    fn legacy_and_widened_directories_parse_independently() {
        // 14 = the legacy formula at 768-d; 64 = the measured optimum.
        let legacy = directory_at_width(14);
        let widened = directory_at_width(64);

        let legacy_bytes = legacy.to_bytes().expect("legacy directory must serialize");
        let widened_bytes = widened
            .to_bytes()
            .expect("widened directory must serialize");

        // Different widths ⇒ different lengths, but the SAME layout: each length
        // is exactly what `serialized_len` predicts for its own width.
        assert_ne!(legacy_bytes.len(), widened_bytes.len());
        assert_eq!(
            legacy_bytes.len(),
            CoarseDirectory::serialized_len(4, 64, 14)
        );
        assert_eq!(
            widened_bytes.len(),
            CoarseDirectory::serialized_len(4, 64, 64)
        );

        let parsed_legacy =
            CoarseDirectory::parse(&legacy_bytes).expect("legacy segment must still parse");
        let parsed_widened =
            CoarseDirectory::parse(&widened_bytes).expect("widened segment must parse");

        assert_eq!(parsed_legacy.model.n_comp, 14);
        assert_eq!(parsed_widened.model.n_comp, 64);
        assert_eq!(parsed_legacy, legacy);
        assert_eq!(parsed_widened, widened);
        // Same magic and version: the widened directory is not a new format.
        assert_eq!(&legacy_bytes[..4], A0_MAGIC);
        assert_eq!(&widened_bytes[..4], A0_MAGIC);
        assert_eq!(legacy_bytes[4], widened_bytes[4], "version must not change");
    }

    /// Parsing must be driven by the persisted `n_comp`, not by anything the
    /// caller assumes. Truncating a widened directory to the length a *legacy*
    /// one would occupy must fail closed rather than silently decode a prefix as
    /// a valid narrower model — that would be the silent mis-probe path.
    #[test]
    fn a_widened_directory_truncated_to_legacy_length_fails_closed() {
        let widened = directory_at_width(64);
        let bytes = widened.to_bytes().expect("must serialize");
        let legacy_len = CoarseDirectory::serialized_len(4, 64, 14);
        assert!(legacy_len < bytes.len());
        assert!(
            CoarseDirectory::parse(&bytes[..legacy_len]).is_err(),
            "a truncated directory must fail closed, never decode as a narrower model"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_directory() -> CoarseDirectory {
        let dim = 8u32;
        let n_comp = 3u16;
        let k_c = 4usize;
        let model = CoarseModel {
            dim,
            n_comp,
            pca_mean: (0..dim).map(|i| i as f32 * 0.5).collect(),
            pca_components: (0..n_comp as usize * dim as usize)
                .map(|i| (i as f32).sin())
                .collect(),
            centroids: (0..k_c * n_comp as usize)
                .map(|i| i as f32 * 0.25)
                .collect(),
            radii: vec![1.0, 2.0, 0.0, 3.5],
            cell_rows: vec![100, 50, 0, 25],
            // Arbitrary round-trip fixture value (the serde container is
            // seed-agnostic — it stores whatever u64 the plan trained with).
            seed: 0xABCD_1234_5678_9ABC,
            trained_on: 175,
        };
        let mut row = 0u64;
        let mut cells = Vec::new();
        for (i, &rows) in model.cell_rows.iter().enumerate() {
            cells.push(CoarseCellEntry {
                row_begin: row,
                row_end: row + rows,
                a_off: 1000 + row * 24,
                a_len: rows * 24,
                b_off: 9000 + row * 8,
                b_len: rows * 8,
                c_off: 0,
                c_len: 0,
                d_block_begin: i as u32,
                d_block_end: i as u32 + u32::from(rows > 0),
            });
            row += rows;
        }
        CoarseDirectory { model, cells }
    }

    #[test]
    fn round_trips_bytes_and_fields() {
        let dir = sample_directory();
        let bytes = dir.to_bytes().unwrap();
        assert_eq!(
            bytes.len(),
            CoarseDirectory::serialized_len(4, 8, 3),
            "serialized_len must predict the exact byte length"
        );
        let parsed = CoarseDirectory::parse(&bytes).unwrap();
        assert_eq!(parsed, dir);
        // Determinism: serialize again ⇒ identical bytes.
        assert_eq!(parsed.to_bytes().unwrap(), bytes);
    }

    #[test]
    fn checksum_detects_corruption() {
        let bytes = sample_directory().to_bytes().unwrap();
        // Flip one payload byte (inside centroids, past the fixed head).
        let mut corrupt = bytes.clone();
        corrupt[A0_FIXED_HEAD_LEN + 10] ^= 0x01;
        let err = CoarseDirectory::parse(&corrupt).unwrap_err();
        assert!(err.to_string().contains("checksum"), "{err}");
    }

    #[test]
    fn parse_fail_closed_on_magic_version_truncation() {
        let bytes = sample_directory().to_bytes().unwrap();
        // Bad magic.
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert!(CoarseDirectory::parse(&bad).is_err());
        // Unknown version.
        let mut v = bytes.clone();
        v[4] = 99;
        assert!(CoarseDirectory::parse(&v).is_err());
        // Truncation (any prefix must error, never panic).
        for cut in [0usize, 5, A0_FIXED_HEAD_LEN, bytes.len() - 1] {
            assert!(CoarseDirectory::parse(&bytes[..cut]).is_err(), "cut={cut}");
        }
    }

    #[test]
    fn validate_rejects_mismatched_lengths() {
        let mut dir = sample_directory();
        dir.model.centroids.pop();
        assert!(dir.to_bytes().is_err());
        let mut dir2 = sample_directory();
        dir2.cells.pop();
        assert!(dir2.to_bytes().is_err());
    }

    #[test]
    fn project_uses_mean_and_components() {
        let dir = sample_directory();
        let query: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let proj = dir.project(&query);
        assert_eq!(proj.len(), 3);
        // Manual check of component 0.
        let m = &dir.model;
        let expect: f64 = (0..8)
            .map(|j| (query[j] as f64 - m.pca_mean[j] as f64) * m.pca_components[j] as f64)
            .sum();
        assert!((proj[0] as f64 - expect).abs() < 1e-4);
    }
}
