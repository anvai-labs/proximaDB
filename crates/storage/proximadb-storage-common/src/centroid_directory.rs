//! Collection-global PAX centroid directory (TD-PAXDIR-1, slice D1).
//!
//! Every coalesced PAX segment carries its own Region A0 coarse directory
//! ([`crate::coarse_directory`]): a PCA model, `k_c` centroids and per-cell
//! byte extents. A query therefore pays each segment's header and A0 before it
//! can probe anything, and `nprobe` is chosen per segment. This module is the
//! **collection-level** view of those A0s: one small object, published by
//! compaction beside the segments, that lists every covered segment's cells so
//! a query ranks cells across the whole collection in RAM and probes only the
//! segments that own the winners (design:
//! `docs/12-design/P5B_PAX_CENTROID_DIRECTORY_DESIGN.adoc`).
//!
//! The directory is an S7 projection: derived only from segment bytes,
//! rebuildable at any time, never authoritative. Its **stamp** is the segment
//! table — the exact `(name, size)` set it examined. A reader trusts an entry
//! only for a segment that is still live with the same size; everything else
//! falls back to the per-segment cascade, so a stale or missing directory can
//! change a query's cost, never its correctness.
//!
//! ## Object naming
//!
//! `centroids-<generation:016x>.pxcd` in the collection data directory. The
//! extension is deliberately not `.pax` (discovery treats every `*.pax` as a
//! segment); the LIST that discovery already issues returns the directory too.
//!
//! ## Byte layout (fixed, little-endian, deterministic)
//!
//! ```text
//! [magic "PXCD" 4][version u8][flags u8][reserved u16]
//! [generation u64][dim u32][n_models u32][n_segments u32][n_cells u32]
//! models:   n_models × [model_id u64][n_comp u16]
//!                       [pca_mean dim × f32][pca_components n_comp × dim × f32]
//! segments: n_segments × [name_len u32][name utf-8][size_bytes u64]
//!                         [rows_covered u64][model_idx u32 (u32::MAX = uncovered)]
//! cells:    n_cells × [segment_idx u32][a0_cell u32]
//!                      [centroid n_comp × f32][radius f32][CoarseCellEntry 72 B]
//! [checksum u64]   (FNV-1a over all preceding bytes, as Region A0)
//! ```
//!
//! Segments are sorted by name and cells by `(segment, a0_cell)`, so identical
//! inputs always serialize to identical bytes regardless of discovery order.
//!
//! ## Departure from the TD-PAXDIR-1 §3.1 wire draft
//!
//! The D1 format drops the draft's `built_at_unix_ms` field and the per-table
//! `metric u8`: staleness is carried by the `(name, size)` stamp plus the
//! generation (until TD-PAXMETA-1 adds a snapshot id), and the metric is a
//! collection-level property the caller already holds. `flags` and `reserved`
//! are therefore written as zero and `parse` REJECTS non-zero values there, so
//! a future revision that gives them meaning must bump [`DIRECTORY_VERSION`]
//! rather than silently reinterpret v1 bytes.

#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};

use crate::coarse_directory::{
    CoarseCellEntry, CoarseDirectory, fnv1a64, project_with_model, read_f32s, read_u16, read_u32,
    read_u64,
};

/// Directory magic — first 4 bytes of the serialized object.
pub const DIRECTORY_MAGIC: &[u8; 4] = b"PXCD";

/// Directory serde version (single version pre-GA, as Region A0).
pub const DIRECTORY_VERSION: u8 = 1;

/// Object-name extension. Not `.pax`: discovery lists `*.pax` as segments.
pub const DIRECTORY_EXT: &str = ".pxcd";

const OBJECT_PREFIX: &str = "centroids-";

/// `model_idx` of a segment the directory examined but cannot address (no A0,
/// a non-coalesced layout, or a dimension mismatch). Readers scan such
/// segments with the per-segment path.
pub const NO_MODEL: u32 = u32::MAX;

/// Fixed head: magic(4) + version(1) + flags(1) + reserved(2) + generation(8)
/// + dim(4) + n_models(4) + n_segments(4) + n_cells(4).
const FIXED_HEAD_LEN: usize = 32;

/// Serialized [`CoarseCellEntry`] length (8 × u64 + 2 × u32).
const CELL_ENTRY_LEN: usize = 8 * 8 + 4 * 2;

const CHECKSUM_LEN: usize = 8;

/// Object name for a directory generation.
pub fn object_name(generation: u64) -> String {
    format!("{OBJECT_PREFIX}{generation:016x}{DIRECTORY_EXT}")
}

/// Generation encoded in a directory object name, or `None` if `name` is not
/// one (segments, sidecars and foreign objects all return `None`).
pub fn parse_object_name(name: &str) -> Option<u64> {
    let hex = name
        .strip_prefix(OBJECT_PREFIX)?
        .strip_suffix(DIRECTORY_EXT)?;
    if hex.len() != 16 {
        return None;
    }
    u64::from_str_radix(hex, 16).ok()
}

/// One PCA model shared by the segments that were clustered with it.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectoryModel {
    /// FNV-1a of the model's serialized `(n_comp, pca_mean, pca_components)`;
    /// segments written with the same model share it, and therefore a cell
    /// space.
    pub model_id: u64,
    pub n_comp: u16,
    pub pca_mean: Vec<f32>,
    pub pca_components: Vec<f32>,
}

impl DirectoryModel {
    fn from_a0(a0: &CoarseDirectory) -> Self {
        let m = &a0.model;
        let mut bytes = Vec::with_capacity(2 + 4 * (m.pca_mean.len() + m.pca_components.len()));
        bytes.extend_from_slice(&m.n_comp.to_le_bytes());
        for v in m.pca_mean.iter().chain(&m.pca_components) {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Self {
            model_id: fnv1a64(&bytes),
            n_comp: m.n_comp,
            pca_mean: m.pca_mean.clone(),
            pca_components: m.pca_components.clone(),
        }
    }
}

/// One segment the directory examined — the stamp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectorySegment {
    /// Object name relative to the collection data directory.
    pub name: String,
    /// Object size when examined; a size change means the entry is stale.
    pub size_bytes: u64,
    /// Rows covered by the segment's cells (`0` when uncovered).
    pub rows_covered: u64,
    /// Index into the model table, or [`NO_MODEL`].
    pub model_idx: u32,
}

impl DirectorySegment {
    /// Whether the directory addresses this segment's cells.
    pub fn is_covered(&self) -> bool {
        self.model_idx != NO_MODEL
    }
}

/// One coarse cell of one segment.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectoryCell {
    pub segment_idx: u32,
    /// The cell's index in the segment's own Region A0.
    pub a0_cell: u32,
    /// Centroid in the segment model's PCA space (`n_comp` values).
    pub centroid: Vec<f32>,
    /// Max member→centroid distance in PCA space.
    pub radius: f32,
    /// Row range and absolute byte extents, copied verbatim from A0.
    pub entry: CoarseCellEntry,
}

impl DirectoryCell {
    fn rows(&self) -> u64 {
        self.entry.row_end.saturating_sub(self.entry.row_begin)
    }
}

/// A cell and its global ranking score (squared L2 distance from the query to
/// the centroid's PCA reconstruction in the original space).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredCell {
    pub cell: usize,
    pub score: f32,
}

/// A query's use of the directory: for every usable segment (by object
/// name), the Region A0 cells to probe, in rank order. An empty list means the
/// segment holds none of the globally nearest cells and is skipped. Segments
/// absent from the map are not usable and keep the per-segment path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbePlan {
    pub cells_by_segment: HashMap<String, Vec<u32>>,
    /// Usable non-empty cells the ranking chose from.
    pub usable_cells: usize,
}

/// The parsed collection directory.
#[derive(Debug, Clone, PartialEq)]
pub struct PaxCentroidDirectory {
    pub generation: u64,
    pub dim: u32,
    pub models: Vec<DirectoryModel>,
    pub segments: Vec<DirectorySegment>,
    pub cells: Vec<DirectoryCell>,
}

impl PaxCentroidDirectory {
    /// Serialize (deterministic).
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let dim = self.dim as usize;
        let mut buf = Vec::new();
        buf.extend_from_slice(DIRECTORY_MAGIC);
        buf.push(DIRECTORY_VERSION);
        buf.push(0); // flags (reserved; D6 decode keys, TD-PAXMETA-1 snapshot id)
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&self.generation.to_le_bytes());
        buf.extend_from_slice(&self.dim.to_le_bytes());
        for count in [self.models.len(), self.segments.len(), self.cells.len()] {
            let count = u32::try_from(count)
                .map_err(|_| anyhow::anyhow!("centroid directory table exceeds u32"))?;
            buf.extend_from_slice(&count.to_le_bytes());
        }
        for model in &self.models {
            buf.extend_from_slice(&model.model_id.to_le_bytes());
            buf.extend_from_slice(&model.n_comp.to_le_bytes());
            debug_assert_eq!(model.pca_mean.len(), dim);
            push_f32s(&mut buf, &model.pca_mean);
            push_f32s(&mut buf, &model.pca_components);
        }
        for segment in &self.segments {
            let name_len = u32::try_from(segment.name.len())
                .map_err(|_| anyhow::anyhow!("segment name exceeds u32"))?;
            buf.extend_from_slice(&name_len.to_le_bytes());
            buf.extend_from_slice(segment.name.as_bytes());
            buf.extend_from_slice(&segment.size_bytes.to_le_bytes());
            buf.extend_from_slice(&segment.rows_covered.to_le_bytes());
            buf.extend_from_slice(&segment.model_idx.to_le_bytes());
        }
        for cell in &self.cells {
            buf.extend_from_slice(&cell.segment_idx.to_le_bytes());
            buf.extend_from_slice(&cell.a0_cell.to_le_bytes());
            push_f32s(&mut buf, &cell.centroid);
            buf.extend_from_slice(&cell.radius.to_le_bytes());
            cell.entry.write_to(&mut buf);
        }
        let checksum = fnv1a64(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        Ok(buf)
    }

    /// Parse and verify. Fail-closed on magic, version, truncation, checksum
    /// or any structural inconsistency; the caller treats an error as "no
    /// directory" and uses the per-segment path.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < FIXED_HEAD_LEN + CHECKSUM_LEN {
            bail!("centroid directory too short: {}", bytes.len());
        }
        if &bytes[..4] != DIRECTORY_MAGIC {
            bail!("not a centroid directory (bad magic)");
        }
        if bytes[4] != DIRECTORY_VERSION {
            bail!("unsupported centroid directory version {}", bytes[4]);
        }
        // v1 assigns no meaning to flags or reserved: reject non-zero values
        // so a later revision must bump the version instead of silently
        // reinterpreting v1 bytes.
        if bytes[5] != 0 {
            bail!(
                "unknown centroid directory flags 0x{:02x} (version {})",
                bytes[5],
                bytes[4]
            );
        }
        if bytes[6] != 0 || bytes[7] != 0 {
            bail!("centroid directory reserved bytes must be zero");
        }
        let body_len = bytes.len() - CHECKSUM_LEN;
        let stored = u64::from_le_bytes(bytes[body_len..].try_into()?);
        if stored != fnv1a64(&bytes[..body_len]) {
            bail!("centroid directory checksum mismatch");
        }
        let body = &bytes[..body_len];
        let mut p = 8usize; // magic + version + flags + reserved
        let generation = read_u64(body, &mut p)?;
        let dim = read_u32(body, &mut p)?;
        let n_models = read_u32(body, &mut p)? as usize;
        let n_segments = read_u32(body, &mut p)? as usize;
        let n_cells = read_u32(body, &mut p)? as usize;
        let d = dim as usize;

        // Every count is bounded by the remaining bytes before allocating.
        let mut models = Vec::with_capacity(n_models.min(body.len() / 10));
        for _ in 0..n_models {
            let model_id = read_u64(body, &mut p)?;
            let n_comp = read_u16(body, &mut p)?;
            let pca_mean = read_f32s(body, &mut p, d)?;
            let pca_components = read_f32s(body, &mut p, n_comp as usize * d)?;
            models.push(DirectoryModel {
                model_id,
                n_comp,
                pca_mean,
                pca_components,
            });
        }
        let mut segments = Vec::with_capacity(n_segments.min(body.len() / 24));
        for _ in 0..n_segments {
            let name_len = read_u32(body, &mut p)? as usize;
            let end = p
                .checked_add(name_len)
                .filter(|end| *end <= body.len())
                .ok_or_else(|| anyhow::anyhow!("centroid directory truncated (segment name)"))?;
            let name = std::str::from_utf8(&body[p..end])
                .map_err(|_| anyhow::anyhow!("centroid directory segment name is not UTF-8"))?
                .to_owned();
            p = end;
            segments.push(DirectorySegment {
                name,
                size_bytes: read_u64(body, &mut p)?,
                rows_covered: read_u64(body, &mut p)?,
                model_idx: read_u32(body, &mut p)?,
            });
        }
        let mut cells = Vec::with_capacity(n_cells.min(body.len() / (8 + CELL_ENTRY_LEN)));
        for _ in 0..n_cells {
            let segment_idx = read_u32(body, &mut p)?;
            let a0_cell = read_u32(body, &mut p)?;
            let n_comp = segments
                .get(segment_idx as usize)
                .and_then(|s| models.get(s.model_idx as usize))
                .map(|m| m.n_comp as usize)
                .ok_or_else(|| anyhow::anyhow!("centroid directory cell has no model"))?;
            let centroid = read_f32s(body, &mut p, n_comp)?;
            let radius = f32::from_le_bytes(read_u32(body, &mut p)?.to_le_bytes());
            let entry = CoarseCellEntry::read_from(body, &mut p)?;
            cells.push(DirectoryCell {
                segment_idx,
                a0_cell,
                centroid,
                radius,
                entry,
            });
        }
        if p != body.len() {
            bail!("centroid directory has {} trailing bytes", body.len() - p);
        }
        let dir = Self {
            generation,
            dim,
            models,
            segments,
            cells,
        };
        dir.validate()?;
        Ok(dir)
    }

    /// Structural invariants shared by serialize and parse.
    fn validate(&self) -> Result<()> {
        let d = self.dim as usize;
        for model in &self.models {
            if model.n_comp == 0 || u32::from(model.n_comp) > self.dim {
                bail!(
                    "centroid directory model has invalid n_comp {}",
                    model.n_comp
                );
            }
            if model.pca_mean.len() != d || model.pca_components.len() != model.n_comp as usize * d
            {
                bail!("centroid directory model length mismatch");
            }
            // A single non-finite model value makes every cell score of the
            // model non-finite, which would let one poisoned segment occupy
            // all nprobe slots (fail closed instead).
            if model.pca_mean.iter().any(|v| !v.is_finite())
                || model.pca_components.iter().any(|v| !v.is_finite())
            {
                bail!("centroid directory model has a non-finite value");
            }
        }
        let mut names = HashSet::with_capacity(self.segments.len());
        for segment in &self.segments {
            if !names.insert(segment.name.as_str()) {
                bail!("centroid directory lists segment {} twice", segment.name);
            }
            if segment.is_covered() && segment.model_idx as usize >= self.models.len() {
                bail!("centroid directory segment {} has no model", segment.name);
            }
        }
        for cell in &self.cells {
            let segment = self
                .segments
                .get(cell.segment_idx as usize)
                .filter(|s| s.is_covered())
                .ok_or_else(|| anyhow::anyhow!("centroid directory cell on uncovered segment"))?;
            if cell.centroid.len() != self.models[segment.model_idx as usize].n_comp as usize {
                bail!("centroid directory cell centroid length mismatch");
            }
            if cell.centroid.iter().any(|v| !v.is_finite()) || !cell.radius.is_finite() {
                bail!("centroid directory cell has a non-finite value");
            }
            if cell.entry.row_end < cell.entry.row_begin {
                bail!("centroid directory cell has an inverted row range");
            }
        }
        Ok(())
    }

    /// Covered segments by name — the entries a reader may use, provided the
    /// live object still has the recorded size.
    pub fn covered_segments(&self) -> impl Iterator<Item = (usize, &DirectorySegment)> {
        self.segments
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_covered())
    }

    /// Rank every cell across every model by the squared distance from
    /// `query` to the centroid's PCA reconstruction `μ + Pᵀc`:
    ///
    /// `‖P(q − μ) − c‖² + (‖q − μ‖² − ‖P(q − μ)‖²)`
    ///
    /// The first term is the in-segment A0 distance; the second is one value
    /// per model, so within one model the order is exactly A0's order, and
    /// across models the scores are commensurable. Ties break on cell index.
    /// `None` when the query's dimension does not match.
    pub fn rank_cells(&self, query: &[f32]) -> Option<Vec<ScoredCell>> {
        if query.len() != self.dim as usize {
            return None;
        }
        let projected: Vec<(Vec<f32>, f32)> = self
            .models
            .iter()
            .map(|m| {
                let proj =
                    project_with_model(&m.pca_mean, &m.pca_components, m.n_comp as usize, query);
                let centered: f64 = query
                    .iter()
                    .zip(&m.pca_mean)
                    .map(|(q, mu)| {
                        let v = *q as f64 - *mu as f64;
                        v * v
                    })
                    .sum();
                let explained: f64 = proj.iter().map(|v| (*v as f64) * (*v as f64)).sum();
                (proj, (centered - explained).max(0.0) as f32)
            })
            .collect();
        let mut scored: Vec<ScoredCell> = self
            .cells
            .iter()
            .enumerate()
            .map(|(cell, c)| {
                let model = self.segments[c.segment_idx as usize].model_idx as usize;
                let (proj, residual) = &projected[model];
                let d: f32 = proj
                    .iter()
                    .zip(&c.centroid)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                // A non-finite score must never win a nprobe slot: total_cmp
                // ranks -NaN first, so clamp it to +inf (sorts last) instead.
                // v1 files are rejected at parse for non-finite values; this
                // guards directories written before that check existed.
                let score = d + residual;
                ScoredCell {
                    cell,
                    score: if score.is_finite() {
                        score
                    } else {
                        f32::INFINITY
                    },
                }
            })
            .collect();
        scored.sort_by(|a, b| a.score.total_cmp(&b.score).then(a.cell.cmp(&b.cell)));
        Some(scored)
    }

    /// Nearest non-empty cells, in rank order, until at least `nprobe` cells
    /// **and** at least `min_rows` rows are selected (the survivor-pool floor
    /// that keeps recall when cells are small). `None` on a dimension
    /// mismatch.
    pub fn select_cells(&self, query: &[f32], nprobe: usize, min_rows: u64) -> Option<Vec<usize>> {
        self.select_usable_cells(query, nprobe, min_rows, |_| true)
    }

    /// [`Self::select_cells`] restricted to cells of segments `usable` accepts.
    fn select_usable_cells(
        &self,
        query: &[f32],
        nprobe: usize,
        min_rows: u64,
        usable: impl Fn(u32) -> bool,
    ) -> Option<Vec<usize>> {
        let ranked = self.rank_cells(query)?;
        let mut selected = Vec::new();
        let mut rows = 0u64;
        for scored in ranked {
            if selected.len() >= nprobe && rows >= min_rows {
                break;
            }
            let cell = &self.cells[scored.cell];
            let cell_rows = cell.rows();
            if cell_rows == 0 || !usable(cell.segment_idx) {
                continue;
            }
            selected.push(scored.cell);
            rows += cell_rows;
        }
        Some(selected)
    }

    /// Plan a query against the live segment set (TD-PAXDIR-1 D2).
    ///
    /// Only entries for segments that are live **with the recorded size** are
    /// usable; every other live segment is absent from the plan and the
    /// caller scans it per segment. `nprobe` and `min_rows` are computed from
    /// the usable totals (cells, covered rows), so a partly stale directory
    /// probes proportionally. `None` on a dimension mismatch or when nothing
    /// is usable.
    pub fn plan_probe(
        &self,
        query: &[f32],
        live: &[LiveSegment],
        nprobe_for_cells: impl Fn(usize) -> usize,
        min_rows_for_rows: impl Fn(u64) -> u64,
    ) -> Option<ProbePlan> {
        let live_sizes: HashMap<&str, u64> = live
            .iter()
            .map(|s| (s.name.as_str(), s.size_bytes))
            .collect();
        let usable: Vec<bool> = self
            .segments
            .iter()
            .map(|s| s.is_covered() && live_sizes.get(s.name.as_str()) == Some(&s.size_bytes))
            .collect();
        let usable_cells = self
            .cells
            .iter()
            .filter(|c| usable[c.segment_idx as usize] && c.rows() > 0)
            .count();
        if usable_cells == 0 {
            return None;
        }
        let usable_rows: u64 = self
            .segments
            .iter()
            .zip(&usable)
            .filter(|(_, u)| **u)
            .map(|(s, _)| s.rows_covered)
            .sum();
        let selected = self.select_usable_cells(
            query,
            nprobe_for_cells(usable_cells),
            min_rows_for_rows(usable_rows),
            |segment_idx| usable[segment_idx as usize],
        )?;
        let mut cells_by_segment: HashMap<String, Vec<u32>> = self
            .segments
            .iter()
            .zip(&usable)
            .filter(|(_, u)| **u)
            .map(|(s, _)| (s.name.clone(), Vec::new()))
            .collect();
        for cell in selected {
            let cell = &self.cells[cell];
            if let Some(cells) =
                cells_by_segment.get_mut(&self.segments[cell.segment_idx as usize].name)
            {
                cells.push(cell.a0_cell);
            }
        }
        Some(ProbePlan {
            cells_by_segment,
            usable_cells,
        })
    }

    /// Start a builder that keeps the segments `keep` accepts, carrying their
    /// models and cells over without re-reading any segment.
    pub fn retain_into_builder(
        &self,
        mut keep: impl FnMut(&DirectorySegment) -> bool,
    ) -> DirectoryBuilder {
        let mut builder = DirectoryBuilder::default();
        let mut kept: HashMap<u32, PendingSegment> = HashMap::new();
        for (idx, segment) in self.segments.iter().enumerate() {
            if !keep(segment) {
                continue;
            }
            let model = segment
                .is_covered()
                .then(|| self.models[segment.model_idx as usize].clone());
            kept.insert(
                idx as u32,
                PendingSegment {
                    name: segment.name.clone(),
                    size_bytes: segment.size_bytes,
                    rows_covered: segment.rows_covered,
                    model,
                    cells: Vec::new(),
                },
            );
        }
        for cell in &self.cells {
            if let Some(pending) = kept.get_mut(&cell.segment_idx) {
                pending.cells.push(cell.clone());
            }
        }
        // Inherit the base dimension only from a kept segment that is actually
        // covered (carries a model). A base directory of uncovered segments
        // legitimately has dim 0; propagating that 0 would make every later
        // add_covered bail on "dim differs from 0" and silently turn coverage
        // off for the rest of the collection's life.
        if builder.dim.is_none() && kept.values().any(|p| p.model.is_some()) {
            builder.dim = Some(self.dim);
        }
        for (_, pending) in kept {
            builder.pending.insert(pending.name.clone(), pending);
        }
        builder
    }
}

fn push_f32s(buf: &mut Vec<u8>, values: &[f32]) {
    for v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

#[derive(Debug, Clone)]
struct PendingSegment {
    name: String,
    size_bytes: u64,
    rows_covered: u64,
    model: Option<DirectoryModel>,
    cells: Vec<DirectoryCell>,
}

/// Accumulates segments into a directory. Order-independent: [`Self::build`]
/// sorts segments by name and deduplicates models by `model_id`.
#[derive(Debug, Clone, Default)]
pub struct DirectoryBuilder {
    dim: Option<u32>,
    pending: HashMap<String, PendingSegment>,
}

impl DirectoryBuilder {
    /// Add a segment whose Region A0 the directory will address. Fails on a
    /// dimension that disagrees with the segments already added; the caller
    /// then records the segment with [`Self::add_uncovered`].
    pub fn add_covered(&mut self, name: &str, size_bytes: u64, a0: &CoarseDirectory) -> Result<()> {
        let dim = a0.model.dim;
        if let Some(expected) = self.dim
            && expected != dim
        {
            bail!("segment {name}: dim {dim} differs from the directory's {expected}");
        }
        if a0.cells.len() != a0.model.k_c() {
            bail!("segment {name}: A0 cell table disagrees with k_c");
        }
        let n_comp = a0.model.n_comp as usize;
        let cells = a0
            .cells
            .iter()
            .enumerate()
            .map(|(c, entry)| DirectoryCell {
                segment_idx: 0, // assigned in build()
                a0_cell: c as u32,
                centroid: a0.model.centroids[c * n_comp..(c + 1) * n_comp].to_vec(),
                radius: a0.model.radii[c],
                entry: *entry,
            })
            .collect();
        self.dim = Some(dim);
        self.pending.insert(
            name.to_owned(),
            PendingSegment {
                name: name.to_owned(),
                size_bytes,
                rows_covered: a0.model.rows_covered(),
                model: Some(DirectoryModel::from_a0(a0)),
                cells,
            },
        );
        Ok(())
    }

    /// Record a segment the directory examined but cannot address, so later
    /// publishes do not re-read it and readers know to scan it per segment.
    pub fn add_uncovered(&mut self, name: &str, size_bytes: u64) {
        self.pending.insert(
            name.to_owned(),
            PendingSegment {
                name: name.to_owned(),
                size_bytes,
                rows_covered: 0,
                model: None,
                cells: Vec::new(),
            },
        );
    }

    /// Whether any segment has been added.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Assemble the directory for `generation`.
    pub fn build(self, generation: u64) -> Result<PaxCentroidDirectory> {
        let mut pending: Vec<PendingSegment> = self.pending.into_values().collect();
        pending.sort_by(|a, b| a.name.cmp(&b.name));
        let mut models: Vec<DirectoryModel> = Vec::new();
        let mut model_index: HashMap<u64, u32> = HashMap::new();
        let mut segments = Vec::with_capacity(pending.len());
        let mut cells = Vec::new();
        for (segment_idx, segment) in pending.into_iter().enumerate() {
            let model_idx = match segment.model {
                Some(model) => *model_index.entry(model.model_id).or_insert_with(|| {
                    models.push(model);
                    (models.len() - 1) as u32
                }),
                None => NO_MODEL,
            };
            let mut segment_cells = segment.cells;
            segment_cells.sort_by_key(|c| c.a0_cell);
            for mut cell in segment_cells {
                cell.segment_idx = segment_idx as u32;
                cells.push(cell);
            }
            segments.push(DirectorySegment {
                name: segment.name,
                size_bytes: segment.size_bytes,
                rows_covered: segment.rows_covered,
                model_idx,
            });
        }
        let dir = PaxCentroidDirectory {
            generation,
            dim: self.dim.unwrap_or(0),
            models,
            segments,
            cells,
        };
        dir.validate()?;
        Ok(dir)
    }
}

/// A live segment as discovery sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveSegment {
    pub name: String,
    pub size_bytes: u64,
}

/// What a publish must do to bring a directory up to date with the live set.
#[derive(Debug)]
pub struct ReconcilePlan {
    /// Base entries still valid, carried over.
    pub builder: DirectoryBuilder,
    /// Live segments the base does not describe (new, or changed size); the
    /// publisher reads their A0 and adds them.
    pub to_load: Vec<LiveSegment>,
    /// Whether the result differs from the base (a new generation is due).
    pub changed: bool,
}

/// Plan an incremental rebuild: keep base entries for segments that are still
/// live with the same size, load the rest. With no base this is a full build.
pub fn plan_reconcile(base: Option<&PaxCentroidDirectory>, live: &[LiveSegment]) -> ReconcilePlan {
    let live_sizes: HashMap<&str, u64> = live
        .iter()
        .map(|s| (s.name.as_str(), s.size_bytes))
        .collect();
    let Some(base) = base else {
        return ReconcilePlan {
            builder: DirectoryBuilder::default(),
            to_load: live.to_vec(),
            changed: true,
        };
    };
    let mut kept = HashSet::new();
    let mut dropped = false;
    let builder = base.retain_into_builder(|segment| {
        let keep = live_sizes.get(segment.name.as_str()) == Some(&segment.size_bytes);
        if keep {
            kept.insert(segment.name.clone());
        } else {
            dropped = true;
        }
        keep
    });
    let to_load: Vec<LiveSegment> = live
        .iter()
        .filter(|s| !kept.contains(&s.name))
        .cloned()
        .collect();
    let changed = dropped || !to_load.is_empty();
    ReconcilePlan {
        builder,
        to_load,
        changed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coarse_directory::CoarseModel;

    /// A0 for a segment whose PCA keeps the first `n_comp` axes of `dim`
    /// (orthonormal components) with `mean`, and cells at `centroids`.
    fn a0(
        dim: usize,
        n_comp: usize,
        mean: f32,
        centroids: &[&[f32]],
        rows: &[u64],
    ) -> CoarseDirectory {
        let mut components = vec![0f32; n_comp * dim];
        for c in 0..n_comp {
            components[c * dim + c] = 1.0;
        }
        let k_c = centroids.len();
        let mut cells = Vec::with_capacity(k_c);
        let mut row = 0u64;
        for (c, &n) in rows.iter().enumerate() {
            cells.push(CoarseCellEntry {
                row_begin: row,
                row_end: row + n,
                a_off: 1000 + c as u64 * 64,
                a_len: n * 8,
                b_off: 9000 + c as u64 * 64,
                b_len: n * dim as u64,
                c_off: 0,
                c_len: 0,
                d_block_begin: c as u32,
                d_block_end: c as u32 + 1,
            });
            row += n;
        }
        CoarseDirectory {
            model: CoarseModel {
                dim: dim as u32,
                n_comp: n_comp as u16,
                pca_mean: vec![mean; dim],
                pca_components: components,
                centroids: centroids.iter().flat_map(|c| c.iter().copied()).collect(),
                radii: vec![0.5; k_c],
                cell_rows: rows.to_vec(),
                seed: 7,
                trained_on: rows.iter().sum(),
            },
            cells,
        }
    }

    fn two_model_directory() -> PaxCentroidDirectory {
        let mut b = DirectoryBuilder::default();
        b.add_covered(
            "b.pax",
            200,
            &a0(4, 2, 0.0, &[&[0.0, 0.0], &[5.0, 5.0]], &[10, 20]),
        )
        .unwrap();
        b.add_covered(
            "a.pax",
            100,
            &a0(
                4,
                2,
                1.0,
                &[&[1.0, -1.0], &[-3.0, 2.0], &[0.0, 0.0]],
                &[5, 0, 7],
            ),
        )
        .unwrap();
        b.add_uncovered("l0.pax", 50);
        b.build(3).unwrap()
    }

    #[test]
    fn object_names_round_trip_and_reject_other_objects() {
        assert_eq!(object_name(0x2a), "centroids-000000000000002a.pxcd");
        assert_eq!(parse_object_name(&object_name(u64::MAX)), Some(u64::MAX));
        for other in [
            "centroids-2a.pxcd",
            "centroids-000000000000002a.pax",
            "seg-000000000000002a.pxcd",
            "retire-pending.json",
            "centroids-00000000000000zz.pxcd",
        ] {
            assert_eq!(parse_object_name(other), None, "{other}");
        }
    }

    #[test]
    fn serialization_round_trips_and_is_order_independent() {
        let dir = two_model_directory();
        let bytes = dir.to_bytes().unwrap();
        assert_eq!(PaxCentroidDirectory::parse(&bytes).unwrap(), dir);

        let mut b = DirectoryBuilder::default();
        b.add_uncovered("l0.pax", 50);
        b.add_covered(
            "a.pax",
            100,
            &a0(
                4,
                2,
                1.0,
                &[&[1.0, -1.0], &[-3.0, 2.0], &[0.0, 0.0]],
                &[5, 0, 7],
            ),
        )
        .unwrap();
        b.add_covered(
            "b.pax",
            200,
            &a0(4, 2, 0.0, &[&[0.0, 0.0], &[5.0, 5.0]], &[10, 20]),
        )
        .unwrap();
        assert_eq!(b.build(3).unwrap().to_bytes().unwrap(), bytes);

        assert_eq!(
            dir.segments
                .iter()
                .map(|s| s.name.as_str())
                .collect::<Vec<_>>(),
            ["a.pax", "b.pax", "l0.pax"]
        );
        assert_eq!(dir.models.len(), 2);
        assert!(!dir.segments[2].is_covered());
        assert_eq!(dir.cells.len(), 5);
        assert_eq!(dir.segments[0].rows_covered, 12);
    }

    #[test]
    fn segments_sharing_a_model_share_one_model_entry() {
        let mut b = DirectoryBuilder::default();
        b.add_covered("a.pax", 1, &a0(4, 2, 0.0, &[&[0.0, 0.0]], &[3]))
            .unwrap();
        b.add_covered("b.pax", 1, &a0(4, 2, 0.0, &[&[1.0, 1.0]], &[4]))
            .unwrap();
        let dir = b.build(1).unwrap();
        assert_eq!(dir.models.len(), 1);
        assert!(dir.segments.iter().all(|s| s.model_idx == 0));
    }

    #[test]
    fn parse_fails_closed_on_corruption() {
        let bytes = two_model_directory().to_bytes().unwrap();
        for i in [0usize, 4, 9, bytes.len() / 2, bytes.len() - 1] {
            let mut bad = bytes.clone();
            bad[i] ^= 0x40;
            assert!(PaxCentroidDirectory::parse(&bad).is_err(), "flip at {i}");
        }
        for len in [0, 10, bytes.len() - 1] {
            assert!(
                PaxCentroidDirectory::parse(&bytes[..len]).is_err(),
                "len {len}"
            );
        }
    }

    #[test]
    fn mismatched_dimension_is_rejected_for_coverage() {
        let mut b = DirectoryBuilder::default();
        b.add_covered("a.pax", 1, &a0(4, 2, 0.0, &[&[0.0, 0.0]], &[3]))
            .unwrap();
        assert!(
            b.add_covered("b.pax", 1, &a0(6, 2, 0.0, &[&[0.0, 0.0]], &[3]))
                .is_err()
        );
    }

    /// The score is the squared distance to the centroid's reconstruction.
    #[test]
    fn score_is_the_full_space_distance_to_the_reconstructed_centroid() {
        let dir = two_model_directory();
        let query = [0.3f32, -2.0, 4.0, 1.5];
        for scored in dir.rank_cells(&query).unwrap() {
            let cell = &dir.cells[scored.cell];
            let model = &dir.models[dir.segments[cell.segment_idx as usize].model_idx as usize];
            // μ + Pᵀc with P = the first n_comp axes.
            let mut recon = model.pca_mean.clone();
            for (c, v) in cell.centroid.iter().enumerate() {
                recon[c] += v;
            }
            let expected: f32 = query
                .iter()
                .zip(&recon)
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            assert!(
                (scored.score - expected).abs() < 1e-4,
                "{scored:?} vs {expected}"
            );
        }
    }

    #[test]
    fn within_one_model_rank_order_equals_a0_order() {
        let seg = a0(
            4,
            2,
            0.5,
            &[&[0.0, 0.0], &[2.0, 2.0], &[-1.0, 3.0], &[4.0, -4.0]],
            &[1, 1, 1, 1],
        );
        let mut b = DirectoryBuilder::default();
        b.add_covered("s.pax", 1, &seg).unwrap();
        let dir = b.build(1).unwrap();
        let query = [1.7f32, 0.2, -9.0, 3.0];
        let q = seg.project(&query);
        let mut a0_order: Vec<usize> = (0..4).collect();
        let a0_dist = |c: usize| -> f32 {
            q.iter()
                .zip(&seg.model.centroids[c * 2..c * 2 + 2])
                .map(|(a, b)| (a - b) * (a - b))
                .sum()
        };
        a0_order.sort_by(|a, b| a0_dist(*a).total_cmp(&a0_dist(*b)));
        let ranked: Vec<usize> = dir
            .rank_cells(&query)
            .unwrap()
            .iter()
            .map(|s| dir.cells[s.cell].a0_cell as usize)
            .collect();
        assert_eq!(ranked, a0_order);
    }

    #[test]
    fn selection_honours_nprobe_and_the_row_floor_and_skips_empty_cells() {
        let dir = two_model_directory();
        let query = [1.0f32, -1.0, 1.0, 1.0];
        assert_eq!(dir.rank_cells(&[0.0; 3]), None);
        let one = dir.select_cells(&query, 1, 0).unwrap();
        assert_eq!(one.len(), 1);
        let all = dir.select_cells(&query, 1, u64::MAX).unwrap();
        // Every non-empty cell, never the empty one.
        assert_eq!(all.len(), 4);
        assert!(all.iter().all(|&c| dir.cells[c].rows() > 0));
        let floor = dir.select_cells(&query, 1, 13).unwrap();
        assert!(floor.iter().map(|&c| dir.cells[c].rows()).sum::<u64>() >= 13);
    }

    #[test]
    fn reconcile_keeps_unchanged_segments_and_reloads_the_rest() {
        let base = two_model_directory();
        let live = |items: &[(&str, u64)]| -> Vec<LiveSegment> {
            items
                .iter()
                .map(|(n, s)| LiveSegment {
                    name: (*n).to_owned(),
                    size_bytes: *s,
                })
                .collect()
        };

        let same = plan_reconcile(
            Some(&base),
            &live(&[("a.pax", 100), ("b.pax", 200), ("l0.pax", 50)]),
        );
        assert!(!same.changed);
        assert!(same.to_load.is_empty());
        assert_eq!(same.builder.build(base.generation).unwrap(), base);

        // b.pax retired, a.pax rewritten in place (size change), c.pax new.
        let plan = plan_reconcile(
            Some(&base),
            &live(&[("a.pax", 101), ("c.pax", 9), ("l0.pax", 50)]),
        );
        assert!(plan.changed);
        let names: Vec<&str> = plan.to_load.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["a.pax", "c.pax"]);
        let rebuilt = plan.builder.build(4).unwrap();
        assert_eq!(
            rebuilt
                .segments
                .iter()
                .map(|s| s.name.as_str())
                .collect::<Vec<_>>(),
            ["l0.pax"]
        );

        // Retirement alone is a change.
        let shrink = plan_reconcile(Some(&base), &live(&[("a.pax", 100)]));
        assert!(shrink.changed && shrink.to_load.is_empty());
        let full = plan_reconcile(None, &live(&[("a.pax", 100)]));
        assert!(full.changed && full.to_load.len() == 1);
    }

    /// Incremental reconcile produces exactly the directory a full build does.
    #[test]
    fn incremental_build_equals_full_build() {
        let seg_c = a0(4, 3, 0.0, &[&[1.0, 2.0, 3.0]], &[9]);
        let base = two_model_directory();
        let live = vec![
            LiveSegment {
                name: "a.pax".into(),
                size_bytes: 100,
            },
            LiveSegment {
                name: "c.pax".into(),
                size_bytes: 9,
            },
        ];
        let mut plan = plan_reconcile(Some(&base), &live);
        assert_eq!(plan.to_load.len(), 1);
        plan.builder.add_covered("c.pax", 9, &seg_c).unwrap();
        let incremental = plan.builder.build(4).unwrap();

        let mut full = DirectoryBuilder::default();
        // `a.pax` in the base was built from this A0's first model/cells.
        full.add_covered(
            "a.pax",
            100,
            &a0(
                4,
                2,
                1.0,
                &[&[1.0, -1.0], &[-3.0, 2.0], &[0.0, 0.0]],
                &[5, 0, 7],
            ),
        )
        .unwrap();
        full.add_covered("c.pax", 9, &seg_c).unwrap();
        assert_eq!(
            incremental.to_bytes().unwrap(),
            full.build(4).unwrap().to_bytes().unwrap()
        );
    }

    #[test]
    fn retain_from_uncovered_base_leaves_dim_unset() {
        // A base directory of uncovered-only segments has dim 0; retaining it
        // must not pin the builder's dim to 0 (which would make every later
        // add_covered bail and silently turn coverage off for good).
        let mut base_builder = DirectoryBuilder::default();
        base_builder.add_uncovered("l0.pax", 50);
        let base = base_builder.build(1).unwrap();
        assert_eq!(base.dim, 0);
        let mut builder = base.retain_into_builder(|_| true);
        assert!(
            builder.dim.is_none(),
            "uncovered-only base must not set the builder's dim"
        );
        builder
            .add_covered(
                "c.pax",
                9,
                &a0(4, 2, 0.0, &[&[0.0, 0.0], &[5.0, 5.0]], &[10, 20]),
            )
            .expect("covered segment after uncovered-only base must be accepted");
        let rebuilt = builder.build(4).unwrap();
        assert_eq!(rebuilt.dim, 4);
        assert_eq!(rebuilt.covered_segments().count(), 1);
    }

    #[test]
    fn parse_rejects_non_finite_model_values() {
        let dir = two_model_directory();
        let mut bytes = dir.to_bytes().unwrap();
        // Flip one pca_mean f32 inside the first model to a bit pattern for
        // NaN, then re-stamp the checksum so ONLY the finite check can catch
        // it (exercises a valid checksum over structurally bad values).
        let model_mean_off = FIXED_HEAD_LEN
            + 8 /* model_id */
            + 2 /* n_comp */;
        let f = f32::NAN.to_le_bytes();
        for (i, b) in f.iter().enumerate() {
            bytes[model_mean_off + i] = *b;
        }
        let body_len = bytes.len() - CHECKSUM_LEN;
        let sum = fnv1a64(&bytes[..body_len]);
        bytes[body_len..].copy_from_slice(&sum.to_le_bytes());
        let err = PaxCentroidDirectory::parse(&bytes).unwrap_err().to_string();
        assert!(
            err.contains("non-finite"),
            "expected the finite check to fire, got: {err}"
        );
    }

    #[test]
    fn parse_rejects_unknown_flags_and_nonzero_reserved() {
        let bytes = two_model_directory().to_bytes().unwrap();

        let mut flagged = bytes.clone();
        flagged[5] = 0b0000_0001;
        let body_len = flagged.len() - CHECKSUM_LEN;
        let sum = fnv1a64(&flagged[..body_len]);
        flagged[body_len..].copy_from_slice(&sum.to_le_bytes());
        let err = PaxCentroidDirectory::parse(&flagged)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("unknown centroid directory flags"),
            "expected the flags check to fire, got: {err}"
        );

        let mut reserved = bytes;
        reserved[6] = 0x7f;
        let sum = fnv1a64(&reserved[..body_len]);
        reserved[body_len..].copy_from_slice(&sum.to_le_bytes());
        let err = PaxCentroidDirectory::parse(&reserved)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("reserved bytes must be zero"),
            "expected the reserved check to fire, got: {err}"
        );
    }

    #[test]
    fn rank_cells_clamps_non_finite_scores_to_infinity() {
        // A directory written before the parse-time finite check (or poisoned
        // in memory) must not let its NaN-scored cells win nprobe slots:
        // total_cmp ranks -NaN FIRST, so the score is clamped to +inf.
        // Assembled as a struct literal because build()/parse() now reject
        // non-finite values outright.
        let identity_2x4 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let models = vec![
            DirectoryModel {
                model_id: 1,
                n_comp: 2,
                pca_mean: vec![f32::NAN, 0.0, 0.0, 0.0],
                pca_components: identity_2x4.clone(),
            },
            DirectoryModel {
                model_id: 2,
                n_comp: 2,
                pca_mean: vec![0.0; 4],
                pca_components: identity_2x4,
            },
        ];
        let segments = vec![
            DirectorySegment {
                name: "bad.pax".to_string(),
                size_bytes: 100,
                rows_covered: 10,
                model_idx: 0,
            },
            DirectorySegment {
                name: "good.pax".to_string(),
                size_bytes: 100,
                rows_covered: 10,
                model_idx: 1,
            },
        ];
        let entry = CoarseCellEntry {
            row_begin: 0,
            row_end: 10,
            a_off: 0,
            a_len: 80,
            b_off: 0,
            b_len: 320,
            c_off: 0,
            c_len: 0,
            d_block_begin: 0,
            d_block_end: 1,
        };
        let cells = vec![
            DirectoryCell {
                segment_idx: 0,
                a0_cell: 0,
                centroid: vec![0.0, 0.0],
                radius: 0.5,
                entry,
            },
            DirectoryCell {
                segment_idx: 1,
                a0_cell: 0,
                centroid: vec![0.0, 0.0],
                radius: 0.5,
                entry,
            },
        ];
        let poisoned = PaxCentroidDirectory {
            generation: 7,
            dim: 4,
            models,
            segments,
            cells,
        };

        let ranked = poisoned
            .rank_cells(&[0.0, 0.0, 0.0, 0.0])
            .expect("dimension matches");
        assert_eq!(
            ranked.last().expect("cells ranked").score,
            f32::INFINITY,
            "the NaN-model cell must be clamped to +inf"
        );
        assert!(
            ranked[..ranked.len() - 1]
                .iter()
                .all(|s| s.score.is_finite()),
            "only the poisoned cell may sit at +inf"
        );
    }

    #[test]
    fn a_probe_plan_uses_only_live_unchanged_segments() {
        let dir = two_model_directory();
        let query = [1.0f32, -1.0, 1.0, 1.0];
        let live = |items: &[(&str, u64)]| -> Vec<LiveSegment> {
            items
                .iter()
                .map(|(n, s)| LiveSegment {
                    name: (*n).to_owned(),
                    size_bytes: *s,
                })
                .collect()
        };
        // Every usable segment appears; the uncovered one never does.
        let all = dir
            .plan_probe(
                &query,
                &live(&[("a.pax", 100), ("b.pax", 200), ("l0.pax", 50)]),
                |_| 99,
                |_| 0,
            )
            .unwrap();
        assert_eq!(all.usable_cells, 4);
        assert_eq!(all.cells_by_segment.len(), 2);
        assert_eq!(
            all.cells_by_segment["a.pax"].len(),
            2,
            "the empty A0 cell is never probed"
        );
        assert_eq!(all.cells_by_segment["b.pax"].len(), 2);

        // One probe: one segment gets the nearest cell, the other is skipped.
        let one = dir
            .plan_probe(
                &query,
                &live(&[("a.pax", 100), ("b.pax", 200)]),
                |_| 1,
                |_| 0,
            )
            .unwrap();
        let mut sizes: Vec<usize> = one.cells_by_segment.values().map(Vec::len).collect();
        sizes.sort();
        assert_eq!(sizes, [0, 1]);

        // A rewritten (size changed) segment is unusable; a retired one too.
        let stale = dir.plan_probe(
            &query,
            &live(&[("a.pax", 101), ("c.pax", 9)]),
            |_| 99,
            |_| 0,
        );
        assert_eq!(stale, None, "nothing usable");
        let partial = dir
            .plan_probe(
                &query,
                &live(&[("a.pax", 101), ("b.pax", 200)]),
                |n| n,
                |_| 0,
            )
            .unwrap();
        assert_eq!(partial.usable_cells, 2);
        assert!(!partial.cells_by_segment.contains_key("a.pax"));
        assert_eq!(
            dir.plan_probe(&[0.0; 3], &live(&[("b.pax", 200)]), |n| n, |_| 0),
            None
        );
    }
}
