//! `.tq` file format — persist + restore a `TurboQuantStore`.
//!
//! Wire format is byte-exact per `TURBOQUANT_LLD_2026_05_30.adoc` §3:
//!
//! ```text
//! +--------------------------------------------------------+
//! | File header (64 bytes, 16-byte aligned)                |
//! |   [0..4]   magic = b"PQTQ"                              |
//! |   [4..6]   version = u16 = 1                            |
//! |   [6..7]   bit_width = u8 ∈ {2, 4}                      |
//! |   [7..8]   calibration_mode = u8 (0=Identity, 1=TqPlus) |
//! |   [8..16]  rotation_seed = u64                          |
//! |   [16..24] dim = u64                                    |
//! |   [24..32] n_vectors = u64                              |
//! |   [32..40] encoded_epoch = u64                          |
//! |   [40..48] codes_offset = u64                           |
//! |   [48..56] scales_offset = u64                          |
//! |   [56..64] calibration_offset = u64 (0 if Identity)     |
//! +--------------------------------------------------------+
//! | Bit-packed codes (codes_offset..scales_offset)         |
//! | Per-vector scales (scales_offset..calibration_offset)   |
//! | Calibration shift+scale_tq (calibration_offset..EOF), if TqPlus only |
//! +--------------------------------------------------------+
//! ```
//!
//! All multi-byte integers little-endian per LLD Q8. Loader rejects
//! mismatched magic / version / dim / bit_width / length-invariant
//! violations with `TurboQuantError::InvalidFileFormat`.

use std::io::{Read, Write};

use proximadb_quantization_types::CalibrationMode;

use super::{Calibration, TurboQuantError, check_bit_width, check_dim};

/// File-format magic. Tripwire for accidental loads of unrelated files.
pub const MAGIC: &[u8; 4] = b"PQTQ";

/// Format version. Increment requires a backwards-compatibility plan.
pub const VERSION: u16 = 1;

/// Magic for `.tvim` (TurboQuant with `IdMapIndex`). Distinct from
/// [`MAGIC`] so the loader rejects accidental cross-loads at the
/// magic-check stage. Same 64-byte header layout as `.tq`; body is
/// `.tq`-body + 8 × n_vectors bytes of little-endian u64 IDs as a
/// trailing footer.
pub const ID_MAP_MAGIC: &[u8; 4] = b"PQTI";

/// Version of the `.tvim` format. Independent of [`VERSION`] so the
/// two formats can evolve separately.
pub const ID_MAP_VERSION: u16 = 1;

/// Header byte length. The body starts at offset 64 to keep code/scale
/// reads naturally aligned.
pub const HEADER_LEN: usize = 64;

const CALIB_IDENTITY: u8 = 0;
const CALIB_TQPLUS: u8 = 1;

/// All fields the file format carries. Lifted into a struct so the
/// reader and writer can be tested independently of `TurboQuantStore`.
#[derive(Debug, Clone, PartialEq)]
pub struct PersistedStore {
    pub bit_width: u8,
    pub calibration_mode: CalibrationMode,
    pub rotation_seed: u64,
    pub dim: usize,
    pub n_vectors: usize,
    pub encoded_epoch: u64,
    pub codes: Vec<u8>,
    pub scales: Vec<f32>,
    /// `Some` iff `calibration_mode == TqPlus` AND the calibration has
    /// been fit. `None` for identity mode or pre-fit TqPlus.
    pub calibration: Option<Calibration>,
}

impl PersistedStore {
    fn bytes_per_vec(&self) -> usize {
        (self.dim * self.bit_width as usize).div_ceil(8)
    }
}

fn calibration_tag(mode: CalibrationMode) -> u8 {
    match mode {
        CalibrationMode::Identity => CALIB_IDENTITY,
        CalibrationMode::TqPlus => CALIB_TQPLUS,
    }
}

fn calibration_from_tag(tag: u8) -> Result<CalibrationMode, TurboQuantError> {
    match tag {
        CALIB_IDENTITY => Ok(CalibrationMode::Identity),
        CALIB_TQPLUS => Ok(CalibrationMode::TqPlus),
        other => Err(TurboQuantError::InvalidFileFormat(format!(
            "unknown calibration mode tag: {other}",
        ))),
    }
}

/// The 64-byte file header shared by the `.tq` and `.tvim` formats
/// (LLD §3). The two formats differ ONLY in magic and version — the field
/// set and the fixed little-endian offset layout are identical — so the
/// serialize/parse/invariant logic lives here once.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TqHeader {
    bit_width: u8,
    calibration_mode: CalibrationMode,
    rotation_seed: u64,
    dim: usize,
    n_vectors: usize,
    encoded_epoch: u64,
    codes_offset: u64,
    scales_offset: u64,
    calibration_offset: u64,
}

impl TqHeader {
    /// Serialize into the fixed 64-byte little-endian layout with the
    /// caller's format identity (`.tq` or `.tvim` magic/version).
    fn to_bytes(&self, magic: &[u8; 4], version: u16) -> [u8; HEADER_LEN] {
        let mut h = [0u8; HEADER_LEN];
        h[0..4].copy_from_slice(magic);
        h[4..6].copy_from_slice(&version.to_le_bytes());
        h[6] = self.bit_width;
        h[7] = calibration_tag(self.calibration_mode);
        h[8..16].copy_from_slice(&self.rotation_seed.to_le_bytes());
        h[16..24].copy_from_slice(&(self.dim as u64).to_le_bytes());
        h[24..32].copy_from_slice(&(self.n_vectors as u64).to_le_bytes());
        h[32..40].copy_from_slice(&self.encoded_epoch.to_le_bytes());
        h[40..48].copy_from_slice(&self.codes_offset.to_le_bytes());
        h[48..56].copy_from_slice(&self.scales_offset.to_le_bytes());
        h[56..64].copy_from_slice(&self.calibration_offset.to_le_bytes());
        h
    }

    /// Parse + validate the fixed 64-byte layout against the expected
    /// `magic`/`version`. `format_label` (`""` for `.tq`, `".tvim "` for
    /// `.tvim`) rides the version-mismatch message so each format keeps
    /// its distinct historical error text.
    fn from_bytes(
        buf: &[u8; HEADER_LEN],
        magic: &[u8; 4],
        version: u16,
        format_label: &str,
    ) -> Result<Self, TurboQuantError> {
        if &buf[0..4] != magic {
            return Err(TurboQuantError::InvalidFileFormat(format!(
                "bad magic: expected {:?}, got {:?}",
                magic,
                &buf[0..4],
            )));
        }
        let file_version = u16::from_le_bytes([buf[4], buf[5]]);
        if file_version != version {
            return Err(TurboQuantError::InvalidFileFormat(format!(
                "unsupported {format_label}version: {file_version} \
                 (this build supports {version})",
            )));
        }
        let bit_width = buf[6];
        check_bit_width(bit_width)?;
        let calibration_mode = calibration_from_tag(buf[7])?;
        let rotation_seed = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        let dim = u64::from_le_bytes(buf[16..24].try_into().unwrap()) as usize;
        check_dim(dim)?;
        let n_vectors = u64::from_le_bytes(buf[24..32].try_into().unwrap()) as usize;
        Ok(Self {
            bit_width,
            calibration_mode,
            rotation_seed,
            dim,
            n_vectors,
            encoded_epoch: u64::from_le_bytes(buf[32..40].try_into().unwrap()),
            codes_offset: u64::from_le_bytes(buf[40..48].try_into().unwrap()),
            scales_offset: u64::from_le_bytes(buf[48..56].try_into().unwrap()),
            calibration_offset: u64::from_le_bytes(buf[56..64].try_into().unwrap()),
        })
    }

    /// Cross-check the body offsets against the length invariants derived
    /// from `(dim, n_vectors, bit_width, calibration_mode)`. Returns the
    /// expected `(codes_len, scales_bytes)` section lengths for the body
    /// read.
    fn check_offsets(&self) -> Result<(usize, usize), TurboQuantError> {
        if self.codes_offset != HEADER_LEN as u64 {
            return Err(TurboQuantError::InvalidFileFormat(format!(
                "codes_offset {} != HEADER_LEN {}",
                self.codes_offset, HEADER_LEN,
            )));
        }

        let bytes_per_vec = (self.dim * self.bit_width as usize).div_ceil(8);
        let expected_codes_len = self.n_vectors * bytes_per_vec;
        let expected_scales_offset = self.codes_offset + expected_codes_len as u64;
        if self.scales_offset != expected_scales_offset {
            return Err(TurboQuantError::InvalidFileFormat(format!(
                "scales_offset {} != codes_offset + n*bytes_per_vec = {}",
                self.scales_offset, expected_scales_offset,
            )));
        }

        let expected_scales_bytes = self.n_vectors * 4;
        let expected_calibration_offset =
            if matches!(self.calibration_mode, CalibrationMode::TqPlus) {
                self.scales_offset + expected_scales_bytes as u64
            } else {
                0
            };
        if self.calibration_offset != expected_calibration_offset {
            return Err(TurboQuantError::InvalidFileFormat(format!(
                "calibration_offset {} != expected {} for mode {:?}",
                self.calibration_offset, expected_calibration_offset, self.calibration_mode,
            )));
        }
        Ok((expected_codes_len, expected_scales_bytes))
    }

    /// Assemble the parsed header + body back into a [`PersistedStore`].
    fn into_store(
        self,
        codes: Vec<u8>,
        scales: Vec<f32>,
        calibration: Option<Calibration>,
    ) -> PersistedStore {
        PersistedStore {
            bit_width: self.bit_width,
            calibration_mode: self.calibration_mode,
            rotation_seed: self.rotation_seed,
            dim: self.dim,
            n_vectors: self.n_vectors,
            encoded_epoch: self.encoded_epoch,
            codes,
            scales,
            calibration,
        }
    }
}

/// Validate a store's length invariants and derive its header (field values
/// + body offsets). Shared by both writers — `.tq` and `.tvim` serialize the
/// same `PersistedStore` body.
fn store_header(store: &PersistedStore) -> Result<TqHeader, TurboQuantError> {
    check_dim(store.dim)?;
    check_bit_width(store.bit_width)?;
    let bytes_per_vec = store.bytes_per_vec();
    let expected_codes_len = store.n_vectors * bytes_per_vec;
    if store.codes.len() != expected_codes_len {
        return Err(TurboQuantError::InvalidFileFormat(format!(
            "codes length {} mismatch with n_vectors={} bit_width={} dim={} (expected {})",
            store.codes.len(),
            store.n_vectors,
            store.bit_width,
            store.dim,
            expected_codes_len,
        )));
    }
    if store.scales.len() != store.n_vectors {
        return Err(TurboQuantError::InvalidFileFormat(format!(
            "scales length {} != n_vectors {}",
            store.scales.len(),
            store.n_vectors,
        )));
    }
    if matches!(store.calibration_mode, CalibrationMode::TqPlus) {
        if let Some(cal) = &store.calibration {
            if cal.shift.len() != store.dim || cal.scale_tq.len() != store.dim {
                return Err(TurboQuantError::InvalidFileFormat(format!(
                    "calibration vectors have length ({}, {}) but expected {}",
                    cal.shift.len(),
                    cal.scale_tq.len(),
                    store.dim,
                )));
            }
        }
    } else if store.calibration.is_some() {
        return Err(TurboQuantError::InvalidFileFormat(
            "calibration body present but calibration_mode is Identity".to_string(),
        ));
    }

    let codes_offset = HEADER_LEN as u64;
    let scales_offset = codes_offset + store.codes.len() as u64;
    let calibration_offset = if store.calibration.is_some() {
        scales_offset + (store.scales.len() * 4) as u64
    } else {
        0
    };

    Ok(TqHeader {
        bit_width: store.bit_width,
        calibration_mode: store.calibration_mode,
        rotation_seed: store.rotation_seed,
        dim: store.dim,
        n_vectors: store.n_vectors,
        encoded_epoch: store.encoded_epoch,
        codes_offset,
        scales_offset,
        calibration_offset,
    })
}

/// Write the `.tq` body shared by both formats: bit-packed codes, per-vector
/// scales as little-endian f32, then the calibration body when present
/// (shift first, then scale_tq).
fn write_body<W: Write>(w: &mut W, store: &PersistedStore) -> Result<(), TurboQuantError> {
    w.write_all(&store.codes).map_err(io_err)?;

    // Per-vector scales as little-endian f32.
    let mut scales_bytes = Vec::with_capacity(store.scales.len() * 4);
    for &s in &store.scales {
        scales_bytes.extend_from_slice(&s.to_le_bytes());
    }
    w.write_all(&scales_bytes).map_err(io_err)?;

    // Calibration body, when present: shift first, then scale_tq.
    if let Some(cal) = &store.calibration {
        let mut cal_bytes = Vec::with_capacity(2 * cal.shift.len() * 4);
        for &s in &cal.shift {
            cal_bytes.extend_from_slice(&s.to_le_bytes());
        }
        for &s in &cal.scale_tq {
            cal_bytes.extend_from_slice(&s.to_le_bytes());
        }
        w.write_all(&cal_bytes).map_err(io_err)?;
    }

    Ok(())
}

/// Read the `.tq` body described by `header`: bit-packed codes, per-vector
/// LE f32 scales, then the calibration body for TqPlus. Shared by both
/// readers.
fn read_body<R: Read>(
    r: &mut R,
    header: &TqHeader,
    expected_codes_len: usize,
    expected_scales_bytes: usize,
) -> Result<(Vec<u8>, Vec<f32>, Option<Calibration>), TurboQuantError> {
    let (dim, n_vectors) = (header.dim, header.n_vectors);

    let mut codes = vec![0u8; expected_codes_len];
    r.read_exact(&mut codes).map_err(io_err)?;

    let mut scales_bytes = vec![0u8; expected_scales_bytes];
    r.read_exact(&mut scales_bytes).map_err(io_err)?;
    let mut scales = Vec::with_capacity(n_vectors);
    for chunk in scales_bytes.as_chunks::<4>().0 {
        scales.push(f32::from_le_bytes(*chunk));
    }

    let calibration = if matches!(header.calibration_mode, CalibrationMode::TqPlus) {
        let mut cal_bytes = vec![0u8; 2 * dim * 4];
        r.read_exact(&mut cal_bytes).map_err(io_err)?;
        let mut shift = Vec::with_capacity(dim);
        let mut scale_tq = Vec::with_capacity(dim);
        for chunk in cal_bytes[..dim * 4].as_chunks::<4>().0 {
            shift.push(f32::from_le_bytes(*chunk));
        }
        for chunk in cal_bytes[dim * 4..].as_chunks::<4>().0 {
            scale_tq.push(f32::from_le_bytes(*chunk));
        }
        Some(Calibration { shift, scale_tq })
    } else {
        None
    };

    Ok((codes, scales, calibration))
}

/// Serialize a `PersistedStore` into the LLD §3 byte sequence.
pub fn write_to<W: Write>(w: &mut W, store: &PersistedStore) -> Result<(), TurboQuantError> {
    let header = store_header(store)?;
    w.write_all(&header.to_bytes(MAGIC, VERSION))
        .map_err(io_err)?;
    write_body(w, store)
}

/// Parse a `PersistedStore` from the LLD §3 byte sequence.
pub fn read_from<R: Read>(r: &mut R) -> Result<PersistedStore, TurboQuantError> {
    let mut buf = [0u8; HEADER_LEN];
    r.read_exact(&mut buf).map_err(io_err)?;
    let header = TqHeader::from_bytes(&buf, MAGIC, VERSION, "")?;
    let (codes_len, scales_bytes) = header.check_offsets()?;
    let (codes, scales, calibration) = read_body(r, &header, codes_len, scales_bytes)?;
    Ok(header.into_store(codes, scales, calibration))
}

/// Wrap `std::io::Error` into `TurboQuantError` without exposing the
/// `io::Error` API surface in the error enum.
fn io_err(err: std::io::Error) -> TurboQuantError {
    TurboQuantError::InvalidFileFormat(format!("I/O: {err}"))
}

// ============================================================================
// .tvim wire format — IdMapIndex persistence
// ============================================================================

/// All fields the `.tvim` file format carries. Superset of
/// [`PersistedStore`] with a trailing `ids` vector keyed by slot.
#[derive(Debug, Clone, PartialEq)]
pub struct PersistedIdMap {
    pub store: PersistedStore,
    pub ids: Vec<u64>,
}

/// Serialize a `PersistedIdMap` into the `.tvim` byte sequence.
///
/// Wire shape: 64-byte header (same field set as [`write_to`] but with
/// `ID_MAP_MAGIC` and `ID_MAP_VERSION`) + bit-packed codes + per-vec
/// scales + optional calibration body + `n_vectors × 8` bytes of
/// little-endian u64 IDs.
pub fn write_id_map_to<W: Write>(
    w: &mut W,
    persisted: &PersistedIdMap,
) -> Result<(), TurboQuantError> {
    if persisted.ids.len() != persisted.store.n_vectors {
        return Err(TurboQuantError::InvalidFileFormat(format!(
            "ids.len()={} != store.n_vectors={}",
            persisted.ids.len(),
            persisted.store.n_vectors,
        )));
    }
    // Reject duplicate IDs at write time — the in-memory invariant the
    // loader will assume. Catches caller bugs that would produce a
    // corrupt file otherwise.
    let mut seen = std::collections::HashSet::with_capacity(persisted.ids.len());
    for &id in &persisted.ids {
        if !seen.insert(id) {
            return Err(TurboQuantError::InvalidFileFormat(format!(
                "duplicate id {id} in persisted slot table",
            )));
        }
    }

    // Header — same layout as `.tq` but with the IdMap magic + version —
    // then the shared `.tq` body, then the IDs footer.
    let header = store_header(&persisted.store)?;
    w.write_all(&header.to_bytes(ID_MAP_MAGIC, ID_MAP_VERSION))
        .map_err(io_err)?;
    write_body(w, &persisted.store)?;

    let mut id_bytes = Vec::with_capacity(persisted.ids.len() * 8);
    for &id in &persisted.ids {
        id_bytes.extend_from_slice(&id.to_le_bytes());
    }
    w.write_all(&id_bytes).map_err(io_err)?;
    Ok(())
}

/// Parse a `PersistedIdMap` from the `.tvim` byte sequence.
pub fn read_id_map_from<R: Read>(r: &mut R) -> Result<PersistedIdMap, TurboQuantError> {
    let mut buf = [0u8; HEADER_LEN];
    r.read_exact(&mut buf).map_err(io_err)?;
    let header = TqHeader::from_bytes(&buf, ID_MAP_MAGIC, ID_MAP_VERSION, ".tvim ")?;
    let (codes_len, scales_bytes) = header.check_offsets()?;
    let (codes, scales, calibration) = read_body(r, &header, codes_len, scales_bytes)?;

    // IDs footer — n_vectors u64s.
    let mut id_bytes = vec![0u8; header.n_vectors * 8];
    r.read_exact(&mut id_bytes).map_err(io_err)?;
    let mut ids = Vec::with_capacity(header.n_vectors);
    for chunk in id_bytes.as_chunks::<8>().0 {
        ids.push(u64::from_le_bytes(*chunk));
    }
    // Reject duplicate IDs in the persisted footer — corresponds to the
    // in-memory `id_to_slot` HashMap uniqueness invariant.
    let mut seen = std::collections::HashSet::with_capacity(ids.len());
    for &id in &ids {
        if !seen.insert(id) {
            return Err(TurboQuantError::InvalidFileFormat(format!(
                "duplicate id {id} in persisted slot table",
            )));
        }
    }

    Ok(PersistedIdMap {
        store: header.into_store(codes, scales, calibration),
        ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_store_identity() -> PersistedStore {
        // dim=8, bit_width=4 → 1 byte per code per vector; 4-bit, 2 codes
        // per byte, so 4 bytes per vector. 3 vectors total.
        PersistedStore {
            bit_width: 4,
            calibration_mode: CalibrationMode::Identity,
            rotation_seed: 0xdead_beef_cafe_babe,
            dim: 8,
            n_vectors: 3,
            encoded_epoch: 7,
            codes: vec![
                0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC,
            ],
            scales: vec![1.0, 0.5, 2.0],
            calibration: None,
        }
    }

    fn sample_store_tq_plus() -> PersistedStore {
        let dim = 16;
        let n = 2;
        // bit_width=2, dim=16 → 4 bytes per vec.
        PersistedStore {
            bit_width: 2,
            calibration_mode: CalibrationMode::TqPlus,
            rotation_seed: 42,
            dim,
            n_vectors: n,
            encoded_epoch: 1,
            codes: vec![0xAA; n * 4],
            scales: vec![1.0, 0.9],
            calibration: Some(Calibration {
                shift: (0..dim).map(|i| (i as f32) * 0.01).collect(),
                scale_tq: (0..dim).map(|i| 1.0 + (i as f32) * 0.05).collect(),
            }),
        }
    }

    #[test]
    fn round_trip_identity() {
        let store = sample_store_identity();
        let mut buf = Vec::new();
        write_to(&mut buf, &store).unwrap();
        let mut cur = std::io::Cursor::new(buf);
        let restored = read_from(&mut cur).unwrap();
        assert_eq!(store, restored);
    }

    #[test]
    fn round_trip_tq_plus() {
        let store = sample_store_tq_plus();
        let mut buf = Vec::new();
        write_to(&mut buf, &store).unwrap();
        let mut cur = std::io::Cursor::new(buf);
        let restored = read_from(&mut cur).unwrap();
        assert_eq!(store, restored);
    }

    #[test]
    fn header_size_locked_at_64_bytes() {
        // Tripwire: any future field addition that grows the header is a
        // wire-format breaking change and must bump VERSION.
        let store = sample_store_identity();
        let mut buf = Vec::new();
        write_to(&mut buf, &store).unwrap();
        let codes_offset = u64::from_le_bytes(buf[40..48].try_into().unwrap());
        assert_eq!(codes_offset, HEADER_LEN as u64);
        assert_eq!(HEADER_LEN, 64);
    }

    #[test]
    fn header_magic_is_pqtq() {
        let store = sample_store_identity();
        let mut buf = Vec::new();
        write_to(&mut buf, &store).unwrap();
        assert_eq!(&buf[0..4], b"PQTQ");
    }

    #[test]
    fn version_is_locked_at_1() {
        // Same tripwire as header_size: a version bump is a deliberate
        // wire contract change.
        assert_eq!(VERSION, 1);
        let store = sample_store_identity();
        let mut buf = Vec::new();
        write_to(&mut buf, &store).unwrap();
        let version = u16::from_le_bytes([buf[4], buf[5]]);
        assert_eq!(version, 1);
    }

    #[test]
    fn rejects_bad_magic() {
        let store = sample_store_identity();
        let mut buf = Vec::new();
        write_to(&mut buf, &store).unwrap();
        buf[0] = b'X';
        let err = read_from(&mut std::io::Cursor::new(buf)).unwrap_err();
        assert!(
            matches!(err, TurboQuantError::InvalidFileFormat(ref s) if s.contains("bad magic"))
        );
    }

    #[test]
    fn rejects_unsupported_version() {
        let store = sample_store_identity();
        let mut buf = Vec::new();
        write_to(&mut buf, &store).unwrap();
        buf[4] = 99; // version low byte
        buf[5] = 0;
        let err = read_from(&mut std::io::Cursor::new(buf)).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::InvalidFileFormat(ref s) if s.contains("unsupported version")
        ));
    }

    #[test]
    fn rejects_bad_dim() {
        let store = sample_store_identity();
        let mut buf = Vec::new();
        write_to(&mut buf, &store).unwrap();
        // Corrupt dim to 7 (not multiple of 8). u64 little-endian at
        // offset 16; low byte becomes 7.
        buf[16] = 7;
        for i in 17..24 {
            buf[i] = 0;
        }
        let err = read_from(&mut std::io::Cursor::new(buf)).unwrap_err();
        assert!(matches!(err, TurboQuantError::DimNotMultipleOf8(7)));
    }

    #[test]
    fn rejects_bad_bit_width() {
        let store = sample_store_identity();
        let mut buf = Vec::new();
        write_to(&mut buf, &store).unwrap();
        buf[6] = 5; // bit_width
        let err = read_from(&mut std::io::Cursor::new(buf)).unwrap_err();
        assert!(matches!(err, TurboQuantError::BitWidthOutOfRange(5)));
    }

    #[test]
    fn rejects_inconsistent_scales_offset() {
        let store = sample_store_identity();
        let mut buf = Vec::new();
        write_to(&mut buf, &store).unwrap();
        // Bump scales_offset by 1, breaking the codes-length invariant.
        let mut scales_off = u64::from_le_bytes(buf[48..56].try_into().unwrap());
        scales_off += 1;
        buf[48..56].copy_from_slice(&scales_off.to_le_bytes());
        let err = read_from(&mut std::io::Cursor::new(buf)).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::InvalidFileFormat(ref s) if s.contains("scales_offset")
        ));
    }

    #[test]
    fn rejects_truncated_body() {
        let store = sample_store_identity();
        let mut buf = Vec::new();
        write_to(&mut buf, &store).unwrap();
        // Drop the last 2 bytes — should fail on read_exact for scales.
        buf.truncate(buf.len() - 2);
        let err = read_from(&mut std::io::Cursor::new(buf)).unwrap_err();
        assert!(matches!(err, TurboQuantError::InvalidFileFormat(ref s) if s.contains("I/O")));
    }

    #[test]
    fn rejects_calibration_present_in_identity_mode() {
        let mut store = sample_store_identity();
        // Smuggle a calibration into an Identity-mode store.
        store.calibration = Some(Calibration {
            shift: vec![0.0; store.dim],
            scale_tq: vec![1.0; store.dim],
        });
        let mut buf = Vec::new();
        let err = write_to(&mut buf, &store).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::InvalidFileFormat(ref s) if s.contains("Identity")
        ));
    }

    #[test]
    fn rejects_calibration_dim_mismatch() {
        let mut store = sample_store_tq_plus();
        if let Some(cal) = &mut store.calibration {
            cal.shift.pop(); // length now != dim
        }
        let err = write_to(&mut Vec::new(), &store).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::InvalidFileFormat(ref s) if s.contains("calibration vectors")
        ));
    }

    #[test]
    fn rejects_scales_count_mismatch() {
        let mut store = sample_store_identity();
        store.scales.push(0.5);
        let err = write_to(&mut Vec::new(), &store).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::InvalidFileFormat(ref s) if s.contains("scales length")
        ));
    }

    // ------------------------------------------------------------------
    // .tvim (PersistedIdMap) round-trips
    // ------------------------------------------------------------------

    fn sample_id_map_identity() -> PersistedIdMap {
        PersistedIdMap {
            store: sample_store_identity(),
            ids: vec![1001, 1002, 1003],
        }
    }

    fn sample_id_map_tq_plus() -> PersistedIdMap {
        PersistedIdMap {
            store: sample_store_tq_plus(),
            ids: vec![42, 43],
        }
    }

    #[test]
    fn id_map_round_trip_identity() {
        let p = sample_id_map_identity();
        let mut buf = Vec::new();
        write_id_map_to(&mut buf, &p).unwrap();
        let mut cur = std::io::Cursor::new(buf);
        let restored = read_id_map_from(&mut cur).unwrap();
        assert_eq!(p, restored);
    }

    #[test]
    fn id_map_round_trip_tq_plus() {
        let p = sample_id_map_tq_plus();
        let mut buf = Vec::new();
        write_id_map_to(&mut buf, &p).unwrap();
        let mut cur = std::io::Cursor::new(buf);
        let restored = read_id_map_from(&mut cur).unwrap();
        assert_eq!(p, restored);
    }

    #[test]
    fn id_map_magic_is_pqti() {
        let p = sample_id_map_identity();
        let mut buf = Vec::new();
        write_id_map_to(&mut buf, &p).unwrap();
        assert_eq!(&buf[0..4], b"PQTI");
    }

    #[test]
    fn id_map_rejects_tq_magic_load() {
        // A `.tq` file must NOT load via `.tvim` reader.
        let store = sample_store_identity();
        let mut buf = Vec::new();
        write_to(&mut buf, &store).unwrap();
        let err = read_id_map_from(&mut std::io::Cursor::new(buf)).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::InvalidFileFormat(ref s) if s.contains("bad magic")
        ));
    }

    #[test]
    fn tq_reader_rejects_tvim_magic() {
        // Reverse: a `.tvim` file must NOT load via the `.tq` reader.
        let p = sample_id_map_identity();
        let mut buf = Vec::new();
        write_id_map_to(&mut buf, &p).unwrap();
        let err = read_from(&mut std::io::Cursor::new(buf)).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::InvalidFileFormat(ref s) if s.contains("bad magic")
        ));
    }

    #[test]
    fn id_map_rejects_ids_count_mismatch() {
        let mut p = sample_id_map_identity();
        p.ids.push(9999); // now 4 IDs but store says 3 vectors
        let err = write_id_map_to(&mut Vec::new(), &p).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::InvalidFileFormat(ref s) if s.contains("ids.len()")
        ));
    }

    #[test]
    fn id_map_rejects_duplicate_ids_at_write() {
        let mut p = sample_id_map_identity();
        p.ids[1] = p.ids[0]; // make first and second duplicate
        let err = write_id_map_to(&mut Vec::new(), &p).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::InvalidFileFormat(ref s) if s.contains("duplicate id")
        ));
    }

    #[test]
    fn id_map_rejects_duplicate_ids_at_read() {
        // Hand-craft a buffer with a known dup in the ID footer to
        // exercise the loader's dedup check (the write path's check
        // wouldn't fire on a maliciously constructed file).
        let p = sample_id_map_identity();
        let mut buf = Vec::new();
        write_id_map_to(&mut buf, &p).unwrap();
        // Overwrite the second 8-byte ID with the first one (slots
        // 0 and 1 now share id 1001).
        let n_id_bytes = p.ids.len() * 8;
        let footer_start = buf.len() - n_id_bytes;
        let first_id_bytes: [u8; 8] = buf[footer_start..footer_start + 8].try_into().unwrap();
        buf[footer_start + 8..footer_start + 16].copy_from_slice(&first_id_bytes);
        let err = read_id_map_from(&mut std::io::Cursor::new(buf)).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::InvalidFileFormat(ref s) if s.contains("duplicate id")
        ));
    }

    #[test]
    fn id_map_rejects_truncated_id_footer() {
        let p = sample_id_map_identity();
        let mut buf = Vec::new();
        write_id_map_to(&mut buf, &p).unwrap();
        // Drop the last 8 bytes — one full ID. read_exact on the footer
        // must fail.
        buf.truncate(buf.len() - 8);
        let err = read_id_map_from(&mut std::io::Cursor::new(buf)).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::InvalidFileFormat(ref s) if s.contains("I/O")
        ));
    }

    #[test]
    fn id_map_rejects_unsupported_version() {
        let p = sample_id_map_identity();
        let mut buf = Vec::new();
        write_id_map_to(&mut buf, &p).unwrap();
        buf[4] = 99;
        buf[5] = 0;
        let err = read_id_map_from(&mut std::io::Cursor::new(buf)).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::InvalidFileFormat(ref s) if s.contains("unsupported .tvim version")
        ));
    }

    // ------------------------------------------------------------------
    // Golden wire bytes — pinned BEFORE the shared-header extraction so the
    // `.tq`/`.tvim` refactor is provably byte-identical. Header fields
    // hand-derived from LLD §3 (all multi-byte ints little-endian);
    // identity fixtures are fully deterministic so the whole file pins.
    //
    //   identity header (shared layout, magic/version differ per format):
    //     [0..4]   magic "PQTQ"/"PQTI"           50 51 54 51 / 50 51 54 49
    //     [4..6]   version 1                      01 00
    //     [6]      bit_width 4                    04
    //     [7]      calibration Identity           00
    //     [8..16]  seed 0xdead_beef_cafe_babe     be ba fe ca ef be ad de
    //     [16..24] dim 8                           08 00*7
    //     [24..32] n_vectors 3                     03 00*7
    //     [32..40] encoded_epoch 7                 07 00*7
    //     [40..48] codes_offset 64                 40 00*7
    //     [48..56] scales_offset 64+12 = 76        4c 00*7
    //     [56..64] calibration_offset 0 (Identity) 00*8
    //   body: 12 code bytes, then LE f32 scales 1.0 / 0.5 / 2.0.
    //   `.tvim` appends ids 1001..1003 as LE u64s.
    //
    //   tq_plus header (fixtures compute the calibration floats, so only
    //   the header pins byte-exactly):
    //     [6] bit_width 2, [7] 01 (TqPlus), [8..16] seed 42,
    //     dim 16, n_vectors 2, epoch 1, codes_offset 64,
    //     scales_offset 72, calibration_offset 80.
    // ------------------------------------------------------------------

    #[test]
    fn golden_tq_identity_bytes() {
        let store = sample_store_identity();
        let mut buf = Vec::new();
        write_to(&mut buf, &store).unwrap();
        let expected: Vec<u8> = [
            // Header
            &b"PQTQ"[..],
            &[0x01, 0x00, 0x04, 0x00],
            &0xdead_beef_cafe_babeu64.to_le_bytes(),
            &8u64.to_le_bytes(),
            &3u64.to_le_bytes(),
            &7u64.to_le_bytes(),
            &64u64.to_le_bytes(), // codes_offset
            &76u64.to_le_bytes(), // scales_offset = 64 + 12 code bytes
            &0u64.to_le_bytes(),  // calibration_offset (Identity)
            // Body: codes
            &[
                0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC,
            ],
            // Body: scales 1.0, 0.5, 2.0 as LE f32
            &1.0f32.to_le_bytes(),
            &0.5f32.to_le_bytes(),
            &2.0f32.to_le_bytes(),
        ]
        .concat();
        assert_eq!(buf, expected);
        assert_eq!(buf.len(), 88);
    }

    #[test]
    fn golden_tvim_identity_bytes() {
        let p = sample_id_map_identity();
        let mut buf = Vec::new();
        write_id_map_to(&mut buf, &p).unwrap();
        let expected: Vec<u8> = [
            // Header — same field values as the `.tq` golden, PQTI magic.
            &b"PQTI"[..],
            &[0x01, 0x00, 0x04, 0x00],
            &0xdead_beef_cafe_babeu64.to_le_bytes(),
            &8u64.to_le_bytes(),
            &3u64.to_le_bytes(),
            &7u64.to_le_bytes(),
            &64u64.to_le_bytes(),
            &76u64.to_le_bytes(),
            &0u64.to_le_bytes(),
            // Body: codes + scales (identical to `.tq`)
            &[
                0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC,
            ],
            &1.0f32.to_le_bytes(),
            &0.5f32.to_le_bytes(),
            &2.0f32.to_le_bytes(),
            // IDs footer: 1001, 1002, 1003 as LE u64
            &1001u64.to_le_bytes(),
            &1002u64.to_le_bytes(),
            &1003u64.to_le_bytes(),
        ]
        .concat();
        assert_eq!(buf, expected);
        assert_eq!(buf.len(), 112);
    }

    #[test]
    fn golden_tq_plus_header_bytes() {
        let store = sample_store_tq_plus();
        let mut buf = Vec::new();
        write_to(&mut buf, &store).unwrap();
        assert_eq!(&buf[0..4], b"PQTQ");
        assert_eq!(&buf[4..6], &1u16.to_le_bytes());
        assert_eq!(buf[6], 2); // bit_width
        assert_eq!(buf[7], 1); // TqPlus calibration tag
        assert_eq!(&buf[8..16], &42u64.to_le_bytes());
        assert_eq!(&buf[16..24], &16u64.to_le_bytes());
        assert_eq!(&buf[24..32], &2u64.to_le_bytes());
        assert_eq!(&buf[32..40], &1u64.to_le_bytes());
        assert_eq!(&buf[40..48], &64u64.to_le_bytes()); // codes_offset
        assert_eq!(&buf[48..56], &72u64.to_le_bytes()); // scales_offset
        assert_eq!(&buf[56..64], &80u64.to_le_bytes()); // calibration_offset
        // Body: 8×0xAA codes, then 2 LE f32 scales, then 32 LE f32
        // calibration floats → 64 + 8 + 8 + 128 = 208 bytes.
        assert_eq!(buf.len(), 208);
        assert_eq!(&buf[64..72], &[0xAA; 8]);
    }

    #[test]
    fn golden_tvim_tq_plus_header_bytes() {
        let p = sample_id_map_tq_plus();
        let mut buf = Vec::new();
        write_id_map_to(&mut buf, &p).unwrap();
        assert_eq!(&buf[0..4], b"PQTI");
        assert_eq!(&buf[4..6], &1u16.to_le_bytes());
        assert_eq!(buf[6], 2);
        assert_eq!(buf[7], 1);
        assert_eq!(&buf[8..16], &42u64.to_le_bytes());
        assert_eq!(&buf[16..24], &16u64.to_le_bytes());
        assert_eq!(&buf[24..32], &2u64.to_le_bytes());
        assert_eq!(&buf[40..48], &64u64.to_le_bytes());
        assert_eq!(&buf[48..56], &72u64.to_le_bytes());
        assert_eq!(&buf[56..64], &80u64.to_le_bytes());
        // `.tq` body + 2×8 id bytes: 208 + 16 = 224.
        assert_eq!(buf.len(), 224);
        assert_eq!(&buf[208..216], &42u64.to_le_bytes());
        assert_eq!(&buf[216..224], &43u64.to_le_bytes());
    }
}
