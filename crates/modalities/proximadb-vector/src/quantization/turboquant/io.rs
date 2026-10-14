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

/// Serialize a `PersistedStore` into the LLD §3 byte sequence.
pub fn write_to<W: Write>(w: &mut W, store: &PersistedStore) -> Result<(), TurboQuantError> {
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

    // Header
    let mut header = [0u8; HEADER_LEN];
    header[0..4].copy_from_slice(MAGIC);
    header[4..6].copy_from_slice(&VERSION.to_le_bytes());
    header[6] = store.bit_width;
    header[7] = calibration_tag(store.calibration_mode);
    header[8..16].copy_from_slice(&store.rotation_seed.to_le_bytes());
    header[16..24].copy_from_slice(&(store.dim as u64).to_le_bytes());
    header[24..32].copy_from_slice(&(store.n_vectors as u64).to_le_bytes());
    header[32..40].copy_from_slice(&store.encoded_epoch.to_le_bytes());
    header[40..48].copy_from_slice(&codes_offset.to_le_bytes());
    header[48..56].copy_from_slice(&scales_offset.to_le_bytes());
    header[56..64].copy_from_slice(&calibration_offset.to_le_bytes());

    w.write_all(&header).map_err(io_err)?;
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

/// Parse a `PersistedStore` from the LLD §3 byte sequence.
pub fn read_from<R: Read>(r: &mut R) -> Result<PersistedStore, TurboQuantError> {
    let mut header = [0u8; HEADER_LEN];
    r.read_exact(&mut header).map_err(io_err)?;

    if &header[0..4] != MAGIC {
        return Err(TurboQuantError::InvalidFileFormat(format!(
            "bad magic: expected {:?}, got {:?}",
            MAGIC,
            &header[0..4],
        )));
    }
    let version = u16::from_le_bytes([header[4], header[5]]);
    if version != VERSION {
        return Err(TurboQuantError::InvalidFileFormat(format!(
            "unsupported version: {version} (this build supports {VERSION})",
        )));
    }
    let bit_width = header[6];
    check_bit_width(bit_width)?;
    let calibration_mode = calibration_from_tag(header[7])?;
    let rotation_seed = u64::from_le_bytes(header[8..16].try_into().unwrap());
    let dim = u64::from_le_bytes(header[16..24].try_into().unwrap()) as usize;
    check_dim(dim)?;
    let n_vectors = u64::from_le_bytes(header[24..32].try_into().unwrap()) as usize;
    let encoded_epoch = u64::from_le_bytes(header[32..40].try_into().unwrap());
    let codes_offset = u64::from_le_bytes(header[40..48].try_into().unwrap());
    let scales_offset = u64::from_le_bytes(header[48..56].try_into().unwrap());
    let calibration_offset = u64::from_le_bytes(header[56..64].try_into().unwrap());

    if codes_offset != HEADER_LEN as u64 {
        return Err(TurboQuantError::InvalidFileFormat(format!(
            "codes_offset {} != HEADER_LEN {}",
            codes_offset, HEADER_LEN,
        )));
    }

    let bytes_per_vec = (dim * bit_width as usize).div_ceil(8);
    let expected_codes_len = n_vectors * bytes_per_vec;
    let expected_scales_offset = codes_offset + expected_codes_len as u64;
    if scales_offset != expected_scales_offset {
        return Err(TurboQuantError::InvalidFileFormat(format!(
            "scales_offset {} != codes_offset + n*bytes_per_vec = {}",
            scales_offset, expected_scales_offset,
        )));
    }

    let expected_scales_bytes = n_vectors * 4;
    let expected_calibration_offset = if matches!(calibration_mode, CalibrationMode::TqPlus) {
        scales_offset + expected_scales_bytes as u64
    } else {
        0
    };
    if calibration_offset != expected_calibration_offset {
        return Err(TurboQuantError::InvalidFileFormat(format!(
            "calibration_offset {} != expected {} for mode {:?}",
            calibration_offset, expected_calibration_offset, calibration_mode,
        )));
    }

    let mut codes = vec![0u8; expected_codes_len];
    r.read_exact(&mut codes).map_err(io_err)?;

    let mut scales_bytes = vec![0u8; expected_scales_bytes];
    r.read_exact(&mut scales_bytes).map_err(io_err)?;
    let mut scales = Vec::with_capacity(n_vectors);
    for chunk in scales_bytes.as_chunks::<4>().0 {
        scales.push(f32::from_le_bytes(*chunk));
    }

    let calibration = if matches!(calibration_mode, CalibrationMode::TqPlus) {
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

    Ok(PersistedStore {
        bit_width,
        calibration_mode,
        rotation_seed,
        dim,
        n_vectors,
        encoded_epoch,
        codes,
        scales,
        calibration,
    })
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

    let store = &persisted.store;
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

    // Header — same layout as `.tq` but with the IdMap magic + version.
    let mut header = [0u8; HEADER_LEN];
    header[0..4].copy_from_slice(ID_MAP_MAGIC);
    header[4..6].copy_from_slice(&ID_MAP_VERSION.to_le_bytes());
    header[6] = store.bit_width;
    header[7] = calibration_tag(store.calibration_mode);
    header[8..16].copy_from_slice(&store.rotation_seed.to_le_bytes());
    header[16..24].copy_from_slice(&(store.dim as u64).to_le_bytes());
    header[24..32].copy_from_slice(&(store.n_vectors as u64).to_le_bytes());
    header[32..40].copy_from_slice(&store.encoded_epoch.to_le_bytes());
    header[40..48].copy_from_slice(&codes_offset.to_le_bytes());
    header[48..56].copy_from_slice(&scales_offset.to_le_bytes());
    header[56..64].copy_from_slice(&calibration_offset.to_le_bytes());

    w.write_all(&header).map_err(io_err)?;
    w.write_all(&store.codes).map_err(io_err)?;

    let mut scales_bytes = Vec::with_capacity(store.scales.len() * 4);
    for &s in &store.scales {
        scales_bytes.extend_from_slice(&s.to_le_bytes());
    }
    w.write_all(&scales_bytes).map_err(io_err)?;

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

    // IDs footer.
    let mut id_bytes = Vec::with_capacity(persisted.ids.len() * 8);
    for &id in &persisted.ids {
        id_bytes.extend_from_slice(&id.to_le_bytes());
    }
    w.write_all(&id_bytes).map_err(io_err)?;
    Ok(())
}

/// Parse a `PersistedIdMap` from the `.tvim` byte sequence.
pub fn read_id_map_from<R: Read>(r: &mut R) -> Result<PersistedIdMap, TurboQuantError> {
    let mut header = [0u8; HEADER_LEN];
    r.read_exact(&mut header).map_err(io_err)?;

    if &header[0..4] != ID_MAP_MAGIC {
        return Err(TurboQuantError::InvalidFileFormat(format!(
            "bad magic: expected {:?}, got {:?}",
            ID_MAP_MAGIC,
            &header[0..4],
        )));
    }
    let version = u16::from_le_bytes([header[4], header[5]]);
    if version != ID_MAP_VERSION {
        return Err(TurboQuantError::InvalidFileFormat(format!(
            "unsupported .tvim version: {version} (this build supports {ID_MAP_VERSION})",
        )));
    }
    let bit_width = header[6];
    check_bit_width(bit_width)?;
    let calibration_mode = calibration_from_tag(header[7])?;
    let rotation_seed = u64::from_le_bytes(header[8..16].try_into().unwrap());
    let dim = u64::from_le_bytes(header[16..24].try_into().unwrap()) as usize;
    check_dim(dim)?;
    let n_vectors = u64::from_le_bytes(header[24..32].try_into().unwrap()) as usize;
    let encoded_epoch = u64::from_le_bytes(header[32..40].try_into().unwrap());
    let codes_offset = u64::from_le_bytes(header[40..48].try_into().unwrap());
    let scales_offset = u64::from_le_bytes(header[48..56].try_into().unwrap());
    let calibration_offset = u64::from_le_bytes(header[56..64].try_into().unwrap());

    if codes_offset != HEADER_LEN as u64 {
        return Err(TurboQuantError::InvalidFileFormat(format!(
            "codes_offset {} != HEADER_LEN {}",
            codes_offset, HEADER_LEN,
        )));
    }
    let bytes_per_vec = (dim * bit_width as usize).div_ceil(8);
    let expected_codes_len = n_vectors * bytes_per_vec;
    let expected_scales_offset = codes_offset + expected_codes_len as u64;
    if scales_offset != expected_scales_offset {
        return Err(TurboQuantError::InvalidFileFormat(format!(
            "scales_offset {} != codes_offset + n*bytes_per_vec = {}",
            scales_offset, expected_scales_offset,
        )));
    }
    let expected_scales_bytes = n_vectors * 4;
    let expected_calibration_offset = if matches!(calibration_mode, CalibrationMode::TqPlus) {
        scales_offset + expected_scales_bytes as u64
    } else {
        0
    };
    if calibration_offset != expected_calibration_offset {
        return Err(TurboQuantError::InvalidFileFormat(format!(
            "calibration_offset {} != expected {} for mode {:?}",
            calibration_offset, expected_calibration_offset, calibration_mode,
        )));
    }

    let mut codes = vec![0u8; expected_codes_len];
    r.read_exact(&mut codes).map_err(io_err)?;

    let mut scales_bytes = vec![0u8; expected_scales_bytes];
    r.read_exact(&mut scales_bytes).map_err(io_err)?;
    let mut scales = Vec::with_capacity(n_vectors);
    for chunk in scales_bytes.as_chunks::<4>().0 {
        scales.push(f32::from_le_bytes(*chunk));
    }

    let calibration = if matches!(calibration_mode, CalibrationMode::TqPlus) {
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

    // IDs footer — n_vectors u64s.
    let mut id_bytes = vec![0u8; n_vectors * 8];
    r.read_exact(&mut id_bytes).map_err(io_err)?;
    let mut ids = Vec::with_capacity(n_vectors);
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
        store: PersistedStore {
            bit_width,
            calibration_mode,
            rotation_seed,
            dim,
            n_vectors,
            encoded_epoch,
            codes,
            scales,
            calibration,
        },
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
}
