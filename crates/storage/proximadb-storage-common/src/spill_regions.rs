// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Disk-backed construction of the coalesced PAX vector regions.
//!
//! Region A (RaBitQ) and Region B (SQ8) both require corpus-wide fitted
//! parameters, but neither requires the corpus to remain in RAM. This module
//! writes ordered f32 rows once, accumulates the exact canonical reductions,
//! then sequentially encodes each fixed-width row into local region files.

#![forbid(unsafe_code)]

use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use proximadb_codec::RaBitQParams;
use proximadb_codec::functions::rabitq::{RaBitQEncodeScratch, build_rotation, encode_into};
use proximadb_codec::functions::sq8::{Sq8Params, quantize_one};

const VECTOR_BUFFER_BYTES: usize = 1024 * 1024;

/// Final disk regions plus the small fitted metadata needed by the PAX footer.
/// The task directory is reclaimed when this value is dropped.
pub struct DiskEncodedRegions {
    task_directory: tempfile::TempDir,
    rabitq_path: PathBuf,
    sq8_path: PathBuf,
    /// Region C (exact f32) spooled file — `Some` only when the finish was
    /// asked to emit it (`emit_exact`); payload is byte-for-byte the wire form
    /// of the in-memory encoder's Region C (TD-PAXRG-1).
    exact_path: Option<PathBuf>,
    pub row_count: u32,
    pub dim: u32,
    pub centroid: Vec<f32>,
    pub sq8_params: Sq8Params,
    pub rabitq_len: u64,
    pub sq8_len: u64,
}

impl DiskEncodedRegions {
    pub fn rabitq_path(&self) -> &Path {
        &self.rabitq_path
    }

    pub fn sq8_path(&self) -> &Path {
        &self.sq8_path
    }

    /// Region C (exact f32) spooled path + length, when emitted.
    pub fn exact(&self) -> Option<(&Path, u64)> {
        self.exact_path.as_ref().map(|p| {
            let len = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
            (p.as_path(), len)
        })
    }

    pub fn scratch_path(&self) -> &Path {
        self.task_directory.path()
    }
}

/// One-pass vector spool followed by bounded sequential Region-A/B encoding.
pub struct DiskVectorSpool {
    task_directory: tempfile::TempDir,
    vectors_path: PathBuf,
    vectors: BufWriter<File>,
    validity: Vec<u8>,
    row_count: u32,
    dim: usize,
    present_count: u32,
    centroid_sum: Vec<f32>,
    sq8_min: f32,
    sq8_max: f32,
}

impl DiskVectorSpool {
    pub fn new(scratch_root: &Path) -> Result<Self> {
        let task_directory = tempfile::Builder::new()
            .prefix("proximadb-pax-regions-")
            .tempdir_in(scratch_root)?;
        let vectors_path = task_directory.path().join("ordered-vectors.f32");
        let vectors = BufWriter::with_capacity(
            VECTOR_BUFFER_BYTES,
            OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&vectors_path)?,
        );
        Ok(Self {
            task_directory,
            vectors_path,
            vectors,
            validity: Vec::new(),
            row_count: 0,
            dim: 0,
            present_count: 0,
            centroid_sum: Vec::new(),
            sq8_min: f32::INFINITY,
            sq8_max: f32::NEG_INFINITY,
        })
    }

    /// Append one already-ordered vector row. `None` preserves a null row and
    /// writes a zero-filled fixed-width slot once dimensionality is known.
    pub fn push(&mut self, vector: Option<&[f32]>) -> Result<()> {
        if self.row_count == u32::MAX {
            bail!("PAX coalesced region row count exceeds u32");
        }
        if let Some(vector) = vector {
            if vector.is_empty() {
                bail!("PAX coalesced spill vector dimension must be greater than zero");
            }
            if self.dim != 0 && vector.len() != self.dim {
                bail!(
                    "PAX coalesced spill vector dim {} != established dim {}",
                    vector.len(),
                    self.dim
                );
            }
        }

        // Mutate the spool only after all fallible row-shape validation. A
        // rejected row must leave both the byte stream and validity bitmap
        // unchanged so the caller can fail the task without corrupting state.
        if self.row_count.is_multiple_of(8) {
            self.validity.push(0);
        }

        if let Some(vector) = vector {
            if self.dim == 0 {
                self.dim = vector.len();
                self.centroid_sum.resize(self.dim, 0.0);
                self.write_zero_rows(self.row_count as usize)?;
            }

            if let Some(byte) = self.validity.get_mut((self.row_count / 8) as usize) {
                *byte |= 1u8 << (self.row_count & 7);
            }
            for (sum, &value) in self.centroid_sum.iter_mut().zip(vector) {
                *sum += value;
                if value.is_finite() {
                    self.sq8_min = self.sq8_min.min(value);
                    self.sq8_max = self.sq8_max.max(value);
                }
                self.vectors.write_all(&value.to_le_bytes())?;
            }
            self.present_count = self.present_count.saturating_add(1);
        } else if self.dim > 0 {
            self.write_zero_rows(1)?;
        }

        self.row_count = self.row_count.saturating_add(1);
        Ok(())
    }

    fn write_zero_rows(&mut self, rows: usize) -> Result<()> {
        let bytes = rows
            .checked_mul(self.dim)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| anyhow::anyhow!("PAX vector spill zero-fill size overflows usize"))?;
        const ZEROES: [u8; 8192] = [0; 8192];
        let mut remaining = bytes;
        while remaining > 0 {
            let take = remaining.min(ZEROES.len());
            self.vectors.write_all(&ZEROES[..take])?;
            remaining -= take;
        }
        Ok(())
    }

    pub fn row_count(&self) -> u32 {
        self.row_count
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Encode the spooled corpus into the coalesced regions. When `emit_exact`
    /// is set (TD-PAXRG-1: row-group layout ∧ f32 tier), the spooled raw-f32
    /// file is materialized as **Region C** — `[row_count u32][dim u32]
    /// [validity][n·dim·f32]` — by header + copy; the spool's encoding is
    /// byte-for-byte the wire form already.
    pub fn finish(mut self, seed: u64, emit_exact: bool) -> Result<DiskEncodedRegions> {
        if self.row_count == 0 || self.dim == 0 || self.present_count == 0 {
            bail!("cannot encode empty PAX coalesced spill regions");
        }
        self.vectors.flush()?;
        self.vectors.get_ref().sync_data()?;

        let inv = 1.0 / self.present_count as f32;
        let centroid = self
            .centroid_sum
            .iter()
            .map(|sum| *sum * inv)
            .collect::<Vec<_>>();
        let sq8_params = fitted_sq8_params(self.sq8_min, self.sq8_max);
        let rabitq_path = self.task_directory.path().join("region-a.rabitq");
        let sq8_path = self.task_directory.path().join("region-b.sq8");
        let rabitq_len = self.encode_rabitq(&rabitq_path, seed, &centroid)?;
        let sq8_len = self.encode_sq8(&sq8_path, &sq8_params)?;
        let exact_path = if emit_exact {
            let path = self.task_directory.path().join("region-c.f32");
            let len = self.encode_exact(&path)?;
            Some((path, len))
        } else {
            None
        };
        let dim = u32::try_from(self.dim)
            .map_err(|_| anyhow::anyhow!("PAX coalesced spill dimension exceeds u32"))?;

        Ok(DiskEncodedRegions {
            task_directory: self.task_directory,
            rabitq_path,
            sq8_path,
            exact_path: exact_path.map(|(path, _)| path),
            row_count: self.row_count,
            dim,
            centroid,
            sq8_params,
            rabitq_len,
            sq8_len,
        })
    }

    /// Materialize Region C: `[row_count u32][dim u32][validity]` header + a
    /// sequential copy of the spooled raw-f32 corpus (its encoding already
    /// matches the wire form row-for-row).
    fn encode_exact(&self, output: &Path) -> Result<u64> {
        let mut writer = BufWriter::with_capacity(VECTOR_BUFFER_BYTES, create_new(output)?);
        writer.write_all(&self.row_count.to_le_bytes())?;
        writer.write_all(
            &u32::try_from(self.dim)
                .map_err(|_| anyhow::anyhow!("PAX coalesced spill dimension exceeds u32"))?
                .to_le_bytes(),
        )?;
        writer.write_all(&self.validity)?;

        writer.flush()?;
        let mut input = vector_reader(&self.vectors_path)?;
        std::io::copy(&mut input, &mut writer)?;
        writer.flush()?;
        writer.get_ref().sync_data()?;
        Ok(writer.get_ref().metadata()?.len())
    }

    fn encode_rabitq(&self, output: &Path, seed: u64, centroid: &[f32]) -> Result<u64> {
        let dim = u32::try_from(self.dim)
            .map_err(|_| anyhow::anyhow!("PAX coalesced spill dimension exceeds u32"))?;
        let mut writer = BufWriter::with_capacity(VECTOR_BUFFER_BYTES, create_new(output)?);
        writer.write_all(&self.row_count.to_le_bytes())?;
        writer.write_all(&dim.to_le_bytes())?;
        writer.write_all(&seed.to_le_bytes())?;
        for value in centroid {
            writer.write_all(&value.to_le_bytes())?;
        }
        writer.write_all(&self.validity)?;

        let params = RaBitQParams {
            dim: self.dim,
            seed,
            centroid: centroid.to_vec(),
        };
        let rotation = build_rotation(self.dim, seed);
        let mut scratch = RaBitQEncodeScratch::new(self.dim);
        let mut input = vector_reader(&self.vectors_path)?;
        let mut vector = vec![0.0f32; self.dim];
        let mut input_bytes = vec![0u8; self.dim * std::mem::size_of::<f32>()];
        let mut code = vec![0u8; 8 + self.dim.div_ceil(8)];
        for row in 0..self.row_count as usize {
            read_vector_row(&mut input, &mut input_bytes, &mut vector)?;
            code.fill(0);
            if is_present(&self.validity, row) {
                let (scalars, bits) = code.split_at_mut(8);
                let (distance, inverse) =
                    encode_into(&vector, &params, &rotation, bits, &mut scratch)?;
                scalars[..4].copy_from_slice(&distance.to_le_bytes());
                scalars[4..].copy_from_slice(&inverse.to_le_bytes());
            }
            writer.write_all(&code)?;
        }
        writer.flush()?;
        writer.get_ref().sync_data()?;
        Ok(writer.get_ref().metadata()?.len())
    }

    fn encode_sq8(&self, output: &Path, params: &Sq8Params) -> Result<u64> {
        let dim = u32::try_from(self.dim)
            .map_err(|_| anyhow::anyhow!("PAX coalesced spill dimension exceeds u32"))?;
        let mut writer = BufWriter::with_capacity(VECTOR_BUFFER_BYTES, create_new(output)?);
        writer.write_all(&self.row_count.to_le_bytes())?;
        writer.write_all(&dim.to_le_bytes())?;
        writer.write_all(&params.scale.to_le_bytes())?;
        writer.write_all(&params.offset.to_le_bytes())?;
        writer.write_all(&params.vmin.to_le_bytes())?;
        writer.write_all(&params.vmax.to_le_bytes())?;
        writer.write_all(&self.validity)?;

        let mut input = vector_reader(&self.vectors_path)?;
        let mut vector = vec![0.0f32; self.dim];
        let mut input_bytes = vec![0u8; self.dim * std::mem::size_of::<f32>()];
        let mut codes = vec![0u8; self.dim];
        for row in 0..self.row_count as usize {
            read_vector_row(&mut input, &mut input_bytes, &mut vector)?;
            codes.fill(0);
            if is_present(&self.validity, row) {
                for (code, value) in codes.iter_mut().zip(&vector) {
                    *code = quantize_one(*value, params);
                }
            }
            writer.write_all(&codes)?;
        }
        writer.flush()?;
        writer.get_ref().sync_data()?;
        Ok(writer.get_ref().metadata()?.len())
    }
}

fn fitted_sq8_params(mut minimum: f32, mut maximum: f32) -> Sq8Params {
    if !minimum.is_finite() || !maximum.is_finite() {
        minimum = 0.0;
        maximum = 0.0;
    }
    proximadb_codec::functions::sq8::fit_params(&[minimum, maximum])
}

fn create_new(path: &Path) -> Result<File> {
    Ok(OpenOptions::new().create_new(true).write(true).open(path)?)
}

fn vector_reader(path: &Path) -> Result<BufReader<File>> {
    Ok(BufReader::with_capacity(
        VECTOR_BUFFER_BYTES,
        File::open(path)?,
    ))
}

fn read_vector_row(
    reader: &mut BufReader<File>,
    input_bytes: &mut [u8],
    vector: &mut [f32],
) -> Result<()> {
    reader.read_exact(input_bytes)?;
    for (value, bytes) in vector.iter_mut().zip(input_bytes.as_chunks::<4>().0) {
        *value = f32::from_le_bytes(*bytes);
    }
    Ok(())
}

fn is_present(validity: &[u8], row: usize) -> bool {
    validity
        .get(row >> 3)
        .is_some_and(|byte| (byte >> (row & 7)) & 1 == 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_block_format::coalesced_rabitq::{RABITQ_SEED_BASE, encode_region};
    use proximadb_block_format::coalesced_sq8::encode_region as encode_sq8_region;

    #[test]
    fn disk_regions_are_byte_identical_to_canonical_memory_encoders() {
        let root = tempfile::tempdir().expect("scratch root");
        let vectors = [
            None,
            Some(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            Some(vec![-3.0, 0.5, 9.0, 2.5, 1.5, 8.0, -1.0]),
            None,
            Some(vec![4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0]),
        ];
        let mut spool = DiskVectorSpool::new(root.path()).expect("spool");
        for vector in &vectors {
            spool
                .push(vector.as_ref().map(Vec::as_slice))
                .expect("push");
        }
        let regions = spool
            .finish(RABITQ_SEED_BASE, false)
            .expect("encode regions");
        let refs = vectors
            .iter()
            .map(|vector| vector.as_ref().map(Vec::as_slice))
            .collect::<Vec<_>>();
        let (expected_a, expected_centroid) =
            encode_region(&refs, 7, RABITQ_SEED_BASE).expect("memory A");
        let (expected_b, expected_sq8) = encode_sq8_region(&refs, 7).expect("memory B");

        assert_eq!(
            std::fs::read(regions.rabitq_path()).expect("read A"),
            expected_a
        );
        assert_eq!(
            std::fs::read(regions.sq8_path()).expect("read B"),
            expected_b
        );
        assert_eq!(regions.centroid, expected_centroid);
        assert_eq!(regions.sq8_params, expected_sq8);
        assert_eq!(regions.rabitq_len, expected_a.len() as u64);
        assert_eq!(regions.sq8_len, expected_b.len() as u64);
    }

    #[test]
    fn dropping_regions_reclaims_all_region_scratch() {
        let root = tempfile::tempdir().expect("scratch root");
        let task_path = {
            let mut spool = DiskVectorSpool::new(root.path()).expect("spool");
            spool.push(Some(&[1.0, 2.0])).expect("push");
            let regions = spool.finish(7, false).expect("finish");
            regions.scratch_path().to_path_buf()
        };
        assert!(!task_path.exists());
    }

    #[test]
    fn rejects_dimension_drift_before_writing_a_mixed_row() {
        let root = tempfile::tempdir().expect("scratch root");
        let mut spool = DiskVectorSpool::new(root.path()).expect("spool");
        spool.push(Some(&[1.0, 2.0])).expect("first");
        assert!(spool.push(Some(&[1.0, 2.0, 3.0])).is_err());
        assert_eq!(spool.row_count(), 1);
        assert_eq!(spool.dim(), 2);
    }
}
