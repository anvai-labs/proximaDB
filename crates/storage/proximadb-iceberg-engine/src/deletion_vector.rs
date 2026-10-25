//! # deletion_vector — Iceberg v3 `deletion-vector-v1` blobs (P2.4 slice 3)
//!
//! A v3 table deletes rows of a data file with a **deletion vector**: the set of
//! deleted row positions, stored as one blob in a Puffin file. The blob is the
//! spec's `deletion-vector-v1`:
//!
//! ```text
//! u32 BE  length of (magic + vector)
//! 4 B     magic D1 D3 39 64
//! ...     the positions, as a Roaring bitmap in the portable 64-bit format
//! u32 BE  CRC-32 (IEEE) of (magic + vector)
//! ```
//!
//! The portable 64-bit format is a `u64` LE bucket count, then per bucket (ascending)
//! a `u32` LE key holding the positions' high 32 bits and that bucket's low 32 bits as
//! a standard 32-bit Roaring bitmap (the `RoaringFormatSpec` portable layout). The
//! in-repo [`RoaringBitmap`] holds each bucket in memory; its own `serialize` is a
//! private layout, so the portable encoding lives here.
//!
//! Decoding accepts every container kind (array, bitmap, run) and both cookies, since
//! other writers emit run containers. Encoding writes array and bitmap containers
//! under the no-run cookie, which every reader accepts.

use std::collections::BTreeMap;

use proximadb_kernel::error::StorageError;
use proximadb_storage_common::bitmap::RoaringBitmap;

/// The `deletion-vector-v1` magic.
const DV_MAGIC: [u8; 4] = [0xD1, 0xD3, 0x39, 0x64];

/// 32-bit Roaring cookie for a bitmap without run containers.
const SERIAL_COOKIE_NO_RUNCONTAINER: u32 = 12346;
/// 32-bit Roaring cookie (low 16 bits) for a bitmap that may hold run containers.
const SERIAL_COOKIE: u16 = 12347;
/// With [`SERIAL_COOKIE`], the offset header is present only from this many containers.
const NO_OFFSET_THRESHOLD: usize = 4;
/// An array container holds at most this many values; more is a bitmap container.
const ARRAY_MAX: usize = 4096;
/// A bitmap container is 2^16 bits.
const BITMAP_BYTES: usize = 8192;

fn dv_err(context: &str) -> StorageError {
    StorageError::Serialization(format!("iceberg deletion vector: {context}"))
}

/// The deleted row positions of one data file.
#[derive(Debug, Clone, Default)]
pub struct PositionDeletes {
    /// High 32 bits of a position → its low 32 bits.
    buckets: BTreeMap<u32, RoaringBitmap>,
}

impl PositionDeletes {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark `position` deleted. `true` if it was not already.
    pub fn insert(&mut self, position: u64) -> bool {
        self.buckets
            .entry((position >> 32) as u32)
            .or_default()
            .insert(position as u32)
    }

    pub fn contains(&self, position: u64) -> bool {
        self.buckets
            .get(&((position >> 32) as u32))
            .is_some_and(|b| b.contains(position as u32))
    }

    /// The number of deleted positions (the delete file's `record_count`).
    pub fn cardinality(&self) -> u64 {
        self.buckets.values().map(RoaringBitmap::cardinality).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.buckets.values().all(RoaringBitmap::is_empty)
    }

    /// Add every position of `other` (a new DV must carry the previous one's).
    pub fn union_with(&mut self, other: &PositionDeletes) {
        for (key, bitmap) in &other.buckets {
            let merged = match self.buckets.get(key) {
                Some(mine) => mine.union(bitmap),
                None => bitmap.clone(),
            };
            self.buckets.insert(*key, merged);
        }
    }

    /// The deleted positions, ascending.
    pub fn positions(&self) -> Vec<u64> {
        let mut out = Vec::with_capacity(self.cardinality() as usize);
        for (key, bitmap) in &self.buckets {
            let mut low: Vec<u32> = bitmap.to_vec();
            low.sort_unstable();
            out.extend(
                low.into_iter()
                    .map(|l| (u64::from(*key) << 32) | u64::from(l)),
            );
        }
        out
    }

    /// The portable 64-bit Roaring encoding.
    pub fn encode_portable(&self) -> Vec<u8> {
        let buckets: Vec<(u32, Vec<u32>)> = self
            .buckets
            .iter()
            .filter(|(_, b)| !b.is_empty())
            .map(|(key, bitmap)| {
                let mut low = bitmap.to_vec();
                low.sort_unstable();
                (*key, low)
            })
            .collect();
        let mut out = Vec::new();
        out.extend_from_slice(&(buckets.len() as u64).to_le_bytes());
        for (key, low) in &buckets {
            out.extend_from_slice(&key.to_le_bytes());
            encode_roaring32(low, &mut out);
        }
        out
    }

    /// Decode the portable 64-bit Roaring encoding; rejects trailing bytes.
    pub fn decode_portable(bytes: &[u8]) -> Result<Self, StorageError> {
        let mut cur = Cursor { bytes, pos: 0 };
        let count = cur.u64_le()?;
        let mut out = Self::new();
        let mut previous: Option<u32> = None;
        for _ in 0..count {
            let key = cur.u32_le()?;
            if previous.is_some_and(|p| p >= key) {
                return Err(dv_err("bucket keys are not ascending"));
            }
            previous = Some(key);
            let bitmap = decode_roaring32(&mut cur)?;
            out.buckets.insert(key, bitmap);
        }
        if cur.pos != bytes.len() {
            return Err(dv_err("trailing bytes after the bitmap"));
        }
        Ok(out)
    }

    /// The `deletion-vector-v1` blob.
    pub fn encode_blob(&self) -> Vec<u8> {
        let vector = self.encode_portable();
        let mut checked = Vec::with_capacity(4 + vector.len());
        checked.extend_from_slice(&DV_MAGIC);
        checked.extend_from_slice(&vector);
        let mut out = Vec::with_capacity(checked.len() + 8);
        out.extend_from_slice(&(checked.len() as u32).to_be_bytes());
        out.extend_from_slice(&checked);
        out.extend_from_slice(&crc32fast::hash(&checked).to_be_bytes());
        out
    }

    /// Decode a `deletion-vector-v1` blob, checking its length, magic and CRC.
    pub fn decode_blob(blob: &[u8]) -> Result<Self, StorageError> {
        if blob.len() < 12 {
            return Err(dv_err("blob shorter than its framing"));
        }
        let length = u32::from_be_bytes([blob[0], blob[1], blob[2], blob[3]]) as usize;
        if length + 8 != blob.len() {
            return Err(dv_err("blob length does not match its framing"));
        }
        let checked = &blob[4..4 + length];
        if checked[..4] != DV_MAGIC {
            return Err(dv_err("bad magic"));
        }
        let crc = &blob[4 + length..];
        if crc32fast::hash(checked).to_be_bytes() != crc {
            return Err(dv_err("CRC mismatch"));
        }
        Self::decode_portable(&checked[4..])
    }
}

/// Append one 32-bit Roaring bitmap (portable layout, no-run cookie) of the
/// ascending, distinct `values`.
fn encode_roaring32(values: &[u32], out: &mut Vec<u8>) {
    let mut containers: Vec<(u16, &[u32])> = Vec::new();
    let mut start = 0;
    while start < values.len() {
        let key = (values[start] >> 16) as u16;
        let end = start + values[start..].partition_point(|v| (v >> 16) as u16 == key);
        containers.push((key, &values[start..end]));
        start = end;
    }
    out.extend_from_slice(&SERIAL_COOKIE_NO_RUNCONTAINER.to_le_bytes());
    out.extend_from_slice(&(containers.len() as u32).to_le_bytes());
    for (key, members) in &containers {
        out.extend_from_slice(&key.to_le_bytes());
        out.extend_from_slice(&((members.len() - 1) as u16).to_le_bytes());
    }
    // Offsets are relative to the start of this 32-bit bitmap.
    let header = 8 + containers.len() * 8;
    let mut offset = header;
    let base = out.len() - 8 - containers.len() * 4;
    for (_, members) in &containers {
        out.extend_from_slice(&(offset as u32).to_le_bytes());
        offset += container_size(members.len());
    }
    for (_, members) in &containers {
        if members.len() <= ARRAY_MAX {
            for v in *members {
                out.extend_from_slice(&(*v as u16).to_le_bytes());
            }
        } else {
            let mut words = [0u64; BITMAP_BYTES / 8];
            for v in *members {
                let low = (*v & 0xFFFF) as usize;
                words[low / 64] |= 1 << (low % 64);
            }
            for word in words {
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
    }
    debug_assert_eq!(out.len() - base, offset);
}

fn container_size(cardinality: usize) -> usize {
    if cardinality <= ARRAY_MAX {
        cardinality * 2
    } else {
        BITMAP_BYTES
    }
}

/// Decode one 32-bit Roaring bitmap (portable layout, either cookie).
fn decode_roaring32(cur: &mut Cursor<'_>) -> Result<RoaringBitmap, StorageError> {
    let cookie = cur.u32_le()?;
    let (count, run_flags) = if cookie == SERIAL_COOKIE_NO_RUNCONTAINER {
        (cur.u32_le()? as usize, None)
    } else if (cookie & 0xFFFF) as u16 == SERIAL_COOKIE {
        let count = (cookie >> 16) as usize + 1;
        let flags = cur.take(count.div_ceil(8))?.to_vec();
        (count, Some(flags))
    } else {
        return Err(dv_err("unknown Roaring cookie"));
    };
    if count > 1 << 16 {
        return Err(dv_err("more containers than 16-bit keys"));
    }
    let mut descriptors = Vec::with_capacity(count);
    for _ in 0..count {
        let key = cur.u16_le()?;
        let cardinality = usize::from(cur.u16_le()?) + 1;
        descriptors.push((key, cardinality));
    }
    if run_flags.is_none() || count >= NO_OFFSET_THRESHOLD {
        // The offset header only speeds up random access; containers follow in order.
        cur.take(count * 4)?;
    }
    let mut bitmap = RoaringBitmap::new();
    for (i, (key, cardinality)) in descriptors.into_iter().enumerate() {
        let high = u32::from(key) << 16;
        let is_run = run_flags
            .as_ref()
            .is_some_and(|flags| flags[i / 8] & (1 << (i % 8)) != 0);
        if is_run {
            let runs = usize::from(cur.u16_le()?);
            for _ in 0..runs {
                let start = u32::from(cur.u16_le()?);
                let length = u32::from(cur.u16_le()?);
                if start + length > 0xFFFF {
                    return Err(dv_err("run past the end of its container"));
                }
                for low in start..=start + length {
                    bitmap.insert(high | low);
                }
            }
        } else if cardinality <= ARRAY_MAX {
            for _ in 0..cardinality {
                bitmap.insert(high | u32::from(cur.u16_le()?));
            }
        } else {
            let words = cur.take(BITMAP_BYTES)?;
            let (chunks, _) = words.as_chunks::<8>();
            for (w, chunk) in chunks.iter().enumerate() {
                let mut word = u64::from_le_bytes(*chunk);
                while word != 0 {
                    let bit = word.trailing_zeros();
                    bitmap.insert(high | (w as u32 * 64 + bit));
                    word &= word - 1;
                }
            }
        }
    }
    Ok(bitmap)
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], StorageError> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| dv_err("truncated bitmap"))?;
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }
    fn u16_le(&mut self) -> Result<u16, StorageError> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("2 bytes"),
        ))
    }
    fn u32_le(&mut self) -> Result<u32, StorageError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }
    fn u64_le(&mut self) -> Result<u64, StorageError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }
}

#[cfg(test)]
#[path = "deletion_vector_tests.rs"]
mod tests;
