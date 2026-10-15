// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Persistent, rebuildable local-disk byte tier.
//!
//! The store is deliberately a cache, never a correctness authority. Entries
//! are immutable, atomically published files addressed by a stable SHA-256 of
//! their logical key. Startup reconstructs the in-memory index from those
//! files. A malformed or checksum-failing entry degrades to a miss.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crc32fast::Hasher as Crc32;
use dashmap::DashMap;
use sha2::{Digest, Sha256};

/// File-family discriminator, not a compatibility version. This cache is
/// rebuildable and has no pre-GA legacy readers.
const MAGIC: &[u8; 8] = b"PXLCACHE";
const HEADER_LEN: u64 = 8 + 1 + 4 + 8 + 4 + 4;
/// Integrity and read-amplification unit for verified ranged reads. This
/// follows local-disk range locality, not the object-store billing unit: a
/// 4 MiB integrity chunk made ~1 MiB survivor reads hash 4 MiB each. The CRC
/// table costs only 4 bytes per 64 KiB (0.006%).
const CHECKSUM_CHUNK_BYTES: u64 = 64 * 1024;
const SHARDS: usize = 16;
const MAX_KEY_BYTES: usize = 16 * 1024 * 1024;
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Eviction class for the shared disk budget. Lower-valued classes are
/// discarded first; immutable search control/invariants survive survivor churn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum L2Class {
    Survivor = 0,
    Invariants = 1,
}

impl L2Class {
    fn parse(value: u8) -> std::io::Result<Self> {
        match value {
            0 => Ok(Self::Survivor),
            1 => Ok(Self::Invariants),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unknown persistent cache class {value}"),
            )),
        }
    }
}

#[derive(Debug)]
struct EntryMeta {
    path: PathBuf,
    value_offset: u64,
    value_len: u64,
    file_len: u64,
    value_crc: u32,
    chunk_crcs: Arc<[u32]>,
    class: L2Class,
    touch: AtomicU64,
}

#[derive(Debug)]
struct StoreInner {
    base: PathBuf,
    max_bytes: u64,
    resident_bytes: AtomicU64,
    class_resident_bytes: [AtomicU64; 2],
    clock: AtomicU64,
    entries: DashMap<String, Arc<EntryMeta>>,
    mutation: tokio::sync::Mutex<()>,
}

/// A persistent raw-byte cache for a local filesystem.
///
/// Clones share one index and byte budget. Reads never become object-store
/// correctness dependencies: any corrupt/missing local entry returns a miss.
/// Fast SSD/NVMe media is recommended for query latency, but it is not part of
/// the storage contract.
#[derive(Clone, Debug)]
pub struct PersistentByteStore {
    inner: Arc<StoreInner>,
}

impl PersistentByteStore {
    /// Open or rebuild a store rooted at `base`, bounded by `max_bytes`.
    pub fn open(base: impl AsRef<Path>, max_bytes: u64) -> std::io::Result<Self> {
        let base = base.as_ref().to_path_buf();
        std::fs::create_dir_all(&base)?;
        for shard in 0..SHARDS {
            std::fs::create_dir_all(base.join(format!("shard_{shard:02}")))?;
        }
        let inner = Arc::new(StoreInner {
            base,
            max_bytes,
            resident_bytes: AtomicU64::new(0),
            class_resident_bytes: [AtomicU64::new(0), AtomicU64::new(0)],
            clock: AtomicU64::new(1),
            entries: DashMap::new(),
            mutation: tokio::sync::Mutex::new(()),
        });
        let store = Self { inner };
        store.rebuild_index()?;
        store.evict_blocking();
        Ok(store)
    }

    /// Bytes occupied by valid indexed entry files.
    pub fn resident_bytes(&self) -> u64 {
        self.inner.resident_bytes.load(Ordering::Relaxed)
    }

    pub fn resident_bytes_for(&self, class: L2Class) -> u64 {
        self.inner.class_resident_bytes[class as usize].load(Ordering::Relaxed)
    }

    /// Number of indexed entries.
    pub fn entry_count(&self) -> usize {
        self.inner.entries.len()
    }

    /// Atomically persist an immutable byte value.
    pub async fn put(
        &self,
        key: impl Into<String>,
        class: L2Class,
        value: Arc<[u8]>,
    ) -> std::io::Result<()> {
        let key = key.into();
        if key.len() > MAX_KEY_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "persistent cache key exceeds 16 MiB",
            ));
        }
        let _guard = self.inner.mutation.lock().await;
        if let Some(existing) = self.inner.entries.get(&key) {
            existing.touch.store(self.next_tick(), Ordering::Relaxed);
            return Ok(());
        }
        let final_path = self.entry_path(&key);
        let temp_path = final_path.with_extension(format!(
            "tmp-{}-{}",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let key_for_write = key.clone();
        let value_for_write = value.clone();
        let final_for_write = final_path.clone();
        let temp_for_write = temp_path.clone();
        let write_meta = tokio::task::spawn_blocking(move || {
            write_entry_file(
                &temp_for_write,
                &final_for_write,
                &key_for_write,
                class,
                &value_for_write,
            )
        })
        .await
        .map_err(join_error)??;

        if let Some(meta) = write_meta {
            let file_len = meta.file_len;
            if let Some(old) = self.inner.entries.insert(key, Arc::new(meta)) {
                self.inner
                    .resident_bytes
                    .fetch_sub(old.file_len, Ordering::Relaxed);
                self.inner.class_resident_bytes[old.class as usize]
                    .fetch_sub(old.file_len, Ordering::Relaxed);
            }
            self.inner
                .resident_bytes
                .fetch_add(file_len, Ordering::Relaxed);
            self.inner.class_resident_bytes[class as usize].fetch_add(file_len, Ordering::Relaxed);
        } else if !self.inner.entries.contains_key(&key) {
            // A prior process may already have published this immutable key.
            if let Ok((stored_key, meta)) = read_entry_metadata(&final_path)
                && stored_key == key
            {
                self.inner
                    .resident_bytes
                    .fetch_add(meta.file_len, Ordering::Relaxed);
                self.inner.class_resident_bytes[meta.class as usize]
                    .fetch_add(meta.file_len, Ordering::Relaxed);
                self.inner.entries.insert(key, Arc::new(meta));
            }
        }
        self.evict_blocking();
        Ok(())
    }

    /// Atomically persist one byte range from an existing local file without
    /// materializing the value in memory.
    ///
    /// This is the write-time promotion seam for disk-backed segment writers:
    /// the segment is already present in local staging, so copying its immutable
    /// Region A/B bytes directly into L2 avoids both an object-store GET and a
    /// segment-sized `Arc<[u8]>`. The cache entry uses the same checksummed file
    /// format as [`Self::put`].
    pub async fn put_file_range(
        &self,
        key: impl Into<String>,
        class: L2Class,
        source: impl AsRef<Path>,
        offset: u64,
        len: u64,
    ) -> std::io::Result<()> {
        let key = key.into();
        if key.len() > MAX_KEY_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "persistent cache key exceeds 16 MiB",
            ));
        }
        let _guard = self.inner.mutation.lock().await;
        if let Some(existing) = self.inner.entries.get(&key) {
            existing.touch.store(self.next_tick(), Ordering::Relaxed);
            return Ok(());
        }
        let final_path = self.entry_path(&key);
        let temp_path = final_path.with_extension(format!(
            "tmp-{}-{}",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let source = source.as_ref().to_path_buf();
        let key_for_write = key.clone();
        let final_for_write = final_path.clone();
        let temp_for_write = temp_path.clone();
        let write_meta = tokio::task::spawn_blocking(move || {
            write_entry_file_range(
                &temp_for_write,
                &final_for_write,
                &key_for_write,
                class,
                &source,
                offset,
                len,
            )
        })
        .await
        .map_err(join_error)??;

        if let Some(meta) = write_meta {
            let file_len = meta.file_len;
            if let Some(old) = self.inner.entries.insert(key, Arc::new(meta)) {
                self.inner
                    .resident_bytes
                    .fetch_sub(old.file_len, Ordering::Relaxed);
                self.inner.class_resident_bytes[old.class as usize]
                    .fetch_sub(old.file_len, Ordering::Relaxed);
            }
            self.inner
                .resident_bytes
                .fetch_add(file_len, Ordering::Relaxed);
            self.inner.class_resident_bytes[class as usize].fetch_add(file_len, Ordering::Relaxed);
        } else if !self.inner.entries.contains_key(&key)
            && let Ok((stored_key, meta)) = read_entry_metadata(&final_path)
            && stored_key == key
        {
            self.inner
                .resident_bytes
                .fetch_add(meta.file_len, Ordering::Relaxed);
            self.inner.class_resident_bytes[meta.class as usize]
                .fetch_add(meta.file_len, Ordering::Relaxed);
            self.inner.entries.insert(key, Arc::new(meta));
        }
        self.evict_blocking();
        Ok(())
    }

    /// Read and checksum-verify a complete value.
    pub async fn get(&self, key: &str) -> std::io::Result<Option<Arc<[u8]>>> {
        let Some(meta) = self.inner.entries.get(key).map(|entry| entry.clone()) else {
            return Ok(None);
        };
        let meta_for_read = meta.clone();
        let result = tokio::task::spawn_blocking(move || read_full_value(&meta_for_read))
            .await
            .map_err(join_error)?;
        match result {
            Ok(bytes) => {
                meta.touch.store(self.next_tick(), Ordering::Relaxed);
                Ok(Some(Arc::from(bytes)))
            }
            Err(error) if is_rebuildable_miss(&error) => {
                self.remove_indexed(key, true);
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    /// Read a byte subrange relative to the cached value.
    ///
    /// Every touched 64 KiB chunk is checked against its persisted CRC before
    /// any requested bytes are returned. All touched chunks are fetched as one
    /// contiguous envelope, avoiding a seek/read syscall per checksum chunk.
    pub async fn get_range(
        &self,
        key: &str,
        offset: u64,
        len: u64,
    ) -> std::io::Result<Option<Arc<[u8]>>> {
        let Some(meta) = self.inner.entries.get(key).map(|entry| entry.clone()) else {
            return Ok(None);
        };
        let Some(end) = offset.checked_add(len) else {
            return Ok(None);
        };
        if end > meta.value_len {
            return Ok(None);
        }
        let meta_for_read = meta.clone();
        let result =
            tokio::task::spawn_blocking(move || read_verified_range(&meta_for_read, offset, len))
                .await
                .map_err(join_error)?;
        match result {
            Ok(bytes) => {
                meta.touch.store(self.next_tick(), Ordering::Relaxed);
                Ok(Some(Arc::from(bytes)))
            }
            Err(error) if is_rebuildable_miss(&error) => {
                self.remove_indexed(key, true);
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    /// Remove a single logical entry. Failure to delete a cache file is
    /// intentionally best-effort; the next startup can reclaim it.
    pub fn remove(&self, key: &str) -> bool {
        self.remove_indexed(key, true)
    }

    /// Remove every indexed entry whose logical key matches `predicate`.
    ///
    /// The mutation lock serializes the sweep with atomic publication so a
    /// compaction cannot leave behind a persistent-only entry that was not
    /// resident in the DRAM tier at invalidation time.
    pub async fn remove_where(&self, predicate: impl Fn(&str) -> bool) -> usize {
        let _guard = self.inner.mutation.lock().await;
        let victims: Vec<String> = self
            .inner
            .entries
            .iter()
            .filter(|entry| predicate(entry.key()))
            .map(|entry| entry.key().clone())
            .collect();
        let mut removed = 0;
        for key in victims {
            removed += usize::from(self.remove_indexed(&key, true));
        }
        removed
    }

    fn next_tick(&self) -> u64 {
        self.inner.clock.fetch_add(1, Ordering::Relaxed)
    }

    fn entry_path(&self, key: &str) -> PathBuf {
        let digest = Sha256::digest(key.as_bytes());
        let shard = (digest[0] as usize) % SHARDS;
        let mut name = String::with_capacity(digest.len() * 2 + 5);
        for byte in digest {
            use std::fmt::Write as _;
            let _ = write!(name, "{byte:02x}");
        }
        name.push_str(".pxl2");
        self.inner.base.join(format!("shard_{shard:02}")).join(name)
    }

    fn rebuild_index(&self) -> std::io::Result<()> {
        for shard in 0..SHARDS {
            let shard_path = self.inner.base.join(format!("shard_{shard:02}"));
            for dir_entry in std::fs::read_dir(shard_path)? {
                let Ok(dir_entry) = dir_entry else {
                    continue;
                };
                let path = dir_entry.path();
                if path.extension().and_then(|ext| ext.to_str()) != Some("pxl2") {
                    let _ = std::fs::remove_file(path);
                    continue;
                }
                match read_entry_metadata(&path) {
                    Ok((key, meta)) => {
                        let file_len = meta.file_len;
                        let class = meta.class;
                        if let Some(old) = self.inner.entries.insert(key, Arc::new(meta)) {
                            self.inner
                                .resident_bytes
                                .fetch_sub(old.file_len, Ordering::Relaxed);
                            self.inner.class_resident_bytes[old.class as usize]
                                .fetch_sub(old.file_len, Ordering::Relaxed);
                        }
                        self.inner
                            .resident_bytes
                            .fetch_add(file_len, Ordering::Relaxed);
                        self.inner.class_resident_bytes[class as usize]
                            .fetch_add(file_len, Ordering::Relaxed);
                    }
                    Err(_) => {
                        let _ = std::fs::remove_file(path);
                    }
                }
            }
        }
        Ok(())
    }

    fn evict_blocking(&self) {
        while self.resident_bytes() > self.inner.max_bytes {
            let victim = self
                .inner
                .entries
                .iter()
                .min_by_key(|entry| {
                    (
                        entry.value().class,
                        entry.value().touch.load(Ordering::Relaxed),
                    )
                })
                .map(|entry| entry.key().clone());
            let Some(victim) = victim else {
                break;
            };
            if !self.remove_indexed(&victim, true) {
                break;
            }
        }
    }

    fn remove_indexed(&self, key: &str, delete_file: bool) -> bool {
        let Some((_, meta)) = self.inner.entries.remove(key) else {
            return false;
        };
        let _ =
            self.inner
                .resident_bytes
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                    Some(current.saturating_sub(meta.file_len))
                });
        let _ = self.inner.class_resident_bytes[meta.class as usize].try_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |current| Some(current.saturating_sub(meta.file_len)),
        );
        if delete_file {
            let _ = std::fs::remove_file(&meta.path);
        }
        true
    }

    #[cfg(test)]
    fn entry_path_for_test(&self, key: &str) -> PathBuf {
        self.entry_path(key)
    }
}

fn write_entry_file(
    temp_path: &Path,
    final_path: &Path,
    key: &str,
    class: L2Class,
    value: &[u8],
) -> std::io::Result<Option<EntryMeta>> {
    if final_path.exists() {
        return Ok(None);
    }
    let mut checksum = Crc32::new();
    checksum.update(value);
    let value_crc = checksum.finalize();
    let chunk_crcs: Vec<u32> = value
        .chunks(CHECKSUM_CHUNK_BYTES as usize)
        .map(|chunk| {
            let mut checksum = Crc32::new();
            checksum.update(chunk);
            checksum.finalize()
        })
        .collect();
    let chunk_count = u32::try_from(chunk_crcs.len()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "persistent cache chunk count exceeds u32",
        )
    })?;
    let key_len = u32::try_from(key.len()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "persistent cache key length exceeds u32",
        )
    })?;
    let value_len = value.len() as u64;
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(temp_path)?;
    let result = (|| {
        file.write_all(MAGIC)?;
        file.write_all(&[class as u8])?;
        file.write_all(&key_len.to_le_bytes())?;
        file.write_all(&value_len.to_le_bytes())?;
        file.write_all(&value_crc.to_le_bytes())?;
        file.write_all(&chunk_count.to_le_bytes())?;
        file.write_all(key.as_bytes())?;
        for crc in &chunk_crcs {
            file.write_all(&crc.to_le_bytes())?;
        }
        file.write_all(value)?;
        file.sync_all()?;
        std::fs::rename(temp_path, final_path)?;
        let crc_table_len = u64::from(chunk_count) * 4;
        let value_offset = HEADER_LEN + key.len() as u64 + crc_table_len;
        let file_len = value_offset + value_len;
        Ok(Some(EntryMeta {
            path: final_path.to_path_buf(),
            value_offset,
            value_len,
            file_len,
            value_crc,
            chunk_crcs: Arc::from(chunk_crcs),
            class,
            touch: AtomicU64::new(1),
        }))
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(temp_path);
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn write_entry_file_range(
    temp_path: &Path,
    final_path: &Path,
    key: &str,
    class: L2Class,
    source_path: &Path,
    source_offset: u64,
    value_len: u64,
) -> std::io::Result<Option<EntryMeta>> {
    if final_path.exists() {
        return Ok(None);
    }
    let source_end = source_offset.checked_add(value_len).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "persistent cache source range overflows u64",
        )
    })?;
    let mut source = std::fs::File::open(source_path)?;
    if source.metadata()?.len() < source_end {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "persistent cache source range exceeds local file",
        ));
    }
    source.seek(SeekFrom::Start(source_offset))?;

    let chunk_count_u64 = if value_len == 0 {
        0
    } else {
        value_len.div_ceil(CHECKSUM_CHUNK_BYTES)
    };
    let chunk_count = u32::try_from(chunk_count_u64).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "persistent cache chunk count exceeds u32",
        )
    })?;
    let key_len = u32::try_from(key.len()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "persistent cache key length exceeds u32",
        )
    })?;
    let crc_table_len = u64::from(chunk_count) * 4;
    let value_offset = HEADER_LEN
        .checked_add(key.len() as u64)
        .and_then(|offset| offset.checked_add(crc_table_len))
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "persistent cache entry offset overflows u64",
            )
        })?;
    let file_len = value_offset.checked_add(value_len).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "persistent cache entry length overflows u64",
        )
    })?;
    let mut output = std::fs::OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(temp_path)?;
    let result = (|| {
        output.write_all(MAGIC)?;
        output.write_all(&[class as u8])?;
        output.write_all(&key_len.to_le_bytes())?;
        output.write_all(&value_len.to_le_bytes())?;
        output.write_all(&0u32.to_le_bytes())?;
        output.write_all(&chunk_count.to_le_bytes())?;
        output.write_all(key.as_bytes())?;
        output.seek(SeekFrom::Start(value_offset))?;

        let mut value_checksum = Crc32::new();
        let mut chunk_crcs = Vec::with_capacity(chunk_count as usize);
        let mut remaining = value_len;
        let mut buffer = vec![0u8; CHECKSUM_CHUNK_BYTES as usize];
        while remaining > 0 {
            let take = usize::try_from(remaining.min(CHECKSUM_CHUNK_BYTES)).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "persistent cache chunk length exceeds usize",
                )
            })?;
            source.read_exact(&mut buffer[..take])?;
            value_checksum.update(&buffer[..take]);
            let mut chunk_checksum = Crc32::new();
            chunk_checksum.update(&buffer[..take]);
            chunk_crcs.push(chunk_checksum.finalize());
            output.write_all(&buffer[..take])?;
            remaining -= take as u64;
        }
        let value_crc = value_checksum.finalize();
        output.seek(SeekFrom::Start(21))?;
        output.write_all(&value_crc.to_le_bytes())?;
        output.seek(SeekFrom::Start(HEADER_LEN + key.len() as u64))?;
        for crc in &chunk_crcs {
            output.write_all(&crc.to_le_bytes())?;
        }
        output.sync_all()?;
        std::fs::rename(temp_path, final_path)?;
        Ok(Some(EntryMeta {
            path: final_path.to_path_buf(),
            value_offset,
            value_len,
            file_len,
            value_crc,
            chunk_crcs: Arc::from(chunk_crcs),
            class,
            touch: AtomicU64::new(1),
        }))
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(temp_path);
    }
    result
}

fn read_entry_metadata(path: &Path) -> std::io::Result<(String, EntryMeta)> {
    let mut file = std::fs::File::open(path)?;
    let actual_len = file.metadata()?.len();
    let mut header = [0u8; HEADER_LEN as usize];
    file.read_exact(&mut header)?;
    if &header[..8] != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "persistent cache magic mismatch",
        ));
    }
    let class = L2Class::parse(header[8])?;
    let key_len = u32::from_le_bytes(header[9..13].try_into().map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid key length field")
    })?) as usize;
    if key_len > MAX_KEY_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "persistent cache key length exceeds limit",
        ));
    }
    let value_len = u64::from_le_bytes(header[13..21].try_into().map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid value length field",
        )
    })?);
    let value_crc = u32::from_le_bytes(header[21..25].try_into().map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid checksum field")
    })?);
    let chunk_count = u32::from_le_bytes(header[25..29].try_into().map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid checksum chunk-count field",
        )
    })?);
    let expected_chunk_count = if value_len == 0 {
        0
    } else {
        value_len.div_ceil(CHECKSUM_CHUNK_BYTES)
    };
    if u64::from(chunk_count) != expected_chunk_count {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "persistent cache checksum chunk count mismatch",
        ));
    }
    let crc_table_len = u64::from(chunk_count) * 4;
    let expected_len = HEADER_LEN
        .checked_add(key_len as u64)
        .and_then(|len| len.checked_add(crc_table_len))
        .and_then(|len| len.checked_add(value_len))
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "persistent cache entry length overflow",
            )
        })?;
    if actual_len != expected_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("persistent cache length {actual_len} != expected {expected_len}"),
        ));
    }
    let mut key_bytes = vec![0u8; key_len];
    file.read_exact(&mut key_bytes)?;
    let key = String::from_utf8(key_bytes).map_err(|error| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("persistent cache key is not UTF-8: {error}"),
        )
    })?;
    let crc_table_bytes = usize::try_from(crc_table_len).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "persistent cache checksum table does not fit address space",
        )
    })?;
    let mut encoded_crcs = vec![0u8; crc_table_bytes];
    file.read_exact(&mut encoded_crcs)?;
    let mut chunk_crcs = Vec::with_capacity(chunk_count as usize);
    for encoded in encoded_crcs.as_chunks::<4>().0 {
        chunk_crcs.push(u32::from_le_bytes(*encoded));
    }
    let value_offset = HEADER_LEN + key_len as u64 + crc_table_len;
    Ok((
        key,
        EntryMeta {
            path: path.to_path_buf(),
            value_offset,
            value_len,
            file_len: actual_len,
            value_crc,
            chunk_crcs: Arc::from(chunk_crcs),
            class,
            touch: AtomicU64::new(1),
        },
    ))
}

fn read_full_value(meta: &EntryMeta) -> std::io::Result<Vec<u8>> {
    let len = usize::try_from(meta.value_len).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "persistent cache value does not fit address space",
        )
    })?;
    let value = read_value_range(meta, 0, meta.value_len)?;
    if value.len() != len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "persistent cache value was truncated",
        ));
    }
    let mut checksum = Crc32::new();
    checksum.update(&value);
    if checksum.finalize() != meta.value_crc {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "persistent cache value checksum mismatch",
        ));
    }
    Ok(value)
}

fn read_value_range(meta: &EntryMeta, offset: u64, len: u64) -> std::io::Result<Vec<u8>> {
    let len = usize::try_from(len).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "persistent cache range does not fit address space",
        )
    })?;
    let mut file = std::fs::File::open(&meta.path)?;
    file.seek(SeekFrom::Start(meta.value_offset + offset))?;
    let mut bytes = vec![0u8; len];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn read_verified_range(meta: &EntryMeta, offset: u64, len: u64) -> std::io::Result<Vec<u8>> {
    if len == 0 {
        return Ok(Vec::new());
    }
    let end = offset.checked_add(len).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "persistent cache range overflow",
        )
    })?;
    let expected_len = usize::try_from(len).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "persistent cache range does not fit address space",
        )
    })?;
    let first_chunk = offset / CHECKSUM_CHUNK_BYTES;
    let last_chunk = (end - 1) / CHECKSUM_CHUNK_BYTES;
    let verified_start = first_chunk * CHECKSUM_CHUNK_BYTES;
    let verified_end = ((last_chunk + 1) * CHECKSUM_CHUNK_BYTES).min(meta.value_len);
    let verified_len = usize::try_from(verified_end - verified_start).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "persistent cache verified envelope does not fit address space",
        )
    })?;
    let mut file = std::fs::File::open(&meta.path)?;
    file.seek(SeekFrom::Start(meta.value_offset + verified_start))?;
    let mut verified = vec![0u8; verified_len];
    file.read_exact(&mut verified)?;

    for chunk_index in first_chunk..=last_chunk {
        let chunk_start = chunk_index * CHECKSUM_CHUNK_BYTES;
        let chunk_end = (chunk_start + CHECKSUM_CHUNK_BYTES).min(meta.value_len);
        let relative_start = usize::try_from(chunk_start - verified_start).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "persistent cache checksum start does not fit address space",
            )
        })?;
        let relative_end = usize::try_from(chunk_end - verified_start).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "persistent cache checksum end does not fit address space",
            )
        })?;
        let chunk = verified.get(relative_start..relative_end).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "persistent cache checksum chunk lies outside verified envelope",
            )
        })?;
        let mut checksum = Crc32::new();
        checksum.update(chunk);
        let expected_crc = meta
            .chunk_crcs
            .get(usize::try_from(chunk_index).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "persistent cache chunk index does not fit address space",
                )
            })?)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "persistent cache checksum chunk is missing",
                )
            })?;
        if checksum.finalize() != *expected_crc {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "persistent cache chunk checksum mismatch",
            ));
        }
    }
    let requested_start = usize::try_from(offset - verified_start).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "persistent cache result start does not fit address space",
        )
    })?;
    let requested_end = requested_start.checked_add(expected_len).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "persistent cache result range overflow",
        )
    })?;
    if requested_end > verified.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "persistent cache verified range was truncated",
        ));
    }
    verified.copy_within(requested_start..requested_end, 0);
    verified.truncate(expected_len);
    Ok(verified)
}

fn is_rebuildable_miss(error: &std::io::Error) -> bool {
    matches!(
        error.kind(),
        std::io::ErrorKind::NotFound
            | std::io::ErrorKind::InvalidData
            | std::io::ErrorKind::UnexpectedEof
    )
}

fn join_error(error: tokio::task::JoinError) -> std::io::Error {
    std::io::Error::other(format!("persistent cache worker failed: {error}"))
}

#[cfg(test)]
mod tests {
    use super::{CHECKSUM_CHUNK_BYTES, L2Class, PersistentByteStore};
    use std::io::{Seek, SeekFrom, Write};
    use std::sync::Arc;

    #[tokio::test]
    async fn value_survives_store_reopen() {
        let dir = tempfile::tempdir().expect("test tempdir");
        let store = PersistentByteStore::open(dir.path(), 1 << 20).expect("open");
        store
            .put(
                "tenant-a/segment-a",
                L2Class::Survivor,
                Arc::from(&b"persistent"[..]),
            )
            .await
            .expect("put");
        drop(store);

        let reopened = PersistentByteStore::open(dir.path(), 1 << 20).expect("reopen");
        assert!(reopened.resident_bytes_for(L2Class::Survivor) > 0);
        assert_eq!(
            reopened
                .get("tenant-a/segment-a")
                .await
                .expect("get")
                .as_deref(),
            Some(&b"persistent"[..])
        );
    }

    #[tokio::test]
    async fn verified_range_crosses_checksum_chunks_without_full_value_read() {
        let dir = tempfile::tempdir().expect("test tempdir");
        let budget = CHECKSUM_CHUNK_BYTES * 2;
        let store = PersistentByteStore::open(dir.path(), budget).expect("open");
        let mut value = vec![0u8; CHECKSUM_CHUNK_BYTES as usize + 8];
        let boundary = CHECKSUM_CHUNK_BYTES as usize;
        value[boundary - 2..boundary + 2].copy_from_slice(&[1, 2, 3, 4]);
        store
            .put("segment/sq8", L2Class::Survivor, Arc::from(value))
            .await
            .expect("put");

        assert_eq!(
            store
                .get_range("segment/sq8", CHECKSUM_CHUNK_BYTES - 2, 4)
                .await
                .expect("range")
                .as_deref(),
            Some(&[1, 2, 3, 4][..])
        );
    }

    #[tokio::test]
    async fn local_file_range_is_streamed_into_reopenable_entry() {
        let dir = tempfile::tempdir().expect("test tempdir");
        let source_path = dir.path().join("segment.pax");
        let value_len = CHECKSUM_CHUNK_BYTES * 2 + 17;
        let mut source = vec![0xA5; 31 + value_len as usize + 29];
        for (index, byte) in source[31..31 + value_len as usize].iter_mut().enumerate() {
            *byte = (index % 251) as u8;
        }
        std::fs::write(&source_path, &source).expect("write source");

        let cache_path = dir.path().join("cache");
        let store = PersistentByteStore::open(&cache_path, 1 << 20).expect("open");
        store
            .put_file_range(
                "segment/region-b",
                L2Class::Survivor,
                &source_path,
                31,
                value_len,
            )
            .await
            .expect("stream file range");
        drop(store);

        let reopened = PersistentByteStore::open(&cache_path, 1 << 20).expect("reopen");
        let offset = CHECKSUM_CHUNK_BYTES - 9;
        let len = CHECKSUM_CHUNK_BYTES + 13;
        let actual = reopened
            .get_range("segment/region-b", offset, len)
            .await
            .expect("range read")
            .expect("cached range");
        let expected_start = 31 + offset as usize;
        assert_eq!(
            actual.as_ref(),
            &source[expected_start..expected_start + len as usize]
        );
    }

    #[tokio::test]
    async fn out_of_bounds_local_file_range_does_not_publish_entry() {
        let dir = tempfile::tempdir().expect("test tempdir");
        let source_path = dir.path().join("short.pax");
        std::fs::write(&source_path, b"short").expect("write source");
        let store = PersistentByteStore::open(dir.path().join("cache"), 1 << 20).expect("open");

        let error = store
            .put_file_range("segment/region-a", L2Class::Invariants, &source_path, 3, 8)
            .await
            .expect_err("range must fail");
        assert_eq!(error.kind(), std::io::ErrorKind::UnexpectedEof);
        assert_eq!(store.entry_count(), 0);
    }

    #[tokio::test]
    async fn corrupt_requested_chunk_is_a_range_miss() {
        let dir = tempfile::tempdir().expect("test tempdir");
        let store = PersistentByteStore::open(dir.path(), 1 << 20).expect("open");
        store
            .put(
                "segment/sq8",
                L2Class::Survivor,
                Arc::from(&b"0123456789"[..]),
            )
            .await
            .expect("put");
        let value_offset = store
            .inner
            .entries
            .get("segment/sq8")
            .map(|entry| entry.value_offset)
            .expect("indexed entry");
        let path = store.entry_path_for_test("segment/sq8");
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .expect("open entry");
        file.seek(SeekFrom::Start(value_offset))
            .expect("seek value");
        file.write_all(b"x").expect("corrupt value");
        file.sync_all().expect("sync corruption");

        assert!(
            store
                .get_range("segment/sq8", 0, 4)
                .await
                .expect("range")
                .is_none()
        );
        assert_eq!(store.entry_count(), 0);
    }

    #[tokio::test]
    async fn corruption_is_a_cache_miss_not_bad_data() {
        let dir = tempfile::tempdir().expect("test tempdir");
        let store = PersistentByteStore::open(dir.path(), 1 << 20).expect("open");
        store
            .put("segment/a", L2Class::Invariants, Arc::from(&b"valid"[..]))
            .await
            .expect("put");
        let path = store.entry_path_for_test("segment/a");
        std::fs::write(path, b"corrupt").expect("corrupt test entry");

        assert!(store.get("segment/a").await.expect("get").is_none());
    }

    #[tokio::test]
    async fn pressure_evicts_survivors_before_invariants() {
        let dir = tempfile::tempdir().expect("test tempdir");
        let store = PersistentByteStore::open(dir.path(), 220).expect("open");
        store
            .put("inv", L2Class::Invariants, Arc::from(vec![1u8; 80]))
            .await
            .expect("put invariant");
        store
            .put("survivor-old", L2Class::Survivor, Arc::from(vec![2u8; 80]))
            .await
            .expect("put survivor");
        store
            .put("survivor-new", L2Class::Survivor, Arc::from(vec![3u8; 80]))
            .await
            .expect("put survivor");

        assert!(store.get("inv").await.expect("get inv").is_some());
        assert!(store.get("survivor-old").await.expect("get old").is_none());
    }

    #[tokio::test]
    async fn remove_where_purges_persistent_only_entries() {
        let dir = tempfile::tempdir().expect("test tempdir");
        let store = PersistentByteStore::open(dir.path(), 1 << 20).expect("open");
        store
            .put(
                "segment-a/control",
                L2Class::Invariants,
                Arc::from(&b"a"[..]),
            )
            .await
            .expect("put a");
        store
            .put(
                "segment-b/control",
                L2Class::Invariants,
                Arc::from(&b"b"[..]),
            )
            .await
            .expect("put b");

        assert_eq!(
            store
                .remove_where(|key| key.starts_with("segment-a/"))
                .await,
            1
        );
        assert!(
            store
                .get("segment-a/control")
                .await
                .expect("get a")
                .is_none()
        );
        assert!(
            store
                .get("segment-b/control")
                .await
                .expect("get b")
                .is_some()
        );
    }
}
