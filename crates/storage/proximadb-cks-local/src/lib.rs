// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! # `proximadb-cks-local` — the generation-fenced, MVCC local-WAL `ConditionalKeyStore`
//!
//! F1b + **F1d** (ADR-072 D5/D11): the hot, correctness-carrying default
//! implementation of [`ConditionalKeyStore`], with an **append-only MVCC**
//! reclaim model.
//!
//! - **Versioned, not slot-reclaiming (D11/D14).** Each key keeps an
//!   append-ordered version history (`Some(oid)` = a claim, `None` = a
//!   tombstone). A PK update/delete + re-insert appends *new* versions; a slot is
//!   never overwritten.
//! - **Monotonic generation (ABA defense).** The store stamps every mutation with
//!   a strictly increasing generation (≥ the caller's fence), so a reused key can
//!   never masquerade as an older version.
//! - **Snapshot reads.** [`LocalWalKeyStore::get_at`] resolves the version with
//!   the greatest generation ≤ a snapshot — the MVCC visibility the record store
//!   reads against.
//! - **Reader-watermark GC.** [`LocalWalKeyStore::compact`] rewrites the WAL,
//!   dropping versions no active reader can still observe (below the
//!   oldest-active-reader watermark). Physical scheduling is the F6 background
//!   plane; the mechanism lives here.
//!
//! Durable via a CRC-framed, torn-tail-tolerant WAL (ledger framing):
//! `[len u32-le][crc32 u32-le][payload]`,
//! payload = `op u8 | generation u64-le | key_len u32-le | key | oid_len u32-le | oid`.

use anyhow::{Context, Result};
use proximadb_storage_ports::{
    AtomicScope, ConditionalKeyStore, Generation, Identity, Oid, PutOutcome,
};
use std::collections::{BTreeMap, HashMap};
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// WAL fsync policy (mirrors the ledger's lever).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncPolicy {
    /// `fsync` after every append — strict durability (the default).
    PerOp,
    /// Buffer appends; the caller flushes at a checkpoint (throughput).
    Deferred,
}

const OP_PUT: u8 = 0;
const OP_TOMBSTONE: u8 = 1;

/// IEEE CRC-32 (reflected, poly `0xEDB88320`). Inline (no dependency); identical
/// to the ledger WAL's framing.
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// One version of a key. `oid = None` is a tombstone.
#[derive(Clone)]
struct Version {
    generation: Generation,
    oid: Option<Oid>,
}

struct Inner {
    /// Per key, versions in ascending generation order (the WAL append order).
    map: HashMap<Identity, Vec<Version>>,
    /// Monotonic generation clock; the next stamp is `> clock` and `>= fence`.
    clock: u64,
    /// Active reader snapshots -> refcount (the GC watermark source).
    readers: BTreeMap<u64, usize>,
    log: BufWriter<File>,
    sync: SyncPolicy,
    path: PathBuf,
}

/// A durable, generation-fenced, MVCC [`ConditionalKeyStore`] backed by a local WAL.
pub struct LocalWalKeyStore {
    inner: Mutex<Inner>,
}

impl LocalWalKeyStore {
    /// Open (or create) the store at `path`, replaying any existing WAL.
    pub fn open(path: impl AsRef<Path>, sync: SyncPolicy) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let (map, clock) =
            replay(&path).with_context(|| format!("replaying WAL at {}", path.display()))?;
        let log = open_append(&path)?;
        Ok(Self {
            inner: Mutex::new(Inner {
                map,
                clock,
                readers: BTreeMap::new(),
                log: BufWriter::new(log),
                sync,
                path,
            }),
        })
    }

    /// Flush buffered appends to stable storage (a no-op under [`SyncPolicy::PerOp`]).
    pub fn flush(&self) -> Result<()> {
        let mut g = self.inner.lock().unwrap();
        g.log.flush()?;
        g.log.get_ref().sync_all()?;
        Ok(())
    }

    /// The current snapshot generation (the latest committed generation).
    pub fn snapshot(&self) -> Generation {
        Generation(self.inner.lock().unwrap().clock)
    }

    /// Begin a read at the current snapshot; the returned guard pins the GC
    /// watermark until dropped, so [`Self::compact`] cannot reclaim versions this
    /// reader can still observe.
    pub fn begin_read(&self) -> ReadGuard<'_> {
        let mut g = self.inner.lock().unwrap();
        let snap = g.clock;
        *g.readers.entry(snap).or_insert(0) += 1;
        ReadGuard {
            store: self,
            snapshot: Generation(snap),
        }
    }

    /// Resolve `key` at `snapshot`: the version with the greatest generation
    /// `<= snapshot`, mapped to its holder (`None` if that version is a tombstone
    /// or the key did not exist yet).
    pub fn get_at(&self, key: &Identity, snapshot: Generation) -> Option<Oid> {
        let g = self.inner.lock().unwrap();
        resolve_at(g.map.get(key)?, snapshot).cloned()
    }

    /// Compact the WAL: drop every version no active reader can still observe
    /// (strictly below the oldest-active-reader watermark, except the single
    /// latest version at-or-below it that a watermark read must resolve), then
    /// rewrite the log. Bounds WAL growth without losing visible state.
    pub fn compact(&self) -> Result<()> {
        let mut g = self.inner.lock().unwrap();
        let watermark = g.oldest_active();

        let mut kept: HashMap<Identity, Vec<Version>> = HashMap::new();
        for (id, versions) in &g.map {
            let pruned = prune(versions, watermark);
            if !pruned.is_empty() {
                kept.insert(id.clone(), pruned);
            }
        }

        // Rewrite to a temp file, fsync, then atomically rename over the log.
        let tmp = g.path.with_extension("wal.compact");
        {
            let mut w = BufWriter::new(
                OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(&tmp)?,
            );
            for (id, versions) in &kept {
                for v in versions {
                    let (op, oid): (u8, &[u8]) = match &v.oid {
                        Some(o) => (OP_PUT, o.0.as_bytes()),
                        None => (OP_TOMBSTONE, &[]),
                    };
                    write_frame(&mut w, op, v.generation, id.as_bytes(), oid)?;
                }
            }
            w.flush()?;
            w.get_ref().sync_all()?;
        }
        std::fs::rename(&tmp, &g.path)?;

        g.map = kept;
        g.log = BufWriter::new(open_append(&g.path)?);
        Ok(())
    }
}

impl Inner {
    /// The GC watermark: the oldest active reader snapshot, or the current clock
    /// when no reads are in flight (everything superseded is then reclaimable).
    fn oldest_active(&self) -> Generation {
        Generation(self.readers.keys().next().copied().unwrap_or(self.clock))
    }

    /// Stamp the next monotonic generation, respecting the caller's fence.
    fn tick(&mut self, fence: Generation) -> Generation {
        self.clock = (self.clock + 1).max(fence.0);
        Generation(self.clock)
    }

    fn append(&mut self, op: u8, stamp: Generation, key: &[u8], oid: &[u8]) -> io::Result<()> {
        write_frame(&mut self.log, op, stamp, key, oid)?;
        if self.sync == SyncPolicy::PerOp {
            self.log.flush()?;
            self.log.get_ref().sync_all()?;
        }
        Ok(())
    }
}

/// A live read reservation; pins the GC watermark until dropped.
pub struct ReadGuard<'a> {
    store: &'a LocalWalKeyStore,
    snapshot: Generation,
}

impl ReadGuard<'_> {
    /// The snapshot this read is pinned at.
    pub fn snapshot(&self) -> Generation {
        self.snapshot
    }
}

impl Drop for ReadGuard<'_> {
    fn drop(&mut self) {
        let mut g = self.store.inner.lock().unwrap();
        if let Some(n) = g.readers.get_mut(&self.snapshot.0) {
            *n -= 1;
            if *n == 0 {
                g.readers.remove(&self.snapshot.0);
            }
        }
    }
}

fn latest(versions: &[Version]) -> Option<&Version> {
    versions.last()
}

fn resolve_at(versions: &[Version], snapshot: Generation) -> Option<&Oid> {
    versions
        .iter()
        .rev()
        .find(|v| v.generation <= snapshot)
        .and_then(|v| v.oid.as_ref())
}

/// Keep every version `>= watermark`, plus the single latest version strictly
/// below it (what a read at the watermark must still resolve). Drop the rest. If
/// the sole survivor is a tombstone, the key is fully dead -> drop it.
fn prune(versions: &[Version], watermark: Generation) -> Vec<Version> {
    let split = versions.partition_point(|v| v.generation < watermark);
    let mut out = Vec::new();
    if split > 0 {
        out.push(versions[split - 1].clone());
    }
    out.extend_from_slice(&versions[split..]);
    if out.len() == 1 && out[0].oid.is_none() {
        return Vec::new();
    }
    out
}

fn write_frame(
    w: &mut impl Write,
    op: u8,
    stamp: Generation,
    key: &[u8],
    oid: &[u8],
) -> io::Result<()> {
    let mut payload = Vec::with_capacity(1 + 8 + 4 + key.len() + 4 + oid.len());
    payload.push(op);
    payload.extend_from_slice(&stamp.0.to_le_bytes());
    payload.extend_from_slice(&(key.len() as u32).to_le_bytes());
    payload.extend_from_slice(key);
    payload.extend_from_slice(&(oid.len() as u32).to_le_bytes());
    payload.extend_from_slice(oid);

    let len = u32::try_from(payload.len()).map_err(io::Error::other)?;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(&crc32(&payload).to_le_bytes())?;
    w.write_all(&payload)?;
    Ok(())
}

fn open_append(path: &Path) -> Result<File> {
    OpenOptions::new()
        .create(true)
        .append(true)
        .read(true)
        .open(path)
        .with_context(|| format!("opening WAL at {}", path.display()))
}

/// Rebuild version histories from the WAL, stopping at the first torn/short/bad-CRC
/// frame. Returns the map and the max generation seen (the clock).
fn replay(path: &Path) -> Result<(HashMap<Identity, Vec<Version>>, u64)> {
    let mut map: HashMap<Identity, Vec<Version>> = HashMap::new();
    let mut clock = 0u64;
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok((map, clock)),
        Err(e) => return Err(e.into()),
    };
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    let mut pos = 0usize;
    while pos + 8 <= buf.len() {
        let len = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
        let crc = u32::from_le_bytes(buf[pos + 4..pos + 8].try_into().unwrap());
        let start = pos + 8;
        if start + len > buf.len() || crc32(&buf[start..start + len]) != crc {
            break; // torn tail or corrupt frame
        }
        let Some((op, stamp, key, oid)) = decode(&buf[start..start + len]) else {
            break;
        };
        let version = match op {
            OP_PUT => match String::from_utf8(oid) {
                Ok(s) => Version {
                    generation: stamp,
                    oid: Some(Oid(s)),
                },
                Err(_) => break, // non-UTF-8 oid -> corrupt
            },
            OP_TOMBSTONE => Version {
                generation: stamp,
                oid: None,
            },
            _ => break, // unknown op
        };
        clock = clock.max(stamp.0);
        map.entry(Identity::from_bytes(key))
            .or_default()
            .push(version);
        pos = start + len;
    }
    Ok((map, clock))
}

fn decode(p: &[u8]) -> Option<(u8, Generation, Vec<u8>, Vec<u8>)> {
    let mut i = 0usize;
    let op = *p.get(i)?;
    i += 1;
    let gen_bytes: [u8; 8] = p.get(i..i + 8)?.try_into().ok()?;
    let stamp = Generation(u64::from_le_bytes(gen_bytes));
    i += 8;
    let key_len = u32::from_le_bytes(p.get(i..i + 4)?.try_into().ok()?) as usize;
    i += 4;
    let key = p.get(i..i + key_len)?.to_vec();
    i += key_len;
    let oid_len = u32::from_le_bytes(p.get(i..i + 4)?.try_into().ok()?) as usize;
    i += 4;
    let oid = p.get(i..i + oid_len)?.to_vec();
    i += oid_len;
    if i != p.len() {
        return None;
    }
    Some((op, stamp, key, oid))
}

#[async_trait::async_trait]
impl ConditionalKeyStore for LocalWalKeyStore {
    fn atomic_scope(&self) -> AtomicScope {
        AtomicScope::PerKey
    }

    async fn put_if_absent(
        &self,
        key: &Identity,
        oid: &Oid,
        fence: Generation,
    ) -> Result<PutOutcome> {
        // Blocking WAL I/O under a std mutex; no `.await` is held across the lock.
        let mut g = self.inner.lock().unwrap();
        if let Some(v) = g.map.get(key).and_then(|vs| latest(vs))
            && let Some(holder) = &v.oid
        {
            return Ok(PutOutcome::Conflict {
                holder: holder.clone(),
                generation: v.generation,
            });
        }
        let stamp = g.tick(fence);
        g.append(OP_PUT, stamp, key.as_bytes(), oid.0.as_bytes())?;
        g.map.entry(key.clone()).or_default().push(Version {
            generation: stamp,
            oid: Some(oid.clone()),
        });
        Ok(PutOutcome::Committed { generation: stamp })
    }

    async fn get(&self, key: &Identity) -> Result<Option<Oid>> {
        let g = self.inner.lock().unwrap();
        Ok(g.map
            .get(key)
            .and_then(|vs| latest(vs))
            .and_then(|v| v.oid.clone()))
    }

    async fn live_claims(&self) -> Result<Option<Vec<(Identity, Oid)>>> {
        let g = self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("conditional key store lock poisoned"))?;
        Ok(Some(
            g.map
                .iter()
                .filter_map(|(key, versions)| {
                    latest(versions)
                        .and_then(|v| v.oid.clone())
                        .map(|oid| (key.clone(), oid))
                })
                .collect(),
        ))
    }

    async fn tombstone(&self, key: &Identity, fence: Generation) -> Result<()> {
        let mut g = self.inner.lock().unwrap();
        let live = g
            .map
            .get(key)
            .and_then(|vs| latest(vs))
            .map(|v| v.oid.is_some())
            .unwrap_or(false);
        if live {
            let stamp = g.tick(fence);
            g.append(OP_TOMBSTONE, stamp, key.as_bytes(), &[])?;
            g.map.entry(key.clone()).or_default().push(Version {
                generation: stamp,
                oid: None,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(s: &str) -> Identity {
        Identity::from_bytes(s.as_bytes().to_vec())
    }
    fn tmp() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("cks-{}-{}.wal", std::process::id(), n))
    }

    #[tokio::test]
    async fn mvcc_snapshot_resolution_and_monotonic_generations() {
        let path = tmp();
        let s = LocalWalKeyStore::open(&path, SyncPolicy::PerOp).unwrap();
        let a = Oid("a".into());
        let b = Oid("b".into());

        // put(a), tombstone, put(b) -> strictly increasing generations.
        let g1 = match s.put_if_absent(&id("k"), &a, Generation(0)).await.unwrap() {
            PutOutcome::Committed { generation } => generation,
            other => panic!("{other:?}"),
        };
        s.tombstone(&id("k"), Generation(0)).await.unwrap();
        let g3 = match s.put_if_absent(&id("k"), &b, Generation(0)).await.unwrap() {
            PutOutcome::Committed { generation } => generation,
            other => panic!("{other:?}"),
        };
        assert!(g1 < g3, "generations must be monotonic: {g1:?} < {g3:?}");

        // Snapshot reads see the version visible at that generation.
        assert_eq!(s.get_at(&id("k"), g1), Some(a.clone())); // at g1, 'a' is live
        assert_eq!(s.get_at(&id("k"), Generation(g3.0 - 1)), None); // after tombstone, before g3
        assert_eq!(s.get_at(&id("k"), g3), Some(b.clone())); // at g3, 'b' is live
        assert_eq!(s.get(&id("k")).await.unwrap(), Some(b)); // current

        // A higher fence jumps the clock (fence is respected).
        match s
            .put_if_absent(&id("k2"), &a, Generation(1000))
            .await
            .unwrap()
        {
            PutOutcome::Committed { generation } => assert!(generation.0 >= 1000),
            other => panic!("{other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn compaction_bounds_history_but_preserves_visible_state_and_survives_restart() {
        let path = tmp();
        {
            let s = LocalWalKeyStore::open(&path, SyncPolicy::PerOp).unwrap();
            let a = Oid("a".into());
            let b = Oid("b".into());
            // Churn a key through several versions with no active readers.
            s.put_if_absent(&id("k"), &a, Generation(0)).await.unwrap();
            s.tombstone(&id("k"), Generation(0)).await.unwrap();
            s.put_if_absent(&id("k"), &b, Generation(0)).await.unwrap();
            // A separate key that ends dead should be reclaimed entirely.
            s.put_if_absent(&id("dead"), &a, Generation(0))
                .await
                .unwrap();
            s.tombstone(&id("dead"), Generation(0)).await.unwrap();

            s.compact().unwrap(); // watermark = current clock (no readers)

            assert_eq!(s.get(&id("k")).await.unwrap(), Some(b));
            assert_eq!(s.get(&id("dead")).await.unwrap(), None);
        }
        // Compacted WAL replays to the same visible state.
        {
            let s = LocalWalKeyStore::open(&path, SyncPolicy::PerOp).unwrap();
            assert_eq!(s.get(&id("k")).await.unwrap(), Some(Oid("b".into())));
            assert_eq!(s.get(&id("dead")).await.unwrap(), None);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn active_reader_pins_the_watermark() {
        let path = tmp();
        let s = LocalWalKeyStore::open(&path, SyncPolicy::PerOp).unwrap();
        let a = Oid("a".into());
        let b = Oid("b".into());
        s.put_if_absent(&id("k"), &a, Generation(0)).await.unwrap();
        let snap = s.snapshot();
        let reader = s.begin_read(); // pins `snap`
        s.tombstone(&id("k"), Generation(0)).await.unwrap();
        s.put_if_absent(&id("k"), &b, Generation(0)).await.unwrap();

        s.compact().unwrap(); // must NOT drop the version visible at `snap`
        assert_eq!(s.get_at(&id("k"), snap), Some(a)); // reader still sees 'a'
        drop(reader);
        let _ = std::fs::remove_file(&path);
    }
}
