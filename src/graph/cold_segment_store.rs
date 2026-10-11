//! Cold graph-payload **segment** store (TD-168 follow-up #3, Phase 1 — capability).
//!
//! [`ColdGraphRecordStore`](crate::graph::ColdGraphRecordStore) writes one object
//! per node/edge (`graph-cold/{oid}`) — simple, but one PUT per create and one GET
//! per cold fetch. This store **batches** many records into one object so the
//! object-store **op count** drops (the KRU/$ lever the ADR-034 I/O-trace audit
//! flagged): one PUT per *segment* (~thousands of records) instead of per record.
//!
//! ## What this is (Phase 1, unconditional win)
//! - **Write:** records buffer in RAM and flush to one segment object on a size /
//!   count threshold (or an explicit [`flush`](Self::flush)). ⇒ ~Nx fewer PUTs.
//! - **Read:** a point-get reads ONLY the record's bytes via the oid→byte-range
//!   index + a ranged GET — never the whole segment — so there is **no read
//!   regression** vs one-object-per-record. A batched [`get_records`] groups by
//!   segment and coalesces the ranges into one `get_ranges` per segment (so a
//!   frontier that happens to share a segment collapses to ~one round-trip).
//! - **Mixed-read-safe:** an oid not in the segment index falls back to the
//!   legacy `graph-cold/{oid}` point GET, so old data + a partial migration read
//!   correctly.
//!
//! Deferred to Phase 2 (the *conditional* read win): write-time **locality**
//! clustering (insertion-order → Louvain compaction) so a traversal frontier's
//! nodes co-locate in a segment and the GET-*count* drops on read. Phase 1 makes
//! that free to add later (the format + index already support range coalescing).
//!
//! Capability only (not yet wired into production) — like `put_with_tier` (#468)
//! was. Gating + the periodic/​shutdown flush + replacing `ColdGraphRecordStore`
//! in `shared_services` are a separate, separately-reviewed slice.
//!
//! ## Segment format (self-describing, little-endian)
//! ```text
//! [ rec_0 ][ rec_1 ] … [ rec_{n-1} ][ directory ][ dir_len: u64 ][ MAGIC: 8 ]
//! ```
//! `rec_i` = `bincode(ProximaRecordV2)`. `directory` = `bincode(Vec<(oid, off, len)>)`.
//! The trailer (`dir_len` + `MAGIC`) makes a segment self-describing so the index
//! can be rebuilt by scanning segment tails if the sidecar is lost.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use bytes::Bytes;
use dashmap::DashMap;
use object_store::path::Path as ObjectPath;

use proximadb_kernel::error::StorageError;
use proximadb_object_store::ProximaObjectStore;
use proximadb_records::wire_v2::ProximaRecordV2;
use proximadb_records::{ProximaRecord, RecordKey, RecordStore, RecordStoreResult};
use proximadb_storage_filesystem_types::ObjectAccessTier;

const SEG_MAGIC: &[u8; 8] = b"GCSEGv1\0";
const SEG_PREFIX: &str = "graph-cold-seg";
const INDEX_KEY: &str = "graph-cold-seg/index.bin";
/// Legacy one-object-per-record prefix (mixed-read fallback).
const LEGACY_PREFIX: &str = "graph-cold";

const DEFAULT_FLUSH_BYTES: u64 = 16 * 1024 * 1024; // 16 MiB
const DEFAULT_FLUSH_COUNT: usize = 8192;

/// Where a record's bytes live within a segment object.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct RecordLoc {
    segment: String,
    offset: u64,
    len: u64,
}

#[derive(Default)]
struct Buffer {
    /// (oid, encoded ProximaRecordV2 bytes) in arrival order.
    records: Vec<(String, Vec<u8>)>,
    bytes: u64,
}

/// Segment-batched, object-storage-backed [`RecordStore`] for cold graph payloads.
pub struct ColdGraphSegmentStore {
    store: ProximaObjectStore,
    tier: ObjectAccessTier,
    /// oid → location within a segment (the read-side fast path).
    index: DashMap<String, RecordLoc>,
    buffer: Mutex<Buffer>,
    seq: AtomicU64,
    flush_bytes: u64,
    flush_count: usize,
}

impl ColdGraphSegmentStore {
    /// Open a segment store over the object-storage root `url`, writing segments at
    /// `tier`. Loads the persisted oid→segment index if present (so reads work
    /// across restarts); absent index ⇒ empty (all reads fall back to legacy).
    pub async fn from_storage_root(url: &str, tier: ObjectAccessTier) -> RecordStoreResult<Self> {
        let store = ProximaObjectStore::from_url(url)
            .map_err(|e| anyhow::anyhow!("cold segment store: open `{url}` failed: {e}"))?;
        let me = Self::new(store, tier);
        me.load_index().await?;
        Ok(me)
    }

    /// Wrap an existing [`ProximaObjectStore`] (no index load — for tests).
    pub fn new(store: ProximaObjectStore, tier: ObjectAccessTier) -> Self {
        Self {
            store,
            tier,
            index: DashMap::new(),
            buffer: Mutex::new(Buffer::default()),
            seq: AtomicU64::new(0),
            flush_bytes: DEFAULT_FLUSH_BYTES,
            flush_count: DEFAULT_FLUSH_COUNT,
        }
    }

    /// Override the flush thresholds (mainly for tests).
    pub fn with_flush_thresholds(mut self, bytes: u64, count: usize) -> Self {
        self.flush_bytes = bytes.max(1);
        self.flush_count = count.max(1);
        self
    }

    fn lock_buffer(&self) -> RecordStoreResult<std::sync::MutexGuard<'_, Buffer>> {
        self.buffer
            .lock()
            .map_err(|_| anyhow::anyhow!("cold segment store: buffer mutex poisoned"))
    }

    /// Flush any buffered records into a segment. No-op when the buffer is empty.
    pub async fn flush(&self) -> RecordStoreResult<()> {
        let pending = {
            let mut b = self.lock_buffer()?;
            b.bytes = 0;
            std::mem::take(&mut b.records)
        };
        self.write_segment(pending).await
    }

    /// Build + publish one segment from `pending`, then update the index + sidecar.
    async fn write_segment(&self, pending: Vec<(String, Vec<u8>)>) -> RecordStoreResult<()> {
        if pending.is_empty() {
            return Ok(());
        }
        // `fetch_update`, not `fetch_add`. Atomics are NOT covered by
        // `overflow-checks`, so `fetch_add` wraps silently in debug as well as
        // release — which left the resume-site refusals off by one against their
        // own stated invariant: a name carrying `u64::MAX - 1` resumes to
        // `u64::MAX` without refusing, the next write consumes `MAX`, the
        // counter wraps to 0, and the write after that re-issues
        // `seg-0000000000000000.gcseg` and silently overwrites a live segment
        // (`put_with_tier` is an unconditional PUT). Refusing at the point of
        // CONSUMPTION closes the class; refusing only at resume cannot.
        //
        // `fetch_update` returns the PREVIOUS value on success, matching
        // `fetch_add`'s contract, and leaves the counter untouched on refusal.
        //
        // Note it refuses once the counter REACHES `u64::MAX`, so that sequence
        // is never issued — the closure must be able to produce the next value,
        // and a counter that cannot advance has no way to represent
        // "MAX consumed" without a second flag. Burning one sequence out of
        // 2^64 to make a wrap structurally impossible is the right trade.
        let seq = match self
            .seq
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                cur.checked_add(1)
            }) {
            Ok(previous) => previous,
            Err(current) => {
                // Requeue, as every other failure path in this function does:
                // the records were `mem::take`-n out of the buffer by the
                // caller, so dropping them here would lose buffered writes.
                self.requeue(pending)?;
                return Err(anyhow::anyhow!(
                    "cold segment store: segment sequence space exhausted (counter at \
                     {current}); refusing to write rather than wrap onto a live segment name"
                ));
            }
        };
        // Deterministic, collision-free key; not time-based (offline-build constraint).
        let seg_path = format!("{SEG_PREFIX}/seg-{seq:016x}.gcseg");

        let mut body: Vec<u8> = Vec::new();
        let mut dir: Vec<(String, u64, u64)> = Vec::with_capacity(pending.len());
        for (oid, bytes) in &pending {
            let offset = body.len() as u64;
            body.extend_from_slice(bytes);
            dir.push((oid.clone(), offset, bytes.len() as u64));
        }
        let dir_bytes = match bincode::serialize(&dir) {
            Ok(dir_bytes) => dir_bytes,
            Err(e) => {
                self.requeue(pending)?;
                return Err(anyhow::anyhow!(
                    "cold segment store: encode directory failed: {e}"
                ));
            }
        };
        body.extend_from_slice(&dir_bytes);
        body.extend_from_slice(&(dir_bytes.len() as u64).to_le_bytes());
        body.extend_from_slice(SEG_MAGIC);

        // The records were `mem::take`-n out of the buffer by the caller BEFORE this
        // PUT; if the PUT fails, re-queue them so a later flush retries rather than
        // dropping them — a transient object-store error must not silently lose
        // buffered writes (the engine is authoritative and recovery re-population is
        // the backstop, but a blip shouldn't require a full recovery cycle).
        if let Err(e) = self
            .store
            .put_with_tier(
                &ObjectPath::from(seg_path.clone()),
                Bytes::from(body),
                self.tier,
            )
            .await
        {
            self.requeue(pending)?;
            return Err(anyhow::anyhow!(
                "cold segment store: put `{seg_path}` failed: {e}"
            ));
        }

        for (oid, offset, len) in dir {
            self.index.insert(
                oid,
                RecordLoc {
                    segment: seg_path.clone(),
                    offset,
                    len,
                },
            );
        }
        // The segment is now durable. An index-sidecar failure here leaves the
        // records durable-but-unfindable until recovery re-population rewrites them
        // (the segment-tail index rebuild that would heal this directly is a tracked
        // follow-up). Do NOT re-queue — re-flushing would duplicate the durable
        // segment.
        self.persist_index().await
    }

    /// Re-insert a failed flush batch at the FRONT of the buffer so a later flush
    /// retries it. The records left the buffer (`mem::take`) before the segment PUT,
    /// so on a transient PUT/encode failure this is what prevents silent data loss.
    fn requeue(&self, mut pending: Vec<(String, Vec<u8>)>) -> RecordStoreResult<()> {
        let restored: u64 = pending.iter().map(|(_, bytes)| bytes.len() as u64).sum();
        let mut b = self.lock_buffer()?;
        // Failed batch first, then any writes that landed since the take (FIFO-ish).
        pending.append(&mut b.records);
        b.records = pending;
        b.bytes += restored;
        Ok(())
    }

    /// Persist the oid→location index as a bincode sidecar (best-effort durable map
    /// so reads survive restart without rescanning every segment tail).
    async fn persist_index(&self) -> RecordStoreResult<()> {
        let snapshot: Vec<(String, RecordLoc)> = self
            .index
            .iter()
            .map(|e| (e.key().clone(), e.value().clone()))
            .collect();
        let bytes = bincode::serialize(&snapshot)
            .map_err(|e| anyhow::anyhow!("cold segment store: encode index failed: {e}"))?;
        self.store
            .put_with_tier(&ObjectPath::from(INDEX_KEY), Bytes::from(bytes), self.tier)
            .await
            .map_err(|e| anyhow::anyhow!("cold segment store: persist index failed: {e}"))
    }

    /// Load the persisted oid→location index sidecar over the current backing.
    /// `pub(crate)` so a caller that constructs the store with [`Self::new`] over a
    /// shared backing (e.g. crash-recovery tests / a reopen path) can rehydrate it,
    /// mirroring what [`Self::from_storage_root`] does internally.
    pub(crate) async fn load_index(&self) -> RecordStoreResult<()> {
        // 1. Fast path: load the durable sidecar if present.
        match self.store.get(&ObjectPath::from(INDEX_KEY)).await {
            Ok(bytes) => {
                let snapshot: Vec<(String, RecordLoc)> = bincode::deserialize(&bytes)
                    .map_err(|e| anyhow::anyhow!("cold segment store: decode index failed: {e}"))?;
                // Resume the segment sequence past the highest seen, so new segments
                // never clobber existing ones.
                let mut max_seq: u64 = 0;
                for (oid, loc) in snapshot {
                    if let Some(seq) = parse_seg_seq(&loc.segment) {
                        // `checked_add`: `seq` comes from a NAME, so
                        // `seg-ffffffffffffffff.gcseg` would otherwise panic in
                        // debug and WRAP in release — and a wrap resumes at 0,
                        // where the next `put_with_tier` (an unconditional PUT,
                        // with no conditional-create backstop) silently
                        // overwrites a live segment.
                        let Some(next) = seq.checked_add(1) else {
                            return Err(anyhow::anyhow!(
                                "cold segment store: segment sequence space exhausted \
                                 ('{}' carries the maximum); refusing to resume rather \
                                 than reuse a live segment name",
                                loc.segment
                            ));
                        };
                        max_seq = max_seq.max(next);
                    }
                    self.index.insert(oid, loc);
                }
                self.seq.store(max_seq, Ordering::Relaxed);
            }
            // No sidecar yet ⇒ rebuild entirely from segments below (or empty).
            Err(StorageError::NotFound(_)) => {}
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "cold segment store: load index failed: {e}"
                ));
            }
        }
        // 2. Heal: a segment whose body PUT succeeded but whose sidecar update did not
        //    (crash in `write_segment` between the two PUTs) is durable-but-unfindable
        //    via the sidecar alone. Scan any segment NOT covered by the sidecar and
        //    rebuild its entries from the self-describing tail directory — closing that
        //    window without waiting for a recovery re-population cycle. The segment
        //    format `[records][dir][dir_len u64][MAGIC]` was designed for exactly this.
        self.heal_index_from_segments().await
    }

    /// Merge into the index any segment with `seq >= self.seq` — i.e. written after the
    /// sidecar's last persist (or all of them if the sidecar was missing). Processed in
    /// ascending `seq` so a newer segment's entry for an oid wins, mirroring write order.
    async fn heal_index_from_segments(&self) -> RecordStoreResult<()> {
        let covered_through = self.seq.load(Ordering::Relaxed);
        let metas = self
            .store
            .list(Some(&ObjectPath::from(SEG_PREFIX)))
            .await
            .map_err(|e| anyhow::anyhow!("cold segment store: list segments failed: {e}"))?;
        // `list` yields BASE-prefixed locations; the rest of the store keys by the
        // caller-relative segment path (and `get_range` re-applies the base). Parse the
        // seq from the listed name but RECONSTRUCT the canonical relative path — exactly
        // `write_segment`'s format — so reads and the stored index entry stay consistent
        // on a non-empty base (file://, s3://). The `index.bin` sidecar (also under the
        // prefix) fails the `seg-` parse and is skipped.
        let mut uncovered: Vec<(u64, ObjectPath)> = metas
            .into_iter()
            .filter_map(|meta| {
                let seq = parse_seg_seq(meta.location.as_ref())?;
                (seq >= covered_through).then(|| {
                    (
                        seq,
                        ObjectPath::from(format!("{SEG_PREFIX}/seg-{seq:016x}.gcseg")),
                    )
                })
            })
            .collect();
        if uncovered.is_empty() {
            return Ok(());
        }
        uncovered.sort_by_key(|(seq, _)| *seq);
        let mut max_seq = covered_through;
        for (seq, path) in &uncovered {
            self.merge_segment_directory(path).await?;
            // See the `checked_add` note in the index-snapshot path: the
            // sequence is name-derived, and a wrap would resume onto a live name
            // that an unconditional PUT then overwrites.
            let Some(next) = seq.checked_add(1) else {
                return Err(anyhow::anyhow!(
                    "cold segment store: segment sequence space exhausted ('{path}' \
                     carries the maximum); refusing to resume rather than reuse a live \
                     segment name"
                ));
            };
            max_seq = max_seq.max(next);
        }
        self.seq.store(max_seq, Ordering::Relaxed);
        tracing::info!(
            healed = uncovered.len(),
            "cold segment store: rebuilt index entries from segment tails missing from the sidecar"
        );
        Ok(())
    }

    /// Read one segment's trailing `[dir][dir_len u64 LE][MAGIC]` directory and insert
    /// its oid→location entries (no record bodies are fetched).
    async fn merge_segment_directory(&self, path: &ObjectPath) -> RecordStoreResult<()> {
        const TRAILER: u64 = 16; // dir_len (8) + MAGIC (8)
        let size = self
            .store
            .object_size(path)
            .await
            .map_err(|e| anyhow::anyhow!("cold segment store: size `{path}` failed: {e}"))?;
        if size < TRAILER {
            return Err(anyhow::anyhow!(
                "cold segment store: segment `{path}` smaller than its trailer"
            ));
        }
        let tail = self.store.get_suffix(path, TRAILER).await.map_err(|e| {
            anyhow::anyhow!("cold segment store: read trailer `{path}` failed: {e}")
        })?;
        if tail.len() != TRAILER as usize || &tail[8..16] != SEG_MAGIC {
            return Err(anyhow::anyhow!(
                "cold segment store: segment `{path}` has a bad trailer magic"
            ));
        }
        let dir_len =
            u64::from_le_bytes(tail[0..8].try_into().map_err(|_| {
                anyhow::anyhow!("cold segment store: segment `{path}` bad dir_len")
            })?);
        // `checked_add`/`checked_sub`, because `dir_len` is read from the
        // SEGMENT'S OWN TRAILER — the same untrusted input this module's
        // sequence handling now guards. `TRAILER + dir_len` wraps in release
        // (`overflow-checks` is off there, on in dev/test), and a wrap makes the
        // truncation guard *pass*: for `size = 1000`, `dir_len = 0xFFFF…F5` the
        // sum wraps small, `dir_start` wraps to 995, and the range below becomes
        // `995..984` — reversed — instead of the refusal this guard exists to
        // produce. Debug builds would panic instead (mandate #4).
        let Some(dir_end) = TRAILER.checked_add(dir_len) else {
            return Err(anyhow::anyhow!(
                "cold segment store: segment `{path}` declares a directory length that \
                 overflows ({dir_len}); refusing to trust the trailer"
            ));
        };
        if size < dir_end {
            return Err(anyhow::anyhow!(
                "cold segment store: segment `{path}` directory is truncated"
            ));
        }
        let dir_start = size - dir_end;
        let dir_bytes = self
            .store
            .get_range(path, dir_start..(size - TRAILER))
            .await
            .map_err(|e| anyhow::anyhow!("cold segment store: read dir `{path}` failed: {e}"))?;
        let dir: Vec<(String, u64, u64)> = bincode::deserialize(&dir_bytes)
            .map_err(|e| anyhow::anyhow!("cold segment store: decode dir `{path}` failed: {e}"))?;
        let segment = path.to_string();
        for (oid, offset, len) in dir {
            self.index.insert(
                oid,
                RecordLoc {
                    segment: segment.clone(),
                    offset,
                    len,
                },
            );
        }
        Ok(())
    }

    fn decode(bytes: &[u8], oid: &str) -> RecordStoreResult<ProximaRecord> {
        let wire: ProximaRecordV2 = bincode::deserialize(bytes)
            .map_err(|e| anyhow::anyhow!("cold segment store: decode `{oid}` failed: {e}"))?;
        Ok(ProximaRecord::from(wire))
    }

    /// Legacy one-object-per-record fallback (mixed-read-safety with the
    /// `ColdGraphRecordStore` format). Returns `None` if absent.
    async fn legacy_get(&self, oid: &str) -> RecordStoreResult<Option<ProximaRecord>> {
        let key = ObjectPath::from(format!("{LEGACY_PREFIX}/{oid}"));
        match self.store.get(&key).await {
            Ok(bytes) => Ok(Some(Self::decode(&bytes, oid)?)),
            Err(StorageError::NotFound(_)) => Ok(None),
            Err(e) => Err(anyhow::anyhow!(
                "cold segment store: legacy get `{oid}` failed: {e}"
            )),
        }
    }

    /// Buffered (not-yet-flushed) record bytes for `oid`, if present.
    fn buffered(&self, oid: &str) -> RecordStoreResult<Option<Vec<u8>>> {
        let b = self.lock_buffer()?;
        Ok(b.records
            .iter()
            .rev() // last write wins
            .find(|(o, _)| o == oid)
            .map(|(_, bytes)| bytes.clone()))
    }
}

#[async_trait]
impl RecordStore for ColdGraphSegmentStore {
    async fn upsert_record(&self, record: ProximaRecord) -> RecordStoreResult<ProximaRecord> {
        let wire = ProximaRecordV2::from(&record);
        let bytes = bincode::serialize(&wire).map_err(|e| {
            anyhow::anyhow!("cold segment store: encode `{}` failed: {e}", record.oid)
        })?;
        let len = bytes.len() as u64;
        crate::metrics::consumption_metrics::record_object_store_write_bytes_by_tier(
            &record.tenant_id,
            self.tier.as_str(),
            len,
        );
        let pending = {
            let mut b = self.lock_buffer()?;
            b.records.push((record.oid.clone(), bytes));
            b.bytes += len;
            if b.bytes >= self.flush_bytes || b.records.len() >= self.flush_count {
                b.bytes = 0;
                std::mem::take(&mut b.records)
            } else {
                Vec::new()
            }
        };
        self.write_segment(pending).await?;
        Ok(record)
    }

    /// Force the in-memory buffer durable (segment + index sidecar). Overrides the
    /// `RecordStore` default no-op so the graph checkpoint / graceful-shutdown path
    /// (`flush_wal`) can flush this buffered store. Delegates to the inherent
    /// [`Self::flush`].
    async fn flush(&self) -> RecordStoreResult<()> {
        ColdGraphSegmentStore::flush(self).await
    }

    async fn get_record(&self, key: &RecordKey) -> RecordStoreResult<Option<ProximaRecord>> {
        // 1. Buffered (created but not yet flushed).
        if let Some(bytes) = self.buffered(&key.oid)? {
            return Ok(Some(Self::decode(&bytes, &key.oid)?));
        }
        // 2. Segment index → ranged GET of just this record's bytes.
        if let Some(loc) = self.index.get(&key.oid) {
            // `loc.offset`/`loc.len` are bincode-decoded from the segment
            // directory, i.e. untrusted for the same reason as `dir_len` above.
            let Some(end) = loc.offset.checked_add(loc.len) else {
                return Err(anyhow::anyhow!(
                    "cold segment store: record location in `{}` overflows \
                     (offset {} + len {})",
                    loc.segment,
                    loc.offset,
                    loc.len
                ));
            };
            let range = loc.offset..end;
            let bytes = self
                .store
                .get_range(&ObjectPath::from(loc.segment.clone()), range)
                .await
                .map_err(|e| {
                    anyhow::anyhow!("cold segment store: range get `{}` failed: {e}", key.oid)
                })?;
            return Ok(Some(Self::decode(&bytes, &key.oid)?));
        }
        // 3. Legacy one-object-per-record (mixed-read-safety).
        self.legacy_get(&key.oid).await
    }

    async fn get_records(
        &self,
        keys: &[RecordKey],
    ) -> RecordStoreResult<Vec<Option<ProximaRecord>>> {
        let mut out: Vec<Option<ProximaRecord>> = vec![None; keys.len()];
        // Group index-resident keys by segment so each segment is ONE coalesced
        // ranged read (the depth-collapse a co-located frontier earns).
        let mut by_segment: std::collections::HashMap<String, Vec<(usize, RecordLoc)>> =
            std::collections::HashMap::new();
        let mut misses: Vec<usize> = Vec::new();
        for (i, key) in keys.iter().enumerate() {
            if let Some(bytes) = self.buffered(&key.oid)? {
                out[i] = Some(Self::decode(&bytes, &key.oid)?);
            } else if let Some(loc) = self.index.get(&key.oid) {
                by_segment
                    .entry(loc.segment.clone())
                    .or_default()
                    .push((i, loc.clone()));
            } else {
                misses.push(i);
            }
        }
        for (segment, items) in by_segment {
            // Same untrusted input as the single-record path: `offset`/`len`
            // come from the segment directory, so the sum is checked. Collected
            // through a `Result` rather than a plain `map` so one corrupt entry
            // refuses the batch instead of wrapping into a reversed range.
            let ranges: Vec<std::ops::Range<u64>> = items
                .iter()
                .map(|(_, loc)| {
                    loc.offset
                        .checked_add(loc.len)
                        .map(|end| loc.offset..end)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "cold segment store: record location in `{segment}` overflows \
                             (offset {} + len {})",
                                loc.offset,
                                loc.len
                            )
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let bufs = self
                .store
                .get_ranges(&ObjectPath::from(segment.clone()), &ranges)
                .await
                .map_err(|e| {
                    anyhow::anyhow!("cold segment store: get_ranges `{segment}` failed: {e}")
                })?;
            for ((slot, _), bytes) in items.into_iter().zip(bufs) {
                out[slot] = Some(Self::decode(&bytes, &keys[slot].oid)?);
            }
        }
        // Index misses → legacy fallback (concurrently).
        let legacy =
            futures::future::try_join_all(misses.iter().map(|&i| self.legacy_get(&keys[i].oid)))
                .await?;
        for (&i, rec) in misses.iter().zip(legacy) {
            out[i] = rec;
        }
        Ok(out)
    }

    async fn delete_record(&self, key: &RecordKey) -> RecordStoreResult<bool> {
        // Remove from the buffer and the index (segment bytes become garbage,
        // reclaimed by a future compaction — Phase 2). Also best-effort delete a
        // legacy object if one exists.
        let in_buffer = {
            let mut b = self.lock_buffer()?;
            let before = b.records.len();
            b.records.retain(|(o, _)| o != &key.oid);
            before != b.records.len()
        };
        let in_index = self.index.remove(&key.oid).is_some();
        if in_index {
            self.persist_index().await?;
        }
        let legacy_existed = match self
            .store
            .delete(&ObjectPath::from(format!("{LEGACY_PREFIX}/{}", key.oid)))
            .await
        {
            Ok(()) => true,
            Err(StorageError::NotFound(_)) => false,
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "cold segment store: legacy delete `{}` failed: {e}",
                    key.oid
                ));
            }
        };
        Ok(in_buffer || in_index || legacy_existed)
    }
}

/// Width of the hex sequence in a segment name, fixed so a lexicographic sort of
/// names equals their numeric order.
const SEG_SEQ_HEX_WIDTH: usize = 16;

/// Sequence encoded in a segment's name, or `None` if the name is not one this
/// store could have written.
///
/// Strict on purpose. The writer emits exactly `seg-{seq:016x}.gcseg`, so
/// sixteen LOWERCASE hex digits is the whole language; anything else is not ours
/// and must not be assigned a sequence.
///
/// Two gaps this closes, both of which broke the one property the name exists to
/// carry — that a name maps to exactly one sequence, and that sorting names
/// equals sorting sequences:
///
/// * **No width check.** `seg-9.gcseg` parsed to 9 while sorting *after*
///   `seg-10.gcseg`.
/// * **No alphabet check.** `from_str_radix` accepts either case AND a leading
///   `+`, so `seg-00000000000000FF.gcseg` and `seg-+0000000000000ff.gcseg` each
///   claimed the sequence `seg-00000000000000ff.gcseg` owns. Note the shape of
///   the fix: it requires the ALLOWED alphabet rather than rejecting classes
///   someone happened to think of — "reject uppercase" would still have
///   admitted `+`, and `+` is 0x2B, sorting ahead of `0`.
fn parse_seg_seq(path: &str) -> Option<u64> {
    let name = path.rsplit('/').next()?;
    let hex = name.strip_prefix("seg-")?.strip_suffix(".gcseg")?;
    if hex.len() != SEG_SEQ_HEX_WIDTH {
        return None;
    }
    if !hex
        .bytes()
        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return None;
    }
    u64::from_str_radix(hex, 16).ok()
}

#[cfg(test)]
mod tests {

    /// The name format must carry its sequence unambiguously, and sorting names
    /// must equal sorting sequences.
    ///
    /// Asserts the PROPERTY, not just a table of rejections: the previous guard
    /// (none at all) and a half-guard that merely rejected uppercase would both
    /// pass a rejection list that happened to omit the input which breaks them.
    #[test]
    fn seg_name_maps_to_exactly_one_sequence_and_sorts_by_it() {
        // Round-trip over a range plus the boundaries.
        for seq in [0u64, 1, 9, 10, 15, 16, 255, 256, 4095, 4096, u64::MAX] {
            let name = format!("seg-{seq:016x}.gcseg");
            assert_eq!(parse_seg_seq(&name), Some(seq), "{name} must round-trip");
            assert_eq!(
                parse_seg_seq(&format!("{SEG_PREFIX}/{name}")),
                Some(seq),
                "a full path must parse by its final component"
            );
        }

        // Injectivity: no name outside the canonical rendering may claim a
        // sequence that a canonical name owns.
        for seq in [0u64, 15, 255, 4096, u64::MAX] {
            let canonical = format!("seg-{seq:016x}.gcseg");
            for candidate in [
                format!("seg-+{seq:015x}.gcseg"),
                format!("seg-{seq:015X}.gcseg"),
                format!("seg-{seq:016X}.gcseg"),
                format!("seg-{seq:x}.gcseg"),
            ] {
                if candidate == canonical {
                    continue;
                }
                assert_ne!(
                    parse_seg_seq(&candidate),
                    Some(seq),
                    "{candidate} must not claim the sequence {canonical} owns"
                );
            }
        }

        // Lexicographic order equals numeric order.
        let names: Vec<String> = (0..600u64).map(|q| format!("seg-{q:016x}.gcseg")).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(sorted, names, "name order must equal sequence order");

        // Shapes that are not ours.
        for bad in [
            "seg-9.gcseg",                  // unpadded
            "seg-000000000000000.gcseg",    // 15 digits
            "seg-00000000000000000.gcseg",  // 17 digits
            "seg-.gcseg",                   // no sequence
            "seg-00000000000000FF.gcseg",   // uppercase
            "seg-+0000000000000ff.gcseg",   // leading plus: parses via from_str_radix
            "seg--0000000000000ff.gcseg",   // leading minus
            "seg- 0000000000000ff.gcseg",   // leading space
            "seg-0x00000000000000.gcseg",   // radix prefix
            "seg-zzzzzzzzzzzzzzzz.gcseg",   // not hex
            "other-0000000000000001.gcseg", // not our prefix
            "seg-0000000000000001.sst",     // not our extension
        ] {
            assert_eq!(parse_seg_seq(bad), None, "{bad} must not parse");
        }
    }
    use super::*;

    fn mem_store() -> ColdGraphSegmentStore {
        let store = ProximaObjectStore::from_url("memory://").expect("memory store");
        ColdGraphSegmentStore::new(store, ObjectAccessTier::Cool)
    }

    fn record(oid: &str) -> ProximaRecord {
        ProximaRecord {
            oid: oid.to_string(),
            tenant_id: "t".to_string(),
            ..ProximaRecord::default()
        }
    }

    #[tokio::test]
    async fn buffered_record_is_readable_before_flush() {
        let store = mem_store(); // high thresholds ⇒ stays buffered
        store
            .upsert_record(record("graph/g/node/a"))
            .await
            .expect("upsert");
        let got = store
            .get_record(&RecordKey::new("graph/g/node/a"))
            .await
            .expect("get")
            .expect("present in buffer");
        assert_eq!(got.oid, "graph/g/node/a");
        // Nothing flushed yet.
        assert!(store.index.is_empty());
    }

    #[tokio::test]
    async fn flush_batches_into_one_segment_and_reads_via_index() {
        let store = mem_store();
        for id in ["a", "b", "c"] {
            store
                .upsert_record(record(&format!("graph/g/node/{id}")))
                .await
                .expect("upsert");
        }
        store.flush().await.expect("flush");
        // All three share ONE segment.
        assert_eq!(
            store.seq.load(Ordering::Relaxed),
            1,
            "exactly one segment written"
        );
        for id in ["a", "b", "c"] {
            let oid = format!("graph/g/node/{id}");
            let got = store
                .get_record(&RecordKey::new(oid.clone()))
                .await
                .expect("get")
                .expect("present");
            assert_eq!(got.oid, oid);
        }
    }

    #[tokio::test]
    async fn count_threshold_triggers_flush() {
        let store = mem_store().with_flush_thresholds(u64::MAX, 2);
        store
            .upsert_record(record("graph/g/node/a"))
            .await
            .expect("a");
        assert_eq!(store.seq.load(Ordering::Relaxed), 0, "not flushed at 1");
        store
            .upsert_record(record("graph/g/node/b"))
            .await
            .expect("b");
        assert_eq!(store.seq.load(Ordering::Relaxed), 1, "flushed at count=2");
        assert!(store.index.contains_key("graph/g/node/a"));
    }

    #[tokio::test]
    async fn get_records_batches_one_segment_in_order_with_miss() {
        let store = mem_store();
        for id in ["a", "c"] {
            store
                .upsert_record(record(&format!("graph/g/node/{id}")))
                .await
                .expect("upsert");
        }
        store.flush().await.expect("flush");
        let keys = [
            RecordKey::new("graph/g/node/a"),
            RecordKey::new("graph/g/node/b"), // absent
            RecordKey::new("graph/g/node/c"),
        ];
        let got = store.get_records(&keys).await.expect("get_records");
        assert_eq!(
            got[0].as_ref().map(|r| r.oid.as_str()),
            Some("graph/g/node/a")
        );
        assert!(got[1].is_none());
        assert_eq!(
            got[2].as_ref().map(|r| r.oid.as_str()),
            Some("graph/g/node/c")
        );
    }

    /// The WRITE site must refuse at exhaustion too, and requeue the batch.
    ///
    /// Refusing only at the resume sites was off by one: `fetch_add` on an
    /// `AtomicU64` is not covered by `overflow-checks`, so it wraps silently in
    /// BOTH profiles. A name carrying `u64::MAX - 1` resumes to `u64::MAX`
    /// without tripping the resume refusal, the next write consumes `MAX`, the
    /// counter wraps to 0, and the write after that re-issues
    /// `seg-0000000000000000.gcseg` over a live segment.
    ///
    /// Also asserts the records are REQUEUED rather than dropped — every other
    /// failure path in `write_segment` does that, because the caller has already
    /// `mem::take`-n them out of the buffer.
    #[tokio::test]
    async fn write_refuses_at_sequence_exhaustion_and_requeues_the_batch() {
        let backing = ProximaObjectStore::from_url("memory://").expect("mem");
        let store = ColdGraphSegmentStore::new(backing, ObjectAccessTier::Cool);

        // Parked one below the ceiling: that write succeeds and leaves the
        // counter at `u64::MAX`, after which the next must refuse rather than
        // wrap. (Sequence `u64::MAX` is deliberately never issued — see the
        // `fetch_update` note at the write site.)
        store.seq.store(u64::MAX - 1, Ordering::Relaxed);
        store
            .upsert_record(record("graph/g/node/a"))
            .await
            .expect("upsert");
        store
            .flush()
            .await
            .expect("the write at MAX-1 is still legal");

        store
            .upsert_record(record("graph/g/node/b"))
            .await
            .expect("upsert");
        let err = store
            .flush()
            .await
            .expect_err("the write after exhaustion must refuse, not wrap to 0");
        assert!(
            err.to_string().contains("sequence space exhausted"),
            "the error must name the cause; got: {err}"
        );

        // The refused batch is back in the buffer, not lost.
        assert_eq!(
            store.lock_buffer().expect("buffer").records.len(),
            1,
            "a refused write must requeue its records"
        );
    }

    /// A sequence read from a NAME must not be incremented blindly.
    ///
    /// `seg-ffffffffffffffff.gcseg` is a well-formed name carrying `u64::MAX`,
    /// so `seq + 1` panicked in debug (mandate #4) and WRAPPED in release — and
    /// a wrap resumes at 0, where the next segment write silently overwrites a
    /// live object, because this store writes with `put_with_tier` (an
    /// unconditional PUT) and not `put_if_absent`.
    ///
    /// Site 1 (the sidecar load) is reachable with no I/O beyond one planted
    /// index entry, which is why this exercises that path: the heal-from-tails
    /// site would need a VALID segment body under the max name, so it fails on
    /// the trailer before reaching the resume.
    #[tokio::test]
    async fn index_load_refuses_an_exhausted_sequence_rather_than_overflowing() {
        let backing = ProximaObjectStore::from_url("memory://").expect("mem");

        // A sidecar whose only entry points at the maximum-sequence name.
        let snapshot: Vec<(String, RecordLoc)> = vec![(
            "graph/g/node/a".to_string(),
            RecordLoc {
                segment: format!("{SEG_PREFIX}/seg-{:016x}.gcseg", u64::MAX),
                offset: 0,
                len: 1,
            },
        )];
        let bytes = bincode::serialize(&snapshot).expect("encode sidecar");
        backing
            .put_with_tier(
                &ObjectPath::from(INDEX_KEY),
                Bytes::from(bytes),
                ObjectAccessTier::Cool,
            )
            .await
            .expect("plant sidecar");

        let store = ColdGraphSegmentStore::new(backing, ObjectAccessTier::Cool);
        let err = store
            .load_index()
            .await
            .expect_err("an exhausted sequence must refuse, not overflow or wrap");
        let msg = err.to_string();
        assert!(
            msg.contains("sequence space exhausted"),
            "the error must name the cause; got: {msg}"
        );
    }

    #[tokio::test]
    async fn index_reloads_from_sidecar_across_reopen() {
        // Share one in-memory object store across two store instances.
        let backing = ProximaObjectStore::from_url("memory://").expect("mem");
        let s1 = ColdGraphSegmentStore::new(backing.clone(), ObjectAccessTier::Cool);
        s1.upsert_record(record("graph/g/node/a"))
            .await
            .expect("upsert");
        s1.flush().await.expect("flush");

        let s2 = ColdGraphSegmentStore::new(backing, ObjectAccessTier::Cool);
        s2.load_index().await.expect("load index");
        let got = s2
            .get_record(&RecordKey::new("graph/g/node/a"))
            .await
            .expect("get")
            .expect("present after reload");
        assert_eq!(got.oid, "graph/g/node/a");
        // Sequence resumed past the loaded segment.
        assert_eq!(s2.seq.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn index_rebuilds_from_segment_tails_when_sidecar_lost() {
        // The durable-but-unfindable window: segment bodies are durable but the index
        // sidecar is missing/stale (crash between the segment PUT and `persist_index`).
        // `load_index` must rebuild from the self-describing segment tails. Newer
        // segments win on oid collisions (ascending-seq merge).
        let backing = ProximaObjectStore::from_url("memory://").expect("mem");
        let s1 = ColdGraphSegmentStore::new(backing.clone(), ObjectAccessTier::Cool)
            .with_flush_thresholds(u64::MAX, usize::MAX);
        // seg0: a(v1), b, c.
        for id in ["a", "b", "c"] {
            s1.upsert_record(record(&format!("graph/g/node/{id}")))
                .await
                .expect("upsert");
        }
        s1.flush().await.expect("flush seg0");
        // seg1: a(v2) — a newer copy in a higher-seq segment — and d.
        s1.upsert_record(ProximaRecord {
            oid: "graph/g/node/a".to_string(),
            tenant_id: "t".to_string(),
            record_version: 2,
            ..ProximaRecord::default()
        })
        .await
        .expect("upsert a v2");
        s1.upsert_record(record("graph/g/node/d"))
            .await
            .expect("upsert d");
        s1.flush().await.expect("flush seg1");

        // Lose the sidecar (simulates the never-persisted / stale index window).
        s1.store
            .delete(&ObjectPath::from(INDEX_KEY))
            .await
            .expect("delete sidecar");

        // Reopen: no sidecar ⇒ rebuild entirely from both segment tails.
        let s2 = ColdGraphSegmentStore::new(backing, ObjectAccessTier::Cool);
        s2.load_index().await.expect("rebuild from tails");
        for id in ["a", "b", "c", "d"] {
            assert!(
                s2.get_record(&RecordKey::new(format!("graph/g/node/{id}")))
                    .await
                    .expect("get")
                    .is_some(),
                "{id} recovered from segment-tail rebuild"
            );
        }
        // Newer segment wins: a resolves to v2 (seg1), not v1 (seg0).
        let a = s2
            .get_record(&RecordKey::new("graph/g/node/a"))
            .await
            .expect("get a")
            .expect("a present");
        assert_eq!(a.record_version, 2, "higher-seq segment wins for a");
        // Sequence resumed past the highest segment (seg1).
        assert_eq!(s2.seq.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn flush_via_record_store_trait_makes_buffer_durable_across_reopen() {
        // The checkpoint/shutdown path flushes through the `RecordStore` trait
        // object, so the trait override (not just the inherent method) must dispatch
        // to the real flush. High thresholds keep writes buffered until we flush.
        let backing = ProximaObjectStore::from_url("memory://").expect("mem");
        let s1 = ColdGraphSegmentStore::new(backing.clone(), ObjectAccessTier::Cool)
            .with_flush_thresholds(u64::MAX, usize::MAX);
        for id in ["a", "b", "c"] {
            s1.upsert_record(record(&format!("graph/g/node/{id}")))
                .await
                .expect("upsert");
        }
        assert!(s1.index.is_empty(), "buffered, nothing flushed yet");

        // Flush THROUGH the trait object — proves `RecordStore::flush` dispatches.
        let dyn_store: &dyn RecordStore = &s1;
        dyn_store.flush().await.expect("trait flush");

        // Reopen over the same backing: every record is durable + index-findable.
        let s2 = ColdGraphSegmentStore::new(backing, ObjectAccessTier::Cool);
        s2.load_index().await.expect("load index");
        for id in ["a", "b", "c"] {
            let oid = format!("graph/g/node/{id}");
            assert!(
                s2.get_record(&RecordKey::new(oid))
                    .await
                    .expect("get")
                    .is_some(),
                "record durable after trait flush + reopen"
            );
        }
    }

    #[tokio::test]
    async fn requeue_restores_failed_batch_to_buffer() {
        // The buffer is `mem::take`-n before the segment PUT; `requeue` must put a
        // failed batch back so a later flush retries instead of dropping it.
        let store = mem_store().with_flush_thresholds(u64::MAX, usize::MAX);
        store
            .upsert_record(record("graph/g/node/a"))
            .await
            .expect("upsert");
        let pending = {
            let mut b = store.lock_buffer().expect("lock");
            b.bytes = 0;
            std::mem::take(&mut b.records)
        };
        assert_eq!(pending.len(), 1);
        store.requeue(pending).expect("requeue");
        // The record is back in the buffer and a real flush now persists it.
        store.flush().await.expect("flush");
        assert!(store.index.contains_key("graph/g/node/a"));
    }

    #[tokio::test]
    async fn delete_removes_from_index() {
        let store = mem_store();
        store
            .upsert_record(record("graph/g/node/a"))
            .await
            .expect("upsert");
        store.flush().await.expect("flush");
        assert!(
            store
                .delete_record(&RecordKey::new("graph/g/node/a"))
                .await
                .expect("delete")
        );
        assert!(
            store
                .get_record(&RecordKey::new("graph/g/node/a"))
                .await
                .expect("get")
                .is_none()
        );
    }
}
